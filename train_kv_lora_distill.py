"""Train a KV-only LoRA adapter by distilling Wan cross-attention projections.

This script loads a Wan diffusion backbone, injects LoRA modules into all
cross-attention K and V projections and optimizes those adapters so that the
keys/values produced from a bridge encoder match those from the original
UMT5 encoder.

It intentionally keeps the diffusion transformer frozen and only trains the
LoRA parameters, allowing the resulting weights to be exported in a
PEFT-compatible format.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List, Tuple
from datetime import datetime, timezone
import shutil
from tqdm import tqdm
import time
import gc
from glob import glob
import subprocess
import shlex

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from safetensors.torch import load_file, safe_open

from kv_lora_inject import (
    export_peft_adapter,
    inject_lora_kv,
    lora_parameters,
    set_lora_enabled,
    load_peft_adapter,
)

# Force the training process to use UMT5 as teacher, regardless of shell env
os.environ["WAN_USE_BRIDGE"] = "0"
print(f"[env] WAN_USE_BRIDGE (trainer) = {os.environ.get('WAN_USE_BRIDGE','<unset>')}")

# Wan imports live under the inference package.
# The Wan2.2 inference package uses a directory name with a dot. To make it
# importable as a Python package we append it to ``sys.path`` and then import
# from the ``wan`` submodule.
WAN22_DIR = Path(__file__).resolve().parent / "inference" / "Wan2.2"
sys.path.append(str(WAN22_DIR))
from wan.modules.model import WanModel  # type: ignore  # noqa: E402
from wan.modules.t5 import T5EncoderModel  # type: ignore  # noqa: E402
from wan.modules.t5_bridge import BridgeEncoderModel  # type: ignore  # noqa: E402


# -----------------------------------------------------------------------------
# Data helpers
# -----------------------------------------------------------------------------


class PromptDataset(Dataset):
    """
    Reads prompts from either a .txt (one per line) or a .jsonl with a text field.
    Supports optional shuffle and max_samples for quick subsampling.
    """

    def __init__(
        self,
        path: str,
        jsonl_field: str | None = None,
        shuffle: bool = False,
        max_samples: int | None = None,
    ):
        p = Path(path)
        ex = p.suffix.lower()
        prompts: list[str] = []
        if ex == ".jsonl":
            field = jsonl_field or "caption"
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        txt = obj.get(field, "")
                        if isinstance(txt, str) and txt.strip():
                            prompts.append(txt.strip())
                    except Exception:
                        continue
        else:
            with open(p, "r", encoding="utf-8") as f:
                prompts = [ln.strip() for ln in f if ln.strip()]
        if shuffle:
            random.shuffle(prompts)
        if max_samples is not None:
            prompts = prompts[:max_samples]
        self.prompts = prompts

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.prompts)

    def __getitem__(self, idx: int) -> str:  # pragma: no cover - trivial
        return self.prompts[idx]


# -----------------------------------------------------------------------------
# Encoding utilities
# -----------------------------------------------------------------------------


def pad_stack(
    tokens: List[torch.Tensor], max_len: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad variable length token sequences to `max_len`.

    Returns tuple (padded, mask) with shapes [B, max_len, C] and [B, max_len, 1].
    """

    b = len(tokens)
    d = tokens[0].shape[-1]
    padded = torch.zeros(b, max_len, d, device=tokens[0].device, dtype=tokens[0].dtype)
    mask = torch.zeros(b, max_len, 1, device=tokens[0].device, dtype=torch.float32)
    for i, t in enumerate(tokens):
        length = min(t.shape[0], max_len)
        padded[i, :length] = t[:length]
        mask[i, :length, 0] = 1.0
    return padded, mask


def encode_teacher(
    prompts: List[str], encoder: T5EncoderModel, device: torch.device, max_len: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    tokens = encoder(prompts, device)
    return pad_stack(tokens, max_len)


def encode_bridge(
    prompts: List[str], encoder: BridgeEncoderModel, device: torch.device, max_len: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    tokens = encoder(prompts, device)
    # Bridge returns variable-length lists; pad and return its own mask.
    padded, mask = pad_stack(tokens, max_len)
    return padded, mask


# -----------------------------------------------------------------------------
# Training helpers
# -----------------------------------------------------------------------------


def compute_kv(
    model: WanModel, context: torch.Tensor, use_norm: bool
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    k_list: List[torch.Tensor] = []
    v_list: List[torch.Tensor] = []
    for blk in model.blocks:
        ca = blk.cross_attn
        k = ca.k(context)
        v = ca.v(context)
        if use_norm and hasattr(ca, "norm_k"):
            k = ca.norm_k(k)
        k_list.append(k)
        v_list.append(v)
    return k_list, v_list


def _strip_prefix(k: str) -> str:
    """Normalize pretrained keys: drop common prefixes to match WanModel attr names."""
    for p in ("diffusion_model.", "transformer.", "model."):
        if k.startswith(p):
            return k[len(p) :]
    return k


def _needed_key(name: str) -> bool:
    """We only need text_embedding and cross-attn K/V (& norms)."""
    if name.startswith("text_embedding"):
        return True
    if ".cross_attn." in name:
        # keep k, v, and q/k norms (q not strictly needed for this trainer but harmless)
        return any(
            sub in name
            for sub in (
                ".cross_attn.k",
                ".cross_attn.v",
                ".cross_attn.norm_k",
                ".cross_attn.norm_q",
            )
        )
    return False


def _load_filtered_state(
    weights_dir: str | None, weights_file: str | None
) -> dict[str, torch.Tensor]:
    """
    Load only the tensors we need from Wan 5B sharded weights.
    Accepts either a directory containing shards+index or a single .safetensors file.
    """
    state: dict[str, torch.Tensor] = {}
    if weights_dir:
        weights_dir = os.fspath(weights_dir)
        # Prefer index if present, otherwise glob shards.
        index_path = (
            Path(weights_dir) / "diffusion_pytorch_model.safetensors.index.json"
        )
        shard_paths = []
        if index_path.exists():
            with open(index_path, "r", encoding="utf-8") as f:
                idx = json.load(f)
            # Collect only files referenced by needed keys
            needed = set()
            for k, fn in idx.get("weight_map", {}).items():
                name = _strip_prefix(k)
                if _needed_key(name):
                    needed.add(fn)
            shard_paths = sorted({os.path.join(weights_dir, fn) for fn in needed})
        if not shard_paths:
            shard_paths = sorted(
                str(p)
                for p in Path(weights_dir).glob("diffusion_pytorch_model-*.safetensors")
            )
        for sp in shard_paths:
            with safe_open(sp, framework="pt", device="cpu") as f:
                for k in f.keys():
                    name = _strip_prefix(k)
                    if _needed_key(name) and name not in state:
                        state[name] = f.get_tensor(k)
    else:
        assert (
            weights_file is not None
        ), "Provide either --transformer_weights_dir or --transformer_weights"
        sd = load_file(weights_file, device="cpu")
        for k, v in sd.items():
            name = _strip_prefix(k)
            if _needed_key(name):
                state[name] = v
    return state


def _parse_step_from_filename(path: str) -> int:
    import os
    import re

    m = re.search(r"step_(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else 0


def _free_bridge_encoder(obj):
    """Force-free VRAM held by BridgeEncoderModel's LLM (Accelerate-sharded)."""
    try:
        if hasattr(obj, "model") and hasattr(obj.model, "bridge"):
            try:
                obj.model.bridge.to("cpu")
            except Exception:
                pass
        if hasattr(obj, "model") and hasattr(obj.model, "llm"):
            obj.model.llm = None
        if hasattr(obj, "llm"):
            obj.llm = None
    except Exception as e:  # pragma: no cover - best effort
        print(f"[free_bridge] warning: {e}")
    import gc  # local import to avoid global cuda init
    import torch

    torch.cuda.empty_cache()
    gc.collect()


def _release_training_gpu(*objs):
    for o in objs:
        try:
            getattr(o, "to", lambda *_a, **_k: None)("cpu")
        except Exception:
            pass
        try:
            if hasattr(o, "llm") and hasattr(o, "model"):
                _free_bridge_encoder(o)
        except Exception:
            pass
    import gc
    import torch

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()


def _rebuild_training(cfg, args):
    """Re-create WanModel + encoders on GPU and reload latest adapter + optimizer."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from wan.modules.model import WanModel

    torch.cuda.empty_cache()
    model = WanModel(**cfg).to(dev)
    filtered = _load_filtered_state(
        getattr(args, "transformer_weights_dir", None),
        getattr(args, "transformer_weights", None),
    )
    model.load_state_dict(filtered, strict=False)
    model.requires_grad_(False)

    def _inject_into(obj) -> dict:
        return inject_lora_kv(
            obj,
            blocks_attr=getattr(args, "blocks_attr", "blocks"),
            cross_attr=getattr(args, "cross_attr", "cross_attn"),
            rank=args.rank,
            alpha=args.alpha,
            dropout=0.0,
            blocks_range=None,
        )

    loras = _inject_into(model)
    if len(loras) == 0 and hasattr(model, "diffusion_model"):
        loras = _inject_into(model.diffusion_model)
    if getattr(args, "verify_injection", False) and len(loras) == 0:
        raise RuntimeError(
            "LoRA injection found zero targets. "
            "Check --blocks_attr/--cross_attr (defaults: blocks / cross_attn)."
        )

    # Reload latest adapter
    latest_adapters = sorted(
        glob(os.path.join(args.out_dir, "adapter_step_*.safetensors"))
    )
    if latest_adapters:
        load_peft_adapter(
            model,
            path=latest_adapters[-1],
            prefix=args.adapter_prefix,
            alpha=args.alpha,
        )

    # Rebuild encoders (back to training enc_device)
    enc_device = torch.device(args.enc_device if args.enc_device == "cpu" else "cuda")
    enc_dtype = (
        torch.bfloat16 if (args.bf16 and enc_device.type == "cuda") else torch.float32
    )
    teacher_enc = T5EncoderModel(
        text_len=getattr(model, "text_len", 512),
        checkpoint_path=args.t5_checkpoint,
        tokenizer_path=args.t5_tokenizer_dir or args.t5_tokenizer,
        device=enc_device,
        dtype=enc_dtype,
    )
    bridge_enc = BridgeEncoderModel(
        text_len=getattr(model, "text_len", 512),
        device=enc_device,
        dtype=enc_dtype,
    )

    # Rebuild optimizer + reload
    params = lora_parameters(model)
    if not params:
        raise RuntimeError(
            "No LoRA trainable parameters found (injection failed). "
            "Confirm the model has `blocks.*.cross_attn.{k,v}` linears."
        )
    optim = AdamW(params, lr=args.lr, weight_decay=0.0)
    latest_opt = sorted(glob(os.path.join(args.out_dir, "optimizer_step_*.pt")))
    if latest_opt:
        st = torch.load(latest_opt[-1], map_location="cpu")
        optim.load_state_dict(st.get("state_dict", st))

    return model, teacher_enc, bridge_enc, optim


def _run_sampler_batch(
    gen_script,
    prompts,
    ckpt_dir,
    save_dir,
    args,
    adapter_path=None,
    mode="lora_only",
):
    os.makedirs(save_dir, exist_ok=True)
    env = os.environ.copy()
    # Bridge on for sampling, regardless of training setting
    env["WAN_USE_BRIDGE"] = "1"
    env["WAN_BRIDGE_CKPT"] = args.bridge_ckpt
    env["WAN_BRIDGE_LLM_DIR"] = args.llm_dir
    env["WAN_BRIDGE_GLOBAL_SCALE"] = str(args.global_scale)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    t5flag = "" if args.sample_gpu_text_encoder else "--t5_cpu"
    for i, prompt in enumerate(prompts):
        base_out = os.path.join(save_dir, f"p{i:02d}_baseline.mp4")
        cmd_base = (
            f'python "{gen_script}" --task {args.sample_task} --ckpt_dir "{ckpt_dir}" '
            f"--size {args.sample_size} --frame_num {args.sample_frame_num} "
            f"--sample_steps {args.sample_steps} --sample_guide_scale {args.sample_guide_scale} "
            f"--base_seed {args.sample_seed} --prompt {shlex.quote(prompt)} "
            f'--save_file "{base_out}" --offload_model True {t5flag}'
        )
        if mode in ("both", "baseline_only"):
            print(f"[sample] baseline -> {base_out}")
            subprocess.run(shlex.split(cmd_base), env=env, check=True)

        if adapter_path and mode in ("both", "lora_only"):
            lora_out = base_out.replace("baseline.mp4", "lora.mp4")
            cmd_lora = (
                cmd_base
                + f' --lora_adapter_path "{adapter_path}"'
                + f" --lora_prefix {args.adapter_prefix} --lora_alpha {args.alpha} "
                + f' --save_file "{lora_out}"'
            )
            print(f"[sample] lora     -> {lora_out} (adapter={adapter_path})")
            subprocess.run(shlex.split(cmd_lora), env=env, check=True)


# -----------------------------------------------------------------------------
# Main training routine
# -----------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc_device = torch.device(
        args.enc_device
        if args.enc_device == "cpu" or torch.cuda.is_available()
        else "cpu"
    )
    enc_dtype = (
        torch.bfloat16 if (args.bf16 and enc_device.type == "cuda") else torch.float32
    )
    print(f"[enc] teacher on {enc_device}, bridge on {enc_device} (dtype={enc_dtype})")

    # --- load Wan model ---------------------------------------------------
    # reports dir (AGENTS.md policy)
    run_tag = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    reports_root = args.report_dir or f"reports/dit_lora_translator/{run_tag}"
    Path(reports_root).mkdir(parents=True, exist_ok=True)
    # persist run args
    with open(os.path.join(reports_root, "run_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)
    progress_path = os.path.join(reports_root, "progress.jsonl")

    def _append_progress(**kv):
        with open(progress_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(kv) + "\n")

    with open(args.transformer_config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    model = WanModel(**cfg).to(device)

    # Load sharded or single-file weights; keep only text_embed + cross_attn K/V (+ norms)
    filtered = _load_filtered_state(
        getattr(args, "transformer_weights_dir", None),
        getattr(args, "transformer_weights", None),
    )
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if unexpected:
        print(
            f"[warn] unexpected keys while loading filtered state: {len(unexpected)} (ignored)"
        )
    if missing:
        # It's okay to miss non-text/cross-attn params; we never use them here.
        print(f"[info] missing keys (not loaded): {len(missing)}")
    model.requires_grad_(False)

    # --- inject LoRA modules BEFORE loading any resume adapter -------------
    def _inject_into(obj) -> dict:
        return inject_lora_kv(
            obj,
            blocks_attr=getattr(args, "blocks_attr", "blocks"),
            cross_attr=getattr(args, "cross_attr", "cross_attn"),
            rank=args.rank,
            alpha=args.alpha,
            dropout=0.0,
            blocks_range=None,
        )

    loras = _inject_into(model)
    if len(loras) == 0 and hasattr(model, "diffusion_model"):
        # Some Wan builds hang the transformer under .diffusion_model
        loras = _inject_into(model.diffusion_model)

    print(f"[reports] writing to: {reports_root}")
    print(f"[lora] injected {len(loras)} modules:")
    for name in sorted(loras.keys())[:4]:
        print("   ", name)
    if len(loras) > 4:
        print("   ...")
    if getattr(args, "verify_injection", False) and len(loras) == 0:
        raise RuntimeError(
            "LoRA injection found zero targets. "
            "Check --blocks_attr/--cross_attr (defaults: blocks / cross_attn)."
        )

    # --- optionally load a resume adapter now that wrappers exist ----------
    if getattr(args, "resume_adapter", None):
        loaded = load_peft_adapter(
            model,
            path=args.resume_adapter,
            prefix=args.adapter_prefix,
            alpha=args.alpha,
        )
        print(
            f"[resume] loaded adapter: {args.resume_adapter}  (modules={len(loaded)})"
        )

    # --- encoders ---------------------------------------------------------
    # robust text length getter
    text_len = getattr(model, "config", None)
    text_len = getattr(text_len, "text_len", None) if text_len is not None else None
    if text_len is None:
        text_len = getattr(model, "text_len", 512)

    teacher_enc = T5EncoderModel(
        text_len=text_len,
        checkpoint_path=args.t5_checkpoint,
        tokenizer_path=args.t5_tokenizer_dir or args.t5_tokenizer,
        device=enc_device,
        dtype=enc_dtype,
    )

    # bridge env variables
    os.environ["WAN_BRIDGE_CKPT"] = args.bridge_ckpt
    os.environ["WAN_BRIDGE_LLM_DIR"] = args.llm_dir
    os.environ["WAN_BRIDGE_GLOBAL_SCALE"] = str(args.global_scale)
    bridge_enc = BridgeEncoderModel(
        text_len=text_len,
        device=enc_device,
        dtype=enc_dtype,
    )

    dataset = PromptDataset(
        args.prompts_file,
        jsonl_field=args.jsonl_field,
        shuffle=args.shuffle,
        max_samples=args.max_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=True,
        persistent_workers=True,
    )

    params = lora_parameters(model)
    if not params:
        raise RuntimeError(
            "No LoRA trainable parameters found (injection failed). "
            "Confirm the model has `blocks.*.cross_attn.{k,v}` linears."
        )
    optim = AdamW(params, lr=args.lr, weight_decay=0.0)
    optim.zero_grad(set_to_none=True)
    # Optionally resume optimizer
    if args.resume_optimizer and os.path.exists(args.resume_optimizer):
        state = torch.load(args.resume_optimizer, map_location="cpu")
        sdict = state.get("state_dict", state)  # accept raw or wrapped
        optim.load_state_dict(sdict)
        print(f"[resume] loaded optimizer: {args.resume_optimizer}")

    # Initial step (for numbering/checkpoint cadence)
    steps_total = int(args.start_step) if args.start_step is not None else 0
    if steps_total == 0 and args.resume_adapter:
        steps_total = _parse_step_from_filename(args.resume_adapter)
        if steps_total:
            print(f"[resume] inferred start_step={steps_total} from adapter filename")

    ema = None
    ema_beta = 0.98  # smoothing
    last_print = time.perf_counter()

    if args.sample_at_first and args.sample_prompts_file:
        # Optional: write a step0000000 adapter for consistent A/B layout
        zero_path = None
        if args.sample_lora_at_first:
            zero_path = os.path.join(args.out_dir, "adapter_step_0000000.safetensors")
            export_peft_adapter(model, zero_path, prefix=args.adapter_prefix)
        gen = os.path.join(Path(__file__).parent, "inference", "Wan2.2", "generate.py")
        with open(args.sample_prompts_file, "r", encoding="utf-8") as f:
            first_prompts = [ln.strip() for ln in f if ln.strip()][: args.sample_n]
        samp_dir = os.path.join(args.out_dir, "sample", "step_0000000")
        if args.sample_pause_training:
            _release_training_gpu(model, teacher_enc, bridge_enc, optim)
        _run_sampler_batch(
            gen_script=gen,
            prompts=first_prompts,
            ckpt_dir=(
                args.transformer_weights_dir
                or os.path.dirname(args.transformer_weights)
            ),
            save_dir=samp_dir,
            args=args,
            adapter_path=(zero_path if args.sample_lora_at_first else None),
            mode=("both" if args.sample_lora_at_first else "baseline_only"),
        )
        if args.sample_pause_training:
            model, teacher_enc, bridge_enc, optim = _rebuild_training(cfg, args)
            optim.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"epoch {epoch+1}/{args.epochs}", leave=True)
        for batch_prompts in pbar:
            if isinstance(batch_prompts, tuple):
                batch_prompts = list(batch_prompts)
            # teacher pass (LoRA disabled)
            set_lora_enabled(model, False)
            with torch.no_grad():
                teacher_tokens, mask = encode_teacher(
                    batch_prompts, teacher_enc, enc_device, text_len
                )
                teacher_tokens = teacher_tokens.to(device)
                mask = mask.to(device)
                teacher_context = model.text_embedding(teacher_tokens.float())
                k_t, v_t = compute_kv(model, teacher_context, args.use_normed_targets)

            # student pass (LoRA enabled)
            set_lora_enabled(model, True)
            bridge_tokens, mask_bridge = encode_bridge(
                batch_prompts, bridge_enc, enc_device, text_len
            )
            bridge_tokens = bridge_tokens.to(device)
            mask_bridge = mask_bridge.to(device)
            bridge_context = model.text_embedding(bridge_tokens.float())
            k_s, v_s = compute_kv(model, bridge_context, args.use_normed_targets)

            # loss
            loss = torch.tensor(0.0, device=device)
            # intersect masks (teacher & bridge) to avoid counting pad/trim mismatches
            mask_inter = (mask > 0) & (mask_bridge > 0)
            denom = mask_inter.sum().clamp_min(1.0)
            for kt, ks, vt, vs in zip(k_t, k_s, v_t, v_s):
                mse_k = ((ks - kt) ** 2) * mask_inter
                mse_v = ((vs - vt) ** 2) * mask_inter
                loss = loss + mse_k.sum() / denom + mse_v.sum() / denom

            loss_raw = loss
            loss = loss / args.grad_accum
            loss.backward()
            if ((steps_total + 1) % args.grad_accum) == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optim.step()
                optim.zero_grad(set_to_none=True)
            # progress
            loss_val = float(loss_raw.detach().item())
            loss = loss_raw
            ema = (
                (ema_beta * ema + (1 - ema_beta) * loss_val)
                if ema is not None
                else loss_val
            )
            steps_total += 1
            if steps_total % 50 == 0:
                now = time.perf_counter()
                dt = now - last_print
                it_per_s = 50.0 / max(dt, 1e-6)
                last_print = now
                # Tiny delta budget check
                tot_sq, base_sq = 0.0, 0.0
                with torch.no_grad():
                    for m in model.modules():
                        if hasattr(m, "base") and hasattr(m, "lora_A"):
                            dW = (m.lora_B.weight @ m.lora_A.weight) * m.scaling
                            tot_sq += float((dW**2).sum().item())
                            base_sq += float((m.base.weight**2).sum().item())
                delta_ratio = (tot_sq**0.5 / base_sq**0.5) if base_sq > 0 else 0.0
                print(
                    f"[{epoch+1}/{args.epochs}] step={steps_total} loss={loss_val:.1f} ema={ema:.1f} it/s={it_per_s:.2f} ΔW/W={delta_ratio:.3e}"
                )
            pbar.set_postfix_str(f"loss={loss_val:.5f}")
            _append_progress(epoch=epoch + 1, step=steps_total, loss=loss_val)

            # optional step checkpoint
            if args.save_every_steps and (steps_total % args.save_every_steps == 0):
                ck_step = os.path.join(
                    args.out_dir, f"adapter_step_{steps_total:07d}.safetensors"
                )
                export_peft_adapter(model, ck_step, prefix=args.adapter_prefix)
                opt_step = os.path.join(
                    args.out_dir, f"optimizer_step_{steps_total:07d}.pt"
                )
                torch.save({"state_dict": optim.state_dict()}, opt_step)
                shutil.copy2(
                    ck_step, os.path.join(args.out_dir, "adapter_latest.safetensors")
                )
                shutil.copy2(
                    opt_step, os.path.join(args.out_dir, "optimizer_latest.pt")
                )
                print(f"[ckpt] wrote {ck_step}")
                print(f"[ckpt] wrote {opt_step}")
                # Optional sampling (with pause/rebuild)
                if (
                    args.sample_every_steps > 0
                    and args.sample_prompts_file
                    and (steps_total % args.sample_every_steps == 0)
                ):
                    try:
                        gen = os.path.join(
                            Path(__file__).parent, "inference", "Wan2.2", "generate.py"
                        )
                        with open(args.sample_prompts_file, "r", encoding="utf-8") as f:
                            prompts = [ln.strip() for ln in f if ln.strip()][
                                : args.sample_n
                            ]
                        samp_dir = os.path.join(
                            args.out_dir, "sample", f"step_{steps_total:07d}"
                        )
                        if args.sample_pause_training:
                            _release_training_gpu(model, teacher_enc, bridge_enc, optim)
                        _run_sampler_batch(
                            gen_script=gen,
                            prompts=prompts,
                            ckpt_dir=(
                                args.transformer_weights_dir
                                or os.path.dirname(args.transformer_weights)
                            ),
                            save_dir=samp_dir,
                            args=args,
                            adapter_path=ck_step,
                            mode=args.sample_mode,
                        )
                    except Exception as e:
                        print(f"[sample] failed to auto-sample: {e}")
                    finally:
                        if args.sample_pause_training:
                            try:
                                _free_bridge_encoder(bridge_enc)
                            except Exception:
                                pass
                            del model, teacher_enc, bridge_enc, optim
                            torch.cuda.empty_cache()
                            gc.collect()
                            model, teacher_enc, bridge_enc, optim = _rebuild_training(
                                cfg, args
                            )

        print(f"Epoch {epoch+1}/{args.epochs} - loss {loss.item():.4f}")
        # epoch checkpoint
        if args.save_every_epochs and ((epoch + 1) % args.save_every_epochs == 0):
            ck_ep = os.path.join(
                args.out_dir, f"adapter_epoch{epoch+1:02d}.safetensors"
            )
            export_peft_adapter(model, ck_ep, prefix=args.adapter_prefix)
            shutil.copy2(
                ck_ep, os.path.join(args.out_dir, "adapter_latest.safetensors")
            )

    # save adapter + optimizer
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"adapter_step_{steps_total:07d}.safetensors")
    export_peft_adapter(
        model, out_path, prefix=getattr(args, "adapter_prefix", "diffusion_model.")
    )
    opt_path = os.path.join(args.out_dir, f"optimizer_step_{steps_total:07d}.pt")
    torch.save({"state_dict": optim.state_dict()}, opt_path)
    print(f"[ckpt] wrote {out_path}")
    print(f"[ckpt] wrote {opt_path}")
    # final summary
    with open(os.path.join(reports_root, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "steps_total": steps_total,
                "final_loss": float(loss.detach().item()),
                "out_adapter": out_path,
            },
            f,
            indent=2,
        )
    with open(os.path.join(reports_root, "report.md"), "w", encoding="utf-8") as f:
        f.write("# KV‑LoRA Phase‑1 run\n\n")
        f.write(f"- out_dir: `{args.out_dir}`\n")
        f.write(f"- adapter: `{out_path}`\n")
        f.write(f"- steps_total: `{steps_total}`\n")
        f.write(f"- final_loss: `{float(loss.detach().item()):.6f}`\n")


# -----------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="KV LoRA distillation trainer")
    p.add_argument("--transformer_config", type=str, required=True)
    # Accept either a directory with shards+index or a single .safetensors file
    p.add_argument(
        "--transformer_weights_dir",
        type=str,
        default=None,
        help="Directory containing diffusion_pytorch_model-*.safetensors (Wan 5B).",
    )
    p.add_argument(
        "--transformer_weights",
        type=str,
        default=None,
        help="Single .safetensors file (optional if --transformer_weights_dir is set).",
    )
    p.add_argument("--t5_checkpoint", type=str, required=True)
    p.add_argument(
        "--t5_tokenizer_dir",
        type=str,
        default=None,
        help="Local tokenizer directory (preferred).",
    )
    p.add_argument(
        "--t5_tokenizer",
        type=str,
        default=None,
        help="HF name or path (fallback if *_dir not provided).",
    )
    p.add_argument("--prompts_file", type=str, required=True)
    p.add_argument(
        "--jsonl_field",
        type=str,
        default="caption",
        help="Field name for JSONL files (default: caption).",
    )
    p.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional subsample cap for quick runs.",
    )
    p.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle prompts before subsampling.",
    )
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--bridge_ckpt", type=str, required=True)
    p.add_argument("--llm_dir", type=str, required=True)
    p.add_argument("--global_scale", type=float, default=1.0)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=float, default=32.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument(
        "--grad_accum",
        type=int,
        default=1,
        help="Gradient accumulation steps.",
    )
    p.add_argument("--use_normed_targets", action="store_true")
    p.add_argument(
        "--bf16", action="store_true", help="Use bfloat16 precision for encoders"
    )
    p.add_argument(
        "--enc_device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for teacher/bridge encoders during TRAINING (CUDA is fastest).",
    )
    # Resume options
    p.add_argument(
        "--resume_adapter",
        type=str,
        default=None,
        help="Path to adapter_step_*.safetensors to resume from.",
    )
    p.add_argument(
        "--resume_optimizer",
        type=str,
        default=None,
        help="Path to optimizer_step_*.pt to resume from (optional).",
    )
    p.add_argument(
        "--start_step",
        type=int,
        default=None,
        help="Global step to start numbering at (inferred from adapter filename if omitted).",
    )
    p.add_argument(
        "--adapter_prefix",
        type=str,
        default="diffusion_model.",
        help="Key prefix when exporting LoRA adapter (e.g., diffusion_model. or transformer.)",
    )
    p.add_argument(
        "--blocks_attr",
        type=str,
        default="blocks",
        help="Model attribute containing transformer blocks (default: blocks).",
    )
    p.add_argument(
        "--cross_attr",
        type=str,
        default="cross_attn",
        help="Attribute name for cross-attention modules (default: cross_attn).",
    )
    p.add_argument(
        "--verify_injection",
        action="store_true",
        help="Error if LoRA injection finds zero target modules.",
    )
    p.add_argument(
        "--save_every_steps",
        type=int,
        default=0,
        help="If >0, write adapter_step_*.safetensors every N steps.",
    )
    # On-save sampling
    p.add_argument(
        "--sample_every_steps",
        type=int,
        default=0,
        help="0=off; sample every N steps when a checkpoint is saved",
    )
    p.add_argument(
        "--sample_prompts_file",
        type=str,
        default=None,
        help="Text file of prompts; first K are used per sample",
    )
    p.add_argument(
        "--sample_n",
        type=int,
        default=2,
        help="How many prompts from the file to sample each time",
    )
    p.add_argument("--sample_task", type=str, default="ti2v-5B")
    p.add_argument("--sample_size", type=str, default="768*432")
    p.add_argument("--sample_steps", type=int, default=20)
    p.add_argument("--sample_guide_scale", type=float, default=5.5)
    p.add_argument("--sample_frame_num", type=int, default=17)
    p.add_argument("--sample_seed", type=int, default=12345)
    p.add_argument(
        "--sample_mode",
        type=str,
        default="lora_only",
        choices=["both", "lora_only", "baseline_only"],
        help="What to render at sampling points. Default lora_only (after the first baseline).",
    )
    p.add_argument(
        "--sample_at_first",
        action="store_true",
        help="Run baseline (and optional LoRA) samples before the first step.",
    )
    p.add_argument(
        "--sample_lora_at_first",
        action="store_true",
        help="If set, also sample with the step0000000 LoRA (no-op but keeps layout consistent).",
    )
    p.add_argument(
        "--sample_pause_training",
        action="store_true",
        help="Free GPU memory before sampling, then rebuild training state on GPU.",
    )
    p.add_argument(
        "--sample_gpu_text_encoder",
        action="store_true",
        help="Place text encoder/LLM on GPU during sampling (faster but heavier). If false, we pass --t5_cpu to sampler.",
    )
    p.add_argument(
        "--save_every_epochs",
        type=int,
        default=1,
        help="Write adapter_epoch*.safetensors every N epochs (0 to disable).",
    )
    p.add_argument(
        "--report_dir",
        type=str,
        default=None,
        help=(
            "Optional path for reports/; defaults to reports/dit_lora_translator/<timestamp>."
        ),
    )
    return p


if __name__ == "__main__":
    parser = build_argparser()
    train(parser.parse_args())
