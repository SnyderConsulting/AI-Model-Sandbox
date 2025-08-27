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

import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Tuple
from datetime import datetime, timezone
import shutil
from tqdm import tqdm

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from safetensors.torch import load_file, safe_open

from kv_lora_inject import (
    export_peft_adapter,
    inject_lora_kv,
    lora_parameters,
    set_lora_enabled,
)

# Wan imports live under the inference package.
import sys

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


# -----------------------------------------------------------------------------
# Main training routine
# -----------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    steps_total = 0

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

    # inject LoRA modules
    loras = inject_lora_kv(model, rank=args.rank, alpha=args.alpha)
    # Preflight: list injected modules
    print(f"[reports] writing to: {reports_root}")
    print(f"[lora] injected {len(loras)} modules:")
    for name in sorted(loras.keys())[:4]:
        print("   ", name)
    if len(loras) > 4:
        print("   ...")

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
        device=device,
        dtype=torch.bfloat16 if args.bf16 else torch.float32,
    )

    # bridge env variables
    os.environ["WAN_BRIDGE_CKPT"] = args.bridge_ckpt
    os.environ["WAN_BRIDGE_LLM_DIR"] = args.llm_dir
    os.environ["WAN_BRIDGE_GLOBAL_SCALE"] = str(args.global_scale)
    bridge_enc = BridgeEncoderModel(
        text_len=text_len,
        device=device,
        dtype=torch.bfloat16 if args.bf16 else torch.float32,
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
    optim = AdamW(params, lr=args.lr, weight_decay=0.0)

    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"epoch {epoch+1}/{args.epochs}", leave=True)
        for batch_prompts in pbar:
            if isinstance(batch_prompts, tuple):
                batch_prompts = list(batch_prompts)
            # teacher pass (LoRA disabled)
            set_lora_enabled(model, False)
            with torch.no_grad():
                teacher_tokens, mask = encode_teacher(
                    batch_prompts, teacher_enc, device, text_len
                )
                teacher_context = model.text_embedding(teacher_tokens.float())
                k_t, v_t = compute_kv(model, teacher_context, args.use_normed_targets)

            # student pass (LoRA enabled)
            set_lora_enabled(model, True)
            bridge_tokens, mask_bridge = encode_bridge(
                batch_prompts, bridge_enc, device, text_len
            )
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

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optim.step()
            steps_total += 1
            # progress
            loss_val = float(loss.detach().item())
            pbar.set_postfix_str(f"loss={loss_val:.5f}")
            _append_progress(epoch=epoch + 1, step=steps_total, loss=loss_val)

            # optional step checkpoint
            if args.save_every_steps and (steps_total % args.save_every_steps == 0):
                ck_step = os.path.join(
                    args.out_dir, f"adapter_step_{steps_total:07d}.safetensors"
                )
                export_peft_adapter(model, ck_step, prefix=args.adapter_prefix)
                shutil.copy2(
                    ck_step, os.path.join(args.out_dir, "adapter_latest.safetensors")
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

    # save adapter
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(args.out_dir, "adapter_model.safetensors")
    # If your loader expects a different prefix, change it here (e.g., "transformer.")
    export_peft_adapter(
        model, out_path, prefix=getattr(args, "adapter_prefix", "diffusion_model.")
    )
    print(f"Saved adapter to {out_path}")
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
        "--adapter_prefix",
        type=str,
        default="diffusion_model.",
        help="Key prefix when exporting LoRA adapter (e.g., diffusion_model. or transformer.)",
    )
    p.add_argument(
        "--save_every_steps",
        type=int,
        default=0,
        help="If >0, write adapter_step_*.safetensors every N steps.",
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
