#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage-2 LoRA trainer for Wan DiT cross-attention (Q/K/V/O) on captioned image+video data.

- Continues from a KV-only Stage-1 adapter (optional --resume_adapter).
- Expands scope to train Q/O (and optionally keep KV trainable) using a diffusion loss.
- Reads your dataset layout:
    /data/image/480p/*.png + .txt
    /data/image/720p/*.png + .txt
    /data/video/480p/{17,33,49,65,81}frames/*.mp4 + .txt
    /data/video/720p/{17,33,49,65,81}frames/*.mp4 + .txt
"""

from __future__ import annotations
import glob
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from kv_lora_inject import (
    inject_lora_kv,
    set_lora_enabled,
    lora_parameters,
    load_peft_adapter,
    export_peft_adapter,
    LoRALinear,
)
from mixed_media import MixedCaptioned, BucketBatchSampler

# --- Configuration / args
import argparse


def build_argparser():
    p = argparse.ArgumentParser("Stage-2 Q/K/V/O LoRA trainer for Wan 2.2")
    # Wan backbone
    p.add_argument(
        "--transformer_config",
        type=str,
        required=True,
        help="Wan2.2 config json (same used in Stage-1).",
    )
    p.add_argument(
        "--transformer_weights_dir",
        type=str,
        default=None,
        help="Dir with diffusion_pytorch_model-*.safetensors",
    )
    p.add_argument(
        "--transformer_weights",
        type=str,
        default=None,
        help="Single .safetensors (optional if --transformer_weights_dir set).",
    )
    p.add_argument("--text_len", type=int, default=512)

    # Bridge (Qwen -> Wan) encoder
    p.add_argument(
        "--bridge_ckpt",
        type=str,
        required=True,
        help="Your trained Bridge checkpoint (.pt or .safetensors) incl. affine.",
    )
    p.add_argument(
        "--llm_dir",
        type=str,
        required=True,
        help="HF dir for Qwen2.5-VL (or your LLM).",
    )
    p.add_argument(
        "--global_scale",
        type=float,
        default=1.0,
        help="Optional scalar applied by Bridge (kept for continuity with Stage-1).",
    )

    # LoRA
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=float, default=32.0)
    p.add_argument(
        "--targets",
        type=str,
        default="q,k,v,o",
        help="Comma list from {q,k,v,o}. Default: q,k,v,o",
    )
    p.add_argument(
        "--freeze_kv", action="store_true", help="Freeze K/V LoRA (train Q/O only)."
    )
    p.add_argument(
        "--resume_adapter",
        type=str,
        default=None,
        help="Path to Stage-1 adapter_model.safetensors to initialize from.",
    )

    # Data
    p.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root folder (contains /image, /video).",
    )
    p.add_argument(
        "--resolutions",
        type=str,
        default="480,720",
        help="Comma list of resolutions to include (e.g. 480,720).",
    )
    p.add_argument(
        "--frames",
        type=str,
        default="1,17",
        help="Which frame-count buckets to include (must include 1 for images).",
    )
    p.add_argument(
        "--base_batch",
        type=int,
        default=16,
        help="Reference batch size for 480p images; other buckets scale from this.",
    )
    p.add_argument("--workers", type=int, default=4, help="DataLoader workers.")

    # Training
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--clip_grad", type=float, default=1.0)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    # Saving / logging
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument(
        "--adapter_prefix",
        type=str,
        default="diffusion_model.",
        help="Key prefix to export adapter (e.g. diffusion_model. or transformer.)",
    )
    return p


# ---- Wan imports
def _import_wan22():
    # mirror Stage-1 import path
    WAN22_DIR = Path(__file__).resolve().parent / "inference" / "Wan2.2"
    if not WAN22_DIR.exists():
        raise FileNotFoundError(f"Expected Wan2.2 at {WAN22_DIR}")
    sys.path.append(str(WAN22_DIR))
    from wan.modules.model import WanModel  # type: ignore
    from wan.modules.t5_bridge import BridgeEncoderModel  # type: ignore

    return WanModel, BridgeEncoderModel


# ---- Text (Bridge) helpers


def pad_stack(
    tokens: List[torch.Tensor], max_len: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    b = len(tokens)
    d = tokens[0].shape[-1]
    out = torch.zeros(b, max_len, d, device=tokens[0].device, dtype=tokens[0].dtype)
    mask = torch.zeros(b, max_len, 1, device=tokens[0].device, dtype=torch.float32)
    for i, t in enumerate(tokens):
        L = min(t.shape[0], max_len)
        out[i, :L] = t[:L]
        mask[i, :L, 0] = 1.0
    return out, mask


# ---- Model I/O helpers (Wan)


def _load_filtered_state(
    weights_dir: Optional[str], weights_file: Optional[str]
) -> Dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    if weights_dir:
        # merge shards
        state = {}
        for f in sorted(
            glob.glob(
                os.path.join(weights_dir, "diffusion_pytorch_model-*.safetensors")
            )
        ):
            state.update(load_file(f))
        return state
    if weights_file:
        return load_file(weights_file)
    raise ValueError("Provide --transformer_weights_dir or --transformer_weights")


def _apply_filtered_state(model: nn.Module, state: Dict[str, torch.Tensor]) -> None:
    # Load only the keys we need (as Stage-1 did): text_embedding + attn projections.
    keep = {}
    for k, v in state.items():
        if (".cross_attn." in k) or k.startswith("text_embedding"):
            keep[k] = v
    missing, unexpected = model.load_state_dict(keep, strict=False)
    if False:  # debug
        print("missing:", missing)
        print("unexpected:", unexpected)


# graceful attempt to call DiT forward under various repos
def call_dit(
    model: nn.Module,
    x_t: torch.Tensor,
    t: torch.Tensor,
    context: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> torch.Tensor:
    # Common signatures:
    # y/context mask names vary a lot; try a few.
    try:
        return model(x_t, t, context, mask=mask)
    except Exception:
        pass
    try:
        return model(x_t, t, context=context, mask=mask)
    except Exception:
        pass
    try:
        return model(x_t, t, y=context, y_mask=mask)
    except Exception:
        pass
    try:
        return model.diffusion_model(x_t, t, context, mask)  # Wan style
    except Exception:
        pass
    # As a last resort:
    return model(x_t, t, context)


def encode_video_to_latents(model, video_bthwc: torch.Tensor) -> torch.Tensor:
    """
    video_bthwc: [B, T, H, W, 3] in [-1,1]
    Returns latents with same B,T,H',W',C' as Wan expects.
    """
    # Try common Wan VAEs: model.vae.encode_video or model.vae.encode
    vae = getattr(model, "vae", None)
    if vae is None:
        raise RuntimeError("WanModel has no .vae; please adapt encode path.")

    # Prefer a vectorized encode if available
    if hasattr(vae, "encode_video"):
        return vae.encode_video(video_bthwc)  # type: ignore

    if hasattr(vae, "encode"):
        B, T, H, W, C = video_bthwc.shape
        outs = []
        for b in range(B):
            per = []
            for t in range(T):
                img = video_bthwc[b, t].permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
                lat = vae.encode(img)  # [1,C',H',W'] (assumed)
                per.append(lat)
            outs.append(torch.stack(per, dim=1))  # [1,T,C',H',W']
        return torch.cat(outs, dim=0)  # [B,T,C',H',W']
    raise RuntimeError("Don't know how to encode latents; add your repo's method here.")


def make_velocity_training_pair(
    x1: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Flow-matching style targets:
      x_t = t*x1 + (1-t)*x0,  v = x1 - x0,  with x0 ~ N(0,I), t~U(0,1)
    Shapes: x1 [B,T,C,H,W] (latents). Returns x_t, v, t
    """
    B = x1.shape[0]
    x0 = torch.randn_like(x1)
    t = torch.rand(B, device=x1.device)
    # reshape t for broadcasting
    t_ = t.view(B, 1, 1, 1, 1)
    x_t = t_ * x1 + (1.0 - t_) * x0
    v = x1 - x0
    return x_t, v, t


def count_params(params: List[torch.nn.Parameter]) -> int:
    return sum(p.numel() for p in params)


# ---- Main train


def main(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    WanModel, BridgeEncoderModel = _import_wan22()

    # Env toggles so Wan code uses the Bridge implementation
    os.environ["WAN_USE_BRIDGE"] = "1"
    os.environ["WAN_BRIDGE_CKPT"] = args.bridge_ckpt
    os.environ["WAN_BRIDGE_LLM_DIR"] = args.llm_dir
    os.environ["WAN_BRIDGE_DTYPE"] = "bf16" if getattr(args, "bf16", False) else "fp16"
    os.environ["WAN_BRIDGE_GLOBAL_SCALE"] = str(getattr(args, "global_scale", 1.0))
    os.environ["WAN_BRIDGE_FORCE_VL"] = "1"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (args.bf16 and device.type == "cuda") else torch.float32

    # Load Wan model
    with open(args.transformer_config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    model = WanModel(**cfg).to(device)
    state = _load_filtered_state(args.transformer_weights_dir, args.transformer_weights)
    _apply_filtered_state(model, state)
    model.requires_grad_(False)

    # Inject LoRA into cross-attn targets (default q/k/v/o)
    targets = tuple([s.strip() for s in args.targets.split(",") if s.strip()])
    inject_lora_kv(
        model,
        blocks_attr="blocks",
        cross_attr="cross_attn",
        targets=targets,
        rank=args.rank,
        alpha=args.alpha,
        dropout=0.0,
        blocks_range=None,
    )

    # Resume from Stage-1 adapter (KV only) if provided
    if args.resume_adapter and os.path.exists(args.resume_adapter):
        load_peft_adapter(
            model, args.resume_adapter, prefix=args.adapter_prefix, alpha=args.alpha
        )
        print(f"[resume] Loaded Stage-1 adapter from: {args.resume_adapter}")

    # Optionally freeze K/V LoRA parameters (train Q/O only)
    if args.freeze_kv:
        for name, module in model.named_modules():
            if isinstance(module, LoRALinear):
                if name.endswith(".k") or name.endswith(".v"):
                    for p in module.trainable_parameters():
                        p.requires_grad_(False)
    # gather params
    params = [p for p in lora_parameters(model) if p.requires_grad]
    n_params = count_params(params)
    print(
        f"[lora] Trainable LoRA params: {n_params/1e6:.3f} M across targets={targets} (freeze_kv={args.freeze_kv})"
    )

    # Text encoder (Bridge) runs inside Wan via environment, but we use it here to produce tokens
    # so we can control masking & pass through model.text_embedding.
    bridge = BridgeEncoderModel(text_len=args.text_len, device=device, dtype=dtype)

    # Data
    frames = [int(x) for x in args.frames.split(",") if x.strip()]
    reses = [int(x) for x in args.resolutions.split(",") if x.strip()]
    dataset = MixedCaptioned(
        root=args.data_root,
        frames_options=frames,
        resolutions=reses,
        center_crop=True,
        seed=args.seed,
    )
    if len(dataset) == 0:
        raise RuntimeError("No samples found under the given config.")
    print(f"[data] total samples: {len(dataset)}")

    def guess_bs(res, T, base=16):
        scale = (1.0 + T / 4.0) * (res / 16.0) ** 2
        ref = (1.0 + 1 / 4.0) * (480 / 16.0) ** 2
        return max(1, int(base * ref / scale))

    batch_sizes = {
        (r, f): guess_bs(r, f, base=args.base_batch) for r in reses for f in frames
    }
    sampler = BucketBatchSampler(dataset, batch_sizes, seed=args.seed)
    loader = DataLoader(
        dataset, batch_sampler=sampler, num_workers=args.workers, pin_memory=True
    )

    # Optimizer
    opt = torch.optim.AdamW(
        params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95)
    )
    _ = torch.cuda.amp.GradScaler(enabled=False)  # bf16 uses autocast without scaler
    model.train()
    set_lora_enabled(model, True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    running = 0.0
    pbar = tqdm(total=args.steps, desc="train-stage2")
    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            vids = batch["pixel"].permute(0, 1, 3, 4, 2).to(device)  # [B,T,H,W,3]
            caps: List[str] = batch["caption"]

            # text → tokens → context
            tokens_list = bridge(caps, device)  # list of [L_i, d]
            tokens, mask = pad_stack(tokens_list, args.text_len)  # [B,L,d], [B,L,1]
            tokens = tokens.to(device=device, dtype=dtype)
            mask = mask.to(device=device, dtype=torch.float32)
            context = model.text_embedding(tokens.float())  # [B,L,d_model]

            # encode latents
            vids = vids.to(dtype=torch.float32)  # VAE usually wants fp32
            x1 = encode_video_to_latents(model, vids)  # shape [B,T,C',H',W']
            x1 = x1.to(device=device, dtype=dtype)

            # flow-matching pair
            x_t, v, t = make_velocity_training_pair(x1)  # x_t/v are dtype=dtype
            # run DiT
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=(dtype == torch.bfloat16),
            ):
                pred = call_dit(model, x_t, t, context, mask)  # predict velocity
                loss = F.mse_loss(pred, v)

            (loss / args.grad_accum).backward()
            running += loss.item()

            if (step + 1) % args.grad_accum == 0:
                if args.clip_grad is not None and args.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(params, args.clip_grad)
                opt.step()
                opt.zero_grad(set_to_none=True)

            step += 1
            pbar.update(1)
            if step % args.log_every == 0:
                avg = running / args.log_every
                pbar.set_postfix(loss=f"{avg:.4f}")
                running = 0.0

            if step % args.save_every == 0 or step == args.steps:
                save_path = out_dir / f"adapter_step_{step:07d}.safetensors"
                export_peft_adapter(model, str(save_path), prefix=args.adapter_prefix)
                # also export rolling latest
                export_peft_adapter(
                    model,
                    str(out_dir / "adapter_model.safetensors"),
                    prefix=args.adapter_prefix,
                )
                print(f"[save] {save_path}")

    # final save
    export_peft_adapter(
        model, str(out_dir / "adapter_model.safetensors"), prefix=args.adapter_prefix
    )
    print("[done] Saved final adapter_model.safetensors")


if __name__ == "__main__":
    args = build_argparser().parse_args()
    main(args)
