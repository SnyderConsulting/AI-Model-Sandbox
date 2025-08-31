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
    inject_lora_visual,
    set_lora_enabled,
    load_peft_adapter,
    export_peft_adapter,
    LoRALinear,
)
from mixed_media import MixedCaptioned, BucketBatchSampler
from wan_vae_loader import load_wan_vae

# --- Configuration / args
import argparse


def _tstats(x: torch.Tensor):
    x = x.detach().float()
    return {
        "shape": tuple(x.shape),
        "mean": float(x.mean()),
        "std": float(x.std(unbiased=False)),
        "absmean": float(x.abs().mean()),
        "min": float(x.min()),
        "max": float(x.max()),
        "nonzero_%": float((x != 0).float().mean()) * 100.0,
    }


def debug_step(
    step,
    vids=None,
    x1=None,
    x_t=None,
    v=None,
    pred=None,
    loss=None,
    t=None,
    model=None,
    grad_accum=1,
    every=200,
):
    if step % every != 0:
        return
    print(f"\n[debug] step={step} grad_accum={grad_accum}")
    if vids is not None:
        print("vids     ", _tstats(vids))
    if x1 is not None:
        print("latents x1", _tstats(x1))
    if x_t is not None:
        print("x_t      ", _tstats(x_t))
    if v is not None:
        print("v        ", _tstats(v))
    if pred is not None:
        print("pred     ", _tstats(pred))
    if t is not None:
        try:
            tu = torch.unique(t.detach().to(torch.int64), sorted=True)
            print("t unique (first 8):", tu[:8].tolist(), " total:", int(tu.numel()))
            print(
                "[t] mean=",
                float(t.detach().float().mean()),
                "std=",
                float(t.detach().float().std(unbiased=False)),
                "min=",
                float(t.detach().float().min()),
                "max=",
                float(t.detach().float().max()),
            )
        except Exception:
            pass
    if (pred is not None) and (v is not None):
        raw_mse = F.mse_loss(pred.float(), v.float()).item()
        print(
            f"raw_mse(fp32)={raw_mse:.8f}  scaled(raw_mse/accum)={raw_mse/grad_accum:.8f}"
        )
    if loss is not None and torch.is_tensor(loss):
        print(f"reported loss tensor={float(loss):.8f}")
    if model is not None:
        gmeans = []
        for n, p in model.named_parameters():
            if p.requires_grad and (p.grad is not None):
                gmeans.append(p.grad.detach().abs().mean().item())
        if gmeans:
            print(
                f"grad_abs_mean={sum(gmeans)/len(gmeans):.8e} over {len(gmeans)} trainable params"
            )


def setup_amp_and_models(args, model, vae):
    want_bf16 = (
        bool(getattr(args, "bf16", False))
        or os.environ.get("WAN_BRIDGE_DTYPE", "").lower() == "bf16"
    )
    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    amp_dtype = torch.bfloat16 if (want_bf16 and bf16_ok) else torch.float16
    if want_bf16 and not bf16_ok:
        print("[warn] BF16 requested but not supported on this GPU; switching to FP16.")
        os.environ["WAN_BRIDGE_DTYPE"] = "fp16"

    # Cast models to the chosen compute dtype
    model.to(dtype=amp_dtype)
    vae.to(dtype=amp_dtype)
    if hasattr(vae, "dtype"):
        try:
            vae.dtype = amp_dtype  # for VAEs that read self.dtype inside autocast
        except Exception:
            pass

    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))
    return amp_dtype, scaler


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
    p.add_argument(
        "--vae_dir",
        type=str,
        default=None,
        help="Folder that contains 'wan/' (e.g., .../inference/Wan2.2).",
    )
    p.add_argument(
        "--vae_module",
        type=str,
        default=None,
        help="Python module for Wan VAE, e.g. 'wan.modules.vae3d' or 'wan.modules.vae'.",
    )
    p.add_argument(
        "--vae_ckpt",
        type=str,
        default=None,
        help="Path to Wan VAE weights (.pth or .safetensors).",
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
        "--freeze_cross",
        action="store_true",
        help="Freeze cross-attn LoRA so only visual stream adapts.",
    )
    p.add_argument(
        "--lr_visual",
        type=float,
        default=None,
        help="Learning rate for visual LoRA layers (self-attn/ffn).",
    )
    p.add_argument(
        "--enable_self_attn",
        action="store_true",
        help="Inject LoRA into blocks.*.attn.{q,k,v,o}",
    )
    p.add_argument(
        "--enable_ffn",
        action="store_true",
        help="Inject LoRA into blocks.*.ffn linears",
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
    # OLD: only text_embedding + cross_attn
    # NEW: at least also load head + patch embedding
    keep = {}
    for k, v in state.items():
        if (
            k.startswith("text_embedding")
            or ".cross_attn." in k
            or k.startswith("patch_embedding")
            or k.startswith("head.")
        ):
            keep[k] = v
    missing, unexpected = model.load_state_dict(keep, strict=False)
    print(
        f"[load] loaded {len(state)} keys; missing={len(missing)} unexpected={len(unexpected)}"
    )
    print("[head abs-mean]", float(model.head.head.weight.abs().mean()))


def call_dit(model, x_t, t, context, mask=None, debug: bool = False):
    """
    Normalizes inputs for WanModel.forward, computes seq_len from Conv3d patcher,
    picks the correct prediction tensor from outputs, and returns a Tensor with
    the same layout as x_t.
    """
    pe = getattr(model, "patch_embedding", None)
    assert pe is not None, "model.patch_embedding (Conv3d) not found"
    c_in = int(getattr(pe, "in_channels", 48))

    # ---- 1) Normalize x_t to [B,C,T,H,W], remember original layout ----
    if x_t.dim() == 5:
        if x_t.size(1) == c_in:  # [B,C,T,H,W]
            B, C, T, H, W = x_t.shape
            x_bcthw, orig = x_t, "BCTHW"
        elif x_t.size(2) == c_in:  # [B,T,C,H,W] -> [B,C,T,H,W]
            B, T, C, H, W = x_t.shape
            x_bcthw, orig = x_t.permute(0, 2, 1, 3, 4).contiguous(), "BTCHW"
        else:
            # Heuristic fallback
            if x_t.size(1) in (16, 32, 48):
                B, C, T, H, W = x_t.shape
                x_bcthw, orig = x_t, "BCTHW"
            else:
                B, T, C, H, W = x_t.shape
                x_bcthw, orig = x_t.permute(0, 2, 1, 3, 4).contiguous(), "BTCHW"
    elif x_t.dim() == 4:  # [B,C,H,W] -> [B,C,1,H,W]
        B, C, H, W = x_t.shape
        T = 1
        x_bcthw, orig = x_t.unsqueeze(2).contiguous(), "BCHW"
    else:
        raise AssertionError(f"Unexpected x_t shape {tuple(x_t.shape)}")

    assert x_bcthw.size(1) == c_in, f"Expected channels={c_in}, got {x_bcthw.size(1)}"
    device = next(model.parameters()).device
    mdtype = next(model.parameters()).dtype
    x_bcthw = x_bcthw.to(device=device, dtype=mdtype)

    # ---- 2) seq_len from Conv3d patcher (exact) ----
    def _as3(v, default):
        if isinstance(v, int):
            return (v, v, v)
        if isinstance(v, (list, tuple)) and len(v) == 3:
            return tuple(int(x) for x in v)
        return default

    kT, kH, kW = _as3(getattr(pe, "kernel_size", (1, 2, 2)), (1, 2, 2))
    sT, sH, sW = _as3(getattr(pe, "stride", (1, 2, 2)), (1, 2, 2))
    pT, pH, pW = _as3(getattr(pe, "padding", (0, 0, 0)), (0, 0, 0))
    dT, dH, dW = _as3(getattr(pe, "dilation", (1, 1, 1)), (1, 1, 1))

    def _conv_out(n, k, s, p, d):
        return (n + 2 * p - d * (k - 1) - 1) // s + 1

    T_out = _conv_out(T, kT, sT, pT, dT)
    H_out = _conv_out(H, kH, sH, pH, dH)
    W_out = _conv_out(W, kW, sW, pW, dW)
    seq_len = int(T_out * H_out * W_out)
    assert seq_len > 0

    # ---- 3) Build forward inputs ----
    x_list = [x_bcthw[i] for i in range(B)]  # each [C,T,H,W]

    t = t.reshape(-1).to(device)
    if t.numel() == 1:
        t = t.repeat(B)
    elif t.numel() != B:
        t = t[:1].repeat(B)

    # context -> List[Tensor [L, D_exp]]
    te = getattr(model, "text_embedding", None)
    D_exp = None
    if te is not None:
        for m in te.modules():
            if isinstance(m, torch.nn.Linear):
                D_exp = int(m.in_features)
                break
    if D_exp is None:
        D_exp = 4096

    def _pad_trunc_lastdim(c: torch.Tensor) -> torch.Tensor:
        c = c.to(device)
        d = c.size(-1)
        if d == D_exp:
            return c
        if d < D_exp:
            return F.pad(c, (0, D_exp - d))
        return c[..., :D_exp]

    if isinstance(context, torch.Tensor):
        if context.dim() == 3:  # [B,L,D]
            context_list = [
                _pad_trunc_lastdim(context[i]) for i in range(min(B, context.size(0)))
            ]
            if len(context_list) < B:
                context_list += [context_list[0]] * (B - len(context_list))
        elif context.dim() == 2:  # [L,D]
            c = _pad_trunc_lastdim(context)
            context_list = [c for _ in range(B)]
        else:
            raise AssertionError(
                f"Unexpected context tensor shape {tuple(context.shape)}"
            )
    else:
        tmp = []
        for c in context:
            c = torch.as_tensor(c, device=device)
            if c.dim() != 2:
                raise AssertionError(
                    f"Each context item must be [L,D], got {tuple(c.shape)}"
                )
            tmp.append(_pad_trunc_lastdim(c))
        if len(tmp) != B:
            tmp = [tmp[0] for _ in range(B)]
        context_list = tmp

    context_list = [c.to(dtype=mdtype) for c in context_list]

    # ---- 4) Forward ----
    out = model(x_list, t, context_list, seq_len=seq_len)

    # ---- 5) Extract prediction tensor and stack correctly ----
    def _ensure_4d_sample(x):
        """Per-sample: want [C,T,H,W]. If [C,H,W], add T=1; if [1,C,T,H,W], squeeze B."""
        if x.dim() == 4:  # [C,T,H,W]
            return x
        if x.dim() == 3:  # [C,H,W] -> [C,1,H,W]
            return x.unsqueeze(1)
        if x.dim() == 5 and x.size(0) == 1 and x.size(1) == c_in:  # [1,C,T,H,W]
            return x.squeeze(0)
        raise AssertionError(
            f"Unexpected per-sample tensor rank {x.dim()} in model output"
        )

    def _pick_best_sample_tensor(container):
        # Flatten one level and pick best by abs-mean, preferring channel match.
        cands = []

        def _collect(z):
            if torch.is_tensor(z):
                cands.append(z)
            elif isinstance(z, (list, tuple)):
                for y in z:
                    _collect(y)

        _collect(container)
        assert cands, "No tensor candidate found in model outputs"

        best, best_score = None, -1.0
        for tns in cands:
            # Score
            score = float(tns.detach().abs().mean())
            # Bonus if the first dim (or second when batched sample) equals c_in
            bonus = 0.0
            if tns.dim() == 4 and tns.size(0) == c_in:
                bonus = 10.0
            if tns.dim() == 5 and tns.size(0) == 1 and tns.size(1) == c_in:
                bonus = 10.0
            sc = score * (1.0 + bonus)

            if debug:
                print(
                    "cand",
                    tns.shape,
                    "absmean=",
                    float(tns.detach().abs().mean()),
                    "bonus=",
                    bonus,
                    "score=",
                    sc,
                )
            if sc > best_score:
                try:
                    candidate = _ensure_4d_sample(tns)
                    best, best_score = candidate, sc
                except AssertionError:
                    continue
        assert best is not None, "Failed to coerce any model output to [C,T,H,W]"
        return best

    if torch.is_tensor(out):
        # Batched: want [B,C,T,H,W]. If [B,C,H,W], add T=1 at dim=2.
        if out.dim() == 5:
            y_bcthw = out
        elif out.dim() == 4:
            y_bcthw = out.unsqueeze(2)  # [B,C,H,W] -> [B,C,1,H,W]
        else:
            raise AssertionError(f"Unexpected batched output rank {out.dim()}")
    elif isinstance(out, (list, tuple)):
        # Per-sample outputs: each to [C,T,H,W], then stack to [B,C,T,H,W]
        items = []
        for s in out:
            items.append(_pick_best_sample_tensor(s))
        y_bcthw = torch.stack(items, dim=0)
    else:
        raise AssertionError(f"Unexpected model output type {type(out)}")

    # ---- 6) Restore original layout ----
    if orig == "BCTHW":
        y = y_bcthw
    elif orig == "BTCHW":
        y = y_bcthw.permute(0, 2, 1, 3, 4).contiguous()
    else:  # "BCHW"
        y = y_bcthw.squeeze(2).contiguous()

    return y.to(x_t.dtype)


@torch.no_grad()
def _norm_to_neg1_pos1(x: torch.Tensor) -> torch.Tensor:
    # x uint8 [0,255] or float [0,1] -> float [-1,1]
    if x.dtype in (torch.uint8, torch.int16, torch.int32, torch.int64):
        x = x.float() / 255.0
    return x.mul_(2.0).sub_(1.0)


def _to_bthwc(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize pixel tensor layouts to [B, T, H, W, C].
    Accepts BTCHW, BTHWC, BCHW, BHWC.
    """
    if x.dim() == 5:
        # BTCHW -> BTHWC
        if x.size(2) in (1, 3):
            return x.permute(0, 1, 3, 4, 2).contiguous()
        # already BTHWC
        if x.size(-1) in (1, 3):
            return x.contiguous()
    elif x.dim() == 4:
        # BCHW -> BTHWC (T=1)
        if x.size(1) in (1, 3):
            return x.permute(0, 2, 3, 1).unsqueeze(1).contiguous()
        # BHWC -> BTHWC (T=1)
        if x.size(-1) in (1, 3):
            return x.unsqueeze(1).contiguous()
    raise AssertionError(
        f"Unexpected pixel tensor shape {tuple(x.shape)}; expected BCHW/BTCHW/BHWC/BTHWC."
    )


@torch.no_grad()
def encode_pixels_to_latents(vae, pixels):
    x = _to_bthwc(pixels)
    return vae.encode(x)


def parse_allowed_frames(arg: str) -> List[int]:
    return [int(x) for x in arg.split(",") if x.strip()]


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

    # build/load VAE explicitly (Wan keeps it separate from the DiT)
    vae = load_wan_vae(
        args.vae_dir, args.vae_module, args.vae_ckpt, device=device, dtype=dtype
    )

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
    if args.enable_self_attn or args.enable_ffn:
        inject_lora_visual(
            model,
            blocks_attr="blocks",
            attn_attr="attn",
            ffn_attr="ffn",
            targets=targets,
            rank=args.rank,
            alpha=args.alpha,
            dropout=0.0,
            enable_attn=args.enable_self_attn,
            enable_ffn=args.enable_ffn,
        )

    hit_counts = {}

    def _fw_hook(mod, inp, out):
        hit_counts[id(mod)] = hit_counts.get(id(mod), 0) + 1

    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.register_forward_hook(_fw_hook)

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
    # Freeze cross-attn LoRA if requested
    if args.freeze_cross:
        for name, module in model.named_modules():
            if isinstance(module, LoRALinear) and ".cross_attn." in name:
                for p in module.trainable_parameters():
                    p.requires_grad_(False)

    # Build param groups: text vs visual
    text_params, visual_params = [], []
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            bucket = (
                visual_params if (".attn." in name or ".ffn." in name) else text_params
            )
            for p in module.trainable_parameters():
                if p.requires_grad:
                    bucket.append(p)

    params = text_params + visual_params
    n_params = count_params(params)
    print(
        f"[lora] Trainable LoRA params: {n_params/1e6:.3f} M across targets={targets} "
        f"(freeze_kv={args.freeze_kv} freeze_cross={args.freeze_cross})"
    )

    opt_groups = []
    if text_params:
        opt_groups.append({"params": text_params, "lr": args.lr})
    if visual_params:
        opt_groups.append({"params": visual_params, "lr": (args.lr_visual or args.lr)})

    # Text encoder (Bridge) runs inside Wan via environment, but we use it here to produce tokens
    # so we can control masking & pass through model.text_embedding.
    bridge = BridgeEncoderModel(text_len=args.text_len, device=device, dtype=dtype)

    # Data
    frames = parse_allowed_frames(args.frames)
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
        opt_groups, weight_decay=args.weight_decay, betas=(0.9, 0.95)
    )
    model.train()
    set_lora_enabled(model, True)

    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    print(
        f"[sanity] trainable tensors: {len(trainable)}  total_params={sum(p.numel() for _,p in trainable):,}"
    )
    for n, p in trainable[:8]:
        print("   ", n, tuple(p.shape))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ==== USAGE IN TRAIN LOOP (replace your loss/backward block) ====
    # After you have constructed model/vae/optimizer and parsed args:
    AMP_DTYPE, scaler = setup_amp_and_models(args, model, vae)

    # Keep only the small LoRA matrices in fp32 for stability
    from kv_lora_inject import LoRALinear as _LoRALinear

    for m in model.modules():
        if isinstance(m, _LoRALinear) and m.lora_A is not None:
            m.lora_A.data = m.lora_A.data.float()
            m.lora_B.data = m.lora_B.data.float()

    step = 0
    running = 0.0
    pbar = tqdm(total=args.steps, desc="train-stage2")
    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            vids = batch["pixel"].to(device)  # [B,T,3,H,W]
            caps: List[str] = batch["caption"]

            if step == 0:
                try:
                    print(
                        "[once] vids shape/dtype:",
                        tuple(vids.shape),
                        vids.dtype,
                        " min/max:",
                        float(vids.min()),
                        float(vids.max()),
                    )
                except Exception:
                    pass

            # text → tokens → context
            tokens_list = bridge(caps, device)  # list of [L_i, d]
            tokens, mask = pad_stack(tokens_list, args.text_len)  # [B,L,d], [B,L,1]
            tokens = tokens.to(device=device, dtype=dtype)
            mask = mask.to(device=device, dtype=torch.float32)
            context = tokens  # [B,L,d_model]

            # encode latents
            vids = vids.to(torch.float32)
            vids = vids * 2.0 - 1.0  # [0,1] -> [-1,1]
            x1 = encode_pixels_to_latents(vae, vids)

            # flow-matching pair
            x_t, v, t = make_velocity_training_pair(x1)  # x_t/v are dtype=dtype

            with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
                pred = call_dit(model, x_t, t, context, mask, debug=(step % 200 == 0))
                mse_fp32 = F.mse_loss(pred.float(), v.float())
                loss = mse_fp32 / args.grad_accum
            running += float(mse_fp32.item())

            debug_step(
                step,
                vids=vids,
                x1=x1,
                x_t=x_t,
                v=v,
                pred=pred,
                loss=loss.detach(),
                t=t,
                model=model,
                grad_accum=args.grad_accum,
                every=200,
            )

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % 200 == 0:
                print(
                    "[lora] modules hit in forward:",
                    sum(hit_counts.values()),
                    "unique:",
                    len(hit_counts),
                )

            if step % 200 == 0:
                gmeans = []
                for n, p in model.named_parameters():
                    if p.requires_grad and (p.grad is not None):
                        gmeans.append(p.grad.detach().abs().mean().item())
                if gmeans:
                    print(
                        f"[grads] mean(|grad|)={sum(gmeans)/len(gmeans):.3e} over {len(gmeans)} tensors"
                    )

            # Gradient step on accumulation boundary:
            if (step + 1) % args.grad_accum == 0:
                if args.clip_grad and args.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(params, args.clip_grad)
                if scaler.is_enabled():
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)

            step += 1
            pbar.update(1)
            if step % args.log_every == 0:
                avg = running / args.log_every
                try:
                    pbar.set_postfix(
                        mse=f"{avg:.6f}",
                        loss=f"{(avg/args.grad_accum):.8f}",
                    )
                except Exception:
                    pass
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
