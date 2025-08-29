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
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode
from torchvision.io import read_video  # fallback video decoder
from torchvision.transforms.functional import resize, center_crop

# ---- torchvision dtype shim (works across torchvision versions) ----
try:
    # Newer builds may have to_dtype; keep its behavior if present.
    from torchvision.transforms.functional import to_dtype as _tv_to_dtype

    def to_dtype(img, dtype, scale: bool = True):
        # mirror the signature we use later in the file
        return _tv_to_dtype(img, dtype=dtype, scale=scale)

except Exception:
    # Fallback for common builds: use convert_image_dtype (scales ints->[0,1] when going to float)
    from torchvision.transforms.functional import (
        convert_image_dtype as _convert_image_dtype,
    )
    import torch

    def to_dtype(img, dtype, scale: bool = True):
        # If we're converting integer tensors to float and scale=True, convert_image_dtype
        # will scale to [0,1]. If scale=False, just cast.
        if scale:
            return _convert_image_dtype(img, dtype)
        else:
            return img.to(dtype)


from tqdm import tqdm

# ---- local imports
from kv_lora_inject import (
    inject_lora_kv,
    set_lora_enabled,
    lora_parameters,
    load_peft_adapter,
    export_peft_adapter,
    LoRALinear,
)

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
    p.add_argument("--use_480p", action="store_true")
    p.add_argument("--use_720p", action="store_true")
    p.add_argument(
        "--frames",
        type=str,
        default="17,33,49,65,81",
        help="Which frame-count folders to include for videos.",
    )
    p.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional cap on total samples (images+videos).",
    )
    p.add_argument("--shuffle", action="store_true")
    p.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size in videos usually must be small.",
    )
    p.add_argument("--num_workers", type=int, default=4)

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


# ---- Data helpers


def _read_caption(txt_path: str) -> str:
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            t = f.read().strip()
            return t if t else " "
    except Exception:
        return " "


def _norm_to_minus1_1(t: torch.Tensor) -> torch.Tensor:
    # t in [0,1] float -> [-1,1]
    return t * 2.0 - 1.0


def _load_image(path: str, side: int) -> torch.Tensor:
    # returns [T=1, H, W, C]
    img = read_image(path, mode=ImageReadMode.RGB)  # [3,H,W], uint8
    img = to_dtype(img, torch.float32, scale=True)  # -> [0,1]
    # make square crop to preserve aspect before resizing (Wan uses fixed res)
    H, W = img.shape[1], img.shape[2]
    s = min(H, W)
    img = center_crop(img, [s, s])
    img = resize(img, [side, side], antialias=True)
    img = img.permute(1, 2, 0).contiguous()  # [H,W,3]
    img = _norm_to_minus1_1(img)
    return img.unsqueeze(0)  # [1,H,W,3]


def _ensure_nframes(frames: torch.Tensor, target_T: int) -> torch.Tensor:
    # frames: [T,H,W,3] in [-1,1]
    T = frames.shape[0]
    if T == target_T:
        return frames
    if T > target_T:
        # uniform downsample
        idx = torch.linspace(0, T - 1, target_T).round().long()
        return frames.index_select(0, idx)
    # pad last frame
    last = frames[-1:].repeat(target_T - T, 1, 1, 1)
    return torch.cat([frames, last], dim=0)


def _load_video(path: str, side: int, target_T: int) -> torch.Tensor:
    # returns [T,H,W,3]
    # Note: torchvision.io.read_video returns T,H,W,C in uint8 (or float)
    vframes, _, _ = read_video(path, pts_unit="sec")
    vframes = to_dtype(vframes, torch.float32, scale=True)  # [T,H,W,3] in [0,1]
    # center-crop shortest side then resize square
    H, W = int(vframes.shape[1]), int(vframes.shape[2])
    s = min(H, W)
    top = (H - s) // 2
    left = (W - s) // 2
    vframes = vframes[:, top : top + s, left : left + s, :]
    # resize per-frame
    vframes = torch.stack(
        [
            resize(f.permute(2, 0, 1), [side, side], antialias=True).permute(1, 2, 0)
            for f in vframes
        ],
        dim=0,
    )
    vframes = _norm_to_minus1_1(vframes)
    vframes = _ensure_nframes(vframes, target_T)
    return vframes  # [T,H,W,3]


@dataclass
class MediaSample:
    kind: str  # "image" or "video"
    path: str
    caption: str
    reso: int  # 480 or 720
    frames: int  # 1 for images, else one of {17,33,49,65,81}


class MediaDataset(Dataset):
    def __init__(
        self,
        root: str,
        use_480p: bool,
        use_720p: bool,
        frames_list: List[int],
        max_samples: Optional[int] = None,
        shuffle: bool = False,
    ):
        root = Path(root)
        items: List[MediaSample] = []

        def scan_img(reso: int):
            d = root / "image" / f"{reso}p"
            for img in glob.glob(str(d / "*")):
                if os.path.splitext(img)[1].lower() not in (
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".bmp",
                    ".webp",
                ):
                    continue
                cap = os.path.splitext(img)[0] + ".txt"
                items.append(MediaSample("image", img, _read_caption(cap), reso, 1))

        def scan_vid(reso: int):
            vd = root / "video" / f"{reso}p"
            for T in frames_list:
                td = vd / f"{T}frames"
                for mp4 in glob.glob(str(td / "*.mp4")):
                    cap = os.path.splitext(mp4)[0] + ".txt"
                    items.append(MediaSample("video", mp4, _read_caption(cap), reso, T))

        if use_480p:
            scan_img(480)
            scan_vid(480)
        if use_720p:
            scan_img(720)
            scan_vid(720)

        if shuffle:
            random.shuffle(items)
        if max_samples is not None:
            items = items[:max_samples]

        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        s = self.items[idx]
        side = s.reso
        if s.kind == "image":
            frames = _load_image(s.path, side)  # [1,H,W,3]
        else:
            frames = _load_video(s.path, side, s.frames)  # [T,H,W,3]
        return {
            "kind": s.kind,
            "path": s.path,
            "caption": s.caption,
            "video": frames,  # [T,H,W,3], in [-1,1]
            "T": frames.shape[0],
            "H": frames.shape[1],
            "W": frames.shape[2],
        }


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
    os.environ["WAN_LLM_DIR"] = args.llm_dir
    os.environ["WAN_BRIDGE_GLOBAL_SCALE"] = str(args.global_scale)

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
    frames_list = [int(x) for x in args.frames.split(",") if x.strip()]
    ds = MediaDataset(
        args.data_root,
        args.use_480p,
        args.use_720p,
        frames_list=frames_list,
        max_samples=args.max_samples,
        shuffle=args.shuffle,
    )
    if len(ds) == 0:
        raise RuntimeError("No samples found under the given config.")
    print(f"[data] total samples: {len(ds)}")

    def _collate(batch):
        # batch is list of dicts
        caps = [b["caption"] for b in batch]
        # pad to max T in batch; stack videos
        T_max = max(b["video"].shape[0] for b in batch)
        vids = []
        for b in batch:
            v = b["video"]
            if v.shape[0] != T_max:
                v = _ensure_nframes(v, T_max)
            vids.append(v)
        vids = torch.stack(vids, dim=0)  # [B,T,H,W,3]
        return {"captions": caps, "video": vids}

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=_collate,
        drop_last=True,
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
            vids = batch["video"].to(device)  # [B,T,H,W,3], in [-1,1]
            caps: List[str] = batch["captions"]

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
