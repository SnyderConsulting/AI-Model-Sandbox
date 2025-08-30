# wan_vae_loader.py  — supports models.wan.vae2_2.Wan2_2_VAE and unifies encode API
from __future__ import annotations
import importlib, os, sys
from pathlib import Path
from typing import Optional, List
import torch
import torch.nn as nn

def _ensure_on_path(vae_dir: Optional[str]):
    # Put a folder that contains "models/" or "wan/" on sys.path
    if vae_dir:
        p = Path(vae_dir)
        if p.exists():
            sys.path.insert(0, str(p))

def load_wan_vae(vae_dir: Optional[str],
                 vae_module: Optional[str],
                 ckpt_path: str,
                 device: torch.device,
                 dtype: torch.dtype):
    """
    Returns an object with a unified:
        encode(x) -> latents
    where:
        x is [B,T,H,W,3] in [-1,1] or [B,3,H,W] in [-1,1]
        latents is [B, 1+T/4, C', H', W'] (Wan-VAE style)
    """
    _ensure_on_path(vae_dir or os.getcwd())

    if not vae_module:
        raise RuntimeError("Pass --vae_module, e.g. models.wan.vae2_2")

    try:
        mod = importlib.import_module(vae_module)
    except Exception as e:
        raise RuntimeError(f"Could not import {vae_module}: {e}")

    # ---- special-case your VAE class ----
    if hasattr(mod, "Wan2_2_VAE"):
        # instantiate your wrapper; it expects 'vae_pth', 'dtype', 'device'
        inner = mod.Wan2_2_VAE(
            vae_pth=ckpt_path,
            dtype=(torch.bfloat16 if dtype == torch.bfloat16 else torch.float32),
            device=("cuda" if device.type == "cuda" else "cpu"),
        )

        class _Wrapper(nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            @torch.no_grad()
            def encode(self, x: torch.Tensor) -> torch.Tensor:
                """
                x: [B,T,H,W,3] or [B,3,H,W]  in [-1,1]
                returns [B, F, C, H', W'] with F = 1 + T/4
                """
                if x.dim() == 4:  # images → [B,3,H,W] -> [B,1,H,W,3]
                    x = x.permute(0, 2, 3, 1)  # [B,H,W,3]
                    x = x.unsqueeze(1)

                assert x.dim() == 5 and x.size(-1) == 3, f"Expected [B,T,H,W,3], got {tuple(x.shape)}"
                B, T, H, W, _ = x.shape
                # per-sample list of [3,T,H,W]
                lst: List[torch.Tensor] = [x[b].permute(3,0,1,2).contiguous() for b in range(B)]
                # your class returns a list of [C', F', H', W'] tensors
                z_list = self.inner.encode(lst)
                # stack to [B, F', C', H', W']
                z = torch.stack([z.permute(1,0,2,3).contiguous() for z in z_list], dim=0)
                return z

        vae = _Wrapper(inner).to(device=device)
        return vae

    raise RuntimeError(
        f"Module '{vae_module}' imported, but no supported class was found. "
        f"Expected 'Wan2_2_VAE' in that module."
    )
