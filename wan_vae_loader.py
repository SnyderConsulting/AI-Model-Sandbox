from __future__ import annotations
import importlib
from pathlib import Path
from typing import Optional
import torch

# We try a few common module paths used by Wan 2.x repos.
CANDIDATES = [
    ("inference.Wan2.2.wan.modules.vae.video_vae", "WanVAE", "from_pretrained"),
    ("inference.Wan2.2.wan.modules.vae", "WanVAE", "from_pretrained"),
    ("inference.Wan2.2.wan.modules.vae", "load_vae", None),
    ("wan.modules.vae.video_vae", "WanVAE", "from_pretrained"),
    ("wan.modules.vae", "WanVAE", "from_pretrained"),
    ("wan.modules.vae", "load_vae", None),
]


def _maybe_find_vae_file(root: str | Path) -> Optional[str]:
    root = Path(root)
    if not root.exists():
        return None
    # look for common filenames
    for name in ["vae.safetensors", "video_vae.safetensors", "wan_vae.safetensors"]:
        p = root / name
        if p.exists():
            return str(p)
    # last resort: the only .safetensors under root
    ss = list(root.rglob("*.safetensors"))
    return str(ss[0]) if ss else None


def load_wan_vae(
    vae_dir_or_ckpt: Optional[str], device: torch.device, dtype: torch.dtype
):
    """
    Returns an object with a callable 'encode' that accepts:
      - image tensor  [B,3,H,W] in [-1,1]  -> [B, 1, 16, H/8, W/8]
      - video tensor  [B,T,3,H,W] in [-1,1] -> [B, 1+T/4, 16, H/8, W/8]
    """
    # Determine resource
    ckpt = None
    if vae_dir_or_ckpt:
        p = Path(vae_dir_or_ckpt)
        if p.is_file():
            ckpt = str(p)
        else:
            ckpt = _maybe_find_vae_file(p)

    # Try to import from repo
    last_err = None
    for mod_name, attr, ctor in CANDIDATES:
        try:
            m = importlib.import_module(mod_name)
            obj = getattr(m, attr)
            if callable(obj) and ctor is None:
                vae = obj(ckpt or vae_dir_or_ckpt) if ckpt or vae_dir_or_ckpt else obj()
            elif hasattr(obj, ctor or ""):
                vae = (
                    getattr(obj, ctor)(ckpt or vae_dir_or_ckpt)
                    if ckpt or vae_dir_or_ckpt
                    else getattr(obj, ctor)()
                )
            else:
                vae = obj  # already an instance
            # move / cast if methods exist
            if hasattr(vae, "to"):
                vae.to(device=device, dtype=dtype)
            if hasattr(vae, "eval"):
                vae.eval()
            break
        except Exception as e:
            last_err = e
            vae = None
            continue

    if vae is None:
        raise RuntimeError(
            f"Could not import Wan VAE. Last error: {last_err}\n"
            f"Set --vae_dir to the Wan VAE folder or --vae_ckpt to its .safetensors."
        )

    class _Wrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        @torch.no_grad()
        def encode(self, x: torch.Tensor) -> torch.Tensor:
            # Accept [B,3,H,W] or [B,T,3,H,W], values in [-1,1]
            if x.dim() == 4:  # image
                # common interfaces: encode returns either Tensor or a struct with .latent
                if hasattr(self.inner, "encode"):
                    y = self.inner.encode(x)
                elif hasattr(self.inner, "forward"):
                    y = self.inner(x)
                else:
                    raise RuntimeError("Wan VAE has no callable encode/forward")
                return getattr(y, "latent", y)
            elif x.dim() == 5:  # video
                if hasattr(self.inner, "encode_video"):
                    y = self.inner.encode_video(x)
                    return getattr(y, "latent", y)
                # fallback: run chunked along time using .encode
                B, T, _, H, W = x.shape
                outs = []
                for t0 in range(0, T, 4):
                    xt = x[:, t0 : t0 + 4]  # [B,<=4,3,H,W]
                    xt = xt.reshape(-1, 3, H, W)  # merge time
                    yt = self.encode(xt)  # [B*<=4, 1, 16, H/8, W/8]
                    # reassemble with 3D causal compression (1 frame spatial-only, rest temporal 4x)
                    outs.append(yt)
                return torch.cat(outs, dim=1)
            else:
                raise ValueError(f"Unexpected input rank {x.dim()} for VAE.encode()")

    return _Wrapper(vae)
