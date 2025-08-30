from __future__ import annotations
import importlib
import os  # noqa: F401
import sys
from pathlib import Path
from typing import Optional
import torch

CANDIDATE_MODULES = (
    "wan.modules.vae",  # many releases
    "wan.modules.vae3d",  # some releases
    "wan.modules.stvae",  # some releases
    "wan.modules.video_vae",  # some forks
)


def _ensure_on_path(vae_dir: Optional[str]):
    if vae_dir:
        p = Path(vae_dir)
        if (p / "wan").exists():
            sys.path.insert(0, str(p))
        elif (p / "inference" / "Wan2.2" / "wan").exists():
            sys.path.insert(0, str(p / "inference" / "Wan2.2"))
        else:
            # still add; user may have already put correct root
            sys.path.insert(0, str(p))


def _import_vae_module(name: str):
    try:
        return importlib.import_module(name)
    except Exception:
        return None


def _first_existing_module(preferred: Optional[str]) -> str:
    if preferred:
        mod = _import_vae_module(preferred)
        if mod is not None:
            return preferred
    for m in CANDIDATE_MODULES:
        mod = _import_vae_module(m)
        if mod is not None:
            return m
    raise RuntimeError(
        "Could not import any Wan VAE module. "
        f"Tried preferred={preferred!r} and candidates={CANDIDATE_MODULES}. "
        "Check that your Wan repo (the folder that contains 'wan/modules') "
        "is on PYTHONPATH, or pass --vae_dir to the correct folder."
    )


def load_wan_vae(
    vae_dir: Optional[str],
    vae_module: Optional[str],
    ckpt_path: str,
    device: torch.device,
    dtype: torch.dtype,
):
    # 1) Make sure the Wan repo root (the directory that contains 'wan/') is on sys.path.
    _ensure_on_path(vae_dir)

    # 2) Find an importable module name
    module_name = _first_existing_module(vae_module)

    # 3) Import and construct the VAE
    mod = importlib.import_module(module_name)

    # Common class/function names in different Wan drops:
    ctor_names = ["VAE", "WanVAE", "VideoVAE", "STVAE", "build_vae", "load_vae"]
    obj = None
    for name in ctor_names:
        if hasattr(mod, name):
            obj = getattr(mod, name)
            break
    if obj is None:
        raise RuntimeError(
            f"Loaded module '{module_name}' but couldn't find a VAE constructor "
            f"(tried {ctor_names}). Please open {module_name} and pick the class/function."
        )

    # 4) Instantiate and load weights (support .safetensors and .pth)
    if callable(obj) and not isinstance(obj, type):
        vae = obj()  # build_vae()/load_vae() returning nn.Module
    else:
        try:
            vae = obj()
        except TypeError as e:
            # fallback: some ctors need no args but disallow missing; try defaults
            try:
                vae = obj  # maybe it's already a module instance
            except Exception:
                raise e

    # 5) Load checkpoint
    ckpt_path = str(ckpt_path)
    if ckpt_path.endswith(".safetensors"):
        from safetensors.torch import load_file as st_load

        sd = st_load(ckpt_path)
        missing, unexpected = vae.load_state_dict(sd, strict=False)
    else:
        sd = torch.load(ckpt_path, map_location="cpu")
        # some Wan checkpoints wrap {"state_dict":..., "module":...}
        state = sd.get("state_dict", sd.get("module", sd))
        missing, unexpected = vae.load_state_dict(state, strict=False)

    if len(unexpected) > 0:
        print(f"[wan_vae_loader] unexpected keys: {unexpected[:8]} ...")
    if len(missing) > 0:
        print(f"[wan_vae_loader] missing keys   : {missing[:8]} ...")

    vae.to(device=device, dtype=dtype).eval()
    return vae
