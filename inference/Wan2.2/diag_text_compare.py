#!/usr/bin/env python3
"""
diag_text_compare.py — Deep diagnostics for a single prompt (Bridge vs UMT5)

What it does
------------
- Loads Wan's stock UMT5 encoder and your Bridge (by toggling WAN_USE_BRIDGE)
- Encodes the SAME prompt with both
- Computes:
  * token-level cosine (vector) + summary stats
  * per-token norm ratio (bridge/teacher) + summary
  * per-dimension mean delta and std ratio
  * sequence-level cosine (flattened)
  * mu-vector cosine
- Saves:
  * report.json + report.txt (human readable)
  * per-token cosine and norm ratio CSV
  * per-dim std-ratio CSV
  * teacher/bridge tensors as .npy
  * 3 figures (matplotlib): token-cos vs position, norm-ratio vs position, std-ratio histogram

Usage
-----
export WAN_T5_CKPT=/workspace/models/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth
python diag_text_compare.py --prompt "a neon-lit alley at night" --device cpu
python diag_text_compare.py --prompt "..." --device cuda:0 --outdir /tmp/text_diag

Notes
-----
- No need to load the DiT; this isolates the text interface.
- If you want multiple prompts, prefer diag_text_ab_v2.py; this tool focuses on one prompt with deeper plots.
"""

from __future__ import annotations
import os, json, math, argparse, importlib
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

WAN_T5_MODULE_CANDIDATES = ["wan.models.t5", "wan.modules.t5"]

def _import_wan_t5():
    last=None
    for name in WAN_T5_MODULE_CANDIDATES:
        try:
            return importlib.import_module(name)
        except Exception as e:
            last=e
    raise RuntimeError(f"Could not import Wan t5 module: {last}")

def _reload_wan_t5():
    mod = _import_wan_t5()
    return importlib.reload(mod)

def _find_ckpt(filename: str) -> Optional[str]:
    env = os.environ.get("WAN_T5_CKPT")
    if env and Path(env).exists():
        return env
    for c in [
        Path.cwd()/filename,
        Path.cwd().parent/filename,
        Path("/workspace/models")/filename,
        Path("/workspace/models/Wan2.2-TI2V-5B")/filename,
        Path("/workspace/AI-Model-Sandbox/inference/Wan2.2")/filename,
    ]:
        if c.exists(): return str(c)
    return None

def build_encoder(use_bridge: bool, text_len: int, dtype: torch.dtype, device: str):
    os.environ["WAN_USE_BRIDGE"] = "1" if use_bridge else "0"
    t5_mod = _reload_wan_t5()
    kwargs=dict(text_len=text_len, dtype=dtype, device=device)
    if not use_bridge:
        ckpt = _find_ckpt("models_t5_umt5-xxl-enc-bf16.pth")
        tok  = os.environ.get("WAN_T5_TOKENIZER", "google/umt5-xxl")
        if ckpt is None:
            raise SystemExit("Set WAN_T5_CKPT to the UMT5 encoder checkpoint path.")
        kwargs.update(checkpoint_path=ckpt, tokenizer_path=tok)
    try:
        return t5_mod.T5EncoderModel(**kwargs, t5_cpu=(device=="cpu"))
    except TypeError:
        return t5_mod.T5EncoderModel(**kwargs)

@torch.no_grad()
def encode(enc, prompt: str, device: str) -> torch.Tensor:
    lst = enc([prompt], device=device)
    assert isinstance(lst, list) and len(lst)==1 and torch.is_tensor(lst[0])
    return lst[0].float().contiguous()  # [L, D]

def _metrics(T: torch.Tensor, B: torch.Tensor) -> Dict[str, Any]:
    L = min(T.size(0), B.size(0))
    T = T[:L]; B=B[:L]
    # token cos
    tn = T / (T.norm(dim=-1, keepdim=True)+1e-8)
    bn = B / (B.norm(dim=-1, keepdim=True)+1e-8)
    cos_tok = (tn*bn).sum(dim=-1)              # [L]
    # norms
    norm_ratio = (B.norm(dim=-1)/(T.norm(dim=-1)+1e-8)) # [L]
    # per-dim stats
    mu_T = T.mean(dim=0); mu_B = B.mean(dim=0)
    std_T = T.std(dim=0, unbiased=False)+1e-8
    std_B = B.std(dim=0, unbiased=False)+1e-8
    std_ratio = (std_B/std_T)                  # [D]
    mu_delta = (mu_B - mu_T)                   # [D]
    # sequence level
    seq_cos = F.cosine_similarity(T.flatten(), B.flatten(), dim=0).item()
    mu_cos  = F.cosine_similarity(mu_T, mu_B, dim=0).item()
    return dict(
        L=int(L), D=int(T.size(1)),
        cos_tok=cos_tok.cpu(), norm_ratio=norm_ratio.cpu(),
        std_ratio=std_ratio.cpu(), mu_delta=mu_delta.cpu(),
        seq_cos=float(seq_cos), mu_cos=float(mu_cos),
        token_cos_mean=float(cos_tok.mean().item()),
        token_cos_median=float(cos_tok.median().item()),
        norm_ratio_mean=float(norm_ratio.mean().item()),
        std_ratio_median=float(std_ratio.median().item()),
        mu_abs_delta=float(mu_delta.abs().mean().item()),
        global_scale_suggestion=float(1.0/max(1e-8, norm_ratio.mean().item()))
    )

def _save_arrays(outdir: Path, name: str, arr: torch.Tensor):
    np.save(outdir / f"{name}.npy", arr.cpu().numpy())

def _plot_token_curve(y: np.ndarray, title: str, out: Path, ylabel: str):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(np.arange(len(y)), y)
    plt.xlabel("token position")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

def _plot_hist(y: np.ndarray, title: str, out: Path, xlabel: str, bins: int = 80):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.hist(y, bins=bins)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True, type=str)
    ap.add_argument("--text_len", type=int, default=512)
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16","fp16","fp32"])
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--outdir", type=str, default="/tmp/text_diag")
    args = ap.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Build encoders
    teacher = build_encoder(False, args.text_len, dtype, args.device)
    bridge  = build_encoder(True,  args.text_len, dtype, args.device)

    # Encode
    T = encode(teacher, args.prompt, args.device)
    B = encode(bridge,  args.prompt, args.device)

    # Save raw tensors
    _save_arrays(outdir, "teacher_tokens", T)
    _save_arrays(outdir, "bridge_tokens",  B)

    # Metrics
    m = _metrics(T, B)

    # Save CSVs
    import pandas as pd
    tok_df = pd.DataFrame({
        "token_index": np.arange(m["L"]),
        "token_cos": m["cos_tok"].numpy(),
        "norm_ratio": m["norm_ratio"].numpy(),
    })
    tok_df.to_csv(outdir / "per_token_metrics.csv", index=False)

    dim_df = pd.DataFrame({
        "dim": np.arange(m["D"]),
        "std_ratio": m["std_ratio"].numpy(),
        "mu_delta": m["mu_delta"].numpy(),
    })
    dim_df.to_csv(outdir / "per_dim_metrics.csv", index=False)

    # Plots
    _plot_token_curve(tok_df["token_cos"].to_numpy(), "Per-token cosine", outdir/"token_cosine_curve.png", "cosine")
    _plot_token_curve(tok_df["norm_ratio"].to_numpy(), "Per-token norm ratio (bridge/teacher)", outdir/"token_norm_ratio_curve.png", "ratio")
    _plot_hist(dim_df["std_ratio"].to_numpy(), "Per-dimension std ratio (bridge/teacher)", outdir/"std_ratio_hist.png", "std_ratio")

    # Report
    report = {
        "prompt": args.prompt,
        "env": {
            "WAN_BRIDGE_CKPT": os.environ.get("WAN_BRIDGE_CKPT",""),
            "WAN_BRIDGE_LLM_DIR": os.environ.get("WAN_BRIDGE_LLM_DIR",""),
            "WAN_BRIDGE_DTYPE": os.environ.get("WAN_BRIDGE_DTYPE",""),
            "WAN_BRIDGE_GLOBAL_SCALE": os.environ.get("WAN_BRIDGE_GLOBAL_SCALE",""),
            "WAN_BRIDGE_GLOBAL_BIAS": os.environ.get("WAN_BRIDGE_GLOBAL_BIAS",""),
            "WAN_T5_CKPT": os.environ.get("WAN_T5_CKPT",""),
        },
        "metrics": {
            k: (float(v) if isinstance(v, (int,float)) else None)
            for k,v in m.items()
            if k in ["L","D","seq_cos","mu_cos","token_cos_mean","token_cos_median","norm_ratio_mean","std_ratio_median","mu_abs_delta","global_scale_suggestion"]
        },
        "files": {
            "teacher_tokens.npy": str(outdir/"teacher_tokens.npy"),
            "bridge_tokens.npy": str(outdir/"bridge_tokens.npy"),
            "per_token_metrics.csv": str(outdir/"per_token_metrics.csv"),
            "per_dim_metrics.csv": str(outdir/"per_dim_metrics.csv"),
            "token_cosine_curve.png": str(outdir/"token_cosine_curve.png"),
            "token_norm_ratio_curve.png": str(outdir/"token_norm_ratio_curve.png"),
            "std_ratio_hist.png": str(outdir/"std_ratio_hist.png"),
        }
    }
    with open(outdir/"report.json","w") as f:
        json.dump(report, f, indent=2)

    # Human-readable text
    txt = []
    txt.append(f"Prompt: {args.prompt}")
    txt.append(f"L={m['L']}  D={m['D']}")
    txt.append("---- Summary ----")
    txt.append(f"seq_cos={m['seq_cos']:.4f}  mu_cos={m['mu_cos']:.4f}")
    txt.append(f"token_cos_mean={m['token_cos_mean']:.4f}  token_cos_median={m['token_cos_median']:.4f}")
    txt.append(f"norm_ratio_mean={m['norm_ratio_mean']:.4f}  std_ratio_median={m['std_ratio_median']:.4f}")
    txt.append(f"mu_abs_delta={m['mu_abs_delta']:.5f}")
    txt.append(f"global_scale_suggestion≈ {m['global_scale_suggestion']:.4f}   (set WAN_BRIDGE_GLOBAL_SCALE to this for a crude sanity check)")
    (outdir/"report.txt").write_text("\n".join(txt))
    print("\n".join(txt))
    print(f"\nWrote report and artifacts to {outdir}")

if __name__ == "__main__":
    main()
