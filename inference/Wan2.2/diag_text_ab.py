
#!/usr/bin/env python3
"""
diag_text_ab_v2.py — A/B diagnostics for Wan text encoders (stock UMT5 vs Bridge)

Usage examples:
  python diag_text_ab_v2.py --prompts "neon reflections" "A serene sunset"
  WAN_T5_CKPT=/workspace/models/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth \
    python diag_text_ab_v2.py --prompts_file prompts.txt --device cuda:0

If the stock encoder checkpoint can't be found, set WAN_T5_CKPT explicitly.
Tokenizer defaults to google/umt5-xxl; override with WAN_T5_TOKENIZER.
"""

from __future__ import annotations
import os, argparse, importlib, math
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import torch
import torch.nn.functional as F

DEFAULT_OUTDIR = Path(os.environ.get("TEXT_AB_OUTDIR", "/tmp/text_ab")).expanduser()

WAN_T5_MODULE_CANDIDATES = ["wan.models.t5", "wan.modules.t5"]

def _import_wan_t5():
    last = None
    for name in WAN_T5_MODULE_CANDIDATES:
        try:
            return importlib.import_module(name)
        except Exception as e:
            last = e
    raise RuntimeError(f"Could not import wan t5 module from {WAN_T5_MODULE_CANDIDATES}: {last}")

def _reload_wan_t5():
    mod = _import_wan_t5()
    return importlib.reload(mod)

def _find_ckpt(filename: str) -> Optional[str]:
    env = os.environ.get("WAN_T5_CKPT")
    if env and Path(env).exists():
        return env
    candidates = [
        Path.cwd()/filename,
        Path.cwd().parent/filename,
        Path("/workspace/models")/filename,
        Path("/workspace/models/Wan2.2-TI2V-5B")/filename,
        Path("/workspace/AI-Model-Sandbox/inference/Wan2.2")/filename,
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None

def _find_tokenizer(default="google/umt5-xxl") -> str:
    return os.environ.get("WAN_T5_TOKENIZER", default)

def build_encoder(use_bridge: bool, text_len: int, dtype: torch.dtype, device: str, t5_cpu: bool=False):
    os.environ["WAN_USE_BRIDGE"] = "1" if use_bridge else "0"
    t5_mod = _reload_wan_t5()

    kwargs = dict(text_len=text_len, dtype=dtype, device=device)
    if not use_bridge:
        ckpt = _find_ckpt("models_t5_umt5-xxl-enc-bf16.pth")
        tok  = _find_tokenizer()
        if ckpt is None:
            raise SystemExit("Could not locate UMT5 encoder checkpoint. Set WAN_T5_CKPT, "
                             "e.g. /workspace/models/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth")
        kwargs.update(checkpoint_path=ckpt, tokenizer_path=tok)

    try:
        enc = t5_mod.T5EncoderModel(**kwargs, t5_cpu=t5_cpu)
    except TypeError:
        enc = t5_mod.T5EncoderModel(**kwargs)
    return enc

@torch.no_grad()
def encode(enc, prompts: List[str], device: str):
    out = enc(prompts, device=device)
    assert isinstance(out, list) and all(torch.is_tensor(x) for x in out), "Encoder did not return List[Tensor]"
    return [x.float().contiguous() for x in out]

def _safe_min_len(a: torch.Tensor, b: torch.Tensor) -> int:
    return int(min(a.size(0), b.size(0)))

def _per_prompt_metrics(t: torch.Tensor, b: torch.Tensor) -> Dict[str, Any]:
    L = _safe_min_len(t, b)
    t = t[:L].float()
    b = b[:L].float()
    D = t.size(1)

    t_flat = t.flatten()
    b_flat = b.flatten()
    seq_cos = F.cosine_similarity(t_flat, b_flat, dim=0).item()

    t_norm = t.norm(dim=-1) + 1e-8
    b_norm = b.norm(dim=-1) + 1e-8
    token_cos = (t * b).sum(dim=-1) / (t_norm * b_norm)
    token_cos_mean = token_cos.mean().item()
    token_cos_med  = token_cos.median().item()
    token_cos_p10  = token_cos.kthvalue(max(1, int(0.10 * L))).values.item()
    token_cos_p90  = token_cos.kthvalue(max(1, int(0.90 * L))).values.item()

    norm_ratio = (b_norm / (t_norm + 1e-8)).mean().item()

    mu_t = t.mean(dim=0)
    mu_b = b.mean(dim=0)
    std_t = t.std(dim=0, unbiased=False) + 1e-12
    std_b = b.std(dim=0, unbiased=False) + 1e-12

    mu_abs_delta = (mu_b - mu_t).abs().mean().item()
    std_ratio_med = (std_b / std_t).median().item()
    std_ratio_mean = (std_b / std_t).mean().item()
    mu_cos = F.cosine_similarity(mu_t, mu_b, dim=0).item()

    return dict(
        Lt=int(t.size(0)), Lb=int(b.size(0)), D=int(D),
        seq_cos=seq_cos,
        token_cos_mean=token_cos_mean,
        token_cos_median=token_cos_med,
        token_cos_p10=token_cos_p10,
        token_cos_p90=token_cos_p90,
        norm_ratio=norm_ratio,
        mu_abs_delta=mu_abs_delta,
        mu_cos=mu_cos,
        std_ratio_median=std_ratio_med,
        std_ratio_mean=std_ratio_mean,
    )

def run(prompts: List[str], text_len: int, dtype: torch.dtype, device: str, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)

    teacher = build_encoder(use_bridge=False, text_len=text_len, dtype=dtype, device=device)
    bridge  = build_encoder(use_bridge=True,  text_len=text_len, dtype=dtype, device=device)

    T_list = encode(teacher, prompts, device=device)
    B_list = encode(bridge,  prompts, device=device)

    import pandas as pd
    rows = []
    for p, t, b in zip(prompts, T_list, B_list):
        m = _per_prompt_metrics(t, b)
        m["prompt"] = p
        rows.append(m)
    df = pd.DataFrame(rows)

    summary = {
        "n_prompts": len(prompts),
        "seq_cos_mean": float(df["seq_cos"].mean()),
        "seq_cos_median": float(df["seq_cos"].median()),
        "token_cos_mean": float(df["token_cos_mean"].mean()),
        "token_cos_median": float(df["token_cos_median"].median()),
        "norm_ratio_mean": float(df["norm_ratio"].mean()),
        "std_ratio_median_mean": float(df["std_ratio_median"].mean()),
        "mu_abs_delta_mean": float(df["mu_abs_delta"].mean()),
        "mu_cos_mean": float(df["mu_cos"].mean()),
    }

    csv_path = outdir / "text_ab_metrics.csv"
    json_path = outdir / "text_ab_metrics.json"
    df.to_csv(csv_path, index=False)
    import json as _json
    with open(json_path, "w") as f:
        _json.dump({"summary": summary, "rows": rows}, f, indent=2)

    print("\n=== Text Encoder A/B Diagnostics ===")
    print(f"Prompts: {len(prompts)}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved JSON: {json_path}\n")
    print(df[["prompt","seq_cos","token_cos_mean","token_cos_median","norm_ratio","std_ratio_median","mu_abs_delta","mu_cos"]])
    print("\n--- Overall summary ---")
    for k,v in summary.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

def parse_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name in ("bf16","bfloat16"): return torch.bfloat16
    if name in ("fp16","float16","half"): return torch.float16
    if name in ("fp32","float32","full"): return torch.float32
    raise ValueError(f"Unknown dtype: {name}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=str, help="Single prompt")
    ap.add_argument("--prompts", nargs="*", default=[], help="Multiple prompts")
    ap.add_argument("--prompts_file", type=str, help="File with one prompt per line")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16","fp16","fp32"])
    ap.add_argument("--text_len", type=int, default=512)
    ap.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    args = ap.parse_args()

    plist: List[str] = []
    if args.prompt: plist.append(args.prompt.strip())
    if args.prompts: plist.extend([p.strip() for p in args.prompts if p.strip()])
    if args.prompts_file:
        with open(args.prompts_file, "r") as f:
            lines = [ln.strip() for ln in f.readlines()]
        plist.extend([ln for ln in lines if ln])

    if not plist:
        raise SystemExit("No prompts provided. Use --prompt, --prompts, or --prompts_file.")

    dtype = parse_dtype(args.dtype)
    outdir = Path(args.outdir)

    run(plist, text_len=args.text_len, dtype=dtype, device=args.device, outdir=outdir)

if __name__ == "__main__":
    main()
