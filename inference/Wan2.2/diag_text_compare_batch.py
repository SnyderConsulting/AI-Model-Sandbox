
#!/usr/bin/env python3
"""
diag_text_compare_batch.py — Batched diagnostics for Bridge vs UMT5 (keeps models warm)

What it does
------------
- Loads BOTH encoders once (stock UMT5 "teacher" + Bridge) and keeps them in memory.
- Processes many prompts in batches to avoid reloading and to maximize throughput.
- For each prompt, computes the same metrics as diag_text_compare.py:
  * per-token cosine and norm ratio
  * per-dimension std ratio and mean delta
  * sequence cosine and μ-vector cosine
  * global scale suggestion = 1 / mean(norm_ratio)
- Saves per-prompt artifacts (CSVs + PNGs) and a top-level manifest CSV/JSONL.
- Does NOT load the DiT/vae; this isolates text-conditioning only.

Usage examples
--------------
export WAN_T5_CKPT=/workspace/models/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth

# 1) From a plain text file (one prompt per line)
python diag_text_compare_batch.py \
  --prompts_file /workspace/eval_runs/bridge_20000/prompts_used.txt \
  --outdir /workspace/reports/diag_batch \
  --device cpu

# 2) Sample from your NSFW JSONL (uses "caption" field), GPU for speed
python diag_text_compare_batch.py \
  --captions_jsonl /workspace/AI-Model-Sandbox/datasets/captions.jsonl \
  --sample_n 200 \
  --outdir /workspace/reports/diag_batch_nsfw \
  --device cuda:0 \
  --batch_size 64

# Notes
# - Bridge config is taken from env (WAN_BRIDGE_CKPT, WAN_BRIDGE_LLM_DIR, WAN_BRIDGE_GLOBAL_SCALE, ...).
# - The new calibrated checkpoint can be tested by setting WAN_BRIDGE_CKPT before running.
"""

from __future__ import annotations
import os, json, argparse, importlib, hashlib, random
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

# ----------------------- Wan T5 loader helpers -----------------------

WAN_T5_MODULE_CANDIDATES = ["wan.models.t5", "wan.modules.t5"]

def _import_wan_t5():
    last = None
    for name in WAN_T5_MODULE_CANDIDATES:
        try:
            return importlib.import_module(name)
        except Exception as e:
            last = e
    raise RuntimeError(f"Could not import Wan t5 module: {last}")

def _reload_wan_t5():
    mod = _import_wan_t5()
    return importlib.reload(mod)

def _find_ckpt(default_name: str = "models_t5_umt5-xxl-enc-bf16.pth") -> Optional[str]:
    env = os.environ.get("WAN_T5_CKPT")
    if env and Path(env).exists():
        return env
    for c in [
        Path.cwd()/default_name,
        Path.cwd().parent/default_name,
        Path("/workspace/models")/default_name,
        Path("/workspace/models/Wan2.2-TI2V-5B")/default_name,
        Path("/workspace/AI-Model-Sandbox/inference/Wan2.2")/default_name,
    ]:
        if c.exists():
            return str(c)
    return None

def build_encoder(use_bridge: bool, text_len: int, dtype: torch.dtype, device: str,
                  t5_ckpt: Optional[str] = None, tokenizer: str = "google/umt5-xxl"):
    os.environ["WAN_USE_BRIDGE"] = "1" if use_bridge else "0"
    t5_mod = _reload_wan_t5()
    kwargs = dict(text_len=text_len, dtype=dtype, device=device)
    if not use_bridge:
        ckpt = t5_ckpt or _find_ckpt()
        if ckpt is None:
            raise SystemExit("Set --t5_ckpt or WAN_T5_CKPT to the UMT5 encoder checkpoint path.")
        kwargs.update(checkpoint_path=ckpt, tokenizer_path=tokenizer)
    try:
        return t5_mod.T5EncoderModel(**kwargs, t5_cpu=(device=="cpu"))
    except TypeError:
        return t5_mod.T5EncoderModel(**kwargs)

@torch.no_grad()
def encode_list(enc, prompts: List[str], device: str, microbatch: int = 64) -> List[torch.Tensor]:
    """Encode a list of prompts in microbatches -> list of [L_i, D] tensors in the same order."""
    out: List[torch.Tensor] = []
    i = 0
    n = len(prompts)
    while i < n:
        mb = prompts[i:i+microbatch]
        lst = enc(mb, device=device)
        assert isinstance(lst, list) and len(lst)==len(mb)
        out.extend([x.float().contiguous() for x in lst])
        i += len(mb)
    return out

# ----------------------- Data plumbing -----------------------

def load_prompts_from_file(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"prompts_file not found: {path}")
    return [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]

def sample_captions_from_jsonl(jsonl_path: str, n: int, seed: int = 0) -> List[str]:
    """Reservoir sample n 'caption' strings from a large JSONL without loading fully."""
    rng = random.Random(seed)
    samples: List[str] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for t, line in enumerate(f, start=1):
            try:
                obj = json.loads(line)
            except Exception:
                continue
            cap = obj.get("caption")
            if not isinstance(cap, str):
                continue
            cap = cap.strip()
            if not cap:
                continue
            if len(samples) < n:
                samples.append(cap)
            else:
                j = rng.randint(1, t)
                if j <= n:
                    samples[j - 1] = cap
    # de-dup preserving order
    seen=set(); out=[]
    for s in samples:
        if s not in seen:
            seen.add(s); out.append(s)
    return out

def safe_stub(prompt: str, maxlen: int = 64) -> str:
    s = prompt.replace("/", "_").replace("\\", "_").replace(":", "_").replace("|","_")
    s = s.replace('"', "'").replace("<","(").replace(">",")")
    if len(s) > maxlen:
        s = s[:maxlen].rstrip()
    return s

def short_hash(prompt: str) -> str:
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12]

# ----------------------- Metrics & plotting -----------------------

def metrics_for_pair(T: torch.Tensor, B: torch.Tensor) -> Dict[str, Any]:
    L = min(T.size(0), B.size(0))
    T = T[:L]; B = B[:L]
    tn = T / (T.norm(dim=-1, keepdim=True)+1e-8)
    bn = B / (B.norm(dim=-1, keepdim=True)+1e-8)
    cos_tok = (tn*bn).sum(dim=-1).cpu()                    # [L]
    norm_ratio = (B.norm(dim=-1)/(T.norm(dim=-1)+1e-8)).cpu()   # [L]

    mu_T = T.mean(dim=0); mu_B = B.mean(dim=0)
    std_T = T.std(dim=0, unbiased=False)+1e-8
    std_B = B.std(dim=0, unbiased=False)+1e-8
    std_ratio = (std_B/std_T).cpu()                        # [D]
    mu_delta = (mu_B - mu_T).cpu()                         # [D]

    seq_cos = F.cosine_similarity(T.flatten(), B.flatten(), dim=0).item()
    mu_cos  = F.cosine_similarity(mu_T, mu_B, dim=0).item()

    return dict(
        L=int(L), D=int(T.size(1)),
        cos_tok=cos_tok, norm_ratio=norm_ratio,
        std_ratio=std_ratio, mu_delta=mu_delta,
        seq_cos=float(seq_cos), mu_cos=float(mu_cos),
        token_cos_mean=float(cos_tok.mean().item()),
        token_cos_median=float(cos_tok.median().item()),
        norm_ratio_mean=float(norm_ratio.mean().item()),
        std_ratio_median=float(std_ratio.median().item()),
        mu_abs_delta=float(mu_delta.abs().mean().item()),
        global_scale_suggestion=float(1.0/max(1e-8, norm_ratio.mean().item())),
    )

def plot_token_curve(y: np.ndarray, title: str, out: Path, ylabel: str):
    plt.figure()
    plt.plot(np.arange(len(y)), y)
    plt.xlabel("token position")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

def plot_hist(y: np.ndarray, title: str, out: Path, xlabel: str, bins: int = 80):
    plt.figure()
    plt.hist(y, bins=bins)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

# ----------------------- Main -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts_file", type=str, default=None, help="One prompt per line.")
    ap.add_argument("--captions_jsonl", type=str, default=None, help="JSONL with a 'caption' field.")
    ap.add_argument("--sample_n", type=int, default=0, help="If using JSONL, how many to sample.")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--text_len", type=int, default=512)
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16","fp16","fp32"])
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--t5_ckpt", type=str, default=None, help="UMT5 teacher checkpoint; else WAN_T5_CKPT/env search.")
    ap.add_argument("--tokenizer", type=str, default="google/umt5-xxl")
    ap.add_argument("--save_raw", action="store_true", help="Save teacher/bridge .npy per prompt")
    ap.add_argument("--no_plots", action="store_true", help="Skip PNG plots.")
    ap.add_argument("--manifest_name", type=str, default="manifest_diag")
    args = ap.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    out_root = Path(args.outdir); out_root.mkdir(parents=True, exist_ok=True)

    # Build prompt list
    prompts: List[str] = []
    if args.prompts_file:
        prompts = load_prompts_from_file(args.prompts_file)
    elif args.captions_jsonl:
        n = args.sample_n if args.sample_n > 0 else 200
        prompts = sample_captions_from_jsonl(args.captions_jsonl, n=n, seed=args.seed)
    else:
        raise SystemExit("Provide --prompts_file or --captions_jsonl")
    if len(prompts)==0:
        raise SystemExit("No prompts found.")

    # Save prompts for provenance
    (out_root/"prompts.txt").write_text("\n".join(prompts))

    # Instantiate teacher and bridge ONCE
    teacher = build_encoder(False, args.text_len, dtype, args.device, t5_ckpt=args.t5_ckpt, tokenizer=args.tokenizer)
    bridge  = build_encoder(True,  args.text_len, dtype, args.device)

    # Encode all prompts with both encoders using microbatches
    print(f"[diag-batch] encoding teacher ... ({len(prompts)} prompts, batch={args.batch_size})")
    T_all = encode_list(teacher, prompts, device=args.device, microbatch=args.batch_size)
    print(f"[diag-batch] encoding bridge ...  ({len(prompts)} prompts, batch={args.batch_size})")
    B_all = encode_list(bridge,  prompts, device=args.device, microbatch=args.batch_size)

    # Prepare manifest writers
    import csv, pandas as pd, time
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    out_csv = out_root / f"{args.manifest_name}.csv"
    out_jsonl = out_root / f"{args.manifest_name}.jsonl"
    fcsv = open(out_csv, "w", newline=""); writer = csv.writer(fcsv)
    writer.writerow([
        "ts","prompt","hash","L","D","seq_cos","mu_cos",
        "token_cos_mean","token_cos_median",
        "norm_ratio_mean","std_ratio_median","mu_abs_delta","global_scale_suggestion",
        "folder","per_token_csv","per_dim_csv","token_cosine_curve","token_norm_ratio_curve","std_ratio_hist"
    ])
    fjsonl = open(out_jsonl, "w")

    # Per-prompt processing
    summary_rows = []
    agg = []  # for overall stats
    for idx, (p, T, B) in enumerate(zip(prompts, T_all, B_all)):
        h = short_hash(p)
        stub = safe_stub(p, 64)
        pdir = out_root / f"{idx:04d}__{h}"
        pdir.mkdir(exist_ok=True, parents=True)

        m = metrics_for_pair(T, B)
        agg.append(m)

        # Save CSVs
        import pandas as pd
        tok_df = pd.DataFrame({
            "token_index": np.arange(m["L"]),
            "token_cos": m["cos_tok"].numpy(),
            "norm_ratio": m["norm_ratio"].numpy(),
        })
        tok_csv = pdir / "per_token_metrics.csv"
        tok_df.to_csv(tok_csv, index=False)

        dim_df = pd.DataFrame({
            "dim": np.arange(m["D"]),
            "std_ratio": m["std_ratio"].numpy(),
            "mu_delta": m["mu_delta"].numpy(),
        })
        dim_csv = pdir / "per_dim_metrics.csv"
        dim_df.to_csv(dim_csv, index=False)

        # Plots
        tok_cos_png = pdir/"token_cosine_curve.png"
        tok_norm_png = pdir/"token_norm_ratio_curve.png"
        std_hist_png = pdir/"std_ratio_hist.png"
        if not args.no_plots:
            plot_token_curve(tok_df["token_cos"].to_numpy(), f"Per-token cosine — {stub}", tok_cos_png, "cosine")
            plot_token_curve(tok_df["norm_ratio"].to_numpy(), f"Per-token norm ratio (bridge/teacher) — {stub}", tok_norm_png, "ratio")
            plot_hist(dim_df["std_ratio"].to_numpy(), f"Per-dim std ratio (bridge/teacher) — {stub}", std_hist_png, "std_ratio")

        # Optional raw tensors
        if args.save_raw:
            np.save(pdir/"teacher_tokens.npy", T.cpu().numpy())
            np.save(pdir/"bridge_tokens.npy",  B.cpu().numpy())

        # Small per-prompt JSON
        rep = {
            "prompt": p,
            "hash": h,
            "metrics": {
                k: (float(v) if isinstance(v, (int,float)) else None)
                for k,v in m.items()
                if k in ["L","D","seq_cos","mu_cos","token_cos_mean","token_cos_median","norm_ratio_mean","std_ratio_median","mu_abs_delta","global_scale_suggestion"]
            },
            "files": {
                "per_token_metrics.csv": str(tok_csv),
                "per_dim_metrics.csv": str(dim_csv),
                "token_cosine_curve.png": str(tok_cos_png),
                "token_norm_ratio_curve.png": str(tok_norm_png),
                "std_ratio_hist.png": str(std_hist_png),
            }
        }
        (pdir/"report.json").write_text(json.dumps(rep, indent=2))

        writer.writerow([
            ts, p, h, m["L"], m["D"], m["seq_cos"], m["mu_cos"],
            m["token_cos_mean"], m["token_cos_median"],
            m["norm_ratio_mean"], m["std_ratio_median"], m["mu_abs_delta"], m["global_scale_suggestion"],
            str(pdir), str(tok_csv), str(dim_csv), str(tok_cos_png), str(tok_norm_png), str(std_hist_png)
        ]); fcsv.flush()

        fjsonl.write(json.dumps(rep)+"\n"); fjsonl.flush()

    fcsv.close(); fjsonl.close()

    # Overall summary
    if len(agg) > 0:
        import statistics as st
        def _avg(key): return float(sum(a[key] for a in agg) / len(agg))
        def _med(key): return float(st.median([a[key] for a in agg]))
        overall = {
            "n_prompts": len(agg),
            "seq_cos_mean": _avg("seq_cos"),
            "token_cos_mean_mean": _avg("token_cos_mean"),
            "token_cos_median_median": _med("token_cos_median"),
            "norm_ratio_mean_mean": _avg("norm_ratio_mean"),
            "std_ratio_median_median": _med("std_ratio_median"),
            "mu_cos_mean": _avg("mu_cos"),
            "mu_abs_delta_mean": _avg("mu_abs_delta"),
            "global_scale_suggestion_mean": _avg("global_scale_suggestion"),
        }
        (out_root/"overall_summary.json").write_text(json.dumps(overall, indent=2))
        print(json.dumps(overall, indent=2))

    print(f"[diag-batch] wrote manifest:\n  {out_csv}\n  {out_jsonl}\nOutputs under: {out_root}")

if __name__ == "__main__":
    main()
