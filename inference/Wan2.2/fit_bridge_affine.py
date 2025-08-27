
#!/usr/bin/env python3
"""
fit_bridge_affine_progress.py — same as fit_bridge_affine.py, but with live progress logs.

Adds:
  --log_every N      : print "processed/total" every N prompts (default 25)
  --dry_run          : just load prompts and print how many will be processed, then exit

Everything else matches the original script.
"""

from __future__ import annotations
import os, json, argparse, importlib, random, time
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import torch

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

def _find_ckpt_from_env_or_common(default_name: str) -> Optional[str]:
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
        ckpt = t5_ckpt or _find_ckpt_from_env_or_common("models_t5_umt5-xxl-enc-bf16.pth")
        if ckpt is None:
            raise SystemExit("Set --t5_ckpt or WAN_T5_CKPT to the UMT5 encoder checkpoint path.")
        kwargs.update(checkpoint_path=ckpt, tokenizer_path=tokenizer)
    try:
        return t5_mod.T5EncoderModel(**kwargs, t5_cpu=(device == "cpu"))
    except TypeError:
        return t5_mod.T5EncoderModel(**kwargs)

@torch.no_grad()
def encode_batch(enc, prompts: List[str], device: str) -> List[torch.Tensor]:
    lst = enc(prompts, device=device)
    assert isinstance(lst, list) and all(torch.is_tensor(x) for x in lst)
    return [x.float().contiguous() for x in lst]

def load_prompts_from_file(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"prompts_file not found: {path}")
    return [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]

def sample_captions_from_jsonl(jsonl_path: str, n: int, seed: int = 0) -> List[str]:
    rng = random.Random(seed)
    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for t, line in enumerate(f, start=1):
            try:
                obj = json.loads(line)
            except Exception:
                continue
            cap = obj.get("caption", None)
            if not cap or not isinstance(cap, str):
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
    seen = set(); out = []
    for s in samples:
        if s not in seen:
            seen.add(s); out.append(s)
    return out

def fit_affine(stats: Dict[str, torch.Tensor], method: str, eps: float, min_scale: float, max_scale: float):
    n = stats["N"].clamp_min(1.0)
    sum_B = stats["sum_B"]; sum_T = stats["sum_T"]
    sum_B2 = stats["sum_B2"]; sum_T2 = stats["sum_T2"]; sum_BT = stats["sum_BT"]

    mean_B = sum_B / n
    mean_T = sum_T / n

    if method == "ols":
        cov_BT = sum_BT - (sum_B * sum_T) / n
        var_B  = sum_B2 - (sum_B * sum_B) / n
        s = cov_BT / (var_B + eps)
        b = mean_T - s * mean_B
    else:
        var_B = (sum_B2 / n) - mean_B * mean_B
        var_T = (sum_T2 / n) - mean_T * mean_T
        std_B = torch.sqrt(torch.clamp(var_B, min=0.0)) + eps
        std_T = torch.sqrt(torch.clamp(var_T, min=0.0))
        s = std_T / std_B
        b = mean_T - s * mean_B

    s = torch.clamp(s, min=min_scale, max=max_scale)
    return s, b

def update_running_stats(stats: Dict[str, torch.Tensor], B: torch.Tensor, T: torch.Tensor):
    sum_B  = B.sum(dim=0)
    sum_T  = T.sum(dim=0)
    sum_B2 = (B * B).sum(dim=0)
    sum_T2 = (T * T).sum(dim=0)
    sum_BT = (B * T).sum(dim=0)
    n      = torch.tensor([B.size(0)], dtype=torch.float32)

    for k, v in dict(sum_B=sum_B, sum_T=sum_T, sum_B2=sum_B2, sum_T2=sum_T2, sum_BT=sum_BT).items():
        stats[k] += v
    stats["N"] += n

def init_stats(D: int, device: torch.device = torch.device("cpu")) -> Dict[str, torch.Tensor]:
    z = torch.zeros(D, dtype=torch.float32, device=device)
    stats = dict(sum_B=z.clone(), sum_T=z.clone(), sum_B2=z.clone(),
                 sum_T2=z.clone(), sum_BT=z.clone(), N=torch.tensor([0.0], dtype=torch.float32, device=device))
    return stats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts_file", type=str, default=None)
    ap.add_argument("--captions_jsonl", type=str, default=None)
    ap.add_argument("--sample_n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bridge_ckpt", type=str, default=os.environ.get("WAN_BRIDGE_CKPT", ""))
    ap.add_argument("--llm_dir", type=str, default=os.environ.get("WAN_BRIDGE_LLM_DIR", "/workspace/models/MythoMax-L2-13B"))
    ap.add_argument("--t5_ckpt", type=str, default=os.environ.get("WAN_T5_CKPT", None))
    ap.add_argument("--out_ckpt", type=str, required=True)
    ap.add_argument("--text_len", type=int, default=512)
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16","fp16","fp32"])
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--method", type=str, default="ols", choices=["ols","moments"])
    ap.add_argument("--min_scale", type=float, default=0.2)
    ap.add_argument("--max_scale", type=float, default=5.0)
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--report", type=str, default=None)
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    if not args.bridge_ckpt or not Path(args.bridge_ckpt).exists():
        raise SystemExit("--bridge_ckpt must point to an existing Bridge checkpoint (.pth).")
    if args.t5_ckpt is None:
        # search common default
        def _find_ckpt(default_name="models_t5_umt5-xxl-enc-bf16.pth"):
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
        args.t5_ckpt = _find_ckpt()
        if args.t5_ckpt is None:
            raise SystemExit("Set --t5_ckpt or WAN_T5_CKPT to teacher UMT5 checkpoint path.")

    # Prompts
    if args.prompts_file:
        prompts = [ln.strip() for ln in Path(args.prompts_file).read_text().splitlines() if ln.strip()]
    elif args.captions_jsonl:
        # Reservoir sample 'caption' fields
        rng = random.Random(args.seed)
        prompts = []
        with open(args.captions_jsonl, "r", encoding="utf-8") as f:
            for t, line in enumerate(f, start=1):
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                cap = obj.get("caption", None)
                if not cap or not isinstance(cap, str):
                    continue
                cap = cap.strip()
                if not cap:
                    continue
                if len(prompts) < args.sample_n:
                    prompts.append(cap)
                else:
                    j = rng.randint(1, t)
                    if j <= args.sample_n:
                        prompts[j - 1] = cap
        # dedup
        seen=set(); prompts=[(seen.add(p) or p) for p in prompts if p not in seen]
    else:
        raise SystemExit("Provide either --prompts_file or --captions_jsonl.")

    total = len(prompts)
    if args.dry_run:
        print(f"[affine-fit] DRY RUN — would process {total} prompts (batch_size={args.batch_size})")
        return

    # Build encoders
    os.environ["WAN_BRIDGE_LLM_DIR"] = args.llm_dir
    os.environ["WAN_BRIDGE_CKPT"] = args.bridge_ckpt
    os.environ["WAN_BRIDGE_DTYPE"] = "bf16" if dtype == torch.bfloat16 else ("fp16" if dtype == torch.float16 else "fp32")

    teacher = build_encoder(False, args.text_len, dtype, args.device, t5_ckpt=args.t5_ckpt)
    bridge  = build_encoder(True,  args.text_len, dtype, args.device)

    # Accumulate stats with progress
    print(f"[affine-fit] prompts={total} method={args.method} device={args.device} dtype={dtype} batch_size={args.batch_size}")
    stats = None
    D = 4096
    cursor = 0
    t0 = time.time()
    printed = 0
    while cursor < total:
        batch = prompts[cursor: cursor + args.batch_size]
        cursor += len(batch)
        T_list = encode_batch(teacher, batch, args.device)
        B_list = encode_batch(bridge,  batch, args.device)
        for T, B in zip(T_list, B_list):
            L = min(T.size(0), B.size(0))
            if L <= 0: 
                continue
            T = T[:L]; B = B[:L]
            if stats is None:
                D = T.size(1)
                stats = {
                    "sum_B": torch.zeros(D),
                    "sum_T": torch.zeros(D),
                    "sum_B2": torch.zeros(D),
                    "sum_T2": torch.zeros(D),
                    "sum_BT": torch.zeros(D),
                    "N": torch.tensor([0.0], dtype=torch.float32),
                }
            update_running_stats(stats, B.cpu(), T.cpu())

        processed = min(cursor, total)
        if processed - printed >= args.log_every or processed == total:
            dt = time.time() - t0
            print(f"[affine-fit] progress {processed}/{total}  ({processed/total:.1%})  elapsed={dt:.1f}s")
            printed = processed

    if stats is None:
        raise SystemExit("No tokens processed — something went wrong.")

    # Fit
    s, b = fit_affine(stats, args.method, args.eps, args.min_scale, args.max_scale)
    print(f"[affine-fit] scale stats: mean={s.mean():.3f} median={s.median():.3f} min={s.min():.3f} max={s.max():.3f}")

    # Load and write ckpt
    ckpt_path = Path(args.bridge_ckpt)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    def _get_sd(ckpt_obj):
        if isinstance(ckpt_obj, dict) and "bridge" in ckpt_obj and isinstance(ckpt_obj["bridge"], dict):
            return ckpt_obj["bridge"], True
        return ckpt_obj, False

    sd, has_bridge_key = _get_sd(ckpt)
    scale_key = next((k for k in sd.keys() if k.endswith("out_scale")), None)
    shift_key = next((k for k in sd.keys() if k.endswith("out_shift")), None)
    if scale_key is None or shift_key is None:
        raise SystemExit("Could not find 'out_scale'/'out_shift' in bridge checkpoint.")

    old_scale = sd[scale_key]; old_shift = sd[shift_key]
    if old_scale.shape[-1] != s.numel():
        raise SystemExit(f"Dim mismatch: ckpt out_scale has D={old_scale.shape[-1]} but fitted has D={s.numel()}.")
    new_scale = s.view(1,1,-1).to(dtype=old_scale.dtype).contiguous()
    new_shift = b.view(1,1,-1).to(dtype=old_shift.dtype).contiguous()
    sd[scale_key] = new_scale
    sd[shift_key] = new_shift
    if has_bridge_key:
        ckpt["bridge"] = sd

    out_path = Path(args.out_ckpt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_path)
    print(f"[affine-fit] wrote calibrated bridge → {out_path}")

if __name__ == "__main__":
    main()
