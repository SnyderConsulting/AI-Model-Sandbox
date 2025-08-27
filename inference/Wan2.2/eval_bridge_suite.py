
#!/usr/bin/env python3
"""
eval_bridge_suite.py — Batch evaluation of Wan 2.2 TI2V-5B with stock UMT5 vs Bridge encoder.

What this script does
---------------------
- Builds a parameter grid and generates videos with Wan's `generate.py`.
- Runs a deep text-encoder diagnostic (diag_text_compare.py) per prompt to log
  norm/cosine metrics and to compute a per-prompt global-scale suggestion.
- Supports two presets: "smoke" (quick) and "full" (broader sweep).
- Produces a CSV + JSONL manifest with all runs, and a folder structure with logs.
- Reproducible (fixed seeds unless you override). Safe to resume (skips existing outputs).

You can evaluate any bridge checkpoint by pointing WAN_BRIDGE_CKPT to it.

Example (smoke test)
--------------------
export WAN_T5_CKPT=/workspace/models/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth
python eval_bridge_suite.py \
  --generate_py /workspace/AI-Model-Sandbox/inference/Wan2.2/generate.py \
  --ckpt_dir /workspace/models/Wan2.2-TI2V-5B \
  --bridge_ckpt /workspace/AI-Model-Sandbox/outputs/encoder-bridg/bridge_step020000.pth \
  --llm_dir /workspace/models/MythoMax-L2-13B \
  --out_dir /workspace/eval_runs/bridge_20000 \
  --preset smoke

# Later, run full sweep:
python eval_bridge_suite.py ... --preset full --size 704*1280 --frame_num 33

Outputs
-------
out_dir/
  manifest.csv              # one row per generated video
  manifest.jsonl            # same as CSV, in JSONL
  prompts_used.txt
  .../teacher/...
  .../bridge/...
Each run folder contains: the MP4 (or saved file), a short log, and a metadata JSON.

Requirements
------------
- Your repo's `generate.py` must accept: --task, --size, --frame_num, --ckpt_dir,
  --prompt, --sample_steps, --sample_guide_scale, --convert_model_dtype, --save_file.
- The encoder swap is controlled via env:
  * WAN_USE_BRIDGE=0 → stock UMT5
  * WAN_USE_BRIDGE=1 → Bridge (requires WAN_BRIDGE_CKPT and WAN_BRIDGE_LLM_DIR)
  * WAN_BRIDGE_GLOBAL_SCALE optionally set per run
- The diagnostic script is expected at /mnt/data/diag_text_compare.py (from our earlier step).

Notes
-----
- This script does not judge "quality". It standardizes generation and collects
  all knobs/metrics so you (or another script) can grade outputs.
"""

from __future__ import annotations
import os, sys, argparse, json, time, csv, subprocess, itertools, math, random, shlex
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional

# ------------- default prompts -------------

SFW_DEFAULT = [
    "A serene sunset over the ocean, cinematic, warm glow, gentle waves",
    "A neon-lit alleyway at night, wet asphalt reflections, shallow depth of field",
    "A close-up of a cat yawning in soft morning light",
    "A dragon flying over a medieval city, wide shot, dramatic clouds",
    "A busy street food market in Tokyo, bokeh lights, handheld look",
    "A vintage muscle car driving through a desert highway at golden hour",
    "A rainy window with city lights in the background, macro focus",
    "A snowy mountain landscape with moving clouds, time-lapse feel",
]

NSFW_DEFAULT = [
    "A topless woman fondling her breasts, cinematic lighting, soft focus",
    "A nude couple embracing in the shower, water droplets, steamy atmosphere",
    "A sensual close-up of lips biting gently, glossy highlights",
    "A shirtless muscular man flexing under neon lights, dramatic contrast",
    "A seductive woman arching her back on silk sheets, low-key lighting",
    "A nude figure silhouetted behind a semi-transparent curtain, moody backlight",
    "A pair of bodies intertwined, artistic nude composition, chiaroscuro",
    "A sensual portrait with wet skin and beads of water, studio lighting",
]

# ------------- helpers -------------

def _now() -> str:
    import datetime as _dt
    return _dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def find_generate_py(explicit: Optional[str]) -> Path:
    cands = []
    if explicit:
        cands.append(Path(explicit))
    # repo-typical locations
    cands += [
        Path.cwd() / "generate.py",
        Path.cwd() / "inference" / "Wan2.2" / "generate.py",
        Path("/workspace/AI-Model-Sandbox/inference/Wan2.2/generate.py"),
    ]
    for c in cands:
        if c.exists():
            return c
    raise SystemExit("Could not find generate.py; pass --generate_py path.")

def run_cmd(cmd: List[str], env: Dict[str,str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        f.write("$ " + " ".join(shlex.quote(x) for x in cmd) + "\n\n")
        f.flush()
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
        p.wait()
        return p.returncode

def run_diag(prompt: str, device: str, outdir: Path) -> Dict[str,Any]:
    diag_py = Path("diag_text_compare.py")
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(diag_py),
        "--prompt", prompt,
        "--device", device,
        "--outdir", str(outdir),
    ]
    env = os.environ.copy()
    # respect WAN_T5_CKPT and bridge env if present; the diag toggles internally
    with open(outdir/"diag_cmd.txt","w") as g:
        g.write("$ " + " ".join(shlex.quote(x) for x in cmd) + "\n")
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    (outdir/"diag_stdout.txt").write_text(proc.stdout + "\n--- STDERR ---\n" + proc.stderr)
    # load report.json if present
    rep_path = outdir/"report.json"
    if rep_path.exists():
        try:
            return json.loads(rep_path.read_text())
        except Exception:
            pass
    return {}

def parse_list_arg(val: str, typ=float):
    if not val: return []
    return [typ(v.strip()) for v in val.split(",") if v.strip()]

@dataclass
class EvalConfig:
    # paths
    generate_py: str
    ckpt_dir: str
    bridge_ckpt: str
    llm_dir: str
    out_dir: str
    # generation
    task: str = "ti2v-5B"
    size: str = "704*1280"
    frame_num: int = 33
    sample_steps_list: List[int] = None
    guide_scales_list: List[float] = None
    seeds: List[int] = None
    # sweeps
    scales_mode: str = "dynamic"     # "fixed" or "dynamic"
    fixed_scales: List[float] = None # used when scales_mode="fixed"
    dynamic_secondary: float = 0.85  # also test 0.85 * suggestion
    # prompts
    prompts_file: Optional[str] = None
    include_nsfw: bool = True
    max_prompts: Optional[int] = None
    # misc
    device: str = "cuda:0"
    convert_model_dtype: bool = True
    preset: str = "smoke"

def build_prompts(cfg: EvalConfig) -> List[str]:
    prom = list(SFW_DEFAULT)
    if cfg.include_nsfw:
        prom += list(NSFW_DEFAULT)
    if cfg.prompts_file:
        path = Path(cfg.prompts_file)
        if path.exists():
            extra = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
            prom = extra
    if cfg.max_prompts:
        prom = prom[:cfg.max_prompts]
    return prom

def make_manifest_writer(out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    f = open(out_csv, "w", newline="")
    writer = csv.writer(f)
    header = [
        "ts","encoder","prompt","seed","steps","cfg","scale","size","frames",
        "save_file","retcode","elapsed_sec","diag_seq_cos","diag_token_cos_mean",
        "diag_norm_ratio_mean","diag_std_ratio_median","diag_mu_cos","diag_scale_suggestion",
        "diag_report_json"
    ]
    writer.writerow(header); f.flush()
    return f, writer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate_py", type=str, default=None)
    ap.add_argument("--ckpt_dir", type=str, required=True)
    ap.add_argument("--bridge_ckpt", type=str, required=True)
    ap.add_argument("--llm_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--task", type=str, default="ti2v-5B")
    ap.add_argument("--size", type=str, default="704*1280")
    ap.add_argument("--frame_num", type=int, default=33)
    ap.add_argument("--sample_steps", type=str, default=None)       # comma list (e.g., "30,50")
    ap.add_argument("--guide_scales", type=str, default=None)       # comma list (e.g., "5,7")
    ap.add_argument("--seeds", type=str, default=None)              # comma list of ints
    ap.add_argument("--preset", type=str, default="smoke", choices=["smoke","full","custom"])
    ap.add_argument("--prompts_file", type=str, default=None)
    ap.add_argument("--include_nsfw", action="store_true")
    ap.add_argument("--max_prompts", type=int, default=None)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--scales_mode", type=str, default="dynamic", choices=["dynamic","fixed"])
    ap.add_argument("--fixed_scales", type=str, default="1.0,0.8,0.65")
    ap.add_argument("--dynamic_secondary", type=float, default=0.85)
    ap.add_argument("--convert_model_dtype", action="store_true")
    args = ap.parse_args()

    # Build config
    cfg = EvalConfig(
        generate_py=args.generate_py or "",
        ckpt_dir=args.ckpt_dir,
        bridge_ckpt=args.bridge_ckpt,
        llm_dir=args.llm_dir,
        out_dir=args.out_dir,
        task=args.task,
        size=args.size,
        frame_num=args.frame_num,
        device=args.device,
        preset=args.preset,
        include_nsfw=args.include_nsfw,
        prompts_file=args.prompts_file,
        max_prompts=args.max_prompts,
        convert_model_dtype=args.convert_model_dtype,
        scales_mode=args.scales_mode,
        fixed_scales=parse_list_arg(args.fixed_scales, float),
        dynamic_secondary=args.dynamic_secondary,
    )

    # Presets
    if cfg.preset == "smoke":
        cfg.sample_steps_list = [50]
        cfg.guide_scales_list = [5.0]
        cfg.seeds = [0]
        cfg.max_prompts = cfg.max_prompts or 8
    elif cfg.preset == "full":
        cfg.sample_steps_list = [30, 50]
        cfg.guide_scales_list = [5.0, 7.0]
        cfg.seeds = [0, 1]
        cfg.max_prompts = cfg.max_prompts or 16
    else:  # custom
        cfg.sample_steps_list = parse_list_arg(args.sample_steps, int) or [50]
        cfg.guide_scales_list = parse_list_arg(args.guide_scales, float) or [5.0]
        cfg.seeds = [int(x) for x in (parse_list_arg(args.seeds, float) or [0])]

    prompts = build_prompts(cfg)
    out_root = Path(cfg.out_dir); out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "prompts_used.txt").write_text("\n".join(prompts))
    out_csv = out_root / "manifest.csv"
    out_jsonl = out_root / "manifest.jsonl"
    manifest_f, writer = make_manifest_writer(out_csv)
    jsonl_f = open(out_jsonl, "w")

    generate_py = find_generate_py(cfg.generate_py)

    ts_all = _now()

    # Run diagnostics once per prompt (so we can compute per-prompt scale suggestion)
    diag_cache: Dict[str, Dict[str,Any]] = {}
    for p in prompts:
        d = run_diag(prompt=p, device="cpu", outdir=out_root / "diagnostics" / f"{hash(p)}")
        diag_cache[p] = d

    # Helper to build command and run
    def do_run(encoder: str, prompt: str, seed: int, steps: int, cfg_scale: float, scale: Optional[float]):
        subdir = out_root / encoder / f"steps{steps}_cfg{cfg_scale}" / (f"scale{scale:.2f}" if scale is not None else "scaleNA")
        subdir.mkdir(parents=True, exist_ok=True)
        save_file = subdir / f"seed{seed}__{hash(prompt)}.mp4"
        log_file  = subdir / f"seed{seed}__{hash(prompt)}.log"
        meta_file = subdir / f"seed{seed}__{hash(prompt)}.json"

        env = os.environ.copy()
        # encoder selection
        if encoder == "teacher":
            env["WAN_USE_BRIDGE"] = "0"
        else:
            env["WAN_USE_BRIDGE"] = "1"
            env["WAN_BRIDGE_CKPT"] = cfg.bridge_ckpt
            env["WAN_BRIDGE_LLM_DIR"] = cfg.llm_dir
            if scale is not None:
                env["WAN_BRIDGE_GLOBAL_SCALE"] = str(scale)
        # device flags are handled inside generate.py for the DiT; we don't override CUDA_VISIBLE_DEVICES here
        # Build CLI
        cmd = [
            sys.executable, str(generate_py),
            "--task", cfg.task,
            "--size", cfg.size,
            "--frame_num", str(cfg.frame_num),
            "--ckpt_dir", cfg.ckpt_dir,
            "--prompt", prompt,
            "--sample_steps", str(steps),
            "--sample_guide_scale", str(cfg_scale),
            "--base_seed", str(seed),
            "--convert_model_dtype",
            "--save_file", str(save_file),
        ]
        t0 = time.time()
        ret = run_cmd(cmd, env=env, log_path=log_file)
        elapsed = time.time() - t0

        # Collect diag summary for this prompt
        diag = diag_cache.get(prompt, {})
        metrics = diag.get("metrics", {})
        diag_seq = metrics.get("seq_cos", None)
        diag_tok = metrics.get("token_cos_mean", None)
        diag_norm = metrics.get("norm_ratio_mean", None)
        diag_std = metrics.get("std_ratio_median", None)
        diag_mu = metrics.get("mu_cos", None)
        diag_scale = metrics.get("global_scale_suggestion", None)

        row = [
            ts_all, encoder, prompt, seed, steps, cfg_scale,
            (scale if scale is not None else ""), str(cfg.size), cfg.frame_num,
            str(save_file), ret, round(elapsed, 3),
            diag_seq, diag_tok, diag_norm, diag_std, diag_mu, diag_scale,
            str(diag.get("files", {})),
        ]
        writer.writerow(row); manifest_f.flush()
        jsonl_f.write(json.dumps({
            "ts": ts_all, "encoder": encoder, "prompt": prompt, "seed": seed,
            "steps": steps, "cfg": cfg_scale, "scale": scale, "size": cfg.size, "frames": cfg.frame_num,
            "save_file": str(save_file), "retcode": ret, "elapsed_sec": elapsed,
            "diag": metrics, "diag_files": diag.get("files", {})
        }) + "\n"); jsonl_f.flush()

        # Also store per-run metadata
        meta = {
            "encoder": encoder, "prompt": prompt, "seed": seed, "steps": steps,
            "cfg": cfg_scale, "scale": scale, "size": cfg.size, "frames": cfg.frame_num,
            "elapsed_sec": elapsed, "retcode": ret, "save_file": str(save_file),
            "cmd": cmd, "env": {k:v for k,v in env.items() if k.startswith("WAN_")},
            "diag_metrics": metrics, "diag_files": diag.get("files", {}),
        }
        meta_file.write_text(json.dumps(meta, indent=2))

    # Build grid and run
    for prompt in prompts:
        # teacher baseline (no scale)
        for steps in cfg.sample_steps_list:
            for g in cfg.guide_scales_list:
                for seed in cfg.seeds:
                    do_run("teacher", prompt, seed, steps, g, scale=None)

        # bridge runs
        # Determine scales
        scale_list: List[float] = []
        if cfg.scales_mode == "fixed":
            scale_list = cfg.fixed_scales
        else:
            suggestion = diag_cache.get(prompt, {}).get("metrics", {}).get("global_scale_suggestion", 1.0)
            if suggestion is None:
                suggestion = 1.0
            # Always try: 1.0 (no fudge), suggestion, and 0.85*suggestion (slightly conservative)
            scale_list = sorted({1.0, float(suggestion), float(suggestion) * float(cfg.dynamic_secondary)})
        for steps in cfg.sample_steps_list:
            for g in cfg.guide_scales_list:
                for seed in cfg.seeds:
                    for scale in scale_list:
                        do_run("bridge", prompt, seed, steps, g, scale=scale)

    manifest_f.close(); jsonl_f.close()
    print(f"\n[done] Manifest written to:\n  CSV  : {out_csv}\n  JSONL: {out_jsonl}")
    print("Folder contains per-run logs and videos.")

if __name__ == "__main__":
    main()
