#!/usr/bin/env python3
import argparse, json, re, sys
from pathlib import Path
from collections import defaultdict, Counter

# --- Minimal .safetensors header reader (no tensor loading) ---
def read_safetensors_header(p: Path):
    with p.open("rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
        header = f.read(header_len)
    j = json.loads(header.decode("utf-8"))
    meta = j.get("__metadata__", {})
    tensors = {k: v for k, v in j.items() if k != "__metadata__"}
    # tensors[k] = {"dtype":"F16","shape":[...],"data_offsets":[start,end]}
    return meta, tensors

STRIP_PREFIX = re.compile(r"^model\.diffusion_model\.")

ATTN_PAT = re.compile(
    r"^(?P<prefix>.*?blocks\.(?P<block>\d+)\.)"
    r"(?P<which>(self_attn|cross_attn))\.(?P<comp>q|k|v|o)\.weight$"
)
FFN_PAT = re.compile(
    r"^(?P<prefix>.*?blocks\.(?P<block>\d+)\.)ffn\.(?P<which>0|2)\.weight$"
)
I2V_HINT_PAT = re.compile(r"\.cross_attn\.k_img\.weight$")

def scan_checkpoint(root: Path, exclude_patterns=("adapter_model",)):
    root = root.resolve()
    ckpt_name = root.name
    # find config.json candidates
    config_paths = []
    if root.is_dir():
        # common places
        for cand in [
            root / "config.json",
            root / "low_noise_model" / "config.json",
            root / "high_noise_model" / "config.json",
        ]:
            if cand.exists():
                config_paths.append(cand)
        # fallback: any config.json within depth 2
        if not config_paths:
            for cand in root.rglob("config.json"):
                config_paths.append(cand)
    else:
        # single file: config likely sits next to parent dir
        p = root.parent / "config.json"
        if p.exists():
            config_paths.append(p)

    # load config(s) if present
    cfgs = []
    for c in config_paths:
        try:
            cfgs.append(json.loads(c.read_text()))
        except Exception:
            pass
    # pick the "most specific" config (prefer non-empty, last one)
    cfg = cfgs[-1] if cfgs else {}

    # collect safetensors files
    files = []
    if root.is_file() and root.suffix == ".safetensors":
        files = [root]
    elif root.is_dir():
        files = sorted(root.rglob("*.safetensors"))
    else:
        raise FileNotFoundError(f"{root} is not a .safetensors file or directory")

    # optional exclude
    def keep(path: Path):
        s = str(path.as_posix()).lower()
        return not any(pat in s for pat in exclude_patterns)

    files = [f for f in files if keep(f)]
    if not files:
        raise RuntimeError(f"No .safetensors found under {root}")

    # scan
    keymap = {}  # (expert_tag, key) -> (shape, dtype, file)
    i2v_hints = False
    experts_seen = set()

    for f in files:
        # best-effort expert tag from path
        tag = None
        lower = f.as_posix().lower()
        if "high_noise" in lower:
            tag = "high_noise"
        elif "low_noise" in lower:
            tag = "low_noise"
        elif "expert" in lower or "experts" in lower:
            tag = "expert_path"
        else:
            tag = "default"

        meta, tensors = read_safetensors_header(f)
        for k, v in tensors.items():
            k2 = STRIP_PREFIX.sub("", k)
            shape = tuple(v.get("shape", []))
            dtype = v.get("dtype", "unknown")
            if I2V_HINT_PAT.search(k2):
                i2v_hints = True
            # heuristic: some drops might actually encode expert in the key
            key_tag = tag
            if "expert" in k2 or "experts" in k2:
                key_tag = "expert_key"
            experts_seen.add(key_tag)
            keymap[(key_tag, k2)] = (shape, dtype, f)

    # block / module summaries
    blocks = defaultdict(lambda: {
        "self_attn": {"q": None, "k": None, "v": None, "o": None},
        "cross_attn": {"q": None, "k": None, "v": None, "o": None},
        "ffn": {"0": None, "2": None},
    })
    attn_counts = Counter()
    ffn_counts = Counter()

    for (tag, key), (shape, dtype, f) in keymap.items():
        m = ATTN_PAT.match(key)
        if m:
            b = int(m["block"])
            which = m["which"]
            comp = m["comp"]
            blocks[b][which][comp] = shape
            attn_counts[(which, comp)] += 1
            continue
        m = FFN_PAT.match(key)
        if m:
            b = int(m["block"])
            which = m["which"]
            blocks[b]["ffn"][which] = shape
            ffn_counts[which] += 1

    # guess num_blocks / d_model
    if blocks:
        max_b = max(blocks.keys())
        num_blocks = max_b + 1
    else:
        num_blocks = None

    # guess d_model from any attn/ffn shape
    d_model = None
    for b in sorted(blocks.keys()):
        for which in ("self_attn", "cross_attn"):
            for comp in ("q", "k", "v", "o"):
                shp = blocks[b][which][comp]
                if shp and len(shp) == 2:
                    d_model = shp[0]
                    break
            if d_model:
                break
        if d_model:
            break
    # fallback: ffn.0 weight is (4*d_model, d_model) typically
    if d_model is None:
        for b in sorted(blocks.keys()):
            shp = blocks[b]["ffn"]["0"]
            if shp and len(shp) == 2:
                d_model = shp[1]
                break

    # try to read config hints
    summary = {
        "ckpt_name": ckpt_name,
        "root": str(root),
        "model_type_cfg": cfg.get("model_type"),
        "dim_cfg": cfg.get("dim"),
        "num_layers_cfg": cfg.get("num_layers") or cfg.get("layers"),
        "text_len_cfg": cfg.get("text_len"),
        "num_heads_cfg": cfg.get("num_heads") or cfg.get("heads"),
        "experts_detected": sorted(experts_seen),
        "i2v_keys_present": i2v_hints,
        "num_blocks_guessed": num_blocks,
        "d_model_guessed": d_model,
        "attn_key_counts": {f"{k[0]}.{k[1]}": v for k, v in attn_counts.items()},
        "ffn_key_counts": {k: v for k, v in ffn_counts.items()},
    }

    return keymap, blocks, summary


def write_csv_lines(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="+", help="Path(s) to Wan checkpoints or .safetensors files")
    ap.add_argument("-o", "--outdir", type=Path, required=True, help="Output directory")
    ap.add_argument("--include-keys", default="blocks\\.", help="Regex to include keys (default: blocks\\.)")
    ap.add_argument("--exclude", nargs="*", default=["adapter_model"], help="Exclude paths containing these fragments")
    args = ap.parse_args()

    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    global_summary = []
    for c in args.ckpt:
        root = Path(c)
        keymap, blocks, summary = scan_checkpoint(root, exclude_patterns=tuple(args.exclude))
        ckpt_tag = summary["ckpt_name"]

        # 1) Dump key list
        rows = []
        for (expert, key), (shape, dtype, f) in sorted(keymap.items(), key=lambda kv: kv[0][1]):
            if not re.search(args.include_keys, key):
                continue
            rows.append([ckpt_tag, expert, key, f"{list(shape)}", dtype, f.name])
        write_csv_lines(outdir / f"{ckpt_tag}__keys.csv",
                        ["ckpt", "expert", "key", "shape", "dtype", "file"], rows)

        # 2) Dump block summary
        brow = []
        for b in range(0, max(blocks.keys()) + 1 if blocks else 0):
            blk = blocks.get(b, None)
            if not blk:
                continue
            def s(x): return "" if x is None else "x".join(map(str, x))
            brow.append([
                ckpt_tag, b,
                s(blk["self_attn"]["q"]), s(blk["self_attn"]["k"]), s(blk["self_attn"]["v"]), s(blk["self_attn"]["o"]),
                s(blk["cross_attn"]["q"]), s(blk["cross_attn"]["k"]), s(blk["cross_attn"]["v"]), s(blk["cross_attn"]["o"]),
                s(blk["ffn"]["0"]), s(blk["ffn"]["2"]),
            ])
        write_csv_lines(outdir / f"{ckpt_tag}__blocks_summary.csv",
                        ["ckpt","block",
                         "sa.q","sa.k","sa.v","sa.o",
                         "ca.q","ca.k","ca.v","ca.o",
                         "ffn.0","ffn.2"], brow)

        # 3) Write model summary JSON
        (outdir / f"{ckpt_tag}__model_summary.json").write_text(json.dumps(summary, indent=2))
        global_summary.append(summary)

    (outdir / "ALL__model_summaries.json").write_text(json.dumps(global_summary, indent=2))
    print(f"Done. Wrote outputs to: {outdir}")

if __name__ == "__main__":
    main()
