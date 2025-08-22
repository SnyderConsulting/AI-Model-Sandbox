#!/usr/bin/env python3
import argparse, csv, re
from pathlib import Path
from collections import defaultdict

def load_keys_csv(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            # normalize shape string like "[1536, 1536]" -> tuple
            shp = tuple(int(x) for x in re.findall(r"\d+", r["shape"]))
            rows.append((r["key"], shp))
    return dict(rows)

def build_from_ckpt_dump_dir(d: Path):
    # find the single keys csv in a dump dir
    csvs = sorted(d.glob("*__keys.csv"))
    if len(csvs) == 0:
        raise RuntimeError(f"No __keys.csv found under {d}")
    # if there are multiple (e.g., MoE split), merge
    merged = {}
    for c in csvs:
        merged.update(load_keys_csv(c))
    return merged

def detect_source(s: str):
    p = Path(s)
    if p.is_file() and p.suffix == ".csv":
        return "csv", p
    if p.is_dir():
        # either a dump dir or a checkpoint dir — in both cases we expect keys CSVs if it's a dump dir
        csvs = list(p.glob("*__keys.csv"))
        if csvs:
            return "dumpdir", p
        # raw ckpt dir: user should run inspector first
    raise RuntimeError(f"Unsupported input: {s}. Pass a dump dir (with __keys.csv) or a keys CSV.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a", help="Path to A (dump dir or keys CSV)")
    ap.add_argument("b", help="Path to B (dump dir or keys CSV)")
    args = ap.parse_args()

    typ_a, pa = detect_source(args.a)
    typ_b, pb = detect_source(args.b)

    if typ_a == "csv": A = load_keys_csv(pa)
    else: A = build_from_ckpt_dump_dir(pa)
    if typ_b == "csv": B = load_keys_csv(pb)
    else: B = build_from_ckpt_dump_dir(pb)

    ka, kb = set(A.keys()), set(B.keys())
    only_a = sorted(ka - kb)
    only_b = sorted(kb - ka)

    mismatches = []
    for k in sorted(ka & kb):
        if A[k] != B[k]:
            mismatches.append((k, A[k], B[k]))

    print(f"Total keys A: {len(ka)} | B: {len(kb)}")
    print(f"Only in A: {len(only_a)} | Only in B: {len(only_b)} | Shape mismatches: {len(mismatches)}")

    # quick per-module counts
    mod_re = re.compile(r"(self_attn|cross_attn)\.(q|k|v|o)\.weight$|ffn\.(0|2)\.weight$")
    bucket = defaultdict(int)
    for k, sa, sb in mismatches:
        m = mod_re.search(k)
        tag = m.group(0) if m else "other"
        bucket[tag] += 1

    print("\n— Mismatch buckets —")
    for tag, n in sorted(bucket.items(), key=lambda x: (-x[1], x[0])):
        print(f"{tag:24s} : {n}")

    # print a few examples for each category
    def preview(lst, name):
        print(f"\n— {name} (showing up to 20) —")
        for k in lst[:20]:
            if name == "mismatches":
                print(f"{k} : {A[k]} vs {B[k]}")
            else:
                print(k)

    preview(only_a, "only in A")
    preview(only_b, "only in B")
    preview(mismatches, "mismatches")

if __name__ == "__main__":
    main()
