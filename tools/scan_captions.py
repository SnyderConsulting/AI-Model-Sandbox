import argparse, json
from pathlib import Path
from statistics import mean

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/workspace/data")
    ap.add_argument("--out",  default="/workspace/data/captions.jsonl")
    args = ap.parse_args()

    root = Path(args.root)
    files = sorted(root.rglob("*.txt"))
    n = 0; lengths = []
    with open(args.out, "w", encoding="utf-8") as w:
        for p in files:
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore").strip()
                if not txt: continue
                rec = {"path": str(p), "caption": txt, "n_chars": len(txt)}
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1; lengths.append(len(txt))
            except Exception as e:
                print(f"skip {p}: {e}")
    print(f"wrote {n} captions to {args.out}")
    if lengths:
        print(f"avg chars: {mean(lengths):.1f}, min: {min(lengths)}, max: {max(lengths)}")

if __name__ == "__main__":
    main()
