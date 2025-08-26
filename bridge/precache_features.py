import argparse
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from teachers.wan_text_teacher_from_wan import WanTextTeacher


def iter_captions(jsonl_path, start=0, count=0):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < start:
                continue
            if count and (i - start) >= count:
                break
            j = json.loads(line)
            cap = j.get("caption", "").strip()
            if cap:
                yield i, cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", required=True)
    ap.add_argument("--llm_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--shard_size", type=int, default=2048)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=0, help="0=all")
    ap.add_argument("--llm_max_length", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.llm_dir, use_fast=True)
    llm = AutoModelForCausalLM.from_pretrained(
        args.llm_dir, torch_dtype=torch.bfloat16, device_map="auto"
    )
    llm.eval().requires_grad_(False)

    teacher = WanTextTeacher(L_wan=512, d_wan=None, device=args.device).eval()

    buf_idx, buf_caps = [], []
    shard_id, total = 0, 0

    def flush():
        nonlocal shard_id, total, buf_idx, buf_caps
        if not buf_caps:
            return
        with torch.no_grad():
            enc = tok(
                buf_caps,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.llm_max_length,
            ).to(llm.device)
            out = llm(**enc, output_hidden_states=True, use_cache=False)
            llm_h = out.hidden_states[-1].to(torch.bfloat16).cpu()  # [B, Lt, d_llm]
            llm_msk = enc["attention_mask"].bool().cpu()

            wan_h, wan_m = teacher(buf_caps)  # [B, 512, 4096], [B, 512]
            wan_h = wan_h.to(torch.bfloat16).cpu()
            wan_m = wan_m.cpu()

        shard = {
            "idx": buf_idx,
            "captions": buf_caps,
            "llm_h": llm_h,
            "llm_mask": llm_msk,
            "wan_h": wan_h,
            "wan_mask": wan_m,
        }
        path = out / f"shard_{shard_id:05d}.pt"
        torch.save(shard, path)
        print(
            f"[precache] wrote {path}  (n={len(buf_caps)}, total={total + len(buf_caps)})"
        )
        shard_id += 1
        total += len(buf_caps)
        buf_idx, buf_caps = [], []

    for i, cap in iter_captions(args.captions, args.start, args.count):
        buf_idx.append(i)
        buf_caps.append(cap)
        if len(buf_caps) >= args.shard_size:
            flush()

    flush()
    print(f"[precache] done. total cached={total}")


if __name__ == "__main__":
    main()
