import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

# Prefer TF32 kernels on A100 (throughput boost)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# Optional SDP kernel hints (PyTorch 2.1+)
try:
    from torch.backends.cuda import sdp_kernel

    sdp_kernel.enable_flash(True)
    sdp_kernel.enable_mem_efficient(True)
    sdp_kernel.enable_math(False)
except Exception:
    pass


def iter_captions(jsonl_path, start=0, count=0):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        i0 = 0
        for line in f:
            if i0 < start:
                i0 += 1
                continue
            j = json.loads(line)
            cap = (j.get("caption") or "").strip()
            if cap:
                yield i0, cap
            i0 += 1
            if count and (i0 - start) >= count:
                break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", required=True)
    ap.add_argument("--llm_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--shard_size", type=int, default=2000)
    ap.add_argument(
        "--microbatch", type=int, default=64, help="per-GPU microbatch during caching"
    )
    ap.add_argument("--llm_max_length", type=int, default=512)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=0, help="0=all")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("[precache] loading LLM (base decoder) …")
    tok = AutoTokenizer.from_pretrained(args.llm_dir, use_fast=True)
    # Use AutoModel (LlamaModel), not ForCausalLM, to avoid logits & extra buffers
    llm = AutoModel.from_pretrained(
        args.llm_dir, torch_dtype=torch.bfloat16, device_map="auto"
    )
    llm.eval().requires_grad_(False)

    print("[precache] loading Wan teacher …")
    # Your Wan teacher wrapper (GPU). Adjust import path if needed.
    from teachers.wan_text_teacher_from_wan import WanTextTeacher

    teacher = WanTextTeacher(L_wan=512, d_wan=None, device=args.device).eval()

    # Buffers for one shard
    buf_idx, buf_caps = [], []
    shard_id, total = 0, 0

    @torch.inference_mode()
    def encode_llm(batch_caps):
        enc = tok(
            batch_caps,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.llm_max_length,
        ).to(llm.device)
        # AutoModel returns BaseModelOutputWithPast; use last_hidden_state directly
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = llm(
                **enc,
                output_hidden_states=False,
                use_cache=False,
                return_dict=True,
            )
        h = out.last_hidden_state.to(torch.bfloat16)  # [b, Lt, d_llm]
        m = enc["attention_mask"].bool()
        return h, m

    @torch.inference_mode()
    def encode_teacher(batch_caps):
        # Wan teacher already returns bf16 on CUDA
        return teacher(batch_caps)

    def flush_shard():
        nonlocal shard_id, total, buf_idx, buf_caps
        if not buf_caps:
            return
        print(
            f"[precache] shard {shard_id:05d} — computing {len(buf_caps)} samples "
            f"(microbatch={args.microbatch})"
        )

        llm_h_list, llm_m_list = [], []
        wan_h_list, wan_m_list = [], []

        # process in micro-batches to keep peak VRAM low
        for i in range(0, len(buf_caps), args.microbatch):
            caps_mb = buf_caps[i : i + args.microbatch]

            # LLM
            h_llm, m_llm = encode_llm(caps_mb)
            # move to CPU immediately to free GPU
            llm_h_list.append(h_llm.cpu())
            llm_m_list.append(m_llm.cpu())
            del h_llm, m_llm

            # Teacher
            h_wan, m_wan = encode_teacher(caps_mb)
            wan_h_list.append(h_wan.to(torch.bfloat16).cpu())
            wan_m_list.append(m_wan.cpu())
            del h_wan, m_wan

            # help the allocator between micro-batches
            torch.cuda.empty_cache()

        # Pad LLM sequences in shard to a common Lt (per shard)
        llm_h = torch.nn.utils.rnn.pad_sequence(llm_h_list, batch_first=True)
        llm_m = torch.nn.utils.rnn.pad_sequence(llm_m_list, batch_first=True)
        wan_h = torch.cat(wan_h_list, dim=0).contiguous()
        wan_m = torch.cat(wan_m_list, dim=0).contiguous()

        shard = {
            "idx": buf_idx,
            "captions": buf_caps,
            "llm_h": llm_h,  # [N, Lt_shard, d_llm] (bf16, CPU)
            "llm_mask": llm_m,  # [N, Lt_shard] (bool, CPU)
            "wan_h": wan_h,  # [N, 512, 4096] (bf16, CPU)
            "wan_mask": wan_m,  # [N, 512] (bool, CPU)
        }
        path = out / f"shard_{shard_id:05d}.pt"
        torch.save(shard, path)
        total += len(buf_caps)
        print(f"[precache] wrote {path}  (n={len(buf_caps)}, total={total})")

        # reset buffers
        shard_id += 1
        buf_idx, buf_caps = [], []

    # fill shards
    for i, cap in iter_captions(args.captions, start=args.start, count=args.count):
        buf_idx.append(i)
        buf_caps.append(cap)
        if len(buf_caps) >= args.shard_size:
            flush_shard()
    flush_shard()
    print(f"[precache] done. total cached={total}")


if __name__ == "__main__":
    main()
