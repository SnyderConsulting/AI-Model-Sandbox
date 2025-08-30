import argparse
import json
import gc
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModel

try:  # optional VL support
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
except Exception:  # pragma: no cover - optional
    AutoProcessor = None
    Qwen2_5_VLForConditionalGeneration = None

# Throughput knobs on A100
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
        i = 0
        for line in f:
            if i < start:
                i += 1
                continue
            j = json.loads(line)
            cap = (j.get("caption") or "").strip()
            if cap:
                yield i, cap
            i += 1
            if count and (i - start) >= count:
                break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", required=True)
    ap.add_argument("--llm_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--shard_size", type=int, default=2000)
    ap.add_argument("--microbatch", type=int, default=64, help="per-GPU microbatch")
    ap.add_argument("--llm_max_length", type=int, default=512)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=0, help="0=all")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    # Use distinct names so we never shadow them
    out_dir_path = Path(args.out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    cfg = AutoConfig.from_pretrained(args.llm_dir)
    is_vl = "qwen2_vl" in getattr(cfg, "model_type", "") or "qwen2_5_vl" in getattr(
        cfg, "model_type", ""
    )
    if is_vl and AutoProcessor and Qwen2_5_VLForConditionalGeneration:
        print("[precache] loading LLM (Qwen-VL)…")
        proc = AutoProcessor.from_pretrained(args.llm_dir, trust_remote_code=True)
        vl = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.llm_dir,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        vl.eval().requires_grad_(False)

        @torch.inference_mode()
        def encode_llm(batch_caps):
            inputs = proc(
                text=batch_caps,
                images=None,
                videos=None,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.llm_max_length,
            )
            inputs = {
                k: (v.to(next(vl.parameters()).device) if hasattr(v, "to") else v)
                for k, v in inputs.items()
            }
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = vl(
                    **inputs,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
            # Robustly get token embeddings for both causal LMs and Qwen-VL
            h = getattr(out, "last_hidden_state", None)
            if h is None:
                hs = getattr(out, "hidden_states", None)
                if hs is None:
                    hs = getattr(out, "text_hidden_states", None)
                if hs is None or len(hs) == 0:
                    raise RuntimeError(
                        "Model did not return hidden states; upgrade transformers or check model id."
                    )
                h = hs[-1]
            h = h.to(torch.bfloat16)
            m = inputs["attention_mask"].bool()
            return h, m

    else:
        print("[precache] loading LLM (base decoder)…")
        tok = AutoTokenizer.from_pretrained(args.llm_dir, use_fast=True)
        llm_model = AutoModel.from_pretrained(
            args.llm_dir, torch_dtype=torch.bfloat16, device_map="auto"
        )
        llm_model.eval().requires_grad_(False)

        @torch.inference_mode()
        def encode_llm(batch_caps):
            enc = tok(
                batch_caps,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.llm_max_length,
            ).to(llm_model.device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                llm_out = llm_model(
                    **enc, output_hidden_states=True, use_cache=False, return_dict=True
                )
            # Robustly get token embeddings for both causal LMs and Qwen-VL
            h = getattr(llm_out, "last_hidden_state", None)
            if h is None:
                hs = getattr(llm_out, "hidden_states", None)
                if hs is None:
                    hs = getattr(llm_out, "text_hidden_states", None)
                if hs is None or len(hs) == 0:
                    raise RuntimeError(
                        "Model did not return hidden states; upgrade transformers or check model id."
                    )
                h = hs[-1]
            h = h.to(torch.bfloat16)
            m = enc["attention_mask"].bool()
            return h, m

    print("[precache] loading Wan teacher…")
    from teachers.wan_text_teacher_from_wan import WanTextTeacher

    teacher = WanTextTeacher(L_wan=512, d_wan=None, device=args.device).eval()

    buf_idx, buf_caps = [], []
    shard_id, total = 0, 0

    @torch.inference_mode()
    def encode_teacher(batch_caps):
        return teacher(batch_caps)  # [b, 512, 4096], [b, 512]

    def flush_shard():
        nonlocal shard_id, total, buf_idx, buf_caps
        if not buf_caps:
            return
        print(
            f"[precache] shard {shard_id:05d} — computing {len(buf_caps)} samples (microbatch={args.microbatch})"
        )

        # Per-sample lists (NOT per-microbatch tensors)
        llm_h_list, llm_m_list = [], []
        wan_h_list, wan_m_list = [], []

        for i in range(0, len(buf_caps), args.microbatch):
            caps_mb = buf_caps[i : i + args.microbatch]

            # ---- LLM (bf16) ----
            h_llm, m_llm = encode_llm(caps_mb)  # [Bmb, Lt, d_llm], [Bmb, Lt]
            h_llm_cpu = h_llm.cpu()
            m_llm_cpu = m_llm.cpu()
            for b in range(h_llm_cpu.shape[0]):
                llm_h_list.append(h_llm_cpu[b])  # [Lt_i, d_llm]
                llm_m_list.append(m_llm_cpu[b])  # [Lt_i]

            # ---- Teacher (bf16) ----
            h_wan, m_wan = encode_teacher(caps_mb)  # [Bmb, 512, 4096], [Bmb, 512]
            wan_h_list.append(h_wan.to(torch.bfloat16).cpu())
            wan_m_list.append(m_wan.cpu())

            # Free scratch
            del h_llm, m_llm, h_wan, m_wan, h_llm_cpu, m_llm_cpu
            torch.cuda.empty_cache()
            gc.collect()

        # Pad per-sample LLM sequences to shard max Lt
        llm_h = torch.nn.utils.rnn.pad_sequence(
            llm_h_list, batch_first=True
        )  # [N, Lt_shard, d_llm]
        llm_m = torch.nn.utils.rnn.pad_sequence(
            llm_m_list, batch_first=True
        )  # [N, Lt_shard]
        # Concatenate teacher (fixed length)
        wan_h = torch.cat(wan_h_list, dim=0).contiguous()  # [N, 512, 4096]
        wan_m = torch.cat(wan_m_list, dim=0).contiguous()  # [N, 512]

        shard = {
            "idx": buf_idx,
            "captions": buf_caps,
            "llm_h": llm_h,  # bf16, CPU
            "llm_mask": llm_m,  # bool, CPU
            "wan_h": wan_h,  # bf16, CPU
            "wan_mask": wan_m,  # bool, CPU
        }
        path = out_dir_path / f"shard_{shard_id:05d}.pt"
        torch.save(shard, path)
        total += len(buf_caps)
        print(f"[precache] wrote {path}  (n={len(buf_caps)}, total={total})")

        shard_id += 1
        buf_idx, buf_caps = [], []

    # Fill shards
    for i, cap in iter_captions(args.captions, start=args.start, count=args.count):
        buf_idx.append(i)
        buf_caps.append(cap)
        if len(buf_caps) >= args.shard_size:
            flush_shard()
    flush_shard()
    print(f"[precache] done. total cached={total}")


if __name__ == "__main__":
    # Helpful alloc flags — set once in your shell if you haven’t:
    #   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256
    #   export TOKENIZERS_PARALLELISM=true
    main()
