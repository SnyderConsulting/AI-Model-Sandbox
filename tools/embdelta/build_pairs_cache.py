# tools/embdelta/build_pairs_cache.py
import argparse, json, os, time, torch
from pathlib import Path
from typing import List, Dict
from datetime import datetime

# Repo-local imports (Wan UMT5 encoder and tokenizer)
from models.wan.t5 import T5EncoderModel

def load_jsonl(path: Path) -> List[Dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            items.append(json.loads(line))
    return items

@torch.inference_mode()
def encode_batch(encoder, tokenizer, texts: List[str], device: torch.device):
    ids, mask = tokenizer(texts, return_mask=True, add_special_tokens=True)
    ids = ids.to(device)
    mask = mask.to(device)
    seq_lens = mask.gt(0).sum(dim=1).long()
    # encoder(...) returns a list[Tensor(seq_len_i, d_model)] for variable lengths
    outs = encoder(ids, mask)
    return outs, seq_lens

def pad_stack(seq_list: List[torch.Tensor], max_len: int = None):
    # Stack variable-length sequences (L_i, D) into (B, Lmax, D) + lengths
    lens = [t.size(0) for t in seq_list]
    D = seq_list[0].size(1)
    Lmax = max(lens) if max_len is None else max_len
    B = len(seq_list)
    out = seq_list[0].new_zeros((B, Lmax, D))
    for i, t in enumerate(seq_list):
        L = min(lens[i], Lmax)
        out[i, :L] = t[:L]
    return out, torch.tensor(lens)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_jsonl", required=True, help="JSONL with fields: base_prompt, rewritten_prompt")
    ap.add_argument("--ckpt_dir", required=True, help="Path to Wan 2.2 checkpoint root that contains UMT5-XXL assets")
    ap.add_argument("--output_path", default="cache/embdelta_pairs.pt")
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    # Instantiate UMT5 from your repo’s wrapper (same as WanPipeline uses)
    # We only need the encoder+tokenizer; inference on CPU is fine for small batches.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = T5EncoderModel(
        text_len=512,                     # Wan 2.2 uses 512 by default
        dtype=torch.bfloat16,            # match your training dtype
        device=device,
        checkpoint_path=os.path.join(args.ckpt_dir, "umt5_xxl.safetensors"),  # adjust if named differently
        tokenizer_path=Path(args.ckpt_dir) / "tokenizer_umt5_xxl",            # adjust to your layout
        shard_fn=None,
    )
    encoder.model.eval()

    items = load_jsonl(Path(args.pairs_jsonl))
    base_texts = [it["base_prompt"] for it in items]
    rew_texts  = [it["rewritten_prompt"] for it in items]

    # Encode in chunks
    base_embs, base_lens, rew_embs, rew_lens = [], [], [], []
    for start in range(0, len(items), args.batch_size):
        sl = slice(start, start+args.batch_size)
        outs_b, lens_b = encode_batch(encoder.model, encoder.tokenizer, base_texts[sl], device)
        outs_r, lens_r = encode_batch(encoder.model, encoder.tokenizer, rew_texts[sl],  device)
        base_embs.extend(outs_b); base_lens.extend(lens_b.tolist())
        rew_embs.extend(outs_r);  rew_lens.extend(lens_r.tolist())

    # Pad & stack for easy training
    base_stack, base_stack_lens = pad_stack(base_embs)
    rew_stack,  rew_stack_lens  = pad_stack(rew_embs, max_len=base_stack.size(1))  # align to same Lmax

    out_dir = Path(args.output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "base": base_stack.to(torch.float32).cpu(),
            "base_lens": base_stack_lens.cpu(),
            "rewrite": rew_stack.to(torch.float32).cpu(),
            "rewrite_lens": torch.tensor(rew_lens).cpu(),
            "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
        },
        args.output_path,
    )
    print(f"Saved pair cache -> {args.output_path}")

if __name__ == "__main__":
    main()
