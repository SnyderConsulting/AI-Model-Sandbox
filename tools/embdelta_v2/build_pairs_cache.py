import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

# Ensure repo root on path if running as a module this isn't needed
try:
    from models.wan.t5 import T5EncoderModel
except ModuleNotFoundError:
    import sys

    ROOT = Path(__file__).resolve().parents[2]
    sys.path.append(str(ROOT))
    from models.wan.t5 import T5EncoderModel


@torch.no_grad()
def encode_batch(
    te: T5EncoderModel, prompts: List[str], device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      E:  [B, L, D] float32
      M:  [B, L]    bool
      L:  [B]       int64 (sequence lengths)
    """
    # Tokenize (padding requires a pad_token to be present on the HF tokenizer)
    ids, mask = te.tokenizer(prompts, return_mask=True, add_special_tokens=True)
    ids = ids.to(device)
    mask = mask.to(device)

    # Forward through the TE (encoder-only)
    # te.model(...) returns a list of per-item tensors or a padded batch depending on implementation.
    # The Wan wrapper returns a list of [len_i, d_model] tensors; we convert to a padded tensor.
    embs = te.model(ids, mask)  # list(T_i, D) or [B, L, D]

    if isinstance(embs, list):
        # Pad to max length
        max_len = max(e.shape[0] for e in embs)
        d_model = embs[0].shape[-1]
        B = len(embs)
        E = ids.new_zeros((B, max_len, d_model), dtype=torch.float32)  # fp32 for numpy
        M = torch.zeros((B, max_len), dtype=torch.bool, device=ids.device)
        L = torch.zeros((B,), dtype=torch.long, device=ids.device)
        for i, e in enumerate(embs):
            L[i] = e.shape[0]
            E[i, : e.shape[0]] = e.to(torch.float32)
            M[i, : e.shape[0]] = True
    else:
        # Already a batch tensor [B, L, D] (likely bf16); cast to fp32
        E = embs.to(torch.float32)
        M = mask.bool()
        L = M.long().sum(dim=1)

    return E.cpu().numpy(), M.cpu().numpy(), L.cpu().numpy()


def ensure_pad_token_is_alias(te: T5EncoderModel) -> None:
    """
    Make sure the underlying HF tokenizer has a pad token without changing vocab size.
    Prefer aliasing pad->eos (or ->unk) to avoid resizing embeddings.
    """
    # te.tokenizer is our wrapper; the underlying HF tokenizer is at .tokenizer
    hf_tok = getattr(te.tokenizer, "tokenizer", None)
    if hf_tok is None:
        return

    # If already set, we're done.
    if getattr(hf_tok, "pad_token_id", None) is not None:
        return

    # Prefer EOS, else UNK. These already exist in vocab.
    if getattr(hf_tok, "eos_token", None) is not None:
        hf_tok.pad_token = hf_tok.eos_token
    elif getattr(hf_tok, "unk_token", None) is not None:
        hf_tok.pad_token = hf_tok.unk_token
    else:
        # Last resort: set pad_token_id to 0 (common for T5), without adding a new token.
        # This assumes id 0 exists in the vocab (true for T5-family).
        hf_tok.pad_token_id = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--jsonl",
        type=str,
        required=True,
        help="Path to JSONL with fields base_prompt, rewritten_prompt",
    )
    ap.add_argument(
        "--ckpt_root",
        type=str,
        required=True,
        help="Directory containing Wan 2.2 5B assets",
    )
    ap.add_argument("--out", type=str, required=True, help="Output .npz cache path")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    ckpt_root = Path(args.ckpt_root)
    # Prefer Wan 2.2 5B TI2V encoder assets
    #  - weights: models_t5_umt5-xxl-enc-bf16.pth  (or .safetensors if you converted)
    #  - tokenizer dir: tokenizer_umt5_xxl
    weight_path = ckpt_root / "models_t5_umt5-xxl-enc-bf16.pth"
    tok_path = ckpt_root / "tokenizer_umt5_xxl"

    print(f"[embdelta_v2] Using TE checkpoint: {weight_path}")

    device = torch.device(args.device)

    # Load TE
    te = T5EncoderModel(
        text_len=512,
        dtype=torch.bfloat16,
        device="cpu",  # we move later
        checkpoint_path=str(weight_path),
        tokenizer_path=ckpt_root
        / "google/umt5-xxl",  # adjust if your layout differs
        shard_fn=None,
    )
    ensure_pad_token_is_alias(te)  # <---- IMPORTANT
    te.model.eval().to(device)

    # Read the pairs
    base_prompts, rew_prompts = [], []
    with open(args.jsonl, "r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            if "base_prompt" in j and "rewritten_prompt" in j:
                base_prompts.append(j["base_prompt"])
                rew_prompts.append(j["rewritten_prompt"])

    assert len(base_prompts) == len(rew_prompts) and len(base_prompts) > 0
    N = len(base_prompts)
    print(f"[embdelta_v2] Loaded {N} preference pairs.")

    all_E_base, all_E_rew, all_M, all_L = [], [], [], []

    bs = args.batch_size
    for i in range(0, N, bs):
        b = slice(i, min(N, i + bs))
        E_base, M_base, L_base = encode_batch(te, base_prompts[b], device)
        E_rew, M_rew, L_rew = encode_batch(te, rew_prompts[b], device)
        # Sanity (masks/lengths should match per-pair after tokenization settings)
        assert (M_base.shape == M_rew.shape) and (L_base.shape == L_rew.shape)

        all_E_base.append(E_base)
        all_E_rew.append(E_rew)
        all_M.append(M_base)  # either is fine; same shapes
        all_L.append(L_base)

        if (i // bs) % 20 == 0:
            print(f"Encoded {i + len(base_prompts[b])}/{N}")

    E_base = np.concatenate(all_E_base, axis=0)
    E_rew = np.concatenate(all_E_rew, axis=0)
    M = np.concatenate(all_M, axis=0)
    L = np.concatenate(all_L, axis=0)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, E_base=E_base, E_rew=E_rew, M=M, L=L)

    print(
        f"[embdelta_v2] Saved cache: {out_path}  "
        f"(E_base={E_base.shape} E_rew={E_rew.shape} M={M.shape} L={L.shape})"
    )


if __name__ == "__main__":
    main()
