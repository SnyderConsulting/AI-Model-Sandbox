import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

# Local imports (repo root added by launcher script in our previous setup)
from models.wan.t5 import T5EncoderModel


@torch.no_grad()
def encode_batch(te, prompts, device):
    """
    Returns:
      E: [B, T, H] float32 (encoder hidden/context)
      M: [B, T]     float32 (0/1 mask)
      L: [B]        int64   (sequence lengths)
    """
    # tokenizer returns padded ids + mask to text_len
    ids, mask = te.tokenizer(prompts, return_mask=True, add_special_tokens=True)
    ids = ids.to(device)
    mask = mask.to(device)

    # No autocast needed; the TE already runs in its native dtype (often bf16).
    # We only cast *outputs* to fp32 for NumPy compatibility.
    E = te.model(ids, mask)  # [B, T, H], likely bf16
    L = mask.gt(0).sum(dim=1).to(torch.long)  # [B]
    M = mask.to(torch.float32)  # [B, T] -> fp32 mask

    # Cast encoder output to fp32 *before* numpy()
    E = E.to(torch.float32)

    return E.cpu().numpy(), M.cpu().numpy(), L.cpu().numpy()


def find_t5_checkpoint(root: Path) -> Path:
    """
    Heuristic finder for the UMT5-XXL checkpoint inside a Wan 2.2 folder.
    Works whether it's a single .safetensors or a .pth/pt torch state dict.
    """
    cand = []
    for p in root.rglob("*"):
        if p.is_file():
            name = p.name.lower()
            if ("t5" in name or "umt5" in name or "text_encoder" in name) and (
                name.endswith(".safetensors")
                or name.endswith(".pth")
                or name.endswith(".pt")
            ):
                cand.append(p)
    if not cand:
        raise FileNotFoundError(f"No T5/UMT5 checkpoint found under {root}")
    # Prefer safetensors if present
    cand.sort(key=lambda x: (not x.name.endswith(".safetensors"), len(x.as_posix())))
    return cand[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--jsonl", required=True, help="JSONL with fields base_prompt, rewritten_prompt"
    )
    ap.add_argument(
        "--ckpt_root", required=True, help="Path to Wan 2.2 model root (folder)"
    )
    ap.add_argument("--out", required=True, help="Output .npz path")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--text_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()

    ckpt_root = Path(args.ckpt_root)
    te_ckpt = find_t5_checkpoint(ckpt_root)
    print(f"[embdelta_v2] Using TE checkpoint: {te_ckpt}")

    # Initialize encoder (keep weights in native dtype to avoid RAM blow-up)
    te = T5EncoderModel(
        text_len=args.text_len,
        dtype=torch.bfloat16,  # keep weights in bf16
        device="cpu",  # we’ll .to(device) below if needed
        checkpoint_path=os.fspath(te_ckpt),  # PathLike-safe
        tokenizer_path=ckpt_root
        / "tokenizer_umt5_xxl",  # adjust if your layout differs
        shard_fn=None,
    )
    # Move module to compute device (bf16 on CUDA is fine; CPU will run in fp32 compute)
    te.model.to(args.device)

    # Load pairs
    pairs = []
    with open(args.jsonl, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            base = rec.get("base_prompt", "").strip()
            rew = rec.get("rewritten_prompt", "").strip()
            if base and rew:
                pairs.append((base, rew))
    if not pairs:
        raise RuntimeError("No (base_prompt, rewritten_prompt) pairs found.")

    E_base_list, E_rew_list, M_list, L_list = [], [], [], []
    B = args.batch_size
    for i in range(0, len(pairs), B):
        chunk = pairs[i : i + B]
        base_prompts = [b for (b, _) in chunk]
        rew_prompts = [r for (_, r) in chunk]

        E_base, M, L = encode_batch(te, base_prompts, args.device)
        E_rew, _, _ = encode_batch(te, rew_prompts, args.device)

        E_base_list.append(E_base)  # fp32
        E_rew_list.append(E_rew)  # fp32
        M_list.append(M)  # fp32
        L_list.append(L)  # int64

        if (i // B) % 20 == 0:
            print(f"Encoded {i+len(chunk)}/{len(pairs)}")

    E_base = np.concatenate(E_base_list, axis=0).astype(np.float32)
    E_rew = np.concatenate(E_rew_list, axis=0).astype(np.float32)
    M = np.concatenate(M_list, axis=0).astype(np.float32)
    L = np.concatenate(L_list, axis=0).astype(np.int64)

    os.makedirs(Path(args.out).parent, exist_ok=True)
    np.savez_compressed(args.out, E_base=E_base, E_rew=E_rew, M=M, L=L)
    print(f"[embdelta_v2] Saved cache: {args.out}")
    print(f"  E_base: {E_base.shape} {E_base.dtype}")
    print(f"  E_rew : {E_rew.shape} {E_rew.dtype}")
    print(f"  M     : {M.shape} {M.dtype}")
    print(f"  L     : {L.shape} {L.dtype}")


if __name__ == "__main__":
    main()
