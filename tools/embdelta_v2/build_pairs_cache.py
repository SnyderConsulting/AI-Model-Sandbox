from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Make repo importable from tools/*
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from models.wan.t5 import T5EncoderModel  # noqa: E402
from models.wan import configs as wan_configs  # noqa: E402
from tools.embdelta_v2.utils_io import read_jsonl, ensure_dir  # noqa: E402


@torch.no_grad()
def encode_batch(te: T5EncoderModel, texts: list[str], device: torch.device):
    # Match Wan pipeline: add_special_tokens=True, return mask
    ids, mask = te.tokenizer(texts, return_mask=True, add_special_tokens=True)
    ids = ids.to(device)
    mask = mask.to(device)
    lens = mask.gt(0).sum(dim=1).long()
    embs = te.model(ids, mask)  # list of [Li, D] tensors
    D = embs[0].size(-1)
    L = int(mask.size(1))
    B = len(embs)

    # pack to padded for saving
    E = torch.zeros(B, L, D, device=device, dtype=embs[0].dtype)
    M = torch.zeros(B, L, device=device, dtype=torch.bool)
    for i, (e, length) in enumerate(zip(embs, lens)):
        li = int(length.item())
        E[i, :li] = e[:li]
        M[i, :li] = True
    return E.cpu().numpy(), M.cpu().numpy(), lens.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--jsonl",
        type=Path,
        required=True,
        help="JSONL with fields base_prompt, rewritten_prompt",
    )
    ap.add_argument(
        "--ckpt_root", type=Path, required=True, help="Path to Wan 2.2 5B (TI2V) root"
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("reports/embdelta_v2/cache/pairs_5b_2.2.npz"),
    )
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    pairs = read_jsonl(args.jsonl)
    base_prompts = [p["base_prompt"] for p in pairs]
    rew_prompts = [p["rewritten_prompt"] for p in pairs]

    # Resolve UMT5 paths via Wan config (TI2V_5B)
    ckpt_dir = args.ckpt_root
    wan_cfg = wan_configs.ti2v_5B
    t5_ckpt = str(ckpt_dir / wan_cfg.t5_checkpoint)
    t5_tok = str(ckpt_dir / wan_cfg.t5_tokenizer)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    te = T5EncoderModel(
        text_len=wan_cfg.text_len,
        dtype=torch.bfloat16,
        device=device,
        checkpoint_path=t5_ckpt,
        tokenizer_path=t5_tok,
        shard_fn=None,
    )
    te.model.eval()

    # Encode in mini-batches
    all_base_E, all_base_M, all_base_lens = [], [], []
    all_rew_E, all_rew_M, all_rew_lens = [], [], []

    bs = args.batch_size
    for i in range(0, len(base_prompts), bs):
        chunk = base_prompts[i : i + bs]
        E, M, L = encode_batch(te, chunk, device)
        all_base_E.append(E)
        all_base_M.append(M)
        all_base_lens.append(L)

    for i in range(0, len(rew_prompts), bs):
        chunk = rew_prompts[i : i + bs]
        E, M, L = encode_batch(te, chunk, device)
        all_rew_E.append(E)
        all_rew_M.append(M)
        all_rew_lens.append(L)

    base_E = np.concatenate(all_base_E, axis=0)
    base_M = np.concatenate(all_base_M, axis=0)
    base_L = np.concatenate(all_base_lens, axis=0)
    rew_E = np.concatenate(all_rew_E, axis=0)
    rew_M = np.concatenate(all_rew_M, axis=0)
    rew_L = np.concatenate(all_rew_lens, axis=0)

    out_path = Path(args.out)
    ensure_dir(out_path.parent)
    np.savez_compressed(
        out_path,
        base_E=base_E,
        base_M=base_M,
        base_L=base_L,
        rew_E=rew_E,
        rew_M=rew_M,
        rew_L=rew_L,
    )
    print(
        f"Saved cache: {out_path}  "
        f"(N={base_E.shape[0]}, L={base_E.shape[1]}, D={base_E.shape[2]})"
    )


if __name__ == "__main__":
    main()
