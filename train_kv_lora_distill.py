"""Train a KV-only LoRA adapter by distilling Wan cross-attention projections.

This script loads a Wan diffusion backbone, injects LoRA modules into all
cross-attention K and V projections and optimizes those adapters so that the
keys/values produced from a bridge encoder match those from the original
UMT5 encoder.

It intentionally keeps the diffusion transformer frozen and only trains the
LoRA parameters, allowing the resulting weights to be exported in a
PEFT-compatible format.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from safetensors.torch import load_file

from kv_lora_inject import (
    export_peft_adapter,
    inject_lora_kv,
    lora_parameters,
    set_lora_enabled,
)

# Wan imports live under the inference package.
import sys

# The Wan2.2 inference package uses a directory name with a dot. To make it
# importable as a Python package we append it to ``sys.path`` and then import
# from the ``wan`` submodule.
WAN22_DIR = Path(__file__).resolve().parent / "inference" / "Wan2.2"
sys.path.append(str(WAN22_DIR))
from wan.modules.model import WanModel  # type: ignore  # noqa: E402
from wan.modules.t5 import T5EncoderModel  # type: ignore  # noqa: E402
from wan.modules.t5_bridge import BridgeEncoderModel  # type: ignore  # noqa: E402


# -----------------------------------------------------------------------------
# Data helpers
# -----------------------------------------------------------------------------


class PromptDataset(Dataset):
    """Simple dataset reading prompts from a text file."""

    def __init__(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            self.prompts = [line.strip() for line in f.readlines() if line.strip()]

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.prompts)

    def __getitem__(self, idx: int) -> str:  # pragma: no cover - trivial
        return self.prompts[idx]


# -----------------------------------------------------------------------------
# Encoding utilities
# -----------------------------------------------------------------------------


def pad_stack(
    tokens: List[torch.Tensor], max_len: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad variable length token sequences to `max_len`.

    Returns tuple (padded, mask) with shapes [B, max_len, C] and [B, max_len, 1].
    """

    b = len(tokens)
    d = tokens[0].shape[-1]
    padded = torch.zeros(b, max_len, d, device=tokens[0].device, dtype=tokens[0].dtype)
    mask = torch.zeros(b, max_len, 1, device=tokens[0].device, dtype=torch.float32)
    for i, t in enumerate(tokens):
        length = min(t.shape[0], max_len)
        padded[i, :length] = t[:length]
        mask[i, :length, 0] = 1.0
    return padded, mask


def encode_teacher(
    prompts: List[str], encoder: T5EncoderModel, device: torch.device, max_len: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    tokens = encoder(prompts, device)
    return pad_stack(tokens, max_len)


def encode_bridge(
    prompts: List[str], encoder: BridgeEncoderModel, device: torch.device, max_len: int
) -> torch.Tensor:
    tokens = encoder(prompts, device)
    # Bridge already returns fixed length tensors but pad just in case
    padded, _ = pad_stack(tokens, max_len)
    return padded


# -----------------------------------------------------------------------------
# Training helpers
# -----------------------------------------------------------------------------


def compute_kv(
    model: WanModel, context: torch.Tensor, use_norm: bool
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    k_list: List[torch.Tensor] = []
    v_list: List[torch.Tensor] = []
    for blk in model.blocks:
        ca = blk.cross_attn
        k = ca.k(context)
        v = ca.v(context)
        if use_norm and hasattr(ca, "norm_k"):
            k = ca.norm_k(k)
        k_list.append(k)
        v_list.append(v)
    return k_list, v_list


# -----------------------------------------------------------------------------
# Main training routine
# -----------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- load Wan model ---------------------------------------------------
    with open(args.transformer_config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    model = WanModel(**cfg).to(device)

    # load only required weights
    sd = load_file(args.transformer_weights)
    filtered = {
        k: v
        for k, v in sd.items()
        if k.startswith("text_embedding") or ".cross_attn." in k
    }
    model.load_state_dict(filtered, strict=False)
    model.requires_grad_(False)

    # inject LoRA modules
    inject_lora_kv(model, rank=args.rank, alpha=args.alpha)

    # --- encoders ---------------------------------------------------------
    teacher_enc = T5EncoderModel(
        text_len=model.config.text_len,
        checkpoint_path=args.t5_checkpoint,
        tokenizer_path=args.t5_tokenizer,
        device=device,
        dtype=torch.bfloat16 if args.bf16 else torch.float32,
    )

    # bridge env variables
    os.environ["WAN_BRIDGE_CKPT"] = args.bridge_ckpt
    os.environ["WAN_BRIDGE_LLM_DIR"] = args.llm_dir
    os.environ["WAN_BRIDGE_GLOBAL_SCALE"] = str(args.global_scale)
    bridge_enc = BridgeEncoderModel(
        text_len=model.config.text_len,
        device=device,
        dtype=torch.bfloat16 if args.bf16 else torch.float32,
    )

    dataset = PromptDataset(args.prompts_file)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=False
    )

    params = lora_parameters(model)
    optim = AdamW(params, lr=args.lr, weight_decay=0.0)

    for epoch in range(args.epochs):
        for batch_prompts in loader:
            if isinstance(batch_prompts, tuple):
                batch_prompts = list(batch_prompts)
            # teacher pass (LoRA disabled)
            set_lora_enabled(model, False)
            with torch.no_grad():
                teacher_tokens, mask = encode_teacher(
                    batch_prompts, teacher_enc, device, model.config.text_len
                )
                teacher_context = model.text_embedding(teacher_tokens.float())
                k_t, v_t = compute_kv(model, teacher_context, args.use_normed_targets)

            # student pass (LoRA enabled)
            set_lora_enabled(model, True)
            bridge_tokens = encode_bridge(
                batch_prompts, bridge_enc, device, model.config.text_len
            )
            bridge_context = model.text_embedding(bridge_tokens.float())
            k_s, v_s = compute_kv(model, bridge_context, args.use_normed_targets)

            # loss
            loss = torch.tensor(0.0, device=device)
            for kt, ks, vt, vs in zip(k_t, k_s, v_t, v_s):
                mse_k = ((ks - kt) ** 2) * mask
                mse_v = ((vs - vt) ** 2) * mask
                denom = mask.sum().clamp_min(1.0)
                loss = loss + mse_k.sum() / denom + mse_v.sum() / denom

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optim.step()

        print(f"Epoch {epoch+1}/{args.epochs} - loss {loss.item():.4f}")

    # save adapter
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(args.out_dir, "adapter_model.safetensors")
    export_peft_adapter(model, out_path)
    print(f"Saved adapter to {out_path}")


# -----------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="KV LoRA distillation trainer")
    p.add_argument("--transformer_config", type=str, required=True)
    p.add_argument("--transformer_weights", type=str, required=True)
    p.add_argument("--t5_checkpoint", type=str, required=True)
    p.add_argument("--t5_tokenizer", type=str, required=True)
    p.add_argument("--prompts_file", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--bridge_ckpt", type=str, required=True)
    p.add_argument("--llm_dir", type=str, required=True)
    p.add_argument("--global_scale", type=float, default=1.0)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=float, default=32.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--use_normed_targets", action="store_true")
    p.add_argument(
        "--bf16", action="store_true", help="Use bfloat16 precision for encoders"
    )
    return p


if __name__ == "__main__":
    parser = build_argparser()
    train(parser.parse_args())
