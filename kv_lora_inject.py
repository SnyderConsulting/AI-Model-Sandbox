# kv_lora_inject.py
# Minimal LoRA for Wan cross-attention K/V projections (single adapter file).
# Works without PEFT; produces a PEFT-compatible state dict for adapter_model.safetensors.

from __future__ import annotations
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
from safetensors.torch import save_file as safetensors_save
import math


# ---- LoRA modules ----


class LoRALinear(nn.Module):
    def __init__(
        self, base: nn.Linear, rank: int = 8, alpha: float = 32.0, dropout: float = 0.0
    ):
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank if rank > 0 else 0.0
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        # Keep a reference to the base linear
        self.base = base
        # LoRA weights (A: in->r, B: r->out). Zero-init so it's identity at start.
        if rank > 0:
            self.lora_A = nn.Linear(self.in_features, rank, bias=False)
            self.lora_B = nn.Linear(rank, self.out_features, bias=False)
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)  # ensures initial delta ~ 0
        else:
            # Rank 0 -> disabled lora (for completeness)
            self.lora_A = None
            self.lora_B = None

        # Switch to enable/disable LoRA at forward
        self.enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.lora_A is None or not self.enabled:
            return y
        return y + self.dropout(self.lora_B(self.lora_A(x))) * self.scaling

    @property
    def weight(self):
        # Proxy to match nn.Linear API (read-only)
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        if self.lora_A is not None:
            yield from self.lora_A.parameters()
            yield from self.lora_B.parameters()


def _replace_linear_with_lora(
    module: nn.Module, attr: str, rank: int, alpha: float, dropout: float
) -> LoRALinear:
    base = getattr(module, attr)
    assert isinstance(
        base, nn.Linear
    ), f"Expected nn.Linear at {attr}, got {type(base)}"
    lora = LoRALinear(base, rank=rank, alpha=alpha, dropout=dropout)
    setattr(module, attr, lora)
    return lora


def inject_lora_kv(
    model: nn.Module,
    blocks_attr: str = "blocks",
    cross_attr: str = "cross_attn",
    targets: Tuple[str, ...] = ("k", "v"),
    rank: int = 8,
    alpha: float = 32.0,
    dropout: float = 0.0,
    blocks_range: Tuple[int, int] | None = None,
) -> Dict[str, LoRALinear]:
    """
    Replace cross-attn .k and .v linears with LoRALinear in all WanAttentionBlocks.

    Returns: dict mapping module path -> LoRALinear
    """
    loras: Dict[str, LoRALinear] = {}
    blocks = getattr(model, blocks_attr)
    n = len(blocks)
    start, end = (0, n - 1) if blocks_range is None else blocks_range
    for i in range(start, end + 1):
        blk = blocks[i]
        ca = getattr(blk, cross_attr, None)
        if ca is None:
            continue
        for tgt in targets:
            if hasattr(ca, tgt):
                path = f"{blocks_attr}.{i}.{cross_attr}.{tgt}"
                loras[path] = _replace_linear_with_lora(ca, tgt, rank, alpha, dropout)
    return loras


def set_lora_enabled(model: nn.Module, enabled: bool = True):
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.enabled = enabled


def lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    params: List[nn.Parameter] = []
    for m in model.modules():
        if isinstance(m, LoRALinear):
            params += list(m.trainable_parameters())
    return params


# ---- Save / load in PEFT-ish format ----


def export_peft_adapter(
    model: nn.Module, save_path: str, prefix: str = "diffusion_model."
) -> None:
    """
    Export only LoRA tensors in a PEFT-compatible key format so existing loaders can consume it:
      diffusion_model.blocks.{i}.cross_attn.{k,v}.lora_A.weight
      diffusion_model.blocks.{i}.cross_attn.{k,v}.lora_B.weight

    `prefix` should match what your inference loader expects (e.g., ``"transformer."``).
    """
    state: Dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            # name ends with "...blocks.{i}.cross_attn.k"
            key_A = f"{prefix}{name}.lora_A.weight"
            key_B = f"{prefix}{name}.lora_B.weight"
            state[key_A] = module.lora_A.weight.detach().cpu()
            state[key_B] = module.lora_B.weight.detach().cpu()
    safetensors_save(state, save_path, metadata={"format": "pt"})
