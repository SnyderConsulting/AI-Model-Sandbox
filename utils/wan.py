from __future__ import annotations

from typing import List

import torch.nn as nn


def get_lora_module_names(self: nn.Module) -> List[str]:
    """Return Wan module names eligible for LoRA injection.

    We enumerate linear-like submodules under ``self.transformer.blocks`` and keep
    attention and feed-forward paths. Names are normalized to start with ``blocks.``
    so they match configs/tests like ``cross_attn.(k|v)`` and ``ffn.0``.
    """
    names: List[str] = []
    # Ensure transformer is present/loaded
    if not hasattr(self, "transformer"):
        return names

    for full_name, module in self.transformer.named_modules():
        # Target leaf linear-like modules
        if hasattr(module, "weight") and getattr(module.weight, "ndim", 0) == 2:
            if (
                ".self_attn." in full_name
                or ".cross_attn." in full_name
                or ".attn." in full_name
                or ".ffn." in full_name
            ):
                # Normalize prefix: "transformer.blocks.X..." -> "blocks.X..."
                short = full_name
                if short.startswith("transformer."):
                    short = short[len("transformer.") :]
                names.append(short)

    # Deduplicate and provide stable order
    return sorted(set(names))
