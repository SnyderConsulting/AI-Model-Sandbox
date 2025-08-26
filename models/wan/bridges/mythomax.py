from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bridge.adapter import PerceiverBridge


class MythoMaxBridgeEncoder:
    """Map MythoMax-L2-13B hidden states to Wan's token space using a trained Perceiver bridge."""

    def __init__(
        self,
        text_len: int = 512,
        d_wan: int = 3072,
        llm_dir: str | Path = Path(__file__).resolve().parents[3]
        / "models"
        / "MythoMax-L2-13B",
        bridge_ckpt: str | Path = Path(__file__).resolve().parents[3]
        / "checkpoints"
        / "bridge"
        / "bridge_epoch01.pth",
        d_mid: int = 1024,
        n_blocks: int = 3,
        heads_mid: int = 16,
        dtype: torch.dtype = torch.bfloat16,
        device: int | str = torch.cuda.current_device(),
    ) -> None:
        self.text_len = text_len
        self.d_wan = d_wan
        self.device = device
        self.dtype = dtype

        self.tok = AutoTokenizer.from_pretrained(str(llm_dir), use_fast=True)
        self.llm = AutoModelForCausalLM.from_pretrained(
            str(llm_dir), torch_dtype=dtype, device_map="auto"
        )
        self.llm.eval().requires_grad_(False)
        d_llm = self.llm.config.hidden_size

        self.bridge = (
            PerceiverBridge(
                d_llm=d_llm,
                d_wan=d_wan,
                L_wan=text_len,
                d_mid=d_mid,
                n_heads=heads_mid,
                n_blocks=n_blocks,
            )
            .to(device)
            .eval()
        )
        ckpt = torch.load(str(bridge_ckpt), map_location="cpu")
        self.bridge.load_state_dict(ckpt["bridge"], strict=True)

    @torch.no_grad()
    def __call__(
        self, texts: list[str], device: int | str | None = None
    ) -> list[torch.Tensor]:
        device = device or self.device
        enc = self.tok(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=4096
        ).to(self.llm.device)
        out = self.llm(**enc, output_hidden_states=True, use_cache=False)
        h_llm = out.hidden_states[-1]
        mask_llm = enc["attention_mask"].bool()
        h_wan = self.bridge(h_llm, mask_llm)
        lengths = mask_llm.sum(dim=1).clamp(max=self.text_len).tolist()
        out_list: list[torch.Tensor] = []
        for i, L in enumerate(lengths):
            L = L if L > 0 else min(8, self.text_len)
            out_list.append(h_wan[i, :L, :].to(device))
        return out_list
