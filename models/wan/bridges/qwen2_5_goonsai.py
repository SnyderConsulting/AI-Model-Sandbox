from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bridge.adapter import PerceiverBridge


class Qwen2_5GoonsaiBridgeEncoder:
    """Map Qwen2.5-3B-goonsai-nsfw-100k hidden states to Wan's token space using a trained Perceiver bridge."""

    def __init__(
        self,
        llm_dir: str | Path = Path(__file__).resolve().parents[3]
        / "models"
        / "qwen2.5-3B-goonsai-nsfw-100k",
        bridge_ckpt: str | Path = Path(__file__).resolve().parents[3]
        / "checkpoints"
        / "bridge_qwen2.5"
        / "bridge_epoch01.pth",
        d_mid: int = 1024,
        n_blocks: int = 3,
        heads_mid: int = 16,
        dtype: torch.dtype = torch.bfloat16,
        device: int | str = torch.cuda.current_device(),
    ) -> None:
        self.device = device
        self.dtype = dtype

        self.tok = AutoTokenizer.from_pretrained(str(llm_dir), use_fast=True)
        self.llm = AutoModelForCausalLM.from_pretrained(
            str(llm_dir), torch_dtype=dtype, device_map="auto"
        )
        self.llm.eval().requires_grad_(False)
        d_llm = self.llm.config.hidden_size

        ckpt = torch.load(str(bridge_ckpt), map_location="cpu")
        cfg = ckpt.get("cfg", {})
        self.text_len = int(cfg.get("L_wan", 512))
        self.d_wan = int(cfg.get("d_wan", 4096))

        self.bridge = (
            PerceiverBridge(
                d_llm=d_llm,
                d_wan=self.d_wan,
                L_wan=self.text_len,
                d_mid=d_mid,
                n_heads=heads_mid,
                n_blocks=n_blocks,
            )
            .to(device)
            .eval()
        )
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
