import sys
from pathlib import Path

import torch

# Ensure repo root is on sys.path to import models.wan.t5
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from models.wan.t5 import T5EncoderModel  # noqa: E402


class WanTextTeacher(torch.nn.Module):
    """Wraps Wan's UMT5 encoder to provide Wan-space tokens."""

    def __init__(
        self,
        ckpt: str | Path = REPO_ROOT
        / "models"
        / "Wan2.2-TI2V-5B"
        / "models_t5_umt5-xxl-enc-bf16.pth",
        tok_dir: str | Path = REPO_ROOT
        / "models"
        / "Wan2.2-TI2V-5B"
        / "google"
        / "umt5-xxl",
        L_wan: int = 512,
        d_wan: int = 3072,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.L_wan = L_wan
        self.d_wan = d_wan
        self.device = device
        self.enc = T5EncoderModel(
            text_len=L_wan,
            dtype=dtype,
            device=device,
            checkpoint_path=str(ckpt),
            tokenizer_path=str(tok_dir),
            shard_fn=None,
        )
        self.requires_grad_(False)

    @torch.no_grad()
    def forward(self, captions: list[str]) -> tuple[torch.Tensor, torch.BoolTensor]:
        outs = self.enc(captions, self.device)
        B = len(outs)
        h = torch.zeros(
            B, self.L_wan, self.d_wan, dtype=torch.bfloat16, device=self.device
        )
        m = torch.zeros(B, self.L_wan, dtype=torch.bool, device=self.device)
        for i, t in enumerate(outs):
            L = min(t.shape[0], self.L_wan)
            h[i, :L, :] = t[:L].to(h.dtype).to(h.device)
            m[i, :L] = True
        return h, m
