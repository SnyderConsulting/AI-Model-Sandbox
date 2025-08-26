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
        ckpt: str | Path = Path("/workspace")
        / "models"
        / "Wan2.2-TI2V-5B"
        / "models_t5_umt5-xxl-enc-bf16.pth",
        tok_dir: str | Path = Path("/workspace")
        / "models"
        / "Wan2.2-TI2V-5B"
        / "google"
        / "umt5-xxl",
        L_wan: int = 512,
        d_wan: int | None = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.L_wan, self.d_wan = L_wan, d_wan
        self.device = device
        self.enc = T5EncoderModel(
            text_len=L_wan,
            dtype=dtype,
            device=device,
            checkpoint_path=str(ckpt),
            tokenizer_path=str(tok_dir),
            shard_fn=None,
        )
        self._ensure_pad_token()
        self.eval().requires_grad_(False)

    def _ensure_pad_token(self) -> None:
        """
        Ensure the underlying HF tokenizer has a pad token set. Some saved
        tokenizers omit this, which causes errors when padding captions. Use the
        eos token if available; otherwise, add a new ``[PAD]`` token. Also set
        ``padding_side='right'`` to match common T5 encoder usage.
        """
        try:
            tok = getattr(self.enc, "tokenizer", None)
            hf_tok = getattr(tok, "tokenizer", None) if tok is not None else None
            if hf_tok is None:
                return

            if getattr(hf_tok, "padding_side", None) != "right":
                hf_tok.padding_side = "right"

            if getattr(hf_tok, "pad_token_id", None) is None:
                if getattr(hf_tok, "eos_token", None) is not None:
                    hf_tok.pad_token = hf_tok.eos_token
                else:
                    hf_tok.add_special_tokens({"pad_token": "[PAD]"})

                if hasattr(tok, "pad_token_id"):
                    tok.pad_token_id = hf_tok.pad_token_id
                if hasattr(tok, "pad_token"):
                    tok.pad_token = hf_tok.pad_token
        except Exception as e:  # pragma: no cover - warn only
            print(f"[WanTextTeacher] Warning: could not ensure pad_token: {e}")

    @torch.no_grad()
    def forward(self, captions: list[str]) -> tuple[torch.Tensor, torch.BoolTensor]:
        outs = self.enc(captions, self.device)  # list of [L_i, d_enc]
        B = len(outs)
        if self.d_wan is None:
            self.d_wan = outs[0].shape[1]
        h = torch.zeros(
            B, self.L_wan, self.d_wan, dtype=torch.bfloat16, device=self.device
        )
        m = torch.zeros(B, self.L_wan, dtype=torch.bool, device=self.device)
        for i, t in enumerate(outs):
            L = min(t.shape[0], self.L_wan)
            D = t.shape[1]
            if D != self.d_wan:
                raise RuntimeError(
                    f"WanTextTeacher width changed within a batch: got {D}, expected {self.d_wan}"
                )
            h[i, :L, :] = t[:L].to(h.dtype).to(h.device)
            m[i, :L] = True
        return h, m
