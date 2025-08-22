from __future__ import annotations

import torch
import torch.nn as nn
from typing import List, Tuple


class EmbDeltaAdapter(nn.Module):
    """
    Token-level residual projector that:
      - projects (low rank) + orthogonalizes delta w.r.t. base token embedding
      - caps per-token delta norm
      - applies a content-adaptive prompt-wise scalar gate in (0, 1)
    Lives strictly between TE and DiT: E_out = E_in + g(prompt) * Δ(E_in)
    """

    def __init__(self, d_model: int, rank: int = 64, cap: float = 0.22):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.cap = cap

        # Low-rank projector
        self.ln = nn.LayerNorm(d_model, elementwise_affine=False)
        self.down = nn.Linear(d_model, rank, bias=False)
        self.up = nn.Linear(rank, d_model, bias=False)

        # Prompt-wise gate: features = [pooled_emb, len_norm, token_var]
        hidden = max(64, d_model // 32)
        self.gate = nn.Sequential(
            nn.Linear(d_model + 2, hidden), nn.SiLU(), nn.Linear(hidden, 1)
        )

    @torch.no_grad()
    def _seq_mean(self, E: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # E: [B, L, D], mask: [B, L]
        m = mask.float().unsqueeze(-1)  # [B, L, 1]
        denom = m.sum(dim=1).clamp_min(1.0)
        return (E * m).sum(dim=1) / denom

    def _prompt_gate(self, E: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Features: pooled embedding + (len_norm, token_var)
        pooled = self._seq_mean(E, mask)  # [B, D]
        len_norm = mask.sum(dim=1).float() / mask.size(1)  # [B]
        # cheap statistic of within-prompt variance
        m = mask.float().unsqueeze(-1)  # [B, L, 1]
        centered = (E - pooled.unsqueeze(1)) * m
        token_var = centered.pow(2).mean(dim=(1, 2))  # [B]

        feats = torch.cat(
            [pooled, len_norm.unsqueeze(-1), token_var.unsqueeze(-1)], dim=-1
        )
        gate = torch.sigmoid(self.gate(feats))  # [B, 1]
        return gate  # (0,1)

    def _orthogonalize(
        self, delta: torch.Tensor, base: torch.Tensor, eps: float = 1e-6
    ) -> torch.Tensor:
        # Remove component of delta along base: Δ <- Δ - proj_base(Δ)
        dot = (delta * base).sum(dim=-1, keepdim=True)
        base_norm_sq = base.pow(2).sum(dim=-1, keepdim=True).clamp_min(eps)
        return delta - (dot / base_norm_sq) * base

    def forward(
        self, E: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        E:    [B, L, D], float
        mask: [B, L],    bool/int
        Returns (E_out, delta, gate) where:
          E_out = E + gate * capped_orth_delta
          delta = uncapped, pre-gated delta (for diagnostics)
          gate  = [B, 1] content-adaptive scalar in (0,1)
        """
        assert E.ndim == 3, "E must be [B, L, D]"
        assert mask.ndim == 2 and mask.shape[:1] == E.shape[:1], "mask must be [B, L]"

        B, L, D = E.shape
        assert D == self.d_model, f"d_model mismatch: got {D} vs {self.d_model}"

        # Low-rank delta
        Z = self.up(self.down(self.ln(E)))  # [B, L, D]
        Z = self._orthogonalize(Z, E)  # geometry-preserving direction

        # Per-token norm cap (soft)
        # First squash with tanh to keep values tame, then cap by per-token L2
        Z = torch.tanh(Z)
        z_norm = Z.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        scale = torch.clamp(self.cap / z_norm, max=1.0)
        Z = Z * scale

        # Prompt-wise gate in (0,1)
        g = self._prompt_gate(E, mask).unsqueeze(1)  # [B,1,1]
        E_out = E + g * Z

        return E_out, Z, g.squeeze(-1)

    # ---------- helper for list-of-tensors style ----------
    @torch.no_grad()
    def apply_on_list(
        self, seqs: List[torch.Tensor], lens: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        seqs: list of [Li, D] float tensors (CPU/GPU ok)
        lens: [B] lengths (int)
        Returns list of same shapes, post-adapter.
        """
        device = seqs[0].device
        B = len(seqs)
        L = int(lens.max().item())
        D = seqs[0].size(-1)

        # pack to padded batch
        E = torch.zeros(B, L, D, device=device, dtype=seqs[0].dtype)
        mask = torch.zeros(B, L, dtype=torch.bool, device=device)
        for i, t in enumerate(seqs):
            li = int(lens[i].item())
            E[i, :li] = t[:li]
            mask[i, :li] = True

        E_out, _, _ = self.forward(E, mask)

        # unpack
        out = []
        for i in range(B):
            li = int(lens[i].item())
            out.append(E_out[i, :li].contiguous())
        return out


def load_embdelta_adapter(
    path: str, d_model: int, map_location: str | torch.device = "cuda"
) -> EmbDeltaAdapter:
    ckpt = torch.load(path, map_location=map_location)
    cfg = ckpt["config"]
    model = EmbDeltaAdapter(d_model=cfg["d_model"], rank=cfg["rank"], cap=cfg["cap"])
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval().to(map_location)
    return model


# Optional convenience hook you can call from an inference script (no core edits):
def apply_to_text_embeddings(
    text_embeddings: List[torch.Tensor],
    seq_lens: torch.Tensor,
    adapter: EmbDeltaAdapter,
) -> List[torch.Tensor]:
    """
    text_embeddings: list([Li, D]) from UMT5-XXL
    seq_lens:        [B] lengths (int)
    adapter:         loaded EmbDeltaAdapter
    """
    with torch.no_grad():
        return adapter.apply_on_list(text_embeddings, seq_lens)
