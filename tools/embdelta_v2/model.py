from __future__ import annotations

import torch
import torch.nn as nn


class EmbDeltaAdapter(nn.Module):
    def __init__(self, d_model: int, rank: int = 64, cap: float = 0.22):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.cap = cap

        self.ln = nn.LayerNorm(d_model, elementwise_affine=False)
        self.down = nn.Linear(d_model, rank, bias=False)
        self.up = nn.Linear(rank, d_model, bias=False)

        hidden = max(64, d_model // 32)
        self.gate = nn.Sequential(
            nn.Linear(d_model + 2, hidden), nn.SiLU(), nn.Linear(hidden, 1)
        )

    def _seq_mean(self, E: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.float().unsqueeze(-1)
        denom = m.sum(dim=1).clamp_min(1.0)
        return (E * m).sum(dim=1) / denom

    def _prompt_gate(self, E: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pooled = self._seq_mean(E, mask)  # [B, D]
        len_norm = mask.sum(dim=1).float() / mask.size(1)  # [B]
        centered = (E - pooled.unsqueeze(1)) * mask.float().unsqueeze(-1)
        token_var = centered.pow(2).mean(dim=(1, 2))  # [B]
        feats = torch.cat(
            [pooled, len_norm.unsqueeze(-1), token_var.unsqueeze(-1)], dim=-1
        )
        return torch.sigmoid(self.gate(feats))  # [B, 1]

    def _orth(
        self, delta: torch.Tensor, base: torch.Tensor, eps: float = 1e-6
    ) -> torch.Tensor:
        dot = (delta * base).sum(dim=-1, keepdim=True)
        base_norm_sq = base.pow(2).sum(dim=-1, keepdim=True).clamp_min(eps)
        return delta - (dot / base_norm_sq) * base

    def forward(self, E: torch.Tensor, mask: torch.Tensor):
        Z = self.up(self.down(self.ln(E)))  # [B, L, D]
        Z = self._orth(Z, E)
        Z = torch.tanh(Z)
        z_norm = Z.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        Z = Z * torch.clamp(self.cap / z_norm, max=1.0)

        g = self._prompt_gate(E, mask).unsqueeze(1)  # [B,1,1]
        E_out = E + g * Z
        return E_out, Z, g
