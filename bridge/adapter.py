import math
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, d: int, mult: int = 4, act: type[nn.Module] = nn.GELU):
        super().__init__()
        self.fc1 = nn.Linear(d, int(mult * d))
        self.act = act()
        self.fc2 = nn.Linear(int(mult * d), d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class CrossBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        return_attn: bool = False,
    ) -> None:
        super().__init__()
        self.q_ln = nn.LayerNorm(d_model, eps=1e-5)
        self.kv_ln = nn.LayerNorm(d_model, eps=1e-5)
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.mlp_ln = nn.LayerNorm(d_model, eps=1e-5)
        self.mlp = MLP(d_model, mult=4)
        self.return_attn = return_attn

    def forward(
        self,
        queries: torch.Tensor,
        tokens: torch.Tensor,
        tokens_mask: torch.BoolTensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        q = self.q_ln(queries)
        kv = self.kv_ln(tokens)
        attn_out, attn_w = self.mha(
            q,
            kv,
            kv,
            key_padding_mask=(~tokens_mask) if tokens_mask is not None else None,
            need_weights=self.return_attn,
            average_attn_weights=True,
        )
        x = queries + attn_out
        x = x + self.mlp(self.mlp_ln(x))
        if self.return_attn:
            return x, attn_w
        return x


class PerceiverBridge(nn.Module):
    def __init__(
        self,
        d_llm: int = 5120,
        d_wan: int = 4096,
        L_wan: int = 512,
        d_mid: int = 1024,
        n_heads: int = 16,
        n_blocks: int = 3,
        return_attn: bool = False,
    ) -> None:
        super().__init__()
        self.L_wan = L_wan
        self.return_attn = return_attn
        self.query = nn.Parameter(torch.randn(L_wan, d_mid) / math.sqrt(d_mid))
        self.in_proj = nn.Linear(d_llm, d_mid)
        self.blocks = nn.ModuleList(
            [
                CrossBlock(d_mid, n_heads, return_attn=return_attn)
                for _ in range(n_blocks)
            ]
        )
        self.out_ln = nn.LayerNorm(d_mid, eps=1e-5)
        self.out_proj = nn.Linear(d_mid, d_wan)
        self.out_scale = nn.Parameter(torch.ones(1, 1, d_wan))
        self.out_shift = nn.Parameter(torch.zeros(1, 1, d_wan))

    def forward(
        self, llm_tokens: torch.Tensor, llm_mask: torch.BoolTensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, list[float]]:
        B, Lt, _ = llm_tokens.shape
        x_tokens = self.in_proj(llm_tokens)
        q = self.query.unsqueeze(0).expand(B, -1, -1).contiguous()
        attn_entropies: list[float] = []
        for blk in self.blocks:
            if self.return_attn:
                q, w = blk(q, x_tokens, tokens_mask=llm_mask)
                with torch.no_grad():
                    p = torch.nan_to_num(w, nan=0.0)
                    p = p.clamp_min(1e-8)
                    ent = -(p * p.log()).sum(dim=-1).mean()
                    attn_entropies.append(float(ent.item()))
            else:
                q = blk(q, x_tokens, tokens_mask=llm_mask)
        h = self.out_proj(self.out_ln(q))
        h = h * self.out_scale + self.out_shift
        if self.return_attn:
            return h, attn_entropies
        return h
