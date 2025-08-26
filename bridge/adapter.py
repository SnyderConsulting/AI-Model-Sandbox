import math
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, d, mult: int = 4, act: type[nn.Module] = nn.GELU):
        super().__init__()
        self.fc1 = nn.Linear(d, int(mult * d))
        self.act = act()
        self.fc2 = nn.Linear(int(mult * d), d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class CrossBlock(nn.Module):
    """Cross-attend learned queries (Lw × dm) to LLM tokens (Lt × dm), then MLP."""

    def __init__(
        self, d_model: int, n_heads: int, mlp_mult: int = 4, dropout: float = 0.0
    ):
        super().__init__()
        self.q_ln = nn.LayerNorm(d_model)
        self.kv_ln = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.mlp_ln = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, mult=mlp_mult)

    def forward(
        self,
        queries: torch.Tensor,
        tokens: torch.Tensor,
        tokens_mask: torch.BoolTensor | None = None,
    ) -> torch.Tensor:
        # queries: [B, Lw, dm], tokens: [B, Lt, dm]
        q = self.q_ln(queries)
        kv = self.kv_ln(tokens)
        attn_out, _ = self.mha(
            q,
            kv,
            kv,
            key_padding_mask=(~tokens_mask) if tokens_mask is not None else None,
            need_weights=False,
        )
        x = queries + attn_out
        x = x + self.mlp(self.mlp_ln(x))
        return x


class PerceiverBridge(nn.Module):
    """
    LLM last hidden states (B, Lt, d_llm)  →  Wan tokens (B, Lw, d_wan)
    Perceiver-style: learned queries (Lw, dm) attend over LLM tokens; N cross blocks; final proj to d_wan.
    """

    def __init__(
        self,
        d_llm: int = 5120,
        d_wan: int = 3072,
        L_wan: int = 512,
        d_mid: int = 1024,
        n_heads: int = 16,
        n_blocks: int = 3,
    ):
        super().__init__()
        self.L_wan = L_wan
        self.query = nn.Parameter(torch.randn(L_wan, d_mid) / math.sqrt(d_mid))
        self.in_proj = nn.Linear(d_llm, d_mid)  # project LLM tokens to mid
        self.blocks = nn.ModuleList(
            [CrossBlock(d_mid, n_heads) for _ in range(n_blocks)]
        )
        self.out_ln = nn.LayerNorm(d_mid)
        self.out_proj = nn.Linear(d_mid, d_wan)
        # learned scale/shift to match Wan feature stats
        self.out_scale = nn.Parameter(torch.ones(1, 1, d_wan))
        self.out_shift = nn.Parameter(torch.zeros(1, 1, d_wan))

    def forward(
        self, llm_tokens: torch.Tensor, llm_mask: torch.BoolTensor | None = None
    ) -> torch.Tensor:
        # llm_tokens: [B, Lt, d_llm]
        B, Lt, _ = llm_tokens.shape
        x_tokens = self.in_proj(llm_tokens)  # [B, Lt, d_mid]
        # expand learned queries for B
        queries = self.query.unsqueeze(0).expand(B, -1, -1).contiguous()
        for blk in self.blocks:
            queries = blk(queries, x_tokens, tokens_mask=llm_mask)
        h = self.out_proj(self.out_ln(queries))
        return h * self.out_scale + self.out_shift  # [B, L_wan, d_wan]
