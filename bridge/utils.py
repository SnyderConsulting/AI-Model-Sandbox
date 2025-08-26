import time
import torch


def cosine_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.BoolTensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    # x,y: [B, L, D]; mask: [B, L] (bool)
    x = torch.nn.functional.normalize(x, dim=-1)
    y = torch.nn.functional.normalize(y, dim=-1)
    if mask is not None:
        m = mask.unsqueeze(-1).float()
        return ((1.0 - (x * y).sum(dim=-1)) * m.squeeze(-1)).sum() / (m.sum() + 1e-8)
    return (1.0 - (x * y).sum(dim=-1)).mean()


def mse_loss(
    x: torch.Tensor, y: torch.Tensor, mask: torch.BoolTensor | None = None
) -> torch.Tensor:
    if mask is not None:
        m = mask.unsqueeze(-1).float()
        return ((x - y) ** 2 * m).sum() / (m.sum() * x.shape[-1] + 1e-8)
    return torch.nn.functional.mse_loss(x, y)


def match_stats_loss(
    x: torch.Tensor, y: torch.Tensor, mask: torch.BoolTensor | None = None
) -> torch.Tensor:
    # channel mean/var match; x,y: [B,L,D]
    if mask is not None:
        m = mask.unsqueeze(-1).float()
        xm = (x * m).sum(dim=(0, 1)) / (m.sum(dim=(0, 1)) + 1e-8)
        ym = (y * m).sum(dim=(0, 1)) / (m.sum(dim=(0, 1)) + 1e-8)
        xv = (((x - xm) * m) ** 2).sum(dim=(0, 1)) / (m.sum(dim=(0, 1)) + 1e-8)
        yv = (((y - ym) * m) ** 2).sum(dim=(0, 1)) / (m.sum(dim=(0, 1)) + 1e-8)
    else:
        xm, xv = x.mean(dim=(0, 1)), x.var(dim=(0, 1), unbiased=False)
        ym, yv = y.mean(dim=(0, 1)), y.var(dim=(0, 1), unbiased=False)
    return torch.nn.functional.l1_loss(xm, ym) + torch.nn.functional.l1_loss(
        torch.sqrt(xv + 1e-8), torch.sqrt(yv + 1e-8)
    )


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")
