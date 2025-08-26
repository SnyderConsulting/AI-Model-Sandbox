import json
import os
import time
from contextlib import contextmanager

import torch


def cosine_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.BoolTensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Cosine distance averaged over tokens.

    Args:
        x: Predicted features ``[B, L, D]``.
        y: Target features ``[B, L, D]``.
        mask: Optional boolean mask ``[B, L]`` where ``True`` marks valid tokens.
        eps: Numerical stability constant.

    Returns:
        Scalar tensor of the mean cosine distance.
    """

    x = torch.nn.functional.normalize(x, dim=-1)
    y = torch.nn.functional.normalize(y, dim=-1)
    if mask is not None:
        m = mask.unsqueeze(-1).float()
        return ((1.0 - (x * y).sum(dim=-1)) * m.squeeze(-1)).sum() / (m.sum() + eps)
    return (1.0 - (x * y).sum(dim=-1)).mean()


def mse_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.BoolTensor | None = None,
) -> torch.Tensor:
    """Mean squared error with optional masking."""

    if mask is not None:
        m = mask.unsqueeze(-1).float()
        return ((x - y) ** 2 * m).sum() / (m.sum() * x.shape[-1] + 1e-8)
    return torch.nn.functional.mse_loss(x, y)


def match_stats_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.BoolTensor | None = None,
) -> torch.Tensor:
    """Match channel means and variances between tensors."""

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
    """Return the current timestamp formatted for logs."""

    return time.strftime("%Y-%m-%d %H:%M:%S")


def tensor_stats(
    x: torch.Tensor,
    mask: torch.BoolTensor | None = None,
    name: str | None = None,
    max_channels: int = 8,
) -> dict:
    """Return robust numeric statistics for ``x`` suitable for logging."""

    with torch.no_grad():
        info: dict[str, object] = {"name": name or "tensor"}
        info["dtype"] = str(x.dtype).replace("torch.", "")
        info["shape"] = list(x.shape)
        xf = x
        if (
            mask is not None
            and x.dim() >= 2
            and mask.shape[0] == x.shape[0]
            and mask.shape[1] == x.shape[1]
        ):
            m = mask.unsqueeze(-1)
            denom = m.sum().item()
            if denom > 0:
                xf = x * m
                xf = xf[m.expand_as(x)].view(-1, x.shape[-1]) if x.dim() == 3 else xf[m]
        if xf.numel() == 0:
            xf = x

        finite = torch.isfinite(xf)
        info["n"] = int(xf.numel())
        info["n_finite"] = int(finite.sum().item())
        info["n_inf"] = int((~torch.isfinite(xf) & torch.isinf(xf)).sum().item())
        info["n_nan"] = int(torch.isnan(xf).sum().item())
        xf = torch.nan_to_num(xf, nan=0.0, posinf=0.0, neginf=0.0)
        info["min"] = float(xf.min().item()) if xf.numel() else 0.0
        info["max"] = float(xf.max().item()) if xf.numel() else 0.0
        info["mean"] = float(xf.mean().item()) if xf.numel() else 0.0
        info["std"] = float(xf.std(unbiased=False).item()) if xf.numel() else 0.0

        if x.dim() >= 2:
            z = x.float().reshape(-1, x.shape[-1])
            z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
            D = z.shape[-1]
            take = min(D, max_channels)
            idx = torch.linspace(0, D - 1, steps=take).round().long()
            ch_mean = z[:, idx].mean(0)
            ch_std = z[:, idx].std(0, unbiased=False)
            info["channels_probe_idx"] = idx.tolist()
            info["channels_mean"] = [float(v) for v in ch_mean]
            info["channels_std"] = [float(v) for v in ch_std]
        return info


def grad_norms(model: torch.nn.Module, topk: int = 10) -> dict:
    """Return gradient L2 norms for model parameters."""

    norms = []
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.grad is None:
                continue
            g = p.grad.detach()
            if g.numel() == 0:
                continue
            finite = torch.isfinite(g).all().item()
            norms.append((n, float(g.norm(p=2).item()), bool(finite), list(g.shape)))
    norms.sort(key=lambda t: t[1], reverse=True)
    head = [
        {"name": n, "l2": v, "finite": f, "shape": s} for (n, v, f, s) in norms[:topk]
    ]
    total = sum(v for (_, v, _, _) in norms)
    return {"topk": head, "total_l2": float(total), "count": len(norms)}


class JsonlLogger:
    """Minimal JSONL logger writing one object per line."""

    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(self.path, "a"):
            pass

    def log(self, obj: dict) -> None:
        obj["_ts"] = now()
        with open(self.path, "a") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


@contextmanager
def detect_anomaly_if(flag: bool = True):
    """Conditional wrapper around ``torch.autograd.detect_anomaly``."""

    if flag:
        with torch.autograd.detect_anomaly():
            yield
    else:
        yield
