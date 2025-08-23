from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from tools.embdelta_v2.model import EmbDeltaAdapter  # noqa: E402


def load_pairs_npz(path: str):
    data = np.load(path)
    E_base = data["E_base"].astype(np.float32)
    E_rew = data["E_rew"].astype(np.float32)
    M = data["M"].astype(np.float32)
    L = data["L"].astype(np.int64)
    return E_base, E_rew, M, L


class PairCache(Dataset):
    def __init__(self, npz_path: str):
        E_base, E_rew, M, _ = load_pairs_npz(npz_path)
        self.base_E = torch.from_numpy(E_base)  # [N, L, D]
        self.rew_E = torch.from_numpy(E_rew)
        self.M = torch.from_numpy(M)
        assert self.base_E.shape == self.rew_E.shape
        self.N, self.L, self.D = self.base_E.shape

    def __len__(self):
        return self.N

    def __getitem__(self, i):
        return self.base_E[i], self.rew_E[i], self.M[i]


def seq_mean(E, M):
    m = M.float().unsqueeze(-1)
    denom = m.sum(dim=1).clamp_min(1.0)
    return (E * m).sum(dim=1) / denom


def _save_checkpoint(adapter, out_dir: Path, step: int, meta: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"embdelta_step{step:06d}.pt"
    meta_path = out_dir / f"embdelta_step{step:06d}.meta.json"

    state = {k: v.cpu() for k, v in adapter.state_dict().items()}
    torch.save(state, ckpt_path)

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[embdelta_v2] Saved checkpoint @ step {step} -> {ckpt_path}")


def train_one_epoch(
    model,
    loader,
    opt,
    device,
    w_anchor,
    w_orth,
    w_l2,
    *,
    save_every=0,
    out_dir=None,
    args=None,
    epoch=0,
    global_step=0,
):
    model.train()
    total = 0.0
    for base_E, rew_E, M in loader:
        base_E, rew_E, M = base_E.to(device), rew_E.to(device), M.to(device)

        adj_E, delta, _ = model(base_E, M)

        pooled_adj = seq_mean(adj_E, M)
        pooled_rew = seq_mean(rew_E, M)
        loss_align = 1.0 - F.cosine_similarity(pooled_adj, pooled_rew, dim=-1).mean()

        pooled_base = seq_mean(base_E, M)
        loss_anchor = (
            1.0 - F.cosine_similarity(pooled_adj, pooled_base, dim=-1)
        ).mean()

        dot = (delta * base_E).sum(dim=-1)
        base_n = base_E.norm(dim=-1).clamp_min(1e-6)
        delta_n = delta.norm(dim=-1).clamp_min(1e-6)
        loss_orth = (
            (dot / (base_n * delta_n)) ** 2 * M.float()
        ).sum() / M.float().sum().clamp_min(1.0)

        loss_l2 = (
            delta.pow(2).sum(dim=-1) * M.float()
        ).sum() / M.float().sum().clamp_min(1.0)

        loss = loss_align + w_anchor * loss_anchor + w_orth * loss_orth + w_l2 * loss_l2

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += float(loss.item())
        global_step += 1

        if save_every > 0 and (global_step % save_every == 0):
            meta = {
                "step": global_step,
                "epoch": epoch,
                "rank": args.rank,
                "cap": args.cap,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "val_frac": args.val_frac,
                "device": args.device,
                "seed": args.seed,
            }
            _save_checkpoint(model, out_dir, global_step, meta)
    return total / len(loader), global_step


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    base_cos, new_cos = [], []
    for base_E, rew_E, M in loader:
        base_E, rew_E, M = base_E.to(device), rew_E.to(device), M.to(device)
        adj_E, _, _ = model(base_E, M)

        pooled_rew = seq_mean(rew_E, M)
        pooled_base = seq_mean(base_E, M)
        pooled_adj = seq_mean(adj_E, M)

        base_cos.append(F.cosine_similarity(pooled_base, pooled_rew, dim=-1))
        new_cos.append(F.cosine_similarity(pooled_adj, pooled_rew, dim=-1))

    base_cos = torch.cat(base_cos)
    new_cos = torch.cat(new_cos)
    delta = new_cos - base_cos
    return {
        "mean_uplift": float(delta.mean().item()),
        "p50_uplift": float(delta.median().item()),
        "p90_uplift": float(delta.quantile(0.9).item()),
        "mean_base_cos": float(base_cos.mean().item()),
        "mean_new_cos": float(new_cos.mean().item()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="NPZ from build_pairs_cache.py")
    ap.add_argument("--out_dir", default="reports/embdelta_v2/run")
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--cap", type=float, default=0.22)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--w_anchor", type=float, default=0.05)
    ap.add_argument("--w_orth", type=float, default=0.02)
    ap.add_argument("--w_l2", type=float, default=1e-4)
    ap.add_argument(
        "--save_every",
        type=int,
        default=0,
        help="If > 0, save an adapter checkpoint every N optimizer steps.",
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ds = PairCache(args.cache)
    N_val = max(1, int(len(ds) * args.val_frac))
    N_tr = len(ds) - N_val
    tr_ds, va_ds = random_split(
        ds, [N_tr, N_val], generator=torch.Generator().manual_seed(args.seed)
    )

    tr_ld = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    va_ld = DataLoader(
        va_ds, batch_size=args.batch_size, shuffle=False, drop_last=False
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = EmbDeltaAdapter(d_model=ds.D, rank=args.rank, cap=args.cap).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.01
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_uplift, best_path = -1e9, None
    log = []
    global_step = 0

    for step in range(1, args.epochs + 1):
        tr_loss, global_step = train_one_epoch(
            model,
            tr_ld,
            opt,
            device,
            args.w_anchor,
            args.w_orth,
            args.w_l2,
            save_every=args.save_every,
            out_dir=out_dir,
            args=args,
            epoch=step,
            global_step=global_step,
        )
        metrics = evaluate(model, va_ld, device)
        metrics.update({"step": step, "train_loss": tr_loss})
        log.append(metrics)
        if step % 50 == 0:
            print(
                f"[{step}] loss={tr_loss:.4f}  uplift(mean/p50/p90)={metrics['mean_uplift']:.4f}/"
                f"{metrics['p50_uplift']:.4f}/{metrics['p90_uplift']:.4f}  "
                f"base={metrics['mean_base_cos']:.4f} new={metrics['mean_new_cos']:.4f}"
            )

        if metrics["mean_uplift"] > best_uplift:
            best_uplift = metrics["mean_uplift"]
            ck = {
                "config": {"d_model": ds.D, "rank": args.rank, "cap": args.cap},
                "state_dict": model.state_dict(),
                "metrics": metrics,
            }
            best_path = out_dir / "embdelta_adapter.pt"
            torch.save(ck, best_path)

    with (out_dir / "train_log.json").open("w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    print(f"Best adapter saved to: {best_path}")


if __name__ == "__main__":
    main()
