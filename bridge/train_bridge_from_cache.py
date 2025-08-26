import argparse
import os
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from adapter import PerceiverBridge
from utils import (
    mse_loss,
    cosine_loss,
    match_stats_loss,
    now,
    JsonlLogger,
    grad_norms,
    tensor_stats,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


class CacheIndex(Dataset):
    def __init__(self, cache_dir):
        self.files = sorted(Path(cache_dir).glob("shard_*.pt"))
        self.index = []  # (file_idx, local_idx)
        self.meta = []
        for fi, f in enumerate(self.files):
            sh = torch.load(f, map_location="cpu")
            n = sh["wan_h"].shape[0]
            self.meta.append(
                {
                    "path": f,
                    "n": n,
                    "shape_llm": list(sh["llm_h"].shape),
                    "shape_wan": list(sh["wan_h"].shape),
                }
            )
            self.index.extend([(fi, j) for j in range(n)])
        random.shuffle(self.index)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        return self.index[i]


def collate_fetch(batch, cache):
    # batch: list of (file_idx, local_idx)
    by_file = {}
    for fi, li in batch:
        by_file.setdefault(fi, []).append(li)
    # load per-file once
    X = []
    for fi, lis in by_file.items():
        sh = cache._loaded.get(fi)
        if sh is None:
            sh = torch.load(cache.files[fi], map_location="cpu")
            cache._loaded[fi] = sh
        for li in lis:
            X.append(
                (
                    sh["llm_h"][li],
                    sh["llm_mask"][li],
                    sh["wan_h"][li],
                    sh["wan_mask"][li],
                )
            )
    # pad llm to max Lt in batch
    Lt = max(x[0].shape[0] for x in X)
    B = len(X)
    d_llm = X[0][0].shape[-1]
    llm_h = torch.zeros(B, Lt, d_llm, dtype=X[0][0].dtype)
    llm_mask = torch.zeros(B, Lt, dtype=torch.bool)
    wan_h = torch.stack([x[2] for x in X], dim=0).contiguous()
    wan_mask = torch.stack([x[3] for x in X], dim=0).contiguous()
    for i, (h, m, _, _) in enumerate(X):
        li = h.shape[0]
        llm_h[i, :li, :] = h
        llm_mask[i, :li] = m
    return llm_h, llm_mask, wan_h, wan_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--adamw", type=float, default=0.01)
    ap.add_argument("--amp", choices=["bf16", "fp16", "off"], default="bf16")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--save_every_steps", type=int, default=500)
    ap.add_argument("--save_keep_last_k", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    logger = JsonlLogger(os.path.join(args.save_dir, "bridge_debug.jsonl"))

    # build index and lazily load shards
    cache = CacheIndex(args.cache_dir)
    cache._loaded = {}
    print(f"[{now()}] cached samples: {len(cache)} | shards: {len(cache.files)}")
    print(f"[{now()}] first shard shapes: {cache.meta[0]}")

    dl = DataLoader(
        cache,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=lambda b: collate_fetch(b, cache),
        drop_last=True,
    )

    # infer dims from first item
    llm_h, llm_m, wan_h, wan_m = next(iter(dl))
    d_llm = llm_h.shape[-1]
    L_wan = wan_h.shape[1]
    d_wan = wan_h.shape[2]
    print(f"[{now()}] inferred dims: d_llm={d_llm}, L_wan={L_wan}, d_wan={d_wan}")

    # bridge
    bridge = PerceiverBridge(
        d_llm=d_llm, d_wan=d_wan, L_wan=L_wan, d_mid=1024, n_heads=16, n_blocks=3
    ).to(args.device)
    optim = torch.optim.AdamW(bridge.parameters(), lr=args.lr, weight_decay=args.adamw)

    # AMP
    if args.amp == "fp16":
        autocast_cm = torch.amp.autocast("cuda", dtype=torch.float16)
        scaler = torch.amp.GradScaler("cuda")
    elif args.amp == "bf16":
        autocast_cm = torch.amp.autocast("cuda", dtype=torch.bfloat16)
        scaler = None
    else:

        class NullCtx:
            def __enter__(self):
                return None

            def __exit__(self, *a):
                return False

        autocast_cm, scaler = NullCtx(), None

    global_step = 0
    for ep in range(args.epochs):
        for it, (llm_h, llm_m, wan_h, wan_m) in enumerate(dl):
            llm_h, llm_m = llm_h.to(args.device, non_blocking=True), llm_m.to(
                args.device, non_blocking=True
            )
            wan_h, wan_m = wan_h.to(args.device, non_blocking=True), wan_m.to(
                args.device, non_blocking=True
            )

            with autocast_cm:
                h_hat = bridge(llm_h, llm_m)
                # same losses as live trainer
                loss_mse = mse_loss(h_hat, wan_h, mask=wan_m)
                loss_cos = cosine_loss(h_hat, wan_h, mask=wan_m)
                loss_stats = match_stats_loss(h_hat, wan_h, mask=wan_m)
                loss = 1.0 * loss_mse + 0.5 * loss_cos + 0.2 * loss_stats

            if scaler is not None:
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
                optim.step()
            optim.zero_grad(set_to_none=True)

            global_step += 1
            if global_step % args.log_every == 0:
                bstats = tensor_stats(h_hat, wan_m, "bridge")
                tstats = tensor_stats(wan_h, wan_m, "teacher")
                ratio = (
                    (bstats["std"] / tstats["std"]) if tstats["std"] else float("nan")
                )
                logger.log(
                    {
                        "step": global_step,
                        "epoch": ep,
                        "losses": {
                            "mse": float(loss_mse.item()),
                            "cos": float(loss_cos.item()),
                            "stats": float(loss_stats.item()),
                        },
                        "bridge": bstats,
                        "teacher": tstats,
                        "sigma_ratio": ratio,
                        "grads": grad_norms(bridge, topk=8),
                    }
                )
                print(
                    f"[{now()}] ep{ep} it{it} step{global_step} loss={float(loss.item()):.6f} σ_ratio={ratio:.3f}"
                )

            if args.save_every_steps and (global_step % args.save_every_steps == 0):
                ck = Path(args.save_dir) / f"bridge_step{global_step:06d}.pth"
                torch.save(
                    {
                        "bridge": bridge.state_dict(),
                        "cfg": {
                            "L_wan": L_wan,
                            "d_wan": d_wan,
                            "d_llm": d_llm,
                            "d_mid": 1024,
                            "n_blocks": 3,
                            "heads_mid": 16,
                        },
                    },
                    ck,
                )
                print(f"[{now()}] saved {ck}")
                if args.save_keep_last_k > 0:
                    step_ckpts = sorted(Path(args.save_dir).glob("bridge_step*.pth"))
                    for old in step_ckpts[: -args.save_keep_last_k]:
                        try:
                            old.unlink()
                        except Exception:
                            pass

        # epoch end
        ck = Path(args.save_dir) / f"bridge_epoch{ep:02d}.pth"
        torch.save(
            {
                "bridge": bridge.state_dict(),
                "cfg": {
                    "L_wan": L_wan,
                    "d_wan": d_wan,
                    "d_llm": d_llm,
                    "d_mid": 1024,
                    "n_blocks": 3,
                    "heads_mid": 16,
                },
            },
            ck,
        )
        print(f"[{now()}] saved {ck}")


if __name__ == "__main__":
    main()
