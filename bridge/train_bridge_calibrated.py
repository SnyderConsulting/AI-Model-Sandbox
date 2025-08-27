#!/usr/bin/env python3
"""
train_bridge_calibrated.py — final bridge training (from cached shards)

- Fixes distribution mismatch (μ/σ, norms) + aligns directions.
- Composite loss:
    * token-wise cosine (content)
    * sequence MSE (stability)
    * moment matching on μ/σ (per-dim)
    * norm-ratio penalty (||pred|| ≈ ||teacher|| per token)
- Streams cached shards from `precache_features.py`.

Understands shard keys like:
  ['idx', 'captions', 'llm_h', 'llm_mask', 'wan_h', 'wan_mask']
and several aliases; derives token lengths from masks if needed.

ENV expected by your Wan model:
  WAN_TEXT_LEN=512
  WAN_TEXT_DIM=4096
  WAN_LLM_DIM=5120           # only used for logging

Example:
  python train_bridge_calibrated.py \
    --cache_dir /workspace/cache/bridge \
    --out_dir   /workspace/checkpoints/bridge_final \
    --init_ckpt /workspace/checkpoints/bridge/bridge_step004000.pth \
    --device cuda:0 --epochs 1 --max_steps 20000 \
    --batch_size 32 --lr 2e-4 --amp bf16 \
    --validate_every 500 --log_every 50 \
    --w_cos 1.0 --w_mse 0.25 --w_mom 0.75 --w_norm 0.25
"""

from __future__ import annotations
import os, argparse, time, random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader

# Use the same bridge module you use in inference.
from adapter import PerceiverBridge


# --------------------------- data ---------------------------

# Variants present in your cache + a few aliases
KEYS_LLM   = ["llm_h", "h_llm", "llm_states", "llm", "hidden_states"]
KEYS_LMASK = ["llm_mask", "mask", "attention_mask"]
KEYS_TEACH = ["wan_h", "teacher", "wan", "t5", "t5_tokens"]
KEYS_TMASK = ["wan_mask", "teacher_mask", "t5_mask"]
KEYS_LEN   = ["lengths", "llm_lengths"]
KEYS_PROM  = ["captions", "prompts"]

def _first_key(d: Dict, keys: List[str]) -> Optional[str]:
    for k in keys:
        if k in d: return k
    return None

class CachedPairs(IterableDataset):
    def __init__(self, cache_dir: str | Path, shuffle: bool = True, repeat: bool = True, verbose_first: bool = True):
        super().__init__()
        self.cache_dir = Path(cache_dir)
        self.shuffle = shuffle
        self.repeat = repeat
        self.verbose_first = verbose_first
        self.paths = sorted(self.cache_dir.glob("shard_*.pt"))
        if not self.paths:
            raise FileNotFoundError(f"No shards found in {self.cache_dir} (expected shard_*.pt)")
        print(f"[data] found {len(self.paths)} shards")

    def _yield_from_file(self, path: Path, idx_file: int):
        pkg = torch.load(path, map_location="cpu")

        k_llm   = _first_key(pkg, KEYS_LLM)
        k_lmask = _first_key(pkg, KEYS_LMASK)
        k_tch   = _first_key(pkg, KEYS_TEACH)
        k_tmask = _first_key(pkg, KEYS_TMASK)
        if k_llm is None or k_tch is None:
            raise KeyError(f"{path} missing expected keys; found {list(pkg.keys())}")

        k_len = _first_key(pkg, KEYS_LEN)
        k_pr  = _first_key(pkg, KEYS_PROM)

        H = pkg[k_llm]            # [N, Lt, d_llm]
        T = pkg[k_tch]            # [N, Lw, d_wan]
        M_llm = pkg[k_lmask] if k_lmask else None   # [N, Lt]
        M_tch = pkg[k_tmask] if k_tmask else None   # [N, Lw]
        L     = pkg.get(k_len, None)
        P     = pkg.get(k_pr, None)

        if self.verbose_first and idx_file == 0:
            print(f"[data] using keys: llm={k_llm}, llm_mask={k_lmask}, teacher={k_tch}, teacher_mask={k_tmask}, "
                  f"lengths={'yes' if L is not None else 'no'}, prompts={'yes' if P is not None else 'no'}")
            self.verbose_first = False

        # sanitize types
        if M_llm is not None and M_llm.dtype != torch.bool:
            M_llm = M_llm != 0
        if M_tch is not None and M_tch.dtype != torch.bool:
            M_tch = M_tch != 0

        N = H.shape[0]
        idxs = list(range(N))
        if self.shuffle:
            random.shuffle(idxs)
        for i in idxs:
            # derive length
            if L is not None:
                L_i = int(L[i])
            else:
                L_llm = int(M_llm[i].sum().item()) if M_llm is not None else int(H[i].shape[0])
                L_tch = int(M_tch[i].sum().item()) if M_tch is not None else int(T[i].shape[0])
                L_i = min(L_llm, L_tch)
            yield {
                "h_llm":   H[i],                       # [Lt, d_llm]
                "mask":    (M_llm[i] if M_llm is not None else torch.ones(H[i].shape[0], dtype=torch.bool)),
                "teacher": T[i],                       # [Lw, d_wan]
                "length":  L_i,
                "prompt":  (P[i] if (P is not None) else ""),
            }

    def __iter__(self):
        # infinite stream over shards
        file_idx = 0
        while True:
            order = list(self.paths)
            if self.shuffle:
                random.shuffle(order)
            for p in order:
                yield from self._yield_from_file(p, idx_file=file_idx)
                file_idx += 1
            if not self.repeat:
                break


def collate(batch: List[Dict]):
    # variable length; pad within batch to max Lt and Lw (we trim by lengths later)
    H = [b["h_llm"] for b in batch]
    M = [b["mask"]  for b in batch]
    T = [b["teacher"] for b in batch]
    # FIXED: use batch[i]["length"] (not b["length"])
    L = [min(batch[i]["length"], H[i].shape[0], T[i].shape[0]) for i in range(len(batch))]

    Lt_max = max(h.shape[0] for h in H)
    Lw_max = max(t.shape[0] for t in T)

    def pad_stack(seq_list, target_len):
        out = []
        for x in seq_list:
            pad_len = target_len - x.shape[0]
            if pad_len > 0:
                # handle 1D masks vs 2D sequences
                if x.dim() == 2:
                    pad = torch.zeros((pad_len, x.shape[1]), dtype=x.dtype)
                else:
                    pad = torch.zeros((pad_len,), dtype=x.dtype)
                out.append(torch.cat([x, pad], dim=0))
            else:
                out.append(x[:target_len])
        return torch.stack(out, dim=0)

    Hs = pad_stack(H, Lt_max)                                  # [B, Lt_max, d_llm]
    # pad masks as float then cast back to bool
    Ms = pad_stack([m.float().unsqueeze(-1) for m in M], Lt_max).squeeze(-1).bool()  # [B, Lt_max]
    Ts = pad_stack(T, Lw_max)                                  # [B, Lw_max, d_wan]
    Ls = torch.tensor(L, dtype=torch.int32)

    return Hs, Ms, Ts, Ls


# --------------------------- losses & metrics ---------------------------

def token_cosine_loss(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    pn = pred / (pred.norm(dim=-1, keepdim=True) + 1e-8)
    tn = tgt  / (tgt.norm(dim=-1, keepdim=True) + 1e-8)
    return (1.0 - (pn * tn).sum(dim=-1)).mean()

def mse_loss(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, tgt)

def moment_loss(pred_seq: torch.Tensor, tgt_seq: torch.Tensor):
    mu_p = pred_seq.mean(dim=0)
    mu_t = tgt_seq.mean(dim=0)
    std_p = pred_seq.std(dim=0, unbiased=False) + 1e-8
    std_t = tgt_seq.std(dim=0, unbiased=False) + 1e-8

    mu_l2 = F.mse_loss(mu_p, mu_t)
    log_ratio = (std_p.log() - std_t.log())
    std_pen = (log_ratio ** 2).mean()

    metrics = {
        "mu_abs_delta": float((mu_p - mu_t).abs().mean().item()),
        "std_ratio_median": float((std_p / std_t).median().item()),
    }
    return mu_l2 + std_pen, metrics

def norm_ratio_penalty(pred_seq: torch.Tensor, tgt_seq: torch.Tensor) -> torch.Tensor:
    np_ = pred_seq.norm(dim=-1) + 1e-8
    nt_ = tgt_seq.norm(dim=-1) + 1e-8
    return ((np_.log() - nt_.log()) ** 2).mean()

@torch.no_grad()
def ab_metrics(pred_list: List[torch.Tensor], tgt_list: List[torch.Tensor]) -> Dict[str,float]:
    P = torch.cat(pred_list, dim=0).float()
    T = torch.cat(tgt_list, dim=0).float()
    seq_cos = F.cosine_similarity(P.flatten(), T.flatten(), dim=0).item()
    pn = P / (P.norm(dim=-1, keepdim=True) + 1e-8)
    tn = T / (T.norm(dim=-1, keepdim=True) + 1e-8)
    token_cos = (pn * tn).sum(dim=-1)
    std_ratio_median = (P.std(dim=0, unbiased=False) / (T.std(dim=0, unbiased=False) + 1e-8)).median().item()
    norm_ratio = (P.norm(dim=-1) / (T.norm(dim=-1) + 1e-8)).mean().item()
    mu_cos = F.cosine_similarity(P.mean(dim=0), T.mean(dim=0), dim=0).item()
    return dict(
        seq_cos=float(seq_cos),
        token_cos_mean=float(token_cos.mean().item()),
        token_cos_median=float(token_cos.median().item()),
        std_ratio_median=float(std_ratio_median),
        norm_ratio=float(norm_ratio),
        mu_cos=float(mu_cos),
    )


# --------------------------- training ---------------------------

def train(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device)
    d_wan = int(os.environ.get("WAN_TEXT_DIM", "4096"))
    L_wan = int(os.environ.get("WAN_TEXT_LEN", "512"))
    d_llm = int(os.environ.get("WAN_LLM_DIM", "5120"))  # just for logging

    ds = CachedPairs(args.cache_dir, shuffle=True, repeat=True)
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=0, collate_fn=collate)

    model = PerceiverBridge(
        d_llm=d_llm, d_wan=d_wan, L_wan=L_wan, d_mid=args.d_mid, n_heads=args.heads, n_blocks=args.blocks
    ).to(device)

    if args.init_ckpt and Path(args.init_ckpt).exists():
        sd = torch.load(args.init_ckpt, map_location="cpu")
        sd = sd.get("bridge", sd)
        _miss, _unexp = model.load_state_dict(sd, strict=False)
        print(f"[init] loaded {args.init_ckpt} (missing={len(_miss)}, unexpected={len(_unexp)})")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.wd)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.amp == "fp16"))
    use_autocast = (device.type == "cuda") and (args.amp in ("bf16","fp16"))
    autocast_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16

    step = 0
    best_token_cos = -1.0
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    start = time.time()
    for epoch in range(args.epochs):
        for Hs, Ms, Ts, Ls in dl:
            step += 1
            Hs = Hs.to(device)
            Ms = Ms.to(device)
            Ts = Ts.to(device)
            Ls = Ls.to(device)

            with torch.cuda.amp.autocast(enabled=use_autocast, dtype=autocast_dtype):
                Pred = model(Hs, Ms)  # [B, L_wan, d_wan]

                pred_list, tgt_list = [], []
                token_losses_cos, token_losses_mse = [], []
                for i in range(Pred.size(0)):
                    L = int(Ls[i].item())
                    p = Pred[i, :L, :]
                    t = Ts[i,   :L, :]
                    pred_list.append(p)
                    tgt_list.append(t)
                    token_losses_cos.append(token_cosine_loss(p, t))
                    token_losses_mse.append(mse_loss(p, t))

                tok_cos = torch.stack(token_losses_cos).mean()
                tok_mse = torch.stack(token_losses_mse).mean()

                Pcat = torch.cat(pred_list, dim=0)
                Tcat = torch.cat(tgt_list, dim=0)
                mom, mom_stats = moment_loss(Pcat, Tcat)
                nrm = norm_ratio_penalty(Pcat, Tcat)

                loss = (
                    args.w_cos * tok_cos +
                    args.w_mse * tok_mse +
                    args.w_mom * mom +
                    args.w_norm * nrm
                )

            opt.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                opt.step()

            if step % args.log_every == 0:
                elapsed = time.time() - start
                print(f"[{step:06d}] loss={loss.item():.4f}  cos={tok_cos.item():.4f}  mse={tok_mse.item():.5f}  "
                      f"mom_muΔ={mom_stats['mu_abs_delta']:.4f}  mom_std≈{mom_stats['std_ratio_median']:.3f}  "
                      f"norm_pen={nrm.item():.4f}  {elapsed/60:.1f}m")

            if step % args.validate_every == 0:
                model.eval()
                with torch.no_grad():
                    metrics = ab_metrics(pred_list, tgt_list)  # last batch proxy
                model.train()
                tok_mean = metrics["token_cos_mean"]
                if tok_mean > best_token_cos:
                    best_token_cos = tok_mean
                    ck = out_dir / f"bridge_step{step:06d}_best.pth"
                    torch.save({"bridge": model.state_dict(), "cfg": vars(args)}, ck)
                    print(f"[save] new best token_cos_mean={tok_mean:.4f} → {ck}")

            if step >= args.max_steps:
                break
        if step >= args.max_steps:
            break

    ck = out_dir / f"bridge_step{step:06d}.pth"
    torch.save({"bridge": model.state_dict(), "cfg": vars(args)}, ck)
    print(f"[done] wrote {ck}")


# --------------------------- args ---------------------------

def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--init_ckpt", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=20000)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--amp", type=str, default="bf16", choices=["off","bf16","fp16"])
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--validate_every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--d_mid", type=int, default=1024)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--heads", type=int, default=16)
    # loss weights
    ap.add_argument("--w_cos", type=float, default=1.0)
    ap.add_argument("--w_mse", type=float, default=0.25)
    ap.add_argument("--w_mom", type=float, default=0.75)
    ap.add_argument("--w_norm", type=float, default=0.25)
    return ap.parse_args()

def main():
    args = build_args()
    train(args)

if __name__ == "__main__":
    main()
