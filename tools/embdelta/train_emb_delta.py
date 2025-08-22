# tools/embdelta/train_emb_delta.py
import argparse, math, os, time, json
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

def l2_normalize(x, dim=-1, eps=1e-8):
    return x / (x.norm(dim=dim, keepdim=True) + eps)

class LowRankAdapter(nn.Module):
    def __init__(self, d_model: int, rank: int = 16):
        super().__init__()
        self.U = nn.Linear(d_model, rank, bias=False)  # V^T in math (B, L, D) -> (B, L, r)
        self.V = nn.Linear(rank, d_model, bias=False)  # U in math   (B, L, r) -> (B, L, D)

    def forward(self, x):
        # y = x + V(U(x))
        return x + self.V(self.U(x))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_cache", required=True, help="Path produced by build_pairs_cache.py (pt)")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max_token_shift", type=float, default=0.75, help="cap per-token delta norm (absolute units)")
    ap.add_argument("--cosine_weight", type=float, default=1.0)
    ap.add_argument("--l2_weight", type=float, default=0.25)
    ap.add_argument("--output_dir", default="outputs/embdelta")
    args = ap.parse_args()

    data = torch.load(args.pairs_cache, map_location="cpu")
    X = data["base"].float()     # (N, L, D)
    Y = data["rewrite"].float()  # (N, L, D)
    N, L, D = X.shape

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LowRankAdapter(d_model=D, rank=args.rank).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)

    # Simple random sampler by index
    idxs = torch.arange(N)
    num_steps = math.ceil(N / args.batch_size)

    def batchify(t, indices):
        return t.index_select(0, indices.to(t.device))

    for epoch in range(1, args.epochs+1):
        perm = idxs[torch.randperm(N)]
        total = 0.0
        for s in range(num_steps):
            sl = perm[s*args.batch_size:(s+1)*args.batch_size]
            xb = batchify(X, sl).to(device)   # (B, L, D)
            yb = batchify(Y, sl).to(device)

            yhat = model(xb)                  # (B, L, D)
            # Optional per-token shift cap
            delta = yhat - xb
            if args.max_token_shift > 0:
                delta_norm = delta.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                scale = (args.max_token_shift / delta_norm).clamp(max=1.0)
                delta = delta * scale
                yhat = xb + delta

            # Loss: multi-term
            # 1) Cosine closeness to target embeddings
            cos = (l2_normalize(yhat) * l2_normalize(yb)).sum(dim=-1)   # (B, L)
            loss_cos = 1.0 - cos.mean()

            # 2) L2 closeness to target
            loss_l2 = (yhat - yb).pow(2).mean()

            loss = args.cosine_weight * loss_cos + args.l2_weight * loss_l2

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item()

        avg = total / num_steps
        print(f"[epoch {epoch:03d}] loss={avg:.5f}")

    out_dir = Path(args.output_dir) / datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "d_model": D, "rank": args.rank}, out_dir / "embdelta_adapter.pt")
    with open(out_dir / "train_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Saved adapter -> {out_dir}")

if __name__ == "__main__":
    main()
