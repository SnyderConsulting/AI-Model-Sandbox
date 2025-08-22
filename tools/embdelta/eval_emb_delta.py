# tools/embdelta/eval_emb_delta.py
import argparse, os, json
from pathlib import Path
from datetime import datetime
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class LowRankAdapter(torch.nn.Module):
    def __init__(self, d_model: int, rank: int):
        super().__init__()
        self.U = torch.nn.Linear(d_model, rank, bias=False)
        self.V = torch.nn.Linear(rank, d_model, bias=False)
    def forward(self, x):
        return x + self.V(self.U(x))

def l2n(t, dim=-1, eps=1e-8):
    return t / (t.norm(dim=dim, keepdim=True) + eps)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_cache", required=True)
    ap.add_argument("--adapter_pt", required=True)
    ap.add_argument("--report_root", default="reports/embdelta_probe")
    ap.add_argument("--max_token_shift", type=float, default=0.75)
    args = ap.parse_args()

    data = torch.load(args.pairs_cache, map_location="cpu")
    X = data["base"].float()     # (N,L,D)
    Y = data["rewrite"].float()  # (N,L,D)
    N, L, D = X.shape

    ckpt = torch.load(args.adapter_pt, map_location="cpu")
    model = LowRankAdapter(D, ckpt["rank"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    with torch.no_grad():
        Yhat = model(X)
        # apply the same cap used during training, so eval mirrors runtime
        delta = Yhat - X
        if args.max_token_shift > 0:
            dn = delta.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            scale = (args.max_token_shift / dn).clamp(max=1.0)
            delta = delta * scale
            Yhat = X + delta

        # Cosine improvement: cos(Yhat, Y) vs cos(X, Y)
        cos_base = (l2n(X) * l2n(Y)).sum(dim=-1)      # (N,L)
        cos_new  = (l2n(Yhat) * l2n(Y)).sum(dim=-1)
        uplift   = (cos_new - cos_base).mean().item()

        # Token-wise norm of shift
        token_shift = delta.norm(dim=-1).numpy()      # (N,L)

    # Prepare report dir
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.report_root) / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    # Histograms
    plt.figure()
    plt.hist((cos_new - cos_base).flatten().numpy(), bins=80)
    plt.title("Token-wise cosine uplift (new - base)")
    plt.xlabel("Δcos"); plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_dir / "hist_cosine_uplift.png"); plt.close()

    plt.figure()
    plt.hist(token_shift.flatten(), bins=80)
    plt.title("Token-wise norm of applied shift ||Δ||")
    plt.xlabel("||Δ||"); plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_dir / "hist_token_norm_shift.png"); plt.close()

    # Summary
    summary = {
        "num_pairs": int(N),
        "seq_len": int(L),
        "d_model": int(D),
        "mean_token_cos_uplift": float((cos_new - cos_base).mean().item()),
        "p50_token_cos_uplift": float(np.percentile((cos_new - cos_base).numpy(), 50.0)),
        "p90_token_cos_uplift": float(np.percentile((cos_new - cos_base).numpy(), 90.0)),
        "mean_token_shift_norm": float(token_shift.mean()),
        "p90_token_shift_norm": float(np.percentile(token_shift, 90.0)),
        "adapter_path": str(args.adapter_pt),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Human-readable markdown
    with open(out_dir / "summary.md", "w") as f:
        f.write("# Embedding Delta Adapter — Evaluation Summary\n\n")
        for k, v in summary.items():
            f.write(f"- **{k}**: {v}\n")
        f.write("\nArtifacts:\n")
        f.write("- hist_cosine_uplift.png\n")
        f.write("- hist_token_norm_shift.png\n")

    print(f"Wrote report -> {out_dir}")

if __name__ == "__main__":
    main()
