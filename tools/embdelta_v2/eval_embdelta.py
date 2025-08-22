from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from tools.embdelta_v2.model import EmbDeltaAdapter  # noqa: E402
from tools.embdelta_v2.utils_io import ensure_dir  # noqa: E402


def seq_mean(E, M):
    m = M.float().unsqueeze(-1)
    denom = m.sum(dim=1).clamp_min(1.0)
    return (E * m).sum(dim=1) / denom


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out_dir", default="reports/embdelta_v2/eval")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    data = np.load(args.cache)
    base_E = torch.from_numpy(data["base_E"]).to(torch.float32)
    base_M = torch.from_numpy(data["base_M"]).bool()
    rew_E = torch.from_numpy(data["rew_E"]).to(torch.float32)
    rew_M = torch.from_numpy(data["rew_M"]).bool()
    N, L, D = base_E.shape

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.adapter, map_location=device)
    model = EmbDeltaAdapter(
        d_model=ckpt["config"]["d_model"],
        rank=ckpt["config"]["rank"],
        cap=ckpt["config"]["cap"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    # batched eval
    bs = 64
    base_cos_list, new_cos_list = [], []
    with torch.no_grad():
        for i in range(0, N, bs):
            E0 = base_E[i : i + bs].to(device)
            M0 = base_M[i : i + bs].to(device)
            E1 = rew_E[i : i + bs].to(device)
            M1 = rew_M[i : i + bs].to(device)
            Adj, _, _ = model(E0, M0)
            base_cos_list.append(
                F.cosine_similarity(seq_mean(E0, M0), seq_mean(E1, M1), dim=-1).cpu()
            )
            new_cos_list.append(
                F.cosine_similarity(seq_mean(Adj, M0), seq_mean(E1, M1), dim=-1).cpu()
            )

    base_cos = torch.cat(base_cos_list)
    new_cos = torch.cat(new_cos_list)
    uplift = (new_cos - base_cos).numpy()

    out_dir = ensure_dir(args.out_dir)
    # histogram of uplift
    plt.figure()
    plt.hist(uplift, bins=40)
    plt.title("Prompt-level cosine uplift (new - base)")
    plt.xlabel("Δcos")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_dir / "hist_prompt_uplift.png")
    plt.close()

    summary = {
        "N": int(N),
        "L": int(L),
        "D": int(D),
        "adapter_path": str(Path(args.adapter).resolve()),
        "mean_uplift": float(uplift.mean()),
        "p50_uplift": float(np.quantile(uplift, 0.5)),
        "p90_uplift": float(np.quantile(uplift, 0.9)),
        "mean_base_cos": float(base_cos.mean().item()),
        "mean_new_cos": float(new_cos.mean().item()),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Artifacts written under: {out_dir}")


if __name__ == "__main__":
    main()
