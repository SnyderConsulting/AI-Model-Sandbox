import os
import argparse
import random
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from adapter import PerceiverBridge
from data import CaptionsJSONL
from utils import (
    cosine_loss,
    mse_loss,
    match_stats_loss,
    now,
    tensor_stats,
    grad_norms,
    JsonlLogger,
    detect_anomaly_if,
)


try:
    from teachers.wan_text_teacher_from_wan import WanTextTeacher
except Exception:  # pragma: no cover - optional dependency
    WanTextTeacher = None

try:
    from teachers.umt5_hf_projection import HFUMT5Teacher
except Exception:  # pragma: no cover - optional dependency
    HFUMT5Teacher = None


def seed_all(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_text(batch: list[str]) -> list[str]:
    return batch


def get_llm_hidden(model, tok, texts, device: str):
    with torch.no_grad():
        enc = tok(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=4096
        ).to(device)
        out = model(**enc, output_hidden_states=True, use_cache=False)
        h = out.hidden_states[-1]
        mask = enc["attention_mask"].bool()
    return h, mask


def build_autocast_and_scaler(amp_mode: str):
    amp_mode = (amp_mode or "bf16").lower()
    if amp_mode == "fp16":
        autocast_cm = torch.amp.autocast("cuda", dtype=torch.float16)
        scaler = torch.amp.GradScaler("cuda")
    elif amp_mode == "bf16":
        autocast_cm = torch.amp.autocast("cuda", dtype=torch.bfloat16)
        scaler = None
    else:

        class _NullCtx:
            def __enter__(self):
                return None

            def __exit__(self, *args):
                return False

        autocast_cm = _NullCtx()
        scaler = None
    return autocast_cm, scaler, amp_mode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", default="/workspace/data/captions.jsonl")
    ap.add_argument("--model_dir", default="/workspace/models/MythoMax-L2-13B")
    ap.add_argument("--save_dir", default="/workspace/checkpoints/bridge")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--adamw", type=float, default=0.01)
    ap.add_argument("--warmup_steps", type=int, default=200)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--L_wan", type=int, default=512)
    ap.add_argument("--d_wan", type=int, default=4096)
    ap.add_argument("--teacher", choices=["wan", "umt5_hf", "none"], default="wan")
    ap.add_argument("--teacher_hf_name", default="google/umt5-xxl")
    ap.add_argument("--d_mid", type=int, default=1024)
    ap.add_argument("--n_blocks", type=int, default=3)
    ap.add_argument("--heads_mid", type=int, default=16)
    ap.add_argument("--amp", choices=["bf16", "fp16", "off"], default="bf16")
    ap.add_argument("--debug_log_name", default="bridge_debug.jsonl")
    ap.add_argument("--debug_dump_every", type=int, default=0)
    ap.add_argument("--detect_anomaly_steps", type=int, default=2)
    args = ap.parse_args()

    seed_all(123)
    os.makedirs(args.save_dir, exist_ok=True)
    logger = JsonlLogger(os.path.join(args.save_dir, args.debug_log_name))
    dumps_dir = Path(args.save_dir) / "dumps"
    dumps_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{now()}] loading LLM from {args.model_dir}")
    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    llm_dtype = torch.bfloat16 if args.amp != "off" else torch.float32
    llm = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=llm_dtype, device_map="auto"
    )
    llm.eval().requires_grad_(False)

    teacher = None
    if args.teacher == "wan" and WanTextTeacher is not None:
        teacher = WanTextTeacher(L_wan=args.L_wan, d_wan=None, device=args.device).to(
            args.device
        )
        for p in teacher.parameters():
            p.requires_grad = False
        print(f"[{now()}] Using WAN teacher (real UMT5_Wan).")
    elif args.teacher == "umt5_hf" and HFUMT5Teacher is not None:
        teacher = HFUMT5Teacher(
            hf_name=args.teacher_hf_name,
            L_wan=args.L_wan,
            d_wan=args.d_wan,
            device=args.device,
        ).to(args.device)
        print(f"[{now()}] Using HF UMT5 + proj teacher.")
    else:
        print(f"[{now()}] No teacher; training with shape/stat constraints only.")

    if teacher is not None:
        with torch.no_grad():
            H_probe, M_probe = teacher(["."])
        auto_L, auto_D = int(H_probe.shape[1]), int(H_probe.shape[2])
        if args.L_wan != auto_L or args.d_wan != auto_D:
            print(
                f"[{now()}] Auto-detected Wan text interface: L={auto_L}, D={auto_D} (overriding CLI defaults)"
            )
            args.L_wan, args.d_wan = auto_L, auto_D

    bridge = PerceiverBridge(
        d_llm=llm.config.hidden_size,
        d_wan=args.d_wan,
        L_wan=args.L_wan,
        d_mid=args.d_mid,
        n_heads=args.heads_mid,
        n_blocks=args.n_blocks,
        return_attn=False,
    ).to(args.device)

    optim = torch.optim.AdamW(bridge.parameters(), lr=args.lr, weight_decay=args.adamw)
    autocast_cm, scaler, amp_mode = build_autocast_and_scaler(args.amp)

    ds = CaptionsJSONL(args.captions, min_chars=1)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_text,
        drop_last=True,
    )

    global_step = 0
    anomaly_left = max(0, args.detect_anomaly_steps)

    for epoch in range(args.epochs):
        for batch_i, texts in enumerate(dl):
            use_anomaly = anomaly_left > 0
            anomaly_left -= 1 if use_anomaly else 0

            with detect_anomaly_if(use_anomaly):
                llm_h, llm_mask = get_llm_hidden(llm, tok, texts, device=args.device)

                with autocast_cm:
                    h_hat = bridge(llm_h.to(args.device), llm_mask.to(args.device))
                    loss = torch.zeros([], device=args.device)

                    stats_record: dict[str, object] = {
                        "step": global_step + 1,
                        "epoch": epoch,
                        "amp": amp_mode,
                        "batch_size": len(texts),
                    }
                    if teacher is not None:
                        with torch.no_grad():
                            h_t, m_t = teacher(texts)
                        if h_hat.dtype != h_t.dtype:
                            h_hat = h_hat.to(h_t.dtype)

                        loss_mse = mse_loss(h_hat, h_t, mask=m_t)
                        loss_cos = cosine_loss(h_hat, h_t, mask=m_t)
                        loss_stats = match_stats_loss(h_hat, h_t, mask=m_t)
                        loss = loss + (
                            1.0 * loss_mse + 0.5 * loss_cos + 0.1 * loss_stats
                        )

                        stats_record["llm_h"] = tensor_stats(llm_h, llm_mask, "llm_h")
                        stats_record["h_hat"] = tensor_stats(h_hat, None, "bridge_out")
                        stats_record["h_t"] = tensor_stats(h_t, m_t, "teacher_out")
                        stats_record["losses"] = {
                            "mse": float(loss_mse.item()),
                            "cos": float(loss_cos.item()),
                            "stats": float(loss_stats.item()),
                        }
                    else:
                        mu = h_hat.mean(dim=(0, 1))
                        sigma = h_hat.std(dim=(0, 1))
                        loss = loss + 0.1 * (
                            torch.abs(mu).mean() + torch.abs(sigma - 1.0).mean()
                        )
                        stats_record["llm_h"] = tensor_stats(llm_h, llm_mask, "llm_h")
                        stats_record["h_hat"] = tensor_stats(h_hat, None, "bridge_out")
                        stats_record["losses"] = {"dist_only": float(loss.item())}

                bad = (
                    not torch.isfinite(loss).item()
                    or stats_record["h_hat"]["n_nan"] > 0
                    or (teacher is not None and stats_record["h_t"]["n_nan"] > 0)
                )
                if bad:
                    dump_path = dumps_dir / f"nan_step{global_step + 1:06d}.pt"
                    torch.save(
                        {
                            "texts": texts,
                            "llm_h": llm_h.cpu(),
                            "llm_mask": llm_mask.cpu(),
                            "h_hat": h_hat.detach().cpu(),
                            "teacher_out": (
                                h_t.detach().cpu() if teacher is not None else None
                            ),
                            "teacher_mask": (
                                m_t.cpu() if teacher is not None else None
                            ),
                            "bridge_state": bridge.state_dict(),
                        },
                        dump_path,
                    )
                    print(
                        f"[{now()}] 🚨 Non-finite detected at step {global_step + 1}. Dumped to {dump_path}. Aborting."
                    )
                    return

                optim.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
                    scaler.step(optim)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
                    optim.step()

                stats_record["grads"] = grad_norms(bridge, topk=8)
                logger.log(stats_record)

                global_step += 1
                if global_step % args.log_every == 0:
                    print(
                        f"[{now()}] ep{epoch} it{batch_i} step{global_step} "
                        f"loss={float(loss.item()):.6f} "
                        f"| h_hat μ={stats_record['h_hat']['mean']:.4f} σ={stats_record['h_hat']['std']:.4f} "
                        f"| teacher μ={stats_record.get('h_t', {}).get('mean', '-')}"
                    )

                if args.debug_dump_every and (global_step % args.debug_dump_every == 0):
                    dump_path = dumps_dir / f"step{global_step:06d}.pt"
                    torch.save(
                        {
                            "texts": texts,
                            "llm_h": llm_h.cpu(),
                            "h_hat": h_hat.detach().cpu(),
                            "teacher_out": (
                                h_t.detach().cpu() if teacher is not None else None
                            ),
                        },
                        dump_path,
                    )

        ckpt = {"bridge": bridge.state_dict(), "cfg": vars(args)}
        outp = Path(args.save_dir) / f"bridge_epoch{epoch:02d}.pth"
        torch.save(ckpt, outp)
        print(f"[{now()}] saved {outp}")

    print(f"[{now()}] done.")


if __name__ == "__main__":
    main()
