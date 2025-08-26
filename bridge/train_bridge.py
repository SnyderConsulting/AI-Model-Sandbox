import argparse
import os
import random
import signal
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

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

# Teacher imports (Wan is preferred)
try:
    from teachers.wan_text_teacher_from_wan import WanTextTeacher
except Exception:
    WanTextTeacher = None
try:
    from teachers.umt5_hf_projection import HFUMT5Teacher
except Exception:
    HFUMT5Teacher = None

# --- performance knobs: TF32 on A100 ---
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


def seed_all(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_text(batch):
    return batch


def get_llm_hidden(model, tok, texts, device, llm_max_length):
    with torch.no_grad():
        enc = tok(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=llm_max_length,
        ).to(device)
        out = model(**enc, output_hidden_states=True, use_cache=False)
        h = out.hidden_states[-1]  # [B, Lt, d_llm]
        mask = enc["attention_mask"].bool()
    return h, mask


def build_autocast_and_scaler(amp_mode):
    amp_mode = (amp_mode or "bf16").lower()
    if amp_mode == "fp16":
        autocast_cm = torch.amp.autocast("cuda", dtype=torch.float16)
        scaler = torch.amp.GradScaler("cuda")
    elif amp_mode == "bf16":
        autocast_cm = torch.amp.autocast("cuda", dtype=torch.bfloat16)
        scaler = None
    else:  # off

        class NullCtx:
            def __enter__(self):
                return None

            def __exit__(self, *args):
                return False

        autocast_cm, scaler = NullCtx(), None
    return autocast_cm, scaler, amp_mode


def main():
    ap = argparse.ArgumentParser()
    # data / models
    ap.add_argument("--captions", default="/workspace/data/captions.jsonl")
    ap.add_argument("--model_dir", default="/workspace/models/MythoMax-L2-13B")
    ap.add_argument("--save_dir", default="/workspace/checkpoints/bridge")
    ap.add_argument("--teacher", choices=["wan", "umt5_hf", "none"], default="wan")
    ap.add_argument("--teacher_hf_name", default="google/umt5-xxl")
    # dims
    ap.add_argument("--L_wan", type=int, default=512)
    ap.add_argument(
        "--d_wan", type=int, default=4096
    )  # auto-overridden by teacher probe
    ap.add_argument("--d_mid", type=int, default=1024)
    ap.add_argument("--n_blocks", type=int, default=3)
    ap.add_argument("--heads_mid", type=int, default=16)
    # training
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument(
        "--max_steps", type=int, default=0, help="Stop after N steps (0=off)."
    )
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--adamw", type=float, default=0.01)
    ap.add_argument("--amp", choices=["bf16", "fp16", "off"], default="bf16")
    ap.add_argument(
        "--llm_max_length", type=int, default=512, help="LLM token cap (was 4096)."
    )
    ap.add_argument("--device", default="cuda")
    # logging / saving
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--debug_log_name", default="bridge_debug.jsonl")
    ap.add_argument("--debug_dump_every", type=int, default=0)
    ap.add_argument(
        "--save_every_steps", type=int, default=1000, help="0=off; save every N steps."
    )
    ap.add_argument("--save_keep_last_k", type=int, default=5)
    ap.add_argument("--detect_anomaly_steps", type=int, default=2)
    args = ap.parse_args()

    seed_all(123)
    os.makedirs(args.save_dir, exist_ok=True)
    logger = JsonlLogger(os.path.join(args.save_dir, args.debug_log_name))
    dumps_dir = Path(args.save_dir) / "dumps"
    dumps_dir.mkdir(parents=True, exist_ok=True)

    # support on-demand checkpointing via USR1
    _save_now = {"flag": False}

    def _sigusr1_handler(sig, frame):
        _save_now["flag"] = True

    try:
        signal.signal(signal.SIGUSR1, _sigusr1_handler)
    except Exception:
        pass

    print(f"[{now()}] loading LLM from {args.model_dir}")
    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    llm_dtype = torch.bfloat16 if args.amp != "off" else torch.float32
    llm = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=llm_dtype, device_map="auto"
    )
    llm.eval().requires_grad_(False)

    # Teacher
    teacher = None
    if args.teacher == "wan" and WanTextTeacher is not None:
        teacher = WanTextTeacher(L_wan=args.L_wan, d_wan=None, device=args.device).to(
            args.device
        )
        for p in teacher.parameters():
            p.requires_grad = False
        print(f"[{now()}] Using WAN teacher (real UMT5).")
    elif args.teacher == "umt5_hf" and HFUMT5Teacher is not None:
        teacher = HFUMT5Teacher(
            hf_name=args.teacher_hf_name,
            L_wan=args.L_wan,
            d_wan=args.d_wan,
            device=args.device,
        ).to(args.device)
        print(f"[{now()}] Using HF UMT5 + proj teacher.")
    else:
        print(f"[{now()}] No teacher; distribution constraints only.")

    # Auto-detect Wan dims from teacher
    if teacher is not None:
        with torch.no_grad():
            H_probe, M_probe = teacher(["."])
        auto_L, auto_D = int(H_probe.shape[1]), int(H_probe.shape[2])
        if args.L_wan != auto_L or args.d_wan != auto_D:
            print(
                f"[{now()}] Auto-detected Wan interface: L={auto_L}, D={auto_D} (overriding)."
            )
            args.L_wan, args.d_wan = auto_L, auto_D

    # Bridge
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

    # Data
    ds = CaptionsJSONL(args.captions, min_chars=1)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        collate_fn=collate_text,
        drop_last=True,
    )
    print(f"[{now()}] batches_per_epoch={len(dl)} (batch_size={args.batch_size})")

    global_step = 0
    anomaly_left = max(0, args.detect_anomaly_steps)
    accum_step = 0

    for epoch in range(args.epochs):
        for batch_i, texts in enumerate(dl):
            if args.max_steps and global_step >= args.max_steps:
                ck_step = Path(args.save_dir) / f"bridge_step{global_step:06d}.pth"
                torch.save({"bridge": bridge.state_dict(), "cfg": vars(args)}, ck_step)
                print(
                    f"[{now()}] Reached max_steps={args.max_steps}. Saved {ck_step} and exiting."
                )
                return

            use_anomaly = anomaly_left > 0
            anomaly_left -= 1 if use_anomaly else 0

            with detect_anomaly_if(use_anomaly):
                # LLM features
                llm_h, llm_mask = get_llm_hidden(
                    llm,
                    tok,
                    texts,
                    device=args.device,
                    llm_max_length=args.llm_max_length,
                )

                with autocast_cm:
                    h_hat = bridge(llm_h.to(args.device), llm_mask.to(args.device))
                    # teacher supervision
                    loss = torch.zeros([], device=args.device)
                    stats_record = {
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
                        # weights: slightly higher stats to tighten scale
                        loss_total = 1.0 * loss_mse + 0.5 * loss_cos + 0.2 * loss_stats
                        loss = loss + loss_total / args.grad_accum

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
                        loss = (
                            loss
                            + (
                                0.1
                                * (torch.abs(mu).mean() + torch.abs(sigma - 1.0).mean())
                            )
                            / args.grad_accum
                        )
                        stats_record["llm_h"] = tensor_stats(llm_h, llm_mask, "llm_h")
                        stats_record["h_hat"] = tensor_stats(h_hat, None, "bridge_out")
                        stats_record["losses"] = {
                            "dist_only": float(loss.item() * args.grad_accum)
                        }

                # NaN guard
                bad = (
                    (not torch.isfinite(loss).item())
                    or (stats_record["h_hat"]["n_nan"] > 0)
                    or (teacher is not None and stats_record["h_t"]["n_nan"] > 0)
                )
                if bad:
                    dump_path = dumps_dir / f"nan_step{global_step+1:06d}.pt"
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
                        f"[{now()}] 🚨 Non-finite at step {global_step+1}. Dumped {dump_path}. Aborting."
                    )
                    return

                # Backprop (with grad accumulation)
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                accum_step += 1
                do_step = (accum_step % args.grad_accum) == 0

                if do_step:
                    torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
                    if scaler is not None:
                        scaler.step(optim)
                        scaler.update()
                    else:
                        optim.step()
                    optim.zero_grad(set_to_none=True)
                    accum_step = 0
                    global_step += 1

                    # log
                    stats_record["grads"] = grad_norms(bridge, topk=8)
                    # add sigma ratio to console visibility
                    if "h_t" in stats_record:
                        tstd = stats_record["h_t"]["std"]
                        bstd = stats_record["h_hat"]["std"]
                        stats_record["sigma_ratio"] = (
                            (bstd / tstd) if tstd else float("nan")
                        )
                    logger.log(stats_record)

                    if global_step % args.log_every == 0:
                        comp = stats_record.get("losses", {})
                        tstd = stats_record.get("h_t", {}).get("std", float("nan"))
                        bstd = stats_record["h_hat"]["std"]
                        ratio = (
                            (bstd / tstd)
                            if (isinstance(tstd, float) and tstd != 0)
                            else float("nan")
                        )
                        print(
                            f"[{now()}] ep{epoch} it{batch_i} step{global_step} "
                            f"loss={float((sum(comp.values()) if comp else loss.item())):.6f} "
                            f"| mse={comp.get('mse','-'):.6f} cos={comp.get('cos','-'):.6f} stats={comp.get('stats','-'):.6f} "
                            f"| ĥ σ={bstd:.4f}  teacher σ={tstd:.4f}  σ_ratio={ratio:.3f}"
                        )

                    # periodic checkpoint
                    if args.save_every_steps and (
                        global_step % args.save_every_steps == 0
                    ):
                        ck = Path(args.save_dir) / f"bridge_step{global_step:06d}.pth"
                        torch.save(
                            {"bridge": bridge.state_dict(), "cfg": vars(args)}, ck
                        )
                        print(f"[{now()}] saved {ck}")
                        if args.save_keep_last_k > 0:
                            step_ckpts = sorted(
                                Path(args.save_dir).glob("bridge_step*.pth")
                            )
                            for old in step_ckpts[: -args.save_keep_last_k]:
                                try:
                                    old.unlink()
                                except Exception:
                                    pass

                    # on-demand checkpoint (kill -USR1 <pid>)
                    if _save_now["flag"]:
                        _save_now["flag"] = False
                        ck = Path(args.save_dir) / f"bridge_step{global_step:06d}.pth"
                        torch.save(
                            {"bridge": bridge.state_dict(), "cfg": vars(args)}, ck
                        )
                        print(f"[{now()}] Saved on USR1 -> {ck}")

        # end epoch save
        ckpt = {"bridge": bridge.state_dict(), "cfg": vars(args)}
        outp = Path(args.save_dir) / f"bridge_epoch{epoch:02d}.pth"
        torch.save(ckpt, outp)
        print(f"[{now()}] saved {outp}")

    print(f"[{now()}] done.")


if __name__ == "__main__":
    main()
