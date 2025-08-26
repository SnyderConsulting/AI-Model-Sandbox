import argparse
import os
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from adapter import PerceiverBridge
from data import CaptionsJSONL
from utils import cosine_loss, match_stats_loss, mse_loss, now
from teachers.wan_text_teacher_from_wan import WanTextTeacher


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


def default_paths() -> tuple[Path, Path, Path]:
    repo_root = Path(__file__).resolve().parents[1]
    captions = repo_root / "data" / "captions.jsonl"
    model_dir = repo_root / "models" / "MythoMax-L2-13B"
    save_dir = repo_root / "checkpoints" / "bridge"
    return captions, model_dir, save_dir


def main() -> None:
    captions_def, model_def, save_def = map(str, default_paths())
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", default=captions_def)
    ap.add_argument("--model_dir", default=model_def)
    ap.add_argument("--save_dir", default=save_def)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--adamw", type=float, default=0.01)
    ap.add_argument("--warmup_steps", type=int, default=200)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--L_wan", type=int, default=512)
    ap.add_argument("--d_wan", type=int, default=3072)
    ap.add_argument("--teacher", choices=["wan", "none"], default="wan")
    ap.add_argument("--d_mid", type=int, default=1024)
    ap.add_argument("--n_blocks", type=int, default=3)
    ap.add_argument("--heads_mid", type=int, default=16)
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()

    seed_all(123)
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"[{now()}] loading LLM from {args.model_dir}")
    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    llm = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=(
            torch.bfloat16
            if (torch.cuda.is_available() and not args.fp16)
            else torch.float16
        ),
        device_map="auto",
    )
    llm.eval().requires_grad_(False)

    teacher = None
    if args.teacher == "wan":
        teacher = WanTextTeacher(L_wan=args.L_wan, d_wan=None, device=args.device).to(
            args.device
        )
        for p in teacher.parameters():
            p.requires_grad = False
        print(f"[{now()}] Using WAN teacher (real UMT5→Wan).")
    else:
        print(
            f"[{now()}] No teacher; training with distribution/shape constraints only."
        )

    if teacher is not None:
        with torch.no_grad():
            h_probe, _ = teacher(["."])
        auto_L, auto_D = h_probe.shape[1], h_probe.shape[2]
        if args.L_wan != auto_L or args.d_wan != auto_D:
            print(
                f"[{now()}] Auto-detected Wan text interface: L={auto_L}, D={auto_D} (overriding CLI defaults)"
            )
            args.L_wan, args.d_wan = int(auto_L), int(auto_D)

    bridge = PerceiverBridge(
        d_llm=llm.config.hidden_size,
        d_wan=args.d_wan,
        L_wan=args.L_wan,
        d_mid=args.d_mid,
        n_heads=args.heads_mid,
        n_blocks=args.n_blocks,
    ).to(args.device)

    ds = CaptionsJSONL(args.captions, min_chars=1)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_text,
        drop_last=True,
    )

    optim = torch.optim.AdamW(bridge.parameters(), lr=args.lr, weight_decay=args.adamw)
    scaler = torch.amp.GradScaler("cuda") if args.fp16 else None
    autocast_dtype = torch.float16 if args.fp16 else torch.bfloat16

    global_step = 0
    for epoch in range(args.epochs):
        for batch_i, texts in enumerate(dl):
            bridge.train()
            llm_h, llm_mask = get_llm_hidden(llm, tok, texts, device=args.device)
            with torch.amp.autocast("cuda", dtype=autocast_dtype):
                h_hat = bridge(llm_h.to(args.device), llm_mask.to(args.device))
                loss = 0.0
                if teacher is not None:
                    h_t, m_t = teacher(texts)
                    loss_mse = mse_loss(h_hat, h_t, mask=m_t)
                    loss_cos = cosine_loss(h_hat, h_t, mask=m_t)
                    loss_stats = match_stats_loss(h_hat, h_t, mask=m_t)
                    loss = loss + (1.0 * loss_mse + 0.5 * loss_cos + 0.1 * loss_stats)
                else:
                    mu = h_hat.mean(dim=(0, 1))
                    sigma = h_hat.std(dim=(0, 1))
                    loss = loss + 0.1 * (
                        torch.abs(mu).mean() + torch.abs(sigma - 1.0).mean()
                    )

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

            global_step += 1
            if global_step % args.log_every == 0:
                print(
                    f"[{now()}] ep{epoch} it{batch_i} step{global_step} loss={loss.item():.4f}"
                )

        ckpt = {"bridge": bridge.state_dict(), "cfg": vars(args)}
        if teacher is not None and hasattr(teacher, "proj"):
            ckpt["teacher_proj"] = teacher.proj.state_dict()
        outp = Path(args.save_dir) / f"bridge_epoch{epoch:02d}.pth"
        torch.save(ckpt, outp)
        print(f"[{now()}] saved {outp}")

    print(f"[{now()}] done.")


if __name__ == "__main__":
    main()
