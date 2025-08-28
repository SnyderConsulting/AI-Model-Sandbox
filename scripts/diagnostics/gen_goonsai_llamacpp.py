#!/usr/bin/env python3
from __future__ import annotations
import argparse
from llama_cpp import Llama

SYS = (
    "You are a prompt engineer for diffusion/video models. "
    "Given a short idea, expand it into a CivitAI/Stable-Diffusion style prompt: "
    "quality tags, subject, anatomy (if relevant), pose/action, setting, lighting, camera/lens, "
    "technical specs. End with an optional 'Negative prompt:' line. Use comma-separated phrases."
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--idea", required=True)
    ap.add_argument("--n_ctx", type=int, default=8192)
    ap.add_argument(
        "--n_gpu_layers",
        type=int,
        default=-1,
        help="-1 = full GPU offload if possible",
    )
    ap.add_argument("--max_tokens", type=int, default=256)
    args = ap.parse_args()

    llm = Llama(
        model_path=args.gguf,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        logits_all=False,
        seed=0,
        verbose=False,
    )

    prompt = f"{SYS}\n\nIdea: {args.idea}\nPrompt:"
    out = llm.create_completion(
        prompt=prompt,
        max_tokens=args.max_tokens,
        temperature=0.4,
        top_p=0.85,
        top_k=40,
        repeat_penalty=1.3,
        presence_penalty=0.5,
        repeat_last_n=64,
    )
    print(out["choices"][0]["text"].strip())


if __name__ == "__main__":
    main()
