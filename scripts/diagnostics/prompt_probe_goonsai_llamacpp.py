#!/usr/bin/env python3
"""Probe Goonsai Qwen2.5-3B NSFW-100k (GGUF via llama-cpp) using your captions as seeds.
Scores: token count, comma-phrases, photography terms, NSFW terms, 'Negative prompt:' presence, refusals.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional, Dict, Any

import pandas as pd
from llama_cpp import Llama

PHOTOGRAPHY_TERMS = {
    "35mm",
    "50mm",
    "85mm",
    "105mm",
    "200mm",
    "anamorphic",
    "aperture",
    "bokeh",
    "cinematic",
    "close-up",
    "composition",
    "depth of field",
    "dslr",
    "hdr",
    "high key",
    "low key",
    "f/1.4",
    "f/1.8",
    "f/2.8",
    "f/4",
    "f/8",
    "iso",
    "lens flare",
    "macro",
    "noise",
    "overhead",
    "portrait",
    "prime lens",
    "rim light",
    "softbox",
    "studio lighting",
    "telephoto",
    "ultra-wide",
    "volumetric",
    "wide angle",
    "white balance",
    "8k",
    "raw photo",
    "rule of thirds",
    "exposure",
    "shutter speed",
    "tripod",
    "diffused light",
    "tilt-shift",
    "dof",
    "film grain",
    "chromatic aberration",
    "motion blur",
    "cinestill",
}
NSFW_TERMS = {
    "nude",
    "topless",
    "breasts",
    "nipples",
    "areola",
    "vulva",
    "labia",
    "clitoris",
    "penis",
    "erection",
    "cum",
    "semen",
    "oral",
    "doggy",
    "cowgirl",
    "missionary",
    "anal",
    "threesome",
    "orgasm",
    "aroused",
    "explicit",
    "deepthroat",
    "creampie",
    "squirt",
    "spitroast",
    "lingerie",
    "butt",
    "ass",
    "vagina",
    "groping",
    "fondling",
}
REFUSAL_PATTERNS = [
    r"\b(as an ai|i am unable|i'm unable|i do not|i cannot|i can('|)t|cannot comply)\b",
    r"\b(content policy|guidelines|safety)\b",
    r"\b(refuse|not appropriate|cannot help with)\b",
]
STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "for",
    "to",
    "in",
    "on",
    "at",
    "with",
    "by",
    "from",
    "is",
    "are",
    "be",
    "this",
    "that",
    "these",
    "those",
    "as",
    "into",
    "it",
    "its",
    "their",
    "his",
    "her",
    "your",
    "you",
    "we",
    "they",
    "them",
    "he",
    "she",
    "i",
    "my",
    "me",
    "our",
    "ours",
    "yourself",
    "themselves",
    "himself",
    "herself",
}

SYS = (
    "You are a prompt engineer for diffusion/video models. "
    "Given a short idea, expand it into a CivitAI/Stable-Diffusion style prompt: "
    "quality tags, subject, anatomy (if relevant), pose/action, setting, lighting, camera/lens, "
    "technical specs. End with an optional 'Negative prompt:' line. Use comma-separated phrases."
)


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _condense_to_idea(text: str, max_tokens: int = 16) -> str:
    low = text.lower()
    if "negative prompt:" in low:
        text = text[: low.index("negative prompt:")]
    parts = re.split(r"[.;\n]+", text)
    s = parts[0] if parts else text
    drop = {
        "masterpiece",
        "best quality",
        "8k",
        "4k",
        "dslr",
        "hdr",
        "raw",
        "highres",
        "nsfw",
    }
    tokens = [t for t in re.split(r"[,\s]+", s) if t and t.lower() not in drop]
    return " ".join(tokens[:max_tokens])


def _reservoir_sample_jsonl(path: Path, key: str, n: int, seed: int = 0) -> List[str]:
    random.seed(seed)
    out: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if key not in obj or not obj[key]:
                continue
            txt = str(obj[key]).strip()
            if not txt:
                continue
            if len(out) < n:
                out.append(txt)
            else:
                j = random.randint(1, i)
                if j <= n:
                    out[j - 1] = txt
    return out


def _count_terms(text: str, vocab: Iterable[str]) -> int:
    low = text.lower()
    return sum(1 for term in vocab if term in low)


@dataclass
class Score:
    tokens: int
    phrases: int
    photo_terms: int
    nsfw_terms: int
    has_negative: int
    refusal: int


REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE)


def score_prompt(text: str, nsfw_vocab: Optional[set[str]] = None) -> Score:
    nsfw_vocab = nsfw_vocab or NSFW_TERMS
    toks = [t for t in re.split(r"[,\s]+", text) if t]
    phrases = len([p for p in text.split(",") if p.strip()])
    low = text.lower()
    photo = _count_terms(low, PHOTOGRAPHY_TERMS)
    nsfw = _count_terms(low, nsfw_vocab)
    has_neg = int("negative prompt:" in low)
    refusal = int(bool(REFUSAL_RE.search(low)))
    return Score(
        tokens=len(toks),
        phrases=phrases,
        photo_terms=photo,
        nsfw_terms=nsfw,
        has_negative=has_neg,
        refusal=refusal,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument(
        "--sample_jsonl",
        type=str,
        default="/workspace/AI-Model-Sandbox/datasets/captions.jsonl",
    )
    ap.add_argument("--key", default="caption")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--use_condense", action="store_true")
    ap.add_argument("--nsfw_lexicon", type=str)
    ap.add_argument("--outdir", type=str, default="/tmp/prompt_probe_goonsai")
    ap.add_argument("--n_ctx", type=int, default=8192)
    ap.add_argument("--n_gpu_layers", type=int, default=-1)
    ap.add_argument("--max_tokens", type=int, default=256)
    args = ap.parse_args()

    nsfw_vocab = None
    if args.nsfw_lexicon and Path(args.nsfw_lexicon).exists():
        nsfw_vocab = {
            ln.strip()
            for ln in Path(args.nsfw_lexicon).read_text().splitlines()
            if ln.strip()
        }
        print(f"[lexicon] loaded {len(nsfw_vocab)} terms from {args.nsfw_lexicon}")

    ideas = _reservoir_sample_jsonl(
        Path(args.sample_jsonl), args.key, args.n, seed=args.seed
    )
    if args.use_condense:
        ideas = [_condense_to_idea(x) for x in ideas]

    llm = Llama(
        model_path=args.gguf,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        logits_all=False,
        seed=0,
        verbose=False,
    )

    rows: List[Dict[str, Any]] = []
    for i, idea in enumerate(ideas, 1):
        prompt = f"{SYS}\n\nIdea: {idea}\nPrompt:"
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
        gen = (out["choices"][0]["text"] or "").strip()
        sc = score_prompt(gen, nsfw_vocab)
        row = {
            "when": time.strftime("%Y-%m-%d %H:%M:%S"),
            "idx": i,
            "base_idea": idea,
            "gen": gen,
        }
        row.update(asdict(sc))
        rows.append(row)
        print(
            f"[{i:03d}/{len(ideas)}] tokens={sc.tokens} photo={sc.photo_terms} nsfw={sc.nsfw_terms} "
            f"neg={sc.has_negative} refusal={sc.refusal}"
        )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "prompt_probe.csv", index=False)
    (outdir / "prompt_probe.json").write_text(
        json.dumps({"rows": rows}, indent=2, ensure_ascii=False)
    )
    agg = (
        df[
            [
                "tokens",
                "phrases",
                "photo_terms",
                "nsfw_terms",
                "has_negative",
                "refusal",
            ]
        ]
        .mean()
        .to_dict()
    )
    pd.DataFrame([agg]).to_csv(outdir / "prompt_probe_agg.csv", index=False)

    print("\n=== Aggregate (means) ===")
    for k, v in agg.items():
        print(f"{k:>14}: {v:.2f}")
    print(
        f"\nWrote:\n - {outdir/'prompt_probe.csv'}\n - {outdir/'prompt_probe.json'}\n - {outdir/'prompt_probe_agg.csv'}"
    )


if __name__ == "__main__":
    main()
