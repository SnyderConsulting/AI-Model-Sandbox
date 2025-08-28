#!/usr/bin/env python3
"""
prompt_probe_goonsai.py — quick probe for Goonsai Qwen2.5 NSFW prompt writer

What it does
------------
- Expands short "ideas" into CivitAI/Stable‑Diffusion style prompts via
  HuggingFace Transformers:
    goonsai-com/civitaiprompts/qwen2.5-3B-goonsai-nsfw-100k
- Scores generations with simple, diffusion-relevant text features:
    • token count
    • photography/camera/lighting lexicon hits
    • NSFW anatomy/action lexicon hits (your own, or a built-in starter list)
    • presence of 'Negative prompt:' line
    • refusal heuristics ("I can't…", "As an AI…")
- Saves CSV/JSON results and a small aggregate table (means per run).
- Can sample your JSONL dataset (key='caption') to use *your* domain text as seeds.
- Can auto-build a domain NSFW lexicon from your JSONL sample for better scoring.

Prereqs
-------
- Requires `transformers>=4.53.0` and the Goonsai model downloaded from
  HuggingFace.

Examples
--------
# 1) Use built-in base prompts
python prompt_probe_goonsai.py

# 2) Use your captions.jsonl (sample 500 ideas)
python prompt_probe_goonsai.py \
  --sample_jsonl /workspace/AI-Model-Sandbox/datasets/captions.jsonl \
  --key caption --n 500 --seed 42

# 3) Build a domain NSFW lexicon from your JSONL (sample 5k rows), then use it
python prompt_probe_goonsai.py \
  --build_lexicon_from_jsonl /workspace/AI-Model-Sandbox/datasets/captions.jsonl \
  --key caption --n 5000 --lexicon_out /tmp/nsfw_terms.txt

python prompt_probe_goonsai.py --nsfw_lexicon /tmp/nsfw_terms.txt --n 500 \
  --sample_jsonl /workspace/AI-Model-Sandbox/datasets/captions.jsonl --key caption

Outputs (default /tmp/prompt_probe_goonsai):
- prompt_probe.csv          # per-generation rows + scores
- prompt_probe.json         # same as JSON (with raw text)
- prompt_probe_agg.csv      # simple means across all generations

Notes
-----
- This script is content-agnostic but assumes *adult* NSFW data. Ensure your data is 18+.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ------------------------- lexicons -------------------------

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

# Safe, adult-only seed list (no minor-coded terms).
NSFW_SEED_SUBSTR = {
    "breast",
    "boob",
    "nipple",
    "areola",
    "pussy",
    "vulva",
    "labia",
    "clitoris",
    "clit",
    "vagina",
    "penis",
    "cock",
    "dick",
    "shaft",
    "glans",
    "scrotum",
    "balls",
    "cum",
    "semen",
    "ejaculat",
    "orgasm",
    "aroused",
    "erect",
    "hard-on",
    "handjob",
    "blowjob",
    "oral",
    "deepthroat",
    "spitroast",
    "missionary",
    "doggy",
    "cowgirl",
    "anal",
    "threesome",
    "creampie",
    "squirt",
    "butt",
    "ass",
    "titty",
    "tits",
    "milf",
    "bbw",
    "lingerie",
    "nude",
    "topless",
    "fondl",
    "grop",
    "masturbat",
}

# A small starter explicit vocab; you can override/extend via --nsfw_lexicon
NSFW_TERMS_STARTER = {
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
    "over",
    "under",
    "very",
    "just",
    "also",
    "still",
    "more",
    "most",
    "less",
    "least",
    "some",
    "any",
}

# ------------------------- utils -------------------------


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _safe_write_json(obj: Any, path: Path):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def _tok(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s\-]", " ", text)
    toks = [t for t in text.split() if t and len(t) > 2 and t not in STOPWORDS]
    return toks


def _contains_seed(text: str) -> bool:
    low = text.lower()
    return any(s in low for s in NSFW_SEED_SUBSTR)


def _condense_to_idea(text: str, max_tokens: int = 16) -> str:
    """Heuristic: drop 'negative prompt' blocks, camera tech tags, keep first clause and trim."""
    low = text.lower()
    if "negative prompt:" in low:
        text = text[: low.index("negative prompt:")]
    # Split on sentence-ish boundaries
    parts = re.split(r"[.;\n]+", text)
    s = parts[0] if parts else text
    # Remove common tech tokens
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
    """Reservoir-sample n lines of `key` from a large JSONL without loading all into memory."""
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


def build_lexicon_from_jsonl(
    jsonl: Path, key: str, n: int, out_path: Path, seed: int = 0, max_terms: int = 300
) -> Path:
    """Make a domain NSFW lexicon by contrasting seed-bearing captions vs background captions."""
    caps = _reservoir_sample_jsonl(jsonl, key, n, seed=seed)
    nsfw_counts = Counter()
    bg_counts = Counter()
    for c in caps:
        toks = _tok(c)
        (nsfw_counts if _contains_seed(c) else bg_counts).update(toks)
    # score = nsfw_count - 0.35 * background_count
    scored: List[Tuple[str, float]] = []
    for t, cnt in nsfw_counts.items():
        if t in STOPWORDS:
            continue
        if any(t in p for p in PHOTOGRAPHY_TERMS):
            continue
        score = cnt - 0.35 * bg_counts.get(t, 0)
        if score > 0:
            scored.append((t, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    terms = [t for t, _ in scored[:max_terms]]
    # ensure seeds and starter terms are included
    seeds = sorted({s for s in NSFW_SEED_SUBSTR} | {t for t in NSFW_TERMS_STARTER})
    uniq = list(dict.fromkeys(seeds + terms))  # preserve order, dedupe
    out_path.write_text("\n".join(uniq))
    return out_path


# ------------------------- generation -------------------------


def expand(
    tok: AutoTokenizer,
    model: AutoModelForCausalLM,
    idea: str,
    max_new_tokens: int = 240,
    temperature: float = 0.4,
) -> str:
    sys_prompt = (
        "You are a prompt engineer for diffusion/video models. "
        "Given a short idea, expand it into a CivitAI/Stable‑Diffusion style prompt: "
        "quality tags, subject, anatomy (if relevant), pose/action, setting, lighting, camera/lens, "
        "technical specs. End with an optional 'Negative prompt:' line. Use comma‑separated phrases."
    )
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": idea},
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.85,
        top_k=40,
        repetition_penalty=1.3,
    )
    return tok.decode(out[0], skip_special_tokens=True).strip()


# ------------------------- scoring -------------------------

REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE)


@dataclass
class Score:
    tokens: int
    phrases: int
    photo_terms: int
    nsfw_terms: int
    has_negative: int
    refusal: int


def score_prompt(text: str, nsfw_vocab: Optional[set[str]] = None) -> Score:
    nsfw_vocab = nsfw_vocab or NSFW_TERMS_STARTER
    toks = [t for t in re.split(r"[,\s]+", text) if t]
    phrases = len([p for p in text.split(",") if p.strip()])
    low = text.lower()

    def _count(vocab: Iterable[str]) -> int:
        return sum(1 for term in vocab if term.lower() in low)

    photo = _count(PHOTOGRAPHY_TERMS)
    nsfw = _count(nsfw_vocab)
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


# ------------------------- main -------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default="goonsai-com/civitaiprompts/qwen2.5-3B-goonsai-nsfw-100k",
        help="HuggingFace model repo id",
    )
    ap.add_argument(
        "--prompts_file", type=str, help="Text file with base ideas (one per line)"
    )
    ap.add_argument(
        "--sample_jsonl",
        type=str,
        help="JSONL to sample ideas from (e.g., your captions.jsonl)",
    )
    ap.add_argument(
        "--key", default="caption", help="Field name in JSONL when using --sample_jsonl"
    )
    ap.add_argument(
        "--n", type=int, default=32, help="Number of ideas to test (prompts or samples)"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--use_condense",
        action="store_true",
        help="Condense sampled captions to short 'idea' phrases before expansion",
    )
    ap.add_argument(
        "--nsfw_lexicon",
        type=str,
        help="Optional path to extra NSFW terms (one per line)",
    )
    ap.add_argument(
        "--build_lexicon_from_jsonl",
        type=str,
        help="Build a domain NSFW lexicon from this JSONL (uses --key and --n)",
    )
    ap.add_argument("--lexicon_out", type=str, default="/tmp/nsfw_terms.txt")
    ap.add_argument("--outdir", type=str, default="/tmp/prompt_probe_goonsai")
    ap.add_argument("--max_new_tokens", type=int, default=240)
    ap.add_argument("--temperature", type=float, default=0.4)
    args = ap.parse_args()

    model_id = "goonsai-com/civitaiprompts"
    subfolder = "qwen2.5-3B-goonsai-nsfw-100k"
    tok = AutoTokenizer.from_pretrained(model_id, subfolder=subfolder, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, subfolder=subfolder, use_safetensors=True, torch_dtype=torch.bfloat16, device_map="auto")

    random.seed(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Step 0: optionally build domain lexicon
    nsfw_vocab: Optional[set[str]] = None
    if args.build_lexicon_from_jsonl:
        path = build_lexicon_from_jsonl(
            Path(args.build_lexicon_from_jsonl),
            args.key,
            args.n,
            Path(args.lexicon_out),
            seed=args.seed,
        )
        print(f"[lexicon] wrote {path}")
    if args.nsfw_lexicon and Path(args.nsfw_lexicon).exists():
        nsfw_vocab = {
            ln.strip()
            for ln in Path(args.nsfw_lexicon).read_text().splitlines()
            if ln.strip()
        }
        print(f"[lexicon] loaded {len(nsfw_vocab)} NSFW terms from {args.nsfw_lexicon}")

    # Step 1: collect base ideas
    ideas: List[str] = []
    if args.prompts_file:
        ideas = [
            ln.strip()
            for ln in Path(args.prompts_file).read_text().splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
    elif args.sample_jsonl:
        caps = _reservoir_sample_jsonl(
            Path(args.sample_jsonl), args.key, args.n, seed=args.seed
        )
        ideas = [_condense_to_idea(c) if args.use_condense else c for c in caps]
    else:
        # built-in tiny defaults
        ideas = [
            "serene sunset over city skyline",
            "neon-lit alleyway with puddles",
            "topless woman at the pool, looking over shoulder",
            "fit man in locker room, towel around waist",
            "passionate kiss in the rain, two lovers",
            "sensual boudoir shot on silk sheets",
            "close-up of hands exploring a body",
            "artistic nude in chiaroscuro lighting",
        ]

    if args.n and len(ideas) > args.n:
        # fast subsample to requested n
        random.shuffle(ideas)
        ideas = ideas[: args.n]

    print(f"[probe] ideas: {len(ideas)}  model: {args.model}")

    rows: List[Dict[str, Any]] = []
    for i, idea in enumerate(ideas, 1):
        gen = expand(
            tok,
            model,
            idea,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        sc = score_prompt(gen, nsfw_vocab)
        row = {
            "when": _now(),
            "idx": i,
            "base_idea": idea,
            "model": args.model,
            "gen": gen,
        }
        row.update(asdict(sc))
        rows.append(row)
        print(
            f"[{i:03d}/{len(ideas)}] tokens={sc.tokens}  photo={sc.photo_terms}  nsfw={sc.nsfw_terms}  "
            f"neg={sc.has_negative}  refusal={sc.refusal}"
        )

    df = pd.DataFrame(rows)
    csv_path = Path(outdir) / "prompt_probe.csv"
    json_path = Path(outdir) / "prompt_probe.json"
    df.to_csv(csv_path, index=False)
    _safe_write_json({"rows": rows}, json_path)

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
    agg_csv = Path(outdir) / "prompt_probe_agg.csv"
    pd.DataFrame([agg]).to_csv(agg_csv, index=False)

    print("\n=== Aggregate (means) ===")
    for k, v in agg.items():
        print(f"{k:>14}: {v:.2f}")
    print(f"\nWrote:\n - {csv_path}\n - {json_path}\n - {agg_csv}")


if __name__ == "__main__":
    main()
