#!/usr/bin/env python3
# thesby_vl_probe.py — batch probe for Thesby Qwen2.5-VL-7B-NSFW-Caption-V3 (text-only)
# - Expands short ideas into diffusion-style prompts (comma-separated phrases, optional 'Negative prompt:')
# - Scores outputs on tokens, phrases, photo_terms, nsfw_terms, has_negative, refusal
# - Can sample your JSONL dataset (key='caption') or read a prompts file
# - Optional comparison to a local MythoMax LLaMA model (HF directory) for side-by-side results

from __future__ import annotations
import argparse, json, random, re, time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional, Dict, Any

import pandas as pd
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM

PHOTOGRAPHY_TERMS = {
    "35mm","50mm","85mm","105mm","200mm","anamorphic","aperture","bokeh","cinematic","close-up",
    "composition","depth of field","dslr","hdr","high key","low key","f/1.4","f/1.8","f/2.8","f/4",
    "f/8","iso","lens flare","macro","noise","overhead","portrait","prime lens","rim light",
    "softbox","studio lighting","telephoto","ultra-wide","volumetric","wide angle","white balance",
    "8k","raw photo","rule of thirds","exposure","shutter speed","tripod","diffused light",
    "tilt-shift","dof","film grain","chromatic aberration","motion blur","cinestill"
}
NSFW_TERMS = {
    "nude","topless","breasts","nipples","areola","vulva","labia","clitoris","penis","erection","cum",
    "semen","oral","doggy","cowgirl","missionary","anal","threesome","orgasm","aroused","explicit",
    "deepthroat","creampie","squirt","spitroast","lingerie","butt","ass","vagina","groping","fondling"
}
REFUSAL_PATTERNS = [
    r"\b(as an ai|i am unable|i'm unable|i do not|i cannot|i can('|)t|cannot comply)\b",
    r"\b(content policy|guidelines|safety)\b",
    r"\b(refuse|not appropriate|cannot help with)\b",
]
STOPWORDS = {"the","a","an","and","or","of","for","to","in","on","at","with","by","from","is","are","be",
             "this","that","these","those","as","into","it","its","their","his","her","your","you","we",
             "they","them","he","she","i","my","me","our","ours","yourself","themselves","himself","herself"}

SYS = ("You are a prompt engineer for diffusion/video models. "
       "Given a short idea, expand it into a CivitAI/Stable‑Diffusion style prompt: "
       "quality tags, subject, anatomy (if relevant), pose/action, setting, lighting, camera/lens, "
       "technical specs. End with an optional 'Negative prompt:' line. Use comma‑separated phrases.")

def _now() -> str: return time.strftime("%Y-%m-%d %H:%M:%S")

def _condense_to_idea(text: str, max_tokens: int = 16) -> str:
    low = text.lower()
    if "negative prompt:" in low:
        text = text[:low.index("negative prompt:")]
    parts = re.split(r"[.;\n]+", text)
    s = parts[0] if parts else text
    drop = {"masterpiece","best quality","8k","4k","dslr","hdr","raw","highres","nsfw"}
    tokens = [t for t in re.split(r"[,\s]+", s) if t and t.lower() not in drop]
    return " ".join(tokens[:max_tokens])

def _reservoir_sample_jsonl(path: Path, key: str, n: int, seed: int = 0) -> List[str]:
    random.seed(seed)
    out: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            try: obj = json.loads(line)
            except Exception: continue
            if key not in obj or not obj[key]: continue
            txt = str(obj[key]).strip()
            if not txt: continue
            if len(out) < n: out.append(txt)
            else:
                j = random.randint(1, i)
                if j <= n: out[j - 1] = txt
    return out

def _count_terms(text: str, vocab: Iterable[str]) -> int:
    low = text.lower()
    return sum(1 for term in vocab if term in low)

@dataclass
class Score:
    tokens: int; phrases: int; photo_terms: int; nsfw_terms: int; has_negative: int; refusal: int

REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE)

def score_prompt(text: str, nsfw_vocab: Optional[set[str]] = None) -> Score:
    nsfw_vocab = nsfw_vocab or NSFW_TERMS
    toks = [t for t in re.split(r"[,\s]+", text) if t]
    phrases = len([p for p in text.split(",") if p.strip()])
    low = text.lower()
    photo = _count_terms(low, PHOTOGRAPHY_TERMS)
    nsfw  = _count_terms(low, nsfw_vocab)
    has_neg = int("negative prompt:" in low)
    refusal = int(bool(REFUSAL_RE.search(low)))
    return Score(tokens=len(toks), phrases=phrases, photo_terms=photo, nsfw_terms=nsfw,
                 has_negative=has_neg, refusal=refusal)

def load_thesby(model_id: str):
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype="auto", device_map="auto"
    )
    return model, processor

def gen_thesby_text(model, processor, idea: str, max_new_tokens: int = 256) -> str:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYS}]},
        {"role": "user",   "content": [{"type": "text", "text": idea}]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=None, videos=None, return_tensors="pt")
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.no_grad():
        gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True,
                                 temperature=0.4, top_p=0.85, top_k=40, repetition_penalty=1.3)
        out_ids = gen_ids[0][len(inputs["input_ids"][0]):]
        out = processor.decode(out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return out.strip()

def load_mytho(model_dir: str):
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    mdl = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype="auto", device_map="auto")
    return mdl, tok

def gen_mytho_text(model, tok, idea: str, max_new_tokens: int = 256) -> str:
    prompt = ( "You are a prompt engineer for diffusion/video models. "
               "Given a short idea, expand it into a comma-separated CivitAI/Stable-Diffusion style prompt; "
               "include optional 'Negative prompt:' at the end.\n\n"
               f"Idea: {idea}\nPrompt:" )
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True,
                                 temperature=0.4, top_p=0.85, top_k=40, repetition_penalty=1.3)[0]
    return tok.decode(out_ids, skip_special_tokens=True).strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="thesby/Qwen2.5-VL-7B-NSFW-Caption-V3")
    ap.add_argument("--prompts_file", type=str)
    ap.add_argument("--sample_jsonl", type=str)
    ap.add_argument("--key", default="caption")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--use_condense", action="store_true")
    ap.add_argument("--nsfw_lexicon", type=str)
    ap.add_argument("--compare_mytho", type=str, help="Path to local HF model dir for MythoMax (optional)")
    ap.add_argument("--outdir", type=str, default="/tmp/thesby_vl_probe")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    args = ap.parse_args()

    # Load ideas
    if args.prompts_file:
        ideas = [ln.strip() for ln in Path(args.prompts_file).read_text().splitlines()
                 if ln.strip() and not ln.strip().startswith("#")]
    elif args.sample_jsonl:
        ideas = _reservoir_sample_jsonl(Path(args.sample_jsonl), args.key, args.n, seed=args.seed)
        if args.use_condense:
            ideas = [_condense_to_idea(x) for x in ideas]
    else:
        ideas = [
            "neon-lit alleyway with puddles",
            "topless woman at the pool, looking over shoulder",
            "fit man in locker room, towel around waist",
            "passionate kiss in the rain, two lovers",
            "sensual boudoir shot on silk sheets",
            "close-up of hands exploring a body",
            "artistic nude in chiaroscuro lighting",
        ]
    if args.n and len(ideas) > args.n:
        random.seed(args.seed); random.shuffle(ideas); ideas = ideas[:args.n]

    # Optional domain lexicon
    nsfw_vocab = None
    if args.nsfw_lexicon and Path(args.nsfw_lexicon).exists():
        nsfw_vocab = {ln.strip() for ln in Path(args.nsfw_lexicon).read_text().splitlines() if ln.strip()}
        print(f"[lexicon] loaded {len(nsfw_vocab)} terms from {args.nsfw_lexicon}")

    # Load models
    model, processor = load_thesby(args.model_id)
    mytho = None
    if args.compare_mytho:
        mytho = load_mytho(args.compare_mytho)

    rows: List[Dict[str, Any]] = []
    for i, idea in enumerate(ideas, 1):
        # Thesby
        gen_t = gen_thesby_text(model, processor, idea, max_new_tokens=args.max_new_tokens)
        sc_t = score_prompt(gen_t, nsfw_vocab)
        rows.append({"when": _now(), "idx": i, "which":"thesby", "model": args.model_id,
                     "base_idea": idea, "gen": gen_t, **asdict(sc_t)})
        print(f"[{i:03d}] Thesby  tokens={sc_t.tokens} photo={sc_t.photo_terms} nsfw={sc_t.nsfw_terms} "
              f"neg={sc_t.has_negative} refusal={sc_t.refusal}")

        # Mytho (optional)
        if mytho:
            gen_m = gen_mytho_text(mytho[0], mytho[1], idea, max_new_tokens=args.max_new_tokens)
            sc_m = score_prompt(gen_m, nsfw_vocab)
            rows.append({"when": _now(), "idx": i, "which":"mytho", "model": args.compare_mytho,
                         "base_idea": idea, "gen": gen_m, **asdict(sc_m)})
            print(f"[{i:03d}] Mytho   tokens={sc_m.tokens} photo={sc_m.photo_terms} nsfw={sc_m.nsfw_terms} "
                  f"neg={sc_m.has_negative} refusal={sc_m.refusal}")

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows); df.to_csv(outdir/"probe_rows.csv", index=False)
    (outdir/"probe_rows.json").write_text(json.dumps({"rows": rows}, indent=2, ensure_ascii=False))

    # Aggregate by 'which'
    agg = (df.groupby("which")[["tokens","phrases","photo_terms","nsfw_terms","has_negative","refusal"]]
             .mean().reset_index())
    agg.to_csv(outdir/"probe_agg.csv", index=False)

    print("\n=== Aggregate (means) ===")
    print(agg)

    print(f"\nWrote:\n - {outdir/'probe_rows.csv'}\n - {outdir/'probe_rows.json'}\n - {outdir/'probe_agg.csv'}")

if __name__ == "__main__":
    main()
