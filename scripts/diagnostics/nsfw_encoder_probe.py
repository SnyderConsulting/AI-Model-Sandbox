#!/usr/bin/env python3
# nsfw_encoder_probe.py  (updated: Goonsai subfolder load)
# Compare encoders on NSFW semantics: MythoMax-L2-13B, Goonsai Qwen2.5-3B NSFW-100k, Thesby Qwen2.5-VL-7B NSFW (text-only).

from __future__ import annotations
import argparse, json, math, time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any

import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except Exception:
    Qwen2_5_VLForConditionalGeneration = None

# ------------------------ probe sets ------------------------

PROBE = {
  "ambiguous": [
    {"base": "cock",   "sexual": ["penis", "erect penis", "male genitalia"], "sfw": ["rooster", "chicken"]},
    {"base": "pussy",  "sexual": ["vagina", "vulva", "female genitalia"],    "sfw": ["cat", "kitten"]},
    {"base": "cowgirl","sexual": ["woman riding a man, sex position", "cowgirl sex position"],
                       "sfw": ["woman riding a horse", "horse riding"]},
    {"base": "doggy",  "sexual": ["doggy style sex position from behind", "rear-entry sex position"],
                       "sfw": ["dog", "puppy"]},
    {"base": "cum",    "sexual": ["semen ejaculation", "male ejaculation"],
                       "sfw": ["come here", "arrival"]},
    {"base": "beaver", "sexual": ["slang vagina with pubic hair"],            "sfw": ["beaver animal"]},
  ],
  "implications": [
    {"phrase": "a woman fingering herself",
     "positives": ["vagina visible", "female masturbation", "clitoris"],
     "negatives": ["fully clothed", "no nudity"]},
    {"phrase": "a woman giving a blowjob",
     "positives": ["penis in mouth", "oral sex"],
     "negatives": ["eating food", "speaking into microphone"]},
    {"phrase": "a couple having sex",
     "positives": ["intercourse", "penetration penis vagina"],
     "negatives": ["hugging", "kissing only"]},
    {"phrase": "anal sex",
     "positives": ["penis in anus", "rear-entry sex"],
     "negatives": ["medical exam", "back massage"]},
    {"phrase": "a man ejaculating",
     "positives": ["semen", "cum on body"],
     "negatives": ["sweat", "water splash"]},
  ],
  "composition": [
    {"phrase": "woman riding a man", "parts": ["woman", "riding", "man"], "foil": "woman riding a horse"},
    {"phrase": "man performing oral sex on a woman", "parts": ["man", "oral sex", "woman"], "foil": "man eating a hot dog"},
    {"phrase": "woman fingering herself on a bed", "parts": ["woman", "fingering", "self", "on a bed"], "foil": "woman pointing at herself on a bed"},
    {"phrase": "two women kissing naked in the shower", "parts": ["two women", "kissing", "naked", "in the shower"], "foil": "two women talking in the shower"},
  ],
  "style_pairs": [
    {"prose": "a nude woman fondling her breasts under soft window light",
     "tags":  "woman, nude, fondling breasts, soft lighting, window light, close-up, boudoir"},
    {"prose": "a couple having sex in the missionary position",
     "tags":  "missionary, couple, sex, man on top, penetration"},
    {"prose": "a woman riding a man in cowgirl position",
     "tags":  "cowgirl sex position, woman riding man"},
    {"prose": "a man ejaculating semen on a woman's stomach",
     "tags":  "male ejaculation, semen, cum on stomach"},
  ]
}

# ------------------------ embedding helpers ------------------------

@dataclass
class ModelSpec:
    name: str
    kind: str           # "causal" or "vl"
    tok: Any
    mdl: Any
    device: torch.device
    hidden_size: int

def _pad_token_fix(tok):
    if getattr(tok, "pad_token_id", None) is None and getattr(tok, "eos_token", None) is not None:
        tok.pad_token = tok.eos_token

def mean_pool(last_hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    mask = attn_mask.float().unsqueeze(-1)
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom

def layer_mix(hs: tuple, top_k: int) -> torch.Tensor:
    k = min(top_k, len(hs))
    return torch.stack(hs[-k:], dim=0).mean(dim=0)

def embed_batch_causal(spec: ModelSpec, texts: List[str], max_length: int, mix: str, pool: str) -> torch.Tensor:
    tok, mdl = spec.tok, spec.mdl
    _pad_token_fix(tok)
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    enc = {k: v.to(spec.device) for k, v in enc.items()}
    with torch.no_grad():
        out = mdl(**enc, output_hidden_states=True, use_cache=False)
    hs = layer_mix(out.hidden_states, 4) if mix=="last4" else layer_mix(out.hidden_states, 2) if mix=="last2" else out.hidden_states[-1]
    if pool=="lasttok":
        lengths = enc["attention_mask"].sum(dim=1) - 1
        vecs = hs[torch.arange(hs.size(0), device=hs.device), lengths, :]
    else:
        vecs = mean_pool(hs, enc["attention_mask"])
    return F.normalize(vecs, p=2, dim=1)

def embed_batch_vl(spec: ModelSpec, texts: List[str], max_length: int, mix: str, pool: str) -> torch.Tensor:
    proc, mdl = spec.tok, spec.mdl
    inputs = proc(text=texts, images=None, videos=None, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    inputs = {k:(v.to(spec.device) if hasattr(v,"to") else v) for k,v in inputs.items()}
    with torch.no_grad():
        out = mdl(**inputs, output_hidden_states=True, use_cache=False)
    hs = layer_mix(out.hidden_states, 4) if mix=="last4" else layer_mix(out.hidden_states, 2) if mix=="last2" else out.hidden_states[-1]
    attn = inputs["attention_mask"]
    if pool=="lasttok":
        lengths = attn.sum(dim=1) - 1
        vecs = hs[torch.arange(hs.size(0), device=hs.device), lengths, :]
    else:
        vecs = mean_pool(hs, attn)
    return F.normalize(vecs, p=2, dim=1)

def embed_texts(spec: ModelSpec, texts: List[str], batch: int, max_length: int, mix: str, pool: str) -> torch.Tensor:
    out = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i+batch]
        if spec.kind=="causal":
            vec = embed_batch_causal(spec, chunk, max_length, mix, pool)
        else:
            vec = embed_batch_vl(spec, chunk, max_length, mix, pool)
        out.append(vec)
    return torch.cat(out, dim=0)

# ------------------------ loaders (with subfolder support) ------------------------

def _maybe_kw_subfolder(subfolder: str | None) -> dict:
    return {"subfolder": subfolder} if subfolder else {}

def load_causal(model_id: str, subfolder: str | None, load_8bit: bool, load_4bit: bool) -> ModelSpec:
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True,
                                        **({"subfolder": subfolder} if subfolder else {}))
    kwargs = dict(device_map="auto", low_cpu_mem_usage=True,
                  **({"subfolder": subfolder} if subfolder else {}))

    if load_4bit:
        kwargs.update(dict(load_in_4bit=True))
    elif load_8bit:
        kwargs.update(dict(load_in_8bit=True))
    else:
        # Let Transformers choose .bin or .safetensors based on what's present.
        kwargs.update(dict(torch_dtype=torch.bfloat16))

    mdl = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    device = next(mdl.parameters()).device
    hidden = int(getattr(mdl.config, "hidden_size", 0) or getattr(mdl.config, "n_embd", 0))
    return ModelSpec(name=(model_id if not subfolder else f"{model_id}/{subfolder}"),
                     kind="causal", tok=tok, mdl=mdl, device=device, hidden_size=hidden)

def load_vl(model_id: str, load_8bit: bool, load_4bit: bool) -> ModelSpec:
    if Qwen2_5_VLForConditionalGeneration is None:
        raise RuntimeError("transformers missing Qwen2_5_VLForConditionalGeneration; upgrade transformers.")
    proc = AutoProcessor.from_pretrained(model_id)
    kwargs = dict(device_map="auto")
    if load_4bit: kwargs.update(dict(load_in_4bit=True))
    elif load_8bit: kwargs.update(dict(load_in_8bit=True))
    else: kwargs.update(dict(torch_dtype=torch.bfloat16))
    mdl = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
    device = next(mdl.parameters()).device
    hidden = int(getattr(mdl.config, "hidden_size", 0))
    return ModelSpec(name=model_id, kind="vl", tok=proc, mdl=mdl, device=device, hidden_size=hidden)

# ------------------------ scoring ------------------------

def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (F.normalize(a, dim=-1) * F.normalize(b, dim=-1)).sum(dim=-1)

def score_ambiguous(batcher) -> List[Dict[str, Any]]:
    rows=[]
    for it in PROBE["ambiguous"]:
        texts = [it["base"]] + it["sexual"] + it["sfw"]
        E = batcher(texts)
        base = E[0:1]; e_sex = E[1:1+len(it["sexual"])].mean(dim=0, keepdim=True); e_sfw = E[1+len(it["sexual"]):].mean(dim=0, keepdim=True)
        rows.append({"probe":"ambiguous","base":it["base"],
                     "sexual_vs_sfw_margin": float(cosine(base, e_sex) - cosine(base, e_sfw))})
    return rows

def score_implications(batcher) -> List[Dict[str, Any]]:
    rows=[]
    for it in PROBE["implications"]:
        texts = [it["phrase"]] + it["positives"] + it["negatives"]
        E = batcher(texts)
        p = E[0:1]; pos = E[1:1+len(it["positives"])].mean(dim=0, keepdim=True); neg = E[1+len(it["positives"]):].mean(dim=0, keepdim=True)
        rows.append({"probe":"implication","phrase":it["phrase"],
                     "implication_margin": float(cosine(p, pos) - cosine(p, neg))})
    return rows

def score_composition(batcher) -> List[Dict[str, Any]]:
    rows=[]
    for it in PROBE["composition"]:
        texts = [it["phrase"], it["foil"]] + it["parts"]
        E = batcher(texts)
        phr = E[0:1]; foil = E[1:2]; parts = E[2:].sum(dim=0, keepdim=True)
        rows.append({"probe":"composition","phrase":it["phrase"],"foil":it["foil"],
                     "additivity_margin": float(cosine(phr, parts) - cosine(foil, parts))})
    return rows

def score_style(batcher) -> List[Dict[str, Any]]:
    rows=[]
    for it in PROBE["style_pairs"]:
        E = batcher([it["prose"], it["tags"]])
        rows.append({"probe":"style","prose":it["prose"],"tags":it["tags"],
                     "prose_vs_tags_cos": float(cosine(E[0:1], E[1:2]))})
    return rows

# ------------------------ main ------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mytho",  default="/workspace/models/MythoMax-L2-13B", help="Path or HF id")
    ap.add_argument("--goonsai", default="goonsai-com/civitaiprompts", help="HF repo id (parent)")
    ap.add_argument("--goonsai_subfolder", default="qwen2.5-3B-goonsai-nsfw-100k", help="HF subfolder inside the repo")
    ap.add_argument("--thesby", default="thesby/Qwen2.5-VL-7B-NSFW-Caption-V3")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=96)
    ap.add_argument("--mix", choices=["last","last2","last4"], default="last4")
    ap.add_argument("--pool", choices=["mean","lasttok"], default="mean")
    ap.add_argument("--include_style", action="store_true")
    ap.add_argument("--load_8bit", action="store_true")
    ap.add_argument("--load_4bit", action="store_true")
    ap.add_argument("--outdir", default="/tmp/nsfw_encoder_probe")
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    specs=[]
    print(f"[{time.strftime('%H:%M:%S')}] Loading MythoMax from {args.mytho}")
    specs.append(load_causal(args.mytho, None, args.load_8bit, args.load_4bit))

    print(f"[{time.strftime('%H:%M:%S')}] Loading Goonsai from {args.goonsai}/{args.goonsai_subfolder}")
    specs.append(load_causal(args.goonsai, args.goonsai_subfolder, args.load_8bit, args.load_4bit))

    print(f"[{time.strftime('%H:%M:%S')}] Loading Thesby from {args.thesby}")
    specs.append(load_vl(args.thesby, args.load_8bit, args.load_4bit))

    scoreboard_rows=[]; detail_rows=[]

    for spec in specs:
        print(f"\n=== {spec.name} | kind={spec.kind} | hidden={spec.hidden_size} | device={spec.device} ===")
        def batcher(texts: List[str]) -> torch.Tensor:
            return embed_texts(spec, texts, args.batch, args.max_length, args.mix, args.pool)

        amb = score_ambiguous(batcher)
        imp = score_implications(batcher)
        cmp = score_composition(batcher)
        sty = score_style(batcher) if args.include_style else []

        detail_rows.extend([{"model": spec.name, **r} for r in (amb+imp+cmp+sty)])

        import math
        amb_mean = float(pd.DataFrame(amb)["sexual_vs_sfw_margin"].mean())
        imp_mean = float(pd.DataFrame(imp)["implication_margin"].mean())
        cmp_mean = float(pd.DataFrame(cmp)["additivity_margin"].mean())
        sty_mean = float(pd.DataFrame(sty)["prose_vs_tags_cos"].mean()) if sty else float("nan")

        scoreboard_rows.append({
            "model": spec.name,
            "sexual_bias_mean": round(amb_mean, 4),
            "implication_mean": round(imp_mean, 4),
            "composition_mean": round(cmp_mean, 4),
            "style_invariance_mean": (round(sty_mean, 4) if not math.isnan(sty_mean) else "")
        })

        print(f"  sexual_bias_mean: {amb_mean:.3f}")
        print(f"  implication_mean: {imp_mean:.3f}")
        print(f"  composition_mean: {cmp_mean:.3f}")
        if not math.isnan(sty_mean):
            print(f"  style_invariance_mean: {sty_mean:.3f}")

    score_df = pd.DataFrame(scoreboard_rows).sort_values(by=["sexual_bias_mean","implication_mean","composition_mean"], ascending=False)
    score_df.to_csv(Path(args.outdir)/"scoreboard.csv", index=False)

    det_csv = Path(args.outdir)/"detail_rows.csv"
    det_json = Path(args.outdir)/"detail_rows.json"
    pd.DataFrame(detail_rows).to_csv(det_csv, index=False)
    Path(det_json).write_text(json.dumps({"rows": detail_rows}, indent=2, ensure_ascii=False))

    print("\n=== SCOREBOARD (higher is better) ===")
    print(score_df.to_string(index=False))
    print(f"\nWrote:\n - {Path(args.outdir)/'scoreboard.csv'}\n - {det_csv}\n - {det_json}")

if __name__ == "__main__":
    main()
