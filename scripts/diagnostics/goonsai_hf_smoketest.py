#!/usr/bin/env python3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ATTN_IMPL = "flash_attention_2"
try:
    import flash_attn  # noqa: F401
except Exception:
    ATTN_IMPL = "eager"

MODEL_ID = "goonsai-com/civitaiprompts/qwen2.5-3B-goonsai-nsfw-100k"

# Load


tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation=ATTN_IMPL,
)

# Use the repo's chat template
messages = [
    {
        "role": "system",
        "content": (
            "You are a prompt engineer for diffusion/video models. "
            "Given a short idea, expand it into a CivitAI/Stable‑Diffusion style prompt: "
            "quality tags, subject, anatomy (if relevant), pose/action, setting, lighting, camera/lens, "
            "technical specs. End with an optional 'Negative prompt:' line. Use comma‑separated phrases."
        ),
    },
    {"role": "user", "content": "neon‑lit alleyway with puddles"},
]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tok(text, return_tensors="pt").to(model.device)

# Generation params: start with their generation_config and override a few
out = model.generate(
    **inputs,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.4,
    top_p=0.85,
    top_k=40,
    repetition_penalty=1.3,
)
print(tok.decode(out[0], skip_special_tokens=True))

# Hidden states for bridging / diagnostics
with torch.no_grad():
    hs = model(**inputs, output_hidden_states=True, use_cache=False).hidden_states
print("n_layers:", len(hs), " last shape:", hs[-1].shape)
