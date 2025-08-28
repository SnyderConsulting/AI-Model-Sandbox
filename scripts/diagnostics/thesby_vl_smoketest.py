#!/usr/bin/env python3
# thesby_vl_smoketest.py
# Quick text-only sanity check for thesby/Qwen2.5-VL-7B-NSFW-Caption-V3

import argparse, torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

SYS = ("You are a prompt engineer for diffusion/video models. "
       "Given a short idea, expand it into a CivitAI/Stable‑Diffusion style prompt: "
       "quality tags, subject, anatomy (if relevant), pose/action, setting, lighting, camera/lens, "
       "technical specs. End with an optional 'Negative prompt:' line. Use comma‑separated phrases.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="thesby/Qwen2.5-VL-7B-NSFW-Caption-V3")
    ap.add_argument("--idea", default="neon-lit alleyway with puddles")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    args = ap.parse_args()

    print(f"[load] {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype="auto", device_map="auto"
    )

    # text-only chat
    messages = [{
        "role": "system", "content": [{"type": "text", "text": SYS}]
    },{
        "role": "user", "content": [{"type": "text", "text": args.idea}]
    }]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=None, videos=None, return_tensors="pt")
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.no_grad():
        gen_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=True,
                                 temperature=0.4, top_p=0.85, top_k=40, repetition_penalty=1.3)
        out_ids = gen_ids[0][len(inputs["input_ids"][0]):]
        text_out = processor.decode(out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    print("\n=== OUTPUT ===\n")
    print(text_out.strip())

    # inspect config
    cfg = model.config
    print("\n=== CONFIG SNAPSHOT ===")
    print("arch:", getattr(cfg, "architectures", None))
    print("vocab_size:", cfg.vocab_size, " hidden_size:", cfg.hidden_size,
          " layers:", cfg.num_hidden_layers, " heads:", cfg.num_attention_heads)
    print("rope_theta:", getattr(cfg, "rope_theta", None))

if __name__ == "__main__":
    main()
