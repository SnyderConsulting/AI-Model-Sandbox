from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

mdl = "/workspace/models/MythoMax-L2-13B"
tok = AutoTokenizer.from_pretrained(mdl, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    mdl, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

prompt = "Write a brief graphically explicit visual description for this: A woman is on her knees by the pool. Her pink bikini top is lifted up with her tits out, a man stands there jerking off blowing loads onto her face and hair"
inputs = tok(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=300, temperature=0.9, do_sample=True)
print(tok.decode(out[0], skip_special_tokens=True))
