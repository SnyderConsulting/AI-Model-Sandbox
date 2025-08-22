# AI Model Sandbox

**Purpose:** A focused lab for experimenting with **Wan 2.2** text‑to‑image/video systems—especially the **UMT5‑XXL text encoder** and **DiT LoRA “translator” adapters**—using a fast, pipeline‑parallel training loop.  
**Scope:** Short‑cycle research runs, diagnostics, and evaluations that *improve how Wan listens to NSFW tokens* while keeping image/video fidelity stable.

> We keep the scope intentionally tight: Wan 2.2 (TI2V‑5B and T2V‑A14B) + UMT5‑XXL. Additional models can be added later, but are out‑of‑scope for now.

---

## Why this repo exists

1. **Encoder alignment for NSFW semantics**  
   Enrich UMT5‑XXL’s coverage of NSFW vocabulary while preserving geometric compatibility with Wan’s DiT.

2. **DiT LoRA “translator”**  
   Train small, reversible adapters that **improve how the DiT consumes the updated encoder’s embeddings**—without changing Wan’s drawing style.

3. **Diagnostics-first workflow**  
   Every change (encoder or LoRA) is accompanied by light‑weight probes (key/shape scans, delta budgets, attention‑logit sanity checks).

---

## Status & guardrails (hard‑won lessons)

- **Text Encoder FT works well** when trained in isolation (DiT frozen). Gains in single‑token fidelity carry over to short/verbose prompts.
- **Self‑attention LoRA in DiT is risky.** In multiple tests, enabling `self_attn.*` adapters caused artifacts and destabilization.  
  **Recommendation:** **target only `cross_attn.{k,v}`** for DiT LoRA “translator” runs.
- **Block targeting matters.** Upper third of blocks may be useful, but avoid blanket enabling of every module. Prefer small, auditable coverage.
- **Cache strategy matters.** If you train the text encoder, **do not cache text embeddings**; if you train DiT LoRA, **do cache** them (driven by config below).

---

## Repository overview

```

.
├─ train.py                     # Deepspeed pipeline-parallel training entrypoint
├─ models/
│  ├─ wan/                      # Wan-specific pipeline + UMT5-XXL wrapper
│  │  ├─ wan.py                 # WanPipeline: loading, caching, staging, saving
│  │  ├─ t5.py                  # UMT5-XXL text encoder wrapper
│  │  ├─ model.py               # DiT blocks (Wan 2.x variants)
│  │  └─ ...                    # Attention, VAE wrappers, CLIP (i2v/flf2v)
├─ utils/                       # Dataset, saver, offloading, patches, etc.
├─ configs/                     # Place your TOML configs here
├─ examples/                    # Example TOMLs (use as starting points)
└─ scripts/
└─ diagnostics/              # (Recommended) lightweight weight/key inspectors

````

> If `scripts/diagnostics/` isn’t in your clone yet, create it and drop in your tools (e.g. key/shape inspectors, LoRA delta probes).

---

## Environment

- **OS:** Linux recommended. WSL2 works on Windows. Native Windows + Deepspeed is not supported.
- **Python:** 3.12 recommended.
- **CUDA/PyTorch:** Install a PyTorch/CUDA combo suitable for your GPU. Then install repo requirements.

### Quick setup

```bash
# (conda or venv—your choice)
conda create -n ai-sandbox python=3.12 -y
conda activate ai-sandbox

# 1) Install PyTorch matching your CUDA
# Example (adjust for your system / CUDA):
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

# 2) Install repo deps
pip install -r requirements.txt
````

> If you plan to train with bitsandbytes or special optimizers (e.g., AdamW8bitKahan), ensure compatible wheels are installed for your platform.

---

## Dataset format

* **Images or videos** with a side‑car `.txt` caption file per media item (same basename).
* Mixed images/videos are supported; bucketing handles aspect ratios/frame counts.
* Caching uses Hugging Face Datasets locally (a `cache/` folder is created per dataset root).

Example layout:

```
/data/nsfw/
  0001.jpg
  0001.txt
  0002.mp4
  0002.txt
  ...
```

> Missing captions are allowed (empty caption is used), but discouraged for encoder work.

---

## Caching & training modes

* **Encoder FT (UMT5‑XXL only):**
  `cache_text_embeddings = false` and `train_text_encoder = true`.
  DiT remains frozen.

* **DiT LoRA “translator”:**
  `cache_text_embeddings = true` and `train_text_encoder = false`.
  DiT base frozen; LoRA layers trained.

> Caching happens up‑front. Use `--cache_only` to precompute latents/embeddings and exit.

---

## Minimal config examples

> Copy one of these into `configs/` and adjust paths.

### A) Text‑Encoder fine‑tune (DiT frozen)

```toml
# configs/wan22_te_ft.toml
output_dir = "/workspace/output/wan22_te_ft"
dataset    = "/workspace/datasets/dataset.toml"

epochs                          = 10
micro_batch_size_per_gpu        = 8
gradient_accumulation_steps     = 1
save_every_n_epochs             = 1
steps_per_print                 = 1
mixed_precision                 = "bf16"
pipeline_stages                 = 1
warmup_steps                    = 500

[model]
type                  = "wan"
ckpt_path             = "/workspace/ckpts/Wan2.2-TI2V-5B"  # or A14B top-level dir
dtype                 = "bfloat16"
cache_text_embeddings = false          # << text encoder training
train_text_encoder    = true
text_train_blocks     = 0              # 0 = train all blocks; or train top-N blocks
freeze_transformer    = true           # DiT stays frozen
save_text_encoder_full= true           # write text_encoder.safetensors each save
save_diffusion_model  = false

[optimizer]
type         = "AdamW8bitKahan"
lr           = 8e-6
weight_decay = 0.01
betas        = [0.9, 0.999]

# Optional: mild caption trimming for single-token calibration
[model.caption_aug]
enable           = true
trim_prob        = 0.20
case_insensitive = true
single_token     = true
keywords         = [ "token1","token2","token3" ]  # keep generic in this repo
```

### B) DiT LoRA “translator” (encoder frozen, cache text)

```toml
# configs/wan22_dit_lora_translator.toml
output_dir = "/workspace/output/wan22_dit_lora"
dataset    = "/workspace/datasets/dataset.toml"

epochs                          = 6
micro_batch_size_per_gpu        = 8
image_micro_batch_size_per_gpu  = 32
gradient_accumulation_steps     = 1
save_every_n_epochs             = 1
steps_per_print                 = 1
mixed_precision                 = "bf16"
pipeline_stages                 = 1
warmup_steps                    = 500
disable_block_swap_for_eval     = true
blocks_to_swap                  = 24    # only valid for LoRA; keep ≤ num_layers-2

[model]
type                  = "wan"
ckpt_path             = "/workspace/ckpts/Wan2.2-TI2V-5B"  # or A14B dir
dtype                 = "bfloat16"
cache_text_embeddings = true           # << DiT-only training
train_text_encoder    = false
freeze_transformer    = true
save_text_encoder_full= false
save_diffusion_model  = false          # LoRA saves to adapter file

[adapter]
type    = "lora"
rank    = 16
dropout = 0.10
# Optional: limit training to upper‑blocks only (adjust after inspection)
train_blocks_range = [20, 29]

[optimizer]
type         = "AdamW8bitKahan"
lr           = 2e-5
weight_decay = 0.01
betas        = [0.9, 0.999]

# IMPORTANT QUALITY GUARDRAIL:
# Target only cross-attn.{k,v} modules for LoRA (avoid self_attn.*).
# See docs/recipes/dit_lora_targets.md (or code comments) for how to configure/patch targets.
```

> **Guardrail:** For DiT LoRA, avoid `self_attn.*`. Empirically this caused artifacts and “glitching.” Prefer a surgical target list limited to `cross_attn.k` and `cross_attn.v` (and, if absolutely needed, `cross_attn.q/o` with small ranks). We keep this repo’s examples conservative.

---
## Running

```bash
# Pre-cache (optional; exits after caching)
deepspeed --num_gpus=1 train.py --deepspeed --config configs/wan22_te_ft.toml --cache_only

# Train
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
deepspeed --num_gpus=1 train.py --deepspeed --config configs/wan22_te_ft.toml

# Resume
deepspeed --num_gpus=1 train.py --deepspeed --config configs/wan22_te_ft.toml --resume_from_checkpoint
```

> On RTX 40xx cards, the `NCCL_*` environment variables often avoid transport issues. Adjust for your system.

---

## Outputs

Each run creates a timestamped folder in `output_dir/` with:

* **Checkpoints** (`global_step*`) – full training state (for resuming).
* **Saved artifacts** (`epochN/`) – `text_encoder.safetensors` for encoder runs, `adapter_model.safetensors` (+ PEFT config) for LoRA runs.
* **TensorBoard logs**.

---

## Diagnostics & evaluation (recommended)

Place small utilities in `scripts/diagnostics/` to keep changes auditable:

* **Key/shape inspector:** enumerate `.safetensors` shards, strip `model.diffusion_model.` prefixes, and dump per‑block shapes (helps A14B MoE vs 5B single model).
* **LoRA delta budget:** compute `||ΔW|| / ||W||` per targeted weight; cap large deltas; summarize by type (`cross_attn`, `self_attn`, `ffn`).
* **Attention logit probe:** sample `Q` projections before/after LoRA and compare logit std ratios (sanity: drift should be small).

These scripts are small, header‑only readers and are safe to run on large checkpoints.

---

## Dataset guidance (high‑level)

For encoder FT runs:

* Mix **single‑token “calibration” captions** (e.g., one salient token) with **short multi‑token** and **fully‑verbose** captions.
* A reasonable starting split is **15% single‑token / 35% short / 50% verbose**, adjusted based on downstream probes.
* Keep non‑domain data, but use light caption trimming/keyword retention to prevent over‑reliance on context.
* Always hold out a small **NSFW prompt suite** for frozen‑seed qualitative checks.

For DiT LoRA translator runs:

* Reuse the **updated encoder** to produce cached embeddings.
* Prefer **broad, realistic captions** (the LoRA learns to receive embeddings, not to change style).
* Keep LoRA coverage **small and specific** (cross‑attn k/v; limited blocks).

---

## Known pitfalls

* Enabling LoRA on `self_attn.*` has repeatedly caused image artifacts. Avoid by default.
* If you set `cache_text_embeddings = true` while attempting to train the text encoder, it won’t actually train (the cache bypasses it).
* When switching between 5B and 14B, always re‑inspect keys and shapes before reusing a LoRA target list.

---

## License & attribution

This repo started as a fork‑style sandbox of a pipeline‑parallel diffusion trainer. Credit to original authors for Deepspeed pipeline scaffolding. This sandbox narrows focus to Wan 2.2 + UMT5‑XXL research and adds diagnostics/recipes around that theme.

````
