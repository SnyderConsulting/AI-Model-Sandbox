# AGENTS.md — Working Agreements for Automation & Contributors

This document sets expectations for AI agents and humans collaborating in **AI Model Sandbox**. It encodes **guardrails**, **workflows**, and **definitions of done** so changes remain safe, auditable, and useful.

---

## Mission (optimize for)

* Improve **UMT5-XXL (Wan 2.2)** understanding of NSFW tokens while preserving **DiT compatibility**.
* Build small **DiT LoRA “translator”** adapters that make Wan **listen better** to the updated encoder **without** changing how Wan draws.
* Ship small, testable increments. Every feature/change must include a **diagnostic or probe**.

**Out of scope (for now):** non-Wan models, large refactors unrelated to encoder/DiT alignment, dataset crawlers, web UIs.

---

## Non-negotiable guardrails

1. **Do not enable LoRA on `self_attn.*` by default.**
   Evidence: repeated artifacting/quality collapse. Only touch `self_attn.*` with explicit approval, narrow scope, and low rank.

2. **Keep DiT geometry stable.**
   For DiT LoRA, prefer **`cross_attn.{k,v}` targets only**. Consider minimal `out_proj` (and later specific temporal blocks) **only** with strict delta budgets.

3. **Honor training mode switches:**

   * **Encoder FT** ⇒ `cache_text_embeddings=false`, `train_text_encoder=true`, `freeze_transformer=true`.
   * **DiT LoRA** ⇒ `cache_text_embeddings=true`, `train_text_encoder=false`, `freeze_transformer=true`.

4. **Diagnostics with every change.**
   Add/update a probe under `scripts/diagnostics/` and write a short run report (inputs/outputs, how to read it).

5. **Small diffs, reversible changes.**
   Prefer new modules/flags over invasive rewrites. Document flags in `README.md` or a short `docs/` note.

---

## Environment & install policy (agents MUST follow)

**Why:** Our CI/agents often failed because accelerator builds import `torch` during build-time. Fix = install **PyTorch first**, then optional accelerators **without build isolation**.

### Toolchain matrix (reference)

* **Python** 3.12
* **Torch** 2.8.0, **TorchVision** 0.23.0
* **CUDA** 12.x runtime (GPU nodes)
* **FlashAttention** 2.8.3 (optional)
* **DeepSpeed** 0.17.0 (optional)

### Rules

1. **Install PyTorch first.**
   GPU nodes: `pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.8.0 torchvision==0.23.0`
   CPU nodes: same without the CUDA index-url.

2. **Accelerators are optional.**
   Install with `--no-build-isolation`; do **not** block PRs if they fail.

   * Skip entirely with env **`WAN_SKIP_ACCEL=1`**.
   * If flash-attn import/build fails, set **`WAN_DISABLE_FLASH_ATTN=1`** and continue.

3. **Import guards are required.**
   Any accelerator import must be wrapped so the code runs without it:

   ```python
   USE_FLASH_ATTN = os.getenv("WAN_DISABLE_FLASH_ATTN", "0") != "1"
   try:
       import flash_attn  # noqa: F401
   except Exception:
       USE_FLASH_ATTN = False
   ```

4. **Repo install order (for CI/agents).**

   ```bash
   pip install --upgrade pip setuptools wheel
   # torch first
   pip install torch==2.8.0 torchvision==0.23.0  # + CUDA index-url on GPU nodes
   # optional accelerators
   pip install --no-build-isolation "flash-attn==2.8.3" || export WAN_DISABLE_FLASH_ATTN=1
   DS_BUILD_OPS=0 pip install --no-build-isolation "deepspeed==0.17.0" || true
   # project deps last
   pip install -r requirements.txt -r inference/Wan2.2/requirements.txt
   ```

5. **Definition of done (env).**

   * `python -c "import torch; print(torch.__version__)"` prints a version.
   * Absence of `flash_attn` does **not** break tests or training; fallback path works.

### Recommended bootstrap scripts (copy into your platform)

**Setup script (after clone):**

```bash
#!/usr/bin/env bash
set -euxo pipefail
export PIP_PREFER_BINARY=1
pip install --upgrade pip setuptools wheel packaging
if command -v nvidia-smi >/dev/null 2>&1; then
  pip install --index-url "https://download.pytorch.org/whl/cu124" torch==2.8.0 torchvision==0.23.0
else
  pip install torch==2.8.0 torchvision==0.23.0
fi
if [ "${WAN_SKIP_ACCEL:-0}" != "1" ]; then
  pip install --no-build-isolation -U "flash-attn==2.8.3" || export WAN_DISABLE_FLASH_ATTN=1
  DS_BUILD_OPS=0 pip install --no-build-isolation "deepspeed==0.17.0" || true
fi
pip install -r requirements.txt || true
pip install -r inference/Wan2.2/requirements.txt || true
pip install -r requirements.txt -r inference/Wan2.2/requirements.txt
```

**Maintenance script (after resume):**

```bash
#!/usr/bin/env bash
set -euxo pipefail
export PIP_PREFER_BINARY=1
python -c "import torch; print(torch.__version__)" || {
  if command -v nvidia-smi >/dev/null 2>&1; then
    pip install --index-url "https://download.pytorch.org/whl/cu124" torch==2.8.0 torchvision==0.23.0
  else
    pip install torch==2.8.0 torchvision==0.23.0
  fi
}
if [ "${WAN_SKIP_ACCEL:-0}" != "1" ]; then
  pip install --no-build-isolation -U "flash-attn==2.8.3" || export WAN_DISABLE_FLASH_ATTN=1
  DS_BUILD_OPS=0 pip install --no-build-isolation "deepspeed==0.17.0" || true
fi
pip install -r requirements.txt -r inference/Wan2.2/requirements.txt
```

---

## Repository conventions

* **Python** 3.12, prefer type hints.
* **Style:** `black` + `ruff` defaults.
* **Deepspeed** pipeline parallel is supported; keep code runnable on single-GPU too.
* **Configs** live in `configs/` (minimal, commented).
* **Diagnostics** in `scripts/diagnostics/`.
* **Training artifacts** under each run’s `output_dir/…`; **do not** place artifacts in `reports/`.

---

## Reports directory policy (REQUIRED)

All **diagnostic / inspection / research** scripts must write outputs to **`reports/`**.

* Subdirectory per tool/experiment: `reports/<tool_or_experiment>/<run_tag>/`
  Example: `reports/te_eval/2025-08-22T10-31Z/` or `reports/dit_lora_translator/run_0012/`
* Include:

  * `report.md` — short summary (what/why/how to read it)
  * A metrics file `summary.json` or `summary.csv` (the numbers referenced in the summary)
  * Pointers to inputs (paths to weights, configs, seeds)
* Large binaries small & compressed; use Git LFS if committed.

---

## Typical workflows

### A) Text-encoder FT experiment

1. Add `configs/wan22_te_ft.toml`.
2. Set `cache_text_embeddings=false`, `train_text_encoder=true`, `freeze_transformer=true`.
3. Smoke ≤1 epoch on a tiny dataset; verify caching/training paths.
4. Diagnostics in `scripts/diagnostics/`:

   * Per-prompt embedding cosine vs base
   * Injected context signal / nearest-neighbor drift
5. Write a report to `reports/te_eval/<run_tag>/` with `report.md` + metrics.
   **DoD:** Loss stable; single-token probes improve; report exists.

### B) DiT LoRA “translator” run (Phase-1 default)

1. Add `configs/wan22_dit_lora_translator.toml`.
2. Set `cache_text_embeddings=true`, `train_text_encoder=false`, `freeze_transformer=true`.
3. **Targets:** restrict to `cross_attn.{k,v}`. Implement include-filter if needed; **do not** touch `self_attn.*`.
4. Optional: limit `train_blocks_range` to an upper-block band.
5. Diagnostics:

   * LoRA delta budget (`||ΔW||/||W||`) summary
   * Attention logit std ratios (base vs adapted) on a few blocks
6. Write a report to `reports/dit_lora_translator/<run_tag>/` with `report.md` + metrics.
7. Smoke ≤2 epochs; inspect artifacts before longer runs.
   **DoD:** No obvious artifacts on frozen-seed grids; delta budgets within caps; report exists.

### C) Key/shape inspection (5B vs A14B)

1. Add/run an inspector under `scripts/diagnostics/` dumping `keys.csv` + shapes.
2. Report to `reports/weights_inspect/<run_tag>/` with notes and CSVs.
3. Build LoRA target list from the dump.
   **DoD:** Shape parity confirmed; target list reflects actual keys; report exists.

---

## Phase-1 KV-LoRA trainer (what agents should run)

**Goal:** Distill the **text interface** so DiT reads bridged tokens. This trainer matches **K/V (optionally after `norm_k`)** across all cross-attn blocks. It does **not** require VAE/noise latents.

**Inputs:**

* Wan 2.2-5B **single model** (sharded): `config.json` + `diffusion_pytorch_model-0000x-of-00003.safetensors`
* Teacher encoder: `models_t5_umt5-xxl-enc-bf16.pth` + local tokenizer dir `google/umt5-xxl/`
* Bridge encoder: affine-calibrated bridge checkpoint + LLM dir
* Prompts: use the large JSONL (`/workspace/AI-Model-Sandbox/datasets/captions.jsonl`, field `caption`)

**Run example:**

```bash
WAN5B_DIR=/workspace/models/Wan2.2-TI2V-5B
WAN_CFG=$WAN5B_DIR/config.json
T5_PTH=$WAN5B_DIR/models_t5_umt5-xxl-enc-bf16.pth
T5_TOK_DIR=$WAN5B_DIR/google/umt5-xxl
BRIDGE_PTH=/workspace/AI-Model-Sandbox/outputs/encoder-bridg/bridge_20000_affine.pth
LLM_DIR=/workspace/models/MythoMax-L2-13B
PROMPTS=/workspace/AI-Model-Sandbox/datasets/captions.jsonl
OUT=/workspace/runs/wan5b_kv_lora_stage1

python train_kv_lora_distill.py \
  --transformer_config $WAN_CFG \
  --transformer_weights_dir $WAN5B_DIR \
  --t5_checkpoint $T5_PTH \
  --t5_tokenizer_dir $T5_TOK_DIR \
  --prompts_file $PROMPTS \
  --out_dir $OUT \
  --bridge_ckpt $BRIDGE_PTH \
  --llm_dir $LLM_DIR \
  --global_scale 1.8 \
  --rank 8 --alpha 32 --lr 1e-4 \
  --epochs 1 --batch_size 6 --grad_accum 4 \
  --use_normed_targets --bf16 \
  --adapter_prefix diffusion_model.
```

**Expected:** log of `injected N modules` where `N = (#blocks * 2)`; adapter saved at `$OUT/adapter_model.safetensors`.

**Distribution:** Ship a **single** LoRA file. If the inference loader expects a different key prefix, export with `--adapter_prefix transformer.` accordingly.

---

## Safety & NSFW handling

* Research focus is NSFW encoder alignment, but **examples/docs must remain neutral**.
* Avoid explicit sexual content in comments, configs, or tests.
* Token lists can be referenced indirectly (e.g., “domain keywords”) or read from private files; do not hard-code explicit terms in repo.

---

## Change review checklist (for agents & humans)

* [ ] Guardrails respected (no default `self_attn.*` LoRA).
* [ ] Training toggles correct for the mode (encoder FT vs DiT LoRA).
* [ ] New/updated diagnostic + **report written to `reports/<tool>/<run_tag>/`** with `report.md` + metrics.
* [ ] Training artifacts under `output_dir/...` (not mixed into `reports/`).
* [ ] Configs minimal and commented.
* [ ] Diff is small and reversible (feature-flagged if needed).
* [ ] Environment follows **install policy**; accelerator imports are guarded.

---

## Known pitfalls / tips

* If encoder FT “does nothing,” you likely left `cache_text_embeddings=true`.
* When switching between 5B and 14B, re-derive the LoRA target list; do not reuse blindly.
* If artifacts appear after a LoRA change, first disable all `self_attn.*` adapters and retest.
* Keep reports lightweight and textual; compress image grids; use Git LFS for large binaries.

---

## Roadmap (near-term)

* A14B-specific LoRA targeting recipe (per-expert bands).
* First-class include/exclude support for LoRA targets via config.
* Automated post-epoch probe runner that writes directly to `reports/…`.

---

## Contacts

Open an issue with a short description, your config, and the smallest `reports/…` artifact that shows the behavior (`report.md` + metrics). Keep reproduction runs short when possible.
