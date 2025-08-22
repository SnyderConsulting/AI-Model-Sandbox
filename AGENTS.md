# AGENTS.md — Working Agreements for Automation & Contributors

This document sets expectations for AI agents and humans collaborating in **AI Model Sandbox**. It encodes **guardrails**, **workflows**, and **definitions of done** so changes remain safe, auditable, and useful.

---

## Mission (what to optimize for)

- Improve **UMT5‑XXL (Wan 2.2)** understanding of NSFW tokens while preserving **DiT compatibility**.
- Build small **DiT LoRA “translator”** adapters that make Wan **listen better** to the updated encoder, **without** changing how Wan draws.
- Ship small, testable increments. Every feature/change should come with a **diagnostic or probe**.

Out of scope (for now): non‑Wan models, large refactors unrelated to encoder/DiT alignment, dataset crawlers, web UIs.

---

## Non‑negotiable guardrails

1. **Do not enable LoRA on `self_attn.*` by default.**  
   Evidence: repeated artifacting and quality collapse. Only touch `self_attn` with explicit approval, narrow scope, and low rank.

2. **Keep DiT geometry stable.**  
   For DiT LoRA, prefer **`cross_attn.{k,v}` targets only**; consider minimal `q/o` only with strict delta budgets.

3. **Honor training mode switches:**
   - Encoder FT ⇒ `cache_text_embeddings=false`, `train_text_encoder=true`, `freeze_transformer=true`.
   - DiT LoRA ⇒ `cache_text_embeddings=true`, `train_text_encoder=false`, `freeze_transformer=true`.

4. **Diagnostics with every change.**  
   Add/update a probe under `scripts/diagnostics/` and include a short markdown note (inputs/outputs, how to read it).

5. **Small diffs, reversible changes.**  
   Prefer new modules or flags over invasive rewrites. Document flags in the README or a short `docs/` note.

---

## Repository conventions

- **Python** 3.12, prefer type hints.  
- **Style:** `black` + `ruff` defaults.  
- **Deepspeed** pipeline parallel is the backbone; keep surfaces compatible with single‑GPU runs too.  
- **Configs** live in `configs/`. Keep examples minimal and commented.  
- **Diagnostics** live in `scripts/diagnostics/`.

---

## Typical workflows
### A) Add a text‑encoder FT experiment
1. Create `configs/wan22_te_ft.toml` (see README example).
2. Ensure `cache_text_embeddings=false` and `train_text_encoder=true`.
3. Run a short smoke (≤ 1 epoch) on a tiny dataset to verify caching/training paths.
4. Add/refresh diagnostics:
   - Per‑prompt embedding cosine vs base (`scripts/diagnostics/te_eval.py`).
   - “Injected context signal” or nearest‑neighbor drift.
5. Commit: config + diagnostics + brief results summary.

**Definition of Done**
- Loss is stable; TE produces plausible gains on single‑token probes.
- Summary artifact (JSON/CSV) written to `output/…/epochN_*` and referenced from commit message.

---

### B) Add a DiT LoRA “translator” run
1. Create `configs/wan22_dit_lora_translator.toml` (see README example).
2. Set `cache_text_embeddings=true`, `train_text_encoder=false`, `freeze_transformer=true`.
3. **Target list:** restrict to `cross_attn.{k,v}`. If code lacks targeting controls, implement a minimal include‑filter near LoRA init, gated by config.
4. Optional: limit `train_blocks_range` to an upper‑block band.
5. Add/refresh diagnostics:
   - LoRA delta budget (`||ΔW||/||W||`) summary JSON.
   - Attention logit std ratios (base vs adapted) on a few blocks.
6. Run a short smoke (≤ 2 epochs) and inspect for artifacts before longer runs.

**Definition of Done**
- No obvious artifacts on frozen‑seed prompt grid.
- Delta budgets remain within configured caps; attention probes near 1.0× std.

---

### C) Key/shape inspection (5B vs A14B)
1. Add or run a header‑only inspector under `scripts/diagnostics/` to dump `__keys.csv` and per‑block summaries.
2. Use the dump to build a LoRA target list (regex or explicit keys).
3. Store the dump next to run outputs for traceability.

**Definition of Done**
- Shape parity confirmed; MoE split identified (`high_noise`, `low_noise`, etc.).
- Target list reflects actual keys present in the checkpoint(s).

---

## Safety & NSFW handling

- The research focus is NSFW encoder alignment, but **examples and docs must remain neutral**.  
- Avoid explicit sexual content or images in code comments, configs, or tests.  
- Token lists may be referenced indirectly (e.g., “domain keywords”) or read from external private files, not hard‑coded in the repo.

---

## Change review checklist (for agents & humans)

- [ ] Does the change respect the guardrails (no default `self_attn` LoRA)?
- [ ] Are the relevant training toggles (`cache_text_embeddings`, `train_text_encoder`, `freeze_transformer`) correct?
- [ ] Is there a new/updated diagnostic proving the change did what it claims?
- [ ] Are outputs written to `output_dir/…/epochN/` with machine‑readable summaries (JSON/CSV)?
- [ ] Are configs minimal and commented?
- [ ] Is the diff small and reversible (feature‑flagged if needed)?

---

## Known pitfalls / tips

- If encoder training “does nothing,” you likely left `cache_text_embeddings=true`.
- When moving between 5B and 14B, re‑derive the LoRA target list—don’t reuse blindly.
- If artifacts appear after a LoRA update, first disable all `self_attn.*` adapters and retest.

---

## Roadmap (near‑term)

- A14B‑specific LoRA targeting recipe (per‑expert bands).
- First‑class include/exclude support for LoRA targets via config.
- Automated post‑epoch probe runner (writes summaries under `epochN/`).

---

## Contact

Open an issue with a short description, your config, and the smallest diagnostic artifact that shows the behavior (JSON/CSV). Keep reproduction runs short when possible.
