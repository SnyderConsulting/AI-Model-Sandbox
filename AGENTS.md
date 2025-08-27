# AGENTS.md — Working Agreements for Automation & Contributors

This document defines how humans and AI agents collaborate in **AI Model Sandbox**. It sets **guardrails**, **workflow rules**, and **definitions of done** so changes remain safe, auditable, and useful across all workstreams.

---

## Mission (what to optimize for)

- Improve model **text–vision alignment** (prompt controllability, stability, faithfulness).
- Keep the **diffusion backbone stable** while iterating on adapters/encoders.
- Ship **small, testable increments**: every change comes with a diagnostic/probe and a short report.

**Out of scope (for now):** non-repo models, large refactors unrelated to encoder/DiT alignment, dataset crawlers, UI work.

---

## Non‑negotiable guardrails

1. **Minimize surface area of trainable changes.** Prefer small adapters; avoid invasive rewrites.
2. **Keep the diffusion transformer geometry stable.**  
   - Default for adapter work: target **cross‑attention** pathways.  
   - Do **not** enable `self_attn.*` adapters by default; expand scope only with explicit approval + tight delta budgets.
3. **Training‑mode toggles must be explicit and correct.**  
   - If training a text encoder: `cache_text_embeddings=false`, `train_text_encoder=true`, `freeze_transformer=true`.  
   - If training a DiT adapter: `cache_text_embeddings=true`, `train_text_encoder=false`, `freeze_transformer=true`.
4. **Feature‑flags over rewrites.** Add a config/flag/env var instead of changing behavior globally.
5. **Diagnostics with every change.** Add/refresh a probe and write a short report (what changed, how to read it).
6. **Small diffs, reversible changes.** Land incremental PRs; document flags in README or a short `docs/` note.

---

## Environment & dependency policy

**Why:** extension packages often import `torch` during build; to prevent flaky setups, follow this order.

1. **Install PyTorch first**, then the rest.  
   - GPU nodes: use the appropriate CUDA wheel index; CPU nodes: regular wheels.
2. **Accelerator libs are optional.**  
   - Never make core flows depend on FlashAttention/DeepSpeed.  
   - Always guard imports: if unavailable, fall back without failing.
3. **Import guards are required** for optional deps:
   ```python
   USE_ACCEL = os.getenv("DISABLE_ACCEL", "0") != "1"
   try:
       import flash_attn  # noqa: F401
   except Exception:
       USE_ACCEL = False
````

4. **Pip order (for CI/agents):**

   * `pip install --upgrade pip setuptools wheel`
   * install **torch/torchvision**
   * (optional) accelerators **without** build isolation; do not fail the build if they’re missing
   * project requirements last
5. **Definition of done (env):**

   * `python -c "import torch"` succeeds
   * tests/diagnostics run without optional accelerators

---

## Repository conventions

* **Language:** Python 3.12, type hints preferred.
* **Style:** `black` and `ruff` defaults.
* **Configs:** live in `configs/` (minimal, commented).
* **Diagnostics/scripts:** live in `scripts/diagnostics/`.
* **Training artifacts:** always under an `output_dir/...` for the run; **never** inside `reports/`.
* **Env vars:** prefix with `WAN_` for repo‑wide switches.

---

## Reports directory policy (REQUIRED)

All **diagnostic / inspection / research** outputs must go under **`reports/`**.

* Structure: `reports/<tool_or_experiment>/<run_tag>/`
* Minimum contents:

  * `report.md` — short narrative (what changed, how to read it)
  * `summary.json` or `summary.csv` — the numbers referenced in the report
  * Pointers to inputs (paths to weights, configs, seeds)
* Keep binaries small; if committed, use Git LFS.

---

## Workflow templates

### A) Experiment that trains/changes text embeddings

1. Create a minimal config in `configs/`.
2. Verify toggles (see Guardrail #3).
3. Run a short smoke; add/update diagnostic(s).
4. Write `reports/<tool>/<run_tag>/report.md` + metrics file; link inputs.
   **Definition of Done:** loss stable, probe improvements visible, report exists.

### B) Experiment that adapts the diffusion model with small adapters

1. Minimal config with flags + target list; default to cross‑attention pathways.
2. Keep adapter surface small; document any expansions.
3. Add diagnostics: delta budgets, attention statistics, etc.
4. Write `reports/<tool>/<run_tag>/report.md` + metrics.
   **Definition of Done:** no obvious artifacts in fixed‑seed grids; delta budgets within caps; report exists.

### C) Inspection / introspection task

1. Write head‑only inspectors (no heavy weights) under `scripts/diagnostics/`.
2. Dump keys/shapes/summaries and write a short report in `reports/...`.
   **Definition of Done:** shapes verified; target list derived; report exists.

---

## Review checklist (for agents & humans)

* [ ] Guardrails respected (adapter scope; stable geometry).
* [ ] Training toggles correct for the task type.
* [ ] New/updated diagnostic present and saved to `reports/...`.
* [ ] Training outputs in `output_dir/...`, not mixed into `reports/`.
* [ ] Configs minimal and commented; behavior behind flags.
* [ ] Style checks pass (`black`, `ruff`); unit tests/quick probes pass.
* [ ] Optional deps are import‑guarded; core path runs without them.

---

## Safety & content policy

* Focus is alignment and controllability; **keep examples/docs neutral**.
* Avoid explicit sexual content in code/comments/tests.
* If domain‑specific token sets are required, read from external/private files rather than hard‑coding explicit terms.

---

## Data & secrets

* Never commit secrets or private datasets.
* Reference datasets/configs by path; keep private lists/files out of the repo.

---

## Contacts

Open an issue with: short description, config snippet, smallest `reports/...` artifact that reproduces the behavior. Keep reproduction runs short where possible.
