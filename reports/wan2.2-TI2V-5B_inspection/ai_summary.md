Here’s what your **Wan 2.2 TI2V‑5B** introspection is telling us, and why it matters:

### What the report says (facts)

* **Checkpoint & variant.** It’s the `Wan2.2‑TI2V‑5B` model; the config identifies the variant as **`ti2v`**, not i2v. The inspector also flags **no i2v‑specific keys** (e.g., no `k_img` projections), so this ckpt is *not* the image‑conditioned i2v branch.&#x20;
* **Depth / width.** The transformer has **30 layers** (`num_layers_cfg=30`) with **model width `d_model=3072`** and **24 attention heads** (`num_heads_cfg=24`). With 3072/24, each head is **128‑dim**.&#x20;
* **Text budget.** **Text length is 512 tokens** (`text_len_cfg=512`).&#x20;
* **No experts/MoE.** **`experts_detected: ["default"]`** indicates a single monolithic DiT (no mixture‑of‑experts or dual‑stage towers in this 5B ckpt).&#x20;
* **Block makeup (what’s trainable).** Every one of the **30 blocks** exposes a full set of **self‑attention and cross‑attention** projections: **q, k, v, o** are present for both, and the **FFN** has two linear weights (indexed as `0` and `2`). In short, you’ve got **self\_attn.{q,k,v,o} ×30**, **cross\_attn.{q,k,v,o} ×30**, and **ffn.{0,2} ×30** as addressable targets.&#x20;
* The same summary appears in your combined `ALL__model_summaries.json`, confirming this layout is what your scripts detected.&#x20;

### What that implies (practical takeaways)

* **Architecture scale vs Wan 2.1 (1.3B).** Compared with your earlier 1.3B runs (where block weights were `1536×1536` and heads were 12), this 5B keeps the **per‑head size at 128** but **doubles the width (1536→3072) and heads (12→24)** while keeping **30 layers**. That means each attention projection is now **3072×3072**—**4×** the parameters per matrix relative to 1.3B. (This lines up with the distortion sensitivity you saw when over‑adapting attention at 1.3B; the 5B has much more capacity per projection, so LoRA placement/magnitude matters even more.)
* **LoRA sizing math (helps plan what to adapt).** For a square 3072×3072 projection, a rank‑**r** LoRA has **r × (3072+3072) = r × 6144** trainable params per matrix.

  * With **r=16**, that’s **98,304** params per matrix.
  * If you adapt **cross\_attn.{k,v} only** across 30 blocks (60 matrices total): **≈ 5.90 M** LoRA params (\~**11.25 MiB** at bf16).
  * If you adapt **cross\_attn.{q,k,v,o}** across 30 blocks (120 matrices): **≈ 11.80 M** (\~**22.5 MiB**).
  * If you adapted **both self & cross (all 8 per block)** across 30 blocks (240 matrices): **≈ 23.59 M** (\~**45 MiB**).
    These budgets are modest, but the *behavioral* impact differs a lot (as your experiments showed).
* **Targets for a “translator” LoRA.** Because **every block** has cross‑attention and self‑attention, you can precisely select what to touch. Your earlier findings (disabling `self_attn.*` removed artifacts and improved clarity) align with prioritizing **cross\_attn.K/V** as the safest dials for “listening better” to the encoder while avoiding global self‑attn instabilities.
* **512‑token context matters.** The encoder→DiT interface expects **up to 512 tokens**, so the encoder work you did (strong single‑token semantics, better NSFW coverage) can be expressed without truncation in this ckpt, and prompt‑rewriter ideas can safely expand short prompts when useful.

### TL;DR

* **Wan 2.2 TI2V‑5B** is a **single‑tower, 30‑layer DiT** with **3072‑dim width**, **24 heads (128 per head)**, **512‑token text**, and **full self+cross attention** plus FFN in every block—**no MoE** here. This gives you clean, uniform hooks for LoRA targeting per layer. &#x20;
* Given your prior results, a translator LoRA that **avoids self‑attention** and focuses on **cross\_attn (especially K/V)** is the most consistent starting point for aligning the 5B DiT to your improved NSFW encoder without degrading image fidelity.
