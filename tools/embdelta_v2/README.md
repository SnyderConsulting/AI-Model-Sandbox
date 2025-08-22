# Embedding Delta v2 (between TE and DiT)

**Goal:** Learn a small residual adapter that adjusts UMT5‑XXL token embeddings *before* they reach Wan’s DiT, using ~750 `(base → rewritten)` prompts. No touching the encoder or the DiT.

**Key ideas**
- Low‑rank per‑token residual + **orthogonalization** to preserve encoder geometry
- **Token‑wise norm cap** for stability
- **Content‑adaptive prompt gate** (0–1) learned from pooled features for better gains on terse/single‑token prompts
- Simple sequence‑level cosine objective vs. rewritten prompts + small geometry/magnitude regularizers

**Outputs go to** `reports/embdelta_v2/*`.

## Quickstart

1. **Cache pairs (UMT5‑XXL encodings)**
   ```bash
   python tools/embdelta_v2/build_pairs_cache.py \
     --jsonl datasets/nsfw_rewrites.jsonl \
     --ckpt_root /workspace/models/Wan2.2-TI2V-5B \
     --out reports/embdelta_v2/cache/pairs_5b_2.2.npz
   ```

2. **Train adapter**

   ```bash
   python tools/embdelta_v2/train_embdelta.py \
     --cache reports/embdelta_v2/cache/pairs_5b_2.2.npz \
     --out_dir reports/embdelta_v2/run1 \
     --rank 64 --cap 0.22 --epochs 6000 --lr 3e-4 --batch_size 64
   ```

3. **Evaluate**

   ```bash
   python tools/embdelta_v2/eval_embdelta.py \
     --cache reports/embdelta_v2/cache/pairs_5b_2.2.npz \
     --adapter reports/embdelta_v2/run1/embdelta_adapter.pt \
     --out_dir reports/embdelta_v2/run1/eval
   ```

4. **Use at inference (example)**

   ```python
   # Get token embeddings from your UMT5-XXL as usual:
   ids, mask = t5.tokenizer([prompt], return_mask=True, add_special_tokens=True)
   seq_lens  = mask.gt(0).sum(dim=1).long()
   embs      = t5.model(ids.to(device), mask.to(device))  # list([Li, D])

   # Load adapter and apply:
   from models.wan.embdelta_adapter import load_embdelta_adapter, apply_to_text_embeddings
   adapter = load_embdelta_adapter("reports/embdelta_v2/run1/embdelta_adapter.pt", d_model=embs[0].size(-1), map_location=device)
   adj_embs = apply_to_text_embeddings(embs, seq_lens.to(device), adapter)
   # Feed adj_embs to the rest of your pipeline in place of TE outputs.
   ```

**Notes**

* Keep `cache_text_embeddings = False` if you integrate into a training/inference pipeline that would otherwise bypass the TE call. The adapter must see token embeddings at runtime.
* All scripts write reports/artifacts under `reports/embdelta_v2/*`.
