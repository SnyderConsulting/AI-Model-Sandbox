# wan/modules/attention.py
# FlashAttention (v3/v2) when available; graceful SDPA fallback otherwise.

import warnings
import torch

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

__all__ = ["flash_attention", "attention"]


def _sdpa_fallback(
    q, k, v, dropout_p=0.0, softmax_scale=None, q_scale=None, causal=False, dtype=torch.bfloat16
):
    """
    Fallback using torch.nn.functional.scaled_dot_product_attention.
    Shapes:
      q,k,v: [B, L, H, C]  -> SDPA expects [B, H, L, C]
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    out_dtype = q.dtype

    # Move to half precision if needed
    tgt_dtype = dtype if dtype in half_dtypes else torch.bfloat16
    q = q.transpose(1, 2).to(tgt_dtype).contiguous()
    k = k.transpose(1, 2).to(tgt_dtype).contiguous()
    v = v.transpose(1, 2).to(tgt_dtype).contiguous()

    # Emulate optional scales
    if q_scale is not None:
        q = q * q_scale
    if softmax_scale is not None:
        q = q * softmax_scale

    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=causal, dropout_p=dropout_p
    )
    out = out.transpose(1, 2).contiguous()
    return out.to(out_dtype)


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.0,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q: [B, Lq, Nq, C1]
    k: [B, Lk, Nk, C1]
    v: [B, Lk, Nk, C2]
    Uses FA3/FA2 if installed; otherwise auto-fallback to SDPA.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    out_dtype = q.dtype
    b, lq, lk = q.size(0), q.size(1), k.size(1)

    # Prefer FA3, then FA2
    use_fa3 = (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE
    use_fa2 = (not use_fa3) and FLASH_ATTN_2_AVAILABLE

    # Conditions where FA cannot/should not be used → SDPA fallback
    if q.device.type != "cuda" or q.size(-1) > 256 or (not use_fa3 and not use_fa2):
        return _sdpa_fallback(q, k, v, dropout_p, softmax_scale, q_scale, causal, dtype)

    # Helper cast
    def _half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # Pack varlen Q
    if q_lens is None:
        q_lens = torch.full((b,), lq, dtype=torch.int32, device=q.device)
        q_flat = _half(q.flatten(0, 1))
    else:
        q_flat = _half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # Pack varlen K/V
    if k_lens is None:
        k_lens = torch.full((b,), lk, dtype=torch.int32, device=k.device)
        k_flat = _half(k.flatten(0, 1))
        v_flat = _half(v.flatten(0, 1))
    else:
        k_flat = _half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v_flat = _half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    # Align dtypes
    q_flat = q_flat.to(v_flat.dtype)
    k_flat = k_flat.to(v_flat.dtype)
    if q_scale is not None:
        q_flat = q_flat * q_scale

    # Try FA3
    if use_fa3:
        try:
            x = flash_attn_interface.flash_attn_varlen_func(
                q=q_flat,
                k=k_flat,
                v=v_flat,
                cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens])
                .cumsum(0, dtype=torch.int32)
                .to(q.device, non_blocking=True),
                cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens])
                .cumsum(0, dtype=torch.int32)
                .to(q.device, non_blocking=True),
                seqused_q=None,
                seqused_k=None,
                max_seqlen_q=lq,
                max_seqlen_k=lk,
                softmax_scale=softmax_scale,
                causal=causal,
                deterministic=deterministic,
            )[0].unflatten(0, (b, lq))
            return x.type(out_dtype)
        except Exception as e:
            warnings.warn(f"FlashAttention v3 failed ({e}); falling back to v2/SDPA.")

    # Try FA2
    if use_fa2:
        try:
            x = flash_attn.flash_attn_varlen_func(
                q=q_flat,
                k=k_flat,
                v=v_flat,
                cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens])
                .cumsum(0, dtype=torch.int32)
                .to(q.device, non_blocking=True),
                cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens])
                .cumsum(0, dtype=torch.int32)
                .to(q.device, non_blocking=True),
                max_seqlen_q=lq,
                max_seqlen_k=lk,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                deterministic=deterministic,
            ).unflatten(0, (b, lq))
            return x.type(out_dtype)
        except Exception as e:
            warnings.warn(f"FlashAttention v2 failed ({e}); using SDPA fallback.")

    # Final guard: SDPA fallback
    return _sdpa_fallback(q, k, v, dropout_p, softmax_scale, q_scale, causal, dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.0,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    # Keep original wrapper semantics, but route through the resilient implementation above
    return flash_attention(
        q=q,
        k=k,
        v=v,
        q_lens=q_lens,
        k_lens=k_lens,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        q_scale=q_scale,
        causal=causal,
        window_size=window_size,
        deterministic=deterministic,
        dtype=dtype,
        version=fa_version,
    )
