import torch
import torch.nn.functional as F


def _match_dtype(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """Ensure K/V match Q dtype for deterministic math parity."""
    if q.dtype == k.dtype == v.dtype:
        return q, k, v
    dtype = q.dtype
    return q, k.to(dtype), v.to(dtype)


def _scaled_attention(q, k, v, attn_mask=None, is_causal=False):
    q, k, v = _match_dtype(q, k, v)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)


def sdpa_flash(q, k, v, attn_mask=None, is_causal=False):
    """Force PyTorch SDPA to use flash or mem-efficient kernels."""
    if not (q.is_cuda and torch.cuda.is_available()):
        return _scaled_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
    q, k, v = _match_dtype(q, k, v)
    ctx = getattr(torch.backends, "cuda", None)
    kernel = getattr(ctx, "sdp_kernel", None)
    if callable(kernel):
        with kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False):
            return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
    return _scaled_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)


def sdpa_mem_efficient(q, k, v, attn_mask=None, is_causal=False):
    if not (q.is_cuda and torch.cuda.is_available()):
        return _scaled_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
    q, k, v = _match_dtype(q, k, v)
    ctx = getattr(torch.backends, "cuda", None)
    kernel = getattr(ctx, "sdp_kernel", None)
    if callable(kernel):
        with kernel(enable_flash=False, enable_mem_efficient=True, enable_math=False):
            return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
    return _scaled_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)


def attention(q, k, v, *, backend="auto", attn_mask=None, is_causal=False):
    """Select the fastest mathematically equivalent SDPA backend."""
    backend = (backend or "auto").lower()
    if backend == "sdpa_flash":
        return sdpa_flash(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
    if backend in ("sdpa_mem", "mem_efficient"):
        return sdpa_mem_efficient(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
    if backend == "math":
        return _scaled_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
    if backend != "auto":
        raise ValueError(f"Unknown attention backend: {backend}")

    # Auto-select: try flash, fall back to mem-efficient, then math.
    try:
        return sdpa_flash(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
    except Exception:
        pass

    try:
        return sdpa_mem_efficient(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
    except Exception:
        pass

    return _scaled_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
