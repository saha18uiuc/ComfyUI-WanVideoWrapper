import torch
import torch.nn.functional as F
from contextlib import contextmanager

try:
    from torch.nn.attention import sdpa_kernel as torch_sdpa_kernel
    from torch.nn.attention import SDPBackend
except Exception:
    torch_sdpa_kernel = None
    SDPBackend = None


def _match_dtype(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """Ensure K/V match Q dtype for deterministic math parity."""
    if q.dtype == k.dtype == v.dtype:
        return q, k, v
    dtype = q.dtype
    return q, k.to(dtype), v.to(dtype)


def _scaled_attention(q, k, v, attn_mask=None, is_causal=False):
    q, k, v = _match_dtype(q, k, v)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)


@contextmanager
def _sdpa_context(preferred_backends, *, fallback_flash=False, fallback_mem=False):
    if torch_sdpa_kernel is not None and SDPBackend is not None:
        backend_objs = []
        for name in preferred_backends:
            backend = getattr(SDPBackend, name, None)
            if backend is not None:
                backend_objs.append(backend)
        if backend_objs:
            with torch_sdpa_kernel(backend_objs):
                yield
                return
    ctx = getattr(torch.backends, "cuda", None)
    sdp_ctx = getattr(ctx, "sdp_kernel", None)
    if callable(sdp_ctx):
        with sdp_ctx(
            enable_flash=fallback_flash,
            enable_mem_efficient=fallback_mem,
            enable_math=False,
        ):
            yield
            return
    yield


def sdpa_flash(q, k, v, attn_mask=None, is_causal=False):
    """Force PyTorch SDPA to use flash or mem-efficient kernels."""
    if not (q.is_cuda and torch.cuda.is_available()):
        return _scaled_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
    q, k, v = _match_dtype(q, k, v)
    with _sdpa_context(
        preferred_backends=["FLASH_ATTENTION", "EFFICIENT_ATTENTION", "MATH"],
        fallback_flash=True,
        fallback_mem=True,
    ):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)


def sdpa_mem_efficient(q, k, v, attn_mask=None, is_causal=False):
    if not (q.is_cuda and torch.cuda.is_available()):
        return _scaled_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
    q, k, v = _match_dtype(q, k, v)
    with _sdpa_context(
        preferred_backends=["EFFICIENT_ATTENTION", "MATH"],
        fallback_flash=False,
        fallback_mem=True,
    ):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)


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
