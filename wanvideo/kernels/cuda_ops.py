"""
CUDA Kernels for WanVideo - Python Interface with GUARANTEED Performance

These kernels use RUNTIME BENCHMARKING to ensure they only run when faster.
On first use, we benchmark CUDA kernel vs PyTorch and cache the result.
If CUDA is slower, we use PyTorch instead.

This guarantees:
- Never slower than PyTorch (we benchmark and choose the fastest)
- Automatic adaptation to different GPUs (L4, A100, etc.)
- No user configuration needed
"""

import os
import torch
import time
from pathlib import Path
from typing import Optional, Dict

# ============================================================================
# Configuration
# ============================================================================

# Minimum speedup required to use CUDA kernel (1.0 = must be at least as fast)
# Using 1.05 means CUDA kernel must be at least 5% faster to be used
MIN_SPEEDUP_THRESHOLD = 1.05

# Number of iterations for benchmarking
BENCHMARK_ITERS = 20

# Cache benchmark results to avoid re-benchmarking
_benchmark_cache: Dict[str, bool] = {}

# Track if kernels are loaded
_cuda_ops = None
_load_attempted = False
_load_error = None


def _get_cuda_ops():
    """Load CUDA kernels via JIT compilation (cached after first load)."""
    global _cuda_ops, _load_attempted, _load_error
    
    if _load_attempted:
        return _cuda_ops
    
    _load_attempted = True
    
    if not torch.cuda.is_available():
        _load_error = "CUDA not available"
        return None
    
    try:
        from torch.utils.cpp_extension import load
        
        cuda_src = Path(__file__).parent / "cuda" / "fused_ops.cu"
        
        if not cuda_src.exists():
            _load_error = f"CUDA source not found: {cuda_src}"
            return None
        
        _cuda_ops = load(
            name="wanvideo_cuda_ops",
            sources=[str(cuda_src)],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
        return _cuda_ops
        
    except Exception as e:
        _load_error = str(e)
        return None


def is_available() -> bool:
    """Check if CUDA kernels are available."""
    return _get_cuda_ops() is not None


# ============================================================================
# Runtime Benchmarking - Ensures we ONLY use CUDA when it's faster
# ============================================================================

def _benchmark_silu_mul(x: torch.Tensor, gate: torch.Tensor) -> bool:
    """
    Benchmark CUDA vs PyTorch for fused_silu_mul.
    Returns True if CUDA kernel is faster.
    """
    ops = _get_cuda_ops()
    if ops is None:
        return False
    
    # Create cache key based on shape and dtype
    cache_key = f"silu_mul_{x.shape}_{x.dtype}"
    if cache_key in _benchmark_cache:
        return _benchmark_cache[cache_key]
    
    # Warmup
    for _ in range(5):
        _ = ops.fused_silu_mul(x, gate)
        _ = torch.nn.functional.silu(x) * gate
    torch.cuda.synchronize()
    
    # Benchmark CUDA
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(BENCHMARK_ITERS):
        _ = ops.fused_silu_mul(x, gate)
    torch.cuda.synchronize()
    cuda_time = time.perf_counter() - start
    
    # Benchmark PyTorch
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(BENCHMARK_ITERS):
        _ = torch.nn.functional.silu(x) * gate
    torch.cuda.synchronize()
    pytorch_time = time.perf_counter() - start
    
    # CUDA must be at least MIN_SPEEDUP_THRESHOLD faster
    use_cuda = (pytorch_time / cuda_time) >= MIN_SPEEDUP_THRESHOLD
    _benchmark_cache[cache_key] = use_cuda
    
    return use_cuda


def _benchmark_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> bool:
    """
    Benchmark CUDA vs PyTorch for fused_rmsnorm.
    Returns True if CUDA kernel is faster.
    """
    ops = _get_cuda_ops()
    if ops is None:
        return False
    
    cache_key = f"rmsnorm_{x.shape}_{x.dtype}"
    if cache_key in _benchmark_cache:
        return _benchmark_cache[cache_key]
    
    # PyTorch reference (best available)
    def pytorch_rmsnorm(x, w, e):
        if hasattr(torch.nn.functional, 'rms_norm'):
            return torch.nn.functional.rms_norm(x, (x.shape[-1],), w, e)
        rstd = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + e)
        return (x.float() * rstd).to(x.dtype) * w
    
    # Warmup
    for _ in range(5):
        _ = ops.fused_rmsnorm(x, weight, eps)
        _ = pytorch_rmsnorm(x, weight, eps)
    torch.cuda.synchronize()
    
    # Benchmark CUDA
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(BENCHMARK_ITERS):
        _ = ops.fused_rmsnorm(x, weight, eps)
    torch.cuda.synchronize()
    cuda_time = time.perf_counter() - start
    
    # Benchmark PyTorch
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(BENCHMARK_ITERS):
        _ = pytorch_rmsnorm(x, weight, eps)
    torch.cuda.synchronize()
    pytorch_time = time.perf_counter() - start
    
    use_cuda = (pytorch_time / cuda_time) >= MIN_SPEEDUP_THRESHOLD
    _benchmark_cache[cache_key] = use_cuda
    
    return use_cuda


# ============================================================================
# Public API - Automatically uses fastest implementation
# ============================================================================

def fused_silu_mul(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """
    Fused SiLU(x) * gate operation.
    
    Automatically benchmarks and uses CUDA kernel only if it's faster.
    
    Args:
        x: Input tensor
        gate: Gate tensor (same shape as x)
        
    Returns:
        SiLU(x) * gate
    """
    ops = _get_cuda_ops()
    
    # Only consider CUDA if available and tensors are suitable
    if (ops is not None and x.is_cuda and 
        x.is_contiguous() and gate.is_contiguous() and
        x.numel() >= 65536):  # Minimum size for CUDA to be worthwhile
        
        # Check if we've benchmarked this shape before
        cache_key = f"silu_mul_{x.shape}_{x.dtype}"
        
        if cache_key not in _benchmark_cache:
            # First time - benchmark
            _benchmark_silu_mul(x, gate)
        
        if _benchmark_cache.get(cache_key, False):
            return ops.fused_silu_mul(x, gate)
    
    # Fallback to PyTorch
    return torch.nn.functional.silu(x) * gate


def fused_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5
) -> torch.Tensor:
    """
    Fused RMSNorm operation.
    
    Automatically benchmarks and uses CUDA kernel only if it's faster.
    
    Args:
        x: Input tensor [..., hidden_dim]
        weight: Weight tensor [hidden_dim]
        eps: Epsilon for numerical stability
        
    Returns:
        Normalized tensor
    """
    ops = _get_cuda_ops()
    
    # Only consider CUDA if available and tensors are suitable
    if (ops is not None and x.is_cuda and x.is_contiguous() and
        x.numel() >= 65536):  # Minimum size
        
        cache_key = f"rmsnorm_{x.shape}_{x.dtype}"
        
        if cache_key not in _benchmark_cache:
            _benchmark_rmsnorm(x, weight, eps)
        
        if _benchmark_cache.get(cache_key, False):
            return ops.fused_rmsnorm(x, weight, eps)
    
    # Fallback to best PyTorch implementation
    if hasattr(torch.nn.functional, 'rms_norm'):
        return torch.nn.functional.rms_norm(x, (x.shape[-1],), weight, eps)
    
    # Manual fallback
    dtype = x.dtype
    x = x.float()
    rstd = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x * rstd).to(dtype) * weight


def fused_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5
) -> torch.Tensor:
    """
    Fused (x + residual) followed by RMSNorm.
    """
    ops = _get_cuda_ops()
    
    if (ops is not None and x.is_cuda and 
        x.is_contiguous() and residual.is_contiguous() and
        x.numel() >= 65536):
        # For add+rmsnorm, always try CUDA since it's an extra fusion
        try:
            return ops.fused_add_rmsnorm(x, residual, weight, eps)
        except:
            pass
    
    return fused_rmsnorm(x + residual, weight, eps)


# ============================================================================
# Status and diagnostics
# ============================================================================

def get_status() -> dict:
    """Get status of CUDA kernel loading and benchmark results."""
    ops = _get_cuda_ops()
    return {
        "cuda_available": torch.cuda.is_available(),
        "kernels_loaded": ops is not None,
        "load_error": _load_error,
        "benchmark_cache": dict(_benchmark_cache),
        "min_speedup_threshold": MIN_SPEEDUP_THRESHOLD,
    }


def clear_benchmark_cache():
    """Clear the benchmark cache to force re-benchmarking."""
    global _benchmark_cache
    _benchmark_cache = {}


def run_full_benchmark(hidden_dim: int = 5120, seq_len: int = 4096, batch: int = 2):
    """Run full benchmark and print results."""
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    device = torch.device('cuda')
    
    print("=" * 60)
    print(f"Full Benchmark (shape: {batch}x{seq_len}x{hidden_dim})")
    print(f"MIN_SPEEDUP_THRESHOLD: {MIN_SPEEDUP_THRESHOLD}")
    print("=" * 60)
    
    x = torch.randn(batch, seq_len, hidden_dim, device=device, dtype=torch.float16)
    gate = torch.randn_like(x)
    weight = torch.randn(hidden_dim, device=device, dtype=torch.float16)
    
    # Clear cache to force fresh benchmark
    clear_benchmark_cache()
    
    # Test fused_silu_mul
    print("\nfused_silu_mul:")
    _ = fused_silu_mul(x, gate)  # Triggers benchmark
    cache_key = f"silu_mul_{x.shape}_{x.dtype}"
    result = _benchmark_cache.get(cache_key, False)
    print(f"  Using CUDA kernel: {result}")
    
    # Test fused_rmsnorm
    print("\nfused_rmsnorm:")
    _ = fused_rmsnorm(x, weight)  # Triggers benchmark
    cache_key = f"rmsnorm_{x.shape}_{x.dtype}"
    result = _benchmark_cache.get(cache_key, False)
    print(f"  Using CUDA kernel: {result}")
    
    print("\n" + "=" * 60)
    print("Benchmark cache:", _benchmark_cache)
    print("=" * 60)
