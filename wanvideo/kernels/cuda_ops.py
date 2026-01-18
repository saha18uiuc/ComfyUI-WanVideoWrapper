"""
CUDA Kernels for WanVideo

Simple wrapper that compiles CUDA kernels once and uses them.
No runtime benchmarking - just use CUDA if it compiles, else fallback.
"""

import os
import torch
from pathlib import Path

_cuda_ops = None
_compilation_attempted = False


def _compile_cuda_ops():
    """Compile CUDA kernels (only called once)."""
    global _cuda_ops, _compilation_attempted
    
    if _compilation_attempted:
        return _cuda_ops
    
    _compilation_attempted = True
    
    if not torch.cuda.is_available():
        return None
    
    try:
        from torch.utils.cpp_extension import load
        
        cuda_src = Path(__file__).parent / "cuda" / "fused_ops.cu"
        if not cuda_src.exists():
            return None
        
        # Simple compilation flags - should work on any CUDA version
        _cuda_ops = load(
            name="wanvideo_fused_ops",
            sources=[str(cuda_src)],
            extra_cuda_cflags=["-O3"],  # Just optimization, no fancy flags
            verbose=False,
        )
        print("[WanVideo] CUDA kernels compiled successfully")
        return _cuda_ops
        
    except Exception as e:
        # Silently fall back to PyTorch
        print(f"[WanVideo] CUDA kernel compilation failed, using PyTorch: {e}")
        return None


def fused_silu_mul(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """
    Fused SiLU(x) * gate.
    Uses CUDA kernel if available, else PyTorch.
    """
    ops = _compile_cuda_ops()
    
    if ops is not None and x.is_cuda and x.is_contiguous() and gate.is_contiguous():
        try:
            return ops.fused_silu_mul(x.contiguous(), gate.contiguous())
        except Exception:
            pass
    
    return torch.nn.functional.silu(x) * gate


def fused_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Fused RMSNorm.
    Uses CUDA kernel if available, else PyTorch F.rms_norm, else manual.
    """
    ops = _compile_cuda_ops()
    
    if ops is not None and x.is_cuda and x.is_contiguous():
        try:
            return ops.fused_rmsnorm(x.contiguous(), weight, eps)
        except Exception:
            pass
    
    # Fallback to PyTorch's F.rms_norm (2.4+)
    if hasattr(torch.nn.functional, 'rms_norm'):
        return torch.nn.functional.rms_norm(x, (x.shape[-1],), weight, eps)
    
    # Manual fallback
    dtype = x.dtype
    x = x.float()
    rstd = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x * rstd).to(dtype) * weight


def is_available() -> bool:
    """Check if CUDA kernels compiled successfully."""
    return _compile_cuda_ops() is not None
