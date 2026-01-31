"""
Fused Triton Kernels for Video Diffusion

These kernels target bandwidth-bound operations that PyTorch often launches
as many tiny kernels. By fusing operations, we reduce:
1. Kernel launch overhead
2. Memory bandwidth (fewer global memory round-trips)
3. GPU occupancy issues from small kernels

Kernels provided:
- rmsnorm_triton: Fused RMSNorm with weight multiplication
- swiglu_triton: Fused SwiGLU activation (SiLU(x) * y)
- rope_triton: Fused Rotary Position Embedding (optional)

These are portable across A100 (SM80) and L4 (SM89) and have minimal
cold-start cost (single Triton compilation per kernel).
"""

try:
    from .rmsnorm_triton import rmsnorm, RMSNormTriton
    HAS_RMSNORM = True
except ImportError:
    HAS_RMSNORM = False
    rmsnorm = None
    RMSNormTriton = None

try:
    from .swiglu_triton import swiglu, swiglu_fused
    HAS_SWIGLU = True
except ImportError:
    HAS_SWIGLU = False
    swiglu = None
    swiglu_fused = None

__all__ = [
    'rmsnorm',
    'RMSNormTriton',
    'swiglu',
    'swiglu_fused',
    'HAS_RMSNORM',
    'HAS_SWIGLU',
]
