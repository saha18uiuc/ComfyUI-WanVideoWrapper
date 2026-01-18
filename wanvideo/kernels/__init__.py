# Kernel optimizations for WanVideo
# Uses CUDA kernels when available, falls back to PyTorch

from .cuda_ops import fused_silu_mul, fused_rmsnorm, is_available
