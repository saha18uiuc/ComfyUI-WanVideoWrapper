# Kernel optimizations for WanVideo
# Only includes PROVEN optimizations with minimal overhead

# JIT-compiled fused operations (torch.jit.script based - zero overhead)
from .jit_ops import fused_silu_mul, fused_rope_apply, fused_ln_modulate

# Triton RMSNorm kernel - fuses 4-5 operations into 1 kernel
try:
    from .rmsnorm_triton import triton_rms_norm, validate_rmsnorm_kernel
except ImportError:
    # Triton not available
    pass
