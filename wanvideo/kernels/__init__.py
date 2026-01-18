# Kernel optimizations for WanVideo
# Only includes PROVEN optimizations that ALWAYS help on L4 and A100

# JIT-compiled fused operations (torch.jit.script based)
# - Zero overhead after first call (JIT compilation)
# - PyTorch automatically fuses element-wise ops
# - Always enabled, no flags needed
from .jit_ops import fused_silu_mul, fused_rope_apply, fused_ln_modulate
