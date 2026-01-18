# Kernel optimizations for WanVideo
# Uses PyTorch's built-in optimizations - no custom kernels needed

import torch
import torch.nn.functional as F

# Simple fused SiLU*mul using PyTorch (no JIT overhead)
def fused_silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Fused SiLU(gate) * up - uses PyTorch's optimized F.silu."""
    return F.silu(gate) * up
