"""
Fused RMSNorm Triton Kernel

RMSNorm is bandwidth-bound: we read x, compute variance, normalize, multiply by weight.
PyTorch's default implementation may use multiple kernels.

This kernel fuses everything into one pass:
1. Load x and weight
2. Compute sum(x^2) / n
3. Compute x * rsqrt(var + eps) * weight
4. Store result

Compatible with A100 (SM80) and L4 (SM89).
"""

import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def rmsnorm_kernel(
        X,          # Input tensor pointer
        W,          # Weight tensor pointer
        Y,          # Output tensor pointer
        stride_x,   # Stride for X rows
        stride_y,   # Stride for Y rows
        n_cols: tl.constexpr,     # Number of columns
        eps: tl.constexpr,        # Epsilon for numerical stability
        BLOCK: tl.constexpr,      # Block size (power of 2 >= n_cols)
    ):
        """
        Fused RMSNorm kernel.
        
        For each row:
            y = x * rsqrt(mean(x^2) + eps) * weight
        """
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols
        
        # Load input row
        x_ptr = X + row * stride_x + cols
        x = tl.load(x_ptr, mask=mask, other=0.0).to(tl.float32)
        
        # Load weight (same for all rows)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        
        # Compute RMS: sqrt(mean(x^2))
        x_sq = x * x
        var = tl.sum(x_sq, axis=0) / n_cols
        inv_rms = tl.rsqrt(var + eps)
        
        # Normalize and apply weight
        y = (x * inv_rms) * w
        
        # Store output (cast back to input dtype)
        y_ptr = Y + row * stride_y + cols
        tl.store(y_ptr, y.to(tl.float16), mask=mask)


    @triton.jit
    def rmsnorm_kernel_bf16(
        X, W, Y,
        stride_x, stride_y,
        n_cols: tl.constexpr,
        eps: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """BFloat16 output variant."""
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols
        
        x_ptr = X + row * stride_x + cols
        x = tl.load(x_ptr, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        
        x_sq = x * x
        var = tl.sum(x_sq, axis=0) / n_cols
        inv_rms = tl.rsqrt(var + eps)
        y = (x * inv_rms) * w
        
        y_ptr = Y + row * stride_y + cols
        tl.store(y_ptr, y.to(tl.bfloat16), mask=mask)


def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Fused RMSNorm using Triton kernel.
    
    Args:
        x: Input tensor [..., hidden_dim]
        weight: Learnable scale [hidden_dim]
        eps: Epsilon for numerical stability
    
    Returns:
        Normalized tensor, same shape as x
    """
    if not HAS_TRITON:
        # Fallback to PyTorch
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + eps)
        return x_normed * weight
    
    assert x.is_cuda and weight.is_cuda, "Triton kernel requires CUDA tensors"
    
    # Flatten to 2D: [batch*seq, hidden]
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    n_rows, n_cols = x_2d.shape
    
    # Allocate output
    if x.dtype == torch.bfloat16:
        y = torch.empty_like(x_2d, dtype=torch.bfloat16)
    else:
        y = torch.empty_like(x_2d, dtype=torch.float16)
    
    # Compute block size (power of 2 >= n_cols)
    BLOCK = triton.next_power_of_2(n_cols)
    
    # Launch kernel
    grid = (n_rows,)
    
    if x.dtype == torch.bfloat16:
        rmsnorm_kernel_bf16[grid](
            x_2d, weight, y,
            x_2d.stride(0), y.stride(0),
            n_cols=n_cols,
            eps=eps,
            BLOCK=BLOCK,
        )
    else:
        rmsnorm_kernel[grid](
            x_2d, weight, y,
            x_2d.stride(0), y.stride(0),
            n_cols=n_cols,
            eps=eps,
            BLOCK=BLOCK,
        )
    
    return y.reshape(orig_shape)


class RMSNormTriton(nn.Module):
    """
    Drop-in replacement for RMSNorm using Triton kernel.
    
    Falls back to PyTorch if Triton is not available.
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rmsnorm(x, self.weight, self.eps)
    
    def _fallback_forward(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch fallback implementation."""
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.eps)
        return x_normed * self.weight


# Utility to patch existing RMSNorm modules
def patch_rmsnorm_triton(model: nn.Module) -> int:
    """
    Replace RMSNorm modules with Triton-accelerated versions.
    
    Returns number of modules patched.
    """
    if not HAS_TRITON:
        return 0
    
    patched = 0
    for name, module in model.named_modules():
        # Check for common RMSNorm class names
        class_name = module.__class__.__name__
        if 'RMSNorm' in class_name and hasattr(module, 'weight'):
            # Store original forward for fallback
            module._original_forward = module.forward
            
            # Create closure to capture module
            def make_forward(m):
                def triton_forward(x, num_chunks=1):
                    # Ignore num_chunks, we handle full tensor
                    return rmsnorm(x, m.weight, getattr(m, 'eps', 1e-6))
                return triton_forward
            
            module.forward = make_forward(module)
            patched += 1
    
    return patched
