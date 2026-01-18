"""
Fused RMSNorm Triton Kernel
============================

This is a REAL kernel-level optimization that fuses:
  - x.pow(2)
  - .mean(dim=-1)
  - rsqrt(... + eps)
  - x * rsqrt_result
  - result * weight

Into a SINGLE CUDA kernel, saving 4-5 kernel launches per RMSNorm call.

Since RMSNorm is called ~80+ times per diffusion step (2x per transformer block,
40 blocks), this saves ~320+ kernel launches per step.

At ~5μs per kernel launch overhead, this saves ~1.6ms per step.
Over 30 steps, this saves ~48ms total.

Reference: NVIDIA Apex FusedRMSNorm
"""

import torch
import os

# Environment variable to disable this kernel if needed
ENABLE_TRITON_RMSNORM = os.environ.get("WAN_ENABLE_TRITON_RMSNORM", "1").strip().lower() in ("1", "true", "yes")

_HAS_TRITON = False
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    pass


if _HAS_TRITON:
    
    @triton.jit
    def _rms_norm_fwd_kernel(
        X,  # input pointer
        W,  # weight pointer  
        Y,  # output pointer
        stride_x,  # stride for batch dimension
        N,  # hidden dimension
        eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Fused RMSNorm forward kernel.
        
        For each row (batch element):
        1. Compute sum of squares
        2. Compute rsqrt(mean_sq + eps)
        3. Multiply x * rsqrt * weight
        
        All in one kernel!
        """
        row_idx = tl.program_id(0)
        
        # Pointer to start of this row
        X_row = X + row_idx * stride_x
        Y_row = Y + row_idx * stride_x
        
        # Compute sum of squares in blocks
        sum_sq = 0.0
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            x = tl.load(X_row + cols, mask=mask, other=0.0).to(tl.float32)
            sum_sq += tl.sum(x * x, axis=0)
        
        # Compute normalization factor: rsqrt(mean_sq + eps)
        mean_sq = sum_sq / N
        rstd = tl.rsqrt(mean_sq + eps)
        
        # Apply normalization and weight
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            x = tl.load(X_row + cols, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(W + cols, mask=mask, other=1.0).to(tl.float32)
            
            # Fused: x * rstd * w
            y = x * rstd * w
            
            tl.store(Y_row + cols, y, mask=mask)


def triton_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Fused RMSNorm using Triton.
    
    Args:
        x: Input tensor of shape [..., hidden_dim]
        weight: Weight tensor of shape [hidden_dim]
        eps: Epsilon for numerical stability
        
    Returns:
        Normalized tensor of same shape as x
    """
    if not _HAS_TRITON or not ENABLE_TRITON_RMSNORM or not x.is_cuda:
        # Fallback to PyTorch
        return _pytorch_rms_norm(x, weight, eps)
    
    # Flatten to 2D for kernel
    orig_shape = x.shape
    x_2d = x.view(-1, x.shape[-1])
    
    # Allocate output
    y = torch.empty_like(x_2d)
    
    # Grid: one program per row
    n_rows = x_2d.shape[0]
    hidden_dim = x_2d.shape[1]
    
    # Choose block size based on hidden dimension
    # Larger hidden dims can use larger blocks
    if hidden_dim <= 1024:
        BLOCK_SIZE = 1024
    elif hidden_dim <= 2048:
        BLOCK_SIZE = 1024
    elif hidden_dim <= 4096:
        BLOCK_SIZE = 1024
    else:
        BLOCK_SIZE = 2048
    
    # Ensure x is contiguous
    x_2d = x_2d.contiguous()
    
    _rms_norm_fwd_kernel[(n_rows,)](
        x_2d, weight, y,
        x_2d.stride(0),
        hidden_dim,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return y.view(orig_shape)


def _pytorch_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """PyTorch reference implementation of RMSNorm."""
    dtype = x.dtype
    x = x.float()
    rstd = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x * rstd).to(dtype) * weight


# ============================================================================
# Validation
# ============================================================================

def validate_rmsnorm_kernel():
    """Validate Triton kernel against PyTorch reference."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping validation")
        return False
    
    print("=" * 50)
    print("Validating Triton RMSNorm Kernel")
    print("=" * 50)
    
    torch.manual_seed(42)
    device = torch.device('cuda')
    
    # Test different shapes
    test_cases = [
        (2, 1024, 5120),   # Typical transformer
        (1, 4096, 5120),   # Longer sequence
        (4, 512, 2048),    # Smaller hidden
    ]
    
    all_passed = True
    
    for batch, seq, hidden in test_cases:
        x = torch.randn(batch, seq, hidden, device=device, dtype=torch.float16)
        weight = torch.randn(hidden, device=device, dtype=torch.float16)
        
        # Reference
        ref = _pytorch_rms_norm(x, weight, eps=1e-5)
        
        # Triton (if available)
        if _HAS_TRITON and ENABLE_TRITON_RMSNORM:
            out = triton_rms_norm(x, weight, eps=1e-5)
            max_diff = (ref - out).abs().max().item()
            passed = max_diff < 1e-2  # FP16 tolerance
            print(f"Shape {batch}x{seq}x{hidden}: max_diff={max_diff:.2e}, passed={passed}")
            all_passed = all_passed and passed
        else:
            print(f"Shape {batch}x{seq}x{hidden}: Triton not available, using PyTorch")
    
    print("=" * 50)
    print(f"Overall: {'PASSED' if all_passed else 'FAILED'}")
    print("=" * 50)
    
    return all_passed


def benchmark_rmsnorm():
    """Benchmark Triton vs PyTorch RMSNorm."""
    if not torch.cuda.is_available() or not _HAS_TRITON:
        print("CUDA/Triton not available")
        return
    
    import time
    
    print("=" * 50)
    print("Benchmarking RMSNorm")
    print("=" * 50)
    
    device = torch.device('cuda')
    batch, seq, hidden = 2, 4096, 5120
    
    x = torch.randn(batch, seq, hidden, device=device, dtype=torch.float16)
    weight = torch.randn(hidden, device=device, dtype=torch.float16)
    
    # Warmup
    for _ in range(10):
        _ = triton_rms_norm(x, weight)
        _ = _pytorch_rms_norm(x, weight)
    torch.cuda.synchronize()
    
    # Benchmark Triton
    n_iters = 100
    start = time.perf_counter()
    for _ in range(n_iters):
        _ = triton_rms_norm(x, weight)
    torch.cuda.synchronize()
    triton_time = (time.perf_counter() - start) / n_iters * 1000
    
    # Benchmark PyTorch
    start = time.perf_counter()
    for _ in range(n_iters):
        _ = _pytorch_rms_norm(x, weight)
    torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - start) / n_iters * 1000
    
    print(f"Shape: {batch}x{seq}x{hidden}")
    print(f"Triton:  {triton_time:.3f}ms")
    print(f"PyTorch: {pytorch_time:.3f}ms")
    print(f"Speedup: {pytorch_time/triton_time:.2f}x")


if __name__ == "__main__":
    validate_rmsnorm_kernel()
    print()
    benchmark_rmsnorm()
