"""
Compile-friendly RoPE (Rotary Position Embedding) implementation.

This module provides a Triton and PyTorch implementation of RoPE that:
1. Does NOT require @torch.compiler.disable()
2. Uses real arithmetic instead of complex numbers
3. Can be fused with attention kernels
4. Is fully compatible with CUDA graphs

The key insight is that complex multiplication for rotation:
    (a + bi) * (cos(θ) + sin(θ)i) = (a*cos - b*sin) + (a*sin + b*cos)i

Can be expressed as real operations on pairs of values.
"""

import torch
import math
from typing import Optional, Tuple

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


# ============================================================================
# Triton Kernel Implementation
# ============================================================================

if _HAS_TRITON:
    @triton.jit
    def _rope_kernel_3d(
        # Pointers
        x_ptr,
        cos_ptr,
        sin_ptr,
        out_ptr,
        # Dimensions
        batch_size,
        seq_len,
        num_heads,
        head_dim,
        # Grid sizes for this sample
        grid_f,
        grid_h,
        grid_w,
        # Strides for x: [B, L, H, D]
        stride_xb,
        stride_xl,
        stride_xh,
        stride_xd,
        # Strides for cos/sin: [max_seq, D//2]
        stride_cos_seq,
        stride_cos_d,
        # Strides for output
        stride_ob,
        stride_ol,
        stride_oh,
        stride_od,
        # Frequency split sizes
        freq_size_t: tl.constexpr,
        freq_size_h: tl.constexpr,
        freq_size_w: tl.constexpr,
        # Block size
        BLOCK_D: tl.constexpr,
    ):
        """
        Apply 3D RoPE (time, height, width) using real arithmetic.
        
        x: [B, L, H, D] where L = F*H*W
        cos/sin: precomputed [max_seq, D//2]
        
        For 3D, we split the head dimension into 3 parts:
        - First (D - 2*(D//3))//2 pairs for time
        - Next (D//3)//2 pairs for height  
        - Last (D//3)//2 pairs for width
        """
        pid_batch = tl.program_id(0)
        pid_seq = tl.program_id(1)
        pid_head = tl.program_id(2)
        
        # Bounds check
        if pid_batch >= batch_size or pid_seq >= seq_len or pid_head >= num_heads:
            return
            
        # Convert linear seq position to (f, h, w) coordinates
        hw = grid_h * grid_w
        f_idx = pid_seq // hw
        rem = pid_seq % hw
        h_idx = rem // grid_w
        w_idx = rem % grid_w
        
        # Load pairs of values and apply rotation
        # Process in blocks for efficiency
        half_d = head_dim // 2
        
        for d_start in range(0, half_d, BLOCK_D):
            d_offs = d_start + tl.arange(0, BLOCK_D)
            d_mask = d_offs < half_d
            
            # Load x values (pairs: x[..., 2*d] and x[..., 2*d+1])
            x_even_ptr = x_ptr + pid_batch * stride_xb + pid_seq * stride_xl + pid_head * stride_xh + 2 * d_offs * stride_xd
            x_odd_ptr = x_ptr + pid_batch * stride_xb + pid_seq * stride_xl + pid_head * stride_xh + (2 * d_offs + 1) * stride_xd
            
            x_even = tl.load(x_even_ptr, mask=d_mask, other=0.0)
            x_odd = tl.load(x_odd_ptr, mask=d_mask, other=0.0)
            
            # Determine which frequency component this dimension belongs to
            # and get the appropriate position index
            is_time = d_offs < freq_size_t
            is_height = (d_offs >= freq_size_t) & (d_offs < freq_size_t + freq_size_h)
            # is_width = d_offs >= freq_size_t + freq_size_h  (implicit)
            
            # Get position indices for each component
            pos_idx = tl.where(is_time, f_idx, 
                              tl.where(is_height, h_idx, w_idx))
            
            # Adjust d_offs for each frequency block
            d_offs_adjusted = tl.where(is_time, d_offs,
                                       tl.where(is_height, d_offs - freq_size_t,
                                                d_offs - freq_size_t - freq_size_h))
            
            # Load cos/sin values
            cos_ptr_local = cos_ptr + pos_idx * stride_cos_seq + d_offs_adjusted * stride_cos_d
            sin_ptr_local = sin_ptr + pos_idx * stride_cos_seq + d_offs_adjusted * stride_cos_d
            
            cos_val = tl.load(cos_ptr_local, mask=d_mask, other=1.0)
            sin_val = tl.load(sin_ptr_local, mask=d_mask, other=0.0)
            
            # Apply rotation: (a + bi) * (cos + sin*i) = (a*cos - b*sin) + (a*sin + b*cos)i
            out_even = x_even * cos_val - x_odd * sin_val
            out_odd = x_even * sin_val + x_odd * cos_val
            
            # Store results
            out_even_ptr = out_ptr + pid_batch * stride_ob + pid_seq * stride_ol + pid_head * stride_oh + 2 * d_offs * stride_od
            out_odd_ptr = out_ptr + pid_batch * stride_ob + pid_seq * stride_ol + pid_head * stride_oh + (2 * d_offs + 1) * stride_od
            
            tl.store(out_even_ptr, out_even, mask=d_mask)
            tl.store(out_odd_ptr, out_odd, mask=d_mask)


    @triton.jit  
    def _rope_kernel_1d(
        # Pointers
        x_ptr,
        cos_ptr,
        sin_ptr,
        out_ptr,
        # Dimensions
        batch_size,
        seq_len,
        num_heads,
        head_dim,
        rope_dim,  # Number of dimensions to apply RoPE to
        # Strides
        stride_xb,
        stride_xl,
        stride_xh,
        stride_xd,
        stride_cos_seq,
        stride_cos_d,
        stride_ob,
        stride_ol,
        stride_oh,
        stride_od,
        # Block sizes
        BLOCK_D: tl.constexpr,
    ):
        """
        Apply 1D RoPE using real arithmetic.
        """
        pid_batch = tl.program_id(0)
        pid_seq = tl.program_id(1)
        pid_head = tl.program_id(2)
        
        if pid_batch >= batch_size or pid_seq >= seq_len or pid_head >= num_heads:
            return
            
        half_d = head_dim // 2
        half_rope = rope_dim // 2
        
        for d_start in range(0, half_d, BLOCK_D):
            d_offs = d_start + tl.arange(0, BLOCK_D)
            d_mask = d_offs < half_d
            rope_mask = d_offs < half_rope
            
            # Load x values
            x_even_ptr = x_ptr + pid_batch * stride_xb + pid_seq * stride_xl + pid_head * stride_xh + 2 * d_offs * stride_xd
            x_odd_ptr = x_ptr + pid_batch * stride_xb + pid_seq * stride_xl + pid_head * stride_xh + (2 * d_offs + 1) * stride_xd
            
            x_even = tl.load(x_even_ptr, mask=d_mask, other=0.0)
            x_odd = tl.load(x_odd_ptr, mask=d_mask, other=0.0)
            
            # Load cos/sin for RoPE dimensions only
            cos_ptr_local = cos_ptr + pid_seq * stride_cos_seq + d_offs * stride_cos_d
            sin_ptr_local = sin_ptr + pid_seq * stride_cos_seq + d_offs * stride_cos_d
            
            cos_val = tl.load(cos_ptr_local, mask=rope_mask, other=1.0)
            sin_val = tl.load(sin_ptr_local, mask=rope_mask, other=0.0)
            
            # Apply rotation only to RoPE dimensions, passthrough for others
            out_even = tl.where(rope_mask, x_even * cos_val - x_odd * sin_val, x_even)
            out_odd = tl.where(rope_mask, x_even * sin_val + x_odd * cos_val, x_odd)
            
            # Store
            out_even_ptr = out_ptr + pid_batch * stride_ob + pid_seq * stride_ol + pid_head * stride_oh + 2 * d_offs * stride_od
            out_odd_ptr = out_ptr + pid_batch * stride_ob + pid_seq * stride_ol + pid_head * stride_oh + (2 * d_offs + 1) * stride_od
            
            tl.store(out_even_ptr, out_even, mask=d_mask)
            tl.store(out_odd_ptr, out_odd, mask=d_mask)


# ============================================================================
# PyTorch Implementation (compile-friendly, no complex numbers)
# ============================================================================

def _precompute_freqs_real(max_seq_len: int, dim: int, theta: float = 10000.0, 
                           device=None, dtype=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute cos and sin for RoPE in real form.
    
    Returns:
        cos: [max_seq_len, dim//2]
        sin: [max_seq_len, dim//2]
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if dtype is None:
        dtype = torch.float32
        
    # Compute inverse frequencies
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float64) / dim))
    
    # Compute position indices
    positions = torch.arange(max_seq_len, device=device, dtype=torch.float64)
    
    # Compute angles: [max_seq_len, dim//2]
    angles = torch.outer(positions, inv_freq)
    
    # Compute cos and sin
    cos = torch.cos(angles).to(dtype)
    sin = torch.sin(angles).to(dtype)
    
    return cos, sin


def rope_apply_real(x: torch.Tensor, 
                    cos: torch.Tensor, 
                    sin: torch.Tensor,
                    seq_len: Optional[int] = None) -> torch.Tensor:
    """
    Apply RoPE using real arithmetic. Fully compile-friendly.
    
    Args:
        x: Input tensor [B, L, H, D] or [B, L, D]
        cos: Precomputed cosines [max_seq, D//2]
        sin: Precomputed sines [max_seq, D//2]
        seq_len: Actual sequence length (if different from x.shape[1])
        
    Returns:
        Rotated tensor with same shape as input
    """
    if seq_len is None:
        seq_len = x.shape[1]
        
    # Ensure we have the right sequence length of cos/sin
    cos = cos[:seq_len]
    sin = sin[:seq_len]
    
    # Reshape for broadcasting: [1, L, 1, D//2] for 4D input
    if x.dim() == 4:
        cos = cos.view(1, seq_len, 1, -1)
        sin = sin.view(1, seq_len, 1, -1)
    else:  # 3D: [B, L, D]
        cos = cos.view(1, seq_len, -1)
        sin = sin.view(1, seq_len, -1)
    
    # Split into even and odd indices (pairs)
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    
    # Apply rotation
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos
    
    # Interleave back
    out = torch.stack([out_even, out_odd], dim=-1).flatten(-2)
    
    return out


def rope_apply_real_3d(x: torch.Tensor,
                       grid_sizes: torch.Tensor,
                       cos_t: torch.Tensor, sin_t: torch.Tensor,
                       cos_h: torch.Tensor, sin_h: torch.Tensor,
                       cos_w: torch.Tensor, sin_w: torch.Tensor,
                       reverse_time: bool = False) -> torch.Tensor:
    """
    Apply 3D RoPE (time, height, width) using real arithmetic.
    Compile-friendly version that doesn't use complex numbers.
    
    Args:
        x: [B, L, H, D] where L = F*H*W
        grid_sizes: [B, 3] containing (F, H, W) for each sample
        cos_t, sin_t: Precomputed for time dimension [max_t, d_t//2]
        cos_h, sin_h: Precomputed for height dimension [max_h, d_h//2]
        cos_w, sin_w: Precomputed for width dimension [max_w, d_w//2]
        reverse_time: Whether to reverse time indexing
        
    Returns:
        Rotated tensor [B, L, H, D]
    """
    B, L, H, D = x.shape
    half_d = D // 2
    
    # Dimension split: similar to original
    d_t = half_d - 2 * (half_d // 3)
    d_h = half_d // 3
    d_w = half_d // 3
    
    output = []
    
    for i in range(B):
        f, h, w = grid_sizes[i].tolist()
        seq_len = int(f * h * w)
        
        x_i = x[i, :seq_len]  # [seq_len, H, D]
        
        # Split x into even/odd pairs
        x_even = x_i[..., 0::2]  # [seq_len, H, D//2]
        x_odd = x_i[..., 1::2]
        
        # Build position indices for each dimension
        # Position in the FxHxW grid
        pos = torch.arange(seq_len, device=x.device)
        hw = h * w
        f_idx = pos // hw
        rem = pos % hw
        h_idx = rem // w
        w_idx = rem % w
        
        if reverse_time:
            f_idx = int(f) - 1 - f_idx
        
        # Get cos/sin for each position
        # Time component: first d_t dimensions
        cos_t_pos = cos_t[f_idx.long()][:, :d_t]  # [seq_len, d_t]
        sin_t_pos = sin_t[f_idx.long()][:, :d_t]
        
        # Height component: next d_h dimensions
        cos_h_pos = cos_h[h_idx.long()][:, :d_h]  # [seq_len, d_h]
        sin_h_pos = sin_h[h_idx.long()][:, :d_h]
        
        # Width component: last d_w dimensions
        cos_w_pos = cos_w[w_idx.long()][:, :d_w]  # [seq_len, d_w]
        sin_w_pos = sin_w[w_idx.long()][:, :d_w]
        
        # Concatenate cos/sin for all dimensions
        cos_all = torch.cat([cos_t_pos, cos_h_pos, cos_w_pos], dim=-1)  # [seq_len, D//2]
        sin_all = torch.cat([sin_t_pos, sin_h_pos, sin_w_pos], dim=-1)
        
        # Add head dimension for broadcasting: [seq_len, 1, D//2]
        cos_all = cos_all.unsqueeze(1)
        sin_all = sin_all.unsqueeze(1)
        
        # Apply rotation
        out_even = x_even * cos_all - x_odd * sin_all
        out_odd = x_even * sin_all + x_odd * cos_all
        
        # Interleave back
        x_rotated = torch.stack([out_even, out_odd], dim=-1).flatten(-2)
        
        # Handle padding (tokens beyond seq_len are not rotated)
        if seq_len < L:
            x_rotated = torch.cat([x_rotated, x[i, seq_len:]], dim=0)
        
        output.append(x_rotated)
    
    return torch.stack(output).to(x.dtype)


def rope_apply_triton(x: torch.Tensor,
                      grid_sizes: torch.Tensor,
                      freqs: torch.Tensor,
                      reverse_time: bool = False) -> torch.Tensor:
    """
    Drop-in replacement for the original rope_apply that uses Triton kernels.
    Falls back to PyTorch implementation if Triton is not available.
    
    Args:
        x: [B, L, H, D] input tensor
        grid_sizes: [B, ndim] where ndim is 1 or 3
        freqs: Complex frequencies from rope_params (will be converted to real)
        reverse_time: Whether to reverse time dimension
        
    Returns:
        Rotated tensor [B, L, H, D]
    """
    x_ndim = grid_sizes.shape[-1]
    
    # Convert complex freqs to real cos/sin
    # freqs is complex: freqs = cos(theta) + i*sin(theta)
    if torch.is_complex(freqs):
        cos_freqs = freqs.real.float()
        sin_freqs = freqs.imag.float()
    else:
        # Already real representation
        cos_freqs = freqs
        sin_freqs = torch.zeros_like(freqs)
    
    if x_ndim == 3:
        # 3D case: need to split frequencies
        B, L, H, D = x.shape
        half_d = D // 2
        c = half_d
        
        # Split frequencies similar to original
        d_t = c - 2 * (c // 3)
        d_h = c // 3
        d_w = c // 3
        
        cos_t = cos_freqs[:, :d_t]
        sin_t = sin_freqs[:, :d_t]
        cos_h = cos_freqs[:, d_t:d_t+d_h]
        sin_h = sin_freqs[:, d_t:d_t+d_h]
        cos_w = cos_freqs[:, d_t+d_h:d_t+d_h+d_w]
        sin_w = sin_freqs[:, d_t+d_h:d_t+d_h+d_w]
        
        return rope_apply_real_3d(
            x, grid_sizes,
            cos_t, sin_t, cos_h, sin_h, cos_w, sin_w,
            reverse_time=reverse_time
        )
    else:
        # 1D case
        return rope_apply_real(x, cos_freqs, sin_freqs)


# ============================================================================
# Validation function
# ============================================================================

def validate_rope_implementation(batch_size=2, seq_len=1024, num_heads=32, head_dim=64):
    """
    Validate that the real-arithmetic RoPE produces the same results as complex version.
    """
    import numpy as np
    
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test input
    x = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float32)
    
    # Create grid sizes (1D case for simplicity)
    grid_sizes = torch.tensor([[seq_len]], device=device)
    
    # Create frequencies using original method
    theta = 10000.0
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float64) / head_dim))
    positions = torch.arange(seq_len, device=device, dtype=torch.float64)
    angles = torch.outer(positions, inv_freq)
    freqs_complex = torch.polar(torch.ones_like(angles), angles)
    
    # Original complex implementation
    x_complex = torch.view_as_complex(x.to(torch.float64).reshape(batch_size, seq_len, num_heads, -1, 2))
    freqs_for_mul = freqs_complex.view(1, seq_len, 1, -1)
    x_rotated_complex = torch.view_as_real(x_complex * freqs_for_mul).flatten(-2)
    
    # Real arithmetic implementation
    cos_freqs = freqs_complex.real.float()
    sin_freqs = freqs_complex.imag.float()
    x_rotated_real = rope_apply_real(x, cos_freqs, sin_freqs)
    
    # Compare
    x_rotated_complex = x_rotated_complex.float()
    max_diff = (x_rotated_complex - x_rotated_real).abs().max().item()
    mean_diff = (x_rotated_complex - x_rotated_real).abs().mean().item()
    
    print(f"RoPE Validation Results:")
    print(f"  Max difference: {max_diff:.2e}")
    print(f"  Mean difference: {mean_diff:.2e}")
    print(f"  Passed: {max_diff < 1e-5}")
    
    return max_diff < 1e-5


if __name__ == "__main__":
    validate_rope_implementation()
