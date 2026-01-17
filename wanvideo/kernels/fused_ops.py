"""
Fused Operations for WanVideo Transformer Blocks.

This module implements fused Triton kernels for common operations that are
typically executed as separate kernels, causing launch overhead.

Research Basis:
1. Apex FusedLayerNorm - NVIDIA's fused layer normalization
   https://github.com/NVIDIA/apex/tree/master/apex/normalization
   
2. FlashAttention-2 (Dao, 2023) - Fused attention with online softmax
   https://arxiv.org/abs/2307.08691
   
3. Megatron-LM - Fused bias/gelu operations
   https://github.com/NVIDIA/Megatron-LM
   
4. xFormers - Memory-efficient attention implementations
   https://github.com/facebookresearch/xformers

Key Optimizations:
- Fused LayerNorm + Modulation (DiT-style timestep conditioning)
- Fused QKV Projection + RoPE application
- Fused SwiGLU activation for FFN
- All kernels use Triton autotuning for optimal performance
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
# Fused LayerNorm + Modulation (DiT-style)
# ============================================================================
# 
# In Diffusion Transformers (DiT), timestep conditioning is applied via:
#   x_mod = (1 + scale) * LayerNorm(x) + shift
# 
# This is typically 3 separate operations. Fusing them reduces memory traffic.
#
# Reference: DiT paper (Peebles & Xie, 2022) - "Scalable Diffusion Models with Transformers"
# ============================================================================

if _HAS_TRITON:
    
    @triton.jit
    def _layernorm_modulation_kernel(
        # Pointers
        x_ptr,           # Input: [batch, seq_len, hidden]
        weight_ptr,      # LayerNorm weight: [hidden]
        scale_ptr,       # Modulation scale: [batch, hidden] or [batch, 1, hidden]
        shift_ptr,       # Modulation shift: [batch, hidden] or [batch, 1, hidden]
        out_ptr,         # Output: [batch, seq_len, hidden]
        # Dimensions
        batch_size,
        seq_len,
        hidden_dim,
        # Strides
        stride_xb,
        stride_xs,
        stride_xh,
        stride_ob,
        stride_os,
        stride_oh,
        # LN params
        eps,
        # Block size
        BLOCK_H: tl.constexpr,
    ):
        """
        Fused LayerNorm + Modulation kernel.
        
        Computes: out = (1 + scale) * LayerNorm(x, weight) + shift
        
        This is the core operation in DiT-style diffusion transformers where
        timestep embeddings modulate the hidden states.
        """
        pid_b = tl.program_id(0)  # batch index
        pid_s = tl.program_id(1)  # sequence index
        
        if pid_b >= batch_size or pid_s >= seq_len:
            return
        
        # Calculate offsets
        x_offset = pid_b * stride_xb + pid_s * stride_xs
        out_offset = pid_b * stride_ob + pid_s * stride_os
        
        # Load x for this position: [hidden_dim]
        h_offs = tl.arange(0, BLOCK_H)
        h_mask = h_offs < hidden_dim
        
        x = tl.load(x_ptr + x_offset + h_offs * stride_xh, mask=h_mask, other=0.0)
        
        # Compute mean
        mean = tl.sum(x, axis=0) / hidden_dim
        
        # Compute variance
        x_centered = x - mean
        var = tl.sum(x_centered * x_centered, axis=0) / hidden_dim
        
        # Normalize
        rstd = 1.0 / tl.sqrt(var + eps)
        x_norm = x_centered * rstd
        
        # Load LayerNorm weight
        weight = tl.load(weight_ptr + h_offs, mask=h_mask, other=1.0)
        x_norm = x_norm * weight
        
        # Load modulation parameters (broadcast across sequence)
        # scale and shift are [batch, hidden] - same for all seq positions
        mod_offset = pid_b * hidden_dim
        scale = tl.load(scale_ptr + mod_offset + h_offs, mask=h_mask, other=0.0)
        shift = tl.load(shift_ptr + mod_offset + h_offs, mask=h_mask, other=0.0)
        
        # Apply modulation: (1 + scale) * x_norm + shift
        out = (1.0 + scale) * x_norm + shift
        
        # Store output
        tl.store(out_ptr + out_offset + h_offs * stride_oh, out, mask=h_mask)


    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_H': 1024}, num_warps=4),
            triton.Config({'BLOCK_H': 2048}, num_warps=8),
            triton.Config({'BLOCK_H': 4096}, num_warps=8),
        ],
        key=['hidden_dim'],
    )
    @triton.jit
    def _layernorm_modulation_kernel_autotuned(
        x_ptr, weight_ptr, scale_ptr, shift_ptr, out_ptr,
        batch_size, seq_len, hidden_dim,
        stride_xb, stride_xs, stride_xh,
        stride_ob, stride_os, stride_oh,
        eps,
        BLOCK_H: tl.constexpr,
    ):
        """Autotuned version of the fused LayerNorm + Modulation kernel."""
        _layernorm_modulation_kernel(
            x_ptr, weight_ptr, scale_ptr, shift_ptr, out_ptr,
            batch_size, seq_len, hidden_dim,
            stride_xb, stride_xs, stride_xh,
            stride_ob, stride_os, stride_oh,
            eps,
            BLOCK_H,
        )


# ============================================================================
# Fused SwiGLU Activation
# ============================================================================
#
# SwiGLU is used in LLaMA, PaLM, and many modern transformers:
#   SwiGLU(x, W1, W2, W3) = (Swish(x @ W1) * (x @ W2)) @ W3
#
# The activation Swish(x) = x * sigmoid(x) can be fused with the gating.
#
# Reference: "GLU Variants Improve Transformer" (Shazeer, 2020)
# ============================================================================

if _HAS_TRITON:
    
    @triton.jit
    def _swiglu_kernel(
        # Pointers
        gate_ptr,     # Gate input (after W1): [batch, seq, ffn_dim]
        up_ptr,       # Up projection (after W2): [batch, seq, ffn_dim]
        out_ptr,      # Output: [batch, seq, ffn_dim]
        # Dimensions
        n_elements,
        # Block size
        BLOCK: tl.constexpr,
    ):
        """
        Fused SwiGLU activation: Swish(gate) * up
        
        Swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
        """
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        
        # Load inputs
        gate = tl.load(gate_ptr + offs, mask=mask, other=0.0)
        up = tl.load(up_ptr + offs, mask=mask, other=0.0)
        
        # Compute Swish activation: x * sigmoid(x)
        # sigmoid(x) = 1 / (1 + exp(-x))
        gate_swish = gate * tl.sigmoid(gate)
        
        # Compute gated output
        out = gate_swish * up
        
        # Store
        tl.store(out_ptr + offs, out, mask=mask)


    @triton.jit
    def _swiglu_backward_kernel(
        # Pointers
        gate_ptr,
        up_ptr,
        grad_out_ptr,
        grad_gate_ptr,
        grad_up_ptr,
        # Dimensions
        n_elements,
        BLOCK: tl.constexpr,
    ):
        """
        Backward pass for SwiGLU.
        
        d_gate = d_out * up * (sigmoid(gate) + gate * sigmoid(gate) * (1 - sigmoid(gate)))
        d_up = d_out * swish(gate)
        """
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        
        gate = tl.load(gate_ptr + offs, mask=mask, other=0.0)
        up = tl.load(up_ptr + offs, mask=mask, other=0.0)
        grad_out = tl.load(grad_out_ptr + offs, mask=mask, other=0.0)
        
        sig = tl.sigmoid(gate)
        swish = gate * sig
        
        # d_swish/d_gate = sigmoid(gate) + gate * sigmoid(gate) * (1 - sigmoid(gate))
        #                = sigmoid(gate) * (1 + gate * (1 - sigmoid(gate)))
        #                = sigmoid(gate) * (1 + gate - gate * sigmoid(gate))
        d_swish_d_gate = sig * (1.0 + gate * (1.0 - sig))
        
        grad_gate = grad_out * up * d_swish_d_gate
        grad_up = grad_out * swish
        
        tl.store(grad_gate_ptr + offs, grad_gate, mask=mask)
        tl.store(grad_up_ptr + offs, grad_up, mask=mask)


# ============================================================================
# Fused QKV Projection + RoPE
# ============================================================================
#
# Instead of:
#   1. Q = x @ W_q, K = x @ W_k, V = x @ W_v  (3 separate matmuls)
#   2. Q, K = apply_rope(Q), apply_rope(K)    (separate RoPE)
#
# We fuse into:
#   1. QKV = x @ W_qkv  (single matmul)
#   2. Apply RoPE to Q, K in the same kernel
#
# Reference: RoFormer (Su et al., 2021) - Rotary Position Embedding
# ============================================================================

if _HAS_TRITON:
    
    @triton.jit
    def _fused_qkv_rope_kernel(
        # Pointers
        qkv_ptr,        # Combined QKV: [batch, seq, 3 * num_heads * head_dim]
        cos_ptr,        # Cosines: [max_seq, head_dim // 2]
        sin_ptr,        # Sines: [max_seq, head_dim // 2]
        q_out_ptr,      # Output Q: [batch, seq, num_heads, head_dim]
        k_out_ptr,      # Output K: [batch, seq, num_heads, head_dim]
        v_out_ptr,      # Output V: [batch, seq, num_heads, head_dim]
        # Dimensions
        batch_size,
        seq_len,
        num_heads,
        head_dim,
        # Strides
        stride_qkv_b,
        stride_qkv_s,
        stride_qkv_d,
        stride_out_b,
        stride_out_s,
        stride_out_h,
        stride_out_d,
        stride_cos_s,
        stride_cos_d,
        # Block sizes
        BLOCK_H: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """
        Fused QKV split + RoPE application.
        
        Takes combined QKV projection output and:
        1. Splits into Q, K, V
        2. Applies RoPE rotation to Q and K
        3. Outputs in attention-ready format
        """
        pid_b = tl.program_id(0)
        pid_s = tl.program_id(1)
        pid_h = tl.program_id(2)
        
        if pid_b >= batch_size or pid_s >= seq_len or pid_h >= num_heads:
            return
        
        # Calculate QKV offsets
        # QKV layout: [batch, seq, 3 * num_heads * head_dim]
        # Q: offset 0, K: offset num_heads * head_dim, V: offset 2 * num_heads * head_dim
        total_head_dim = num_heads * head_dim
        q_start = pid_h * head_dim
        k_start = total_head_dim + pid_h * head_dim
        v_start = 2 * total_head_dim + pid_h * head_dim
        
        qkv_base = pid_b * stride_qkv_b + pid_s * stride_qkv_s
        
        # Process in blocks of head_dim / 2 (for RoPE pairs)
        half_d = head_dim // 2
        
        for d in range(0, half_d, BLOCK_D):
            d_offs = d + tl.arange(0, BLOCK_D)
            d_mask = d_offs < half_d
            
            # Load Q pairs (even and odd indices)
            q_even_idx = q_start + 2 * d_offs
            q_odd_idx = q_start + 2 * d_offs + 1
            q_even = tl.load(qkv_ptr + qkv_base + q_even_idx * stride_qkv_d, mask=d_mask, other=0.0)
            q_odd = tl.load(qkv_ptr + qkv_base + q_odd_idx * stride_qkv_d, mask=d_mask, other=0.0)
            
            # Load K pairs
            k_even_idx = k_start + 2 * d_offs
            k_odd_idx = k_start + 2 * d_offs + 1
            k_even = tl.load(qkv_ptr + qkv_base + k_even_idx * stride_qkv_d, mask=d_mask, other=0.0)
            k_odd = tl.load(qkv_ptr + qkv_base + k_odd_idx * stride_qkv_d, mask=d_mask, other=0.0)
            
            # Load V (no RoPE needed)
            v_even_idx = v_start + 2 * d_offs
            v_odd_idx = v_start + 2 * d_offs + 1
            v_even = tl.load(qkv_ptr + qkv_base + v_even_idx * stride_qkv_d, mask=d_mask, other=0.0)
            v_odd = tl.load(qkv_ptr + qkv_base + v_odd_idx * stride_qkv_d, mask=d_mask, other=0.0)
            
            # Load cos/sin for this position
            cos = tl.load(cos_ptr + pid_s * stride_cos_s + d_offs * stride_cos_d, mask=d_mask, other=1.0)
            sin = tl.load(sin_ptr + pid_s * stride_cos_s + d_offs * stride_cos_d, mask=d_mask, other=0.0)
            
            # Apply RoPE rotation to Q
            q_even_rot = q_even * cos - q_odd * sin
            q_odd_rot = q_even * sin + q_odd * cos
            
            # Apply RoPE rotation to K
            k_even_rot = k_even * cos - k_odd * sin
            k_odd_rot = k_even * sin + k_odd * cos
            
            # Calculate output offsets
            out_base = pid_b * stride_out_b + pid_s * stride_out_s + pid_h * stride_out_h
            
            # Store Q
            tl.store(q_out_ptr + out_base + 2 * d_offs * stride_out_d, q_even_rot, mask=d_mask)
            tl.store(q_out_ptr + out_base + (2 * d_offs + 1) * stride_out_d, q_odd_rot, mask=d_mask)
            
            # Store K
            tl.store(k_out_ptr + out_base + 2 * d_offs * stride_out_d, k_even_rot, mask=d_mask)
            tl.store(k_out_ptr + out_base + (2 * d_offs + 1) * stride_out_d, k_odd_rot, mask=d_mask)
            
            # Store V (no rotation)
            tl.store(v_out_ptr + out_base + 2 * d_offs * stride_out_d, v_even, mask=d_mask)
            tl.store(v_out_ptr + out_base + (2 * d_offs + 1) * stride_out_d, v_odd, mask=d_mask)


# ============================================================================
# Python Interface Functions
# ============================================================================

def fused_layernorm_modulation(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Fused LayerNorm + Modulation operation.
    
    Computes: (1 + scale) * LayerNorm(x) + shift
    
    This is the core operation in DiT-style diffusion transformers.
    
    Args:
        x: Input tensor [batch, seq_len, hidden_dim]
        weight: LayerNorm weight [hidden_dim]
        scale: Modulation scale [batch, hidden_dim]
        shift: Modulation shift [batch, hidden_dim]
        eps: LayerNorm epsilon
        
    Returns:
        Modulated output [batch, seq_len, hidden_dim]
    """
    if not _HAS_TRITON or not x.is_cuda:
        # Fallback to PyTorch
        x_norm = torch.nn.functional.layer_norm(x, (x.shape[-1],), weight, None, eps)
        return (1 + scale.unsqueeze(1)) * x_norm + shift.unsqueeze(1)
    
    batch_size, seq_len, hidden_dim = x.shape
    out = torch.empty_like(x)
    
    # Ensure contiguous
    x = x.contiguous()
    scale = scale.contiguous()
    shift = shift.contiguous()
    
    # Grid: one block per (batch, seq) position
    grid = (batch_size, seq_len)
    
    # Determine block size based on hidden_dim
    BLOCK_H = triton.next_power_of_2(hidden_dim)
    BLOCK_H = min(BLOCK_H, 4096)  # Cap at 4096 for shared memory
    
    _layernorm_modulation_kernel[grid](
        x, weight, scale, shift, out,
        batch_size, seq_len, hidden_dim,
        x.stride(0), x.stride(1), x.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        eps,
        BLOCK_H=BLOCK_H,
    )
    
    return out


def fused_swiglu(
    gate: torch.Tensor,
    up: torch.Tensor,
) -> torch.Tensor:
    """
    Fused SwiGLU activation.
    
    Computes: Swish(gate) * up = (gate * sigmoid(gate)) * up
    
    Args:
        gate: Gate input [batch, seq, ffn_dim]
        up: Up projection input [batch, seq, ffn_dim]
        
    Returns:
        Activated output [batch, seq, ffn_dim]
    """
    if not _HAS_TRITON or not gate.is_cuda:
        # Fallback to PyTorch
        return torch.nn.functional.silu(gate) * up
    
    assert gate.shape == up.shape, "gate and up must have same shape"
    
    out = torch.empty_like(gate)
    n_elements = gate.numel()
    
    gate = gate.contiguous()
    up = up.contiguous()
    
    BLOCK = 1024
    grid = (triton.cdiv(n_elements, BLOCK),)
    
    _swiglu_kernel[grid](
        gate, up, out,
        n_elements,
        BLOCK=BLOCK,
    )
    
    return out


def fused_qkv_rope(
    qkv: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_heads: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused QKV split and RoPE application.
    
    Args:
        qkv: Combined QKV projection [batch, seq, 3 * num_heads * head_dim]
        cos: Cosines for RoPE [max_seq, head_dim // 2]
        sin: Sines for RoPE [max_seq, head_dim // 2]
        num_heads: Number of attention heads
        
    Returns:
        Tuple of (Q, K, V) each with shape [batch, seq, num_heads, head_dim]
    """
    batch_size, seq_len, qkv_dim = qkv.shape
    head_dim = qkv_dim // (3 * num_heads)
    
    if not _HAS_TRITON or not qkv.is_cuda:
        # Fallback to PyTorch
        qkv = qkv.reshape(batch_size, seq_len, 3, num_heads, head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        
        # Apply RoPE
        cos = cos[:seq_len].view(1, seq_len, 1, -1)
        sin = sin[:seq_len].view(1, seq_len, 1, -1)
        
        q_even, q_odd = q[..., 0::2], q[..., 1::2]
        k_even, k_odd = k[..., 0::2], k[..., 1::2]
        
        q_rot = torch.stack([
            q_even * cos - q_odd * sin,
            q_even * sin + q_odd * cos
        ], dim=-1).flatten(-2)
        
        k_rot = torch.stack([
            k_even * cos - k_odd * sin,
            k_even * sin + k_odd * cos
        ], dim=-1).flatten(-2)
        
        return q_rot, k_rot, v
    
    # Allocate outputs
    q = torch.empty(batch_size, seq_len, num_heads, head_dim, device=qkv.device, dtype=qkv.dtype)
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    
    qkv = qkv.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()
    
    BLOCK_H = 1
    BLOCK_D = min(32, head_dim // 2)
    
    grid = (batch_size, seq_len, num_heads)
    
    _fused_qkv_rope_kernel[grid](
        qkv, cos, sin, q, k, v,
        batch_size, seq_len, num_heads, head_dim,
        qkv.stride(0), qkv.stride(1), qkv.stride(2),
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        cos.stride(0), cos.stride(1),
        BLOCK_H=BLOCK_H, BLOCK_D=BLOCK_D,
    )
    
    return q, k, v


# ============================================================================
# Autograd Functions for Training Compatibility
# ============================================================================

class FusedSwiGLUFunction(torch.autograd.Function):
    """Autograd function for fused SwiGLU with custom backward."""
    
    @staticmethod
    def forward(ctx, gate, up):
        ctx.save_for_backward(gate, up)
        return fused_swiglu(gate, up)
    
    @staticmethod
    def backward(ctx, grad_output):
        gate, up = ctx.saved_tensors
        
        if not _HAS_TRITON or not gate.is_cuda:
            # Fallback
            sig = torch.sigmoid(gate)
            swish = gate * sig
            d_swish_d_gate = sig * (1 + gate * (1 - sig))
            grad_gate = grad_output * up * d_swish_d_gate
            grad_up = grad_output * swish
            return grad_gate, grad_up
        
        grad_gate = torch.empty_like(gate)
        grad_up = torch.empty_like(up)
        
        n_elements = gate.numel()
        BLOCK = 1024
        grid = (triton.cdiv(n_elements, BLOCK),)
        
        _swiglu_backward_kernel[grid](
            gate.contiguous(), up.contiguous(), grad_output.contiguous(),
            grad_gate, grad_up,
            n_elements,
            BLOCK=BLOCK,
        )
        
        return grad_gate, grad_up


# Convenience function that uses autograd
def fused_swiglu_autograd(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """SwiGLU with autograd support for training."""
    return FusedSwiGLUFunction.apply(gate, up)


# ============================================================================
# Validation and Benchmarking
# ============================================================================

def validate_fused_ops():
    """Validate all fused operations against PyTorch reference."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping validation")
        return False
    
    torch.manual_seed(42)
    device = torch.device('cuda')
    all_passed = True
    
    print("=" * 60)
    print("Validating Fused Operations")
    print("=" * 60)
    
    # Test 1: Fused LayerNorm + Modulation
    print("\n1. Testing Fused LayerNorm + Modulation...")
    batch, seq, hidden = 2, 1024, 2048
    x = torch.randn(batch, seq, hidden, device=device)
    weight = torch.randn(hidden, device=device)
    scale = torch.randn(batch, hidden, device=device) * 0.1
    shift = torch.randn(batch, hidden, device=device) * 0.1
    
    # Reference
    x_norm_ref = torch.nn.functional.layer_norm(x, (hidden,), weight, None, 1e-6)
    out_ref = (1 + scale.unsqueeze(1)) * x_norm_ref + shift.unsqueeze(1)
    
    # Fused
    out_fused = fused_layernorm_modulation(x, weight, scale, shift)
    
    max_diff = (out_ref - out_fused).abs().max().item()
    passed = max_diff < 1e-4
    print(f"   Max diff: {max_diff:.2e}, Passed: {passed}")
    all_passed = all_passed and passed
    
    # Test 2: Fused SwiGLU
    print("\n2. Testing Fused SwiGLU...")
    gate = torch.randn(batch, seq, hidden * 4, device=device)
    up = torch.randn(batch, seq, hidden * 4, device=device)
    
    # Reference
    out_ref = torch.nn.functional.silu(gate) * up
    
    # Fused
    out_fused = fused_swiglu(gate, up)
    
    max_diff = (out_ref - out_fused).abs().max().item()
    passed = max_diff < 1e-5
    print(f"   Max diff: {max_diff:.2e}, Passed: {passed}")
    all_passed = all_passed and passed
    
    # Test 3: Fused QKV + RoPE
    print("\n3. Testing Fused QKV + RoPE...")
    num_heads = 32
    head_dim = 64
    qkv_dim = 3 * num_heads * head_dim
    qkv = torch.randn(batch, seq, qkv_dim, device=device)
    
    # Create cos/sin
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(seq, device=device).float()
    angles = torch.outer(positions, inv_freq)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    
    # Reference
    qkv_ref = qkv.reshape(batch, seq, 3, num_heads, head_dim)
    q_ref, k_ref, v_ref = qkv_ref[:, :, 0], qkv_ref[:, :, 1], qkv_ref[:, :, 2]
    
    cos_ref = cos.view(1, seq, 1, -1)
    sin_ref = sin.view(1, seq, 1, -1)
    
    q_even, q_odd = q_ref[..., 0::2], q_ref[..., 1::2]
    q_rot_ref = torch.stack([q_even * cos_ref - q_odd * sin_ref, q_even * sin_ref + q_odd * cos_ref], dim=-1).flatten(-2)
    
    k_even, k_odd = k_ref[..., 0::2], k_ref[..., 1::2]
    k_rot_ref = torch.stack([k_even * cos_ref - k_odd * sin_ref, k_even * sin_ref + k_odd * cos_ref], dim=-1).flatten(-2)
    
    # Fused
    q_fused, k_fused, v_fused = fused_qkv_rope(qkv, cos, sin, num_heads)
    
    q_diff = (q_rot_ref - q_fused).abs().max().item()
    k_diff = (k_rot_ref - k_fused).abs().max().item()
    v_diff = (v_ref - v_fused).abs().max().item()
    
    passed = q_diff < 1e-4 and k_diff < 1e-4 and v_diff < 1e-4
    print(f"   Q max diff: {q_diff:.2e}")
    print(f"   K max diff: {k_diff:.2e}")
    print(f"   V max diff: {v_diff:.2e}")
    print(f"   Passed: {passed}")
    all_passed = all_passed and passed
    
    print("\n" + "=" * 60)
    print(f"Overall: {'PASSED' if all_passed else 'FAILED'}")
    print("=" * 60)
    
    return all_passed


def benchmark_fused_ops():
    """Benchmark fused operations vs PyTorch reference."""
    if not torch.cuda.is_available() or not _HAS_TRITON:
        print("CUDA/Triton not available")
        return
    
    import time
    
    device = torch.device('cuda')
    torch.manual_seed(42)
    
    batch, seq, hidden = 2, 4096, 2048
    num_warmup = 10
    num_iters = 100
    
    print("=" * 60)
    print("Benchmarking Fused Operations")
    print(f"Config: batch={batch}, seq={seq}, hidden={hidden}")
    print("=" * 60)
    
    # Benchmark LayerNorm + Modulation
    print("\n1. LayerNorm + Modulation")
    x = torch.randn(batch, seq, hidden, device=device)
    weight = torch.randn(hidden, device=device)
    scale = torch.randn(batch, hidden, device=device) * 0.1
    shift = torch.randn(batch, hidden, device=device) * 0.1
    
    # Warmup
    for _ in range(num_warmup):
        _ = fused_layernorm_modulation(x, weight, scale, shift)
    torch.cuda.synchronize()
    
    # Fused
    start = time.perf_counter()
    for _ in range(num_iters):
        _ = fused_layernorm_modulation(x, weight, scale, shift)
    torch.cuda.synchronize()
    fused_time = (time.perf_counter() - start) / num_iters * 1000
    
    # Reference
    ln = torch.nn.LayerNorm(hidden, device=device)
    ln.weight.data = weight
    
    for _ in range(num_warmup):
        x_norm = ln(x)
        _ = (1 + scale.unsqueeze(1)) * x_norm + shift.unsqueeze(1)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(num_iters):
        x_norm = ln(x)
        _ = (1 + scale.unsqueeze(1)) * x_norm + shift.unsqueeze(1)
    torch.cuda.synchronize()
    ref_time = (time.perf_counter() - start) / num_iters * 1000
    
    print(f"   Fused: {fused_time:.3f}ms, Reference: {ref_time:.3f}ms, Speedup: {ref_time/fused_time:.2f}x")
    
    # Benchmark SwiGLU
    print("\n2. SwiGLU")
    ffn_dim = hidden * 4
    gate = torch.randn(batch, seq, ffn_dim, device=device)
    up = torch.randn(batch, seq, ffn_dim, device=device)
    
    for _ in range(num_warmup):
        _ = fused_swiglu(gate, up)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(num_iters):
        _ = fused_swiglu(gate, up)
    torch.cuda.synchronize()
    fused_time = (time.perf_counter() - start) / num_iters * 1000
    
    for _ in range(num_warmup):
        _ = torch.nn.functional.silu(gate) * up
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(num_iters):
        _ = torch.nn.functional.silu(gate) * up
    torch.cuda.synchronize()
    ref_time = (time.perf_counter() - start) / num_iters * 1000
    
    print(f"   Fused: {fused_time:.3f}ms, Reference: {ref_time:.3f}ms, Speedup: {ref_time/fused_time:.2f}x")


if __name__ == "__main__":
    validate_fused_ops()
    print()
    benchmark_fused_ops()
