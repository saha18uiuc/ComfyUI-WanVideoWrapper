"""Fused AdaLN (Adaptive Layer Normalization) kernel.

Combines LayerNorm (no affine) + shift + scale into a single operation,
eliminating intermediate tensor materialization and reducing kernel launch count.

Falls back to standard PyTorch ops if Triton is unavailable.
"""
import torch
import torch.nn.functional as F

_HAS_TRITON = False
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    pass


if _HAS_TRITON:
    @triton.jit
    def _fused_adaln_kernel(
        X_ptr, Shift_ptr, Scale_ptr, Out_ptr,
        stride_x_row,
        N: tl.constexpr,  # hidden dimension
        eps: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Fused: out = LayerNorm(x) * (1 + scale) + shift
        
        LayerNorm (no affine): (x - mean) / sqrt(var + eps)
        Each program instance handles one token (one row of the [M, N] input).
        """
        row = tl.program_id(0)
        
        # Pointer offsets for this row
        x_offset = row * stride_x_row
        col_offsets = tl.arange(0, BLOCK_N)
        mask = col_offsets < N
        
        # Load input row
        x = tl.load(X_ptr + x_offset + col_offsets, mask=mask, other=0.0).to(tl.float32)
        
        # LayerNorm: (x - mean) / sqrt(var + eps)
        # Note: masked elements are 0, divide by N (not BLOCK_N) for correct mean
        mean = tl.sum(x, axis=0) / N
        x_centered = x - mean
        # Zero out masked elements before variance computation to avoid
        # masked positions contributing (-mean)^2 to the variance
        x_centered = tl.where(mask, x_centered, 0.0)
        var = tl.sum(x_centered * x_centered, axis=0) / N
        rstd = 1.0 / tl.sqrt(var + eps)
        x_norm = x_centered * rstd
        
        # Load shift and scale (same for all tokens in this batch element)
        shift = tl.load(Shift_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(Scale_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
        
        # AdaLN modulation: x_norm * (1 + scale) + shift
        out = x_norm * (1.0 + scale) + shift
        
        # Store - Triton auto-casts fp32 to output tensor's dtype (bf16/fp16)
        tl.store(Out_ptr + x_offset + col_offsets, out, mask=mask)


def fused_adaln_layernorm(x, shift, scale, eps=1e-6):
    """Fused LayerNorm (no affine) + AdaLN modulation.
    
    Computes: LayerNorm(x) * (1 + scale) + shift
    
    This replaces the pattern:
        self.modulate(self.norm1(x), shift, scale)
    where norm1 is WanLayerNorm(elementwise_affine=False).
    
    Args:
        x: Input tensor [B, L, N]
        shift: Shift tensor broadcastable to [B, L, N]
        scale: Scale tensor broadcastable to [B, L, N]
        eps: Epsilon for LayerNorm
        
    Returns:
        Output tensor same shape as x
    """
    if not _HAS_TRITON or x.device.type != 'cuda':
        return _adaln_pytorch_fallback(x, shift, scale, eps)
    
    B, L, N = x.shape
    
    # Shift/scale must be broadcastable. Common shapes:
    # [1, 1, N], [B, 1, N], or [1, 6, N] (but only one chunk used)
    # We need them as flat [N] for the kernel (per-row broadcast)
    if shift.dim() == 3:
        if shift.shape[0] == 1 and shift.shape[1] == 1:
            shift_flat = shift.reshape(N).contiguous()
            scale_flat = scale.reshape(N).contiguous()
        elif shift.shape[1] == 1:
            # [B, 1, N] - need to handle B>1 case
            if B == 1:
                shift_flat = shift.reshape(N).contiguous()
                scale_flat = scale.reshape(N).contiguous()
            else:
                return _adaln_pytorch_fallback(x, shift, scale, eps)
        else:
            return _adaln_pytorch_fallback(x, shift, scale, eps)
    elif shift.dim() == 2:
        if shift.shape[0] == 1:
            shift_flat = shift.reshape(N).contiguous()
            scale_flat = scale.reshape(N).contiguous()
        else:
            return _adaln_pytorch_fallback(x, shift, scale, eps)
    else:
        return _adaln_pytorch_fallback(x, shift, scale, eps)
    
    # Flatten to 2D for kernel
    x_2d = x.reshape(-1, N)
    if not x_2d.is_contiguous():
        x_2d = x_2d.contiguous()
    
    num_rows = x_2d.shape[0]
    out = torch.empty_like(x_2d)
    
    # Choose block size as next power of 2 >= N, capped at 8192
    BLOCK_N = triton.next_power_of_2(N)
    if BLOCK_N > 8192:
        return _adaln_pytorch_fallback(x, shift, scale, eps)
    
    _fused_adaln_kernel[(num_rows,)](
        x_2d, shift_flat, scale_flat, out,
        stride_x_row=N,
        N=N, eps=eps, BLOCK_N=BLOCK_N,
    )
    
    return out.reshape(B, L, N)


def _adaln_pytorch_fallback(x, shift, scale, eps=1e-6):
    """Standard PyTorch implementation of LayerNorm + AdaLN modulation."""
    x_norm = F.layer_norm(x.float(), (x.shape[-1],), eps=eps)
    return (x_norm * (1 + scale.float()) + shift.float()).to(x.dtype)
