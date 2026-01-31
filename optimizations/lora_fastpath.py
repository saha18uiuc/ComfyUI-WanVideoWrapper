"""
LoRA Fast-path Implementation

Instead of materializing a dense ΔW = A @ B every forward pass, we apply LoRA
as two skinny matrix multiplications on the activations:

    y = xW^T + α * (xB^T)A^T

This is mathematically equivalent but computationally more efficient when:
- The batch size (tokens) is large
- The rank r is small relative to input/output dimensions

Based on RunLoRA paper (arXiv:2312.03415) insights about optimal computation graphs.

The key insight is that LoRA's forward pass can be computed as:
    forward1: Y = (XW) + ((XA)B)  - Default, computes XA intermediate
    forward2: Y = X(W + AB)       - Pre-merge, forms full delta matrix

For inference with large token counts (video diffusion), forward1 with optimized
skinny GEMMs is typically faster than materializing the full AB product.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union


@dataclass
class PackedLoRA:
    """
    Packed LoRA parameters ready for efficient application.
    
    Attributes:
        down: Down projection [r, in_features], already on target device
        up: Up projection [out_features, r], already on target device  
        base_scale: Pre-computed scale = alpha / r
        strength: Strength value (float) or per-step schedule (list/tuple)
    """
    down: torch.Tensor  # [r, in]
    up: torch.Tensor    # [out, r]
    base_scale: float
    strength: Union[float, List[float], Tuple[float, ...]]
    
    def scale_for_step(self, step: int) -> float:
        """Get the scale factor for a given step (supports scheduled strengths)."""
        s = self.strength
        if isinstance(s, (list, tuple)):
            return float(s[min(step, len(s) - 1)])
        return float(s)


def _to_2d(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
    """Flatten tensor to 2D for GEMM, return original shape."""
    orig_shape = x.shape
    if x.ndim == 2:
        return x, orig_shape
    return x.reshape(-1, orig_shape[-1]), orig_shape


@torch.no_grad()
def pack_loras_for_layer(
    lora_list: List[Tuple],
    device: torch.device,
    dtype: torch.dtype
) -> List[PackedLoRA]:
    """
    Pack LoRA parameters for efficient application.
    
    Pre-moves tensors to device and pre-computes scales to avoid
    per-forward overhead.
    
    Args:
        lora_list: List of tuples (up, down, alpha, strength) matching
                   the repo's LoRA storage format
        device: Target device (cuda:0, etc.)
        dtype: Target dtype (float16, bfloat16)
    
    Returns:
        List of PackedLoRA objects ready for efficient application
    """
    packed = []
    for lora_tuple in lora_list:
        if len(lora_tuple) == 4:
            up, down, alpha, strength = lora_tuple
        elif len(lora_tuple) == 3:
            up, down, alpha = lora_tuple
            strength = 1.0
        else:
            # Handle single diff case
            continue
            
        # Flatten and move to device exactly once
        up2 = up.flatten(start_dim=1).to(device=device, dtype=dtype, non_blocking=True)
        dn2 = down.flatten(start_dim=1).to(device=device, dtype=dtype, non_blocking=True)
        
        r = dn2.shape[0]
        base_scale = float(alpha) / float(r) if alpha != 0.0 else 1.0 / float(r)
        
        packed.append(PackedLoRA(
            down=dn2,  # [r, in]
            up=up2,    # [out, r]
            base_scale=base_scale,
            strength=strength
        ))
    
    return packed


def lora_delta_lowrank(
    x: torch.Tensor,
    packed: List[PackedLoRA],
    step: int = 0
) -> torch.Tensor:
    """
    Compute LoRA delta using low-rank form: (x @ down.T) @ up.T
    
    This avoids materializing the full [out, in] delta matrix, instead
    computing two skinny GEMMs:
        1. tmp = x @ down.T  -> shape [..., r]
        2. delta = tmp @ up.T -> shape [..., out]
    
    Total FLOPs: 2 * tokens * (in*r + r*out) vs 2*in*out*r + 2*tokens*in*out
    The low-rank form wins when tokens >> 1 (which is always true in video diffusion)
    
    Args:
        x: Input activations [..., in_features]
        packed: List of PackedLoRA from pack_loras_for_layer
        step: Current denoising step (for scheduled strengths)
    
    Returns:
        LoRA delta to add to base output [..., out_features]
    """
    if not packed:
        return None
        
    x2, orig_shape = _to_2d(x)
    out = None
    
    for p in packed:
        scale = p.base_scale * p.scale_for_step(step)
        if scale == 0.0:
            continue
            
        # Two skinny GEMMs instead of one fat GEMM
        # tmp: [tokens, r] = [tokens, in] @ [in, r]
        tmp = x2 @ p.down.T
        # delta: [tokens, out] = [tokens, r] @ [r, out]
        delta = (tmp @ p.up.T) * scale
        
        if out is None:
            out = delta
        else:
            out = out + delta
    
    if out is None:
        # Return zeros if all scales were 0
        out_features = packed[0].up.shape[0]
        return x.new_zeros((*orig_shape[:-1], out_features))
    
    return out.reshape(*orig_shape[:-1], -1)


def apply_lora_residual_optimized(
    x: torch.Tensor,
    base_output: torch.Tensor,
    down: torch.Tensor,
    up: torch.Tensor,
    scale: float
) -> torch.Tensor:
    """
    Optimized LoRA residual application using fused operations.
    
    Computes: base_output + scale * ((x @ down.T) @ up.T)
    
    Uses torch.addmm when possible for fused multiply-add.
    
    Args:
        x: Input activations [batch, tokens, in_features]
        base_output: Output from base linear [batch, tokens, out_features]
        down: Down projection [r, in_features]
        up: Up projection [out_features, r]
        scale: Combined alpha/rank * strength scale
    
    Returns:
        base_output + LoRA delta
    """
    if scale == 0.0:
        return base_output
    
    x2, orig_shape = _to_2d(x)
    out2, _ = _to_2d(base_output)
    
    # Compute low-rank product
    tmp = x2 @ down.T  # [tokens, r]
    
    # Fused: out2 = 1*out2 + scale*(tmp @ up.T)
    # Note: addmm_ works on 2D tensors
    out2 = torch.addmm(out2, tmp, up.T, beta=1.0, alpha=scale)
    
    return out2.reshape(*orig_shape[:-1], -1)


class LoRAFastLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with optimized LoRA support.
    
    Instead of the default approach that computes:
        W' = W + alpha * (A @ B)
        y = x @ W'.T
    
    This computes:
        y = x @ W.T + alpha * ((x @ B.T) @ A.T)
    
    Which avoids materializing the full [out, in] delta matrix.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter('bias', None)
            
        self._packed_loras: Optional[List[PackedLoRA]] = None
        self._current_step = 0
        
    def set_packed_loras(self, packed: List[PackedLoRA]):
        """Set pre-packed LoRA parameters."""
        self._packed_loras = packed
        
    def set_step(self, step: int):
        """Set current denoising step for scheduled strengths."""
        self._current_step = step
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base linear
        y = nn.functional.linear(x, self.weight, self.bias)
        
        # Add LoRA delta if present
        if self._packed_loras:
            delta = lora_delta_lowrank(x, self._packed_loras, self._current_step)
            if delta is not None:
                y = y + delta
                
        return y


# Optimized batch LoRA application for multiple layers
@torch.no_grad()
def batch_pack_loras(
    patches: dict,
    module_to_key_map: dict,
    device: torch.device,
    dtype: torch.dtype
) -> dict:
    """
    Pack LoRA parameters for multiple layers in batch.
    
    This pre-processes all LoRA parameters once at model load time,
    avoiding per-forward overhead.
    
    Args:
        patches: Dict mapping keys to LoRA patch tuples
        module_to_key_map: Dict mapping module instances to patch keys
        device: Target device
        dtype: Target dtype
    
    Returns:
        Dict mapping modules to their packed LoRA lists
    """
    result = {}
    
    # Use CUDA streams for async transfers if available
    if device.type == 'cuda':
        stream = torch.cuda.Stream()
        ctx = torch.cuda.stream(stream)
    else:
        from contextlib import nullcontext
        ctx = nullcontext()
        stream = None
    
    with ctx:
        for module, key in module_to_key_map.items():
            patch = patches.get(key, [])
            if patch:
                lora_tuples = []
                for p in patch:
                    strength = p[0]
                    lora_obj = p[1]
                    
                    if hasattr(lora_obj, 'weights'):
                        weights = lora_obj.weights
                        if len(weights) >= 2:
                            up, down = weights[0], weights[1]
                            alpha = weights[2] if len(weights) > 2 else 0.0
                            lora_tuples.append((up, down, alpha, strength))
                    elif isinstance(lora_obj, tuple) and lora_obj[0] == "diff":
                        # Pre-computed diff, handle differently
                        continue
                        
                if lora_tuples:
                    result[module] = pack_loras_for_layer(lora_tuples, device, dtype)
    
    # Wait for all transfers
    if stream is not None:
        torch.cuda.current_stream().wait_stream(stream)
    
    return result
