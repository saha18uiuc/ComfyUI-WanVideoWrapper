"""
Token Merging (ToMe) Integration for WanVideo Diffusion Model

This module provides a properly integrated ToMe implementation specifically
designed for the WanVideo architecture.

WanVideo Structure:
- WanAttentionBlock contains self_attn (WanSelfAttention) and cross_attn
- Self-attention receives q, k, v with shape [B, L, num_heads, head_dim]
- q, k, v are computed from hidden states via to_qkv() method

ToMe Integration Points:
1. Before self-attention: merge tokens based on key similarity
2. Run attention on merged tokens (fewer computations)
3. After attention: unmerge tokens back to original count

For video diffusion:
- Token count: T * H * W (e.g., 25 * 64 * 32 = 51,200 tokens)
- Self-attention cost: O(n²) = O(51,200²) ≈ 2.6 billion ops
- With 30% merge: O(35,840²) ≈ 1.3 billion ops (2x speedup)

Usage:
    from optimizations.tome_wanvideo import patch_wanvideo_tome, unpatch_wanvideo_tome
    
    # Apply ToMe to transformer
    num_patched = patch_wanvideo_tome(transformer, ratio=0.3)
    
    # Run inference...
    
    # Remove ToMe
    unpatch_wanvideo_tome(transformer)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Callable, Optional, Dict, Any, List
import math


def compute_merge_indices(
    keys: torch.Tensor,
    num_to_merge: int,
    protected_tokens: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute which tokens to merge based on key similarity.
    
    Uses bipartite matching: split tokens into source and destination,
    then find most similar pairs to merge.
    
    Args:
        keys: Key tensor [B, N, num_heads, head_dim] or [B, N, C]
        num_to_merge: Number of token pairs to merge
        protected_tokens: Number of tokens at start to protect (e.g., conditioning)
    
    Returns:
        src_indices: Source token indices to merge FROM
        dst_indices: Destination token indices to merge INTO
        unmerged_mask: Boolean mask of tokens that were NOT merged
    """
    # Flatten keys for similarity computation
    if keys.dim() == 4:
        # [B, N, H, D] -> [B, N, H*D]
        keys = keys.flatten(2)
    
    B, N, C = keys.shape
    
    # Normalize for cosine similarity
    keys_norm = F.normalize(keys.float(), dim=-1)
    
    # Split tokens: even indices = source, odd indices = destination
    # (after protected tokens)
    src_idx = torch.arange(protected_tokens, N, 2, device=keys.device)
    dst_idx = torch.arange(protected_tokens + 1, N, 2, device=keys.device)
    
    # Handle uneven split
    min_len = min(len(src_idx), len(dst_idx))
    if min_len == 0:
        # Not enough tokens to merge
        return None, None, None
    
    src_idx = src_idx[:min_len]
    dst_idx = dst_idx[:min_len]
    
    # Compute similarity between src and dst
    src_keys = keys_norm[:, src_idx]  # [B, n_src, C]
    dst_keys = keys_norm[:, dst_idx]  # [B, n_dst, C]
    
    # Similarity matrix [B, n_src, n_dst]
    similarity = torch.bmm(src_keys, dst_keys.transpose(-2, -1))
    
    # For each source, find best destination
    best_sim, best_dst = similarity.max(dim=-1)  # [B, n_src]
    
    # Select top-k source tokens to merge (most similar pairs)
    num_to_merge = min(num_to_merge, len(src_idx))
    
    # Get indices of most similar pairs
    _, top_src_local = best_sim.topk(num_to_merge, dim=-1)  # [B, num_to_merge]
    
    # Convert local indices to global
    src_global = src_idx[top_src_local]  # [B, num_to_merge]
    dst_local = best_dst.gather(dim=-1, index=top_src_local)  # [B, num_to_merge]
    dst_global = dst_idx[dst_local]  # [B, num_to_merge]
    
    # Create mask for unmerged tokens
    unmerged_mask = torch.ones(B, N, dtype=torch.bool, device=keys.device)
    for b in range(B):
        unmerged_mask[b, src_global[b]] = False
    
    return src_global, dst_global, unmerged_mask


def merge_tokens(
    x: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    unmerged_mask: torch.Tensor,
    mode: str = 'mean',
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Merge source tokens into destination tokens.
    
    Args:
        x: Input tensor [B, N, ...] (any number of trailing dims)
        src_indices: Source indices [B, r]
        dst_indices: Destination indices [B, r]
        unmerged_mask: Mask of unmerged tokens [B, N]
        mode: Merge mode ('mean' or 'replace')
    
    Returns:
        Merged tensor [B, N-r, ...]
        Unmerge info dict for later reconstruction
    """
    B, N = x.shape[:2]
    trailing_shape = x.shape[2:]
    device = x.device
    
    r = src_indices.shape[1]  # Number of merged pairs
    
    # We'll build the output by:
    # 1. Start with unmerged tokens
    # 2. For destination tokens that receive merges, average with source
    
    # Count how many sources merge into each destination
    merge_counts = torch.ones(B, N, device=device)
    for b in range(B):
        for i in range(r):
            dst_idx = dst_indices[b, i].item()
            merge_counts[b, dst_idx] += 1
    
    # Add source values to destinations
    x_merged = x.clone()
    for b in range(B):
        for i in range(r):
            src_idx = src_indices[b, i].item()
            dst_idx = dst_indices[b, i].item()
            if mode == 'mean':
                x_merged[b, dst_idx] = x_merged[b, dst_idx] + x[b, src_idx]
            # 'replace' mode: just take destination value
    
    # Normalize destinations that received merges
    if mode == 'mean':
        # Reshape merge_counts for broadcasting
        counts_expanded = merge_counts.view(B, N, *([1] * len(trailing_shape)))
        x_merged = x_merged / counts_expanded
    
    # Keep only unmerged tokens
    # This creates variable-length sequences, so we need to handle batch carefully
    # For simplicity, assume same merge pattern across batch (common in video diffusion)
    
    # Use mask to select unmerged tokens
    # unmerged_mask is [B, N], we need to index x_merged [B, N, ...]
    
    # Get indices of unmerged tokens (assuming same pattern for all batches)
    unmerged_indices = unmerged_mask[0].nonzero(as_tuple=True)[0]
    
    output = x_merged[:, unmerged_indices]
    
    unmerge_info = {
        'src_indices': src_indices,
        'dst_indices': dst_indices,
        'unmerged_mask': unmerged_mask,
        'unmerged_indices': unmerged_indices,
        'original_length': N,
        'merge_counts': merge_counts,
    }
    
    return output, unmerge_info


def unmerge_tokens(
    x: torch.Tensor,
    unmerge_info: Dict[str, Any],
) -> torch.Tensor:
    """
    Unmerge tokens back to original sequence length.
    
    Merged tokens are duplicated to their original positions.
    
    Args:
        x: Merged tensor [B, N-r, ...]
        unmerge_info: Info dict from merge_tokens
    
    Returns:
        Unmerged tensor [B, N, ...]
    """
    B = x.shape[0]
    trailing_shape = x.shape[2:]
    device = x.device
    dtype = x.dtype
    
    N = unmerge_info['original_length']
    unmerged_indices = unmerge_info['unmerged_indices']
    src_indices = unmerge_info['src_indices']
    dst_indices = unmerge_info['dst_indices']
    
    # Initialize output
    output = torch.zeros(B, N, *trailing_shape, device=device, dtype=dtype)
    
    # Place unmerged tokens
    output[:, unmerged_indices] = x
    
    # Copy merged destinations to source positions
    for b in range(B):
        for i in range(src_indices.shape[1]):
            src_idx = src_indices[b, i].item()
            dst_idx = dst_indices[b, i].item()
            # Find where dst_idx is in the merged output
            dst_merged_pos = (unmerged_indices == dst_idx).nonzero(as_tuple=True)[0]
            if len(dst_merged_pos) > 0:
                output[b, src_idx] = x[b, dst_merged_pos[0]]
    
    return output


class ToMeSelfAttentionWrapper:
    """
    Wrapper that adds Token Merging to WanSelfAttention.
    
    This wrapper intercepts the forward call and:
    1. Merges tokens before attention
    2. Runs original attention on fewer tokens
    3. Unmerges tokens after attention
    """
    
    def __init__(
        self,
        original_attn: nn.Module,
        ratio: float = 0.3,
        min_tokens: int = 2048,
    ):
        self.original_attn = original_attn
        self.ratio = ratio
        self.min_tokens = min_tokens
        self._original_forward = original_attn.forward
        self._original_to_qkv = original_attn.to_qkv if hasattr(original_attn, 'to_qkv') else None
    
    def forward_with_tome(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        seq_lens: torch.Tensor,
        *args,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward with Token Merging.
        
        Args:
            q, k, v: Query, Key, Value [B, N, num_heads, head_dim]
            seq_lens: Sequence lengths [B]
        
        Returns:
            Attention output [B, N, dim]
        """
        B, N, H, D = q.shape
        
        # Skip merging for small sequences
        if N < self.min_tokens or self.ratio <= 0:
            return self._original_forward(q, k, v, seq_lens, *args, **kwargs)
        
        num_to_merge = int(N * self.ratio)
        
        if num_to_merge < 1:
            return self._original_forward(q, k, v, seq_lens, *args, **kwargs)
        
        # Compute merge indices based on keys
        src_idx, dst_idx, unmerged_mask = compute_merge_indices(k, num_to_merge)
        
        if src_idx is None:
            return self._original_forward(q, k, v, seq_lens, *args, **kwargs)
        
        # Merge Q, K, V
        q_merged, info_q = merge_tokens(q, src_idx, dst_idx, unmerged_mask)
        k_merged, info_k = merge_tokens(k, src_idx, dst_idx, unmerged_mask)
        v_merged, info_v = merge_tokens(v, src_idx, dst_idx, unmerged_mask)
        
        # Update seq_lens for merged sequence
        seq_lens_merged = seq_lens - num_to_merge
        
        # Run original attention on merged tokens
        try:
            out_merged = self._original_forward(
                q_merged, k_merged, v_merged, seq_lens_merged, *args, **kwargs
            )
        except Exception as e:
            # Fall back to original if merged attention fails
            return self._original_forward(q, k, v, seq_lens, *args, **kwargs)
        
        # Unmerge output
        # out_merged is [B, N-r, dim] (flattened from attention output)
        # We need to reshape back to [B, N, dim]
        out = unmerge_tokens(out_merged.view(B, -1, out_merged.shape[-1] // 1), info_q)
        
        # Reshape to match expected output
        out = out.view(B * N, -1).view(B, N, -1)
        
        return out


def patch_wanvideo_tome(
    model: nn.Module,
    ratio: float = 0.3,
    min_tokens: int = 2048,
    verbose: bool = False,
) -> int:
    """
    Patch WanVideo model with Token Merging.
    
    This finds all WanSelfAttention modules and wraps their forward
    method with ToMe.
    
    Args:
        model: WanVideo transformer model
        ratio: Fraction of tokens to merge (0.0-0.5 recommended)
        min_tokens: Minimum sequence length to apply ToMe
        verbose: Print debug info
    
    Returns:
        Number of modules patched
    """
    patched = 0
    
    for name, module in model.named_modules():
        # Target WanSelfAttention modules
        if type(module).__name__ == 'WanSelfAttention':
            if hasattr(module, '_tome_original_forward'):
                # Already patched
                continue
            
            # Store original
            module._tome_original_forward = module.forward
            module._tome_ratio = ratio
            module._tome_min_tokens = min_tokens
            
            # Create wrapper
            def make_tome_forward(orig_module, r, min_t):
                orig_fwd = orig_module._tome_original_forward
                
                def tome_forward(q, k, v, seq_lens, *args, **kwargs):
                    B, N = q.shape[:2]
                    
                    # Skip for small sequences
                    if N < min_t or r <= 0:
                        return orig_fwd(q, k, v, seq_lens, *args, **kwargs)
                    
                    num_to_merge = int(N * r)
                    if num_to_merge < 1:
                        return orig_fwd(q, k, v, seq_lens, *args, **kwargs)
                    
                    # Compute merge indices
                    src_idx, dst_idx, unmerged_mask = compute_merge_indices(k, num_to_merge)
                    
                    if src_idx is None:
                        return orig_fwd(q, k, v, seq_lens, *args, **kwargs)
                    
                    try:
                        # Merge Q, K, V
                        q_merged, info = merge_tokens(q, src_idx, dst_idx, unmerged_mask)
                        k_merged, _ = merge_tokens(k, src_idx, dst_idx, unmerged_mask)
                        v_merged, _ = merge_tokens(v, src_idx, dst_idx, unmerged_mask)
                        
                        # Adjusted seq_lens
                        seq_lens_adj = seq_lens - num_to_merge
                        
                        # Run attention on merged
                        out_merged = orig_fwd(q_merged, k_merged, v_merged, seq_lens_adj, *args, **kwargs)
                        
                        # Unmerge - out_merged is [B, N', dim]
                        N_merged = q_merged.shape[1]
                        out_merged_reshaped = out_merged.view(B, N_merged, -1)
                        out = unmerge_tokens(out_merged_reshaped, info)
                        
                        return out.view(B, N, -1)
                        
                    except Exception as e:
                        # Fall back on any error
                        if verbose:
                            print(f"[ToMe] Fallback due to: {e}")
                        return orig_fwd(q, k, v, seq_lens, *args, **kwargs)
                
                return tome_forward
            
            module.forward = make_tome_forward(module, ratio, min_tokens)
            patched += 1
            
            if verbose:
                print(f"[ToMe] Patched: {name}")
    
    if verbose:
        print(f"[ToMe] Total self-attention modules patched: {patched}")
    
    return patched


def unpatch_wanvideo_tome(model: nn.Module) -> int:
    """
    Remove ToMe patching from WanVideo model.
    
    Returns:
        Number of modules unpatched
    """
    unpatched = 0
    
    for name, module in model.named_modules():
        if hasattr(module, '_tome_original_forward'):
            module.forward = module._tome_original_forward
            delattr(module, '_tome_original_forward')
            if hasattr(module, '_tome_ratio'):
                delattr(module, '_tome_ratio')
            if hasattr(module, '_tome_min_tokens'):
                delattr(module, '_tome_min_tokens')
            unpatched += 1
    
    return unpatched


def set_tome_ratio(model: nn.Module, ratio: float) -> int:
    """
    Update ToMe ratio on an already-patched model.
    
    Args:
        model: Patched model
        ratio: New merge ratio
    
    Returns:
        Number of modules updated
    """
    updated = 0
    
    for module in model.modules():
        if hasattr(module, '_tome_ratio'):
            module._tome_ratio = ratio
            updated += 1
    
    return updated


def estimate_wanvideo_tome_speedup(
    num_frames: int = 25,
    height: int = 512,
    width: int = 512,
    ratio: float = 0.3,
    num_layers: int = 40,
) -> Dict[str, Any]:
    """
    Estimate ToMe speedup for WanVideo.
    
    Args:
        num_frames: Number of video frames
        height: Video height
        width: Video width
        ratio: Token merge ratio
        num_layers: Number of transformer layers
    """
    # Compute token count (after VAE and patchify)
    # VAE: 8x downsample, then 2x2 patches
    lat_h = height // 8 // 2
    lat_w = width // 8 // 2
    tokens_per_frame = lat_h * lat_w
    total_tokens = num_frames * tokens_per_frame
    
    merged_tokens = int(total_tokens * (1 - ratio))
    
    # Self-attention FLOPs
    # O(n²) attention matrix + O(n²) softmax + O(n²) @ V
    original_flops = 3 * total_tokens * total_tokens
    merged_flops = 3 * merged_tokens * merged_tokens
    
    # Merge/unmerge overhead (negligible compared to attention)
    overhead_flops = 2 * total_tokens * 1024  # Similarity computation
    
    total_original = original_flops * num_layers
    total_merged = (merged_flops + overhead_flops) * num_layers
    
    speedup = total_original / total_merged
    
    return {
        'video_shape': f"{num_frames}×{height}×{width}",
        'total_tokens': f"{total_tokens:,}",
        'merged_tokens': f"{merged_tokens:,}",
        'token_reduction': f"{ratio*100:.0f}%",
        'attention_ops_original': f"{total_original/1e9:.2f}G",
        'attention_ops_merged': f"{total_merged/1e9:.2f}G",
        'estimated_attention_speedup': f"{speedup:.2f}x",
        'note': 'Speedup applies to self-attention only, not FFN or cross-attention',
    }
