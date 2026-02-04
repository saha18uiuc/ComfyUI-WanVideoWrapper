"""
SMOOTHCACHE FOR WAN VIDEO DIFFUSION
Based on: SmoothCache: A Universal Inference Acceleration Technique for Diffusion Transformers
Paper: https://arxiv.org/abs/2411.10510
GitHub: https://github.com/roblox/smoothcache

This optimization exploits high cosine similarity between layer outputs across
consecutive diffusion timesteps. Instead of recomputing layers at each timestep,
we cache and reuse outputs when similarity is high.

EXACT when threshold is 1.0 (no caching), slight approximation when < 1.0.
For production with quality requirements, use threshold >= 0.995 for minimal error.
"""

import torch
from typing import Dict, List, Optional, Callable
import math


class SmoothCacheState:
    """
    Manages cached layer outputs across timesteps.
    
    For each layer, we track:
    - Previous output tensor
    - Whether the cache is valid for reuse
    - Cumulative error estimate
    """
    
    def __init__(self, num_layers: int, similarity_threshold: float = 0.995):
        """
        Args:
            num_layers: Number of transformer layers to cache
            similarity_threshold: Minimum cosine similarity to reuse cache (0.0-1.0)
                                 Higher = more conservative, lower error
                                 0.995 recommended for near-exact output
        """
        self.num_layers = num_layers
        self.similarity_threshold = similarity_threshold
        self.cache: Dict[int, torch.Tensor] = {}
        self.prev_input: Dict[int, torch.Tensor] = {}
        self.reuse_count: Dict[int, int] = {i: 0 for i in range(num_layers)}
        self.compute_count: Dict[int, int] = {i: 0 for i in range(num_layers)}
        self.current_timestep = -1
        
    def reset(self):
        """Clear all cached values."""
        self.cache.clear()
        self.prev_input.clear()
        for i in range(self.num_layers):
            self.reuse_count[i] = 0
            self.compute_count[i] = 0
        self.current_timestep = -1
    
    def update_timestep(self, timestep: int):
        """Called at the start of each diffusion timestep."""
        self.current_timestep = timestep
    
    def should_reuse(self, layer_idx: int, current_input: torch.Tensor) -> bool:
        """
        Check if we should reuse the cached output for this layer.
        
        Uses cosine similarity between current input and previous input
        to determine if the layer output would be similar enough to reuse.
        """
        if layer_idx not in self.cache or layer_idx not in self.prev_input:
            return False
        
        prev_input = self.prev_input[layer_idx]
        
        # Check shape compatibility
        if prev_input.shape != current_input.shape:
            return False
        
        # Compute cosine similarity (efficient implementation)
        # Flatten for similarity computation
        curr_flat = current_input.view(-1).float()
        prev_flat = prev_input.view(-1).float()
        
        # Cosine similarity = (a · b) / (||a|| * ||b||)
        dot_product = torch.dot(curr_flat, prev_flat)
        norm_curr = torch.norm(curr_flat)
        norm_prev = torch.norm(prev_flat)
        
        if norm_curr == 0 or norm_prev == 0:
            return False
        
        similarity = dot_product / (norm_curr * norm_prev)
        
        return similarity.item() >= self.similarity_threshold
    
    def get_cached(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Get cached output for a layer."""
        self.reuse_count[layer_idx] += 1
        return self.cache.get(layer_idx)
    
    def update_cache(self, layer_idx: int, input_tensor: torch.Tensor, output_tensor: torch.Tensor):
        """Update cache with new input/output pair."""
        self.prev_input[layer_idx] = input_tensor.detach().clone()
        self.cache[layer_idx] = output_tensor.detach().clone()
        self.compute_count[layer_idx] += 1
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        total_reuse = sum(self.reuse_count.values())
        total_compute = sum(self.compute_count.values())
        total_ops = total_reuse + total_compute
        
        return {
            "total_reuses": total_reuse,
            "total_computes": total_compute,
            "reuse_ratio": total_reuse / total_ops if total_ops > 0 else 0,
            "per_layer_reuse": dict(self.reuse_count),
            "per_layer_compute": dict(self.compute_count),
        }


class SmoothCacheHelper:
    """
    Helper class to apply SmoothCache to a Wan transformer model.
    
    Usage:
        cache_helper = SmoothCacheHelper(transformer, similarity_threshold=0.995)
        cache_helper.enable()
        
        # ... run inference ...
        
        cache_helper.disable()
        print(cache_helper.get_stats())
    """
    
    def __init__(
        self, 
        transformer,
        similarity_threshold: float = 0.995,
        cache_self_attn: bool = True,
        cache_cross_attn: bool = False,  # Usually less similar across timesteps
        cache_ffn: bool = True,
        verbose: bool = False,
    ):
        """
        Args:
            transformer: The Wan transformer model
            similarity_threshold: Minimum cosine similarity for cache reuse
            cache_self_attn: Whether to cache self-attention outputs
            cache_cross_attn: Whether to cache cross-attention outputs  
            cache_ffn: Whether to cache FFN outputs
            verbose: Print cache statistics during inference
        """
        self.transformer = transformer
        self.similarity_threshold = similarity_threshold
        self.cache_self_attn = cache_self_attn
        self.cache_cross_attn = cache_cross_attn
        self.cache_ffn = cache_ffn
        self.verbose = verbose
        
        self.num_layers = len(transformer.blocks) if hasattr(transformer, 'blocks') else 40
        
        # Separate cache states for different module types
        self.self_attn_cache = SmoothCacheState(self.num_layers, similarity_threshold)
        self.cross_attn_cache = SmoothCacheState(self.num_layers, similarity_threshold)
        self.ffn_cache = SmoothCacheState(self.num_layers, similarity_threshold)
        
        self._enabled = False
        self._original_forwards = {}
        
    def enable(self):
        """Enable caching by wrapping layer forward methods."""
        if self._enabled:
            return
            
        self._enabled = True
        self.self_attn_cache.reset()
        self.cross_attn_cache.reset()
        self.ffn_cache.reset()
        
        # Wrap each block's forward method
        if hasattr(self.transformer, 'blocks'):
            for idx, block in enumerate(self.transformer.blocks):
                self._wrap_block(idx, block)
    
    def disable(self):
        """Disable caching and restore original forward methods."""
        if not self._enabled:
            return
            
        self._enabled = False
        
        # Restore original forwards
        for (idx, module_type), original_forward in self._original_forwards.items():
            if hasattr(self.transformer, 'blocks'):
                block = self.transformer.blocks[idx]
                if module_type == 'self_attn' and hasattr(block, 'self_attn'):
                    block.self_attn.forward = original_forward
                elif module_type == 'cross_attn' and hasattr(block, 'cross_attn'):
                    block.cross_attn.forward = original_forward
                elif module_type == 'ffn' and hasattr(block, 'ffn'):
                    block.ffn.forward = original_forward
        
        self._original_forwards.clear()
        
        if self.verbose:
            stats = self.get_stats()
            print(f"[SmoothCache] Stats: {stats['total_reuses']} reuses, "
                  f"{stats['total_computes']} computes, "
                  f"{stats['reuse_ratio']*100:.1f}% reuse ratio")
    
    def _wrap_block(self, idx: int, block):
        """Wrap a single transformer block's submodules."""
        
        # Wrap self-attention
        if self.cache_self_attn and hasattr(block, 'self_attn'):
            self._original_forwards[(idx, 'self_attn')] = block.self_attn.forward
            block.self_attn.forward = self._make_cached_forward(
                block.self_attn.forward, 
                idx, 
                self.self_attn_cache,
                'self_attn'
            )
        
        # Wrap cross-attention
        if self.cache_cross_attn and hasattr(block, 'cross_attn'):
            self._original_forwards[(idx, 'cross_attn')] = block.cross_attn.forward
            block.cross_attn.forward = self._make_cached_forward(
                block.cross_attn.forward,
                idx,
                self.cross_attn_cache,
                'cross_attn'
            )
        
        # Wrap FFN
        if self.cache_ffn and hasattr(block, 'ffn'):
            self._original_forwards[(idx, 'ffn')] = block.ffn.forward
            block.ffn.forward = self._make_cached_forward(
                block.ffn.forward,
                idx,
                self.ffn_cache,
                'ffn'
            )
    
    def _make_cached_forward(
        self, 
        original_forward: Callable,
        layer_idx: int,
        cache_state: SmoothCacheState,
        module_type: str,
    ) -> Callable:
        """Create a cached version of a forward method."""
        
        def cached_forward(*args, **kwargs):
            # Get the main input (first positional arg)
            if len(args) > 0:
                main_input = args[0]
            elif 'x' in kwargs:
                main_input = kwargs['x']
            elif 'hidden_states' in kwargs:
                main_input = kwargs['hidden_states']
            else:
                # Can't determine input, fall back to original
                return original_forward(*args, **kwargs)
            
            # Check if we can reuse cached output
            if cache_state.should_reuse(layer_idx, main_input):
                cached_output = cache_state.get_cached(layer_idx)
                if cached_output is not None:
                    return cached_output
            
            # Compute fresh output
            output = original_forward(*args, **kwargs)
            
            # Update cache
            if isinstance(output, tuple):
                cache_state.update_cache(layer_idx, main_input, output[0])
            else:
                cache_state.update_cache(layer_idx, main_input, output)
            
            return output
        
        return cached_forward
    
    def update_timestep(self, timestep: int):
        """Call at the start of each diffusion timestep."""
        self.self_attn_cache.update_timestep(timestep)
        self.cross_attn_cache.update_timestep(timestep)
        self.ffn_cache.update_timestep(timestep)
    
    def get_stats(self) -> dict:
        """Get combined cache statistics."""
        self_attn_stats = self.self_attn_cache.get_stats()
        cross_attn_stats = self.cross_attn_cache.get_stats()
        ffn_stats = self.ffn_cache.get_stats()
        
        total_reuses = (
            self_attn_stats["total_reuses"] + 
            cross_attn_stats["total_reuses"] + 
            ffn_stats["total_reuses"]
        )
        total_computes = (
            self_attn_stats["total_computes"] + 
            cross_attn_stats["total_computes"] + 
            ffn_stats["total_computes"]
        )
        total_ops = total_reuses + total_computes
        
        return {
            "total_reuses": total_reuses,
            "total_computes": total_computes,
            "reuse_ratio": total_reuses / total_ops if total_ops > 0 else 0,
            "self_attn": self_attn_stats,
            "cross_attn": cross_attn_stats,
            "ffn": ffn_stats,
        }


def apply_smooth_cache(
    transformer,
    similarity_threshold: float = 0.995,
    verbose: bool = False,
) -> SmoothCacheHelper:
    """
    Apply SmoothCache to a Wan transformer model.
    
    Args:
        transformer: The Wan transformer model
        similarity_threshold: Minimum cosine similarity for cache reuse (0.995 recommended)
        verbose: Print cache statistics
    
    Returns:
        SmoothCacheHelper instance (call .enable() to activate)
    """
    helper = SmoothCacheHelper(
        transformer,
        similarity_threshold=similarity_threshold,
        cache_self_attn=True,
        cache_cross_attn=False,  # Cross-attn changes more across timesteps
        cache_ffn=True,
        verbose=verbose,
    )
    return helper


__all__ = ['SmoothCacheState', 'SmoothCacheHelper', 'apply_smooth_cache']
