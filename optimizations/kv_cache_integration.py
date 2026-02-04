"""
CROSS-ATTENTION KV CACHE INTEGRATION
=====================================

This module patches the WanI2VCrossAttention class to cache K/V projections
for static conditioning (text, image). This is a ZERO-OVERHEAD optimization
because:
1. The cache is a simple Python dict (no compilation)
2. Cache lookup is O(1) hash lookup
3. Cache storage happens naturally on first forward pass

Expected savings:
- 40 layers × 4 projections × (num_timesteps - 1) = massive matmul savings
- For 3-step inference: Skip 320 matrix multiplications per sample
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import logging

log = logging.getLogger("WanVideoWrapper")


class CrossAttnKVCacheManager:
    """
    Manages KV caching for cross-attention layers.
    
    Design principles:
    - Zero overhead: Just dict lookups
    - Automatic invalidation: Hash-based context change detection
    - Non-intrusive: Works via simple forward pre-hook
    """
    
    def __init__(self):
        self.enabled = False
        self.cache: Dict[int, Dict[str, torch.Tensor]] = {}
        self.context_hashes: Dict[str, int] = {}
        self.stats = {"hits": 0, "computes": 0}
        self._hooked_modules = []
    
    def enable(self, transformer):
        """Enable KV caching on transformer's cross-attention layers."""
        if self.enabled:
            return
        
        self.enabled = True
        self.cache.clear()
        self.context_hashes.clear()
        
        # Find and hook cross-attention layers
        if hasattr(transformer, 'blocks'):
            for idx, block in enumerate(transformer.blocks):
                if hasattr(block, 'cross_attn') and block.cross_attn is not None:
                    self._hook_cross_attn(idx, block.cross_attn)
        
        log.info(f"[KV Cache] Enabled on {len(self._hooked_modules)} cross-attn layers")
    
    def disable(self):
        """Disable KV caching and remove hooks."""
        if not self.enabled:
            return
        
        self.enabled = False
        
        # Remove hooks
        for hook in self._hooked_modules:
            hook.remove()
        self._hooked_modules.clear()
        
        # Log stats
        total = self.stats["hits"] + self.stats["computes"]
        if total > 0:
            log.info(f"[KV Cache] Disabled. Stats: {self.stats['hits']} hits, "
                    f"{self.stats['computes']} computes "
                    f"({self.stats['hits']/total*100:.1f}% hit rate)")
        
        self.cache.clear()
    
    def _hook_cross_attn(self, layer_idx: int, cross_attn: nn.Module):
        """Add forward pre-hook to cache KV computations."""
        
        # Store original forward
        original_forward = cross_attn.forward
        cache_mgr = self
        
        def cached_forward(x, context, **kwargs):
            # Skip caching if disabled
            if not cache_mgr.enabled:
                return original_forward(x, context, **kwargs)
            
            # Check if context changed (new sample or window)
            ctx_hash = hash((context.data_ptr(), context.shape))
            clip_embed = kwargs.get('clip_embed')
            clip_hash = hash((clip_embed.data_ptr(), clip_embed.shape)) if clip_embed is not None else 0
            combined_hash = hash((ctx_hash, clip_hash))
            
            cache_key = f"layer_{layer_idx}"
            
            # If context changed, invalidate cache for this layer
            if cache_mgr.context_hashes.get(cache_key) != combined_hash:
                cache_mgr.context_hashes[cache_key] = combined_hash
                if layer_idx in cache_mgr.cache:
                    del cache_mgr.cache[layer_idx]
            
            # Check cache
            if layer_idx in cache_mgr.cache:
                cache_mgr.stats["hits"] += 1
                cached = cache_mgr.cache[layer_idx]
                
                # Use cached K/V - patch the module temporarily
                return _forward_with_cached_kv(
                    cross_attn, x, context, 
                    cached.get('k'), cached.get('v'),
                    cached.get('k_img'), cached.get('v_img'),
                    **kwargs
                )
            
            # Compute and cache
            cache_mgr.stats["computes"] += 1
            
            # Compute K/V and cache them
            b, n, d = x.size(0), cross_attn.num_heads, cross_attn.head_dim
            
            # Text K/V
            k = cross_attn.norm_k(cross_attn.k(context).to(cross_attn.norm_k.weight.dtype)).view(b, -1, n, d).to(x.dtype)
            v = cross_attn.v(context).view(b, -1, n, d)
            
            # Image K/V (if available)
            k_img, v_img = None, None
            if clip_embed is not None and hasattr(cross_attn, 'k_img'):
                k_img = cross_attn.norm_k_img(cross_attn.k_img(clip_embed).to(cross_attn.norm_k_img.weight.dtype)).view(b, -1, n, d).to(x.dtype)
                v_img = cross_attn.v_img(clip_embed).view(b, -1, n, d)
            
            # Store in cache
            cache_mgr.cache[layer_idx] = {
                'k': k.detach(),
                'v': v.detach(),
                'k_img': k_img.detach() if k_img is not None else None,
                'v_img': v_img.detach() if v_img is not None else None,
            }
            
            # Continue with original forward using cached values
            return _forward_with_cached_kv(
                cross_attn, x, context, k, v, k_img, v_img, **kwargs
            )
        
        # Replace forward method
        cross_attn._original_forward = original_forward
        cross_attn.forward = cached_forward
        
        # Track for cleanup
        class HookHandle:
            def __init__(self, module, original):
                self.module = module
                self.original = original
            def remove(self):
                self.module.forward = self.original
        
        self._hooked_modules.append(HookHandle(cross_attn, original_forward))
    
    def clear_cache(self):
        """Clear all cached values (call when starting new sample)."""
        self.cache.clear()
        self.context_hashes.clear()


def _forward_with_cached_kv(cross_attn, x, context, k, v, k_img, v_img, **kwargs):
    """
    Forward pass using pre-computed K/V.
    
    This avoids recomputing:
    - k = norm_k(k_proj(context))
    - v = v_proj(context)
    - k_img = norm_k_img(k_img_proj(clip_embed))
    - v_img = v_img_proj(clip_embed)
    """
    from ..wanvideo.modules.attention import attention
    
    b, n, d = x.size(0), cross_attn.num_heads, cross_attn.head_dim
    rope_func = kwargs.get('rope_func', 'comfy')
    
    # Compute query (this always depends on x which changes)
    q = cross_attn.norm_q(
        cross_attn.q(x).to(cross_attn.norm_q.weight.dtype),
        num_chunks=2 if rope_func == "comfy_chunked" else 1
    ).view(b, -1, n, d).to(x.dtype)
    
    # Text attention with cached K/V
    nag_context = kwargs.get('nag_context')
    if nag_context is not None:
        nag_params = kwargs.get('nag_params', {})
        x_text = cross_attn.normalized_attention_guidance(b, n, d, q, context, nag_context, nag_params)
    else:
        x_text = attention(q, k, v, attention_mode=cross_attn.attention_mode, heads=cross_attn.num_heads).flatten(2)
    
    # Image attention with cached K/V
    clip_embed = kwargs.get('clip_embed')
    if clip_embed is not None and k_img is not None and v_img is not None:
        img_x = attention(q, k_img, v_img, attention_mode=cross_attn.attention_mode, heads=cross_attn.num_heads).flatten(2)
        x_text = x_text + img_x
    
    x = x_text
    
    # Audio attention (not cached - audio features change per frame)
    audio_proj = kwargs.get('audio_proj')
    audio_scale = kwargs.get('audio_scale', 1.0)
    num_latent_frames = kwargs.get('num_latent_frames', 21)
    
    if audio_proj is not None:
        if len(audio_proj.shape) == 4:
            audio_q = q.view(b * num_latent_frames, -1, n, d)
            ip_key = cross_attn.k_proj(audio_proj).view(b * num_latent_frames, -1, n, d)
            ip_value = cross_attn.v_proj(audio_proj).view(b * num_latent_frames, -1, n, d)
            audio_x = attention(audio_q, ip_key, ip_value, attention_mode=cross_attn.attention_mode, heads=cross_attn.num_heads)
            audio_x = audio_x.view(b, q.size(1), n, d).flatten(2)
        elif len(audio_proj.shape) == 3:
            ip_key = cross_attn.k_proj(audio_proj).view(b, -1, n, d)
            ip_value = cross_attn.v_proj(audio_proj).view(b, -1, n, d)
            audio_x = attention(q, ip_key, ip_value, attention_mode=cross_attn.attention_mode, heads=cross_attn.num_heads).flatten(2)
        x = x + audio_x * audio_scale
    
    # Adapter attention (not cached - may change)
    adapter_proj = kwargs.get('adapter_proj')
    ip_scale = kwargs.get('ip_scale', 1.0)
    orig_seq_len = kwargs.get('orig_seq_len')
    
    if adapter_proj is not None and hasattr(cross_attn, 'ip_adapter_single_stream_k_proj'):
        if len(adapter_proj.shape) == 4:
            adapter_q = q.view(b * num_latent_frames, -1, n, d)
            ip_key = cross_attn.ip_adapter_single_stream_k_proj(adapter_proj).view(b * num_latent_frames, -1, n, d)
            ip_value = cross_attn.ip_adapter_single_stream_v_proj(adapter_proj).view(b * num_latent_frames, -1, n, d)
            adapter_x = attention(adapter_q, ip_key, ip_value, attention_mode=cross_attn.attention_mode, heads=cross_attn.num_heads)
            adapter_x = adapter_x.view(b, q.size(1), n, d).flatten(2)
        elif len(adapter_proj.shape) == 3:
            ip_key = cross_attn.ip_adapter_single_stream_k_proj(adapter_proj).view(b, -1, n, d)
            ip_value = cross_attn.ip_adapter_single_stream_v_proj(adapter_proj).view(b, -1, n, d)
            adapter_x = attention(q, ip_key, ip_value, attention_mode=cross_attn.attention_mode, heads=cross_attn.num_heads).flatten(2)
        
        if orig_seq_len is not None:
            x[:, :orig_seq_len] = x[:, :orig_seq_len] + adapter_x * ip_scale
        else:
            x = x + adapter_x * ip_scale
    
    return cross_attn.o(x)


# Global cache manager instance
_kv_cache_manager: Optional[CrossAttnKVCacheManager] = None


def get_kv_cache_manager() -> Optional[CrossAttnKVCacheManager]:
    """Get global KV cache manager."""
    return _kv_cache_manager


def enable_kv_cache(transformer) -> CrossAttnKVCacheManager:
    """Enable KV caching on a transformer."""
    global _kv_cache_manager
    if _kv_cache_manager is None:
        _kv_cache_manager = CrossAttnKVCacheManager()
    _kv_cache_manager.enable(transformer)
    return _kv_cache_manager


def disable_kv_cache():
    """Disable KV caching."""
    global _kv_cache_manager
    if _kv_cache_manager is not None:
        _kv_cache_manager.disable()


__all__ = [
    'CrossAttnKVCacheManager',
    'get_kv_cache_manager',
    'enable_kv_cache',
    'disable_kv_cache',
]
