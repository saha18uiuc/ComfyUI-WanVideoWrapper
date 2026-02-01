"""
Optimization Integration for WanVideoWrapper

This module provides a unified interface for enabling and configuring
all math-heavy optimizations. It integrates:

1. LoRA fast-path (low-rank activation form)
2. Cross-attention K/V caching
3. Fused Triton kernels (RMSNorm, SwiGLU)
4. Model warmup
5. Attention backend selection

Usage:
    from optimizations.integration import OptimizationManager
    
    # Create manager with default settings
    opt_mgr = OptimizationManager()
    
    # Apply optimizations to model
    opt_mgr.optimize_model(transformer)
    
    # Enable specific optimizations
    opt_mgr.enable_lowrank_lora()
    opt_mgr.enable_kv_cache()
    
    # Get stats
    print(opt_mgr.get_stats())
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field
import time
from contextlib import contextmanager

# Import optimization modules
try:
    from .kv_cache import enable_crossattn_kv_cache, clear_kv_cache, get_kv_cache_stats
    HAS_KV_CACHE = True
except ImportError:
    HAS_KV_CACHE = False

try:
    from .kernels import HAS_RMSNORM, HAS_SWIGLU
    from .kernels.rmsnorm_triton import patch_rmsnorm_triton
    from .kernels.swiglu_triton import patch_swiglu_triton
except ImportError:
    HAS_RMSNORM = False
    HAS_SWIGLU = False
    patch_rmsnorm_triton = None
    patch_swiglu_triton = None

try:
    from .warmup import warmup_model, create_dummy_inputs
    HAS_WARMUP = True
except ImportError:
    HAS_WARMUP = False

try:
    from .tome_wanvideo import patch_wanvideo_tome, unpatch_wanvideo_tome, set_tome_ratio
    HAS_TOME = True
except ImportError:
    HAS_TOME = False
    patch_wanvideo_tome = None
    unpatch_wanvideo_tome = None
    set_tome_ratio = None


@dataclass
class OptimizationConfig:
    """Configuration for WanVideo optimizations."""
    
    # LoRA optimizations
    enable_lowrank_lora: bool = True  # Use low-rank activation form
    
    # K/V caching
    enable_kv_cache: bool = True  # Cache cross-attention K/V projections
    
    # Triton kernels
    enable_triton_rmsnorm: bool = True  # Fused RMSNorm
    enable_triton_swiglu: bool = True   # Fused SwiGLU
    
    # Token Merging (ToMe)
    enable_tome: bool = False  # Token merging for attention speedup (disabled by default - experimental)
    tome_ratio: float = 0.3   # Fraction of tokens to merge (0.0-0.5)
    tome_min_tokens: int = 2048  # Minimum tokens to apply ToMe
    
    # Warmup
    enable_warmup: bool = True  # Run warmup to eliminate compilation overhead
    warmup_runs: int = 2
    
    # Debug/stats
    verbose: bool = False
    collect_stats: bool = True


class OptimizationManager:
    """
    Central manager for WanVideo optimizations.
    
    Handles enabling/disabling optimizations and collecting statistics.
    """
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        self._stats: Dict[str, Any] = {
            'optimizations_applied': [],
            'patched_modules': 0,
            'warmup_time_ms': 0,
            'kv_cache_stats': {},
        }
        self._model_ref = None
    
    def optimize_model(
        self,
        model: nn.Module,
        warmup_shapes: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        """
        Apply all enabled optimizations to the model.
        
        Args:
            model: The diffusion transformer to optimize
            warmup_shapes: Optional dict with keys like 'batch_size', 'num_frames',
                          'height', 'width' for warmup
        
        Returns:
            Dict with optimization results and statistics
        """
        self._model_ref = model
        results = {'success': True, 'optimizations': []}
        
        # 1. Enable low-rank LoRA globally
        if self.config.enable_lowrank_lora:
            self._enable_lowrank_lora(model)
            results['optimizations'].append('lowrank_lora')
            if self.config.verbose:
                print("[Optimization] Enabled low-rank LoRA (activation-based)")
        
        # 2. Enable K/V caching for cross-attention
        if self.config.enable_kv_cache and HAS_KV_CACHE:
            patched = enable_crossattn_kv_cache(model, verbose=self.config.verbose)
            results['optimizations'].append(f'kv_cache ({patched} modules)')
            self._stats['patched_modules'] += patched
            if self.config.verbose:
                print(f"[Optimization] Enabled K/V cache for {patched} cross-attention modules")
        
        # 3. Patch RMSNorm with Triton kernel
        if self.config.enable_triton_rmsnorm and HAS_RMSNORM and patch_rmsnorm_triton:
            try:
                patched = patch_rmsnorm_triton(model)
                results['optimizations'].append(f'triton_rmsnorm ({patched} modules)')
                self._stats['patched_modules'] += patched
                if self.config.verbose:
                    print(f"[Optimization] Patched {patched} RMSNorm modules with Triton")
            except Exception as e:
                if self.config.verbose:
                    print(f"[Optimization] RMSNorm patch failed: {e}")
        
        # 4. Patch SwiGLU with Triton kernel
        if self.config.enable_triton_swiglu and HAS_SWIGLU and patch_swiglu_triton:
            try:
                patched = patch_swiglu_triton(model)
                results['optimizations'].append(f'triton_swiglu ({patched} modules)')
                self._stats['patched_modules'] += patched
                if self.config.verbose:
                    print(f"[Optimization] Patched {patched} SwiGLU modules with Triton")
            except Exception as e:
                if self.config.verbose:
                    print(f"[Optimization] SwiGLU patch failed: {e}")
        
        # 5. Token Merging (ToMe) for self-attention speedup
        if self.config.enable_tome and HAS_TOME and patch_wanvideo_tome:
            try:
                patched = patch_wanvideo_tome(
                    model,
                    ratio=self.config.tome_ratio,
                    min_tokens=self.config.tome_min_tokens,
                    verbose=self.config.verbose,
                )
                results['optimizations'].append(f'tome ({patched} modules, {self.config.tome_ratio*100:.0f}% merge)')
                self._stats['patched_modules'] += patched
                self._stats['tome_ratio'] = self.config.tome_ratio
                if self.config.verbose:
                    print(f"[Optimization] Enabled ToMe for {patched} self-attention modules")
            except Exception as e:
                if self.config.verbose:
                    print(f"[Optimization] ToMe patch failed: {e}")
        
        # 6. Model warmup
        if self.config.enable_warmup and HAS_WARMUP and warmup_shapes:
            try:
                start = time.perf_counter()
                device = next(model.parameters()).device
                dtype = next(model.parameters()).dtype
                
                dummy = create_dummy_inputs(
                    batch_size=warmup_shapes.get('batch_size', 1),
                    num_frames=warmup_shapes.get('num_frames', 21),
                    height=warmup_shapes.get('height', 512),
                    width=warmup_shapes.get('width', 512),
                    hidden_dim=warmup_shapes.get('hidden_dim', 3072),
                    context_len=warmup_shapes.get('context_len', 512),
                    device=device,
                    dtype=dtype,
                )
                
                warmup_stats = warmup_model(
                    model, dummy,
                    num_warmup_runs=self.config.warmup_runs,
                    verbose=self.config.verbose,
                    catch_errors=True,
                )
                
                self._stats['warmup_time_ms'] = (time.perf_counter() - start) * 1000
                results['optimizations'].append('warmup')
                results['warmup_stats'] = warmup_stats
                
            except Exception as e:
                if self.config.verbose:
                    print(f"[Optimization] Warmup failed: {e}")
        
        self._stats['optimizations_applied'] = results['optimizations']
        return results
    
    def _enable_lowrank_lora(self, model: nn.Module):
        """Enable low-rank LoRA for all CustomLinear modules."""
        from ..custom_linear import set_lora_optimization, CustomLinear
        
        # Set global flag
        set_lora_optimization(True)
        
        # Set per-module flag
        for module in model.modules():
            if isinstance(module, CustomLinear):
                module.use_lowrank_lora = True
    
    def disable_lowrank_lora(self, model: Optional[nn.Module] = None):
        """Disable low-rank LoRA optimization."""
        from ..custom_linear import set_lora_optimization, CustomLinear
        
        set_lora_optimization(False)
        
        model = model or self._model_ref
        if model:
            for module in model.modules():
                if isinstance(module, CustomLinear):
                    module.use_lowrank_lora = False
    
    def clear_caches(self, model: Optional[nn.Module] = None):
        """Clear all caches (K/V cache, etc.)."""
        model = model or self._model_ref
        if model and HAS_KV_CACHE:
            clear_kv_cache(model)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        stats = dict(self._stats)
        
        # Add K/V cache stats if available
        if self._model_ref and HAS_KV_CACHE:
            try:
                stats['kv_cache_stats'] = get_kv_cache_stats(self._model_ref)
            except:
                pass
        
        return stats
    
    def print_summary(self):
        """Print a summary of applied optimizations."""
        print("\n" + "="*60)
        print("WanVideo Optimization Summary")
        print("="*60)
        
        stats = self.get_stats()
        
        print(f"\nOptimizations applied: {len(stats['optimizations_applied'])}")
        for opt in stats['optimizations_applied']:
            print(f"  - {opt}")
        
        print(f"\nModules patched: {stats['patched_modules']}")
        
        if stats['warmup_time_ms'] > 0:
            print(f"Warmup time: {stats['warmup_time_ms']:.1f} ms")
        
        if stats.get('kv_cache_stats'):
            kv = stats['kv_cache_stats']
            print(f"\nK/V Cache:")
            print(f"  Hit rate: {kv.get('overall_hit_rate', 0):.1%}")
            print(f"  Total hits: {kv.get('total_hits', 0)}")
            print(f"  Total misses: {kv.get('total_misses', 0)}")
        
        print("="*60 + "\n")


# Convenience functions
def apply_default_optimizations(
    model: nn.Module,
    verbose: bool = True
) -> OptimizationManager:
    """
    Apply default optimizations to a model.
    
    This is the simplest way to enable optimizations:
        from optimizations.integration import apply_default_optimizations
        opt_mgr = apply_default_optimizations(transformer)
    """
    config = OptimizationConfig(
        enable_lowrank_lora=True,
        enable_kv_cache=True,
        enable_triton_rmsnorm=True,
        enable_triton_swiglu=True,
        enable_warmup=False,  # Skip warmup for faster startup
        verbose=verbose,
    )
    
    mgr = OptimizationManager(config)
    mgr.optimize_model(model)
    
    return mgr


@contextmanager
def optimized_inference(model: nn.Module, config: Optional[OptimizationConfig] = None):
    """
    Context manager for optimized inference.
    
    Applies optimizations on entry, cleans up on exit.
    
    Usage:
        with optimized_inference(transformer) as opt_mgr:
            output = sample(transformer, ...)
        print(opt_mgr.get_stats())
    """
    mgr = OptimizationManager(config)
    try:
        mgr.optimize_model(model)
        yield mgr
    finally:
        mgr.clear_caches(model)


def estimate_optimization_benefit(
    num_frames: int,
    height: int,
    width: int,
    num_steps: int,
    num_windows: int,
    num_layers: int = 40,
    hidden_dim: int = 3072,
    lora_rank: int = 64,
    use_cfg: bool = True,
) -> Dict[str, Any]:
    """
    Estimate the benefit of optimizations for given settings.
    
    Returns estimates of FLOP savings and expected speedup.
    """
    # Compute token count
    lat_h, lat_w = height // 8 // 2, width // 8 // 2  # VAE + patch
    lat_t = num_frames  # Approx
    tokens_per_window = lat_t * lat_h * lat_w
    
    # Total forward passes
    cfg_factor = 2 if use_cfg else 1
    total_forwards = num_steps * num_windows * cfg_factor
    
    # LoRA savings per layer
    # Dense: 2 * in * out * rank + 2 * tokens * in * out (matmul)
    # Low-rank: 2 * tokens * (in*rank + rank*out)
    dense_flops = 2 * hidden_dim * hidden_dim * lora_rank
    lowrank_flops = 2 * tokens_per_window * (hidden_dim * lora_rank + lora_rank * hidden_dim)
    
    lora_savings_per_layer = dense_flops * total_forwards - lowrank_flops * total_forwards
    total_lora_savings = lora_savings_per_layer * num_layers
    
    # K/V cache savings
    # Each layer: 2 GEMMs (K and V projection) saved for steps > 1
    kv_proj_flops = 2 * 2 * tokens_per_window * hidden_dim * hidden_dim  # 2 proj, 2 for matmul
    kv_savings = kv_proj_flops * num_layers * (total_forwards - num_windows)  # First step of each window can't be cached
    
    return {
        'tokens_per_window': tokens_per_window,
        'total_forward_passes': total_forwards,
        'lora_savings_gflops': total_lora_savings / 1e9,
        'kv_cache_savings_gflops': kv_savings / 1e9,
        'total_savings_gflops': (total_lora_savings + kv_savings) / 1e9,
        'estimated_speedup': 1.0 + (total_lora_savings + kv_savings) / (total_forwards * tokens_per_window * hidden_dim * hidden_dim * num_layers * 2) * 0.5,  # Conservative estimate
    }
