"""
Math-Heavy Optimizations for WanVideoWrapper

This module provides significant speed optimizations for video diffusion inference,
drawing from research papers on efficient LoRA computation, attention caching,
and fused kernel implementations.

Key optimizations:
1. LoRA Fast-path: Apply LoRA as low-rank activations instead of dense ΔW (RunLoRA-inspired)
2. Cross-attention K/V caching: Cache constant conditioning projections across steps
3. Fused Triton kernels: RMSNorm and SwiGLU with reduced memory bandwidth
4. Model warmup: Eliminate first-window compilation overhead

References:
- RunLoRA: https://arxiv.org/pdf/2312.03415 (Faster LoRA computation graphs)
- FlashAttention: IO-aware attention algorithms
- SageAttention: Fast attention via quantization-friendly design

Usage:
    from optimizations import OptimizationManager, apply_default_optimizations
    
    # Simple usage:
    opt_mgr = apply_default_optimizations(transformer)
    
    # Or with custom config:
    from optimizations import OptimizationConfig, OptimizationManager
    config = OptimizationConfig(enable_lowrank_lora=True, enable_kv_cache=True)
    mgr = OptimizationManager(config)
    mgr.optimize_model(transformer)
"""

# LoRA fast-path
from .lora_fastpath import (
    PackedLoRA,
    pack_loras_for_layer,
    lora_delta_lowrank,
    apply_lora_residual_optimized,
)

# K/V caching
from .kv_cache import (
    enable_crossattn_kv_cache,
    clear_kv_cache,
    CrossAttnKVCache,
    KVCacheContext,
)

# LoRA merge
from .lora_merge import (
    apply_lora_linear_inplace,
    apply_lora_conv2d_inplace,
    merge_all_loras_fused,
    LoRAMergeManager,
    estimate_merge_benefit,
)

# Warmup
from .warmup import (
    warmup_model,
    create_dummy_inputs,
    WarmupContext,
)

# Integration
from .integration import (
    OptimizationConfig,
    OptimizationManager,
    apply_default_optimizations,
    optimized_inference,
    estimate_optimization_benefit,
)

__all__ = [
    # LoRA fast-path
    'PackedLoRA',
    'pack_loras_for_layer', 
    'lora_delta_lowrank',
    'apply_lora_residual_optimized',
    # K/V caching
    'enable_crossattn_kv_cache',
    'clear_kv_cache',
    'CrossAttnKVCache',
    'KVCacheContext',
    # LoRA merge
    'apply_lora_linear_inplace',
    'apply_lora_conv2d_inplace',
    'merge_all_loras_fused',
    'LoRAMergeManager',
    'estimate_merge_benefit',
    # Warmup
    'warmup_model',
    'create_dummy_inputs',
    'WarmupContext',
    # Integration
    'OptimizationConfig',
    'OptimizationManager',
    'apply_default_optimizations',
    'optimized_inference',
    'estimate_optimization_benefit',
]
