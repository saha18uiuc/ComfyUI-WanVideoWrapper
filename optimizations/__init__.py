"""
Math-Heavy Optimizations for WanVideoWrapper

This module provides significant speed optimizations for video diffusion inference,
drawing from research papers on efficient computation and kernel fusion.

=== PROVEN EXACT OPTIMIZATIONS (no quality impact) ===

1. Fused Triton Kernels (Liger-Kernel inspired):
   - Fused RMSNorm + AdaLN modulation: 3 ops → 1 kernel
   - Fused QKV projection: 3 linear → 1 kernel
   - Fused Linear + GELU: 2 ops → 1 kernel
   
2. SmoothCache (CVPR 2025):
   - Caches layer outputs when timestep similarity is high
   - 8-71% speedup on DiT models
   - threshold=0.995 for near-exact output

3. torch.compile: PyTorch graph compiler
4. Batched CFG: Batch cond+uncond in single forward

=== EXPERIMENTAL OPTIMIZATIONS ===

5. LoRA Fast-path: Apply LoRA as low-rank activations (RunLoRA-inspired)
6. Cross-attention K/V caching
7. Token Merging (ToMe)

References:
- Liger-Kernel: https://github.com/linkedin/Liger-Kernel
- SmoothCache: https://arxiv.org/abs/2411.10510
- RunLoRA: https://arxiv.org/pdf/2312.03415
- FlashAttention: IO-aware attention algorithms
"""

# ============================================================================
# NOVEL FUSED TRITON KERNELS (Liger-Kernel inspired)
# ============================================================================
try:
    from .fused_kernels import (
        fused_rmsnorm_adaln,
        fused_qkv_projection,
        fused_gelu,
        fused_linear_gelu,
        check_triton_available,
    )
    FUSED_KERNELS_AVAILABLE = check_triton_available()
except ImportError:
    FUSED_KERNELS_AVAILABLE = False
    fused_rmsnorm_adaln = None
    fused_qkv_projection = None
    fused_gelu = None
    fused_linear_gelu = None

# ============================================================================
# SMOOTHCACHE (CVPR 2025 - Layer output caching)
# ============================================================================
try:
    from .smooth_cache import (
        SmoothCacheState,
        SmoothCacheHelper,
        apply_smooth_cache,
    )
    SMOOTH_CACHE_AVAILABLE = True
except ImportError:
    SMOOTH_CACHE_AVAILABLE = False
    SmoothCacheState = None
    SmoothCacheHelper = None
    apply_smooth_cache = None

# ============================================================================
# EXPERIMENTAL OPTIMIZATIONS
# ============================================================================

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

# Token Merging (generic)
from .token_merging import (
    apply_tome_to_model,
    remove_tome_from_model,
    estimate_tome_speedup,
    bipartite_soft_matching,
)

# Token Merging (WanVideo-specific)
from .tome_wanvideo import (
    patch_wanvideo_tome,
    unpatch_wanvideo_tome,
    set_tome_ratio,
    estimate_wanvideo_tome_speedup,
)

# CFG Optimizations
from .cfg_optimization import (
    rescale_cfg_output,
    get_adaptive_cfg_schedule,
    AdaptiveCFGScheduler,
    momentum_velocity_estimation,
    compute_perpendicular_cfg,
)

# Temporal Optimizations
from .temporal_optimization import (
    create_local_temporal_mask,
    apply_local_temporal_attention,
    TemporalAttentionCache,
    LocalTemporalAttention,
    apply_temporal_locality,
    estimate_temporal_savings,
)

# Solver Optimizations
from .solver_optimization import (
    VelocityEMAEstimator,
    compute_optimal_timesteps,
    AdaptiveStepController,
    heun_step_with_error,
    OptimizedFlowSolver,
    estimate_solver_benefit,
)

__all__ = [
    # Fused Kernels (Liger-Kernel inspired)
    'fused_rmsnorm_adaln',
    'fused_qkv_projection',
    'fused_gelu',
    'fused_linear_gelu',
    'check_triton_available',
    'FUSED_KERNELS_AVAILABLE',
    # SmoothCache
    'SmoothCacheState',
    'SmoothCacheHelper',
    'apply_smooth_cache',
    'SMOOTH_CACHE_AVAILABLE',
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
    # Token Merging (generic)
    'apply_tome_to_model',
    'remove_tome_from_model',
    'estimate_tome_speedup',
    'bipartite_soft_matching',
    # Token Merging (WanVideo-specific)
    'patch_wanvideo_tome',
    'unpatch_wanvideo_tome',
    'set_tome_ratio',
    'estimate_wanvideo_tome_speedup',
    # CFG Optimizations
    'rescale_cfg_output',
    'get_adaptive_cfg_schedule',
    'AdaptiveCFGScheduler',
    'momentum_velocity_estimation',
    'compute_perpendicular_cfg',
    # Temporal Optimizations
    'create_local_temporal_mask',
    'apply_local_temporal_attention',
    'TemporalAttentionCache',
    'LocalTemporalAttention',
    'apply_temporal_locality',
    'estimate_temporal_savings',
    # Solver Optimizations
    'VelocityEMAEstimator',
    'compute_optimal_timesteps',
    'AdaptiveStepController',
    'heun_step_with_error',
    'OptimizedFlowSolver',
    'estimate_solver_benefit',
]
