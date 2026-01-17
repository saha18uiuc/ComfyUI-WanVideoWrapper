# WanVideo optimized kernels
"""
WanVideo Performance Optimization Kernels

Research Basis:
1. FlashAttention-2 (Dao, 2023) - Fused attention with online softmax
2. Punica/S-LoRA (Chen et al., 2023) - Multi-LoRA batching with custom kernels  
3. RoFormer (Su et al., 2021) - Rotary Position Embedding
4. DiT (Peebles & Xie, 2022) - Diffusion Transformers with modulation
5. GLU Variants (Shazeer, 2020) - SwiGLU activation
6. Apex FusedLayerNorm - NVIDIA's fused normalization
7. Megatron-LM - Fused MLP operations
"""

from .rope_triton import rope_apply_triton, rope_apply_real
from .lora_optimized import (
    grouped_lora_forward_optimized,
    single_lora_forward_optimized,
)
from .fused_ops import (
    fused_layernorm_modulation,
    fused_swiglu,
    fused_swiglu_autograd,
    fused_qkv_rope,
    validate_fused_ops,
    benchmark_fused_ops,
)
from .fused_block import (
    FusedWanBlock,
    create_fused_block_from_attention_block,
    patch_model_with_fused_blocks,
    validate_fused_block,
    benchmark_fused_block,
    ENABLE_FUSED_BLOCK,
)
from .integration import (
    OptimizationConfig,
    OptimizationLevel,
    setup_cuda_graph_environment,
    optimize_transformer,
    get_optimization_summary,
    quick_setup_auto,
    quick_setup_for_a100,
    quick_setup_for_l4,
    quick_setup_for_h100,
    enable_all_fused_ops,
    get_active_optimizations,
)

__all__ = [
    # RoPE
    'rope_apply_triton', 
    'rope_apply_real',
    # LoRA
    'grouped_lora_forward_optimized',
    'single_lora_forward_optimized',
    # Fused Ops
    'fused_layernorm_modulation',
    'fused_swiglu',
    'fused_swiglu_autograd',
    'fused_qkv_rope',
    'validate_fused_ops',
    'benchmark_fused_ops',
    # Fused Block (Phase 1)
    'FusedWanBlock',
    'create_fused_block_from_attention_block',
    'patch_model_with_fused_blocks',
    'validate_fused_block',
    'benchmark_fused_block',
    'ENABLE_FUSED_BLOCK',
    # Integration
    'OptimizationConfig',
    'OptimizationLevel',
    'setup_cuda_graph_environment',
    'optimize_transformer',
    'get_optimization_summary',
    'quick_setup_auto',
    'quick_setup_for_a100',
    'quick_setup_for_l4',
    'quick_setup_for_h100',
    'enable_all_fused_ops',
    'get_active_optimizations',
]
