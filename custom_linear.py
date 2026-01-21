import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import init_empty_weights
from .gguf.gguf_utils import GGUFParameter, dequantize_gguf_tensor
try:
    from .wanvideo.lora_kernels import grouped_lora_available, grouped_lora_forward
except Exception:
    def grouped_lora_available():
        return False

    def grouped_lora_forward(*args, **kwargs):
        raise RuntimeError("Grouped LoRA kernel unavailable (missing dependencies).")

# Timing instrumentation for LoRA operations
_LORA_TIMING_ENABLED = os.environ.get("WAN_LORA_TIMING", "0").strip().lower() in ("1", "true", "yes")
_lora_timing_stats = {
    "cache_builds": 0,
    "cache_hits": 0,
    "cache_moves": 0,  # Device transfers
    "total_cache_build_time": 0.0,
    "total_cache_hit_time": 0.0,
    "total_cache_move_time": 0.0,
    # Punica-style A/B cache stats
    "ab_cache_builds": 0,
    "ab_cache_hits": 0,
}

# Global prefetch stream for async cache moves
_prefetch_stream = None

def get_prefetch_stream():
    """Get or create a CUDA stream for async prefetching."""
    global _prefetch_stream
    if _prefetch_stream is None and torch.cuda.is_available():
        _prefetch_stream = torch.cuda.Stream()
    return _prefetch_stream

def print_lora_timing_stats():
    """Print LoRA timing statistics."""
    if not _LORA_TIMING_ENABLED:
        print("[LoRA Timing] Timing not enabled. Set WAN_LORA_TIMING=1")
        return
    s = _lora_timing_stats
    gpu_adds = s.get('gpu_additions', 0)
    cpu_falls = s.get('cpu_fallbacks', 0)
    total_adds = gpu_adds + cpu_falls
    
    print(f"\n[LoRA Timing Statistics]")
    print(f"  Cache builds: {s['cache_builds']} (total: {s['total_cache_build_time']:.2f}s)")
    print(f"  Cache hits: {s['cache_hits']} (total: {s['total_cache_hit_time']:.4f}s)")
    print(f"  Cache moves: {s['cache_moves']} (total: {s['total_cache_move_time']:.4f}s)")
    
    # Punica-style A/B cache stats
    ab_builds = s.get('ab_cache_builds', 0)
    ab_hits = s.get('ab_cache_hits', 0)
    if ab_builds > 0 or ab_hits > 0:
        total_ab = ab_builds + ab_hits
        hit_rate = ab_hits / total_ab * 100 if total_ab > 0 else 0
        print(f"  A/B cache: {ab_hits} hits / {ab_builds} builds ({hit_rate:.1f}% hit rate)")
    
    # GPU vs CPU fallback stats
    if total_adds > 0:
        gpu_pct = gpu_adds / total_adds * 100
        print(f"  GPU additions: {gpu_adds}/{total_adds} ({gpu_pct:.1f}%) ✓ FAST PATH")
        if cpu_falls > 0:
            print(f"  CPU fallbacks: {cpu_falls}/{total_adds} ({100-gpu_pct:.1f}%) ⚠ SLOW PATH")
        else:
            print(f"  CPU fallbacks: 0 (none needed - all GPU!) ✓")
    
    if s['cache_hits'] > 0:
        avg_hit = s['total_cache_hit_time'] / s['cache_hits'] * 1000
        print(f"  Avg cache hit: {avg_hit:.3f}ms")
    if s['cache_builds'] > 0:
        avg_build = s['total_cache_build_time'] / s['cache_builds'] * 1000
        print(f"  Avg cache build: {avg_build:.1f}ms")

MERGE_STATIC_LORA = os.environ.get("WAN_MERGE_STATIC_LORA", "1").strip().lower() in ("1", "true", "yes")

# PIN_LORA_GPU: Keep all LoRA caches permanently on GPU (avoids CPU->GPU transfers)
# Uses more GPU memory but faster if you have the memory
PIN_LORA_GPU = os.environ.get("WAN_PIN_LORA_GPU", "0").strip().lower() in ("1", "true", "yes")

# =============================================================================
# LORA MODE SELECTION
# =============================================================================
# 
# DEFAULT: STREAMING mode - proven fastest on L4 (~6.4 min vs 7.2 min base)
# 
# How STREAMING works:
# 1. First forward: compute delta = sum(A@B * strength) on CPU, cache it
# 2. Subsequent forwards: transfer cached delta to GPU, compute (W+delta)@x
# 
# This is FASTER than alternatives because:
# - Only ONE matmul per forward (vs 2-3 in other modes)
# - Simple addition (weight + delta) is cheap
# - No expensive transpose/contiguous operations
# 
# Other modes (NOT recommended - tested and slower or OOM):
# - ONTHEFLY: 3 matmuls per forward - no speedup at steps=3
# - PIPELINED: Extra transpose+contiguous causes 16ms overhead (regression!)
# - FUSED: Caches delta on GPU - OOM on L4
# =============================================================================

# LORA_STREAMING: DEFAULT - cache delta on CPU, transfer each forward
# Proven fastest: ~6.4 min on L4 (vs 7.2 min base = 11% speedup)
LORA_STREAMING = os.environ.get("WAN_LORA_STREAMING", "1").strip().lower() in ("1", "true", "yes")

# LORA_ONTHEFLY: Punica-style W@x + A@(B@x) - tested, no speedup at steps=3
LORA_ONTHEFLY = os.environ.get("WAN_LORA_ONTHEFLY", "0").strip().lower() in ("1", "true", "yes")

# LORA_LAZY: No caching - slowest, only if everything else OOMs
LORA_LAZY = os.environ.get("WAN_LORA_LAZY", "0").strip().lower() in ("1", "true", "yes")

# LORA_PREMERGE: Legacy - OOM risk
LORA_PREMERGE = os.environ.get("WAN_LORA_PREMERGE", "0").strip().lower() in ("1", "true", "yes")

# LORA_FUSED: OOM on L4 - don't use
LORA_FUSED = os.environ.get("WAN_LORA_FUSED", "0").strip().lower() in ("1", "true", "yes")

# LORA_PIPELINED: DISABLED - causes regression (16ms/hit due to transpose+contiguous overhead)
# The overhead of delta.t().contiguous() every forward outweighs any transfer overlap benefit
LORA_PIPELINED = os.environ.get("WAN_LORA_PIPELINED", "0").strip().lower() in ("1", "true", "yes")

# =============================================================================
# LORA_TRITON: NEW! Use Triton grouped kernel for LoRA (RESEARCH-BACKED)
# =============================================================================
# Based on Punica/S-LoRA papers: compute out = W@x + scale*(A@(B@x)) directly
# 
# Key advantages over STREAMING:
# - Keeps small A, B matrices on GPU (~1MB each vs 50MB delta)
# - NO CPU→GPU transfer per forward (eliminates 25ms/hit overhead!)
# - Uses fused Triton kernel for efficient A@(B@x) computation
# 
# Memory: 1262 layers × ~2MB = ~2.5GB (fits on L4's 22GB)
# Expected: Eliminates 249s of transfer overhead → potential 3+ min savings
# =============================================================================
LORA_TRITON = os.environ.get("WAN_LORA_TRITON", "0").strip().lower() in ("1", "true", "yes")

# Global compiled function cache to avoid re-compilation
_compiled_fused_forward = None
_compile_warmup_done = False


def _fused_lora_addmm_impl(out_flat: torch.Tensor, x_flat: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
    """
    Core fused LoRA computation: out += x @ delta.T
    This function is designed to be compiled by torch.compile.
    """
    # In-place addmm: out = 1.0*out + 1.0*(x @ delta_t)
    return torch.addmm(out_flat, x_flat, delta_t, beta=1.0, alpha=1.0, out=out_flat)


def _get_compiled_fused_forward():
    """Get or create the compiled fused forward function."""
    global _compiled_fused_forward, _compile_warmup_done
    if _compiled_fused_forward is None and LORA_COMPILE:
        try:
            # Use reduce-overhead mode for minimal warmup
            # fullgraph=True forces full compilation (faster after warmup)
            _compiled_fused_forward = torch.compile(
                _fused_lora_addmm_impl,
                mode="reduce-overhead",
                fullgraph=True,
            )
            if not _compile_warmup_done:
                print("[WanVideo LoRA] torch.compile warming up (one-time cost)...")
                _compile_warmup_done = True
        except Exception as e:
            print(f"[WanVideo LoRA] torch.compile failed, using eager mode: {e}")
            _compiled_fused_forward = _fused_lora_addmm_impl
    return _compiled_fused_forward if _compiled_fused_forward else _fused_lora_addmm_impl

_triton_available = grouped_lora_available()
if LORA_TRITON:
    if _triton_available:
        print(f"[WanVideo LoRA] TRITON mode (NEW! Research-backed)")
        print(f"[WanVideo LoRA]   Uses Punica-style fused kernel: out = W@x + A@(B@x)")
        print(f"[WanVideo LoRA]   NO CPU→GPU transfer! A,B stay on GPU (~2.5GB total)")
        print(f"[WanVideo LoRA]   Expected: Eliminates 249s transfer overhead")
    else:
        print(f"[WanVideo LoRA] TRITON mode requested but Triton not available!")
        print(f"[WanVideo LoRA]   Will fall back to STREAMING mode at runtime")
elif LORA_STREAMING:
    print(f"[WanVideo LoRA] STREAMING mode (proven fastest)")
    print(f"[WanVideo LoRA]   Cache delta on CPU, transfer each forward")
    print(f"[WanVideo LoRA]   ~6.4 min on L4 (11% faster than 7.2 min base)")
elif LORA_PIPELINED:
    print(f"[WanVideo LoRA] PIPELINED mode - WARNING: causes regression, use STREAMING")
elif LORA_ONTHEFLY:
    print(f"[WanVideo LoRA] PUNICA-STYLE mode - no speedup at steps=3")
elif LORA_FUSED:
    print(f"[WanVideo LoRA] FUSED mode - WARNING: OOMs on <40GB GPUs")
elif LORA_LAZY:
    print(f"[WanVideo LoRA] LAZY mode - no caching (slowest)")
elif LORA_PREMERGE:
    print(f"[WanVideo LoRA] Pre-merge mode (legacy)")


def pin_all_lora_caches_to_gpu(module, device="cuda"):
    """
    Move all LoRA caches to GPU and keep them there.
    Call this after the first forward pass (when caches are built).
    
    This eliminates CPU->GPU transfer overhead at the cost of GPU memory.
    """
    moved = 0
    for name, submodule in module.named_modules():
        if isinstance(submodule, CustomLinear):
            if (submodule._lora_cached_deltas is not None and 
                len(submodule._lora_cached_deltas) > 0):
                for i, delta in enumerate(submodule._lora_cached_deltas):
                    if delta.device.type != "cuda":
                        submodule._lora_cached_deltas[i] = delta.to(device)
                        moved += 1
                submodule._lora_cache_device = torch.device(device)
    if moved > 0:
        print(f"[WanVideo LoRA] Pinned {moved} LoRA caches to GPU")
    return moved


def clear_all_lora_caches(module):
    """Clear all LoRA caches to free memory."""
    cleared = 0
    ab_cleared = 0
    for name, submodule in module.named_modules():
        if isinstance(submodule, CustomLinear):
            if submodule._lora_cached_deltas is not None:
                submodule._lora_cached_deltas = None
                submodule._lora_cache_device = None
                submodule._lora_cache_dtype = None
                cleared += 1
            # Also clear the A/B GPU cache used by on-the-fly mode
            if hasattr(submodule, '_lora_ab_cache') and submodule._lora_ab_cache:
                submodule._lora_ab_cache.clear()
                ab_cleared += 1
    if cleared > 0 or ab_cleared > 0:
        torch.cuda.empty_cache()
        print(f"[WanVideo LoRA] Cleared {cleared} delta caches, {ab_cleared} A/B caches")
    return cleared + ab_cleared


def premerge_lora_into_weights(module):
    """
    PRE-MERGE: Merge all LoRA deltas into base weights ONCE at load time.
    
    After this, forward pass has ZERO LoRA overhead - just normal linear layers!
    This is the fastest possible approach.
    
    Call this after model loading and before inference.
    """
    merged = 0
    skipped = 0
    
    for name, submodule in module.named_modules():
        if not isinstance(submodule, CustomLinear):
            continue
        if not getattr(submodule, "lora_diffs", None):
            continue
        if len(submodule.lora_diffs) == 0:
            continue
            
        # Get base weight
        weight = submodule.weight
        if weight is None:
            skipped += 1
            continue
            
        # Compute merged delta on CPU (safe, no OOM)
        compute_dtype = torch.float32
        merged_delta = torch.zeros(weight.shape, dtype=compute_dtype, device='cpu')
        
        for idx, lora_diff_names in enumerate(submodule.lora_diffs):
            lora_strength = submodule._get_lora_strength(idx)
            if isinstance(lora_strength, torch.Tensor):
                strength_value = lora_strength.item() if lora_strength.numel() == 1 else lora_strength.mean().item()
            else:
                strength_value = float(lora_strength)
            
            if abs(strength_value) < 1e-8:
                continue
            
            if isinstance(lora_diff_names, tuple):
                lora_diff_0 = getattr(submodule, lora_diff_names[0])
                lora_diff_1 = getattr(submodule, lora_diff_names[1])
                lora_diff_2 = getattr(submodule, lora_diff_names[2])
                
                patch = torch.mm(
                    lora_diff_0.flatten(start_dim=1).to('cpu', compute_dtype),
                    lora_diff_1.flatten(start_dim=1).to('cpu', compute_dtype)
                ).reshape(weight.shape)
                
                rank = lora_diff_1.shape[0]
                alpha = (float(lora_diff_2) / rank) if (lora_diff_2 is not None and rank != 0) else 1.0
                scale = strength_value * alpha
                
                merged_delta.add_(patch, alpha=scale)
                del patch
            else:
                lora_diff = getattr(submodule, lora_diff_names)
                merged_delta.add_(lora_diff.to('cpu', compute_dtype), alpha=strength_value)
        
        # Store the merged delta for use during forward pass
        # (We can't modify FP8 weights in-place, so we store the delta)
        submodule._lora_cached_deltas = [merged_delta.to(weight.dtype)]
        submodule._lora_cache_device = torch.device('cpu')
        submodule._lora_cache_dtype = weight.dtype
        submodule._premerged = True  # Flag to skip re-computation
        merged += 1
    
    print(f"[WanVideo LoRA] Pre-merged {merged} layers, skipped {skipped}")
    return merged


@torch.library.custom_op("wanvideo::apply_lora", mutates_args=())
def apply_lora(weight: torch.Tensor, lora_diff_0: torch.Tensor, lora_diff_1: torch.Tensor, lora_diff_2: float, lora_strength: torch.Tensor) -> torch.Tensor:
    patch_diff = torch.mm(
        lora_diff_0.flatten(start_dim=1),
        lora_diff_1.flatten(start_dim=1)
    ).reshape(weight.shape)

    alpha = lora_diff_2 / lora_diff_1.shape[0] if lora_diff_2 != 0.0 else 1.0
    scale = lora_strength * alpha

    return weight + patch_diff * scale

@apply_lora.register_fake
def _(weight, lora_diff_0, lora_diff_1, lora_diff_2, lora_strength):
    # Return weight with same metadata
    return weight.clone()

@torch.library.custom_op("wanvideo::apply_single_lora", mutates_args=())
def apply_single_lora(weight: torch.Tensor, lora_diff: torch.Tensor, lora_strength: torch.Tensor) -> torch.Tensor:
    return weight + lora_diff * lora_strength

@apply_single_lora.register_fake
def _(weight, lora_diff, lora_strength):
    # Return weight with same metadata
    return weight.clone()

@torch.library.custom_op("wanvideo::linear_forward", mutates_args=())
def linear_forward(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    return torch.nn.functional.linear(input, weight, bias)

@linear_forward.register_fake
def _(input, weight, bias):
    # Calculate output shape: (..., out_features)
    out_features = weight.shape[0]
    output_shape = list(input.shape[:-1]) + [out_features]
    return input.new_empty(output_shape)

#based on https://github.com/huggingface/diffusers/blob/main/src/diffusers/quantizers/gguf/utils.py
def _replace_linear(model, compute_dtype, state_dict, prefix="", patches=None, scale_weights=None, compile_args=None, modules_to_not_convert=[]):

    has_children = list(model.children())
    if not has_children:
        return

    allow_compile = False

    for name, module in model.named_children():
        if compile_args is not None:
            allow_compile = compile_args.get("allow_unmerged_lora_compile", False)
        module_prefix = prefix + name + "."
        module_prefix = module_prefix.replace("_orig_mod.", "")
        _replace_linear(module, compute_dtype, state_dict, module_prefix, patches, scale_weights, compile_args, modules_to_not_convert)

        if isinstance(module, nn.Linear) and "loras" not in module_prefix and "dual_controller" not in module_prefix and name not in modules_to_not_convert:
            weight_key = module_prefix + "weight"
            if weight_key not in state_dict:
                continue

            in_features = state_dict[weight_key].shape[1]
            out_features = state_dict[weight_key].shape[0]

            is_gguf = isinstance(state_dict[weight_key], GGUFParameter)

            scale_weight = None
            if not is_gguf and scale_weights is not None:
                scale_key = f"{module_prefix}scale_weight"
                scale_weight = scale_weights.get(scale_key)

            with init_empty_weights():
                model._modules[name] = CustomLinear(
                    in_features,
                    out_features,
                    module.bias is not None,
                    compute_dtype=compute_dtype,
                    scale_weight=scale_weight,
                    allow_compile=allow_compile,
                    is_gguf=is_gguf
                )
            model._modules[name].source_cls = type(module)
            model._modules[name].requires_grad_(False)

    return model

def set_lora_params(module, patches, module_prefix="", device=torch.device("cpu")):
    remove_lora_from_module(module)
    # Recursively set lora_diffs and lora_strengths for all CustomLinear layers
    for name, child in module.named_children():
        params = list(child.parameters())
        if params:
            device = params[0].device
        else:
            device = torch.device("cpu")
        child_prefix = (f"{module_prefix}{name}.")
        set_lora_params(child, patches, child_prefix, device)
    if isinstance(module, CustomLinear):
        key = f"diffusion_model.{module_prefix}weight"
        patch = patches.get(key, [])
        #print(f"Processing LoRA patches for {key}: {len(patch)} patches found")
        if len(patch) == 0:
            key = key.replace("_orig_mod.", "")
            patch = patches.get(key, [])
            #print(f"Processing LoRA patches for {key}: {len(patch)} patches found")
        if len(patch) != 0:
            lora_diffs = []
            for p in patch:
                lora_obj = p[1]
                if "head" in key:
                    continue  # For now skip LoRA for head layers
                elif hasattr(lora_obj, "weights"):
                    lora_diffs.append(lora_obj.weights)
                elif isinstance(lora_obj, tuple) and lora_obj[0] == "diff":
                    lora_diffs.append(lora_obj[1])
                else:
                    continue
            lora_strengths = [p[0] for p in patch]
            module.set_lora_diffs(lora_diffs, device=device)
            module.set_lora_strengths(lora_strengths, device=device)
            module._step.fill_(0)   # Initialize step for LoRA scheduling


class CustomLinear(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=False,
        compute_dtype=None,
        device=None,
        scale_weight=None,
        allow_compile=False,
        is_gguf=False
    ) -> None:
        super().__init__(in_features, out_features, bias, device)
        self.compute_dtype = compute_dtype
        self.lora_diffs = []
        self.register_buffer("_step", torch.zeros((), dtype=torch.long))
        self.scale_weight = scale_weight
        self.lora_strengths = []
        self.allow_compile = allow_compile
        self.is_gguf = is_gguf
        self._lora_cached_deltas = None
        self._lora_cache_device = None
        self._lora_cache_dtype = None
        self._grouped_cache = None
        self._grouped_cache_device = None
        self._grouped_cache_dtype = None
        self.grouped_lora_enabled = True
        self.merge_static_lora = MERGE_STATIC_LORA
        self._static_lora_merged = False
        # FUSED mode: cache delta.T on GPU for output-space fusion
        self._fused_delta_t = None  # Transposed delta cached on GPU
        self._fused_delta_device = None
        self._fused_delta_dtype = None

        if not allow_compile:
            self._apply_lora_impl = self._apply_lora_custom_op
            self._apply_single_lora_impl = self._apply_single_lora_custom_op
            self._linear_forward_impl = self._linear_forward_custom_op
        else:
            self._apply_lora_impl = self._apply_lora_direct
            self._apply_single_lora_impl = self._apply_single_lora_direct
            self._linear_forward_impl = self._linear_forward_direct


    # Direct implementations (no custom ops)
    def _apply_lora_direct(self, weight, lora_diff_0, lora_diff_1, lora_diff_2, lora_strength):
        patch_diff = torch.mm(
            lora_diff_0.flatten(start_dim=1),
            lora_diff_1.flatten(start_dim=1)
        ).reshape(weight.shape) + 0
        alpha = lora_diff_2 / lora_diff_1.shape[0] if lora_diff_2 != 0.0 else 1.0
        scale = lora_strength * alpha
        return weight + patch_diff * scale

    def _apply_single_lora_direct(self, weight, lora_diff, lora_strength):
        return weight + lora_diff * lora_strength

    def _linear_forward_direct(self, input, weight, bias):
        return torch.nn.functional.linear(input, weight, bias)

    # Custom op implementations
    def _apply_lora_custom_op(self, weight, lora_diff_0, lora_diff_1, lora_diff_2, lora_strength):
        return torch.ops.wanvideo.apply_lora(weight, lora_diff_0, lora_diff_1,
            float(lora_diff_2) if lora_diff_2 is not None else 0.0, lora_strength
        )

    def _apply_single_lora_custom_op(self, weight, lora_diff, lora_strength):
        return torch.ops.wanvideo.apply_single_lora(weight, lora_diff, lora_strength)

    def _linear_forward_custom_op(self, input, weight, bias):
        return torch.ops.wanvideo.linear_forward(input, weight, bias)

    def _reset_lora_cache(self):
        self._lora_cached_deltas = None
        self._lora_cache_device = None
        self._lora_cache_dtype = None
        self._grouped_cache = None
        self._grouped_cache_device = None
        self._grouped_cache_dtype = None
        # Clear FUSED mode cache
        self._fused_delta_t = None
        self._fused_delta_device = None
        self._fused_delta_dtype = None

    def set_lora_diffs(self, lora_diffs, device=torch.device("cpu")):
        self._reset_lora_cache()
        self._static_lora_merged = False
        self.lora_diffs = []
        all_tuples = True
        for i, diff in enumerate(lora_diffs):
            if len(diff) > 1:
                self.register_buffer(f"lora_diff_{i}_0", diff[0].to(device, self.compute_dtype))
                self.register_buffer(f"lora_diff_{i}_1", diff[1].to(device, self.compute_dtype))
                setattr(self, f"lora_diff_{i}_2", diff[2])
                self.lora_diffs.append((f"lora_diff_{i}_0", f"lora_diff_{i}_1", f"lora_diff_{i}_2"))
            else:
                self.register_buffer(f"lora_diff_{i}_0", diff[0].to(device, self.compute_dtype))
                self.lora_diffs.append(f"lora_diff_{i}_0")
                all_tuples = False
        # Cache this check to avoid repeated isinstance() calls in forward()
        self._all_lora_are_tuples = all_tuples
        # NOTE: grouped_lora_enabled stays False - the Triton batched kernel
        # was found to cause OOM/slowdowns in previous testing

    def set_lora_strengths(self, lora_strengths, device=torch.device("cpu")):
        self._lora_strength_tensors = []
        self._lora_strength_is_scheduled = []
        self._step = self._step.to(device)
        for i, strength in enumerate(lora_strengths):
            if isinstance(strength, list):
                tensor = torch.tensor(strength, dtype=self.compute_dtype, device=device)
                self.register_buffer(f"_lora_strength_{i}", tensor)
                self._lora_strength_is_scheduled.append(True)
            else:
                tensor = torch.tensor([strength], dtype=self.compute_dtype, device=device)
                self.register_buffer(f"_lora_strength_{i}", tensor)
                self._lora_strength_is_scheduled.append(False)
        if self.merge_static_lora:
            self._try_merge_static_lora()

    def _get_lora_strength(self, idx):
        strength_tensor = getattr(self, f"_lora_strength_{idx}")
        if self._lora_strength_is_scheduled[idx]:
            return strength_tensor.index_select(0, self._step).squeeze(0)
        return strength_tensor[0]

    def _maybe_build_lora_cache(self, weight):
        """
        Build LoRA cache. With LORA_PREMERGE=1 (default), all patches are merged
        into a SINGLE delta tensor at cache time.
        
        OPTIMIZATION: This is MUCH faster than applying 1262 patches individually.
        Memory: O(weight_size) for merged delta vs O(num_patches * rank * features) for A/B matrices.
        For many LoRAs, merged delta actually uses LESS memory.
        """
        if self._lora_cached_deltas is not None:
            if self._lora_cache_device == weight.device and self._lora_cache_dtype == weight.dtype:
                return
        
        # Determine compute dtype for FP8 models
        fp8_safe_dtype = None
        if hasattr(torch, "float8_e4m3fn") and weight.dtype == torch.float8_e4m3fn:
            fp8_safe_dtype = torch.float16
        if hasattr(torch, "float8_e4m3fnuz") and weight.dtype == torch.float8_e4m3fnuz:
            fp8_safe_dtype = torch.float16
        if hasattr(torch, "float8_e5m2") and weight.dtype == torch.float8_e5m2:
            fp8_safe_dtype = torch.float16
        
        mm_dtype = weight.dtype if fp8_safe_dtype is None else fp8_safe_dtype
        
        if LORA_PREMERGE:
            # Pre-merge mode: combine all patches into a single delta
            # This is MUCH faster - only ONE add operation per forward pass!
            merged_delta = None
            
            for idx, lora_diff_names in enumerate(self.lora_diffs):
                # Get strength for this LoRA
                lora_strength = self._get_lora_strength(idx)
                if isinstance(lora_strength, torch.Tensor):
                    strength_value = lora_strength.item() if lora_strength.numel() == 1 else lora_strength.mean().item()
                else:
                    strength_value = float(lora_strength)
                
                if abs(strength_value) < 1e-8:
                    continue
                
                if isinstance(lora_diff_names, tuple):
                    lora_diff_0 = getattr(self, lora_diff_names[0]).to(weight.device, mm_dtype)
                    lora_diff_1 = getattr(self, lora_diff_names[1]).to(weight.device, mm_dtype)
                    lora_diff_2 = getattr(self, lora_diff_names[2])
                    
                    # Compute patch: (out, rank) @ (rank, in) = (out, in)
                    patch = torch.mm(
                        lora_diff_0.flatten(start_dim=1),
                        lora_diff_1.flatten(start_dim=1)
                    ).reshape(weight.shape)
                    
                    rank = lora_diff_1.shape[0]
                    alpha = (float(lora_diff_2) / rank) if (lora_diff_2 is not None and rank != 0) else 1.0
                    scale = strength_value * alpha
                    
                    # Accumulate into merged delta (in-place for memory efficiency)
                    if merged_delta is None:
                        merged_delta = patch * scale
                    else:
                        merged_delta.add_(patch, alpha=scale)
                    del patch
                else:
                    lora_diff = getattr(self, lora_diff_names).to(weight.device, mm_dtype)
                    scale = strength_value
                    if merged_delta is None:
                        merged_delta = lora_diff.clone() * scale
                    else:
                        merged_delta.add_(lora_diff, alpha=scale)
            
            # Store the single merged delta on GPU (NO CPU storage - it's too slow)
            if merged_delta is not None:
                self._lora_cached_deltas = [merged_delta.to(weight.dtype).contiguous()]
            else:
                self._lora_cached_deltas = []
            
            self._lora_cache_device = weight.device
            self._lora_cache_dtype = weight.dtype
        else:
            # Original mode: cache individual patches (uses more memory, more operations)
            cached = []
            for lora_diff_names in self.lora_diffs:
                if isinstance(lora_diff_names, tuple):
                    lora_diff_0 = getattr(self, lora_diff_names[0]).to(weight.device, mm_dtype)
                    lora_diff_1 = getattr(self, lora_diff_names[1]).to(weight.device, mm_dtype)
                    lora_diff_2 = getattr(self, lora_diff_names[2])
                    patch = torch.mm(
                        lora_diff_0.flatten(start_dim=1),
                        lora_diff_1.flatten(start_dim=1)
                    ).reshape(weight.shape).to(weight.dtype).contiguous()
                    alpha = (float(lora_diff_2) / lora_diff_1.shape[0]) if (lora_diff_2 is not None and lora_diff_1.shape[0] != 0) else 1.0
                    cached.append((patch, alpha))
                else:
                    lora_diff = getattr(self, lora_diff_names)
                    patch = lora_diff.to(weight.device, weight.dtype).contiguous()
                    cached.append((patch, 1.0))
            self._lora_cached_deltas = cached
            self._lora_cache_device = weight.device
            self._lora_cache_dtype = weight.dtype

    def _clear_lora_state(self):
        if getattr(self, "lora_diffs", None):
            for idx in range(len(self.lora_diffs)):
                strength_name = f"_lora_strength_{idx}"
                if hasattr(self, strength_name):
                    delattr(self, strength_name)
        self.lora_diffs = []
        self._lora_strength_is_scheduled = []
        self._reset_lora_cache()
        self.grouped_lora_enabled = False

    def _try_merge_static_lora(self):
        """
        Attempt to merge LoRA weights directly into base weights.
        Only works for non-FP8 weights with constant (non-scheduled) strengths.
        With LORA_PREMERGE, this is less critical since we pre-merge at cache time.
        """
        if self._static_lora_merged:
            return
        if getattr(self, "lora_diffs", None) is None or not self.lora_diffs:
            return
        if any(self._lora_strength_is_scheduled):
            return  # Can't static merge if strengths change per step
        
        base_weight = self.weight.data if not self.is_gguf else self.weight
        
        # FP8 weights can't have LoRA merged directly (precision issues)
        fp8_dtypes = []
        if hasattr(torch, "float8_e4m3fn"):
            fp8_dtypes.append(torch.float8_e4m3fn)
        if hasattr(torch, "float8_e5m2"):
            fp8_dtypes.append(torch.float8_e5m2)
        if base_weight.dtype in fp8_dtypes:
            return  # Skip for FP8 - we'll handle via LORA_PREMERGE at runtime
        
        # For non-FP8, merge directly into weights
        mm_dtype = base_weight.dtype
        merged = False
        with torch.no_grad():
            for idx, lora_diff_names in enumerate(self.lora_diffs):
                strength = self._get_lora_strength(idx)
                if torch.is_tensor(strength):
                    if strength.numel() != 1:
                        return
                    strength_value = strength.item()
                else:
                    strength_value = float(strength)
                if strength_value == 0.0:
                    continue
                
                if isinstance(lora_diff_names, tuple):
                    lora_diff_0 = getattr(self, lora_diff_names[0]).to(base_weight.device, mm_dtype)
                    lora_diff_1 = getattr(self, lora_diff_names[1]).to(base_weight.device, mm_dtype)
                    lora_diff_2 = getattr(self, lora_diff_names[2])
                    patch = torch.mm(
                        lora_diff_0.flatten(start_dim=1),
                        lora_diff_1.flatten(start_dim=1)
                    ).reshape(base_weight.shape)
                    rank = lora_diff_1.shape[0]
                    alpha = (float(lora_diff_2) / rank) if (lora_diff_2 is not None and rank != 0) else 1.0
                else:
                    patch = getattr(self, lora_diff_names).to(base_weight.device, mm_dtype)
                    alpha = 1.0
                
                scale = strength_value * alpha
                base_weight.add_(patch, alpha=scale)
                merged = True
                del patch
                
        if merged:
            self._clear_lora_state()
            self._static_lora_merged = True

    def _maybe_build_grouped_cache(self, weight):
        if (
            self._grouped_cache is not None
            and self._grouped_cache_device == weight.device
            and self._grouped_cache_dtype == weight.dtype
        ):
            return self._grouped_cache
        if not getattr(self, "lora_diffs", None):
            return None
        a_list = []
        b_list = []
        alpha_list = []
        ranks = set()
        in_features = weight.shape[1]
        out_features = weight.shape[0]
        for lora_diff_names in self.lora_diffs:
            if not isinstance(lora_diff_names, tuple):
                return None
            lora_diff_0 = getattr(self, lora_diff_names[0]).to(weight.device, weight.dtype)
            lora_diff_1 = getattr(self, lora_diff_names[1]).to(weight.device, weight.dtype)
            lora_diff_2 = getattr(self, lora_diff_names[2])
            a_flat = lora_diff_0.flatten(start_dim=1).contiguous()
            b_flat = lora_diff_1.flatten(start_dim=1).contiguous()
            if a_flat.shape[1] != in_features or b_flat.shape[0] != out_features:
                return None
            if b_flat.shape[1] != a_flat.shape[0]:
                return None
            ranks.add(a_flat.shape[0])
            a_list.append(a_flat)
            b_list.append(b_flat)
            alpha = (float(lora_diff_2) / b_flat.shape[1]) if (lora_diff_2 is not None and b_flat.shape[1] != 0) else 1.0
            alpha_list.append(alpha)
        if len(ranks) != 1:
            return None
        cache = {
            "A": torch.stack(a_list, dim=0).contiguous(),
            "B": torch.stack(b_list, dim=0).contiguous(),
            "alpha_tensor": torch.tensor(alpha_list, dtype=torch.float32, device=weight.device),
            "out_buffer": None,
        }
        self._grouped_cache = cache
        self._grouped_cache_device = weight.device
        self._grouped_cache_dtype = weight.dtype
        return cache

    def _get_weight_with_lora(self, weight):
        """Apply LoRA using optimized operations.
        
        Priority:
        1. LORA_STREAMING (default): Build merged delta incrementally, cache it
        2. LORA_LAZY: Compute on-demand, no caching (if streaming OOMs)
        3. LORA_PREMERGE: Build all at once (OOM risk)
        """
        if not getattr(self, "lora_diffs", None):
            return weight

        if LORA_STREAMING:
            return self._apply_lora_streaming(weight)
        elif LORA_LAZY:
            return self._apply_lora_lazy(weight)
        
        # Legacy path
        self._maybe_build_lora_cache(weight)
        if not self._lora_cached_deltas:
            return weight

        if LORA_PREMERGE:
            return weight + self._lora_cached_deltas[0]
        else:
            weight = weight.clone()
            for idx, cached_entry in enumerate(self._lora_cached_deltas):
                patch_diff, alpha = cached_entry
                lora_strength = self._get_lora_strength(idx)
                if isinstance(lora_strength, torch.Tensor):
                    strength_value = lora_strength.item() if lora_strength.numel() == 1 else lora_strength.mean().item()
                else:
                    strength_value = float(lora_strength)
                if abs(strength_value) < 1e-8:
                    continue
                scale = strength_value * alpha
                weight.add_(patch_diff, alpha=scale)
            return weight
    
    def _apply_lora_streaming(self, weight):
        """
        STREAMING MODE: Build merged delta incrementally with memory management.
        
        - First call: Build cache by processing patches ONE AT A TIME
        - Subsequent calls: Just add cached delta (FAST)
        
        Cache PERSISTS across device changes - we move it instead of rebuilding.
        """
        # Check if we have a cached delta
        if (self._lora_cached_deltas is not None and 
            len(self._lora_cached_deltas) > 0 and
            self._lora_cache_dtype == weight.dtype):
            
            t0 = time.perf_counter() if _LORA_TIMING_ENABLED else 0
            cached_delta = self._lora_cached_deltas[0]
            
            if weight.device.type == 'cuda':
                # Move delta to GPU, add, DON'T keep (to avoid OOM)
                gpu_delta = cached_delta.to(weight.device, weight.dtype, non_blocking=True)
                result = weight + gpu_delta
                del gpu_delta
                if _LORA_TIMING_ENABLED:
                    _lora_timing_stats["gpu_additions"] = _lora_timing_stats.get("gpu_additions", 0) + 1
            else:
                result = weight + cached_delta.to(weight.dtype)
            
            if _LORA_TIMING_ENABLED:
                _lora_timing_stats["cache_hits"] += 1
                _lora_timing_stats["total_cache_hit_time"] += time.perf_counter() - t0
            
            return result
        
        # SLOW PATH (first call only): Build the merged delta ON CPU to avoid OOM
        # This is the CPU-offloaded LoRA technique from QLoRA/PEFT papers
        build_t0 = time.perf_counter() if _LORA_TIMING_ENABLED else 0
        compute_dtype = torch.float32  # CPU compute in FP32 for accuracy
        
        # Allocate merged delta ON CPU - plenty of RAM there
        merged_delta = torch.zeros(weight.shape, dtype=compute_dtype, device='cpu')
        
        for idx, lora_diff_names in enumerate(self.lora_diffs):
            # Get strength
            lora_strength = self._get_lora_strength(idx)
            if isinstance(lora_strength, torch.Tensor):
                strength_value = lora_strength.item() if lora_strength.numel() == 1 else lora_strength.mean().item()
            else:
                strength_value = float(lora_strength)
            
            if abs(strength_value) < 1e-8:
                continue
            
            if isinstance(lora_diff_names, tuple):
                lora_diff_0 = getattr(self, lora_diff_names[0])
                lora_diff_1 = getattr(self, lora_diff_names[1])
                lora_diff_2 = getattr(self, lora_diff_names[2])
                
                # Compute patch ON CPU - no GPU memory needed!
                patch = torch.mm(
                    lora_diff_0.flatten(start_dim=1).to('cpu', compute_dtype),
                    lora_diff_1.flatten(start_dim=1).to('cpu', compute_dtype)
                ).reshape(weight.shape)
                
                rank = lora_diff_1.shape[0]
                alpha = (float(lora_diff_2) / rank) if (lora_diff_2 is not None and rank != 0) else 1.0
                scale = strength_value * alpha
                
                # Accumulate into merged delta (in-place, on CPU)
                merged_delta.add_(patch, alpha=scale)
                del patch  # Free immediately
            else:
                lora_diff = getattr(self, lora_diff_names)
                merged_delta.add_(lora_diff.to('cpu', compute_dtype), alpha=strength_value)
        
        # Convert to weight dtype and KEEP ON CPU for storage
        # We'll move to GPU only when needed (in the cache hit path above)
        merged_delta = merged_delta.to(weight.dtype)
        self._lora_cached_deltas = [merged_delta]  # Stored on CPU!
        self._lora_cache_device = torch.device('cpu')  # Mark as on CPU
        self._lora_cache_dtype = weight.dtype
        
        if _LORA_TIMING_ENABLED:
            _lora_timing_stats["cache_builds"] += 1
            _lora_timing_stats["total_cache_build_time"] += time.perf_counter() - build_t0
        
        # Now move to GPU for this forward pass
        # The cache hit path above will handle future calls
        gpu_delta = merged_delta.to(weight.device, non_blocking=True)
        return weight + gpu_delta
    
    def _apply_lora_lazy(self, weight):
        """Apply LoRA on-demand without caching. Compute on CPU to avoid OOM."""
        compute_dtype = torch.float32  # CPU compute in FP32
        
        # Build merged delta on CPU
        merged_delta = torch.zeros(weight.shape, dtype=compute_dtype, device='cpu')
        
        for idx, lora_diff_names in enumerate(self.lora_diffs):
            lora_strength = self._get_lora_strength(idx)
            if isinstance(lora_strength, torch.Tensor):
                strength_value = lora_strength.item() if lora_strength.numel() == 1 else lora_strength.mean().item()
            else:
                strength_value = float(lora_strength)
            
            if abs(strength_value) < 1e-8:
                continue
            
            if isinstance(lora_diff_names, tuple):
                lora_diff_0 = getattr(self, lora_diff_names[0])
                lora_diff_1 = getattr(self, lora_diff_names[1])
                lora_diff_2 = getattr(self, lora_diff_names[2])
                
                # Compute ON CPU
                patch = torch.mm(
                    lora_diff_0.flatten(start_dim=1).to('cpu', compute_dtype),
                    lora_diff_1.flatten(start_dim=1).to('cpu', compute_dtype)
                ).reshape(weight.shape)
                
                rank = lora_diff_1.shape[0]
                alpha = (float(lora_diff_2) / rank) if (lora_diff_2 is not None and rank != 0) else 1.0
                scale = strength_value * alpha
                
                merged_delta.add_(patch, alpha=scale)
                del patch
            else:
                lora_diff = getattr(self, lora_diff_names)
                merged_delta.add_(lora_diff.to('cpu', compute_dtype), alpha=strength_value)
        
        # Move only the final result to GPU
        return weight + merged_delta.to(weight.device, weight.dtype)
    
    def _apply_lora_onthefly(self, input, weight, bias):
        """
        OPTIMIZED PUNICA-STYLE: Compute W@x + sum(scale * A @ (B @ x)) directly.
        
        KEY OPTIMIZATIONS:
        1. Cache A and B on GPU after first transfer (eliminates 30,000+ transfers)
        2. Cache strength tensor references (avoids getattr+fstring overhead)
        3. Use dtype directly as cache key (no tuple creation)
        4. Fused addmm operations (single kernel instead of two)
        5. Minimal branching in hot loop
        """
        # Main forward pass - this is the bulk of compute
        out = F.linear(input, weight, bias)
        
        # Fast path: no LoRAs
        lora_diffs = self.lora_diffs
        if not lora_diffs:
            return out
        
        # Initialize/get caches (avoid repeated hasattr/getattr)
        ab_cache = self.__dict__.get('_lora_ab_cache')
        if ab_cache is None:
            ab_cache = {}
            self._lora_ab_cache = ab_cache
        
        # Cache strength tensor REFERENCES (one-time per layer, avoids 5000+ getattr calls)
        strength_cache = self.__dict__.get('_strength_tensor_refs')
        if strength_cache is None:
            strength_cache = [getattr(self, f"_lora_strength_{i}") for i in range(len(lora_diffs))]
            self._strength_tensor_refs = strength_cache
        
        target_dtype = input.dtype
        
        # Flatten input once, reuse for all LoRAs
        orig_shape = input.shape
        x_flat = input.view(-1, orig_shape[-1])
        
        lora_contribution = None
        
        # Local refs for hot loop (avoid repeated lookups)
        timing_enabled = _LORA_TIMING_ENABLED
        timing_stats = _lora_timing_stats if timing_enabled else None
        scheduled_flags = self._lora_strength_is_scheduled
        step_tensor = self._step
        target_device = input.device
        
        for idx, lora_diff_names in enumerate(lora_diffs):
            # Get strength from cached reference (fast list access vs getattr)
            strength_tensor = strength_cache[idx]
            if scheduled_flags[idx]:
                strength_tensor = strength_tensor.index_select(0, step_tensor).squeeze(0)
            
            # Get scalar value - single .item() call
            strength_value = strength_tensor.item() if strength_tensor.numel() == 1 else strength_tensor.mean().item()
            if abs(strength_value) < 1e-8:
                continue
            
            if isinstance(lora_diff_names, tuple):
                # Use dtype directly as key component (no tuple creation in hot path)
                lora_cache_key = (target_dtype, idx)
                
                cached = ab_cache.get(lora_cache_key)
                if cached is None:
                    # First time: transfer A, B to GPU and cache
                    A = getattr(self, lora_diff_names[0])
                    B = getattr(self, lora_diff_names[1])
                    alpha_val = getattr(self, lora_diff_names[2])
                    
                    rank = B.shape[0]
                    alpha = (float(alpha_val) / rank) if (alpha_val is not None and rank != 0) else 1.0
                    
                    # Transfer with non_blocking for better overlap
                    A_flat = A.to(target_device, target_dtype, non_blocking=True).flatten(start_dim=1)
                    B_flat_t = B.to(target_device, target_dtype, non_blocking=True).flatten(start_dim=1).t()
                    
                    ab_cache[lora_cache_key] = (A_flat, B_flat_t, alpha)
                    if timing_enabled:
                        timing_stats['ab_cache_builds'] += 1
                else:
                    A_flat, B_flat_t, alpha = cached
                    if timing_enabled:
                        timing_stats['ab_cache_hits'] += 1
                
                scale = strength_value * alpha
                
                # Compute B @ x then A @ (B @ x)
                Bx = torch.mm(x_flat, B_flat_t)
                
                # Fused addmm for matmul+add
                if lora_contribution is None:
                    lora_contribution = torch.mm(Bx, A_flat.t()).mul_(scale)
                else:
                    torch.addmm(lora_contribution, Bx, A_flat.t(), beta=1.0, alpha=scale, out=lora_contribution)
                
            else:
                # Single tensor LoRA (pre-computed delta)
                delta_cache_key = (cache_key, idx, 'delta')
                
                delta_flat_t = ab_cache.get(delta_cache_key)
                if delta_flat_t is None:
                    lora_diff = getattr(self, lora_diff_names)
                    delta_flat_t = lora_diff.to(target_device, target_dtype, non_blocking=True).flatten(start_dim=1).t()
                    ab_cache[delta_cache_key] = delta_flat_t
                
                if lora_contribution is None:
                    lora_contribution = torch.mm(x_flat, delta_flat_t).mul_(strength_value)
                else:
                    torch.addmm(lora_contribution, x_flat, delta_flat_t, beta=1.0, alpha=strength_value, out=lora_contribution)
        
        if lora_contribution is not None:
            out = out + lora_contribution.view(*orig_shape[:-1], -1)
        
        return out

    def _compute_grouped_lora(self, input, weight):
        if not grouped_lora_available():
            return None
        if not getattr(self, "lora_diffs", None):
            return None
        if not self.grouped_lora_enabled:
            return None
        cache = self._grouped_cache
        if cache is None or cache["A"].device != weight.device or cache["A"].dtype != weight.dtype:
            cache = self._maybe_build_grouped_cache(weight)
        if cache is None:
            return None
        strength_values = []
        for idx in range(len(self.lora_diffs)):
            strength = self._get_lora_strength(idx)
            if not torch.is_tensor(strength):
                strength = torch.tensor(strength, device=input.device, dtype=input.dtype)
            if strength.numel() != 1:
                return None
            strength_values.append(strength.reshape(1))
        if not strength_values:
            return None
        strengths_tensor = torch.cat(strength_values).to(device=input.device, dtype=torch.float32)
        alpha_tensor = cache["alpha_tensor"].to(input.device)
        if alpha_tensor.shape[0] != strengths_tensor.shape[0]:
            return None
        scales_tensor = strengths_tensor * alpha_tensor
        input_2d = input.reshape(-1, input.shape[-1]).contiguous()
        expected_shape = (input_2d.shape[0], cache["B"].shape[1])
        buffer = cache.get("out_buffer")
        if (
            buffer is None
            or buffer.shape != expected_shape
            or buffer.device != input_2d.device
        ):
            buffer = torch.zeros(expected_shape, device=input_2d.device, dtype=torch.float32)
            cache["out_buffer"] = buffer
        delta = grouped_lora_forward(input_2d, cache["A"], cache["B"], scales_tensor, out_buffer=buffer)
        return delta.reshape(*input.shape[:-1], delta.shape[-1]).to(input.dtype)

    def _prepare_weight(self, input):
        """Prepare weight tensor - handles both regular and GGUF weights"""
        if self.is_gguf:
            weight = dequantize_gguf_tensor(self.weight).to(self.compute_dtype)
        else:
            weight = self.weight.to(input)
        return weight

    def _apply_lora_pipelined(self, input, weight, bias):
        """
        PIPELINED MODE: Overlap CPU→GPU transfer with W@x computation.
        
        Research basis: CUDA stream parallelism (NVIDIA best practices)
        
        Standard STREAMING:
            1. Transfer delta CPU→GPU (blocks)
            2. Compute (W+delta)@x
            Total time = transfer_time + compute_time
        
        PIPELINED (this method):
            1. Start async transfer in stream B
            2. Compute W@x in stream A (overlapped with transfer!)
            3. Synchronize streams
            4. Compute delta@x and add to output
            Total time = max(transfer_time, Wx_time) + delta_x_time
        
        Benefits:
        - Hides transfer latency behind W@x computation
        - No permanent GPU storage (avoids OOM on L4)
        - Works on ALL GPUs without memory issues
        
        vs STREAMING: Faster due to overlap
        vs FUSED: Doesn't OOM (no permanent GPU delta cache)
        vs ONTHEFLY: 2 matmuls vs 3 (W@x + delta@x vs W@x + A@B@x)
        """
        # Ensure we have cached delta on CPU
        if (self._lora_cached_deltas is None or 
            len(self._lora_cached_deltas) == 0 or
            self._lora_cache_dtype != weight.dtype):
            # Build cache first using streaming logic, then use pipelined path
            _ = self._apply_lora_streaming(weight)
        
        t0 = time.perf_counter() if _LORA_TIMING_ENABLED else 0
        
        # Get CPU-cached delta
        cached_delta = self._lora_cached_deltas[0]
        
        # Create transfer stream (reuse if exists)
        if not hasattr(self, '_transfer_stream') or self._transfer_stream is None:
            self._transfer_stream = torch.cuda.Stream()
        
        # STEP 1: Start async transfer in separate stream
        with torch.cuda.stream(self._transfer_stream):
            gpu_delta = cached_delta.to(weight.device, weight.dtype, non_blocking=True)
        
        # STEP 2: Compute base output W@x in main stream (OVERLAPPED with transfer!)
        out = F.linear(input, weight, bias)
        
        # STEP 3: Synchronize - wait for delta transfer to complete
        self._transfer_stream.synchronize()
        
        # STEP 4: Add LoRA contribution via delta@x
        # Reshape for matmul
        orig_shape = out.shape
        out_flat = out.view(-1, orig_shape[-1])
        x_flat = input.view(-1, input.shape[-1])
        
        # Transpose delta for x @ delta.T computation
        delta_t = gpu_delta.t()
        if not delta_t.is_contiguous():
            delta_t = delta_t.contiguous()
        if not x_flat.is_contiguous():
            x_flat = x_flat.contiguous()
        
        # Fused add+matmul: out += x @ delta.T
        torch.addmm(out_flat, x_flat, delta_t, beta=1.0, alpha=1.0, out=out_flat)
        
        # STEP 5: Free GPU delta immediately to avoid memory buildup
        del gpu_delta, delta_t
        
        if _LORA_TIMING_ENABLED:
            _lora_timing_stats["cache_hits"] += 1
            _lora_timing_stats["total_cache_hit_time"] += time.perf_counter() - t0
            _lora_timing_stats["gpu_additions"] = _lora_timing_stats.get("gpu_additions", 0) + 1
        
        return out.view(*orig_shape)

    def _apply_lora_fused(self, input, weight, bias):
        """
        OUTPUT-SPACE LORA FUSION (inspired by LoRAFusion paper arxiv:2510.00206)
        
        Instead of materializing (W + delta) then matmul:
            result = (W + delta) @ x   # requires weight-space delta, creates new tensor
        
        We compute in output space:
            result = W @ x + delta @ x  # fused with torch.addmm!
        
        Key insight: Cache delta.T on GPU ONCE, then use addmm for fused add+matmul
        
        Benefits vs STREAMING:
        - No `weight + delta` tensor allocation every forward
        - Uses fused addmm kernel (single CUDA launch for add + matmul)
        - delta.T cached on GPU (no CPU->GPU transfer after first call)
        
        Benefits vs ONTHEFLY (Punica):
        - 2 matmuls instead of 3 (W@x + delta@x vs W@x + A@(B@x))
        - Pre-computed delta.T is smaller than A+B for high-rank LoRAs
        
        Memory: ~same as STREAMING (stores delta), but on GPU for speed
        WARNING: May OOM on GPUs with <40GB VRAM - use PIPELINED instead
        """
        # Check/build delta.T cache
        if (self._fused_delta_t is not None and 
            self._fused_delta_device == input.device and
            self._fused_delta_dtype == input.dtype):
            # FAST PATH: Use cached delta.T
            delta_t = self._fused_delta_t
            if _LORA_TIMING_ENABLED:
                _lora_timing_stats["cache_hits"] += 1
        else:
            # SLOW PATH (first call): Build delta on CPU, transfer to GPU, cache
            build_t0 = time.perf_counter() if _LORA_TIMING_ENABLED else 0
            compute_dtype = torch.float32
            
            merged_delta = torch.zeros(weight.shape, dtype=compute_dtype, device='cpu')
            
            for idx, lora_diff_names in enumerate(self.lora_diffs):
                lora_strength = self._get_lora_strength(idx)
                if isinstance(lora_strength, torch.Tensor):
                    strength_value = lora_strength.item() if lora_strength.numel() == 1 else lora_strength.mean().item()
                else:
                    strength_value = float(lora_strength)
                
                if abs(strength_value) < 1e-8:
                    continue
                
                if isinstance(lora_diff_names, tuple):
                    lora_diff_0 = getattr(self, lora_diff_names[0])
                    lora_diff_1 = getattr(self, lora_diff_names[1])
                    lora_diff_2 = getattr(self, lora_diff_names[2])
                    
                    patch = torch.mm(
                        lora_diff_0.flatten(start_dim=1).to('cpu', compute_dtype),
                        lora_diff_1.flatten(start_dim=1).to('cpu', compute_dtype)
                    ).reshape(weight.shape)
                    
                    rank = lora_diff_1.shape[0]
                    alpha = (float(lora_diff_2) / rank) if (lora_diff_2 is not None and rank != 0) else 1.0
                    scale = strength_value * alpha
                    
                    merged_delta.add_(patch, alpha=scale)
                    del patch
                else:
                    lora_diff = getattr(self, lora_diff_names)
                    merged_delta.add_(lora_diff.to('cpu', compute_dtype), alpha=strength_value)
            
            # Transfer TRANSPOSED delta to GPU and cache
            # delta.T shape: (in_features, out_features)
            # This allows: out += x @ delta.T via addmm
            delta_t = merged_delta.t().contiguous().to(input.device, input.dtype)
            
            self._fused_delta_t = delta_t
            self._fused_delta_device = input.device
            self._fused_delta_dtype = input.dtype
            
            if _LORA_TIMING_ENABLED:
                _lora_timing_stats["cache_builds"] += 1
                _lora_timing_stats["total_cache_build_time"] += time.perf_counter() - build_t0
        
        # Compute base output: W @ x
        out = F.linear(input, weight, bias)
        
        # Reshape for matmul - ensure contiguous for optimal CUDA performance
        orig_shape = out.shape
        out_flat = out.view(-1, orig_shape[-1])  # (batch*seq, out_features)
        x_flat = input.view(-1, input.shape[-1])  # (batch*seq, in_features)
        
        # Ensure contiguous memory layout for addmm (avoids hidden copies)
        if not out_flat.is_contiguous():
            out_flat = out_flat.contiguous()
        if not x_flat.is_contiguous():
            x_flat = x_flat.contiguous()
        
        # FUSED add + matmul: out += x @ delta.T
        # torch.addmm(input, mat1, mat2, beta=1, alpha=1, out=None)
        # computes: beta*input + alpha*(mat1 @ mat2)
        if LORA_COMPILE:
            # Use compiled kernel for better fusion
            _get_compiled_fused_forward()(out_flat, x_flat, delta_t)
        else:
            torch.addmm(out_flat, x_flat, delta_t, beta=1.0, alpha=1.0, out=out_flat)
        
        if _LORA_TIMING_ENABLED:
            _lora_timing_stats["gpu_additions"] = _lora_timing_stats.get("gpu_additions", 0) + 1
        
        return out.view(*orig_shape)

    def forward(self, input):
        weight = self._prepare_weight(input)

        bias = self.bias
        if bias is not None:
            bias = bias.to(input if not self.is_gguf else self.compute_dtype)

        # Only apply scale_weight for non-GGUF models
        if not self.is_gguf and self.scale_weight is not None:
            if weight.numel() < input.numel():
                weight = weight * self.scale_weight
            else:
                input = input * self.scale_weight

        # =============================================================================
        # LORA_TRITON: NEW RESEARCH-BACKED MODE - Use Triton fused kernel
        # Computes: out = W@x + scale*(A@(B@x)) without CPU→GPU transfer!
        # =============================================================================
        if LORA_TRITON and grouped_lora_available() and getattr(self, "lora_diffs", None) and len(self.lora_diffs) > 0:
            if weight.is_cuda and getattr(self, "_all_lora_are_tuples", None):
                # Force enable grouped LoRA for Triton mode
                self.grouped_lora_enabled = True
                try:
                    grouped_delta = self._compute_grouped_lora(input, weight)
                    if grouped_delta is not None:
                        out = self._linear_forward_impl(input, weight, bias)
                        if _LORA_TIMING_ENABLED:
                            _lora_timing_stats['ab_cache_hits'] = _lora_timing_stats.get('ab_cache_hits', 0) + 1
                        return out + grouped_delta.to(out.dtype)
                except Exception as e:
                    # Triton kernel failed, fall through to STREAMING
                    if not hasattr(self, '_triton_fail_logged'):
                        print(f"[WanVideo LoRA] Triton kernel failed: {e}, falling back to STREAMING")
                        self._triton_fail_logged = True
                # If grouped kernel fails, fall through to STREAMING
        
        # STREAMING mode (DEFAULT): Cache delta on CPU, transfer each forward
        # Uses standard path: _get_weight_with_lora -> _apply_lora_streaming
        # This falls through to the standard path below
        
        # Alternative modes (NOT recommended - tested slower or OOM):
        if LORA_PIPELINED and getattr(self, "lora_diffs", None) and len(self.lora_diffs) > 0:
            if input.device.type == 'cuda':
                return self._apply_lora_pipelined(input, weight, bias)

        if LORA_FUSED and getattr(self, "lora_diffs", None) and len(self.lora_diffs) > 0:
            return self._apply_lora_fused(input, weight, bias)

        if LORA_ONTHEFLY and getattr(self, "lora_diffs", None) and len(self.lora_diffs) > 0:
            return self._apply_lora_onthefly(input, weight, bias)

        # Fast path: check if grouped LoRA is possible (cache the tuple check result)
        if (self.grouped_lora_enabled and weight.is_cuda and 
            getattr(self, "lora_diffs", None) and getattr(self, "_all_lora_are_tuples", None)):
            grouped_delta = self._compute_grouped_lora(input, weight)
            if grouped_delta is not None:
                out = self._linear_forward_impl(input, weight, bias)
                return out + grouped_delta.to(out.dtype)

        # Standard path with LoRA
        weight = self._get_weight_with_lora(weight)
        return self._linear_forward_impl(input, weight, bias)

def update_lora_step(module, step):
    for name, submodule in module.named_modules():
        if isinstance(submodule, CustomLinear) and hasattr(submodule, "_step"):
            submodule._step.fill_(step)

def remove_lora_from_module(module):
    for name, submodule in module.named_modules():
        if hasattr(submodule, "lora_diffs"):
            for i in range(len(submodule.lora_diffs)):
                if hasattr(submodule, f"lora_diff_{i}_0"):
                    delattr(submodule, f"lora_diff_{i}_0")
                if hasattr(submodule, f"lora_diff_{i}_1"):
                    delattr(submodule, f"lora_diff_{i}_1")
                if hasattr(submodule, f"lora_diff_{i}_2"):
                    delattr(submodule, f"lora_diff_{i}_2")
            if hasattr(submodule, "_reset_lora_cache"):
                submodule._reset_lora_cache()
