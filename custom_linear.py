import os
import time
import torch
import torch.nn as nn
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
}

def print_lora_timing_stats():
    """Print LoRA timing statistics."""
    if not _LORA_TIMING_ENABLED:
        print("[LoRA Timing] Timing not enabled. Set WAN_LORA_TIMING=1")
        return
    s = _lora_timing_stats
    print(f"\n[LoRA Timing Statistics]")
    print(f"  Cache builds: {s['cache_builds']} (total: {s['total_cache_build_time']:.2f}s)")
    print(f"  Cache hits: {s['cache_hits']} (total: {s['total_cache_hit_time']:.4f}s)")
    print(f"  Cache moves: {s['cache_moves']} (total: {s['total_cache_move_time']:.4f}s)")
    if s['cache_hits'] > 0:
        avg_hit = s['total_cache_hit_time'] / s['cache_hits'] * 1000
        print(f"  Avg cache hit: {avg_hit:.3f}ms")
    if s['cache_builds'] > 0:
        avg_build = s['total_cache_build_time'] / s['cache_builds'] * 1000
        print(f"  Avg cache build: {avg_build:.1f}ms")

MERGE_STATIC_LORA = os.environ.get("WAN_MERGE_STATIC_LORA", "1").strip().lower() in ("1", "true", "yes")

# LORA_STREAMING: Build merged delta incrementally with memory cleanup (DEFAULT)
# Avoids OOM while still caching for speed
LORA_STREAMING = os.environ.get("WAN_LORA_STREAMING", "1").strip().lower() in ("1", "true", "yes")

# LORA_LAZY: Compute on-demand without ANY caching (fallback if streaming still OOMs)
LORA_LAZY = os.environ.get("WAN_LORA_LAZY", "0").strip().lower() in ("1", "true", "yes")

# LORA_PREMERGE: Old approach - builds cache all at once (OOM risk)
LORA_PREMERGE = os.environ.get("WAN_LORA_PREMERGE", "0").strip().lower() in ("1", "true", "yes")

if LORA_STREAMING:
    print(f"[WanVideo LoRA] Streaming merge ENABLED - incremental cache build")
elif LORA_LAZY:
    print(f"[WanVideo LoRA] Lazy mode ENABLED - compute on-demand")
elif LORA_PREMERGE:
    print(f"[WanVideo LoRA] Pre-merge ENABLED")


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
        # Check if we have a cached delta (don't invalidate on device change!)
        if (self._lora_cached_deltas is not None and 
            len(self._lora_cached_deltas) > 0 and
            self._lora_cache_dtype == weight.dtype):
            
            t0 = time.perf_counter() if _LORA_TIMING_ENABLED else 0
            cached_delta = self._lora_cached_deltas[0]
            
            # Move cache to correct device if needed (non-blocking for speed)
            if cached_delta.device != weight.device:
                move_t0 = time.perf_counter() if _LORA_TIMING_ENABLED else 0
                cached_delta = cached_delta.to(weight.device, non_blocking=True)
                self._lora_cached_deltas[0] = cached_delta
                self._lora_cache_device = weight.device
                if _LORA_TIMING_ENABLED:
                    _lora_timing_stats["cache_moves"] += 1
                    _lora_timing_stats["total_cache_move_time"] += time.perf_counter() - move_t0
            
            # FAST PATH: Use cached delta
            result = weight + cached_delta
            
            if _LORA_TIMING_ENABLED:
                _lora_timing_stats["cache_hits"] += 1
                _lora_timing_stats["total_cache_hit_time"] += time.perf_counter() - t0
            
            return result
        
        # SLOW PATH (first call only): Build the merged delta incrementally
        build_t0 = time.perf_counter() if _LORA_TIMING_ENABLED else 0
        compute_dtype = torch.float16  # Use FP16 to reduce memory
        
        # Allocate merged delta ONCE - same size as weight
        merged_delta = torch.zeros(weight.shape, dtype=compute_dtype, device=weight.device)
        
        num_patches = len(self.lora_diffs)
        
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
                
                # Compute patch in FP16
                patch = torch.mm(
                    lora_diff_0.flatten(start_dim=1).to(weight.device, compute_dtype),
                    lora_diff_1.flatten(start_dim=1).to(weight.device, compute_dtype)
                ).reshape(weight.shape)
                
                rank = lora_diff_1.shape[0]
                alpha = (float(lora_diff_2) / rank) if (lora_diff_2 is not None and rank != 0) else 1.0
                scale = strength_value * alpha
                
                # Accumulate into merged delta (in-place)
                merged_delta.add_(patch, alpha=scale)
                del patch  # Free immediately
            else:
                lora_diff = getattr(self, lora_diff_names)
                merged_delta.add_(lora_diff.to(weight.device, compute_dtype), alpha=strength_value)
            
            # Periodic memory cleanup to prevent fragmentation
            if idx % 100 == 99:
                torch.cuda.empty_cache()
        
        # Convert to weight dtype and cache
        merged_delta = merged_delta.to(weight.dtype)
        self._lora_cached_deltas = [merged_delta]
        self._lora_cache_device = weight.device
        self._lora_cache_dtype = weight.dtype
        
        # Final cleanup
        torch.cuda.empty_cache()
        
        if _LORA_TIMING_ENABLED:
            _lora_timing_stats["cache_builds"] += 1
            _lora_timing_stats["total_cache_build_time"] += time.perf_counter() - build_t0
        
        return weight + merged_delta
    
    def _apply_lora_lazy(self, weight):
        """Apply LoRA on-demand without caching. Lowest memory, slowest."""
        compute_dtype = torch.float16 if weight.is_cuda else weight.dtype
        result = weight.clone()
        
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
                    lora_diff_0.flatten(start_dim=1).to(compute_dtype),
                    lora_diff_1.flatten(start_dim=1).to(compute_dtype)
                ).reshape(weight.shape)
                
                rank = lora_diff_1.shape[0]
                alpha = (float(lora_diff_2) / rank) if (lora_diff_2 is not None and rank != 0) else 1.0
                scale = strength_value * alpha
                
                result.add_(patch.to(result.dtype), alpha=scale)
                del patch
            else:
                lora_diff = getattr(self, lora_diff_names)
                result.add_(lora_diff.to(result.dtype), alpha=strength_value)
        
        return result
    

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
