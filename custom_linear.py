import os
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

MERGE_STATIC_LORA = os.environ.get("WAN_MERGE_STATIC_LORA", "1").strip().lower() in ("1", "true", "yes")
# Pre-merge LoRA: combine all patches into a single delta at cache build time
# Uses O(weight_size) memory instead of O(num_patches * weight_size)
LORA_PREMERGE = os.environ.get("WAN_LORA_PREMERGE", "1").strip().lower() in ("1", "true", "yes")
# Store merged delta on CPU to save GPU memory (default ON for memory safety)
LORA_PREMERGE_CPU = os.environ.get("WAN_LORA_PREMERGE_CPU", "1").strip().lower() in ("1", "true", "yes")


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
        for i, diff in enumerate(lora_diffs):
            if len(diff) > 1:
                self.register_buffer(f"lora_diff_{i}_0", diff[0].to(device, self.compute_dtype))
                self.register_buffer(f"lora_diff_{i}_1", diff[1].to(device, self.compute_dtype))
                setattr(self, f"lora_diff_{i}_2", diff[2])
                self.lora_diffs.append((f"lora_diff_{i}_0", f"lora_diff_{i}_1", f"lora_diff_{i}_2"))
            else:
                self.register_buffer(f"lora_diff_{i}_0", diff[0].to(device, self.compute_dtype))
                self.lora_diffs.append(f"lora_diff_{i}_0")

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
        into a SINGLE delta tensor at cache time. This uses O(weight_size) memory
        instead of O(num_patches * weight_size).
        """
        if self._lora_cached_deltas is not None:
            # For CPU-cached pre-merge, just check dtype (we'll move to GPU during forward)
            if LORA_PREMERGE and LORA_PREMERGE_CPU:
                if self._lora_cache_dtype == weight.dtype:
                    return
            elif self._lora_cache_device == weight.device and self._lora_cache_dtype == weight.dtype:
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
            # This is memory efficient AND fast - only ONE tensor stored
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
                    
                    # Compute patch
                    patch = torch.mm(
                        lora_diff_0.flatten(start_dim=1),
                        lora_diff_1.flatten(start_dim=1)
                    ).reshape(weight.shape)
                    
                    rank = lora_diff_1.shape[0]
                    alpha = (float(lora_diff_2) / rank) if (lora_diff_2 is not None and rank != 0) else 1.0
                    scale = strength_value * alpha
                    
                    # Accumulate into merged delta
                    if merged_delta is None:
                        merged_delta = patch * scale
                    else:
                        merged_delta.add_(patch, alpha=scale)
                    del patch
                else:
                    lora_diff = getattr(self, lora_diff_names).to(weight.device, mm_dtype)
                    scale = strength_value
                    if merged_delta is None:
                        merged_delta = lora_diff * scale
                    else:
                        merged_delta.add_(lora_diff, alpha=scale)
            
            # Store just the single merged delta (on CPU if memory-safe mode)
            if merged_delta is not None:
                merged_delta = merged_delta.to(weight.dtype).contiguous()
                if LORA_PREMERGE_CPU:
                    merged_delta = merged_delta.cpu()
                self._lora_cached_deltas = [merged_delta]
            else:
                self._lora_cached_deltas = []
        else:
            # Original mode: cache individual patches (uses more memory)
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
        
        # Track where cache is stored (may be CPU for memory safety)
        if LORA_PREMERGE and LORA_PREMERGE_CPU:
            self._lora_cache_device = torch.device('cpu')
        else:
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
        """Apply LoRA using memory-efficient operations"""
        if not getattr(self, "lora_diffs", None):
            return weight

        self._maybe_build_lora_cache(weight)
        if not self._lora_cached_deltas:
            return weight

        if LORA_PREMERGE:
            # Pre-merge mode: single merged delta, already scaled
            merged_delta = self._lora_cached_deltas[0]
            # Move from CPU to GPU if needed (for memory-safe mode)
            if merged_delta.device != weight.device:
                merged_delta = merged_delta.to(weight.device, non_blocking=True)
            return weight + merged_delta
        else:
            # Original mode: apply each cached patch with its strength
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

        if self.bias is not None:
            bias = self.bias.to(input if not self.is_gguf else self.compute_dtype)
        else:
            bias = None

        # Only apply scale_weight for non-GGUF models
        if not self.is_gguf and self.scale_weight is not None:
            if weight.numel() < input.numel():
                weight = weight * self.scale_weight
            else:
                input = input * self.scale_weight

        use_grouped = (
            self.grouped_lora_enabled
            and weight.is_cuda
            and getattr(self, "lora_diffs", None)
            and all(isinstance(names, tuple) for names in self.lora_diffs)
        )
        grouped_delta = None
        if use_grouped:
            grouped_delta = self._compute_grouped_lora(input, weight)

        if grouped_delta is not None:
            out = self._linear_forward_impl(input, weight, bias)
            out = out + grouped_delta.to(out.dtype)
        else:
            weight = self._get_weight_with_lora(weight)
            out = self._linear_forward_impl(input, weight, bias)
        del weight, input, bias
        return out

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
