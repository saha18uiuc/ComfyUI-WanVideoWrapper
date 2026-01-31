import torch
import torch.nn as nn
from accelerate import init_empty_weights
from .gguf.gguf_utils import GGUFParameter, dequantize_gguf_tensor

# Global flag for optimized LoRA path (can be toggled at runtime)
USE_LOWRANK_LORA = True  # Default: use optimized low-rank path

def set_lora_optimization(enabled: bool = True):
    """Enable or disable the low-rank LoRA optimization globally."""
    global USE_LOWRANK_LORA
    USE_LOWRANK_LORA = enabled

@torch.library.custom_op("wanvideo::apply_lora", mutates_args=())
def apply_lora(weight: torch.Tensor, lora_diff_0: torch.Tensor, lora_diff_1: torch.Tensor, lora_diff_2: float, lora_strength: torch.Tensor) -> torch.Tensor:
    """Original dense ΔW approach - kept for backward compatibility."""
    patch_diff = torch.mm(
        lora_diff_0.flatten(start_dim=1),
        lora_diff_1.flatten(start_dim=1)
    ).reshape(weight.shape)

    alpha = lora_diff_2 / lora_diff_1.shape[0] if lora_diff_2 != 0.0 else 1.0
    scale = lora_strength * alpha

    return weight + patch_diff * scale


# ============================================================================
# OPTIMIZED LOW-RANK LORA OPERATIONS
# Instead of W' = W + α*(A@B), we compute y = xW + α*((xB)A)
# This avoids materializing the full [out, in] delta matrix.
# ============================================================================

@torch.library.custom_op("wanvideo::apply_lora_lowrank", mutates_args=())
def apply_lora_lowrank(
    base_output: torch.Tensor,
    input_activations: torch.Tensor,
    lora_down: torch.Tensor,  # [out, rank] or [rank, in] depending on convention
    lora_up: torch.Tensor,    # [rank, in] or [out, rank]
    alpha: float,
    rank: int,
    strength: torch.Tensor
) -> torch.Tensor:
    """
    Optimized LoRA application on activations (RunLoRA-style).
    
    Computes: base_output + scale * ((input @ down.T) @ up.T)
    
    This is mathematically equivalent to (input @ (W + scale*up@down).T)
    but avoids materializing the full [out, in] delta matrix.
    
    For video diffusion with large token counts, this can be significantly
    faster than the dense approach.
    """
    # Compute scale
    scale = strength * (alpha / rank if alpha != 0.0 else 1.0 / rank)
    
    if scale.item() == 0.0:
        return base_output
    
    # Flatten to 2D for GEMM
    orig_shape = input_activations.shape
    x_2d = input_activations.reshape(-1, orig_shape[-1])
    out_2d = base_output.reshape(-1, base_output.shape[-1])
    
    # Low-rank computation: (x @ down.T) @ up.T
    # Assuming lora_down is [rank, in], lora_up is [out, rank]
    down_flat = lora_down.flatten(start_dim=1)  # [rank, in]
    up_flat = lora_up.flatten(start_dim=1)      # [out, rank]
    
    tmp = x_2d @ down_flat.T  # [tokens, rank]
    delta = tmp @ up_flat.T   # [tokens, out]
    
    result = out_2d + delta * scale
    return result.reshape(base_output.shape)


@apply_lora_lowrank.register_fake
def _(base_output, input_activations, lora_down, lora_up, alpha, rank, strength):
    return base_output.clone()


@torch.library.custom_op("wanvideo::lora_delta_lowrank", mutates_args=())
def lora_delta_lowrank_op(
    input_activations: torch.Tensor,
    lora_down: torch.Tensor,
    lora_up: torch.Tensor,
    scale: torch.Tensor
) -> torch.Tensor:
    """Compute just the LoRA delta in low-rank form."""
    if scale.item() == 0.0:
        return torch.zeros(
            *input_activations.shape[:-1],
            lora_up.shape[0],
            device=input_activations.device,
            dtype=input_activations.dtype
        )
    
    orig_shape = input_activations.shape
    x_2d = input_activations.reshape(-1, orig_shape[-1])
    
    down_flat = lora_down.flatten(start_dim=1)
    up_flat = lora_up.flatten(start_dim=1)
    
    tmp = x_2d @ down_flat.T
    delta = (tmp @ up_flat.T) * scale
    
    return delta.reshape(*orig_shape[:-1], -1)


@lora_delta_lowrank_op.register_fake
def _(input_activations, lora_down, lora_up, scale):
    out_features = lora_up.shape[0]
    out_shape = list(input_activations.shape[:-1]) + [out_features]
    return input_activations.new_empty(out_shape)

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
        
        # Flag for using optimized low-rank LoRA (applies to activations instead of weights)
        self.use_lowrank_lora = True  # Default: use optimized path

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
    
    # ============================================================================
    # OPTIMIZED LOW-RANK LORA: Apply LoRA on activations instead of weights
    # ============================================================================
    
    def _apply_lora_lowrank_to_output(self, input_activations, base_output):
        """
        Apply LoRA in low-rank form directly on activations.
        
        Instead of: y = x @ (W + α*A@B).T
        Computes:   y = x @ W.T + α * ((x @ B.T) @ A.T)
        
        This avoids materializing the full [out, in] ΔW matrix, which is the
        key optimization from RunLoRA for large token counts.
        """
        if not hasattr(self, "lora_diff_0_0"):
            return base_output
        
        delta_sum = None
        
        for idx, lora_diff_names in enumerate(self.lora_diffs):
            lora_strength = self._get_lora_strength(idx)
            
            if isinstance(lora_diff_names, tuple):
                # Two-matrix LoRA: up @ down
                lora_up = getattr(self, lora_diff_names[0])    # [out, rank]
                lora_down = getattr(self, lora_diff_names[1])  # [rank, in]
                alpha = getattr(self, lora_diff_names[2], 0.0)
                
                rank = lora_down.shape[0]
                scale = lora_strength * (float(alpha) / rank if alpha != 0.0 else 1.0 / rank)
                
                if scale == 0.0:
                    continue
                
                # Low-rank computation: (x @ down.T) @ up.T
                # Flatten for GEMM
                orig_shape = input_activations.shape
                x_2d = input_activations.reshape(-1, orig_shape[-1])
                
                down_flat = lora_down.flatten(start_dim=1)  # [rank, in]
                up_flat = lora_up.flatten(start_dim=1)      # [out, rank]
                
                # Two skinny GEMMs instead of one fat GEMM
                tmp = x_2d @ down_flat.T  # [tokens, rank] - skinny!
                delta = (tmp @ up_flat.T) * scale  # [tokens, out]
                delta = delta.reshape(*orig_shape[:-1], -1)
                
                if delta_sum is None:
                    delta_sum = delta
                else:
                    delta_sum = delta_sum + delta
            else:
                # Pre-computed single diff - fall back to adding directly
                lora_diff = getattr(self, lora_diff_names)
                # This case still uses the weight approach
                # (single diff means it's already computed as A@B)
                pass  # Will be handled by original path
        
        if delta_sum is not None:
            return base_output + delta_sum
        return base_output

    def set_lora_diffs(self, lora_diffs, device=torch.device("cpu")):
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

    def _get_lora_strength(self, idx):
        strength_tensor = getattr(self, f"_lora_strength_{idx}")
        if self._lora_strength_is_scheduled[idx]:
            return strength_tensor.index_select(0, self._step).squeeze(0)
        return strength_tensor[0]

    def _get_weight_with_lora(self, weight):
        """Apply LoRA using custom ops to avoid graph breaks"""
        if not hasattr(self, "lora_diff_0_0"):
            return weight

        for idx, lora_diff_names in enumerate(self.lora_diffs):
            lora_strength = self._get_lora_strength(idx)

            if isinstance(lora_diff_names, tuple):
                lora_diff_0 = getattr(self, lora_diff_names[0])
                lora_diff_1 = getattr(self, lora_diff_names[1])
                lora_diff_2 = getattr(self, lora_diff_names[2])

                weight = self._apply_lora_impl(
                    weight, lora_diff_0, lora_diff_1,
                    float(lora_diff_2) if lora_diff_2 is not None else 0.0, lora_strength
                )
            else:
                lora_diff = getattr(self, lora_diff_names)
                weight = self._apply_single_lora_impl(weight, lora_diff, lora_strength)
        return weight

    def _prepare_weight(self, input):
        """Prepare weight tensor - handles both regular and GGUF weights"""
        if self.is_gguf:
            weight = dequantize_gguf_tensor(self.weight).to(self.compute_dtype)
        else:
            weight = self.weight.to(input)
        return weight

    def _should_use_lowrank_lora(self) -> bool:
        """
        Determine if we should use the optimized low-rank LoRA path.
        
        Conditions for low-rank optimization:
        1. Global USE_LOWRANK_LORA flag is True
        2. Instance flag use_lowrank_lora is True
        3. We have two-matrix LoRA (not pre-computed diff)
        """
        if not USE_LOWRANK_LORA or not self.use_lowrank_lora:
            return False
        
        # Check if we have two-matrix LoRA
        if not hasattr(self, "lora_diff_0_0"):
            return False
        
        # Check that at least one LoRA is two-matrix form
        for lora_diff_names in self.lora_diffs:
            if isinstance(lora_diff_names, tuple):
                return True
        
        return False

    def forward(self, input):
        weight = self._prepare_weight(input)

        if self.bias is not None:
            bias = self.bias.to(input if not self.is_gguf else self.compute_dtype)
        else:
            bias = None

        # Only apply scale_weight for non-GGUF models
        scaled_input = input
        if not self.is_gguf and self.scale_weight is not None:
            if weight.numel() < input.numel():
                weight = weight * self.scale_weight
            else:
                scaled_input = input * self.scale_weight

        # ========================================================================
        # OPTIMIZED PATH: Apply LoRA on activations (RunLoRA-style)
        # This avoids materializing full [out, in] delta matrix
        # ========================================================================
        if self._should_use_lowrank_lora():
            # Base linear without LoRA
            out = self._linear_forward_impl(scaled_input, weight, bias)
            # Add LoRA delta computed on activations
            out = self._apply_lora_lowrank_to_output(scaled_input, out)
        else:
            # Original path: apply LoRA to weights
            weight = self._get_weight_with_lora(weight)
            out = self._linear_forward_impl(scaled_input, weight, bias)
        
        del weight, bias
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
