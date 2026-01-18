import torch
from ..custom_linear import CustomLinear
from ..utils import log


def merge_static_lora_weights(module, min_patches=4):
    """
    Merge constant-strength LoRAs directly into CustomLinear weights.

    Args:
        module: root module containing CustomLinear layers.
        min_patches (int): only merge when a layer has at least this many patches.
    """
    merged_layers = 0
    with torch.no_grad():
        for _, submodule in module.named_modules():
            if not isinstance(submodule, CustomLinear):
                continue
            if not getattr(submodule, "lora_diffs", None):
                continue
            if not hasattr(submodule, "_lora_strength_is_scheduled"):
                continue
            if any(submodule._lora_strength_is_scheduled):
                continue
            if len(submodule.lora_diffs) < min_patches:
                continue
            weight = submodule.weight.data
            device = weight.device
            dtype = weight.dtype
            updated = False
            for idx, diff_names in enumerate(submodule.lora_diffs):
                if not isinstance(diff_names, tuple):
                    # skip non-linear LoRAs
                    updated = False
                    break
                lora_diff_0 = getattr(submodule, diff_names[0]).to(device, dtype)
                lora_diff_1 = getattr(submodule, diff_names[1]).to(device, dtype)
                lora_diff_2 = getattr(submodule, diff_names[2])
                a = lora_diff_0.flatten(start_dim=1)
                b = lora_diff_1.flatten(start_dim=1)
                if a.shape[1] != weight.shape[1] or b.shape[0] != weight.shape[0]:
                    updated = False
                    break
                rank = b.shape[1]
                alpha = (float(lora_diff_2) / rank) if (lora_diff_2 is not None and rank != 0) else 1.0
                strength = submodule._get_lora_strength(idx)
                strength = strength.to(dtype=torch.float32, device=device)
                if strength.numel() != 1:
                    updated = False
                    break
                delta = torch.mm(a, b).reshape_as(weight)
                weight.add_(delta, alpha=float(strength.item() * alpha))
                updated = True
            if updated:
                merged_layers += 1
                from ..custom_linear import remove_lora_from_module

                remove_lora_from_module(submodule)
    if merged_layers > 0:
        log.info(f"Merged static LoRA weights into {merged_layers} layers.")
