"""
Model Warmup for Eliminating First-Window Compilation Overhead

In diffusion inference, the first window often takes 2-4x longer than
subsequent windows due to:
1. JIT compilation of custom ops (Triton, torch.compile)
2. cuBLAS/cuDNN kernel autotuning
3. CUDA kernel cache misses

This module provides warmup utilities that run a dummy forward pass
with the same shapes as real inference, forcing all compilations to
happen before timing starts.

After warmup:
- All Triton kernels are compiled
- cuBLAS has selected optimal GEMM algorithms
- torch.compile graphs are traced and cached
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Any
import gc


def create_dummy_inputs(
    batch_size: int,
    num_frames: int,
    height: int,
    width: int,
    hidden_dim: int,
    context_len: int,
    device: torch.device,
    dtype: torch.dtype,
    vae_downscale: int = 8,
    patch_size: Tuple[int, int, int] = (1, 2, 2),
) -> Dict[str, torch.Tensor]:
    """
    Create dummy input tensors matching real inference shapes.
    
    Args:
        batch_size: Batch size (usually 1 or 2 for CFG)
        num_frames: Number of frames in a window
        height: Video height in pixels
        width: Video width in pixels
        hidden_dim: Model hidden dimension
        context_len: Context/conditioning sequence length
        device: Target device
        dtype: Target dtype
        vae_downscale: VAE spatial downscale factor
        patch_size: Patch size (T, H, W)
    
    Returns:
        Dict of dummy tensors for model forward
    """
    # Compute latent dimensions
    lat_h = height // vae_downscale // patch_size[1]
    lat_w = width // vae_downscale // patch_size[2]
    lat_t = num_frames // patch_size[0]
    
    # Sequence length = flattened latent patches
    seq_len = lat_t * lat_h * lat_w
    
    return {
        'x': torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=dtype),
        'timestep': torch.ones(batch_size, device=device, dtype=torch.long) * 500,
        'context': torch.randn(batch_size, context_len, hidden_dim, device=device, dtype=dtype),
        'seq_lens': [seq_len] * batch_size,
        'grid_sizes': [(lat_t, lat_h, lat_w)],
    }


def warmup_model(
    model: nn.Module,
    dummy_inputs: Dict[str, torch.Tensor],
    num_warmup_runs: int = 2,
    verbose: bool = True,
    catch_errors: bool = True,
) -> Dict[str, Any]:
    """
    Run warmup forward passes to trigger all JIT compilations.
    
    Args:
        model: The diffusion transformer to warmup
        dummy_inputs: Dict of dummy tensors from create_dummy_inputs
        num_warmup_runs: Number of warmup iterations
        verbose: Print warmup progress
        catch_errors: If True, catch and report errors without raising
    
    Returns:
        Dict with warmup statistics
    """
    stats = {
        'success': False,
        'runs_completed': 0,
        'errors': [],
        'memory_before_mb': 0,
        'memory_after_mb': 0,
    }
    
    if verbose:
        print("[Warmup] Starting model warmup...")
    
    # Record initial memory
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        stats['memory_before_mb'] = torch.cuda.memory_allocated() / 1024**2
    
    model.eval()
    
    with torch.no_grad():
        for i in range(num_warmup_runs):
            try:
                # Attempt forward pass
                # The exact call depends on model architecture
                # Try common interfaces
                if hasattr(model, 'forward'):
                    _ = model(**dummy_inputs)
                elif callable(model):
                    _ = model(**dummy_inputs)
                
                stats['runs_completed'] += 1
                
                if verbose:
                    print(f"[Warmup] Run {i+1}/{num_warmup_runs} completed")
                    
            except Exception as e:
                error_msg = f"Run {i+1} failed: {str(e)}"
                stats['errors'].append(error_msg)
                if verbose:
                    print(f"[Warmup] {error_msg}")
                if not catch_errors:
                    raise
    
    # Sync and record final memory
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        stats['memory_after_mb'] = torch.cuda.memory_allocated() / 1024**2
    
    # Force garbage collection to free warmup tensors
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    stats['success'] = stats['runs_completed'] > 0
    
    if verbose:
        print(f"[Warmup] Completed {stats['runs_completed']}/{num_warmup_runs} runs")
        print(f"[Warmup] Memory: {stats['memory_before_mb']:.1f} MB -> {stats['memory_after_mb']:.1f} MB")
    
    return stats


def warmup_transformer_block(
    block: nn.Module,
    hidden_dim: int,
    seq_len: int,
    num_heads: int,
    device: torch.device,
    dtype: torch.dtype,
    verbose: bool = False,
) -> bool:
    """
    Warmup a single transformer block (attention + FFN).
    
    Useful for block-by-block warmup or testing.
    
    Args:
        block: Transformer block module
        hidden_dim: Hidden dimension
        seq_len: Sequence length
        num_heads: Number of attention heads
        device: Target device
        dtype: Target dtype
        verbose: Print progress
    
    Returns:
        True if warmup succeeded
    """
    try:
        x = torch.randn(1, seq_len, hidden_dim, device=device, dtype=dtype)
        
        # Try to call block forward
        with torch.no_grad():
            if hasattr(block, 'forward'):
                # May need additional args depending on block type
                _ = block(x)
        
        if verbose:
            print(f"[Warmup] Block {block.__class__.__name__} warmed up")
        return True
        
    except Exception as e:
        if verbose:
            print(f"[Warmup] Block warmup failed: {e}")
        return False


class WarmupContext:
    """
    Context manager for model warmup with automatic cleanup.
    
    Usage:
        with WarmupContext(model, dummy_inputs) as ctx:
            # Model is now warmed up
            output = model(real_inputs)
        print(ctx.stats)
    """
    
    def __init__(
        self,
        model: nn.Module,
        dummy_inputs: Optional[Dict[str, torch.Tensor]] = None,
        num_warmup_runs: int = 2,
        verbose: bool = True,
    ):
        self.model = model
        self.dummy_inputs = dummy_inputs
        self.num_warmup_runs = num_warmup_runs
        self.verbose = verbose
        self.stats = {}
    
    def __enter__(self):
        if self.dummy_inputs is not None:
            self.stats = warmup_model(
                self.model,
                self.dummy_inputs,
                self.num_warmup_runs,
                self.verbose,
            )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return False


def estimate_warmup_shapes(
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Estimate shapes needed for warmup from model config.
    
    Args:
        config: Model configuration dict with keys like:
            - hidden_dim, num_heads, num_layers
            - max_frames, max_height, max_width
    
    Returns:
        Dict with estimated shape parameters
    """
    hidden_dim = config.get('hidden_dim', 3072)
    num_heads = config.get('num_heads', 40)
    
    # Use minimum practical sizes for warmup
    # (enough to trigger compilation, not too big)
    return {
        'batch_size': 1,
        'num_frames': 21,  # Typical frame window
        'height': 256,     # Small but valid size
        'width': 256,
        'hidden_dim': hidden_dim,
        'context_len': 512,  # Typical context length
        'num_heads': num_heads,
    }


# Integration with existing codebase
def integrate_warmup_hook(sampler_class):
    """
    Decorator to add automatic warmup to a sampler class.
    
    Usage:
        @integrate_warmup_hook
        class WanVideoSampler:
            ...
    """
    original_process = sampler_class.process
    
    def process_with_warmup(self, *args, **kwargs):
        model = kwargs.get('model') or (args[0] if args else None)
        
        if model is not None and not getattr(model, '_warmup_done', False):
            # Get shapes from args
            # This is model-specific, simplified here
            try:
                device = next(model.parameters()).device
                dtype = next(model.parameters()).dtype
                
                dummy = create_dummy_inputs(
                    batch_size=1,
                    num_frames=21,
                    height=512,
                    width=512,
                    hidden_dim=3072,
                    context_len=512,
                    device=device,
                    dtype=dtype,
                )
                
                warmup_model(model, dummy, num_warmup_runs=1, verbose=True)
                model._warmup_done = True
                
            except Exception as e:
                print(f"[Warmup] Auto-warmup failed: {e}")
        
        return original_process(self, *args, **kwargs)
    
    sampler_class.process = process_with_warmup
    return sampler_class
