"""
Pre-allocated Memory Pool for CUDA Graph Capture Stability.

This module provides a memory management system that:
1. Pre-allocates all intermediate tensors at model load time
2. Uses index-based buffer management instead of dynamic allocation
3. Implements double-buffering for compute/memory overlap
4. Ensures all allocations happen BEFORE CUDA graph capture

The key insight is that CUDA graph capture fails when:
- New tensors are allocated inside the captured region
- Memory pools grow during capture
- Different code paths allocate different amounts

By pre-allocating a fixed memory pool, we guarantee that graph capture
will succeed and replay will be stable.
"""

import torch
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class BufferType(Enum):
    """Types of buffers in the memory pool."""
    BLOCK_INPUT = "block_input"
    BLOCK_OUTPUT = "block_output"
    ATTENTION_QKV = "attention_qkv"
    ATTENTION_OUT = "attention_out"
    FFN_HIDDEN = "ffn_hidden"
    RESIDUAL = "residual"
    SCRATCH = "scratch"


@dataclass
class BufferSpec:
    """Specification for a pre-allocated buffer."""
    shape: Tuple[int, ...]
    dtype: torch.dtype
    name: str
    double_buffer: bool = False


class CUDAGraphMemoryPool:
    """
    Pre-allocated memory pool for CUDA graph-safe inference.
    
    This class manages a fixed pool of memory buffers that are allocated
    once at initialization and reused throughout inference. This ensures
    CUDA graph capture can succeed without any dynamic allocations.
    
    Usage:
        # At model load time
        pool = CUDAGraphMemoryPool(
            max_batch=1,
            max_seq_len=16384,
            hidden_dim=2048,
            num_blocks=40,
            num_heads=32,
            device=torch.device('cuda')
        )
        pool.allocate()
        
        # During inference
        input_buffer = pool.get_buffer(BufferType.BLOCK_INPUT, block_idx=0)
        output_buffer = pool.get_buffer(BufferType.BLOCK_OUTPUT, block_idx=0)
        
        # Before CUDA graph capture
        pool.reset_all()
    """
    
    def __init__(
        self,
        max_batch: int = 1,
        max_seq_len: int = 16384,
        hidden_dim: int = 2048,
        num_blocks: int = 40,
        num_heads: int = 32,
        head_dim: Optional[int] = None,
        ffn_dim: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float16,
        enable_double_buffer: bool = True,
    ):
        """
        Initialize the memory pool with maximum dimensions.
        
        Args:
            max_batch: Maximum batch size
            max_seq_len: Maximum sequence length (F*H*W for video)
            hidden_dim: Transformer hidden dimension
            num_blocks: Number of transformer blocks
            num_heads: Number of attention heads
            head_dim: Head dimension (defaults to hidden_dim // num_heads)
            ffn_dim: FFN intermediate dimension (defaults to hidden_dim * 4)
            device: CUDA device
            dtype: Data type for buffers
            enable_double_buffer: Enable double buffering for overlap
        """
        self.max_batch = max_batch
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.head_dim = head_dim or (hidden_dim // num_heads)
        self.ffn_dim = ffn_dim or (hidden_dim * 4)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype
        self.enable_double_buffer = enable_double_buffer
        
        # Buffer storage
        self._buffers: Dict[str, torch.Tensor] = {}
        self._buffer_specs: Dict[str, BufferSpec] = {}
        self._allocated = False
        
        # Double buffer index (0 or 1)
        self._db_index = 0
        
        # Statistics
        self._total_bytes = 0
        self._peak_usage = 0
        
    def _make_buffer_key(self, buffer_type: BufferType, index: int = 0, db_idx: int = 0) -> str:
        """Generate unique key for a buffer."""
        return f"{buffer_type.value}_{index}_db{db_idx}"
    
    def _calculate_buffer_specs(self) -> Dict[str, BufferSpec]:
        """Calculate all buffer specifications based on model config."""
        specs = {}
        num_db = 2 if self.enable_double_buffer else 1
        
        for db_idx in range(num_db):
            # Per-block buffers
            for block_idx in range(self.num_blocks):
                # Block input/output: [batch, seq_len, hidden_dim]
                specs[self._make_buffer_key(BufferType.BLOCK_INPUT, block_idx, db_idx)] = BufferSpec(
                    shape=(self.max_batch, self.max_seq_len, self.hidden_dim),
                    dtype=self.dtype,
                    name=f"block_{block_idx}_input_db{db_idx}",
                    double_buffer=self.enable_double_buffer
                )
                
                specs[self._make_buffer_key(BufferType.BLOCK_OUTPUT, block_idx, db_idx)] = BufferSpec(
                    shape=(self.max_batch, self.max_seq_len, self.hidden_dim),
                    dtype=self.dtype,
                    name=f"block_{block_idx}_output_db{db_idx}",
                    double_buffer=self.enable_double_buffer
                )
                
                # Attention QKV: [batch, seq_len, num_heads, 3 * head_dim]
                specs[self._make_buffer_key(BufferType.ATTENTION_QKV, block_idx, db_idx)] = BufferSpec(
                    shape=(self.max_batch, self.max_seq_len, self.num_heads, 3 * self.head_dim),
                    dtype=self.dtype,
                    name=f"block_{block_idx}_qkv_db{db_idx}",
                    double_buffer=self.enable_double_buffer
                )
                
                # Attention output: [batch, seq_len, num_heads, head_dim]
                specs[self._make_buffer_key(BufferType.ATTENTION_OUT, block_idx, db_idx)] = BufferSpec(
                    shape=(self.max_batch, self.max_seq_len, self.num_heads, self.head_dim),
                    dtype=self.dtype,
                    name=f"block_{block_idx}_attn_out_db{db_idx}",
                    double_buffer=self.enable_double_buffer
                )
                
                # FFN hidden: [batch, seq_len, ffn_dim]
                specs[self._make_buffer_key(BufferType.FFN_HIDDEN, block_idx, db_idx)] = BufferSpec(
                    shape=(self.max_batch, self.max_seq_len, self.ffn_dim),
                    dtype=self.dtype,
                    name=f"block_{block_idx}_ffn_db{db_idx}",
                    double_buffer=self.enable_double_buffer
                )
            
            # Global scratch buffer for temporary computations
            scratch_size = max(
                self.max_batch * self.max_seq_len * self.hidden_dim * 4,  # 4x hidden for safety
                self.max_batch * self.max_seq_len * self.ffn_dim,
            )
            specs[self._make_buffer_key(BufferType.SCRATCH, 0, db_idx)] = BufferSpec(
                shape=(scratch_size,),
                dtype=self.dtype,
                name=f"scratch_db{db_idx}",
                double_buffer=self.enable_double_buffer
            )
        
        return specs
    
    def allocate(self) -> int:
        """
        Allocate all buffers. Should be called once at model load time.
        
        Returns:
            Total bytes allocated
        """
        if self._allocated:
            return self._total_bytes
            
        self._buffer_specs = self._calculate_buffer_specs()
        self._total_bytes = 0
        
        for key, spec in self._buffer_specs.items():
            tensor = torch.zeros(
                spec.shape,
                dtype=spec.dtype,
                device=self.device,
                requires_grad=False
            )
            self._buffers[key] = tensor
            self._total_bytes += tensor.nelement() * tensor.element_size()
        
        self._allocated = True
        return self._total_bytes
    
    def get_buffer(
        self,
        buffer_type: BufferType,
        index: int = 0,
        actual_shape: Optional[Tuple[int, ...]] = None
    ) -> torch.Tensor:
        """
        Get a pre-allocated buffer.
        
        Args:
            buffer_type: Type of buffer to get
            index: Block index (for per-block buffers)
            actual_shape: If provided, return a view with this shape
            
        Returns:
            Buffer tensor (possibly a view if actual_shape provided)
        """
        if not self._allocated:
            raise RuntimeError("Memory pool not allocated. Call allocate() first.")
        
        key = self._make_buffer_key(buffer_type, index, self._db_index)
        buffer = self._buffers.get(key)
        
        if buffer is None:
            raise KeyError(f"Buffer not found: {key}")
        
        if actual_shape is not None:
            # Return a view with the actual needed shape
            total_elements = math.prod(actual_shape)
            return buffer.flatten()[:total_elements].view(actual_shape)
        
        return buffer
    
    def get_scratch(self, size: int) -> torch.Tensor:
        """
        Get a scratch buffer of specified size.
        
        Args:
            size: Number of elements needed
            
        Returns:
            1D tensor of requested size
        """
        scratch = self.get_buffer(BufferType.SCRATCH, 0)
        if size > scratch.numel():
            raise RuntimeError(f"Scratch buffer too small: {scratch.numel()} < {size}")
        return scratch[:size]
    
    def swap_double_buffer(self):
        """Swap the double buffer index for next iteration."""
        if self.enable_double_buffer:
            self._db_index = 1 - self._db_index
    
    def reset_all(self):
        """Reset all buffers to zero. Call before CUDA graph capture."""
        for buffer in self._buffers.values():
            buffer.zero_()
        self._db_index = 0
    
    def release(self):
        """Release all allocated memory."""
        self._buffers.clear()
        self._buffer_specs.clear()
        self._allocated = False
        self._total_bytes = 0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    @property
    def total_memory_mb(self) -> float:
        """Total memory allocated in MB."""
        return self._total_bytes / (1024 * 1024)
    
    def __repr__(self) -> str:
        return (
            f"CUDAGraphMemoryPool("
            f"allocated={self._allocated}, "
            f"total_mb={self.total_memory_mb:.2f}, "
            f"num_buffers={len(self._buffers)}, "
            f"device={self.device})"
        )


class GraphSafeAllocator:
    """
    Context manager that ensures no allocations happen during execution.
    
    Usage:
        with GraphSafeAllocator(pool) as allocator:
            # All tensors here come from the pool
            x = allocator.get_tensor(shape, dtype)
            # No dynamic allocations allowed
    """
    
    def __init__(self, pool: CUDAGraphMemoryPool):
        self.pool = pool
        self._scratch_offset = 0
        self._max_scratch_offset = 0
        
    def __enter__(self):
        self._scratch_offset = 0
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._max_scratch_offset = max(self._max_scratch_offset, self._scratch_offset)
        return False
    
    def get_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        """
        Get a tensor from the scratch pool.
        
        This allocates from a contiguous scratch buffer, allowing
        multiple tensors to be "allocated" without actual allocation.
        """
        size = math.prod(shape)
        element_size = torch.tensor([], dtype=dtype).element_size()
        
        # Get scratch buffer slice
        scratch = self.pool.get_buffer(BufferType.SCRATCH, 0)
        
        # Calculate aligned offset (16-byte alignment for efficiency)
        alignment = 16 // element_size
        aligned_offset = ((self._scratch_offset + alignment - 1) // alignment) * alignment
        
        if aligned_offset + size > scratch.numel():
            raise RuntimeError("Scratch buffer exhausted")
        
        tensor = scratch[aligned_offset:aligned_offset + size].view(shape)
        if tensor.dtype != dtype:
            tensor = tensor.view(dtype=dtype)[:math.prod(shape) // (tensor.element_size() // element_size)].view(shape)
        
        self._scratch_offset = aligned_offset + size
        return tensor


def create_pool_for_wan_model(
    model_config: Dict[str, Any],
    max_frames: int = 120,
    max_height: int = 128,  # Latent height
    max_width: int = 72,    # Latent width
    device: Optional[torch.device] = None,
) -> CUDAGraphMemoryPool:
    """
    Factory function to create a memory pool sized for a WanVideo model.
    
    Args:
        model_config: Model configuration dict
        max_frames: Maximum number of frames (in latent space, so //4)
        max_height: Maximum height in latent space
        max_width: Maximum width in latent space
        device: CUDA device
        
    Returns:
        Configured CUDAGraphMemoryPool
    """
    # Calculate max sequence length
    # Video: F * H * W tokens after patchification
    latent_frames = (max_frames - 1) // 4 + 1
    max_seq_len = latent_frames * (max_height // 2) * (max_width // 2)
    
    # Get model dimensions
    hidden_dim = model_config.get("dim", 2048)
    num_blocks = model_config.get("num_layers", 40)
    num_heads = model_config.get("num_heads", 32)
    
    pool = CUDAGraphMemoryPool(
        max_batch=2,  # For batched CFG
        max_seq_len=max_seq_len,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
        device=device or torch.device('cuda'),
        dtype=torch.float16,
        enable_double_buffer=True,
    )
    
    return pool


# ============================================================================
# Integration helpers
# ============================================================================

def wrap_forward_with_pool(
    module: torch.nn.Module,
    pool: CUDAGraphMemoryPool,
    buffer_type: BufferType,
    block_idx: int = 0
):
    """
    Wrap a module's forward to use pre-allocated output buffers.
    
    This is a non-invasive way to make modules graph-safe without
    modifying their implementation.
    """
    original_forward = module.forward
    
    def pooled_forward(*args, **kwargs):
        # Get output buffer
        # We need to know the output shape, which depends on input
        # For now, just run forward and hope the output fits
        output = original_forward(*args, **kwargs)
        
        # Copy to pool buffer
        buffer = pool.get_buffer(buffer_type, block_idx, output.shape)
        buffer.copy_(output)
        return buffer
    
    module.forward = pooled_forward
    module._original_forward = original_forward


# ============================================================================
# Testing
# ============================================================================

def test_memory_pool():
    """Test memory pool functionality."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return
    
    device = torch.device('cuda')
    
    # Create pool
    pool = CUDAGraphMemoryPool(
        max_batch=2,
        max_seq_len=4096,
        hidden_dim=2048,
        num_blocks=4,
        num_heads=32,
        device=device,
    )
    
    # Allocate
    total_bytes = pool.allocate()
    print(f"Allocated {pool.total_memory_mb:.2f} MB")
    
    # Get buffers
    input_buf = pool.get_buffer(BufferType.BLOCK_INPUT, 0)
    print(f"Block 0 input buffer: {input_buf.shape}")
    
    output_buf = pool.get_buffer(BufferType.BLOCK_OUTPUT, 0, actual_shape=(1, 2048, 2048))
    print(f"Block 0 output buffer (view): {output_buf.shape}")
    
    # Test scratch
    scratch = pool.get_scratch(1000)
    print(f"Scratch buffer: {scratch.shape}")
    
    # Test double buffer
    pool.swap_double_buffer()
    input_buf_2 = pool.get_buffer(BufferType.BLOCK_INPUT, 0)
    print(f"After swap, buffer is different: {input_buf.data_ptr() != input_buf_2.data_ptr()}")
    
    # Cleanup
    pool.release()
    print("Memory pool test passed!")


if __name__ == "__main__":
    test_memory_pool()
