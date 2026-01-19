ComfyUI-WanVideoWrapper
=======================

> High-performance ComfyUI wrapper nodes for [WanVideo](https://github.com/Wan-Video/Wan2.1) with extensive runtime optimizations.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)


Installation
------------

1. Clone this repo into `custom_nodes` folder:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/kijai/ComfyUI-WanVideoWrapper
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
   For portable installs (Windows):
   ```bash
   python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-WanVideoWrapper\requirements.txt
   ```

3. **(Optional)** Install ninja for CUDA kernel compilation:
   ```bash
   pip install ninja
   ```


Models
------

Download models from: https://huggingface.co/Kijai/WanVideo_comfy/tree/main

**Recommended fp8 scaled models:** https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled

| Model Type | Location |
| ---------- | -------- |
| Text encoders | `ComfyUI/models/text_encoders` |
| CLIP vision | `ComfyUI/models/clip_vision` |
| Transformer | `ComfyUI/models/diffusion_models` |
| VAE | `ComfyUI/models/vae` |


Benchmark Results
-----------------

### Audio-Guided Video Generation (InfiniteTalk/MultiTalk)

**Test Configuration:** 512×512, 5 seconds (120 frames)

```
Environment: Google Colab
Models: Wan2.1-I2V-14B (fp8), LightX2V LoRA
Audio: 5-second speech clip
LoRA Mode: Streaming (default, faster than On-the-Fly)
```

#### NVIDIA A100 (80GB VRAM) — 3 Steps (Same Quality)

| Version | Total Time | Speedup |
| ------- | ---------- | ------- |
| Base    | 1.5 min    | —       |
| **Optimized** | **1.5 min** | Same (no regression) |

#### NVIDIA A100 (80GB VRAM) — 2 Steps (Speed Priority)

| Version | Total Time | Speedup |
| ------- | ---------- | ------- |
| Base    | 3.2 min    | —       |
| **Optimized** | **2.8 min** | **12% faster** |

#### NVIDIA L4 (22GB VRAM) — 2 Steps

| Version | Total Time | Speedup |
| ------- | ---------- | ------- |
| Base    | 7.8 min    | —       |
| **Optimized** | **6.4 min** | **18% faster** |


### LoRA Mode Comparison (A100)

| Mode | Memory | Speed | Recommended For |
| ---- | ------ | ----- | --------------- |
| **Streaming** (default) | Higher | **Faster** | A100, H100, high-VRAM GPUs |
| On-the-Fly | Lower | Slower | L4, consumer GPUs with OOM |
| A/B cache hit rate | 0% | 91.7% | — |
| LoRA overhead per window | ~40s | ~3s | **92% reduction** |


Summary of Optimizations
------------------------

### Smart LoRA Mode Selection

- **Streaming Mode (Default, Fastest)** — Builds merged delta `ΔW = Σ(scale × A @ B)` incrementally and caches it. Forward pass uses single `F.linear(x, W + ΔW)` matmul. **Recommended for high-VRAM GPUs (A100, H100).**

- **On-the-Fly Mode (Lower Memory)** — Computes `W@x + Σ(scale × A @ (B @ x))` directly without materializing delta. Uses 3 matmuls instead of 1, so **slower but uses less peak memory**. Auto-disabled on high-VRAM GPUs unless explicitly requested via `WAN_LORA_ONTHEFLY=1`.

- **A/B GPU Caching** — When using On-the-Fly mode, caches LoRA A and B matrices on GPU with `non_blocking=True` transfers. Achieves 91.7% cache hit rate, reducing CPU→GPU transfers from ~30,000 to ~482.

```python
# Streaming Mode (DEFAULT - FASTER):
# Single matmul with pre-computed delta
out = F.linear(input, weight + cached_delta, bias)  # 1 matmul

# On-the-Fly Mode (OPTIONAL - LOWER MEMORY):
# Three matmuls, no delta materialization  
out = F.linear(input, weight, bias)  # matmul 1
Bx = torch.mm(x_flat, B_flat_t)      # matmul 2
out += torch.mm(Bx, A_flat.t()) * s  # matmul 3
```

### Hot Path Micro-Optimizations

- **Simplified Cache Keys** — Reduced cache key from `(device_type, device_index, dtype)` tuple to `(dtype,)` for faster hash computation and lookup in the critical LoRA application loop.

- **Single-Lookup Pattern** — Changed `if key in dict` followed by `dict[key]` to single `dict.get(key)` call, eliminating redundant hash computations in the inner loop executed ~10,000 times per generation.

- **Tuple Storage** — Replaced dictionary storage `{'A_flat': ..., 'B_flat_t': ...}` with tuples `(A_flat, B_flat_t, alpha)` for faster unpacking and reduced memory overhead.

- **Local Variable Caching** — Caches `self.lora_diffs`, `self._lora_ab_cache`, and strength tensors into local variables to reduce attribute lookup overhead in Python's `__getattr__` chain.

- **Strength Tensor Caching** — Pre-caches strength tensor references in `set_lora_strengths()` to avoid repeated `getattr()` with f-string formatting (`f"_lora_strength_{i}"`) in the hot loop.

### Architecture-Aware GPU Detection

- **Split Detection (`GPUConfig`)** — Silent detection at import time applies BF16 optimization before first matmul (critical for A100 performance), while logging is deferred until audio detection to keep imports clean.

- **Compute Capability Detection** — Identifies GPU architecture by SM version:
  - SM 8.0 → Ampere datacenter (A100)
  - SM 8.9 → Ada Lovelace (L4, RTX 4090)
  - SM 9.0 → Hopper (H100)

- **Architecture-Specific Optimizations:**

| GPU | Architecture | BF16 Reduced Precision | Scheduler CUDA Graphs |
| --- | ------------ | ---------------------- | --------------------- |
| A100 | Ampere DC | ✅ Enabled | ✅ Enabled |
| H100 | Hopper | ✅ Enabled | ✅ Enabled |
| L4 | Ada | ❌ Disabled | ❌ Disabled |
| RTX 3090 | Ampere | ✅ Enabled | ✅ If VRAM ≥24GB |

### Custom CUDA Kernels

- **Fused SiLU×Gate Kernel (`fused_ops.cu`)** — Single-pass computation of `SiLU(x) × gate` with half2 vectorization for FP16, processing two elements per thread for improved memory bandwidth utilization.

- **Fused RMSNorm Kernel** — Warp-level parallel reduction for variance computation with `__shfl_down_sync()`, avoiding global memory round-trips for intermediate results.

- **Silent Fallback** — Automatic fallback to PyTorch operations if CUDA kernel compilation fails (missing ninja, incompatible CUDA version), ensuring zero regressions.

```cuda
// Vectorized FP16 processing - 2 elements per thread
template <>
__global__ void fused_silu_mul_kernel<half>(...) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N / 2) {
        half2 x_val = ((half2*)x)[idx];
        half2 gate_val = ((half2*)gate)[idx];
        half2 silu_x = /* vectorized SiLU */;
        ((half2*)out)[idx] = __hmul2(silu_x, gate_val);
    }
}
```

### Memory Management

- **CUDA Allocator Configuration** — Sets `expandable_segments:True` and `garbage_collection_threshold:0.8` via `PYTORCH_CUDA_ALLOC_CONF` to reduce memory fragmentation and OOM risk during long video generation.

- **Selective Graph Disabling** — For audio-guided modes with variable-length inputs, disables only transformer CUDA graphs while keeping scheduler graphs enabled on high-VRAM GPUs (A100/H100), reducing kernel launch overhead for the fixed-shape scheduler step.

### Attention Optimizations

- **Flash SDP** — Enables `torch.backends.cuda.enable_flash_sdp(True)` for fused attention kernels on Ampere+ architectures.

- **Memory-Efficient SDP** — Enables `torch.backends.cuda.enable_mem_efficient_sdp(True)` as fallback for sequences exceeding Flash Attention limits.

- **cuDNN Benchmark** — Enables `torch.backends.cudnn.benchmark = True` for automatic convolution algorithm selection, with `deterministic = False` for maximum performance.


Environment Variables
---------------------

| Variable | Default | Description |
| -------- | ------- | ----------- |
| `WAN_LORA_ONTHEFLY` | `0` | Punica-style LoRA (lower memory, slower). Only enable if OOM. |
| `WAN_LORA_STREAMING` | `1` | Streaming delta merge (faster, default). Recommended. |
| `WAN_LORA_TIMING` | `0` | Print LoRA timing statistics after generation |
| `MAX_STEPS` | `3` | Diffusion steps (2 for speed, 3+ for quality) |
| `MOTION_FRAME` | `25` | Frame overlap between windows (lower = faster) |
| `WAN_TORCH_COMPILE` | `0` | Enable torch.compile (experimental, adds warmup) |


Optimized Run Script Example
----------------------------

```python
import os

# CUDA memory optimization (MUST be before torch import)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.8"

# Speed optimizations
os.environ["MAX_STEPS"] = "2"           # Reduced from 3 (saves ~33%)
os.environ["MOTION_FRAME"] = "5"        # Reduced overlap (fewer windows)
os.environ["WAN_LORA_ONTHEFLY"] = "1"   # Punica-style LoRA
os.environ["WAN_LORA_TIMING"] = "1"     # Show timing stats

# Video settings
os.environ["WIDTH"] = "512"
os.environ["HEIGHT"] = "512"
os.environ["FRAME_RATE"] = "24"
os.environ["MAX_FRAMES"] = "120"        # 5 seconds
```


Supported Models
----------------

### Core WanVideo Models
- WanVideo 2.1 T2V/I2V (1.3B, 14B)
- WanVideo 2.2 Animate

### Audio-Guided Generation
- [InfiniteTalk](https://github.com/MeiGen-AI/MultiTalk) — Single-speaker talking photo
- [MultiTalk](https://github.com/MeiGen-AI/MultiTalk) — Multi-speaker conversation
- [FantasyTalking](https://github.com/Fantasy-AMAP/fantasy-talking) — Expression-driven animation
- [HuMo](https://github.com/Phantom-video/HuMo) — Human motion synthesis

### Camera & Motion Control
- [ReCamMaster](https://github.com/KwaiVGI/ReCamMaster) — Camera trajectory control
- [SkyReels](https://huggingface.co/collections/Skywork/skyreels-v2-6801b1b93df627d441d0d0d9) — Cinematic camera moves
- [MoCha](https://github.com/Orange-3DV-Team/MoCha) — Motion character control

### Subject Consistency
- [Phantom](https://huggingface.co/bytedance-research/Phantom) — Identity preservation
- [EchoShot](https://github.com/D2I-ai/EchoShot) — Subject reference
- [Stand-In](https://github.com/WeChatCV/Stand-In) — Actor replacement
- [MAGREF](https://huggingface.co/MAGREF-Video/MAGREF) — Reference-guided generation

### Video Editing
- [VACE](https://github.com/ali-vilab/VACE) — Video editing framework
- [MiniMaxRemover](https://huggingface.co/zibojia/minimax-remover) — Object removal
- [WanVideoFun](https://huggingface.co/collections/alibaba-pai/wan21-fun-v11-680f514c89fe7b4df9d44f17) — Extended features

### Conditioning & Control
- [ATI](https://huggingface.co/bytedance-research/ATI) — Attention-based injection
- [Uni3C](https://github.com/alibaba-damo-academy/Uni3C) — 3D-consistent generation
- [UniLumos](https://github.com/alibaba-damo-academy/Lumos-Custom) — Lighting control
- [Lynx](https://github.com/bytedance/lynx) — Layout control
- [Bindweave](https://github.com/bytedance/BindWeave) — Compositional binding

### Training-Free Techniques
- [TimeToMove](https://github.com/time-to-move/TTM) — Motion transfer
- [SteadyDancer](https://github.com/MCG-NJU/SteadyDancer) — Dance stabilization
- [One-to-All-Animation](https://github.com/ssj9596/One-to-All-Animation) — Single-image animation
- [SCAIL](https://github.com/zai-org/SCAIL) — Scale-consistent generation

### Extended Architecture
- [LongCat-Video](https://meituan-longcat.github.io/LongCat-Video/) — Extended context


Troubleshooting
---------------

### VRAM Issues with torch.compile

After updates modifying model code, torch.compile may exhibit increased first-run memory usage due to stale Triton caches. Clear caches:

**Windows:**
```
del /s /q C:\Users\<username>\.triton\*
del /s /q C:\Users\<username>\AppData\Local\Temp\torchinductor_<username>\*
```

**Linux/macOS:**
```bash
rm -rf ~/.triton/cache
rm -rf /tmp/torchinductor_*
```

### LoRA Memory Overhead

LoRA weights are now assigned as module buffers for unified offloading. If not using block swap, expect increased VRAM usage. With block swap enabled:

```
Additional VRAM ≈ (LoRA size) × (swapped blocks) / (total blocks)
Example: 1GB LoRA, 20/40 blocks swapped → ~500MB additional
Compensation: Swap 2 additional blocks
```

### CUDA Kernel Compilation

If you see `[WanVideo] CUDA kernels compiled successfully`, custom kernels are active. If compilation fails, the code silently falls back to PyTorch operations with zero performance regression.

To enable CUDA kernels, ensure `ninja` is installed:
```bash
pip install ninja
```


License
-------

This project is licensed under the MIT License.


Acknowledgments
---------------

- [WanVideo](https://github.com/Wan-Video/Wan2.1) — Base video generation model
- [Kijai](https://github.com/kijai) — Original ComfyUI wrapper implementation
- [Punica](https://arxiv.org/abs/2310.18547) — Inspiration for on-the-fly LoRA computation