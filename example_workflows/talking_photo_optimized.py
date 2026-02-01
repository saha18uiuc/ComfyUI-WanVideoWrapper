#!/usr/bin/env python3
"""
Optimized Talking Photo Generation Script for WanVideoWrapper

This script runs the talking photo pipeline with all math-heavy optimizations:
1. Low-rank LoRA (RunLoRA-style activation-based)
2. Cross-attention K/V caching
3. Fused Triton kernels (RMSNorm, SwiGLU)
4. Token Merging (ToMe) for 2x attention speedup
5. CFG optimizations

Expected speedup: 30-50% faster than baseline

Usage:
    python talking_photo_optimized.py
    
Environment variables for optimization control:
    WAN_OPT_LOWRANK_LORA=1    - Low-rank LoRA (default: on)
    WAN_OPT_KV_CACHE=1        - K/V caching (default: on)
    WAN_OPT_TRITON_RMSNORM=1  - Fused RMSNorm (default: on)
    WAN_OPT_TRITON_SWIGLU=1   - Fused SwiGLU (default: on)
    WAN_OPT_TOME=1            - Token Merging (default: on in this script)
    WAN_OPT_TOME_RATIO=0.25   - Fraction to merge (default: 0.25)
    WAN_OPT_VERBOSE=1         - Verbose logging (default: off)
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# =============================================================================
# MEMORY OPTIMIZATION
# =============================================================================
alloc_conf = "expandable_segments:True,garbage_collection_threshold:0.8"
os.environ["PYTORCH_ALLOC_CONF"] = alloc_conf
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = alloc_conf

# =============================================================================
# PATHS
# =============================================================================
WORKFLOW_PATH = Path("/content/workflow_api.py")
COMFY_DIR = Path("/content/ComfyUI")
OUTPUT_DIR = COMFY_DIR / "output"

# =============================================================================
# VIDEO SETTINGS
# =============================================================================
FPS = 24
DURATION_S = 5
MAX_FRAMES = FPS * DURATION_S  # 120 frames

os.environ["FRAME_RATE"] = str(FPS)
os.environ["MAX_FRAMES"] = str(MAX_FRAMES)
os.environ["WIDTH"] = "512"
os.environ["HEIGHT"] = "512"
os.environ["FRAME_WINDOW_SIZE"] = "25"
os.environ["MAX_STEPS"] = "3"

# Audio/visual settings
os.environ["AUDIO_SCALE_STRENGTH"] = "2"
os.environ["WAN_LORA_TIMING"] = "1"

# =============================================================================
# MATH-HEAVY OPTIMIZATIONS (NEW)
# =============================================================================
# These control the new optimization modules in optimizations/

# 1. Low-rank LoRA: Apply LoRA as activations instead of dense weight delta
#    Speedup: 1.5-3x on LoRA layers
os.environ["WAN_OPT_LOWRANK_LORA"] = "1"

# 2. K/V Caching: Cache cross-attention K/V for constant conditioning
#    Speedup: 5-10% overall (saves 2 GEMMs per layer per step)
os.environ["WAN_OPT_KV_CACHE"] = "1"

# 3. Triton Kernels: Fused RMSNorm and SwiGLU
#    Speedup: 5-10% on these operations
os.environ["WAN_OPT_TRITON_RMSNORM"] = "1"
os.environ["WAN_OPT_TRITON_SWIGLU"] = "1"

# 4. Token Merging (ToMe): Merge similar tokens before self-attention
#    Speedup: ~2x on self-attention (biggest single optimization)
#    Set to 0 if you notice quality degradation
os.environ["WAN_OPT_TOME"] = "1"
os.environ["WAN_OPT_TOME_RATIO"] = "0.25"  # Merge 25% of tokens (conservative)

# 5. Verbose logging (set to 1 to see which optimizations are applied)
os.environ["WAN_OPT_VERBOSE"] = "0"

# =============================================================================
# MODEL PATHS
# =============================================================================
os.environ["INFINATE_TALK_MODEL"] = "Wan2_1-InfiniTetalk-Single_fp16.safetensors"
os.environ["VAE_MODEL"] = "Wan2_1_VAE_bf16.safetensors"
os.environ["LORA_MODEL"] = "Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors"
os.environ["CLIP_NAME"] = "clip_vision_h.safetensors"
os.environ["TEXT_ENCODE_MODEL"] = "umt5-xxl-enc-bf16.safetensors"
os.environ["WAN_MODEL"] = "Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors"
os.environ["POS_PROMPT"] = ""

# =============================================================================
# INPUT FILES
# =============================================================================
ref_audio_candidates = [
    Path("/content/audio.mp3"),
    Path("/content/ComfyUI/input/audio.mp3")
]
ref_image_candidates = [
    Path("/content/512.jpg"),
    Path("/content/448.jpg"),
    Path("/content/ComfyUI/input/512.jpg")
]

ref_audio = next((p for p in ref_audio_candidates if p.is_file()), None)
ref_image = next((p for p in ref_image_candidates if p.is_file()), None)

if ref_audio is None:
    raise FileNotFoundError(
        "Missing audio file.\n"
        "Put your audio at /content/audio.mp3\n"
        f"Tried: {[str(p) for p in ref_audio_candidates]}"
    )
if ref_image is None:
    raise FileNotFoundError(
        "Missing image file.\n"
        "Put your image at /content/512.jpg\n"
        f"Tried: {[str(p) for p in ref_image_candidates]}"
    )

os.environ["REF_AUDIO"] = str(ref_audio)
os.environ["REF_IMAGE"] = str(ref_image)

# =============================================================================
# SANITY CHECKS
# =============================================================================
if not WORKFLOW_PATH.is_file():
    raise FileNotFoundError(f"Workflow file not found: {WORKFLOW_PATH}")
if not COMFY_DIR.is_dir():
    raise FileNotFoundError(f"ComfyUI directory not found: {COMFY_DIR}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# RUN
# =============================================================================
print("=" * 70)
print("OPTIMIZED TALKING PHOTO GENERATION")
print("=" * 70)
print()
print(f"Video: {DURATION_S}s @ {FPS}fps = {MAX_FRAMES} frames, {os.environ['WIDTH']}x{os.environ['HEIGHT']}")
print(f"Steps: {os.environ['MAX_STEPS']}, Window: {os.environ['FRAME_WINDOW_SIZE']} frames")
print()
print("Optimizations enabled:")
print(f"  - Low-rank LoRA:    {os.environ.get('WAN_OPT_LOWRANK_LORA', '1') == '1'}")
print(f"  - K/V Caching:      {os.environ.get('WAN_OPT_KV_CACHE', '1') == '1'}")
print(f"  - Triton RMSNorm:   {os.environ.get('WAN_OPT_TRITON_RMSNORM', '1') == '1'}")
print(f"  - Triton SwiGLU:    {os.environ.get('WAN_OPT_TRITON_SWIGLU', '1') == '1'}")
print(f"  - Token Merging:    {os.environ.get('WAN_OPT_TOME', '0') == '1'} (ratio={os.environ.get('WAN_OPT_TOME_RATIO', '0.25')})")
print()
print(f"Audio: {ref_audio}")
print(f"Image: {ref_image}")
print("=" * 70)

# Update to latest optimizations branch
wrapper_dir = COMFY_DIR / "custom_nodes" / "ComfyUI-WanVideoWrapper"
if wrapper_dir.is_dir():
    subprocess.run(
        ["git", "-C", str(wrapper_dir), "pull", "origin", "math-heavy-optimizations"],
        capture_output=True
    )

# Run the workflow
t0 = time.perf_counter()
try:
    completed = subprocess.run(
        ["python", str(WORKFLOW_PATH)],
        cwd=str(WORKFLOW_PATH.parent),
        check=True,
        text=True,
        capture_output=True,
    )
    if completed.stdout:
        print(completed.stdout)
    if completed.stderr:
        print(completed.stderr)
except subprocess.CalledProcessError as e:
    print(f"FAILED with return code: {e.returncode}")
    print("\n----- STDOUT -----\n", e.stdout or "(empty)")
    print("\n----- STDERR -----\n", e.stderr or "(empty)")
    raise
t1 = time.perf_counter()

elapsed_min = (t1 - t0) / 60
print()
print("=" * 70)
print(f"TOTAL TIME: {elapsed_min:.2f} minutes ({t1-t0:.1f} seconds)")
print("=" * 70)

# Find and download output
mp4s = sorted(OUTPUT_DIR.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
audio_mp4s = [p for p in mp4s if "audio" in p.name.lower()]
target = audio_mp4s[-1] if audio_mp4s else (mp4s[-1] if mp4s else None)

print(f"Output: {target}")

# Download in Colab
try:
    from google.colab import files
    if target:
        files.download(str(target))
except ImportError:
    pass  # Not in Colab
