# --- OPTIMIZED TALKING PHOTO ---
# Uses only ZERO-OVERHEAD optimizations that don't add cold-start time
# Expected: ~10-15% speedup with no compilation delay

import os
import sys
import subprocess
import time
from pathlib import Path

# Memory optimization
alloc_conf = "expandable_segments:True,garbage_collection_threshold:0.8"
os.environ["PYTORCH_ALLOC_CONF"] = alloc_conf
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = alloc_conf

# Paths
WORKFLOW_PATH = Path("/content/workflow_api.py")
COMFY_DIR = Path("/content/ComfyUI")
OUTPUT_DIR = COMFY_DIR / "output"

# =============================================================================
# VIDEO SETTINGS
# =============================================================================
FPS = 24
DURATION_S = 5
MAX_FRAMES = FPS * DURATION_S

os.environ["FRAME_RATE"] = str(FPS)
os.environ["MAX_FRAMES"] = str(MAX_FRAMES)
os.environ["WIDTH"] = "512"
os.environ["HEIGHT"] = "512"
os.environ["FRAME_WINDOW_SIZE"] = "25"
os.environ["MAX_STEPS"] = "3"
os.environ["AUDIO_SCALE_STRENGTH"] = "2"
os.environ["WAN_LORA_TIMING"] = "1"

# =============================================================================
# ZERO-OVERHEAD OPTIMIZATIONS (no cold-start cost)
# =============================================================================
# Only enabling BATCHED_CFG - the ONLY optimization with zero cold-start

# BATCHED_CFG: Batches conditional + unconditional in single forward pass
# ~10-15% speedup, EXACT same output, NO compilation overhead
os.environ["BATCHED_CFG"] = "1"

# =============================================================================
# DISABLED: These have cold-start costs that negate speedups
# =============================================================================
# TORCH_COMPILE: Adds 30-60s compilation time - NOT worth it for single runs
# os.environ["TORCH_COMPILE"] = "1"  # DISABLED - cold start too high

# SMOOTH_CACHE: Requires similarity computation overhead
# os.environ["SMOOTH_CACHE"] = "1"  # DISABLED - overhead for short runs

# FUSED_KERNELS: Triton kernel compilation takes time
# os.environ["FUSED_KERNELS"] = "1"  # DISABLED - compilation overhead

# =============================================================================
# MODELS
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
ref_audio = next((p for p in [Path("/content/audio.mp3"), Path("/content/ComfyUI/input/audio.mp3")] if p.is_file()), None)
ref_image = next((p for p in [Path("/content/512.jpg"), Path("/content/ComfyUI/input/512.jpg")] if p.is_file()), None)
if not ref_audio: raise FileNotFoundError("Missing audio at /content/audio.mp3")
if not ref_image: raise FileNotFoundError("Missing image at /content/512.jpg")
os.environ["REF_AUDIO"] = str(ref_audio)
os.environ["REF_IMAGE"] = str(ref_image)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# RUN
# =============================================================================
print("=" * 70)
print("OPTIMIZED TALKING PHOTO GENERATION")
print("=" * 70)
print(f"{DURATION_S}s @ {FPS}fps = {MAX_FRAMES} frames, 512x512, 3 steps")
print()
print("Active Optimizations (zero cold-start):")
print("  [x] BATCHED_CFG - Batch cond+uncond (~10-15% faster)")
print()
print("Disabled (cold-start too expensive):")
print("  [ ] TORCH_COMPILE - 30-60s compilation overhead")
print("  [ ] SMOOTH_CACHE - Similarity computation overhead")
print("  [ ] FUSED_KERNELS - Triton compilation overhead")
print()
print("Expected: ~5.5-5.8 min (vs ~6.4 min baseline)")
print("=" * 70)

# Pull latest optimizations
subprocess.run(["git", "-C", str(COMFY_DIR / "custom_nodes" / "ComfyUI-WanVideoWrapper"), 
                "pull", "origin", "math-heavy-optimizations"], capture_output=True)

t0 = time.perf_counter()
try:
    completed = subprocess.run(["python", str(WORKFLOW_PATH)], cwd=str(WORKFLOW_PATH.parent), 
                               check=True, text=True, capture_output=True)
    print(completed.stdout)
    if completed.stderr: print(completed.stderr)
except subprocess.CalledProcessError as e:
    print(f"FAILED: {e.returncode}\n{e.stdout}\n{e.stderr}")
    raise
t1 = time.perf_counter()

elapsed_min = (t1 - t0) / 60
print("=" * 70)
print(f"TOTAL: {elapsed_min:.2f} minutes")
if elapsed_min < 6.4:
    saved = 6.4 - elapsed_min
    pct = (saved / 6.4) * 100
    print(f"SAVED: {saved:.2f} min ({pct:.1f}% faster than baseline)")
print("=" * 70)

mp4s = sorted(OUTPUT_DIR.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
target = next((p for p in reversed(mp4s) if "audio" in p.name.lower()), mp4s[-1] if mp4s else None)
print("Output:", target)

try:
    from google.colab import files
    if target: files.download(str(target))
except ImportError:
    pass  # Not in Colab
