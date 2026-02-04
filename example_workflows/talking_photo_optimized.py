# --- OPTIMIZED TALKING PHOTO with CFG-Free Inference ---
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
# SPEED OPTIMIZATION: CFG-FREE INFERENCE (~2x speedup!)
# =============================================================================
# The distilled LoRA (lightx2v_cfg_step_distill) was trained for CFG=1.0
# This skips the unconditional pass entirely: 2 forward passes -> 1
#
# CFG=1.0: ~3-4 min (fastest, distilled model handles it)
# CFG=1.5: ~4-5 min (slight quality boost if needed)
# CFG=7.5: ~6-7 min (baseline, full CFG)
os.environ["CFG_SCALE"] = "1.0"

# Audio CFG stays higher to maintain lip-sync quality
os.environ["AUDIO_CFG_SCALE"] = "2.0"

# =============================================================================
# MATH-HEAVY OPTIMIZATIONS (DISABLED - caused 2x slowdown!)
# =============================================================================
# Testing showed these cause regressions on small batch sizes (CFG batch=2).
# The overhead of Python/CUDA dispatch exceeds compute savings.
# Keeping them disabled until batch-size-aware implementation.
os.environ["WAN_OPT_LOWRANK_LORA"] = "0"
os.environ["WAN_OPT_KV_CACHE"] = "0"
os.environ["WAN_OPT_TRITON_RMSNORM"] = "0"
os.environ["WAN_OPT_TRITON_SWIGLU"] = "0"
os.environ["WAN_OPT_TOME"] = "0"
os.environ["WAN_OPT_VERBOSE"] = "0"

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
cfg_val = os.environ.get("CFG_SCALE", "1.0")
print("=" * 60)
print("CFG-FREE OPTIMIZED GENERATION")
print("=" * 60)
print(f"{DURATION_S}s @ {FPS}fps = {MAX_FRAMES} frames, 512x512, 3 steps")
print()
print("Speed Optimization:")
if cfg_val == "1.0":
    print(f"  ✓ CFG-FREE INFERENCE (cfg={cfg_val})")
    print("    Skips unconditional pass: 2 forward passes -> 1")
    print("    ~2x speedup on the denoising loop!")
else:
    print(f"  - CFG={cfg_val} (set CFG_SCALE=1.0 for 2x speedup)")
print()
print("Expected: ~3-4 min (vs ~6-7 min baseline)")
print("=" * 60)

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
print("=" * 60)
print(f"TOTAL: {elapsed_min:.2f} minutes")
if elapsed_min < 5.0:
    speedup = 6.4 / elapsed_min
    print(f"SPEEDUP: {speedup:.1f}x faster than baseline!")
print("=" * 60)

mp4s = sorted(OUTPUT_DIR.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
target = next((p for p in reversed(mp4s) if "audio" in p.name.lower()), mp4s[-1] if mp4s else None)
print("Output:", target)

try:
    from google.colab import files
    if target: files.download(str(target))
except ImportError:
    pass  # Not in Colab
