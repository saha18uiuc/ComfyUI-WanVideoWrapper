# --- MAXIMUM SPEED OPTIMIZED RUN SCRIPT ---
import os
import sys
import subprocess
import time
from pathlib import Path

# ============================================================
# STEP 1: Install ninja for CUDA kernel compilation
# ============================================================
print("Installing ninja...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ninja"], check=True)

# ============================================================
# STEP 2: CUDA Memory Optimization (MUST be before torch import!)
# ============================================================
alloc_conf = "expandable_segments:True,garbage_collection_threshold:0.8"
os.environ["PYTORCH_ALLOC_CONF"] = alloc_conf
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = alloc_conf

WORKFLOW_PATH = Path("/content/workflow_api.py")
COMFY_DIR = Path("/content/ComfyUI")
OUTPUT_DIR = COMFY_DIR / "output"

FPS = 24
DURATION_S = 5
MAX_FRAMES = FPS * DURATION_S

# ============================================================
# STEP 3: Video Settings
# ============================================================
os.environ["FRAME_RATE"] = str(FPS)
os.environ["MAX_FRAMES"] = str(MAX_FRAMES)
os.environ["WIDTH"] = "512"
os.environ["HEIGHT"] = "512"
os.environ["AUDIO_SCALE_STRENGTH"] = "2"
os.environ["FRAME_WINDOW_SIZE"] = "25"

# ============================================================
# STEP 4: SPEED OPTIMIZATIONS
# ============================================================

# 1. DIFFUSION STEPS: Keep at 3 for quality (user requirement)
os.environ["MAX_STEPS"] = "3"

# 2. WINDOW OVERLAP: Default 25 frames for smooth transitions
# os.environ["MOTION_FRAME"] = "5"  # Uncomment to reduce windows (faster but less smooth)

# 3. STREAMING LoRA MODE (DEFAULT - PROVEN FASTEST)
#    How it works:
#      - First forward: compute delta = sum(A@B * strength) on CPU, cache it
#      - Subsequent forwards: transfer delta to GPU, compute (W+delta)@x
#    This is faster because: only ONE matmul, simple addition is cheap
#    Result: ~6.4 min on L4 (vs 7.2 min base = 11% speedup)
os.environ["WAN_LORA_TIMING"] = "1"
# STREAMING is ON by default (WAN_LORA_STREAMING=1)

# ============================================================
# STEP 5: Models
# ============================================================
os.environ["INFINATE_TALK_MODEL"] = "Wan2_1-InfiniTetalk-Single_fp16.safetensors"
os.environ["VAE_MODEL"] = "Wan2_1_VAE_bf16.safetensors"
os.environ["LORA_MODEL"] = "Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors"
os.environ["CLIP_NAME"] = "clip_vision_h.safetensors"
os.environ["TEXT_ENCODE_MODEL"] = "umt5-xxl-enc-bf16.safetensors"
os.environ["WAN_MODEL"] = "Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors"
os.environ["POS_PROMPT"] = ""

ref_audio = next((p for p in [Path("/content/audio.mp3"), Path("/content/ComfyUI/input/audio.mp3")] if p.is_file()), None)
ref_image = next((p for p in [Path("/content/512.jpg"), Path("/content/ComfyUI/input/512.jpg")] if p.is_file()), None)
if not ref_audio: raise FileNotFoundError("Missing audio")
if not ref_image: raise FileNotFoundError("Missing image")
os.environ["REF_AUDIO"] = str(ref_audio)
os.environ["REF_IMAGE"] = str(ref_image)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("STREAMING LoRA MODE (PROVEN FASTEST)")
print("=" * 60)
print(f"512x512 @ 5s, {MAX_FRAMES} frames, 3 steps (full quality)")
print()
print("How STREAMING works:")
print("  1. First forward: compute delta on CPU, cache it")
print("  2. Subsequent: transfer delta, compute (W+delta)@x")
print()
print("Why it's fastest:")
print("  - Only ONE matmul per forward pass")
print("  - Addition (W+delta) is cheap")
print("  - No transpose/contiguous overhead")
print()
print("Expected: ~6.4 min on L4 (11% faster than 7.2 min base)")
print("=" * 60)

# Pull latest optimizations
import subprocess as sp
sp.run(["git", "-C", str(COMFY_DIR / "custom_nodes" / "ComfyUI-WanVideoWrapper"), "pull", "origin", "performance-optimizations-same-steps"], capture_output=True)

t0 = time.perf_counter()
try:
    completed = subprocess.run(["python", str(WORKFLOW_PATH)], cwd=str(WORKFLOW_PATH.parent), check=True, text=True, capture_output=True)
    print(completed.stdout)
    if completed.stderr: print(completed.stderr)
except subprocess.CalledProcessError as e:
    print(f"FAILED: {e.returncode}")
    print(e.stdout or "")
    print(e.stderr or "")
    raise
t1 = time.perf_counter()

print("=" * 60)
print(f"TOTAL TIME: {(t1-t0)/60:.2f} minutes")
print("=" * 60)

mp4s = sorted(OUTPUT_DIR.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
target = next((p for p in reversed(mp4s) if "audio" in p.name.lower()), mp4s[-1] if mp4s else None)
print("Output:", target)

from google.colab import files
if target: files.download(str(target))
