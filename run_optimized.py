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

# 1. REDUCE DIFFUSION STEPS: 3 -> 2 (saves ~33% sampling time!)
os.environ["MAX_STEPS"] = "2"

# 2. REDUCE WINDOW OVERLAP: 25 -> 5 frames overlap
#    This reduces windows from 8 to ~5, saving ~3 windows worth of compute
#    Formula: windows = total_frames / (window_size - overlap) + 1
#    120 / (25-5) + 1 = 7 windows (vs 8 with overlap=9)
os.environ["MOTION_FRAME"] = "5"

# 3. PUNICA-STYLE LORA (already optimized)
os.environ["WAN_LORA_ONTHEFLY"] = "1"
os.environ["WAN_LORA_TIMING"] = "1"

# 4. OPTIONAL: torch.compile (experimental - has warmup cost)
# Uncomment to try - might help on subsequent runs
# os.environ["WAN_TORCH_COMPILE"] = "1"

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
print("MAXIMUM SPEED OPTIMIZATION RUN")
print("=" * 60)
print(f"512x512 @ 5s, {MAX_FRAMES} frames")
print()
print("SPEED OPTIMIZATIONS:")
print(f"  ✓ Reduced steps: 3 → 2 (saves ~33% sampling time)")
print(f"  ✓ Reduced overlap: MOTION_FRAME=5 (fewer windows)")
print(f"  ✓ Punica-style LoRA with A/B GPU caching")
print(f"  ✓ torch.addmm fused operations")
print()
print("EXPECTED: ~5-6 minutes (down from 7.8 min)")
print("=" * 60)

# Pull latest optimizations
import subprocess as sp
sp.run(["git", "-C", str(COMFY_DIR / "custom_nodes" / "ComfyUI-WanVideoWrapper"), "pull", "origin", "talking-photo-optimizations"], capture_output=True)

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
