#!/usr/bin/env bash
set -euo pipefail

# --- Paths (edit if your layout differs) ---
CKPT_DIR="/workspace/models/Wan2.2-TI2V-5B"           # base Wan checkpoints dir with diffusion_pytorch_model-*.safetensors
ADAPTER_PATH="outputs/wan_lora_stage_3_visual/adapter_model.safetensors"  # latest exported adapter
BRIDGE_CKPT="outputs/encoder_bridge_thesby/bridge_step010000-affine.pth"  # your trained bridge
LLM_DIR="thesby/Qwen2.5-VL-7B-NSFW-Caption-V3"                              # your local/HF path

# --- Bridge envs ---
export WAN_USE_BRIDGE=1
export WAN_BRIDGE_CKPT="${BRIDGE_CKPT}"
export WAN_BRIDGE_LLM_DIR="${LLM_DIR}"
export WAN_BRIDGE_DTYPE="bf16"        # set to "fp16" if your GPU lacks bf16
export WAN_BRIDGE_FORCE_VL=1
export WAN_BRIDGE_GLOBAL_SCALE="1.0"

# --- Prompt (safe, adult-only wording) ---
PROMPT="A photorealistic portrait of an adult woman wearing a stylish bikini on a sunny beach; confident pose, natural skin, detailed lighting, cinematic color grading, tasteful; no nudity, no minors."

# --- Sampler settings ---
TASK="ti2v-5B"
SIZE="1280*704"        # ti2v-5B supports: 704*1280 or 1280*704
FRAMES=1               # single-frame still (must be 4n+1 -> 1 is valid)
STEPS=50
GUIDE=5.5              # CFG (try 4.5, 5.5, 6.5)
SEED=12345

OUT_DIR="outputs/infer"
mkdir -p "${OUT_DIR}"
OUT="${OUT_DIR}/bikini_${TASK}_${SIZE}_s${SEED}_st${STEPS}_cfg${GUIDE}.mp4"

# --- Run ---
python inference/Wan2.2/generate.py \
  --task "${TASK}" \
  --ckpt_dir "${CKPT_DIR}" \
  --size "${SIZE}" \
  --frame_num ${FRAMES} \
  --sample_solver "unipc" \
  --sample_steps ${STEPS} \
  --sample_guide_scale ${GUIDE} \
  --base_seed ${SEED} \
  --lora_adapter_path "${ADAPTER_PATH}" \
  --lora_alpha 32 \
  --lora_prefix "diffusion_model." \
  --save_file "${OUT}" \
  --prompt "${PROMPT}"

echo "Saved: ${OUT}"
