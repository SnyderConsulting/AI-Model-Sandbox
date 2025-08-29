#!/usr/bin/env bash
set -euo pipefail

: "${BRIDGE_PTH:?set BRIDGE_PTH to your .pth path}"
: "${LLM_DIR:?set LLM_DIR to your LLM id/path (e.g. thesby/Qwen2.5-VL-7B-NSFW-Caption-V3)}"

echo "===== SHELL ENV ====="
printf 'WAN_BRIDGE_CKPT (shell) = %q\n' "$BRIDGE_PTH"
printf 'WAN_BRIDGE_LLM_DIR (shell) = %q\n' "$LLM_DIR"

python - <<'PY'
import os, sys, torch
from pathlib import Path

# make sure we import the exact module your trainer uses
sys.path.append("inference/Wan2.2")
print("\n===== PYTHON CHECK =====")
ck = os.environ.get("BRIDGE_PTH","")
llm= os.environ.get("LLM_DIR","")
print("os.path.exists(BRIDGE_PTH) =", os.path.exists(ck))
print("BRIDGE_PTH =", ck)
print("LLM_DIR    =", llm)

# Set the env that t5_bridge reads
os.environ["WAN_BRIDGE_CKPT"] = ck
os.environ["WAN_BRIDGE_LLM_DIR"] = llm
os.environ["WAN_BRIDGE_DTYPE"] = "bf16"

from wan.modules.t5_bridge import BridgeEncoderModel
try:
    enc = BridgeEncoderModel(text_len=512, dtype=torch.bfloat16, device="cpu", t5_cpu=True)
    print("BridgeEncoderModel init: OK")
except Exception as e:
    print("BridgeEncoderModel init: FAILED ->", repr(e))
PY
