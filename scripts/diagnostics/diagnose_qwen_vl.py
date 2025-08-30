#!/usr/bin/env python3
import argparse, os, sys, json, platform, traceback
from pathlib import Path

import torch
print = lambda *a, **k: __import__("builtins").print(*a, **k, flush=True)

def section(t):
    print("=" * 80)
    print(t)
    print("=" * 80)

def _try_imports():
    info = {}
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except Exception as e:
        info["transformers"] = f"ERROR: {e}"
    try:
        import qwen_vl_utils  # optional; not always required on transformers>=4.43
        info["qwen_vl_utils"] = getattr(qwen_vl_utils, "__version__", "present")
    except Exception:
        info["qwen_vl_utils"] = None
    try:
        import torchvision
        info["torchvision"] = torchvision.__version__
    except Exception:
        info["torchvision"] = None
    try:
        import decord
        info["decord"] = decord.__version__
    except Exception:
        info["decord"] = None
    return info

def env_report():
    section("ENVIRONMENT")
    info = _try_imports()
    print(f"Python      : {platform.python_version()}  ({sys.executable})")
    print(f"Platform    : {platform.platform()}")
    print(f"torch       : {torch.__version__}  (cuda.is_available={torch.cuda.is_available()})")
    if torch.cuda.is_available():
        print(f"CUDA        : {torch.version.cuda}  GPU={torch.cuda.get_device_name(0)}")
    for k in ("transformers","qwen_vl_utils","torchvision","decord"):
        print(f"{k:<12}: {info[k]}")
    return info

def load_vl_model(llm_dir, use_cpu=False, verbose=False):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    section("ATTEMPTING TO LOAD Qwen2_5_VLForConditionalGeneration")
    # Processor
    proc = AutoProcessor.from_pretrained(llm_dir, trust_remote_code=True)
    print(f"Processor  : {proc.__class__.__name__}")
    # Model
    dtype = torch.float32 if use_cpu else torch.bfloat16
    device_map = {"": "cpu"} if use_cpu else "auto"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        llm_dir,
        torch_dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    cfg = model.config
    print(f"Model      : {model.__class__.__name__}")
    print(f"Hidden size: {getattr(cfg,'hidden_size','?')}")
    if verbose:
        print("architectures:", getattr(cfg, "architectures", None))
        print("model_type  :", getattr(cfg, "model_type", None))
    return proc, model

def try_causallm_fallback(llm_dir, use_cpu=False):
    """Mimic the fallback many scripts use and show what hidden_size that gives."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    section("FALLBACK CHECK (AutoModelForCausalLM)")
    tok = AutoTokenizer.from_pretrained(llm_dir, use_fast=True, trust_remote_code=True)
    dtype = torch.float32 if use_cpu else torch.bfloat16
    device_map = {"": "cpu"} if use_cpu else "auto"
    mdl = AutoModelForCausalLM.from_pretrained(
        llm_dir,
        torch_dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    hs = getattr(mdl.config, "hidden_size", None)
    print(f"AutoModelForCausalLM -> hidden_size={hs}")
    return mdl

def inspect_bridge_ckpt(path):
    section("BRIDGE CHECKPOINT INSPECTION")
    if not path:
        print("No --bridge_ckpt provided; skipping.")
        return None
    if not Path(path).exists():
        print(f"ERROR: {path} not found.")
        return None
    ckpt = torch.load(path, map_location="cpu")
    sd = ckpt.get("bridge", ckpt)
    # Find an in-proj weight no matter how it was saved
    cand_keys = [k for k in sd.keys() if k.endswith("in_proj.weight")]
    if not cand_keys:
        print("Could not find 'in_proj.weight' in ckpt (is this the right bridge file?).")
        return None
    k = cand_keys[0]
    in_dim = sd[k].shape[1]
    print(f"Found {k} with shape {tuple(sd[k].shape)}  -> expected LLM hidden_size={in_dim}")
    return in_dim

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm_dir", required=True, help="HF path or local dir for Qwen2.5-VL")
    ap.add_argument("--bridge_ckpt", default=None, help="Path to your trained bridge *.pth")
    ap.add_argument("--cpu", action="store_true", help="Force CPU load for the VL model")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    env_report()

    # 1) Try the VL path (no fallbacks).
    try:
        _, vl_model = load_vl_model(args.llm_dir, use_cpu=args.cpu, verbose=args.verbose)
        vl_hs = getattr(vl_model.config, "hidden_size", None)
    except Exception as e:
        print("\n!!! FAILED TO LOAD Qwen2_5_VLForConditionalGeneration !!!")
        print("Traceback:")
        traceback.print_exc()
        vl_hs = None

    # 2) Show what the CausalLM fallback would give.
    try:
        clm = try_causallm_fallback(args.llm_dir, use_cpu=args.cpu)
        clm_hs = getattr(clm.config, "hidden_size", None)
    except Exception as e:
        print("\nAutoModelForCausalLM fallback also failed:", repr(e))
        clm_hs = None

    # 3) Bridge checkpoint expectation.
    ckpt_hs = inspect_bridge_ckpt(args.bridge_ckpt)

    section("DIAGNOSIS")
    if vl_hs is not None:
        print(f"Qwen2.5-VL hidden_size = {vl_hs}")
    if clm_hs is not None:
        print(f"CausalLM fallback hidden_size = {clm_hs}")
    if ckpt_hs is not None:
        print(f"Bridge ckpt expects hidden_size = {ckpt_hs}")

    print("\nSummary:")
    if ckpt_hs is not None and vl_hs is not None and ckpt_hs == vl_hs:
        print("✔ Your bridge ckpt matches Qwen2.5-VL. Use the VL loader. Do NOT fall back.")
    if ckpt_hs is not None and clm_hs is not None and ckpt_hs != clm_hs:
        print("✖ Bridge ckpt does NOT match the CausalLM fallback. If you see 5120 vs 3584,"
              " you are accidentally using the fallback.")
    if vl_hs is None:
        print("→ The VL model failed to load in this environment. Fix that first (see traceback above).")
    print("\nActionable next step:")
    print("  Run training with WAN_BRIDGE_FORCE_VL=1 so the code raises instead of silently falling back.")
    print("  Example:")
    print("    WAN_BRIDGE_FORCE_VL=1 python train_lora_stage_2.py ... --llm_dir <your-qwen-vl-repo> ...")

if __name__ == "__main__":
    main()
