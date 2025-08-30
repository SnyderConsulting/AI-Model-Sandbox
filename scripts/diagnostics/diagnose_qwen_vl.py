#!/usr/bin/env python3
"""
diagnose_qwen_vl.py

Purpose
-------
Quickly pinpoint why Qwen2.5-VL isn't loading as a *vision-language* model
(e.g., code falls back to LLaMA / AutoModelForCausalLM and prints
"You are using a model of type llama to instantiate a model of type qwen2_5_vl.").

What it does
------------
1) Prints Python + key library versions and CUDA capability.
2) Reads your model's config.json and shows model_type / architectures.
3) Verifies that Transformers exposes Qwen2_5_VLForConditionalGeneration (or older alias).
4) Tries to build the proper *processor* for Qwen2.5-VL (AutoProcessor).
5) Explains the most likely root cause(s) and prints FIX hints.
6) Returns non-zero exit code when something is wrong.

Usage
-----
python diagnose_qwen_vl.py --llm_dir </path/or/hub-id/of/Qwen2.5-VL>

Examples:
  python diagnose_qwen_vl.py --llm_dir thesby/Qwen2.5-VL-7B-NSFW-Caption-V3
  python diagnose_qwen_vl.py --llm_dir /workspace/models/Qwen2.5-VL-7B

Notes
-----
- The script never attempts to load full 7B weights onto GPU.
- It only imports classes and reads config/processor; safe on CPU-only boxes.
"""

from __future__ import annotations
import argparse, json, os, sys, platform, importlib, traceback
from pathlib import Path

def _try_import(name: str):
    try:
        mod = importlib.import_module(name)
        return mod, None
    except Exception as e:
        return None, e

def _get_ver(mod):
    v = getattr(mod, "__version__", None)
    if v is None:
        # some pkgs store version elsewhere
        v = getattr(mod, "version", None)
    return v

def section(title: str):
    print("="*80)
    print(f"{title}")
    print("="*80)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm_dir", required=True, help="Local path or HF hub id of the Qwen2.5-VL model")
    ap.add_argument("--verbose", action="store_true", help="Print tracebacks for failures")
    args = ap.parse_args()

    problems = []

    # ----------------------------------------------------------------------------------
    # 0) Environment report
    # ----------------------------------------------------------------------------------
    section("ENVIRONMENT")
    print(f"Python      : {platform.python_version()}  ({sys.executable})")
    print(f"Platform    : {platform.platform()}")
    # torch
    torch, torch_err = _try_import("torch")
    if torch:
        print(f"torch       : {_get_ver(torch)}  (cuda.is_available={torch.cuda.is_available() if hasattr(torch, 'cuda') else False})")
        if getattr(torch, 'cuda', None) and torch.cuda.is_available():
            try:
                dev_name = torch.cuda.get_device_name(0)
            except Exception:
                dev_name = "unknown"
            print(f"CUDA        : {torch.version.cuda}  GPU={dev_name}")
    else:
        print("torch       : NOT INSTALLED")
        problems.append("torch not installed")

    # transformers
    transformers, tx_err = _try_import("transformers")
    if transformers:
        print(f"transformers: {_get_ver(transformers)}")
    else:
        print("transformers: NOT INSTALLED")
        problems.append("transformers not installed")

    # qwen-vl-utils (optional but required for Qwen-VL processors / video)
    qvu, qvu_err = _try_import("qwen_vl_utils")
    if qvu:
        print(f"qwen_vl_utils: {_get_ver(qvu)}  (required for Qwen2.5-VL image/video processing)")
    else:
        print("qwen_vl_utils: NOT INSTALLED (often required for Qwen2.5-VL)")

    # pillow, torchvision, decord (common deps for VL)
    pillow, _ = _try_import("PIL")
    if pillow:
        print("Pillow      : present")
    else:
        print("Pillow      : NOT INSTALLED")

    tv, _ = _try_import("torchvision")
    if tv:
        print(f"torchvision : {_get_ver(tv)}")
    else:
        print("torchvision : NOT INSTALLED")

    decord, _ = _try_import("decord")
    if decord:
        print(f"decord      : {_get_ver(decord)}")
    else:
        print("decord      : NOT INSTALLED (only needed for some video loaders)")

    # accelerate (we won't *need* it, but some pipelines rely on it)
    accelerate, _ = _try_import("accelerate")
    if accelerate:
        print(f"accelerate  : {_get_ver(accelerate)}")
    else:
        print("accelerate  : NOT INSTALLED")

    # ----------------------------------------------------------------------------------
    # 1) Inspect model config to ensure you're pointing at a *VL* checkpoint
    # ----------------------------------------------------------------------------------
    section("MODEL CONFIG BASIC CHECKS")
    path = args.llm_dir
    is_local = Path(path).exists()
    print(f"llm_dir     : {path}  (local={is_local})")

    cfg = None
    cfg_err = None
    model_type = None
    architectures = None
    try:
        if is_local:
            # local config.json
            cfg_path = Path(path) / "config.json"
            if not cfg_path.exists():
                print(f"config.json : MISSING at {cfg_path}")
                problems.append("config.json missing")
            else:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
        else:
            # try HF hub download of only config.json
            from huggingface_hub import hf_hub_download
            cfg_file = hf_hub_download(path, "config.json", local_dir=os.getcwd())
            with open(cfg_file, "r", encoding="utf-8") as f:
                cfg = json.load(f)
    except Exception as e:
        cfg_err = e

    if cfg is None:
        print("Could not read config.json")
        if args.verbose and cfg_err:
            traceback.print_exception(cfg_err)
        problems.append("cannot read config.json")
    else:
        model_type = cfg.get("model_type", None)
        architectures = cfg.get("architectures", None)
        print(f"model_type  : {model_type}")
        print(f"architectures: {architectures}")
        # sanity: Qwen2.5-VL family should report e.g. 'qwen2_5_vl' (or 'qwen2_vl' for older)
        if model_type not in {"qwen2_5_vl", "qwen2_vl"}:
            problems.append(f"config.model_type is '{model_type}', not a Qwen2.5-VL type")
        # quick file presence check for vision bits
        if is_local:
            files = [p.name for p in Path(path).glob("*") if p.is_file()]
            has_vision = any(n.startswith("visual") or "vision" in n or "projector" in n for n in files)
            if not has_vision:
                print("note         : did not detect obvious 'vision/*' assets in directory listing (may still be present inside model files)")

    # ----------------------------------------------------------------------------------
    # 2) Does this Transformers build expose the right class?
    # ----------------------------------------------------------------------------------
    section("TRANSFORMERS CLASS AVAILABILITY")
    have_qwen25_vl = False
    alias_tried = False
    qwen_class_name = None

    ModelClass = None
    err_primary = None
    err_alias = None

    if transformers:
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as _MC
            ModelClass = _MC
            have_qwen25_vl = True
            qwen_class_name = "Qwen2_5_VLForConditionalGeneration"
        except Exception as e:
            err_primary = e
            # Try older alias
            try:
                from transformers import Qwen2VLForConditionalGeneration as _ALIAS
                ModelClass = _ALIAS
                have_qwen25_vl = True
                alias_tried = True
                qwen_class_name = "Qwen2VLForConditionalGeneration (alias)"
            except Exception as e2:
                err_alias = e2

    if have_qwen25_vl:
        print(f"✓ Found class in transformers: {qwen_class_name}")
    else:
        print("✗ Could NOT import Qwen2.5-VL model class from transformers")
        if args.verbose:
            if err_primary:
                print("-- primary import error (Qwen2_5_VLForConditionalGeneration) --")
                traceback.print_exception(err_primary)
            if err_alias:
                print("-- alias import error (Qwen2VLForConditionalGeneration) --")
                traceback.print_exception(err_alias)
        problems.append("transformers build lacks Qwen2.5-VL model class (too old)")

    # ----------------------------------------------------------------------------------
    # 3) Can we build the *processor*? (frequent failure when qwen_vl_utils missing)
    # ----------------------------------------------------------------------------------
    section("PROCESSOR CHECK (AutoProcessor)")
    proc_ok = False
    proc_type = None
    proc_err = None
    if transformers:
        try:
            from transformers import AutoProcessor
            proc = AutoProcessor.from_pretrained(path, trust_remote_code=True)
            proc_type = type(proc).__name__
            print(f"✓ AutoProcessor.from_pretrained OK → {proc_type}")
            # On Qwen2.5-VL you typically see 'Qwen2_5_VLProcessor' or similar.
            proc_ok = True
        except Exception as e:
            proc_err = e
            print("✗ AutoProcessor.from_pretrained FAILED")
            if args.verbose:
                traceback.print_exception(proc_err)
            # Heuristic: many failures here are "ModuleNotFoundError: No module named 'qwen_vl_utils'"
            msg = str(proc_err)
            if "qwen_vl_utils" in msg:
                problems.append("qwen_vl_utils is missing (required for Qwen2.5-VL processor/image/video)")
            else:
                problems.append("AutoProcessor.from_pretrained failed (see --verbose)")
    else:
        print("transformers not installed → cannot check processor")
        problems.append("no transformers → processor check skipped")

    # ----------------------------------------------------------------------------------
    # 4) Summarize diagnosis + suggested fixes
    # ----------------------------------------------------------------------------------
    section("DIAGNOSIS & FIX")
    if not problems:
        print("All critical checks passed. If your training still prints the LLaMA warning,")
        print("the culprit is likely your *bridge loader* falling back to AutoModelForCausalLM.")
        print()
        print("Actionable next step:")
        print("  • Ensure the code path uses `Qwen2_5_VLForConditionalGeneration` directly")
        print("    (or `Qwen2VLForConditionalGeneration` on older builds), NOT AutoModelForCausalLM.")
        print("  • Pass `trust_remote_code=True` and keep using the VL processor returned above.")
        rc = 0
    else:
        for i, p in enumerate(problems, 1):
            print(f"{i}) {p}")
        print()
        print("Most common root causes and how to fix:")
        print("  - transformers too old and does not expose Qwen2.5-VL classes →")
        print("      pip install -U 'transformers>=4.44.0'  # 4.45+ recommended")
        print("  - qwen_vl_utils missing (processor import fails) →")
        print("      pip install -U qwen-vl-utils")
        print("  - config.json is not a *VL* checkpoint (model_type != qwen2_5_vl/qwen2_vl) →")
        print("      double-check --llm_dir points at a Qwen2.5-VL repo, not the text-only Qwen2.5")
        print()
        print("IMPORTANT: Even if both the class and the processor are available, using")
        print("AutoModelForCausalLM on a qwen2_5_vl config will trigger the LLaMA warning and")
        print("instantiate the wrong class. Use the explicit VL class instead.")
        rc = 2

    print()
    print("Done.")
    sys.exit(rc)

if __name__ == "__main__":
    main()
