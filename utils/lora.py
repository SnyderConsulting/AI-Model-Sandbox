import re
from typing import List, Pattern, Dict, Any


def _compile_patterns(patterns: List[str]) -> List[Pattern[str]]:
    return [re.compile(p) for p in patterns]


def filter_lora_targets(model, adapter_config: Dict[str, Any]) -> Dict[str, Any]:
    """Filter LoRA target modules based on include/exclude patterns and block range.

    Parameters
    ----------
    model: object
        Model providing ``get_lora_module_names`` which returns candidate module names.
    adapter_config: dict
        Configuration dictionary containing ``include``, ``exclude`` and ``train_blocks_range``.

    Returns
    -------
    dict
        Adapter config updated with ``target_modules``.
    """
    include = _compile_patterns(adapter_config.get("include", []))
    exclude = _compile_patterns(adapter_config.get("exclude", ["self_attn"]))
    block_range = adapter_config.get("train_blocks_range", None)

    target_modules: List[str] = []
    warned_self_attn = False
    for name in getattr(model, "get_lora_module_names", lambda: [])():
        if block_range is not None:
            match = re.search(r"blocks\.(\d+)\.", name)
            if match:
                idx = int(match.group(1))
                if idx < block_range[0] or idx > block_range[1]:
                    continue
        if include and not any(p.search(name) for p in include):
            continue
        if any(p.search(name) for p in exclude):
            if "self_attn" in name and not warned_self_attn:
                print("Skipping self_attn module '" + name + "' due to safety filter")
                warned_self_attn = True
            continue
        target_modules.append(name)

    if not target_modules:
        raise ValueError("No target modules remaining after applying include/exclude filters.")

    adapter_config["target_modules"] = target_modules
    return adapter_config
