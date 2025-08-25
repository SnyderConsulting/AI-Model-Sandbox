#!/usr/bin/env python3
"""Inspect MythoMax-L2-13B architecture without loading weights.

This utility loads a Hugging Face Transformers config and tokenizer,
builds the model on a meta device to enumerate parameter shapes, and
emits a short report with key statistics.

Outputs are written to ``reports/mythomax/<run_tag>/`` where
``<run_tag>`` defaults to an ISO-8601 timestamp. The directory contains:

- ``report.md`` – human-readable summary
- ``summary.json`` – machine-readable key metrics
- ``tensor_index.csv`` – parameter names, shapes, layers, kinds
- ``module_tree.txt`` – pretty-printed module tree

Example:
```
python tools/inspect_mythomax.py \
  --model-dir /workspace/models/MythoMax-L2-13B
```
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from accelerate import init_empty_weights
from rich.console import Console
from rich.table import Table
from rich.tree import Tree
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

console = Console(highlight=False)


def _human(n: float) -> str:
    for unit in ["", "K", "M", "B", "T"]:
        if abs(n) < 1000.0:
            return f"{n:,.2f}{unit}"
        n /= 1000.0
    return f"{n:.2f}P"


def _mem_bytes(numel: int, dtype: str = "float16") -> int:
    bps = {
        "float32": 4,
        "float16": 2,
        "bfloat16": 2,
        "float64": 8,
        "int8": 1,
        "uint8": 1,
    }.get(str(dtype), 2)
    return numel * bps


def _layer_id(name: str) -> Optional[int]:
    m = re.search(r"layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def _kind_from_name(name: str) -> str:
    if ".self_attn.q_proj" in name:
        return "attn.q_proj"
    if ".self_attn.k_proj" in name:
        return "attn.k_proj"
    if ".self_attn.v_proj" in name:
        return "attn.v_proj"
    if ".self_attn.o_proj" in name:
        return "attn.o_proj"
    if ".mlp.up_proj" in name:
        return "mlp.up_proj"
    if ".mlp.gate_proj" in name:
        return "mlp.gate_proj"
    if ".mlp.down_proj" in name:
        return "mlp.down_proj"
    if "embed_tokens" in name:
        return "embeddings"
    if "lm_head" in name:
        return "lm_head"
    if "input_layernorm" in name:
        return "pre_attn_norm"
    if "post_attention_layernorm" in name:
        return "post_attn_norm"
    if name.endswith("weight") or name.endswith("bias"):
        return "other"
    return "buffer_or_other"


def _build_module_tree(model: torch.nn.Module) -> Tree:
    tree = Tree("model")
    top: Dict[str, Dict] = {}
    for n, _ in model.named_parameters():
        parts = n.split(".")
        node = top
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node.setdefault(parts[-1], None)

    def _add_to_tree(parent: Tree, d: Dict[str, Dict]) -> None:
        for k, v in d.items():
            if isinstance(v, dict):
                child = parent.add(k)
                _add_to_tree(child, v)
            else:
                parent.add(k)

    _add_to_tree(tree, top)
    return tree


def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model-dir",
        type=str,
        default="/workspace/models/MythoMax-L2-13B",
        help="Hugging Face model ID or local path",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory for outputs; default reports/mythomax/<timestamp>",
    )
    args = ap.parse_args()

    run_tag = _timestamp()
    base_out = Path("reports") / "mythomax"
    out_dir = Path(args.out_dir) if args.out_dir else base_out / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    model_dir = args.model_dir
    console.print(f"[bold]Loading config & tokenizer from[/bold] {model_dir}")
    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=False)
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=False)

    rows: List[Tuple[str, List[int], int, Optional[int], str]] = []
    by_kind: Counter[str] = Counter()
    by_layer: Counter[int] = Counter()
    total = 0
    for name, p in model.named_parameters():
        shape = list(p.shape)
        numel = int(p.numel())
        lay = _layer_id(name)
        kind = _kind_from_name(name)
        rows.append((name, shape, numel, lay, kind))
        by_kind[kind] += numel
        if lay is not None:
            by_layer[lay] += numel
        total += numel

    dtype = getattr(cfg, "torch_dtype", None)
    dtype = str(dtype).replace("torch.", "") if dtype else "float16 (assumed)"
    mem16 = _mem_bytes(total, "float16")
    membf = _mem_bytes(total, "bfloat16")

    d_model = getattr(cfg, "hidden_size", None)
    n_layers = getattr(cfg, "num_hidden_layers", None)
    n_heads = getattr(cfg, "num_attention_heads", None)
    n_kv = getattr(cfg, "num_key_value_heads", None)
    head_dim = (d_model // n_heads) if (d_model and n_heads) else None
    inter = getattr(cfg, "intermediate_size", None)
    act = getattr(cfg, "hidden_act", None)
    rms_eps = getattr(cfg, "rms_norm_eps", None)
    rope_theta = getattr(cfg, "rope_theta", None)
    rope_scaling = getattr(cfg, "rope_scaling", None)
    vocab = getattr(cfg, "vocab_size", None)
    bos_id = getattr(tok, "bos_token_id", None)
    eos_id = getattr(tok, "eos_token_id", None)
    pad_id = getattr(tok, "pad_token_id", None)

    report_path = out_dir / "report.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# MythoMax-L2-13B Architecture Report\n")
        f.write(f"Model dir: {model_dir}\n\n")
        f.write("## Config\n")
        f.write("```json\n" + json.dumps(cfg.to_dict(), indent=2) + "\n```\n\n")
        f.write("## Tokenizer\n")
        f.write(
            f"vocab_size={vocab}, bos={bos_id}, eos={eos_id}, pad={pad_id}, tokenizer_class={tok.__class__.__name__}\n\n"
        )
        f.write("## LLaMA core\n")
        f.write(
            f"hidden_size={d_model}, num_hidden_layers={n_layers}, num_attention_heads={n_heads}, num_key_value_heads={n_kv}, "
            f"head_dim={head_dim}, intermediate_size={inter}, activation={act}, rms_norm_eps={rms_eps}\n"
        )
        f.write(f"rope_theta={rope_theta}, rope_scaling={rope_scaling}\n\n")
        f.write("## Parameters\n")
        f.write(
            f"total_params: {total:,}\n~memory (fp16): {mem16 / (1024**3):.2f} GB, ~memory (bf16): {membf / (1024**3):.2f} GB\n"
        )
        f.write(f"default/declared dtype: {dtype}\n\n")
        f.write("### By kind\n")
        for k, v in by_kind.most_common():
            f.write(f"- {k}: {v:,}\n")
        f.write("\n### Per layer param counts\n")
        for layer_idx in range(n_layers or 0):
            f.write(f"- layer {layer_idx:02d}: {by_layer.get(layer_idx,0):,}\n")

    summary = {
        "model_dir": model_dir,
        "total_params": total,
        "dtype": dtype,
        "memory_fp16_bytes": mem16,
        "memory_bf16_bytes": membf,
        "hidden_size": d_model,
        "num_hidden_layers": n_layers,
        "num_attention_heads": n_heads,
        "num_key_value_heads": n_kv,
        "head_dim": head_dim,
        "intermediate_size": inter,
        "rms_norm_eps": rms_eps,
        "rope_theta": rope_theta,
        "rope_scaling": rope_scaling,
        "vocab_size": vocab,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with (out_dir / "tensor_index.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "shape", "numel", "layer", "kind"])
        for name, shape, numel, lay, kind in rows:
            w.writerow(
                [
                    name,
                    "x".join(map(str, shape)),
                    numel,
                    lay if lay is not None else "",
                    kind,
                ]
            )

    tree = _build_module_tree(model)
    with (out_dir / "module_tree.txt").open("w", encoding="utf-8") as f:
        f.write(tree.__str__())

    table = Table(title="MythoMax-L2-13B — high-level")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("hidden_size", str(d_model))
    table.add_row("#layers", str(n_layers))
    table.add_row("#heads", str(n_heads))
    table.add_row("#kv_heads", str(n_kv))
    table.add_row("head_dim", str(head_dim))
    table.add_row("intermediate", str(inter))
    table.add_row("rope_theta", str(rope_theta))
    table.add_row("rope_scaling", json.dumps(rope_scaling))
    table.add_row("vocab_size", str(vocab))
    table.add_row("total params", f"{total:,}")
    console.print(table)
    console.print(
        f"[green]Wrote[/green] report.md, summary.json, tensor_index.csv, module_tree.txt to {out_dir}"
    )


if __name__ == "__main__":
    main()
