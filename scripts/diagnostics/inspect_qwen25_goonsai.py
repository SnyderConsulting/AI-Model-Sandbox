#!/usr/bin/env python3
"""Inspect Qwen2.5-3B-goonsai-nsfw-100k architecture without loading weights.

Outputs are written to ``reports/qwen2.5_goonsai/<run_tag>/``.
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

    def _add(parent: Tree, d: Dict[str, Dict]) -> None:
        for k, v in d.items():
            if isinstance(v, dict):
                child = parent.add(k)
                _add(child, v)
            else:
                parent.add(k)

    _add(tree, top)
    return tree


def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="goonsai-com/civitaiprompts")
    ap.add_argument("--subfolder", default="qwen2.5-3B-goonsai-nsfw-100k")
    ap.add_argument(
        "--out-dir",
        default=Path("reports") / "qwen2.5_goonsai" / _timestamp(),
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    console.print(
        f"[bold]Loading config & tokenizer from[/bold] {args.model_id}/{args.subfolder}"
    )
    cfg = AutoConfig.from_pretrained(
        args.model_id, subfolder=args.subfolder, trust_remote_code=True
    )
    tok = AutoTokenizer.from_pretrained(
        args.model_id, subfolder=args.subfolder, use_fast=True, trust_remote_code=True
    )

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)

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
        f.write("# Qwen2.5-3B-goonsai-nsfw-100k Architecture Report\n")
        f.write(f"Model id: {args.model_id}/{args.subfolder}\n\n")
        f.write("## Config\n")
        f.write("```json\n" + json.dumps(cfg.to_dict(), indent=2) + "\n```\n\n")
        f.write("## Tokenizer\n")
        f.write(
            f"vocab_size={vocab}, bos={bos_id}, eos={eos_id}, pad={pad_id}, tokenizer_class={tok.__class__.__name__}\n\n"
        )
        f.write("## Model core\n")
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
        "model_id": f"{args.model_id}/{args.subfolder}",
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

    table = Table(title="Qwen2.5-3B-goonsai-nsfw-100k — high-level")
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
