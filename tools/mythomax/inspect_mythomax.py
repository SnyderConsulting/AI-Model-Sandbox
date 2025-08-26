#!/usr/bin/env python3
import argparse, os, re, json, math
from pathlib import Path
from collections import defaultdict, Counter

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from accelerate import init_empty_weights

# Optional: pretty printing
from rich.console import Console
from rich.table import Table
from rich.tree import Tree

console = Console(highlight=False)

def human(n):
    for unit in ["", "K", "M", "B", "T"]:
        if abs(n) < 1000.0:
            return f"{n:,.2f}{unit}"
        n /= 1000.0
    return f"{n:.2f}P"

def mem_bytes(numel, dtype="float16"):
    bps = {"float32":4,"float16":2,"bfloat16":2,"float64":8,"int8":1,"uint8":1}.get(str(dtype), 2)
    return numel * bps

def layer_id(name:str):
    m = re.search(r"layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None

def kind_from_name(name:str):
    if ".self_attn.q_proj" in name: return "attn.q_proj"
    if ".self_attn.k_proj" in name: return "attn.k_proj"
    if ".self_attn.v_proj" in name: return "attn.v_proj"
    if ".self_attn.o_proj" in name: return "attn.o_proj"
    if ".mlp.up_proj" in name:     return "mlp.up_proj"
    if ".mlp.gate_proj" in name:   return "mlp.gate_proj"
    if ".mlp.down_proj" in name:   return "mlp.down_proj"
    if "embed_tokens" in name:     return "embeddings"
    if "lm_head" in name:          return "lm_head"
    if "input_layernorm" in name:  return "pre_attn_norm"
    if "post_attention_layernorm" in name: return "post_attn_norm"
    if name.endswith("weight") or name.endswith("bias"):
        return "other"
    return "buffer_or_other"

def build_module_tree(model):
    tree = Tree("model")
    # best-effort shallow tree
    top = {}
    for n, _ in model.named_parameters():
        parts = n.split(".")
        node = top
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node.setdefault(parts[-1], None)
    def add_to_tree(parent, d):
        for k, v in d.items():
            if isinstance(v, dict):
                child = parent.add(k)
                add_to_tree(child, v)
            else:
                parent.add(k)
    add_to_tree(tree, top)
    return tree

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=str, default="/workspace/models/MythoMax-L2-13B")
    ap.add_argument("--out-dir", type=str, default="/workspace/reports/mythomax")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    model_dir = args.model_dir

    console.print(f"[bold]Loading config & tokenizer from[/bold] {model_dir}")
    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=False)
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)

    # Meta init to enumerate params without allocating weights
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=False)

    # Collect parameter entries
    rows = []
    by_kind = Counter()
    by_layer = Counter()
    total = 0
    for name, p in model.named_parameters():
        shape = list(p.shape)
        numel = int(p.numel())
        lay = layer_id(name)
        kind = kind_from_name(name)
        rows.append((name, shape, numel, lay, kind))
        by_kind[kind] += numel
        if lay is not None:
            by_layer[lay] += numel
        total += numel

    # Dtype guess (from config if present)
    dtype = getattr(cfg, "torch_dtype", None)
    dtype = str(dtype).replace("torch.", "") if dtype else "float16 (assumed)"
    mem16 = mem_bytes(total, "float16")
    membf = mem_bytes(total, "bfloat16")

    # High-level LLaMA details
    d_model = getattr(cfg, "hidden_size", None)
    n_layers = getattr(cfg, "num_hidden_layers", None)
    n_heads = getattr(cfg, "num_attention_heads", None)
    n_kv    = getattr(cfg, "num_key_value_heads", None)
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

    # Write summary report
    rep = out / "architecture_report.txt"
    with rep.open("w") as f:
        f.write("# MythoMax-L2-13B Architecture Report\n")
        f.write(f"Model dir: {model_dir}\n")
        f.write("\n[Config]\n")
        f.write(json.dumps(cfg.to_dict(), indent=2))
        f.write("\n\n[Tokenizer]\n")
        f.write(f"vocab_size={vocab}, bos={bos_id}, eos={eos_id}, pad={pad_id}\n")
        f.write(f"tokenizer_class={tok.__class__.__name__}\n")
        f.write("\n[LLaMA core]\n")
        f.write(f"hidden_size={d_model}, num_hidden_layers={n_layers}, num_attention_heads={n_heads}, "
                f"num_key_value_heads={n_kv}, head_dim={head_dim}, intermediate_size={inter}, "
                f"activation={act}, rms_norm_eps={rms_eps}\n")
        f.write(f"rope_theta={rope_theta}, rope_scaling={rope_scaling}\n")
        f.write("\n[Parameters]\n")
        f.write(f"total_params: {total:,}\n")
        f.write(f"~memory (fp16): {mem16/ (1024**3):.2f} GB, ~memory (bf16): {membf/(1024**3):.2f} GB\n")
        f.write(f"default/declared dtype: {dtype}\n")
        f.write("\n[By kind]\n")
        for k, v in by_kind.most_common():
            f.write(f"{k:20s} {v:>12,}\n")
        f.write("\n[Per layer param counts]\n")
        for l in range(n_layers or 0):
            f.write(f"layer {l:02d}: {by_layer.get(l,0):>12,}\n")
        f.write("\n[Notes]\n- Keys/Shapes listed in tensor_index.csv\n- Module tree in module_tree.txt\n")

    # Tensor index CSV
    import csv
    with (out / "tensor_index.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name","shape","numel","layer","kind"])
        for name, shape, numel, lay, kind in rows:
            w.writerow([name, "x".join(map(str,shape)), numel, lay if lay is not None else "", kind])

    # Module tree
    tree = build_module_tree(model)
    with (out / "module_tree.txt").open("w") as f:
        f.write(tree.__str__())

    # Console summary
    table = Table(title="MythoMax-L2-13B — high-level")
    table.add_column("Field"); table.add_column("Value")
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
    console.print(f"[green]Wrote[/green] {rep}, tensor_index.csv, module_tree.txt to {out}")

if __name__ == "__main__":
    main()
