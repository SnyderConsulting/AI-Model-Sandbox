# KV LoRA Distillation Trainer

`train_kv_lora_distill.py` trains key/value LoRA adapters for Wan 2.2's cross-attention.

## Dataset

The `--prompts_file` flag accepts either a newline-delimited text file or a
JSONL file. For JSONL, specify the caption field via `--jsonl_field` (default:
`caption`).

Useful options:

- `--max_samples N` – limit the dataset size for quick tests. Combine with
  `--shuffle` to select a random subset.
- `--shuffle` – shuffle prompts before applying `--max_samples`.

## Training

- `--grad_accum` controls gradient accumulation to reach larger effective batch
  sizes without increasing memory.

The data loader now uses multiple workers and pinned memory for faster I/O.
