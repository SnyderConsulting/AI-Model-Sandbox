from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p
