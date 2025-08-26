import json
import random
from pathlib import Path
from torch.utils.data import Dataset


class CaptionsJSONL(Dataset):
    def __init__(
        self, jsonl_path: str | Path, min_chars: int = 1, max_chars: int | None = None
    ):
        self.recs: list[str] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                cap = rec.get("caption", "")
                if not cap:
                    continue
                if max_chars and len(cap) > max_chars:
                    continue
                if len(cap) >= min_chars:
                    self.recs.append(cap)
        random.shuffle(self.recs)

    def __len__(self) -> int:
        return len(self.recs)

    def __getitem__(self, idx: int) -> str:
        return self.recs[idx]
