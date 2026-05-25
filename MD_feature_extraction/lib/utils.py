
from __future__ import annotations

import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def get_input_text(row: dict) -> str:

    return str(row.get("head") or row.get("text") or row.get("input_text") or "").strip()


def normalize_head_text(head: str) -> str:
    head = (head or "").strip()
    if not head:
        return head
    if head[-1] not in ".!?":
        return head + "."
    return head


def slugify_prompt(text: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")
    slug = "_".join(part for part in slug.split("_") if part)
    return slug[:80] or "prompt"
