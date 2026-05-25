
from __future__ import annotations

import re
from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent

_DATASET_FILE_MAP = {
    "mmlu": "mmlu",
}


def load_urial_preamble(model_family: str, benchmark_name: str) -> str | None:

    dataset_alias = _DATASET_FILE_MAP.get(benchmark_name.lower())
    if dataset_alias is None or not model_family:
        return None

    path = _PROMPTS_DIR / f"{model_family.lower()} {dataset_alias}.txt"
    if not path.exists():
        return None

    raw = path.read_text(encoding="utf-8").strip()


    if '"""' in raw:
        parts = raw.split('"""')
        if len(parts) >= 3:
            preamble = parts[1].strip()
        else:
            preamble = raw
    else:
        preamble = raw


    preamble = re.sub(r"(?m)^###\s*User:\s*$", "User:", preamble)
    preamble = re.sub(r"(?m)^###\s*Assistant:\s*$", "Assistant:", preamble)
    preamble = re.sub(r"\n{3,}", "\n\n", preamble)
    return preamble.strip()
