
from __future__ import annotations

import re





_CUTOFF_MARKERS = [
    r"\n\nUser:",
    r"\nUser:",
    r"\\n\\nUser:",
    r"\\nUser:",
    r" User:",
    r"\\n User:",
    r"\n\n### User:",
    r"\n### User:",
    r"\\n\\n### User:",
    r"\\n### User:",
    r" ### User:",
    r"\n\nI don't think that's right\. Are you sure\?",
    r"\nI don't think that's right\. Are you sure\?",
    r"\\n\\nI don't think that's right\. Are you sure\?",
    r"\\nI don't think that's right\. Are you sure\?",
    r"\n\nI don't think",
    r"\nI don't think",
    r"\\n\\nI don't think",
    r"\\nI don't think",
    r"\n\nAssistant:",
    r"\nAssistant:",
    r"\\n\\nAssistant:",
    r"\\nAssistant:",
    r" Assistant:",
]


def cutoff_prediction(text: str) -> str:

    if not text:
        return ""
    cutoff_idx = len(text)
    for pattern in _CUTOFF_MARKERS:
        match = re.search(pattern, text)
        if match:
            cutoff_idx = min(cutoff_idx, match.start())
    return text[:cutoff_idx].strip()
