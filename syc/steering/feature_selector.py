
from __future__ import annotations

import json
from pathlib import Path


def load_mean_diff_feature(mean_diff_dir: str, layer: int, direction: str = "pos") -> tuple[int, dict]:

    p = Path(mean_diff_dir) / f"layer_{layer:02d}_mean_diff.json"
    if not p.exists():
        raise FileNotFoundError(f"Mean-diff file not found: {p}")
    with p.open(encoding="utf-8") as f:
        data = json.load(f)
    key = "top_positive" if direction == "pos" else "top_negative"
    top = data["modes"]["output"][key][0]
    return int(top["feature_id"]), top


def load_maximin_feature(
    mean_diff_dir_pt: str,
    mean_diff_dir_it: str,
    layer: int,
    direction: str = "pos",
) -> tuple[int, dict]:

    import numpy as np

    pt_npy = Path(mean_diff_dir_pt) / "npy" / "both" / f"layer_{layer:02d}_mean_diff.npy"
    it_npy = Path(mean_diff_dir_it) / "npy" / "both" / f"layer_{layer:02d}_mean_diff.npy"

    if not pt_npy.exists():
        raise FileNotFoundError(f"PT mean-diff npy not found: {pt_npy}")
    if not it_npy.exists():
        raise FileNotFoundError(f"IT mean-diff npy not found: {it_npy}")

    score_pt = np.load(pt_npy)
    score_it = np.load(it_npy)

    if score_pt.shape != score_it.shape:
        raise ValueError(
            f"PT/IT mean-diff shape mismatch: {score_pt.shape} vs {score_it.shape}. "
            "Ensure both use the same SAE (same d_sae)."
        )

    if direction == "pos":

        combined = np.minimum(score_pt, score_it)
        feature_id = int(np.argmax(combined))
    else:

        combined = np.maximum(score_pt, score_it)
        feature_id = int(np.argmin(combined))

    info = {
        "feature_id": feature_id,
        "score_pt": float(score_pt[feature_id]),
        "score_it": float(score_it[feature_id]),
        "combined_score": float(combined[feature_id]),
        "selection": "maximin" if direction == "pos" else "minimax",
        "direction": direction,
    }
    return feature_id, info


def available_mean_diff_layers(mean_diff_dir: str) -> list[int]:

    p = Path(mean_diff_dir)
    layers = []
    for f in p.glob("layer_*_mean_diff.json"):
        part = f.stem.split("_")[1]
        try:
            layers.append(int(part))
        except ValueError:
            pass
    return sorted(layers)
