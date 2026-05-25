
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve(value: str, base_dir: Path) -> str:
    if isinstance(value, str):
        value = value.replace("{work_dir}", str(base_dir))
    return value


def resolve_config(cfg: dict[str, Any], config_path: Path) -> dict[str, Any]:
    base_dir = config_path.resolve().parent

    main_sets = cfg.get("main_sets", {})
    for key, val in main_sets.items():
        main_sets[key] = _resolve(str(val), base_dir) if val else val

    output = cfg.get("output", {})
    if "base_dir" in output:
        output["base_dir"] = _resolve(str(output["base_dir"]), base_dir)

    sae = cfg.get("sae", {})
    if sae.get("checkpoint_path"):
        sae["checkpoint_path"] = _resolve(str(sae["checkpoint_path"]), base_dir)

    steering = cfg.get("steering", {})
    if steering.get("mean_diff_path"):
        steering["mean_diff_path"] = _resolve(str(steering["mean_diff_path"]), base_dir)

    return cfg


def parse_layers(cfg: dict[str, Any], available_layers: list[int] | None = None) -> list[int]:
    raw = cfg.get("steering", {}).get("layers", "all")
    if raw == "all":
        if available_layers is None:
            raise ValueError("layers='all' requires available_layers to be provided.")
        return sorted(available_layers)
    if isinstance(raw, list):
        return [int(x) for x in raw]
    if isinstance(raw, str):
        return [int(x.strip()) for x in raw.split(",") if x.strip()]
    raise ValueError(f"Unsupported layers format: {raw!r}")


def parse_list(value: Any, cast=float) -> list:
    if isinstance(value, list):
        return [cast(v) for v in value]
    if isinstance(value, str):
        return [cast(v.strip()) for v in value.split(",") if v.strip()]
    return [cast(value)]
