
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _resolve_local_snapshot(model_name: str) -> str:

    if os.path.isdir(model_name):
        return model_name
    cache_roots = []
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        cache_roots.append(Path(hf_home) / "hub")
    tc = os.environ.get("TRANSFORMERS_CACHE")
    if tc:
        cache_roots.append(Path(tc))
    cache_roots += [
        Path.home() / ".cache" / "huggingface" / "hub",
    ]
    repo_dir = f"models--{model_name.replace('/', '--')}"
    for root in cache_roots:
        snapshots = root / repo_dir / "snapshots"
        if snapshots.exists():
            candidates = sorted(p for p in snapshots.iterdir() if p.is_dir())
            if candidates:
                return str(candidates[-1])
    return model_name


def load_model_and_tokenizer(
    cfg: dict[str, Any],
    device: str = "cuda",
    local_files_only: bool = False,
) -> tuple[Any, Any]:
    model_cfg = cfg["model"]
    model_name = model_cfg["name"]

    dtype_str = model_cfg.get("dtype", "bfloat16")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype_str]

    if local_files_only:
        model_name = _resolve_local_snapshot(model_name)

    print(f"[loader] Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=local_files_only,
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"





    use_device_map = device == "cuda"
    print(f"[loader] Loading model: {model_name}  dtype={dtype}  device={device}  device_map={'auto' if use_device_map else 'none'}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if use_device_map else None,
        local_files_only=local_files_only,
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )
    if not use_device_map:
        model = model.to(device)
    model.eval()
    return model, tokenizer


def get_model_input_device(model: Any) -> str:
    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        return "cpu"
