
from __future__ import annotations

import re
from typing import Any

import torch


def _build_sae_id(base_sae_id: str, layer: int) -> str:


    result = re.sub(r"(?<=l)\d+(?=r)", str(layer), base_sae_id)
    if result != base_sae_id:
        return result

    result = re.sub(r"(?<=layer_)\d+", str(layer), base_sae_id)
    if result == base_sae_id:

        result = re.sub(r"(?<=layer_)\d+(?=/)", str(layer), base_sae_id)
    return result


def load_sae(cfg: dict[str, Any], layer: int, device: str = "cuda") -> Any:

    try:
        from sae_lens import SAE
    except ImportError as exc:
        raise SystemExit("sae_lens is required: pip install sae_lens") from exc

    sae_cfg = cfg.get("sae", {})
    release = sae_cfg.get("release", "")
    base_sae_id = sae_cfg.get("base_sae_id", sae_cfg.get("sae_id", ""))
    sae_id = _build_sae_id(base_sae_id, layer)

    print(f"[sae_loader] loading layer={layer}  release={release}  sae_id={sae_id}")
    sae, _, _ = SAE.from_pretrained(
        release=release,
        sae_id=sae_id,
        device=device,
    )
    sae.eval()
    return sae


def get_decoder_directions(sae: Any, feature_ids: list[int], device: str = "cuda") -> torch.Tensor:

    W_dec = sae.W_dec
    ids_tensor = torch.tensor(feature_ids, dtype=torch.long, device=W_dec.device)
    directions = W_dec[ids_tensor]

    norms = directions.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    directions = directions / norms
    summed = directions.sum(dim=0)
    return summed.to(device)
