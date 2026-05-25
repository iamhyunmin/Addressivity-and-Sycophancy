
from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import torch


def _get_layer_module(model: Any, layer: int) -> Any:

    try:
        return model.model.layers[layer]
    except (AttributeError, IndexError) as exc:
        raise ValueError(
            f"Cannot access model.model.layers[{layer}]. "
            "Ensure the model is Llama or Gemma compatible."
        ) from exc


class AllTokenSteerer:


    def __init__(self, direction: torch.Tensor, alpha: float):
        self.direction = direction
        self.alpha = alpha

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        delta = self.alpha * self.direction.to(hidden_states.device, hidden_states.dtype)
        hidden_states = hidden_states + delta.unsqueeze(0).unsqueeze(0)

        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        return hidden_states


@contextmanager
def all_token_steering_hook(model: Any, layer: int, direction: torch.Tensor, alpha: float):

    steerer = AllTokenSteerer(direction=direction, alpha=alpha)
    layer_module = _get_layer_module(model, layer)
    handle = layer_module.register_forward_hook(steerer)
    try:
        yield
    finally:
        handle.remove()


class NegSteerer:


    def __init__(self, sae: Any, feature_id: int, direction: torch.Tensor):
        self.sae = sae
        self.feature_id = feature_id
        self.direction = direction

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        orig_shape = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        flat = hidden_states.reshape(-1, orig_shape[-1]).to(self.sae.device, self.sae.dtype)

        with torch.inference_mode():
            acts = self.sae.encode(flat)
            a_j = acts[:, self.feature_id]


        delta = a_j.unsqueeze(1) * self.direction.unsqueeze(0).to(flat.device, flat.dtype)
        hidden_states = (flat - delta).reshape(orig_shape).to(device, dtype)

        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        return hidden_states


@contextmanager
def neg_steer_hook(model: Any, layer: int, sae: Any, feature_id: int, direction: torch.Tensor):

    steerer = NegSteerer(sae=sae, feature_id=feature_id, direction=direction)
    layer_module = _get_layer_module(model, layer)
    handle = layer_module.register_forward_hook(steerer)
    try:
        yield
    finally:
        handle.remove()


class NormPreservingAblator:


    def __init__(self, sae: Any, feature_ids: list[int]):
        self.sae = sae
        self.feature_ids = feature_ids

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        orig_shape = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        flat = hidden_states.reshape(-1, orig_shape[-1])
        orig_norms = flat.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        flat_sae = flat.to(self.sae.device, self.sae.dtype)

        with torch.inference_mode():
            acts = self.sae.encode(flat_sae)
            for fid in self.feature_ids:
                acts[:, fid] = 0.0
            reconstructed = self.sae.decode(acts)


        recon_norms = reconstructed.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        rescaled = reconstructed * (orig_norms.to(reconstructed.device, reconstructed.dtype) / recon_norms)

        hidden_states = rescaled.reshape(orig_shape).to(device, dtype)

        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        return hidden_states


@contextmanager
def norm_preserving_ablation_hook(model: Any, layer: int, sae: Any, feature_ids: list[int]):

    ablator = NormPreservingAblator(sae=sae, feature_ids=feature_ids)
    layer_module = _get_layer_module(model, layer)
    handle = layer_module.register_forward_hook(ablator)
    try:
        yield
    finally:
        handle.remove()


class ZeroAblator:


    def __init__(self, sae: Any, feature_ids: list[int]):
        self.sae = sae
        self.feature_ids = feature_ids

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        orig_shape = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype


        flat = hidden_states.reshape(-1, orig_shape[-1]).to(self.sae.device, self.sae.dtype)

        with torch.inference_mode():

            acts = self.sae.encode(flat)


            for fid in self.feature_ids:
                acts[:, fid] = 0.0


            reconstructed = self.sae.decode(acts)


        hidden_states = reconstructed.reshape(orig_shape).to(device, dtype)

        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        return hidden_states


@contextmanager
def zero_ablation_hook(model: Any, layer: int, sae: Any, feature_ids: list[int]):

    ablator = ZeroAblator(sae=sae, feature_ids=feature_ids)
    layer_module = _get_layer_module(model, layer)
    handle = layer_module.register_forward_hook(ablator)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def no_steering():

    yield
