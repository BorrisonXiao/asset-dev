"""Small, testable utilities for first-order bilevel reward lookahead.

The production trainer keeps hard boundary sampling and GRPO.  A temporary SGD
step adapts selected inner parameters on a support batch, the adapted model is
scored on a disjoint query batch, and every parameter is restored before the
real optimizer update.  No gradient is differentiated through the inner step.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import torch


def support_query_slices(batch_size: int, support_fraction: float = 0.5):
    """Return non-empty contiguous support/query slices for one batch."""
    if batch_size < 2:
        raise ValueError(
            "Bilevel support/query training needs at least two utterances per "
            f"batch, got {batch_size}. Increase max_batch_length_train."
        )
    if not 0.0 < support_fraction < 1.0:
        raise ValueError(
            "bilevel_support_fraction must be strictly between zero and one, "
            f"got {support_fraction}"
        )
    split = round(batch_size * support_fraction)
    split = min(max(int(split), 1), batch_size - 1)
    return slice(0, split), slice(split, batch_size)


def _slice_value(value, selection):
    """Slice tensors, PaddedData namedtuples, or Python sequences."""
    if hasattr(value, "data") and hasattr(value, "lengths"):
        return type(value)(value.data[selection], value.lengths[selection])
    if torch.is_tensor(value):
        return value[selection]
    if isinstance(value, list):
        return value[selection]
    if isinstance(value, tuple):
        return value[selection]
    raise TypeError(f"Cannot slice batch value of type {type(value).__name__}")


def decoder_batch_view(batch, selection):
    """Make the minimal batch view consumed by decoder/reward helpers."""
    view = SimpleNamespace()
    for name in ("tokens_bos", "tokens_eos", "prompt_len", "wrd", "id"):
        if hasattr(batch, name):
            setattr(view, name, _slice_value(getattr(batch, name), selection))
    return view


def unique_trainable_parameters(*modules):
    """Collect trainable parameters once, preserving module/parameter order."""
    parameters = []
    seen = set()
    for module in modules:
        if module is None:
            continue
        for parameter in module.parameters():
            if parameter.requires_grad and id(parameter) not in seen:
                parameters.append(parameter)
                seen.add(id(parameter))
    return parameters


def gradient_global_norm(gradients):
    """FP32 global L2 norm over a list containing tensors or ``None``."""
    terms = [
        gradient.detach().float().square().sum()
        for gradient in gradients
        if gradient is not None
    ]
    if not terms:
        return torch.zeros(())
    return torch.stack(terms).sum().sqrt()


def gradient_diagnostics(gradients):
    """Return precise coverage and magnitude diagnostics for inner gradients."""
    gradients = list(gradients)
    used = [gradient for gradient in gradients if gradient is not None]
    nonzero = [gradient for gradient in used if torch.count_nonzero(gradient)]
    max_abs = max(
        (float(gradient.detach().float().abs().max()) for gradient in used),
        default=0.0,
    )
    return {
        "parameter_tensors": len(gradients),
        "gradient_tensors": len(used),
        "nonzero_gradient_tensors": len(nonzero),
        "gradient_max_abs": max_abs,
    }


def capture_rng_state(device):
    """Capture CPU and, when applicable, current-device CUDA RNG state."""
    device = torch.device(device)
    state = {"cpu": torch.random.get_rng_state()}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state


@contextmanager
def replay_rng_state(state, device):
    """Replay a forward's RNG state without rewinding the caller afterward."""
    device = torch.device(device)
    devices = []
    if device.type == "cuda":
        devices = [
            device.index
            if device.index is not None
            else torch.cuda.current_device()
        ]
    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.random.set_rng_state(state["cpu"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda"], device)
        yield


@contextmanager
def temporary_sgd_update(
    parameters,
    gradients,
    learning_rate: float,
    max_grad_norm: float = 0.0,
):
    """Apply one reversible SGD step and restore bit-identical values on exit.

    The helper never touches ``parameter.grad`` or optimizer state, so an outer
    gradient-accumulation schedule remains intact.  Gradient clipping is global
    and optional; a non-positive ``max_grad_norm`` disables it.
    """
    parameters = list(parameters)
    gradients = list(gradients)
    if len(parameters) != len(gradients):
        raise ValueError("parameters and gradients must have equal length")
    if learning_rate < 0.0:
        raise ValueError("learning_rate must be non-negative")

    originals = [parameter.detach().clone() for parameter in parameters]
    grad_norm = gradient_global_norm(gradients)
    diagnostics = gradient_diagnostics(gradients)
    grad_norm_value = float(grad_norm)
    if not torch.isfinite(grad_norm):
        raise FloatingPointError("Non-finite inner-loop gradient norm")
    scale = 1.0
    if max_grad_norm > 0.0 and grad_norm_value > max_grad_norm:
        scale = max_grad_norm / (grad_norm_value + 1.0e-12)

    try:
        with torch.no_grad():
            for parameter, gradient in zip(parameters, gradients):
                if gradient is not None:
                    parameter.add_(
                        gradient,
                        alpha=-float(learning_rate) * scale,
                    )
        yield {
            "grad_norm": grad_norm_value,
            "grad_scale": scale,
            "update_norm": float(learning_rate) * scale * grad_norm_value,
            **diagnostics,
        }
    finally:
        with torch.no_grad():
            for parameter, original in zip(parameters, originals):
                parameter.copy_(original)


def reward_rank_diagnostics(current_rewards, adapted_rewards):
    """Summarize how one-step adaptation changes K-rollout reward rankings."""
    if current_rewards.shape != adapted_rewards.shape:
        raise ValueError(
            "current and adapted rewards must have the same shape, got "
            f"{tuple(current_rewards.shape)} and {tuple(adapted_rewards.shape)}"
        )
    if current_rewards.ndim != 2 or current_rewards.size(1) < 2:
        raise ValueError("reward tensors must have shape (batch, K>=2)")

    current = current_rewards.detach().float()
    adapted = adapted_rewards.detach().float()
    top_flip = (current.argmax(dim=1) != adapted.argmax(dim=1)).float().mean()

    current_rank = torch.argsort(torch.argsort(current, dim=1), dim=1).float()
    adapted_rank = torch.argsort(torch.argsort(adapted, dim=1), dim=1).float()
    current_rank = current_rank - current_rank.mean(dim=1, keepdim=True)
    adapted_rank = adapted_rank - adapted_rank.mean(dim=1, keepdim=True)
    rank_corr = (
        (current_rank * adapted_rank).sum(dim=1)
        / (
            current_rank.square().sum(dim=1).sqrt()
            * adapted_rank.square().sum(dim=1).sqrt()
        ).clamp(min=1.0e-8)
    ).mean()
    delta = adapted - current
    return {
        "reward_delta": float(delta.mean()),
        "reward_delta_abs": float(delta.abs().mean()),
        "top_rollout_flip": float(top_flip),
        "rank_correlation": float(rank_corr),
    }
