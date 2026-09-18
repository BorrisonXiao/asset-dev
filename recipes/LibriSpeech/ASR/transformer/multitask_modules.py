"""Task-routed decoder adapters and warm-start helpers for multi-task runs.

Two properties matter more than anything else here, because the whole study
rests on them:

1. At step zero, ``task_id=asr`` must reproduce the warm-started ASR model
   exactly -- same boundaries, same pooled states, same logits. The segmenter
   FiLM is identity initialized already (``segmenter.TaskFiLM``); this module
   supplies the matching decoder-side property by copying the trained ASR LoRA
   into task slot 0 verbatim.
2. The ST slot must start from a stable small initialization. LoRA's up
   projection is zero initialized, so a fresh ST slot starts as the frozen base
   LLM and cannot inject noise into the shared acoustic interface on step one.

Nothing in this module is imported by the ASR-only recipes.
"""

import logging

import torch
import torch.nn as nn

from speechbrain.nnet.adapters import LoRA

logger = logging.getLogger(__name__)


class TaskRouter:
    """Shared mutable pointer to the task all routed adapters should use.

    Batches are task homogeneous, so one active index per forward is exactly
    right: a single ``set_task`` call switches every adapter in the model at
    once, with no per-example gather in the hot path.
    """

    def __init__(self, num_tasks, active=0):
        if num_tasks < 1:
            raise ValueError(f"num_tasks must be >= 1, got {num_tasks}")
        self.num_tasks = int(num_tasks)
        self.active = int(active)

    def set_task(self, task_id):
        """Point every routed adapter at ``task_id``."""
        task_id = int(task_id)
        if not 0 <= task_id < self.num_tasks:
            raise ValueError(
                f"task_id {task_id} out of range for {self.num_tasks} tasks"
            )
        self.active = task_id
        return self.active


class TaskRoutedLoRA(nn.Module):
    """LoRA with one independent low-rank pair per task, selected by a router.

    Slot ``t`` is used whenever ``router.active == t``. The frozen pretrained
    module is shared across tasks, so the extra cost is only
    ``num_tasks x rank x (in + out)`` parameters per adapted layer.

    Arguments
    ---------
    target_module : nn.Module
        Pretrained linear layer to wrap.
    router : TaskRouter
        Shared task pointer.
    rank : int
        LoRA rank.
    alpha : float
        LoRA scaling numerator.
    """

    def __init__(self, target_module, router, rank=16, alpha=1.0):
        super().__init__()
        self.router = router
        self.pretrained_module = target_module
        for param in self.pretrained_module.parameters():
            param.requires_grad = False

        input_size = target_module.weight.data.shape[1]
        output_size = target_module.weight.data.shape[0]
        device = target_module.weight.device

        self.adapter_down_proj = nn.ModuleList(
            nn.Linear(input_size, rank, bias=False, device=device)
            for _ in range(router.num_tasks)
        )
        self.adapter_up_proj = nn.ModuleList(
            nn.Linear(rank, output_size, bias=False, device=device)
            for _ in range(router.num_tasks)
        )
        for up in self.adapter_up_proj:
            up.weight.data.fill_(0.0)

        self.scaling = alpha / rank

    def forward(self, x):
        """Apply the frozen layer plus the active task's low-rank update."""
        task = self.router.active
        delta = self.adapter_up_proj[task](self.adapter_down_proj[task](x))
        return self.pretrained_module(x) + delta * self.scaling


def make_task_routed_lora(router):
    """Adapter factory for ``AdaptedModel(adapter_class=...)``."""

    def _factory(target_module, **kwargs):
        return TaskRoutedLoRA(target_module, router, **kwargs)

    return _factory


def copy_asr_lora_into_task_slot(model, state_dict, task_id=0, strict=True):
    """Copy a single-task ASR LoRA checkpoint into one routed task slot.

    The saved ASR checkpoint stores ``...adapter_down_proj.weight``; a routed
    model stores ``...adapter_down_proj.<task>.weight``. Mapping is purely
    mechanical, but doing it explicitly (rather than via ``strict=False``) is
    what makes a silent mismatch impossible -- and a silent mismatch here would
    invalidate the entire frozen-equivalence argument.

    Returns
    -------
    dict
        ``{"copied": int, "skipped": list}``
    """
    own_state = model.state_dict()
    copied = 0
    skipped = []
    for name, tensor in state_dict.items():
        routed = None
        for stem in ("adapter_down_proj", "adapter_up_proj"):
            marker = f".{stem}."
            if marker in name:
                head, tail = name.split(marker, 1)
                routed = f"{head}.{stem}.{task_id}.{tail}"
                break
        if routed is None:
            # Non-LoRA trainables (e.g. unfrozen layers) keep their name.
            routed = name
        if routed not in own_state:
            skipped.append(name)
            continue
        if own_state[routed].shape != tensor.shape:
            raise ValueError(
                f"Shape mismatch restoring {name} -> {routed}: "
                f"checkpoint {tuple(tensor.shape)} vs "
                f"model {tuple(own_state[routed].shape)}"
            )
        own_state[routed].copy_(tensor)
        copied += 1

    if skipped and strict:
        raise ValueError(
            f"{len(skipped)} checkpoint tensors had no routed destination, "
            f"first few: {skipped[:5]}"
        )
    logger.info(
        "Copied %d ASR LoRA tensors into task slot %d (%d skipped)",
        copied,
        task_id,
        len(skipped),
    )
    return {"copied": copied, "skipped": skipped}


def expand_film_to_tasks(segmenter, num_tasks):
    """Grow an identity ``TaskFiLM`` to ``num_tasks``, preserving identity.

    A single-task warm start has FiLM disabled (pure identity). Enabling it for
    the multi-task run must not change the function: every task starts at
    scale 1 / bias 0, so at step zero each task reproduces the warm-started
    computation and any later divergence is attributable to training.
    """
    film = segmenter.film
    if film.enabled and film.gamma.num_embeddings == num_tasks:
        return False
    dim = _film_dim(segmenter)
    device = next(segmenter.parameters()).device
    film.gamma = nn.Embedding(num_tasks, dim, device=device)
    film.beta = nn.Embedding(num_tasks, dim, device=device)
    nn.init.ones_(film.gamma.weight)
    nn.init.zeros_(film.beta.weight)
    film.enabled = True
    logger.info(
        "Expanded segmenter TaskFiLM to %d tasks (identity initialized, dim=%d)",
        num_tasks,
        dim,
    )
    return True


def _film_dim(segmenter):
    """Feature dimension the segmenter FiLM modulates."""
    film = segmenter.film
    if getattr(film, "enabled", False):
        return film.gamma.embedding_dim
    # FiLM sits on the raw segmenter input, i.e. the boundary backbone's input.
    for module in segmenter.net.modules():
        if isinstance(module, nn.Conv1d):
            return module.in_channels
        if isinstance(module, nn.Linear):
            return module.in_features
    raise ValueError("Could not infer the segmenter FiLM feature dimension")


def film_parameters(segmenter):
    """Trainable task-FiLM parameters (the Phase B policy parameters)."""
    film = segmenter.film
    if not getattr(film, "enabled", False):
        return []
    return list(film.gamma.parameters()) + list(film.beta.parameters())


def routed_lora_parameters(model, task_id=None):
    """Task-routed adapter parameters, optionally restricted to one task."""
    params = []
    for module in model.modules():
        if not isinstance(module, TaskRoutedLoRA):
            continue
        slots = (
            range(module.router.num_tasks)
            if task_id is None
            else [int(task_id)]
        )
        for slot in slots:
            params.extend(module.adapter_down_proj[slot].parameters())
            params.extend(module.adapter_up_proj[slot].parameters())
    return params


@torch.no_grad()
def count_task_conditioned_parameters(segmenter, llm):
    """Parameter accounting the plan requires reporting for the main system."""
    film = sum(p.numel() for p in film_parameters(segmenter))
    per_task = {}
    for module in llm.modules():
        if isinstance(module, TaskRoutedLoRA):
            for slot in range(module.router.num_tasks):
                per_task[slot] = per_task.get(slot, 0) + sum(
                    p.numel()
                    for p in list(module.adapter_down_proj[slot].parameters())
                    + list(module.adapter_up_proj[slot].parameters())
                )
    return {
        "task_film": film,
        "routed_lora_per_task": per_task,
        "routed_lora_total": sum(per_task.values()),
    }


__all__ = [
    "LoRA",
    "TaskRouter",
    "TaskRoutedLoRA",
    "make_task_routed_lora",
    "copy_asr_lora_into_task_slot",
    "expand_film_to_tasks",
    "film_parameters",
    "routed_lora_parameters",
    "count_task_conditioned_parameters",
]
