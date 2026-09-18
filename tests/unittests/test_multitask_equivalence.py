"""Frozen-equivalence guarantees for the task-conditioned multi-task extension.

The scientific claim only holds if ``task_id=asr`` reproduces the warm-started
single-task ASR model *exactly* at step zero. Otherwise any measured change --
in WER, in BLEU, or in where boundaries land -- could be an artifact of the
warm start rather than of task conditioning. These tests pin that property down
on both sides of the model: the decoder adapters and the segmenter policy.
"""

import os
import sys
from collections import OrderedDict

import pytest
import torch
import torch.nn as nn

RECIPE_DIR = os.path.join(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ),
    "recipes",
    "LibriSpeech",
    "ASR",
    "transformer",
)
if RECIPE_DIR not in sys.path:
    sys.path.insert(0, RECIPE_DIR)

from multitask_modules import (  # noqa: E402
    TaskRoutedLoRA,
    TaskRouter,
    copy_asr_lora_into_task_slot,
    count_task_conditioned_parameters,
    expand_film_to_tasks,
    film_parameters,
    make_task_routed_lora,
)
from segmenter import Segmenter  # noqa: E402

from speechbrain.nnet.adapters import AdaptedModel, LoRA  # noqa: E402


def _base_model():
    return nn.Sequential(
        OrderedDict(
            [
                ("layer1", nn.Linear(12, 16)),
                ("act", nn.ReLU()),
                ("layer2", nn.Linear(16, 12)),
            ]
        )
    )


def _trained_single_task_lora(seed=0):
    """A single-task ASR LoRA model with non-trivial adapter weights."""
    torch.manual_seed(seed)
    model = AdaptedModel(
        model_to_adapt=_base_model(),
        adapter_class=LoRA,
        all_linear=True,
        adapter_kwargs={"rank": 4},
    )
    # LoRA up projections start at zero; training moves them off zero, and the
    # equivalence test is only meaningful against a non-identity adapter.
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, LoRA):
                module.adapter_up_proj.weight.normal_(std=0.05)
                module.adapter_down_proj.weight.normal_(std=0.05)
    return model


def _routed_model(router, seed=1):
    torch.manual_seed(seed)
    return AdaptedModel(
        model_to_adapt=_base_model(),
        adapter_class=make_task_routed_lora(router),
        all_linear=True,
        adapter_kwargs={"rank": 4},
    )


def _copy_frozen_base(source, target):
    """Give both models identical frozen pretrained weights."""
    src = {
        name: param
        for name, param in source.state_dict().items()
        if "pretrained_module" in name
    }
    missing = target.load_state_dict(src, strict=False)
    assert not [k for k in missing.unexpected_keys], missing.unexpected_keys


def test_asr_slot_reproduces_the_single_task_model_exactly():
    single = _trained_single_task_lora()
    router = TaskRouter(num_tasks=2)
    routed = _routed_model(router)
    _copy_frozen_base(single, routed)

    trained_state = {
        name: param.detach()
        for name, param in single.state_dict(keep_vars=True).items()
        if param.requires_grad
    }
    report = copy_asr_lora_into_task_slot(routed, trained_state, task_id=0)
    assert report["copied"] == len(trained_state)
    assert report["skipped"] == []

    x = torch.randn(3, 5, 12)
    single.eval()
    routed.eval()
    with torch.no_grad():
        expected = single(x)
        router.set_task(0)
        actual = routed(x)
    # Bit-for-bit, not approximately: the warm start must be the identity.
    assert torch.equal(expected, actual)


def test_st_slot_starts_as_the_frozen_base_model():
    single = _trained_single_task_lora()
    router = TaskRouter(num_tasks=2)
    routed = _routed_model(router)
    _copy_frozen_base(single, routed)
    trained_state = {
        name: param.detach()
        for name, param in single.state_dict(keep_vars=True).items()
        if param.requires_grad
    }
    copy_asr_lora_into_task_slot(routed, trained_state, task_id=0)

    x = torch.randn(3, 5, 12)
    routed.eval()
    with torch.no_grad():
        router.set_task(1)
        st_out = routed(x)
        base_out = single.adapted_model.layer2.pretrained_module(
            torch.relu(single.adapted_model.layer1.pretrained_module(x))
        )
    # A fresh ST slot has zero up projections, so it cannot perturb the shared
    # acoustic interface on the first optimizer step.
    assert torch.equal(st_out, base_out)


def test_task_slots_are_independent():
    router = TaskRouter(num_tasks=2)
    routed = _routed_model(router)
    with torch.no_grad():
        for module in routed.modules():
            if isinstance(module, TaskRoutedLoRA):
                module.adapter_up_proj[1].weight.normal_(std=0.1)
                module.adapter_down_proj[1].weight.normal_(std=0.1)
    x = torch.randn(2, 4, 12)
    routed.eval()
    with torch.no_grad():
        router.set_task(0)
        asr_out = routed(x)
        router.set_task(1)
        st_out = routed(x)
    assert not torch.allclose(asr_out, st_out)


def test_router_rejects_out_of_range_task():
    router = TaskRouter(num_tasks=2)
    with pytest.raises(ValueError, match="out of range"):
        router.set_task(2)
    with pytest.raises(ValueError, match="out of range"):
        router.set_task(-1)


def test_copy_reports_unmatched_checkpoint_tensors():
    router = TaskRouter(num_tasks=2)
    routed = _routed_model(router)
    with pytest.raises(ValueError, match="no routed destination"):
        copy_asr_lora_into_task_slot(
            routed, {"nonexistent.adapter_down_proj.weight": torch.zeros(4, 12)}
        )


def test_copy_rejects_shape_mismatch():
    router = TaskRouter(num_tasks=2)
    routed = _routed_model(router)
    name = next(
        n
        for n in routed.state_dict()
        if n.endswith("adapter_down_proj.0.weight")
    )
    single_name = name.replace("adapter_down_proj.0.", "adapter_down_proj.")
    with pytest.raises(ValueError, match="Shape mismatch"):
        copy_asr_lora_into_task_slot(routed, {single_name: torch.zeros(3, 3)})


def _segmenter(seed=0, num_tasks=1):
    torch.manual_seed(seed)
    return Segmenter(
        input_dim=24,
        backbone="transformer_ar",
        hidden_dim=16,
        ar_hidden_dim=16,
        ar_num_layers=2,
        ar_nhead=2,
        ar_ffn_dim=32,
        ar_history_window=8,
        ar_cache_mode="preallocated",
        num_tasks=num_tasks,
        pooling="bigru_residual",
        pooling_input_dim=24,
        pooling_hidden_dim=8,
    )


def test_film_expansion_preserves_the_warm_started_policy():
    segmenter = _segmenter()
    segmenter.eval()
    features = torch.randn(2, 24, 24)
    pad_mask = torch.zeros(2, 24, dtype=torch.bool)
    pad_mask[1, 18:] = True

    with torch.no_grad():
        before = segmenter.sample_boundaries(
            features, pad_mask, deterministic=True
        )
        pooled_before, _ = segmenter.pool(features, pad_mask, before.boundaries)

    assert not segmenter.film.enabled  # single-task warm start
    assert expand_film_to_tasks(segmenter, 2)
    assert segmenter.film.enabled

    with torch.no_grad():
        for task_id in (0, 1):
            segmenter.active_task_id = torch.full(
                (2,), task_id, dtype=torch.long
            )
            after = segmenter.sample_boundaries(
                features, pad_mask, deterministic=True
            )
            pooled_after, _ = segmenter.pool(
                features, pad_mask, after.boundaries
            )
            # Identity FiLM: same mask, same logits, same pooled states.
            assert torch.equal(before.boundaries, after.boundaries)
            assert torch.equal(before.logits, after.logits)
            assert torch.equal(pooled_before, pooled_after)


def test_active_task_id_none_is_the_untouched_single_task_path():
    segmenter = _segmenter()
    segmenter.eval()
    features = torch.randn(2, 20, 24)
    pad_mask = torch.zeros(2, 20, dtype=torch.bool)
    with torch.no_grad():
        reference = segmenter.sample_boundaries(
            features, pad_mask, deterministic=True
        )
    expand_film_to_tasks(segmenter, 2)
    segmenter.active_task_id = None  # the ASR-only recipes never set this
    with torch.no_grad():
        result = segmenter.sample_boundaries(
            features, pad_mask, deterministic=True
        )
    assert torch.equal(reference.boundaries, result.boundaries)
    assert torch.equal(reference.logits, result.logits)


def test_trained_film_can_change_boundaries_per_task():
    segmenter = _segmenter(seed=3)
    expand_film_to_tasks(segmenter, 2)
    segmenter.eval()
    features = torch.randn(2, 32, 24)
    pad_mask = torch.zeros(2, 32, dtype=torch.bool)

    # Simulate a trained ST FiLM slot; the ASR slot stays identity.
    with torch.no_grad():
        segmenter.film.gamma.weight[1].normal_(mean=1.0, std=0.5)
        segmenter.film.beta.weight[1].normal_(mean=0.0, std=0.5)
        segmenter.active_task_id = torch.zeros(2, dtype=torch.long)
        asr = segmenter.sample_boundaries(
            features, pad_mask, deterministic=True
        )
        segmenter.active_task_id = torch.ones(2, dtype=torch.long)
        st = segmenter.sample_boundaries(features, pad_mask, deterministic=True)
    # Conditioning must be able to move the mask, or the study has no mechanism.
    assert not torch.equal(asr.boundaries, st.boundaries)


def test_film_parameters_and_accounting():
    segmenter = _segmenter()
    assert film_parameters(segmenter) == []
    expand_film_to_tasks(segmenter, 2)
    params = film_parameters(segmenter)
    assert len(params) == 2
    assert sum(p.numel() for p in params) == 2 * 2 * 24

    router = TaskRouter(num_tasks=2)
    routed = _routed_model(router)
    accounting = count_task_conditioned_parameters(segmenter, routed)
    assert accounting["task_film"] == 2 * 2 * 24
    assert set(accounting["routed_lora_per_task"]) == {0, 1}
    assert (
        accounting["routed_lora_per_task"][0]
        == accounting["routed_lora_per_task"][1]
    )
    assert (
        accounting["routed_lora_total"]
        == 2 * accounting["routed_lora_per_task"][0]
    )


def test_routed_lora_gradients_reach_only_the_active_task():
    router = TaskRouter(num_tasks=2)
    routed = _routed_model(router)
    router.set_task(1)
    routed(torch.randn(2, 3, 12)).sum().backward()
    for module in routed.modules():
        if isinstance(module, TaskRoutedLoRA):
            # The inactive task's adapter must not accumulate gradient from a
            # batch of the other task.
            assert module.adapter_down_proj[0].weight.grad is None
            assert module.adapter_down_proj[1].weight.grad is not None


def test_film_accepts_scalar_and_replicated_task_ids():
    """The GRPO rollout paths replicate the batch K times before conditioning.

    A per-example task tensor sized to the original batch then no longer matches
    the replicated features, so FiLM must accept a scalar (and tile a short
    per-example tensor) rather than requiring an exact-length index.
    """
    segmenter = _segmenter(seed=5)
    expand_film_to_tasks(segmenter, 2)
    with torch.no_grad():
        segmenter.film.gamma.weight[1].normal_(mean=1.0, std=0.3)
        segmenter.film.beta.weight[1].normal_(mean=0.0, std=0.3)

    features = torch.randn(3, 16, 24)
    scalar = segmenter.film(features, torch.tensor(1))
    per_example = segmenter.film(features, torch.ones(3, dtype=torch.long))
    plain_int = segmenter.film(features, 1)
    assert torch.equal(scalar, per_example)
    assert torch.equal(plain_int, per_example)

    # K=4 replication: 3 utterances -> 12 rows, conditioned by one scalar.
    replicated = features.repeat(4, 1, 1)
    out = segmenter.film(replicated, torch.tensor(1))
    assert out.shape == replicated.shape
    assert torch.equal(out[:3], per_example)
    # A short per-example tensor is tiled in the same order replication uses.
    tiled = segmenter.film(replicated, torch.ones(3, dtype=torch.long))
    assert torch.equal(tiled, out)


def test_film_rejects_unalignable_task_ids():
    segmenter = _segmenter(seed=6)
    expand_film_to_tasks(segmenter, 2)
    features = torch.randn(5, 8, 24)
    with pytest.raises(ValueError, match="whole multiple"):
        segmenter.film(features, torch.zeros(3, dtype=torch.long))


def test_film_and_router_stay_the_same_size_across_the_task_table():
    """The 2-vs-5 trap: FiLM and the routed adapters have separate sizes.

    ``segmenter_num_tasks`` sizes both at construction, but the warm-start path
    then resizes FiLM to ``NUM_TASKS``. If those two numbers ever disagree,
    FiLM is silently rebuilt at the wrong width against a router of the other
    width, and the mismatch surfaces as an out-of-range index at the first
    batch of a high-numbered task -- not at startup.
    """
    from multitask_data import NUM_TASKS

    segmenter = _segmenter(num_tasks=NUM_TASKS)
    router = TaskRouter(num_tasks=NUM_TASKS)
    assert segmenter.film.enabled
    assert segmenter.film.gamma.num_embeddings == NUM_TASKS

    # Expanding to the same count is a no-op that leaves the width alone.
    assert expand_film_to_tasks(segmenter, NUM_TASKS) is False
    assert segmenter.film.gamma.num_embeddings == NUM_TASKS

    # Every task id in the table addresses both structures.
    for task_id in range(NUM_TASKS):
        assert router.set_task(task_id) == task_id
    with pytest.raises(ValueError, match="out of range"):
        router.set_task(NUM_TASKS)


def test_film_expansion_to_a_wider_table_stays_identity():
    """Growing the table must not change what any existing task computes."""
    segmenter = _segmenter(num_tasks=2)
    segmenter.eval()
    features = torch.randn(2, 24, 24)
    pad_mask = torch.zeros(2, 24, dtype=torch.bool)

    # transformer_ar has no prefix-free logits; the deterministic rollout is
    # the comparable quantity, as in the warm-start equivalence test above.
    with torch.no_grad():
        before = segmenter.sample_boundaries(
            features, pad_mask, deterministic=True, task_id=0
        )
    assert expand_film_to_tasks(segmenter, 5) is True
    assert segmenter.film.gamma.num_embeddings == 5
    with torch.no_grad():
        after = segmenter.sample_boundaries(
            features, pad_mask, deterministic=True, task_id=0
        )
    assert torch.equal(before.boundaries, after.boundaries)
