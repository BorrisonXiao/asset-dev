"""Regression tests for the LS960 optimizer-step training schedule."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import speechbrain as sb


RECIPE_DIR = (
    Path(__file__).resolve().parents[2]
    / "recipes"
    / "LibriSpeech"
    / "ASR"
    / "transformer"
)
sys.path.insert(0, str(RECIPE_DIR))

from segment_pooling import fixed_rate_boundary_targets  # noqa: E402
from segmenter import Segmenter  # noqa: E402
from train_speechllm_with_segmenter import (  # noqa: E402
    SegmenterASR,
    alignment_boundary_targets,
    freeze_boundary_policy_parameters,
    resolve_effective_grad_accumulation,
    resolve_fixed_rate_k,
    resolve_fractional_warmup_steps,
    resolve_joint_boundary_source,
    resolve_optimizer_step_limit,
    step_validation_reason,
)


def test_ls960_warmup_uses_effective_cli_accumulation():
    effective = resolve_effective_grad_accumulation(
        hparams_value=4,
        run_option_value=1,
        run_option_was_overridden=True,
    )
    assert effective == 1
    assert resolve_fractional_warmup_steps(11977, effective, 0.20) == 2395


def test_yaml_accumulation_is_used_without_cli_override():
    assert resolve_effective_grad_accumulation(4, 1, False) == 4
    with pytest.raises(ValueError, match="positive"):
        resolve_effective_grad_accumulation(0, 1, False)


def test_optional_cli_step_limit_is_normalized_from_string():
    assert resolve_optimizer_step_limit("24000") == 24000
    assert resolve_optimizer_step_limit(None) is None
    with pytest.raises(ValueError, match="positive"):
        resolve_optimizer_step_limit("0")


def test_optional_fixed_rate_is_normalized_and_validated():
    assert resolve_fixed_rate_k(None) is None
    assert resolve_fixed_rate_k("5") == 5
    with pytest.raises(ValueError, match="positive"):
        resolve_fixed_rate_k(0)


def test_joint_external_boundary_source_is_normalized_and_validated():
    assert resolve_joint_boundary_source(None) is None
    assert resolve_joint_boundary_source("none") is None
    assert resolve_joint_boundary_source("ALIGNMENT") == "alignment"
    with pytest.raises(ValueError, match="support"):
        resolve_joint_boundary_source("posterior")


def test_external_alignment_boundaries_are_validated_and_pad_masked():
    targets = torch.tensor([[1, 0, 1, 0], [1, 1, 0, 0]])
    pad_mask = torch.tensor(
        [[False, False, False, False], [False, False, True, True]]
    )
    boundary = alignment_boundary_targets(targets, pad_mask)
    assert boundary.tolist() == [[1, 0, 1, 0], [1, 1, -1, -1]]

    with pytest.raises(ValueError, match="frame zero"):
        alignment_boundary_targets(torch.tensor([[0, 1]]), torch.zeros(1, 2, dtype=torch.bool))


def test_freeze_boundary_policy_retains_trainable_pooler():
    class ToySegmenter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.film = torch.nn.Linear(3, 3)
            self.net = torch.nn.Linear(3, 1)
            self.pooler = torch.nn.Linear(3, 3)

    segmenter = ToySegmenter()
    frozen, trainable = freeze_boundary_policy_parameters(segmenter)

    assert frozen > 0
    assert trainable > 0
    assert all(
        not parameter.requires_grad
        for name, parameter in segmenter.named_parameters()
        if not name.startswith("pooler.")
    )
    assert all(
        parameter.requires_grad
        for name, parameter in segmenter.named_parameters()
        if name.startswith("pooler.")
    )


def test_freeze_boundary_policy_accepts_parameter_free_mean_pooling():
    class MeanPoolSegmenter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.film = torch.nn.Linear(3, 3)
            self.net = torch.nn.Linear(3, 1)

    segmenter = MeanPoolSegmenter()
    frozen, trainable = freeze_boundary_policy_parameters(segmenter)

    assert frozen > 0
    assert trainable == 0
    assert all(
        not parameter.requires_grad for parameter in segmenter.parameters()
    )


def test_joint_optimizer_allows_frozen_policy_with_parameter_free_pooling():
    segmenter = torch.nn.Linear(3, 1)
    for parameter in segmenter.parameters():
        parameter.requires_grad = False
    brain = SimpleNamespace(
        modules=SimpleNamespace(
            segmenter=segmenter,
            proj=torch.nn.Linear(3, 3),
            llm=torch.nn.Linear(3, 3),
        ),
        hparams=SimpleNamespace(
            segmenter_mode="joint",
            freeze_boundary_policy=True,
            lr_segmenter=5.0e-5,
            lr_decoder=2.0e-4,
            weight_decay=0.0,
        ),
        checkpointer=None,
    )

    SegmenterASR.init_optimizers(brain)

    assert len(brain.optimizer.param_groups) == 2
    assert brain.optimizer.param_groups[0]["params"] == []
    assert brain.optimizer.param_groups[1]["params"]


def test_fixed_boundaries_train_bigru_without_policy_gradients():
    segmenter = Segmenter(
        input_dim=8,
        backbone="transformer_ar",
        hidden_dim=8,
        num_layers=1,
        nhead=2,
        ffn_dim=16,
        ar_hidden_dim=8,
        ar_num_layers=1,
        ar_nhead=2,
        ar_ffn_dim=16,
        ar_dropout=0.0,
        ar_history_window=4,
        pooling="bigru_residual",
        pooling_input_dim=8,
        pooling_hidden_dim=4,
    )
    freeze_boundary_policy_parameters(segmenter)
    features = torch.randn(2, 12, 8)
    pad_mask = torch.zeros(2, 12, dtype=torch.bool)
    pad_mask[1, 9:] = True
    boundary = fixed_rate_boundary_targets(pad_mask, 5)

    pooled, pooled_pad = segmenter.pool(features, pad_mask, boundary)
    pooled.masked_fill(pooled_pad.unsqueeze(-1), 0.0).square().mean().backward()

    policy_parameters = [
        parameter
        for name, parameter in segmenter.named_parameters()
        if not name.startswith("pooler.")
    ]
    pooler_parameters = list(segmenter.pooler.parameters())
    assert all(parameter.grad is None for parameter in policy_parameters)
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in pooler_parameters
    )


def test_frozen_boundary_policy_uses_decoder_pooler_path_for_all_train_epochs():
    brain = SegmenterASR.__new__(SegmenterASR)
    brain.hparams = SimpleNamespace(
        segmenter_mode="joint",
        freeze_boundary_policy=True,
        warmup_optimizer_steps=None,
        warmup_epochs=0,
    )
    brain.current_epoch = 99
    brain.optimizer_step = 999

    assert brain._is_warmup(sb.Stage.TRAIN)
    assert not brain._is_warmup(sb.Stage.VALID)


@pytest.mark.parametrize(
    ("step", "expected"),
    [
        (2395, "warmup_end"),
        (4000, "interval"),
        (8000, "interval"),
        (23999, None),
        (24000, "interval"),
    ],
)
def test_ls960_step_validation_schedule(step, expected):
    assert (
        step_validation_reason(
            step,
            4000,
            warmup_step=2395,
            max_steps=24000,
        )
        == expected
    )


def test_final_and_duplicate_validation_handling():
    assert (
        step_validation_reason(
            23954,
            4000,
            warmup_step=2395,
            max_steps=23954,
        )
        == "final"
    )
    assert (
        step_validation_reason(
            4000,
            4000,
            warmup_step=2395,
            max_steps=24000,
            last_validated_step=4000,
        )
        is None
    )


def test_fit_batch_hook_validates_at_warmup_boundary():
    brain = SegmenterASR.__new__(SegmenterASR)
    brain.hparams = SimpleNamespace(
        validation_interval_optimizer_steps=4000,
        warmup_optimizer_steps=2395,
        validate_at_warmup_end=True,
    )
    brain.optimizer_step = 2395
    brain.optimizer_step_limit = 24000
    brain._step_validation_set = object()
    brain._last_step_validation = None
    reasons = []
    brain._run_step_validation = reasons.append

    brain.on_fit_batch_end(None, None, None, should_step=True)

    assert reasons == ["warmup_end"]
