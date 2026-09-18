"""Tests for the rate-term definitions, per-task bands, and collapse logging.

Three things are pinned here, all of which were silent failures before:

* the two rate channels measure the SAME quantity (they feed one shared band);
* per-task bands do not leak into the matched-rate phase, which would dissolve
  the study's central comparison;
* the within-group reward spread is recorded, because a spread of zero is the
  absorbing state that ended the first RL runs and it took seven epochs to spot.
"""

import os
import sys
import types

import pytest
import torch

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

from segment_pooling import compute_segment_ids  # noqa: E402
from segmenter import expected_kept_ratio, realized_kept_ratio  # noqa: E402
from train_speechllm_multitask import (  # noqa: E402
    PHASE_BRIDGE,
    PHASE_FILM,
    PHASE_UNFREEZE,
    MultitaskSegmenterASR,
    validate_task_rate_bands,
)

# ------------------------------------------------------- rate-term agreement


def test_realized_rho_equals_the_true_segment_count():
    # rho must be exactly what pooling produces, or the band is aiming at a
    # quantity the decoder never sees.
    boundary = torch.tensor([[0, 0, 1, 0, 1, 0]])
    pad = torch.tensor([[False, False, False, False, False, True]])
    _, num_segments, _ = compute_segment_ids(boundary, pad)
    n_valid = (~pad).sum(dim=1).float()
    assert torch.allclose(
        realized_kept_ratio(boundary, pad), num_segments.float() / n_valid
    )


def test_a_boundary_at_frame_zero_is_not_double_counted():
    # compute_segment_ids' frame-0 guard: frame 0 already opens a segment, so a
    # boundary predicted there must not open a second one.
    pad = torch.zeros(1, 4, dtype=torch.bool)
    with_b0 = torch.tensor([[1, 0, 1, 0]])
    without = torch.tensor([[0, 0, 1, 0]])
    assert torch.allclose(
        realized_kept_ratio(with_b0, pad), realized_kept_ratio(without, pad)
    )
    _, n_with, _ = compute_segment_ids(with_b0, pad)
    assert float(realized_kept_ratio(with_b0, pad)) == pytest.approx(
        float(n_with) / 4
    )


def test_expected_and_realized_rho_agree_on_a_deterministic_policy():
    # The two rate channels are held to ONE band, so a systematic offset between
    # them means the reward and the auxiliary loss are aiming at different rates.
    pad = torch.tensor([[False] * 5 + [True]])
    boundary = torch.tensor([[0, 0, 1, 0, 1, 0]])
    probs = torch.tensor([[0.0, 0.0, 1.0, 0.0, 1.0, 0.0]])
    _, rho_bar, _ = expected_kept_ratio(None, pad, probabilities=probs)
    assert torch.allclose(rho_bar, realized_kept_ratio(boundary, pad))


def test_expected_rho_counts_the_implicit_first_segment():
    # A policy that never fires a boundary still yields exactly one segment.
    pad = torch.zeros(1, 10, dtype=torch.bool)
    probs = torch.zeros(1, 10)
    _, rho_bar, _ = expected_kept_ratio(None, pad, probabilities=probs)
    assert float(rho_bar) == pytest.approx(1 / 10)
    assert float(
        realized_kept_ratio(torch.zeros(1, 10, dtype=torch.long), pad)
    ) == pytest.approx(1 / 10)


def test_expected_rho_ignores_frame_zero_probability():
    pad = torch.zeros(1, 4, dtype=torch.bool)
    base = torch.tensor([[0.0, 0.5, 0.0, 0.0]])
    withp0 = torch.tensor([[0.9, 0.5, 0.0, 0.0]])
    assert torch.allclose(
        expected_kept_ratio(None, pad, probabilities=base)[1],
        expected_kept_ratio(None, pad, probabilities=withp0)[1],
    )


# --------------------------------------------------------- per-task bands


def _brain(bands, phases=None, phase=PHASE_UNFREEZE, task="emotion"):
    brain = MultitaskSegmenterASR.__new__(MultitaskSegmenterASR)
    brain.active_task_name = task
    brain.hparams = types.SimpleNamespace(
        rho_lo=0.15,
        rho_hi=0.25,
        task_rate_bands=bands,
        task_rate_bands_phases=phases,
    )
    brain._phase = lambda: phase
    return brain


def test_per_task_band_is_used_in_the_allowed_phase():
    brain = _brain({"emotion": [0.04, 0.10]})
    assert brain._rate_band() == (0.04, 0.10)


def test_per_task_bands_do_not_leak_into_the_matched_rate_phase():
    # Phase B is the placement comparison: every task must share one band, or a
    # "better placement" result is really "spent more audio tokens".
    for phase in (PHASE_BRIDGE, PHASE_FILM):
        brain = _brain({"emotion": [0.04, 0.10]}, phase=phase)
        assert brain._rate_band() == (0.15, 0.25)


def test_a_task_with_no_band_falls_back_to_the_global_one():
    brain = _brain({"emotion": [0.04, 0.10]}, task="asr")
    assert brain._rate_band() == (0.15, 0.25)
    brain = _brain(None, task="emotion")
    assert brain._rate_band() == (0.15, 0.25)


def test_phases_are_configurable():
    brain = _brain(
        {"emotion": [0.04, 0.10]},
        phases=[PHASE_FILM, PHASE_UNFREEZE],
        phase=PHASE_FILM,
    )
    assert brain._rate_band() == (0.04, 0.10)


def test_band_validation_rejects_bad_configurations():
    validate_task_rate_bands({"task_rate_bands": {"emotion": [0.04, 0.10]}})
    with pytest.raises(ValueError, match="unknown task"):
        validate_task_rate_bands({"task_rate_bands": {"nope": [0.1, 0.2]}})
    with pytest.raises(ValueError, match=r"\[rho_lo, rho_hi\]"):
        validate_task_rate_bands({"task_rate_bands": {"emotion": [0.1]}})
    with pytest.raises(ValueError, match="rho_lo <= rho_hi"):
        validate_task_rate_bands({"task_rate_bands": {"emotion": [0.3, 0.1]}})
    with pytest.raises(ValueError, match="unknown phases"):
        validate_task_rate_bands(
            {
                "task_rate_bands": {"emotion": [0.04, 0.10]},
                "task_rate_bands_phases": ["phase_b"],
            }
        )


def test_the_shipped_priors_are_ordered_and_in_range():
    import yaml

    text = open(
        os.path.join(RECIPE_DIR, "hparams", "speechllm_multitask.yaml"),
        encoding="utf-8",
    ).read()
    block = text.split("task_rate_bands:")[1].split("entropy_coeff_init:")[0]
    bands = yaml.safe_load("task_rate_bands:" + block)["task_rate_bands"]
    validate_task_rate_bands({"task_rate_bands": bands})
    hz = {k: (50 * v[0], 50 * v[1]) for k, v in bands.items()}
    # The prior's hypothesis: lexical content needs the finest rate, affect the
    # coarsest. Recorded as a test so a later edit has to restate the claim.
    assert hz["asr"][1] > hz["intent"][1] > hz["speaker_count"][1]
    assert hz["speaker_count"][1] > hz["emotion"][1]
    for task, (lo, hi) in hz.items():
        assert 1.0 <= lo < hi <= 12.5, task


# ----------------------------------------------------- reward-spread logging


def test_zero_spread_is_exactly_the_dead_policy():
    from segmenter import group_advantage

    identical = torch.full((3, 4), -1.3)
    spread = identical.std(dim=1)
    assert torch.allclose(spread, torch.zeros(3))
    # std == 0 -> advantage == 0 -> no gradient, ever. This is the quantity the
    # new logging watches.
    assert torch.allclose(group_advantage(identical), torch.zeros(3, 4))
    assert float((spread <= 1e-6).float().mean()) == 1.0


def test_reward_spread_tracker_accumulates_globally_and_per_task():
    brain = MultitaskSegmenterASR.__new__(MultitaskSegmenterASR)
    brain._reward_std_samples = []
    from collections import defaultdict

    brain._task_reward_std = defaultdict(list)
    brain.active_task_name = "emotion"
    brain._track_reward_std(torch.tensor([0.0, 0.5]))
    brain.active_task_name = "asr"
    brain._track_reward_std(torch.tensor([1.5]))
    assert brain._reward_std_samples == [0.0, 0.5, 1.5]
    assert brain._task_reward_std["emotion"] == [0.0, 0.5]
    assert brain._task_reward_std["asr"] == [1.5]


# --------------------------------------------------------- rate channels


def _channel_brain(choice, full_ar=True):
    from train_speechllm_with_segmenter import SegmenterASR

    brain = SegmenterASR.__new__(SegmenterASR)
    brain.hparams = types.SimpleNamespace(rate_channel=choice)
    brain.modules = types.SimpleNamespace(
        segmenter=types.SimpleNamespace(is_full_ar=full_ar)
    )
    return brain


def test_auto_preserves_the_historical_per_backbone_behaviour():
    # The default must not silently change what the completed ASR runs did:
    # the full-AR path took the reward channel only, bernoulli took both.
    assert _channel_brain("auto", full_ar=True)._rate_channels() == {"reward"}
    assert _channel_brain("auto", full_ar=False)._rate_channels() == {
        "reward",
        "aux",
    }


def test_explicit_channels_override_the_backbone_default():
    for full_ar in (True, False):
        assert _channel_brain("reward", full_ar)._rate_channels() == {"reward"}
        assert _channel_brain("aux", full_ar)._rate_channels() == {"aux"}
        assert _channel_brain("both", full_ar)._rate_channels() == {
            "reward",
            "aux",
        }


def test_a_missing_setting_is_auto_not_a_crash():
    from train_speechllm_with_segmenter import SegmenterASR

    brain = SegmenterASR.__new__(SegmenterASR)
    brain.hparams = types.SimpleNamespace()
    brain.modules = types.SimpleNamespace(
        segmenter=types.SimpleNamespace(is_full_ar=True)
    )
    assert brain._rate_channels() == {"reward"}


def test_an_unknown_channel_is_rejected_by_name():
    with pytest.raises(ValueError, match="auto/reward/aux/both"):
        _channel_brain("neither")._rate_channels()


def test_the_shipped_configs_default_to_auto():
    import re

    for name in (
        "speechllm_multitask.yaml",
        "speechllm_multitask_mustc.yaml",
        "speechllm_multitask_nonasr.yaml",
    ):
        text = open(
            os.path.join(RECIPE_DIR, "hparams", name), encoding="utf-8"
        ).read()
        assert re.search(r"^rate_channel: auto$", text, re.M), name


def _ar_brain(choice, rho_lo=0.15, rho_hi=0.25):
    """A SegmenterASR carrying only what the AR aux rate term touches."""
    from train_speechllm_with_segmenter import SegmenterASR

    brain = SegmenterASR.__new__(SegmenterASR)
    brain.hparams = types.SimpleNamespace(
        rate_channel=choice,
        rate_mode="band",
        rho_star=0.2,
        lambda_cap=1.0,
        lambda_press=0.02,
        rho_floor=0.08,
        lambda_floor=0.0,
        rho_lo=rho_lo,
        rho_hi=rho_hi,
        rate_two_sided=False,
    )
    brain.modules = types.SimpleNamespace(
        segmenter=types.SimpleNamespace(is_full_ar=True)
    )
    return brain


def _ar_score(seed=0):
    """Real Transformer-AR conditional logits for a sampled history."""
    import sys as _sys

    if RECIPE_DIR not in _sys.path:
        _sys.path.insert(0, RECIPE_DIR)
    from segmenter import Segmenter

    torch.manual_seed(seed)
    seg = Segmenter(
        input_dim=24,
        backbone="transformer_ar",
        hidden_dim=16,
        ar_hidden_dim=16,
        ar_num_layers=2,
        ar_nhead=2,
        ar_ffn_dim=32,
        ar_history_window=8,
        ar_cache_mode="preallocated",
        num_tasks=1,
        pooling="bigru_residual",
        pooling_input_dim=24,
        pooling_hidden_dim=8,
    )
    seg.eval()
    feats = torch.randn(2, 24, 24)
    pad = torch.zeros(2, 24, dtype=torch.bool)
    pad[1, 20:] = True
    with torch.no_grad():
        sampled = seg.sample_boundaries(feats, pad, deterministic=True)
    return seg.score_boundaries(feats, pad, sampled.boundaries)


def test_ar_aux_rate_loss_is_exactly_zero_when_the_channel_is_off():
    # The AR default routes rate through the reward, so this term must not
    # perturb the loss -- and must stay differentiable-shaped, not a bare float.
    score = _ar_score()
    loss = _ar_brain("reward")._full_ar_rate_loss(
        score.logits, score.valid_mask
    )
    assert float(loss) == 0.0
    assert loss.requires_grad


def test_ar_aux_rate_loss_runs_and_penalizes_only_outside_the_band():
    # This is the path a rate_channel=aux run actually executes; the dispatch
    # test alone would not have caught a shape or API error here.
    score = _ar_score()
    wide = _ar_brain("aux", rho_lo=0.0, rho_hi=1.0)._full_ar_rate_loss(
        score.logits, score.valid_mask
    )
    assert float(wide) == pytest.approx(0.0)

    narrow = _ar_brain("aux", rho_lo=0.9, rho_hi=0.99)._full_ar_rate_loss(
        score.logits, score.valid_mask
    )
    assert float(narrow) > 0.0
    assert torch.isfinite(narrow)


def test_ar_aux_rate_loss_carries_gradient_to_the_policy():
    # The whole point of the aux channel: unlike the reward channel it is
    # differentiable, so the rate weight actually scales a gradient.
    score = _ar_score()
    logits = score.logits.detach().clone().requires_grad_(True)
    loss = _ar_brain("aux", rho_lo=0.9, rho_hi=0.99)._full_ar_rate_loss(
        logits, score.valid_mask
    )
    loss.backward()
    assert logits.grad is not None
    assert float(logits.grad.abs().sum()) > 0.0


def test_ar_aux_rate_loss_scales_with_lambda():
    # Contrast with the reward channel, where lambda cancels under
    # std-normalized GRPO whenever quality is flat across the group.
    score = _ar_score()
    small = _ar_brain("aux", rho_lo=0.9, rho_hi=0.99)
    big = _ar_brain("aux", rho_lo=0.9, rho_hi=0.99)
    big.hparams.lambda_cap = 10.0
    a = float(small._full_ar_rate_loss(score.logits, score.valid_mask))
    b = float(big._full_ar_rate_loss(score.logits, score.valid_mask))
    assert b == pytest.approx(10.0 * a, rel=1e-5)
