"""Tests for the same-audio ASR/ST boundary comparison math.

These are the statistics the scientific gate rests on -- if Jaccard or
tolerance-F1 is wrong, a null result and a real task-conditioning effect become
indistinguishable. They are tested here without models so the arithmetic is
pinned independently of any checkpoint.
"""

import os
import sys

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

from analyze_multitask_boundaries import (  # noqa: E402
    FRAME_SECONDS,
    _jaccard,
    _segment_durations,
    _summary,
    _tolerance_f1,
)


def _mask(bits):
    return torch.tensor(bits, dtype=torch.long)


def _valid(n, used=None):
    valid = torch.zeros(n, dtype=torch.bool)
    valid[: (used if used is not None else n)] = True
    return valid


def test_identical_masks_score_one():
    mask = _mask([1, 0, 0, 1, 0, 1, 0, 0])
    valid = _valid(8)
    assert _jaccard(mask, mask, valid) == 1.0
    assert _tolerance_f1(mask, mask, valid, 0) == 1.0


def test_disjoint_masks_score_zero():
    left = _mask([1, 0, 1, 0, 0, 0])
    right = _mask([0, 1, 0, 0, 0, 0])
    valid = _valid(6)
    assert _jaccard(left, right, valid) == 0.0
    assert _tolerance_f1(left, right, valid, 0) == 0.0


def test_jaccard_is_intersection_over_union():
    # left {0,2,4}, right {0,2,5} -> intersection 2, union 4
    left = _mask([1, 0, 1, 0, 1, 0])
    right = _mask([1, 0, 1, 0, 0, 1])
    assert _jaccard(left, right, _valid(6)) == pytest.approx(2 / 4)


def test_tolerance_window_forgives_a_one_frame_shift():
    left = _mask([1, 0, 0, 1, 0, 0])
    right = _mask([1, 0, 0, 0, 1, 0])  # boundary moved one frame later
    valid = _valid(6)
    assert _tolerance_f1(left, right, valid, 0) < 1.0
    # A 20 ms shift is not a meaningful placement difference.
    assert _tolerance_f1(left, right, valid, 1) == 1.0


def test_padding_is_excluded_from_every_statistic():
    # The two masks differ only inside the padded region.
    left = _mask([1, 0, 1, 0, 1, 1])
    right = _mask([1, 0, 1, 0, 0, 0])
    valid = _valid(6, used=4)
    assert _jaccard(left, right, valid) == 1.0
    assert _tolerance_f1(left, right, valid, 0) == 1.0


def test_empty_boundary_sets_do_not_crash():
    empty = _mask([0, 0, 0, 0])
    valid = _valid(4)
    assert _jaccard(empty, empty, valid) == 1.0
    assert _tolerance_f1(empty, empty, valid, 0) == 1.0
    assert _tolerance_f1(empty, _mask([1, 0, 0, 0]), valid, 0) == 0.0


def test_segment_durations_partition_the_valid_region():
    mask = _mask([1, 0, 0, 1, 0, 1, 0, 0])
    valid = _valid(8)
    durations = _segment_durations(mask, valid)
    # Frame 0 always opens a segment, so boundaries at 3 and 5 give 3 segments.
    assert len(durations) == 3
    assert sum(durations) == pytest.approx(8 * FRAME_SECONDS)
    assert durations == pytest.approx([3 * 0.02, 2 * 0.02, 3 * 0.02])


def test_segment_durations_respect_padding():
    mask = _mask([1, 0, 1, 0, 1, 1])
    valid = _valid(6, used=4)
    durations = _segment_durations(mask, valid)
    assert sum(durations) == pytest.approx(4 * FRAME_SECONDS)


def test_summary_reports_spread_not_just_a_mean():
    stats = _summary([1.0, 2.0, 3.0, 4.0])
    assert stats["n"] == 4
    assert stats["mean"] == pytest.approx(2.5)
    assert stats["median"] == pytest.approx(2.5)
    assert stats["p10"] < stats["median"] < stats["p90"]
    assert _summary([]) == {"n": 0}
    assert _summary([5.0])["std"] == 0.0


def test_position_cap_matches_the_policy_limit():
    """The data filter must be derived from the model's own position limit."""
    from train_speechllm_multitask import max_utterance_seconds

    # 4096 positions on a 50 Hz grid, minus one frame of slack.
    assert max_utterance_seconds(
        {"segmenter_backbone": "transformer_ar", "segmenter_ar_max_positions": 4096}
    ) == pytest.approx(81.9)
    # Non-autoregressive backbones have no positional limit to respect.
    assert (
        max_utterance_seconds(
            {"segmenter_backbone": "cnn", "segmenter_ar_max_positions": 4096}
        )
        is None
    )
    assert (
        max_utterance_seconds(
            {"segmenter_backbone": "transformer_ar", "segmenter_ar_max_positions": None}
        )
        is None
    )
