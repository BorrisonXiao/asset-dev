"""Tests for transcript-free BPE-CTC state-run boundaries."""

import sys
from pathlib import Path

import torch


RECIPE_DIR = (
    Path(__file__).resolve().parents[2]
    / "recipes"
    / "LibriSpeech"
    / "ASR"
    / "transformer"
)
sys.path.insert(0, str(RECIPE_DIR))

from bpe_ctc_posterior_collapse import (  # noqa: E402
    boundary_from_argmax,
    map_starts_to_target,
    nonblank_run_starts,
    state_run_starts,
)


def test_state_runs_keep_blank_intervals():
    labels = torch.tensor([0, 0, 4, 4, 0, 9, 9, 0])
    assert state_run_starts(labels).tolist() == [0, 2, 4, 5, 7]


def test_nonblank_runs_merge_blanks_but_keep_ctc_repetition():
    labels = torch.tensor([0, 0, 4, 4, 0, 4, 4, 9, 9, 0])
    assert nonblank_run_starts(labels).tolist() == [0, 5, 7]


def test_all_blank_path_has_one_segment_for_both_rules():
    labels = torch.zeros(4, dtype=torch.long)
    assert state_run_starts(labels).tolist() == [0]
    assert nonblank_run_starts(labels).tolist() == [0]


def test_map_starts_doubles_grid_without_collisions():
    starts = torch.tensor([0, 2, 4, 5, 7])
    assert map_starts_to_target(starts, 8, 16).tolist() == [0, 4, 8, 10, 14]


def test_state_run_boundary_uses_exact_target_length():
    labels = torch.tensor([0, 0, 4, 4, 0, 9, 9, 0])
    boundary = boundary_from_argmax(labels, target_frames=16, rule="state_runs")
    assert boundary.dtype == torch.uint8
    assert boundary.tolist() == [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0]


def test_empty_ctc_path_is_rejected():
    for function in (state_run_starts, nonblank_run_starts):
        try:
            function(torch.tensor([], dtype=torch.long))
        except ValueError as error:
            assert "at least one frame" in str(error)
        else:
            raise AssertionError("empty CTC path should fail")
