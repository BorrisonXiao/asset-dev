"""Tests for transcript-free CTC posterior-collapse boundaries."""

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

from ctc_posterior_collapse import nonblank_run_boundaries  # noqa: E402


def test_ctc_collapse_merges_blanks_and_repetitions():
    labels = torch.tensor([0, 0, 3, 3, 0, 4, 4, 0, 0, 5])
    boundary = nonblank_run_boundaries(labels)
    assert boundary.tolist() == [1, 0, 0, 0, 0, 1, 0, 0, 0, 1]
    assert int(boundary.sum()) == 3


def test_ctc_collapse_keeps_repeated_symbol_separated_by_blank():
    labels = torch.tensor([7, 7, 0, 7, 7])
    boundary = nonblank_run_boundaries(labels)
    assert boundary.tolist() == [1, 0, 0, 1, 0]


def test_ctc_collapse_handles_all_blank_audio():
    boundary = nonblank_run_boundaries(torch.zeros(4, dtype=torch.long))
    assert boundary.dtype == torch.uint8
    assert boundary.tolist() == [1, 0, 0, 0]


def test_ctc_collapse_rejects_empty_path():
    try:
        nonblank_run_boundaries(torch.tensor([], dtype=torch.long))
    except ValueError as error:
        assert "at least one frame" in str(error)
    else:
        raise AssertionError("empty CTC path should fail")
