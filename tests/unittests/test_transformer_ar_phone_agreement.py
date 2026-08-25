import math
import sys
from pathlib import Path

import torch

RECIPE = Path(__file__).resolve().parents[2] / "recipes/LibriSpeech/ASR/transformer"
sys.path.insert(0, str(RECIPE))

from audit_transformer_ar_phone_agreement import (  # noqa: E402
    _empty_counts,
    _positions,
    _score,
    _update_counts,
)


def test_harsh_and_lenient_duplicate_matching():
    counts = {scheme: _empty_counts() for scheme in ("harsh", "lenient")}
    _update_counts(counts, pred=[9, 10], gold=[10], tolerance=1)

    harsh = _score(counts["harsh"])
    assert harsh["precision"] == 0.5
    assert harsh["recall"] == 1.0
    assert math.isclose(harsh["f1"], 2 / 3)

    lenient = _score(counts["lenient"])
    assert lenient["precision"] == 1.0
    assert lenient["recall"] == 1.0
    assert lenient["f1"] == 1.0
    assert math.isclose(lenient["r_value"], 1.0, abs_tol=1e-8)


def test_implicit_frame_zero_is_added_once():
    assert _positions(torch.tensor([0, 0, 1, 0]), True) == [0, 2]
    assert _positions(torch.tensor([1, 0, 1, 0]), True) == [0, 2]
    assert _positions(torch.tensor([1, 0, 1, 0]), False) == [0, 2]


def test_harsh_matching_is_ordered_not_nearest_first():
    counts = {scheme: _empty_counts() for scheme in ("harsh", "lenient")}
    _update_counts(counts, pred=[2, 3], gold=[0, 3], tolerance=2)
    harsh = _score(counts["harsh"])
    assert harsh["precision"] == 1.0
    assert harsh["recall"] == 1.0
