import sys
from pathlib import Path

RECIPE = Path(__file__).resolve().parents[2] / "recipes/LibriSpeech/ASR/transformer"
sys.path.insert(0, str(RECIPE))

from audit_fixed_rate_phone_agreement import _fixed_positions  # noqa: E402


def test_fixed_positions_match_pooler_semantics():
    assert _fixed_positions(0, 5) == []
    assert _fixed_positions(1, 5) == [0]
    assert _fixed_positions(10, 5) == [0, 5]
    assert _fixed_positions(11, 5) == [0, 5, 10]


def test_fixed_positions_reject_invalid_k():
    try:
        _fixed_positions(10, 0)
    except ValueError as error:
        assert "at least one" in str(error)
    else:
        raise AssertionError("k=0 should fail")
