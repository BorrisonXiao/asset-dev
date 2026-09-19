"""Unit tests for the cross-lingual step24k data/alignment tooling."""

import os
import sys

import pytest
import torch

RECIPE = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "recipes",
    "LibriSpeech",
    "ASR",
    "transformer",
)
sys.path.insert(0, os.path.abspath(RECIPE))

from reazonspeech_prepare import (  # noqa: E402
    kana_to_moras,
    normalize_text,
    split_of,
)
from train_speechllm_with_segmenter import (  # noqa: E402
    resolve_selection_metric,
)
from xling_alignment_gate import boundary_agreement, coverage  # noqa: E402
from xling_ctc_boundary_align import (  # noqa: E402
    build_vocab,
    duration_batches,
    num_encoder_frames,
    _edit_distance,
)
from xling_textgrid_boundaries import phones_tier_intervals  # noqa: E402


def test_selection_metric_normalizes_and_rejects_unknown_values():
    assert resolve_selection_metric("cer") == "CER"
    assert resolve_selection_metric("WER") == "WER"
    assert resolve_selection_metric(None) == "WER"
    with pytest.raises(ValueError):
        resolve_selection_metric("bleu")


def test_normalize_text_strips_punctuation_and_width_variants():
    assert normalize_text("今日は、いい天気です。") == "今日はいい天気です"
    # NFKC folds full-width ASCII; punctuation and spaces are removed.
    assert normalize_text("ＡＢＣ！ 「テスト」") == "ABCテスト"


def test_kana_to_moras_attaches_small_kana_and_keeps_special_moras():
    # キョ is one mora; ッ, ー and ン are their own moras.
    assert kana_to_moras("キョウ") == ["キョ", "ウ"]
    assert kana_to_moras("ガッコー") == ["ガ", "ッ", "コ", "ー"]
    assert kana_to_moras("ニッポン") == ["ニ", "ッ", "ポ", "ン"]


def test_split_of_is_deterministic_and_mostly_train():
    names = [f"000/{i:011x}.flac" for i in range(20000)]
    splits = [split_of(n) for n in names]
    assert splits == [split_of(n) for n in names]
    frac_train = splits.count("train") / len(splits)
    assert 0.93 < frac_train < 0.97
    assert splits.count("dev") > 0 and splits.count("test") > 0


def test_num_encoder_frames_matches_conv_stack():
    # 1 second of 16 kHz audio -> 49 frames for the wav2vec2/HuBERT stack.
    assert num_encoder_frames(16000) == 49
    assert num_encoder_frames(16400) == 51
    # Monotone, ~50 Hz: 10 s of audio adds ~450 frames over 1 s.
    assert num_encoder_frames(160000) == 499


def test_build_vocab_reserves_blank_zero():
    vocab = build_vocab({"u1": ["b", "a"], "u2": ["c", "a"]})
    assert 0 not in vocab.values()
    assert sorted(vocab) == ["a", "b", "c"]
    assert sorted(vocab.values()) == [1, 2, 3]


def test_duration_batches_respects_budget_and_count():
    rows = [(f"u{i}", 10.0, "w") for i in range(10)]
    batches = list(duration_batches(rows, batch_seconds=25.0, max_utts=8))
    assert all(sum(r[1] for r in b) <= 25.0 for b in batches)
    assert sum(len(b) for b in batches) == 10


def test_edit_distance():
    assert _edit_distance(list("abc"), list("abc")) == 0
    assert _edit_distance(list("abc"), list("axc")) == 1
    assert _edit_distance([], list("ab")) == 2


TEXTGRID = """File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 1.0
tiers? <exists>
size = 2
item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        xmin = 0
        xmax = 1.0
        intervals: size = 1
        intervals [1]:
            xmin = 0
            xmax = 1.0
            text = "word"
    item [2]:
        class = "IntervalTier"
        name = "phones"
        xmin = 0
        xmax = 1.0
        intervals: size = 3
        intervals [1]:
            xmin = 0
            xmax = 0.30
            text = ""
        intervals [2]:
            xmin = 0.30
            xmax = 0.58
            text = "a"
        intervals [3]:
            xmin = 0.58
            xmax = 1.0
            text = "b"
"""


def test_phones_tier_parsing_reads_only_the_phones_tier(tmp_path):
    path = tmp_path / "u.TextGrid"
    path.write_text(TEXTGRID)
    intervals = phones_tier_intervals(str(path))
    assert [t for _, t in intervals] == ["", "a", "b"]
    # 0.58 * 50 floats to 28.999...; the epsilon must land it on frame 29.
    assert int(intervals[2][0] * 50 + 1e-6) == 29


def test_gate_coverage_and_agreement(tmp_path):
    char_dir = tmp_path / "char"
    phone_dir = tmp_path / "phone"
    char_dir.mkdir()
    phone_dir.mkdir()
    char = torch.zeros(50, dtype=torch.uint8)
    char[[0, 10, 20, 40]] = 1
    phone = torch.zeros(50, dtype=torch.uint8)
    phone[[0, 9, 21, 30]] = 1
    torch.save(char, char_dir / "u1.pt")
    torch.save(phone, phone_dir / "u1.pt")
    assert coverage(["u1"], str(char_dir)) == 1.0
    assert coverage(["u1", "u2"], str(char_dir)) == 0.5
    # char boundaries 0,10,20 are within 2 frames of phone ones; 40 is not.
    agree, used = boundary_agreement(
        ["u1"], str(char_dir), str(phone_dir), tol_frames=2, sample=10
    )
    assert used == 1
    assert agree == pytest.approx(3 / 4)


def test_forced_alignment_boundaries_on_synthetic_emissions():
    """End-to-end shape check of the span->boundary convention."""
    from torchaudio.functional import forced_align, merge_tokens

    T, V = 20, 4
    emission = torch.full((1, T, V), -10.0)
    # blank everywhere except two unit spans: unit 1 at frames 4-8,
    # unit 2 at frames 12-15.
    emission[:, :, 0] = -0.5
    emission[:, 4:9, 1] = 0.0
    emission[:, 12:16, 2] = 0.0
    log_probs = torch.log_softmax(emission, dim=-1)
    targets = torch.tensor([[1, 2]], dtype=torch.int32)
    frames, scores = forced_align(log_probs, targets, blank=0)
    spans = merge_tokens(frames[0], scores[0])
    boundary = torch.zeros(T, dtype=torch.uint8)
    boundary[0] = 1
    for span in spans:
        boundary[span.start] = 1
    starts = boundary.nonzero().flatten().tolist()
    assert starts[0] == 0
    assert len(starts) == 3  # frame 0 + two unit onsets
    assert 4 in starts and 12 in starts
