"""Tests for the CREMA-D and Fluent Speech Commands manifest preparation.

These two scripts decide what the emotion and intent tasks are actually trained
on, and their mistakes are the quiet kind: a label that never matches a
candidate, a speaker appearing in both train and test, a tie silently resolved
to one side. Each of those would show up much later as an unexplained score
rather than as an error, so they are pinned here.
"""

import csv
import os
import sys

import pytest

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

import cremad_prepare as cremad  # noqa: E402
import fsc_prepare as fsc  # noqa: E402
from multitask_data import (  # noqa: E402
    EMOTION_CANDIDATES,
    INTENT_CANDIDATES,
    MULTITASK_FIELDS,
)

# ------------------------------------------------------------------- CREMA-D


def _record(utt_key, speaker, sentence, acted, voice, duration=2.5):
    return {
        "utt_key": utt_key,
        "speaker": speaker,
        "sentence": sentence,
        "acted_label": acted,
        "voice_label": voice,
        "voice_vote_raw": voice or "N:S",
        "wav": f"/corpus/{utt_key}.wav",
        "duration": duration,
    }


def test_every_sentence_and_emotion_code_maps_into_the_task_table():
    # The 12 documented sentences and the 6 candidate labels are what pair the
    # ASR view to the emotion view; a gap here breaks the same-audio analysis.
    assert len(cremad.SENTENCES) == 12
    assert set(cremad.VOICE_CODES.values()) == set(EMOTION_CANDIDATES)
    assert set(cremad.ACTED_CODES.values()) == set(EMOTION_CANDIDATES)


def test_voice_labelling_drops_ties_and_acted_labelling_keeps_them():
    records = [
        _record("1001_IEO_NEU_XX", "1001", "IEO", "neutral", "neutral"),
        # No unique audible majority: dropped from the voice view, kept by the
        # acted view, which is why the two views have different clip counts.
        _record("1001_IEO_SAD_XX", "1001", "IEO", "sad", None),
    ]
    voice = cremad.emotion_rows(records, "voice")
    acted = cremad.emotion_rows(records, "acted")
    assert [r["wrd"] for r in voice] == ["neutral"]
    assert [r["wrd"] for r in acted] == ["neutral", "sad"]
    assert all(r["task_name"] == "emotion" and r["task_id"] == 2 for r in acted)


def test_asr_view_covers_exactly_the_clips_the_emotion_view_keeps():
    records = [
        _record("1001_DFA_ANG_XX", "1001", "DFA", "angry", "angry"),
        _record("1001_IEO_SAD_XX", "1001", "IEO", "sad", None),
    ]
    emotion = cremad.emotion_rows(records, "voice")
    asr = cremad.asr_rows(records, "voice")
    # The paired counterfactual compares boundaries on the SAME recordings, so
    # the two views must not drift apart.
    assert [r["utt_key"] for r in emotion] == [r["utt_key"] for r in asr]
    assert asr[0]["wrd"] == "DON'T FORGET A JACKET"
    assert asr[0]["task_name"] == "asr" and asr[0]["task_id"] == 0


def test_emotion_rows_reject_a_label_outside_the_candidate_set():
    records = [_record("1001_IEO_NEU_XX", "1001", "IEO", "neutral", "furious")]
    with pytest.raises(ValueError, match="not in the task table"):
        cremad.emotion_rows(records, "voice")


def test_splits_are_speaker_disjoint_and_ordered():
    records = [
        _record(
            f"{1000 + s}_IEO_NEU_XX", str(1000 + s), "IEO", "neutral", "neutral"
        )
        for s in range(1, 7)
    ]
    buckets = cremad.assign_splits(records, (3, 1, 2))
    assert buckets == {
        "train": ["1001", "1002", "1003"],
        "valid": ["1004"],
        "test": ["1005", "1006"],
    }
    by_split = {}
    for record in records:
        by_split.setdefault(record["split"], set()).add(record["speaker"])
    # Emotion recognition on held-out clips from a *seen* speaker measures
    # something much weaker than generalization to a new voice.
    assert by_split["train"].isdisjoint(by_split["test"])
    assert by_split["train"].isdisjoint(by_split["valid"])


def test_split_sizes_must_account_for_every_speaker():
    records = [
        _record(
            f"{1000 + s}_IEO_NEU_XX", str(1000 + s), "IEO", "neutral", "neutral"
        )
        for s in range(1, 7)
    ]
    with pytest.raises(ValueError, match="sums to"):
        cremad.assign_splits(records, (3, 1, 1))


def test_the_distributed_filename_inconsistency_is_aliased_not_guessed():
    # The summary table says 1040_ITH_SAD_XX; the audio file is
    # 1040_ITH_SAD_X.wav. Aliased explicitly so any OTHER missing file stays a
    # hard error rather than being skipped.
    assert cremad.FILENAME_ALIASES == {"1040_ITH_SAD_XX": "1040_ITH_SAD_X"}
    path = cremad._audio_path("/corpus", "1040_ITH_SAD_XX")
    assert path.endswith("AudioWAV/1040_ITH_SAD_X.wav")
    assert cremad._audio_path("/corpus", "1001_IEO_NEU_XX").endswith(
        "AudioWAV/1001_IEO_NEU_XX.wav"
    )


def test_cremad_rows_carry_the_full_multitask_schema():
    rows = cremad.emotion_rows(
        [_record("1001_IEO_NEU_XX", "1001", "IEO", "neutral", "neutral")],
        "voice",
    )
    assert set(rows[0]) == set(MULTITASK_FIELDS)


# ------------------------------------------------------------------------ FSC


def test_canonical_intent_joins_only_the_filled_slots():
    assert fsc.canonical_intent("activate", "lights", "kitchen") == (
        "activate lights kitchen"
    )
    assert fsc.canonical_intent("change language", "none", "none") == (
        "change language"
    )
    assert fsc.canonical_intent("decrease", "volume", "none") == (
        "decrease volume"
    )


def test_every_canonical_intent_is_in_the_pinned_candidate_set():
    # The corpus's own triples are the source of truth for the tuple; if these
    # ever diverge, a label exists that the decoder never scores.
    assert len(INTENT_CANDIDATES) == 31
    assert fsc.canonical_intent("activate", "lamp", "none") in INTENT_CANDIDATES


def test_read_split_rejects_an_intent_outside_the_candidate_set(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    with open(data / "train_data.csv", "w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "",
                "path",
                "speakerId",
                "transcription",
                "action",
                "object",
                "location",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "": "0",
                "path": "wavs/speakers/spk1/utt1.wav",
                "speakerId": "spk1",
                "transcription": "Launch the rocket",
                "action": "launch",
                "object": "rocket",
                "location": "none",
            }
        )
    with pytest.raises(ValueError, match="not in the task table"):
        fsc.read_split(str(tmp_path), "train")


def test_speaker_disjointness_is_checked_not_assumed():
    shared = [{"speaker": "spk1"}, {"speaker": "spk2"}]
    ok = {
        "train": [{"speaker": "spk1"}],
        "valid": [{"speaker": "spk9"}],
        "test": [{"speaker": "spk8"}],
    }
    assert fsc.check_speaker_disjoint(ok) == {
        "train": 1,
        "valid": 1,
        "test": 1,
    }
    bad = {"train": shared, "valid": [{"speaker": "spk2"}], "test": []}
    with pytest.raises(ValueError, match="not speaker disjoint"):
        fsc.check_speaker_disjoint(bad)


def test_fsc_rows_carry_the_full_multitask_schema_for_both_views():
    record = {
        "utt_key": "uuid-1",
        "wav": "/corpus/wavs/speakers/spk1/uuid-1.wav",
        "speaker": "spk1",
        "intent": "decrease volume",
        "transcription": "Turn it down",
        "duration": 2.7,
    }
    intent = fsc._row(record, "intent", "fsc_intent", record["intent"])
    asr = fsc._row(record, "asr", "fsc_asr", "TURN IT DOWN")
    assert set(intent) == set(MULTITASK_FIELDS)
    assert intent["task_id"] == 4 and asr["task_id"] == 0
    # Same audio, same pairing key, distinct manifest IDs.
    assert intent["utt_key"] == asr["utt_key"] == "uuid-1"
    assert intent["ID"] != asr["ID"]
