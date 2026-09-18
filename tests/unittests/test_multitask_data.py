"""Tests for the multi-task manifest schema and task-homogeneous batching."""

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

from multitask_data import (  # noqa: E402
    MULTITASK_FIELDS,
    TASK_SPECS,
    MultiSourceDynamicBatchSampler,
    build_multitask_manifest,
    candidates_of,
    is_classification,
    prompt_of,
    resolve_candidates,
    task_id_of,
)


def _row(index, source_key, task_name, duration):
    task_id = task_id_of(task_name)
    return {
        "ID": f"{task_name}-utt{index}",
        "duration": f"{duration}",
        "wav": f"/tmp/audio/utt{index}.flac",
        "spk_id": f"spk{index % 7}",
        "wrd": f"text {index}",
        "task_name": task_name,
        "task_id": task_id,
        "source_key": source_key,
        "source_dataset": "synthetic",
        "source_lang": "en",
        "target_lang": "de" if task_name.startswith("st_") else "en",
        "utt_key": f"utt{index}",
    }


def _write(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MULTITASK_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


class _FakeDataset:
    """Minimal stand-in exposing the two attributes the sampler reads."""

    def __init__(self, rows):
        self.data_ids = [row["ID"] for row in rows]
        self.data = {row["ID"]: row for row in rows}

    def __len__(self):
        return len(self.data_ids)


def _dataset(counts, durations=None):
    rows = []
    index = 0
    for source_key, (task_name, count) in counts.items():
        for _ in range(count):
            duration = durations[source_key] if durations else 4.0
            rows.append(_row(index, source_key, task_name, duration))
            index += 1
    return _FakeDataset(rows), rows


def test_task_schema_is_stable():
    # task_id 0 must be ASR: an ASR warm start is only the identity at step
    # zero if the ASR slot is the one the checkpoint is copied into. The
    # non-ASR tasks were appended after ids 0/1 already existed in
    # checkpoints, so those two must keep their ids forever.
    assert task_id_of("asr") == 0
    assert task_id_of("st_en_de") == 1
    assert task_id_of("emotion") == 2
    assert task_id_of("speaker_count") == 3
    assert task_id_of("intent") == 4
    ids = sorted(spec[0] for spec in TASK_SPECS.values())
    assert ids == list(range(len(TASK_SPECS)))
    assert "German" in prompt_of("st_en_de")
    assert prompt_of("asr") != prompt_of("st_en_de")
    # Every task's prompt is distinct: the prompt is what tells the decoder
    # which question it is answering.
    prompts = [prompt_of(name) for name in TASK_SPECS]
    assert len(set(prompts)) == len(prompts)


def test_generative_and_classification_tasks_are_separated():
    assert not is_classification("asr")
    assert not is_classification("st_en_de")
    assert candidates_of("asr") == ()
    for name in ("emotion", "speaker_count", "intent"):
        labels = candidates_of(name)
        assert is_classification(name)
        assert len(set(labels)) == len(labels)
    assert len(candidates_of("emotion")) == 6
    # 31 Fluent Speech Commands intents, one per (action, object, location).
    assert len(candidates_of("intent")) == 31
    # The full speaker-count vocabulary is 0-10 even while the pilot ranks 0-4.
    assert len(candidates_of("speaker_count")) == 11


def test_candidate_subset_restricts_without_changing_the_table():
    pilot = resolve_candidates(
        "speaker_count", {"speaker_count": ["zero", "one", "two"]}
    )
    assert pilot == ("zero", "one", "two")
    # Falling through leaves the canonical set untouched.
    assert resolve_candidates("speaker_count") == candidates_of("speaker_count")
    assert resolve_candidates("emotion", {"speaker_count": ["zero"]}) == (
        candidates_of("emotion")
    )


def test_candidate_subset_rejects_labels_outside_the_task():
    # A typo here would otherwise become a label the model can never be right
    # about, showing up only as an unexplained accuracy ceiling.
    with pytest.raises(ValueError, match="outside the task table"):
        resolve_candidates("emotion", {"emotion": ["angry", "furious"]})
    with pytest.raises(ValueError, match="Empty candidate subset"):
        resolve_candidates("emotion", {"emotion": []})


def test_build_manifest_merges_and_stamps_sources(tmp_path):
    st = tmp_path / "st.csv"
    asr = tmp_path / "asr.csv"
    _write(st, [_row(i, "covost_st", "st_en_de", 3.0) for i in range(5)])
    _write(asr, [_row(i, "covost_asr", "asr", 3.0) for i in range(5)])

    out = tmp_path / "merged.csv"
    counts = build_multitask_manifest(
        [
            {"csv": str(st), "source_key": "covost_st"},
            {"csv": str(asr), "source_key": "covost_asr"},
        ],
        str(out),
    )
    assert counts == {"covost_st": 5, "covost_asr": 5}

    with open(out, encoding="utf-8") as stream:
        merged = list(csv.DictReader(stream))
    assert len(merged) == 10
    assert {row["source_key"] for row in merged} == {"covost_st", "covost_asr"}
    # The two views of one utterance must remain linkable by utt_key.
    st_keys = {r["utt_key"] for r in merged if r["task_name"] == "st_en_de"}
    asr_keys = {r["utt_key"] for r in merged if r["task_name"] == "asr"}
    assert st_keys == asr_keys


def test_build_manifest_rejects_duplicate_ids(tmp_path):
    same = tmp_path / "same.csv"
    _write(same, [_row(i, "covost_st", "st_en_de", 3.0) for i in range(4)])
    out = tmp_path / "merged.csv"
    with pytest.raises(ValueError, match="Duplicate utterance ID"):
        build_multitask_manifest(
            [
                {"csv": str(same), "source_key": "covost_st"},
                {"csv": str(same), "source_key": "covost_asr"},
            ],
            str(out),
        )


def test_limit_selects_the_same_utterances_across_views(tmp_path):
    st = tmp_path / "st.csv"
    asr = tmp_path / "asr.csv"
    _write(st, [_row(i, "covost_st", "st_en_de", 3.0) for i in range(50)])
    _write(asr, [_row(i, "covost_asr", "asr", 3.0) for i in range(50)])

    out = tmp_path / "merged.csv"
    build_multitask_manifest(
        [
            {"csv": str(st), "source_key": "covost_st", "limit": 10},
            {"csv": str(asr), "source_key": "covost_asr", "limit": 10},
        ],
        str(out),
    )
    with open(out, encoding="utf-8") as stream:
        merged = list(csv.DictReader(stream))
    st_keys = [r["utt_key"] for r in merged if r["task_name"] == "st_en_de"]
    asr_keys = [r["utt_key"] for r in merged if r["task_name"] == "asr"]
    assert len(st_keys) == len(asr_keys) == 10
    # Paired analysis requires the dev subsets to cover the same audio.
    assert st_keys == asr_keys


def test_librispeech_replay_rows_absolutize_paths(tmp_path):
    ls = tmp_path / "ls.csv"
    with open(ls, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["ID", "duration", "wav", "spk_id", "wrd"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "ID": "1272-128104-0000",
                "duration": "5.855",
                "wav": "$data_root//dev-clean/1272/128104/1272-128104-0000.flac",
                "spk_id": "1272-128104",
                "wrd": "MISTER QUILTER IS THE APOSTLE",
            }
        )
    out = tmp_path / "merged.csv"
    build_multitask_manifest(
        [
            {
                "csv": str(ls),
                "source_key": "librispeech_asr",
                "kind": "librispeech",
            }
        ],
        str(out),
        data_folder="/corpora/LibriSpeech",
    )
    with open(out, encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert row["wav"] == (
        "/corpora/LibriSpeech/dev-clean/1272/128104/1272-128104-0000.flac"
    )
    assert row["task_name"] == "asr"
    assert int(row["task_id"]) == 0
    assert row["utt_key"] == "1272-128104-0000"
    assert "$data_root" not in row["wav"]


def test_batches_are_task_homogeneous():
    dataset, rows = _dataset(
        {
            "covost_st": ("st_en_de", 200),
            "covost_asr": ("asr", 200),
            "librispeech_asr": ("asr", 400),
        }
    )
    sampler = MultiSourceDynamicBatchSampler(
        dataset,
        source_keys=["covost_st", "covost_asr", "librispeech_asr"],
        probabilities=[0.5, 0.25, 0.25],
        max_batch_length=32,
        num_buckets=5,
        max_batch_ex=8,
        seed=1234,
    )
    seen_sources = set()
    for batch in sampler:
        sources = {rows[i]["source_key"] for i in batch}
        tasks = {rows[i]["task_name"] for i in batch}
        # This is the invariant the whole design rests on: adapter routing and
        # GRPO reward normalization are only well defined per task.
        assert len(sources) == 1, sources
        assert len(tasks) == 1, tasks
        seen_sources |= sources
    assert seen_sources == {"covost_st", "covost_asr", "librispeech_asr"}


def test_sampling_probabilities_are_respected():
    dataset, rows = _dataset(
        {
            "covost_st": ("st_en_de", 400),
            "covost_asr": ("asr", 400),
            "librispeech_asr": ("asr", 400),
        }
    )
    sampler = MultiSourceDynamicBatchSampler(
        dataset,
        source_keys=["covost_st", "covost_asr", "librispeech_asr"],
        probabilities=[0.5, 0.25, 0.25],
        max_batch_length=16,
        num_buckets=4,
        max_batch_ex=4,
        batches_per_epoch=4000,
        seed=7,
    )
    counts = dict.fromkeys(sampler.source_keys, 0)
    for batch in sampler:
        counts[sampler.source_of_batch(batch)] += 1
    total = sum(counts.values())
    assert total == 4000
    assert counts["covost_st"] / total == pytest.approx(0.5, abs=0.03)
    assert counts["covost_asr"] / total == pytest.approx(0.25, abs=0.03)
    assert counts["librispeech_asr"] / total == pytest.approx(0.25, abs=0.03)


def test_small_source_repeats_instead_of_starving():
    # ST is 20x smaller than the replay corpus but must still supply half the
    # batches; the sampler regenerates an exhausted source rather than stopping.
    dataset, rows = _dataset(
        {
            "covost_st": ("st_en_de", 20),
            "librispeech_asr": ("asr", 400),
        }
    )
    sampler = MultiSourceDynamicBatchSampler(
        dataset,
        source_keys=["covost_st", "librispeech_asr"],
        probabilities=[0.5, 0.5],
        max_batch_length=8,
        num_buckets=2,
        max_batch_ex=2,
        batches_per_epoch=600,
        seed=11,
    )
    counts = dict.fromkeys(sampler.source_keys, 0)
    for batch in sampler:
        counts[sampler.source_of_batch(batch)] += 1
    assert sum(counts.values()) == 600
    assert counts["covost_st"] / 600 == pytest.approx(0.5, abs=0.05)


def test_exhaustive_mode_is_deterministic_and_complete():
    dataset, rows = _dataset(
        {"covost_st": ("st_en_de", 60), "librispeech_asr": ("asr", 40)}
    )
    sampler = MultiSourceDynamicBatchSampler(
        dataset,
        source_keys=["covost_st", "librispeech_asr"],
        probabilities=[1.0, 1.0],
        max_batch_length=16,
        num_buckets=3,
        max_batch_ex=4,
        seed=3,
        exhaustive=True,
    )
    first = [list(batch) for batch in sampler]
    second = [list(batch) for batch in sampler]
    # A validation set must produce identical batches at every validation point,
    # or per-task metrics are not comparable across optimizer steps.
    assert first == second
    covered = sorted(i for batch in first for i in batch)
    assert covered == list(range(len(dataset)))
    assert len(first) == len(sampler)
    sampler.set_epoch(5)
    assert [list(batch) for batch in sampler] == first


def test_rejects_mismatched_probability_length():
    dataset, _ = _dataset({"covost_st": ("st_en_de", 10)})
    with pytest.raises(ValueError, match="same length"):
        MultiSourceDynamicBatchSampler(
            dataset,
            source_keys=["covost_st"],
            probabilities=[0.5, 0.5],
            max_batch_length=8,
            num_buckets=2,
        )


def test_rejects_unknown_source():
    dataset, _ = _dataset({"covost_st": ("st_en_de", 10)})
    with pytest.raises(ValueError, match="no rows"):
        MultiSourceDynamicBatchSampler(
            dataset,
            source_keys=["covost_st", "missing_source"],
            probabilities=[0.5, 0.5],
            max_batch_length=8,
            num_buckets=2,
        )


def test_max_duration_filters_utterances_beyond_the_policy_limit(tmp_path):
    """Over-long utterances must be removed when the manifest is built.

    The autoregressive boundary policy raises once a rollout passes its
    positional limit. If such an utterance survives into a test-only split, the
    failure appears hours into a run, at final evaluation, after all training is
    done -- so the filter belongs at manifest-build time.
    """
    src = tmp_path / "src.csv"
    rows = [_row(i, "covost_st", "st_en_de", 5.0) for i in range(8)]
    rows[3]["duration"] = "142.5"  # a mis-segmented Common Voice clip
    rows[6]["duration"] = "95.0"
    _write(src, rows)

    out = tmp_path / "merged.csv"
    counts = build_multitask_manifest(
        [{"csv": str(src), "source_key": "covost_st"}],
        str(out),
        max_duration=81.9,
    )
    assert counts == {"covost_st": 6}
    with open(out, encoding="utf-8") as stream:
        kept = list(csv.DictReader(stream))
    assert all(float(r["duration"]) <= 81.9 for r in kept)
    assert {r["utt_key"] for r in kept} == {
        f"utt{i}" for i in range(8) if i not in (3, 6)
    }


def test_max_duration_none_keeps_everything(tmp_path):
    src = tmp_path / "src.csv"
    rows = [_row(i, "covost_st", "st_en_de", 5.0) for i in range(4)]
    rows[1]["duration"] = "300.0"
    _write(src, rows)
    out = tmp_path / "merged.csv"
    counts = build_multitask_manifest(
        [{"csv": str(src), "source_key": "covost_st"}], str(out)
    )
    assert counts == {"covost_st": 4}


def test_manifest_loader_tolerates_dollar_amounts_in_text(tmp_path):
    """Text containing "$9" must not be read as a variable reference.

    SpeechBrain's from_csv expands ``$name`` in every field for the
    ``$data_root`` mechanism. Merged manifests carry absolute paths and need no
    expansion, but real translations contain currency amounts -- MuST-C German
    has "$9 pro Pfund" -- which that loader raises KeyError on.
    """
    from multitask_data import _load_manifest

    path = tmp_path / "m.csv"
    rows = [_row(0, "mustc_st", "st_en_de", 4.0), _row(1, "mustc_st", "st_en_de", 7.5)]
    rows[0]["wrd"] = "Dieser Ersatzstoff kostet $9 pro Pfund, echtes $900."
    rows[1]["wrd"] = "Kein Preis hier."
    _write(path, rows)

    data = _load_manifest(str(path))
    assert len(data) == 2
    assert data["st_en_de-utt0"]["wrd"] == (
        "Dieser Ersatzstoff kostet $9 pro Pfund, echtes $900."
    )
    # Durations must be numeric, or "10.5" would sort before "9.2".
    assert isinstance(data["st_en_de-utt1"]["duration"], float)
    assert data["st_en_de-utt1"]["duration"] == 7.5
    assert "ID" not in data["st_en_de-utt0"]


def test_manifest_loader_rejects_duplicate_ids(tmp_path):
    from multitask_data import _load_manifest

    path = tmp_path / "dup.csv"
    rows = [_row(0, "mustc_st", "st_en_de", 4.0), _row(0, "mustc_st", "st_en_de", 4.0)]
    _write(path, rows)
    with pytest.raises(ValueError, match="Duplicate id"):
        _load_manifest(str(path))


def test_subset_does_not_alias_against_a_periodic_manifest():
    """A held-out subset must keep every class of a class-interleaved manifest.

    Regression for a real corruption: the subset used to take evenly spaced
    indices, and the speaker-count manifests are interleaved by label with
    period 5. Striding 600 out of 2000 steps by 3.33 and lands only on residues
    {0, 3, 1}, so validation ran on THREE of five classes -- it never saw a
    ``two`` or a ``four``. Validation then measured an easier 3-way problem
    (0.90) than the 5-way test set (0.69), and checkpoint selection was driven
    by that. A 1000-of-2000 subset strides by exactly 2 and covers all five,
    which is why it stayed hidden until the subset size changed.
    """
    from multitask_data import _stride_subset

    labels = ["zero", "one", "two", "three", "four"]
    rows = [{"ID": f"utt{i}", "wrd": labels[i % 5]} for i in range(2000)]

    for limit in (600, 999, 1000, 1001, 1234):
        subset = _stride_subset(rows, limit)
        assert len(subset) == limit
        present = {r["wrd"] for r in subset}
        assert present == set(labels), (
            f"limit={limit} dropped {set(labels) - present} entirely"
        )
        # Near-balanced, not merely present: a class reduced to a handful of
        # examples would make its per-class metric noise.
        counts = [sum(1 for r in subset if r["wrd"] == c) for c in labels]
        assert min(counts) > 0.6 * (limit / len(labels)), counts

    # Deterministic across calls, and a superset request returns everything.
    assert [r["ID"] for r in _stride_subset(rows, 600)] == [
        r["ID"] for r in _stride_subset(rows, 600)
    ]
    assert len(_stride_subset(rows, 5000)) == len(rows)

    # Manifest order is preserved, which downstream sorting relies on.
    picked = [int(r["ID"][3:]) for r in _stride_subset(rows, 600)]
    assert picked == sorted(picked)
