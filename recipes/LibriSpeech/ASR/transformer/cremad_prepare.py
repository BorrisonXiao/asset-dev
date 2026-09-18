#!/usr/bin/env python3
"""Prepare CREMA-D as paired emotion/ASR manifests in the multi-task schema.

CREMA-D is the survey's non-lexical task: six emotions over twelve fixed
sentences, so the label cannot be recovered from the transcript. Because the
sentence set is fixed and known, every clip also has a free ASR view, which is
what makes the same-audio counterfactual (ASR boundaries versus emotion
boundaries on the identical recording) possible. That ASR view is 3.3 h of
twelve sentences and is for analysis only -- it is not a substitute for LS960
replay.

The corpus is already on the cluster at ``/export/corpora6/CREMA-D`` (7,442
16 kHz mono WAV), so this script decodes nothing: it reads durations, resolves
labels, and writes manifests.

**Two labellings, and they are not interchangeable** (see ADR-031):

``voice`` (default)
    The crowd majority vote on the AUDIO ONLY, from
    ``processedResults/summaryTable.csv``. This is what a listener can actually
    hear. 644 of 7,442 clips (8.7%) have no unique majority and are dropped.
    The surviving prior is heavily skewed -- about 57% neutral -- so report
    macro-F1, never raw accuracy: a constant "neutral" predictor scores ~57%
    accuracy and ~12 macro-F1.
``acted``
    The emotion the actor was instructed to perform, from the filename. Near
    uniform by construction, no ties, all 7,442 clips retained. Easier, and a
    weaker claim: it measures recovery of an instruction rather than of audible
    affect.

Both are written on every run, to separate manifests, so switching is a source
path in the hparams rather than a re-prep.

Splits are speaker-disjoint by construction: the 91 speaker ids are sorted
ascending and cut 63/9/19. The cut is recorded in the JSON sidecar so the
numbers in any report can be traced back.
"""

import argparse
import collections
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from covost2_prepare import normalize_english, write_csv  # noqa: E402
from multitask_data import EMOTION_CANDIDATES, task_id_of  # noqa: E402

SOURCE_DATASET = "cremad"

# The 12 sentences, keyed by the three-letter code in filename field 2. Copied
# from the corpus's own docs/README.md; normalized to LibriSpeech conventions
# at write time so this ASR view's WER is comparable to LS960's.
SENTENCES = {
    "DFA": "Don't forget a jacket",
    "IEO": "It's eleven o'clock",
    "IOM": "I'm on my way to the meeting",
    "ITH": "I think I have a doctor's appointment",
    "ITS": "I think I've seen this before",
    "IWL": "I would like a new alarm clock",
    "IWW": "I wonder what this is about",
    "MTI": "Maybe tomorrow it will be cold",
    "TAI": "The airplane is almost full",
    "TIE": "That is exactly what happened",
    "TSI": "The surface is slick",
    "WSI": "We'll stop in a couple of minutes",
}

# Crowd vote codes (single letter) and acted codes (filename field 3) onto the
# candidate label strings the decoder scores.
VOICE_CODES = {
    "A": "angry",
    "D": "disgust",
    "F": "fearful",
    "H": "happy",
    "N": "neutral",
    "S": "sad",
}
ACTED_CODES = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fearful",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}

# One genuine filename inconsistency in the distributed corpus: the summary
# table says ``1040_ITH_SAD_XX`` but the audio file is ``1040_ITH_SAD_X.wav``.
# Aliased explicitly rather than matched fuzzily, so any OTHER missing file
# stays a hard error instead of being silently skipped.
FILENAME_ALIASES = {"1040_ITH_SAD_XX": "1040_ITH_SAD_X"}

DEFAULT_SPLIT_SPEAKERS = (63, 9, 19)
SPLIT_NAMES = ("train", "valid", "test")


def read_summary_table(corpus_dir):
    """Rows of ``processedResults/summaryTable.csv`` with parsed filename fields.

    Returns one record per listed clip, carrying both candidate labellings and
    the pieces of the filename the split and the ASR view need.
    """
    path = os.path.join(corpus_dir, "processedResults", "summaryTable.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CREMA-D summary table not found: {path}")

    records = []
    with open(path, encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            name = row["FileName"].strip()
            fields = name.split("_")
            if len(fields) != 4:
                raise ValueError(f"Unparseable CREMA-D file name: {name!r}")
            speaker, sentence, acted, _intensity = fields
            if sentence not in SENTENCES:
                raise KeyError(
                    f"{name}: sentence code {sentence!r} is not one of the 12 "
                    f"documented sentences {sorted(SENTENCES)}"
                )
            if acted not in ACTED_CODES:
                raise KeyError(f"{name}: unknown acted emotion {acted!r}")
            vote = row["VoiceVote"].strip()
            records.append(
                {
                    "utt_key": name,
                    "speaker": speaker,
                    "sentence": sentence,
                    "acted_label": ACTED_CODES[acted],
                    # None marks a tie, i.e. no unique audible majority.
                    "voice_label": VOICE_CODES.get(vote),
                    "voice_vote_raw": vote,
                }
            )
    return records


def _audio_path(corpus_dir, utt_key):
    return os.path.join(
        corpus_dir, "AudioWAV", f"{FILENAME_ALIASES.get(utt_key, utt_key)}.wav"
    )


def _probe(args):
    """Path and duration for one clip (header read only, no decode)."""
    corpus_dir, utt_key = args
    path = _audio_path(corpus_dir, utt_key)
    if not os.path.isfile(path):
        return utt_key, None, None
    info = sf.info(path)
    return utt_key, path, info.frames / info.samplerate


def attach_audio(records, corpus_dir, workers):
    """Fill in ``wav``/``duration``; raise if any listed clip has no audio."""
    probes = [(corpus_dir, r["utt_key"]) for r in records]
    resolved = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for utt_key, path, duration in pool.map(_probe, probes, chunksize=64):
            resolved[utt_key] = (path, duration)

    missing = [k for k, (path, _) in resolved.items() if path is None]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} clips in the summary table have no audio, first "
            f"few: {missing[:5]}. Add an entry to FILENAME_ALIASES if this is "
            "another distributed-name inconsistency."
        )
    for record in records:
        path, duration = resolved[record["utt_key"]]
        record["wav"] = path
        record["duration"] = round(float(duration), 4)
    return records


def assign_splits(records, split_speakers=DEFAULT_SPLIT_SPEAKERS):
    """Speaker-disjoint split by ascending speaker id.

    Splitting on speakers rather than clips is the whole point: emotion
    recognition on held-out clips from a speaker the model trained on measures
    something much weaker than generalization to a new voice.
    """
    speakers = sorted({r["speaker"] for r in records})
    n_train, n_valid, n_test = (int(n) for n in split_speakers)
    if n_train + n_valid + n_test != len(speakers):
        raise ValueError(
            f"split_speakers {tuple(split_speakers)} sums to "
            f"{n_train + n_valid + n_test} but the corpus has "
            f"{len(speakers)} speakers"
        )
    buckets = {
        "train": set(speakers[:n_train]),
        "valid": set(speakers[n_train : n_train + n_valid]),
        "test": set(speakers[n_train + n_valid :]),
    }
    for record in records:
        for name, members in buckets.items():
            if record["speaker"] in members:
                record["split"] = name
                break
    return {name: sorted(members) for name, members in buckets.items()}


def _row(record, task_name, source_key, text, label_kind):
    """Project one clip onto a single task view."""
    return {
        # The label kind is in the ID because both labellings can be merged
        # into one manifest, and IDs must be unique across views.
        "ID": f"{task_name}-{label_kind}-{record['utt_key']}",
        "duration": record["duration"],
        "wav": record["wav"],
        "spk_id": record["speaker"],
        "wrd": text,
        "task_name": task_name,
        "task_id": task_id_of(task_name),
        "source_key": source_key,
        "source_dataset": SOURCE_DATASET,
        "source_lang": "en",
        "target_lang": "en",
        "utt_key": record["utt_key"],
    }


def emotion_rows(records, label_kind):
    """Emotion-view rows under one labelling, dropping clips with no label."""
    field = "voice_label" if label_kind == "voice" else "acted_label"
    rows = []
    for record in records:
        label = record[field]
        if label is None:
            continue
        if label not in EMOTION_CANDIDATES:
            raise ValueError(
                f"{record['utt_key']}: label {label!r} is not in the task "
                f"table's candidate set {list(EMOTION_CANDIDATES)}"
            )
        rows.append(
            _row(
                record,
                "emotion",
                f"cremad_emotion_{label_kind}",
                label,
                label_kind,
            )
        )
    return rows


def asr_rows(records, label_kind):
    """ASR-view rows over exactly the clips the emotion view keeps.

    Held to the same clip set on purpose: the paired analysis compares ASR and
    emotion boundaries on the same recordings, which is only well defined if
    both views contain the same recordings.
    """
    field = "voice_label" if label_kind == "voice" else "acted_label"
    return [
        _row(
            record,
            "asr",
            f"cremad_asr_{label_kind}",
            normalize_english(SENTENCES[record["sentence"]]),
            label_kind,
        )
        for record in records
        if record[field] is not None
    ]


def _stats(rows):
    counts = collections.Counter(row["wrd"] for row in rows)
    total = sum(counts.values())
    return {
        "clips": total,
        "hours": round(sum(r["duration"] for r in rows) / 3600.0, 3),
        "label_counts": dict(counts.most_common()),
        "majority_share": (
            round(counts.most_common(1)[0][1] / total, 4) if total else 0.0
        ),
    }


def prepare(corpus_dir, manifest_dir, split_speakers, workers, labellings):
    """Write every manifest and the provenance sidecar; return the summary."""
    records = read_summary_table(corpus_dir)
    attach_audio(records, corpus_dir, workers)
    speaker_split = assign_splits(records, split_speakers)
    # Deterministic order regardless of dict/pool ordering.
    records.sort(key=lambda r: r["utt_key"])

    ties = [r for r in records if r["voice_label"] is None]
    summary = {
        "corpus_dir": corpus_dir,
        "clips_listed": len(records),
        "voice_vote_ties_dropped": len(ties),
        "tie_examples": sorted({r["voice_vote_raw"] for r in ties})[:8],
        "speaker_split": speaker_split,
        "splits": {},
    }

    for label_kind in labellings:
        for split in SPLIT_NAMES:
            in_split = [r for r in records if r["split"] == split]
            emotion = emotion_rows(in_split, label_kind)
            asr = asr_rows(in_split, label_kind)
            suffix = "" if label_kind == "voice" else f"_{label_kind}"
            emotion_path = os.path.join(
                manifest_dir, f"cremad_emotion{suffix}_{split}.csv"
            )
            asr_path = os.path.join(
                manifest_dir, f"cremad_asr{suffix}_{split}.csv"
            )
            write_csv(emotion_path, emotion)
            write_csv(asr_path, asr)
            stats = _stats(emotion)
            summary["splits"][f"{label_kind}/{split}"] = {
                **stats,
                "emotion_manifest": emotion_path,
                "asr_manifest": asr_path,
            }
            print(
                f"[{label_kind}/{split}] {stats['clips']} clips, "
                f"{stats['hours']:.2f} h, majority "
                f"{stats['majority_share']:.1%}\n"
                f"  {emotion_path}\n  {asr_path}",
                flush=True,
            )

    sidecar = os.path.join(manifest_dir, "cremad_prep.json")
    with open(sidecar, "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
    print(f"Provenance written to {sidecar}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-dir",
        default="/export/corpora6/CREMA-D",
        help="CREMA-D root (default: the shared CLSP copy)",
    )
    parser.add_argument("--manifest-dir", required=True, help="CSV output dir")
    parser.add_argument(
        "--split-speakers",
        nargs=3,
        type=int,
        default=list(DEFAULT_SPLIT_SPEAKERS),
        metavar=("TRAIN", "VALID", "TEST"),
        help="speaker counts per split, taken in ascending speaker-id order",
    )
    parser.add_argument(
        "--labellings",
        nargs="+",
        default=["voice", "acted"],
        choices=["voice", "acted"],
        help="which labellings to write (both by default; see ADR-031)",
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.manifest_dir, exist_ok=True)
    summary = prepare(
        args.corpus_dir,
        args.manifest_dir,
        args.split_speakers,
        args.workers,
        args.labellings,
    )
    print(
        f"Prepared CREMA-D: {summary['clips_listed']} clips listed, "
        f"{summary['voice_vote_ties_dropped']} voice-vote ties",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
