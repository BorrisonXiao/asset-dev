#!/usr/bin/env python3
"""Prepare Fluent Speech Commands as paired intent/ASR manifests.

FSC is the survey's easy semantic control: 31 closed intents over short spoken
commands, with a gold transcript on every row. That transcript gives two things
the study needs -- a same-audio ASR view for the boundary counterfactual, and a
transcript-only intent pipeline as the strong text control that emotion and
speaker count are expected to fail.

The corpus is already on the cluster at
``/export/corpora6/fluent_speech_commands_dataset`` (16 kHz mono WAV) with its
official speaker-disjoint split in ``data/{train,valid,test}_data.csv``, so this
script decodes nothing: it reads durations, builds the canonical intent string,
and writes manifests.

The intent label is the non-empty slots of the (action, object, location) tuple
joined in that order -- ``activate lights kitchen``, ``change language``,
``decrease volume``. Those 31 strings are pinned in
``multitask_data.INTENT_CANDIDATES``; this script checks the corpus against that
tuple and raises on any disagreement, because a label the decoder never scores
is a label the model can never get right, and that failure is otherwise
invisible.
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
from multitask_data import INTENT_CANDIDATES, task_id_of  # noqa: E402

SOURCE_DATASET = "fluent_speech_commands"
SPLIT_NAMES = ("train", "valid", "test")
SLOT_NONE = "none"


def canonical_intent(action, obj, location):
    """Non-empty slots joined in action-object-location order."""
    slots = [s.strip() for s in (action, obj, location)]
    return " ".join(s for s in slots if s and s != SLOT_NONE)


def read_split(corpus_dir, split):
    """Rows of one official split, with the canonical intent attached."""
    path = os.path.join(corpus_dir, "data", f"{split}_data.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"FSC split not found: {path}")

    records = []
    with open(path, encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            rel = row["path"].strip()
            intent = canonical_intent(
                row["action"], row["object"], row["location"]
            )
            if intent not in INTENT_CANDIDATES:
                raise ValueError(
                    f"{rel}: intent {intent!r} is not in the task table's "
                    f"candidate set. Either the corpus changed or "
                    "multitask_data.INTENT_CANDIDATES is stale; regenerate it "
                    "rather than widening this check."
                )
            records.append(
                {
                    # The uuid basename is unique across all three splits
                    # (verified: 30,043 rows, 30,043 distinct), so it is a
                    # stable pairing key between the intent and ASR views.
                    "utt_key": os.path.splitext(os.path.basename(rel))[0],
                    "wav": os.path.join(corpus_dir, rel),
                    "speaker": row["speakerId"].strip(),
                    "intent": intent,
                    "transcription": row["transcription"].strip(),
                }
            )
    return records


def _probe(path):
    """Duration of one clip (header read only, no decode)."""
    if not os.path.isfile(path):
        return path, None
    info = sf.info(path)
    return path, info.frames / info.samplerate


def attach_durations(records, workers):
    """Fill in ``duration``; raise if any manifest row has no audio."""
    paths = [r["wav"] for r in records]
    resolved = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for path, duration in pool.map(_probe, paths, chunksize=64):
            resolved[path] = duration

    missing = [p for p, d in resolved.items() if d is None]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} FSC rows have no audio, first few: {missing[:5]}"
        )
    for record in records:
        record["duration"] = round(float(resolved[record["wav"]]), 4)
    return records


def _row(record, task_name, source_key, text):
    """Project one clip onto a single task view."""
    return {
        "ID": f"{task_name}-{record['utt_key']}",
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


def _stats(records):
    counts = collections.Counter(r["intent"] for r in records)
    total = len(records)
    return {
        "clips": total,
        "hours": round(sum(r["duration"] for r in records) / 3600.0, 3),
        "speakers": len({r["speaker"] for r in records}),
        "intents_present": len(counts),
        "majority_share": (
            round(counts.most_common(1)[0][1] / total, 4) if total else 0.0
        ),
    }


def check_speaker_disjoint(by_split):
    """Confirm the official split really is speaker disjoint before training.

    The corpus documents this, but an intent model that has heard the test
    speakers is measuring something else entirely, so it is worth one set
    intersection rather than a footnote.
    """
    speakers = {
        split: {r["speaker"] for r in records}
        for split, records in by_split.items()
    }
    overlaps = {}
    for split in ("valid", "test"):
        shared = speakers["train"] & speakers.get(split, set())
        if shared:
            overlaps[f"train&{split}"] = sorted(shared)
    if overlaps:
        raise ValueError(f"FSC splits are not speaker disjoint: {overlaps}")
    return {split: len(members) for split, members in speakers.items()}


def prepare(corpus_dir, manifest_dir, workers, splits):
    """Write every manifest and the provenance sidecar; return the summary."""
    by_split = {}
    for split in splits:
        records = read_split(corpus_dir, split)
        attach_durations(records, workers)
        records.sort(key=lambda r: r["utt_key"])
        by_split[split] = records

    summary = {
        "corpus_dir": corpus_dir,
        "intents": len(INTENT_CANDIDATES),
        "speakers_per_split": check_speaker_disjoint(by_split),
        "splits": {},
    }

    for split, records in by_split.items():
        intent_path = os.path.join(manifest_dir, f"fsc_intent_{split}.csv")
        asr_path = os.path.join(manifest_dir, f"fsc_asr_{split}.csv")
        write_csv(
            intent_path,
            [_row(r, "intent", "fsc_intent", r["intent"]) for r in records],
        )
        write_csv(
            asr_path,
            [
                _row(
                    r,
                    "asr",
                    "fsc_asr",
                    normalize_english(r["transcription"]),
                )
                for r in records
            ],
        )
        stats = _stats(records)
        summary["splits"][split] = {
            **stats,
            "intent_manifest": intent_path,
            "asr_manifest": asr_path,
        }
        print(
            f"[{split}] {stats['clips']} clips, {stats['hours']:.2f} h, "
            f"{stats['speakers']} speakers, "
            f"{stats['intents_present']}/{len(INTENT_CANDIDATES)} intents\n"
            f"  {intent_path}\n  {asr_path}",
            flush=True,
        )

    sidecar = os.path.join(manifest_dir, "fsc_prep.json")
    with open(sidecar, "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
    print(f"Provenance written to {sidecar}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-dir",
        default="/export/corpora6/fluent_speech_commands_dataset",
        help="FSC root (default: the shared CLSP copy)",
    )
    parser.add_argument("--manifest-dir", required=True, help="CSV output dir")
    parser.add_argument("--splits", nargs="+", default=list(SPLIT_NAMES))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.manifest_dir, exist_ok=True)
    summary = prepare(
        args.corpus_dir, args.manifest_dir, args.workers, args.splits
    )
    total = sum(s["clips"] for s in summary["splits"].values())
    print(f"Prepared {total} FSC utterances", file=sys.stderr)


if __name__ == "__main__":
    main()
