#!/usr/bin/env python3
"""Prepare Dynamic-SUPERB's LibriCount folds as a speaker-count eval manifest.

LibriCount is the external reference the plan names for speaker counting, and it
is already on the cluster inside the Dynamic-SUPERB download rather than needing
a separate stage:
``evals/dynamic-superb/data/HEARSpeakerCountIdentification_LibriCount-Fold{1..5}``,
200 clips each.

Why it is the right comparison. The original corpus is 5 s, 16 kHz, 0-10
speakers mixed at 0 dB SNR from LibriSpeech test-clean, and its filename encodes
"the maximum number of concurrent speakers within the 5 seconds" -- the *same*
label semantics, duration, and rate as our synthetic mixtures. Published
accuracy on the full 11-way task is around 86%, which is what makes it a usable
ceiling for judging our own numbers.

Three deliberate normalizations, each of which changes what a number means:

``48 kHz -> 16 kHz``
    The Dynamic-SUPERB pack ships upsampled audio; the original release is
    16 kHz and so is our encoder. Downsampled by the exact 1/3 integer ratio
    with ``resample_poly``, not a resampling library's default.
``integer labels -> our count words``
    The pack's labels are ``0``-``10``; we answer by ranking the candidate
    strings ``"zero"``-``"ten"``. The surface form is an artifact of the
    benchmark's prompt, not of the task, so it is mapped rather than worked
    around. A genuine Dynamic-SUPERB *submission* would instead match their
    surface form exactly.
``--max-count 4``
    Our pilot ranks 0-4. Restricting keeps the label space identical to the
    training task, at the cost of discarding 55% of the clips; the retained 455
    are near-balanced. Pass ``--max-count 10`` once the full range is trained.

**Interpretation.** Any score here is a *transfer* number, not a Dynamic-SUPERB
result: the label surface is remapped and the count range truncated. Say so
wherever it is reported.

**Environment.** This reads Arrow/parquet, and ``pyarrow`` is deliberately absent
from the pinned training venv (ADR-030). Run it with an interpreter that has
pyarrow + soundfile + scipy, e.g.
``/home/cxiao7/miniconda3/envs/scale/bin/python``. It writes plain WAV and CSV,
so training and evaluation use the normal venv afterwards.
"""

import argparse
import collections
import csv
import glob
import io
import json
import os
import sys

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SOURCE_DATASET = "libricount"
TARGET_SR = 16000

# Mirrors multitask_data.SPEAKER_COUNT_CANDIDATES. Duplicated rather than
# imported because this script runs under a different interpreter than the
# training venv; the test below asserts the two agree.
COUNT_WORDS = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
)

MULTITASK_FIELDS = [
    "ID",
    "duration",
    "wav",
    "spk_id",
    "wrd",
    "task_name",
    "task_id",
    "source_key",
    "source_dataset",
    "source_lang",
    "target_lang",
    "utt_key",
]

TASK_SPEAKER_COUNT = 3  # multitask_data.TASK_SPECS["speaker_count"]

PACK = (
    "HEARSpeakerCountIdentification_LibriCount-Fold{fold}"
)


def fold_parquet(superb_dir, fold):
    """The single parquet shard for one LibriCount fold."""
    # pyarrow is imported lazily so this module stays importable from the
    # pinned training venv, which deliberately lacks it (ADR-030). That lets the
    # unit suite assert COUNT_WORDS still matches the task table.
    pattern = os.path.join(
        superb_dir, "data", PACK.format(fold=fold), "data", "*.parquet"
    )
    found = sorted(glob.glob(pattern))
    if not found:
        raise FileNotFoundError(f"No parquet for fold {fold}: {pattern}")
    if len(found) > 1:
        raise ValueError(f"Expected one shard for fold {fold}, got {found}")
    return found[0]


def decode_clip(payload, target_sr=TARGET_SR):
    """Decoded mono waveform at ``target_sr``.

    Resampling uses the exact integer ratio when one exists (48 kHz -> 16 kHz is
    1/3), which avoids the filter-design differences between resampling
    libraries showing up as a domain shift against our 16 kHz training audio.
    """
    wave, sr = sf.read(io.BytesIO(payload), dtype="float32", always_2d=False)
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    if sr == target_sr:
        return wave, sr
    if sr % target_sr == 0:
        return resample_poly(wave, 1, sr // target_sr).astype(np.float32), sr
    from math import gcd

    g = gcd(sr, target_sr)
    return resample_poly(wave, target_sr // g, sr // g).astype(np.float32), sr


def prepare(superb_dir, out_dir, audio_dir, folds, max_count):
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    # Lazily imported so this module stays importable from the pinned training
    # venv, which deliberately lacks pyarrow (ADR-030).
    import pyarrow.parquet as pq

    rows = []
    kept = collections.Counter()
    dropped = collections.Counter()
    source_rates = collections.Counter()

    for fold in folds:
        table = pq.read_table(fold_parquet(superb_dir, fold))
        audio = table.column("audio").to_pylist()
        label = table.column("label").to_pylist()
        name = table.column("file").to_pylist()
        for payload, count, key in zip(audio, label, name):
            count = int(count)
            if count > max_count:
                dropped[count] += 1
                continue
            if count >= len(COUNT_WORDS):
                raise ValueError(
                    f"Count {count} has no word in the task table's "
                    f"{len(COUNT_WORDS)}-label vocabulary"
                )
            wave, sr = decode_clip(payload["bytes"])
            source_rates[sr] += 1
            utt_key = f"libricount-f{fold}-{key}"
            path = os.path.join(audio_dir, f"{utt_key}.wav")
            # Written atomically so an interrupted run cannot leave a
            # truncated WAV a later run would trust. The format is explicit
            # because the ".tmp" suffix defeats extension-based inference.
            tmp = path + ".tmp"
            sf.write(tmp, wave, TARGET_SR, subtype="PCM_16", format="WAV")
            os.replace(tmp, path)
            rows.append(
                {
                    "ID": f"speaker_count-{utt_key}",
                    "duration": round(len(wave) / TARGET_SR, 4),
                    "wav": path,
                    # LibriCount does not expose speaker identities per clip in
                    # this pack; the count is the label, not the identities.
                    "spk_id": "unknown",
                    "wrd": COUNT_WORDS[count],
                    "task_name": "speaker_count",
                    "task_id": TASK_SPEAKER_COUNT,
                    "source_key": "libricount",
                    "source_dataset": SOURCE_DATASET,
                    "source_lang": "en",
                    "target_lang": "en",
                    "utt_key": utt_key,
                }
            )
            kept[count] += 1

    rows.sort(key=lambda r: r["utt_key"])
    manifest = os.path.join(out_dir, f"libricount_test_0to{max_count}.csv")
    with open(manifest, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MULTITASK_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    total_kept = sum(kept.values())
    summary = {
        "superb_dir": superb_dir,
        "folds": list(folds),
        "max_count": max_count,
        "clips_kept": total_kept,
        "clips_dropped_above_max_count": sum(dropped.values()),
        "kept_by_count": dict(sorted(kept.items())),
        "dropped_by_count": dict(sorted(dropped.items())),
        "source_sample_rates": dict(source_rates),
        "target_sample_rate": TARGET_SR,
        "label_surface": "integers remapped to count words; a Dynamic-SUPERB "
        "submission would keep the integers",
        "manifest": manifest,
        "audio_dir": audio_dir,
    }
    with open(
        os.path.join(out_dir, "libricount_prep.json"), "w", encoding="utf-8"
    ) as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)

    majority = max(kept.values()) / total_kept if total_kept else 0.0
    print(
        f"kept {total_kept} clips (dropped {sum(dropped.values())} above count "
        f"{max_count})\n  by count: {dict(sorted(kept.items()))}\n"
        f"  majority-class baseline {majority:.1%}, chance "
        f"{1 / len(kept):.1%}\n  {manifest}",
        flush=True,
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--superb-dir",
        default="/export/jsalt26/omnienc/evals/dynamic-superb",
        help="Dynamic-SUPERB root (read-only shared data)",
    )
    parser.add_argument(
        "--out-dir",
        default="/export/jsalt26/omnienc/users/cxiao/datasets/nonasr_manifests",
    )
    parser.add_argument(
        "--audio-dir",
        default="/export/jsalt26/omnienc/users/cxiao/datasets/libricount_16k",
    )
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument(
        "--max-count",
        type=int,
        default=4,
        help="drop clips above this count (pilot ranks 0-4)",
    )
    args = parser.parse_args()

    summary = prepare(
        args.superb_dir,
        args.out_dir,
        args.audio_dir,
        args.folds,
        args.max_count,
    )
    print(
        f"Prepared LibriCount: {summary['clips_kept']} clips", file=sys.stderr
    )


if __name__ == "__main__":
    main()
