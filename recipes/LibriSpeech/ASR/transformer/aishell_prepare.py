#!/usr/bin/env python3
"""Prepare AISHELL-1 manifests for the cross-lingual step24k replication.

Produces SpeechBrain-style CSVs (``ID,duration,wav,spk_id,wrd``) with ABSOLUTE
wav paths plus, per split, a ``{split}_units_char.tsv`` file
(``utt_id<TAB>space-joined characters``) consumed by the char-CTC aligner
(xling_ctc_boundary_align.py).

Text: the corpus transcript (aishell_transcript_v0.8.txt) is already
word-segmented with spaces. ``wrd`` keeps that segmentation (useful to MFA and
for a meaningful word-level WER); CER is computed from characters and is
whitespace-invariant. The char units drop the spaces.

Usage:
    python aishell_prepare.py \
        --data_folder /path/to/aishell/data_aishell \
        --save_folder /path/to/manifests
"""

import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import soundfile as sf

SPLITS = ("train", "dev", "test")
# Published split sizes (utterances with transcripts); hard-checked below.
EXPECTED = {"train": 120098, "dev": 14326, "test": 7176}


def read_transcripts(path):
    """utt_id -> space-segmented transcript."""
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) >= 2:
                out[parts[0]] = " ".join(parts[1:])
    return out


def _probe(path):
    info = sf.info(path)
    return round(info.frames / info.samplerate, 3), info.samplerate


def collect_split(wav_root, split, transcripts):
    rows, missing_text = [], 0
    split_dir = os.path.join(wav_root, split)
    for spk in sorted(os.listdir(split_dir)):
        spk_dir = os.path.join(split_dir, spk)
        for name in sorted(os.listdir(spk_dir)):
            if not name.endswith(".wav"):
                continue
            utt = name[: -len(".wav")]
            text = transcripts.get(utt)
            if text is None:
                missing_text += 1
                continue
            rows.append((utt, os.path.join(spk_dir, name), spk, text))
    return rows, missing_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_folder", required=True)
    parser.add_argument("--save_folder", required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    transcript = os.path.join(
        args.data_folder, "transcript", "aishell_transcript_v0.8.txt"
    )
    wav_root = os.path.join(args.data_folder, "wav")
    transcripts = read_transcripts(transcript)
    os.makedirs(args.save_folder, exist_ok=True)

    ok = True
    for split in SPLITS:
        rows, missing = collect_split(wav_root, split, transcripts)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            probes = list(pool.map(_probe, [r[1] for r in rows], chunksize=256))
        bad_sr = sum(1 for _, sr in probes if sr != 16000)
        if bad_sr:
            print(f"ERROR: {bad_sr} non-16kHz files in {split}")
            ok = False

        csv_path = os.path.join(args.save_folder, f"{split}.csv")
        units_path = os.path.join(args.save_folder, f"{split}_units_char.tsv")
        with open(csv_path, "w", newline="", encoding="utf-8") as fh, open(
            units_path, "w", encoding="utf-8"
        ) as uh:
            writer = csv.writer(fh)
            writer.writerow(["ID", "duration", "wav", "spk_id", "wrd"])
            for (utt, wav, spk, text), (dur, _) in zip(rows, probes):
                writer.writerow([utt, dur, wav, spk, text])
                chars = " ".join(text.replace(" ", ""))
                uh.write(f"{utt}\t{chars}\n")

        total_h = sum(d for d, _ in probes) / 3600.0
        print(
            f"{split}: {len(rows)} utts, {total_h:.1f} h, "
            f"{missing} wavs without transcript (skipped)"
        )
        if len(rows) != EXPECTED[split]:
            print(
                f"ERROR: {split} has {len(rows)} utts, "
                f"expected {EXPECTED[split]}"
            )
            ok = False
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
