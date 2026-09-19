#!/usr/bin/env python3
"""Prepare JSUT basic5000 as a clean out-of-domain Japanese test set.

JSUT audio is 48 kHz; this script resamples to 16 kHz WAV once (written next
to the corpus under ``wav16/``) and emits ``jsut_test.csv`` +
``jsut_test_units_mora.tsv`` using the same normalization and mora conversion
as reazonspeech_prepare.py.

Usage:
    python jsut_prepare.py --data_folder /path/to/jsut/jsut_ver1.1 \
        --save_folder /path/to/manifests
"""

import argparse
import csv
import os

import soundfile as sf
import torch
import torchaudio

from reazonspeech_prepare import (
    is_japanese_script,
    normalize_text,
    text_to_moras,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_folder", required=True)
    parser.add_argument("--save_folder", required=True)
    args = parser.parse_args()

    import fugashi

    tagger = fugashi.Tagger()
    basic = os.path.join(args.data_folder, "basic5000")
    wav16_dir = os.path.join(basic, "wav16")
    os.makedirs(wav16_dir, exist_ok=True)
    os.makedirs(args.save_folder, exist_ok=True)

    transcripts = {}
    with open(
        os.path.join(basic, "transcript_utf8.txt"), encoding="utf-8"
    ) as fh:
        for line in fh:
            utt, _, text = line.strip().partition(":")
            if utt and text:
                transcripts[utt] = text

    resampler = None
    rows, dropped = [], 0
    for utt in sorted(transcripts):
        src = os.path.join(basic, "wav", f"{utt}.wav")
        dst = os.path.join(wav16_dir, f"{utt}.wav")
        if not os.path.exists(dst):
            audio, sr = torchaudio.load(src)
            if resampler is None:
                resampler = torchaudio.transforms.Resample(sr, 16000)
            assert sr == resampler.orig_freq, f"mixed source rates ({sr})"
            torchaudio.save(dst, resampler(audio).to(torch.float32), 16000)
        text = normalize_text(transcripts[utt])
        if not text or not is_japanese_script(text):
            dropped += 1
            continue
        moras, covered = text_to_moras(text, tagger)
        if not covered or not moras:
            dropped += 1
            continue
        dur = round(sf.info(dst).frames / 16000.0, 3)
        rows.append((utt, dur, dst, "jsut", text, moras))

    csv_path = os.path.join(args.save_folder, "jsut_test.csv")
    units_path = os.path.join(args.save_folder, "jsut_test_units_mora.tsv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh, open(
        units_path, "w", encoding="utf-8"
    ) as uh:
        writer = csv.writer(fh)
        writer.writerow(["ID", "duration", "wav", "spk_id", "wrd"])
        for utt, dur, wav, spk, text, moras in rows:
            writer.writerow([utt, dur, wav, spk, text])
            uh.write(f"{utt}\t{' '.join(moras)}\n")
    hours = sum(r[1] for r in rows) / 3600.0
    print(f"jsut_test: {len(rows)} utts, {hours:.1f} h, dropped {dropped}")


if __name__ == "__main__":
    main()
