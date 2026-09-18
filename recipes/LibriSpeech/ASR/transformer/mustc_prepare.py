#!/usr/bin/env python3
"""Prepare MuST-C v1 En-De as paired ASR/ST manifests in the multi-task schema.

MuST-C ships full-talk WAVs plus a YAML of segment offsets, unlike CoVoST's
pre-segmented clips, so this cuts segments out of each talk. The output schema
is identical to ``covost2_prepare.py`` so the trainer, sampler, and analysis
code need no changes -- the ST corpus is a configuration choice.

Two text conventions, for the same reasons as CoVoST:

* the English side is folded to LibriSpeech style (upper case, no punctuation)
  so ASR WER stays comparable across corpora, and non-speech annotations --
  ``(Laughter)``, ``(Applause)`` and friends, present on ~5% of lines -- are
  removed first. They are not spoken, so leaving them in would put phantom
  words in the reference and inflate WER against any model that never emits
  them.
* the German side is kept verbatim, including its own annotations, because
  sacreBLEU scores case-sensitive detokenized text and published MuST-C BLEU is
  computed on the distributed target as-is.
"""

import argparse
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor

import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from covost2_prepare import (  # noqa: E402
    TASK_ASR,
    TASK_ST_EN_DE,
    normalize_english,
    normalize_german,
    write_csv,
)

TARGET_SR = 16000
SPLITS = ("train", "dev", "tst-COMMON", "tst-HE")

# "- {duration: 28.800000, offset: 6.170000, speaker_id: spk.1, wav: ted_1.wav}"
# Parsed by regex rather than pyyaml: the file is 19 MB / 230k flow-style
# entries and a real YAML parse takes minutes for no benefit.
YAML_ENTRY = re.compile(
    r"duration:\s*([0-9.]+).*?offset:\s*([0-9.]+).*?"
    r"speaker_id:\s*(\S+?),.*?wav:\s*(\S+?)\}"
)
NON_SPEECH = re.compile(r"\([^)]*\)")


def strip_non_speech(text):
    """Drop bracketed non-speech annotations before ASR normalization."""
    return NON_SPEECH.sub(" ", text)


def read_split(root, split):
    """Segment records for one split, joined with their parallel text."""
    txt = os.path.join(root, "data", split, "txt")
    with open(os.path.join(txt, f"{split}.yaml"), encoding="utf-8") as stream:
        entries = YAML_ENTRY.findall(stream.read())
    with open(os.path.join(txt, f"{split}.en"), encoding="utf-8") as stream:
        english = [line.rstrip("\n") for line in stream]
    with open(os.path.join(txt, f"{split}.de"), encoding="utf-8") as stream:
        german = [line.rstrip("\n") for line in stream]

    if not (len(entries) == len(english) == len(german)):
        raise ValueError(
            f"{split}: yaml/en/de disagree -- {len(entries)}/{len(english)}/"
            f"{len(german)}. The three files are positionally aligned, so a "
            "mismatch means the corpus copy is inconsistent."
        )

    per_talk = {}
    for index, ((duration, offset, speaker, wav), en, de) in enumerate(
        zip(entries, english, german)
    ):
        per_talk.setdefault(wav, []).append(
            {
                "index": index,
                "duration": float(duration),
                "offset": float(offset),
                "speaker_id": speaker,
                "wav": wav,
                "english": en,
                "german": de,
            }
        )
    return per_talk


def _cut_talk(job):
    """Cut every segment of one talk; returns manifest rows."""
    wav_name, segments, wav_dir, audio_dir = job
    path = os.path.join(wav_dir, wav_name)
    try:
        info = sf.info(path)
    except RuntimeError:
        return [], [f"unreadable talk {wav_name}"]

    total_frames = info.frames
    rows, problems = [], []
    stem = os.path.splitext(wav_name)[0]

    for order, seg in enumerate(sorted(segments, key=lambda s: s["offset"])):
        utt_key = f"{stem}_{order:04d}"
        out_path = os.path.join(audio_dir, f"{utt_key}.flac")

        english = normalize_english(strip_non_speech(seg["english"]))
        german = normalize_german(seg["german"])
        # The two views are kept independently. A line that was entirely a
        # non-speech annotation ("(Laughter)") leaves no English to transcribe,
        # but its German translation is still a valid ST reference -- dropping
        # the segment outright would silently shrink tst-COMMON below the 2641
        # every published MuST-C BLEU is computed on.
        if not english and not german:
            problems.append(f"{utt_key}: no usable text in either language")
            continue

        if os.path.exists(out_path):
            duration = sf.info(out_path).frames / TARGET_SR
        else:
            start = int(round(seg["offset"] * info.samplerate))
            stop = start + int(round(seg["duration"] * info.samplerate))
            # MuST-C v1 has segments whose offset+duration runs past the talk;
            # clamp instead of failing, and record it.
            if start >= total_frames or seg["duration"] <= 0:
                problems.append(f"{utt_key}: segment outside talk, skipped")
                continue
            if stop > total_frames:
                problems.append(f"{utt_key}: clipped to talk end")
                stop = total_frames

            audio, sample_rate = sf.read(
                path, start=start, stop=stop, dtype="float32", always_2d=False
            )
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if audio.size == 0:
                problems.append(f"{utt_key}: empty segment, skipped")
                continue
            if sample_rate != TARGET_SR:
                import torch
                import torchaudio

                audio = torchaudio.functional.resample(
                    torch.from_numpy(audio), sample_rate, TARGET_SR
                ).numpy()
            tmp_path = out_path + ".tmp"
            sf.write(tmp_path, audio, TARGET_SR, format="FLAC")
            os.replace(tmp_path, out_path)
            duration = audio.size / TARGET_SR

        rows.append(
            {
                "has_asr": bool(english),
                "has_st": bool(german),
                "utt_key": utt_key,
                "duration": round(float(duration), 4),
                "wav": out_path,
                "spk_id": seg["speaker_id"],
                "english": english,
                "german": german,
            }
        )
    return rows, problems


def _view(row, task_name, task_id, source_key, text, target_lang):
    return {
        "ID": f"{task_name}-{row['utt_key']}",
        "duration": row["duration"],
        "wav": row["wav"],
        "spk_id": row["spk_id"],
        "wrd": text,
        "task_name": task_name,
        "task_id": task_id,
        "source_key": source_key,
        "source_dataset": "mustc_v1_en_de",
        "source_lang": "en",
        "target_lang": target_lang,
        "utt_key": row["utt_key"],
    }


def prepare_split(root, split, audio_root, manifest_dir, workers):
    """Cut one split and emit its paired ASR/ST manifests."""
    per_talk = read_split(root, split)
    wav_dir = os.path.join(root, "data", split, "wav")
    audio_dir = os.path.join(audio_root, split)
    os.makedirs(audio_dir, exist_ok=True)

    jobs = [
        (wav, segs, wav_dir, audio_dir)
        for wav, segs in sorted(per_talk.items())
    ]
    rows, problems = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for done, (talk_rows, talk_problems) in enumerate(
            pool.map(_cut_talk, jobs), 1
        ):
            rows.extend(talk_rows)
            problems.extend(talk_problems)
            if done % 50 == 0 or done == len(jobs):
                print(
                    f"  [{split}] talk {done}/{len(jobs)}: "
                    f"{len(rows)} segments so far",
                    flush=True,
                )

    rows.sort(key=lambda r: r["utt_key"])
    asr_rows = [
        _view(r, "asr", TASK_ASR, "mustc_asr", r["english"], "en")
        for r in rows
        if r["has_asr"]
    ]
    st_rows = [
        _view(r, "st_en_de", TASK_ST_EN_DE, "mustc_st", r["german"], "de")
        for r in rows
        if r["has_st"]
    ]
    asr_path = os.path.join(manifest_dir, f"mustc_asr_{split}.csv")
    st_path = os.path.join(manifest_dir, f"mustc_st_{split}.csv")
    write_csv(asr_path, asr_rows)
    write_csv(st_path, st_rows)

    hours = sum(r["duration"] for r in rows) / 3600.0
    print(
        f"[{split}] {len(rows)} segments, {hours:.1f} h -- "
        f"ASR view {len(asr_rows)}, ST view {len(st_rows)} "
        f"({len(problems)} issues)\n  {asr_path}\n  {st_path}",
        flush=True,
    )
    for problem in problems[:10]:
        print(f"  ! {problem}", flush=True)
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mustc-root", required=True, help="the en-de dir")
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--splits", nargs="+", default=list(SPLITS))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.manifest_dir, exist_ok=True)
    total = 0
    for split in args.splits:
        total += prepare_split(
            args.mustc_root,
            split,
            args.audio_root,
            args.manifest_dir,
            args.workers,
        )
    print(f"Prepared {total} paired MuST-C En-De segments", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main() or 0)
