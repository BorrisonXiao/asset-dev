"""Prepare CoVoST 2 En->De as paired ASR/ST manifests for the multi-task study.

MuST-C v1 is distributed behind an FBK registration form, so the paired corpus
for the task-conditioned segmenter study is CoVoST 2 En->De. It preserves the
property the study depends on: one English utterance carries both an English
transcript and a German translation, so ``asr`` and ``st_en_de`` are two views
of the *same audio* and the same-audio boundary counterfactual is well defined.

For each split this script

1. decodes the packed Common Voice MP3 (48 kHz) from the parquet shards,
2. resamples to 16 kHz mono and writes FLAC next to a stable utterance key, and
3. emits two SpeechBrain CSV manifests -- one per task view -- that share
   ``utt_key`` and the audio path.

The English transcript is normalized to LibriSpeech conventions (upper case, no
punctuation) so ASR WER stays comparable across the two ASR sources. The German
translation is written verbatim because sacreBLEU scores case-sensitive
detokenized text.
"""

import argparse
import csv
import io
import os
import re
import sys
import unicodedata
from concurrent.futures import ProcessPoolExecutor

import soundfile as sf
import torch
import torchaudio

TARGET_SR = 16000
SPLITS = ("train", "validation", "test")

# Two task views of one audio file. ``task_id`` is the index used by the
# segmenter FiLM and the task-routed decoder adapters.
TASK_ASR = 0
TASK_ST_EN_DE = 1

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

# Keep the apostrophe: LibriSpeech transcripts contain "DIDN'T", "OLD FELLOW'S".
_PUNCT = re.compile(r"[^A-Z' ]+")
_SPACES = re.compile(r"\s+")


def normalize_english(text):
    """Fold a CoVoST English sentence onto LibriSpeech transcript conventions."""
    # Curly quotes/apostrophes appear throughout Common Voice prompts.
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", " ").replace("”", " ")
    text = text.replace("–", " ").replace("—", " ")
    text = _PUNCT.sub(" ", text.upper())
    return _SPACES.sub(" ", text).strip()


def normalize_german(text):
    """Whitespace-only cleanup; case and punctuation are scored by sacreBLEU."""
    text = unicodedata.normalize("NFKC", text)
    return _SPACES.sub(" ", text).strip()


def _decode_shard(args):
    """Decode one parquet shard to FLAC and return its manifest rows."""
    shard_path, audio_dir, split = args
    import pyarrow.parquet as pq

    rows = []
    resamplers = {}
    parquet = pq.ParquetFile(shard_path)
    for batch in parquet.iter_batches(batch_size=64):
        for record in batch.to_pylist():
            utt_key = record["id"]
            english = normalize_english(record["sentence"] or "")
            german = normalize_german(record["translation"] or "")
            if not english or not german:
                continue

            out_path = os.path.join(audio_dir, f"{utt_key}.flac")
            if os.path.exists(out_path):
                info = sf.info(out_path)
                duration = info.frames / info.samplerate
            else:
                payload = record["audio"]["bytes"]
                if not payload:
                    continue
                wave, sample_rate = sf.read(
                    io.BytesIO(payload), dtype="float32"
                )
                wave = torch.from_numpy(wave)
                if wave.ndim > 1:
                    wave = wave.mean(dim=1)
                if sample_rate != TARGET_SR:
                    if sample_rate not in resamplers:
                        resamplers[sample_rate] = (
                            torchaudio.transforms.Resample(
                                sample_rate, TARGET_SR
                            )
                        )
                    wave = resamplers[sample_rate](wave)
                if wave.numel() == 0:
                    continue
                # Written atomically so an interrupted run cannot leave a
                # truncated FLAC that a later run would trust via sf.info.
                tmp_path = out_path + ".tmp"
                sf.write(tmp_path, wave.numpy(), TARGET_SR, format="FLAC")
                os.replace(tmp_path, out_path)
                duration = wave.numel() / TARGET_SR

            rows.append(
                {
                    "utt_key": utt_key,
                    "duration": round(float(duration), 4),
                    "wav": out_path,
                    "spk_id": record["client_id"][:32],
                    "english": english,
                    "german": german,
                }
            )
    return rows


def _view_row(
    row, task_name, task_id, source_key, text, source_lang, target_lang
):
    """Project one decoded utterance onto a single task view."""
    return {
        "ID": f"{task_name}-{row['utt_key']}",
        "duration": row["duration"],
        "wav": row["wav"],
        "spk_id": row["spk_id"],
        "wrd": text,
        "task_name": task_name,
        "task_id": task_id,
        "source_key": source_key,
        "source_dataset": "covost2_en_de",
        "source_lang": source_lang,
        "target_lang": target_lang,
        "utt_key": row["utt_key"],
    }


def write_csv(path, rows):
    """Write one manifest, creating parent directories as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MULTITASK_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def prepare_split(parquet_dir, split, audio_root, manifest_dir, workers):
    """Decode one CoVoST split and emit its paired ASR/ST manifests."""
    shards = sorted(
        os.path.join(parquet_dir, name)
        for name in os.listdir(parquet_dir)
        if name.startswith(f"{split}-") and name.endswith(".parquet")
    )
    if not shards:
        raise FileNotFoundError(f"No {split} shards under {parquet_dir}")

    audio_dir = os.path.join(audio_root, split)
    os.makedirs(audio_dir, exist_ok=True)

    rows = []
    tasks = [(shard, audio_dir, split) for shard in shards]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for done, shard_rows in enumerate(pool.map(_decode_shard, tasks), 1):
            rows.extend(shard_rows)
            print(
                f"  [{split}] shard {done}/{len(shards)}: "
                f"{len(shard_rows)} utts (total {len(rows)})",
                flush=True,
            )

    # Deterministic order regardless of shard completion order.
    rows.sort(key=lambda r: r["utt_key"])

    asr_rows = [
        _view_row(r, "asr", TASK_ASR, "covost_asr", r["english"], "en", "en")
        for r in rows
    ]
    st_rows = [
        _view_row(
            r, "st_en_de", TASK_ST_EN_DE, "covost_st", r["german"], "en", "de"
        )
        for r in rows
    ]

    asr_path = os.path.join(manifest_dir, f"covost_asr_{split}.csv")
    st_path = os.path.join(manifest_dir, f"covost_st_{split}.csv")
    write_csv(asr_path, asr_rows)
    write_csv(st_path, st_rows)

    hours = sum(r["duration"] for r in rows) / 3600.0
    print(
        f"[{split}] {len(rows)} paired utterances, {hours:.1f} h\n"
        f"  {asr_path}\n  {st_path}",
        flush=True,
    )
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parquet-dir",
        required=True,
        help="directory holding the en_de/*.parquet shards",
    )
    parser.add_argument("--audio-root", required=True, help="FLAC output root")
    parser.add_argument("--manifest-dir", required=True, help="CSV output dir")
    parser.add_argument("--splits", nargs="+", default=list(SPLITS))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.manifest_dir, exist_ok=True)
    total = 0
    for split in args.splits:
        total += prepare_split(
            args.parquet_dir,
            split,
            args.audio_root,
            args.manifest_dir,
            args.workers,
        )
    print(f"Prepared {total} paired CoVoST 2 En-De utterances", file=sys.stderr)


if __name__ == "__main__":
    main()
