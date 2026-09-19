#!/usr/bin/env python3
"""Prepare ReazonSpeech v2 `small` manifests for the cross-lingual step24k study.

Extracts the 12 tar shards, normalizes transcripts, converts them to katakana
mora sequences (fugashi + unidic-lite), carves deterministic held-out dev/test
sets, and writes SpeechBrain-style CSVs (``ID,duration,wav,spk_id,wrd``) with
ABSOLUTE audio paths plus ``{split}_units_mora.tsv``
(``utt_id<TAB>space-joined moras``) for the mora-CTC aligner.

Conventions
-----------
* Text normalization: NFKC, punctuation stripped, whitespace removed. CER is
  computed on these normalized strings.
* ``spk_id``: ReazonSpeech carries no speaker labels; the shard prefix is used
  as a coarse grouping id (never used for training decisions).
* Split: deterministic MD5 hash of the utterance name — dev ≈ 1000 utts,
  test ≈ 2000 utts, remainder train. ReazonSpeech has no speaker/program
  metadata, so cross-split program overlap is possible; JSUT basic5000 is the
  clean out-of-domain test set counterweight.
* Filters: 1.0 s <= duration <= 30.0 s, non-empty normalized text,
  Japanese-script-only (kanji/kana; measured on the small subset, 18.3% of
  subtitle lines contain digits/latin, and unidic-lite provides no kana
  readings for those tokens — rather than fabricating number readings for the
  mora oracle, the corpus is defined as the Japanese-script subset), and mora
  conversion must cover the utterance (counts reported). Every arm trains and
  evaluates on the same filtered manifests, so comparisons are internally
  fair.

Usage:
    python reazonspeech_prepare.py \
        --data_folder /path/to/reazonspeech_small \
        --save_folder /path/to/manifests [--skip_extract]
"""

import argparse
import csv
import hashlib
import os
import re
import sys
import tarfile
import unicodedata
from concurrent.futures import ProcessPoolExecutor

import soundfile as sf

NUM_SHARDS = 12

_PUNCT = re.compile(
    r"[、。．，,\.!！?？「」『』（）()\[\]【】〈〉《》・…‥ー?—–\-:;：；\"'’‘”“〜~/\\|＜＞<>*＊+＝=%％#＃&＆@＠]"
)
# ー (long vowel mark) must NOT be stripped from katakana readings; the
# punctuation regex above is applied to the ORTHOGRAPHIC transcript only,
# where a bare prolonged-sound mark outside katakana context is noise.
_KATAKANA = re.compile(r"^[ァ-ヴー]+$")
_SMALL_KANA = set("ァィゥェォャュョヮ")
# Kanji + kana + iteration/長音 marks; no digits, latin, or symbols.
_JAPANESE_SCRIPT = re.compile(r"^[ぁ-ゖァ-ヴー一-鿿々〆ゝゞヽヾ]+$")


def is_japanese_script(text):
    """True when the string is purely kanji/kana (the corpus definition)."""
    return bool(_JAPANESE_SCRIPT.match(text))


def normalize_text(text):
    """NFKC-normalize, strip punctuation and all whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = _PUNCT.sub("", text)
    return "".join(text.split())


def kana_to_moras(kana):
    """Split a katakana string into moras (small kana attach left)."""
    moras = []
    for ch in kana:
        if ch in _SMALL_KANA and moras:
            moras[-1] += ch
        else:
            moras.append(ch)
    return moras


def text_to_moras(text, tagger):
    """Katakana mora sequence for a normalized Japanese string.

    Returns (moras, covered): ``covered`` is False when any token lacks a
    reading (rare symbols); such utterances are dropped by the caller.
    """
    moras, covered = [], True
    for word in tagger(text):
        kana = word.feature.kana or word.feature.pron
        if not kana:
            if _KATAKANA.match(word.surface):
                kana = word.surface
            else:
                covered = False
                continue
        kana = unicodedata.normalize("NFKC", kana)
        if not _KATAKANA.match(kana):
            kana = "".join(c for c in kana if _KATAKANA.match(c))
            if not kana:
                covered = False
                continue
        moras.extend(kana_to_moras(kana))
    return moras, covered


def split_of(name, dev_per_mille=16, test_per_mille=32):
    """Deterministic split: ~1.6% dev, ~3.2% test of 62k utts."""
    h = int(hashlib.md5(name.encode()).hexdigest(), 16) % 1000
    if h < dev_per_mille:
        return "dev"
    if h < dev_per_mille + test_per_mille:
        return "test"
    return "train"


def _probe(path):
    try:
        info = sf.info(path)
        return round(info.frames / info.samplerate, 3), info.samplerate
    except Exception:
        return None, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_folder", required=True)
    parser.add_argument("--save_folder", required=True)
    parser.add_argument("--skip_extract", action="store_true")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    import fugashi

    tagger = fugashi.Tagger()
    audio_root = os.path.join(args.data_folder, "audio")
    if not args.skip_extract:
        os.makedirs(audio_root, exist_ok=True)
        for i in range(NUM_SHARDS):
            shard = os.path.join(args.data_folder, f"{i:03x}.tar")
            with tarfile.open(shard) as tf:
                tf.extractall(audio_root)
            print("extracted", shard)

    rows = {"train": [], "dev": [], "test": []}
    dropped = {
        "missing_audio": 0,
        "empty_text": 0,
        "duration": 0,
        "non_japanese_script": 0,
        "mora": 0,
    }
    entries = []
    with open(os.path.join(args.data_folder, "small.tsv"), encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                continue
            entries.append(parts)

    paths = [os.path.join(audio_root, name) for name, _ in entries]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        probes = list(pool.map(_probe, paths, chunksize=256))

    for (name, raw_text), (dur, sr) in zip(entries, probes):
        if dur is None:
            dropped["missing_audio"] += 1
            continue
        assert sr == 16000, f"unexpected sample rate {sr} for {name}"
        if not (1.0 <= dur <= 30.0):
            dropped["duration"] += 1
            continue
        text = normalize_text(raw_text)
        if not text:
            dropped["empty_text"] += 1
            continue
        if not _JAPANESE_SCRIPT.match(text):
            dropped["non_japanese_script"] += 1
            continue
        moras, covered = text_to_moras(text, tagger)
        if not covered or not moras:
            dropped["mora"] += 1
            continue
        utt = name.replace("/", "_").replace(".flac", "")
        shard = name.split("/")[0]
        wav = os.path.join(audio_root, name)
        rows[split_of(name)].append((utt, dur, wav, f"reazon_{shard}", text, moras))

    os.makedirs(args.save_folder, exist_ok=True)
    for split, split_rows in rows.items():
        csv_path = os.path.join(args.save_folder, f"{split}.csv")
        units_path = os.path.join(args.save_folder, f"{split}_units_mora.tsv")
        with open(csv_path, "w", newline="", encoding="utf-8") as fh, open(
            units_path, "w", encoding="utf-8"
        ) as uh:
            writer = csv.writer(fh)
            writer.writerow(["ID", "duration", "wav", "spk_id", "wrd"])
            for utt, dur, wav, spk, text, moras in split_rows:
                writer.writerow([utt, dur, wav, spk, text])
                uh.write(f"{utt}\t{' '.join(moras)}\n")
        hours = sum(r[1] for r in split_rows) / 3600.0
        print(f"{split}: {len(split_rows)} utts, {hours:.1f} h")
    print("dropped:", dropped)
    # The duration and Japanese-script filters are part of the corpus
    # DEFINITION (measured: ~18% of subtitle lines contain digits/latin with
    # no unidic kana reading); they do not count as failures. Hard-fail only
    # on unexpected losses: missing audio or residual mora-conversion gaps.
    unexpected = dropped["missing_audio"] + dropped["mora"]
    if unexpected > 0.01 * len(entries):
        print(
            f"ERROR: {unexpected}/{len(entries)} utterances lost to "
            "missing audio or mora-conversion gaps (>1%) — inspect before "
            "training."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
