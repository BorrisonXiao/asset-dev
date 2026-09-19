#!/usr/bin/env python3
"""MFA TextGrid phones tier -> per-utterance phone boundary targets.

Follows the LibriSpeech conventions of prepare_boundary_targets.py
(origin/forced_alignment): one ``{out_dir}/{utt_id}.pt`` per utterance holding
a 1-D ``torch.uint8`` tensor with one label per encoder frame; a boundary at
the start of EVERY phones-tier interval (silence gets its own segment);
silence labels are ``""``, ``"sil"``, ``"sp"`` (``"spn"`` is speech); a unit
starting at time t maps to frame ``int(t * 50 + 1e-6)`` clamped to
``[0, T-1]``; index 0 always 1; same-frame collapses counted, not fatal.

The encoder frame count comes from the audio sample count with the shared
conv formula (identical for WavLM-Large and HuBERT-Large).

Usage:
    python xling_textgrid_boundaries.py --csv m/train.csv m/dev.csv \
        --textgrid_dir aligned/ --out_dir boundaries/phone \
        --stats_out boundaries/phone_stats.json
"""

import argparse
import csv
import json
import os
import re

import soundfile as sf
import torch

FRAME_HZ = 50.0
SILENCE = {"", "sil", "sp"}
CONV = [(10, 5), (3, 2), (3, 2), (3, 2), (3, 2), (2, 2), (2, 2)]

_INTERVAL = re.compile(
    r"intervals \[\d+\]:\s*xmin = ([\d.eE+-]+)\s*xmax = ([\d.eE+-]+)\s*"
    r'text = "((?:[^"]|"")*)"',
    re.S,
)
_ITEM = re.compile(r'item \[\d+\]:\s*class = "IntervalTier"\s*name = "([^"]+)"')


def num_encoder_frames(num_samples):
    L = num_samples
    for k, s in CONV:
        L = (L - k) // s + 1
    return L


def phones_tier_intervals(path):
    """(xmin, text) for each phones-tier interval, in order."""
    text = open(path, encoding="utf-8").read()
    items = list(_ITEM.finditer(text))
    phones_start = phones_end = None
    for i, m in enumerate(items):
        if m.group(1) == "phones":
            phones_start = m.end()
            phones_end = items[i + 1].start() if i + 1 < len(items) else len(text)
    if phones_start is None:
        raise ValueError(f"no phones tier in {path}")
    chunk = text[phones_start:phones_end]
    return [
        (float(m.group(1)), m.group(3).replace('""', '"').strip())
        for m in _INTERVAL.finditer(chunk)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", nargs="+", required=True)
    parser.add_argument("--textgrid_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--stats_out")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    stats = {
        "n": 0,
        "written": 0,
        "missing_textgrid": 0,
        "collapses": 0,
        "rho_sum": 0.0,
    }
    for path in args.csv:
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                stats["n"] += 1
                utt = row["ID"]
                tg = os.path.join(args.textgrid_dir, utt + ".TextGrid")
                if not os.path.exists(tg):
                    stats["missing_textgrid"] += 1
                    continue
                info = sf.info(row["wav"])
                T = num_encoder_frames(info.frames)
                boundary = torch.zeros(T, dtype=torch.uint8)
                boundary[0] = 1
                for xmin, _label in phones_tier_intervals(tg):
                    frame = min(int(xmin * FRAME_HZ + 1e-6), T - 1)
                    if boundary[frame]:
                        stats["collapses"] += 1
                    boundary[frame] = 1
                torch.save(boundary, os.path.join(args.out_dir, utt + ".pt"))
                stats["written"] += 1
                stats["rho_sum"] += float(boundary.sum()) / T
    stats["mean_rho"] = stats["rho_sum"] / max(stats["written"], 1)
    stats["mean_hz"] = FRAME_HZ * stats["mean_rho"]
    del stats["rho_sum"]
    print(json.dumps(stats, indent=2))
    if args.stats_out:
        os.makedirs(os.path.dirname(args.stats_out), exist_ok=True)
        with open(args.stats_out, "w") as fh:
            json.dump(stats, fh, indent=2)


if __name__ == "__main__":
    main()
