#!/usr/bin/env python3
"""Measure oracle boundary kept ratios against fixed-rate pooling.

The corpus ratio is total pooled segments divided by total encoder frames. A
fixed-k row uses the exact segment count produced by ``fixed_rate_boundary_targets``
(one implicit first segment plus boundaries at frames k, 2k, ...).
"""

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch


def load_count(item):
    level_dir, utt_id = item
    target = torch.load(level_dir / f"{utt_id}.pt", map_location="cpu", weights_only=True)
    target = target.view(-1)
    return int(target.sum().item()), int(target.numel())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--boundary-root", type=Path, required=True)
    parser.add_argument(
        "--levels", nargs="+", default=("char", "phone", "syllable", "word", "word_ctc")
    )
    parser.add_argument("--fixed-k", type=int, default=5)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    ids = []
    for manifest in args.manifests:
        with manifest.open(newline="", encoding="utf-8") as stream:
            ids.extend(row["ID"] for row in csv.DictReader(stream))
    if args.limit:
        ids = ids[: args.limit]

    rows = []
    frame_lengths = None
    for level in args.levels:
        level_dir = args.boundary_root / level
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            counts = list(pool.map(load_count, ((level_dir, utt_id) for utt_id in ids)))
        segments = sum(count for count, _ in counts)
        frames = sum(length for _, length in counts)
        utt_mean = sum(count / length for count, length in counts) / len(counts)
        rows.append((level, segments / frames, utt_mean, segments, frames))
        if frame_lengths is None:
            frame_lengths = [length for _, length in counts]
        elif frame_lengths != [length for _, length in counts]:
            raise RuntimeError(f"Frame lengths differ for boundary level {level}")

    fixed_segments = sum((length + args.fixed_k - 1) // args.fixed_k
                         for length in frame_lengths)
    total_frames = sum(frame_lengths)
    target = fixed_segments / total_frames

    print(f"utterances={len(ids)} frames={total_frames} fixed_k={args.fixed_k} "
          f"fixed_corpus_rho={target:.6f}")
    print("level\tcorpus_rho\tutt_mean_rho\tabs_delta_from_fixed")
    for level, corpus_rho, utt_mean, _, _ in sorted(rows, key=lambda row: abs(row[1] - target)):
        print(f"{level}\t{corpus_rho:.6f}\t{utt_mean:.6f}\t{abs(corpus_rho-target):.6f}")


if __name__ == "__main__":
    main()
