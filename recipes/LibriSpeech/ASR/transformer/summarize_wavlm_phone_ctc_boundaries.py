"""Validate and aggregate sharded WavLM phone-CTC boundary generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED_SPLITS = {
    "train-clean-100": 28_539,
    "train-clean-360": 104_014,
    "train-other-500": 148_688,
    "dev-clean": 2_703,
    "dev-other": 2_864,
    "test-clean": 2_620,
    "test-other": 2_939,
}


def run(args: argparse.Namespace) -> None:
    boundary_dir = Path(args.boundary_dir)
    stats_dir = boundary_dir / "_stats"
    paths = sorted(stats_dir.glob("rank_00_shard_*.json"))
    if len(paths) != args.num_shards:
        raise RuntimeError(
            f"Expected {args.num_shards} shard summaries, found {len(paths)}"
        )

    aggregate = {
        split: {
            "utterances": 0,
            "audio_seconds": 0.0,
            "frames": 0,
            "base_segments": 0,
            "segments": 0,
        }
        for split in EXPECTED_SPLITS
    }
    seen_shards = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["num_shards"] != args.num_shards:
            raise RuntimeError(f"Wrong num_shards in {path}")
        if payload["long_segment_split_min_frames"] != args.split_frames:
            raise RuntimeError(f"Wrong split threshold in {path}")
        if payload["transcript_used_for_boundary_inference"]:
            raise RuntimeError(f"Transcript-dependent inference recorded in {path}")
        shard = int(payload["shard"])
        if shard in seen_shards:
            raise RuntimeError(f"Duplicate shard {shard}")
        seen_shards.add(shard)
        for split, values in payload["splits"].items():
            if split not in aggregate:
                raise RuntimeError(f"Unexpected split {split} in {path}")
            for key in aggregate[split]:
                aggregate[split][key] += values[key]

    if seen_shards != set(range(args.num_shards)):
        raise RuntimeError(f"Incomplete shard indices: {sorted(seen_shards)}")
    for split, expected in EXPECTED_SPLITS.items():
        observed = aggregate[split]["utterances"]
        if observed != expected:
            raise RuntimeError(f"{split}: expected {expected} utterances, found {observed}")
        values = aggregate[split]
        values["added_segments"] = values["segments"] - values["base_segments"]
        values["base_boundary_rate_hz"] = (
            values["base_segments"] / values["audio_seconds"]
        )
        values["boundary_rate_hz"] = values["segments"] / values["audio_seconds"]

    expected_total = sum(EXPECTED_SPLITS.values())
    file_count = sum(1 for _ in boundary_dir.glob("*.pt"))
    if file_count != expected_total:
        raise RuntimeError(
            f"Expected {expected_total} boundary tensors, found {file_count}"
        )

    summary = {
        "method": "frozen_wavlm_large_phone_ctc_nonblank_runs_plus_long_segment_midpoints",
        "transcript_used_for_boundary_inference": False,
        "long_segment_split_min_frames": args.split_frames,
        "frame_hop_seconds": 0.02,
        "num_shards": args.num_shards,
        "boundary_tensor_count": file_count,
        "splits": aggregate,
    }
    output = stats_dir / "combined_summary.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(summary, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boundary_dir", required=True)
    parser.add_argument("--num_shards", type=int, required=True)
    parser.add_argument("--split_frames", type=int, default=8)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
