"""Increase a CTC boundary rate by splitting only long predicted segments.

The transformation is audio/transcript independent: it preserves every input
boundary and adds one midpoint boundary to each segment whose duration reaches a
fixed number of encoder frames.  The threshold must be selected by frequency,
before downstream WER is observed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch


def refine_boundary(boundary: torch.Tensor, min_segment_frames: int) -> torch.Tensor:
    """Return a copy with one midpoint added to each sufficiently long segment."""

    boundary = torch.as_tensor(boundary).flatten().to(dtype=torch.uint8, device="cpu")
    if boundary.numel() == 0 or int(boundary[0]) != 1:
        raise ValueError("Boundary tensor must be nonempty and start with one")
    if not torch.all((boundary == 0) | (boundary == 1)):
        raise ValueError("Boundary tensor must contain only zero and one")
    if min_segment_frames < 2:
        raise ValueError("min_segment_frames must be at least two")

    refined = boundary.clone()
    starts = torch.nonzero(boundary, as_tuple=False).flatten().tolist()
    ends = starts[1:] + [boundary.numel()]
    for start, end in zip(starts, ends):
        length = end - start
        if length >= min_segment_frames:
            refined[start + length // 2] = 1
    return refined


def atomic_torch_save(tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    torch.save(tensor, temporary)
    os.replace(temporary, path)


def source_seconds(stats_dir: Path) -> dict[str, float]:
    totals: dict[str, float] = {}
    paths = sorted(stats_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No source statistics under {stats_dir}")
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for split, values in payload["splits"].items():
            totals[split] = totals.get(split, 0.0) + float(values["audio_seconds"])
    return totals


def run(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    audio_root = Path(args.audio_root)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    seconds = source_seconds(input_dir / "_stats")
    summary = {
        "method": "phone_ctc_nonblank_runs_plus_long_segment_midpoints",
        "transcript_used_for_refinement": False,
        "min_segment_frames": args.min_segment_frames,
        "frame_hop_seconds": 0.02,
        "splits": {},
    }

    for split in args.splits:
        audio_paths = sorted((audio_root / split).glob("*/*/*.flac"))
        if not audio_paths:
            raise FileNotFoundError(f"No audio files found for split {split}")
        base_segments = refined_segments = frames = 0
        for index, audio_path in enumerate(audio_paths, start=1):
            source = input_dir / f"{audio_path.stem}.pt"
            if not source.is_file():
                raise FileNotFoundError(source)
            boundary = torch.load(source, map_location="cpu", weights_only=True)
            refined = refine_boundary(boundary, args.min_segment_frames)
            atomic_torch_save(refined, output_dir / source.name)
            base_segments += int(torch.as_tensor(boundary).sum())
            refined_segments += int(refined.sum())
            frames += refined.numel()
            if index % args.log_interval == 0 or index == len(audio_paths):
                print(f"split={split} utterances={index}/{len(audio_paths)}", flush=True)

        duration = seconds[split]
        summary["splits"][split] = {
            "utterances": len(audio_paths),
            "audio_seconds": duration,
            "frames": frames,
            "base_segments": base_segments,
            "refined_segments": refined_segments,
            "added_segments": refined_segments - base_segments,
            "base_rate_hz": base_segments / duration,
            "refined_rate_hz": refined_segments / duration,
        }

    stats_dir = output_dir / "_stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    stats_path = stats_dir / "long_segment_split_summary.json"
    stats_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--audio_root", required=True)
    parser.add_argument("--splits", nargs="+", required=True)
    parser.add_argument("--min_segment_frames", type=int, default=8)
    parser.add_argument("--log_interval", type=int, default=5000)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
