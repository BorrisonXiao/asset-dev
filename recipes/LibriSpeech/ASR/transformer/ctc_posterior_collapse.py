"""Generate transcript-free CTC posterior-collapse boundaries.

This is the deployable counterpart to ``ctc_boundary_align.py``.  It never
opens a transcript or target file.  A frozen pretrained CTC acoustic model is
run on audio, its framewise argmax path is CTC-collapsed, and the resulting
nonblank symbol runs define segments for the downstream WavLM features.

The boundary convention is shared with ``segment_pooling.py``: one uint8 value
per 20 ms encoder frame, with ``1`` opening a segment and frame zero always set
to one.  Blank frames are not emitted as separate tokens.  Leading blank frames
are merged into the first symbol segment and later blank runs are merged into
the preceding symbol segment.  Thus the output has one segment per nonblank run
in the ordinary CTC collapse (repeated symbols separated by blank remain two
runs).

The CTC model itself is already trained (torchaudio's
``WAV2VEC2_ASR_BASE_960H``); this script performs inference only.  Reference
transcripts are used neither for boundary generation nor at deployment.
"""

import argparse
import glob
import json
import os
import time

import torch
import torchaudio

from ctc_boundary_align import num_encoder_frames


SAMPLE_RATE = 16000
BLANK_ID = 0


def collect_audio_files(audio_root, splits):
    """Collect LibriSpeech audio without reading transcript files."""

    utterances = []
    for split in splits:
        split_dir = os.path.join(audio_root, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        pattern = os.path.join(split_dir, "*", "*", "*.flac")
        paths = sorted(glob.glob(pattern))
        if not paths:
            raise FileNotFoundError(f"No FLAC files found for split {split}")
        utterances.extend((split, path) for path in paths)
    return utterances


def nonblank_run_boundaries(argmax_labels, blank_id=BLANK_ID):
    """Return one boundary per nonblank run in a CTC argmax path.

    Blank runs are merged rather than emitted.  Frame zero opens the first
    segment, so the first nonblank run does not add a second boundary after
    leading silence.  Later nonblank run onsets open new segments, including a
    repeated label separated by one or more blanks.
    """

    labels = torch.as_tensor(argmax_labels, dtype=torch.long).flatten()
    if labels.numel() == 0:
        raise ValueError("CTC argmax path must contain at least one frame")

    nonblank = labels.ne(int(blank_id))
    previous_is_blank = torch.ones_like(nonblank)
    previous_is_blank[1:] = labels[:-1].eq(int(blank_id))
    label_changed = torch.ones_like(nonblank)
    label_changed[1:] = labels[1:].ne(labels[:-1])
    run_start = nonblank & (previous_is_blank | label_changed)
    starts = torch.nonzero(run_start, as_tuple=False).flatten()

    boundary = torch.zeros(labels.numel(), dtype=torch.uint8)
    boundary[0] = 1
    if starts.numel() > 1:
        boundary[starts[1:]] = 1
    return boundary


def _write_stats(path, stats):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp_path = f"{path}.tmp.{os.getpid()}"
    with open(temp_path, "w", encoding="utf-8") as stream:
        json.dump(stats, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temp_path, path)


@torch.inference_mode()
def generate_boundaries(
    audio_root,
    out_root,
    splits,
    device="cuda",
    shard=0,
    num_shards=1,
):
    """Run audio-only CTC inference and save sharded boundary tensors."""

    all_utterances = collect_audio_files(audio_root, splits)
    utterances = all_utterances[shard::num_shards]
    if not utterances:
        raise ValueError(
            f"Shard {shard}/{num_shards} has no utterances "
            f"(total={len(all_utterances)})"
        )

    os.makedirs(out_root, exist_ok=True)
    model = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H.get_model()
    model = model.to(device).eval()

    started = time.perf_counter()
    total_audio_seconds = 0.0
    total_frames = 0
    total_segments = 0
    per_split = {}

    for index, (split, audio_path) in enumerate(utterances, start=1):
        utterance_id = os.path.splitext(os.path.basename(audio_path))[0]
        output_path = os.path.join(out_root, f"{utterance_id}.pt")

        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != SAMPLE_RATE:
            raise ValueError(
                f"{audio_path}: sample rate {sample_rate} != {SAMPLE_RATE}"
            )
        if waveform.size(0) != 1:
            raise ValueError(f"{audio_path}: expected mono audio")

        expected_frames = num_encoder_frames(waveform.size(1))
        emissions, _ = model(waveform.to(device))
        if emissions.size(0) != 1 or emissions.size(1) != expected_frames:
            raise RuntimeError(
                f"{utterance_id}: CTC frames={emissions.size(1)} but "
                f"WavLM-compatible frame count={expected_frames}"
            )
        argmax_labels = emissions[0].argmax(dim=-1).cpu()
        boundary = nonblank_run_boundaries(argmax_labels)
        if boundary.numel() != expected_frames or boundary[0].item() != 1:
            raise RuntimeError(f"{utterance_id}: invalid boundary tensor")
        torch.save(boundary, output_path)

        audio_seconds = waveform.size(1) / SAMPLE_RATE
        segments = int(boundary.sum().item())
        total_audio_seconds += audio_seconds
        total_frames += expected_frames
        total_segments += segments
        split_stats = per_split.setdefault(
            split,
            {"utterances": 0, "audio_seconds": 0.0, "frames": 0, "segments": 0},
        )
        split_stats["utterances"] += 1
        split_stats["audio_seconds"] += audio_seconds
        split_stats["frames"] += expected_frames
        split_stats["segments"] += segments

        if index % 250 == 0 or index == len(utterances):
            elapsed = time.perf_counter() - started
            print(
                f"shard={shard}/{num_shards} processed={index}/{len(utterances)} "
                f"audio_hours={total_audio_seconds / 3600:.2f} "
                f"rate_hz={total_segments / total_audio_seconds:.2f} "
                f"elapsed_min={elapsed / 60:.1f}",
                flush=True,
            )

    for values in per_split.values():
        values["rate_hz"] = values["segments"] / values["audio_seconds"]
    stats = {
        "method": "wav2vec2_asr_base_960h_argmax_nonblank_runs",
        "transcript_used": False,
        "blank_rule": "merge blanks; one segment per nonblank CTC run",
        "shard": shard,
        "num_shards": num_shards,
        "utterances": len(utterances),
        "audio_seconds": total_audio_seconds,
        "frames": total_frames,
        "segments": total_segments,
        "rate_hz": total_segments / total_audio_seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "splits": per_split,
    }
    stats_path = os.path.join(out_root, "_stats", f"shard_{shard:02d}.json")
    _write_stats(stats_path, stats)
    print(json.dumps(stats, indent=2, sort_keys=True), flush=True)
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_root", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--splits", nargs="+", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    args = parser.parse_args()
    if args.num_shards <= 0:
        parser.error("--num_shards must be positive")
    if not 0 <= args.shard < args.num_shards:
        parser.error("--shard must satisfy 0 <= shard < num_shards")
    generate_boundaries(
        audio_root=args.audio_root,
        out_root=args.out_root,
        splits=args.splits,
        device=args.device,
        shard=args.shard,
        num_shards=args.num_shards,
    )


if __name__ == "__main__":
    main()
