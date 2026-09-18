"""Generate transcript-free BPE-CTC state-run boundaries.

The acoustic model is the released Icefall LibriSpeech Conformer-CTC model
with a 500-piece SentencePiece vocabulary.  It is frozen: this script only
runs audio inference and never reads reference transcripts.

The checkpoint emits CTC posteriors at roughly 25 Hz, while the downstream
WavLM encoder emits roughly 50 Hz.  State-run onsets are therefore mapped by
normalized frame time to the exact WavLM frame count.  ``state_runs`` opens a
segment whenever the CTC argmax state changes, including transitions into and
out of blank.  This preserves acoustically useful gap intervals and gives a
rate close to the learned segmenter.  ``nonblank_runs`` is also available as
the ordinary BPE-collapse diagnostic, but is expected to be substantially
coarser.
"""

import argparse
import glob
import json
import os
import time

import torch
import torchaudio
from torchaudio.compliance.kaldi import fbank

from ctc_boundary_align import num_encoder_frames


SAMPLE_RATE = 16000
BLANK_ID = 0
NUM_MEL_BINS = 80
BOUNDARY_RULES = ("state_runs", "nonblank_runs")


def collect_audio_files(audio_root, splits):
    """Collect LibriSpeech audio paths without opening transcript files."""

    utterances = []
    for split in splits:
        split_dir = os.path.join(audio_root, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        paths = sorted(glob.glob(os.path.join(split_dir, "*", "*", "*.flac")))
        if not paths:
            raise FileNotFoundError(f"No FLAC files found for split {split}")
        utterances.extend((split, path) for path in paths)
    return utterances


def state_run_starts(argmax_labels):
    """Return starts of every constant-label run, including CTC blank."""

    labels = torch.as_tensor(argmax_labels, dtype=torch.long).flatten()
    if labels.numel() == 0:
        raise ValueError("CTC argmax path must contain at least one frame")
    starts = torch.ones(labels.numel(), dtype=torch.bool)
    starts[1:] = labels[1:].ne(labels[:-1])
    return torch.nonzero(starts, as_tuple=False).flatten()


def nonblank_run_starts(argmax_labels, blank_id=BLANK_ID):
    """Return ordinary CTC nonblank-run starts under the project convention.

    Frame zero always opens the first downstream segment.  Consequently, the
    first nonblank run after leading blank is represented by frame zero and is
    not added as a second boundary.
    """

    labels = torch.as_tensor(argmax_labels, dtype=torch.long).flatten()
    if labels.numel() == 0:
        raise ValueError("CTC argmax path must contain at least one frame")
    run_starts = state_run_starts(labels)
    nonblank = run_starts[labels[run_starts].ne(int(blank_id))]
    if nonblank.numel() <= 1:
        return torch.zeros(1, dtype=torch.long)
    return torch.cat((torch.zeros(1, dtype=torch.long), nonblank[1:]))


def map_starts_to_target(starts, source_frames, target_frames):
    """Map monotonic run onsets between frame grids by normalized time."""

    starts = torch.as_tensor(starts, dtype=torch.long).flatten()
    if source_frames <= 0 or target_frames <= 0:
        raise ValueError("source_frames and target_frames must be positive")
    if starts.numel() == 0 or starts[0].item() != 0:
        raise ValueError("run starts must begin at frame zero")
    if starts.min().item() < 0 or starts.max().item() >= source_frames:
        raise ValueError("run start is outside the source frame grid")
    mapped = torch.div(
        starts * int(target_frames), int(source_frames), rounding_mode="floor"
    )
    return torch.unique_consecutive(mapped.clamp(max=target_frames - 1))


def boundary_from_argmax(
    argmax_labels,
    target_frames,
    rule="state_runs",
    blank_id=BLANK_ID,
):
    """Convert one BPE-CTC argmax path to WavLM-grid boundary labels."""

    labels = torch.as_tensor(argmax_labels, dtype=torch.long).flatten()
    if rule == "state_runs":
        starts = state_run_starts(labels)
    elif rule == "nonblank_runs":
        starts = nonblank_run_starts(labels, blank_id=blank_id)
    else:
        raise ValueError(f"Unknown boundary rule: {rule}")
    mapped = map_starts_to_target(starts, labels.numel(), target_frames)
    boundary = torch.zeros(target_frames, dtype=torch.uint8)
    boundary[mapped] = 1
    boundary[0] = 1
    return boundary


def extract_fbank(waveform):
    """Extract the 80-bin, 10 ms Kaldi-style features used by Icefall."""

    return fbank(
        waveform,
        num_mel_bins=NUM_MEL_BINS,
        sample_frequency=SAMPLE_RATE,
        dither=0.0,
        snip_edges=False,
    )


def _atomic_torch_save(tensor, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp_path = f"{path}.tmp.{os.getpid()}"
    torch.save(tensor, temp_path)
    os.replace(temp_path, path)


def _write_stats(path, stats):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp_path = f"{path}.tmp.{os.getpid()}"
    with open(temp_path, "w", encoding="utf-8") as stream:
        json.dump(stats, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temp_path, path)


def _new_totals():
    return {
        "utterances": 0,
        "audio_seconds": 0.0,
        "ctc_frames": 0,
        "wavlm_frames": 0,
        "state_run_segments": 0,
        "nonblank_run_segments": 0,
        "selected_segments": 0,
    }


def _add_rates(values):
    seconds = values["audio_seconds"]
    values["state_run_rate_hz"] = values["state_run_segments"] / seconds
    values["nonblank_run_rate_hz"] = values["nonblank_run_segments"] / seconds
    values["selected_rate_hz"] = values["selected_segments"] / seconds


@torch.inference_mode()
def generate_boundaries(
    audio_root,
    out_root,
    model_path,
    splits,
    boundary_rule="state_runs",
    device="cuda",
    shard=0,
    num_shards=1,
    max_utterances=None,
):
    """Run frozen BPE-CTC inference and save sharded boundary tensors."""

    if boundary_rule not in BOUNDARY_RULES:
        raise ValueError(f"boundary_rule must be one of {BOUNDARY_RULES}")
    all_utterances = collect_audio_files(audio_root, splits)
    utterances = all_utterances[shard::num_shards]
    if max_utterances is not None:
        utterances = utterances[:max_utterances]
    if not utterances:
        raise ValueError(
            f"Shard {shard}/{num_shards} has no utterances "
            f"(total={len(all_utterances)})"
        )

    os.makedirs(out_root, exist_ok=True)
    model = torch.jit.load(model_path, map_location=device).to(device).eval()
    started = time.perf_counter()
    totals = _new_totals()
    per_split = {}

    for index, (split, audio_path) in enumerate(utterances, start=1):
        utterance_id = os.path.splitext(os.path.basename(audio_path))[0]
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != SAMPLE_RATE:
            raise ValueError(
                f"{audio_path}: sample rate {sample_rate} != {SAMPLE_RATE}"
            )
        if waveform.size(0) != 1:
            raise ValueError(f"{audio_path}: expected mono audio")

        features = extract_fbank(waveform).unsqueeze(0).to(device)
        log_probs, _, _ = model(features, None)
        if log_probs.size(0) != 1 or log_probs.size(2) != 500:
            raise RuntimeError(
                f"{utterance_id}: unexpected CTC output {tuple(log_probs.shape)}"
            )
        labels = log_probs[0].argmax(dim=-1).cpu()
        target_frames = num_encoder_frames(waveform.size(1))
        state_segments = int(state_run_starts(labels).numel())
        nonblank_segments = int(nonblank_run_starts(labels).numel())
        boundary = boundary_from_argmax(
            labels,
            target_frames=target_frames,
            rule=boundary_rule,
        )
        selected_segments = int(boundary.sum().item())
        expected_selected = (
            state_segments if boundary_rule == "state_runs" else nonblank_segments
        )
        if selected_segments != expected_selected:
            raise RuntimeError(
                f"{utterance_id}: mapped {selected_segments} of "
                f"{expected_selected} state runs"
            )
        _atomic_torch_save(
            boundary, os.path.join(out_root, f"{utterance_id}.pt")
        )

        audio_seconds = waveform.size(1) / SAMPLE_RATE
        for values in (totals, per_split.setdefault(split, _new_totals())):
            values["utterances"] += 1
            values["audio_seconds"] += audio_seconds
            values["ctc_frames"] += labels.numel()
            values["wavlm_frames"] += target_frames
            values["state_run_segments"] += state_segments
            values["nonblank_run_segments"] += nonblank_segments
            values["selected_segments"] += selected_segments

        if index % 100 == 0 or index == len(utterances):
            elapsed = time.perf_counter() - started
            print(
                f"shard={shard}/{num_shards} processed={index}/{len(utterances)} "
                f"audio_hours={totals['audio_seconds'] / 3600:.2f} "
                f"state_hz={totals['state_run_segments'] / totals['audio_seconds']:.2f} "
                f"nonblank_hz={totals['nonblank_run_segments'] / totals['audio_seconds']:.2f} "
                f"elapsed_min={elapsed / 60:.1f}",
                flush=True,
            )

    _add_rates(totals)
    for values in per_split.values():
        _add_rates(values)
    stats = {
        "method": "icefall_librispeech_conformer_ctc_bpe500_argmax_runs",
        "model_path": os.path.abspath(model_path),
        "vocabulary": "SentencePiece BPE-500",
        "blank_id": BLANK_ID,
        "transcript_used": False,
        "boundary_rule": boundary_rule,
        "boundary_rule_description": (
            "one segment per argmax state run, including CTC blank"
            if boundary_rule == "state_runs"
            else "one segment per nonblank BPE run; blank runs merged"
        ),
        "frame_mapping": "floor(source_index * wavlm_frames / ctc_frames)",
        "shard": shard,
        "num_shards": num_shards,
        "max_utterances": max_utterances,
        **totals,
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
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--splits", nargs="+", required=True)
    parser.add_argument("--boundary_rule", choices=BOUNDARY_RULES, default="state_runs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--max_utterances", type=int)
    args = parser.parse_args()
    if args.num_shards <= 0:
        parser.error("--num_shards must be positive")
    if not 0 <= args.shard < args.num_shards:
        parser.error("--shard must satisfy 0 <= shard < num_shards")
    if args.max_utterances is not None and args.max_utterances <= 0:
        parser.error("--max_utterances must be positive")
    generate_boundaries(
        audio_root=args.audio_root,
        out_root=args.out_root,
        model_path=args.model_path,
        splits=args.splits,
        boundary_rule=args.boundary_rule,
        device=args.device,
        shard=args.shard,
        num_shards=args.num_shards,
        max_utterances=args.max_utterances,
    )


if __name__ == "__main__":
    main()
