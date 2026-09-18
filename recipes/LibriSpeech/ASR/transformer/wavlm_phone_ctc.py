"""Train and apply a compact phone-CTC head on frozen WavLM-Large.

Training uses ordered phone sequences derived from the MFA phone tier, but no
phone timing information.  Boundary inference is audio-only: ordinary CTC
nonblank-run onsets are converted to the project's 50 Hz WavLM boundary grid.

The deliberately small protocol is intended as a time-limited baseline:
WavLM-Large remains frozen, every epoch checkpoint is retained, dev-clean phone
error rate (PER) ranks checkpoints, and inference can address any retained rank.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import WavLMModel

from ctc_boundary_align import num_encoder_frames
from refine_ctc_boundaries import refine_boundary


SAMPLE_RATE = 16000
BLANK_ID = 0
PHONE_TIER_MARKER = 'name = "phones"'


def atomic_json(payload, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp.{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def atomic_torch_save(payload, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp.{os.getpid()}"
    torch.save(payload, temporary)
    os.replace(temporary, path)


def normalize_phone(phone):
    """Map stress-marked ARPAbet to a compact phone vocabulary."""

    return re.sub(r"\d+$", "", phone.strip().upper())


def phone_sequence(textgrid_path):
    """Read only the ordered labels from the phones tier of a TextGrid."""

    with open(textgrid_path, encoding="utf-8", errors="replace") as stream:
        contents = stream.read()
    if PHONE_TIER_MARKER not in contents:
        raise ValueError(f"No phones tier in {textgrid_path}")
    phone_tier = contents.split(PHONE_TIER_MARKER, 1)[1]
    phones = [
        normalize_phone(value)
        for value in re.findall(r'text\s*=\s*"([^"]*)"', phone_tier)
        if value.strip()
    ]
    if not phones:
        raise ValueError(f"Empty phones tier in {textgrid_path}")
    return phones


@dataclass(frozen=True)
class Utterance:
    uid: str
    split: str
    audio_path: str
    samples: int
    phones: tuple[str, ...] | None = None

    @property
    def duration(self):
        return self.samples / SAMPLE_RATE


def collect_utterances(audio_root, splits, alignment_root=None, shard=0, num_shards=1):
    if num_shards <= 0 or not 0 <= shard < num_shards:
        raise ValueError("Require num_shards > 0 and 0 <= shard < num_shards")
    rows = []
    utterance_index = 0
    for split in splits:
        pattern = os.path.join(audio_root, split, "*", "*", "*.flac")
        paths = sorted(glob.glob(pattern))
        if not paths:
            raise FileNotFoundError(f"No audio found for split {split}: {pattern}")
        for audio_path in paths:
            selected = utterance_index % num_shards == shard
            utterance_index += 1
            if not selected:
                continue
            uid = os.path.splitext(os.path.basename(audio_path))[0]
            info = sf.info(audio_path)
            if info.samplerate != SAMPLE_RATE or info.channels != 1:
                raise ValueError(f"Expected mono 16-kHz audio: {audio_path}")
            phones = None
            if alignment_root is not None:
                relative = os.path.relpath(audio_path, os.path.join(audio_root, split))
                textgrid_path = os.path.join(
                    alignment_root,
                    split,
                    os.path.splitext(relative)[0] + ".TextGrid",
                )
                if not os.path.isfile(textgrid_path):
                    raise FileNotFoundError(textgrid_path)
                phones = tuple(phone_sequence(textgrid_path))
            rows.append(Utterance(uid, split, audio_path, info.frames, phones))
    return rows


class AudioDataset(Dataset):
    def __init__(self, rows, phone_to_id=None):
        self.rows = list(rows)
        self.phone_to_id = phone_to_id

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        waveform, sample_rate = sf.read(row.audio_path, dtype="float32")
        if sample_rate != SAMPLE_RATE or waveform.ndim != 1:
            raise ValueError(f"Expected mono 16-kHz audio: {row.audio_path}")
        target = None
        if row.phones is not None:
            target = torch.tensor(
                [self.phone_to_id[phone] for phone in row.phones],
                dtype=torch.long,
            )
        return row, torch.from_numpy(np.asarray(waveform)), target


class DurationBatchSampler(Sampler[list[int]]):
    """Duration-sorted batches capped by padded audio seconds."""

    def __init__(self, rows, max_padded_seconds, max_batch_size, shuffle, seed):
        self.rows = list(rows)
        self.max_padded_seconds = float(max_padded_seconds)
        self.max_batch_size = int(max_batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        ordered = sorted(range(len(rows)), key=lambda index: rows[index].samples)
        self.batches = []
        current = []
        current_max = 0.0
        for index in ordered:
            duration = rows[index].duration
            next_max = max(current_max, duration)
            if current and (
                next_max * (len(current) + 1) > self.max_padded_seconds
                or len(current) >= self.max_batch_size
            ):
                self.batches.append(current)
                current = []
                current_max = 0.0
            current.append(index)
            current_max = max(current_max, duration)
        if current:
            self.batches.append(current)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        batches = list(self.batches)
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(batches)
        yield from batches

    def __len__(self):
        return len(self.batches)


def collate_audio(examples):
    rows, waveforms, targets = zip(*examples)
    lengths = torch.tensor([waveform.numel() for waveform in waveforms], dtype=torch.long)
    padded = torch.zeros(len(waveforms), int(lengths.max()), dtype=torch.float32)
    for index, waveform in enumerate(waveforms):
        # Match the normalization used by the SpeechBrain WavLM wrapper.
        waveform = F.layer_norm(waveform, waveform.shape)
        padded[index, : waveform.numel()] = waveform
    attention_mask = torch.arange(padded.size(1)).unsqueeze(0) < lengths.unsqueeze(1)
    return rows, padded, attention_mask.long(), lengths, targets


class PhoneCTCHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, vocab_size, dropout):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size),
        )

    def forward(self, features):
        return self.network(features)


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, ref_token in enumerate(reference, start=1):
        current = [i]
        for j, hyp_token in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (ref_token != hyp_token),
                )
            )
        previous = current
    return previous[-1]


def collapse_nonblank(labels, blank_id=BLANK_ID):
    output = []
    previous = blank_id
    for label in labels:
        label = int(label)
        if label != blank_id and label != previous:
            output.append(label)
        previous = label
    return output


def nonblank_run_boundaries(labels, blank_id=BLANK_ID):
    labels = torch.as_tensor(labels, dtype=torch.long).flatten()
    if labels.numel() == 0:
        raise ValueError("CTC path must contain at least one frame")
    previous = torch.full_like(labels, int(blank_id))
    previous[1:] = labels[:-1]
    starts = torch.nonzero(
        labels.ne(int(blank_id)) & labels.ne(previous), as_tuple=False
    ).flatten()
    boundary = torch.zeros(labels.numel(), dtype=torch.uint8)
    boundary[0] = 1
    # Frame zero represents the first nonblank run after any leading silence.
    if starts.numel() > 1:
        boundary[starts[1:]] = 1
    return boundary


def load_wavlm(source, cache_dir, device):
    model = WavLMModel.from_pretrained(
        source,
        cache_dir=cache_dir,
        local_files_only=True,
    )
    model.requires_grad_(False)
    model.eval().to(device)
    return model


def output_lengths(model, sample_lengths):
    return model._get_feat_extract_output_lengths(sample_lengths).long()


def forward_features(model, waveforms, attention_mask):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return model(
            waveforms,
            attention_mask=attention_mask,
        ).last_hidden_state


@torch.inference_mode()
def validate(model, head, loader, device):
    head.eval()
    total_edits = 0
    total_phones = 0
    total_segments = 0
    total_seconds = 0.0
    total_loss = 0.0
    total_examples = 0
    for rows, waveforms, attention_mask, sample_lengths, targets in loader:
        waveforms = waveforms.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        sample_lengths = sample_lengths.to(device)
        features = forward_features(model, waveforms, attention_mask)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = head(features)
        frame_lengths = output_lengths(model, sample_lengths)
        target_lengths = torch.tensor([target.numel() for target in targets], device=device)
        concatenated = torch.cat(targets).to(device)
        log_probs = logits.float().log_softmax(dim=-1)
        loss = F.ctc_loss(
            log_probs.transpose(0, 1),
            concatenated,
            frame_lengths,
            target_lengths,
            blank=BLANK_ID,
            zero_infinity=True,
        )
        total_loss += float(loss) * len(rows)
        total_examples += len(rows)
        argmax = logits.argmax(dim=-1).cpu()
        for index, (row, target) in enumerate(zip(rows, targets)):
            length = int(frame_lengths[index])
            labels = argmax[index, :length]
            hypothesis = collapse_nonblank(labels)
            reference = target.tolist()
            total_edits += edit_distance(reference, hypothesis)
            total_phones += len(reference)
            total_segments += int(nonblank_run_boundaries(labels).sum())
            total_seconds += row.duration
    return {
        "loss": total_loss / total_examples,
        "phone_error_rate": total_edits / total_phones,
        "phone_errors": total_edits,
        "reference_phones": total_phones,
        "boundary_rate_hz": total_segments / total_seconds,
        "audio_seconds": total_seconds,
        "utterances": total_examples,
    }


def checkpoint_paths(output_dir):
    paths = sorted(glob.glob(os.path.join(output_dir, "checkpoints", "epoch_*.pt")))
    if not paths:
        raise FileNotFoundError(f"No epoch checkpoints under {output_dir}")
    records = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        records.append((payload["metrics"]["phone_error_rate"], payload["epoch"], path))
    return [record[2] for record in sorted(records)]


def run_train(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    train_rows = collect_utterances(
        args.audio_root, [args.train_split], alignment_root=args.alignment_root
    )
    valid_rows = collect_utterances(
        args.audio_root, [args.valid_split], alignment_root=args.alignment_root
    )
    phone_set = sorted({phone for row in train_rows for phone in row.phones})
    phone_to_id = {phone: index + 1 for index, phone in enumerate(phone_set)}
    unseen = sorted({phone for row in valid_rows for phone in row.phones} - set(phone_set))
    if unseen:
        raise ValueError(f"Validation contains unseen phones: {unseen}")
    vocabulary = {"blank_id": BLANK_ID, "phone_to_id": phone_to_id}
    atomic_json(vocabulary, os.path.join(args.output_dir, "vocab.json"))

    train_sampler = DurationBatchSampler(
        train_rows, args.max_padded_seconds, args.max_batch_size, True, args.seed
    )
    valid_sampler = DurationBatchSampler(
        valid_rows, args.valid_max_padded_seconds, args.max_batch_size, False, args.seed
    )
    loader_kwargs = {
        "num_workers": args.num_workers,
        "collate_fn": collate_audio,
        "pin_memory": True,
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        AudioDataset(train_rows, phone_to_id), batch_sampler=train_sampler, **loader_kwargs
    )
    valid_loader = DataLoader(
        AudioDataset(valid_rows, phone_to_id), batch_sampler=valid_sampler, **loader_kwargs
    )

    device = torch.device(args.device)
    model = load_wavlm(args.wavlm_source, args.cache_dir, device)
    head = PhoneCTCHead(
        model.config.hidden_size,
        args.hidden_dim,
        len(phone_to_id) + 1,
        args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate)
    print(
        json.dumps(
            {
                "train_utterances": len(train_rows),
                "valid_utterances": len(valid_rows),
                "phone_vocabulary": len(phone_to_id),
                "train_batches": len(train_sampler),
                "wavlm_frozen": True,
                "epochs": args.epochs,
                "output_dir": args.output_dir,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        train_sampler.set_epoch(epoch)
        head.train()
        loss_sum = 0.0
        example_count = 0
        for step, (rows, waveforms, attention_mask, sample_lengths, targets) in enumerate(
            train_loader, start=1
        ):
            waveforms = waveforms.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            sample_lengths = sample_lengths.to(device)
            features = forward_features(model, waveforms, attention_mask)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = head(features)
            frame_lengths = output_lengths(model, sample_lengths)
            target_lengths = torch.tensor(
                [target.numel() for target in targets], device=device
            )
            concatenated = torch.cat(targets).to(device)
            loss = F.ctc_loss(
                logits.float().log_softmax(dim=-1).transpose(0, 1),
                concatenated,
                frame_lengths,
                target_lengths,
                blank=BLANK_ID,
                zero_infinity=True,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), args.max_grad_norm)
            optimizer.step()
            loss_sum += float(loss) * len(rows)
            example_count += len(rows)
            if step % args.log_interval == 0:
                print(
                    f"epoch={epoch} step={step}/{len(train_sampler)} "
                    f"train_loss={loss_sum / example_count:.4f}",
                    flush=True,
                )
        valid_metrics = validate(model, head, valid_loader, device)
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": loss_sum / example_count,
            "elapsed_seconds": time.perf_counter() - started,
            **valid_metrics,
        }
        checkpoint = {
            "epoch": epoch,
            "head_state": head.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "head_config": {
                "input_dim": model.config.hidden_size,
                "hidden_dim": args.hidden_dim,
                "vocab_size": len(phone_to_id) + 1,
                "dropout": args.dropout,
            },
            "vocabulary": vocabulary,
            "metrics": valid_metrics,
            "wavlm_source": args.wavlm_source,
            "target_definition": "ordered MFA phones; stress stripped; timing discarded",
        }
        atomic_torch_save(
            checkpoint,
            os.path.join(args.output_dir, "checkpoints", f"epoch_{epoch:03d}.pt"),
        )
        os.makedirs(args.output_dir, exist_ok=True)
        with open(metrics_path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(epoch_metrics, sort_keys=True) + "\n")
        print(json.dumps(epoch_metrics, sort_keys=True), flush=True)

    ranked = checkpoint_paths(args.output_dir)
    ranking = []
    for rank, path in enumerate(ranked):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        ranking.append(
            {
                "rank": rank,
                "epoch": payload["epoch"],
                "checkpoint": path,
                **payload["metrics"],
            }
        )
    atomic_json({"checkpoints": ranking}, os.path.join(args.output_dir, "ranking.json"))


@torch.inference_mode()
def run_infer(args):
    ranked = checkpoint_paths(args.output_dir)
    if not 0 <= args.checkpoint_rank < len(ranked):
        raise ValueError(
            f"checkpoint_rank={args.checkpoint_rank} outside 0..{len(ranked) - 1}"
        )
    checkpoint_path = ranked[args.checkpoint_rank]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metrics = checkpoint["metrics"]
    if args.require_max_per is not None and metrics["phone_error_rate"] > args.require_max_per:
        raise RuntimeError(
            f"Best available dev PER {metrics['phone_error_rate']:.3f} exceeds "
            f"gate {args.require_max_per:.3f}"
        )
    rows = collect_utterances(
        args.audio_root,
        args.splits,
        shard=args.shard,
        num_shards=args.num_shards,
    )
    sampler = DurationBatchSampler(
        rows, args.max_padded_seconds, args.max_batch_size, False, args.seed
    )
    loader = DataLoader(
        AudioDataset(rows),
        batch_sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_audio,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    device = torch.device(args.device)
    model = load_wavlm(checkpoint["wavlm_source"], args.cache_dir, device)
    head = PhoneCTCHead(**checkpoint["head_config"]).to(device)
    head.load_state_dict(checkpoint["head_state"])
    head.eval()
    started = time.perf_counter()
    split_frames = int(args.long_segment_split_min_frames)
    if split_frames != 0 and split_frames < 2:
        raise ValueError("long_segment_split_min_frames must be zero or at least two")
    totals = {
        "utterances": 0,
        "audio_seconds": 0.0,
        "frames": 0,
        "base_segments": 0,
        "segments": 0,
    }
    per_split = {}
    for batch_index, (batch_rows, waveforms, attention_mask, sample_lengths, _) in enumerate(
        loader, start=1
    ):
        waveforms = waveforms.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        sample_lengths_device = sample_lengths.to(device)
        features = forward_features(model, waveforms, attention_mask)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = head(features)
        frame_lengths = output_lengths(model, sample_lengths_device).cpu()
        argmax = logits.argmax(dim=-1).cpu()
        for index, row in enumerate(batch_rows):
            frames = int(frame_lengths[index])
            expected_frames = num_encoder_frames(row.samples)
            if frames != expected_frames:
                raise RuntimeError(
                    f"{row.uid}: WavLM frames={frames}, expected={expected_frames}"
                )
            base_boundary = nonblank_run_boundaries(argmax[index, :frames])
            boundary = (
                refine_boundary(base_boundary, split_frames)
                if split_frames
                else base_boundary
            )
            if not args.stats_only:
                atomic_torch_save(boundary, os.path.join(args.boundary_dir, f"{row.uid}.pt"))
            values = per_split.setdefault(
                row.split,
                {
                    "utterances": 0,
                    "audio_seconds": 0.0,
                    "frames": 0,
                    "base_segments": 0,
                    "segments": 0,
                },
            )
            for aggregate in (totals, values):
                aggregate["utterances"] += 1
                aggregate["audio_seconds"] += row.duration
                aggregate["frames"] += frames
                aggregate["base_segments"] += int(base_boundary.sum())
                aggregate["segments"] += int(boundary.sum())
        if batch_index % args.log_interval == 0 or batch_index == len(sampler):
            print(
                f"rank={args.checkpoint_rank} shard={args.shard}/{args.num_shards} "
                f"batches={batch_index}/{len(sampler)} rate_hz="
                f"{totals['segments'] / totals['audio_seconds']:.2f}",
                flush=True,
            )
    for values in [totals, *per_split.values()]:
        values["added_segments"] = values["segments"] - values["base_segments"]
        values["base_boundary_rate_hz"] = (
            values["base_segments"] / values["audio_seconds"]
        )
        values["boundary_rate_hz"] = values["segments"] / values["audio_seconds"]
    rate = totals["boundary_rate_hz"]
    if not args.require_rate_min <= rate <= args.require_rate_max:
        raise RuntimeError(
            f"Boundary-rate gate failed: {rate:.3f} Hz not in "
            f"[{args.require_rate_min:.3f}, {args.require_rate_max:.3f}]"
        )
    stats = {
        "method": (
            "frozen_wavlm_large_phone_ctc_nonblank_runs_plus_long_segment_midpoints"
            if split_frames
            else "frozen_wavlm_large_phone_ctc_nonblank_runs"
        ),
        "transcript_used_for_boundary_inference": False,
        "long_segment_split_min_frames": split_frames,
        "training_target": checkpoint["target_definition"],
        "checkpoint": checkpoint_path,
        "checkpoint_rank": args.checkpoint_rank,
        "checkpoint_epoch": checkpoint["epoch"],
        "dev_phone_error_rate": metrics["phone_error_rate"],
        "dev_boundary_rate_hz": metrics["boundary_rate_hz"],
        "shard": args.shard,
        "num_shards": args.num_shards,
        "stats_only": args.stats_only,
        "elapsed_seconds": time.perf_counter() - started,
        **totals,
        "splits": per_split,
    }
    stats_dir = args.boundary_dir if not args.stats_only else args.audit_dir
    atomic_json(
        stats,
        os.path.join(
            stats_dir,
            "_stats",
            f"rank_{args.checkpoint_rank:02d}_shard_{args.shard:02d}.json",
        ),
    )
    print(json.dumps(stats, sort_keys=True), flush=True)


def build_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--audio_root", required=True)
    common.add_argument("--output_dir", required=True)
    common.add_argument("--cache_dir", required=True)
    common.add_argument("--wavlm_source", default="microsoft/wavlm-large")
    common.add_argument("--device", default="cuda")
    common.add_argument("--seed", type=int, default=3407)
    common.add_argument("--num_workers", type=int, default=6)
    common.add_argument("--max_padded_seconds", type=float, default=120.0)
    common.add_argument("--max_batch_size", type=int, default=16)
    common.add_argument("--log_interval", type=int, default=100)

    train = subparsers.add_parser("train", parents=[common])
    train.add_argument("--alignment_root", required=True)
    train.add_argument("--train_split", default="train-clean-100")
    train.add_argument("--valid_split", default="dev-clean")
    train.add_argument("--epochs", type=int, default=3)
    train.add_argument("--hidden_dim", type=int, default=512)
    train.add_argument("--dropout", type=float, default=0.1)
    train.add_argument("--learning_rate", type=float, default=1e-3)
    train.add_argument("--max_grad_norm", type=float, default=5.0)
    train.add_argument("--valid_max_padded_seconds", type=float, default=160.0)

    infer = subparsers.add_parser("infer", parents=[common])
    infer.add_argument("--splits", nargs="+", required=True)
    infer.add_argument("--boundary_dir", required=True)
    infer.add_argument("--audit_dir", required=True)
    infer.add_argument("--checkpoint_rank", type=int, default=0)
    infer.add_argument("--shard", type=int, default=0)
    infer.add_argument("--num_shards", type=int, default=1)
    infer.add_argument("--stats_only", action="store_true")
    infer.add_argument("--require_max_per", type=float, default=None)
    infer.add_argument("--require_rate_min", type=float, default=7.0)
    infer.add_argument("--require_rate_max", type=float, default=14.0)
    infer.add_argument(
        "--long_segment_split_min_frames",
        type=int,
        default=0,
        help=(
            "Preserve every CTC boundary and add one midpoint to segments at "
            "least this many frames long; zero disables refinement."
        ),
    )
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if arguments.command == "train":
        run_train(arguments)
    else:
        run_infer(arguments)
