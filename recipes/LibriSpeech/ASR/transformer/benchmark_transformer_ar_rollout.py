#!/usr/bin/env python3
"""Benchmark isolated full-prefix Transformer boundary rollouts.

This script intentionally does not change the production policy implementation.  It
loads a trained segmenter, compares the current concatenating KV cache against an
exact preallocated cache, and measures local-history variants.  The effective batch
always contains four rollout samples per utterance (GRPO K=4).
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from segmenter import Segmenter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lengths", type=int, nargs="+", default=[256, 512, 1024])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


@torch.inference_mode()
def preallocated_rollout(policy, features, padding_mask, history_window=None):
    """Greedy rollout with indexed KV writes; full history is numerically exact."""
    batch_size, num_frames, _ = features.shape
    previous = torch.full(
        (batch_size,), policy.bos_index, dtype=torch.long, device=features.device
    )
    caches = []
    for layer in policy.layers:
        shape = (
            batch_size,
            layer.nhead,
            num_frames,
            layer.head_dim,
        )
        caches.append(
            (
                torch.empty(shape, device=features.device, dtype=features.dtype),
                torch.empty(shape, device=features.device, dtype=features.dtype),
            )
        )

    boundaries = torch.empty(
        (batch_size, num_frames), dtype=torch.long, device=features.device
    )
    logits = torch.empty(
        (batch_size, num_frames), dtype=features.dtype, device=features.device
    )

    for position in range(num_frames):
        h = policy.audio_proj(features[:, position]).unsqueeze(1)
        h = h + policy.boundary_embedding(previous).unsqueeze(1)
        h = h + policy.pe.pe[:, position : position + 1].to(
            device=h.device, dtype=h.dtype
        )
        start = 0
        if history_window is not None:
            start = max(0, position + 1 - history_window)
        for layer, (key_cache, value_cache) in zip(policy.layers, caches):
            residual = h
            query, key, value = layer._split_qkv(layer.norm1(h))
            key_cache[:, :, position : position + 1].copy_(key)
            value_cache[:, :, position : position + 1].copy_(value)
            attended = F.scaled_dot_product_attention(
                query,
                key_cache[:, :, start : position + 1],
                value_cache[:, :, start : position + 1],
                dropout_p=0.0,
                is_causal=False,
            )
            attended = attended.transpose(1, 2).reshape_as(h)
            h = residual + layer.out(attended)
            h = h + layer.ffn(layer.norm2(h))
        logit = policy.head(policy.norm(h)).squeeze(-1).squeeze(-1)
        active = ~padding_mask[:, position]
        action = (logit > 0).long()
        if position == 0:
            action.zero_()
        action = action.masked_fill(~active, -1)
        boundaries[:, position] = action
        logits[:, position] = logit
        previous = torch.where(active, action.clamp(min=0), previous)
    return boundaries, logits


def synchronize():
    torch.cuda.synchronize()


def measure(label, function, warmup, repeats):
    for _ in range(warmup):
        function()
    synchronize()
    samples = []
    peak_bytes = 0
    for _ in range(repeats):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        synchronize()
        started = time.perf_counter()
        function()
        synchronize()
        samples.append(time.perf_counter() - started)
        peak_bytes = max(peak_bytes, torch.cuda.max_memory_allocated())
    return {
        "label": label,
        "seconds": samples,
        "median_seconds": statistics.median(samples),
        "peak_allocated_gib": peak_bytes / (1024**3),
    }


@torch.inference_mode()
def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.batch_size < 1 or any(length < 2 for length in args.lengths):
        raise ValueError("batch size must be positive and lengths must be >=2")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    grpo_k = 4

    segmenter = Segmenter(
        input_dim=1024,
        backbone="transformer_ar",
        ar_hidden_dim=256,
        ar_num_layers=4,
        ar_nhead=4,
        ar_ffn_dim=1024,
        ar_dropout=0.0,
        ar_max_positions=max(max(args.lengths), 4096),
    )
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    segmenter.load_state_dict(state, strict=True)
    segmenter = segmenter.to(device=device, dtype=dtype).eval()
    policy = segmenter.net

    report = {
        "host": os.uname().nodename,
        "gpu": torch.cuda.get_device_name(),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "dtype": str(dtype),
        "base_batch_size": args.batch_size,
        "grpo_k": grpo_k,
        "effective_rollout_batch": args.batch_size * grpo_k,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "lengths": [],
    }

    for num_frames in args.lengths:
        base_features = torch.randn(
            args.batch_size, num_frames, 1024, device=device, dtype=dtype
        )
        features = base_features.repeat(grpo_k, 1, 1)
        padding_mask = torch.zeros(
            features.shape[:2], dtype=torch.bool, device=device
        )
        if args.batch_size > 1:
            base_lengths = torch.linspace(
                num_frames,
                max(2, int(num_frames * 0.7)),
                args.batch_size,
                device=device,
            ).long()
            lengths = base_lengths.repeat(grpo_k)
            padding_mask = (
                torch.arange(num_frames, device=device).unsqueeze(0)
                >= lengths.unsqueeze(1)
            )

        baseline_output = policy.rollout(
            features, padding_mask, mode="argmax"
        )
        exact_boundaries, exact_logits = preallocated_rollout(
            policy, features, padding_mask
        )
        valid = ~padding_mask
        max_logit_error = (
            (baseline_output.logits - exact_logits).abs()[valid].max().item()
        )
        mismatch_rate = (
            (baseline_output.boundaries[valid] != exact_boundaries[valid])
            .float()
            .mean()
            .item()
        )
        if mismatch_rate != 0.0:
            raise RuntimeError(
                f"Exact preallocated cache changed greedy boundaries at T={num_frames}: "
                f"mismatch_rate={mismatch_rate}"
            )

        variants = []
        variants.append(
            measure(
                "current_concat_full",
                lambda: policy.rollout(features, padding_mask, mode="argmax"),
                args.warmup,
                args.repeats,
            )
        )
        variants.append(
            measure(
                "preallocated_full",
                lambda: preallocated_rollout(policy, features, padding_mask),
                args.warmup,
                args.repeats,
            )
        )
        for window in (64,):
            variants.append(
                measure(
                    f"preallocated_local_{window}",
                    lambda window=window: preallocated_rollout(
                        policy, features, padding_mask, history_window=window
                    ),
                    args.warmup,
                    args.repeats,
                )
            )

        baseline_time = variants[0]["median_seconds"]
        for variant in variants:
            variant["speedup_vs_current"] = (
                baseline_time / variant["median_seconds"]
            )
        item = {
            "num_frames": num_frames,
            "max_exact_logit_error": max_logit_error,
            "exact_boundary_mismatch_rate": mismatch_rate,
            "variants": variants,
        }
        report["lengths"].append(item)
        print(json.dumps(item, sort_keys=True), flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
