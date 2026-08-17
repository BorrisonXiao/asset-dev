#!/usr/bin/env python3
"""Summarize matched short bilevel training pilots from their saved logs."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

FIELD_RE = re.compile(
    r"(train|valid|test) ([A-Za-z0-9_]+): "
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)"
)


def parse_train_log(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    epoch_lines = [line for line in lines if line.startswith("epoch:")]
    if not epoch_lines:
        raise ValueError(f"No completed epoch in {path}")
    line = epoch_lines[-1]
    fields = {
        f"{scope}_{name}": float(value)
        for scope, name, value in FIELD_RE.findall(line)
    }
    epoch = int(line.split(",", 1)[0].split(":", 1)[1])
    tests = []
    for test_line in lines:
        if test_line.startswith("Epoch loaded:"):
            tests.append(
                {
                    f"{scope}_{name}": float(value)
                    for scope, name, value in FIELD_RE.findall(test_line)
                }
            )
    return epoch, fields, tests


def parse_timing(path: Path):
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def parse_diagnostics(path: Path):
    """Load full-precision per-batch records written by the training loop."""
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def mean(records, field):
    values = [row[field] for row in records if field in row]
    return sum(values) / len(values) if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("recipes/LibriSpeech/ASR/transformer/results")
        / "speechllm_segmenter_wavlm"
        / "bilevel_transformer_ar_local64_bigru_pilot",
    )
    parser.add_argument("--tag", default="matched_v3")
    parser.add_argument("--inner-lr", default="0.01")
    parser.add_argument("--seed", default="3407")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=[
            "off",
            "heldout",
            "decoder_lookahead",
            "decoder_pooler_lookahead",
        ],
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/segmenter/bilevel_training_pilot/results.json"),
    )
    args = parser.parse_args()

    runs = []
    for mode in args.modes:
        run_dir = (
            args.root / args.tag / mode / f"lr_{args.inner_lr}" / args.seed
        )
        epoch, fields, tests = parse_train_log(run_dir / "train_log.txt")
        timing = parse_timing(run_dir / "stage_timing.jsonl")
        diagnostics = parse_diagnostics(run_dir / "bilevel_diagnostics.jsonl")
        run = {
            "mode": mode,
            "seed": args.seed,
            "inner_lr": float(args.inner_lr),
            "epoch": epoch,
            "run_dir": str(run_dir),
            "train": fields,
            "timing": timing,
            "batch_diagnostics": diagnostics,
            "debug_test_metrics_in_order": tests,
        }
        if mode.endswith("lookahead"):
            before_values = [
                row["inner_support_ce"]
                for row in diagnostics
                if "inner_support_ce" in row
            ]
            after_values = [
                row["inner_support_ce_after"]
                for row in diagnostics
                if "inner_support_ce_after" in row
            ]
            gradient_norms = [
                row["inner_grad_norm"]
                for row in diagnostics
                if "inner_grad_norm" in row
            ]
            before = (
                sum(before_values) / len(before_values)
                if before_values
                else fields.get("train_bilevel_inner_support_ce")
            )
            after = (
                sum(after_values) / len(after_values)
                if after_values
                else fields.get("train_bilevel_inner_support_ce_after")
            )
            run["checks"] = {
                "nonzero_finite_inner_gradient": bool(gradient_norms)
                and all(
                    math.isfinite(value) and value > 0.0
                    for value in gradient_norms
                ),
                "all_selected_tensors_receive_gradients": bool(diagnostics)
                and all(
                    row.get("inner_gradient_tensors")
                    == row.get("inner_parameter_tensors")
                    for row in diagnostics
                ),
                "support_ce_decreased": (
                    before is not None and after is not None and after < before
                ),
                "rollout_ranking_changed": any(
                    row.get("top_rollout_flip", 0.0) > 0.0
                    for row in diagnostics
                ),
            }
            run["lookahead_summary"] = {
                "mean_support_ce_before": before,
                "mean_support_ce_after": after,
                "support_ce_reduction_percent": (
                    100.0 * (before - after) / before
                    if before and after is not None
                    else None
                ),
                "mean_signed_reward_delta": mean(diagnostics, "reward_delta"),
                "mean_absolute_reward_delta": mean(
                    diagnostics, "reward_delta_abs"
                ),
                "mean_top_rollout_flip_fraction": mean(
                    diagnostics, "top_rollout_flip"
                ),
                "mean_rank_correlation": mean(diagnostics, "rank_correlation"),
            }
        runs.append(run)

    off_train_seconds = next(
        (
            row["seconds"]
            for run in runs
            if run["mode"] == "off"
            for row in run["timing"]
            if row["stage"] == "TRAIN"
        ),
        None,
    )
    for run in runs:
        train_seconds = next(
            (
                row["seconds"]
                for row in run["timing"]
                if row["stage"] == "TRAIN"
            ),
            None,
        )
        run["train_seconds"] = train_seconds
        run["train_time_vs_off"] = (
            train_seconds / off_train_seconds
            if train_seconds is not None and off_train_seconds
            else None
        )

    result = {
        "study": "Short full-loop support/query bilevel implementation pilot",
        "warning": (
            "Debug batches validate mechanics and rollout rankings; their WER "
            "values are not corpus-level experimental results."
        ),
        "runs": runs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
