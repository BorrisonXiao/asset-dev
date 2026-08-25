#!/usr/bin/env python3
"""Evaluate fixed-rate pooling boundaries against phone boundaries.

The fixed-rate systems place boundaries at frames ``k, 2k, ...`` and have an
implicit segment start at frame zero.  Agreement uses the same harsh/lenient
metrics as ``audit_transformer_ar_phone_agreement.py`` so the results can be
compared directly with learned segmenters.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

RECIPE = Path(__file__).resolve().parent
sys.path.insert(0, str(RECIPE))

from audit_transformer_ar_phone_agreement import (  # noqa: E402
    _empty_counts,
    _positions,
    _score,
    _update_counts,
)

DEFAULT_KS = (3, 4, 5, 6, 8)
DEFAULT_SEEDS = (3407, 3408, 3409)


def _load_ids(csv_path: Path) -> list[str]:
    with csv_path.open(newline="") as handle:
        return [row["ID"] for row in csv.DictReader(handle)]


def _load_phone_target(item: tuple[Path, str]) -> tuple[str, torch.Tensor]:
    phone_dir, utterance_id = item
    path = phone_dir / f"{utterance_id}.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    target = torch.load(path, map_location="cpu", weights_only=True).view(-1)
    return utterance_id, target


def _fixed_positions(length: int, k: int) -> list[int]:
    if k < 1:
        raise ValueError(f"k must be at least one, got {k}")
    return [0, *range(k, length, k)] if length else []


def _validate_completed_systems(
    results_root: Path, ks: tuple[int, ...], seeds: tuple[int, ...]
) -> dict[str, list[str]]:
    completed = {}
    for k in ks:
        runs = []
        for seed in seeds:
            run = results_root / f"fixed_rate_k{k}_tc100" / str(seed)
            hparams = run / "hyperparams.yaml"
            train_log = run / "train_log.txt"
            if not hparams.is_file() or not train_log.is_file():
                raise FileNotFoundError(
                    f"Missing completed fixed-k={k}, seed={seed} run under {run}"
                )
            if "Epoch loaded:" not in train_log.read_text(errors="replace"):
                raise ValueError(f"No completed test evaluation found in {train_log}")
            runs.append(str(run))
        completed[f"fixed_k{k}"] = runs
    return completed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--phone-dir", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--ks", type=int, nargs="+", default=list(DEFAULT_KS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--tolerances", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--learned-result",
        type=Path,
        help="Optional Transformer-AR agreement JSON for direct F1 deltas.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    ks = tuple(sorted(set(args.ks)))
    seeds = tuple(args.seeds)
    tolerances = tuple(sorted(set(args.tolerances)))
    if not ks or ks[0] < 1:
        raise ValueError("All fixed-rate k values must be positive")
    if not tolerances or tolerances[0] < 0:
        raise ValueError("All tolerances must be non-negative")
    for path in (args.csv, args.phone_dir, args.results_root):
        if not path.exists():
            raise FileNotFoundError(path)

    completed_runs = _validate_completed_systems(args.results_root, ks, seeds)
    utterance_ids = _load_ids(args.csv)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        loaded = list(
            pool.map(
                _load_phone_target,
                ((args.phone_dir, utterance_id) for utterance_id in utterance_ids),
            )
        )
    targets = dict(loaded)

    aggregate = {
        k: {
            tolerance: {
                scheme: _empty_counts() for scheme in ("harsh", "lenient")
            }
            for tolerance in tolerances
        }
        for k in ks
    }
    total_frames = total_phone_boundaries = 0
    predicted_totals = dict.fromkeys(ks, 0)
    for utterance_id in utterance_ids:
        target = targets[utterance_id]
        gold = _positions(target, implicit_frame_zero=False)
        total_frames += target.numel()
        total_phone_boundaries += len(gold)
        for k in ks:
            pred = _fixed_positions(target.numel(), k)
            predicted_totals[k] += len(pred)
            for tolerance in tolerances:
                _update_counts(aggregate[k][tolerance], pred, gold, tolerance)

    tolerance_names = {
        tolerance: "exact" if tolerance == 0 else f"tol_{20 * tolerance}ms"
        for tolerance in tolerances
    }
    systems = {}
    for k in ks:
        kept_ratio = predicted_totals[k] / total_frames
        systems[f"fixed_k{k}"] = {
            "k": k,
            "segment_width_ms": 20 * k,
            "nominal_token_rate_hz": 50 / k,
            "actual_kept_ratio": kept_ratio,
            "actual_token_rate_hz": 50 * kept_ratio,
            "predicted_boundaries": predicted_totals[k],
            "scores": {
                tolerance_names[tolerance]: {
                    scheme: _score(counts)
                    for scheme, counts in schemes.items()
                }
                for tolerance, schemes in aggregate[k].items()
            },
        }

    comparison = None
    if args.learned_result is not None:
        learned = json.loads(args.learned_result.read_text())
        comparison = {
            "learned_result": str(args.learned_result),
            "learned_system": learned["model"]["architecture"],
            "delta_fixed_minus_learned_mean": {},
        }
        for label, system in systems.items():
            comparison["delta_fixed_minus_learned_mean"][label] = {}
            for tolerance, schemes in system["scores"].items():
                comparison["delta_fixed_minus_learned_mean"][label][tolerance] = {
                    scheme: {
                        metric: scores[metric]
                        - learned["summary"][tolerance][scheme][metric]["mean"]
                        for metric in ("precision", "recall", "f1", "r_value")
                    }
                    for scheme, scores in schemes.items()
                }

    result = {
        "split": args.csv.stem,
        "utterances": len(utterance_ids),
        "frame_rate_hz": 50,
        "reference": {
            "path": str(args.phone_dir),
            "boundaries": total_phone_boundaries,
            "kept_ratio": total_phone_boundaries / total_frames,
            "boundary_rate_hz": 50 * total_phone_boundaries / total_frames,
        },
        "metric": {
            "implementation": "audit_transformer_ar_phone_agreement.py",
            "implicit_frame_zero": "included in every fixed-rate prediction",
        },
        "completed_runs": completed_runs,
        "systems": systems,
        "comparison": comparison,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    printable = {
        label: {
            "actual_token_rate_hz": system["actual_token_rate_hz"],
            "exact_harsh_f1": system["scores"]["exact"]["harsh"]["f1"],
            "tol_20ms_harsh_f1": system["scores"]["tol_20ms"]["harsh"]["f1"],
            "tol_40ms_harsh_f1": system["scores"]["tol_40ms"]["harsh"]["f1"],
        }
        for label, system in systems.items()
    }
    print(json.dumps(printable, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
