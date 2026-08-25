#!/usr/bin/env python3
"""Evaluate Transformer-AR segment boundaries against phone boundaries.

The boundary metrics follow the harsh and lenient matching schemes implemented
by Strgar and Harwath's ``ss-phoneme-seg`` repository.  At the default WavLM
frame rate, one frame of tolerance is approximately 20 ms.

This is an inference-only audit: it loads the WavLM encoder and the segmenter
checkpoints, but never constructs or runs the LLM decoder.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import sys
from collections.abc import Iterable
from pathlib import Path

import torch
import torchaudio

RECIPE = Path(__file__).resolve().parent
sys.path.insert(0, str(RECIPE))
sys.path.insert(0, str(RECIPE.parents[3]))

from segmenter import Segmenter  # noqa: E402

from speechbrain.integrations.huggingface.wav2vec2 import Wav2Vec2  # noqa: E402
from speechbrain.processing.features import InputNormalization  # noqa: E402

DEFAULT_SEEDS = (3407, 3408, 3409)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_rows(path: Path, max_utts: int) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if max_utts > 0:
        rows = rows[:max_utts]
    return rows


def _selected_epoch(train_log: Path) -> int:
    """Return the checkpoint epoch used by the run's final test evaluation."""
    epochs = []
    for line in train_log.read_text(errors="replace").splitlines():
        match = re.match(r"Epoch loaded: (\d+) - test", line)
        if match:
            epochs.append(int(match.group(1)))
    if not epochs:
        raise ValueError(f"No selected test epoch found in {train_log}")
    if len(set(epochs)) != 1:
        raise ValueError(f"Inconsistent selected epochs in {train_log}: {epochs}")
    return epochs[0]


def _checkpoint_epoch(checkpoint_dir: Path) -> int | None:
    metadata = checkpoint_dir / "CKPT.yaml"
    if not metadata.is_file():
        return None
    match = re.search(r"^epoch:\s*(\d+)\s*$", metadata.read_text(), re.MULTILINE)
    return int(match.group(1)) if match else None


def _selected_segmenter(run_dir: Path) -> tuple[int, Path]:
    epoch = _selected_epoch(run_dir / "train_log.txt")
    candidates = [
        checkpoint_dir / "segmenter.ckpt"
        for checkpoint_dir in sorted((run_dir / "save").glob("CKPT*"))
        if _checkpoint_epoch(checkpoint_dir) == epoch
        and (checkpoint_dir / "segmenter.ckpt").is_file()
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one epoch-{epoch} segmenter under {run_dir}, found {candidates}"
        )
    return epoch, candidates[0]


def _build_segmenter(checkpoint: Path, device: torch.device) -> Segmenter:
    """Construct the exact local-64 Transformer-AR + BiGRU architecture."""
    model = Segmenter(
        input_dim=1024,
        backbone="transformer_ar",
        hidden_dim=256,
        num_layers=4,
        nhead=4,
        ffn_dim=1024,
        dropout=0.0,
        ar_hidden_dim=256,
        ar_num_layers=4,
        ar_nhead=4,
        ar_ffn_dim=1024,
        ar_dropout=0.0,
        ar_max_positions=4096,
        ar_history_window=64,
        ar_cache_mode="preallocated",
        pooling="bigru_residual",
        pooling_input_dim=1024,
        pooling_hidden_dim=128,
        pooling_num_layers=1,
        pooling_dropout=0.0,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.eval().to(device)


def _directional_counts(
    reference: list[int], candidates: list[int], tolerance: int
) -> tuple[int, int]:
    """Match candidates to references as in ``ss-phoneme-seg/utils/eval.py``.

    The first result is the number of one-to-one matches.  The second counts
    additional candidates that lie within tolerance of an already-used
    reference; the lenient scheme accepts those duplicates.
    """
    used: set[int] = set()
    matches = duplicates = 0
    for candidate in candidates:
        # The reference implementation sorts eligible reference indices before
        # assignment.  With time-sorted inputs this is an ordered greedy match,
        # rather than a closest-distance match.
        nearby = [
            index
            for index, position in enumerate(reference)
            if abs(position - candidate) <= tolerance
        ]
        if not nearby:
            continue
        unused = next((index for index in nearby if index not in used), None)
        if unused is None:
            duplicates += 1
        else:
            used.add(unused)
            matches += 1
    return matches, duplicates


def _empty_counts() -> dict[str, int]:
    return {
        "precision_hits": 0,
        "recall_hits": 0,
        "predicted": 0,
        "reference": 0,
    }


def _update_counts(
    counts: dict[str, dict[str, int]],
    pred: list[int],
    gold: list[int],
    tolerance: int,
) -> None:
    precision_hits, precision_duplicates = _directional_counts(
        gold, pred, tolerance
    )
    recall_hits, recall_duplicates = _directional_counts(pred, gold, tolerance)
    for scheme, p_hits, r_hits in (
        ("harsh", precision_hits, recall_hits),
        (
            "lenient",
            precision_hits + precision_duplicates,
            recall_hits + recall_duplicates,
        ),
    ):
        counts[scheme]["precision_hits"] += p_hits
        counts[scheme]["recall_hits"] += r_hits
        counts[scheme]["predicted"] += len(pred)
        counts[scheme]["reference"] += len(gold)


def _score(counts: dict[str, int]) -> dict[str, float | int]:
    predicted = counts["predicted"]
    reference = counts["reference"]
    precision = counts["precision_hits"] / predicted if predicted else 0.0
    recall = counts["recall_hits"] / reference if reference else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    # Preserve the reference implementation's finite behavior at zero counts.
    eps = 1e-8
    over_segmentation = recall / (precision + eps) - 1
    r1 = math.sqrt((1 - recall) ** 2 + over_segmentation**2)
    r2 = (-over_segmentation + recall - 1) / math.sqrt(2)
    r_value = 1 - (abs(r1) + abs(r2)) / 2
    return {
        **counts,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "r_value": r_value,
    }


def _positions(target: torch.Tensor, implicit_frame_zero: bool) -> list[int]:
    positions = (target.view(-1) == 1).nonzero().flatten().tolist()
    if implicit_frame_zero and target.numel() and (not positions or positions[0] != 0):
        positions.insert(0, 0)
    return positions


def _load_wave(row: dict[str, str], data_root: Path) -> torch.Tensor:
    path = Path(row["wav"].replace("$data_root", str(data_root)))
    waveform, sample_rate = torchaudio.load(path)
    waveform = waveform.mean(0)
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
    return waveform


def _new_aggregate(seeds: Iterable[int], tolerances: Iterable[int]):
    return {
        seed: {
            tolerance: {
                scheme: _empty_counts() for scheme in ("harsh", "lenient")
            }
            for tolerance in tolerances
        }
        for seed in seeds
    }


def _summarize(per_seed: dict[str, dict]) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    tolerance_names = [
        name
        for name, scores in next(iter(per_seed.values())).items()
        if isinstance(scores, dict) and "harsh" in scores
    ]
    for tolerance in tolerance_names:
        summary[tolerance] = {}
        for scheme in ("harsh", "lenient"):
            summary[tolerance][scheme] = {}
            for metric in ("precision", "recall", "f1", "r_value"):
                values = [seed[tolerance][scheme][metric] for seed in per_seed.values()]
                summary[tolerance][scheme][metric] = {
                    "mean": statistics.mean(values),
                    "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
                }
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--phone-dir", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--ssl-source", default="microsoft/wavlm-large")
    parser.add_argument("--ssl-cache", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-utts", type=int, default=0)
    parser.add_argument("--tolerances", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rows = _load_rows(args.csv, args.max_utts)
    tolerances = tuple(sorted(set(args.tolerances)))
    if not rows:
        raise ValueError(f"No utterances found in {args.csv}")
    if not tolerances or tolerances[0] < 0:
        raise ValueError("Tolerances must be non-negative frame counts")
    for directory in (args.data_root, args.phone_dir, args.run_root, args.ssl_cache):
        if not directory.exists():
            raise FileNotFoundError(directory)

    selected = {}
    for seed in args.seeds:
        epoch, checkpoint = _selected_segmenter(args.run_root / str(seed))
        selected[seed] = {
            "epoch": epoch,
            "path": str(checkpoint),
            "sha256": _sha256(checkpoint),
        }
    missing_targets = [
        row["ID"]
        for row in rows
        if not (args.phone_dir / f"{row['ID']}.pt").is_file()
    ]
    if missing_targets:
        raise FileNotFoundError(
            f"Missing {len(missing_targets)} phone targets; first IDs: "
            f"{missing_targets[:5]}"
        )
    if args.validate_only:
        print(
            json.dumps(
                {
                    "utterances": len(rows),
                    "tolerances_frames": tolerances,
                    "selected_checkpoints": selected,
                },
                indent=2,
            )
        )
        return

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    use_bf16 = args.precision == "bf16" and device.type == "cuda"
    models = {
        seed: _build_segmenter(Path(info["path"]), device)
        for seed, info in selected.items()
    }
    ssl = Wav2Vec2(
        source=args.ssl_source,
        output_norm=True,
        freeze=True,
        save_path=str(args.ssl_cache),
        device_map=device.type,
    ).eval()
    normalization = InputNormalization(norm_type="sentence").to(device).eval()
    aggregate = _new_aggregate(args.seeds, tolerances)
    frame_totals = {seed: {"frames": 0, "segments": 0} for seed in args.seeds}

    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start : start + args.batch_size]
        waves = [_load_wave(row, args.data_root) for row in batch_rows]
        golds = [
            torch.load(
                args.phone_dir / f"{row['ID']}.pt",
                map_location="cpu",
                weights_only=True,
            ).view(-1)
            for row in batch_rows
        ]
        wav_lengths = torch.tensor([wave.numel() for wave in waves], device=device)
        max_wav = int(wav_lengths.max())
        wav_batch = torch.zeros(len(waves), max_wav, device=device)
        for index, waveform in enumerate(waves):
            wav_batch[index, : waveform.numel()] = waveform.to(device)
        relative_lengths = wav_lengths.float() / max_wav

        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            features = ssl(
                normalization(wav_batch, relative_lengths), relative_lengths
            )

        target_lengths = torch.tensor(
            [target.numel() for target in golds], device=device
        )
        if int(target_lengths.max()) != features.size(1):
            raise ValueError(
                f"WavLM returned {features.size(1)} frames; longest phone target "
                f"has {int(target_lengths.max())} frames"
            )
        frame_index = torch.arange(features.size(1), device=device).unsqueeze(0)
        padding_mask = frame_index >= target_lengths.unsqueeze(1)

        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            predictions = {
                seed: model.sample_boundaries(
                    features, padding_mask, deterministic=True
                ).boundaries.cpu()
                for seed, model in models.items()
            }

        for item_index, gold_tensor in enumerate(golds):
            length = gold_tensor.numel()
            gold = _positions(gold_tensor, implicit_frame_zero=False)
            for seed, boundary in predictions.items():
                # Transformer-AR stores zero at frame 0 because there is no
                # decision there, while the pooler always begins a segment at
                # frame 0.  Include that implicit start in the boundary set.
                pred = _positions(
                    boundary[item_index, :length], implicit_frame_zero=True
                )
                frame_totals[seed]["frames"] += length
                frame_totals[seed]["segments"] += len(pred)
                for tolerance in tolerances:
                    _update_counts(
                        aggregate[seed][tolerance], pred, gold, tolerance
                    )
        completed = min(start + len(batch_rows), len(rows))
        if start == 0 or completed == len(rows) or completed % 200 < args.batch_size:
            print(f"evaluated {completed}/{len(rows)} utterances", flush=True)

    tolerance_names = {
        tolerance: "exact" if tolerance == 0 else f"tol_{20 * tolerance}ms"
        for tolerance in tolerances
    }
    per_seed = {
        str(seed): {
            tolerance_names[tolerance]: {
                scheme: _score(counts)
                for scheme, counts in schemes.items()
            }
            for tolerance, schemes in per_tolerance.items()
        }
        for seed, per_tolerance in aggregate.items()
    }
    for seed, totals in frame_totals.items():
        per_seed[str(seed)]["kept_ratio"] = (
            totals["segments"] / totals["frames"] if totals["frames"] else 0.0
        )
    result = {
        "split": args.csv.stem,
        "utterances": len(rows),
        "reference": str(args.phone_dir),
        "frame_rate_hz": 50,
        "metric": {
            "source": {
                "repository": "https://github.com/lstrgar/ss-phoneme-seg",
                "commit": "8092c5778844aaaf0016b0b1ee9756e316310526",
                "implementation": "utils/eval.py",
            },
            "schemes": {
                "harsh": "ordered one-to-one matching within tolerance",
                "lenient": "harsh matching plus duplicate in-tolerance matches",
            },
            "implicit_frame_zero": (
                "included in predictions because it is an unconditional segment "
                "start in the Transformer-AR pooler"
            ),
        },
        "model": {
            "architecture": "WavLM Large + local-64 Transformer-AR + BiGRU residual pooling",
            "run_root": str(args.run_root),
            "selected_checkpoints": {str(k): v for k, v in selected.items()},
        },
        "per_seed": per_seed,
        "summary": _summarize(per_seed),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["summary"], indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
