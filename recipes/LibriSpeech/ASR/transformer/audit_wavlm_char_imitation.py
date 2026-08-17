#!/usr/bin/env python3
"""Audit WavLM segmenter agreement with the char-CTC boundary oracle.

This is intentionally a small, read-only diagnostic.  It evaluates the same
argmax boundary policy used by the ASR recipe, before and after RL, and writes
aggregate exact/near-boundary scores plus representative utterances as JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import torchaudio

RECIPE = Path(__file__).resolve().parent
sys.path.insert(0, str(RECIPE))
sys.path.insert(0, str(RECIPE.parents[3]))

from segmenter import Segmenter  # noqa: E402

from speechbrain.integrations.huggingface.wav2vec2 import Wav2Vec2  # noqa: E402
from speechbrain.processing.features import InputNormalization  # noqa: E402


def _load_rows(csv_path: Path, n: int) -> list[dict]:
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: float(row["duration"]))
    if n <= 0 or n >= len(rows):
        return rows
    # Deterministic coverage of the full duration range.
    indices = torch.linspace(0, len(rows) - 1, n).round().long().tolist()
    return [rows[index] for index in indices]


def _load_segmenter(path: Path, backbone: str, device: torch.device) -> Segmenter:
    model = Segmenter(
        input_dim=1024,
        backbone=backbone,
        hidden_dim=256,
        kernel_size=7,
        num_layers=4,
        nhead=4,
        ffn_dim=1024,
        dropout=0.1,
    )
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.eval().to(device)


def _match_count(pred: list[int], gold: list[int], tolerance: int) -> int:
    """Maximum ordered one-to-one matches for sorted positions within tolerance."""
    i = j = matches = 0
    while i < len(pred) and j < len(gold):
        if abs(pred[i] - gold[j]) <= tolerance:
            matches += 1
            i += 1
            j += 1
        elif pred[i] < gold[j] - tolerance:
            i += 1
        else:
            j += 1
    return matches


def _counts(pred: list[int], gold: list[int], tolerance: int) -> dict[str, int]:
    tp = _match_count(pred, gold, tolerance)
    return {"tp": tp, "pred": len(pred), "gold": len(gold)}


def _score(counts: dict[str, int]) -> dict[str, float | int]:
    tp, pred, gold = counts["tp"], counts["pred"], counts["gold"]
    precision = tp / pred if pred else 0.0
    recall = tp / gold if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        **counts,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _add_counts(total: dict[str, int], item: dict[str, int]) -> None:
    for key in ("tp", "pred", "gold"):
        total[key] += item[key]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--boundary-dir", type=Path, required=True)
    parser.add_argument("--ssl-source", required=True)
    parser.add_argument("--ssl-cache", type=Path, required=True)
    parser.add_argument("--cold-cnn", type=Path, required=True)
    parser.add_argument("--rl-cnn", type=Path, required=True)
    parser.add_argument("--cold-transformer", type=Path, required=True)
    parser.add_argument("--rl-transformer", type=Path, required=True)
    parser.add_argument("--num-utts", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Execution device; auto uses CUDA when available.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device_name = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device=cuda was requested, but CUDA is unavailable.")
    device = torch.device(device_name)
    rows = _load_rows(args.csv, args.num_utts)
    models = {
        "cold_cnn": _load_segmenter(args.cold_cnn, "cnn", device),
        "rl_nll_mt_cnn": _load_segmenter(args.rl_cnn, "cnn", device),
        "cold_transformer": _load_segmenter(
            args.cold_transformer, "transformer", device
        ),
        "rl_nll_mt_transformer": _load_segmenter(
            args.rl_transformer, "transformer", device
        ),
    }

    ssl = Wav2Vec2(
        source=args.ssl_source,
        output_norm=True,
        freeze=True,
        save_path=str(args.ssl_cache),
        device_map=device_name,
    ).eval()
    normalization = InputNormalization(norm_type="sentence").to(device).eval()

    aggregate = {
        name: {
            tol: {"tp": 0, "pred": 0, "gold": 0}
            for tol in ("exact", "tol1", "tol2")
        }
        for name in models
    }
    per_utt: list[dict] = []

    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start : start + args.batch_size]
        waves, golds = [], []
        for row in batch_rows:
            wav_path = Path(row["wav"].replace("$data_root", str(args.data_root)))
            wav, sample_rate = torchaudio.load(wav_path)
            wav = wav.mean(0)
            if sample_rate != 16000:
                wav = torchaudio.functional.resample(wav, sample_rate, 16000)
            waves.append(wav)
            golds.append(
                torch.load(
                    args.boundary_dir / f"{row['ID']}.pt",
                    map_location="cpu",
                    weights_only=True,
                ).view(-1).long()
            )

        wav_lengths = torch.tensor([wav.numel() for wav in waves], device=device)
        max_wav = int(wav_lengths.max())
        wav_batch = torch.zeros(len(waves), max_wav, device=device)
        for index, wav in enumerate(waves):
            wav_batch[index, : wav.numel()] = wav.to(device)
        relative_lengths = wav_lengths.float() / max_wav

        with torch.inference_mode():
            features = ssl(
                normalization(wav_batch, relative_lengths), relative_lengths
            )

        target_lengths = torch.tensor(
            [gold.numel() for gold in golds], device=device
        )
        if int(target_lengths.max()) != features.size(1):
            raise ValueError(
                f"Encoder returned {features.size(1)} frames; longest target has "
                f"{int(target_lengths.max())} frames"
            )
        frame_index = torch.arange(features.size(1), device=device).unsqueeze(0)
        padding_mask = frame_index >= target_lengths.unsqueeze(1)

        predictions: dict[str, torch.Tensor] = {}
        with torch.inference_mode():
            for name, model in models.items():
                logits = model.boundary_logits(features, padding_mask)
                predictions[name] = logits > 0

        for item_index, (row, gold_tensor) in enumerate(zip(batch_rows, golds)):
            gold = (gold_tensor == 1).nonzero().flatten().tolist()
            record = {
                "id": row["ID"],
                "duration": float(row["duration"]),
                "transcript": row["wrd"],
                "frames": gold_tensor.numel(),
                "gold_boundaries": len(gold),
                "models": {},
            }
            for name in models:
                pred_tensor = predictions[name][item_index, : gold_tensor.numel()].cpu()
                pred = pred_tensor.nonzero().flatten().tolist()
                scores = {}
                for tolerance, label in ((0, "exact"), (1, "tol1"), (2, "tol2")):
                    item_counts = _counts(pred, gold, tolerance)
                    _add_counts(aggregate[name][label], item_counts)
                    scores[label] = _score(item_counts)
                scores["rho"] = len(pred) / max(gold_tensor.numel(), 1)
                record["models"][name] = scores
            per_utt.append(record)

    # Pick four examples spanning the cold-CNN exact-F1 distribution: a severe
    # failure, lower quartile, median, and strong case.
    ranked = sorted(
        per_utt,
        key=lambda row: row["models"]["cold_cnn"]["exact"]["f1"],
    )
    example_indices = sorted(set([0, len(ranked) // 4, len(ranked) // 2, len(ranked) - 1]))
    result = {
        "sample": {
            "csv": str(args.csv),
            "num_utterances": len(rows),
            "selection": "evenly spaced after sorting dev-clean by duration",
            "total_hours": sum(float(row["duration"]) for row in rows) / 3600,
        },
        "checkpoints": {
            "cold_cnn": str(args.cold_cnn),
            "rl_nll_mt_cnn": str(args.rl_cnn),
            "cold_transformer": str(args.cold_transformer),
            "rl_nll_mt_transformer": str(args.rl_transformer),
        },
        "aggregate_micro": {
            name: {label: _score(counts) for label, counts in tolerances.items()}
            for name, tolerances in aggregate.items()
        },
        "examples": [ranked[index] for index in example_indices],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["aggregate_micro"], indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
