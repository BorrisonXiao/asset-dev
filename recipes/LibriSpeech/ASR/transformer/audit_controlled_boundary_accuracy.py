#!/usr/bin/env python3
"""Measure each controlled setup's boundary agreement with char-CTC targets.

The audit is evaluation-only. Learned systems use the exact segmenter checkpoint
selected for their reported WER; deterministic boundary sources are constructed
directly. Scores are micro precision/recall/F1 on the full dev-clean split, with
ordered one-to-one matching at exact and +/-1-frame tolerance.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio

RECIPE = Path(__file__).resolve().parent
sys.path.insert(0, str(RECIPE))
sys.path.insert(0, str(RECIPE.parents[3]))

from segmenter import Segmenter  # noqa: E402

from speechbrain.integrations.huggingface.wav2vec2 import Wav2Vec2  # noqa: E402
from speechbrain.processing.features import InputNormalization  # noqa: E402

SEEDS = (3407, 3408, 3409)
RESULTS = RECIPE / "results"


@dataclass(frozen=True)
class SystemSpec:
    label: str
    run_pattern: str
    encoder: str
    input_dim: int
    backbone: str
    sampling: str = "bernoulli"


SYSTEMS = (
    SystemSpec(
        "CNN segmenter on wav2vec2 -> pool WavLM - char decoder",
        "speechllm_segmenter_wavlm/hybrid_w2v2_segmenter_controlled/"
        "oracle_char_decoder/{seed}",
        "wav2vec2",
        768,
        "cnn",
    ),
    SystemSpec(
        "CNN segmenter on wav2vec2 -> pool WavLM - scratch decoder",
        "speechllm_segmenter_wavlm/hybrid_w2v2_segmenter_controlled/"
        "scratch_decoder/{seed}",
        "wav2vec2",
        768,
        "cnn",
    ),
    SystemSpec(
        "CNN segmenter on WavLM -> pool WavLM - Bernoulli",
        "speechllm_segmenter_wavlm/oracleclose_bestinit/bernoulli/{seed}",
        "wavlm",
        1024,
        "cnn",
    ),
    SystemSpec(
        "CNN segmenter on WavLM -> pool WavLM - first-order AR",
        "speechllm_segmenter_wavlm/oracleclose_bestinit/autoregressive/{seed}",
        "wavlm",
        1024,
        "cnn",
        "autoregressive",
    ),
    SystemSpec(
        "Transformer segmenter on WavLM -> pool WavLM - shared warm-up",
        "speechllm_segmenter_wavlm/multiseed_shared_warmup_onpolicy/"
        "nll_mt/transformer/{seed}",
        "wavlm",
        1024,
        "transformer",
    ),
)

BASELINES = (
    "Char-aligned baseline",
    "Phone-aligned baseline",
    "Fixed k=5 baseline",
    "No-downsampling baseline",
)


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _selected_epoch(train_log: Path) -> int:
    for line in train_log.read_text(errors="replace").splitlines():
        match = re.match(r"Epoch loaded: (\d+) - test", line)
        if match:
            return int(match.group(1))
    raise ValueError(f"No selected test epoch found in {train_log}")


def _checkpoint_epoch(checkpoint: Path) -> int | None:
    metadata = checkpoint / "CKPT.yaml"
    if not metadata.is_file():
        return None
    match = re.search(r"^epoch:\s*(\d+)\s*$", metadata.read_text(), re.MULTILINE)
    return int(match.group(1)) if match else None


def _selected_segmenter(run: Path) -> tuple[int, Path]:
    epoch = _selected_epoch(run / "train_log.txt")
    candidates = [
        checkpoint / "segmenter.ckpt"
        for checkpoint in sorted((run / "save").glob("CKPT*"))
        if _checkpoint_epoch(checkpoint) == epoch
        and (checkpoint / "segmenter.ckpt").is_file()
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one epoch-{epoch} segmenter under {run}, found {candidates}"
        )
    return epoch, candidates[0]


def _models_for_encoder(
    encoder: str, device: torch.device
) -> tuple[dict[str, dict[int, Segmenter]], dict[str, dict[int, dict[str, str]]]]:
    models: dict[str, dict[int, Segmenter]] = {}
    checkpoints: dict[str, dict[int, dict[str, str]]] = {}
    for spec in SYSTEMS:
        if spec.encoder != encoder:
            continue
        models[spec.label] = {}
        checkpoints[spec.label] = {}
        for seed in SEEDS:
            run = RESULTS / spec.run_pattern.format(seed=seed)
            epoch, checkpoint = _selected_segmenter(run)
            model = Segmenter(
                input_dim=spec.input_dim,
                backbone=spec.backbone,
                hidden_dim=256,
                kernel_size=7,
                num_layers=4,
                nhead=4,
                ffn_dim=1024,
                dropout=0.1,
                sampling_strategy=spec.sampling,
            )
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            models[spec.label][seed] = model.eval().to(device)
            checkpoints[spec.label][seed] = {
                "epoch": str(epoch),
                "path": str(checkpoint),
            }
    return models, checkpoints


def _match_count(pred: list[int], gold: list[int], tolerance: int) -> int:
    """Maximum ordered one-to-one matching within ``tolerance`` frames."""
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


def _empty_counts() -> dict[str, int]:
    return {"tp": 0, "pred": 0, "gold": 0}


def _add(
    counts: dict[str, int], pred: list[int], gold: list[int], tolerance: int
) -> None:
    counts["tp"] += _match_count(pred, gold, tolerance)
    counts["pred"] += len(pred)
    counts["gold"] += len(gold)


def _score(counts: dict[str, int]) -> dict[str, float | int]:
    precision = counts["tp"] / counts["pred"] if counts["pred"] else 0.0
    recall = counts["tp"] / counts["gold"] if counts["gold"] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {**counts, "precision": precision, "recall": recall, "f1": f1}


def _new_aggregate(labels: tuple[str, ...], seeds: tuple[int, ...]):
    return {
        label: {
            seed: {name: _empty_counts() for name in ("exact", "tol1")}
            for seed in seeds
        }
        for label in labels
    }


def _baseline_scores(
    rows: list[dict[str, str]], char_dir: Path, phone_dir: Path
) -> dict[str, dict[str, dict[str, float | int]]]:
    aggregate = _new_aggregate(BASELINES, (0,))
    for row in rows:
        gold_tensor = torch.load(
            char_dir / f"{row['ID']}.pt", map_location="cpu", weights_only=True
        ).view(-1)
        phone_tensor = torch.load(
            phone_dir / f"{row['ID']}.pt", map_location="cpu", weights_only=True
        ).view(-1)
        gold = (gold_tensor == 1).nonzero().flatten().tolist()
        length = gold_tensor.numel()
        predictions = {
            "Char-aligned baseline": gold,
            "Phone-aligned baseline": (
                (phone_tensor[:length] == 1).nonzero().flatten().tolist()
            ),
            "Fixed k=5 baseline": list(range(5, length, 5)),
            "No-downsampling baseline": list(range(1, length)),
        }
        for label, pred in predictions.items():
            for tolerance, name in ((0, "exact"), (1, "tol1")):
                _add(aggregate[label][0][name], pred, gold, tolerance)
    return {
        label: {name: _score(counts) for name, counts in per_seed[0].items()}
        for label, per_seed in aggregate.items()
    }


def _load_wave(row: dict[str, str], data_root: Path) -> torch.Tensor:
    path = Path(row["wav"].replace("$data_root", str(data_root)))
    waveform, sample_rate = torchaudio.load(path)
    waveform = waveform.mean(0)
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
    return waveform


def _evaluate_encoder(
    encoder: str,
    rows: list[dict[str, str]],
    data_root: Path,
    char_dir: Path,
    source: str,
    cache: Path,
    batch_size: int,
    device: torch.device,
):
    models, checkpoints = _models_for_encoder(encoder, device)
    aggregate = _new_aggregate(tuple(models), SEEDS)
    ssl = Wav2Vec2(
        source=source,
        output_norm=True,
        freeze=True,
        save_path=str(cache),
        device_map=device.type,
    ).eval()
    normalize = InputNormalization(norm_type="sentence").to(device).eval()

    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        waves = [_load_wave(row, data_root) for row in batch_rows]
        golds = [
            torch.load(
                char_dir / f"{row['ID']}.pt",
                map_location="cpu",
                weights_only=True,
            ).view(-1)
            for row in batch_rows
        ]
        wav_lengths = torch.tensor([wav.numel() for wav in waves], device=device)
        max_wav = int(wav_lengths.max())
        wav_batch = torch.zeros(len(waves), max_wav, device=device)
        for index, waveform in enumerate(waves):
            wav_batch[index, : waveform.numel()] = waveform.to(device)
        relative_lengths = wav_lengths.float() / max_wav
        with torch.inference_mode():
            features = ssl(
                normalize(wav_batch, relative_lengths), relative_lengths
            )

        target_lengths = torch.tensor(
            [target.numel() for target in golds], device=device
        )
        if int(target_lengths.max()) != features.size(1):
            raise ValueError(
                f"{encoder} returned {features.size(1)} frames; longest target has "
                f"{int(target_lengths.max())} frames"
            )
        frame_index = torch.arange(features.size(1), device=device).unsqueeze(0)
        padding_mask = frame_index >= target_lengths.unsqueeze(1)

        with torch.inference_mode():
            for label, seed_models in models.items():
                for seed, model in seed_models.items():
                    logits = model.boundary_logits(features, padding_mask)
                    boundary = model.sample_boundary(
                        logits, padding_mask, mode="argmax"
                    ).cpu()
                    for item_index, gold_tensor in enumerate(golds):
                        length = gold_tensor.numel()
                        gold = (gold_tensor == 1).nonzero().flatten().tolist()
                        pred = (
                            (boundary[item_index, :length] == 1)
                            .nonzero()
                            .flatten()
                            .tolist()
                        )
                        for tolerance, name in ((0, "exact"), (1, "tol1")):
                            _add(aggregate[label][seed][name], pred, gold, tolerance)
        if start % (50 * batch_size) == 0:
            print(f"{encoder}: {min(start + batch_size, len(rows))}/{len(rows)}")

    scored = {
        label: {
            seed: {name: _score(counts) for name, counts in tolerances.items()}
            for seed, tolerances in per_seed.items()
        }
        for label, per_seed in aggregate.items()
    }
    return scored, checkpoints


def _summary(per_seed: dict[int, dict[str, dict]]) -> dict[str, dict[str, float]]:
    output = {}
    for tolerance in ("exact", "tol1"):
        output[tolerance] = {}
        for metric in ("precision", "recall", "f1"):
            values = [per_seed[seed][tolerance][metric] for seed in sorted(per_seed)]
            output[tolerance][metric] = {
                "mean": statistics.mean(values),
                "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--char-dir", type=Path, required=True)
    parser.add_argument("--phone-dir", type=Path, required=True)
    parser.add_argument("--wav2vec2-cache", type=Path, required=True)
    parser.add_argument("--wavlm-cache", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = _load_rows(args.csv)
    selected = {
        spec.label: {
            seed: {
                "epoch": epoch,
                "path": str(path),
            }
            for seed in SEEDS
            for epoch, path in [
                _selected_segmenter(
                    RESULTS / spec.run_pattern.format(seed=seed)
                )
            ]
        }
        for spec in SYSTEMS
    }
    for directory in (args.data_root, args.char_dir, args.phone_dir):
        if not directory.exists():
            raise FileNotFoundError(directory)
    if args.validate_only:
        print(json.dumps({"utterances": len(rows), "selected": selected}, indent=2))
        return

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    baselines = _baseline_scores(rows, args.char_dir, args.phone_dir)
    learned: dict[str, dict[int, dict[str, dict]]] = {}
    checkpoints = {}
    for encoder, source, cache in (
        ("wav2vec2", "facebook/wav2vec2-base-960h", args.wav2vec2_cache),
        ("wavlm", "microsoft/wavlm-large", args.wavlm_cache),
    ):
        scores, encoder_checkpoints = _evaluate_encoder(
            encoder,
            rows,
            args.data_root,
            args.char_dir,
            source,
            cache,
            args.batch_size,
            device,
        )
        learned.update(scores)
        checkpoints.update(encoder_checkpoints)
        torch.cuda.empty_cache()

    systems = {
        label: {
            "kind": "deterministic boundary source",
            "n": 1,
            "per_seed": {"deterministic": scores},
            "summary": _summary({0: scores}),
        }
        for label, scores in baselines.items()
    }
    systems.update(
        {
            label: {
                "kind": "learned segmenter",
                "n": len(per_seed),
                "per_seed": {str(seed): scores for seed, scores in per_seed.items()},
                "summary": _summary(per_seed),
            }
            for label, per_seed in learned.items()
        }
    )
    result = {
        "split": "dev-clean",
        "utterances": len(rows),
        "reference": str(args.char_dir),
        "metric": (
            "Micro boundary precision/recall/F1 using ordered one-to-one matches; "
            "exact requires the same frame and tol1 allows +/-1 encoder frame."
        ),
        "selected_checkpoints": checkpoints,
        "systems": systems,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({label: value["summary"] for label, value in systems.items()}, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
