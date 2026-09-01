#!/usr/bin/env python3
"""Evaluate important SpeechLLM boundary streams on TIMIT phone alignments.

TIMIT ships sample-accurate phone start times in its native ``.PHN`` files.
This audit maps them to the WavLM / wav2vec2 50 Hz grid with the same
floor-with-epsilon convention as the project's LibriSpeech phone targets. It
reports micro ordered one-to-one boundary precision, recall, and F1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio

RECIPE = Path(__file__).resolve().parent
PROJECT_ROOT = RECIPE.parents[3]
sys.path.insert(0, str(RECIPE))
sys.path.insert(0, str(PROJECT_ROOT))

from ctc_boundary_align import num_encoder_frames  # noqa: E402
from segmenter import Segmenter  # noqa: E402
from wavlm_phone_ctc import (  # noqa: E402
    PhoneCTCHead,
    load_wavlm,
    nonblank_run_boundaries as wavlm_phone_ctc_boundaries,
)
from speechbrain.integrations.huggingface.wav2vec2 import Wav2Vec2  # noqa: E402
from speechbrain.processing.features import InputNormalization  # noqa: E402

SAMPLE_RATE = 16000
FRAME_RATE_HZ = 50.0
SEEDS = (3407, 3408, 3409)
CNN_ROOT = RECIPE / "results/speechllm_segmenter_wavlm/cnn_first_order_ar_bigru_best_char_decoder/nll_mt"
TRANSFORMER_ROOT = RECIPE / "results/speechllm_segmenter_wavlm/fullprefix_transformer_ar_local64_bigru_best_char_decoder/nll_mt"
WAVLM_PHONE_CTC_CHECKPOINT = PROJECT_ROOT / "artifacts/models/wavlm_phone_ctc_tc100_seed3407/checkpoints/epoch_003.pt"


@dataclass(frozen=True)
class Utterance:
    utt_id: str
    wav: Path
    phn: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selected_epoch(train_log: Path) -> int:
    epochs = [int(match.group(1)) for line in train_log.read_text(errors="replace").splitlines() if (match := re.match(r"Epoch loaded: (\d+) - test", line))]
    if len(set(epochs)) != 1:
        raise ValueError(f"Expected one selected test epoch in {train_log}, got {epochs}")
    return epochs[0]


def _checkpoint_epoch(directory: Path) -> int | None:
    metadata = directory / "CKPT.yaml"
    match = re.search(r"^epoch:\s*(\d+)\s*$", metadata.read_text(), re.MULTILINE) if metadata.is_file() else None
    return int(match.group(1)) if match else None


def _selected_checkpoint(run_dir: Path) -> tuple[int, Path]:
    epoch = _selected_epoch(run_dir / "train_log.txt")
    candidates = [directory / "segmenter.ckpt" for directory in sorted((run_dir / "save").glob("CKPT*")) if _checkpoint_epoch(directory) == epoch and (directory / "segmenter.ckpt").is_file()]
    if len(candidates) != 1:
        raise ValueError(f"Expected one selected epoch-{epoch} segmenter under {run_dir}, got {candidates}")
    return epoch, candidates[0]


def _load_wave(path: Path) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(path)
    waveform = waveform.mean(0)
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"{path}: expected the native TIMIT 16 kHz rate, got {sample_rate}")
    return waveform


def _associated_phn(wav: Path) -> Path:
    for suffix in (".PHN", ".phn"):
        candidate = wav.with_suffix(suffix)
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No .PHN alignment next to {wav}")


def collect_timit(root: Path, split: str, include_sa: bool) -> list[Utterance]:
    """Find standard TIMIT waveforms and their native phone alignments."""
    split_dirs = [path for path in root.iterdir() if path.is_dir() and path.name.lower() == split]
    if len(split_dirs) != 1:
        raise FileNotFoundError(f"Expected one {split.upper()} directory under {root}")
    waves = sorted(path for path in split_dirs[0].rglob("*") if path.is_file() and path.suffix.lower() == ".wav")
    if not include_sa:
        waves = [path for path in waves if path.stem.lower() not in {"sa1", "sa2"}]
    utterances = [Utterance(path.relative_to(root).with_suffix("").as_posix(), path, _associated_phn(path)) for path in waves]
    if not utterances:
        raise ValueError(f"No TIMIT {split.upper()} utterances found under {root}")
    return utterances


def phone_starts_to_frames(phn: Path, num_frames: int) -> list[int]:
    """Map TIMIT's sample-index phone starts onto the project 50 Hz grid."""
    starts = []
    for line_number, line in enumerate(phn.read_text().splitlines(), start=1):
        fields = line.split()
        if len(fields) != 3:
            raise ValueError(f"{phn}:{line_number}: expected '<start> <end> <phone>'")
        sample_start, sample_end = (int(fields[0]), int(fields[1]))
        if sample_start < 0 or sample_end <= sample_start:
            raise ValueError(f"{phn}:{line_number}: invalid sample interval {fields[:2]}")
        # Same as int((samples / 16000) * 50 + 1e-6), used by the existing
        # TextGrid -> WavLM target generator.
        starts.append(min(max(int(sample_start * FRAME_RATE_HZ / SAMPLE_RATE + 1e-6), 0), num_frames - 1))
    if not starts:
        raise ValueError(f"No phone entries in {phn}")
    return sorted(set([0, *starts]))


def _match_count(predicted: list[int], reference: list[int], tolerance: int) -> int:
    """Maximum ordered one-to-one matches; duplicate near-cuts get no credit."""
    i = j = matches = 0
    while i < len(predicted) and j < len(reference):
        if abs(predicted[i] - reference[j]) <= tolerance:
            matches += 1
            i += 1
            j += 1
        elif predicted[i] < reference[j] - tolerance:
            i += 1
        else:
            j += 1
    return matches


def _empty_counts() -> dict[str, int]:
    return {"hits": 0, "predicted": 0, "reference": 0}


def _update(counts: dict[str, int], predicted: list[int], reference: list[int], tolerance: int) -> None:
    counts["hits"] += _match_count(predicted, reference, tolerance)
    counts["predicted"] += len(predicted)
    counts["reference"] += len(reference)


def _score(counts: dict[str, int]) -> dict[str, float | int]:
    precision = counts["hits"] / counts["predicted"] if counts["predicted"] else 0.0
    recall = counts["hits"] / counts["reference"] if counts["reference"] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {**counts, "precision": precision, "recall": recall, "f1": f1}


def _new_counts(labels: list[str], tolerances: tuple[int, ...]) -> dict[str, dict[int, dict[str, int]]]:
    return {label: {tolerance: _empty_counts() for tolerance in tolerances} for label in labels}


def _load_cnn(checkpoint: Path, device: torch.device) -> Segmenter:
    model = Segmenter(input_dim=1024, backbone="cnn", hidden_dim=256, kernel_size=7, dropout=0.1, sampling_strategy="autoregressive", pooling="bigru_residual", pooling_input_dim=1024, pooling_hidden_dim=128, pooling_num_layers=1, pooling_dropout=0.0)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    return model.eval().to(device)


def _load_transformer(checkpoint: Path, device: torch.device) -> Segmenter:
    model = Segmenter(input_dim=1024, backbone="transformer_ar", hidden_dim=256, num_layers=4, nhead=4, ffn_dim=1024, dropout=0.0, ar_hidden_dim=256, ar_num_layers=4, ar_nhead=4, ar_ffn_dim=1024, ar_dropout=0.0, ar_max_positions=4096, ar_history_window=64, ar_cache_mode="preallocated", pooling="bigru_residual", pooling_input_dim=1024, pooling_hidden_dim=128, pooling_num_layers=1, pooling_dropout=0.0)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    return model.eval().to(device)


def _load_wavlm_phone_ctc(cache_dir: Path, device: torch.device):
    """Load the project's best trained WavLM phone-CTC checkpoint."""
    checkpoint = torch.load(WAVLM_PHONE_CTC_CHECKPOINT, map_location="cpu", weights_only=False)
    encoder = load_wavlm(checkpoint["wavlm_source"], str(cache_dir), device)
    head = PhoneCTCHead(**checkpoint["head_config"]).to(device)
    head.load_state_dict(checkpoint["head_state"], strict=True)
    return encoder, head.eval(), checkpoint


def _predict_learned(model: Segmenter, features: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
    if model.is_full_ar:
        return model.sample_boundaries(features, padding_mask, deterministic=True).boundaries
    return model.sample_boundary(model.boundary_logits(features, padding_mask), padding_mask, mode="argmax")


def _summary(per_seed: dict[str, dict[str, dict]]) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for tolerance_name in next(iter(per_seed.values())):
        out[tolerance_name] = {}
        for metric in ("precision", "recall", "f1"):
            values = [score[tolerance_name][metric] for score in per_seed.values()]
            out[tolerance_name][metric] = {"mean": statistics.mean(values), "sd": statistics.stdev(values) if len(values) > 1 else 0.0}
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timit-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--include-sa", action="store_true", help="Include repeated SA1/SA2 prompts.")
    parser.add_argument("--ssl-cache", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tolerances", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--max-utts", type=int, default=0)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    tolerances = tuple(sorted(set(args.tolerances)))
    if not tolerances or tolerances[0] < 0:
        raise ValueError("--tolerances must be non-negative frame counts")
    if not args.timit_root.is_dir() or not args.ssl_cache.is_dir():
        raise FileNotFoundError("--timit-root and --ssl-cache must both be directories")
    utterances = collect_timit(args.timit_root, args.split, args.include_sa)
    if args.max_utts:
        utterances = utterances[:args.max_utts]

    selected = {
        "wavlm_phone_ctc": {
            "checkpoint": str(WAVLM_PHONE_CTC_CHECKPOINT),
            "sha256": _sha256(WAVLM_PHONE_CTC_CHECKPOINT),
        },
        "cnn_first_order_ar_bigru": {},
        "transformer_ar_local64_bigru": {},
    }
    for label, root in (("cnn_first_order_ar_bigru", CNN_ROOT), ("transformer_ar_local64_bigru", TRANSFORMER_ROOT)):
        for seed in SEEDS:
            epoch, checkpoint = _selected_checkpoint(root / str(seed))
            selected[label][str(seed)] = {"epoch": epoch, "path": str(checkpoint), "sha256": _sha256(checkpoint)}
    if args.validate_only:
        print(json.dumps({"utterances": len(utterances), "selected_checkpoints": selected}, indent=2))
        return

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    use_bf16 = args.precision == "bf16" and device.type == "cuda"
    learned = {
        "cnn_first_order_ar_bigru": {seed: _load_cnn(Path(selected["cnn_first_order_ar_bigru"][str(seed)]["path"]), device) for seed in SEEDS},
        "transformer_ar_local64_bigru": {seed: _load_transformer(Path(selected["transformer_ar_local64_bigru"][str(seed)]["path"]), device) for seed in SEEDS},
    }
    phone_ctc_encoder, phone_ctc_head, phone_ctc_checkpoint = _load_wavlm_phone_ctc(args.ssl_cache, device)
    selected["wavlm_phone_ctc"].update(
        {
            "epoch": phone_ctc_checkpoint["epoch"],
            "dev_phone_error_rate": phone_ctc_checkpoint["metrics"]["phone_error_rate"],
            "training_target": phone_ctc_checkpoint["target_definition"],
        }
    )
    ssl = Wav2Vec2(source="microsoft/wavlm-large", output_norm=True, freeze=True, save_path=str(args.ssl_cache), device_map=device.type).eval()
    normalization = InputNormalization(norm_type="sentence").to(device).eval()
    labels = ["fixed_k5", "wavlm_phone_ctc"] + [f"{kind}_seed{seed}" for kind in learned for seed in SEEDS]
    counts = _new_counts(labels, tolerances)
    totals = {label: {"frames": 0, "segments": 0} for label in labels}
    collapsed_phone_starts = 0
    for start in range(0, len(utterances), args.batch_size):
        batch = utterances[start : start + args.batch_size]
        waves = [_load_wave(item.wav) for item in batch]
        lengths = torch.tensor([wave.numel() for wave in waves], device=device)
        frame_lengths = [num_encoder_frames(int(length)) for length in lengths.cpu().tolist()]
        max_samples = int(lengths.max())
        wav_batch = torch.zeros(len(batch), max_samples, device=device)
        for index, wave in enumerate(waves):
            wav_batch[index, : wave.numel()] = wave.to(device)
        relative_lengths = lengths.float() / max_samples
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            features = ssl(normalization(wav_batch, relative_lengths), relative_lengths)
            phone_wavs = torch.zeros_like(wav_batch)
            for index, wave in enumerate(waves):
                phone_wavs[index, : wave.numel()] = F.layer_norm(wave.to(device), wave.shape)
            phone_attention_mask = torch.arange(max_samples, device=device).unsqueeze(0) < lengths.unsqueeze(1)
            phone_features = phone_ctc_encoder(phone_wavs, attention_mask=phone_attention_mask.long()).last_hidden_state
            phone_logits = phone_ctc_head(phone_features)
        phone_frame_lengths = phone_ctc_encoder._get_feat_extract_output_lengths(lengths).long().cpu().tolist()
        if features.size(1) != max(frame_lengths) or phone_frame_lengths != frame_lengths:
            raise RuntimeError("WavLM frame count does not match the validated convolutional frame formula")
        frame_index = torch.arange(features.size(1), device=device).unsqueeze(0)
        padding_mask = frame_index >= torch.tensor(frame_lengths, device=device).unsqueeze(1)
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            predicted_learned = {f"{kind}_seed{seed}": _predict_learned(model, features, padding_mask).cpu() for kind, models in learned.items() for seed, model in models.items()}
        for index, item in enumerate(batch):
            length = frame_lengths[index]
            gold = phone_starts_to_frames(item.phn, length)
            collapsed_phone_starts += len(item.phn.read_text().splitlines()) - len(gold)
            predictions = {
                "fixed_k5": list(range(0, length, 5)),
                **{label: (boundary[index, :length] == 1).nonzero().flatten().tolist() for label, boundary in predicted_learned.items()},
            }
            # Ordinary CTC nonblank-state runs; leading silence belongs to the
            # unconditional segment at frame zero, matching the trained system.
            phone_boundary = wavlm_phone_ctc_boundaries(
                phone_logits[index, :length].argmax(dim=-1).cpu()
            )
            predictions["wavlm_phone_ctc"] = (phone_boundary == 1).nonzero().flatten().tolist()
            for label, predicted in predictions.items():
                totals[label]["frames"] += length
                totals[label]["segments"] += len(predicted)
                for tolerance in tolerances:
                    _update(counts[label][tolerance], predicted, gold, tolerance)
        completed = min(start + len(batch), len(utterances))
        if start == 0 or completed == len(utterances) or completed % 200 < args.batch_size:
            print(f"evaluated {completed}/{len(utterances)} utterances", flush=True)

    tolerance_names = {tol: "exact" if tol == 0 else f"tol_{20 * tol}ms" for tol in tolerances}
    systems: dict[str, dict] = {}
    for label in ("fixed_k5", "wavlm_phone_ctc"):
        systems[label] = {"kind": "deterministic", "n": 1, "per_seed": {"deterministic": {tolerance_names[tol]: _score(counts[label][tol]) for tol in tolerances}}}
        systems[label]["summary"] = _summary(systems[label]["per_seed"])
    for kind in learned:
        per_seed = {str(seed): {tolerance_names[tol]: _score(counts[f"{kind}_seed{seed}"][tol]) for tol in tolerances} for seed in SEEDS}
        systems[kind] = {"kind": "learned segmenter", "n": len(SEEDS), "per_seed": per_seed, "summary": _summary(per_seed)}
    for label, values in totals.items():
        system_label, seed_label = label.rsplit("_seed", 1) if "_seed" in label else (label, "deterministic")
        systems[system_label]["per_seed"][seed_label]["audio_frequency_hz"] = values["segments"] / values["frames"] * FRAME_RATE_HZ
    result = {
        "dataset": "TIMIT", "split": args.split.upper(), "utterances": len(utterances), "include_sa_prompts": args.include_sa,
        "reference": "native .PHN sample-index phone starts", "frame_rate_hz": FRAME_RATE_HZ,
        "collapsed_phone_starts_after_20ms_quantization": collapsed_phone_starts,
        "metric": "Micro precision/recall/F1 with maximum ordered one-to-one matches; exact is same 20 ms frame, tolerances permit ±20 ms per frame.",
        "prediction_rules": {"fixed_k5": "boundaries at frames 0, 5, 10, …", "wavlm_phone_ctc": "trained frozen-WavLM-Large phone-CTC head, ordinary argmax nonblank CTC state runs; blanks merge into adjacent segments"},
        "selected_checkpoints": selected, "systems": systems,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({name: value["summary"] for name, value in systems.items()}, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
