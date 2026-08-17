#!/usr/bin/env python3
"""Evaluate an oracle-trained decoder with oracle/cold/RL boundaries.

This script is deliberately evaluation-only.  It loads the projection and LoRA
weights from one decoder checkpoint, swaps only the boundary source, and reports
WER plus exact/near-boundary agreement on the same dataset and decode settings.
No checkpoint is copied or modified.
"""

from __future__ import annotations

# The recipe and repository roots must be on sys.path before importing the
# in-tree SpeechBrain package and sibling recipe modules.
# ruff: noqa: E402, I001

import argparse
import hashlib
import json
import sys
from pathlib import Path

RECIPE = Path(__file__).resolve().parent
sys.path.insert(0, str(RECIPE))
sys.path.insert(0, str(RECIPE.parents[3]))

import torch
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.utils.distributed import if_main_process

from segment_pooling import lengths_to_padding_mask  # noqa: E402
from train_speechllm import dataio_prepare  # noqa: E402
from train_speechllm_with_segmenter import SegmenterASR  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _match_count(pred: list[int], gold: list[int], tolerance: int) -> int:
    """Maximum ordered one-to-one boundary matches within a frame tolerance."""
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


def _prf(counts: dict[str, int]) -> dict[str, float | int]:
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


class BoundarySwapASR(SegmenterASR):
    """Segmenter ASR whose evaluation boundary source is externally selected."""

    boundary_mode = "oracle"

    def on_stage_start(self, stage, epoch):
        super().on_stage_start(stage, epoch)
        self.boundary_counts = {
            label: {"tp": 0, "pred": 0, "gold": 0}
            for label in ("exact", "tol1", "tol2")
        }

    def _track_rho_from_boundary(self, boundary, pad_mask):
        """Count frame 0 once, matching the pooler's actual segment semantics."""
        valid = ~pad_mask
        frame_index = torch.arange(boundary.size(1), device=boundary.device).unsqueeze(0)
        later_starts = (boundary == 1) & valid & (frame_index > 0)
        n_seg = later_starts.sum(dim=1).float() + valid.any(dim=1).float()
        n_fr = valid.sum(dim=1).float().clamp(min=1.0)
        self._rho_samples.extend((n_seg / n_fr).tolist())

    def _update_boundary_metrics(self, boundary, target, pad_mask):
        for row in range(boundary.size(0)):
            valid_len = int((~pad_mask[row]).sum().item())
            pred = (
                (boundary[row, :valid_len] == 1).nonzero().flatten().tolist()
            )
            gold = ((target[row, :valid_len] == 1).nonzero().flatten().tolist())
            for tolerance, label in ((0, "exact"), (1, "tol1"), (2, "tol2")):
                counts = self.boundary_counts[label]
                counts["tp"] += _match_count(pred, gold, tolerance)
                counts["pred"] += len(pred)
                counts["gold"] += len(gold)

    def compute_forward(self, batch, stage):
        if stage == sb.Stage.TRAIN:
            raise RuntimeError("BoundarySwapASR is evaluation-only.")
        batch = batch.to(self.device)
        feats, feat_lens = self._encoder_features(batch)
        time = feats.size(1)
        pad_mask = lengths_to_padding_mask(feat_lens, time)

        target, _ = batch.boundary_target
        target = target.long()
        if target.size(1) != time:
            raise ValueError(
                f"Boundary target has {target.size(1)} frames, WavLM returned {time}."
            )
        target = target.masked_fill(pad_mask, -1)

        if self.boundary_mode == "oracle":
            boundary = target
        elif self.boundary_mode in ("cold", "rl"):
            logits = self.modules.segmenter.boundary_logits(feats, pad_mask)
            boundary = self.modules.segmenter.sample_boundary(
                logits, pad_mask, mode="argmax"
            )
        else:
            raise ValueError(f"Unknown boundary mode: {self.boundary_mode}")

        self._update_boundary_metrics(boundary, target, pad_mask)
        decoded = self._run_decoder(
            feats, pad_mask, boundary, batch, want_hyps=True
        )
        return {
            "mode": "joint",
            "pad_mask": pad_mask,
            "argmax_boundary": boundary,
            "llm_logits": decoded["llm_logits"],
            "seg_pad": decoded["seg_pad"],
            "hyps": decoded["hyps"],
            "is_warmup": False,
        }


def _require_checkpoint(directory: Path, names: tuple[str, ...]) -> None:
    if not directory.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {directory}")
    missing = [name for name in names if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{directory} is missing: {', '.join(missing)}")


def _load_decoder(brain: BoundarySwapASR, directory: Path) -> dict[str, str]:
    _require_checkpoint(directory, ("llm.ckpt", "proj.ckpt", "normalize.ckpt"))
    # AdaptedModel checkpoints intentionally contain trainable LoRA weights only.
    brain.modules.llm.loader(str(directory / "llm.ckpt"), True)
    brain.modules.proj.load_state_dict(
        torch.load(directory / "proj.ckpt", map_location="cpu", weights_only=True)
    )
    # InputNormalization persists running statistics through SpeechBrain's
    # custom checkpoint hook; these values are attributes, not parameters or
    # buffers accepted by ``load_state_dict``.
    brain.modules.normalize._load(str(directory / "normalize.ckpt"), True)
    return {
        name: _sha256(directory / name)
        for name in ("llm.ckpt", "proj.ckpt", "normalize.ckpt")
    }


def _load_segmenter(brain: BoundarySwapASR, path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Segmenter checkpoint does not exist: {path}")
    brain.modules.segmenter.load_state_dict(
        torch.load(path, map_location="cpu", weights_only=True)
    )
    return _sha256(path)


def _parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--decoder-checkpoint", type=Path, required=True)
    parser.add_argument("--cold-segmenter", type=Path, required=True)
    parser.add_argument("--rl-segmenter", type=Path, required=True)
    parser.add_argument("--backbone", choices=("cnn", "transformer"), required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["dev-clean"])
    parser.add_argument(
        "--max-utts",
        type=int,
        default=0,
        help="Debug-only cap per split; 0 evaluates the full split.",
    )
    parser.add_argument("--summary", type=Path, required=True)
    custom, remaining = parser.parse_known_args()
    hparams_file, run_opts, overrides = sb.parse_arguments(remaining)
    return custom, hparams_file, run_opts, overrides


def main() -> None:
    args, hparams_file, run_opts, overrides = _parse_args()
    with open(hparams_file, encoding="utf-8") as handle:
        hparams = load_hyperpyyaml(handle, overrides)

    if hparams["segmenter_mode"] != "joint":
        raise ValueError("Use --segmenter_mode joint for boundary-swap evaluation.")
    if hparams["segmenter_backbone"] != args.backbone:
        raise ValueError(
            f"CLI backbone {args.backbone} != hparams {hparams['segmenter_backbone']}"
        )
    if hparams["ssl_hub"] != "microsoft/wavlm-large":
        raise ValueError(f"Expected WavLM Large, got {hparams['ssl_hub']}")

    # Reuse the exact manifests from the oracle training run. This avoids any
    # data-preparation mutation and guarantees identical utterance membership.
    hparams["train_csv"] = str(args.manifest_root / "train.csv")
    hparams["valid_csv"] = str(args.manifest_root / "dev-clean.csv")
    hparams["test_csv"] = [
        str(args.manifest_root / f"{split}.csv")
        for split in ("test-clean", "test-other", "dev-other")
    ]
    for csv_path in [hparams["train_csv"], hparams["valid_csv"], *hparams["test_csv"]]:
        if not Path(csv_path).is_file():
            raise FileNotFoundError(f"Missing manifest: {csv_path}")

    tokenizer = hparams["llm"].tokenizer
    train_data, valid_data, test_datasets, tokenizer, _, _ = dataio_prepare(
        hparams, tokenizer
    )
    del train_data
    datasets = {"dev-clean": valid_data, **test_datasets}
    unknown = sorted(set(args.splits) - set(datasets))
    if unknown:
        raise ValueError(f"Unknown splits {unknown}; available: {sorted(datasets)}")
    if args.max_utts > 0:
        datasets = {
            name: dataset.filtered_sorted(select_n=args.max_utts)
            for name, dataset in datasets.items()
        }

    brain = BoundarySwapASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=None,
    )
    brain.tokenizer = tokenizer
    brain.txt_embedding = brain.raw_modules.llm.model.get_input_embeddings()
    brain.current_epoch = 1
    for parameter in brain.modules.ssl.parameters():
        parameter.requires_grad = False
    brain.modules.eval()

    decoder_hashes = _load_decoder(brain, args.decoder_checkpoint)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    wer_dir = args.summary.parent / "wer_results"
    wer_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "decoder_checkpoint": str(args.decoder_checkpoint),
        "decoder_sha256": decoder_hashes,
        "backbone": args.backbone,
        "decode": {
            "max_decode_ratio": float(hparams["max_decode_ratio"]),
            "test_batch_size": int(hparams["test_batch_size"]),
            "max_utts": args.max_utts,
        },
        "cells": {},
    }

    boundary_specs = (
        ("oracle", None),
        ("cold", args.cold_segmenter),
        ("rl", args.rl_segmenter),
    )
    for mode, checkpoint in boundary_specs:
        segmenter_hash = None
        if checkpoint is not None:
            segmenter_hash = _load_segmenter(brain, checkpoint)
        brain.boundary_mode = mode
        result["cells"][mode] = {
            "segmenter_checkpoint": str(checkpoint) if checkpoint else None,
            "segmenter_sha256": segmenter_hash,
            "splits": {},
        }
        for split in args.splits:
            wer_path = wer_dir / f"wer_{mode}_{split}.txt"
            brain.hparams.test_wer_file = str(wer_path)
            brain.evaluate(
                datasets[split],
                test_loader_kwargs=hparams["test_dataloader_opts"],
            )
            rho = torch.tensor(brain._rho_samples)
            result["cells"][mode]["splits"][split] = {
                "WER": float(brain.wer_metric.summarize("error_rate")),
                "CER": float(brain.cer_metric.summarize("error_rate")),
                "rho_mean": float(rho.mean()) if rho.numel() else None,
                "rho_std": float(rho.std()) if rho.numel() > 1 else 0.0,
                "boundary": {
                    label: _prf(counts)
                    for label, counts in brain.boundary_counts.items()
                },
                "wer_file": str(wer_path),
            }
            if if_main_process():
                args.summary.write_text(json.dumps(result, indent=2) + "\n")
        torch.cuda.empty_cache()

    print(json.dumps(result, indent=2))
    print(f"Wrote {args.summary}")


if __name__ == "__main__":
    main()
