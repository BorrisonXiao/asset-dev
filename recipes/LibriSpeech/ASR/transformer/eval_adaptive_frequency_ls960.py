#!/usr/bin/env python3
"""Test final continuation checkpoints using their frozen training implementation."""

import json
import logging
import os
from pathlib import Path
import sys

snapshot = Path(os.environ["ADAPTIVE_EVAL_SOURCE"]).resolve(strict=True)
recipe = snapshot / "recipes/LibriSpeech/ASR/transformer"
# The current working tree must not shadow the frozen recipe or SpeechBrain.
sys.path[:0] = [str(recipe), str(snapshot)]

import speechbrain as sb
from train_speechllm_with_segmenter import SegmenterASR, main

logger = logging.getLogger(__name__)


def configure(hparams):
    if not hparams["eval_only"] or hparams["segmenter_mode"] != "joint":
        raise ValueError("This entry point is test-only")
    hparams["eval_checkpoint_step"] = 8000
    hparams["skip_final_test"] = False
    checkpoints = hparams["checkpointer"].find_checkpoints()
    if len(checkpoints) != 1 or checkpoints[0].meta["optimizer_step"] != 8000:
        raise ValueError("Require a view containing exactly the step-8000 checkpoint")
    hparams["evaluated_checkpoint"] = str(checkpoints[0].path.resolve())
    output = Path(hparams["output_folder"]).resolve()
    if output in Path(hparams["evaluated_checkpoint"]).parents:
        raise ValueError("Evaluation outputs must be separate from training")
    expected_splits = {"test-clean", "test-other"}
    if {Path(p).stem for p in hparams["test_csv"]} != expected_splits:
        raise ValueError("Evaluate both complete LibriSpeech test sets")


class TestOnlyASR(SegmenterASR):
    def fit(self, *args, **kwargs):
        raise RuntimeError("Training is disabled in this entry point")

    def on_evaluate_start(self, max_key=None, min_key=None):
        super().on_evaluate_start(max_key=max_key, min_key=min_key)
        # The continuation checkpoint stores a training minibatch counter.
        # Evaluation loss averaging must start from zero on each test split.
        self.step = 0
        self._test_batches = 0

    def evaluate_batch(self, batch, stage):
        result = super().evaluate_batch(batch, stage)
        self._test_batches += 1
        if self._test_batches % 50 == 0:
            logger.info("Test progress: split=%s batches=%d utterances=%d",
                        self.hparams.evaluation_split, self._test_batches,
                        len(self._rho_samples))
        return result

    def on_stage_end(self, stage, stage_loss, epoch):
        super().on_stage_end(stage, stage_loss, epoch)
        if stage != sb.Stage.TEST:
            raise RuntimeError("Unexpected non-test stage")
        row = {
            "split": self.hparams.evaluation_split,
            "checkpoint": self.hparams.evaluated_checkpoint,
            "continuation_optimizer_step": 8000,
            "source_optimizer_step": 24000,
            "WER": self.wer_metric.summarize("error_rate"),
            "CER": self.cer_metric.summarize("error_rate"),
            "utterances": len(self._rho_samples),
            "rate_hz_50rho": 50.0 * sum(self._rho_samples) / len(self._rho_samples),
            "rate_definition": "50 times the mean per-utterance kept-frame ratio",
        }
        destination = Path(self.hparams.output_folder) / "test_metrics.jsonl"
        with destination.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        logger.info("Full-precision test result: %s", json.dumps(row))


if __name__ == "__main__":
    main(brain_class=TestOnlyASR, configure_hparams=configure)
