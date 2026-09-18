"""CPU integration checks for the continuation hooks and checkpoint semantics."""

import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import speechbrain as sb
from speechbrain.utils.checkpoints import Checkpointer

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "recipes/LibriSpeech/ASR/transformer"))
from adaptive_frequency import FrequencyConfig, FrequencyController
from train_speechllm_adaptive import AdaptiveSegmenterASR, RandomState
from train_speechllm_with_segmenter import SegmenterASR


def test_random_state_checkpoint_roundtrip(tmp_path):
    rng = RandomState()
    checkpointer = Checkpointer(tmp_path, recoverables={})
    controller = FrequencyController(FrequencyConfig())
    controller.observe(0, 3, 6, 10)
    checkpointer.add_recoverable("frequency", controller,
                                custom_save_hook=FrequencyController.save,
                                custom_load_hook=FrequencyController.load)
    checkpointer.add_recoverable("rng", rng, custom_save_hook=RandomState.save, custom_load_hook=RandomState.load)
    checkpointer.save_checkpoint(end_of_epoch=False)
    expected = (random.random(), np.random.rand(), torch.rand(1).item())
    controller.observe(500, 3, 6, 10)
    checkpointer.recover_if_possible()
    assert expected == (random.random(), np.random.rand(), torch.rand(1).item())
    assert controller.state["last_step"] == 0


def test_paired_validation_restores_training_state_before_save(tmp_path):
    b = AdaptiveSegmenterASR.__new__(AdaptiveSegmenterASR)
    b.frequency = FrequencyController(FrequencyConfig())
    b.step = 0
    b.optimizer_step = 0
    b.current_epoch = 1
    b._stage_started_at = 7
    b._rho_samples = [0.2]
    b._reward_std_samples = [0.01]
    b._step_validation_seconds_in_epoch = 0
    b._frequency_loaders = {"dev-clean": "clean", "dev-other": "other"}
    b._step_validation_enable = False
    b.device = "cpu"
    b.device_type = "cpu"
    b.modules = torch.nn.ModuleDict({"layer": torch.nn.Linear(1, 1)})
    b.modules.train()
    b.optimizer = SimpleNamespace(param_groups=[{"lr": 5e-6}, {"lr": 2e-5}])
    b.hparams = SimpleNamespace(
        output_folder=str(tmp_path), save_folder=str(tmp_path / "save"),
        source_optimizer_step=24000, continuation_smoke_pause_step=0,
        train_logger=SimpleNamespace(log_stats=lambda **kw: None),
    )
    saved_steps = []

    def save(**kwargs):
        assert b.step == b.optimizer_step  # must not be validation's reset zero
        assert b.modules.training
        assert kwargs["end_of_epoch"] is False
        saved_steps.append(b.step)

    b.checkpointer = SimpleNamespace(save_checkpoint=save)

    def validate(loader, epoch, enable):
        assert b.step == 0
        b.step = 999
        b._rho_samples = [1.0]
        b._reward_std_samples = []
        b.modules.eval()
        random.random(), np.random.rand(), torch.rand(2)
        b._frequency_metrics[b._frequency_split] = dict(
            WER=3.0 if loader == "clean" else 6.0, CER=1.0,
            rate_hz=10.0, loss=0.5, utterances=2)

    b._fit_valid = validate
    rng = RandomState.capture()
    expected = (random.random(), np.random.rand(), torch.rand(1).item())
    RandomState.restore(rng)
    b._run_step_validation("initial")
    assert expected == (random.random(), np.random.rand(), torch.rand(1).item())
    for step in (500, 1000):
        b.step = b.optimizer_step = step
        b._run_step_validation("interval")
    assert saved_steps == [0, 500, 1000]
    assert b._rho_samples == [0.2] and b._reward_std_samples == [0.01]
    assert b._stage_started_at == 7
    assert b._rate_band() == (0.15, 9.5 / 50)
    events = [json.loads(x) for x in (tmp_path / "frequency_events.jsonl").read_text().splitlines()]
    assert events[-1]["action"] == "lower_target"


def test_changed_band_reaches_the_existing_grpo_penalty():
    b = AdaptiveSegmenterASR.__new__(AdaptiveSegmenterASR)
    b.frequency = FrequencyController(FrequencyConfig())
    b.frequency.observe(0, 3, 6, 10)
    b.hparams = SimpleNamespace(rate_mode="band", rho_star=0.2, lambda_cap=1.0,
                               lambda_press=0.02, rho_floor=0.08, lambda_floor=0.0,
                               rate_channel="reward")
    boundary = torch.tensor([[1, 0, 0, 0, 0] * 4])
    mask = torch.zeros_like(boundary, dtype=torch.bool)
    before, _ = b._sampled_rate_terms([boundary] * 4, mask)
    b.frequency.observe(500, 3, 6, 10)
    b.frequency.observe(1000, 3, 6, 10)
    after, _ = b._sampled_rate_terms([boundary] * 4, mask)
    assert torch.all(before == 0)
    assert torch.all(after > before)
    b.frequency.observe(1500, 3, 7, 10)
    held, _ = b._sampled_rate_terms([boundary] * 4, mask)
    assert torch.equal(held, after)
    assert b._rate_channels() == frozenset({"reward"})


def test_resume_does_not_reset_optimizer_or_ema(monkeypatch):
    b = AdaptiveSegmenterASR.__new__(AdaptiveSegmenterASR)
    b.frequency = FrequencyController(FrequencyConfig())
    b.frequency.observe(0, 3, 6, 10)
    b.frequency.observe(500, 3, 6.5, 10)
    b.optimizer_step = 500
    b.optimizer_step_limit = 8000
    b.distributed_launch = False
    b.data_parallel_backend = False
    b.hparams = SimpleNamespace(lr_segmenter=5e-6, lr_decoder=2e-5, rate_channel="reward")
    b.optimizer = SimpleNamespace(param_groups=[{"lr": 5e-5}, {"lr": 2e-4}])
    b.modules = torch.nn.ModuleDict({name: torch.nn.Linear(1, 1)
                                   for name in ("segmenter", "proj", "llm", "ssl")})
    b.modules.ssl.requires_grad_(False)
    monkeypatch.setattr(SegmenterASR, "on_fit_start", lambda self: None)
    b.on_fit_start()
    assert b._last_step_validation == 500
    assert b.frequency.state["ema_other_wer"] == pytest.approx(6.1)
    assert [g["lr"] for g in b.optimizer.param_groups] == [5e-6, 2e-5]
