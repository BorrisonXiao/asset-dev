"""The controller is pure Python: no GPU or model downloads required."""

import importlib.util
from pathlib import Path
import sys

import pytest

MODULE = Path(__file__).resolve().parents[2] / "recipes/LibriSpeech/ASR/transformer/adaptive_frequency.py"
SPEC = importlib.util.spec_from_file_location("adaptive_frequency", MODULE)
MOD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)
FrequencyConfig = MOD.FrequencyConfig
FrequencyController = MOD.FrequencyController


def controller(**kwargs):
    c = FrequencyController(FrequencyConfig(**kwargs))
    c.observe(0, 3.0, 6.0, 10.0)
    return c


def test_two_passing_checks_then_hold_with_penalty():
    c = controller()
    assert c.observe(500, 3, 6, 10)["action"] == "hold"
    result = c.observe(1000, 3, 6, 10)
    assert result["action"] == "lower_target"
    assert c.rate_band == (0.15, 9.5 / 50)
    bad = c.observe(1500, 3, 7, 9.5)
    assert bad["ema_other_wer"] == pytest.approx(6.2)
    assert not bad["quality_ok"] and not bad["training_done"]
    assert c.rate_band == (0.15, 9.5 / 50)
    for step in range(2000, 8001, 500):
        result = c.observe(step, 3, 7, 12)
        assert c.target_hz == 9.5
        assert result["training_done"] == (step == 8000)


def test_rate_must_reach_target_and_streak_resets():
    c = controller()
    c.observe(500, 3, 6, 10)
    assert c.observe(1000, 3, 6, 10.01)["advance_streak"] == 0
    assert c.observe(1500, 3, 6, 10)["action"] == "hold"
    assert c.observe(2000, 3, 6, 10)["action"] == "lower_target"
    assert c.observe(2500, 3, 6, 9.5)["action"] == "hold"
    assert c.observe(3000, 3, 6, 9.5)["action"] == "lower_target"


@pytest.mark.parametrize("clean,other", [(3.11, 6), (3, 6.11)])
def test_raw_guards_even_when_ema_passes(clean, other):
    c = controller()
    c.observe(500, 3, 6, 10)
    result = c.observe(1000, clean, other, 10)
    assert result["ema_other_wer"] < 6.1
    assert not result["quality_ok"] and c.target_hz == 10


def test_ema_must_recover_with_original_reference():
    c = controller()
    c.observe(500, 3, 8, 10)
    assert not c.observe(1000, 3, 6, 10)["quality_ok"]
    for step in range(1500, 8001, 500):
        result = c.observe(step, 3, 6, 10)
    assert result["quality_ok"]
    assert c.state["initial_other_wer"] == 6
    assert c.target_hz == 9.5


def test_floor_and_final_step_do_not_stop_early_or_lower_unused_target():
    c = controller(min_target_hz=9.5, max_steps=2500)
    for step in range(500, 2501, 500):
        result = c.observe(step, 3, 6, 9.5)
        assert result["training_done"] == (step == 2500)
    assert c.target_hz == 9.5
    c = controller(max_steps=1000)
    c.observe(500, 3, 6, 10)
    assert c.observe(1000, 3, 6, 10)["training_done"]
    assert c.target_hz == 10


@pytest.mark.parametrize("arm,target", [("original", 12.5), ("fixed_lower", 9.5)])
def test_controls_never_change_target(arm, target):
    c = controller(arm=arm)
    for step in range(500, 8001, 500):
        c.observe(step, 3, 6, 9)
        assert c.target_hz == target


def test_best_checkpoint_requires_quality_but_not_current_target():
    c = controller()
    assert c.state["best_step"] == 0
    c.observe(500, 3, 6, 9.8)
    assert c.state["best_step"] == 0
    c.observe(1000, 3, 6, 9.7)
    assert c.state["best_step"] == 1000
    assert c.target_hz == 9.5
    assert c.observe(1500, 3, 6, 9.6)["selected"]
    assert not c.observe(2000, 3, 7, 8)["selected"]


def test_resume_identical_and_reject_changed_configuration(tmp_path):
    c = controller()
    c.observe(500, 3, 6, 10)
    path = tmp_path / "controller.ckpt"
    c.save(path)
    resumed = FrequencyController(c.config)
    resumed.load(path)
    assert c.observe(1000, 3, 6, 10) == resumed.observe(1000, 3, 6, 10)
    changed = FrequencyController(FrequencyConfig(ema_alpha=0.3))
    with pytest.raises(ValueError, match="configuration changed"):
        changed.load(path)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1])
def test_invalid_metrics_fail(value):
    c = controller()
    with pytest.raises(ValueError):
        c.observe(500, 3, value, 10)


def test_duplicate_or_missing_initial_validation_fails():
    with pytest.raises(ValueError, match="step zero"):
        FrequencyController(FrequencyConfig()).observe(500, 3, 6, 10)
    c = controller()
    with pytest.raises(ValueError, match="increase"):
        c.observe(0, 3, 6, 10)
