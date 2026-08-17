"""Unit tests for reversible first-order bilevel lookahead utilities."""

import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from speechbrain.dataio.batch import PaddedBatch

RECIPE_DIR = (
    Path(__file__).resolve().parents[2]
    / "recipes"
    / "LibriSpeech"
    / "ASR"
    / "transformer"
)
sys.path.insert(0, str(RECIPE_DIR))

from bilevel import (  # noqa: E402
    capture_rng_state,
    decoder_batch_view,
    gradient_diagnostics,
    replay_rng_state,
    reward_rank_diagnostics,
    support_query_slices,
    temporary_sgd_update,
    unique_trainable_parameters,
)
from train_speechllm_with_segmenter import SegmenterASR  # noqa: E402


def test_support_query_split_is_disjoint_and_nonempty():
    support, query = support_query_slices(5, support_fraction=0.5)
    assert list(range(5))[support] == [0, 1]
    assert list(range(5))[query] == [2, 3, 4]
    with pytest.raises(ValueError, match="at least two"):
        support_query_slices(1)


def test_decoder_batch_view_slices_padded_and_python_values():
    batch = PaddedBatch(
        [
            {
                "id": f"u{i}",
                "tokens_bos": torch.arange(i + 2),
                "tokens_eos": torch.arange(i + 1),
                "prompt_len": 1,
                "wrd": f"text {i}",
            }
            for i in range(4)
        ]
    )
    view = decoder_batch_view(batch, slice(2, 4))
    assert view.id == ["u2", "u3"]
    assert view.wrd == ["text 2", "text 3"]
    assert view.tokens_bos.data.shape[0] == 2
    assert view.tokens_eos.lengths.shape == (2,)


def test_temporary_sgd_update_improves_loss_and_restores_exactly():
    model = torch.nn.Linear(3, 1, bias=False)
    x = torch.tensor([[1.0, -2.0, 0.5]])
    target = torch.tensor([[0.75]])
    params = unique_trainable_parameters(model)
    before_values = [parameter.detach().clone() for parameter in params]
    sentinel_grads = [torch.full_like(parameter, 7.0) for parameter in params]
    for parameter, sentinel in zip(params, sentinel_grads):
        parameter.grad = sentinel.clone()

    before_loss = torch.nn.functional.mse_loss(model(x), target)
    gradients = torch.autograd.grad(before_loss, params)
    with temporary_sgd_update(params, gradients, learning_rate=0.05) as stats:
        during_loss = torch.nn.functional.mse_loss(model(x), target)
        assert during_loss < before_loss
        assert stats["grad_norm"] > 0.0
        assert stats["grad_scale"] == 1.0
        assert stats["gradient_tensors"] == len(params)
        assert stats["nonzero_gradient_tensors"] == len(params)
        assert stats["gradient_max_abs"] > 0.0

    for parameter, original, sentinel in zip(
        params, before_values, sentinel_grads
    ):
        assert torch.equal(parameter, original)
        assert torch.equal(parameter.grad, sentinel)


def test_temporary_sgd_update_restores_after_exception():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    original = parameter.detach().clone()
    with pytest.raises(RuntimeError, match="probe failure"):
        with temporary_sgd_update(
            [parameter], [torch.ones_like(parameter)], learning_rate=0.1
        ):
            raise RuntimeError("probe failure")
    assert torch.equal(parameter, original)


def test_gradient_diagnostics_counts_unused_and_zero_tensors():
    stats = gradient_diagnostics(
        [None, torch.zeros(2), torch.tensor([0.0, -0.25])]
    )
    assert stats == {
        "parameter_tensors": 3,
        "gradient_tensors": 2,
        "nonzero_gradient_tensors": 1,
        "gradient_max_abs": 0.25,
    }


def test_replay_rng_state_matches_dropout_without_rewinding_caller():
    torch.manual_seed(8)
    dropout = torch.nn.Dropout(0.5).train()
    inputs = torch.ones(16)
    state = capture_rng_state("cpu")
    first = dropout(inputs)
    state_after_first = torch.random.get_rng_state().clone()
    with replay_rng_state(state, "cpu"):
        replay = dropout(inputs)
    assert torch.equal(first, replay)
    assert torch.equal(torch.random.get_rng_state(), state_after_first)


def test_reward_rank_diagnostics_detects_rollout_changes():
    current = torch.tensor([[4.0, 3.0, 2.0, 1.0], [1.0, 3.0, 2.0, 0.0]])
    adapted = torch.tensor([[1.0, 2.0, 3.0, 4.0], [1.0, 3.2, 2.0, 0.0]])
    stats = reward_rank_diagnostics(current, adapted)
    assert stats["top_rollout_flip"] == pytest.approx(0.5)
    assert -1.0 <= stats["rank_correlation"] <= 1.0
    assert stats["reward_delta_abs"] > 0.0


class _ToyBilevelBrain(SegmenterASR):
    """Use the production lookahead method with a tiny differentiable decoder."""

    def __init__(self):
        self.device = "cpu"
        self.training_ctx = nullcontext()
        self.proj = torch.nn.Linear(2, 3)
        self.llm = torch.nn.Linear(3, 1)
        self.modules = SimpleNamespace(
            proj=self.proj,
            llm=self.llm,
            segmenter=SimpleNamespace(
                pooler=torch.nn.Identity(), pool=self._pool
            ),
        )
        self.hparams = SimpleNamespace(
            segmenter_reward="nll",
            bilevel_inner_lr=0.05,
            bilevel_inner_max_grad_norm=10.0,
            bilevel_measure_support_after=True,
        )

    def _run_decoder(self, feats, pad_mask, boundary, batch, want_hyps=False):
        del want_hyps
        segments = self._pool(feats, pad_mask, boundary)
        return self._run_decoder_from_segments(*segments, batch)

    @staticmethod
    def _pool(feats, pad_mask, boundary):
        del pad_mask
        valid_boundary = boundary.clamp(min=0).float().mean(dim=1, keepdim=True)
        pooled = feats.mean(dim=1) + valid_boundary
        return pooled.unsqueeze(1), torch.zeros(
            pooled.size(0), 1, dtype=torch.bool
        )

    def _run_decoder_from_segments(
        self, seg_feats, seg_pad, batch, want_hyps=False
    ):
        del seg_pad, batch, want_hyps
        pooled = seg_feats.squeeze(1)
        projected = torch.tanh(self.proj(pooled))
        return {"llm_logits": self.llm(projected)}

    @staticmethod
    def _per_example_loss(logits, batch):
        return (logits.squeeze(1) - batch.targets).square()

    def _decoder_ce(self, logits, batch):
        return self._per_example_loss(logits, batch).mean()

    def _utterance_nll(self, logits, batch):
        return self._per_example_loss(logits, batch)


def test_production_lookahead_restores_model_and_returns_query_rewards():
    torch.manual_seed(4)
    brain = _ToyBilevelBrain()
    support_feats = torch.randn(2, 4, 2)
    query_feats = torch.randn(3, 4, 2)
    support_pad = torch.zeros(2, 4, dtype=torch.bool)
    query_pad = torch.zeros(3, 4, dtype=torch.bool)
    support_boundaries = [torch.randint(0, 2, (2, 4)) for _ in range(4)]
    query_boundaries = [torch.randint(0, 2, (3, 4)) for _ in range(4)]
    support_batch = SimpleNamespace(targets=torch.tensor([0.2, -0.4]))
    query_batch = SimpleNamespace(targets=torch.tensor([0.1, 0.3, -0.2]))
    parameters = unique_trainable_parameters(brain.proj, brain.llm)
    originals = [parameter.detach().clone() for parameter in parameters]

    current_rewards, rewards, diagnostics = brain._bilevel_lookahead_rewards(
        support_feats,
        support_pad,
        support_boundaries,
        support_batch,
        query_feats,
        query_pad,
        query_boundaries,
        query_batch,
        "decoder_lookahead",
    )

    assert rewards.shape == (3, 4)
    assert current_rewards.shape == rewards.shape
    assert torch.isfinite(rewards).all()
    assert diagnostics["inner_grad_norm"] > 0.0
    assert (
        diagnostics["inner_support_ce_after"] < diagnostics["inner_support_ce"]
    )
    for parameter, original in zip(parameters, originals):
        assert torch.equal(parameter, original)

    brain.hparams.bilevel_inner_lr = 0.0
    no_step_current, no_step_adapted, no_step_diagnostics = (
        brain._bilevel_lookahead_rewards(
            support_feats,
            support_pad,
            support_boundaries,
            support_batch,
            query_feats,
            query_pad,
            query_boundaries,
            query_batch,
            "decoder_lookahead",
        )
    )
    assert torch.equal(no_step_current, no_step_adapted)
    assert (
        no_step_diagnostics["inner_support_ce"]
        == no_step_diagnostics["inner_support_ce_after"]
    )

    pooler_current, pooler_adapted, _ = brain._bilevel_lookahead_rewards(
        support_feats,
        support_pad,
        support_boundaries,
        support_batch,
        query_feats,
        query_pad,
        query_boundaries,
        query_batch,
        "decoder_pooler_lookahead",
    )
    assert torch.equal(pooler_current, pooler_adapted)
