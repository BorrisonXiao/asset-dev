"""Tests for the classification GRPO reward.

The reward is what the boundary policy optimizes, so the properties pinned here
are the ones that decide whether the policy learns anything at all: that a
classification batch is scored by the candidate margin rather than by WER-shaped
machinery, that a generative batch is untouched, that the margin actually varies
with the sampled boundaries (a constant reward gives GRPO zero advantage), and
that the cheap ablation path still works.
"""

import os
import sys
import types

import pytest
import torch

RECIPE_DIR = os.path.join(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ),
    "recipes",
    "LibriSpeech",
    "ASR",
    "transformer",
)
if RECIPE_DIR not in sys.path:
    sys.path.insert(0, RECIPE_DIR)

from candidate_scoring import (  # noqa: E402
    CandidateSet,
    candidate_margin,
    score_candidates,
)
from train_speechllm_multitask import MultitaskSegmenterASR  # noqa: E402

DIM = 8
VOCAB = 40


class _FakeTokenizer:
    pad_token_id = 3

    def __init__(self):
        self.vocab = {}

    def _id(self, word):
        return self.vocab.setdefault(word, 10 + len(self.vocab))

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        return types.SimpleNamespace(
            input_ids=[self._id(w) for w in text.split()]
        )

    def convert_tokens_to_ids(self, token):
        return {"<|start_of_audio|>": 1, "<|end_of_audio|>": 2}[token]


class _FakeLLM(torch.nn.Module):
    """Decoder whose logits depend on the whole prefix, so audio reaches labels."""

    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(DIM, VOCAB)

    def forward(self, inputs_embeds, attention_mask, position_ids):
        mask = attention_mask.unsqueeze(-1).to(inputs_embeds.dtype)
        pooled = (inputs_embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp(
            min=1.0
        )
        return types.SimpleNamespace(
            logits=self.head(inputs_embeds + pooled.unsqueeze(1))
        )


class _FakeSegmenter:
    """``pool`` collapses frames between boundaries; here, a boundary-dependent mean."""

    def pool(self, feats, pad_mask, boundary):
        # Two different boundaries must give two different pooled prefixes,
        # which is what makes the reward informative about segmentation.
        weights = boundary.unsqueeze(-1).to(feats.dtype)
        pooled = feats * (1.0 + weights)
        return pooled, pad_mask


def _brain(candidates=("red", "green", "blue"), reward="margin"):
    """A MultitaskSegmenterASR with only the pieces the reward path touches.

    Built with ``__new__`` rather than ``__init__`` so the real method -- and in
    particular its zero-argument ``super()`` -- is exercised, without standing up
    a SpeechBrain Brain, an optimizer, or a 1B decoder.
    """
    brain = MultitaskSegmenterASR.__new__(MultitaskSegmenterASR)
    tokenizer = _FakeTokenizer()
    torch.manual_seed(0)
    brain.tokenizer = tokenizer
    brain.active_task_name = "emotion"
    brain.task_output_kind = {
        "asr": "text",
        "st_en_de": "translation",
        "emotion": "classification",
    }
    brain.candidate_sets = {
        "emotion": CandidateSet(
            tokenizer,
            "Name the emotion.",
            candidates,
            bos_index=0,
            eos_index=4,
            start_of_audio_index=1,
            end_of_audio_index=2,
            ignore_index=-100,
        )
    }
    brain.candidate_chunk_size = None
    brain.hparams = types.SimpleNamespace(classification_reward=reward)
    brain.modules = types.SimpleNamespace(
        segmenter=_FakeSegmenter(),
        proj=torch.nn.Identity(),
        llm=_FakeLLM(),
    )
    embedding = torch.nn.Embedding(VOCAB, DIM)
    brain.txt_embedding = embedding
    return brain


def _inputs(batch_size=3, frames=6):
    torch.manual_seed(1)
    feats = torch.randn(batch_size, frames, DIM)
    pad_mask = torch.zeros(batch_size, frames, dtype=torch.bool)
    boundary = torch.zeros(batch_size, frames)
    boundary[:, 2] = 1.0
    batch = types.SimpleNamespace(wrd=["red", "green", "red"][:batch_size])
    return feats, pad_mask, boundary, batch


# ------------------------------------------------------------------ dispatch


def test_generative_batches_keep_the_base_class_reward():
    brain = _brain()
    brain.active_task_name = "asr"
    feats, pad_mask, boundary, batch = _inputs()

    called = {}

    def fake_run_decoder(feats, pad_mask, boundary, batch, want_hyps=False):
        called["want_hyps"] = want_hyps
        return {"llm_logits": torch.zeros(3, 4, VOCAB)}

    brain._run_decoder = fake_run_decoder
    brain._utterance_nll = lambda logits, batch: torch.full((3,), 2.0)

    reward = brain._rollout_quality_reward(
        feats, pad_mask, boundary, batch, "nll"
    )
    # The ASR path must be byte-for-byte what it was: -NLL from one teacher-
    # forced decoder pass, no candidate scoring.
    assert torch.allclose(reward, torch.full((3,), -2.0))
    assert called["want_hyps"] is False


def test_classification_batches_use_the_candidate_margin():
    brain = _brain()
    feats, pad_mask, boundary, batch = _inputs()

    # Independently recompute what the reward should be.
    seg_feats, seg_pad = brain.modules.segmenter.pool(feats, pad_mask, boundary)
    with torch.no_grad():
        scores = score_candidates(
            brain, seg_feats, seg_pad, brain.candidate_sets["emotion"]
        )
    gold = brain.candidate_sets["emotion"].gold_indices(batch.wrd)
    expected = candidate_margin(scores, gold)

    reward = brain._rollout_quality_reward(
        feats, pad_mask, boundary, batch, "nll"
    )
    assert torch.allclose(reward, expected, atol=1e-5)
    assert reward.shape == (3,)


def test_classification_reward_never_calls_the_generative_decoder():
    # The margin needs only the pooled prefix, so it skips the extra decoder
    # forward the -NLL/CER rewards run first.
    brain = _brain()

    def explode(*args, **kwargs):
        raise AssertionError("the margin path must not run _run_decoder")

    brain._run_decoder = explode
    feats, pad_mask, boundary, batch = _inputs()
    reward = brain._rollout_quality_reward(
        feats, pad_mask, boundary, batch, "nll"
    )
    assert torch.isfinite(reward).all()


def test_nll_ablation_falls_back_to_the_base_reward():
    brain = _brain(reward="nll")
    feats, pad_mask, boundary, batch = _inputs()
    brain._run_decoder = lambda *a, **k: {
        "llm_logits": torch.zeros(3, 4, VOCAB)
    }
    brain._utterance_nll = lambda logits, batch: torch.full((3,), 1.5)

    reward = brain._rollout_quality_reward(
        feats, pad_mask, boundary, batch, "cer"
    )
    # Even when the run asks for a free-running CER reward, a classification
    # batch has no transcript to score, so the ablation uses -NLL of the label.
    assert torch.allclose(reward, torch.full((3,), -1.5))


# -------------------------------------------------------------- reward shape


def test_the_margin_varies_with_the_sampled_boundaries():
    # GRPO's signal is the spread of rewards WITHIN one utterance's K rollouts.
    # A reward that ignores the segmentation gives zero advantage everywhere and
    # the policy would never move.
    brain = _brain()
    feats, pad_mask, _, batch = _inputs()

    rewards = []
    for position in (1, 3, 5):
        boundary = torch.zeros(feats.shape[0], feats.shape[1])
        boundary[:, position] = 1.0
        rewards.append(
            brain._rollout_quality_reward(
                feats, pad_mask, boundary, batch, "nll"
            )
        )
    stacked = torch.stack(rewards, dim=1)
    assert stacked.shape == (3, 3)
    assert (stacked.std(dim=1) > 1e-6).all()


def test_the_margin_is_signed_around_the_decision():
    brain = _brain()
    feats, pad_mask, boundary, batch = _inputs()
    reward = brain._rollout_quality_reward(
        feats, pad_mask, boundary, batch, "nll"
    )
    seg_feats, seg_pad = brain.modules.segmenter.pool(feats, pad_mask, boundary)
    with torch.no_grad():
        scores = score_candidates(
            brain, seg_feats, seg_pad, brain.candidate_sets["emotion"]
        )
    gold = brain.candidate_sets["emotion"].gold_indices(batch.wrd)
    correct = scores.argmax(dim=1) == gold
    # Positive exactly where the gold label already wins, so the sign carries
    # the 0/1 accuracy signal and the magnitude adds the gradient accuracy lacks.
    assert torch.equal(reward > 0, correct)


def test_a_wrong_gold_label_is_a_loud_error():
    brain = _brain()
    feats, pad_mask, boundary, _ = _inputs()
    batch = types.SimpleNamespace(wrd=["red", "chartreuse", "red"])
    with pytest.raises(KeyError, match="absent from the candidate set"):
        brain._rollout_quality_reward(feats, pad_mask, boundary, batch, "nll")
