"""Tests for closed-set candidate scoring.

The scorer decides every classification answer in the multi-task study, so the
properties tested here are the ones a wrong answer would be blamed on: that a
candidate's score depends only on its own tokens, that length normalization
actually removes the label-length confound, and that the margin reward has the
sign and the zero crossing it is supposed to have.
"""

import os
import sys

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
    CandidateAccuracy,
    CandidateSet,
    candidate_accuracy,
    candidate_margin,
    score_candidates,
    sequence_log_likelihood,
)

IGNORE = -100


class _FakeTokenizer:
    """Whitespace tokenizer with a stable vocabulary.

    Word ids start at 10 so they can never collide with the special ids the
    recipe reserves (bos/eos/pad and the two audio markers).
    """

    pad_token_id = 3

    def __init__(self):
        self.vocab = {}

    def _id(self, word):
        return self.vocab.setdefault(word, 10 + len(self.vocab))

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        ids = [self._id(w) for w in text.split()]
        return type("Enc", (), {"input_ids": ids})()

    def convert_tokens_to_ids(self, token):
        return {"<|start_of_audio|>": 1, "<|end_of_audio|>": 2}[token]


def _candidate_set(candidates, prompt="Name it."):
    return CandidateSet(
        _FakeTokenizer(),
        prompt,
        candidates,
        bos_index=0,
        eos_index=4,
        start_of_audio_index=1,
        end_of_audio_index=2,
        ignore_index=IGNORE,
    )


# --------------------------------------------------------------- CandidateSet


def test_candidate_set_layout_matches_the_recipe_prefix():
    cset = _candidate_set(("red", "dark blue"))
    # [<|start_of_audio|>, <|end_of_audio|>, prompt..., bos, label...]
    assert cset.prompt_len == 2 + 2  # two markers + "Name it." -> 2 words
    assert cset.tokens_bos.shape[0] == 2
    # Both rows are padded to the longest label, and the shorter label's
    # padding scores nothing.
    assert cset.tokens_bos.shape[1] == cset.prompt_len + 1 + 2
    assert cset.label_lens == [1, 2]
    # Targets are label + eos, then ignore padding.
    assert cset.tokens_eos[0].tolist()[-1] == IGNORE
    assert cset.tokens_eos[1].tolist()[-1] == 4


def test_candidate_set_rejects_degenerate_label_sets():
    with pytest.raises(ValueError, match="at least one candidate"):
        _candidate_set(())
    with pytest.raises(ValueError, match="Duplicate candidate"):
        _candidate_set(("red", "red"))


def test_gold_indices_raise_on_a_label_outside_the_set():
    cset = _candidate_set(("red", "green"))
    assert cset.gold_indices(["green", "red"]).tolist() == [1, 0]
    # Silently scoring an unknown gold label as always-wrong would surface only
    # as an unexplained accuracy ceiling.
    with pytest.raises(KeyError, match="absent from the candidate set"):
        cset.gold_indices(["green", "blue"])


# -------------------------------------------------------- log-likelihood math


def test_length_normalization_removes_the_label_length_confound():
    # Two candidates the model likes equally per token, one twice as long.
    vocab = 6
    logits = torch.zeros(2, 2, vocab)
    logits[0, 0, 5] = 1.0  # one scored token
    logits[1, 0, 5] = 1.0  # two scored tokens, equally confident
    logits[1, 1, 5] = 1.0
    targets = torch.tensor([[5, IGNORE], [5, 5]])

    normalized = sequence_log_likelihood(logits, targets, ignore_index=IGNORE)
    summed = sequence_log_likelihood(
        logits, targets, ignore_index=IGNORE, length_normalize=False
    )
    # Per token they are indistinguishable...
    assert normalized[0] == pytest.approx(float(normalized[1]), abs=1e-5)
    # ...but the unnormalized total penalizes the longer label purely for being
    # longer -- exactly twice the cost for exactly twice the tokens, which is
    # what the study must not measure.
    assert float(summed[1]) == pytest.approx(2 * float(summed[0]), rel=1e-5)
    assert float(summed[1]) < float(summed[0])


def test_ignored_positions_do_not_contribute():
    vocab = 6
    logits = torch.randn(1, 4, vocab)
    targets = torch.tensor([[IGNORE, IGNORE, 2, 3]])
    scored = sequence_log_likelihood(logits, targets, ignore_index=IGNORE)
    # Same two real targets, no ignored prefix -> identical score.
    trimmed = sequence_log_likelihood(
        logits[:, 2:], torch.tensor([[2, 3]]), ignore_index=IGNORE
    )
    assert float(scored) == pytest.approx(float(trimmed), abs=1e-5)


# ------------------------------------------------------------- margin/accuracy


def test_margin_is_signed_and_crosses_zero_at_the_decision_boundary():
    scores = torch.tensor(
        [
            [2.0, 1.0, 0.5],  # gold 0 wins by 1.0
            [1.0, 2.0, 0.5],  # gold 0 loses by 1.0
            [1.0, 1.0, 0.5],  # gold 0 exactly ties the runner-up
        ]
    )
    gold = torch.tensor([0, 0, 0])
    margin = candidate_margin(scores, gold)
    assert margin.tolist() == pytest.approx([1.0, -1.0, 0.0])
    # The sign agrees with correctness everywhere it is nonzero, which is what
    # makes the margin a dense stand-in for the sparse accuracy reward.
    assert (margin[0] > 0) and (margin[1] < 0)
    # Rows 1 and 3 are argmax-correct (a tie breaks to the lower index), so
    # accuracy sees 2/3 while the margin still reports row 3 as undecided --
    # the reason the reward is the margin and not the accuracy.
    assert candidate_accuracy(scores, gold) == pytest.approx(2 / 3)


def test_margin_ignores_the_gold_column_when_taking_the_best_incorrect():
    # A runaway-confident gold must not be compared against itself.
    scores = torch.tensor([[9.0, 1.0]])
    gold = torch.tensor([0])
    assert float(candidate_margin(scores, gold)) == pytest.approx(8.0)


def test_margin_is_zero_for_a_single_candidate():
    scores = torch.tensor([[1.5], [-2.0]])
    gold = torch.tensor([0, 0])
    assert candidate_margin(scores, gold).tolist() == [0.0, 0.0]


# ------------------------------------------------------------ accuracy metric


def test_candidate_accuracy_tracks_accuracy_macro_f1_and_confusion():
    metric = CandidateAccuracy(("angry", "neutral", "sad"))
    # Majority-class predictor on a neutral-heavy pass: high accuracy, poor
    # macro-F1. This is the CREMA-D voice-vote failure mode the report has to
    # be able to see.
    metric.append(["a", "b", "c", "d"], [1, 1, 1, 1], [1, 1, 1, 0])
    assert metric.summarize("accuracy") == pytest.approx(0.75)
    assert metric.summarize("error_rate") == pytest.approx(25.0)
    # neutral F1 = 2*.75*1/1.75; angry F1 = 0; sad never appears -> excluded.
    assert metric.summarize("macro_f1") == pytest.approx(
        (2 * 0.75 * 1.0 / 1.75 + 0.0) / 2
    )
    confusion = metric.confusion()
    assert confusion["angry"]["neutral"] == 1
    assert confusion["neutral"]["neutral"] == 3
    assert len(metric.scores) == 4


def test_candidate_accuracy_is_empty_before_anything_is_appended():
    metric = CandidateAccuracy(("yes", "no"))
    assert not metric.scores
    assert metric.summarize("accuracy") == 0.0


# ------------------------------------------------------------ score_candidates


class _FakeProj(torch.nn.Module):
    def forward(self, x):
        return x


class _FakeLLM(torch.nn.Module):
    """Deterministic decoder whose logits depend on the whole prefix.

    A per-position linear head would read only the token at that position, so
    the audio could never reach a label position and every candidate score
    would be identical across utterances. Mixing in the masked mean of the
    sequence is the cheapest stand-in for attention that preserves the property
    the placement claim rests on: change the pooled audio, change the ranking.
    """

    def __init__(self, vocab, dim):
        super().__init__()
        self.head = torch.nn.Linear(dim, vocab)

    def forward(self, inputs_embeds, attention_mask, position_ids):
        mask = attention_mask.unsqueeze(-1).to(inputs_embeds.dtype)
        pooled = (inputs_embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp(
            min=1.0
        )
        mixed = inputs_embeds + pooled.unsqueeze(1)
        return type("Out", (), {"logits": self.head(mixed)})()


class _FakeBrain:
    def __init__(self, vocab, dim):
        torch.manual_seed(0)
        self.modules = type(
            "M", (), {"proj": _FakeProj(), "llm": _FakeLLM(vocab, dim)}
        )()
        self.embedding = torch.nn.Embedding(vocab, dim)

    def txt_embedding(self, tokens):
        return self.embedding(tokens)


def _score(brain, cset, seg_feats, seg_pad, **kwargs):
    with torch.no_grad():
        return score_candidates(brain, seg_feats, seg_pad, cset, **kwargs)


def test_score_candidates_shape_and_chunk_invariance():
    vocab, dim = 40, 8
    cset = _candidate_set(("red", "green", "dark blue"))
    brain = _FakeBrain(vocab, dim)
    seg_feats = torch.randn(4, 5, dim)
    seg_pad = torch.zeros(4, 5, dtype=torch.bool)
    seg_pad[2, 4:] = True  # one shorter utterance

    full = _score(brain, cset, seg_feats, seg_pad)
    assert full.shape == (4, 3)
    # Chunking is a memory strategy, not a modelling choice: it must not move
    # a single score.
    for chunk in (1, 2, 5, 1000):
        chunked = _score(brain, cset, seg_feats, seg_pad, chunk_size=chunk)
        assert torch.allclose(full, chunked, atol=1e-5)


def test_scores_respond_to_the_pooled_prefix():
    vocab, dim = 40, 8
    cset = _candidate_set(("red", "green"))
    brain = _FakeBrain(vocab, dim)
    seg_pad = torch.zeros(2, 5, dtype=torch.bool)

    a = _score(brain, cset, torch.randn(2, 5, dim), seg_pad)
    b = _score(brain, cset, torch.randn(2, 5, dim), seg_pad)
    # Different audio, different ranking evidence. If this ever ties, candidate
    # scoring is not reading the audio at all and every task result is vacuous.
    assert not torch.allclose(a, b, atol=1e-4)


def test_scoring_one_utterance_matches_scoring_it_in_a_batch():
    # Utterances must not influence each other through padding or batching.
    vocab, dim = 40, 8
    cset = _candidate_set(("red", "green", "dark blue"))
    brain = _FakeBrain(vocab, dim)
    seg_feats = torch.randn(3, 6, dim)
    seg_pad = torch.zeros(3, 6, dtype=torch.bool)

    batched = _score(brain, cset, seg_feats, seg_pad)
    for i in range(3):
        alone = _score(brain, cset, seg_feats[i : i + 1], seg_pad[i : i + 1])
        assert torch.allclose(batched[i], alone[0], atol=1e-5)


def test_empty_batch_returns_an_empty_score_matrix():
    vocab, dim = 40, 8
    cset = _candidate_set(("red", "green"))
    brain = _FakeBrain(vocab, dim)
    scores = _score(
        brain,
        cset,
        torch.zeros(0, 5, dim),
        torch.zeros(0, 5, dtype=torch.bool),
    )
    assert scores.shape == (0, 2)


def test_chunk_size_must_be_positive():
    vocab, dim = 40, 8
    cset = _candidate_set(("red", "green"))
    brain = _FakeBrain(vocab, dim)
    with pytest.raises(ValueError, match="chunk_size must be >= 1"):
        _score(
            brain,
            cset,
            torch.randn(2, 5, dim),
            torch.zeros(2, 5, dtype=torch.bool),
            chunk_size=0,
        )
