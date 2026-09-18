"""Closed-set label scoring for the classification tasks.

The generative tasks (``asr``, ``st_en_de``) are decoded freely and scored with
WER/BLEU. The non-ASR tasks added by the 2026-08-31 survey are classifications
over a fixed label set, and free decoding is the wrong instrument for them:
the decoder can emit a string that is not a label at all, and comparing labels
that tokenize to different lengths lets sequence length stand in for evidence.

So a classification answer is chosen by ranking the label set. Every candidate
is scored under the SAME pooled audio prefix, the score is the mean per-token
log-likelihood of the candidate's own tokens, and the prediction is the argmax.
Two quantities come out of that ranking:

``accuracy``
    what the paper reports.
``margin``
    gold log-likelihood minus the best incorrect candidate's -- signed, dense,
    and defined on every utterance including the ones already answered right.
    That is the GRPO task reward; raw accuracy is far too sparse to shape a
    boundary policy with ``K=4`` rollouts.

Nothing here is imported by the ASR-only recipes.
"""

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class CandidateSet:
    """Tokenized label set for one task, built once and reused every batch.

    The decoder prefix layout is fixed by the recipe:

        [<|start_of_audio|>] [audio] [<|end_of_audio|>] [prompt] [bos] [label]

    Only the trailing label varies across candidates, but the tokenizer is the
    expensive part, so all ``M`` variants are tokenized and padded up front.

    Arguments
    ---------
    tokenizer : transformers tokenizer
        The decoder's tokenizer.
    prompt : str
        The task's canonical decoder prompt.
    candidates : sequence of str
        The label strings to rank, in reporting order.
    bos_index : int or None
        Optional BOS token inserted between prompt and label.
    eos_index : int
        Terminates each candidate.
    start_of_audio_index, end_of_audio_index : int
        The audio-span markers.
    ignore_index : int
        Label-padding value; positions holding it score nothing.
    """

    def __init__(
        self,
        tokenizer,
        prompt,
        candidates,
        bos_index,
        eos_index,
        start_of_audio_index,
        end_of_audio_index,
        ignore_index=-100,
    ):
        candidates = tuple(candidates)
        if not candidates:
            raise ValueError("A CandidateSet needs at least one candidate")
        if len(set(candidates)) != len(candidates):
            raise ValueError(f"Duplicate candidate labels: {candidates}")

        self.candidates = candidates
        self.index_of = {label: i for i, label in enumerate(candidates)}
        self.ignore_index = int(ignore_index)

        prompt_ids = tokenizer(
            prompt, return_tensors=None, add_special_tokens=False
        ).input_ids
        prefix = (
            [int(start_of_audio_index), int(end_of_audio_index)]
            + list(prompt_ids)
            + ([int(bos_index)] if bos_index is not None else [])
        )
        # prompt_len counts only what precedes the audio span in the text
        # stream, matching multitask_data.text_pipeline exactly.
        self.prompt_len = 2 + len(prompt_ids)

        label_ids = [
            tokenizer(label, add_special_tokens=False).input_ids
            for label in candidates
        ]
        self.label_lens = [len(ids) for ids in label_ids]
        max_label = max(self.label_lens)

        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = int(eos_index)

        bos_rows, eos_rows = [], []
        for ids in label_ids:
            pad = max_label - len(ids)
            bos_rows.append(prefix + ids + [pad_id] * pad)
            # Targets are the label plus EOS; padding scores nothing.
            eos_rows.append(ids + [int(eos_index)] + [self.ignore_index] * pad)

        self.tokens_bos = torch.tensor(bos_rows, dtype=torch.long)
        self.tokens_eos = torch.tensor(eos_rows, dtype=torch.long)
        # Relative lengths, the convention get_multimodal_attention_mask uses.
        total = self.tokens_bos.shape[1]
        self.tokens_bos_lens = torch.tensor(
            [(len(prefix) + n) / total for n in self.label_lens],
            dtype=torch.float32,
        )
        logger.info(
            "Candidate set for prompt %r: %d labels, %d-%d label tokens",
            prompt,
            len(candidates),
            min(self.label_lens),
            max_label,
        )

    def __len__(self):
        return len(self.candidates)

    def gold_indices(self, labels, device=None):
        """Map gold label strings to candidate indices ``(B,)``.

        Raises rather than falling back to a sentinel: a label outside the set
        means the manifest and the task table disagree, and silently scoring it
        as always-wrong would show up only as an unexplained accuracy ceiling.
        """
        missing = sorted({x for x in labels if x not in self.index_of})
        if missing:
            raise KeyError(
                f"Gold labels absent from the candidate set: {missing[:5]}. "
                f"Known labels: {list(self.candidates)}"
            )
        return torch.tensor(
            [self.index_of[x] for x in labels],
            dtype=torch.long,
            device=device,
        )

    def to(self, device):
        """Move the cached tensors onto ``device`` in place."""
        self.tokens_bos = self.tokens_bos.to(device)
        self.tokens_eos = self.tokens_eos.to(device)
        self.tokens_bos_lens = self.tokens_bos_lens.to(device)
        return self


def sequence_log_likelihood(
    logits, targets, ignore_index=-100, length_normalize=True
):
    """Per-sequence log-likelihood of ``targets`` under ``logits``.

    Arguments
    ---------
    logits : torch.Tensor
        ``(N, T, V)`` decoder logits already aligned to ``targets``.
    targets : torch.Tensor
        ``(N, T)`` token ids, ``ignore_index`` where nothing is scored.
    ignore_index : int
        Positions to skip.
    length_normalize : bool
        Divide by the number of scored tokens. On by default: without it a
        one-token label beats a four-token label on length alone.

    Returns
    -------
    torch.Tensor
        ``(N,)`` log-likelihoods.
    """
    nll = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        targets.reshape(-1),
        ignore_index=ignore_index,
        reduction="none",
    ).reshape(targets.shape)
    counts = (targets != ignore_index).sum(dim=1)
    total = -nll.sum(dim=1)
    if not length_normalize:
        return total
    return total / counts.clamp(min=1).to(total.dtype)


def candidate_margin(scores, gold_index):
    """Gold score minus the best incorrect candidate's ``(B,)``.

    Positive means the gold label already wins; the magnitude says by how much,
    which is what makes this a usable dense reward rather than a 0/1 signal.
    With a single candidate the margin is undefined and returns zeros.
    """
    if scores.shape[1] < 2:
        return torch.zeros_like(scores[:, 0])
    gold = scores.gather(1, gold_index.view(-1, 1)).squeeze(1)
    masked = scores.scatter(
        1, gold_index.view(-1, 1), torch.finfo(scores.dtype).min
    )
    return gold - masked.max(dim=1).values


def candidate_accuracy(scores, gold_index):
    """Fraction of utterances whose argmax candidate is the gold one."""
    if scores.numel() == 0:
        return 0.0
    return float((scores.argmax(dim=1) == gold_index).float().mean())


class CandidateAccuracy:
    """Accumulate closed-set predictions across a validation or test pass.

    Mirrors the interface the recipe's WER/CER metrics expose (``append``,
    truthy ``scores``, ``summarize``) so the per-task stats path treats a
    classification task the same way it treats a generative one.
    """

    def __init__(self, candidates):
        self.candidates = tuple(candidates)
        self.scores = []  # per-utterance 1/0, named to match the WER metrics
        self.ids = []
        self.predictions = []
        self.references = []

    def append(self, ids, predicted_index, gold_index):
        """Record one batch of predictions."""
        predicted_index = [int(i) for i in predicted_index]
        gold_index = [int(i) for i in gold_index]
        for utt_id, pred, gold in zip(ids, predicted_index, gold_index):
            self.ids.append(utt_id)
            self.predictions.append(self.candidates[pred])
            self.references.append(self.candidates[gold])
            self.scores.append(1.0 if pred == gold else 0.0)

    def summarize(self, field="accuracy"):
        """``accuracy``, ``macro_f1``, or ``error_rate`` over everything seen."""
        if field == "error_rate":
            return 100.0 * (1.0 - self._accuracy())
        if field == "macro_f1":
            return self._macro_f1()
        if field == "accuracy":
            return self._accuracy()
        raise KeyError(f"Unknown field {field!r}")

    def _accuracy(self):
        if not self.scores:
            return 0.0
        return sum(self.scores) / len(self.scores)

    def _macro_f1(self):
        """Unweighted mean per-class F1.

        Reported alongside accuracy because CREMA-D's crowd voice labels are
        roughly half neutral: an accuracy that looks respectable can come
        entirely from the majority class.
        """
        if not self.scores:
            return 0.0
        f1s = []
        for label in self.candidates:
            tp = sum(
                1
                for p, r in zip(self.predictions, self.references)
                if p == label and r == label
            )
            fp = sum(
                1
                for p, r in zip(self.predictions, self.references)
                if p == label and r != label
            )
            fn = sum(
                1
                for p, r in zip(self.predictions, self.references)
                if p != label and r == label
            )
            if tp + fn == 0:
                # Class absent from the references: not this pass's business.
                continue
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn)
            f1s.append(
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
        return sum(f1s) / len(f1s) if f1s else 0.0

    def confusion(self):
        """``{reference: {prediction: count}}`` for the report's matrices."""
        matrix = {r: dict.fromkeys(self.candidates, 0) for r in self.candidates}
        for pred, ref in zip(self.predictions, self.references):
            matrix[ref][pred] += 1
        return matrix


def score_candidates(
    brain,
    seg_feats,
    seg_pad,
    candidate_set,
    chunk_size=None,
    length_normalize=True,
):
    """Rank ``candidate_set`` for every utterance in a pooled batch.

    Every candidate is scored against the same pooled audio prefix, so the
    comparison isolates the label: the audio, the boundaries, and the prompt are
    byte-identical across the ``M`` scores of one utterance.

    Arguments
    ---------
    brain : SegmenterASR
        Supplies ``modules.proj``, ``txt_embedding``, ``modules.llm`` and the
        multimodal mask helpers, i.e. exactly the decoder path
        ``_run_decoder_from_segments`` uses.
    seg_feats : torch.Tensor
        ``(B, S, D)`` pooled segment features.
    seg_pad : torch.Tensor
        ``(B, S)`` boolean padding mask, ``True`` at padding.
    candidate_set : CandidateSet
        The label set to rank.
    chunk_size : int or None
        Maximum ``B * M`` rows per decoder forward. ``None`` runs one forward,
        which is fine for a six-way task but not for the 31-way intent set at a
        58-utterance microbatch.
    length_normalize : bool
        Passed to :func:`sequence_log_likelihood`.

    Returns
    -------
    torch.Tensor
        ``(B, M)`` scores, higher is better.
    """
    # Imported here rather than at module import time: train_speechllm pulls in
    # the full recipe stack, and this module is unit tested on its own.
    from batch_invariant_decode import compact_position_ids
    from segment_pooling import padding_mask_to_lengths
    from train_speechllm import get_multimodal_attention_mask

    device = seg_feats.device
    candidate_set.to(device)
    batch_size = seg_feats.shape[0]
    n_candidates = len(candidate_set)
    if batch_size == 0:
        return seg_feats.new_zeros((0, n_candidates))

    projected = brain.modules.proj(seg_feats)
    seg_lens = padding_mask_to_lengths(seg_pad)

    # (B, M) flattened row-major: row b*M + m is utterance b scored as label m.
    proj_rows = projected.repeat_interleave(n_candidates, dim=0)
    len_rows = seg_lens.repeat_interleave(n_candidates, dim=0)
    bos_rows = candidate_set.tokens_bos.repeat(batch_size, 1)
    eos_rows = candidate_set.tokens_eos.repeat(batch_size, 1)
    bos_len_rows = candidate_set.tokens_bos_lens.repeat(batch_size)

    total_rows = batch_size * n_candidates
    # ``None`` means "one forward"; 0 is a misconfiguration, not a synonym.
    step = total_rows if chunk_size is None else int(chunk_size)
    if step < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")

    scores = []
    for start in range(0, total_rows, step):
        stop = min(start + step, total_rows)
        proj = proj_rows[start:stop]
        txt_embds = brain.txt_embedding(bos_rows[start:stop])
        multimodal = torch.cat(
            [txt_embds[:, 0].unsqueeze(1), proj, txt_embds[:, 1:]], dim=1
        )
        attn = get_multimodal_attention_mask(
            proj,
            len_rows[start:stop],
            txt_embds,
            bos_len_rows[start:stop],
            device,
        )
        logits = brain.modules.llm(
            inputs_embeds=multimodal,
            attention_mask=attn,
            position_ids=compact_position_ids(attn),
        ).logits
        targets = eos_rows[start:stop]
        # The audio span occupies the leading positions and scores nothing,
        # exactly as in SegmenterASR._target_tokens -- so slice it off rather
        # than pad it with ignore_index. The two are mathematically identical
        # (ignored positions contribute zero to both the sum and the count),
        # but padding makes cross_entropy materialize a float32
        # (rows, n_audio + n_label, vocab) tensor whose audio part is pure
        # waste. With a 128k vocab at 50 Hz that is ~12 GB for one chunk and it
        # OOMs an 80 GB A100 (job 1784456); the label span is ~4 positions.
        n_audio = logits.shape[1] - targets.shape[1]
        scores.append(
            sequence_log_likelihood(
                logits[:, n_audio:],
                targets,
                ignore_index=candidate_set.ignore_index,
                length_normalize=length_normalize,
            )
        )

    return torch.cat(scores, dim=0).view(batch_size, n_candidates)


__all__ = [
    "CandidateAccuracy",
    "CandidateSet",
    "candidate_accuracy",
    "candidate_margin",
    "score_candidates",
    "sequence_log_likelihood",
]
