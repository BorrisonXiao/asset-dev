#!/usr/bin/env python3
"""Task-conditioned multi-task SpeechLLM training (ASR + En->De speech translation).

This is a strict extension of ``train_speechllm_with_segmenter.py``: the GRPO
rollouts, rate objective, bilevel hooks, step-controlled validation, and
inference benchmarks are inherited unchanged by subclassing ``SegmenterASR``.
Only what multi-task training genuinely requires is overridden here --

* one task per batch, with the task routed to the segmenter FiLM and to the
  decoder's task-routed LoRA before any forward pass;
* per-task validation (WER for ASR, sacreBLEU/chrF++ for ST, closed-set
  accuracy/macro-F1 for the classification tasks) and per-task rate logging, so
  a gain is never read off a mixture average;
* a three-phase schedule (frozen-policy bridge -> FiLM-only policy training ->
  limited backbone unfreezing) driven by optimizer steps.

The ASR-only recipes import none of this and are unaffected.

See plans/task_conditioned_multitask_segmenter.md.
"""

import json
import logging
import os
import sys
from collections import defaultdict

import torch
from candidate_scoring import (  # noqa: E402
    CandidateAccuracy,
    CandidateSet,
    candidate_margin,
    score_candidates,
)
from hyperpyyaml import load_hyperpyyaml
from multitask_data import (  # noqa: E402
    NUM_TASKS,
    TASK_SPECS,
    MultiSourceDynamicBatchSampler,
    build_multitask_manifest,
    is_classification,
    multitask_dataio_prepare,
    prompt_of,
    resolve_candidates,
    task_id_of,
)
from multitask_modules import (  # noqa: E402
    copy_asr_lora_into_task_slot,
    count_task_conditioned_parameters,
    expand_film_to_tasks,
    film_parameters,
)
from train_speechllm_with_segmenter import (  # noqa: E402
    SegmenterASR,
    _attach_segmenter_ssl,
    resolve_effective_grad_accumulation,
    resolve_optimizer_step_limit,
)

import speechbrain as sb
from speechbrain.utils.distributed import if_main_process

logger = logging.getLogger(__name__)

# Phases are defined on optimizer steps, matching the plan's step-controlled
# budget rather than corpus epochs.
PHASE_BRIDGE = "bridge"  # frozen policy; decoder adapters only
PHASE_FILM = "film"  # identity-initialized task FiLM trained with GRPO
PHASE_UNFREEZE = "unfreeze"  # last segmenter block joins at a lower LR


def _translation_metrics():
    """Case-sensitive detokenized sacreBLEU plus chrF++ (chrF with word order 2).

    Built once and cached: the signature must stay identical across every
    validation point, or scores from different steps are not comparable.
    """
    global _BLEU_METRIC, _CHRF_METRIC
    if _BLEU_METRIC is None:
        from sacrebleu.metrics import BLEU, CHRF

        _BLEU_METRIC = BLEU()
        _CHRF_METRIC = CHRF(word_order=2)
    return _BLEU_METRIC, _CHRF_METRIC


_BLEU_METRIC = None
_CHRF_METRIC = None


class _SelectionErrorRate:
    """Adapt a maximize-me task metric to the recipe's min-WER selection.

    The base recipe ranks checkpoints with ``min_keys=["WER"]``. A single-task
    non-ASR baseline has no WER at all, and adding a parallel selection path
    would mean two code paths deciding what "best" means. Instead the task's own
    metric is expressed as an error rate, ``100 * (1 - value)``, so the existing
    machinery selects the best-scoring checkpoint with no changes to it.

    This matters for the baselines specifically: with ASR-only selection every
    arm reports a checkpoint chosen by a criterion the arm does not optimize,
    which is how the multi-task runs ended up reporting pre-policy checkpoints.
    """

    def __init__(self, name, value):
        self.name = str(name)
        self.value = float(value)
        # Non-empty: unlike _NoErrorRate, this metric IS defined for the split.
        self.scores = [1]

    def summarize(self, field=None):
        """The metric as an error rate in percent (lower is better)."""
        del field
        return 100.0 * (1.0 - self.value)

    def write_stats(self, stream):
        """Record the real metric, not just its inverted form."""
        stream.write(
            f"Checkpoint selection metric: {self.name} = {self.value:.4f}\n"
            f"Reported as an error rate of {100.0 * (1.0 - self.value):.2f} so "
            "the recipe's min-WER checkpoint selection maximizes it.\n"
        )


class _NoErrorRate:
    """Placeholder for a split where word error rate is not defined.

    Reporting ``nan`` is deliberate: it propagates into the logs as "not
    measured" rather than silently reading as a perfect or a zero score.
    """

    scores = []

    def summarize(self, field=None):
        """Return the not-measured sentinel."""
        del field
        return float("nan")

    def write_stats(self, stream):
        """Explain the absence rather than writing an empty report."""
        stream.write(
            "Word error rate is undefined for this split: it contains no "
            "same-language transcription task. See task_metrics.jsonl for "
            "sacreBLEU and chrF++.\n"
        )


class MultitaskSegmenterASR(SegmenterASR):
    """Segmenter training over several tasks with task-homogeneous batches."""

    # ------------------------------------------------------------- task routing
    def _batch_task_id(self, batch):
        """The single task id carried by a task-homogeneous batch."""
        task_ids = batch.task_id
        if isinstance(task_ids, torch.Tensor):
            unique = torch.unique(task_ids)
            if unique.numel() != 1:
                raise RuntimeError(
                    "Batches must be task homogeneous; got task ids "
                    f"{unique.tolist()}. Check the multi-task sampler."
                )
            return int(unique.item())
        distinct = set(int(t) for t in task_ids)
        if len(distinct) != 1:
            raise RuntimeError(
                f"Batches must be task homogeneous; got task ids {distinct}"
            )
        return distinct.pop()

    def _route_task(self, batch):
        """Point the segmenter FiLM and the decoder adapters at this batch's task."""
        task_id = self._batch_task_id(batch)
        self.active_task_id = task_id
        self.active_task_name = self.task_names[task_id]
        if self.task_router is not None:
            self.task_router.set_task(task_id)
        # A scalar, not a (B,) tensor: the GRPO rollout paths replicate the
        # batch K times, and a scalar expands to whatever batch the segmenter
        # is handed. Every element of a task-homogeneous batch shares this id.
        self.modules.segmenter.active_task_id = torch.tensor(
            task_id, dtype=torch.long, device=self.device
        )
        return task_id

    # --------------------------------------------------------------- phases
    def _phase(self):
        """Resolve the training phase from the current optimizer step."""
        step = int(getattr(self, "optimizer_step", 0))
        if step < int(self.hparams.bridge_optimizer_steps):
            return PHASE_BRIDGE
        film_end = int(self.hparams.bridge_optimizer_steps) + int(
            self.hparams.film_optimizer_steps
        )
        if step < film_end:
            return PHASE_FILM
        return PHASE_UNFREEZE

    def _is_warmup(self, stage):
        """The frozen-policy bridge is a decoder-only warmup for the base class."""
        if stage != sb.Stage.TRAIN:
            return False
        return self._phase() == PHASE_BRIDGE

    def _apply_phase(self, phase):
        """Set trainability to match ``phase``; a no-op once already applied."""
        if getattr(self, "_active_phase", None) == phase:
            return
        segmenter = self.modules.segmenter
        film_params = set(id(p) for p in film_parameters(segmenter))
        pooler_params = set(id(p) for p in segmenter.pooler.parameters())
        last_block = self._last_policy_block_parameters()
        last_block_ids = set(id(p) for p in last_block)

        for param in segmenter.parameters():
            if id(param) in film_params:
                param.requires_grad = phase in (PHASE_FILM, PHASE_UNFREEZE)
            elif id(param) in pooler_params:
                param.requires_grad = bool(self.hparams.train_shared_pooler)
            elif id(param) in last_block_ids:
                param.requires_grad = phase == PHASE_UNFREEZE
            else:
                # The rest of the boundary policy backbone stays frozen for the
                # whole run: the claim is about conditioning a *shared* policy.
                param.requires_grad = False

        trainable = sum(
            p.numel() for p in segmenter.parameters() if p.requires_grad
        )

        def _trainable(params):
            return sum(p.numel() for p in params if p.requires_grad)

        logger.info(
            "Phase %s at optimizer_step=%d: %d trainable segmenter parameters "
            "(film %d/%d, pooler %d/%d, last_block %d/%d trainable/total)",
            phase,
            int(getattr(self, "optimizer_step", 0)),
            trainable,
            _trainable(film_parameters(segmenter)),
            sum(p.numel() for p in film_parameters(segmenter)),
            _trainable(segmenter.pooler.parameters()),
            sum(p.numel() for p in segmenter.pooler.parameters()),
            _trainable(last_block),
            sum(p.numel() for p in last_block),
        )
        self._active_phase = phase

    def _last_policy_block_parameters(self):
        """Parameters of the final segmenter Transformer block, if it has one."""
        net = self.modules.segmenter.net
        for attr in ("layers", "blocks"):
            layers = getattr(net, attr, None)
            if layers is not None and len(layers) > 0:
                return list(layers[-1].parameters())
        encoder = getattr(net, "encoder", None)
        layers = (
            getattr(encoder, "layers", None) if encoder is not None else None
        )
        if layers is not None and len(layers) > 0:
            return list(layers[-1].parameters())
        return []

    # ---------------------------------------------------------- forward / fit
    def compute_forward(self, batch, stage):
        self._route_task(batch)
        return super().compute_forward(batch, stage)

    def fit_batch(self, batch):
        self._apply_phase(self._phase())
        self._route_task(batch)
        loss = super().fit_batch(batch)
        self._task_batch_counts[self.active_task_name] += 1
        return loss

    def compute_objectives(self, predictions, batch, stage):
        loss = super().compute_objectives(predictions, batch, stage)
        if stage == sb.Stage.TRAIN and predictions["mode"] == "joint":
            self._task_loss_sums[self.active_task_name] += float(loss.detach())
        return loss

    # ------------------------------------------------------ candidate scoring
    def _run_decoder_from_segments(
        self, seg_feats, seg_pad, batch, want_hyps=False
    ):
        """Keep the pooled prefix so candidate scoring can reuse it.

        A classification answer is ranked over the label set from the SAME
        pooled features the CE path just used, which is the whole point: the
        audio, the boundaries and the prompt are identical across candidates.
        The base class returns only what CE and free decoding need, so the
        features are stashed here rather than pooled a second time.

        Detached on purpose. Candidate scoring runs under ``no_grad``, and
        holding a live graph reference past ``fit_batch`` would pin the whole
        training-step graph in memory.
        """
        self._last_segments = (seg_feats.detach(), seg_pad)
        return super()._run_decoder_from_segments(
            seg_feats, seg_pad, batch, want_hyps=want_hyps
        )

    def _rate_band(self):
        """Per-task rate band, but only in the phases that allow one.

        Phase B's matched-rate claim depends on every task being held to ONE
        band: that is what makes a placement difference attributable to
        placement rather than to a task simply spending more audio tokens. So
        per-task bands are consulted only in the phases named by
        ``task_rate_bands_phases`` (default: the Phase-C unfreeze phase), and
        every earlier phase uses the global ``rho_lo``/``rho_hi``.
        """
        bands = getattr(self.hparams, "task_rate_bands", None) or {}
        band = bands.get(self.active_task_name)
        if band is None:
            return super()._rate_band()
        phases = getattr(self.hparams, "task_rate_bands_phases", None) or [
            PHASE_UNFREEZE
        ]
        if self._phase() not in phases:
            return super()._rate_band()
        return float(band[0]), float(band[1])

    def _track_reward_std(self, reward_std):
        """Record the within-group reward spread globally and per task."""
        before = len(getattr(self, "_reward_std_samples", []))
        super()._track_reward_std(reward_std)
        task = getattr(self, "active_task_name", None)
        if task is not None:
            self._task_reward_std[task].extend(
                self._reward_std_samples[before:]
            )

    def _is_classification(self):
        """True when the current batch's task is answered by ranking labels."""
        return (
            self.task_output_kind.get(self.active_task_name) == "classification"
        )

    @torch.no_grad()
    def _score_segments(self, seg_feats, seg_pad, batch):
        """``(scores (B, M), gold (B,))`` for one pooled classification batch."""
        candidate_set = self.candidate_sets[self.active_task_name]
        scores = score_candidates(
            self,
            seg_feats,
            seg_pad,
            candidate_set,
            chunk_size=self.candidate_chunk_size,
        )
        gold = candidate_set.gold_indices(batch.wrd, device=scores.device)
        return scores, gold

    def _score_active_candidates(self, batch):
        """Score the batch from the pooled prefix the decoder forward just used."""
        if getattr(self, "_last_segments", None) is None:
            raise RuntimeError(
                f"No pooled segments cached for task {self.active_task_name!r}; "
                "candidate scoring must follow the decoder forward for the "
                "same batch."
            )
        seg_feats, seg_pad = self._last_segments
        return self._score_segments(seg_feats, seg_pad, batch)

    def _rollout_quality_reward(self, feats, pad_mask, boundary, batch, kind):
        """Per-task GRPO reward for one sampled segmentation.

        Generative tasks keep the base class's reward. A classification task
        instead gets the **margin**: the gold label's normalized log-likelihood
        minus the best incorrect label's, under this rollout's boundaries.

        Why the margin and not accuracy. GRPO's signal is the spread of rewards
        *within* an utterance's ``K`` rollouts. Accuracy is 0/1, and four
        segmentations of the same clip almost always agree on the answer, so the
        within-group advantage would be exactly zero on nearly every utterance
        and the policy would receive no gradient. The margin moves continuously
        with the evidence the boundaries preserve, so a segmentation that makes
        the right label a little more likely is rewarded even when the
        prediction does not flip -- and one that erodes the evidence is
        penalized before the answer breaks.

        Why not ``-NLL`` of the label. That rewards raising the gold label's
        likelihood in isolation, which a boundary policy can do by making the
        decoder more confident about everything. The margin is a *contrast*: it
        only improves when the gold label gains on its competitors, which is the
        quantity the task's accuracy actually depends on.

        Cost note: this scores ``M`` candidates per rollout, so a joint step on
        the 31-way intent task runs ~30x the decoder tokens of an ``nll`` step.
        Set ``classification_reward: nll`` to fall back to the cheap reward.
        """
        if not self._is_classification():
            return super()._rollout_quality_reward(
                feats, pad_mask, boundary, batch, kind
            )
        if str(getattr(self.hparams, "classification_reward", "margin")) != (
            "margin"
        ):
            # Ablation path: treat the label string as an ordinary target.
            return super()._rollout_quality_reward(
                feats, pad_mask, boundary, batch, "nll"
            )
        # Only the pooled prefix is needed, so this skips the decoder forward
        # the generative rewards run before scoring.
        seg_feats, seg_pad = self.modules.segmenter.pool(
            feats, pad_mask, boundary
        )
        scores, gold = self._score_segments(seg_feats, seg_pad, batch)
        return candidate_margin(scores, gold)

    # ----------------------------------------------------------- per-task eval
    def _append_wer(self, hyps, llm_logits, batch):
        """Route each batch's scoring to its own task metric."""
        task = self.active_task_name
        if self._is_classification():
            scores, gold = self._score_active_candidates(batch)
            self._task_acc[task].append(
                batch.id, scores.argmax(dim=1).tolist(), gold.tolist()
            )
            return
        preds = self.tokenizer.batch_decode(hyps[0], skip_special_tokens=True)
        target = self._target_tokens(llm_logits, batch)
        target = target.masked_fill(
            target == self.hparams.ignore_index, self.tokenizer.pad_token_id
        )
        refs = self.tokenizer.batch_decode(target, skip_special_tokens=True)

        if self.task_output_kind[task] == "translation":
            self._bleu_hyps[task].extend(p.strip() for p in preds)
            self._bleu_refs[task].extend(r.strip() for r in refs)
            return

        ids = batch.id
        self._task_wer[task].append(
            ids, [p.split(" ") for p in preds], [r.split(" ") for r in refs]
        )
        self._task_cer[task].append(
            ids, [p.split(" ") for p in preds], [r.split(" ") for r in refs]
        )

    def _track_rho_from_boundary(self, boundary, pad_mask):
        """Record the emitted rate globally and per task."""
        before = len(self._rho_samples)
        super()._track_rho_from_boundary(boundary, pad_mask)
        task = getattr(self, "active_task_name", None)
        if task is not None:
            self._task_rho[task].extend(self._rho_samples[before:])

    def _run_step_validation(self, reason):
        """Preserve per-task training accumulators across a validation pause.

        ``on_stage_start(VALID)`` resets the per-task accumulators, and the base
        class only saves and restores its own global ``_rho_samples``. Without
        this, every step validation would silently wipe the training-side
        per-task rate, loss, and mixture-share statistics.
        """
        saved = (
            self._task_rho,
            self._task_reward_std,
            self._task_batch_counts,
            self._task_loss_sums,
            getattr(self, "active_task_name", "asr"),
            getattr(self, "active_task_id", 0),
        )
        try:
            super()._run_step_validation(reason)
        finally:
            (
                self._task_rho,
                self._task_reward_std,
                self._task_batch_counts,
                self._task_loss_sums,
                self.active_task_name,
                self.active_task_id,
            ) = saved
            # Validation left the router pointing at whichever task ran last.
            if self.task_router is not None:
                self.task_router.set_task(self.active_task_id)

    def on_stage_start(self, stage, epoch):
        super().on_stage_start(stage, epoch)
        if stage == sb.Stage.TRAIN:
            self._task_batch_counts = defaultdict(int)
            self._task_loss_sums = defaultdict(float)
        self._task_rho = defaultdict(list)
        self._task_reward_std = defaultdict(list)
        if stage != sb.Stage.TRAIN:
            self._task_wer = {
                name: self.hparams.error_rate_computer()
                for name, kind in self.task_output_kind.items()
                if kind == "text"
            }
            self._task_cer = {
                name: self.hparams.cer_computer()
                for name, kind in self.task_output_kind.items()
                if kind == "text"
            }
            self._bleu_hyps = defaultdict(list)
            self._bleu_refs = defaultdict(list)
            self._task_acc = {
                name: CandidateAccuracy(self.candidate_sets[name].candidates)
                for name, kind in self.task_output_kind.items()
                if kind == "classification"
            }
        self._last_segments = None

    def _task_stats(self):
        """Per-task quality, rate, and (training) mixture-realization stats."""
        stats = {}
        for task, metric in getattr(self, "_task_wer", {}).items():
            if metric.scores:
                stats[f"WER_{task}"] = metric.summarize("error_rate")
        for task, metric in getattr(self, "_task_cer", {}).items():
            if metric.scores:
                stats[f"CER_{task}"] = metric.summarize("error_rate")
        for task, hyps in getattr(self, "_bleu_hyps", {}).items():
            refs = self._bleu_refs[task]
            if not hyps:
                continue
            bleu_metric, chrf_metric = _translation_metrics()
            stats[f"BLEU_{task}"] = bleu_metric.corpus_score(hyps, [refs]).score
            stats[f"chrF2_{task}"] = chrf_metric.corpus_score(
                hyps, [refs]
            ).score
            # The signature pins tokenization/case handling, which is what
            # makes a reported BLEU comparable to anyone else's.
            stats[f"BLEU_signature_{task}"] = str(bleu_metric.get_signature())
            stats[f"chrF2_signature_{task}"] = str(chrf_metric.get_signature())
        for task, metric in getattr(self, "_task_acc", {}).items():
            if not metric.scores:
                continue
            stats[f"acc_{task}"] = metric.summarize("accuracy")
            # CREMA-D's crowd voice labels are about half neutral, so accuracy
            # alone cannot distinguish a real six-way model from a majority
            # predictor. Macro-F1 travels with it everywhere.
            stats[f"macro_f1_{task}"] = metric.summarize("macro_f1")
            stats[f"n_{task}"] = len(metric.scores)
        for task, samples in getattr(self, "_task_reward_std", {}).items():
            if not samples:
                continue
            spread = torch.tensor(samples)
            stats[f"reward_std_{task}"] = float(spread.mean())
            stats[f"zero_variance_share_{task}"] = float(
                (spread <= 1e-6).float().mean()
            )
        for task, samples in getattr(self, "_task_rho", {}).items():
            if not samples:
                continue
            rho = torch.tensor(samples)
            # f_audio = 50 Hz x rho: reader-facing rate is always in Hz.
            stats[f"f_audio_hz_{task}"] = float(rho.mean()) * 50.0
            stats[f"f_audio_hz_std_{task}"] = (
                float(rho.std()) * 50.0 if rho.numel() > 1 else 0.0
            )
        return stats

    def on_stage_end(self, stage, stage_loss, epoch):
        task_stats = self._task_stats()
        if stage == sb.Stage.TRAIN:
            counts = getattr(self, "_task_batch_counts", {})
            total = sum(counts.values())
            for task, count in counts.items():
                task_stats[f"batch_share_{task}"] = (
                    count / total if total else 0.0
                )
                if count:
                    task_stats[f"loss_{task}"] = (
                        self._task_loss_sums[task] / count
                    )
            task_stats["phase"] = getattr(self, "_active_phase", PHASE_BRIDGE)
        self._pending_task_stats = task_stats

        selection = str(getattr(self.hparams, "selection_metric", "") or "")
        if stage != sb.Stage.TRAIN and selection:
            # Single-task baselines: rank on the task's own metric. Without
            # this, a run with no ASR source has nothing to select on at all.
            if selection not in task_stats:
                if stage == sb.Stage.VALID:
                    raise RuntimeError(
                        f"selection_metric={selection!r} was not produced by "
                        "this validation pass; available metrics: "
                        f"{sorted(task_stats)}"
                    )
                self.wer_metric = _NoErrorRate()
                self.cer_metric = _NoErrorRate()
            else:
                self.wer_metric = _SelectionErrorRate(
                    selection, task_stats[selection]
                )
                # CER has no meaning for a classification task; nan reads as
                # "not measured" rather than as a perfect score.
                self.cer_metric = _NoErrorRate()
        elif stage != sb.Stage.TRAIN:
            # The base class selects checkpoints on the aggregate WER metric.
            # Feed it the ASR task's metric so checkpoint selection stays a
            # pure ASR-retention criterion, and report ST separately.
            asr_metric = getattr(self, "_task_wer", {}).get("asr")
            if asr_metric is not None and asr_metric.scores:
                self.wer_metric = asr_metric
                self.cer_metric = self._task_cer["asr"]
            elif stage == sb.Stage.VALID:
                raise RuntimeError(
                    "The validation mixture contains no ASR utterances, so "
                    "LibriSpeech retention cannot be checked and checkpoint "
                    "selection has nothing to rank. Add an ASR source to "
                    "valid_sources."
                )
            else:
                # A translation-only test split legitimately has no WER; the
                # base class would otherwise divide by zero summarizing an
                # empty metric. BLEU/chrF++ for this split are in task_stats.
                self.wer_metric = _NoErrorRate()
                self.cer_metric = _NoErrorRate()

        super().on_stage_end(stage, stage_loss, epoch)

        if stage == sb.Stage.TRAIN:
            self.train_stats.update(task_stats)
            return
        if if_main_process():
            self._write_task_metrics(stage, epoch, task_stats)

    def _write_task_metrics(self, stage, epoch, task_stats):
        """Append one JSONL row per validation/test point for later analysis."""
        path = getattr(self.hparams, "task_metrics_file", None)
        if not path:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        row = {
            "stage": stage.name,
            "epoch": epoch,
            "optimizer_step": int(getattr(self, "optimizer_step", 0)),
            "phase": getattr(self, "_active_phase", PHASE_BRIDGE),
            **task_stats,
        }
        with open(path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        logger.info(
            "Per-task metrics @ step %d: %s",
            row["optimizer_step"],
            ", ".join(
                f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in task_stats.items()
            ),
        )

    # ------------------------------------------------------------- optimizers
    def _phase_trainable_segmenter_parameters(self):
        """Every segmenter parameter that some phase will train.

        A torch optimizer snapshots its parameter list at construction, so a
        parameter frozen during the bridge but unfrozen in a later phase must
        still be registered now. Freezing is expressed through ``requires_grad``
        alone: AdamW skips parameters whose gradient is ``None``.
        """
        segmenter = self.modules.segmenter
        collected = []
        seen = set()
        groups = [film_parameters(segmenter)]
        if bool(self.hparams.train_shared_pooler):
            groups.append(list(segmenter.pooler.parameters()))
        groups.append(self._last_policy_block_parameters())
        for group in groups:
            for param in group:
                if id(param) not in seen:
                    seen.add(id(param))
                    collected.append(param)
        return collected

    def init_optimizers(self):
        """Segmenter (FiLM/pooler/last block) and decoder (proj + routed LoRA).

        Group order matches the base recipe -- group 0 is the segmenter and
        group 1 the decoder -- because ``_set_decoder_phase_lr`` addresses the
        decoder group by index.
        """
        seg_params = self._phase_trainable_segmenter_parameters()
        if not seg_params:
            raise ValueError(
                "The segmenter optimizer has no trainable parameters"
            )
        dec_params = list(self.modules.proj.parameters()) + [
            p for p in self.modules.llm.parameters() if p.requires_grad
        ]
        if not dec_params:
            raise ValueError(
                "The decoder optimizer has no trainable parameters"
            )
        self.optimizer = torch.optim.AdamW(
            [
                {"params": seg_params, "lr": float(self.hparams.lr_segmenter)},
                {"params": dec_params, "lr": float(self.hparams.lr_decoder)},
            ],
            weight_decay=float(self.hparams.weight_decay),
        )
        self.optimizers_dict = {"model_optimizer": self.optimizer}
        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("model_optimizer", self.optimizer)
            self.checkpointer.add_recoverable("proj", self.modules.proj)
            self.checkpointer.add_recoverable("llm", self.modules.llm)

        parameters = count_task_conditioned_parameters(
            self.modules.segmenter, self.modules.llm
        )
        logger.info("Task-conditioned parameters: %s", parameters)
        path = getattr(self.hparams, "task_metrics_file", None)
        if path and if_main_process():
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps({"event": "parameters", **parameters}) + "\n"
                )


def validate_task_table(hparams):
    """Fail loudly when the YAML task count and ``TASK_SPECS`` disagree.

    These are two independent sources of truth for the same number, and they
    fail asymmetrically: ``segmenter_num_tasks`` sizes the FiLM table and the
    routed-LoRA slots at construction, while ``expand_film_to_tasks`` later
    resizes FiLM to ``NUM_TASKS``. Configure 5 with a 2-task ``TASK_SPECS`` and
    FiLM is silently rebuilt with 2 rows against a 5-slot router, so task ids
    2-4 index out of range -- at the first batch of that task, not at startup.
    Configure 2 with a 5-task table and the router rejects the same ids.
    Neither is worth debugging from a stack trace an hour into a run.
    """
    configured = int(hparams["segmenter_num_tasks"])
    if configured != NUM_TASKS:
        raise ValueError(
            f"segmenter_num_tasks={configured} but multitask_data.TASK_SPECS "
            f"defines {NUM_TASKS} tasks "
            f"({', '.join(f'{n}={s.task_id}' for n, s in TASK_SPECS.items())}). "
            "TASK_SPECS owns the task order; set segmenter_num_tasks to match "
            "it. Note that changing this number changes the FiLM and routed "
            "LoRA shapes, so a checkpoint written under a different task count "
            "cannot be resumed -- warm start from the single-task ASR "
            "checkpoint instead."
        )


def validate_selection_metric(hparams):
    """Fail fast on a mis-typed ``selection_metric``.

    A typo here would not surface until the first validation pass, and for the
    ASR-WER default it would not surface at all -- the run would simply keep
    selecting on the wrong criterion and every reported number would come from
    an arbitrary checkpoint.
    """
    selection = str(hparams.get("selection_metric") or "")
    if not selection:
        return None
    families = ("acc", "macro_f1", "WER", "CER")
    for family in families:
        prefix = family + "_"
        if selection.startswith(prefix):
            task = selection[len(prefix):]
            if task not in TASK_SPECS:
                raise ValueError(
                    f"selection_metric={selection!r} names task {task!r}, "
                    f"which is not in the task table: {sorted(TASK_SPECS)}"
                )
            if family in ("acc", "macro_f1") and not is_classification(task):
                raise ValueError(
                    f"selection_metric={selection!r} asks for a "
                    f"classification metric on task {task!r}, which is "
                    "generative; use WER_/CER_ instead"
                )
            direction = "min" if family in ("WER", "CER") else "max"
            logger.info(
                "Checkpoint selection: %s (%simize), not the default ASR WER",
                selection,
                direction,
            )
            return selection
    raise ValueError(
        f"selection_metric={selection!r} does not name a known metric family; "
        f"expected one of {['%s_<task>' % f for f in families]}"
    )


def validate_task_rate_bands(hparams):
    """Check every configured per-task band before a GPU is touched.

    A band is a dead zone: the rate terms are exactly zero inside
    ``[rho_lo, rho_hi]`` and quadratic outside, so ``rho_hi`` is the rate the
    policy is pushed down to and ``rho_lo`` is the floor below which it is
    pushed back up. Reported in Hz as ``f_audio = 50 * rho``.
    """
    bands = hparams.get("task_rate_bands") or {}
    for name, band in bands.items():
        if name not in TASK_SPECS:
            raise ValueError(
                f"task_rate_bands names unknown task {name!r}; known tasks: "
                f"{sorted(TASK_SPECS)}"
            )
        if len(band) != 2:
            raise ValueError(
                f"task_rate_bands[{name}] must be [rho_lo, rho_hi], got {band}"
            )
        lo, hi = float(band[0]), float(band[1])
        if not 0.0 < lo <= hi < 1.0:
            raise ValueError(
                f"task_rate_bands[{name}] = [{lo}, {hi}] must satisfy "
                "0 < rho_lo <= rho_hi < 1"
            )
    phases = hparams.get("task_rate_bands_phases") or [PHASE_UNFREEZE]
    known = {PHASE_BRIDGE, PHASE_FILM, PHASE_UNFREEZE}
    unknown = [p for p in phases if p not in known]
    if unknown:
        raise ValueError(
            f"task_rate_bands_phases names unknown phases {unknown}; "
            f"known phases: {sorted(known)}"
        )
    if bands and PHASE_FILM in phases:
        logger.warning(
            "task_rate_bands are active during the %s phase. Phase B's "
            "matched-rate comparison requires every task to share ONE band; "
            "a placement result measured this way is not rate-matched.",
            PHASE_FILM,
        )
    if bands:
        logger.info(
            "Per-task rate bands (active in phases %s): %s",
            phases,
            ", ".join(
                f"{n}={50 * float(b[0]):.1f}-{50 * float(b[1]):.1f} Hz"
                for n, b in sorted(bands.items())
            ),
        )


def _build_candidate_sets(hparams, tokenizer):
    """Tokenized label sets for every classification task in the table.

    ``candidate_subsets`` restricts a task to part of its canonical label set
    (the survey's 0-4 speaker-count pilot inside the 0-10 vocabulary) without
    touching task ids or checkpoint shapes.
    """
    subsets = hparams.get("candidate_subsets") or {}
    sets = {}
    for name in TASK_SPECS:
        if not is_classification(name):
            continue
        sets[name] = CandidateSet(
            tokenizer,
            prompt_of(name),
            resolve_candidates(name, subsets),
            bos_index=hparams["bos_index"],
            eos_index=hparams["eos_index"],
            start_of_audio_index=tokenizer.convert_tokens_to_ids(
                "<|start_of_audio|>"
            ),
            end_of_audio_index=tokenizer.convert_tokens_to_ids(
                "<|end_of_audio|>"
            ),
            ignore_index=hparams["ignore_index"],
        )
    return sets


def _setup_task_metadata(brain, hparams):
    """Attach the task table the brain needs before any batch is seen."""
    brain.task_names = {
        task_id: name for name, (task_id, *_) in TASK_SPECS.items()
    }
    brain.task_output_kind = {
        name: (
            "classification"
            if is_classification(name)
            else "translation"
            if name.startswith("st_")
            else "text"
        )
        for name in TASK_SPECS
    }
    brain.candidate_sets = _build_candidate_sets(hparams, brain.tokenizer)
    brain.candidate_chunk_size = hparams.get("candidate_chunk_size")
    brain.active_task_id = task_id_of("asr")
    brain.active_task_name = "asr"
    brain.task_router = hparams.get("task_router")
    brain._active_phase = None
    brain._task_rho = defaultdict(list)
    brain._task_reward_std = defaultdict(list)
    brain._task_batch_counts = defaultdict(int)
    brain._task_loss_sums = defaultdict(float)


def _load_multitask_warm_start(brain, hparams):
    """Warm start from the single-task LS960 ASR checkpoint.

    Restores segmenter, pooler, projection, and normalization verbatim, then
    copies the trained ASR LoRA into task slot 0. The ST slot keeps its
    zero-initialized up projection, so at step zero the ST path is exactly the
    frozen base LLM and the ASR path is exactly the warm-started model.
    """
    ckpt_dir = hparams["warm_start_ckpt_dir"]
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"Warm-start checkpoint not found: {ckpt_dir}")

    seg_path = os.path.join(ckpt_dir, "segmenter.ckpt")
    state = torch.load(seg_path, map_location="cpu", weights_only=True)
    missing, unexpected = brain.modules.segmenter.load_state_dict(
        state, strict=False
    )
    # FiLM is the only thing a single-task checkpoint cannot contain.
    unexpected = [k for k in unexpected if not k.startswith("film.")]
    if unexpected:
        raise ValueError(f"Unexpected segmenter tensors: {unexpected[:5]}")
    non_film_missing = [k for k in missing if not k.startswith("film.")]
    if non_film_missing:
        raise ValueError(
            f"Segmenter checkpoint is missing tensors: {non_film_missing[:5]}"
        )
    logger.info("Restored segmenter from %s", seg_path)

    proj_path = os.path.join(ckpt_dir, "proj.ckpt")
    brain.modules.proj.load_state_dict(
        torch.load(proj_path, map_location="cpu", weights_only=True)
    )
    logger.info("Restored projection from %s", proj_path)

    norm_path = os.path.join(ckpt_dir, "normalize.ckpt")
    if os.path.isfile(norm_path):
        brain.hparams.normalize._load(norm_path, end_of_epoch=False)
        logger.info("Restored input normalization from %s", norm_path)

    llm_path = os.path.join(ckpt_dir, "llm.ckpt")
    lora_state = torch.load(llm_path, map_location="cpu", weights_only=True)
    copy_asr_lora_into_task_slot(
        brain.modules.llm, lora_state, task_id=task_id_of("asr")
    )

    expand_film_to_tasks(brain.modules.segmenter, NUM_TASKS)
    brain.modules.segmenter.to(brain.device)


def max_utterance_seconds(hparams):
    """Longest utterance the boundary policy can roll out, in seconds.

    The autoregressive policy indexes a positional table of
    ``segmenter_ar_max_positions`` entries on a 50 Hz frame grid, and raises
    once a rollout runs past it. Deriving the cap from that number keeps the
    data filter and the model limit from drifting apart.
    """
    if str(hparams.get("segmenter_backbone", "")).lower() != "transformer_ar":
        return None
    positions = hparams.get("segmenter_ar_max_positions")
    if not positions:
        return None
    # One frame of slack: the policy rejects position == max_positions.
    return (int(positions) - 1) / 50.0


def _build_manifests(hparams):
    """Assemble merged train/valid manifests from the configured sources."""
    manifest_dir = hparams["multitask_manifest_dir"]
    os.makedirs(manifest_dir, exist_ok=True)
    cap = max_utterance_seconds(hparams)

    train_csv = os.path.join(manifest_dir, "train_multitask.csv")
    counts = build_multitask_manifest(
        hparams["train_sources"],
        train_csv,
        data_folder=hparams["data_folder"],
        max_duration=cap,
    )
    valid_csv = os.path.join(manifest_dir, "valid_multitask.csv")
    build_multitask_manifest(
        hparams["valid_sources"],
        valid_csv,
        data_folder=hparams["data_folder"],
        max_duration=cap,
    )
    return train_csv, valid_csv, counts


if __name__ == "__main__":
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    validate_task_table(hparams)
    validate_task_rate_bands(hparams)
    validate_selection_metric(hparams)
    _attach_segmenter_ssl(hparams)
    sb.utils.distributed.ddp_init_group(run_opts)
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    train_csv, valid_csv, source_counts = _build_manifests(hparams)
    logger.info("Merged manifest sources: %s", source_counts)

    tokenizer = hparams["llm"].tokenizer
    train_data = multitask_dataio_prepare(
        hparams, tokenizer, train_csv, sorting="random"
    )
    valid_data = multitask_dataio_prepare(
        hparams, tokenizer, valid_csv, sorting="ascending"
    )

    train_sampler = MultiSourceDynamicBatchSampler(
        train_data,
        source_keys=hparams["sampling_source_keys"],
        probabilities=hparams["sampling_probabilities"],
        max_batch_length=hparams["max_batch_length_train"],
        num_buckets=hparams["num_bucket"],
        max_batch_ex=hparams["max_batch_ex"],
        min_batch_ex=hparams["min_batch_ex_train"],
        batches_per_epoch=hparams.get("multitask_batches_per_epoch"),
        seed=hparams["seed"],
    )
    valid_sampler = MultiSourceDynamicBatchSampler(
        valid_data,
        source_keys=hparams["valid_source_keys"],
        probabilities=hparams["valid_source_probabilities"],
        max_batch_length=hparams["max_batch_length_val"],
        num_buckets=hparams["num_bucket"],
        max_batch_ex=hparams["max_batch_ex"],
        seed=hparams["seed"],
        exhaustive=True,
    )

    effective_grad_accumulation = resolve_effective_grad_accumulation(
        hparams["grad_accumulation_factor"],
        run_opts.grad_accumulation_factor,
        "grad_accumulation_factor" in run_opts.overridden_args,
    )
    hparams["grad_accumulation_factor"] = effective_grad_accumulation
    # The base class treats the bridge as its decoder-only warmup.
    hparams["warmup_optimizer_steps"] = int(hparams["bridge_optimizer_steps"])

    asr_brain = MultitaskSegmenterASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    asr_brain.optimizer_step_limit = resolve_optimizer_step_limit(
        asr_brain.optimizer_step_limit
    )
    asr_brain.tokenizer = tokenizer
    asr_brain.txt_embedding = (
        asr_brain.raw_modules.llm.model.get_input_embeddings()
    )
    asr_brain.current_epoch = 1
    _setup_task_metadata(asr_brain, hparams)
    _load_multitask_warm_start(asr_brain, hparams)

    for p in asr_brain.modules.ssl.parameters():
        p.requires_grad = False
    segmenter_ssl = getattr(asr_brain.modules, "segmenter_ssl", None)
    if segmenter_ssl is not None:
        for p in segmenter_ssl.parameters():
            p.requires_grad = False
    # Establish the bridge phase before the optimizer is built so the logged
    # trainable set is the one the first optimizer step will actually see.
    asr_brain._apply_phase(PHASE_BRIDGE)

    loader_options = {
        "num_workers": hparams["num_workers"],
        "pin_memory": hparams["pin_memory"],
        "persistent_workers": hparams["persistent_workers"],
    }
    if hparams["num_workers"] > 0:
        loader_options["prefetch_factor"] = hparams["prefetch_factor"]
    collate_fn = hparams["train_dataloader_opts"]["collate_fn"]
    train_dl = {
        "batch_sampler": train_sampler,
        "collate_fn": collate_fn,
        **loader_options,
    }
    valid_dl = {"batch_sampler": valid_sampler, "collate_fn": collate_fn}

    if bool(hparams.get("eval_only", False)):
        asr_brain.init_optimizers()
        logger.info("Evaluation-only mode: skipping fit")
    else:
        fit_valid_data = valid_data
        if int(hparams.get("validation_interval_optimizer_steps", 0)) > 0:
            asr_brain.configure_step_validation(valid_data, valid_dl)
            fit_valid_data = None
        asr_brain.fit(
            asr_brain.hparams.epoch_counter,
            train_data,
            fit_valid_data,
            train_loader_kwargs=train_dl,
            valid_loader_kwargs=valid_dl,
        )

    os.makedirs(hparams["output_wer_folder"], exist_ok=True)
    for name, spec in hparams["test_source_specs"].items():
        csv_path = hparams["test_sources"][name]
        build_multitask_manifest(
            spec,
            csv_path,
            data_folder=hparams["data_folder"],
            max_duration=max_utterance_seconds(hparams),
        )
        test_data = multitask_dataio_prepare(
            hparams, tokenizer, csv_path, sorting="ascending"
        )
        test_sampler = MultiSourceDynamicBatchSampler(
            test_data,
            source_keys=[hparams["test_source_keys"][name]],
            probabilities=[1.0],
            max_batch_length=hparams["max_batch_length_val"],
            num_buckets=hparams["num_bucket"],
            max_batch_ex=hparams["max_batch_ex"],
            seed=hparams["seed"],
            exhaustive=True,
        )
        asr_brain.hparams.evaluation_split = name
        asr_brain.hparams.test_wer_file = os.path.join(
            hparams["output_wer_folder"], f"wer_{name}.txt"
        )
        asr_brain.evaluate(
            test_data,
            min_key="WER",
            test_loader_kwargs={
                "batch_sampler": test_sampler,
                "collate_fn": collate_fn,
            },
        )
