#!/usr/bin/env python3
"""Train a learned segmenter in front of a SpeechLLM (char cold-start -> joint RL).

Two ``segmenter_mode``s (run as two chained jobs; the joint run loads the cold-start
segmenter via ``coldstart_ckpt_dir``):

  * ``coldstart`` — supervised. Train the boundary policy with BCE against char-level
    CTC boundaries (``boundary_target_dir`` = ``wavlm_boundaries/char``). No LLM.
    Produces the segmenter init AND (evaluated with those boundaries) the B1 baseline.

  * ``joint`` — one run with a warmup front phase:
      - epochs ``<= warmup_epochs``: **decoder-warmup** — segmenter frozen at the
        cold-start, train only the decoder (proj + LoRA) with CE on the *argmax*
        segmentation. Reward model becomes competent at the compressed rate.
      - later epochs: **joint** — decoder CE on the argmax segmentation (grad ->
        proj+LoRA) **+** GRPO policy-gradient on the segmenter (K sampled
        segmentations, reward = ``-NLL`` from the frozen-forward decoder, group-relative
        advantage) **+** the differentiable rate objective (one-sided cap + token-tax)
        **+** an annealed entropy bonus. Encoder (SSL) frozen throughout.

``gamma = 1`` (bandit), pg reduced with ``.mean()``, no KL-to-cold-start. No WER reward
in v1 (that needs decoding — deferred to v2). See ``plans/dynamic_segmenter_rl_v1.md``.

Authors
-------
 * Speech-LLM segmenter, 2026 (based on train_speechllm.py by Adel Moumen, 2025)
"""

import glob
import json
import os
import sys
import time
from contextlib import contextmanager, nullcontext

import torch
import torch.nn.functional as F
from bilevel import (
    capture_rng_state,
    decoder_batch_view,
    replay_rng_state,
    reward_rank_diagnostics,
    support_query_slices,
    temporary_sgd_update,
    unique_trainable_parameters,
)
from hyperpyyaml import load_hyperpyyaml
from segment_pooling import lengths_to_padding_mask, padding_mask_to_lengths
from segmenter import (
    boundary_bce_loss,
    boundary_prf,
    entropy_bonus,
    group_advantage,
    grpo_pg_loss,
    rate_loss,
    sampled_rate_penalty,
)
from torch.nn.attention import SDPBackend, sdpa_kernel
from train_speechllm import (  # reuse the base recipe's plumbing
    ASR,
    dataio_prepare,
    get_multimodal_attention_mask,
)

import speechbrain as sb
from speechbrain.utils.distributed import if_main_process, run_on_main
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


def _levenshtein(a, b):
    """Edit distance between two sequences (lists/strings)."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


class SegmenterASR(ASR):
    """Brain that trains a learned segmenter (cold-start BCE, then joint GRPO)."""

    # ------------------------------------------------------------------ helpers
    def _encoder_features(self, batch):
        """Frozen decoder/segmenter SSL features on one shared frame grid.

        The default path reuses ``modules.ssl`` for both consumers. When a
        ``modules.segmenter_ssl`` is configured, it supplies only the boundary
        policy input while the decoder continues to pool ``modules.ssl`` features.
        Both encoders must emit exactly the same number of frames because a boundary
        index predicted on one stream is applied directly to the other stream.

        Returns ``(decoder_feats, segmenter_feats, rel_lens)``.
        """
        if bool(getattr(self.hparams, "use_feats", False)):
            if getattr(batch, "feats", None) is None:
                raise ValueError("`use_feats=True` but batch has no `feats`.")
            if getattr(self.modules, "segmenter_ssl", None) is not None:
                raise ValueError(
                    "Dual-SSL mode is incompatible with `use_feats=True`: the "
                    "batch provides only one cached feature stream."
                )
            feats, feat_lens = batch.feats
            return feats, feats, feat_lens
        wavs, wav_lens = batch.sig
        wavs = self.hparams.normalize(wavs, wav_lens)
        decoder_feats = self.modules.ssl(wavs, wav_lens)
        segmenter_ssl = getattr(self.modules, "segmenter_ssl", None)
        segmenter_feats = (
            decoder_feats
            if segmenter_ssl is None
            else segmenter_ssl(wavs, wav_lens)
        )
        if decoder_feats.shape[:2] != segmenter_feats.shape[:2]:
            raise RuntimeError(
                "Decoder and segmenter SSL frame grids do not match: "
                f"decoder={tuple(decoder_feats.shape)}, "
                f"segmenter={tuple(segmenter_feats.shape)}. Boundaries cannot be "
                "transferred safely between these streams."
            )
        if segmenter_ssl is not None and not getattr(
            self, "_dual_ssl_grid_logged", False
        ):
            logger.info(
                "Dual-SSL frame grid verified: T=%d; segmenter_dim=%d; "
                "decoder_dim=%d",
                decoder_feats.size(1),
                segmenter_feats.size(2),
                decoder_feats.size(2),
            )
            self._dual_ssl_grid_logged = True
        return decoder_feats, segmenter_feats, wav_lens

    def _run_decoder(self, feats, pad_mask, boundary, batch, want_hyps=False):
        """Pool by ``boundary`` -> proj -> LLM. Returns dict(llm_logits, seg_lens, hyps).

        Runs under the caller's grad context (grad on for the decoder-CE path, wrapped in
        ``no_grad`` by the caller for the reward samples).
        """
        seg_feats, seg_pad = self.modules.segmenter.pool(
            feats, pad_mask, boundary
        )
        return self._run_decoder_from_segments(
            seg_feats, seg_pad, batch, want_hyps=want_hyps
        )

    def _run_decoder_from_segments(
        self, seg_feats, seg_pad, batch, want_hyps=False
    ):
        """Run projection and LLM from an already pooled acoustic prefix."""
        tokens_bos, tokens_bos_lens = batch.tokens_bos
        seg_lens = padding_mask_to_lengths(seg_pad)
        projected = self.modules.proj(seg_feats)
        txt_embds = self.txt_embedding(tokens_bos)
        multimodal = torch.cat(
            [txt_embds[:, 0].unsqueeze(1), projected, txt_embds[:, 1:]], dim=1
        )
        attn = get_multimodal_attention_mask(
            projected, seg_lens, txt_embds, tokens_bos_lens, self.device
        )
        llm_logits = self.modules.llm(
            inputs_embeds=multimodal, attention_mask=attn
        ).logits
        out = {
            "llm_logits": llm_logits,
            "seg_pad": seg_pad,
            "seg_lens": seg_lens,
        }
        if want_hyps:
            prompt_len = batch.prompt_len
            n = projected.shape[1] + int(prompt_len[0].item())
            out["hyps"] = self.modules.searcher(
                multimodal[:, :n], seg_lens, attn[:, :n]
            )
        return out

    def _decode_replicated(
        self,
        feats,
        pad_mask,
        boundary,
        tokens_bos,
        tokens_bos_lens,
        prompt_len,
        want_hyps,
    ):
        """Decode for a K-replicated rollout batch from explicit tensors (no ``batch``).

        Mirrors :meth:`_run_decoder` but takes already-replicated ``(K*B, ...)`` tensors,
        so all K samples of every utterance are pooled + decoded in **one** batched
        forward/generation instead of a Python loop of K sequential decodes.
        """
        seg_feats, seg_pad = self.modules.segmenter.pool(
            feats, pad_mask, boundary
        )
        seg_lens = padding_mask_to_lengths(seg_pad)
        projected = self.modules.proj(seg_feats)
        txt_embds = self.txt_embedding(tokens_bos)
        multimodal = torch.cat(
            [txt_embds[:, 0].unsqueeze(1), projected, txt_embds[:, 1:]], dim=1
        )
        attn = get_multimodal_attention_mask(
            projected, seg_lens, txt_embds, tokens_bos_lens, self.device
        )
        if want_hyps:
            # KV-cached HF generate for the free-running reward decode. The SpeechBrain
            # greedy searcher re-runs the FULL prefix+generated forward every step (no KV
            # cache => O(L^2), compute-bound) which dominated the CER step; generate reuses
            # past_key_values (O(L)). All sequences share prefix length n and end with the
            # prompt, so batched generation is well-formed (interior segment-pad handled by
            # the attention mask). LoRA is active inside adapted_model.
            n = projected.shape[1] + int(prompt_len[0].item())
            max_new = max(
                8,
                int(float(self.hparams.max_decode_ratio) * projected.shape[1]),
            )
            gen = self.modules.llm.adapted_model.generate(
                inputs_embeds=multimodal[:, :n],
                attention_mask=attn[:, :n],
                max_new_tokens=max_new,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                eos_token_id=int(self.hparams.eos_index),
                pad_token_id=int(self.hparams.pad_token),
            )
            return (gen,)  # mimic searcher's hyps[0] == token ids
        return self.modules.llm(
            inputs_embeds=multimodal, attention_mask=attn
        ).logits

    def _utterance_nll_rep(self, llm_logits, tokens_eos):
        """Per-utterance teacher-forced NLL ``(N,)`` for a replicated rollout batch."""
        n_audio = llm_logits.shape[1] - tokens_eos.shape[1]
        target = torch.cat(
            [
                torch.full(
                    (tokens_eos.shape[0], n_audio),
                    self.hparams.ignore_index,
                    device=self.device,
                ),
                tokens_eos,
            ],
            dim=1,
        ).long()
        ce = F.cross_entropy(
            llm_logits.reshape(-1, llm_logits.shape[-1]),
            target.reshape(-1),
            ignore_index=self.hparams.ignore_index,
            reduction="none",
        ).reshape(llm_logits.shape[0], -1)
        valid = (target != self.hparams.ignore_index).float()
        return ce.sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

    def _rollout_rewards(self, feats, pad_mask, sampled, batch, reward_kind):
        """Detached rewards ``(B, K)`` for K sampled segmentations via ONE batched decode.

        Replicates the utterance batch K times along dim 0, concatenates the K sampled
        boundaries, and does a single batched decode (free-running for cer/wer, or a
        teacher-forced forward for nll). This turns K sequential decodes into one wide
        batched generation — the main cost amortization for the CER reward.
        """
        K = len(sampled)
        Bsz = feats.size(0)
        boundary_cat = torch.cat(sampled, dim=0)  # (K*B, T)
        feats_rep = feats.repeat(K, 1, 1)
        pad_rep = pad_mask.repeat(K, 1)
        tokens_bos, tokens_bos_lens = batch.tokens_bos
        tb_rep = tokens_bos.repeat(K, 1)
        tbl_rep = tokens_bos_lens.repeat(K)
        if reward_kind == "nll":
            llm_logits = self._decode_replicated(
                feats_rep,
                pad_rep,
                boundary_cat,
                tb_rep,
                tbl_rep,
                None,
                want_hyps=False,
            )
            tokens_eos, _ = batch.tokens_eos
            err = self._utterance_nll_rep(llm_logits, tokens_eos.repeat(K, 1))
        else:  # cer / wer -- free-running batched generation
            hyps = self._decode_replicated(
                feats_rep,
                pad_rep,
                boundary_cat,
                tb_rep,
                tbl_rep,
                batch.prompt_len.repeat(K),
                want_hyps=True,
            )
            err = self._utterance_error(
                hyps, batch, reward_kind, refs=list(batch.wrd) * K
            )
            if not getattr(self, "_rl_dbg_printed", False):
                self._rl_dbg_printed = True
                preds = self.tokenizer.batch_decode(
                    hyps[0], skip_special_tokens=True
                )
                for i in range(min(3, len(preds))):
                    logger.info(
                        "reward-decode sanity | %s=%.3f | hyp=%r | ref=%r",
                        reward_kind,
                        float(err[i]),
                        preds[i],
                        (list(batch.wrd) * K)[i],
                    )
        # err order is [s0:b0..bB-1, s1:b0..bB-1, ...] -> (K, B) -> (B, K).
        return (-err).reshape(K, Bsz).t().contiguous()

    def _bilevel_mode(self):
        """Resolve and validate the opt-in support/query reward mode."""
        mode = str(getattr(self.hparams, "bilevel_mode", "off")).lower()
        allowed = {
            "off",
            "heldout",
            "decoder_lookahead",
            "decoder_pooler_lookahead",
        }
        if mode not in allowed:
            raise ValueError(
                f"Unknown bilevel_mode={mode!r}; expected one of {sorted(allowed)}"
            )
        return mode

    def _bilevel_inner_parameters(self, mode):
        """Parameters changed by the temporary inner step, never optimizer state."""
        # Use the same module objects as ``_run_decoder``.  This also remains
        # correct if SpeechBrain wraps trainable modules for DP/DDP.
        modules = [self.modules.proj, self.modules.llm]
        if mode == "decoder_pooler_lookahead":
            modules.append(self.modules.segmenter.pooler)
        parameters = unique_trainable_parameters(*modules)
        if not parameters:
            raise RuntimeError(f"No trainable inner parameters for {mode}")
        return parameters

    def _bilevel_sdpa_context(self):
        """Use one stable attention kernel for paired reward comparisons."""
        if bool(getattr(self.hparams, "bilevel_deterministic_sdpa", True)):
            return sdpa_kernel(SDPBackend.MATH)
        return nullcontext()

    def _bilevel_pooler_context(self, mode):
        """Avoid nondeterministic cuDNN BiGRU kernels in paired pooler rewards."""
        if mode == "decoder_pooler_lookahead":
            return torch.backends.cudnn.flags(enabled=False)
        return nullcontext()

    @contextmanager
    def _bilevel_forward_context(self, mode, gradients):
        """Apply the same numerical settings to each paired forward."""
        grad_context = torch.enable_grad() if gradients else torch.no_grad()
        with grad_context:
            with self.training_ctx:
                with self._bilevel_sdpa_context():
                    with self._bilevel_pooler_context(mode):
                        yield

    def _bilevel_lookahead_rewards(
        self,
        support_feats,
        support_pad,
        support_boundaries,
        support_batch,
        query_feats,
        query_pad,
        query_boundaries,
        query_batch,
        mode,
    ):
        """Query ``-NLL`` after one reversible support-set decoder update.

        This is a first-order, score-function bilevel approximation: gradients do
        not pass through the hard boundaries or the temporary inner update.  The
        actual model and optimizer state are restored before the outer GRPO/CE
        backward pass.
        """
        if getattr(self.hparams, "segmenter_reward", "nll") != "nll":
            raise ValueError(
                "The first bilevel pilot supports segmenter_reward=nll only. "
                "CER/WER lookahead would multiply an already expensive decode."
            )
        if len(support_boundaries) != len(query_boundaries):
            raise ValueError(
                "Support and query rollout groups must use the same K"
            )

        parameters = self._bilevel_inner_parameters(mode)
        inner_lr = float(getattr(self.hparams, "bilevel_inner_lr", 1.0e-3))
        max_norm = float(
            getattr(self.hparams, "bilevel_inner_max_grad_norm", 1.0)
        )
        measure_after = bool(
            getattr(self.hparams, "bilevel_measure_support_after", False)
        )
        current_query_rewards = []
        query_rewards = []
        support_before = []
        support_after = []
        grad_norms = []
        grad_scales = []
        update_norms = []
        parameter_tensors = []
        gradient_tensors = []
        nonzero_gradient_tensors = []
        gradient_max_abs = []

        for support_boundary, query_boundary in zip(
            support_boundaries, query_boundaries
        ):
            support_segments = None
            query_segments = None
            if mode == "decoder_lookahead":
                # The pooler is fixed in this arm.  Reuse its exact outputs so
                # paired rewards isolate decoder adaptation rather than cuDNN
                # BiGRU variation across repeated forwards.
                with torch.no_grad(), self.training_ctx:
                    support_segments = self.modules.segmenter.pool(
                        support_feats, support_pad, support_boundary
                    )
                    query_segments = self.modules.segmenter.pool(
                        query_feats, query_pad, query_boundary
                    )
            support_rng_state = capture_rng_state(self.device)
            with self._bilevel_forward_context(mode, gradients=True):
                support_dec = (
                    self._run_decoder_from_segments(
                        *support_segments, support_batch
                    )
                    if support_segments is not None
                    else self._run_decoder(
                        support_feats,
                        support_pad,
                        support_boundary,
                        support_batch,
                    )
                )
                support_loss = self._decoder_ce(
                    support_dec["llm_logits"], support_batch
                )
                gradients = torch.autograd.grad(
                    support_loss,
                    parameters,
                    allow_unused=True,
                )

            # Pair current and adapted query forwards lane-by-lane.  A wide K*B
            # current forward has different BF16 accumulation behavior and can
            # change rankings even when the temporary learning rate is zero.
            query_rng_state = capture_rng_state(self.device)
            with self._bilevel_forward_context(mode, gradients=False):
                current_query_dec = (
                    self._run_decoder_from_segments(
                        *query_segments, query_batch
                    )
                    if query_segments is not None
                    else self._run_decoder(
                        query_feats,
                        query_pad,
                        query_boundary,
                        query_batch,
                    )
                )
                current_query_rewards.append(
                    -self._utterance_nll(
                        current_query_dec["llm_logits"], query_batch
                    )
                )

            with temporary_sgd_update(
                parameters,
                gradients,
                learning_rate=inner_lr,
                max_grad_norm=max_norm,
            ) as update_stats:
                with replay_rng_state(query_rng_state, self.device):
                    with self._bilevel_forward_context(mode, gradients=False):
                        query_dec = (
                            self._run_decoder_from_segments(
                                *query_segments, query_batch
                            )
                            if query_segments is not None
                            else self._run_decoder(
                                query_feats,
                                query_pad,
                                query_boundary,
                                query_batch,
                            )
                        )
                        query_rewards.append(
                            -self._utterance_nll(
                                query_dec["llm_logits"], query_batch
                            )
                        )
                if measure_after:
                    # Match the grad-enabled support forward used to obtain the
                    # gradient.  SDPA may select a different BF16 kernel under
                    # no-grad and hide a small update in kernel-level variation.
                    with replay_rng_state(support_rng_state, self.device):
                        with self._bilevel_forward_context(
                            mode, gradients=True
                        ):
                            adapted_support = (
                                self._run_decoder_from_segments(
                                    *support_segments, support_batch
                                )
                                if support_segments is not None
                                else self._run_decoder(
                                    support_feats,
                                    support_pad,
                                    support_boundary,
                                    support_batch,
                                )
                            )
                            support_after.append(
                                self._decoder_ce(
                                    adapted_support["llm_logits"],
                                    support_batch,
                                ).detach()
                            )

            support_before.append(support_loss.detach())
            grad_norms.append(update_stats["grad_norm"])
            grad_scales.append(update_stats["grad_scale"])
            update_norms.append(update_stats["update_norm"])
            parameter_tensors.append(update_stats["parameter_tensors"])
            gradient_tensors.append(update_stats["gradient_tensors"])
            nonzero_gradient_tensors.append(
                update_stats["nonzero_gradient_tensors"]
            )
            gradient_max_abs.append(update_stats["gradient_max_abs"])

        diagnostics = {
            "inner_support_ce": float(torch.stack(support_before).mean()),
            "inner_grad_norm": sum(grad_norms) / len(grad_norms),
            "inner_grad_scale": sum(grad_scales) / len(grad_scales),
            "inner_update_norm": sum(update_norms) / len(update_norms),
            "inner_parameter_tensors": sum(parameter_tensors)
            / len(parameter_tensors),
            "inner_gradient_tensors": sum(gradient_tensors)
            / len(gradient_tensors),
            "inner_nonzero_gradient_tensors": sum(nonzero_gradient_tensors)
            / len(nonzero_gradient_tensors),
            "inner_gradient_max_abs": sum(gradient_max_abs)
            / len(gradient_max_abs),
        }
        if support_after:
            diagnostics["inner_support_ce_after"] = float(
                torch.stack(support_after).mean()
            )
        return (
            torch.stack(current_query_rewards, dim=1),
            torch.stack(query_rewards, dim=1),
            diagnostics,
        )

    def _write_bilevel_diagnostics(self, mode, diagnostics):
        """Append full-precision per-batch pilot diagnostics on the main process."""
        path = getattr(self.hparams, "bilevel_diagnostics_file", None)
        if not path or not if_main_process():
            return
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        record = {
            "mode": mode,
            "epoch": int(self.current_epoch),
            "minibatch_step": int(self.step),
            **{name: float(value) for name, value in diagnostics.items()},
        }
        with open(path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

    def _target_tokens(self, llm_logits, batch):
        """Text targets aligned to ``llm_logits`` (audio positions -> ignore_index)."""
        tokens_eos, _ = batch.tokens_eos
        n_audio = llm_logits.shape[1] - tokens_eos.shape[1]
        return torch.cat(
            [
                torch.full(
                    (tokens_eos.shape[0], n_audio),
                    self.hparams.ignore_index,
                    device=self.device,
                ),
                tokens_eos,
            ],
            dim=1,
        ).long()

    def _decoder_ce(self, llm_logits, batch):
        """Mean cross-entropy over text tokens (decoder-training loss / monitor)."""
        target = self._target_tokens(llm_logits, batch)
        return F.cross_entropy(
            llm_logits.reshape(-1, llm_logits.shape[-1]),
            target.reshape(-1),
            ignore_index=self.hparams.ignore_index,
        )

    def _utterance_nll(self, llm_logits, batch):
        """Per-utterance teacher-forced NLL ``(B,)`` (the RL reward is ``-NLL``)."""
        target = self._target_tokens(llm_logits, batch)
        ce = F.cross_entropy(
            llm_logits.reshape(-1, llm_logits.shape[-1]),
            target.reshape(-1),
            ignore_index=self.hparams.ignore_index,
            reduction="none",
        ).reshape(llm_logits.shape[0], -1)
        valid = (target != self.hparams.ignore_index).float()
        return ce.sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

    def _utterance_error(self, hyps, batch, kind="cer", refs=None):
        """Per-utterance free-running decode error (CER/WER) ``(N,)`` for the RL reward.

        ``hyps`` are the searcher's decoded token ids; the reference is ``refs`` (defaults
        to ``batch.wrd``; pass an explicit list for a K-replicated rollout batch).
        Character-level (cer) is denser -> lower-variance reward than word-level (wer).
        """
        preds = self.tokenizer.batch_decode(hyps[0], skip_special_tokens=True)
        if refs is None:
            refs = batch.wrd
        errs = []
        for pred, ref in zip(preds, refs):
            if kind == "wer":
                p, r = pred.split(), ref.split()
            else:  # cer
                p, r = list(pred.strip()), list(ref.strip())
            errs.append(_levenshtein(p, r) / max(len(r), 1))
        return torch.tensor(errs, device=self.device, dtype=torch.float32)

    def _entropy_coeff(self):
        """Linearly annealed entropy weight over the joint (post-warmup) epochs."""
        init = float(self.hparams.entropy_coeff_init)
        final = float(self.hparams.entropy_coeff_final)
        warm = int(self.hparams.warmup_epochs)
        total = int(self.hparams.number_of_epochs)
        span = max(total - warm, 1)
        frac = min(max((self.current_epoch - warm) / span, 0.0), 1.0)
        return init + frac * (final - init)

    def _is_warmup(self, stage):
        return (
            self.hparams.segmenter_mode == "joint"
            and stage == sb.Stage.TRAIN
            and self.current_epoch <= int(self.hparams.warmup_epochs)
        )

    def _rate_mode(self):
        """Resolve the current rate objective, including the legacy switch."""
        return getattr(self.hparams, "rate_mode", None) or (
            "pin"
            if bool(getattr(self.hparams, "rate_two_sided", False))
            else "captax"
        )

    def _sampled_rate_terms(self, sampled, pad_mask):
        """Rate penalties and ratios ``(B, K)`` for full-prefix AR rollouts."""
        K = len(sampled)
        B = pad_mask.size(0)
        boundary_cat = torch.cat(sampled, dim=0)
        pad_rep = pad_mask.repeat(K, 1)
        penalty, rho = sampled_rate_penalty(
            boundary_cat,
            pad_rep,
            float(self.hparams.rho_star),
            float(self.hparams.lambda_cap),
            float(self.hparams.lambda_press),
            rho_floor=float(getattr(self.hparams, "rho_floor", 0.0)),
            lambda_floor=float(getattr(self.hparams, "lambda_floor", 0.0)),
            mode=self._rate_mode(),
            rho_lo=float(getattr(self.hparams, "rho_lo", 0.0)),
            rho_hi=float(getattr(self.hparams, "rho_hi", 1.0)),
        )
        return (
            penalty.reshape(K, B).t().contiguous(),
            rho.reshape(K, B).t().contiguous(),
        )

    def _sample_full_ar_k(self, segmenter_feats, pad_mask, K):
        """Sample K histories in one cached ``K*B`` policy rollout."""
        rollout = self.modules.segmenter.sample_boundaries(
            segmenter_feats.repeat(K, 1, 1),
            pad_mask.repeat(K, 1),
            deterministic=False,
        )
        return list(rollout.boundaries.chunk(K, dim=0))

    def _full_ar_pg_entropy(self, segmenter_feats, pad_mask, sampled, adv):
        """Score all K completed histories in one parallel causal forward pass."""
        K = len(sampled)
        boundary_cat = torch.cat(sampled, dim=0)
        score = self.modules.segmenter.score_boundaries(
            segmenter_feats.repeat(K, 1, 1),
            pad_mask.repeat(K, 1),
            boundary_cat,
        )
        advantage_cat = adv.t().reshape(-1)
        pg = grpo_pg_loss(
            score.logits,
            score.boundaries,
            advantage_cat,
            valid_mask=score.valid_mask,
        )
        ent = entropy_bonus(score.logits, ~score.valid_mask)
        return pg, ent

    # ------------------------------------------------------------------ forward
    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        feats, segmenter_feats, feat_lens = self._encoder_features(batch)
        T = feats.size(1)
        pad_mask = lengths_to_padding_mask(feat_lens, T)
        full_ar = bool(getattr(self.modules.segmenter, "is_full_ar", False))

        if self.hparams.segmenter_mode == "coldstart":
            if full_ar:
                targets, _ = batch.boundary_target
                targets = targets.long().masked_fill(pad_mask, -1)
                logits = self.modules.segmenter.teacher_forced_logits(
                    segmenter_feats, pad_mask, targets
                )
            else:
                logits = self.modules.segmenter.boundary_logits(
                    segmenter_feats, pad_mask
                )
            return {
                "mode": "coldstart",
                "logits": logits,
                "pad_mask": pad_mask,
                "segmenter_feats": segmenter_feats,
                "full_ar": full_ar,
            }

        # joint: decoder path on the deterministic (argmax) segmentation.
        if full_ar:
            with torch.no_grad():
                greedy = self.modules.segmenter.sample_boundaries(
                    segmenter_feats, pad_mask, deterministic=True
                )
            argmax_b = greedy.boundaries
            logits = greedy.logits
        else:
            logits = self.modules.segmenter.boundary_logits(
                segmenter_feats, pad_mask
            )
            argmax_b = self.modules.segmenter.sample_boundary(
                logits.detach(), pad_mask, mode="argmax"
            )
        dec = self._run_decoder(
            feats,
            pad_mask,
            argmax_b,
            batch,
            want_hyps=(stage != sb.Stage.TRAIN),
        )
        out = {
            "mode": "joint",
            "logits": logits,
            "pad_mask": pad_mask,
            "argmax_boundary": argmax_b,
            "llm_logits": dec["llm_logits"],
            "seg_pad": dec["seg_pad"],
            "hyps": dec.get("hyps"),
            "is_warmup": self._is_warmup(stage),
            "segmenter_feats": segmenter_feats,
            "full_ar": full_ar,
        }

        # joint (post-warmup) train: K sampled segmentations -> detached rewards.
        #   segmenter_reward=nll: teacher-forced -NLL (cheap, but under-rewards audio).
        #   segmenter_reward=cer/wer: free-running decode error (requires the audio).
        if stage == sb.Stage.TRAIN and not out["is_warmup"]:
            reward_kind = getattr(self.hparams, "segmenter_reward", "nll")
            K = int(self.hparams.grpo_k)
            quality_rewards, sampled = [], []
            if full_ar:
                with torch.no_grad():
                    sampled = self._sample_full_ar_k(
                        segmenter_feats, pad_mask, K
                    )
            for k in range(K):
                if full_ar:
                    bk = sampled[k]
                else:
                    bk = self.modules.segmenter.sample_boundary(
                        logits.detach(), pad_mask, mode="sample"
                    )
                with torch.no_grad():
                    if reward_kind == "nll":
                        deck = self._run_decoder(feats, pad_mask, bk, batch)
                        r = -self._utterance_nll(deck["llm_logits"], batch)
                    else:  # cer / wer -- free-running decode
                        deck = self._run_decoder(
                            feats, pad_mask, bk, batch, want_hyps=True
                        )
                        r = -self._utterance_error(
                            deck["hyps"], batch, reward_kind
                        )
                quality_rewards.append(r)
                if not full_ar:
                    sampled.append(bk)
            quality_rewards = torch.stack(quality_rewards, dim=1)
            if full_ar:
                rate_penalties, sampled_rhos = self._sampled_rate_terms(
                    sampled, pad_mask
                )
                rewards = quality_rewards - rate_penalties
                out["rate_penalties"] = rate_penalties
                out["sampled_rhos"] = sampled_rhos
            else:
                rewards = quality_rewards
            out["quality_rewards"] = quality_rewards
            out["rewards"] = rewards
            out["sampled_boundaries"] = sampled
        return out

    # --------------------------------------------------------------- objectives
    def compute_objectives(self, predictions, batch, stage):
        if predictions["mode"] == "coldstart":
            return self._obj_coldstart(predictions, batch, stage)
        return self._obj_joint(predictions, batch, stage)

    def _obj_coldstart(self, predictions, batch, stage):
        logits, pad_mask = predictions["logits"], predictions["pad_mask"]
        targets, _ = batch.boundary_target
        targets = targets.long().masked_fill(pad_mask, -1)
        if predictions["full_ar"] and targets.size(1):
            targets = targets.clone()
            targets[:, 0] = -1
        loss = boundary_bce_loss(logits, targets)
        if stage != sb.Stage.TRAIN:
            if predictions["full_ar"]:
                pred_b = self.modules.segmenter.sample_boundaries(
                    predictions["segmenter_feats"],
                    pad_mask,
                    deterministic=True,
                ).boundaries
            else:
                pred_b = self.modules.segmenter.sample_boundary(
                    logits, pad_mask, mode="argmax"
                )
            p, r, f1 = boundary_prf(pred_b, targets)
            self.prf_stats["precision"].append(p)
            self.prf_stats["recall"].append(r)
            self.prf_stats["f1"].append(f1)
            self._track_rho_from_boundary(pred_b, pad_mask)
        return loss

    def _obj_joint(self, predictions, batch, stage):
        llm_logits = predictions["llm_logits"]
        dec_ce = self._decoder_ce(llm_logits, batch)

        if stage != sb.Stage.TRAIN:
            self._append_wer(predictions["hyps"], llm_logits, batch)
            self._track_rho_from_boundary(
                predictions["argmax_boundary"], predictions["pad_mask"]
            )
            return dec_ce

        # warmup: decoder only (segmenter gets no gradient).
        if predictions["is_warmup"]:
            self._track_rho_from_boundary(
                predictions["argmax_boundary"], predictions["pad_mask"]
            )
            self._log_train(dec_ce=dec_ce)
            return dec_ce

        # joint: decoder CE + GRPO pg + rate + annealed entropy.
        logits, pad_mask = predictions["logits"], predictions["pad_mask"]
        rewards = predictions["rewards"]  # (B, K)
        adv = group_advantage(
            rewards, normalize_std=bool(self.hparams.grpo_normalize_std)
        )
        if predictions["full_ar"]:
            pg, ent = self._full_ar_pg_entropy(
                predictions["segmenter_feats"],
                pad_mask,
                predictions["sampled_boundaries"],
                adv,
            )
            # The realized rate penalty is already part of each rollout reward,
            # so its gradient is carried by the policy-gradient term.
            rl_rate = pg * 0.0
            rho_bar = predictions["sampled_rhos"].mean(dim=1)
        else:
            pg = logits.sum() * 0.0
            for k, bk in enumerate(predictions["sampled_boundaries"]):
                policy_logits = self.modules.segmenter.conditioned_logits(
                    logits, bk
                )
                pg = pg + grpo_pg_loss(policy_logits, bk, adv[:, k])
            pg = pg / len(predictions["sampled_boundaries"])
            # rate_mode: pin | captax | band. Back-compat: if unset, derive from
            # the old rate_two_sided flag (True -> pin, False -> captax).
            rl_rate, rho_bar = rate_loss(
                logits,
                pad_mask,
                float(self.hparams.rho_star),
                float(self.hparams.lambda_cap),
                float(self.hparams.lambda_press),
                rho_floor=float(getattr(self.hparams, "rho_floor", 0.0)),
                lambda_floor=float(getattr(self.hparams, "lambda_floor", 0.0)),
                mode=self._rate_mode(),
                rho_lo=float(getattr(self.hparams, "rho_lo", 0.0)),
                rho_hi=float(getattr(self.hparams, "rho_hi", 1.0)),
                probabilities=self.modules.segmenter.boundary_marginals(
                    logits, pad_mask
                ),
            )
            ent = self.modules.segmenter.policy_entropy(logits, pad_mask)
        beta_h = self._entropy_coeff()  # 0 when the entropy bonus is dropped
        # Multi-task: add the decoder CE (co-train proj+LoRA) unless the decoder is
        # frozen for this ablation (single-task: RL vs a fixed warmup decoder).
        freeze_dec = bool(
            getattr(self.hparams, "freeze_decoder_in_joint", False)
        )
        total = float(self.hparams.pg_weight) * pg + rl_rate - beta_h * ent
        if not freeze_dec:
            total = total + dec_ce
        self._track_rho_from_boundary(
            predictions["argmax_boundary"], predictions["pad_mask"]
        )
        self._log_train(
            dec_ce=dec_ce,
            pg=pg,
            rate=rl_rate,
            entropy=ent,
            beta_h=beta_h,
            reward=rewards.mean(),
            quality_reward=predictions["quality_rewards"].mean(),
            sampled_rate_penalty=(
                predictions["rate_penalties"].mean()
                if predictions["full_ar"]
                else 0.0
            ),
            exp_rho=rho_bar.mean(),
        )
        return total

    # --------------------------------------------------------- batched fit_batch (RL)
    def fit_batch(self, batch):
        """Select the reference or batched on-policy GRPO update.

        ``on_policy`` is the original, validated implementation and uses SpeechBrain's
        accumulation-aware ``fit_batch``. ``batched_on_policy`` changes only reward
        evaluation: K segmentations are decoded in one wide batch, followed by one
        plain-GRPO backward pass that respects the same gradient accumulation schedule.
        ``combined_on_policy`` is the full-prefix Transformer-AR path: greedy and K
        sampled histories share one sequential policy loop and K rewards share one
        decoder forward. No rollout is reused across optimizer steps.
        """
        is_rl = (
            self.hparams.segmenter_mode == "joint"
            and self.current_epoch > int(self.hparams.warmup_epochs)
        )
        bilevel_mode = self._bilevel_mode()
        update_mode = getattr(self.hparams, "rl_update_mode", "auto")
        full_ar = bool(getattr(self.modules.segmenter, "is_full_ar", False))
        if update_mode == "auto":
            update_mode = "combined_on_policy" if full_ar else "on_policy"
        if is_rl and bilevel_mode != "off":
            if not full_ar or update_mode != "combined_on_policy":
                raise ValueError(
                    "The bilevel pilot requires the optimized full-prefix "
                    "Transformer-AR path (segmenter_backbone=transformer_ar, "
                    "rl_update_mode=combined_on_policy)."
                )
            return self._fit_batch_full_ar_bilevel_on_policy(
                batch, bilevel_mode
            )
        if not is_rl or update_mode == "on_policy":
            return super().fit_batch(batch)
        if update_mode == "combined_on_policy":
            if not full_ar:
                raise ValueError(
                    "combined_on_policy requires segmenter_backbone=transformer_ar"
                )
            return self._fit_batch_full_ar_combined_on_policy(batch)
        if update_mode != "batched_on_policy":
            raise ValueError(
                "rl_update_mode must be 'auto', 'on_policy', "
                "'batched_on_policy', or 'combined_on_policy', "
                f"got {update_mode!r}"
            )
        if full_ar:
            raise ValueError(
                "Use combined_on_policy, not batched_on_policy, for transformer_ar"
            )
        return self._fit_batch_batched_on_policy(batch)

    def _fit_batch_full_ar_bilevel_on_policy(self, batch, mode):
        """Support/query GRPO with an optional one-step inner adaptation.

        ``heldout`` scores query rollouts under the current decoder.  The two
        lookahead modes make one temporary support-set SGD step per rollout lane,
        score the corresponding query lane, then restore the model exactly.  The
        real decoder update is CE on the support half; the query half is reserved
        for the outer reward.
        """
        should_step = (self.step % self.grad_accumulation_factor) == 0
        self.on_fit_batch_start(batch, should_step=should_step)
        batch = batch.to(self.device)
        K = int(self.hparams.grpo_k)
        support_slice, query_slice = support_query_slices(
            batch.batchsize,
            float(getattr(self.hparams, "bilevel_support_fraction", 0.5)),
        )
        support_batch = decoder_batch_view(batch, support_slice)
        query_batch = decoder_batch_view(batch, query_slice)
        freeze_dec = bool(
            getattr(self.hparams, "freeze_decoder_in_joint", False)
        )

        schedule_logs = getattr(self, "_bilevel_schedule_logs", 0)
        if schedule_logs < self.grad_accumulation_factor:
            logger.info(
                "Bilevel Transformer-AR update: mode=%s minibatch_step=%d "
                "optimizer_step=%s K=%d support=%d query=%d inner_lr=%g",
                mode,
                self.step,
                should_step,
                K,
                support_slice.stop - support_slice.start,
                query_slice.stop - query_slice.start,
                float(getattr(self.hparams, "bilevel_inner_lr", 1.0e-3)),
            )
            self._bilevel_schedule_logs = schedule_logs + 1

        # Exit the no-grad autocast scope before the inner forward.  Otherwise
        # autocast may reuse decoder-weight casts created under ``no_grad`` and
        # silently disconnect projection/LoRA parameters from the inner graph.
        with torch.no_grad(), self.training_ctx:
            feats, segmenter_feats, feat_lens = self._encoder_features(batch)
            T = feats.size(1)
            pad_mask = lengths_to_padding_mask(feat_lens, T)
            group = self.modules.segmenter.sample_boundary_group(
                segmenter_feats, pad_mask, num_samples=K
            )
            argmax_b = group.greedy.boundaries
            sampled = list(group.sampled_boundaries)

            support_feats = feats[support_slice]
            support_segmenter_feats = segmenter_feats[support_slice]
            support_pad = pad_mask[support_slice]
            support_argmax = argmax_b[support_slice]
            support_sampled = [item[support_slice] for item in sampled]

            query_feats = feats[query_slice]
            query_segmenter_feats = segmenter_feats[query_slice]
            query_pad = pad_mask[query_slice]
            query_sampled = [item[query_slice] for item in sampled]

            current_quality = None
            if mode == "heldout":
                current_quality = self._rollout_rewards(
                    query_feats,
                    query_pad,
                    query_sampled,
                    query_batch,
                    "nll",
                )
            rate_penalties, sampled_rhos = self._sampled_rate_terms(
                query_sampled, query_pad
            )
            support_rate_penalties = None
            support_sampled_rhos = None
            if mode != "heldout":
                support_rate_penalties, support_sampled_rhos = (
                    self._sampled_rate_terms(support_sampled, support_pad)
                )

        inner_diagnostics = {}
        if mode == "heldout":
            quality_rewards = current_quality
        else:
            current_quality, quality_rewards, inner_diagnostics = (
                self._bilevel_lookahead_rewards(
                    support_feats,
                    support_pad,
                    support_sampled,
                    support_batch,
                    query_feats,
                    query_pad,
                    query_sampled,
                    query_batch,
                    mode,
                )
            )

        current_rewards = current_quality - rate_penalties
        rewards = quality_rewards - rate_penalties
        rank_diagnostics = reward_rank_diagnostics(current_rewards, rewards)
        self._write_bilevel_diagnostics(
            mode,
            {
                "support_examples": support_feats.size(0),
                "query_examples": query_feats.size(0),
                "current_reward": float(current_rewards.mean()),
                "adapted_reward": float(rewards.mean()),
                **rank_diagnostics,
                **inner_diagnostics,
            },
        )
        query_advantage = group_advantage(
            rewards,
            normalize_std=bool(self.hparams.grpo_normalize_std),
        )
        support_advantage = None
        if mode != "heldout":
            # Each support rollout affects mean query quality after the
            # temporary decoder step.  Its own boundary-frequency cost remains
            # per utterance; a query utterance's cost must not be assigned to a
            # support action.
            support_rewards = (
                quality_rewards.mean(dim=0, keepdim=True)
                - support_rate_penalties
            )
            support_advantage = group_advantage(
                support_rewards,
                normalize_std=bool(self.hparams.grpo_normalize_std),
            )

        with self.no_sync(not should_step):
            with self.training_ctx:
                pg_query, entropy_query = self._full_ar_pg_entropy(
                    query_segmenter_feats,
                    query_pad,
                    query_sampled,
                    query_advantage,
                )
                if support_advantage is None:
                    pg_support = pg_query * 0.0
                    entropy = entropy_query
                else:
                    pg_support, entropy_support = self._full_ar_pg_entropy(
                        support_segmenter_feats,
                        support_pad,
                        support_sampled,
                        support_advantage,
                    )
                    support_weight = float(
                        getattr(
                            self.hparams,
                            "bilevel_support_pg_weight",
                            1.0,
                        )
                    )
                    entropy = 0.5 * (entropy_query + entropy_support)
                pg = pg_query + (
                    0.0
                    if support_advantage is None
                    else support_weight * pg_support
                )
                beta_h = self._entropy_coeff()
                total = float(self.hparams.pg_weight) * pg - beta_h * entropy
                if not freeze_dec:
                    dec = self._run_decoder(
                        support_feats,
                        support_pad,
                        support_argmax,
                        support_batch,
                    )
                    dec_ce = self._decoder_ce(dec["llm_logits"], support_batch)
                    total = total + dec_ce
                else:
                    dec_ce = torch.zeros((), device=self.device)
            scaled = self.scaler.scale(total / self.grad_accumulation_factor)
            self.check_loss_isfinite(scaled)
            scaled.backward()

        if should_step:
            self.optimizers_step()

        self._track_rho_from_boundary(argmax_b, pad_mask)
        self._log_train(
            dec_ce=dec_ce,
            pg=pg,
            pg_query=pg_query,
            pg_support=pg_support,
            rate=0.0,
            entropy=entropy,
            beta_h=beta_h,
            reward=rewards.mean(),
            quality_reward=quality_rewards.mean(),
            sampled_rate_penalty=(
                rate_penalties.mean()
                if support_rate_penalties is None
                else 0.5
                * (rate_penalties.mean() + support_rate_penalties.mean())
            ),
            exp_rho=(
                sampled_rhos.mean()
                if support_sampled_rhos is None
                else 0.5 * (sampled_rhos.mean() + support_sampled_rhos.mean())
            ),
            bilevel_current_reward=current_rewards.mean(),
            bilevel_adapted_reward=rewards.mean(),
            bilevel_reward_delta=rank_diagnostics["reward_delta"],
            bilevel_reward_delta_abs=rank_diagnostics["reward_delta_abs"],
            bilevel_top_rollout_flip=rank_diagnostics["top_rollout_flip"],
            bilevel_rank_correlation=rank_diagnostics["rank_correlation"],
            bilevel_inner_support_ce=inner_diagnostics.get(
                "inner_support_ce", 0.0
            ),
            bilevel_inner_support_ce_after=inner_diagnostics.get(
                "inner_support_ce_after", 0.0
            ),
            bilevel_inner_grad_norm=inner_diagnostics.get(
                "inner_grad_norm", 0.0
            ),
            bilevel_inner_grad_scale=inner_diagnostics.get(
                "inner_grad_scale", 0.0
            ),
            bilevel_inner_update_norm=inner_diagnostics.get(
                "inner_update_norm", 0.0
            ),
            bilevel_inner_gradient_tensors=inner_diagnostics.get(
                "inner_gradient_tensors", 0.0
            ),
            bilevel_inner_nonzero_gradient_tensors=inner_diagnostics.get(
                "inner_nonzero_gradient_tensors", 0.0
            ),
            bilevel_inner_gradient_max_abs=inner_diagnostics.get(
                "inner_gradient_max_abs", 0.0
            ),
        )
        self.on_fit_batch_end(batch, {}, total, should_step=should_step)
        return total.detach().cpu()

    def _fit_batch_full_ar_combined_on_policy(self, batch):
        """One exact on-policy update with grouped policy and reward batches.

        The expensive sequential dimension is still the frame axis.  We widen
        its batch to contain one greedy lane and K sampled lanes per utterance,
        then evaluate all K detached rewards in one decoder call.  The policy
        loss remains a differentiable parallel re-score of those same realized
        histories, and SpeechBrain's optimizer/accumulation schedule is unchanged.
        """
        should_step = (self.step % self.grad_accumulation_factor) == 0
        self.on_fit_batch_start(batch, should_step=should_step)
        schedule_logs = getattr(self, "_combined_schedule_logs", 0)
        if schedule_logs < self.grad_accumulation_factor:
            logger.info(
                "Combined Transformer-AR on-policy update: minibatch_step=%d; "
                "optimizer_step=%s; K=%d",
                self.step,
                should_step,
                int(self.hparams.grpo_k),
            )
            self._combined_schedule_logs = schedule_logs + 1

        batch = batch.to(self.device)
        K = int(self.hparams.grpo_k)
        reward_kind = getattr(self.hparams, "segmenter_reward", "nll")
        freeze_dec = bool(
            getattr(self.hparams, "freeze_decoder_in_joint", False)
        )

        # Generate one on-policy group and evaluate each sampled trajectory once.
        with torch.no_grad(), self.training_ctx:
            feats, segmenter_feats, feat_lens = self._encoder_features(batch)
            T = feats.size(1)
            pad_mask = lengths_to_padding_mask(feat_lens, T)
            group = self.modules.segmenter.sample_boundary_group(
                segmenter_feats, pad_mask, num_samples=K
            )
            argmax_b = group.greedy.boundaries
            sampled = list(group.sampled_boundaries)
            quality_rewards = self._rollout_rewards(
                feats, pad_mask, sampled, batch, reward_kind
            )
            rate_penalties, sampled_rhos = self._sampled_rate_terms(
                sampled, pad_mask
            )
            rewards = quality_rewards - rate_penalties
            advantage = group_advantage(
                rewards,
                normalize_std=bool(self.hparams.grpo_normalize_std),
            )

        # Re-score the realized histories once with gradients, exactly as in the
        # reference full-AR objective. The decoder CE still uses the greedy path.
        with self.no_sync(not should_step):
            with self.training_ctx:
                pg, entropy = self._full_ar_pg_entropy(
                    segmenter_feats, pad_mask, sampled, advantage
                )
                beta_h = self._entropy_coeff()
                total = float(self.hparams.pg_weight) * pg - beta_h * entropy
                if not freeze_dec:
                    dec = self._run_decoder(feats, pad_mask, argmax_b, batch)
                    dec_ce = self._decoder_ce(dec["llm_logits"], batch)
                    total = total + dec_ce
                else:
                    dec_ce = torch.zeros((), device=self.device)
            scaled = self.scaler.scale(total / self.grad_accumulation_factor)
            self.check_loss_isfinite(scaled)
            scaled.backward()

        if should_step:
            self.optimizers_step()

        self._track_rho_from_boundary(argmax_b, pad_mask)
        self._log_train(
            dec_ce=dec_ce,
            pg=pg,
            rate=0.0,
            entropy=entropy,
            beta_h=beta_h,
            reward=rewards.mean(),
            quality_reward=quality_rewards.mean(),
            sampled_rate_penalty=rate_penalties.mean(),
            exp_rho=sampled_rhos.mean(),
        )
        self.on_fit_batch_end(batch, {}, total, should_step=should_step)
        return total.detach().cpu()

    def _fit_batch_batched_on_policy(self, batch):
        should_step = (self.step % self.grad_accumulation_factor) == 0
        self.on_fit_batch_start(batch, should_step=should_step)
        schedule_logs = getattr(self, "_rl_schedule_logs", 0)
        if schedule_logs < self.grad_accumulation_factor:
            logger.info(
                "Batched on-policy accumulation: minibatch_step=%d; optimizer_step=%s",
                self.step,
                should_step,
            )
            self._rl_schedule_logs = schedule_logs + 1
        batch = batch.to(self.device)
        K = int(self.hparams.grpo_k)
        reward_kind = getattr(self.hparams, "segmenter_reward", "cer")
        freeze_dec = bool(
            getattr(self.hparams, "freeze_decoder_in_joint", False)
        )
        pg_weight = float(self.hparams.pg_weight)
        rate_mode = getattr(self.hparams, "rate_mode", None) or (
            "pin"
            if bool(getattr(self.hparams, "rate_two_sided", False))
            else "captax"
        )

        # --- 1. rollout: generate ONCE (no grad) -- feats, old policy, K samples, decode.
        with torch.no_grad(), self.training_ctx:
            feats, segmenter_feats, feat_lens = self._encoder_features(batch)
            T = feats.size(1)
            pad_mask = lengths_to_padding_mask(feat_lens, T)
            logits_old = self.modules.segmenter.boundary_logits(
                segmenter_feats, pad_mask
            )
            sampled = [
                self.modules.segmenter.sample_boundary(
                    logits_old, pad_mask, mode="sample"
                )
                for _ in range(K)
            ]
            rewards = self._rollout_rewards(
                feats, pad_mask, sampled, batch, reward_kind
            )  # (B, K)
        adv = group_advantage(
            rewards, normalize_std=bool(self.hparams.grpo_normalize_std)
        )

        # --- 2. exactly one on-policy loss/backward; accumulate like SpeechBrain.
        with self.no_sync(not should_step):
            with self.training_ctx:
                logits = self.modules.segmenter.boundary_logits(
                    segmenter_feats, pad_mask
                )
                pg = logits.sum() * 0.0
                for k, bk in enumerate(sampled):
                    policy_logits = self.modules.segmenter.conditioned_logits(
                        logits, bk
                    )
                    pg = pg + grpo_pg_loss(policy_logits, bk, adv[:, k])
                pg = pg / K
                rl_rate, rho_bar = rate_loss(
                    logits,
                    pad_mask,
                    float(self.hparams.rho_star),
                    float(self.hparams.lambda_cap),
                    float(self.hparams.lambda_press),
                    rho_floor=float(getattr(self.hparams, "rho_floor", 0.0)),
                    lambda_floor=float(
                        getattr(self.hparams, "lambda_floor", 0.0)
                    ),
                    mode=rate_mode,
                    rho_lo=float(getattr(self.hparams, "rho_lo", 0.0)),
                    rho_hi=float(getattr(self.hparams, "rho_hi", 1.0)),
                    probabilities=self.modules.segmenter.boundary_marginals(
                        logits, pad_mask
                    ),
                )
                total = pg_weight * pg + rl_rate
                if not freeze_dec:
                    argmax_b = self.modules.segmenter.sample_boundary(
                        logits.detach(), pad_mask, mode="argmax"
                    )
                    dec = self._run_decoder(feats, pad_mask, argmax_b, batch)
                    dec_ce = self._decoder_ce(dec["llm_logits"], batch)
                    total = total + dec_ce
                else:
                    dec_ce = torch.zeros((), device=self.device)
            scaled = self.scaler.scale(total / self.grad_accumulation_factor)
            self.check_loss_isfinite(scaled)
            scaled.backward()

        if should_step:
            self.optimizers_step()

        # --- logging (once/batch): rho from the argmax path + loss components.
        with torch.no_grad(), self.training_ctx:
            argmax_b = self.modules.segmenter.sample_boundary(
                self.modules.segmenter.boundary_logits(
                    segmenter_feats, pad_mask
                ),
                pad_mask,
                mode="argmax",
            )
        self._track_rho_from_boundary(argmax_b, pad_mask)
        self._log_train(
            dec_ce=dec_ce,
            pg=pg,
            rate=rl_rate,
            entropy=0.0,
            beta_h=0.0,
            reward=rewards.mean(),
            exp_rho=rho_bar.mean(),
        )
        self.on_fit_batch_end(batch, {}, total, should_step=should_step)
        return total.detach().cpu()

    # ------------------------------------------------------------ metric helpers
    def _track_rho_from_boundary(self, boundary, pad_mask):
        valid = ~pad_mask
        n_seg = (boundary == 1).sum(dim=1).float() + 1.0  # + frame-0 segment
        n_fr = valid.sum(dim=1).float().clamp(min=1.0)
        self._rho_samples.extend((n_seg / n_fr).tolist())

    def _append_wer(self, hyps, llm_logits, batch):
        ids = batch.id
        target = self._target_tokens(llm_logits, batch).masked_fill(
            self._target_tokens(llm_logits, batch) == self.hparams.ignore_index,
            self.tokenizer.pad_token_id,
        )
        preds = self.tokenizer.batch_decode(hyps[0], skip_special_tokens=True)
        preds_words = [p.split(" ") for p in preds]
        targets = self.tokenizer.batch_decode(target, skip_special_tokens=True)
        targets_words = [t.split(" ") for t in targets]
        self.cer_metric.append(ids, preds_words, targets_words)
        self.wer_metric.append(ids, preds_words, targets_words)

    def _log_train(self, **kw):
        for k, v in kw.items():
            self._train_accum[k] = self._train_accum.get(k, 0.0) + float(v)
        self._train_accum["n"] = self._train_accum.get("n", 0.0) + 1.0

    # --------------------------------------------------------------- stage hooks
    def on_stage_start(self, stage, epoch):
        self._stage_started_at = time.monotonic()
        self.current_epoch = epoch if epoch is not None else 1
        if (
            stage == sb.Stage.TRAIN
            and self.hparams.segmenter_mode == "joint"
            and hasattr(self, "optimizer")
        ):
            warmup_lr = float(
                getattr(
                    self.hparams,
                    "lr_decoder_warmup",
                    self.hparams.lr_decoder,
                )
            )
            decoder_lr = (
                warmup_lr
                if self.current_epoch <= int(self.hparams.warmup_epochs)
                else float(self.hparams.lr_decoder)
            )
            self.optimizer.param_groups[1]["lr"] = decoder_lr
            logger.info(
                "Epoch %d decoder learning rate=%g (%s)",
                self.current_epoch,
                decoder_lr,
                "warmup" if self._is_warmup(stage) else "joint-RL",
            )
        self._rho_samples = []
        if stage == sb.Stage.TRAIN:
            self._train_accum = {}
        if stage != sb.Stage.TRAIN:
            if self.hparams.segmenter_mode == "coldstart":
                self.prf_stats = {"precision": [], "recall": [], "f1": []}
            else:
                self.cer_metric = self.hparams.cer_computer()
                self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        stage_seconds = time.monotonic() - getattr(
            self, "_stage_started_at", time.monotonic()
        )
        logger.info(
            "Stage timing: stage=%s epoch=%s seconds=%.3f",
            stage.name,
            epoch,
            stage_seconds,
        )
        timing_file = getattr(self.hparams, "stage_timing_file", None)
        if timing_file and if_main_process():
            os.makedirs(os.path.dirname(timing_file), exist_ok=True)
            with open(timing_file, "a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {
                            "stage": stage.name,
                            "epoch": epoch,
                            "seconds": stage_seconds,
                        }
                    )
                    + "\n"
                )
        stats = {"loss": stage_loss}
        if self._rho_samples:
            rho = torch.tensor(self._rho_samples)
            stats["rho_mean"] = float(rho.mean())
            stats["rho_std"] = float(rho.std()) if rho.numel() > 1 else 0.0
        transition_bias = getattr(
            self.modules.segmenter, "transition_bias", None
        )
        if transition_bias is not None:
            bias = transition_bias.detach().float().cpu()
            stats["ar_bias_prev0"] = float(bias[0])
            stats["ar_bias_prev1"] = float(bias[1])
            stats["ar_bias_gap"] = float(bias[1] - bias[0])

        if stage == sb.Stage.TRAIN:
            self.train_stats = stats
            acc = getattr(self, "_train_accum", {})
            n = acc.get("n", 0.0)
            if n > 0:
                for k, v in acc.items():
                    if k != "n":
                        self.train_stats[k] = v / n
            return

        if self.hparams.segmenter_mode == "coldstart":
            for k in ("precision", "recall", "f1"):
                vals = self.prf_stats[k]
                stats["boundary_" + k] = (
                    float(sum(vals) / len(vals)) if vals else 0.0
                )
            key_stat, key_kw = "boundary_f1", {"max_keys": ["boundary_f1"]}
        else:
            stats["CER"] = self.cer_metric.summarize("error_rate")
            stats["WER"] = self.wer_metric.summarize("error_rate")
            key_stat, key_kw = "WER", {"min_keys": ["WER"]}

        if stage == sb.Stage.VALID:
            old_lr = float(
                self.optimizer.param_groups[-1]["lr"]
                if hasattr(self, "optimizer")
                else self.hparams.initial_lr
            )
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=getattr(self, "train_stats", {}),
                valid_stats=stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={key_stat: stats[key_stat], "epoch": epoch}, **key_kw
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )
            if self.hparams.segmenter_mode == "joint" and if_main_process():
                with open(
                    self.hparams.test_wer_file, "w", encoding="utf-8"
                ) as w:
                    self.wer_metric.write_stats(w)

    # --------------------------------------------------------------- optimizers
    def on_fit_start(self):
        """Recover exactly, then make the launched learning rates authoritative.

        PyTorch optimizer recovery restores checkpoint param-group learning rates.
        Reapply the current experiment's values so task-specific tuning is not silently
        replaced by the shared warm-up checkpoint's rates.
        """
        super().on_fit_start()
        self.optimizer.param_groups[0]["lr"] = float(self.hparams.lr_segmenter)
        if self.hparams.segmenter_mode == "joint":
            self.optimizer.param_groups[1]["lr"] = float(
                self.hparams.lr_decoder
            )
        logger.info(
            "RL update mode=%s; grad_accumulation=%d; lr_segmenter=%g; lr_decoder=%g",
            getattr(self.hparams, "rl_update_mode", "on_policy"),
            self.grad_accumulation_factor,
            float(self.hparams.lr_segmenter),
            float(self.hparams.lr_decoder)
            if self.hparams.segmenter_mode == "joint"
            else 0.0,
        )
        bilevel_mode = self._bilevel_mode()
        if bilevel_mode != "off":
            if int(self.hparams.grpo_k) != 4:
                raise ValueError(
                    "The controlled bilevel pilot keeps grpo_k fixed at 4."
                )
            logger.info(
                "Bilevel pilot enabled: mode=%s support_fraction=%g "
                "inner_lr=%g inner_max_grad_norm=%g support_pg_weight=%g",
                bilevel_mode,
                float(getattr(self.hparams, "bilevel_support_fraction", 0.5)),
                float(getattr(self.hparams, "bilevel_inner_lr", 1.0e-3)),
                float(
                    getattr(
                        self.hparams,
                        "bilevel_inner_max_grad_norm",
                        1.0,
                    )
                ),
                float(
                    getattr(
                        self.hparams,
                        "bilevel_support_pg_weight",
                        1.0,
                    )
                ),
            )

    def init_optimizers(self):
        """Single AdamW with per-group LRs (segmenter vs decoder). Mirrors the base
        recipe's ``self.optimizer`` + ``optimizers_dict`` pattern for a safe fit_batch.

        Cold-start trains only the segmenter; joint also trains the decoder (proj +
        LoRA). The SSL encoder is always frozen.
        """
        seg_params = list(self.modules.segmenter.parameters())
        groups = [
            {"params": seg_params, "lr": float(self.hparams.lr_segmenter)}
        ]
        if self.hparams.segmenter_mode == "joint":
            dec_params = list(self.modules.proj.parameters()) + [
                p for p in self.modules.llm.parameters() if p.requires_grad
            ]
            groups.append(
                {"params": dec_params, "lr": float(self.hparams.lr_decoder)}
            )
        self.optimizer = torch.optim.AdamW(
            groups, weight_decay=float(self.hparams.weight_decay)
        )
        self.optimizers_dict = {"model_optimizer": self.optimizer}
        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("model_optimizer", self.optimizer)
            if self.hparams.segmenter_mode == "joint":
                # Save the trained decoder only in joint mode (avoids a 2.5 GB
                # untrained-LLM copy per cold-start checkpoint).
                self.checkpointer.add_recoverable("proj", self.modules.proj)
                self.checkpointer.add_recoverable("llm", self.modules.llm)


def _load_coldstart_segmenter(brain, ckpt_dir):
    """Load a cold-start segmenter state dict from the latest CKPT in ``ckpt_dir``."""
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "CKPT*")))
    if not ckpts:
        raise FileNotFoundError(f"No CKPT* under coldstart_ckpt_dir={ckpt_dir}")
    seg_file = os.path.join(ckpts[-1], "segmenter.ckpt")
    _load_segmenter_checkpoint(brain, seg_file)
    logger.info("Loaded cold-start segmenter from %s", seg_file)


def _load_segmenter_checkpoint(brain, seg_file):
    """Load an exact segmenter checkpoint with safe structured-policy upgrade."""
    if not os.path.isfile(seg_file):
        raise FileNotFoundError(f"Missing segmenter checkpoint: {seg_file}")
    state = torch.load(seg_file, map_location="cpu", weights_only=True)
    incompatible = brain.modules.segmenter.load_state_dict(state, strict=False)
    allowed_missing = set()
    if brain.modules.segmenter.sampling_strategy == "autoregressive":
        allowed_missing.add("transition_bias")
    if brain.modules.segmenter.pooling_name not in (
        "mean",
        "meanpool",
        "mean_pool",
    ):
        # Old boundary-policy checkpoints predate trainable pooling. Every new
        # order-aware pooler is a zero-residual extension, so these initialized
        # parameters reproduce mean pooling exactly and are safe to add.
        allowed_missing.update(
            key
            for key in incompatible.missing_keys
            if key.startswith("pooler.")
        )
    missing = set(incompatible.missing_keys) - allowed_missing
    if missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "Incompatible segmenter initialization: "
            f"missing={sorted(missing)}, "
            f"unexpected={sorted(incompatible.unexpected_keys)}"
        )
    logger.info(
        "Loaded exact segmenter initialization %s; sampling_strategy=%s; "
        "new_parameters=%s",
        seg_file,
        brain.modules.segmenter.sampling_strategy,
        sorted(allowed_missing & set(incompatible.missing_keys)),
    )


def _load_decoder_checkpoint(brain, ckpt_dir):
    """Initialize projection, LoRA, and normalization from an oracle decoder."""
    required = ("llm.ckpt", "proj.ckpt", "normalize.ckpt")
    missing = [
        name
        for name in required
        if not os.path.isfile(os.path.join(ckpt_dir, name))
    ]
    if missing:
        raise FileNotFoundError(
            f"Decoder checkpoint {ckpt_dir} is missing: {', '.join(missing)}"
        )
    brain.modules.llm.loader(os.path.join(ckpt_dir, "llm.ckpt"), True)
    brain.modules.proj.load_state_dict(
        torch.load(
            os.path.join(ckpt_dir, "proj.ckpt"),
            map_location="cpu",
            weights_only=True,
        )
    )
    brain.modules.normalize._load(
        os.path.join(ckpt_dir, "normalize.ckpt"), True
    )
    logger.info("Loaded oracle decoder initialization from %s", ckpt_dir)


def _attach_segmenter_ssl(hparams):
    """Construct the optional frozen SSL encoder used only by the segmenter.

    Keeping this construction outside HyperPyYAML makes the second encoder truly
    optional: ordinary single-encoder experiments do not instantiate or store a
    duplicate SSL model.
    """
    source = hparams.get("segmenter_ssl_hub")
    if not source:
        return
    if bool(hparams.get("use_feats", False)):
        raise ValueError("segmenter_ssl_hub requires use_feats=False")
    from speechbrain.integrations.huggingface.wav2vec2 import Wav2Vec2

    segmenter_ssl = Wav2Vec2(
        source=source,
        output_norm=bool(hparams.get("segmenter_ssl_output_norm", True)),
        freeze=bool(hparams.get("segmenter_ssl_frozen", True)),
        save_path=hparams["segmenter_ssl_folder"],
        device_map=hparams.get("segmenter_ssl_device", "cuda"),
    )
    hparams["modules"]["segmenter_ssl"] = segmenter_ssl
    logger.info(
        "Attached separate segmenter SSL encoder %s (expected feature dim=%s)",
        source,
        hparams.get("segmenter_input_dim"),
    )


if __name__ == "__main__":
    experimental_runtime = os.environ.get("SEGMENTER_EXPERIMENTAL_RUNTIME")
    if experimental_runtime:
        from transformer_ar_experimental_runtime import install

        install(SegmenterASR, experimental_runtime)
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    _attach_segmenter_ssl(hparams)

    sb.utils.distributed.ddp_init_group(run_opts)
    from librispeech_prepare import prepare_librispeech  # noqa

    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )
    run_on_main(
        prepare_librispeech,
        kwargs={
            "data_folder": hparams["data_folder"],
            "tr_splits": hparams["train_splits"],
            "dev_splits": hparams["dev_splits"],
            "te_splits": hparams["test_splits"],
            "save_folder": hparams["output_folder"],
            "merge_lst": hparams["train_splits"],
            "merge_name": "train.csv",
            "skip_prep": hparams["skip_prep"],
        },
    )

    tokenizer = hparams["llm"].tokenizer
    (
        train_data,
        valid_data,
        test_datasets,
        tokenizer,
        train_bsampler,
        valid_bsampler,
    ) = dataio_prepare(hparams, tokenizer)

    asr_brain = SegmenterASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    asr_brain.tokenizer = tokenizer
    asr_brain.txt_embedding = (
        asr_brain.raw_modules.llm.model.get_input_embeddings()
    )
    asr_brain.current_epoch = 1

    # Joint mode can initialize the segmenter and decoder independently. Exact
    # checkpoint paths take precedence over the legacy save-directory lookup.
    if hparams["segmenter_mode"] == "joint":
        if hparams.get("segmenter_init_checkpoint"):
            _load_segmenter_checkpoint(
                asr_brain, hparams["segmenter_init_checkpoint"]
            )
        elif hparams.get("coldstart_ckpt_dir"):
            _load_coldstart_segmenter(asr_brain, hparams["coldstart_ckpt_dir"])
        if hparams.get("decoder_init_ckpt_dir"):
            _load_decoder_checkpoint(
                asr_brain, hparams["decoder_init_ckpt_dir"]
            )
    for p in asr_brain.modules.ssl.parameters():
        p.requires_grad = False
    segmenter_ssl = getattr(asr_brain.modules, "segmenter_ssl", None)
    if segmenter_ssl is not None:
        for p in segmenter_ssl.parameters():
            p.requires_grad = False

    train_dl = hparams["train_dataloader_opts"]
    valid_dl = hparams["valid_dataloader_opts"]
    if train_bsampler is not None:
        cf = train_dl.get("collate_fn")
        train_dl = {
            "batch_sampler": train_bsampler,
            "num_workers": hparams["num_workers"],
        }
        if cf is not None:
            train_dl["collate_fn"] = cf
    if valid_bsampler is not None:
        cf = valid_dl.get("collate_fn")
        valid_dl = {"batch_sampler": valid_bsampler}
        if cf is not None:
            valid_dl["collate_fn"] = cf

    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=train_dl,
        valid_loader_kwargs=valid_dl,
    )

    # WER eval only meaningful in joint mode.
    if hparams["segmenter_mode"] == "joint":
        os.makedirs(hparams["output_wer_folder"], exist_ok=True)
        for k in test_datasets.keys():
            asr_brain.hparams.test_wer_file = os.path.join(
                hparams["output_wer_folder"], f"wer_{k}.txt"
            )
            asr_brain.evaluate(
                test_datasets[k],
                min_key="WER",
                test_loader_kwargs=hparams["test_dataloader_opts"],
            )
