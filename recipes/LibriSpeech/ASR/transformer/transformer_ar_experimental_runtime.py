"""Opt-in Transformer-AR speed experiments.

Nothing in this module is imported unless ``SEGMENTER_EXPERIMENTAL_RUNTIME`` is set.
The production launchers therefore keep their original code path and checkpoints.
All experiments use a fixed 64-frame boundary-history window and GRPO K=4.
"""

import logging

import torch

import segmenter as segmenter_module
from segment_pooling import lengths_to_padding_mask


WINDOW = 64
GRPO_K = 4
LOGGER = logging.getLogger(__name__)
_ORIGINAL_BLOCK_FORWARD = segmenter_module.CausalBoundaryBlock.forward
_FLEX_MASKS = {}
_COMPILED_FLEX = None


def _get_compiled_flex():
    global _COMPILED_FLEX
    if _COMPILED_FLEX is None:
        from torch.nn.attention.flex_attention import flex_attention

        _COMPILED_FLEX = torch.compile(flex_attention, dynamic=True)
    return _COMPILED_FLEX


def _flex_mask(length, device):
    from torch.nn.attention.flex_attention import create_block_mask

    key = (device.type, device.index, length)
    if key not in _FLEX_MASKS:
        def mask_mod(batch, head, query, key_index):
            del batch, head
            delta = query - key_index
            return (delta >= 0) & (delta < WINDOW)

        _FLEX_MASKS[key] = create_block_mask(
            mask_mod,
            B=None,
            H=None,
            Q_LEN=length,
            KV_LEN=length,
            device=str(device),
            BLOCK_SIZE=128,
        )
    return _FLEX_MASKS[key]


def _flex_block_forward(self, x, history_window=None):
    if history_window != WINDOW:
        return _ORIGINAL_BLOCK_FORWARD(
            self, x, history_window=history_window
        )
    if self.dropout:
        raise RuntimeError("FlexAttention benchmark requires policy dropout=0")
    residual = x
    q, k, v = self._split_qkv(self.norm1(x))
    attention = _get_compiled_flex()(
        q,
        k,
        v,
        block_mask=_flex_mask(x.size(1), x.device),
    )
    attention = attention.transpose(1, 2).reshape_as(x)
    x = residual + self.out(attention)
    return x + self.ffn(self.norm2(x))


@torch.no_grad()
def _grouped_greedy_and_samples(segmenter, features, padding_mask, samples):
    """Run greedy plus K stochastic policies in one (K+1)*B sequential loop."""
    policy = segmenter.net
    features = segmenter.film(features)
    batch_size, frames, _ = features.shape
    copies = samples + 1
    features = features.repeat(copies, 1, 1)
    padding_mask = padding_mask.repeat(copies, 1)
    total_batch = features.size(0)
    previous = torch.full(
        (total_batch,),
        policy.bos_index,
        dtype=torch.long,
        device=features.device,
    )
    if policy.cache_mode != "preallocated":
        raise RuntimeError("Grouped rollout benchmark requires preallocated KV")
    caches = []
    for layer in policy.layers:
        shape = (total_batch, layer.nhead, frames, layer.head_dim)
        caches.append(
            (
                torch.empty(shape, device=features.device, dtype=features.dtype),
                torch.empty(shape, device=features.device, dtype=features.dtype),
            )
        )
    boundaries = []
    greedy_logits = []
    for position in range(frames):
        logit, caches = policy._step(
            features[:, position], previous, position, caches
        )
        active = ~padding_mask[:, position]
        if position == 0:
            action = torch.zeros(
                total_batch, dtype=torch.long, device=features.device
            )
        else:
            action = torch.bernoulli(torch.sigmoid(logit.float())).long()
            action[:batch_size] = (logit[:batch_size] > 0).long()
        action = action.masked_fill(~active, -1)
        boundaries.append(action)
        greedy_logits.append(logit[:batch_size])
        previous = torch.where(active, action.clamp(min=0), previous)
    boundary = torch.stack(boundaries, dim=1)
    greedy = boundary[:batch_size]
    sampled = list(boundary[batch_size:].chunk(samples, dim=0))
    return greedy, torch.stack(greedy_logits, dim=1), sampled


def _make_batched_fit(group_rollouts):
    def _fit_batch_batched_on_policy(self, batch):
        should_step = (self.step % self.grad_accumulation_factor) == 0
        self.on_fit_batch_start(batch, should_step=should_step)
        batch = batch.to(self.device)
        samples = int(self.hparams.grpo_k)
        if samples != GRPO_K:
            raise RuntimeError(
                f"Experimental runtime fixes grpo_k={GRPO_K}, got {samples}"
            )
        reward_kind = getattr(self.hparams, "segmenter_reward", "nll")
        freeze_decoder = bool(
            getattr(self.hparams, "freeze_decoder_in_joint", False)
        )

        with torch.no_grad(), self.training_ctx:
            feats, segmenter_feats, feat_lens = self._encoder_features(batch)
            frames = feats.size(1)
            pad_mask = lengths_to_padding_mask(feat_lens, frames)
            if not bool(getattr(self.modules.segmenter, "is_full_ar", False)):
                raise RuntimeError("Experimental batching is Transformer-AR only")
            if group_rollouts:
                argmax_boundary, _, sampled = _grouped_greedy_and_samples(
                    self.modules.segmenter,
                    segmenter_feats,
                    pad_mask,
                    samples,
                )
            else:
                greedy = self.modules.segmenter.sample_boundaries(
                    segmenter_feats, pad_mask, deterministic=True
                )
                argmax_boundary = greedy.boundaries
                sampled = self._sample_full_ar_k(
                    segmenter_feats, pad_mask, samples
                )
            quality_rewards = self._rollout_rewards(
                feats, pad_mask, sampled, batch, reward_kind
            )
            rate_penalties, sampled_rhos = self._sampled_rate_terms(
                sampled, pad_mask
            )
            rewards = quality_rewards - rate_penalties
            advantage = segmenter_module.group_advantage(
                rewards,
                normalize_std=bool(self.hparams.grpo_normalize_std),
            )

        with self.no_sync(not should_step):
            with self.training_ctx:
                pg, entropy = self._full_ar_pg_entropy(
                    segmenter_feats,
                    pad_mask,
                    sampled,
                    advantage,
                )
                beta_h = self._entropy_coeff()
                total = float(self.hparams.pg_weight) * pg - beta_h * entropy
                if not freeze_decoder:
                    decoded = self._run_decoder(
                        feats, pad_mask, argmax_boundary, batch
                    )
                    decoder_ce = self._decoder_ce(
                        decoded["llm_logits"], batch
                    )
                    total = total + decoder_ce
                else:
                    decoder_ce = torch.zeros((), device=self.device)
            scaled = self.scaler.scale(
                total / self.grad_accumulation_factor
            )
            self.check_loss_isfinite(scaled)
            scaled.backward()
        if should_step:
            self.optimizers_step()

        self._track_rho_from_boundary(argmax_boundary, pad_mask)
        self._log_train(
            dec_ce=decoder_ce,
            pg=pg,
            rate=0.0,
            entropy=entropy,
            beta_h=beta_h,
            reward=rewards.mean(),
            quality_reward=quality_rewards.mean(),
            sampled_rate_penalty=rate_penalties.mean(),
            exp_rho=sampled_rhos.mean(),
        )
        self.on_fit_batch_end(
            batch, {}, total, should_step=should_step
        )
        return total.detach().cpu()

    return _fit_batch_batched_on_policy


def install(brain_class, runtime):
    """Install one benchmark-only runtime into ``brain_class``."""
    valid = {
        "flex_ref",
        "dense_reward_batch",
        "dense_group_batch",
        "flex_reward_batch",
        "flex_group_batch",
    }
    if runtime not in valid:
        raise ValueError(f"Unknown experimental runtime {runtime!r}")
    if runtime.startswith("flex"):
        segmenter_module.CausalBoundaryBlock.forward = _flex_block_forward
    if runtime.endswith("reward_batch"):
        brain_class._fit_batch_batched_on_policy = _make_batched_fit(False)
    elif runtime.endswith("group_batch"):
        brain_class._fit_batch_batched_on_policy = _make_batched_fit(True)
    LOGGER.warning(
        "Installed benchmark-only Transformer-AR runtime=%s, window=%d, K=%d",
        runtime,
        WINDOW,
        GRPO_K,
    )
