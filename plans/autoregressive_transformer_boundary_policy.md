# Plan: Autoregressive Transformer Boundary Policy

> **Status:** implemented; three-seed NLL-MT chain running (2026-08-11)  
> **Date:** 2026-08-07  
> **Scope:** learned SpeechLLM segmenter on LibriSpeech-100h, initially with the
> frozen `microsoft/wavlm-large` encoder  
> **Control:** retain the existing independent-Bernoulli `transformer` backbone
> unchanged and add this policy as `transformer_ar`

> **Active precursor (2026-08-10):** jobs 60853–60855 combine the existing
> bidirectional Transformer acoustic backbone with the implemented first-order
> autoregressive boundary transition on frozen wav2vec2 features. They initialize
> from the best completed wav2vec2 Transformer NLL-MT checkpoint and run 30 joint-RL
> epochs. This is a useful long-horizon structured-policy result, but it is **not**
> the decoder-only, full-prefix `transformer_ar` architecture specified below.

> **Production run (2026-08-11):** cold-start job 64597 → shared decoder-warm-up
> job 64598 → NLL-MT seeds 3407/3408/3409 in jobs 64599/64600/64601. Outputs are
> under `results/speechllm_segmenter_wavlm/fullprefix_transformer_ar/`.

> **Best-decoder follow-up (2026-08-11):** jobs 64605/64606/64607 also depend on
> cold-start 64597, initialize the decoder from the best oracle-char checkpoint,
> and run 2 adaptation + 10 joint-RL epochs under the best learned model's tighter
> `[0.15, 0.25]` rate band.

> **Speed follow-up (2026-08-12):** measured epochs are about 4–5.5× slower than
> CNN first-order AR. The top-level TODO now prioritizes profiling, preallocated KV
> storage, combined policy-rollout batching, batched reward decoding, and a
> fixed 64-frame local-attention arm. A custom vLLM/ms-swift rollout
> service is deferred behind a result-and-throughput gate because continuous WavLM
> input at every generated boundary step is not supported as a drop-in text rollout.
> Keep `grpo_k=4` fixed throughout this work. Do not use a smaller rollout group as a
> speed optimization; compare implementation and local-attention changes at the same
> four-rollout setting.

> **Combined rollout implementation (2026-08-12):** the production trainer now has
> an opt-in `combined_on_policy` mode. It uses one `(K+1)·B` cached loop for the greedy
> and sampled histories and one `K·B` decoder reward call, while preserving the same
> parallel policy re-score and on-policy optimizer step. Production smoke 70130 passed;
> matched full one-epoch validation is running in jobs 70131/70132.

## 1. Goal

Replace the current conditionally independent boundary sampler

```text
pi(b_1:T | x) = product_t Bernoulli(b_t; sigmoid(logit_t(x)))
```

with a decoder-only Transformer policy whose decision at frame `t` is conditioned
on the sampled boundary prefix:

```text
pi(b_1:T | x) = product_t pi(b_t | x, b_<t)
```

The immediate scientific question is whether explicit label-history modeling
improves boundary placement and the WER-versus-kept-ratio frontier over the current
independent Transformer. This is an A/B experiment, not an assumption that the
autoregressive policy will win.

## 2. Non-goals and invariants

- Do not replace WavLM, the mean-pooling contract, the projection, the downstream
  Llama decoder, or the NLL/CER reward implementations.
- Do not cancel or overwrite the independent-Bernoulli experiments already running;
  they are the control condition.
- Do not make a streaming claim. WavLM-large is bidirectional, so this policy is
  causal in the **boundary-label history**, not necessarily causal in audio time.
- Do not reuse the existing Transformer cold-start checkpoint as an exact recovery.
  The architecture and policy distribution differ. Partial initialization may be an
  explicit ablation, but the production AR run gets a fresh cold-start and warmup.
- Preserve boundary values and pooling semantics: `1` starts a new segment, `0`
  continues the current segment, and `-1` is padding.

## 3. Proposed architecture

Implement a small decoder-only, Llama-style causal Transformer with native KV-cache
support. At frame `t`:

```text
z_t = audio_proj(wavlm_feature_t) + boundary_embedding(b_{t-1})
z_1 uses boundary_embedding(BOS)
hidden_t = causal_decoder(z_1:t)
logit_t = boundary_head(hidden_t)
b_t ~ Bernoulli(sigmoid(logit_t))
```

Initial configuration:

| Parameter | Initial value |
|---|---:|
| Backbone name | `transformer_ar` |
| WavLM input dimension | 1024 |
| Decoder hidden dimension | 256 |
| Decoder layers | 4 |
| Attention heads | 4 |
| FFN dimension | 1024 |
| Boundary-input vocabulary | `BOS`, `0`, `1` |
| Policy dropout | 0.0 initially |
| Maximum positions | at least 4096 frames |

Use a small Hugging Face causal model or an equivalently tested cached decoder,
rather than naively rerunning `torch.nn.TransformerDecoder` on the full prefix at
every sampled frame. Without KV caching, autoregressive rollout would repeatedly
recompute each prefix and be too slow for `K=4` GRPO.

### 3.1 Frame-zero correction

Frame 0 already opens segment 0 in `compute_segment_ids`; a sampled boundary at frame
0 is therefore a no-op for pooling but currently still affects the rate and policy
loss. In the AR policy:

- set `boundary[:, 0] = 0` deterministically;
- exclude frame 0 from cold-start BCE, policy-gradient log-probability, entropy, and
  kept-ratio calculations;
- still process frame 0 with the `BOS` embedding so its decoder state is available to
  frame 1.

## 4. Segmenter API changes

The current interface produces unconditional logits and then samples them. Keep that
path for `cnn` and `transformer`, but add an AR-specific policy interface:

```python
teacher_forced_logits(features, padding_mask, targets)
sample_boundaries(features, padding_mask, deterministic=False)
score_boundaries(features, padding_mask, boundaries)
```

Expected contracts:

- `teacher_forced_logits`: shift the gold labels right and compute all conditional
  logits in one parallel causal forward pass.
- `sample_boundaries`: sample sequentially with KV caching; return boundaries and the
  conditional log-probabilities generated by the exact rollout policy.
- `score_boundaries`: shift a completed sampled sequence right and re-score all its
  conditional actions in one parallel differentiable pass.
- Deterministic evaluation uses cached greedy decoding, not independent thresholding.
- A small result container should carry `boundaries`, `log_probs`, `logits`, and the
  valid-action mask so callers cannot accidentally use unconditional BCE math.

## 5. Cold-start changes

Train with teacher forcing against the existing char-CTC targets:

```text
input labels  = [BOS, target_0, ..., target_(T-2)]
output target = [target_0, ..., target_(T-1)]
```

Mask padding and frame 0. The training forward is parallel despite the causal
factorization.

Validation must be closed-loop:

1. start from `BOS`;
2. greedily select a boundary;
3. feed that prediction into the next step;
4. compute boundary precision/recall/F1 and kept ratio from the completed sequence.

Do not report teacher-forced argmax F1 as the primary cold-start metric; it has access
to the gold prefix and would overestimate deployment performance. If exposure bias is
large, add scheduled sampling only as a follow-up ablation. The subsequent on-policy
GRPO phase already trains on sampled histories.

## 6. Joint training and GRPO changes

### 6.1 Decoder/warmup path

- When the segmenter is frozen during shared decoder warmup, generate one greedy AR
  segmentation with the cache and pass it through the existing mean pool.
- Keep the shared-warmup handoff and checkpoint staging unchanged in principle, but
  use a new output namespace keyed by `transformer_ar`.

### 6.2 K-sample rollout

For each training batch:

1. Replicate WavLM features and masks to `K * B`.
2. Autoregressively sample all `K * B` boundary sequences together, one frame step at
   a time, using the decoder KV cache.
3. Reshape to the existing `(B, K, T)` logical grouping.
4. Mean-pool and evaluate NLL or CER rewards through the existing batched reward path.
5. Compute group-relative advantages exactly as in the current GRPO implementation.
6. Re-score the sampled boundaries with gradients in one parallel teacher-forced
   causal pass.

The no-gradient cached rollout provides the actions. The differentiable parallel
re-score provides their current conditional log-probabilities. For the current
single-update on-policy path, these represent the same parameters. If rollout reuse
or PPO clipping is re-enabled later, retain the exact detached rollout log-probs as
`old_logp`.

### 6.3 Policy-gradient loss

Replace the independent-logit-specific BCE loss with:

```text
L_pg = -mean_valid[A_utterance * log pi(b_t | x, b_<t)]
```

Every frame may initially retain the same utterance-level advantage (`gamma=1`), so
the factorization changes without simultaneously changing credit assignment. A
prefix-conditioned critic is a separate future experiment.

### 6.4 Dropout and on-policy consistency

Set policy dropout to zero for the first implementation. Sampling under one dropout
mask and re-scoring under another would make the nominally on-policy likelihood
inexact. Reintroduce dropout only with an explicit consistency strategy and test.

## 7. Rate and entropy objectives

The current differentiable expected count, `sum_t sigmoid(logit_t)`, is exact only
for history-independent Bernoulli decisions. A full Transformer history has
exponentially many prefixes, so exact marginal expected count is not tractable with
the current formula.

For the initial AR implementation:

1. Compute each sampled sequence's realized kept ratio, excluding frame 0.
2. Convert `pin`, `captax`, or `band` into a sampled rate penalty.
3. Include it in that rollout's reward before the group-relative advantage:

   ```text
   reward_total = reward_quality - sampled_rate_penalty
   ```

4. Estimate conditional entropy from the logits encountered along sampled histories.

Optionally evaluate an auxiliary differentiable penalty on conditional probabilities
along sampled prefixes, but keep it off by default until its bias and effect are
measured. Log quality reward, rate penalty, sampled ratio, and combined reward
separately so reward scaling remains auditable.

## 8. Configuration and checkpointing

Add hparams without changing existing defaults:

```yaml
segmenter_backbone: transformer_ar
segmenter_ar_hidden_dim: 256
segmenter_ar_num_layers: 4
segmenter_ar_nhead: 4
segmenter_ar_ffn_dim: 1024
segmenter_ar_dropout: 0.0
segmenter_ar_max_positions: 4096
```

Requirements:

- existing `cnn` and `transformer` configs/checkpoints remain loadable;
- `transformer_ar` checkpoints record architecture and frame-zero behavior;
- cold-start, warmup, and joint output directories are separate from the current
  independent-Transformer runs;
- the production AR sweep uses the same WavLM-large model, data, seeds, variants,
  warmup length, and evaluation protocol as the independent control.

## 9. Files expected to change

- `recipes/LibriSpeech/ASR/transformer/segmenter.py`
  - AR decoder module, cached sampling, teacher-forced scoring, result container, and
    conditional-log-probability PG loss.
- `recipes/LibriSpeech/ASR/transformer/train_speechllm_with_segmenter.py`
  - cold-start dispatch, greedy AR path, K-sample rollout, parallel re-score, sampled
    rate reward, and logging.
- `recipes/LibriSpeech/ASR/transformer/hparams/speechllm_segmenter.yaml`
  - `transformer_ar` options and rate-policy switch.
- `recipes/LibriSpeech/ASR/transformer/run_segmenter.slurm`
  - only if validation must admit the new backbone name explicitly.
- a new AR submission script
  - fresh cold-start -> shared six-epoch warmup -> three variants x three seeds.
- focused tests, either beside `segmenter.py` self-tests or under a recipe test file.

## 10. Required tests

### Correctness

- **Causality:** changing future boundary labels does not change earlier logits.
- **History dependence:** changing an earlier boundary can change a later logit.
- **Cache equivalence:** cached stepwise logits match full teacher-forced logits in
  evaluation mode within numerical tolerance.
- **Score equivalence:** rollout log-probs match `score_boundaries` for the sampled
  sequence when parameters and dropout state are unchanged.
- **Padding:** padded frames never contribute to actions, cache-visible attention,
  rate, entropy, or loss.
- **Frame 0:** always zero and excluded from rate/loss.
- **Gradient direction:** positive advantage increases the scored probability of a
  sampled sequence.
- **Checkpoint round trip:** save/recover reproduces greedy boundaries exactly.

### Performance and integration

- Cached sampling is materially faster than full-prefix recomputation.
- One real LibriSpeech batch completes cold-start forward/backward on an A100.
- One `K=4` NLL rollout and one `K=4` CER rollout complete without OOM.
- Existing `cnn` and independent `transformer` self-tests still pass unchanged.

## 11. Execution order and gates

1. **Policy module and tests**
   - implement teacher-forced scoring and cached greedy/stochastic rollout;
   - pass causality, cache-equivalence, padding, and gradient tests.
2. **Cold-start integration**
   - overfit a small batch;
   - run a one-epoch LibriSpeech-100h smoke;
   - verify closed-loop F1, ratio, speed, and VRAM.
3. **Joint GRPO integration**
   - implement K-way cached rollout and parallel re-score;
   - add sampled rate reward and conditional entropy logging;
   - run short NLL and CER smoke jobs.
4. **One-seed production pilot**
   - fresh 10-epoch AR cold-start;
   - shared six-epoch decoder warmup;
   - run `nll_mt`, seed 3407 first.
5. **Go/no-go review**
   - compare cold-start closed-loop F1, kept ratio, wall time, VRAM, dev WER, and
     stability against the independent Transformer;
   - inspect whether later boundary probabilities actually vary with sampled history.
6. **Full matched sweep if healthy**
   - `nll_frozen`, `nll_mt`, `cer_mt` x seeds 3407/3408/3409;
   - use the same tests and result aggregation as the current WavLM sweep.

## 12. Acceptance criteria

The implementation is ready for the full sweep when:

- all correctness tests above pass;
- cached rollout, not prefix recomputation, is used in production;
- cold-start closed-loop predictions do not collapse to all-zero or all-one;
- sampled kept ratios remain controllable under the configured rate objective;
- NLL and CER smoke runs complete without NaNs, OOM, or checkpoint-recovery errors;
- logs distinguish quality reward, rate reward, conditional entropy, and combined
  advantage inputs;
- the independent Transformer remains reproducible as the control.

Scientific success is judged only after the matched A/B: better WER at a comparable
kept ratio, fewer tokens at comparable WER, or a clear stability/placement advantage.
An autoregressive loss to the independent control is still informative and must be
reported rather than hidden.

## 13. Main risks and mitigations

| Risk | Mitigation |
|---|---|
| Sequential sampling is too slow | KV caching from the first implementation; batch `K * B` rollouts |
| Exposure bias after teacher-forced cold-start | closed-loop validation; on-policy GRPO; scheduled sampling only if measured necessary |
| Rate objective changes behavior | sampled rate penalty in reward; separate logging; tune on one-seed pilot |
| On-policy mismatch from stochastic layers | policy dropout 0.0 initially |
| Adjacent/pathological boundaries remain possible | first measure learned dependence; add duration/min-gap constraints only as a separate ablation |
| Existing checkpoints are incompatible | new `transformer_ar` namespace and fresh stage chain |
| Apparent causal claim is overstated | explicitly call it label-autoregressive and offline while WavLM is bidirectional |
