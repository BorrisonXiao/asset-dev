# Learned segmenter cumulative results review

**Date:** 2026-08-10  
**Depth:** standard

## 1. Executive summary

The experiments now support a coherent decomposition. **Evidence-backed:** moderate
WavLM pooling is beneficial on clean speech (fixed k=5: 6.16±0.12 versus no
downsampling: 7.05±0.52 test-clean), and boundary placement adds value beyond token count (phone
alignment at ρ=0.210: 5.49±0.23 versus fixed k=5 at ρ=0.201: 6.16±0.12). The learned
system's gap is concentrated in boundary fidelity and decoder initialization: with one
fixed segmenter that reads wav2vec2 features, initializing from the best char decoder
improves the system from 6.48±0.14/10.04±0.15 to 5.16±0.32/9.44±0.48 clean/other. A same-decoder
boundary swap is more direct: oracle, cold-predicted, and post-RL boundaries score
4.27, 9.14, and 17.37 dev-clean WER.

**Inferred:** the strongest current design is not “use one better encoder everywhere,”
but separate boundary timing from downstream representation: a segmenter using wav2vec2 for
wav2vec2-derived CTC events, WavLM features for pooling, an oracle/char-initialized
decoder, and constrained residual RL. **Open question:** first-order autoregressive
sampling is directionally helpful in one matched WavLM seed (4.86/9.39 versus
5.03/9.45), but it is not replicated on held-out WER.

## 2. Scope and evidence

The primary source was the generated report at
`artifacts/segmenter/report.html`. Headline values were recomputed from the aggregate
edit counts under:

- `results/speechllm_fixed_pooling_wavlm/{fixed_rate_*,no_downsampling_tc100,char_alignment_tc100,phone_alignment_tc100}`;
- `results/speechllm_segmenter_wavlm/multiseed_shared_warmup_onpolicy`;
- `results/speechllm_segmenter_wavlm/hybrid_w2v2_segmenter_controlled`;
- `results/speechllm_segmenter_wavlm/oracleclose_bestinit`;
- `results/speechllm_segmenter/multiseed_shared_warmup_onpolicy`;
- `results/speechllm_segmenter/w2v2_transformer_autoregressive_long/nll_mt`.

The same-decoder intervention comes from
`results/speechllm_boundary_swap_wavlm/oracle_decoder_seed3407/cnn_v2/wer_results`.
Critical paths were spot-checked in `train_speechllm_with_segmenter.py` (dual encoder,
frame-grid assertion, decoder initialization) and `segmenter.py` (first-order
conditional sampling and exact Markov marginals). The long autoregressive jobs were
still training/evaluating at the report snapshot, so their dev curves are provisional
and no missing held-out value was imputed.

## 3. Claims and verdicts

| Claim | Verdict | Primary evidence |
|---|---|---|
| Moderate downsampling can improve WER over no downsampling. | **Confirmed — Evidence-backed.** | Fixed k=5: 6.16 clean, 10.17 other. No downsampling: 7.05 clean, 10.10 other. Three-seed WavLM means. |
| Boundary placement matters at similar kept ratio. | **Confirmed — Evidence-backed.** | Phone alignment ρ=0.210 vs fixed k=5 ρ=0.201: −0.67 clean and −0.96 other WER. |
| The learned/oracle gap is only an encoder-quality issue. | **Refuted — Evidence-backed.** | The segmenter that reads wav2vec2 is fixed while the best char decoder start improves −1.32 clean/−0.61 other. |
| RL preserves the char initialization. | **Refuted — Evidence-backed.** | Same oracle decoder scores cold boundaries at 9.14 and post-RL boundaries at 17.37 dev-clean, versus 4.27 with oracle char boundaries. |
| WavLM always outperforms wav2vec2. | **Refuted — Evidence-backed.** | WavLM helps Transformer test-other but hurts the CNN NLL co-trained arm on both test splits. |
| First-order autoregression improves the boundary policy. | **Partially confirmed.** | One matched WavLM seed improves 0.17 clean/0.06 other; long wav2vec2 best-dev improves early, but held-out replication is incomplete. |
| More epochs cause the structured policy to converge better. | **Refuted for the current schedule.** | Long runs peak at epochs 1/2/4 (mean dev 5.43), then rebound to a latest mean of about 6.23 while the transition gap keeps growing negative. |

## 4. Derived findings

### 4.1 Moderate compression can improve WER and reduce cost

**Evidence-backed.** Fixed k=5 removes roughly 80% of the WavLM frames and improves
test-clean by 0.89 WER points versus no downsampling. Its test-other mean is tied within
seed variance (+0.07). The fixed-rate curve is U-shaped, so both insufficient and
excessive pooling are harmful.

### 4.2 Semantic placement has value at matched cost

**Evidence-backed.** Phone alignment is the closest available oracle to k=5 in corpus
rate (0.210 vs 0.201), but improves 0.67 clean and 0.96 other. Char alignment keeps
more frames (ρ=0.293) and reaches 5.03±0.42/8.07±0.43. Therefore the oracle advantage
cannot be reduced to “more tokens.”

### 4.3 Decoder initialization is the largest gain when the segmenter is fixed

**Evidence-backed.** With segmenter, pooled features, policy, rate band, and RL
budget held fixed, the best-char-decoder arm beats scratch in all three seed means.
The best-char-decoder run is only +0.13 behind the char oracle on test-clean but
+1.37 on test-other. **Inferred:** acoustically difficult speech magnifies segmenter
timing errors and the decoder's boundary-distribution mismatch.

### 4.4 Encoder choice interacts with the policy rather than imposing one ranking

**Evidence-backed.** WavLM Transformer NLL co-training improves test-other from
12.53±0.81 to 10.88±0.50, but WavLM CNN NLL co-training degrades from
6.19±0.58/11.12±0.44 to 6.93±0.47/12.01±0.42. The cold-start segmenter also aligns
better with the wav2vec2-derived targets when its input is wav2vec2 (CNN F1 0.893 vs
0.799 WavLM).

### 4.5 The first-order policy learns spacing, then overfits

**Evidence-backed for dev.** The three long runs improve the 5.85 initialization to
best dev WER 5.50/5.40/5.39, all by epoch 4. Their learned transition gap becomes
negative, lowering the probability of a boundary immediately after another boundary.
Later constant-LR training strengthens that prior while task WER degrades.

## 5. Representative experimental cases

### 5.1 Same-decoder boundary swap

- **Identifier:** `speechllm_boundary_swap_wavlm/oracle_decoder_seed3407/cnn_v2`
- **Expected:** replacing oracle labels with a close learned segmenter should produce a modest
  loss; RL should recover or improve it.
- **Actual:** 4.27 oracle → 9.14 cold → 17.37 post-RL dev-clean WER.
- **Interpretation:** directly observed boundary-stream sensitivity; the split between
  boundary error and out-of-distribution decoder response remains intertwined.

### 5.2 Same segmenter on wav2vec2, different decoder starts

- **Identifier:** `speechllm_segmenter_wavlm/hybrid_w2v2_segmenter_controlled`
- **Expected:** the same segmenter should isolate decoder-start effects.
- **Actual:** oracle init 5.16±0.32/9.44±0.48 vs scratch 6.48±0.14/10.04±0.15.
- **Interpretation:** the char decoder provides a substantially better starting basin;
  this is evidence-backed because the segmenter and RL stage are matched.

### 5.3 Long first-order autoregressive continuation

- **Identifier:** jobs 60853–60855 / `w2v2_transformer_autoregressive_long/nll_mt`
- **Expected:** a 30-epoch budget should reveal whether the structured policy needs
  longer to converge.
- **Actual:** all dev optima occur in epochs 1–4; later dev WER is worse and the
  no-adjacent-boundary bias grows.
- **Interpretation:** the policy learns quickly; the remaining issue is generalization
  and objective scheduling, not too few epochs.

## 6. Failure-mode synthesis

| Failure mode | Prevalence / example | Layer implicated |
|---|---|---|
| Cross-family timing mismatch | WavLM cold F1 0.799/0.777 vs wav2vec2 0.893/0.887 | segmenter input/targets |
| Unanchored semantic drift | 9.14 cold → 17.37 RL in boundary swap | RL objective/action space |
| Decoder starting-point mismatch | +1.32/+0.61 scratch penalty with the same segmenter | decoder training |
| Late structured-policy overfit | all three dev optima by epoch 4 | LR schedule/checkpoint rule |
| Encoder-policy interaction | opposite CNN and Transformer deltas | system integration, not one module |

The dominant pattern is a design problem, not a loading bug: the current action space
can move or delete every frame boundary, the rate objective constrains count rather
than identity, and the decoder co-adapts to the changing segmentation. The dual-grid
assertion and checkpoint checks passed, so there is no evidence for a frame-grid or
checkpoint-recovery implementation failure. The work is ready for a more constrained
RL policy, but not for a claim that first-order autoregression has already won.

## 7. Recommendations

1. **P0 — Preserve the char candidate set during RL.** Make keep/merge decisions only
   on detected char boundaries, initially disallowing relocation/insertion. Include an
   all-keep rollout and char-retention/KL term. **Success:** improve ≥0.3 dev-clean WER
   or reduce ρ meaningfully with ≤0.2 WER regression while retaining ≥0.90 ±1-frame F1.
2. **P0 — Keep the best char decoder and short bridge.** The fixed-segmenter test makes this
   the evidence-backed default initialization. **Success:** three independent seeds
   retain the oracle-init advantage on test-other, not only clean.
3. **P1 — Use early stopping plus LR decay for structured policies.** The long runs
   already peak early. **Success:** no >0.3 WER rebound after the selected checkpoint,
   with transition statistics stable rather than monotonically saturating.
4. **P1 — Replicate policy structure before claiming it.** Compare Bernoulli,
   first-order AR, and the planned full-prefix Transformer with identical segmenter,
   decoder, warmup, rate, and seeds. **Success:** across three seeds, lower WER at the
   same kept ratio, or the same WER at a lower kept ratio.

## 8. Bottom line

The project has established that lower cost and better WER are compatible, and that
boundary semantics contain additional value beyond mean-pooling regularization. The
best next move is to improve RL by preserving the strong char/decoder initialization
and learning conservative merges; more unconstrained epochs are unlikely to close the
oracle gap.
