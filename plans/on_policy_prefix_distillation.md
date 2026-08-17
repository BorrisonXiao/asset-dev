# Plan: On-policy prefix distillation for lower-rate decoder inputs

**Date:** 2026-08-11  
**Status:** proposed; not yet implemented or launched  
**Scope:** WavLM features, CNN segmenter, first-order AR boundary policy, best
WavLM char-level decoder initialization, LibriSpeech-100h

## 1. Question

Can we reduce the decoder's sensitivity to the rate and structure of its audio
prefix by training it on prefixes sampled from the current segmenter, while a
frozen oracle-char decoder supplies a target distribution?

This targets a specific mismatch:

- the best char decoder was trained with oracle char-aligned prefixes at roughly
  `rho=0.293`;
- the highlighted WavLM CNN first-order AR system runs at roughly `rho=0.234`;
- the current joint path trains the decoder with CE on one deterministic
  segmentation, while GRPO evaluates several sampled segmentations under
  `no_grad`.

The hypothesis is that the decoder is not fully trained under the shorter and
more variable prefix distribution it sees from the learned segmenter.

## 2. What the proposal does and does not solve

On-policy prefix distillation **does**:

- expose the decoder to prefixes sampled from the current segmenter;
- teach the decoder to preserve the oracle-char decoder's text distribution when
  the audio prefix is shorter or segmented differently;
- provide a controlled test of prefix input-rate bias while holding the segmenter
  fixed;
- preserve the strong char-level decoder as an anchor during further compression.

It **does not** send a gradient through the current hard mean-pooling operation.
CE and distillation update the projection and LoRA decoder parameters only. The
segmenter still needs GRPO, or another discrete optimization method, during joint
training. A rollout-level distillation score can later be used as a GRPO reward,
but that is a separate mechanism from backpropagating the distillation loss.

## 3. Teacher and student forwards

Use the same utterance, prompt, and gold transcript history for both forwards.

### Frozen teacher

- boundary source: oracle char alignment;
- audio features: WavLM;
- decoder: frozen best WavLM char-level projection + LoRA checkpoint;
- output: detached teacher logits at text-token positions.

### Trainable student

- boundary source: the current WavLM CNN first-order AR segmenter;
- prefix choice: deterministic greedy boundaries or a sampled on-policy boundary
  sequence, depending on the experiment arm;
- audio features: WavLM pooled under those boundaries;
- decoder initialization: the same best char-level decoder checkpoint;
- trainable parameters: student projection + LoRA only in Phase I.

Teacher and student audio-prefix lengths need not match. Their distributions are
compared only at the aligned transcript-token positions under teacher forcing.

## 4. Decoder loss

For transcript-token positions \(\mathcal{M}\), distillation temperature \(T\),
oracle-character prefix \(z_{\mathrm{char}}\), and sampled learned prefix
\(z_{\mathrm{sample}}\):

\[
\begin{aligned}
\mathcal{L}_{\mathrm{decoder}}
  &= \mathcal{L}_{\mathrm{CE}} + \lambda_{\mathrm{KD}}\mathcal{L}_{\mathrm{KD}}, \\
\mathcal{L}_{\mathrm{CE}}
  &= -\sum_{t\in\mathcal{M}} \log p_S\!\left(y_t\mid z_{\mathrm{sample}},y_{<t}\right), \\
\mathcal{L}_{\mathrm{KD}}
  &= T^2\!\sum_{t\in\mathcal{M}} \operatorname{KL}\!\left(
       p_T^{(T)}\!\left(\cdot\mid z_{\mathrm{char}},y_{<t}\right)
       \,\middle\|\,
       p_S^{(T)}\!\left(\cdot\mid z_{\mathrm{sample}},y_{<t}\right)
     \right).
\end{aligned}
\]

The teacher distribution is the first argument of the forward KL. Both sums exclude
audio-prefix and padding positions.

Implementation details:

- mask every audio-prefix and padding position;
- extract the final `N_text` logits from each forward, so different audio-prefix
  lengths do not affect alignment;
- detach all teacher outputs;
- compute forward KL over the vocabulary in fp32 for numerical stability;
- begin with `T=2` and `lambda_KD in {0.25, 0.5}`;
- keep CE weight at 1.0;
- log CE and KD separately, along with teacher and student token entropies.

Do not use distillation alone as the main condition. CE anchors the decoder to the
transcript and prevents it from inheriting teacher errors without correction.
A KD-only arm can be a secondary diagnostic after the main comparison.

## 5. Phase I: frozen-segmenter 2x2 study

Freeze the same selected WavLM CNN first-order AR segmenter in every arm. Initialize
every student decoder from the same best char-level checkpoint. Match data order,
optimizer, number of updates, and checkpoint rule.

| ID | decoder training prefix | decoder loss | purpose |
|---|---|---|---|
| zero-shot | none | none | quantify the initial char-decoder mismatch on learned prefixes |
| det-ce | greedy segmenter boundaries | CE | current deterministic adaptation control |
| det-cekd | greedy segmenter boundaries | CE + KD | isolate the value of the oracle teacher |
| sample-ce | one sampled boundary sequence per utterance | CE | isolate on-policy prefix exposure |
| sample-cekd | one sampled boundary sequence per utterance | CE + KD | proposed method |

The four trained arms form the main 2x2 comparison. `zero-shot` is evaluated but
not trained.

### Sampling

- Sample one boundary sequence per utterance for the decoder update. More samples
  increase cost and are unnecessary for the first test.
- Draw from the exact closed-loop first-order AR policy, not independent logits.
- Force the boundary sequence to respect the same validity mask and frame-0
  convention as joint RL.
- Detach sampled boundaries before pooling. The segmenter is frozen in this phase.
- Log the sampled and greedy kept ratios separately.

### Rate evaluation

Evaluate every trained decoder with the **same fixed boundary sets** at approximately
`rho in {0.29, 0.23, 0.20}`. Generate these sets once on dev-clean by calibrating a
global decision bias/threshold, then reuse them across decoder arms. Also evaluate
the native greedy policy. This prevents decoder comparisons from being confounded by
different segmentations.

### Replication and selection

- Use seeds `3407`, `3408`, and `3409` for decoder training.
- Select checkpoints by dev-clean WER on the native learned-prefix condition.
- Treat dev-other as a robustness metric, not a checkpoint-selection target.
- Decode test-clean/test-other only after the loss and rate choices are fixed.

### Primary comparisons

1. `sample-ce` vs. `det-ce`: does exposure to sampled prefixes help?
2. `det-cekd` vs. `det-ce`: does the oracle teacher help without on-policy sampling?
3. `sample-cekd` vs. `sample-ce`: does distillation add value after on-policy CE?
4. Interaction: is KD most useful specifically under sampled, lower-rate prefixes?

Report mean and sample SD, plus paired per-utterance WER differences. The strongest
evidence for input-rate robustness is an improvement that grows as `rho` decreases
without sacrificing the `rho≈0.29` condition.

## 6. Phase II: integrate with joint RL

Only after Phase I identifies the decoder objective, unfreeze the segmenter and
resume joint training.

Recommended objective split:

\[
\begin{aligned}
\mathcal{L}_{\mathrm{decoder}}
  &= \mathcal{L}_{\mathrm{CE}} + \lambda_{\mathrm{KD}}\mathcal{L}_{\mathrm{KD}}, \\
R_{\mathrm{segmenter}}
  &= -\operatorname{NLL}_{S} - \lambda_{\rho}\mathcal{P}_{\rho}.
\end{aligned}
\]

Here \(\operatorname{NLL}_{S}\) is the detached student transcript loss and
\(\mathcal{P}_{\rho}\) penalizes deviation from the current kept-ratio target.

Keep the roles separate at first. The decoder learns from sampled prefixes; the
segmenter continues to receive the validated discrete policy-gradient update.

Use a gradual rate schedule rather than jumping immediately below the char rate:

```text
rho target: 0.29 -> 0.25 -> 0.23 -> 0.20
```

Advance to the next target only when dev-clean WER at the current target remains
stable. Keep a KL or auxiliary imitation constraint on the segmenter so compression
does not become uncontrolled boundary drift.

### Optional Phase-II reward ablation

After the decoder-only effect is established, test whether the detached per-rollout
teacher/student divergence is useful as a GRPO shaping reward:

\[
R_{mathrm{optional}}
=-\operatorname{NLL}_{S}
-\beta_{\mathrm{KD}}\mathcal{L}_{\mathrm{KD}}
-\lambda_{\rho}\mathcal{P}_{\rho}.
\]

This is optional and should not be mixed into the first joint run. It changes the
segmenter's optimization target and could favor prefixes that are easy to imitate
rather than prefixes that preserve transcript information.

## 7. Implementation plan

### 7.1 Configuration

Add explicit switches to `hparams/speechllm_segmenter.yaml`:

```yaml
decoder_prefix_mode: greedy       # greedy | sample
decoder_loss_mode: ce             # ce | ce_kd | kd
distill_teacher_ckpt_dir: null
distill_temperature: 2.0
distill_weight: 0.5
distill_topk: 0                    # 0 = full-vocabulary KL
```

Keep these independent of `segmenter_reward`; decoder supervision and segmenter
reward are different axes.

### 7.2 Teacher loading

Add a frozen teacher path that loads the oracle-char projection, normalization, and
LoRA state. Prefer one shared frozen Llama backbone with two named PEFT adapters if
the current wrapper supports reliable adapter switching. Otherwise use a second
decoder module and verify memory on one A100 before production.

Teacher requirements:

- permanently `eval()`;
- `requires_grad=False` for every parameter;
- forward under `torch.inference_mode()`;
- no teacher state in the optimizer or student checkpoint;
- checksum/log the teacher checkpoint at startup.

### 7.3 Forward path

Refactor decoder input construction so teacher and student can call the same helper:

1. pool features from explicit boundaries;
2. project pooled features;
3. concatenate prompt, audio prefix, and transcript history;
4. return logits and the slice corresponding to text targets.

For the teacher, read oracle boundaries from `batch.boundary_target`. For the
student, use greedy or sampled first-order AR boundaries according to
`decoder_prefix_mode`.

### 7.4 Loss and logging

Add `_decoder_kd(student_logits, teacher_logits, target_mask, T)`. Validate:

- identical logits give zero KL within tolerance;
- masked audio positions never contribute;
- unequal audio-prefix lengths still produce aligned text-logit shapes;
- gradients reach student projection/LoRA but not teacher or segmenter;
- temperature scaling uses `T^2`;
- fp16/bf16 inputs produce a finite fp32 KL.

Log:

- `train decoder CE`, `train decoder KD`, and total decoder loss;
- greedy and sampled `rho`;
- teacher/student entropy and top-1 agreement on gold text positions;
- WER by evaluation rate;
- peak GPU memory and step time.

### 7.5 Launch structure

Create a launcher that stages the same segmenter and char-decoder checkpoints into
each arm and asserts their checksums. Use separate output directories:

```text
results/speechllm_segmenter_wavlm/onpolicy_prefix_distill/
  det_ce/{3407,3408,3409}/
  det_cekd/{3407,3408,3409}/
  sample_ce/{3407,3408,3409}/
  sample_cekd/{3407,3408,3409}/
```

Run one-seed smoke tests first. Before a production launch, estimate the extra teacher
forward cost and memory and check scratch/home free space under the cluster rules.

## 8. Verification gates

### Correctness gate

- teacher and student text positions align for variable prefix lengths;
- the segmenter has exactly zero gradients in Phase I;
- teacher parameters remain bitwise unchanged after a training step;
- `sample` mode produces more prefix diversity than `greedy` mode;
- with sampling disabled and KD weight zero, the new path reproduces current CE.

### Scientific gate

Proceed to joint RL if `sample-cekd` or another Phase-I arm:

- improves native-rate dev-clean or dev-other WER over `det-ce` across the three
  seeds; or
- holds WER materially better as the evaluation rate moves from `rho≈0.29` toward
  `rho≈0.20`, without a meaningful regression at the char rate.

A null result is still informative: if sampled CE and CE+KD do not beat deterministic
CE at matched boundaries, the remaining gap is less likely to be decoder prefix-rate
bias and more likely to come from boundary information loss or segmenter placement.

## 9. Risks and mitigations

| risk | mitigation |
|---|---|
| teacher doubles memory or forward cost | smoke-test named adapters; use sequential teacher/student forwards; reduce batch size before changing the loss |
| teacher is overconfident | temperature 2; retain CE; monitor entropy and top-1 agreement |
| sampled prefixes are too poor early | initialize from the selected first-order AR segmenter; optionally mix 50% greedy and 50% sampled for the first epoch |
| apparent gain is just more decoder updates | match update counts and include `det-ce` |
| apparent gain is just KD, not on-policy exposure | retain the full 2x2 comparison |
| evaluation changes the boundaries between arms | precompute and reuse identical boundary sets at every target rate |
| KD improves teacher matching but not WER | select by WER, report KL only as a diagnostic |
| joint training destabilizes both modules | establish Phase I first; use smaller decoder LR and gradual rate schedule in Phase II |

## 10. Deliverables

- implementation with unit tests for alignment, masking, gradients, and checkpoint
  loading;
- one-seed memory/runtime smoke report;
- four-arm, three-seed Phase-I table and WER-versus-kept-ratio plot;
- only then, a three-seed joint-RL follow-up using the selected decoder objective;
- update the cumulative HTML report and project memory after results are audited.
