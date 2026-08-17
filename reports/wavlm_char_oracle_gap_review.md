# WavLM char-oracle gap review

**Date:** 2026-08-09  
**Depth:** standard

## 1. Executive summary

The learned policy is not equivalent to the char oracle at either end of the
curriculum. **Evidence-backed:** the char-aligned WavLM baseline reaches
**5.04±0.42 test-clean / 8.07±0.43 test-other WER**, whereas NLL co-trained RL
reaches **6.93±0.47 / 12.01±0.42** with the CNN policy and **6.99±0.70 /
10.88±0.50** with the Transformer policy. The cold-start models only reproduce
the char labels at **0.799 CNN / 0.777 Transformer exact-frame F1** on the full
dev-clean log. More importantly, the joint objective contains no char-BCE or
reference-policy term. In a duration-stratified 16-utterance audit, exact char
F1 falls after NLL RL from **0.799 to 0.551** for CNN and from **0.773 to 0.465**
for Transformer.

**Inferred:** imperfect cold-start imitation and subsequent loss of char
alignment are probably major contributors to the WER gap. **Open question:**
they do not yet establish how much of the gap is causal because the oracle and
learned decoders were trained with different schedules, batch protocols, and
checkpoint staging. A same-decoder boundary-swap evaluation is the shortest
experiment that will separate boundary quality from decoder curriculum.

## 2. Scope and evidence

The primary source was `training_report.html`, specifically its WavLM results,
RL-convergence, pipeline, and cold-start descriptions. The review checked:

- the completed three-seed char-oracle and NLL co-trained WER files under
  `recipes/LibriSpeech/ASR/transformer/results/`;
- full dev-clean cold-start metrics and decoder-warmup logs under
  `results/speechllm_segmenter_wavlm/{coldstart,shared_warmup}`;
- the selected CNN and Transformer NLL co-trained checkpoints and their joint
  logs under `multiseed_shared_warmup_onpolicy/nll_mt`;
- the active objectives in `train_speechllm_with_segmenter.py`, rate settings in
  `hparams/speechllm_segmenter.yaml`, and checkpoint staging in
  `run_segmenter_from_shared_warmup.slurm`;
- a new read-only boundary audit saved as
  `reports/wavlm_char_boundary_audit.json`.

The per-example audit covers 16 dev-clean utterances chosen evenly after sorting
the split by duration. It is diagnostic, not a replacement for full-corpus
post-RL boundary evaluation. The full-corpus cold-start F1 and all WER values
come from completed primary logs/edit-count files.

## 3. Claims and verdicts

| Claim | Verdict | Evidence |
|---|---|---|
| The strongest learned WavLM Transformer arm is NLL co-trained at 6.99±0.70 clean / 10.88±0.50 other. | **Confirmed — Evidence-backed.** | Independently recomputed from the three `wer_test-{clean,other}.txt` files under `.../nll_mt/transformer/{3407,3408,3409}`. |
| RL starts from a char-trained segmenter. | **Confirmed — Evidence-backed.** | The cold objective is char BCE (`train_speechllm_with_segmenter.py:367-379`); job log `slurm_logs/wl_c_w6_3407_51514.out` records the exact cold checkpoint loaded. The CNN cold and both warmup `segmenter.ckpt` files have identical SHA-256. |
| Starting from char supervision should give behavior close to the char oracle. | **Refuted as an implication — Evidence-backed.** | Cold exact-frame F1 is only 0.799/0.777, and the decoder trained on frozen cold predictions is already worse than the oracle before RL. Initialization does not imply oracle boundaries. |
| The learned policy remains char-like during RL. | **Refuted — Evidence-backed for the sampled checkpoints.** | `reports/wavlm_char_boundary_audit.json`: exact F1 drops to 0.551 CNN and 0.465 Transformer after NLL RL. The joint loss has no char-BCE/KL term (`train_speechllm_with_segmenter.py:381-440`). |
| Policy optimization has settled, but reward/generalization alignment has not. | **Partially confirmed — Evidence-backed.** | Rates remain in a narrow band, but NLL-multitask dev WER is non-monotonic and all six selected checkpoints occur before epoch 10. Char alignment was not logged after cold-start, so “settled” does not mean the boundary semantics stabilized. |
| The HTML contains the current baseline state. | **Refuted — Evidence-backed.** | The HTML still says fixed-rate seeds 3408/3409 are running and does not include the now-complete char-oracle results. Its timestamp is 2026-08-09 00:50 EDT. |

## 4. Derived findings

### 4.1 The WER gap is large and consistent

**Evidence-backed.** Values below were recomputed from corpus edit-count files;
± is sample SD over seeds 3407/3408/3409.

| system | test-clean | Δ vs char | test-other | Δ vs char |
|---|---:|---:|---:|---:|
| Char-aligned oracle | **5.04±0.42** | — | **8.07±0.43** | — |
| CNN NLL co-trained | 6.93±0.47 | +1.89 | 12.01±0.42 | +3.95 |
| Transformer NLL co-trained | 6.99±0.70 | +1.95 | 10.88±0.50 | +2.82 |

This is not a small seed fluctuation: the mean gaps are 4.5–9.1 times the
oracle run-to-run SD, depending on split and learned arm.

### 4.2 Cold-start learns a good timing detector, but not an oracle

**Evidence-backed.** On full dev-clean, the final cold-start logs report:

| policy | precision | recall | exact F1 | logged ρ |
|---|---:|---:|---:|---:|
| CNN | 0.785 | 0.814 | 0.799 | 0.310 |
| Transformer | 0.765 | 0.791 | 0.777 | 0.308 |

The char targets themselves have dev-clean macro mean ρ=0.295 (2,703
utterances). The 16-utterance audit reproduces cold exact F1 (0.799 CNN, 0.773
Transformer) and shows that many misses are small timing offsets: CNN F1 rises
to 0.943 at ±1 frame and 0.948 at ±2; Transformer rises to 0.921 and 0.934.

**Inferred:** a one-frame boundary displacement can perturb both adjacent mean
pooled embeddings, so an exact-F1 near 0.8 can still have a larger ASR effect
than its framewise error rate suggests.

### 4.3 RL erases much of the char initialization

**Evidence-backed for the sampled seed-3407 checkpoints.** The same 16
utterances and encoder features give:

| policy checkpoint | exact F1 | ±1-frame F1 | ±2-frame F1 |
|---|---:|---:|---:|
| cold CNN | 0.799 | 0.943 | 0.948 |
| NLL-RL CNN, selected epoch 8 | **0.551** | 0.786 | 0.840 |
| cold Transformer | 0.773 | 0.921 | 0.934 |
| NLL-RL Transformer, selected epoch 9 | **0.465** | 0.733 | 0.807 |

This behavior follows directly from the design: after warmup, the policy is
optimized by GRPO reward plus a permissive rate band [0.10, 0.30], with no
supervised char loss, no KL to the cold policy, and no char-F1 checkpoint
criterion. The selected NLL runs also lower the reported rate from about 0.31
to roughly 0.25–0.27. Therefore “starts at char level” describes initialization
only, not a persistent constraint.

### 4.4 A substantial gap exists before RL, but attribution is confounded

**Evidence-backed.** With the cold CNN frozen, decoder warmup reaches best
dev-clean WER 6.41 at epoch 5 and tests at 7.39/11.27. The corresponding
Transformer numbers are 6.97 dev and 8.19/11.51 test. The char-oracle seed-3407
decoder reaches dev WER 5.01 at epoch 5 and its selected checkpoint tests at
5.03/8.27. This shows the shortfall begins before RL.

**Open question:** it is not a clean segmenter-only comparison. The char oracle
uses ten epochs, max train batch length 200, gradient accumulation 5, and a
decaying LR (5e-4 to 1e-5). Learned warmup uses six epochs, max batch length 100,
accumulation 4, and constant 5e-4. Decode budgets also differ. In addition,
`run_segmenter_from_shared_warmup.slurm:21-55` deliberately stages the exact
final epoch-6 checkpoint rather than best-dev: for CNN, epoch 6 has WER 8.24
versus 6.41 at epoch 5. Finally, all three RL “seeds” share the same seed-3407
cold segmenter and decoder warmup, whereas the char-oracle seeds are independent
end to end. The learned SD therefore understates full-pipeline initialization
variance.

### 4.5 No checkpoint-loading failure was found

**Evidence-backed.** `wl_c_w6_3407_51514.out` records loading the correct CNN
cold checkpoint. The cold checkpoint and both saved warmup segmenters share SHA
`56f6ae465575e33d4a716f084dc147705b8cb62111787b985ab0f7afa3c0a785`.
`wl_c_nm_3407_51516.out` records staging the final warmup and then loading the
cold segmenter; normal checkpoint recovery restores the staged state. The
segmenter is unchanged during warmup as intended.

### 4.6 Minor rate-reporting defect

**Evidence-backed.** `_track_rho_from_boundary` counts all predicted ones and
then adds one for frame 0 (`train_speechllm_with_segmenter.py:557-561`), while
the pooler already treats frame 0 as a segment start and char targets commonly
label it as one. Logged ρ is therefore overcounted by roughly 1/T per utterance.
This slightly biases report rates upward but cannot explain the WER gap.

### 4.7 Clarification: wav2vec2 versus WavLM cold-start convergence

**Evidence-backed.** The earlier wav2vec2 result was indeed approximately 90
F1. Its CNN cold-start peaks at 0.895 exact-frame F1 at epoch 4 and ends at
0.893 at epoch 10; the Transformer ends at 0.887. In contrast, WavLM ends at
0.799 CNN and 0.777 Transformer.

| encoder / boundary policy | epoch 1 F1 | best/final F1 | convergence reading |
|---|---:|---:|---|
| wav2vec2 / CNN | 0.887 | 0.895 / 0.893 | plateaued by epoch 4 |
| wav2vec2 / Transformer | 0.853 | 0.887 / 0.887 | still improves, but slowly late |
| WavLM / CNN | 0.751 | 0.799 / 0.799 | nearly plateaued; +0.006 over epochs 6–10 |
| WavLM / Transformer | 0.661 | 0.777 / 0.777 | not fully plateaued; +0.019 over epochs 6–10 |

The target provenance matters. `ctc_boundary_align.py` generates the char
labels with `torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H`. The earlier
segmenter consumes `facebook/wav2vec2-base-960h` features, whereas the new one
consumes `microsoft/wavlm-large` features. Both have the same 50-Hz frame
count, but their top-layer frame representations need not place CTC token
spikes identically. Thus the wav2vec2 result is partly an in-family imitation
task, and exact-frame F1 is a harsher measure for WavLM. The WavLM CNN's
0.943 F1 at ±1 frame supports this explanation.

**Inferred:** longer WavLM cold-start training may add a few exact-F1 points,
especially for the Transformer, but the current slopes do not support expecting
an automatic jump from 0.80 to 0.89. A tolerance-aware target or alignments
derived from a WavLM CTC model is a more direct test.

### 4.8 Oracle-decoder/learned-segmenter swap

**Evidence-backed as an experimental design; outcome is an open question.**
Using an oracle-trained decoder with the learned segmenter is a valuable
intervention, but that single cross-pair is distribution-shifted: the decoder
was trained only on exact oracle segments. The interpretable experiment is a
2×3 matrix with no additional training:

| decoder checkpoint | oracle boundaries | cold learned boundaries | post-RL boundaries |
|---|---|---|---|
| oracle-char-trained decoder | oracle reference | proposed cross-swap | proposed cross-swap |
| learned-curriculum decoder | reverse cross-swap | frozen-cold control | current system |

This matrix separates three effects:

1. **Cold imitation cost:** oracle versus cold boundaries under the same
   decoder.
2. **RL drift cost:** cold versus post-RL boundaries under the same decoder.
3. **Decoder co-adaptation:** whether the two cross-swaps behave differently.

Interpretation of the proposed oracle-decoder/post-RL-segmenter cell:

- near-oracle WER would weaken the boundary-quality hypothesis and implicate
  decoder curriculum;
- WER near the current learned system would strongly implicate boundary drift;
- WER worse than both diagonal systems would reveal strong decoder/segmenter
  co-adaptation, not prove that either component is independently bad.

The primary score should be dev-clean first, followed by test-clean/test-other
only after the intervention is fixed. Evaluate both CNN and Transformer
seed-3407 selected checkpoints initially; replicate only if the swap is
informative.

## 5. Qualitative examples

All examples below come from `reports/wavlm_char_boundary_audit.json`. Expected
behavior is the saved char-CTC boundary stream. “Cold” and “RL” are the CNN
seed-3407 argmax policies; counts are predicted/gold boundaries. Exact and ±2
F1 provide the trace from strict placement to tolerance-aware placement.

### 5.1 `2803-161169-0002` — difficult cold-start timing

Context: “WHY DID HE GIVE THAT SO ODD A SHAPE OR SO STRANGE A COVERING” (245
frames, 61 oracle boundaries).

| stage | count | exact F1 | ±2 F1 | observed result |
|---|---:|---:|---:|---|
| cold CNN | 58/61 | 0.639 | 0.941 | Most cold errors are within two frames. |
| RL CNN | 57/61 | 0.475 | 0.814 | Similar count, substantially relocated boundaries. |

**Interpretation:** evidence-backed relocation rather than rate reduction alone;
the ASR consequence for this utterance is not available, so causality is open.

### 5.2 `1919-142785-0018` — RL deletes and moves boundaries

Context: “IT IS HARDIER THAN THE ORANGE … BY THE ARABIANS” (352 frames, 101
oracle boundaries).

| stage | count | exact F1 | ±2 F1 | observed result |
|---|---:|---:|---:|---|
| cold CNN | 113/101 | 0.757 | 0.944 | High near-boundary coverage with extra starts. |
| RL CNN | 92/101 | 0.580 | 0.808 | Fewer starts and lower positional agreement. |

**Interpretation:** evidence-backed mixture of deletion and relocation after RL.

### 5.3 `2277-149897-0013` — high-density speech is compressed harder

Context: “FOR RELIEF HE AROSE … WHO WERE DRINKING” (220 frames, 84 oracle
boundaries).

| stage | count | exact F1 | ±2 F1 | observed result |
|---|---:|---:|---:|---|
| cold CNN | 90/84 | 0.816 | 0.954 | Close to the oracle after timing tolerance. |
| RL CNN | 69/84 | 0.575 | 0.824 | About 18% fewer starts than oracle plus relocation. |

**Interpretation:** evidence-backed loss of char coverage; the rate band permits
this behavior by design.

### 5.4 `2277-149896-0028` — a strong cold case is not preserved

Context: “HE PUT ON HIS HAT AND LOOKED AROUND FOR HIS UMBRELLA” (144 frames, 53
oracle boundaries).

| stage | count | exact F1 | ±2 F1 | observed result |
|---|---:|---:|---:|---|
| cold CNN | 56/53 | 0.899 | 0.954 | Strong exact and tolerant imitation. |
| RL CNN | 45/53 | 0.571 | 0.857 | The good initialization is substantially eroded. |

**Interpretation:** evidence-backed counterexample to the assumption that RL
only corrects weak cold-start cases.

## 6. Failure-mode synthesis

### 6.1 Cross-example taxonomy

| failure mode | examples | layer implicated |
|---|---|---|
| One-frame cold-start offsets | all four; especially `2803-161169-0002` | supervised boundary timing / target tolerance |
| RL boundary deletion | `1919-142785-0018`, `2277-149897-0013`, `2277-149896-0028` | rate objective + policy reward |
| RL boundary relocation at similar count | `2803-161169-0002` | unanchored GRPO policy |
| Decoder curriculum confound | aggregate comparison | experimental control / checkpoint staging |
| Slight ρ overcount | all reported learned rates | metric implementation |

### 6.2 Prevalence and priority

**Evidence-backed:** cold-start imperfection is corpus-wide, not anecdotal: full
dev logs are approximately 0.8 exact F1. In the stratified sample, cold CNN has
373 strict errors among 2,027 oracle events plus 459 excess/misaligned predicted
events; ±1 matching recovers most of them. Post-RL CNN recalls only 1,040/2,027
oracle boundaries exactly, and Transformer recalls 895/2,027. The largest
priority is therefore post-RL semantic drift, followed by cold timing precision.

### 6.3 Root-cause clustering

**Inferred:** deletion and relocation share one design cause: after epoch 6,
“char-ness” is not part of either the loss or checkpoint selection. The rate
band controls only how many boundaries survive, while NLL reward controls their
placement through a decoder that is itself changing in the multitask arm.
Boundary identity is consequently free to drift even when aggregate ρ looks
stable.

### 6.4 Design versus implementation failures

- **Design:** cold BCE is treated as initialization rather than a constraint;
  exact-frame BCE ignores tolerance; oracle and learned decoder recipes are not
  matched; RL seeds share the same warmup; final rather than best warmup is
  staged.
- **Implementation:** learned ρ is double-counted at frame 0. No evidence was
  found for a wrong or missing segmenter checkpoint.

### 6.5 Collective assessment

The segmenter does learn a useful char-timing model during cold-start, but the
current RL stage does not learn a demonstrably better char-derived segmentation.
It learns a different lower-rate policy, and task WER does not recover the
oracle advantage. The bottleneck is therefore a combination of boundary
fidelity and objective/curriculum design, not simply insufficient RL epochs.
Training longer under the same unanchored objective is unlikely to make the
policy return to char structure.

## 7. Recommendations

1. **P0 — Run a same-decoder boundary-swap evaluation.** For each selected
   decoder checkpoint, evaluate without retraining using (a) oracle char,
   (b) cold predicted, (c) post-RL predicted, and (d) fixed-rate boundaries
   matched to the post-RL count. This directly addresses relocation/deletion.
   **Success criterion:** obtain a per-split WER decomposition of oracle-vs-cold
   imitation cost and cold-vs-RL drift cost. **Prerequisite:** evaluation code
   must permit boundary-source override while keeping decoder weights fixed.

2. **P0 — Match the decoder protocol for oracle versus cold predictions.** Train
   three independent seeds with identical epochs, LR schedule, dynamic-batch
   limits, accumulation, decode budget, and checkpoint rule; change only the
   boundary stream. **Success criterion:** a defensible causal estimate with
   95% intervals or paired seed differences. **Prerequisite:** decide one common
   protocol before launching.

3. **P1 — Preserve a controllable amount of char structure during RL.** Add an
   annealed char-BCE or KL-to-cold-policy term and checkpoint on both dev WER and
   boundary drift. Prefer a tolerance-aware auxiliary target or local-window
   matching so ±1-frame alternatives are not punished as harshly as unrelated
   boundaries. **Success criterion:** retain at least 0.90 ±1-frame F1 on
   dev-clean while improving WER over the frozen-cold control at matched ρ.
   **Prerequisite:** full-dev post-RL boundary metrics in every epoch.

4. **P1 — Make the three seeds independent and stage best-dev warmup.** Repeat
   cold-start and decoder warmup per seed, or label the existing runs as three
   RL continuations from one shared initialization. Stage the best warmup rather
   than unconditional epoch 6. **Success criterion:** reported variance includes
   the whole curriculum and no run starts from a known worse checkpoint.

5. **P2 — Fix and regenerate rate reporting.** Count frame 0 exactly once.
   **Success criterion:** logged segment counts equal `compute_segment_ids` on a
   unit test with frame-0 labels both 0 and 1.

## 8. Bottom line

**Conclusion:** yes, boundary learning is part of the problem, but “the
segmenter is not learning” is too broad. It learns an approximately char-like
cold policy, then the unanchored RL objective deliberately allows that policy to
move far away from char boundaries. The current evidence does not support a
checkpoint-loading bug or the idea that more epochs alone will close the gap;
the next experiment should isolate boundary placement with a same-decoder swap
before changing the architecture.
