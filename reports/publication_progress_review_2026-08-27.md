# Publication progress review

**Date:** 2026-08-27  
**Depth:** Standard progress review  
**Scope:** Completed LibriSpeech-100h attribution, corrected LibriSpeech-960h
training, memory-limited inference measurements, the current HTML report, and the
publication TODO. Bilevel optimization is excluded because it is no longer on the
publication path.

## 1. Executive summary

The core result is now credible: at 100h, joint RL with the local-history
Transformer segmenter and BiGRU pooling reaches **4.95±0.02 / 9.11±0.18 WER**,
versus **5.17±0.05 / 9.70±0.12** for fixed k=5 + BiGRU and
**5.09±0.34 / 9.52±0.27** for the frozen learned segmenter. At 960h, the CNN and
Transformer learned systems are both complete for three seeds and essentially tie
at about **2.75–2.79 / 5.63–5.67 WER**.

The project no longer needs more policy-architecture exploration. The publication
risks are narrower:

1. there is no transcript-free CTC-compression baseline;
2. the learned 960h runs have a longer training budget and a stronger decoder start
   than the fixed/no-downsampling controls;
3. the learned 100h system emits about 11% more audio tokens/s than fixed k=5, so
   the “same cost” claim needs an exact rate-matched evaluation;
4. the current throughput result is strong, but the decoder is not KV-cached and
   batch-1/latency reporting is incomplete;
5. generalization beyond LibriSpeech is still untested.

## 2. Scope and primary evidence

- Current claims and tables: `artifacts/segmenter/report.html` and
  `recipes/LibriSpeech/ASR/transformer/make_training_report.py`.
- Matched 100h runs:
  `recipes/LibriSpeech/ASR/transformer/results/speechllm_segmenter_wavlm/attribution_100h_transformer_ar_bigru/`.
- Corrected learned 960h runs:
  `recipes/LibriSpeech/ASR/transformer/results/speechllm_ls960_step24k_corrected/`.
- 960h fixed, native, and character-alignment controls:
  `recipes/LibriSpeech/ASR/transformer/results/speechllm_ls960_scaleup_v2_optimized/`.
- Final inference measurements:
  `artifacts/segmenter/inference_eval_memory_limited_batch_v1/` and its
  `retry_batch32_v1/` subtree.
- Forced-alignment implementation:
  `recipes/LibriSpeech/ASR/transformer/ctc_boundary_align.py`.

All WER means below were independently recomputed from corpus edit counts in the
raw WER files. No current SLURM state was used: the controller did not answer a
five-second status query, but every claim below concerns completed artifacts.

## 3. Claims and verdicts

| Claim | Verdict | Evidence |
|---|---|---|
| The learned system scales to 960h. | **Confirmed — evidence-backed.** | CNN is 2.75±0.05 / 5.67±0.10 and Transformer is 2.79±0.02 / 5.63±0.12, each with 3/3 seeds. |
| Joint RL improves over a frozen learned segmenter at 100h. | **Confirmed in aggregate, but modest — evidence-backed.** | Joint is 0.14 clean / 0.41 other WER lower and emits 3.1 fewer audio tokens/s on clean. The paired direction favors joint in 2/3 seeds on both splits. |
| Joint RL improves over fixed k=5 + BiGRU. | **Confirmed at the attained operating points — evidence-backed.** | Joint is 0.22 clean / 0.59 other lower, with lower WER in all three paired seeds, but uses 11.2 rather than 10.1 Hz on clean. |
| Learned segmentation is already fairly compared with fixed controls at 960h. | **Not confirmed.** | Learned runs use 24k steps and the best character-decoder start; reported fixed/native controls use one LS960 pass and mean/no pooling. |
| Learned segmentation reduces end-to-end inference cost. | **Partially confirmed.** | CNN/Transformer + BiGRU reach 0.0236/0.0246 RTF at batch 32 versus 0.0667 for native 50 Hz at batch 16, with segmenter cost included. The current decoder has no KV-cached greedy path, and TTFT/per-utterance latency are not reported. |
| The segmenter is input-adaptive rather than a noisy fixed stride. | **Supported by existing artifacts, not yet reported.** | After controlling linearly for duration, segment count correlates with reference word count at mean Spearman 0.69 for CNN and 0.76 for Transformer across seeds. Faster-speech quartiles receive about 3 more audio tokens/s than slower-speech quartiles. |
| A strong deployable CTC baseline exists. | **Refuted.** | Current character/word CTC boundaries call `forced_align` with transcript-derived targets. No audio-only posterior-collapse baseline was found. |
| The result generalizes beyond LibriSpeech ASR. | **Unsupported.** | No second corpus, corruption suite, or downstream task has final matched evidence. |

## 4. Derived findings

### 4.1 The attribution result is useful, but “fixed budget” is not yet exact

- Joint versus fixed gives relative WER reductions of about **4.3% clean** and
  **6.1% other**, but joint uses about **11% more audio tokens/s** on clean.
- Joint versus frozen improves WER while reducing frequency from 14.4 to 11.2 Hz.
  That is evidence that joint training is not merely keeping more frames.
- A development-set threshold calibration on the existing joint checkpoints is the
  cheapest way to test the fixed-budget claim before training more models.

### 4.2 The 960h headline is strong but does not isolate the source of the gain

- CNN and Transformer learned policies tie within seed variation.
- Both are also effectively tied with the character-alignment oracle
  (2.87±0.08 / 5.66±0.15), while using fewer audio tokens/s.
- The fixed k=5 and no-downsampling rows are much worse, but their one-pass training
  budget and weaker initialization prevent a causal “adaptive placement caused the
  gain” interpretation.

### 4.3 Existing inference records already contain a useful adaptivity result

Across complete test-clean and test-other records:

| system | utterance-frequency p10 / median / p90 | slowest speaking-rate quartile | fastest speaking-rate quartile | duration-adjusted segment/word correlation |
|---|---:|---:|---:|---:|
| CNN AR + BiGRU | 9.4 / 11.7 / 13.6 Hz | 10.0 Hz | 13.0 Hz | 0.69 |
| Transformer AR + BiGRU | 8.0 / 10.3 / 12.1 Hz | 8.5 Hz | 11.7 Hz | 0.76 |

The same pattern appears in every duration quartile, so it is not only the shared
duration denominator. This analysis should become a paper figure before launching
an additional adaptivity experiment.

### 4.4 Current efficiency evidence is meaningful but not deployment-final

- CNN and Transformer + BiGRU have 2.8× and 2.7× lower forward RTF than native
  50-Hz decoding at their largest reliable full-pass batches.
- They remain about 30–35% slower than fixed k=5 because boundary generation and
  BiGRU pooling add work.
- Because the current greedy searcher recomputes the Llama prefix without a KV
  cache, it may overstate how much audio-prefix compression changes a production
  decoder's total latency. A KV-cached benchmark is needed before making a broad
  deployment claim.

## 5. Representative qualitative comparisons

### A. Joint RL repairs a fixed-boundary deletion

**ID:** `1995-1837-0026`, seed 3407, test-clean.  
**Fixed k=5 + BiGRU:** 11/24 errors, including a nine-word deletion.  
**Joint Transformer AR + BiGRU:** 0/24 errors.  
**Interpretation:** The matched decoder/pooler comparison shows a real utterance-level
benefit from learned joint segmentation. The exact boundary responsible is not
recoverable from the WER artifact, so the mechanism remains inferred.

### B. Joint RL removes a long deletion from the frozen policy

**ID:** `6070-86744-0018`, seed 3409, test-other.  
**Frozen learned segmenter:** 24/81 errors, including 19 deletions.  
**Joint RL:** 5/81 errors and no deletions.  
**Interpretation:** This is a concrete example of the 0.41-point aggregate
test-other improvement, but it should not be treated as proof that every utterance
benefits; the paired seed result is mixed.

### C. The two 960h policies tie globally but fail differently

**ID:** `1188-133604-0008`, seed 3407, test-clean. CNN has 17/58 errors while
Transformer has 1/58. On `61-70968-0012`, CNN is exact while Transformer has 4/6
errors.  
**Interpretation:** The near-identical aggregate WER does not mean the policies are
interchangeable per utterance. This favors a paired bootstrap/error-slice analysis
over selecting one architecture from a 0.04-point mean difference.

## 6. Remaining failure modes and evidence gaps

| Gap | Type | Publication consequence |
|---|---|---|
| No transcript-free CTC posterior baseline | Missing external baseline | Reviewers cannot tell whether RL is needed beyond standard dynamic CTC compression. |
| Learned and fixed rates differ by about 1.1 Hz at 100h | Experimental control | “At a fixed token budget” is currently approximate. |
| 960h learned/control optimization budgets differ | Experimental control | The headline scale-up cannot yet attribute its gain to adaptive placement. |
| Non-KV-cached decoder and incomplete latency metrics | Measurement design | Current RTF is valid for this pipeline but not a deployment-complete cost claim. |
| Batch-dependent floating-point hypotheses remain | Implementation/reproducibility | Throughput runs should not replace batch-1 recognition evidence. |
| No robustness or second-corpus result | Generalization | The paper currently supports LibriSpeech ASR only. |

## 7. Recommended next experiments

### P0-A. Exact rate matching and paired statistics using existing checkpoints

Calibrate the joint-policy decision threshold on dev-clean/dev-other to match the
fixed control near 10.1 Hz, freeze it, and evaluate all three test seeds. Compute a
paired utterance bootstrap for joint versus fixed and joint versus frozen.

**Success criterion:** joint retains a clear WER advantage at matched frequency. If
the confidence interval crosses zero, screen two rate-penalty settings on one seed
and replicate only the useful point.

### P0-B. Transcript-free CTC-compression baseline

Use an audio-only CTC posterior stream: collapse repeated argmax labels, test a
small set of blank-run rules on development data, and pool WavLM frames at about
10–12 Hz. Train the same best-character decoder/BiGRU recipe for three seeds and
include its CTC encoder cost in the inference benchmark.

**Success criterion:** a three-seed WER/Hz/RTF row directly comparable with fixed
k=5 + BiGRU and joint Transformer AR + BiGRU. This is the most important missing
external baseline.

### P0-C. One matched fixed-control scale-up

Run **fixed k=5 + BiGRU with the same best-character decoder initialization and
24k-step decoder adaptation budget** as the learned 960h systems, for three seeds.
Do not repeat the full fixed-rate sweep. Add a frozen learned 960h arm only if the
100h paired bootstrap leaves the RL contribution unresolved.

**Success criterion:** the learned LS960 system beats the matched fixed control at
similar frequency. This removes the largest remaining confound in the headline
table.

### P0-D. Production-style efficiency completion

Add a KV-cached Llama evaluation path and benchmark native 50 Hz, fixed k=5 +
BiGRU, transcript-free CTC, CNN AR + BiGRU, and Transformer AR + BiGRU. Report
batch-1 TTFT/total latency, a throughput batch, utterances/s, RTF, component time,
peak memory, and median/p95 latency by duration bucket.

**Success criterion:** learned segmentation still gives a useful wall-clock or
memory benefit when decoder caching and the segmenter cost are both included.

### P1. Adaptivity figure, then one robustness evaluation

First turn the existing per-utterance frequency/word-rate evidence into a generated
figure; this needs no retraining. Then evaluate the locked learned, fixed, and CTC
systems on one standard noise/reverberation suite without method-specific tuning.

**Success criterion:** learned allocation remains input-dependent and its WER
advantage does not disappear under moderate corruption. A new speech-LLM task or
speech-translation task is lower priority.

## 8. Work that is no longer high priority

- More CNN-versus-Transformer policy seeds: both are already complete at 100h and
  960h, and their aggregate WER is tied at scale.
- Additional BiGRU architecture sweeps: mean/BiGRU comparisons are already
  three-seed complete.
- Bilevel optimization, more sampler architectures, or longer RL schedules.
- A full 960h fixed-rate sweep.
- A new downstream speech-LLM task before the CTC, rate-matching, and efficiency
  gaps are closed.

## 9. Bottom line

The paper has moved from “does the method work?” to “is the comparison reviewer
proof?” The answer to the first question is yes: joint learned segmentation is
strong at 100h, scales to 960h, and reduces measured inference cost. The fastest
route to publication is now a transcript-free CTC baseline, exact rate matching,
one matched fixed-control LS960 run, and a production-style latency audit—not more
segmenter architectures or a new task.
