# Transformer Segmenter Training Review

**Date:** 2026-08-05  
**Depth:** Standard  
**Subject:** Transformer-encoder boundary-policy replication of the corrected shared-warm-up segmenter sweep

## 1. Executive Summary

The Transformer boundary encoder trained successfully, but it did not improve the system. All nine production runs reached epoch 10 and all nine produced complete test-clean and test-other evaluations. The best Transformer arm was `nll_mt` at **7.16 +/- 0.37 test-clean WER and 12.53 +/- 0.81 test-other WER**, versus the matched CNN arm at **6.19 +/- 0.58 and 11.12 +/- 0.44**. The gap persists after removing >200%-WER decoding tails, so it is not an outlier artifact. The CER-trained Transformer was unstable after its first RL epoch, and two otherwise-complete jobs OOM'd only while evaluating the final `dev-other` split.

Bottom-line status: training is complete enough to answer the backbone question. Under the current matched recipe, the Transformer is worse than the CNN and should not replace it.

## 2. Scope and Evidence

### Claims reviewed

1. The Transformer shared warm-up produced a valid reusable epoch-6 checkpoint despite its post-training evaluation OOM.
2. The nine stalled production jobs were successfully resubmitted from that checkpoint.
3. The Transformer sweep remained “in progress” and its conclusion was still open.
4. The Transformer backbone trained stably and preserved a viable keep ratio.
5. The Transformer matched or improved on the CNN backbone.
6. CER reward was a viable alternative to NLL reward for this backbone.

### Primary evidence

- Status claims: `docs/project_notes/issues.md`, entries “Review CER-MT and Transformer-backbone sweep progress” and “Resubmit stalled Transformer-backbone production jobs.”
- Stale primary report: `artifacts/segmenter/report.html` and `recipes/LibriSpeech/ASR/transformer/make_training_report.py`.
- Shared warm-up: `recipes/LibriSpeech/ASR/transformer/results/speechllm_segmenter/shared_warmup/transformer/3407/` and `slurm_logs/tf_warm6_3407_45236.out`.
- Production outputs: `recipes/LibriSpeech/ASR/transformer/results/speechllm_segmenter/multiseed_shared_warmup_onpolicy/{nll_frozen,nll_mt,cer_mt}/{cnn,transformer}/{3407,3408,3409}/`.
- Production SLURM logs: `recipes/LibriSpeech/ASR/transformer/slurm_logs/tf_{nf,nm,cm}_340{7,8,9}_466*.out`.
- Critical code paths: `segmenter.py` (`TransformerBoundaryPredictor`), `train_speechllm_with_segmenter.py` (warm-up, joint objectives, checkpoint selection, split evaluation), `run_segmenter_from_shared_warmup.slurm`, and the saved `hyperparams.yaml` files.

### Not reviewed / unavailable

- Current SLURM queue state could not be queried because the login shell could not contact the SLURM controller. Artifact timestamps and terminal log records were used instead.
- Per-utterance boundary tensors were not saved, so the extra ASR deletions cannot be traced directly to individual boundary placements.
- This is a matched-recipe test, not a Transformer-specific hyperparameter search. It does not establish that every Transformer segmenter would fail.

## 3. Claims and Verdicts

| Claim | Verdict | Evidence |
|---|---|---|
| The shared warm-up checkpoint is complete and reusable. | **Confirmed — evidence-backed.** | Epochs 1-6 completed; epoch-6 valid WER was 8.07. The checkpoint `shared_warmup/transformer/3407/save/CKPT+2026-08-05+02-10-59+00` contains all nine required files. Each production log records staging and loading it before epoch 7. |
| Nine Transformer production jobs were resubmitted from the shared warm-up. | **Confirmed — evidence-backed.** | All nine output directories exist; every `train_log.txt` contains epoch 10. Selected checkpoints are epochs 7-10 depending on validation WER. |
| The Transformer sweep is still in progress / conclusions remain open. | **Refuted as stale — evidence-backed.** | All nine runs have test-clean/test-other WER files. Seven have all three evaluation splits. The stale wording remains in `report.html` and `make_training_report.py`. |
| Transformer training collapsed. | **Refuted — evidence-backed.** | NLL arms maintained test keep ratios of about 0.24-0.28, within the configured band, and completed all four joint epochs. There was no zero-rate absorbing state. |
| Transformer matches or beats CNN. | **Refuted — evidence-backed.** | Transformer `nll_mt` is +0.98 WER on test-clean and +1.41 on test-other by three-seed mean. It is worse in five of six paired seed/split comparisons; the one -0.08 clean “win” disappears as evidence of general superiority under per-utterance/outlier analysis. |
| CER is competitive with NLL on the Transformer. | **Refuted for this run — evidence-backed.** | Transformer CER selects epoch 7 for all three seeds; later validation WER degrades. Its mean 7.56/13.06 trails NLL-MT's 7.16/12.53 and costs roughly four times as long for the joint phase. |
| The two OOM jobs failed during training. | **Refuted / misleading — evidence-backed.** | Both reached epoch 10 and wrote test-clean/test-other results. `nll_frozen/3409` and `cer_mt/3408` OOM'd only on the third sequential evaluation (`dev-other`), with 37.9-49.5 GiB reserved but unallocated. |

## 4. Independently Recomputed Results

Means and sample standard deviations below were recomputed from the raw WER edit-count files, not copied from a report.

| Variant | Backbone | test-clean WER | test-other WER | test-clean rho | test-other rho |
|---|---|---:|---:|---:|---:|
| NLL, decoder frozen | CNN | 7.18 +/- 0.09 | 11.86 +/- 0.10 | 0.288 +/- 0.002 | 0.279 +/- 0.001 |
| NLL, decoder frozen | Transformer | 8.62 +/- 0.21 | 14.65 +/- 0.35 | 0.274 +/- 0.005 | 0.264 +/- 0.004 |
| NLL, decoder co-trained | CNN | **6.19 +/- 0.58** | **11.12 +/- 0.44** | 0.265 +/- 0.018 | 0.257 +/- 0.017 |
| NLL, decoder co-trained | Transformer | **7.16 +/- 0.37** | **12.53 +/- 0.81** | 0.272 +/- 0.006 | 0.261 +/- 0.009 |
| CER, decoder co-trained | CNN | 6.67 +/- 0.41 | 11.31 +/- 0.39 | 0.254 +/- 0.007 | 0.245 +/- 0.008 |
| CER, decoder co-trained | Transformer | 7.56 +/- 0.21 | 13.06 +/- 0.45 | 0.250 +/- 0.017 | 0.240 +/- 0.015 |

### Training trajectory

- **Evidence-backed:** Supervised cold-start training worked. Transformer boundary F1 reached 0.887 versus CNN 0.895. The larger 3.36M-parameter Transformer did not improve the boundary proxy over the 1.57M-parameter CNN.
- **Evidence-backed:** Shared decoder warm-up reduced train loss from 3.06 to 0.135 and valid WER from 276 at epoch 1 to 8.07 at epoch 6. Test WER at the warm-up checkpoint was 9.45/14.30.
- **Evidence-backed:** NLL co-training improved the Transformer over its frozen-decoder counterpart for every seed on both test splits: -1.03/-2.70, -2.09/-1.94, and -1.24/-1.72 WER (clean/other for seeds 3407/3408/3409).
- **Evidence-backed:** CER dynamics were unstable after epoch 7. All three seeds selected epoch 7. For example, seed 3407 valid WER moved 6.62 -> 8.12 -> 8.29 -> 11.77 while validation keep ratio fell 0.250 -> 0.227 -> 0.241 -> 0.210.
- **Inferred:** The Transformer is not failing because it simply keeps fewer frames: NLL-MT CNN and Transformer are compared at nearly the same test keep ratios. Architecture/tuning affects the usefulness and placement of boundaries, not just their number.

### Outlier-robust and per-utterance findings

For the best arm (`nll_mt`), across the three seeds:

| Slice | CNN raw WER | Transformer raw WER | CNN WER after removing >200%-WER utterances | Transformer trimmed WER |
|---|---:|---:|---:|---:|
| test-clean | 6.19 | 7.16 | 5.95 | 7.08 |
| test-other | 11.12 | 12.53 | 10.87 | 11.87 |

- **Evidence-backed:** The Transformer deficit survives tail removal: +1.13 clean and +1.00 other.
- **Evidence-backed:** Transformer is worse on 1,705 versus better on 1,122 of 7,860 paired clean seed-utterances; on other it is worse on 2,314 versus better on 1,669 of 8,817.
- **Evidence-backed:** Deletion-dominated utterances (at least five deletions and more deletions than insertions plus substitutions) occur 119 vs 80 times on clean and 93 vs 62 on other for Transformer vs CNN — about 50% more often.
- **Evidence-backed:** Extreme non-EOS repetition tails exist in both backbones. Across three seeds each backbone has one >200%-WER clean case and five other cases. They add variance but do not explain the Transformer-vs-CNN conclusion.

## 5. Qualitative Examples

All examples compare seed 3407 `nll_mt` raw WER artifacts.

### A. Transformer-only non-EOS repetition tail

**ID:** `121-121726-0002`, test-clean  
**Expected:** “ANGOR PAIN PAINFUL TO HEAR.”  
**CNN:** One substitution, 1/5 errors.  
**Transformer:** Repeats “AN GOR” until the batch decode cap; 130/5 errors (125 insertions).  
**Interpretation:** **Evidence-backed** decoder non-termination; **inferred** interaction with the batch-global decode cap. This is a decoding-layer tail, not direct proof of bad boundary placement.

### B. Ordinary contiguous deletion regression

**ID:** `1995-1836-0003`, test-clean  
**Expected:** 21-word sentence ending “…intelligence or whatever looked to her like intelligence in others.”  
**CNN:** Exact transcript, 0/21 errors.  
**Transformer:** Omits the seven-word span “or whatever looked to her like intelligence,” 7/21 errors.  
**Interpretation:** **Evidence-backed** deletion-span regression. The implicated layer is **open** because saved artifacts do not include the selected boundary sequence.

### C. Long-span deletion on an otherwise easy utterance

**ID:** `6930-75918-0003`, test-clean  
**Expected:** 64-word passage.  
**CNN:** Exact transcript, 0/64 errors.  
**Transformer:** Drops a 29-word middle span and substitutes one word, 30/64 errors, then resumes correctly at “too much unhappiness…”.  
**Interpretation:** **Evidence-backed** non-terminal deletion of a coherent acoustic span. It is not explained by a simple end-of-sequence decode-length limit.

### D. Counterexample: Transformer avoids a CNN repetition tail

**ID:** `7902-96595-0003`, test-other  
**Expected:** 17-word sentence.  
**CNN:** Repeats a phrase until the decode cap, 192/17 errors (187 insertions).  
**Transformer:** 4/17 errors.  
**Interpretation:** **Evidence-backed** counterexample showing that non-EOS tails are stochastic/model-specific and can favor either backbone. It explains why seed 3407 test-other is nearly tied despite the Transformer's broader ordinary-error disadvantage.

## 6. Failure-Mode Synthesis

| Failure mode | Examples / prevalence | Layer implicated |
|---|---|---|
| More deletion-dominated utterances | B, C; ~50% more qualifying cases for Transformer | Boundary placement / pooled representation / decoder interaction; exact layer open |
| Non-EOS repetition tail | A and D; rare in both backbones | Greedy evaluation length control / EOS behavior |
| CER late-epoch degradation | All three CER seeds select epoch 7 | RL objective and Transformer policy interaction |
| Sequential-evaluation CUDA OOM | Warm-up plus 2/9 production jobs | Evaluation implementation / allocator fragmentation |

### Prevalence and priority

The primary quality problem is not the rare repetition tail. The Transformer's ordinary paired error rate is worse, its tail-trimmed WER remains worse, and deletion-dominated cases are more common. The OOM is operationally annoying but does not invalidate test-clean/test-other or any trained checkpoint.

### Root-cause clustering

- **Evidence-backed:** The rate is controlled and comparable, so “Transformer used fewer tokens” is not a sufficient explanation.
- **Inferred:** Global contextual mixing or reused CNN-tuned optimization may produce less locally faithful boundary choices, expressed downstream as coherent deletion spans.
- **Open question:** Whether the cause is architecture, segmenter learning rate, over-capacity on train-clean-100, or a particular change in boundary-location distribution. Per-utterance boundary dumps are required to distinguish them.

### Design vs implementation

- **Design/research outcome:** The current 4-layer Transformer is a worse matched backbone than the CNN.
- **Implementation defect:** Evaluation runs three splits sequentially without allocator cleanup or split isolation; this causes post-training OOMs and incomplete `dev-other` coverage.
- **Shared evaluation defect:** The batch-global decode cap permits rare very long non-EOS hypotheses in both backbones.

## 7. Recommendations

1. **Use CNN `nll_mt` as the default result.** It is better on both splits at a comparable keep ratio and is simpler/smaller.  
   **Success criterion:** future main tables and claims identify 6.19/11.12 as the current learned-segmenter result and do not describe Transformer as pending.

2. **Repair evaluation, then evaluate only the two missing `dev-other` cases — do not retrain.** Run splits in isolated processes or clear model/search tensors and CUDA cache between splits; reduce evaluation batch size and/or enable expandable allocator segments.  
   **Success criterion:** `nll_frozen/transformer/3409/wer_dev-other.txt` and `cer_mt/transformer/3408/wer_dev-other.txt` exist with no CUDA OOM.

3. **Do not launch another full Transformer sweep yet.** If rescuing the architecture matters, first run one-seed `nll_mt` diagnostics that save per-utterance boundaries and compare segment-length/location distributions with CNN. Then try a small Transformer-specific learning-rate/capacity test.  
   **Success criterion:** identify a measurable boundary-distribution cause and beat the paired CNN checkpoint on validation before expanding to three seeds.

4. **Fix per-utterance decode termination before interpreting small seed deltas.**  
   **Success criterion:** no hypothesis exceeds a defensible per-utterance cap; report both original and re-decoded WER for auditability.

## 8. Bottom Line

The Transformer encoder trained, stayed within the rate constraint, and generated usable checkpoints; this was not a training collapse. But the completed matched sweep is negative: its best configuration remains roughly **+1.0 clean / +1.4 other WER** behind CNN, with more deletion-span failures and CER instability. Keep the CNN/NLL multi-task system, clean up the evaluation OOMs, and treat any Transformer follow-up as targeted diagnosis rather than another broad replication.
