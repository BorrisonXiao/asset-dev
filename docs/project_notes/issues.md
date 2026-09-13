# Work Log

Quick reference log of completed work. Not a replacement for GitHub issues/PRs.

## Format

Each entry should include:
- Date (YYYY-MM-DD)
- Brief description (1-2 lines)
- Status

## Entries

### 2026-09-06 - Publish the baseline results in the report's WIP section
- **Status**: Completed -- live at
  https://borrisonxiao.github.io/jsalt26-downsampling/reports/segmenter-training.html
  (source `9f6ddb446`, Pages `8f3c27a`), verified byte-identical.
- **Description**: The WIP section reported only the multi-task run, whose
  numbers cannot say what limits the non-ASR tasks. The baselines now lead it
  and the multi-task run follows as the thing they explain. One table covers
  6 arms x 3 tasks with the multi-task cohort and the CountNet ceiling as rows.
- **Notes**: The table reads through `summarize_singletask_baselines.collect()`
  rather than copied numbers, so the report and the CLI cannot drift. One claim
  was weakened in the process: the learned policy's seed-stability advantage was
  computed across all three tasks and read "0.002-0.022 against 0.001-0.136",
  but that 0.001 floor came from saturated intent and had nothing to do with
  segmentation. It is now stated per task on the two with headroom, with intent
  excluded and the reason given.

### 2026-09-04/05 - Single-task baselines: isolate the non-ASR bottleneck
- **Status**: Completed -- 56/56 jobs, 0 failures. Results in `key_facts.md`
  (SB-1/2/3); collector `summarize_singletask_baselines.py`; CSV at
  `artifacts/segmenter/singletask_baselines.csv`.
- **Description**: Six arms per task (fixed grids at 50/25/10/5 Hz, the learned
  policy frozen, and the learned policy trained with K=4 GRPO) x 3 tasks x 3
  seeds, each arm differing in nothing but the segmentation. Answers which of
  rate / placement / mixture limits the non-ASR tasks.
- **Findings**: rate is free (10x compression costs nothing on any task);
  placement buys stability but no accuracy; GRPO compresses at constant quality
  but unevenly (emotion 2x, speaker count 1.07x despite fixed-rate proof that
  5 Hz is fine there); multi-task interference costs ~0.12-0.13 where there is
  headroom; speaker count still trails CountNet 0.773 vs 0.871, so the residual
  is the pooled-prefix interface.
- **Two bugs found and fixed en route**: candidate scoring OOMed at 50 Hz by
  padding the audio prefix into a float32 cross-entropy (~98% waste); and
  held-out subsets aliased against class-interleaved manifests, which silently
  reduced speaker-count validation to 3 of 5 classes and drove every checkpoint
  selection for that task. Both in `bugs.md`, both with regression tests.
- **Known gap**: classification runs write only the selection metric to
  `wer_results/`, no confusion matrix, so per-class diagnosis needs a re-run.
  Worth adding to `_SelectionErrorRate.write_stats`.


### 2026-09-04 - Report the non-ASR multi-task WIP from one selected checkpoint
- **Status**: Completed -- report regenerated, validated, committed and live at https://borrisonxiao.github.io/jsalt26-downsampling/reports/segmenter-training.html (source `ae50afacb`, Pages `22c8294`); NA-2 results logged in `key_facts.md`
- **Description**: The published WIP section compared the two rate-penalty
  channels (`reward` vs `aux`), but ASR-WER-only checkpoint selection keeps a
  pre-policy checkpoint in every arm -- the one regime where the channels are
  identical -- so the comparison could not distinguish them and implied a result
  the design cannot support. Replaced it with seed 3408 at step 3000 reported
  across all four tasks with the emitted rate beside every metric, plus a
  per-task rate trajectory and an explicit pre-compression caveat.
- **Follow-up (same day)**: Dropped both WIP figures at the user's request and
  moved each task's validation quality into the rate-trajectory table, so one
  table now carries the rate/quality trade-off. Side effect worth knowing: the
  two `hk.svg_line` charts were **328 KB each** of inline path data, so the
  standalone page fell from 2.69 MB to 2.05 MB. Every claim the figures carried
  (FiLM-phase degeneracy, the Phase-C rate drop, emotion's best checkpoint) is
  now computed from `train_log.txt`/`task_metrics.jsonl` in the generator instead
  of being hand-typed prose.
- **Notes**: Selected checkpoint: intent macro-F1 0.994 @ 7.55 Hz, speaker count
  0.730 acc @ 10.57 Hz, emotion macro-F1 0.350 @ 8.56 Hz, ASR 2.982/6.058 WER on
  test-clean/other @ ~10.4 Hz. Those rates are the warm start's, not compressed
  ones -- they are identical across all three seeds, which is itself the evidence
  that selection is reporting a pre-policy model. `NONASR_RUN` now resolves
  through `RES` so the generator is cwd-independent.

### 2026-08-30/31 - Make the skipjack -> CLSP port actually runnable and verify it
- **Status**: Completed -- flagship trains and decodes on CLSP; multi-task ST corpora restored; reporting toolchain confirmed working once the skills were ported (2026-08-31), which then exposed a pre-existing three-way drift between generator, local artifact, and published page (see `bugs.md`)
- **Description**: The 2026-08-24..30 rsync port copied data only -- the venv, every launcher, every hparams file and every Slurm flag still named skipjack. Rehomed the environment, rewrote 111 files' paths and 49 files' Slurm identity flags (scripted in `tools/rehome_paths_clsp.sh` / `tools/rehome_slurm_clsp.sh`), restored the data layout, and verified the WavLM Transformer-AR local-64 + BiGRU system trains and decodes on CLSP A100s.
- **Notes**: Verification evidence, the full path/flag map, and the remaining gaps are in `key_facts.md` ("Cluster Migration" section). New helper: `tools/preflight_clsp_assets.py` (CPU asset check) and `recipes/.../verify_clsp_port_{inference,training}.slurm`.

### 2026-08-23 - Correct and launch the LibriSpeech-960h scale-up queue
- **Status**: Optimized v2 queued; superseded v1 pending jobs cancelled
- **Description**: Submitted the requested full-data study using `train-clean-100`, `train-clean-360`, and `train-other-500` for every arm, then replaced the literal 100h schedule reuse with a compute-matched LS960 schedule.
- **Notes**: Optimized jobs 105296–105312 use shared verified manifests, larger BF16 dynamic batches, persistent workers, batched reward evaluation, and a single learned pass with 20% decoder warmup / 80% joint RL. Outputs are isolated under `results/speechllm_ls960_scaleup_v2_optimized/`. The superseded pending v1 jobs were cancelled; v1 jobs 105257/105258/105272 had already started and require explicit cancellation approval. See ADR-012/013 and `bugs.md`.

### 2026-08-21 - Reaffirm A100-only GPU policy during corrected evaluation
- **Status**: Completed
- **Description**: Reconfirmed that every GPU job for this project must use the JSALT account on the `a100` partition; B200, B300, H200, and every other GPU partition are prohibited.
- **Notes**: No non-A100 job was submitted. The existing `ga129` hardware-fault exclusion remains the default. The user later authorized a one-off ga129 retry for these urgent single-GPU evaluations: a real batch-8 canary restored the checkpoint and ran 96 decoding steps without CUDA/NCCL/NVLink/Xid/OOM errors, after which the remaining corrected evaluations were allowed to queue there. This does not remove the future default exclusion without explicit user approval. See `key_facts.md`.

### 2026-08-17 - Document BiGRU residual pooling and standardize reports on Hz
- **Status**: Completed
- **Description**: Added a labeled BiGRU pooling design diagram and component/training-schedule table to the cumulative HTML report, then converted reader-facing compression values across report generators from retained-frame fractions to audio-token frequency in Hz.
- **Notes**: The reporting rule is `f_audio = 50 Hz × rho`; lower Hz means stronger compression. Internal config/log/checkpoint fields retain `rho`. Updated the training report, pipeline/frontier figures, OPD proposal, bilevel analysis, Transformer-AR rollout note, and STE analysis. See ADR-011.

### 2026-08-11 - Propose on-policy prefix distillation
- **Status**: Proposed; not implemented or launched
- **Description**: Wrote a standalone HTML proposal and implementation plan for adapting the decoder to prefixes sampled from the current WavLM CNN first-order AR segmenter, using the frozen oracle-char decoder as a teacher.
- **Notes**: The first study freezes the segmenter and uses a controlled 2x2 comparison: greedy versus sampled decoder prefixes, each trained with CE versus CE+distillation. Evaluation reuses identical boundary sets at approximately rho 0.29/0.23/0.20 so decoder input-rate robustness is separated from boundary placement. HTML: `artifacts/segmenter/on_policy_prefix_distillation.html`; plan: `plans/on_policy_prefix_distillation.md`; generator: `recipes/LibriSpeech/ASR/transformer/make_on_policy_prefix_distillation_proposal.py`. Published in Pages commit `a6badc4` under the new Proposals collection; the existing decoder-segmenter analysis is now an item under the new Analyses collection.

### 2026-08-11 - Finalize structured-policy experiments and refresh report
- **Status**: Completed (jobs 60853–60855 and 63824–63827)
- **Description**: Added the finished three-seed WavLM Bernoulli/first-order AR comparison and the 30-epoch wav2vec2 Transformer AR continuation to project memory and the HTML report.
- **Notes**: First-order AR improves test-clean in all three paired WavLM seeds (5.06±0.18 vs. 5.62±0.52); the long Transformer runs select epochs 1/2/4 and overfit afterward. Published Pages commit `8ac6168`.

### 2026-08-10 - Compile cumulative segmenter results and reorganize HTML report
- **Status**: Completed
- **Description**: Recomputed WER from edit-count artifacts and reorganized the report around five causal questions: compression, boundary placement, decoder initialization, encoder interaction, and label-dependent sampling.
- **Notes**: Added char/phone oracle controls to the WavLM frontier, the six-run hybrid study, same-decoder boundary swap, cold-start F1 comparison, matched encoder table, and live long-AR status. Review: `reports/segmenter_cumulative_results_review.md`; report: `artifacts/segmenter/report.html`; published Pages commit `92c92ce`.

### 2026-08-10 - Complete oracle-close and dual-encoder controlled studies
- **Status**: Completed (jobs 60614/60615 and 60818–60823)
- **Description**: Oracle-close Bernoulli/first-order AR finished at 5.03/9.45 and 4.86/9.39 test WER (one seed); the CNN-segmenter-on-wav2vec2/WavLM-pooling study finished at 5.16±0.32/9.44±0.48 with the best char decoder versus 6.48±0.14/10.04±0.15 from scratch.
- **Notes**: The fixed-segmenter paired comparison attributes −1.32 clean/−0.61 other to decoder initialization. First-order AR remains directional single-seed evidence.

### 2026-08-10 - Launch long-horizon wav2vec2 Transformer autoregressive training
- **Status**: Running (jobs 60853–60855)
- **Description**: Launched seeds 3407–3409 for 30 joint NLL-GRPO epochs from the best completed wav2vec2 Transformer NLL-MT segmenter+decoder checkpoint (seed 3407, epoch 10, dev WER 5.85). The first-order autoregressive transition is zero-initialized, preserving the source model's initial behavior exactly.
- **Notes**: Uses a single frozen wav2vec2-base encoder, Transformer acoustic backbone, co-trained decoder at 2e-4, segmenter at 5e-5, and kept-ratio band `[0.15, 0.25]`; logs now include both transition biases and their gap. Outputs: `results/speechllm_segmenter/w2v2_transformer_autoregressive_long/nll_mt/{3407,3408,3409}/`. This is not the still-planned full decoder-only `transformer_ar` policy.

### 2026-08-10 - Launch controlled wav2vec2-segmenter / WavLM-pooling study
- **Status**: Running (jobs 60818–60823)
- **Description**: Added an optional dual-SSL path with a hard shared-frame-grid check, then launched three seeds each from the best WavLM oracle-char decoder (60818–60820) and a fresh decoder with six warmup epochs (60821–60823). Both arms fix the best wav2vec2 CNN char segmenter (seed 3407, epoch 4, dev F1 0.89454) and receive 10 matched Bernoulli NLL-GRPO epochs.
- **Notes**: All six jobs loaded the exact segmenter, verified 768-d wav2vec2 boundaries on the same frame grid as 1024-d WavLM pooling, and entered epoch 1. Outputs: `results/speechllm_segmenter_wavlm/hybrid_w2v2_segmenter_controlled/{oracle_char_decoder,scratch_decoder}/{3407,3408,3409}/`. Fully independent replication is deferred.

### 2026-08-09 - Launch oracle-close longer RL and structured-policy comparison
- **Status**: Running (jobs 60614 and 60615)
- **Description**: Started matched WavLM-CNN runs from the best char segmenter (dev F1 0.7992) and best oracle-char decoder (seed 3408, epoch 10, dev WER 3.69): 2 decoder-adaptation epochs followed by 10 joint NLL-GRPO epochs, with a `[0.15, 0.25]` kept-ratio band.
- **Notes**: Job 60614 is the independent Bernoulli control; job 60615 uses a first-order autoregressive boundary policy with an exact zero-bias behavior-preserving initialization. Outputs: `results/speechllm_segmenter_wavlm/oracleclose_bestinit/{bernoulli,autoregressive}/3407/`.

### 2026-07-22 - Stand up environment + data for alignment-boundary SpeechLLM training on bluecrab
- **Status**: Completed
- **Description**: Created dedicated venv, located and verified `LibriSpeech`/`LibriSpeech_alignments`/`wavlm_boundaries` data and the shared HF model cache, fixed placeholder paths in `speechllm_fixed_pooling.yaml`, ran a debug smoke test then a full real training run with CTC word boundaries (`word_ctc`).
- **URL**: —
- **Notes**: Branch `blank_analysis`. See `decisions.md` ADR-001–003 and `bugs.md` for the debugging path (torch pin, gated-model access, HF cache_dir gap). Result logged in `key_facts.md` as exp-001 (WER 6.4-9.9% across LibriSpeech dev/test splits).

## Tips

- Keep descriptions brief (1-2 lines max)
- Clean out very old entries periodically (3+ months)

### 2026-07-23 - Build blank-removal ablation infra + launch E-repro + Phase-1 diagnostic
- **Status**: In progress
- **Description**: Investigating why removing CTC "blank"/separator tokens hurts WER (word_ctc pooling). Confirmed word_ctc = strict [sep,word] alternation (N sep + N word segs; "remove blank"=Drop, word embeddings unchanged). Built: (1) `ctc_boundary_align.word_ctc_separator_frames` + `word_ctc_sep` output (regenerated word_ctc verified bit-for-bit vs shared); generated for train-clean-100 + all dev/test (job 25415, 0 failures). (2) `blank_mode: keep|drop|zero|const|shuffle` in segment_pooling.py + train_speechllm.py (+ separator_target_dir). (3) `launch_blank_ablation.sh`, `phase1_blank_diag.py`.
- **Notes**: E-repro keep vs drop @train-clean-100 running (jobs 25461/25462); Phase-1 embedding diagnostic (25465). Plan: `plans/blank_symbol_investigation.md`. See bugs.md for the ssl_folder offline fix.

### 2026-08-02 - Implement learned RL dynamic-downsampling segmenter (v1) + submit pilot
- **Status**: Submitted (training)
- **Description**: Built the learned segmenter per `plans/dynamic_segmenter_rl_v1.md`. New files (recipes/LibriSpeech/ASR/transformer/): `segmenter.py` (single-logit sigmoid boundary policy; CNN + Transformer backbones; GRPO pg loss with gamma=1 and `.mean()`; ratio-space one-sided cap + token-tax rate loss; annealed entropy; cold-start BCE; no-op TaskFiLM), `train_speechllm_with_segmenter.py` (`SegmenterASR`: `coldstart` mode = char-CTC BCE; `joint` mode = decoder-warmup then joint decoder-CE + K-sample GRPO; single AdamW with 2 param groups; SSL frozen; loads cold-start segmenter via `coldstart_ckpt_dir`), `hparams/speechllm_segmenter.yaml`, `run_segmenter.slurm`, `run_fixed_rate.slurm`, `submit_pilot.sh`. Self-tests + `--debug` (coldstart & joint) pass.
- **Notes**: **Gumbel-ST dropped** (can't backprop through hard mean-pool → needs soft pooling/CIF; that's the follow-up differentiable arm). Fixed a rate-loss O(T^2) scaling bug (use ratio space). Used a stable `/weka/scratch/.../cxiao/ssl_cache` for wav2vec2-base-960h (see bugs.md ssl_folder issue). Pilot on train-clean-100, 10 ep, seed 3407: fixed-rate baselines k∈{3,4,5,6,8} (jobs 41285-89), cold-start cnn/tf (41281/41282), joint RL cnn/tf (41283/41284, afterok cold-start). Results under results/speechllm_segmenter/ and results/speechllm_fixed_pooling/fixed_rate_k*_tc100/.

### 2026-08-03 - Reward/multi-task ablations done; optimize CER GRPO (batched + off-policy)
- **Status**: In progress
- **Description**: Finished 3 CNN reward/multi-task ablations (warmup 6 / joint 4, band [0.1,0.3], entropy off): nll_frozen (test 7.55/11.86, ρ0.29, 7h40m), **nll_mt** (test **6.93/11.11**, ρ0.26, 7h17m; multi-task > frozen on every split), cer_mt (decode-bound, cancelled). Optimized the CER path (see ADR-004): `grpo_clipped_pg_loss` + custom `fit_batch` — batched K·B decode + off-policy reuse (`num_iterations=3`, clip 0.2), `max_decode_ratio` 3→2. Relaunched cer_mt (job 43223) **resuming the epoch-6 warmup ckpt** (skips warmup → only joint epochs 7–10). Built the training-dynamics report → hosted at borrisonxiao.github.io/jsalt26-downsampling (Pages) + Artifact; now an `html-report` skill (+ `training-report` subskill).
- **Notes**: `--debug` confirmed the new loop is correct (finite loss, no OOM); its 208 s/it is unrepresentative (cold decoder → max-length decodes). nll_mt is competitive-with/on the fixed-rate frontier, not yet a clear Pareto win. GRPO vs ms-swift: mine == GRPO at μ=1, std-norm advantage, no KL (dropped by design).

### 2026-08-03 - Report: aligned boundary rows + current models on the frontier chart
- **Status**: Completed
- **Description**: `make_training_report.py` section 3 now renders the spectrogram, gold char-CTC alignment, and learned boundaries as three rows sharing one time axis (previously both boundary sets were only overlaid on the spectrogram, with no dedicated per-source view). Section 4's fixed-rate frontier now plots nll_frozen/nll_mt as labeled points (clean + other test WER) via a new `frontier_fig()`, plus matching highlighted rows in the baselines table.
- **Notes**: Added an optional `facecolor` param to the shared `htmlkit.mpl_png` (default unchanged) so new matplotlib figures with visible axis chrome can opt into an opaque white background — a transparent PNG can't react to the page's light/dark theme, so default-black text/ticks risk going illegible in dark mode. Repushed to Pages (borrisonxiao.github.io/jsalt26-downsampling) and the Claude Artifact. cer_mt (job 43226) still training; add it to the frontier once it has test numbers.

### 2026-08-03 - Report: param counts + arch labels + appendix; re-diagnosed the collapse; dropped Claude Artifact hosting
- **Status**: Completed
- **Description**: Added exact trainable-parameter counts (computed from the real modules/config, not estimated) to `make_training_report.py`: segmenter CNN 1,573,633 / Transformer 3,356,161; decoder trainable (proj + LoRA rank-16 all-linear) 19,128,320 (~1.5% of the frozen 1.24B Llama-3.2-1B-Instruct). Labeled the segmenter backbone (CNN) everywhere the 3 ablation models appear. Added an Appendix section with full architecture tables (CNN/Transformer segmenter, proj+LoRA decoder). Added a "1b · Revisiting the failure" subsection: re-examined the collapse history now that non-collapsing runs exist as a contrast, and found it's actually **two distinct collapse mechanisms**, not one — see `docs/project_notes/decisions.md` ADR-005 for the full analysis. Per explicit user request, stopped using the Claude Artifact for this report (hosting "not working" for the user) — removed the now-unused `report.artifact.html` output and purged Artifact-URL references from personal memory; GitHub Pages is now the only distribution channel.
- **Notes**: LLM+LoRA param counts couldn't be gotten by instantiating the real model (loading the tokenizer hung with no error on this login node — likely a blocked network call despite offline env vars — and even a bare `from_config` random-init of the 1B-param model was too slow on CPU); verified analytically instead from the config values `AutoConfig.from_pretrained(..., local_files_only=True)` actually returned (which was instant) against the LoRA/`AdaptedModel` formula read directly from `speechbrain/nnet/adapters.py`. Segmenter and `proj` counts ARE from real instantiated modules (fast, no issue). Repushed to Pages only (commit pending at time of writing).

### 2026-08-03 - Report: Background section + pipeline figure (drawio + png), publish workflow documented
- **Status**: Completed
- **Description**: Added a "0 · Background" section to `make_training_report.py` (task/data/model tables + the cold-start → warmup → joint-RL curriculum), and a new `make_pipeline_figure.py` that emits the architecture figure as **both** an editable `.drawio` and the PNG embedded in the report from a single layout spec (so they can't drift): forward spine with per-block frame rates/dims, a zoom panel showing 28 frames → 7 audio tokens at ρ=0.25, a learned-vs-fixed-rate comparison at equal ρ, and the 3-stage curriculum. Outputs in `artifacts/segmenter/figures/`. Also removed the manual `<br>` in the boundary-figure captions and the stale `max-width:62/66ch` clamp on `.sub`/`.lead` that was still live on Pages.
- **Notes**: Design follows audio-token-compression figure convention (annotate Hz/dims/token counts/ρ; contrast fixed vs learned segmentation) and NeurIPS method-figure guidance (notation legend rather than semantic colour coding). ❄/⚡ used for frozen/trainable because 🔥 (U+1F525) is absent from DejaVu Sans and renders as tofu in matplotlib. Publish workflow now written into `CLAUDE.md` (regenerate → copy to pages repo → commit → push, without being asked). Deleted the stale `report.artifact.html` — it was retired on 2026-08-03 but had survived on disk and was being kept in sync by mistake.

### 2026-08-03 - Refine pipeline figure palette, labels, and layout
- **Status**: Completed
- **Description**: Simplified architecture labels, made frozen/trainable status explicit, reduced the palette to neutral + three semantic accents (segmentation, decoder adaptation, RL), darkened small captions for contrast, and regenerated the synchronized PNG/draw.io/report outputs. Corrected the report caption to distinguish kept-ratio ρ from compression; published to Pages in commit `31e04bb`.

### 2026-08-03 - Fix pipeline figure bounds and spacing after visual review
- **Status**: Completed
- **Description**: Fixed the PNG emitter shrinking rounded boxes by the corner radius (the cause of apparent overflows), moved the token/frame annotations inward, made boundary peaks uniform, shortened the segmenter label, increased stage-card margins, and removed the detached decorative side bars. Regenerated all outputs and published Pages commit `2292134`.

### 2026-08-04 - Polish pipeline merge, token colors, and curriculum
- **Status**: Completed
- **Description**: Replaced the double-ring prompt/audio merge with one centered `⊕`, made speech-token fills uniform, and redesigned the curriculum as neutral cards with numbered stage markers and compact trainable-component pills. Verified at embedded page width and published Pages commit `0979c33`.

### 2026-08-04 - Replace curriculum cards with a paper-style stage rail
- **Status**: Completed
- **Description**: Rebuilt the prompt/audio junction as a 48 px circled-plus gate with arrows terminating on its boundary. Replaced the curriculum cards with a continuous three-stage rail inspired by NeurIPS VRL3/VPGTrans/AttnDreamBooth figures, using concise objectives and component-update pills; verified at embedded width and published Pages commit `4aea085`.

### 2026-08-04 - Show fewer adaptive tokens and use draw.io's native OR gate
- **Status**: Completed
- **Description**: Changed the comparison panel to “Fewer tokens, smarter placement,” with seven fixed-rate segments versus five learned segments. Switched the editable figure to draw.io's native `shape=or`, mirrored it as integrated vector geometry in the PNG, moved the gate right for balanced attached connectors, and published Pages commit `bf50d52`.

### 2026-08-04 - cer_mt_opt (job 43226) completed: joint RL destabilized, did not beat warmup
- **Status**: Completed (job), inconclusive (research question)
- **Description**: The CER-reward multi-task segmenter ablation (resumed from epoch-6 warmup ckpt, off-policy batched GRPO per ADR-004) ran epochs 7-10 and finished cleanly (no crash), but valid loss/WER got worse every joint epoch vs. the epoch-6 warmup checkpoint, so the checkpointer kept epoch 6 — final test numbers (7.53/11.81 clean/other) are essentially the pre-RL baseline, not a result of the CER-reward joint phase. Contrasts with nll_mt, whose joint phase improved valid WER monotonically (6.53→6.29→6.03).
- **Notes**: Logged in `key_facts.md` (RL segmenter reward/multi-task ablation group). No code change made; flagged as needing a follow-up run (more joint epochs, or check off-policy μ=3 variance) before drawing a CER-vs-NLL conclusion.

### 2026-08-04 - Submit three-seed ablations from one exact warm-up checkpoint
- **Status**: Superseded; incorrect jobs 44730–44738 canceled
- **Description**: The first launch used the later three-step off-policy CER optimization, whose custom `fit_batch` bypassed gradient accumulation and took three full optimizer steps per minibatch. That made it a different, unstable algorithm rather than a multi-seed replication of the successful on-policy nll_mt run.
- **Notes**: Incorrect outputs are retained under `results/speechllm_segmenter/multiseed_shared_warmup/` for audit only.

### 2026-08-04 - Relaunch corrected shared-warm-up multi-seed ablations
- **Status**: Running (production jobs 45030–45038)
- **Description**: Restored the successful NLL on-policy/accumulation-aware path. Replaced CER rollout reuse with a single accumulation-aware plain-GRPO update while retaining batched K-reward decoding; tuned CER to `lr_segmenter=5e-5`, `pg_weight=0.5`, and decode ratio 2. NLL retains the successful `lr_segmenter=1e-4`, `lr_decoder=5e-4`, and decode ratio 3.
- **Notes**: All runs recover the exact same epoch-6 checkpoint. Smoke job 45029 completed but SpeechBrain debug mode reset it to a temporary two-epoch run, so it did not exercise post-warm-up RL; live CER job 45032 directly verified the intended accumulation trace (no steps on minibatches 1–3, one step on minibatch 4). Outputs are under `results/speechllm_segmenter/multiseed_shared_warmup_onpolicy/`.

### 2026-08-04 - Record A100-only GPU policy
- **Status**: Completed
- **Description**: Added the user's permanent cluster restriction to `key_facts.md`: GPU authorization is specifically the inseparable pair `--account=jsalt2026-lgarci27 --partition=a100`; never use another GPU partition with that account or another/default account for GPU work. CPU-only `med` with the JSALT account remains allowed.

### 2026-08-04 - Schedule Transformer-backbone shared-warm-up multi-seed sweep
- **Status**: Submitted (warm-up 45236; production 45237–45245)
- **Description**: Added a runtime shared-warm-up handoff and scheduled the Transformer boundary-policy counterpart of the corrected CNN sweep: one six-epoch shared warm-up followed by nll_frozen/nll_mt/cer_mt at seeds 3407/3408/3409. All jobs use the required `jsalt2026-lgarci27` + `a100` pair.
- **Notes**: Production jobs depend on successful warm-up job 45236 and stage its exact final epoch-6 checkpoint. Outputs use `results/speechllm_segmenter/multiseed_shared_warmup_onpolicy/<variant>/transformer/<seed>/`.

### 2026-08-04 - Audit latest segmenter sweep and update the training report
- **Status**: Completed
- **Description**: Independently recomputed the completed three-seed CNN NLL WERs from raw edit counts, inspected paired per-utterance outcomes and critical checkpoint/evaluation code paths, and added an evidence-labeled progress audit to the self-contained report.
- **Notes**: Confirmed decoder co-training wins all three seeds; identified rare non-EOS insertion tails as the main source of seed variance. Updated `artifacts/segmenter/report.html`; CER and Transformer conclusions remain explicitly pending.
### 2026-08-05 - Redesign report progress and claims blocks

- **Status**: Completed
- **Description**: Replaced the side-by-side half-width audit cards with full-width stacked tables, shortened evidence text, fixed column widths, and added wrapping seed-status and verdict pills.
- **Notes**: Eliminated right overflow in Claims & verdicts and improved Sweep progress scanability; regenerated `artifacts/segmenter/report.html`.

### 2026-08-05 - Fold the latest multi-seed results into the main report sections

- **Status**: Completed
- **Description**: Removed the standalone 0b progress-audit section and moved the verified three-seed mean ± SD NLL results into section 2.
- **Notes**: Updated section 4 to use corrected three-seed NLL means with SD error bars; retained the historical single-seed curves for training-dynamics and CER context.

### 2026-08-05 - Add dynamic-downsampling slides to the final presentation

- **Status**: Completed
- **Description**: Replaced the slide-64 placeholder and inserted four slides covering segmentation choices and the speech-LLM pipeline, the three-stage GRPO curriculum, verified three-seed mean ± SD results, a boundary-alignment example, and future work.
- **Notes**: Corrected the draft outline to the implemented frozen wav2vec2 encoder and boundary-BCE cold start. Preserved the original 80-slide deck as `artifacts/segmenter/slides/JSALT2026 - OmniEnc Final Presentation.before-segmenter-slides.pptx`; the updated 84-slide deck keeps the next presenter section at slide 69.

### 2026-08-05 - Simplify the segmenter presentation section

- **Status**: Completed
- **Description**: Rebuilt the section as seven plain JHU-template slides: two overviews, GRPO, training curriculum, quantitative results versus fixed-rate baselines, a spectrogram/boundary example, and takeaways/future work.
- **Notes**: Supersedes the prior five-slide block layout. The final deck has 86 slides, with the segmenter section at slides 64–70 and the next presenter preserved at slide 71; visual QA included a fresh-eyes review and a fix-and-verify pass.

### 2026-08-05 - Export segmenter slides for image review

- **Status**: Completed
- **Description**: Rendered slides 64–70 as individual PNGs, a two-column contact sheet, and a ZIP archive under `artifacts/segmenter/slides/` for rapid review.

### 2026-08-05 - Reset segmenter slides to plain template typography

- **Status**: Completed
- **Description**: Removed inline bold, accent-colored prose, and oversized body labels from slides 64–70 while retaining the original JHU template title treatment. Kept all editable body text regular-weight and dark, then corrected curriculum-note spacing, spectrogram bottom margin, and results-column wrapping.
- **Notes**: Validated the 86-slide deck and confirmed the next presenter remains at slide 71. Regenerated the seven PNG previews, contact sheet, and ZIP; the prior deck is preserved as `artifacts/segmenter/slides/JSALT2026 - OmniEnc Final Presentation.before-plain-text-reset.pptx`.

### 2026-08-05 - Remove generated slide drafts and restore the original template

- **Status**: Completed
- **Description**: Per user request, discarded all generated segmenter slides, draft PPTX files, previews, review exports, slide-generation scripts, and unpacked work directories. Restored `JSALT2026 - OmniEnc Final Presentation.pptx` to the validated original 80-slide deck with the untouched Cihan placeholder at slide 64.
- **Notes**: Exported only seven standalone figures to `artifacts/segmenter/diagrams_highres/`. The downsampling comparison was regenerated from the source diagram at 320 DPI with intact left padding and no outer top or side rule. Do not recreate presentation slides unless explicitly requested.

### 2026-08-05 - Replace the standalone spectrogram example

- **Status**: Completed
- **Description**: Replaced the first learned-boundary example with held-out utterance `2277-149874-0017` and added its transcript as a dedicated fourth row inside the high-resolution figure.
- **Notes**: Kept the transcript as a readable sentence rather than attempting character-to-boundary alignment in this pass. Output: `artifacts/segmenter/diagrams_highres/07_spectrogram_boundaries.png`.

### 2026-08-05 - Clarify the boundary figure and add one plain GRPO slide

- **Status**: Completed
- **Description**: Removed boundary-row shading and color coding from the spectrogram example, leaving only black gold/learned boundary ticks. Replaced the original slide-64 placeholder with one plain JHU-template GRPO slide containing concise hyperparameters, the group-relative policy/rate/optional-CE objective, and NLL/CER reward equations.
- **Notes**: The GRPO objective includes no entropy or KL term; the slide states entropy = 0 and no KL. The presentation remains 80 slides, and no draft decks or preview exports were retained under `artifacts/segmenter/slides/`.

### 2026-08-05 - Simplify slide 64 to three equations

- **Status**: Completed
- **Description**: Removed the advantage-definition, rate-loss, and joint-loss equations from the GRPO slide. Retained only the main GRPO policy-loss equation and the NLL/CER reward equations.
- **Notes**: Refreshed the validated 80-slide presentation and the user-requested `artifacts/segmenter/slides/slide64_grpo_preview.png` image.

### 2026-08-05 - Standardize GRPO equation typography and notation

- **Status**: Completed
- **Description**: Re-rendered the GRPO, NLL, and CER equations with the same STIX math typeface and 18-point source size, reduced the displayed GRPO equation, and adopted the original GRPO paper's `G` group-size and `i`/`o_i` rollout notation.
- **Notes**: Updated slide 64 to `G = 4` and saved the three copy-ready 600-dpi transparent PNGs under `artifacts/segmenter/equations/`.

### 2026-08-05 - Render all GRPO slide equations as single lines

- **Status**: Completed
- **Description**: Re-rendered the GRPO and NLL formulas as one-line equations; CER was already one line. Preserved the shared STIX typography and `G`/`i`/`o_i` notation.
- **Notes**: Refreshed the three standalone equation PNGs, slide 64, and `slide64_grpo_preview.png`.

### 2026-08-05 - Export a combined equation image

- **Status**: Completed
- **Description**: Vertically stacked the GRPO, NLL, and CER equation assets into one transparent 600-dpi PNG.
- **Notes**: Output: `artifacts/segmenter/equations/all_equations.png`.

### 2026-08-05 - Add notation definitions to the combined equation image

- **Status**: Completed
- **Description**: Added a compact legend defining `G` as the four-rollout group size, `o_i` as the sampled boundary sequence, and `A-hat` as the group-normalized rollout advantage shared across frames.
- **Notes**: Updated `artifacts/segmenter/equations/all_equations.png` in place.

### 2026-08-05 - Add the keep-ratio term to the GRPO objective

- **Status**: Completed
- **Description**: Extended the one-line GRPO objective with `lambda_rho L_rho` and defined `L_rho` as the penalty for mean keep ratio outside `[0.10, 0.30]`, with `lambda_rho = 1`.
- **Notes**: Refreshed the standalone objective, combined equation PNG, slide 64, presentation, and slide preview.

### 2026-08-05 - Correct the spectrogram alignment-row terminology

- **Status**: Completed
- **Description**: Replaced the ambiguous `gold char` row label with `char-level CTC alignment` while preserving all spectrogram pixels and boundary positions.
- **Notes**: Updated `artifacts/segmenter/diagrams_highres/07_spectrogram_boundaries.png` in place.

### 2026-08-05 - Review CER-MT and Transformer-backbone sweep progress

- **Status**: Completed
- **Description**: Reviewed the corrected on-policy multi-seed sweep. CNN `cer_mt` (jobs 45032/45035/45038) completed cleanly on all 3 seeds — logged as `cer_mt_shared_3seed` in `key_facts.md`. Found the Transformer-backbone replication (`submit_abl_multiseed_transformer.sh`) stalled: shared warm-up job 45236 ran all 6 epochs and saved a complete, valid epoch-6 checkpoint (`shared_warmup/transformer/3407/save/CKPT+2026-08-05+02-10-59+00`, all 9 required files present), but then hit a CUDA OOM while evaluating the 3rd of 3 test splits post-training, so the job exited FAILED. All 9 dependent production jobs (45237-45245, `afterok:45236`) are stuck in `squeue` as `PD (DependencyNeverSatisfied)` and will never run as submitted.
- **Notes**: No retraining needed — the checkpoint is complete and usable. Fix is to resubmit the 9 production jobs pointing `WARMUP_SOURCE_DIR`/`--dependency` at the existing checkpoint (bypassing the dead `afterok:45236` gate), e.g. via `run_segmenter_from_shared_warmup.slurm` with no dependency.

### 2026-08-05 - Resubmit stalled Transformer-backbone production jobs

- **Status**: Submitted (training)
- **Description**: Cancelled the 9 dead-end pending jobs (45237-45245, stuck on `afterok` of the FAILED warm-up job) and resubmitted all 9 (`nll_frozen`/`nll_mt`/`cer_mt` × seeds 3407/3408/3409, new jobs 46676-46684) via `run_segmenter_from_shared_warmup.slurm`, staging the existing valid epoch-6 checkpoint directly with no dependency gate.
- **Notes**: All 9 confirmed `PENDING (None)` in `squeue` (queued on resources only, no blocker) immediately after submission. Outputs will land under `results/speechllm_segmenter/multiseed_shared_warmup_onpolicy/<variant>/transformer/<seed>/`.

### 2026-08-05 - Restore spectrogram boundary guides and shorten the CTC label

- **Status**: Completed
- **Description**: Changed the reference-row label to `CTC-char` at a size matched to `learned`, and overlaid the 48 learned boundary positions as thin neutral guides on the spectrogram while retaining its original colormap.
- **Notes**: Updated `artifacts/segmenter/diagrams_highres/07_spectrogram_boundaries.png` in place; the boundary lanes and transcript remain unchanged.

### 2026-08-05 - Restore source-specific colors in the spectrogram figure

- **Status**: Completed
- **Description**: Applied a consistent light-blue/blue mapping to CTC boundaries and light-red/red mapping to learned boundaries in both the lanes and spectrogram overlay. Redrew `CTC-char` and `learned` with identical font settings and changed the transcript sentence to gray.
- **Notes**: Updated `artifacts/segmenter/diagrams_highres/07_spectrogram_boundaries.png` in place; CTC has 55 internal boundary ticks and learned has 48.

### 2026-08-05 - Standardize all spectrogram row labels

- **Status**: Completed
- **Description**: Redrew `spectrogram`, `CTC-char`, `learned`, and `transcript` with identical DejaVu Sans 28-point dark-neutral typography. Kept source colors only on the boundary lanes and spectrogram guides.
- **Notes**: Updated `artifacts/segmenter/diagrams_highres/07_spectrogram_boundaries.png` in place; the transcript sentence remains gray.

### 2026-08-05 - Refine transcript styling and boundary weights

- **Status**: Completed
- **Description**: Re-rendered the transcript sentence at a smaller size in uniform neutral gray, thickened the CTC and learned lane ticks, and widened their matching spectrogram guides.
- **Notes**: Updated `artifacts/segmenter/diagrams_highres/07_spectrogram_boundaries.png` in place while preserving the blue/red boundary-source distinction.

### 2026-08-05 - Finalize transcript text and boundary-guide weight

- **Status**: Completed
- **Description**: Reduced the transcript sentence from 40 to 34 points, changed it to solid black, and increased the CTC/learned lane ticks and spectrogram guides by one additional weight step.
- **Notes**: Updated `artifacts/segmenter/diagrams_highres/07_spectrogram_boundaries.png` in place.

### 2026-08-05 - Emphasize spectrogram row captions

- **Status**: Completed
- **Description**: Redrew `spectrogram`, `CTC-char`, `learned`, and `transcript` in 28-point DejaVu Sans Bold with uniform black coloring.
- **Notes**: Updated `artifacts/segmenter/diagrams_highres/07_spectrogram_boundaries.png` without changing the plot, boundary guides, or transcript sentence.

### 2026-08-05 - Replace bold captions with slightly larger regular text

- **Status**: Completed
- **Description**: Changed all four spectrogram row captions from 28-point bold to 30-point regular DejaVu Sans, retaining uniform black coloring.
- **Notes**: Updated `artifacts/segmenter/diagrams_highres/07_spectrogram_boundaries.png` without altering other figure elements.

### 2026-08-05 - Make frozen-decoder error bars visible

- **Status**: Completed
- **Description**: Redrew the quantitative frontier figure with compact frozen-decoder square markers and explicit darker capped error bars for the reported test-clean ±0.09 and test-other ±0.11 sample SDs.
- **Notes**: Updated `artifacts/segmenter/diagrams_highres/05_results_frontier.png` at its original 3233×1263 resolution.

### 2026-08-05 - Restore right-side padding in the frontier figure

- **Status**: Completed
- **Description**: Increased the export's right margin so the final `0.35` tick label is fully visible rather than clipped at the canvas edge.
- **Notes**: Updated `artifacts/segmenter/diagrams_highres/05_results_frontier.png` while preserving its resolution and all plotted values.

### 2026-08-05 - Standardize segmentation diagrams on draw.io green

- **Status**: Completed
- **Description**: Replaced the teal segmentation palette with draw.io's default green family: `#d5e8d4` fill, `#82b366` accent/stroke, and `#2d7600` dark green text.
- **Notes**: Updated high-resolution diagrams 01–04, the shared figure generator, and `artifacts/segmenter/figures/pipeline.drawio`/`pipeline.png`; all crop dimensions and content were preserved.

### 2026-08-05 - STE feasibility analysis for the segmenter + published analysis report

- **Status**: Completed (analysis); implementation decision open
- **Description**: Investigated whether a straight-through estimator (STE) could carry the decoder CE gradient into the segmenter, revisiting the 2026-08-02 "Gumbel-ST dropped" note. Confirmed the note is correct for the current code (a relaxed boundary into `mean_pool_segments` gives no error and no gradient — it is consumed only as an integer index), but is not a mathematical dead end: rewriting pooling as `S = M X` and applying STE at the membership matrix gives an exact hard forward (1.2e-7 vs `mean_pool_segments`) with live gradients, at ~1 fwd+bwd per step. Built two relaxations (Gaussian index / CIF-style, and survival-product) plus fully-soft variants, and ran controlled synthetic experiments (jobs 47302, 47310).
- **Notes**: **Result: no gradient-based arm corrects a systematic 3-frame boundary misalignment** (mean offset 3.34 → 3.90 STE / 3.14 soft / 3.25 GRPO; supervision reaches 0.00). Soft (CIF-style) forward beats hard STE ~2x on the easier test; the survival-product form collapses reproducibly (ρ=0.008, unrescued by `lambda_cap` 1→20 — the one-segment case has exactly zero gradient, an absorbing state). Root cause is geometric: the set of segmentations is finite/zero-dimensional, so the surrogate can only express per-frame mixing weights, never a choice of partition. Plan + numbers in `plans/ste_differentiable_arm.md`; generator `recipes/LibriSpeech/ASR/transformer/make_ste_report.py` → `artifacts/segmenter/ste_analysis/report.html`, published to https://borrisonxiao.github.io/jsalt26-ste-analysis/ (separate repo/subdir from the training report). Probe code on weka at `users/cxiao/ste_probe/` (must NOT live on login-node `/tmp` — compute nodes can't see it; that silently killed job 47252: instant FAILED, zero-byte output. A separate early `srun` failure was unrelated — it was trying to create a step inside a busy interactive allocation). **Follow-up flagged**: run the decoder-co-trained/segmenter-frozen control before any more estimator work.

### 2026-08-05 - Synchronize the full report diagram and neutralize the token-count caption

- **Status**: Completed
- **Description**: Regenerated the full editable Draw.io/PNG pipeline with the default-green palette and changed `N = 7 audio tokens → ρ = N/T = 0.25 (4× fewer)` to regular neutral `#3f4954` text.
- **Notes**: Refreshed the embedded Pipeline image in `artifacts/segmenter/report.html` without changing report claims or progress data, and updated high-resolution diagram 02 to match.

### 2026-08-06 - Add slide-oriented two-row pipeline copy

- **Status**: Completed
- **Description**: Created a two-page Draw.io derivative that preserves the user-edited full source and adds a larger-text, two-row model pipeline with a compact serpentine flow.
- **Notes**: Output: `artifacts/segmenter/figures/pipeline_slide_copy.drawio`; the slide page has additional top whitespace and reduced bottom whitespace. Preview: `artifacts/segmenter/figures/pipeline_slide_copy_preview.png`.

### 2026-08-06 - Add compact slide-oriented training curriculum

- **Status**: Completed
- **Description**: Added a third Draw.io page containing the three-stage training curriculum with larger typography and a moderately compact 1240×410 layout.
- **Notes**: Preserved the user's existing pages and styling in `artifacts/segmenter/figures/pipeline_slide_copy.drawio`. Preview: `artifacts/segmenter/figures/training_curriculum_slide_preview.png`.

### 2026-08-06 - Submit native-rate no-downsampling baseline

- **Status**: Submitted (training)
- **Description**: Submitted the missing SpeechLLM baseline that passes all 50 Hz encoder frames to the frozen decoder (`boundary_source: none`, identity feature downsampler) for seeds 3407, 3408, and 3409 on one A100 each.
- **Notes**: SLURM jobs 49443, 49444, and 49445 write to `results/speechllm_fixed_pooling/no_downsampling_tc100/<seed>/`. The runs use train-clean-100 for 10 epochs, a 0.75 decode ratio (the same absolute cap as the k=4 baseline), batch-duration limits 50/25, gradient accumulation 20, and evaluation batch size 4 to control native-rate memory while preserving the effective training batch. Launch files: `recipes/LibriSpeech/ASR/transformer/run_no_downsampling.slurm` and `recipes/LibriSpeech/ASR/transformer/submit_no_downsampling_multiseed.sh`.

### 2026-08-06 - Annotate qualitative boundaries with per-utterance WER

- **Status**: Completed
- **Description**: Added compact row-aligned WER badges to the qualitative example: CTC-character boundaries score 10.0% WER (1 substitution), while the learned boundaries score 0.0% WER (exact transcript).
- **Notes**: Consolidated the formerly inline/temporary generation steps into `recipes/LibriSpeech/ASR/transformer/make_qualitative_boundary_figure.py` and regenerated `artifacts/segmenter/diagrams_highres/07_spectrogram_boundaries.png` deterministically from utterance `2277-149874-0017`, the char-boundary target, and the exact epoch-10 seed-3407 CNN checkpoint. The figure retains 55 internal CTC boundaries and 48 learned boundaries at 2904×1234.

### 2026-08-06 - Audit and label the qualitative char-CTC alignment

- **Status**: Completed
- **Description**: Re-ran the torchaudio wav2vec2-base-960h forced alignment for `2277-149874-0017`; the regenerated 167-frame char target is bit-for-bit equal to the saved tensor (zero differing frames), with 55 ordered CTC token spans plus the synthetic frame-0 leading-silence boundary.
- **Notes**: Added `--show-char-intervals` to `make_qualitative_boundary_figure.py` and generated `artifacts/segmenter/diagrams_highres/07_spectrogram_boundaries_char_labels.png`, labeling every char interval (`␣` = separator, `sil` = frame-0 segment). The original presentation figure remains byte-identical.

### 2026-08-07 - Clarify the results-frontier training set and precision

- **Status**: Completed
- **Description**: Added a compact `Trained on LibriSpeech-100h` badge and rounded the trained-model mean±SD annotations to one decimal place.
- **Notes**: Added the reproducible generator `make_results_frontier_figure.py` and regenerated `artifacts/segmenter/diagrams_highres/05_results_frontier.png` at its original 3233×1263 resolution.

### 2026-08-10 - Simplify report wording and label comparison values

- **Status**: Completed
- **Description**: Replaced abstract report labels such as “quality-cost frontier,” “benchmark matrix,” and “controlled attribution” with direct descriptions of the experiments.
- **Notes**: The “Does compression help?” row now names each system beside its WER: no downsampling is 7.05 clean / 10.10 other, while fixed k=5 is 6.16 clean / 10.17 other. Project memory now requires this self-labeled format for comparison tables and discourages unlabeled arrows.

### 2026-08-10 - Clarify segmenter/pooling setups and replicate WavLM policies

- **Status**: Report updated; experiments submitted
- **Description**: Replaced “hybrid learned,” “WavLM learned,” and “boundary features” with explicit segmenter-input and pooled-feature descriptions. Added an experiment-setup table and two result plots to the WER-gap section.
- **Notes**: Submitted missing seeds 3408/3409 for the CNN-segmenter-on-WavLM + WavLM-pooling study initialized from the best char segmenter and decoder. Bernoulli jobs: 63824/63826; first-order AR jobs: 63825/63827. Each uses one A100 and writes under `results/speechllm_segmenter_wavlm/oracleclose_bestinit/`.

### 2026-08-10 - Standardize the boundary-model name as “segmenter”

- **Status**: Completed
- **Description**: Removed “detector” as a synonym for the segmenter from the report and current project notes.
- **Notes**: “Segmenter input” now means the encoder features fed to the segmenter; “features pooled for the decoder” names the separate stream averaged using the segmenter's boundaries.

### 2026-08-10 - Widen the HTML report column

- **Status**: Completed
- **Description**: Increased the desktop report width by 25%, from 960 px to 1200 px, to make the setup and result tables easier to read.
- **Notes**: The existing responsive maximum-width behavior and mobile padding are unchanged.

### 2026-08-10 - Number report assets and loosen plot spacing

- **Status**: Completed
- **Description**: Added sequential labels to every table and figure, reduced the desktop report width to 1100 px, and added both page-level and Matplotlib padding before plots.
- **Notes**: Moved WER value annotations past the right error-bar caps so the numbers no longer overlap the bars.

### 2026-08-11 - Add char-CTC boundary agreement plot

- **Status**: Completed and published
- **Description**: Evaluated every setup in Figure 2 against char-CTC boundaries on all 2,703 dev-clean utterances and inserted a two-panel exact/±1-frame boundary-F1 plot directly below Figure 2.
- **Notes**: Learned rows use the same WER-selected checkpoints and show three-seed sample-SD bars; deterministic char, phone, fixed-k=5, and no-downsampling streams use square markers. Added a reproducible audit script and the aggregate JSON artifact. Published as Pages commit `190f143`.

### 2026-08-11 - Launch full-prefix Transformer autoregressive policy

- **Status**: Submitted
- **Description**: Implemented the planned causal boundary-history Transformer policy and launched a WavLM three-seed NLL-MT study. The model conditions each frame decision on the entire preceding boundary-label history, uses cached closed-loop sampling, and re-scores all sampled histories in one parallel causal pass for GRPO.
- **Notes**: CPU correctness tests cover causality, history dependence, cached/parallel equivalence, frame-0 and padding masks, policy-gradient direction, and checkpoint recovery. A100 cold-start smoke job 64593 and joint-RL smoke job 64594 both completed successfully. Production dependency chain: cold start 64597 → shared warm-up 64598 → seeds 3407/3408/3409 in 64599/64600/64601. Each job uses one A100; the three final seeds can run concurrently. Outputs: `results/speechllm_segmenter_wavlm/fullprefix_transformer_ar/`.

### 2026-08-11 - Launch best-char-decoder Transformer-AR follow-up

- **Status**: Submitted
- **Description**: Added a matched three-seed full-prefix Transformer-AR study initialized from the best WavLM oracle-char decoder.
- **Notes**: Corrected jobs 64605/64606/64607 (seeds 3407/3408/3409) depend on causal-policy cold-start job 64597. Each independently runs two decoder-adaptation epochs followed by 10 NLL joint-RL epochs, matching the strongest learned WavLM first-order-AR budget, with decoder/segmenter LRs 2e-4/5e-5 and rate band `[0.15, 0.25]`. The originally submitted 15-RL-epoch jobs 64602/64603/64604 were cancelled while dependency-pending and consumed zero runtime. Outputs: `results/speechllm_segmenter_wavlm/fullprefix_transformer_ar_best_char_decoder/nll_mt/`.

### 2026-08-11 - Highlight the WavLM CNN first-order AR configuration

- **Status**: Completed and published
- **Description**: Revised the training-dynamics report so the WavLM-input CNN segmenter with the best char decoder and first-order AR is the highlighted learned system, replacing the wav2vec2-input hybrid in the top result tiles and controlled-results table.
- **Notes**: The report shows 5.06±0.18 / 9.55±0.17 WER at ρ=0.234 and describes the highlighted configuration directly: WavLM features, CNN segmenter, first-order AR, and the best char decoder. Char alignment is labeled as an oracle reference. The report also clarifies that wav2vec2 supplies cold-start labels only; boundary prediction and pooling both use WavLM at inference. Preserved the newer multi-page Pages layout and published the standalone report plus dashboard update as commit `4d424f5`; follow-up commit `e365a79` versions the wrapper iframe URL to prevent the previously cached top panel from persisting. Commit `12f0115` removes the result-status label “deployable” and names the highlighted system by its configuration instead.

### 2026-08-11 - Bilevel segmenter–decoder pilot and analysis

- **Status**: Completed (retrospective pilot); prospective lookahead test proposed
- **Description**: Mapped the current joint-RL loop to a bilevel objective in which
  the decoder adapts on support speech and the segmenter is judged on held-out query
  speech. Audited all ten RL epochs from six matched WavLM Bernoulli/first-order-AR
  runs. Training decoder CE and recorded reward select epoch 12 in all six runs,
  while dev-clean WER selects epochs 7–9; selecting minimum training CE incurs
  0.42±0.22 WER points of dev-clean regret.
- **Notes**: The mean within-run dev kept-ratio span is only 0.016±0.010, making
  simple rate drift an unlikely full explanation. This is evidence of an inner/outer
  mismatch, not evidence that a bilevel optimizer is already effective. Recommended
  next comparison: current same-batch reward versus held-out current-decoder reward
  versus held-out post-one-step decoder reward. Reproducible outputs are under
  `artifacts/segmenter/bilevel_pilot/`; analysis generator is
  `recipes/LibriSpeech/ASR/transformer/make_bilevel_analysis.py`.

### 2026-08-12 - Benchmark fixed-window-64 attention and rollout batching

- **Status**: Completed (benchmark-only implementation)
- **Description**: Compared fused masked SDPA, forced Flash, and compiled FlexAttention,
  then ran six matched 30-real-batch joint-RL timing jobs at window 64 and `K=4`.
- **Notes**: Flash cannot accept the local mask; Flex helps isolated long sequences but
  not end-to-end training. Combined greedy/sample rollout and NLL-reward batching gives
  the strongest result: 101 s → 52 s training-loop time. The optimized runtime is
  environment-gated and uses separate output paths, so production jobs are unchanged.

### 2026-08-12 - Promote and validate combined Transformer-AR rollout batching

- **Status**: Implemented; full-loop comparison running
- **Description**: Added the opt-in production mode `combined_on_policy`. It advances
  one greedy and four sampled histories in a single cached frame loop, batches all
  four detached NLL rewards, and keeps the existing on-policy GRPO re-score,
  accumulation schedule, and optimizer update.
- **Notes**: All segmenter self-tests pass, including exact matched-seed stochastic
  histories and RNG state on CPU. Production smoke 70130 passed train, validation,
  checkpoint reload, and test. Matched full one-epoch jobs 70131/70132 run on separate
  ga129 A100s and exclude ga132. An HTML implementation note and reproducible generator
  are at `artifacts/segmenter/transformer_ar_combined_rollout.html` and
  `recipes/LibriSpeech/ASR/transformer/make_transformer_ar_combined_rollout_report.py`.

### 2026-08-14 - Launch optimized Transformer-AR with BiGRU segment pooling

- **Status**: Completed
- **Description**: Started a matched three-seed WavLM study combining the local-64
  full-prefix Transformer-AR policy with a one-layer BiGRU residual segment pooler.
- **Notes**: The reported cohort is seeds 3407/3409/3411 (jobs 72012/72014/88597):
  4.89±0.08 test-clean and 9.43±0.80 test-other WER at 10.3±0.5 / 10.0±0.4 Hz.
  Runs use one A100 and 12 CPUs each, preallocated KV, combined on-policy rollout/reward
  batching, `K=4`, the best character decoder, two pooler/decoder adaptation epochs,
  and ten joint-RL epochs. Zero residual initialization exactly preserves the initial
  mean-pooled decoder inputs. Outputs are under
  `results/speechllm_segmenter_wavlm/fullprefix_transformer_ar_local64_bigru_best_char_decoder/nll_mt/`.

### 2026-08-17 - Document BiGRU pooling and standardize reports on Hz

- **Status**: Completed and published
- **Description**: Added an illustrated BiGRU pooling section to the main training report. It shows the ordered-frame BiGRU residual branch, zero-initialized projection, exact mean-pooling initialization, two decoder/pooler adaptation epochs, and the following ten joint-RL epochs. Converted every reader-facing compression value and plot from kept ratio to decoder-side audio-token frequency in Hz.
- **Notes**: The source encoders emit 50 frames/s, so reports use `f_audio = 50 Hz × rho`; lower Hz means stronger compression. Internal configuration names and checkpoint fields still use `rho` for compatibility. Validated all nine main-site pages and the STE analysis page. Published the main report site as commit `f3315d1` and the STE analysis as commit `cc2cea1`.

### 2026-08-24 - Port the priority recovery set to CLSP

- **Status**: Initial static priority transfer completed; live corrected LS960
  follow-up pending
- **Description**: Ranked the project assets by recovery value and copied the
  source/history, project records and artifacts, dedicated environment, archived
  boundary/alignment supervision, selected offline backbones, completed LS960
  family, strongest corrected 100h anchors, and broad three-seed sweep to
  `ctl2:/export/jsalt26/omnienc/users/cxiao/skipjack/jointllm`.
- **Notes**: The destination uses about 49 GiB and has about 30 TiB free. All
  transfers were sequential, low-priority, resumable rsyncs capped at 25 MiB/s.
  Verification found no partial directories or dangling HF links; all six data
  archives passed `gzip -t`. The venv is preserved but not directly runnable
  because CLSP lacks `/usr/bin/python3.11`; the copied `uv` executable runs and
  should be used for a clean rebuild. The corrected step-24k LS960 tree was still
  writing at 16:42 EDT and is guarded behind `CONFIRM_QUIESCENT=1` in
  `tools/port_jointllm_to_clsp.sh`.

### 2026-08-29 - Multi-task task-conditioned ST recipe: data, implementation, tests
- Established that MuST-C cannot be fetched automatically and switched the paired
  ASR/ST corpus to CoVoST 2 En-De (ADR-024); downloaded 13.8 GB of parquet and
  built `covost2_prepare.py` (parquet -> 16 kHz FLAC + paired manifests).
- Added the multi-task path as a separate trainer subclassing `SegmenterASR`
  (ADR-025): `train_speechllm_multitask.py`, `hparams/speechllm_multitask.yaml`,
  `multitask_data.py` (task schema, merged manifests, task-homogeneous
  `MultiSourceDynamicBatchSampler`), `multitask_modules.py` (`TaskRoutedLoRA`,
  FiLM expansion, ASR-LoRA warm-start copy).
- Only shared-file change: `segmenter.py` gained `active_task_id` plus a
  `_resolve_task` fallback, both no-ops for single-task ASR runs.
- Tests: 22 new unit tests (`test_multitask_data.py`,
  `test_multitask_equivalence.py`) plus the existing sampler/bilevel/pooling/
  decode suites, all passing. On-GPU equivalence gate
  (`verify_multitask_warm_start.py`, job 171358) passes against the real
  seed-3408 checkpoint.
- Wrote `analyze_multitask_boundaries.py` for the post-run same-audio analysis:
  ASR-vs-ST boundary Jaccard and tolerance-F1 against a repeat null and a
  shifted-pairing null, per-task f_audio and segment durations, and the FiLM
  gradient-cosine conflict diagnostic.
- Two bugs found by the smoke test before they could reach the flagship: the
  sacrebleu 2.6 signature API and a `ZeroDivisionError` on translation-only
  evaluation splits (both logged in bugs.md). A third, the GRPO K-fold batch
  replication breaking per-example task conditioning, was fixed by making
  `TaskFiLM` accept a scalar task id. A fourth, SLURM `--export` splitting the
  LS960 split list on its commas, was caught by dry-running the launcher.
- `test_multitask_boundary_analysis.py` pins the same-audio statistics
  (Jaccard, tolerance-F1, segment durations, padding exclusion) without needing
  a checkpoint. It caught `_summary` using `torch.median`, which returns the
  lower middle value on an even count and so disagreed with the interpolated
  p10/p90 printed beside it; now `quantile(0.5)`.
- Flagship submitted as job 171461 (seed 3408). It queued behind the account's
  three-GPU cap held by the phone-CTC runs, but outranks the pending eval
  arrays, so it takes the first free slot.
- Flagship completed: job 171461 trained all 15000 steps (marked FAILED only
  because final evaluation hit two over-long CoVoST test clips); job 172481
  reran evaluation from the saved step-13500 checkpoint. Results and bottleneck
  analysis are in key_facts.md under the task-conditioned multi-task group.


## 2026-09-01 — Non-ASR multi-task plan: CLSP adaptation and first code

- Audited `survey/2026-08-31/projects/non_asr_multitask_plan.md` against this
  cluster. Only genuinely broken item was the launch block, which still named
  skipjack (`--account=jsalt2026-lgarci27`, `--partition=a100`,
  `--exclude=ga129`); the multi-task launchers themselves were already rehomed.
- Data turned out better than the plan assumed: CREMA-D
  (`/export/corpora6/CREMA-D`, 7,442 clips, 5.26 h) and Fluent Speech Commands
  (`/export/corpora6/fluent_speech_commands_dataset`, official 23,132/3,118/3,793
  split verified speaker-disjoint, 19.25 h) are both already on the cluster at
  16 kHz mono. The plan's TFDS route was unrunnable anyway — `envs/jointllm` has
  no `tensorflow`/`tensorflow-datasets` (nor `datasets`, `peft`, `librosa`).
- Still to stage: LibriCount10/HEAR, Dynamic-SUPERB, Stress-17K/StressTest.
  Compute nodes do have outbound network (verified from `c25`), but jobs run
  `HF_HUB_OFFLINE=1`, so stage from the login node.
- Patched the plan and the executive summary: CREMA-D/FSC data sections, the
  label contradiction (ADR-031), the Gate-0 threshold, the CREMA-D replay factor
  (~50 passes over 3.31 h), and the CLSP launch flags.
- Code landed: `TASK_SPECS` grown to five tasks with candidate label sets,
  `validate_task_table` startup guard, `segmenter_num_tasks: 5` (ADR-032), and
  `candidate_scoring.py` wired into per-task validation (ADR-033).
- Tests: 15 new in `test_candidate_scoring.py`, 3 new in `test_multitask_data.py`,
  2 new in `test_multitask_equivalence.py`. Full suite 848 passed / 5 skipped.
  Two of the new tests caught real defects while being written — `chunk_size=0`
  was silently treated as "no chunking", and a per-position fake decoder proved
  the audio-sensitivity test needed a prefix-mixing stand-in for attention.
- CREMA-D and FSC manifest prep written and run (`cremad_prepare.py`,
  `fsc_prepare.py`) into `datasets/nonasr_manifests`. Each corpus gets a task
  view and a paired ASR view sharing `utt_key`, plus a JSON provenance sidecar;
  CREMA-D writes both the voice and acted labellings so ADR-031 is reversible by
  changing a source path. 13 new tests; suite now 861 passed / 5 skipped.
- Two things the prep surfaced: the distributed CREMA-D summary table names
  `1040_ITH_SAD_XX` while the file is `1040_ITH_SAD_X.wav` (aliased explicitly,
  so any other missing file stays a hard error), and the acted-label split lands
  within 4 clips of the TFDS counts, confirming the 63/9/19 speaker cut.
- Verified end-to-end against the real Llama tokenizer, not just in unit tests:
  merged manifest loads, audio reads, all 41,988 gold labels resolve inside
  their candidate sets, batches are task homogeneous. Recorded the
  `num_bucket`/batch-fill constraint in key_facts.md.
- Speaker-count mixer built (`speaker_count_prepare.py` + `speaker_count_mixing.py`):
  24,000 five-second mixtures as deterministic recipes rather than audio, wired
  into the dataio pipeline through a `mix:` path scheme. 22 new tests; suite now
  883 passed / 5 skipped. All four tasks verified loading through one merged
  manifest end to end.
- The build-time label assertion earned its keep immediately: the first placement
  let a source shorter than its planned span stop covering the peak window, so a
  4-speaker mixture realized only 3 concurrent speakers. Placement now draws each
  onset no earlier than `peak_end - src_dur`; 0 of 2,000 recipes disagree.
- Measured the energy confound the plan warns about instead of assuming the
  short-peak rule handled it — it did not (RMS alone: 48.3% vs 20% chance). Added
  speech-level normalization before the noise floor; counts 1-4 fell from 43.7%
  to 29.0% against 25% chance. Recorded in key_facts.md.
- Wired the GRPO reward for classification batches to the candidate margin
  (ADR-036). Added `classification_reward: margin | nll` and extracted
  `_rollout_quality_reward` from the K-rollout loop so the multitask trainer can
  override it without forking `compute_forward`; the extracted body is the
  previous inline branch verbatim, so the ASR path is unchanged. 7 new tests;
  suite now 890 passed / 5 skipped.
- Rate-term consistency fix + collapse logging + per-task bands (ADR-037/038).
  `expected_kept_ratio` and `realized_kept_ratio` now both measure what pooling
  actually produces; `reward_std_mean`/`zero_variance_share` are logged every
  step and per task; `task_rate_bands` holds each task to its own band from the
  Phase-C unfreeze phase only, so Phase B stays matched-rate. GRPO loss shape,
  normalization and K=4 deliberately unchanged. 13 new tests; suite 903 passed.
- Band priors (a prior, not a measurement): asr/st 6-9 Hz, intent 4-7 Hz,
  speaker_count 3-6 Hz, emotion 2-5 Hz. ASR moved down from 7.5-12.5 Hz to push
  below the ~10.5 Hz the LS960 runs realize.
- Built `hparams/speechllm_multitask_nonasr.yaml` + `submit_multitask_nonasr.sh`
  (SMOKE=1 mode included) and launched the first integrated non-ASR run,
  job 1780790, seed 3408, into
  `results/speechllm_multitask_ls960_nonasr/3408`. Mixture 0.40 LS960 ASR /
  0.20 emotion / 0.20 speaker_count / 0.20 intent; 15000 steps
  (bridge 3000 / film 6000 / unfreeze 6000); ~15 h estimated, 1 A100.
- Two smoke jobs first. 1780786 validated startup and training but revealed the
  reward-std logging had landed in `_obj_joint`, which the production
  `combined_on_policy` path never executes; fixed across all three fit-batch
  variants. 1780788 (COMPLETED, 11:30) confirmed the fix and exercised the full
  test pass on 48-utterance subsets -- a deliberate change, since the flagship
  previously failed at final evaluation after a complete training run.
- First real diagnostic reading: `zero_variance_share = 0.222` at step 18, i.e.
  22% of utterances had identical rewards across all K=4 rollouts and so
  contributed exactly zero policy gradient. Not collapse (that is 1.0), but a
  fifth of the batch is dead weight and this was invisible before today.
  Suspected mechanism: near-silent audio (count-0 mixtures are 20% of the
  speaker-count task) drives boundary probabilities to ~0, so all four rollouts
  sample the identical all-zero sequence.
- Warm start verified intact through the smoke: `f_audio_hz_asr` held at
  10.45-10.47 Hz and LibriSpeech test WER 3.62 / CER 1.03 on a 48-utterance
  subset.
- **Gap found, not fixed**: the plan requires CREMA-D batches balanced by class,
  and nothing implements it -- the sampler balances by source, not by class
  within a source. The shipped manifest is 55.3% neutral. This is why the first
  run keeps `classification_reward: margin`: on an unbalanced prior, -NLL of the
  gold label is satisfiable by a policy that predicts the majority class.
- NA-1 complete (job 1780836, seed 3408, 8h41m on 1 A100): first integrated
  non-ASR multi-task run. Results and five analysis notes logged in key_facts.md.
  Headline is a design flaw rather than a metric: ASR-WER-only checkpoint
  selection picks step 1500 (pre-policy), so every Phase-C compression result is
  in a checkpoint that is never reported. Reward-arm seeds 3407/3409 still
  running; aux arm queued.

### 2026-09-02 - Set up the ICASSP 2027 paper draft

- **Status**: Completed
- **Description**: Downloaded and extracted the official ICASSP 2027 paper kit
  into `drafts/icassp`, retained the source ZIP and checksum, and added a minimal
  `main.tex`, bibliography, Makefile, and requirement/provenance notes.
- **Notes**: Created a dedicated conda environment at
  `/export/jsalt26/omnienc/users/cxiao/envs/icassp-tex` with Tectonic 0.17 and
  Poppler 26.07. The working scaffold and untouched official example both
  compile; the PDF is US Letter with all fonts embedded/subset, and page 1
  renders to PNG through the checked-in Makefile.

### 2026-09-02 - LibriCount-style placement mode + silence zero-class in the speaker-count mixer

- **Status**: Completed -- code landed, 32/32 mixer tests pass, real-pool generation smoke verified
- **Description**: Extended `speaker_count_prepare.py` / `speaker_count_mixing.py` so the synthetic mixtures can track the external test distribution: `placement: random_offset` draws LibriCount-style independent onsets (each source spans >=3 s of the 5 s clip), accepted by rejection sampling only when realized concurrency reaches the requested count, with the realized full-overlap interval stored in `peak_interval` via a new `full_overlap_interval` helper. `zero_style: noise|silence|mixed` adds a -48 dBFS near-silence floor matching the measured HEAR LibriCount count-0 level (default mixed).
- **Notes**: Motivated by a measured comparison against the staged Dynamic-SUPERB packs (`evals/dynamic-superb/data/`): LibriCount folds are 48 kHz / 5 s / labels 0-10 / flat RMS vs count / silent zeros; LibriTTS pack is 24 kHz / 2-13 s / monotonic energy confound. Placement and zero-style are per-run CLI flags; existing peak-mode behavior and recipes are unchanged. Probe audio extracted during the audit lives in `datasets/spkcount_probe_tmp/` (1,200 WAVs). See `key_facts.md`.

### 2026-09-09 - Refine the ICASSP Method section

- **Status**: Completed
- **Description**: Reframed the Method overview around the hard-boundary gradient problem; generalized the encoder rate; made the segmenter and gradient paths explicit; added a concise three-stage optimization curriculum; refined the mean-pooling claim; and expanded/cited BiGRU and GRPO.
- **Notes**: Removed hard-wrapped Method prose, retained abstract plus Introduction on page 1, rebuilt the five-page draft with page 5 containing references only, and recorded the durable manuscript rules in ADR-041.

### 2026-09-09 - Expand ICASSP 100h results and rewrite efficiency

- **Status**: Completed
- **Description**: Removed Metrics, added a generated ten-configuration 100h WER/rate figure and the missing fixed-mean control, separated pooling/policy/allocation findings, and shortened efficiency around net batched gains. Tightened Introduction/Method and updated abstract/conclusion to make room; corrected accumulation and two SD-rounding overrides from primary evidence.
- **Notes**: Rebuilt `drafts/icassp/main.pdf` and all five page PNGs; inspected every page. Technical content ends on page 4, page 5 contains references only, all fonts are embedded/subset, and no overfull boxes or unresolved references remain. Figure generation checks three seeds, complete test sets, consistent decoding, and WER-file/benchmark agreement. ADR-043 and `drafts/icassp/research/revision_2026-09-09.md` record the choices and sources.

### 2026-09-09 - Refine ICASSP rate presentation, figure spacing, and closing message

- **Status**: Completed
- **Description**: Replaced the merged oracle-rate cell with split-specific measurements; highlighted lower Transformer/BiGRU rates; separated crowded figure markers with overview/detail panels; added phone-boundary F1 results; shortened setup; emphasized adaptivity in the abstract; and expanded the conclusion around task-specific policies.
- **Notes**: Regenerated the table, figure PDF/PNG/evidence, manuscript PDF, and five page previews. The initial expanded layout overflowed; shorter captions and prose restored four technical pages and a references-only fifth page without reducing manuscript fonts. All pages were inspected, and numerical provenance is recorded in ADR-044 and `drafts/icassp/research/revision_2026-09-09_round2.md`. Preview generation now clears stale page images before rendering.

### 2026-09-09 - Assess Table 1 comparability and experiment necessity

- **Status**: Review completed; recommendations awaiting adoption
- **Description**: Audited pooling, initialization, training budgets, and checkpoint selection; verified completed stronger LS960 phone-CTC results; drafted a clearer reference/control table framing and prioritized a matched 100h phone-CTC + BiGRU control.
- **Notes**: The verified two-pass reference is 3.32±0.09 / 6.49±0.06 WER, compared with the manuscript's locked one-pass 4.05/7.61. Audit and primary-evidence JSON are in `drafts/icassp/research/table1_comparability*2026-09-09.*`. No training jobs or manuscript/result-publication changes were made during this advisory review.


### 2026-09-09 - Write fresh BiGRU control execution plan

- **Status**: Plan completed; destination-cluster execution pending
- **Description**: Wrote `plans/icassp_bigru_controls_100h_2026-09-09.md` for phone-CTC and character-oracle BiGRU controls. Revised both to fresh seven-epoch, three-seed 100h runs at the author's request, removing prior phone-run state transfer and unnecessary policy initialization.
- **Notes**: Includes a portable command template, minimal shared inputs, 2/3/6-GPU schedules, matching constraints, smoke checks, and result-return requirements. Bash syntax and trainer/config flags validated. No GPU jobs, manuscript edits, or publication actions were performed for this plan. ADR-045 records the retained oracle cap and fresh-run decision.


### 2026-09-09 - Audit ICASSP citations and refine source attribution

- **Status**: Completed
- **Description**: Checked all 24 original references for title, author order, year, venue, and DOI/arXiv identity; corrected AdaTS and Llama attribution; clarified the GRPO variant; and added foundational and closely related citations.
- **Notes**: Wrote a reference-by-reference audit and dated metadata JSON under drafts/icassp/research/citation_audit_2026-09-09.*. Updated the earlier AdaTS literature notes. Rebuilt and inspected the five-page manuscript and all five PNG previews; verified all 21 cited reference hyperlinks, font embedding/subsetting, and absence of undefined references or overfull boxes. Live OpenReview access limits and the alternative primary evidence used are documented. No experiments, result values, or publication actions were changed.

### 2026-09-09 - Package the current ICASSP draft for Overleaf

- **Status**: Completed
- **Description**: Created drafts/icassp/overleaf/ASSET_ICASSP_2027_overleaf_2026-09-09.zip with all compile dependencies, the editable main diagram, a reference PDF, instructions, and SHA-256 manifest.
- **Notes**: ZIP integrity and file hashes verified. An isolated extraction compiled with cached Tectonic packages to five pages with text/layout identical to the current PDF, all fonts embedded/subset, and all 21 reference-title links. Overleaf instructions specify main.tex and XeLaTeX; the service itself was not accessed. No manuscript content changed.


### 2026-09-09 - Create an editable Figure 1 beta

- **Status**: Completed
- **Description**: Forked the current draw.io source to `JSALT26_beta.drawio.xml`; reorganized the interface and training paths with semantic colors/shapes, grouped frame illustrations, distinct frozen LLM and LoRA blocks, and explicit projection/gradient routing.
- **Notes**: Exported vector PDF/SVG and a PNG preview from the beta XML, inspected the rendered figure, validated native IDs/connections and 17 editable groups, verified the original source hash, and checked label fit/font embedding. Added `make figure-beta` for rendering future manual edits without regenerating the source. The manuscript retains its current Figure 1; design notes are in `drafts/icassp/figures/JSALT26_beta.md`.

### 2026-09-10 - Make ICASSP GRPO wording consistent

- **Status**: Completed
- **Description**: Used GRPO consistently across abstract, introduction, overview caption, Method, and training; removed the REINFORCE label/citation and variant framing; specified zero reference-policy KL regularization and renamed the separate policy-loss weight to alpha. Marked the rejected beta figure as archived.
- **Notes**: Rebuilt the PDF and all five PNG previews, checked the source diff and rendered terminology, and inspected all pages. The draft remains five pages with all technical content on pages 1–4 and 20 cited references; all fonts are embedded/subset, with no overfull boxes or unresolved references. No training code or results changed. ADR-048 records the superseding terminology decision.


### 2026-09-10 — Collaborator-requested segment-unit and pause analysis

- Preserved the original paper's key sources/active figures in local git checkpoint `e312a2677`; origin denied write access and auto-review blocked the alternate asset-dev destination pending explicit approval.
- Completed a preselected 24-utterance/three-LS960-seed inference pilot and saved exact evidence, annotated plots for every utterance, and paired bootstrap intervals. Found stronger phone-transition agreement but mostly merged inter-word pauses, not isolated silence tokens.
- Revised the abstract and boundary-results paragraph to state only supported pilot findings. Preserved GRPO terminology and existing WER results; detailed limitations and full 100h historical evidence remain available separately.
- Validation: five timing/overlap unit tests pass; ordered matching agrees with an independent dynamic-programming optimum on 1,000 randomized cases. Final Tectonic PDF is five pages, technical content on 1–4 and references only on 5, all fonts embedded; abstract 115 words.


### 2026-09-10 — Restore original paper and investigate silence interpretation

- Restored the two changed paper files byte-for-byte to `e312a2677` and rebuilt the PDF at the author's request. Retained analysis/code; no remote push.
- Reanalyzed saved pilot cuts and character-CTC supervision with signed edge errors and a same-segment endpoint-tolerance test. Initialization targets already show essentially the same pause-attachment pattern.
- Prepared `plans/segment_interpretation_followup_2026-09-10.md` with mechanism hypotheses, matched controls, held-out corpus evaluation, decoder interventions, and resource estimates. Larger experiments remain proposed.


### 2026-09-10 — Phone-pattern audit and manuscript wording

Completed the author's requested deeper segmentation analysis on 1,709 utterances, corrected native TIMIT duration handling, retained auditable cuts/controls, and revised abstract/body to emphasize flexible allocation within/across phones. Paper and analysis figures built and inspected; no push. Evidence: drafts/icassp/research/phone_patterns_revision_2026-09-10.md and artifacts/segmenter/ls960_phone_patterns_2026-09-10/README.md.


### 2026-09-11 — Apply the selected abstract oracle comparison

- Updated one abstract sentence to retain 2.79/5.63% WER and state comparable performance to a transcript-assisted character-boundary oracle with approximately 25% fewer audio tokens. Verified the percentage against Table 1; saved the exact source diff and recorded ADR-053.
- Rebuilt and inspected the PDF: 150-word abstract, four technical pages plus one reference page, all fonts embedded/subset, no overfull boxes or unresolved references. Pages 2–5 are pixel-identical to the preceding version. No push.


### 2026-09-11 — Clarify conditional Bernoulli sampling in Section 3.1

- Checked the Transformer and CNN AR samplers and the production LS960 configuration. Both use binary conditional Bernoulli distributions; the AR probabilities depend on previous boundary actions. Made that conditioning explicit and distinguished GRPO sampling from greedy inference.
- Verified edits are confined to Section 3.1 relative to the source at turn start. Rebuilt with a writable temporary TeX cache; inspected changed pages 1–2, with pages 3–5 identical to the previous PDF. Five pages, all fonts embedded/subset, no overfull boxes or unresolved references. No code changes or push.


### 2026-09-11 — Clarify CE/GRPO gradient routing and the policy-loss weight

- Traced the production backward pass and audited the 21 learned-run training configurations used by the manuscript; every `pg_weight` is 1.0. Revised only the final paragraph of Section 3.3 to explain the disjoint gradient paths and unit weights; recorded ADR-054 and source/configuration evidence.
- Rebuilt and visually checked page 3; all other page previews are unchanged. Five pages, technical content on pages 1–4, references on page 5, all fonts embedded/subset, no overfull boxes or unresolved references. No training changes or push.


### 2026-09-11 — Replace vague “interface” terminology

- Checked terminology in related speech papers and replaced the three manuscript occurrences with acoustic sequence length, adaptive segmentation, and adaptive downsampling. Updated Figure 1(a)'s editable label and added a vector title overlay that preserves the manual drawing.
- Rebuilt and inspected pages 1, 2, and 4; pages 3 and 5 are identical to the preceding PDF. Five pages, 25 embedded/subset fonts, no overfull boxes or unresolved references. Abstract, results, and manual PNG unchanged. ADR-055 and the exact source diff record the changes; no push.


### 2026-09-12 — Completed non-ASR boundary experiments; manuscript proposal only

- Audited 54 existing single-task TEST records and ran new shared-audio inference, MFA phone/word alignment, Praat voicing diagnostics, count-matched random/confidence controls and uncertainty/sensitivity analyses on 2216 utterances. Found selective coarsening around shared acoustic boundaries, with vowel/voicing preferences; no basis for a new linguistic-unit label or unconstrained optimum-rate claim.
- Resolved alignment dictionary gaps and identified stale selected-checkpoint phase labels. Produced local review.md, paper_revision_proposal.md, generated emotion tables, six scientific PDF/PNG figures, provenance and rerun instructions under artifacts/segmenter/nonasr_boundary_audit_2026-09-12/. ADR-056 records scope and interpretation.
- The author requested review before changes: all 54 manuscript source/bibliography/figure/PDF snapshots remain unchanged. No push or HTML report update.


### 2026-09-13 — Published non-ASR boundary analysis sub-tab
- Completed Analysis 03 with all task/boundary results, controls, silence checks, six figures, downloadable evidence and the unapplied paper proposal: https://borrisonxiao.github.io/jsalt26-downsampling/research/nonasr-task-boundaries.html.
- Validated desktop/mobile/light/dark, navigation, labels, values, local/live files and Pages workflow; published bef0f59. Cumulative report claims calibrated, its numbers preserved, manuscript unchanged.

## 2026-09-13 — Expresso policy-pattern follow-up

Completed same-audio directional-retention and non-nesting analysis on420 utterances,360 same-speaker/text rendition pairs with alignment/rate controls,480 controlled acoustic probes on six policies, and48 batch-sensitivity repeats. Prepared eight figures (two interactive explorers) and evidence downloads for the existing Analysis03 URL. Recognition accuracy is supporting material. No manuscript edits. Published source66ae664c1 and Pages45ceb4e; deployment34781281482 succeeded. Final byte-level live verification is recorded under html_checks/live_validation.json.
