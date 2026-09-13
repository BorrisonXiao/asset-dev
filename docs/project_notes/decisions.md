# Architectural Decisions

This file logs architectural decisions (ADRs).

## Format

Each decision should include:
- Date and ADR number
- Context (why the decision was needed)
- Decision (what was chosen)
- Alternatives considered
- Consequences (trade-offs, implications)

## Entries

### ADR-001: Dedicated per-project venv instead of the shared jsalt26 environment (2026-07-22)

**Context:**
- Needed a Python env to run this repo's training/eval scripts on the bluecrab cluster.
- A pre-existing shared env (`~/cxiao/envs/jsalt26`, Python 3.11) already exists but has no project dependencies installed and its scope/ownership across other projects is unclear.

**Decision:**
- Created a dedicated `uv`-managed venv at `~/cxiao/envs/jointllm` (Python 3.11, matching this repo's own documented target), installed via the repo's own README steps (`requirements.txt`, editable install, `accelerate`, `h5py`).

**Alternatives Considered:**
- Reuse `~/cxiao/envs/jsalt26` → Rejected: unclear what else depends on it; a dedicated env avoids cross-project dependency conflicts (same reasoning already used for the sibling `disentangle` project's own env).

**Consequences:**
- Clean, reproducible, this-repo-only dependency set.
- One more env to maintain, but isolated from other projects' churn.

### ADR-002: Use the shared local HF model cache instead of substituting a different LLM (2026-07-22)

**Context:**
- This account's own HF token isn't authorized for gated repos (`google/gemma-3-1b-it`, `meta-llama/Llama-3.2-1B-Instruct` both 403'd on real download — see bugs.md).
- A teammate (ytseng21) already has `meta-llama/Llama-3.2-1B-Instruct` downloaded to a shared team cache at `/home/jhu/jsalt2026-ext-cxiao7/scratch_jsalt2026-lgarci27/omnienc/hf/hub`.
- User explicitly directed not to swap to a substitute model to route around the gate.

**Decision:**
- Point the yaml's `llm_save_path` at the shared cache directory and set `HF_HUB_CACHE`/`HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1` so the real target LLM (`meta-llama/Llama-3.2-1B-Instruct`) loads entirely from local disk, no network/gate check at all.

**Alternatives Considered:**
- Swap to an open substitute model (e.g. `Qwen2.5-1.5B-Instruct`) → Rejected by user: changes the actual experiment, not acceptable as a silent substitution.
- Request Llama-3.2 access on this account's own HF token → Still open/recommended for the future, but requires manual action outside this session (visiting the HF model page and accepting the license).

**Consequences:**
- Unblocked immediately, using the actually-intended model.
- Dependent on the shared cache directory remaining available/populated; if it's cleared or this account loses read access to `omnienc/hf/hub`, this breaks and falls back to needing real gated access.

### ADR-003: boundary_target_dir sourced from precomputed wavlm_boundaries, not regenerated (2026-07-22)

**Context:**
- `speechllm_fixed_pooling.yaml`'s `boundary_source: alignment` needs a `boundary_target_dir` of per-utterance `.pt` frame-boundary tensors.
- The generation script (`prepare_boundary_targets.py`) only exists on the unmerged `origin/forced_alignment` branch.
- A teammate (ytseng21) already generated and shared the full-corpus output at `.../omnienc/datasets/wavlm_boundaries/{word,phone,syllable,char,word_ctc}`.

**Decision:**
- Use the precomputed `wavlm_boundaries/` directly — after verifying format/coverage first (correct tensor dtype/values, and file counts matching the full LibriSpeech corpus exactly per split) — rather than merging/running `origin/forced_alignment`'s generation pipeline.

**Alternatives Considered:**
- Merge `origin/forced_alignment` into this branch and regenerate boundaries from scratch → Rejected for now: unnecessary compute, and the branch is unmerged/not vetted against current `main`.

**Consequences:**
- Fast path to a real training run.
- Coupled to that shared directory's continued existence/correctness; if it's ever regenerated or removed, this repo has no in-branch fallback until `origin/forced_alignment` is merged.

### ADR-004: CER-reward GRPO made off-policy (num_iterations) + batched decode (2026-08-03)

**Context:**
- The free-running CER/WER reward requires autoregressively decoding K sampled segmentations
  per utterance. v1 did this as a Python loop of K **sequential** greedy decodes inside
  `compute_forward`, on-policy (one gradient step per generation). cer_mt ran at ~11.6 s/it
  (~18x the NLL reward's 0.65 s/it) — one joint epoch stuck at ~15 h, decode-bound.
- ms-swift / TRL GRPO amortize the expensive generation with `num_generations` (group size)
  + `num_iterations` / `steps_per_generation` (reuse one generation for several gradient
  updates, off-policy, corrected by a PPO-clipped importance ratio).

**Decision:**
- **Batch the K decodes**: replicate the utterance batch K× along dim 0 and decode all K·B in
  ONE wide generation (`_rollout_rewards` / `_decode_replicated`), instead of K sequential calls.
- **Off-policy sample reuse**: override `fit_batch` for the joint-RL phase — generate the
  K-sample rollout + decode ONCE (no grad), then take `num_iterations` (μ=3) gradient steps on
  it with the clipped surrogate `grpo_clipped_pg_loss` (`min(rÂ, clip(r,1±ε)Â)`, ε=0.2, per
  frame). μ=1 reproduces the old on-policy GRPO exactly (ratio≡1 → same gradient; unit-tested).
- **`max_decode_ratio` 3.0→2.0** (a reference transcript is ≲ the compressed prefix, so 2× is
  ample headroom while ~halving decode length).
- My GRPO == ms-swift GRPO at μ=1 with **no KL-to-ref** (dropped by design, see the no-KL
  decision) and std-normalized group advantage; adding μ>1 + clip is the off-policy extension.

**Alternatives Considered:**
- Keep NLL reward only → Rejected: teacher-forced NLL has low conditional MI w.r.t. the audio;
  CER is the intended objective. Optimize CER instead of abandoning it.
- Hybrid (CER every N steps, NLL otherwise) → Rejected: user wants CER always.

**Consequences:**
- Decode amortized ~μ× and batched ~K×; resumes from the epoch-6 warmup ckpt so only joint
  epochs 7–10 re-run. Watch: K·B batched-decode memory; the argmax decoder-CE forward is
  recomputed each of the μ steps (μ× more decoder CE updates per batch than before).

### ADR-005: Re-diagnosed the RL collapse as two distinct mechanisms, not one (2026-08-03)

**Context:**
- The segmenter RL runs collapsed twice, in two rounds: (1) the original joint run
  (no rate floor, entropy still annealing) crashed to ρ≈0 at the first RL epoch and
  never recovered; (2) the A/B/C retry (added a rate floor/band specifically to
  prevent that) collapsed too, in a visibly different way. Both were "fixed" by the
  same change (drop entropy to 0 + extend decoder warmup 2→6 epochs), which worked —
  but changed two variables at once, so the original write-up conflated the two
  failures into one flat "entropy smearing" story without being precise about which
  evidence supports which claim.
- User asked (2026-08-03) to revisit the failure with the benefit of the now-working
  runs and give a evidence-grounded opinion on the primary cause(s).

**Decision (documented finding, not a code change):**
- Treat the two collapses as separate root causes, each with its own direct evidence:
  1. **Failure #1** (original run): a genuine GRPO absorbing state. `grpo_trace.py`
     shows the collapsed policy at mean-p=0.000, entropy=0.0000, with all K=8 sampled
     segmentations per utterance bit-identical → reward std=0.0000 → advantage=+0.000
     for every sample → the policy-gradient is exactly zero, not just small, and can
     never move again (matches the log: ρ pinned at exactly 1.90e-3 for 7 straight
     epochs). Driven by an unbounded one-sided rate objective (no floor at the time).
  2. **Failure #2** (A/B/C retry): the floor/band fix worked exactly as intended — the
     *expected* ρ (Σp/T) never crashes (jointB: 0.184→0.077; jointC: 0.273→0.218, both
     epochs 3-5, staying near/in their guard) — but the *argmax* ρ (what the decoder
     actually gets) still crashes to ~0.002-0.009 in **both** runs, despite the two
     rate objectives being structurally different (soft floor vs. hard band). That
     shared signature isolates the per-frame entropy bonus (which maximizes at p=0.5
     and is unconstrained by a rate loss that only touches the sum, never the shape)
     as the cause — not the rate-objective choice.
  3. A third, separate observation: within jointB's own run, reward improved as ρ
     fell (−3.01→−2.84 as ρ dropped 0.184→0.077) — direct evidence the teacher-forced
     NLL reward pointed the wrong way under the 2-epoch undertrained decoder.
- **Primary-cause judgment** (since the real fix changed entropy AND warmup length at
  once, this isn't a controlled ablation): dropping entropy is the better-evidenced
  primary fix for the argmax-collapse specifically, since it's a direct, within-run,
  rate-objective-independent mechanism (reproduced identically in two different rate
  objectives). The longer warmup fixes the separate reward-misdirection problem (#3)
  and plausibly helps sharpness indirectly too (a better decoder → higher-SNR reward
  gradients that could outcompete entropy's pull) — but no run isolates
  entropy-off-with-short-warmup or entropy-on-with-long-warmup, so that part is an
  informed inference, not a proven fact.

**Alternatives Considered:**
- Keep the original flat "entropy smearing" narrative → Rejected: it doesn't
  distinguish the two failures' different signatures (both-crash-to-zero vs.
  expected-survives-argmax-dies) and overstates how isolated the entropy-vs-warmup
  attribution actually is.

**Consequences:**
- Confirms entropy-off is likely load-bearing for any future segmenter RL config;
  the reward-misdirection problem is a second, independent thing to watch for
  whenever the decoder is undertrained relative to the RL phase's reward.
- Open question for future work: a controlled 2×2 (entropy on/off × warmup
  short/long) would fully disentangle this if it's ever worth the compute.

### ADR-006: RL remains the core method; improve its start, policy, and training (2026-08-09)

**Context:**
- The project goal is a learned segmenter that improves ASR performance while
  reducing decoder-side audio-token cost. Fixed/oracle segmentations are controls
  and initialization sources, not substitutes for learning the policy.
- The first WavLM RL runs drifted away from their char initialization, but this is
  evidence about the current Bernoulli policy and curriculum—not evidence that RL
  should be abandoned.

**Decision:**
- Continue improving RL. The next reference run starts from both the best WavLM
  char segmenter and best WavLM oracle-char decoder, performs a short decoder
  adaptation warmup on predicted boundaries, and then runs a longer joint-RL phase.
- In parallel, replace independent framewise Bernoulli sampling with a structured
  causal policy that models dependencies between consecutive boundary labels while
  preserving the same best char initialization.
- Treat boundary-swap evaluations and fidelity thresholds as diagnostics for
  attribution and model selection, never as a reason to stop pursuing RL.

**Alternatives Considered:**
- Stop RL until a detector reproduces oracle boundaries almost perfectly → Rejected:
  this changes the research goal and prevents RL from addressing the remaining
  quality/compression trade-off.
- Continue only the existing Bernoulli recipe for more epochs → Rejected as the sole
  path: longer training is a useful control, but it does not address independent
  frame decisions or policy drift.

**Consequences:**
- Every new learned-segmentation result must report both WER and compute/kept ratio.
- Oracle, fixed-rate, and random/deterministic merging remain required controls.
- The immediate comparison isolates policy structure by using identical encoder,
  segmenter initialization, decoder initialization, warmup, reward, and epoch budget.

### ADR-007: Full-prefix Transformer AR uses fresh supervised initialization (2026-08-11)

**Context:**
- The first-order policy adds two transition biases to an existing acoustic
  segmenter, but the requested full-prefix policy is a new decoder-only Transformer
  over previous boundary decisions and is not checkpoint-compatible with that model.

**Decision:**
- Implement `transformer_ar` as a four-layer, 256-dimensional causal boundary-history
  Transformer with cached closed-loop sampling and parallel teacher-forced scoring.
- Force frame 0 to zero and exclude it from supervised and policy losses; count the
  implicit initial segment exactly once when measuring the kept ratio.
- Start with a fresh 10-epoch char-CTC cold start, then the same six-epoch decoder
  warm-up and four-epoch NLL-MT RL phase used by the WavLM independent-Transformer
  control. Policy dropout is zero so rollout and re-scoring use the same distribution.
- Apply the rate penalty to each realized rollout reward because exact boundary
  marginals are intractable for a full-history Transformer policy.

**Consequences:**
- The three RL seeds share one supervised segmenter and one decoder warm-up, matching
  the existing replication pattern while keeping all production checkpoints in a new
  `fullprefix_transformer_ar` namespace.
- WER and boundary F1 should be compared with the independent-Transformer control,
  but initialization quality must also be reported because the two architectures
  cannot share an exact segmenter checkpoint.

### ADR-008: Optimize Transformer-AR in process before adopting a vLLM rollout service (2026-08-12)

**Context:**
- Full-prefix Transformer-AR takes about 1 h 49 m per decoder-warm-up epoch and
  3 h 22 m per joint-RL epoch, versus about 27 m and 37 m for CNN first-order AR.
- Its cached step currently grows K/V tensors with `torch.cat` at every 50 Hz frame.
  Joint RL also runs a separate greedy rollout, a `K=4` sampled rollout, sequential
  reward forwards, and a parallel differentiable re-score.
- vLLM/ms-swift accelerates supported language-model rollout through colocated or
  external engines, but this policy must consume a different continuous WavLM vector
  at every generated boundary step. It is therefore not a drop-in vLLM request.

**Decision:**
- Preserve the active jobs as the baseline. First profile and implement exact
  in-process changes: preallocated/indexed KV storage, combined greedy-plus-sampled
  rollout batching where possible, and batched/microbatched reward evaluation.
- Fix the current causal local-attention arm at 64 frames. Use the same window in
  rollout and teacher-forced re-scoring; treat it as a scientific ablation and do not
  vary the window until this control is explicitly reopened.
- Keep the cache interface backend-neutral, but defer a vLLM/ms-swift async server.
  Reconsider it only if the optimized policy remains over 2× slower than CNN and the
  structured policy produces enough accuracy or boundary benefit to justify custom
  model/input integration, weight synchronization, and extra rollout resources.
- Keep `grpo_k=4` fixed for the current Transformer-AR speed study. Throughput work
  must improve the implementation or attention pattern rather than reduce the number
  of rollouts.
- Prefer batching the greedy and four sampled policy rollouts plus batching the four
  NLL reward evaluations. A 30-real-batch pilot reduced the training-loop time from
  101 s to 52 s (1.94×); FlexAttention did not improve steady-state end-to-end speed
  over the already fused masked-SDPA path at the current batch-length mix.

**Alternatives Considered:**
- Adopt ms-swift external async rollout immediately → Deferred: it targets supported
  language-model generation and would require a custom vLLM model runner plus a
  dedicated server/GPU for this continuous per-step audio-conditioned policy.
- Use vLLM only for the Llama reward decoder → Deferred for NLL experiments because
  the measured relative slowdown already appears during decoder warm-up, where the
  extra cost is the segmenter's sequential greedy rollout; it may still help later
  CER/WER reward experiments.
- Replace full history with first-order CNN AR → Retained as the fast control, not a
  replacement for testing whether longer boundary history helps.

**Consequences:**
- Near-term engineering stays inside the existing training process and remains easy
  to compare against current checkpoints.
- Local attention requires a matched WER/rate/variance comparison; the rollout group
  remains fixed at `K=4` in every comparison.
- The installed PyTorch Flash backend cannot accept the sliding-window mask. Compiled
  FlexAttention is useful for long isolated sequences but requires Triton/Python
  headers and adds compile overhead, so it remains an opt-in experiment rather than
  a production default.
- vLLM compatibility remains possible, but a rollout server is not part of the first
  speed milestone.

### ADR-009: Promote combined Transformer-AR rollout and reward batching as an opt-in on-policy mode (2026-08-12)

**Context:**
- The fixed-window-64 pilot showed that the largest end-to-end gain comes from two
  compatible changes: advancing the greedy history and four sampled histories in one
  widened sequential loop, and evaluating four NLL rewards in one decoder forward.
- The pilot implementation was installed through an environment-gated monkeypatch,
  which was appropriate for isolation but not suitable for future training recipes.

**Decision:**
- Add `rl_update_mode=combined_on_policy` for the full-prefix `transformer_ar`
  segmenter. The default remains `on_policy`, so active and historical jobs are
  unchanged.
- In the combined mode, use batch lanes `[:B]` for greedy actions and the remaining
  `K*B` lanes for stochastic actions in one frame-by-frame cached rollout. Draw random
  values only for sampled lanes, preserving the reference sample RNG stream.
- Decode all `K*B` detached rewards in one call, then use the existing parallel
  differentiable re-score, GRPO loss, decoder CE, and SpeechBrain gradient-
  accumulation/optimizer schedule. Keep `K=4`, local history 64, and preallocated KV.
- Retain fused masked SDPA. Do not promote FlexAttention because its isolated long-
  sequence kernel gain did not survive the real training loop.

**Alternatives Considered:**
- Keep the environment monkeypatch as the production interface → Rejected: hidden
  runtime replacement is harder to review, test, and reproduce than a named hparam.
- Combine rollouts but leave four sequential reward forwards → Rejected: this leaves
  a measured, mathematically batchable source of launch overhead in the step.
- Change `K` or reuse rollouts across optimizer steps → Rejected: those alter the
  estimator or on-policy semantics rather than only changing execution layout.

**Consequences:**
- CPU tests now require exact greedy boundaries, matched-seed sampled histories, RNG
  state, frame-0 behavior, and padding semantics; all original segmenter tests remain.
- A production smoke passed as job 70130. Matched full one-epoch train/validation/test
  jobs 70131 (reference) and 70132 (combined) run on separate ga129 A100s and write
  only under `transformer_ar_combined_rollout_validation/`.
- The implementation note is generated at
  `artifacts/segmenter/transformer_ar_combined_rollout.html`; regenerate it after the
  full jobs finish to insert measured stage times and test statistics.

### ADR-010: Use the optimized local-attention path for future full-prefix Transformer segmenters (2026-08-14)

**Context:**
- The window-64, preallocated-KV, combined-rollout implementation passed its
  correctness smoke and was substantially faster in the real-loop pilot.
- The user requested that future Transformer segmenters use this implementation
  unless an experiment explicitly studies another attention/runtime setting.

**Decision:**
- Default full-prefix `transformer_ar` runs to local history 64, preallocated KV,
  `combined_on_policy`, and `K=4`.
- Keep fused masked SDPA. Full history, concatenated KV, reference rollout loops,
  FlexAttention, or a different window require an explicit ablation override.
- `rl_update_mode=auto` selects the combined path for `transformer_ar` and the
  reference on-policy path for CNN/non-full-AR segmenters.

**Alternatives Considered:**
- Make FlexAttention the default -> Rejected because its isolated kernel gain did
  not improve the measured full training loop.
- Change `K` to gain speed -> Rejected because `K=4` is the fixed experimental
  control and changing it alters the estimator.

**Consequences:**
- The main hparams and reusable WavLM Transformer launchers now encode the optimized
  defaults; historical running processes are unchanged.
- Full-history results remain valid controls but are no longer the implicit template
  for new Transformer segmenter jobs.

### ADR-011: Use audio-token frequency in Hz in every reader-facing report (2026-08-17)

**Context:**
- The project historically displayed the retained-frame fraction `rho`. That value is
  convenient inside the training code but is harder to interpret when comparing
  decoder cost across fixed, oracle, and learned segmentations.
- Both wav2vec2 and WavLM emit 50 encoder frames per second in these experiments, so
  the actual decoder-side audio-token frequency has a direct, stable conversion.

**Decision:**
- All reader-facing plots, tables, captions, prose, analyses, and proposals use
  `f_audio = 50 Hz × rho` and report the result in Hz.
- Lower audio-token frequency means fewer decoder-side audio tokens and stronger
  compression. No figure should rely on the direction of an unlabeled fraction.
- Report one decimal place by default and multiply both the mean and sample SD by 50.
  Near-zero collapse diagnostics may use two decimals when one decimal would erase a
  meaningful distinction.
- Keep existing `rho` config keys, log fields, artifact schemas, and checkpoint
  metadata unchanged; this is a reporting convention, not a training-code migration.

**Alternatives Considered:**
- Continue displaying `rho` and define it more carefully → Rejected because readers
  still have to mentally multiply by 50 to understand tokens per second.
- Rename every internal field from `rho` to frequency → Rejected because it would
  break artifact compatibility and create risk without changing the experiments.

**Consequences:**
- Historical results remain numerically identical after the linear conversion.
- Thresholds and uncertainty shown in reports must be converted along with means;
  raw internal values remain available for reproducibility.
- Future report reviews should treat any reader-facing retained-frame fraction as a
  consistency bug.

### ADR-012: “Scale-up” unambiguously means LibriSpeech-960h (2026-08-23)

**Context:**
- A request for the key scale-up experiments was incorrectly implemented as another
  LibriSpeech-100h replication. This changed the requested experimental condition
  and wasted time and compute.

**Decision:**
- In this project, “scale-up” means training on the complete LibriSpeech-960h set:
  `train-clean-100`, `train-clean-360`, and `train-other-500`.
- Every scale-up launcher must set those three splits explicitly, print them before
  submission, encode `ls960` in job/output names, and use a new output root.
- A 100h rerun is called a replication, audit, or corrected retraining—never a
  scale-up. If the requested data scale is ambiguous, verify it before submission.

**Consequences:**
- Dataset scale is a launch-time invariant that must be checked alongside seeds,
  model configuration, and output paths.
- Future agents must not substitute a cheaper 100h experiment for a requested 960h
  run without explicit user approval.

### ADR-013: Compute-match the LS960 schedule and use validated kernels (2026-08-23)

**Context:**
- Reusing epoch counts and small microbatches from the 100h recipes would spend one
  complete 960h pass on decoder warmup alone, repeat CSV generation per job, and
  leave most A100 memory idle.

**Decision:**
- Train each LS960 arm for one full-corpus pass. For learned systems, resolve a
  within-epoch phase boundary from the dynamic sampler: 20% of optimizer updates
  are decoder-only warmup and 80% are joint RL.
- Use 400 s / accumulation 1 for cold starts, 300 s / accumulation 1 for learned
  systems, 500 s / accumulation 2 for compressed baselines, and retain the
  validated 50 s / accumulation 20 native-rate setting.
- Share verified 281,241-utterance LS960 manifests; use BF16, pinned asynchronous
  copies, four persistent workers, and two-way prefetching.
- Use `combined_on_policy` for Transformer AR and `batched_on_policy` for CNN AR.
  Keep K=4 and the 64-frame local window.
- Keep automatic PyTorch SDPA for the local Transformer. Forced Flash Attention is
  unsupported with its arbitrary local mask, and the compiled FlexAttention
  prototype was slower than the dense automatic-SDPA grouped rollout (193 s versus
  186 s for the matched 30-batch benchmark).

**Consequences:**
- The learned LS960 study consumes one pass rather than two while preserving the
  established warmup-to-RL exposure ratio and approximate optimizer-update count.
- The actual shared manifest yields 11,977 learned updates: 2,395 warmup and
  9,582 joint-RL, versus 2,004 and 10,020 in the established 2+10 epoch 100h
  schedule. Cold start yields 8,942 updates; compressed baselines 3,584; native
  50 Hz 3,937. These are within roughly 5–20% of the corresponding 100h totals.
- Stage logs remain the authority for peak GPU memory and timing; a second pass is
  a continuation decision based on seed-3407 dev results, not an automatic cost.

### ADR-014: Keep the one-week pitch RL-centered; include streaming as a gated ASR contribution (2026-08-23)

**Context:**
- The best learned 100 h system reaches 4.80 +/- 0.09 test-clean and 9.24 +/-
  0.36 test-other WER at 10.4 Hz, while the character-aligned oracle reaches
  3.96 +/- 0.41 and 7.86 +/- 0.71 at 14.6 Hz.
- Approximately one week remains for important experiments, current LS960 jobs
  occupy the observed three-A100 project capacity, and translation/diarization
  would require new data, heads, metrics, and matched baselines.
- STAR (IWSLT 2025) already demonstrates dynamic compression for streaming ASR and
  simultaneous speech translation, so streaming adaptive segmentation alone is
  not an unoccupied contribution.

**Decision:**
- Allocate the one-week critical path to finishing LS960, a frozen-segmenter OPD
  screen, constrained joint-RL frequency continuation if OPD passes, and a
  bounded-lookahead streaming evaluation of the same RL segmenter. Streaming is a
  secondary capability of the RL method, not a separate main pitch.
- Use one-seed triage before three-seed replication. For OPD, require at least
  0.3 absolute dev-other WER improvement at 10--12 Hz without more than 0.1
  dev-clean regression at 14.5 Hz. For joint continuation, require either 0.3
  dev-other improvement at similar frequency or WER within 0.1 with at least 15%
  lower frequency.
- Gate streaming with a causality, segment-duration, and hypothetical-latency audit,
  then build a chunked no-future-WavLM + cached segmenter + incremental-prefix
  decoding prototype if the audit passes. Do not call the present
  offline-WavLM/all-audio-prefix-Llama system streaming merely because its boundary
  policy is autoregressive.
- If time remains, permit one frozen-feature cross-task rate-sensitivity probe.
  Do not start a full translation, diarization, or speaker-counting matrix this week.

**Consequences:**
- OPD and frequency control receive compute only when the preceding gate passes;
  inconclusive full grids are avoided.
- The minimum one-week streaming result must use declared bounded lookahead, emit
  partial text before utterance end, and compare learned with fixed boundaries at
  matched frequency. An efficient future system should replace prefix re-decoding
  with an interleaved read/write, transducer, or monotonic encoder-decoder interface.
- The longer-term multi-task question is framed as task-dependent temporal
  resolution with a task-conditioned segmenter, not as a collection of unrelated
  downstream fine-tunes.
- Detailed analysis is in
  `reports/one_week_research_strategy_2026-08-23.md`.

### ADR-015: Control extended LS960 learning by optimizer steps (2026-08-24)

**Context:**
- A single validation after one LS960 pass cannot establish convergence, while
  epoch boundaries provide only two observations in a two-pass continuation.
- The first learned LS960 queue also skipped its intended fractional warmup, so
  extending those jobs would preserve the wrong training condition.

**Decision:**
- Corrected learned runs stop at 24,000 optimizer updates (approximately two LS960
  passes), independent of data-pass boundaries. Validate on dev-clean at the
  corrected warmup boundary (step 2,395) and every 4,000 updates through step
  24,000; select the checkpoint with minimum dev WER.
- Use three safety epochs only to keep the data iterator available until the step
  limit; do not interpret the nominal epoch counter as the stopping criterion.
- Keep the established model controls unchanged: K=4, local history 64, WavLM
  features, oracle-char decoder initialization, BiGRU pooling, 300 s dynamic
  batches, accumulation 1, BF16, and A100-only execution.

**Consequences:**
- Seven validation points expose the warmup-to-RL transition and later convergence
  without validating often enough to dominate training time.
- Mid-pass validation checkpoints are resumable and carry `validation_step` and
  `validation_reason` metadata; test decoding loads the minimum-dev-WER checkpoint.
- Jobs 106155–106160 are the three-seed CNN first-order AR + BiGRU and local-64
  Transformer AR + BiGRU executions of this protocol.

### ADR-016: Port irreplaceable project assets to CLSP in priority order (2026-08-24)

**Context:**
- Bluecrab scratch is 97% full, while the CLSP target filesystem has about 30 TiB
  free. The local results tree is about 143 GiB, of which 136.6 GiB is 2,911
  checkpoint-like files; much of that is repeated frozen encoder state.
- The Git working tree contains consequential uncommitted code, tests, launchers,
  project notes, reports, and artifacts. The corrected 24k-step LS960 runs are
  still producing checkpoints, so that directory cannot yet be treated as a
  quiescent final snapshot.
- The shared Hugging Face cache is 136 GiB, but only about 5.7 GiB of unique model
  files are required by this project. The five expanded boundary-label trees use
  about 6 GiB and 1.46 million files, but complete compressed archives total only
  about 377 MiB.

**Decision:**
- Port assets to
  `ctl2:/export/jsalt26/omnienc/users/cxiao/skipjack/jointllm` in recovery-value
  order: the complete learned-segmenter result lineages; repository state and
  research records; flagship LS960 checkpoints; dedicated environment; selected
  base-model caches and archived boundary labels; public/raw data; then genuinely
  replaceable historical checkpoints. The segmenter lineages are core project
  assets and must be copied wholesale, not reduced to a few selected best runs.
- Copy non-checkpoint experimental material (logs, metrics, predictions,
  manifests, and configs) from every result family even when its checkpoint files
  remain lower priority.
- Use one sequential `rsync` stream with `--bwlimit=25600` (25 MiB/s), resumable
  partial files, `nice -n 10`, and `ionice -c2 -n7`. Do not use parallel transfer
  streams. Rerun rsync after the corrected LS960 jobs become quiescent.
- Preserve the dedicated venv as recovery material, but rebuild from `uv.lock` on
  CLSP when possible because compiled environments and absolute paths are not
  guaranteed to be portable across clusters.
- Copy only the project-relevant WavLM-Large, Llama-3.2-1B-Instruct,
  wav2vec2-base, and wav2vec2-base-960h cache roots. Copy the boundary `.tgz`
  archives instead of the expanded small-file trees. Never copy credentials or
  token stores.

**Consequences:**
- The first operational recovery set is expected to require roughly 50–55 GiB;
  adding LibriSpeech audio and alignments raises this to roughly 115–120 GiB.
  A 250 GiB project allocation leaves headroom for the full historical result tree
  and growth of the corrected LS960 runs. A wholesale copy of the unrelated shared
  Hugging Face cache is outside this project migration.
- The CLSP copy is initially a recovery snapshot, not a validated runnable port.
  Paths, the Python interpreter, CUDA/PyTorch compatibility, and data extraction
  must be checked on CLSP before launching experiments there.

### ADR-017: Prefer the JSALT reservation for long A100 training jobs (2026-08-25)

**Context:**
- Corrected LS960 jobs submitted without an explicit reservation ran on the general
  A100 pool (`ga130`, `ga137`, and `ga143`) and repeatedly waited behind general
  priority. The JSALT reservation currently contains ga129, ga132–ga134, and
  ga138–ga141; ga129 remains prohibited by default.
- At the scheduling audit, the account was using exactly its 64-GPU QOS cap. Five
  healthy GPUs were nevertheless idle inside the reservation, so neither a general
  nor reserved job could start until some account allocation ended.

**Decision:**
- Attach long project A100 training jobs to `--reservation="JSALT 2026"` by default,
  while still specifying the required account, A100 partition, `accept_cost`
  comment, and ga129 exclusion. Permit an empty `RESERVATION_NAME` override after
  the reservation expires or when explicitly testing general capacity.
- Request two days by default for the LS960 queue instead of the 3-day maximum.
  Measured fresh-run times are 8h26m for CNN and 18h16m for Transformer; the extra
  margin protects against slower nodes, repeated validation, and recovery overhead.
- Update pending jobs in place rather than cancel/resubmit so job age, restart
  state, and checkpoints are preserved. Do not maintain duplicate reserved and
  general copies that could write the same output directory.

**Consequences:**
- The reservation cannot bypass `MaxGRESPerAccount`; current jobs still wait until
  the account drops below 64 GPUs. Once it does, reserved capacity avoids competing
  with other accounts for a general-pool node.
- Jobs 106157–106160 now use `JSALT 2026`, retain their original IDs, and continue
  to exclude ga129. Their time limits were reduced in place to two days. Both LS960
  launchers use the same default strategy and expose reservation/time overrides.

### ADR-018: Report only corrected batch-invariant RTF and treat batch sizes as operating points (2026-08-26)

**Context:**
- The HTML report combined corrected batch-invariant WER with legacy speed artifacts
  whose decode budget depended on the padded batch prefix. The corrected suite has
  three seeds at one preselected throughput batch per system, but no usable full
  batch-1 timing or systematic batch sweep.
- Each corrected split already excludes five initial duration-sorted batches. This
  removes cold CUDA startup from the numerator, but the discarded batches are the
  shortest shapes and do not establish steady-state timing robustness.

**Decision:**
- Compute each corrected RTF from summed test-clean and test-other forward time and
  audio duration within a seed, then report the three-seed mean and sample SD. Exclude
  all legacy batch-1 and throughput artifacts from the current report.
- Call the existing batch sizes preselected operating points, not optimal batches.
  Do not infer a batching speedup without corrected measurements at a common batch.
- For a paper-ready throughput claim, select batch size separately for each system by
  a declared search over a common duration-sorted calibration set and fixed candidate
  grid, subject to no OOM and identical decoding semantics. Also report batch-1 latency
  and at least one same-batch comparison so throughput and architecture effects remain
  distinguishable.

**Consequences:**
- The report now gives corrected three-seed RTF values only; the legacy single-seed
  speed figure and its batch-1 comparisons are removed.
- The present values characterize the evaluation pipeline, which includes a
  teacher-forced Llama pass and non-KV-cached greedy generation. They are not labeled
  optimized deployment throughput.

**Revision — simple memory-limited protocol (2026-08-26):**
- Replace the earlier general calibration-grid proposal with two reported operating
  points: batch-1 RTF for every seed and memory-limited throughput. For each system,
  seed 3407 tries power-of-two batches on two batches of the longest dev-other
  utterances. Select the largest batch that completes and stays at or below 90% of
  A100 allocated memory; choose without consulting RTF, then freeze that batch for
  seeds 3407/3408/3409.
- Before each final batch-1 or selected-batch measurement, run ten batches of long
  dev-other utterances in the same process, exclude that pre-warm dataset, and measure
  every test-clean/test-other batch. Save per-utterance batch-1/selected hypothesis
  comparisons so any numerical batch dependence is visible.
- Submitted selector jobs 113881, 113885, 113889, 113893, 113897, 113901, 113905,
  113909, 113913, 113917, 113921, and 113925. Final three-seed jobs 113882–113928
  are wired by per-system `afterok` dependencies. All use one A100, the JSALT account
  and reservation, `accept_cost`, and exclude ga129. Outputs live under
  `artifacts/segmenter/inference_eval_memory_limited_batch_v1/`.

**Revision — conservative batch-32 retry (2026-08-27):**
- After batch-64 full-test OOMs, freeze batch 32 for fixed k=5, fixed k=6,
  character oracle, CNN AR mean, and CNN AR + BiGRU. Repeat all three seeds rather
  than mixing successful batch-64 and retry batch-32 measurements. This also puts
  CNN AR + BiGRU at the same batch as both Transformer AR systems.
- Reuse the completed batch-1 hypothesis artifacts only for numerical comparison;
  run each batch-32 benchmark in a fresh output directory with ten long-utterance
  warm-up batches and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Jobs 118878–118892 were accepted on A100 with the JSALT reservation/account,
  `accept_cost`, and ga129 excluded. Outputs are isolated under
  `inference_eval_memory_limited_batch_v1/retry_batch32_v1/`.

**Revision — common batch-16 comparison (2026-08-27):**
- The memory-limited operating points use batch 16 for no downsampling and batch
  32 for the main compressed systems, so add a same-batch throughput comparison
  at 16, the largest batch already shown stable for the native 50 Hz reference.
- Reuse the completed no-downsampling batch-16 measurements. Run fixed k=5, phone
  oracle, CNN/Transformer AR with mean pooling, and CNN/Transformer AR with BiGRU,
  each with seeds 3407/3408/3409. This directly separates compression/model cost
  from the benefit of allowing compressed systems a larger batch.
- Jobs 119834–119851 use A100 only, the JSALT account/reservation, `accept_cost`,
  eight requested CPUs, 60 GB RAM, an eight-hour limit, and exclude ga129. Outputs
  live under `inference_eval_memory_limited_batch_v1/common_batch16_v1/`.

**Revision — primary efficiency comparison remains max-safe batch (2026-08-27):**
- Per-system memory capacity is part of the benefit of downsampling. Therefore use
  each system's stable throughput batch as the primary comparison: batch 16 for no
  downsampling and batch 32 for the headline compressed systems. A common-batch
  result can only be a secondary decomposition, not the headline benchmark.
- No downsampling at batch 16 peaks at **41.85±0.32 GB allocated** on the long
  prewarm but **78.66±0.04 GB reserved** of 79.25 GB; batch 32 OOMed in selection.
  The apparent spare allocated memory therefore does not imply batch 32 is safe.
- Hold the 14 common-batch jobs that had not started (119838–119851 except the
  already-running 119837). Leave the four short jobs already running (fixed k=5
  seeds 3407/3408/3409 and phone seed 3407) intact rather than killing them.

### ADR-019: Publish completed HTML report updates by default (2026-08-27)

**Context:**
- A validated local training report contained the completed three-seed attribution
  and memory-limited RTF results, while the GitHub Pages copy still showed pending
  Table 3 rows. Stopping after local regeneration made a completed update appear
  unfinished to readers.

**Decision:**
- Unless the user explicitly requests a local draft, an HTML report update includes
  regeneration, validation, a focused source-repository commit and push, copying the
  standalone artifact into the Pages repository, bumping the wrapper's cache key,
  validating the site, committing and pushing `origin/main`, and verifying live
  content after deployment.
- Commit only files belonging to the report update; preserve unrelated WIP in both
  repositories.

**Consequences:**
- “Update the HTML report” now means publish the reviewable result, not only modify a
  local artifact. Deployment verification is part of completion.

### ADR-020: Run one deployable CTC baseline, then prioritize the task-conditioned ASR+ST flagship (2026-08-28)

**Context:**
- The ASR attribution and LS960 learned-system results are mature, while a
  transcript-free CTC posterior-collapse row remains the most obvious missing
  external baseline.
- ICASSP 2027 and ICLR 2027 deadlines are close enough that completing every ASR
  and ST baseline before testing the new research claim would leave insufficient
  time for the main task-conditioned system.
- The segmenter already contains identity-initialized task FiLM, and the completed
  LS960 Transformer-AR + BiGRU system is a substantially safer initialization than
  training a multi-task speech interface from scratch.

**Decision:**
- First run a three-seed 100h transcript-free CTC posterior-collapse baseline. Use
  the cached supervised wav2vec2 CTC model only for audio inference; do not use
  transcripts or forced alignment to produce boundaries. Train the downstream
  BiGRU/projection/LoRA for the same seven-epoch 100h adaptation budget.
- After the CTC jobs are queued, prepare MuST-C En–De and implement the multi-task
  data/task interface. Launch the one-seed best proposed system directly:
  LS960-initialized local-64 Transformer-AR, shared BiGRU, and task-conditioned
  segmenter, with a frozen-policy warmup inside the same run.
- Defer the broad ST baseline grid, independent policies, task-conditioned pooler,
  third task, prompt paraphrases, and full rate sweep until the flagship one-seed
  gate passes.

**Consequences:**
- “Transcript-free” describes inference boundary generation, not the supervision
  history of the pretrained CTC acoustic model or the ASR decoder adaptation.
- Precomputed CTC boundaries make training efficient but do not measure deployable
  CTC cost; any RTF result must run/time the CTC encoder live.
- The accelerated multi-task plan accepts a risk of weaker early attribution in
  exchange for testing the main ICLR-scale claim before the deadline. Required
  controls are added only after a positive signal.
- The authoritative proposal is `plans/task_conditioned_multitask_segmenter.md`;
  the broader literature-driven matrix remains preserved under `survey/2026-08-27/`.

### ADR-021: Use blank-aware BPE-500 CTC as the rate-closer external baseline (2026-08-28)

**Context:**
- The completed transcript-free wav2vec2 CTC baseline uses a character vocabulary
  and consequently emits about 14.6/14.3 audio tokens/s, versus roughly 11.2/10.8
  for the joint Transformer-AR + BiGRU system. Its strong WER is useful, but the
  tokenizer-dependent frequency makes it an unmatched operating point.
- The released ESPnet WavLM-Large+BPE model uses 5,000 unigram pieces and is much
  too coarse. No released WavLM CTC checkpoint with an appropriate smaller BPE
  vocabulary was found in the candidate search. Icefall releases a full-LibriSpeech
  Conformer-CTC checkpoint with 500 SentencePiece units and a portable TorchScript
  inference model.

**Decision:**
- Keep the character-CTC result as a character-scale deployable reference, but use
  Icefall BPE-500 CTC for the closer-frequency comparison.
- Segment at every argmax CTC state transition, including transitions into and out
  of blank. This makes blank/gap intervals explicit decoder segments. Do not use
  reference text, forced alignment, LM decoding, posterior thresholds, or WER to
  tune the boundary rule.
- Lock the rule from a transcript-free dev-clean frequency pilot: 9.06 Hz for state
  runs versus 4.98 Hz for nonblank-only collapse. Run three downstream seeds for
  five epochs as the deadline-driven first pass.

**Consequences:**
- The new baseline is closer but not exactly rate matched; report its measured Hz
  beside WER and compare it with both fixed k=5 and the learned system.
- Full preprocessing confirmed **9.678 Hz overall**, including 9.775 Hz on
  test-clean and 9.443 Hz on test-other. The ordinary nonblank-only diagnostic is
  5.348 Hz overall.
- Blank-aware state runs are not the ordinary nonblank CTC collapse. Name the row
  explicitly “BPE-500 CTC state runs” and retain the 4.98 Hz nonblank rate as a
  diagnostic so the construction is transparent.
- The compressor is Conformer-based, not WavLM-based. Any runtime table must time
  its live fbank+Conformer inference and must not count boundary-file lookup as the
  compressor cost.
- Jobs 135788–135791 implement this decision; if five epochs appear undertrained,
  a seven-epoch extension is the only budget follow-up before adding more CTC rules.

### ADR-022: Replace the external BPE construction with a self-trained WavLM phone-CTC baseline (2026-08-28)

**Context:**
- BPE-500 state runs obtain a useful rate only by treating blank-state changes as
  segments, and the released acoustic model is a Conformer rather than WavLM.
- A phone vocabulary naturally puts ordinary nonblank CTC collapse near the
  phone-oracle frequency and gives a cleaner comparison to the learned segmenter.
- Deadline constraints favor checkpoint selection within one short run over a new
  three-seed training grid.

**Decision:**
- Train a compact phone-CTC MLP head for three epochs on frozen WavLM-Large using
  `train-clean-100`. Targets are ordered MFA phone symbols with stress stripped;
  phone timings are discarded. Rank and retain all three epoch checkpoints by
  dev-clean phone error rate, and run audio-only dev-clean inference for each.
- Generate full train/dev/test boundaries only from the best dev-PER checkpoint,
  using ordinary nonblank CTC runs. Gate downstream launch on dev PER at most 40%
  and a measured boundary rate between 7 and 14 Hz.
- Train one residual-BiGRU + decoder run for four epochs, keep the best three
  validation-WER checkpoints, and evaluate all three separately on the standard
  test-clean/test-other/dev-other splits.

**Consequences:**
- “Transcript-free” applies to boundary inference. The CTC head is supervised by
  phone sequences derived from transcripts/alignments, which must be stated.
- The primary CTC quality metric is PER, not word WER; downstream decoder WER is
  the publication metric.
- This is a 100h rapid baseline, not an LS960 scale-up and not a three-seed estimate.
- The standard headline Transformer-AR + BiGRU value remains the corrected
  **4.80±0.09 / 9.24±0.36** result wherever that system name appears in the report.

### ADR-023: Replicate phone-CTC downstream decoder fine-tuning across three seeds (2026-08-29)

**Context:**
- The phone-CTC head and audio-only boundaries are deterministic shared inputs to
  the downstream comparison, while decoder/BiGRU optimization still has meaningful
  seed variance.
- One seed would provide a rapid checkpoint comparison but not a publication-ready
  uncertainty estimate for downstream WER.

**Decision:**
- Keep the single three-epoch phone-CTC training run and its selected epoch-3
  boundary set fixed.
- Run the four-epoch downstream BiGRU+decoder fine-tuning with seeds
  3407/3408/3409. Retain three checkpoints per seed and evaluate every retained
  checkpoint before selecting the per-seed result.

**Consequences:**
- The final downstream row supports a three-seed mean and sample SD without paying
  for redundant CTC training or boundary generation.
- Jobs 171203, 171226, and 171228 are the three training seeds; arrays 171204,
  171227, and 171229 evaluate their retained checkpoints.

### ADR-024: Report the final downstream checkpoint for all phone-CTC seeds (2026-08-29)

**Context:**
- Seeds 3408 and 3409 selected epoch 4 at validation rank 0, while seed 3407
  selected epoch 2; its epoch-4 checkpoint was retained and evaluated at rank 1.

**Decision:**
- Use epoch 4 for seed 3407 as well, making the reported three-seed comparison a
  consistent final-checkpoint result rather than a per-seed dev-selected result.

**Consequences:**
- The reportable phone-CTC downstream result is **5.09±0.17 / 9.80±0.24**
  test-clean/test-other WER and **9.42±0.27** dev-other WER.
- The prior dev-selected rank-0 aggregation remains available as a checkpoint
  diagnostic but is not the selected headline row.

### ADR-024: CoVoST 2 En-De replaces MuST-C as the paired ASR/ST corpus (2026-08-29)

**Context:**
- `plans/task_conditioned_multitask_segmenter.md` specifies MuST-C v1 En-De.
- MuST-C is distributed only behind an FBK registration form (ict.fbk.eu/must-c);
  there is no automated download. Every Hugging Face mirror checked
  (`enimai/MuST-C-de`, `maxolotl/must-c-en-de-01`, `may-ohta/MUST-C`) carries
  text only or a loader script requiring locally staged audio.
- The study needs one property above all: the SAME English audio must carry both
  an English transcript and a German translation, so ASR and ST are two views of
  one utterance and the same-audio boundary counterfactual is well defined.

**Decision:**
- Use CoVoST 2 En-De (`fixie-ai/covost2`, 13.8 GB of parquet with embedded
  audio), which satisfies that property exactly and is a standard sacreBLEU
  benchmark. Chosen by the user over waiting for an FBK link.
- Keep the corpus a configuration choice (`covost_manifest_dir`, `train_sources`)
  so MuST-C can be added later without touching the trainer.

**Consequences:**
- Speech is Common Voice read speech rather than TED-talk spontaneous speech,
  and results are not directly comparable to MuST-C-reporting prior work. Any
  writeup must name the corpus rather than implying MuST-C.
- 289k paired train utterances (~430 h) and 15.5k/15.5k dev/test pairs, well
  above the plan's sampling needs.
- Data is local: parquet under `datasets/covost2_raw`, 16 kHz FLAC and manifests
  under `datasets/covost2_en_de`.

### ADR-025: Multi-task training lives in a separate trainer subclassing SegmenterASR (2026-08-29)

**Context:**
- The ASR-only recipes are the ICASSP fallback path and must keep working
  unchanged while the multi-task extension is developed.
- `train_speechllm_with_segmenter.py` is ~2.7k lines of validated GRPO,
  rollout, bilevel, step-validation, and benchmark machinery that must not be
  forked.

**Decision:**
- `train_speechllm_multitask.py` subclasses `SegmenterASR`; multi-task config
  lives in `hparams/speechllm_multitask.yaml`. Data/model additions live in new
  modules (`multitask_data.py`, `multitask_modules.py`, `covost2_prepare.py`).
- The only edit to a shared file is `segmenter.py`: a `Segmenter.active_task_id`
  attribute defaulting to `None` plus a `_resolve_task` fallback at the five
  existing FiLM call sites. With `num_tasks=1` and `active_task_id=None`, the
  single-task path is byte-for-byte identical.
- Task ids are fixed by `multitask_data.TASK_SPECS`: `asr=0`, `st_en_de=1`.
  ASR must be task 0 so an ASR warm start is the identity at step zero.

**Consequences:**
- ASR-only runs execute the same code as before apart from one `is None` check.
- Verified: 22 new unit tests plus the existing sampler/bilevel/pooling/decode
  suites (55 tests) all pass, and the on-GPU equivalence gate reproduces the
  warm-started model exactly.

### ADR-026: Warm start the flagship from seed 3408, not 3407 (2026-08-29)

**Context:**
- `plans/task_conditioned_multitask_segmenter.md` names seed 3407;
  `survey/2026-08-27/projects/multitask_st_plan.md` says "selected by dev-clean
  WER". The completed LS960 cohort ranks 3408 (2.879) < 3409 (2.912) < 3407
  (2.939), so the two documents point at different checkpoints.

**Decision:**
- Warm start from seed 3408's 24k-step checkpoint (user decision), i.e.
  `results/speechllm_ls960_step24k_corrected/transformer_ar_local64_bigru/nll_mt/3408/save/CKPT+2026-08-26+02-54-19+00`.

**Consequences:**
- The ST bridge starts from the strongest available ASR model.
- The pilot is no longer seed-matched to artifacts elsewhere in the project that
  used 3407; any cross-comparison must state the seed.

### ADR-027: Rate-match phone-CTC with transcript-free long-segment refinement (2026-08-29)

**Context:**
- Ordinary phone-CTC nonblank runs produce 9.76/9.39 Hz on test-clean/test-other,
  below the requested approximately 11 Hz comparison point.
- Relabeling phones (for example retaining stress) changes token identities but not
  the target sequence length. A genuinely finer phone tokenizer would require a
  second CTC training run and an uncertain subphone design.

**Decision:**
- Preserve every phone-CTC boundary and add one midpoint to predicted segments at
  least eight 20-ms frames long. Select the eight-frame threshold using dev-clean
  frequency only, before downstream WER is observed.
- Label the system “phone-CTC + long-segment split,” not as a new tokenizer. The
  refinement uses only the predicted boundary tensor and segment length.
- Run three downstream seeds for three epochs, retain all three checkpoints per
  seed, and evaluate all nine checkpoints.

**Consequences:**
- Refined rates are 10.77 Hz train-clean-100, 11.06 dev-clean, 10.70 dev-other,
  10.90 test-clean, and 10.49 test-other.
- Training jobs are 202998/203000/203002; evaluation arrays are
  202999/203001/203003. The new boundary set contains 39,665 tensors under
  `boundary_targets/wavlm_phone_ctc_tc100_best_longsplit8`.
- All jobs completed with exit code zero. The final epoch gives
  **5.02±0.12 / 9.34±0.26** test-clean/test-other WER and
  **8.80±0.13** dev-other WER across the three seeds; it is also the strongest
  aggregate epoch of the short run.

### ADR-028: Scale the rate-raised phone-CTC baseline to LS960 with decoder CE only (2026-08-29)

**Context:**
- The three-epoch 100h rate-raised row is close to the learned system but remains
  0.22/0.10 WER points worse on test-clean/test-other.
- The deadline rules out retraining the CTC head or another multi-epoch downstream
  sweep. The requested follow-up is one LS960 pass and three seeds.

**Decision:**
- Keep the 100h-trained WavLM-Large phone-CTC head frozen. Generate boundaries for
  train-clean-100, train-clean-360, train-other-500, dev-clean, dev-other,
  test-clean, and test-other with ordinary nonblank runs plus the locked
  eight-frame midpoint rule. Boundary inference remains transcript-free.
- Use parameter-free mean pooling and regular decoder CE. The only effective
  trainable components are the WavLM-to-LLM projection and Llama LoRA adapters;
  there is no CTC, segmenter, BiGRU, or RL update.
- Train seeds 3407/3408/3409 for exactly one full LS960 pass. Use 500-second
  dynamic batches with accumulation two, matching the established compressed
  one-pass LS960 controls.

**Consequences:**
- This is a true LS960 experiment, explicitly using all three official training
  splits. It is not a strict data-scale-only extension of the 100h CTC + BiGRU
  row because the LS960 follow-up intentionally removes trainable BiGRU pooling.
- Boundary generation is array 204330 (16 shards, at most four A100s), followed by
  CPU gate 204331 and training jobs 204332/204333/204334. Outputs are under
  `results/speechllm_ls960_phone_ctc_longsplit8_mean_ce_1ep/`; the complete
  boundary set is under `boundary_targets/wavlm_phone_ctc_ls960_best_longsplit8/`.
- The complete chain exited successfully. Across three seeds the one-pass result is
  **4.05±0.08 / 7.61±0.06** test-clean/test-other and **7.04±0.26** dev-other.
  At 10.90/10.49 Hz on the test splits, it improves on the established one-pass
  fixed-k=5 mean control (4.37±0.10 / 7.72±0.18), but remains far behind the
  longer-trained LS960 Transformer AR + BiGRU system (2.79±0.02 / 5.63±0.12).

### ADR-029: Spend the remaining allocation on one checkpoint-resumed LS960 continuation (2026-08-30)

**Context:**
- The first LS960 phone-CTC decoder-CE result is complete across three seeds at
  4.05±0.08 / 7.61±0.06 WER, but the user requested one additional short training
  epoch before the project account may expire.
- Each seed has one intact SpeechBrain checkpoint containing the model, optimizer,
  and epoch counter. Reusing it is faster and safer than starting replacement runs.

**Decision:**
- Resume all three existing seed directories in place with the identical full-LS960
  data mixture, frozen phone-CTC boundaries, parameter-free mean pooling, regular
  CE objective, learning rate, batching, and decoder projection + LoRA trainables.
- Keep best-only checkpoint retention. If the new validation state improves, it
  replaces the retained checkpoint and is evaluated; otherwise the prior stronger
  checkpoint remains reportable.
- Do not state the decoder-training epoch count in the reader-facing HTML report;
  describe the optimization recipe and data scale instead.

**Consequences:**
- Continuation jobs are **205552/205553/205554**. Scheduler and logs confirm all
  three are running on A100 node `ga138`, loaded their existing checkpoints, and
  entered the additional epoch without restarting from scratch.
- The completed pre-continuation aggregate remains the report value until the
  continuation produces a verified stronger selected checkpoint.

**Reporting revision (2026-08-30):**
- Treat **4.05±0.08 / 7.61±0.06** as the final reader-facing phone-CTC baseline
  for now. Do not mention the continuation in the HTML report even if its cluster
  status changes; a later replacement requires an explicit reporting decision.
- Present CTC alongside the oracle/fixed reference baselines, not in the header.
  Emphasize instead that the full-LS960 learned Transformer system is 1.25/1.99
  WER points better at essentially the same token frequency.

### ADR-030: Rehome the ported venv instead of rebuilding it from uv.lock on CLSP (2026-08-31)

**Context:**
- The skipjack -> CLSP port copied `~/cxiao/envs/jointllm` verbatim, but CLSP has
  no `/usr/bin/python3.11`, so the copy was not runnable. `reports/clsp_asset_port_2026-08-24.md`
  recommended treating it as recovery material and rebuilding from `uv.lock`.
- The pinned stack is cluster-sensitive by design: `torch==2.7.1+cu118` /
  `triton==3.3.1` were pinned after an unpinned resolve hit a Triton JIT failure
  inside Llama's rotary embedding (see `bugs.md`). A fresh resolve risks silently
  reintroducing that, and a rebuild would also have to re-derive the 236 installed
  distributions rather than reproduce them.

**Decision:**
- Install a managed CPython 3.11.15 with the ported `uv` under
  `/export/jsalt26/omnienc/users/cxiao/uv-python`, copy the venv to
  `/export/jsalt26/omnienc/users/cxiao/envs/jointllm`, and rewrite only its
  location-dependent parts: `pyvenv.cfg` `home`, the `bin/python` symlink, 41
  console-script shebangs, and the editable-speechbrain finder MAPPING.
- Leave `_portability/envs/jointllm` untouched as the pristine recovery copy.
- Keep the option of a `uv.lock` rebuild for later; it is not on the critical path.

**Consequences:**
- Exact package versions are preserved, so any result difference across clusters
  cannot be attributed to a dependency change. That mattered immediately: the
  port verification found identical teacher-forced loss and segmentation rate for
  the flagship checkpoint, isolating the residual WER difference to greedy-decode
  jitter rather than a library change.
- 3.11.15 replaces 3.11.11. Same minor version, so the compiled extensions in
  site-packages remain ABI-compatible; the full SpeechBrain unit suite (828
  passed, 5 skipped) and both GPU verification jobs confirm this in practice.
- The env is now tied to `/export/jsalt26/omnienc/users/cxiao/uv-python`. Deleting
  that interpreter breaks the venv; a `uv python install 3.11` restores it.

### ADR-031: CREMA-D is labelled by crowd voice vote with ties dropped (2026-09-01)

**Context:**
- `survey/2026-08-31/projects/non_asr_multitask_plan.md` asked for "the crowd
  majority **voice** rating rather than the intended acted label" while also
  citing the TFDS split sizes 5,144/738/1,556. Those cannot both hold:
  5,144+738+1,556 = 7,438 = 7,442-4, i.e. that split keeps essentially every
  clip, which is only possible under the acted label.
- The local corpus (`/export/corpora6/CREMA-D`) supplies both: `VoiceVote` in
  `processedResults/summaryTable.csv` and the acted code in the filename.
- The two label sets are shaped very differently. Voice vote: neutral 57.3%,
  angry 14.5%, fearful 9.5%, disgust 8.0%, sad 5.4%, happy 5.2%, plus 644 clips
  (8.7%) with no unique majority. Acted: near-uniform, 17.1% each except neutral
  at 14.6%.

**Decision:**
- Primary labels are the voice vote, with the 644 tied clips dropped (6,798
  remain). The acted label is emitted as a second column so the alternative is a
  config flag, not a re-prep.
- Splits are reconstructed speaker-disjointly from the 91 speaker ids sorted
  ascending, 63/9/19 speakers -> 4,702/665/1,431 clips (3.31/0.47/1.01 h).
- Gate 0 for this task is stated as macro-F1 (or balanced accuracy), never raw
  accuracy.

**Consequences:**
- The choice matches what the study actually claims -- that the segmenter
  preserves what a listener can hear -- rather than what an actor was instructed
  to perform.
- It costs 8.7% of the corpus and leaves a strongly skewed prior. That directly
  invalidates the plan's original Gate-0 wording: a constant "neutral" predictor
  scores 57.3% accuracy, clearing a 55% raw-accuracy bar while scoring about 12
  macro-F1. Reporting raw accuracy alone on this task would be misleading.
- CREMA-D's train split is 3.31 h, so the plan's 167 h per 10k steps is about 50
  passes. The early-stop/augmentation-refresh clause is load-bearing, not
  precautionary.

### ADR-032: Grow TASK_SPECS to five tasks and validate the count at startup (2026-09-01)

**Context:**
- ADR-025 fixed task ids in `multitask_data.TASK_SPECS` (`asr=0`, `st_en_de=1`).
  The 2026-08-31 survey adds `emotion=2`, `speaker_count=3`, `intent=4`.
- The task count had two independent sources of truth: the YAML
  `segmenter_num_tasks` sized the FiLM table and the routed-LoRA slots at
  construction, while `expand_film_to_tasks(segmenter, NUM_TASKS)` resized FiLM
  from the code constant during warm start. A disagreement was silent: FiLM was
  rebuilt at one width against a router of the other, surfacing as an
  out-of-range index at the first batch of a high-numbered task.

**Decision:**
- Append ids 2-4 to `TASK_SPECS`, leaving 0 and 1 untouched, and carry each
  classification task's canonical label set on its `TaskSpec`.
- `speaker_count` holds the full `zero`-`ten` vocabulary; a `candidate_subsets`
  hparam restricts the pilot to `zero`-`four`. Widening the pilot later changes
  no task id and no checkpoint shape.
- `train_speechllm_multitask.validate_task_table` raises at startup if
  `segmenter_num_tasks` and `NUM_TASKS` disagree; `segmenter_num_tasks` is now 5.

**Consequences:**
- The failure mode moves from "wrong tensor an hour into a run" to "named error
  in the first second".
- The task count sizes the FiLM and routed-LoRA tensors, so the completed 2-task
  ST checkpoints (`results/speechllm_multitask_ls960_{covost,mustc}_en_de/3408`)
  cannot be resumed under the 5-task table. They remain readable for analysis;
  new runs warm start from the single-task LS960 ASR checkpoint, which is
  unaffected because it predates task routing entirely.
- Two regression tests pin the invariant: ids 0/1 are frozen, and FiLM stays
  identity when the table is widened.

### ADR-033: Classification answers are ranked, not generated (2026-09-01)

**Context:**
- The recipe scored every task by free-running greedy decode plus WER/BLEU. For
  a closed label set that is the wrong instrument twice over: the decoder can
  emit a string that is not a label at all, and labels that tokenize to
  different lengths let sequence length stand in for evidence.
- Nothing in the repo scored a fixed candidate list; `task_output_kind` was a
  name-prefix hack (`st_` -> translation, else text), so a new task would have
  been WER-scored.

**Decision:**
- `candidate_scoring.py` ranks the label set under the SAME pooled audio prefix,
  scoring each candidate by mean per-token log-likelihood of its own tokens.
- The GRPO task reward is the gold-versus-best-incorrect margin, not accuracy:
  with `K=4` rollouts most rollouts of an utterance agree on the answer, so an
  accuracy reward would have zero within-group advantage almost everywhere.
- A third `task_output_kind`, `classification`, reports accuracy and macro-F1
  per task; checkpoint selection stays ASR-WER-only.

**Consequences:**
- Cost is one extra decoder forward of `B x M` short sequences per validation
  batch, chunked by `candidate_chunk_size` (default 256). The 31-way intent set
  at a ~58-clip microbatch is 1,798 rows, which is why chunking is not optional.
- Length normalization is a modelling commitment, not a detail: without it a
  one-token label beats a four-token label on length alone. It is tested.
- `_candidate_rewards` is implemented and tested but not yet selectable --
  `segmenter_reward` still offers only `nll`/`cer`/`wer`. Wiring it is the next
  step before a first integrated run.


### ADR-034: Speaker-count mixtures are stored as recipes, not audio (2026-09-01)

**Context:**
- The speaker-count task needs tens of thousands of multi-speaker mixtures.
  Materializing them LibriMix-style costs hundreds of GB for audio we want to be
  free to redraw, and `/export/fs05`/`/export/fs06` have no headroom.
- The boundary-alignment analysis scores predicted boundaries against source
  onsets/offsets and concurrent-count change points, so those labels have to
  match the audio the model actually heard, exactly.

**Decision:**
- A mixture is a JSON recipe: source utterances, crops, onsets, gains, noise
  seed, target level, peak window. `speaker_count_mixing.render_mixture`
  reconstructs the waveform deterministically at load time; the manifest `wav`
  column carries a `mix:<recipes>#<utt_key>` pointer that
  `multitask_data.audio_pipeline` dispatches on.
- The label is true by construction — every source spans a common peak window —
  and re-verified from the stored intervals at prep time.

**Consequences:**
- 8.4 MB of recipes instead of hundreds of GB, and rerunning an experiment hears
  byte-identical audio.
- Rendering moves into the dataloader (a few partial FLAC reads per 5 s example),
  which the four persistent workers absorb.
- The verification is not decorative: it caught the first placement rule, where a
  source shorter than its planned span stopped covering the peak window and a
  4-speaker mixture realized 3 concurrent speakers.

### ADR-035: Normalize mixture level so energy cannot encode the speaker count (2026-09-01)

**Context:**
- The plan requires that the full count sometimes be reached for only 0.5–1.5 s
  "so global energy is not enough". That constraint was implemented, and then
  measured rather than assumed.
- On the first generated set a single RMS feature predicted the count at 48.3%
  against a 20% chance baseline (43.7% vs 25% restricted to counts 1–4, Spearman
  +0.595). More speakers means more total speech energy regardless of peak width,
  so the short-peak rule does not address this confound at all.

**Decision:**
- The speech sum is normalized to a per-mixture target level (−20 dBFS, jittered
  ±3 dB) before the noise floor is added. The zero-speaker class keeps only the
  −35 dBFS floor.

**Consequences:**
- RMS on counts 1–4 falls to 29.0% against 25% chance (Spearman +0.137). The
  small residual comes from peak limiting, which trims high-crest single-speaker
  mixtures slightly more often than 4-speaker sums.
- Counts 0 versus non-zero stay trivially separable by level. That is correct;
  silence is silence, and the zero class is a real class.
- Speech *density* still predicts the count (43.3%, Spearman +0.578). Left
  deliberately: density is a genuine cue for genuine speaker counting, and
  removing it would make the task unlike the thing it stands in for. Any claim
  about this task should be read against the density baseline, not chance.
- Level is jittered rather than fixed so the model does not train on an
  unrealistically constant level.


### ADR-036: Classification batches are rewarded by the candidate margin (2026-09-01)

**Context:**
- ADR-033 built the candidate scorer and left the GRPO reward selector offering
  only `nll`/`cer`/`wer`, none of which fit a closed-set task.
- Accuracy is the obvious reward and the wrong one. GRPO's signal is the spread
  of rewards *within* an utterance's `K=4` rollouts; four segmentations of one
  clip almost always agree on the predicted label, so a 0/1 accuracy reward
  leaves the within-group advantage at exactly zero on nearly every utterance
  and the policy receives no gradient.
- `-NLL` of the gold label is dense but not contrastive: a boundary policy can
  raise it by making the decoder more confident about everything, without the
  gold label gaining on its competitors.

**Decision:**
- Reward = gold label's length-normalized log-likelihood minus the best
  incorrect label's, under that rollout's boundaries. Selectable via
  `classification_reward: margin | nll`; generative tasks keep `segmenter_reward`.
- The rollout loop in `train_speechllm_with_segmenter.py` was refactored to call
  `_rollout_quality_reward`, whose body is the previous inline branch verbatim.
  `MultitaskSegmenterASR` overrides it. This is a **second** edit to a shared
  file, amending ADR-025, which allowed only `segmenter.py`: it is a pure
  extract-method refactor with no behavior change on the ASR path, and the
  alternative was forking a 2.7k-line `compute_forward`.

**Consequences:**
- The reward moves continuously with the evidence the boundaries preserve, so a
  segmentation that makes the right label likelier is rewarded before the
  prediction flips, and one that erodes the evidence is penalized before the
  answer breaks. Its sign still carries the 0/1 accuracy signal.
- The margin path skips the generative decoder forward the `-NLL`/CER rewards run
  first: it needs only the pooled prefix.
- Cost: `M` candidate scorings per rollout, ~30x the decoder tokens of an `nll`
  step on the 31-way intent task, ~6x on emotion. Sequences are short so this is
  affordable at the planned mixture, but it is the first thing to profile.
  Prefix KV caching across candidates is the available optimization; `nll` is the
  fallback and doubles as the ablation.
- Reward scale changes: margins are differences of mean per-token
  log-likelihoods, typically a smaller range than `-NLL`. The rate-band penalty
  is subtracted before `group_advantage` standardizes, so its relative weight on
  classification batches differs from ASR batches. Check the realized `f_audio`
  per task before reading any placement result, as the plan already requires.


### ADR-037: Task-specific rate bands as a Phase-C prior, GRPO loss otherwise unchanged (2026-09-01)

**Context:**
- Discussion on 2026-09-01: the study's success criterion is "the minimum
  frequency that retains 99%/95% of the task score", i.e. rate-minimization
  subject to a quality floor, but `rate_mode: band` gives exactly zero rate
  gradient inside `[rho_lo, rho_hi]`, so inside the band all pressure goes to
  being more correct.
- Two findings constrain any fix. (1) Under `grpo_normalize_std: True` the rate
  weight is **inert** whenever quality is flat across the K rollouts -- verified
  in this codebase across a 100x sweep of lambda, and proved as Proposition 1
  (lambda-invariance) in arXiv 2607.21273. (2) The first RL runs collapsed under
  an unbounded one-sided rate objective into a zero-variance absorbing state
  (ADR-005 failure #1).
- User decision: leave the GRPO loss as-is for now (it is what the ASR
  experiments used), get training started, and tune only the bands.

**Decision:**
- Add `task_rate_bands` (task -> `[rho_lo, rho_hi]`) and
  `task_rate_bands_phases` (default `["unfreeze"]`). Resolved through a new
  `SegmenterASR._rate_band()` hook that both rate channels read, overridden in
  `MultitaskSegmenterASR`.
- Starting priors, in `f_audio = 50 * rho`: asr/st_en_de 6-9 Hz, intent 4-7 Hz,
  speaker_count 3-6 Hz, emotion 2-5 Hz.
- Phase gating is load-bearing: Phase B's matched-rate placement claim requires
  one shared band, so per-task bands are off until the Phase-C unfreeze phase.
- Reward shape, normalization, K=4, and the two-channel structure are unchanged.
- The AALC-style accuracy-gated adaptive multiplier is deferred, with the
  lambda-inertness constraint recorded so the deferred work starts from it.

**Consequences:**
- A band is a dead zone, so `rho_hi` is the rate the policy is pushed down to
  and `rho_lo` is the collapse floor. Lowering only the floor permits a lower
  rate without encouraging one; the ASR band was therefore moved down from
  7.5-12.5 Hz to 6-9 Hz, below the ~10.5 Hz the LS960 runs realize.
- These priors are an assumed hierarchy, which the plan explicitly says Phase C
  should avoid ("set each task's band from the Phase-A fixed-rate sweep rather
  than from an assumed hierarchy"). They exist to get a first run moving and
  must be replaced by the sweep before any frontier claim. A test pins the
  ordering so a later edit has to restate the hypothesis.

### ADR-038: One definition of rho, and log the within-group reward spread (2026-09-01)

**Context:**
- `realized_kept_ratio` (reward channel) counted `1 + #boundaries`, while
  `expected_kept_ratio` (auxiliary loss) counted `sum_t p_t` over all frames.
  The two are held to one shared band but measured different quantities, offset
  by `(1 - p_0)/T` -- about 0.4 Hz on a 2.6 s clip.
- **Correction (verified on the running job, 2026-09-01):** on the production
  `transformer_ar` path the auxiliary channel is not active at all.
  `_obj_joint` sets `rl_rate = pg * 0.0` and
  `_fit_batch_combined_on_policy` hardcodes `rate=0.0`, both on the stated
  reasoning that the realized rate penalty is already inside each rollout
  reward. `rate_loss` runs only on the non-AR bernoulli path. Live confirmation
  from job 1780786: `train rate: 0.00e+00` alongside
  `train sampled_rate_penalty: 1.44e-03`.
- Neither matched `segment_pooling.compute_segment_ids`, which opens a segment
  at frame 0 and one more at every *later* boundary.
- ADR-005 failure #1 stayed invisible for seven epochs. Its signature -- all K
  rollouts identical, reward std 0, advantage exactly 0 -- was never logged.

**Decision:**
- Both functions now compute `1 + #{t >= 1 : boundary}` / `1 + sum_{t>=1} p_t`,
  matching pooling exactly. `_track_rho_from_boundary` calls
  `realized_kept_ratio` instead of reimplementing it.
- Log `reward_std_mean` and `zero_variance_share` every step and per task.

**Consequences:**
- No behavior change on the autoregressive production path *at all*: the
  auxiliary channel it would have affected is disabled there. The value is that
  the two definitions can no longer disagree if that channel is ever enabled for
  AR -- which the 2026-09-01 discussion makes likely, since it is the only
  channel where a rate weight is not lambda-invariant. The bernoulli path
  additionally stops double-counting a frame-0 boundary.
- Reported `rho`/`f_audio` are unchanged for AR runs, so past numbers stand.
  Verified live: `f_audio_hz_asr` held at 10.45-10.47 Hz across the smoke run,
  matching the warm start's ~10.5 Hz.
- `zero_variance_share` trending toward 1.0 is the collapse leading indicator;
  it costs nothing and is now visible in `train_log.txt` from step one.


### ADR-039: Speaker-count mixtures are matched to LibriCount, and validated against it (2026-09-04)

**Context:**
- The synthetic speaker-count task stalled at ~0.55 accuracy across the whole
  2x3 matrix, and it was unclear whether that was the data, the pooled-prefix
  architecture, or capacity.
- A CountNet CRNN port (validated to MAE 0.273 against its published 0.27) gave
  an external reference. It scored **0.927** on real LibriCount and **0.564** on
  our mixtures. The data was the dominant problem, not the architecture -- the
  opposite of what a forward-only transfer test had suggested.

**Decision:**
- Match the generator to LibriCount's measured construction, verifying each
  change against the reference rather than by argument. Final profile:
  `--profile libricount --placement libricount --vad-labels`.
- Keep the original `hardened` profile as a controlled ablation.
- The generator profile is recorded in `speaker_count_prep.json`, and the
  launcher refuses to start without it.

**What actually mattered, in order (CountNet accuracy on our data):**
1. **0.564 -> 0.491**: equal-power + no mixture normalization + random offsets
   ALONE made it *worse*. Level policy was not the cause.
2. **-> 0.794**: the dominant bug was ours -- a -35 dBFS Gaussian floor added to
   *every* mixture, not just the zero class. At 10 dB SNR against equal-power
   sources it made single-speaker clips look multi-speaker: count-1 accuracy
   0.55 -> 0.98 once lowered to -60 dBFS. LibriCount adds no floor at all.
3. **-> 0.873**: LibriCount's own VAD annotations (Zenodo 1216072) show every
   speaker vocalizes ~93.5% of the 5 s clip and all N overlap for 77-94% of it.
   Our `random_offset` guaranteed only a single instant of N-way overlap.
   Selecting mostly-voiced full-clip crops matched the duty (93.6% vs 93.5%).
- Hypotheses killed by evidence along the way: edge transients (25 ms fades
  changed nothing), and VAD labelling on its own (relabelling moved 0/2500
  recipes once placement already forced overlap).

**Consequences:**
- Counts 0-3 now match the benchmark within noise; count 4 remains 0.54 vs 0.79
  and is unexplained. Not a blocker, but it should be chased before a
  Dynamic-SUPERB submission.
- **LibriCount is built from LibriSpeech `test-clean` and IS the Dynamic-SUPERB
  eval set.** It can never be training data here. Our generator draws from the
  train splits, which keeps both the submission and our LibriSpeech ASR eval
  clean, and keeps the task pack a genuine held-out test.
- Any speaker-count number is meaningless without its profile: the same
  reference model scores 0.564 and 0.873 on the two.
- NA-1's speaker-count results are superseded, not a baseline: they came from
  the hardened profile.


## Tips

- Number decisions sequentially (ADR-001, ADR-002, etc.)
- Be honest about trade-offs
- Update decisions if they're revisited/changed

### ADR-040: Single-task fixed-rate baselines isolate the bottleneck; checkpoint selection moves to the task's own metric (2026-09-04)

**Context:**
- The multi-task runs conflate three candidate bottlenecks for the non-ASR
  tasks: the audio-token rate, interference from the task mixture, and the
  pooled-prefix interface itself. Speaker count sits at 0.730 against an
  external reference's 0.871 on the same test split (ADR-039), and nothing in
  the multi-task design says which of the three is responsible.
- ASR-WER-only checkpoint selection made every multi-task run report a
  pre-policy checkpoint, so its rates are the warm start's and carry no
  information about the run. A single-task non-ASR run has no ASR source at
  all, and the trainer raised outright: "the validation mixture contains no ASR
  utterances ... checkpoint selection has nothing to rank".

**Decision:**
- Add `selection_metric` to the multi-task recipe: rank checkpoints on a named
  validation metric (`macro_f1_emotion`, `acc_speaker_count`, `acc_intent`)
  instead of ASR WER. Implemented as `_SelectionErrorRate`, which expresses the
  metric as `100*(1-value)` and feeds the existing `min_keys=["WER"]`
  machinery, so there is exactly one code path deciding what "best" means.
  Unset (the default) keeps the ASR-retention behaviour for multi-task runs.
- Run five single-task arms per task that differ in **nothing but the
  segmentation**: fixed grids at k=1/2/5/10 (50/25/10/5 Hz) plus the warm
  start's learned Transformer-AR policy held frozen at argmax.
- `fixed_rate_k: 1` is the no-downsampling arm -- `fixed_rate_boundary_targets`
  marks every frame a boundary, so one mechanism covers all four fixed rates
  and no separate no-downsampling code path is needed.

**Why these rates:** 10 Hz is what the reported multi-task checkpoint actually
emitted (7.6-10.6 Hz across tasks) and 5 Hz is what its Phase C compresses to
(5.4-5.6 Hz), so two arms sit exactly on the learned operating points. 50 Hz is
the interface's no-compression ceiling and 25 Hz fills the gap, giving a 4-point
rate-quality curve at logarithmically spaced k.

**Two learned arms, not one.** `learned_frozen` holds the policy at the warm
start, which makes it comparable in both directions -- to the fixed arms
(segmentation is then the only variable) and to the published multi-task number,
whose selected checkpoint is itself a frozen-policy bridge checkpoint. It
isolates multi-task interference. `learned_grpo` then adds exactly one thing on
top: the policy is trained, with K=4 GRPO, mirroring the multi-task schedule
(bridge 20% / FiLM 40% / unfreeze 40%, per-task band in Phase C) at the *same*
total step budget as the frozen arm. Frozen vs GRPO at matched steps is
therefore a clean answer to whether training the boundary policy buys anything,
which the multi-task runs could not give: there the ASR-selected checkpoint was
always pre-policy, so no reported number ever came from a trained policy.

`learned_grpo` waits on its own `learned_frozen` counterpart per (task, seed)
via `--dependency=afterany`, so the two do not contend for GPUs and a frozen
failure cannot strand the GRPO job. The per-task bands live in the yaml rather
than on the command line because `EXTRA_ARGS` reaches the trainer unquoted and
is word-split, which would break any YAML mapping containing a space; a
single-task run only ever reads its own row.

**Controls that make the arms comparable:**
- Matched effective batch of 300 s audio per optimizer step in every arm. The
  long-sequence arms split the physical batch (k=1: 75 s x 4, k=2: 150 s x 2)
  rather than shrinking it, so rate is not confounded with batch size.
- `multitask_batches_per_epoch` counts dataloader batches, not optimizer steps,
  so the epoch bound is multiplied by the accumulation factor. Without that the
  accumulating arms stop early and silently -- the failure mode that ended job
  1780790 at step 765 of 15000.
- Shared warm start, shared learning rates (decoder 2e-4, segmenter/pooler
  5e-5), and one step budget per task across all its arms.

**Step budgets, sized to the data rather than copied:** emotion 1200 steps
(~30 epochs over 4,702 clips), intent 1500 (~8.5 epochs over 23,132; the task
already saturates at 0.994), speaker count 4000 (~12 epochs over 20,000
mixtures; the hardest task and the study's subject). Ten validation points each,
with best-checkpoint selection standing in for early stopping.

**Consequences:**
- 54 jobs (3 tasks x 6 arms x 3 seeds), each a single A100.
- A flat rate-quality curve would mean the rate is not the bottleneck and the
  interface is; a gap between `learned_frozen` and the published multi-task
  number at matched rate would mean interference is.
- The `selection_metric` mechanism is also the fix the multi-task runs need, but
  changing their criterion would break comparability with the reported cohort,
  so it stays opt-in.
- Verified that a single-task run reports nothing but its own task: the intent
  run's metric keys are exactly `acc_intent`, `macro_f1_intent`,
  `f_audio_hz_intent`, `n_intent` -- no `WER_asr`, no other task's accuracy and
  no BLEU -- and all three samplers are `fsc_intent` at p=1.000.

### ADR-041: ICASSP Method presentation and LaTeX prose style (2026-09-09)

**Context:**
- The Method must explain why each ASSET component exists, especially how ASR feedback reaches a discrete boundary policy, without implying an external reward model.
- Encoder frame rate and segmenter architecture are framework choices rather than fixed assumptions, and mean pooling should not be described as erasing order already encoded in contextual features.

**Decision:**
- Do not hard-wrap prose or insert manual line breaks in the LaTeX manuscript; keep each logical prose paragraph on one source line and let TeX perform visual line breaking.
- Present ASSET as coupled continuous and discrete training paths: CE back-propagates through the pooler/projection/decoder, while detached recognizer returns produce a GRPO score-function gradient through policy log-probabilities.
- Define the segmenter explicitly as a per-frame boundary policy with selectable acoustic and optional boundary-history backbones. Express output rate generically as $f_{audio}=\rho f_{enc}$ rather than assuming a 50-Hz encoder.
- Keep the three-stage optimization curriculum in Method because it makes the policy feedback well-conditioned; reserve dataset-specific stage lengths for Experimental Setup.
- Expand and cite bidirectional gated recurrent unit (BiGRU) and GRPO at first use. State that contextual features retain sequence context, while mean aggregation itself is permutation-invariant within a fixed segment.

**Consequences:**
- Future Method revisions should preserve the motivation, gradient-routing, architecture-generality, and pooling-precision language unless the implementation or paper framing changes.

### ADR-042: ICASSP experimental evidence hierarchy and page-budget presentation (2026-09-09)

**Context:**
- The four technical pages must still identify data, base models, trainable components, training schedule, hardware, metrics, principal LS960 results, controlled attribution, adaptivity, and full-path efficiency.
- The learned LS960 systems use 24k updates and a stronger decoder initialization, while the native/fixed/oracle reference rows use shorter decoder adaptation. The matched train-clean-100 study is the valid causal comparison.
- A separate plot for every analysis would push technical content onto page 5 and expand the bibliography beyond the permitted reference page.

**Decision:**
- Use one generated two-panel full-width table: panel (a) gives the LS960 system-scale comparison; panel (b) gives the matched 100h fixed/frozen/joint attribution. State the LS960 schedule caveat in the caption and prose rather than implying that every gap comes from learned boundaries.
- Present adaptive allocation and inference efficiency as compact quantitative paragraphs. Report both the selected stable batched RTF and the adverse batch-one Transformer result; claim batched throughput only, not latency or streaming.
- Keep the experiment setup in three labeled blocks (data/models, training/baselines, metrics) and the results in three labeled blocks (recognition, controlled attribution/adaptivity, efficiency). Retain 16 directly used references so pages 1--4 contain all technical content and page 5 contains references only.
- Generate the table from saved artifacts via `drafts/icassp/scripts/make_table_results.py`; do not hand-edit `drafts/icassp/tables/main_results.tex`.

**Consequences:**
- Future result updates should change the underlying artifacts or generator and rerun `make -C drafts/icassp tables`, preserving the distinction between system comparison and controlled attribution.
- Adding a new figure or baseline likely requires replacing existing content rather than simply appending it under the current page budget.

### ADR-043: Present the broader 100h evidence and shorten paper efficiency (2026-09-09)

**Context:**
- The user requested removal of the Metrics paragraph, fuller presentation of existing 100h experiments, and a shorter efficiency discussion centered on LLM savings exceeding downsampling overhead. This supersedes ADR-042's presentation-only restrictions.

**Decision:**
- Keep the generated two-panel table, adding fixed-k=5 mean pooling to the seven-epoch 100h controls. Add a column-width figure of the five fixed strides, CNN/Transformer mean/BiGRU systems, and audio-only phone-CTC on both test splits, all with three seeds and corrected decoding.
- Keep the longer 100h system sweep (Transformer/BiGRU 4.80/9.24 WER) distinct from the seven-epoch joint control (4.95/9.11); identify differing adaptation budgets in the figure caption and reserve attribution for matched comparisons.
- Remove the standalone Metrics paragraph; define WER/rate/uncertainty beside their first use. Limit efficiency to the explicitly batched, complete measured path on 100h checkpoints and its 2.8×/2.7× net throughput gain. Preserve the batch-one context and detailed protocol in the revision note rather than a defensive manuscript ending.
- Recompute table SDs from integer edit counts where available; do not force historical rounded SDs. Describe the 960h 300-second batches using the actual accumulation-1 runtime override, not the saved YAML default of 4.

**Consequences:**
- `drafts/icassp/scripts/make_figure_100h.py` generates PDF/PNG plus a JSON evidence record; the Makefile builds both table and figure from their source artifacts. Prose was tightened to retain four technical pages and a fifth references-only page at existing manuscript font sizes.
- Full rationale and source paths: `drafts/icassp/research/revision_2026-09-09.md`.

### ADR-044: Emphasize adaptive rate and add boundary-placement evidence (2026-09-09)

**Context:**
- The author requested split-specific oracle cells, clearer Transformer/BiGRU rate emphasis, less crowded plot markers, more results, a shorter setup, and a fuller conclusion focused on additional tasks and task-specific policies.

**Decision:**
- Replace the merged approximate oracle-rate cell with 14.6/14.3 Hz from corrected test measurements. Reuse only the token counts of the shared precomputed char boundaries; validate their source against the 960h configs and retain the 960h WER.
- Bold the lower-rate ASSET backbone in Table 1(a), explicitly scoped in the caption. State the Transformer/BiGRU reduction from CNN/BiGRU at both scales: 8–9% at 960h and about 13% at 100h.
- Use paired full-sweep and policy-detail plots for each test split. Preserve all ten configurations, actual data positions, three-seed uncertainty, and at least 9-pt labels.
- Add the dev-clean phone-boundary diagnostic (one-to-one matching within ±20 ms). It supplies placement evidence alongside WER and input-dependent rate, without implying that phone-boundary F1 is the training objective.
- Introduce ASSET as adaptive in the abstract. Retain one exact headline WER pair plus a rounded token rate; omit exact results from the expanded conclusion. Future work is additional speech tasks and task-specific policies.
- Preserve the matched/unmatched experiment distinction and the concise net batched-efficiency claim from ADR-043. Shorten setup/captions to keep all technical content on pages 1–4.

**Consequences:**
- Table/figure generators and dependencies now track oracle rate/config evidence and the boundary diagnostic. The figure JSON records recomputed boundary F1 and reduction statistics.
- Rationale, sources, and final render checks: `drafts/icassp/research/revision_2026-09-09_round2.md`.


### ADR-045: Fresh matched 100h BiGRU boundary controls (2026-09-09)

**Context:**
- The author retains Character oracle as the transcript-assisted empirical reference cap and requests phone-CTC + BiGRU and char-boundary + BiGRU controls on another cluster.
- The author explicitly rejects continuing the old phone runs because transferring experiment state costs too much time.

**Decision:**
- Run both arms fresh on train-clean-100, seeds 3407/3408/3409, seven CE epochs each. Preserve the shared character-decoder initialization, BiGRU architecture, optimizer settings, accumulation 4, and dev-clean checkpoint selection of the existing matched controls.
- Transfer no previous phone-run checkpoint, optimizer state, or history. Set both policy initialization paths null: external boundaries bypass the frozen policy, so its cold-start weights are unnecessary. Keep the common decoder initialization (approximately 79 MB) for comparability.
- Reuse the existing phone longsplit8 and transcript-derived char boundaries at their natural rates. Keep the oracle label/test-transcript disclosure and report its measured outcome.
- Write portable commands, asset requirements, scheduling estimates, and checks in `plans/icassp_bigru_controls_100h_2026-09-09.md`. Execution belongs to the destination cluster; no local submissions or 960h runs are requested.

**Consequences:**
- Six fresh jobs require 42 new CE epochs. This supersedes the initial continuation proposal and the comparability review's suggestion to rename the oracle.
- The new results will extend the matched seven-epoch 100h table, distinct from the longer 100h sweep and the 960h system comparison.


### ADR-046: Cite verified sources and qualify the implemented GRPO variant (2026-09-09)

**Context:**
- The user requested an audit of every paper citation, including DOI/arXiv targets, venues, author lists, and missing attribution.
- All 24 original bibliography entries identify real works; all nine original DOIs and seven explicit arXiv identifiers match their titles. The substantive problems were venue/method attribution and model-version specificity.

**Decision:**
- Cite the exact official Llama-3.2-1B-Instruct model card for the decoder; keep the corrected Llama 3 paper as an unused library entry with the current arXiv author prefix.
- Identify AdaTS as parameter-free similarity-threshold grouping, and name its ICLR 2026 Multimodal Intelligence workshop. The accessible PDF verifies the method; coauthor André Martins's announcement explicitly distinguishes this workshop presentation from his main-conference papers.
- Credit DeepSeekMath for group-relative advantages and Williams for REINFORCE. Describe ASSET as an on-policy GRPO variant with fresh samples, one update per group, and no importance-ratio clipping/reference-policy KL; do not imply that the original DeepSeekMath objective is used unchanged.
- Cite Jiang et al. (Interspeech 2021) for prior RL frame-rate control, Bhati et al. for fixed audio-token pooling, Graves et al. for CTC, and Loshchilov/Hutter for AdamW.
- Keep full verified bibliographic metadata and dated source evidence in the citation audit. Use given-name initials and abbreviated IEEE ICASSP/ASRU venue names in print; clickable titles point to the canonical URL/DOI. Preserve the official IEEEbib.bst and use the derived IEEEbib-linked.bst.

**Consequences:**
- The paper has 21 cited references, with technical content on pages 1–4 and references only on page 5. No experimental results or figure/table assets changed.
- Access limits are explicit: live OpenReview metadata was blocked; paper identity and venues were checked through accessible primary PDFs, author announcements, and official conference programs.
- Evidence and per-reference findings: drafts/icassp/research/citation_audit_2026-09-09.{md,json}.


### ADR-047: Separate Figure 1 beta and semantic visual conventions (2026-09-09)

**Context:**
- The author requested an editable `_beta` copy of the draw.io overview with clearer organization, more illustrative detail, and a logical color/shape system.

**Decision:**
- Keep `JSALT26_beta.drawio.xml` and its exports as a separate review candidate. The manuscript continues to use the existing manual Figure 1 export.
- In the beta, gray denotes fixed/frozen components, teal the continuous recognition path, violet the boundary policy, and amber scalar feedback. Rounded shapes are modules, square cards are losses/feedback/gradients, circles are residual addition, and dashed arrows are parameter updates.
- Show projection and LoRA explicitly; separate greedy CE boundaries from sampled policy-training boundaries; put the zero-initialization label on the residual output projection, not the BiGRU weights.
- Generate native editable objects and render the XML itself to PDF/SVG/PNG. Group module labels and shapes. `make figure-beta` uses `--render-only` to preserve subsequent manual edits.

**Consequences:**
- The beta has a roomier 178 × 90 mm canvas with labels at least 9.25 pt; adopting it will require checking the manuscript layout.
- Design/render notes: `drafts/icassp/figures/JSALT26_beta.md`. The generator is `make_asset_overview_beta.py`; the XML records the original source hash.

### ADR-048: Consistent GRPO terminology and unambiguous loss weights (2026-09-10)

**Context:**
- The author rejected the Figure 1 beta and requested consistent GRPO wording, identifying the omitted term as GRPO's reference-policy KL regularizer.

**Decision:**
- Use GRPO throughout the manuscript, including abstract, overview caption, Method, and training schedule. Remove the REINFORCE label/citation and repeated “variant” framing; this supersedes ADR-046's terminology decision.
- State that the reference-policy KL coefficient is `$\beta=0$`. Preserve the implemented policy-gradient expression and the fresh-sample, one-update-per-group description. Rename the separate joint-objective policy-loss weight from beta to alpha.
- Retain the verified Williams entry as unused bibliography metadata. Preserve historical citation-audit evidence; do not change training code or imply a different optimizer.
- Archive the rejected beta as unused; the manuscript continues to use the author's original Figure 1.

**Consequences:**
- Future edits should call the method GRPO with zero reference-policy KL regularization and reserve beta for that coefficient. Alpha denotes the `pg_weight` mixing coefficient.


### ADR-049: Preselected LS960 segment-unit and inter-word-pause pilot (2026-09-10)

**Context:** A collaborator requests a linguistic/acoustic interpretation of the abstract's approximately 11 Hz segments and a check of inter-word silences at least 250 ms long.

**Decision:**
- Preserve the pre-audit paper in commit `e312a2677`, selectively tracking sources, bibliography, the active figures, and their generators; exclude build intermediates and unused assets. The first origin push lacked write access; alternate-remote approval is pending.
- Audit the actual three WER-selected LS960 Transformer-AR/BiGRU checkpoints without the decoder or retraining. Select minimum retained dev-clean WER by checkpoint metadata, not ambiguous epoch counters.
- Freeze a 12-speaker random utterance sample plus a separate 12-speaker pause-enriched sample before inference (RNG 20260910). Analyze cohorts separately, with exact-count uniform grids and shifted-boundary controls.
- Compare internal cuts to continuous-time MFA phone/word edges at ±20/40 ms; exclude obligatory utterance endpoints and check the +12.5 ms timestamp convention. Measure both pure/mostly-silent token coverage and pause fragmentation. Broadband RMS minima do not establish harmonic-energy or demi-syllable identity.

**Consequences:** Job 1790396 runs one A100 on e03; artifacts stay in `artifacts/segmenter/ls960_boundary_audit_2026-09-10/`. This is a descriptive 24-utterance pilot, with repeated model seeds, not a full-corpus or causal RL interpretation study. Paper claims must await measured outcomes.


### ADR-050: Restore manuscript; investigate pause mechanism before new claims (2026-09-10)

- The author rejected the pilot-based wording. Restore `drafts/icassp/main.tex` and its README exactly to `e312a2677`, rebuild the original PDF, and retain the audit as separate exploratory evidence. Do not push the rejected wording or reinsert new claims without author review.
- The follow-up finds the character-CTC supervision already leaves 28/31 pauses uncut, with 1/31 wholly silent segments; final policies have 26–31/31 and 0–1/31. Prioritize tracing targets → actual initialized policy → retained training stages before attributing the pattern to RL.
- A stronger study combines held-out full-corpus statistics with speech-allocation-matched controls, checked pause references, and decoder-scored boundary interventions. The rate band is free internally: do not invoke a per-token tax. Plan: `plans/segment_interpretation_followup_2026-09-10.md`. No full-corpus or decoder intervention job has been launched.


### ADR-051: Cautious phone-boundary manuscript interpretation (2026-09-10)

- The author requests a new version using some LibriSpeech and TIMIT phone-boundary evidence, with an approximately phone-like interpretation and shorter training-detail prose. This authorizes a new local review candidate after ADR-050's restoration.
- Add one abstract sentence and extend the existing Boundary placement paragraph using the full-split diagnostics from the same three 100h-trained Transformer/BiGRU checkpoints: LibriSpeech dev-clean 63.1±1.2% versus fixed-k5 56.9%; TIMIT TEST 66.9±2.5% versus 57.8%, at one-frame (20-ms) tolerance. Keep this scope distinct from the 960h recognition headline.
- Say the partial correspondence is consistent with approximately phone-like units and that boundary agreement alone does not establish token identity. Do not reinsert the rejected pause-pilot claims or infer phoneme discovery or a causal GRPO effect.
- Shorten the setup by omitting exact ratio-band, per-stage step/epoch, and batch-duration details. Preserve the broad-sweep versus matched-control distinction, optimizer settings, and model/method definitions. Add the official TIMIT corpus citation.
- Retain the historical audit values with their endpoint convention documented in the evidence note; do not interpret cross-corpus F1 differences or call the fixed grid exactly count-matched. A future standardized audit should exclude obligatory endpoints consistently.
- Source diff and evidence: `drafts/icassp/research/phone_boundary_revision_2026-09-10.{diff,md,json}`. No remote push is authorized by this wording request.


### ADR-052: Measure structural departures from phone segmentation before naming the units (2026-09-10)

- The author asks to replace approximately phone-like wording with a more informative interpretation supported by further pattern analysis. Freeze five utterances per speaker on each LibriSpeech test split (365 total), plus full TIMIT TEST minus SA (1,344). Evaluate three 100h checkpoints everywhere and three 960h checkpoints on LibriSpeech.
- Retain cuts and annotations; analyze genuine within-phone splits away from uncertain edges, phone-duration/class dependence, spans of multiple phone midpoints, equal-count uniform controls, and encoder-feature changes. Score obligatory endpoints consistently and report timestamp sensitivity.
- Extract WavLM features per utterance; batch only policies, with a batch-sensitivity check. One A100, two-hour limit, no decoder or retraining, estimated outputs below 100 MB under artifacts/segmenter/ls960_phone_patterns_2026-09-10.
- Avoid framing non-phone cuts as causally better or claiming superiority to an exact-phone oracle. Determine manuscript wording only after examining measurements.


**ADR-052 measured outcome and manuscript choice:** The completed 1,709-utterance audit supports sensitivity to phonetic transitions with flexible token allocation within/across phones. Replace the approximately phone-like sentence; add the 960h LibriSpeech 11% deep-phone-split and 15% multi-phone-center-token statistics. Keep existing frame-grid F1 values with explicit ±1-frame scoring. Do not claim acoustic-detail preservation or causal WER superiority to phone boundaries. Timing-shift and feature-change controls limit interpretation. Exact review diff and evidence: drafts/icassp/research/phone_patterns_revision_2026-09-10.{md,diff}.


### ADR-053: Retain WER and the qualified oracle comparison in the abstract (2026-09-11)

- The author selects the proposed sentence with 2.79/5.63% WER, about 11 audio tokens/s, comparable performance to a transcript-assisted character-boundary oracle, and approximately 25% fewer audio tokens. Replace only the abstract's numerical result sentence; retain the cautious phone-pattern wording.
- Table 1 rates imply 25.34% / 25.87% fewer tokens on test-clean/test-other. This is a complete-system comparison with a transcript-assisted reference, not a recognition upper bound or an isolated estimate of boundary-policy benefit; the table retains its pooling and adaptation disclosures.
- The abstract is 150 words. Review diff: drafts/icassp/research/oracle_abstract_revision_2026-09-11.diff. Rebuilt PDF remains five pages; pages 2–5 render identically to the preceding version. No remote push.


### ADR-054: Explain disjoint loss gradients and report their fixed unit weights (2026-09-11)

- The author questions the generic CE + alpha GRPO objective because the losses back-propagate to different parameters. Verified current and archived production code: greedy boundaries/rewards are computed without gradients; CE trains pooler/projection/LoRA, while detached-advantage GRPO re-scoring trains the boundary policy.
- All 21 learned-run training configurations represented in the paper set `pg_weight=1.0`: 12 100h sweep runs, three matched 100h joint runs, and six LS960 runs. Replace the final paragraph of Section 3.3 with explicit gradient routing and unit weights in one backward pass. This supersedes ADR-048's presentation of an unspecified alpha; retain beta = 0 for reference-policy KL.
- A summed loss is valid computational shorthand despite disjoint gradient paths. A nonunit alpha would scale the raw policy gradient before clipping/AdamW; do not equate it exactly with a learning-rate change. The single AdamW's first parameter group contains both boundary policy and pooler, and SpeechBrain clips that group together.
- Evidence: drafts/icassp/research/gradient_routing_evidence_2026-09-11.json; exact edit: gradient_routing_revision_2026-09-11.diff in the same directory. No training-code change or push.


### ADR-055: Use concrete speech-processing terms instead of “interface” (2026-09-11)

- The author dislikes “interface” as vague wording. Prefer adaptive downsampling for the overall reduction operation, adaptive segmentation for boundary prediction, segment pooling for aggregation, and acoustic sequence length for the computational bottleneck. Avoid implying frame selection or phone identity.
- Replaced the three prose occurrences in the introduction, Section 3.1 heading, and conclusion. Figure 1(a) now reads “Adaptive downsampling.” Primary examples of the related terminology are Prabhavalkar et al., frame-rate reduction (https://arxiv.org/abs/2402.17184); Jiang et al., variable frame rate (https://www.isca-archive.org/interspeech_2021/jiang21b_interspeech.html); and CJST, sequence compression (https://arxiv.org/abs/2411.07607). No bibliography change is needed.
- Preserve the author's manual PNG. Both editable XML copies contain the new title; scripts/make_overview_title.py renders that native text into an 8.6 KB vector PDF, and figures/JSALT26.tex places it on the original white title tab. The Makefile tracks both dependencies. This avoids redrawing or altering the rest of the figure.
- Exact diff: drafts/icassp/research/terminology_revision_2026-09-11.diff. Abstract and results unchanged; five-page PDF validated and changed pages inspected. No push.


### ADR-056: Audit non-ASR task adaptation on shared audio before changing the paper (2026-09-12)

- The author requests quantitative single-task non-ASR results, cross-task boundary analysis including Expresso, and a concrete manuscript proposal before any paper edits. Hold all manuscript files fixed; snapshot 54 source/bibliography/figure/PDF files.
- Audit 54 saved TEST records and selected checkpoints. Use the common LS960 ASR parent, three emotion policies, selected intent/count policies, other ASR seeds, a frozen identity control, and separately labeled last-step intent/count sensitivity checkpoints (15 identities). Do not mix last-step rates with selected-checkpoint quality.
- New inference covers full CREMA-D voice-vote test (1431 utterances/19 speakers), matched-text Expresso short reads (420; four speakers, seven styles, 15 texts per speaker), and the prior 365-utterance held-out LibriSpeech selection. Additional task policies use a preselected 342-utterance CREMA-D panel; all policies share the full Expresso/LibriSpeech audio.
- Use one-to-one continuous-time boundary matching, exact per-utterance count controls, circular shifts, random ASR thinning and highest-confidence ASR thinning; exclude compulsory endpoints. Check +12.5-ms timestamps, original-dictionary-covered utterances and low-likelihood alignment removal. Automatic vowel onsets and Praat voicing are diagnostics, not syllable/prosody ground truth.
- Emotion is the cleanest paper extension: all selected checkpoints include final-block updates. Selected speaker-count checkpoints and intent seed 3408 are FiLM-only despite a stale TEST phase field. Task-specific Phase-C bands are priors (emotion 2–5, count 3–6, intent 4–7 Hz), so do not describe resulting rates as unconstrained task optima.
- Deliver the local review, scientific figures and generated table under artifacts/segmenter/nonasr_boundary_audit_2026-09-12/. Proposed addition: compact emotion table and a cautious paragraph near the end of §4.2, preserving the current ASR/oracle abstract. Paper changes remain unapplied pending the author's review.


### ADR-057: Publish the non-ASR audit as an analysis sub-tab (2026-09-13)

- The author requests a published HTML sub-tab containing all completed non-ASR results and analysis. Add Analysis 03, “Non-ASR boundaries,” under Analyses, with a training-report link, six figures, 16 numbered tables, exact numerical downloads, limitations, and the unapplied paper proposal.
- Regenerate the training report and replace the older claims of free compression, causally improved stability, weak count rate pressure, and excluded bottlenecks with the September 12 evidence. All 35 cumulative numerical tables remain identical.
- Publish source-only report/evidence history from an isolated checkout at the existing asset-dev/bilevel-optimization tip, excluding local unpublished manuscript commits. Published source: c0f7407a5 (preceded by 35862b756); current local branch checkpoint: 5810262fb. Pages: bef0f5964910dd71b66e19d9ac9e6f7bdde4060e. No manuscript files changed or pushed.
- Validation: standalone and whole 11-page site; desktop/mobile light/dark rendering, filters, downloads, sub-tab/iframe navigation; sequential captions and source-CSV agreement; successful Pages workflow 34775608028. All 13 live publication files match local SHA256.

### ADR-058: Refocus Expresso analysis on locations and expressive variation (2026-09-13)

- The author prioritizes same-audio cross-task cuts, same-speaker/text expressive renditions, and acoustic/linguistic interpretation over recognition performance. Preserve all manuscript files; revise the published analysis after examining new evidence.
- Freeze the exploratory protocol in artifacts/segmenter/expresso_boundary_patterns_2026-09-13/protocol.md. Reuse all 420 original Expresso utterances and raw cuts, with primary final-block-adapted emotion, last-step intent and last-step count policies; selected FiLM-only policies remain controls. Separate prescribed rate bands and checkpoint stage from task causality.
- Compare sparse policies on common ASR anchors, transition-class retention, and acoustic edge preferences. Same-text comparisons use independent word/phone correspondences with within-phone randomization, clock-grid and timing sensitivity controls; inferred alignment agreement is not phoneme discovery.
- Natural panel: 60 speaker×text groups, four speakers, seven renditions. Main affective contrasts are default/happy/sad/confused; whisper/laughing/enunciated are additional speaking styles. All word sequences agree; 2306/2418 corresponding words have matching stress-stripped phone sequences. Restrict local phone maps accordingly and report exclusions.
- Add a controlled probe on the 60 default recordings: original, Praat resynthesis sham, ±3-semitone pitch shifts, 0.8/1.25 duration factors, flattened pitch, bounded envelope compression. CPU preparation job 1792712; four-core analysis 1792792; frozen decoder-free inference on one A100 job 1792793. Outputs under the new artifact directory, estimated under 500 MB. Pitch/duration contrasts use the sham as their reference.
- Relevant primary work motivates measurements: Oganian & Chang (2019) on envelope rise landmarks; Bhati et al. (2021) on phonetic transition classes; Mo (2008) on duration/intensity and prominence/phrasing; Expresso (Nguyen et al., 2023). These are exploratory diagnostics, not a claimed replication or neural-mechanism analogy.


**ADR-058 measured outcome and publication:** The420-utterance audit shows directional retention: countlast74.8%/2.1% at consonant→vowel/vowel→consonant parent cuts, emotion3408 75.2%/36.5%. Same-text emotion agreement remains67.7–75.2% across the three main affective renditions after local phone mapping, above within-phone placement controls.480 controlled inputs show duration-dependent allocation (×1.25 duration→1.170 emotion token ratio), with most cuts preserved under pitch shifts. These motivate a proposal about selective transition retention and temporal flexibility, not a new unit identity or causal task-reward claim. Published as the existing Analysis03 URL with eight figures, two interactive explorers, eight tables, PDF/numerical/script downloads, and the earlier performance audit as supporting material. Source66ae664c1; Pages45ceb4e; workflow34781281482 succeeded. All54 manuscript snapshot files remain unchanged.

### ADR-059: Let report text wrap naturally (2026-09-13)

- The author requests no manual line breaks in reports and asks that this preference persist in project memory.
- Use semantic paragraphs with normal browser wrapping for prose, headings and captions. Remove artificial text-only width caps and balanced heading wrapping; do not insert forced HTML line breaks or preserve source newlines for layout.
- Apply this to the Expresso analysis generator and republish the same sub-tab. This is a presentation-only change; findings, figures, tables and manuscript text are unchanged.
