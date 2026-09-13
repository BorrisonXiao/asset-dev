# Bug Log

This file logs bugs and their solutions. Keep entries brief and chronological.

## Format

Each bug entry should include:
- Date (YYYY-MM-DD)
- Brief description of the bug/issue
- Solution or fix applied
- Any prevention notes (optional)

## Entries

### 2026-09-04 - Held-out subsets aliased against a class-interleaved manifest
- **Symptom**: speaker-count validation read 0.903 while the *same checkpoint*
  scored 0.689 on test, and macro-F1 (0.943) exceeded accuracy (0.903).
- **Cause**: `_stride_subset` took evenly spaced indices. The speaker-count
  manifests are interleaved by label with period 5; 600-of-2000 strides by 3.33
  and hits only residues {0,3,1}, so validation contained three of five classes
  and never a `two` or `four`. Checkpoint selection ranks on validation, so all
  15 runs selected against a 3-class signal (each picked its first validation
  point).
- **Why it stayed hidden**: the multi-task runs used 1000-of-2000, which strides
  by exactly 2 and covers all five classes. The bug appeared only when the
  subset dropped to 600 in the budget cut, i.e. it was introduced by a change
  that looked purely economic.
- **Ruled out first, in order**: subset class balance (a 600-row *prefix* is
  balanced 120/class, so the manifest looked fine), speaker leakage (train/valid/
  test are 2338/40/40 with zero overlap), test-manifest wiring (2000 rows, all
  from the test CSV, balanced), candidate-set drift (built once at startup and
  shared), and split difficulty -- CountNet scores 0.860 valid vs 0.871 test, so
  the splits are equally hard and the gap was ours.
- **Fix**: seeded permutation instead of a stride -- equally deterministic,
  cannot alias against any ordering, manifest order preserved. Regression test
  in `test_multitask_data.py` asserts every class survives and stays
  near-balanced at several subset sizes; it fails on the old implementation.
- **Prevention**: a "representative subset" check must confirm the *label*
  distribution, not just the row count. Any deterministic subsampler applied to
  a grouped or interleaved manifest is a candidate for this failure.


### 2026-09-04 - Closed-set candidate scoring OOMs at high frame rates
- **Symptom**: `torch.OutOfMemoryError: Tried to allocate 29.70 GiB` on an 80 GB
  A100, in `score_candidates` -> `sequence_log_likelihood` -> `F.cross_entropy`,
  during the TEST pass of the no-downsampling (`fixed_rate_k: 1`, 50 Hz)
  single-task baseline. Training itself completed; only evaluation died
  (job 1784456).
- **Cause**: `score_candidates` padded the label targets with `ignore_index`
  across the whole audio prefix and passed the *full* sequence to
  `sequence_log_likelihood`, which does `logits.reshape(-1, V).float()`. With a
  128,256-token vocab that float32 copy is `rows x (n_audio + n_label) x V`,
  and the audio positions -- 250 of 254 at 50 Hz -- contribute exactly zero to
  both the sum and the token count. ~98% of a 12 GB tensor was pure waste.
- **Fix**: slice the audio prefix off instead of padding it:
  `sequence_log_likelihood(logits[:, n_audio:], targets, ...)`. Mathematically
  identical (verified against the padded form for ragged labels and both
  `length_normalize` settings); the cross-entropy tensor drops from ~254 to ~4
  positions. Also scaled `candidate_chunk_size` per arm (32 at 50 Hz, 64 at
  25 Hz, 256 below) because the LLM's own `.logits` is still
  `rows x positions x 128256`.
- **Prevention**: candidate scoring cost scales with the *audio* rate, not just
  the batch, so any new arm that raises the frame rate must shrink the chunk.
  The launcher now carries the chunk size in its arm table alongside the batch
  split, so the two cannot drift apart. Also note the multi-task runs never hit
  this: at ~10 Hz the wasted tensor was ~5x smaller.


### 2026-08-31 - `artifacts/segmenter/report.html` and the published Pages copy have both drifted ahead of the generator
- **Issue**: With the `html-report` skill restored on CLSP, `make_training_report.py` runs clean again -- but its output is NOT what is currently published. Three versions exist: the generator emits the plainest text; `artifacts/segmenter/report.html` (written 2026-08-30 16:07) additionally carries a hand-edited headline ("15.9% / 13.3% relative WER reduction ... demonstrates the effectiveness of learning content-adaptive boundaries" instead of the generator's "lowers WER by 0.53 points"); and the live Pages copy additionally carries an entire extra section, "Native TIMIT phone-boundary agreement" (Table 5, shifting Tables 5-N to 6-N+1), added by Pages commit `ba1562d` at 20:47 that same day and never mirrored back.
- **Root Cause**: `CLAUDE.md` step 1 permits "Regenerate (**or patch**)". Repeated patching without folding the change into the generator, plus editing the Pages copy directly instead of the source artifact, let three copies diverge. The generator has zero TIMIT references, so that section exists only as a patch.
- **Impact**: A plain `python make_training_report.py --out artifacts/segmenter/report.html` followed by the documented copy-and-push would silently delete the TIMIT section and revert the headline framing that ADR-029 settled.
- **Solution**: Before regenerating, diff the generator's output against BOTH `artifacts/segmenter/report.html` and `<pages-repo>/reports/segmenter-training-standalone.html`, and re-apply anything the generator does not produce. Better: fold the TIMIT table and the headline wording into `make_training_report.py` so a regeneration is lossless, then re-sync all three.
- **Fixed (2026-08-31, commit `d0be3b7ec`)**: Both edits are now derived in the generator. The 15.9% / 13.3% relative reduction is computed from the same LS960 values that already produced the 0.53 / 0.86 absolute points. The TIMIT section renders from `audit_timit_phone_agreement.py`'s output (`artifacts/segmenter/timit_test_phone_agreement.json`), which arrived from skipjack after the first fix attempt; the script and its output are now committed alongside the generator. Verified: the regenerated page is byte-identical to the live one (2,085,302 bytes) apart from the generation timestamp.
- **Why full precision mattered**: three of the six published F1-point gains are NOT recoverable from the rendered table, which keeps one decimal. The Transformer's exact-F1 change against phone-CTC is -0.5 from full precision (28.7537 vs 29.2208) but -0.4 from the rounded 28.8 and 29.2. An interim fix that carried the published deltas through a JSON recovered from the HTML was discarded once the real analysis output arrived; all six gains now reproduce exactly.
- **Prevention**: Patch the source artifact, never the Pages copy, and treat any size mismatch between the two as drift. All three versions passed `scripts/validate_html.py` throughout, so validation will not catch this -- only a diff will. When a report edit is worth keeping, put it in the generator, not in the output.
- **Follow-up**: The TIMIT JSON is a recovery, not a real analysis output. The original script and results are still only on skipjack; TIMIT itself is available on CLSP at `/export/corpora5/LDC/LDC93S1/timit/TIMIT` (6,300 `.PHN` files; TEST minus SA prompts = 1,344, matching the caption), so the analysis is re-runnable here once the script is ported.

### 2026-07-22 - Unpinned torch resolved by uv broke Triton JIT compilation
- **Issue**: `uv pip install -r requirements.txt` (no upper bound on `torch`) resolved torch 2.13.0, which crashed inside Llama's rotary-embedding forward pass with `fatal error: Python.h: No such file or directory` during Triton's JIT compile of a native op kernel.
- **Root Cause**: Triton (bundled with torch 2.13.0) needs Python dev headers to JIT-compile at runtime; not available in this venv/cluster environment. Reproduced identically regardless of which LLM was loaded, so it's an environment issue, not a model/data bug.
- **Solution**: Pinned `torch==2.7.1` / `torchaudio==2.7.1` (`--index-url https://download.pytorch.org/whl/cu118`), matching the version the repo's own root README documents as tested. Triton auto-downgraded to 3.3.1 alongside it and the error disappeared.
- **Prevention**: Don't let `uv`/`pip` resolve `torch` to latest-unbounded on this cluster; follow the README's pinned version, or re-verify a newer torch actually works here before adopting it.

### 2026-07-22 - Gated HF models 403 even though model_info() succeeds
- **Issue**: `google/gemma-3-1b-it` (original config placeholder) and `meta-llama/Llama-3.2-1B-Instruct` (repo's actual target LLM) both returned `403 GatedRepoError` on real file download, despite `huggingface_hub.model_info(repo_id)` succeeding without error for both.
- **Root Cause**: `model_info()` only requires read access to repo metadata, not the file-download grant — it does not reflect whether the account's token has actually been authorized for the gated content. Trusting it burned one avoidable job submission.
- **Solution**: Verify actual access with a real download call (e.g. `huggingface_hub.hf_hub_download(repo_id, filename="config.json")`), not just `model_info()`, before relying on a gated model in a job.
- **Prevention**: Always check gated-model access with a real file fetch, never metadata alone.

### 2026-07-22 - HF tokenizer loading ignores the explicit cache_dir used for model loading
- **Issue**: With `HF_HUB_OFFLINE=1` set and `llm_save_path` pointed at a shared local snapshot, `LlamaForCausalLM` weights loaded fine from the local cache, but the tokenizer load still failed with `LocalEntryNotFoundError` (tried the default cache, found nothing, offline mode blocked the network fallback).
- **Root Cause**: `speechbrain/integrations/huggingface/huggingface.py`'s `load_tokenizer()` calls `AutoTokenizer.from_pretrained(source, **kwarg)` without forwarding `cache_dir=save_path` the way model/config loading does — it silently falls back to the `HF_HUB_CACHE`/`HF_HOME` default instead.
- **Solution**: Also set the `HF_HUB_CACHE` env var to the same shared cache directory, so the default (env-based) resolution matches the explicit `cache_dir` used elsewhere.
- **Prevention**: When pointing this codebase's HF integrations at a non-default cache location, set both the yaml's `*_save_path` AND `HF_HUB_CACHE` — one alone isn't enough.

### 2026-07-22 - merge_csvs() logs "skipping" but always re-merges (upstream, not fixed)
- **Issue**: `speechbrain/dataio/dataio.py`'s `merge_csvs()` logs `"Skipping merging. Completed in previous run."` when the target file already exists, but there's no `return`/`continue` after that log line — it unconditionally re-reads all source CSVs and overwrites the merged file regardless.
- **Root Cause**: Dead conditional; the log message doesn't match the actual behavior.
- **Solution**: Not fixed (out of scope for this session) — but confirmed the actual behavior always regenerates correctly, so there's no stale-`train.csv` risk when `train_splits` changes between runs. Just don't trust the log message itself.
- **Prevention**: If a merged csv (e.g. `train.csv`) ever looks stale, check file contents/mtimes directly rather than trusting this log line.

## Tips

- Keep descriptions under 2-3 lines
- Focus on what was learned, not exhaustive details
- Periodically clean out very old entries (6+ months)

### 2026-07-23 - Fresh output_folder breaks offline SSL load (facebook/wav2vec2-base-960h not in shared cache)
- **Issue**: New training runs with a fresh `--output_folder` crashed in `wav2vec2.py:89` (`AutoConfig.from_pretrained`, `LocalEntryNotFoundError`) under `HF_HUB_OFFLINE=1`. The SSL `ssl_folder` defaults to `<output_folder>/save/ssl_checkpoint` (empty for a new run), and the shared HF cache (`omnienc/hf/hub`) has `wav2vec2-base` and `wav2vec2-large-960h-lv60` but NOT the exact `facebook/wav2vec2-base-960h`.
- **Solution**: Point new runs at exp-001's already-downloaded copy: `--ssl_folder .../results/speechllm_fixed_pooling/alignment/3407/save/ssl_checkpoint` (has `models--facebook--wav2vec2-base-960h`). Baked into `launch_blank_ablation.sh` and `phase1_blank_diag.py`.
- **Prevention**: Any new run of speechllm_fixed_pooling with a fresh output_folder must override `ssl_folder` to a populated SSL cache (or the SSL must be added to the shared HF cache).

### 2026-08-03 - S2SHuggingFaceLLMGreedySearcher has NO KV cache -> O(L^2) decode (CER RL bottleneck)
- **Issue**: The CER/WER-reward RL step was ~12 s/it (~18x the NLL reward), making CER training infeasible. First blamed on doing K decodes sequentially; batching them + off-policy reuse (num_iterations) did NOT help.
- **Root Cause**: `speechbrain/decoders/seq2seq.py::S2SHuggingFaceLLMGreedySearcher.forward_step` (+ `_update_mem_embeddings`) rebuilds `[audio_prefix + ALL generated tokens]` and re-runs the FULL LLM forward every decode step — no `past_key_values`. Decode is O(L^2) and compute-bound, so batching K.B just widens each already-heavy step (no wall-time win).
- **Solution**: For the RL reward decode, bypass the searcher and call the LoRA model's KV-cached HF generate: `self.modules.llm.adapted_model.generate(inputs_embeds=prefix, attention_mask=attn, max_new_tokens=int(max_decode_ratio*S), do_sample=False, use_cache=True, eos_token_id=..., pad_token_id=...)`. All sequences share prefix length n and end with the prompt, so batched generate is well-formed (interior segment-pad handled by the attention mask); LoRA is active inside `adapted_model`. transformers 5.14.1 returns only the generated tokens for the inputs_embeds path. Result: ~12 -> ~3-4 s/it (~3x). See `train_speechllm_with_segmenter.py::_decode_replicated`.
- **Prevention**: This searcher is fine for a one-off eval pass but is quadratic — never use it in a training inner loop. Valid/test still use it (kept for eval-methodology parity with the nll ablations); switch to generate if eval time matters.

### 2026-08-23 - Requested 960h scale-up was incorrectly submitted as 100h replication
- **Issue**: The requested LibriSpeech-960h scale-up was misread and twelve additional 100h runs were submitted instead.
- **Root Cause**: “Scale-up” was treated as scaling the replication matrix rather than scaling the training dataset, despite the explicit 960h context.
- **Solution**: Recorded ADR-012/013 and submitted a dedicated optimized `ls960` launcher with all three training splits explicitly set, shared verified manifests, and a compute-matched one-pass schedule.
- **Prevention**: Before submission, print and verify dataset splits and require `ls960` in every full-data output/job name; never label a 100h run as scale-up.

### 2026-08-23 - External FlashAttention-2 needs an isolated CUDA-extension build
- **Issue**: `flash-attn==2.7.4.post1` initially failed to compile against Torch 2.7.1+cu118 because CUDA 11.8 rejected the default GCC 12 and the system Python 3.11 installation lacks `Python.h`.
- **Solution**: Built only the A100 `sm_80` kernels with GCC 9.3, CUDA 11.8, and headers from an isolated uv-managed CPython 3.11.11 tree. The import-verified package lives at `artifacts/benchmarks/flash_attn2_env/`; the active training environment was not modified.
- **Prevention**: For any rebuild, load `gcc/9.3.0` and `cuda/11.8.0`, set `FLASH_ATTN_CUDA_ARCHS=80`, and expose the isolated Python headers through `CPATH`/`C_INCLUDE_PATH`/`CPLUS_INCLUDE_PATH`.

### 2026-08-24 - LS960 learned runs skipped the fractional decoder warmup
- **Issue**: Jobs 105301–105306 were configured for 20% decoder warmup but entered `joint-RL` at optimizer step 0.
- **Root Cause**: `warmup_optimizer_steps` is written to the source `hparams` after `SegmenterASR` has copied its hparams, so the Brain retains `null`; the resolver also reads the YAML accumulation default (4) instead of the effective CLI value (1).
- **Solution**: Fixed before jobs 106155–106160: resolve with the effective CLI accumulation before Brain construction, copy the integer step limit into the Brain, and assert warmup at optimizer step 0. The corrected LS960 boundary is step 2,395.
- **Prevention**: `test_segmenter_step_schedule.py` covers the CLI/YAML accumulation mismatch, warmup boundary, string-valued optional step limit, and validation trigger.

### 2026-08-25 - ga137 node failure interrupted corrected LS960 seed 3408
- **Issue**: Transformer AR + BiGRU job 106157 ended its first allocation with
  `NODE_FAIL` after 10h29m on ga137.
- **Solution**: SLURM automatically requeued the same job (`Restarts=1`). Its latest
  intact intra-epoch checkpoint is optimizer step 13,867, about 475 steps before
  interruption, so no duplicate was submitted. ga137 rebooted and returned healthy
  before the audit; ga129 remains excluded.
- **Prevention**: Keep periodic intra-epoch checkpoints and inspect duplicate `sacct`
  records after requeues; do not treat the top-level pending state as a fresh job.

### 2026-08-26 - HTML speed section used legacy decoding artifacts
- **Issue**: The report's WER comparisons used corrected batch-invariant evaluation,
  but its RTF section still read the legacy stock-searcher artifacts and presented
  their batch-1/batched comparison as current speed evidence.
- **Solution**: Pointed the generator at the corrected three-seed artifact tree,
  aggregated test-clean and test-other per seed, and removed unsupported corrected
  batch-1 and optimal-batch claims.
- **Prevention**: Require the `batch_invariant_left_packed_duration_cap_v1` protocol
  tag when ingesting report RTF data and fail unless all three corrected seeds exist.

### 2026-08-27 - Memory-limited RTF calibration overestimated safe full-test batches

- **Issue**: The selector chose batch 64 for fixed k=5 and both CNN systems, but
  the final full-test jobs OOMed for all fixed-k=5/CNN-mean seeds and one CNN+BiGRU
  seed. Large amounts of reserved-but-unallocated CUDA memory were present.
- **Root Cause**: A short isolated calibration pass below 90% allocated memory did
  not reproduce allocator fragmentation from sequential batch-1, prewarm, and full
  selected-batch evaluation in one process.
- **Solution**: Submitted a conservative one-power-of-two fallback at batch 32 for
  every affected system and all three seeds. Each retry has a fresh output path,
  avoids re-running batch 1, and enables `expandable_segments`. Jobs 118878–118892
  all completed with exit code zero; peak allocated memory stayed between 25.3 and
  32.7 GB across the affected systems.
- **Prevention**: Do not call a batch memory-safe from a short fresh-process probe
  alone. Require the exact final process lifecycle or a conservative reserve-memory
  rule before launching all seeds.

### 2026-08-29 - Phone-CTC downstream launcher duplicated a HyperPyYAML override

- **Issue**: Downstream job 170353 failed before model construction because
  `coldstart_ckpt_dir` appeared twice in the generated override mapping; dependent
  evaluation array 170354 became `DependencyNeverSatisfied`.
- **Root Cause**: `run_segmenter.slurm` translated the exported `COLDSTART_DIR` to
  `--coldstart_ckpt_dir`, while the launcher's `EXTRA_ARGS` already contained the
  same override.
- **Solution**: Removed `COLDSTART_DIR` from the downstream `--export`, retained the
  single explicit override in `EXTRA_ARGS`, cancelled 170354, and resubmitted only
  downstream training/evaluation as jobs 171034/171035. Completed CTC training,
  checkpoint audits, and boundary tensors were reused.
- **Prevention**: Every option passed by a shared SLURM wrapper must have exactly one
  owner: either a wrapper environment variable or `EXTRA_ARGS`, never both.

### 2026-08-29 - Empty training test split left stale test CSV references

- **Issue**: Corrected job 171034 passed HyperPyYAML/model initialization but failed
  when `dataio_prepare` opened missing `test-clean.csv`; 171035 consequently became
  `DependencyNeverSatisfied`.
- **Root Cause**: Training overrode `test_splits: []` to avoid redundant end-of-fit
  inference but retained the YAML default `test_csv` paths. The evaluation launcher
  also had latent duplicate `output_folder` and `checkpoints_to_keep` overrides.
- **Solution**: Training now sets both `test_splits: []` and `test_csv: []`. Common
  evaluation arguments no longer contain training-only output/checkpoint keys. The
  exact downstream and evaluation override mappings were parsed before submitting
  jobs 171203/171204; 171203 entered epoch 1 successfully.
- **Prevention**: Treat `*_splits` and `*_csv` as paired configuration fields, and
  validate every dependency stage's final override mapping—not only the immediate
  failed stage—before submission.

### 2026-08-29 - sacrebleu 2.6 moved get_signature off the score object
- **Issue**: `sacrebleu.corpus_bleu(...).get_signature()` raised `AttributeError: 'BLEUScore' object has no attribute 'get_signature'`, crashing the first multi-task validation point.
- **Root Cause**: In sacrebleu 2.x the signature belongs to the *metric* (`sacrebleu.metrics.BLEU`), not to the returned score. The module-level `corpus_bleu` helper returns only a score.
- **Solution**: Build `BLEU()` and `CHRF(word_order=2)` once, cache them, and use `metric.corpus_score(hyps, [refs])` plus `metric.get_signature()`. Caching also guarantees the signature is identical at every validation point, which is what makes scores comparable across steps.
- **Prevention**: A BLEU number without its signature is not comparable to anyone else's; log the signature from the metric object every time.

### 2026-08-29 - ErrorRateStats.summarize raises ZeroDivisionError on an empty split
- **Issue**: A translation-only evaluation split (`covost_st_test`) appends nothing to the WER metric, and the inherited `on_stage_end` then calls `summarize("error_rate")` on it, raising `ZeroDivisionError`.
- **Root Cause**: WER is undefined for a split with no same-language transcription task, but the base class computes it unconditionally.
- **Solution**: `_NoErrorRate` placeholder returning `nan` for TEST splits, so the value reads as "not measured" rather than as a zero or perfect score. For VALID the trainer raises instead: a validation mixture without ASR cannot check LibriSpeech retention or rank checkpoints, so that is a config error, not a case to paper over.
- **Prevention**: When a metric is genuinely undefined, emit `nan` with an explanation; never substitute a value that could read as a real score.

### 2026-08-29 - Login-node process pool killed during CoVoST decoding
- **Issue**: `covost2_prepare.py --workers 8` died with `BrokenProcessPool` partway through the 289k-utterance train split.
- **Root Cause**: Eight worker processes each holding torch, torchaudio, and a parquet reader exceeded the login node's per-user limits. Not a data or code fault.
- **Solution**: Run preparation as a SLURM job on the CPU `med` partition (`prepare_covost2.slurm`, 16 CPUs / 96 GB). The script is resumable -- it skips utterances whose FLAC already exists -- so the partial run was not wasted.
- **Prevention**: Multi-hour, multi-process CPU work belongs on `med`, not the login node. Writing output atomically (`.tmp` then `os.replace`) is what makes resumption safe after an abrupt kill.

### 2026-08-29 - HyperPyYAML merges dict overrides instead of replacing them
- **Issue**: The multi-task smoke test overrode `test_source_specs` with three entries but the run evaluated six splits, including the full LibriSpeech dev-clean it was meant to skip.
- **Root Cause**: A command-line override of a mapping is recursively merged into the YAML value; keys not mentioned in the override survive. This is not list/scalar behaviour, where the override replaces.
- **Solution**: For the smoke test the merge is harmless (it exercises the real flagship test path, just slower), so it was documented rather than worked around. Where a true replacement is needed, override every key or restructure the config.
- **Prevention**: When an override is meant to *shrink* a mapping, check the resolved value in the log rather than assuming replacement.

### 2026-08-29 - Two over-long CoVoST test clips killed final evaluation after 9 hours of training
- **Issue**: Flagship job 171461 completed all 15000 optimizer steps and all ten validations, then died in final evaluation with `ValueError: Boundary position 4096 exceeds max_positions=4096`.
- **Root Cause**: The autoregressive boundary policy indexes a 4096-entry positional table on a 50 Hz grid, i.e. 81.92 s. Exactly two of the 15529 CoVoST test utterances exceed it (longest 142.5 s) -- mis-segmented Common Voice clips in a split whose p99.9 is 22 s. Train, validation, and every LibriSpeech split are clean, which is why the failure only appeared at the very end.
- **Solution**: `build_multitask_manifest(..., max_duration=...)` drops and loudly logs such utterances, with the cap derived from `segmenter_ar_max_positions` itself (`max_utterance_seconds`) so the data filter cannot drift from the model limit. Training did not need to be redone -- all checkpoints survived and evaluation was rerun with `eval_only` (job 172481).
- **Prevention**: Validate a split against the model's hard limits when the manifest is built, not when the split is first decoded. A constraint that only a test split violates will otherwise surface after all the expensive work is done. Note the exclusion when reporting CoVoST test BLEU: it is 15527 of 15529 utterances.


## 2026-09-02 — Transient GPU fault killed a run; `--requeue` does not cover it

**Symptom.** Job 1780854 (aux arm, seed 3407) died after 32 min with
`RuntimeError: CUDA error: unspecified launch failure`, with the traceback
pointing at `torch.tensor(...)` in `_route_task` during the first step
validation.

**Diagnosis.** Not a code fault. CUDA reports these asynchronously and the
message says so, so the reported site is not the true one. The discriminating
test was already available: aux seed 3408 (1780855) ran the same new
`rate_channel: aux` code path on a different node (e02) and passed step 1500
normally. That isolated the failure to node **e01**. `sinfo` did not flag e01,
and job 1780836 had completed 8h41m on it earlier the same day, so the fault was
transient rather than a dead node.

**Prevention / workaround.**
- `EXCLUDE_NODES` was added to `submit_multitask_nonasr.sh`; resubmitted as
  1781108 with `--exclude=e01`.
- **`--requeue` does not protect against this.** SLURM requeues on node failure
  and preemption, not on a non-zero application exit, so a mid-run CUDA fault
  costs the entire run. Long runs need internal checkpoint-resume (precedent:
  ADR-029) rather than relying on requeue.
- When a new code path fails once, check whether a sibling job is already
  running the same path elsewhere before assuming a bug — here it answered the
  question for free.

## 2026-09-02 — conda-forge TeX Live could not generate its PDFLaTeX format

**Symptom.** A fresh `texlive-core=20260301` environment exposed `pdflatex` and
`latexmk`, but the first build failed because `mktexfmt` could not locate
`mktexlsr.pl`; consequently `pdflatex.fmt` was absent.

**Diagnosis.** The file is missing from the conda package, matching the open
conda-forge `texlive-core` issue #61. This was an environment-packaging defect,
not an ICASSP source or style-file error.

**Solution / prevention.** Removed only the newly created broken environment
and recreated it with Tectonic 0.17 plus Poppler 26.07. Keep the reproducible
spec in `drafts/icassp/environment.yml`, use the Makefile's absolute tool paths,
and retain the Tectonic package cache under `/export/jsalt26`. Validate future
toolchain changes by compiling both `main.tex` and the untouched official
`Template.tex`, then inspect page size and font embedding with Poppler.

## 2026-09-04 — Noise floor applied to speech mixtures, not just the zero class

**Symptom.** The synthetic speaker-count task stalled at ~0.55 accuracy. A
validated CountNet reference scored 0.927 on real LibriCount but 0.564 on our
mixtures, and only 0.55 on our SINGLE-speaker clips, which should be trivial.

**Cause.** A -35 dBFS Gaussian floor was added so the count-0 class would not be
free digital silence -- correct for count 0, but it was applied to every
mixture. Against equal-power sources at -25 dBFS that is 10 dB SNR of broadband
noise, which reads as additional talkers. LibriCount adds no floor at all.

**Fix.** `SPEECH_FLOOR_DBFS = -60.0` for mixtures containing speech; the audible
floor now applies only to the zero class via `--zero-style`. Count-1 accuracy
0.55 -> 0.98, overall 0.467 -> 0.700.

**Prevention.** A per-class design choice needs a per-class scope. And validate
generated data against an external reference before training on it: three
plausible hypotheses (level policy, edge transients, VAD labelling) were each
disproved by measurement, and the real cause was none of them.

## 2026-09-04 — Ran a heavy data-prep job on the login node

**Symptom.** User noticed there was no Slurm job for the generation. It was
running under `nohup` on `control2`, an 8-core login node shared with 33 users.

**Cause.** The earlier CREMA-D/FSC preps only read audio *headers*, so running
them interactively was harmless, and that habit carried over to a job that
decodes ~40k crops and runs VAD over them.

**Fix/prevention.** `prep_speaker_count.slurm` (partition `cpu`, 16 CPUs, logs
under `/export/jsalt26/.../slurm_logs`), which forwards its arguments so
regenerating never needs a login-node shortcut. Rule of thumb: header reads are
fine interactively; anything that decodes audio goes to `cpu`.


## 2026-09-09 — Real references had incorrect or overbroad manuscript attribution

**Symptom.** The bibliography and prose described AdaTS as learned token sampling at ICLR main conference, cited the Llama 3 paper for a Llama 3.2 checkpoint, and described the implemented policy loss simply as GRPO.

**Cause.** Correct paper titles/identifiers were treated as sufficient verification. PDF conference headers do not distinguish workshops reliably; unversioned arXiv author metadata can change; a source algorithm's name does not imply that our implementation includes its complete objective.

**Fix.** Verified the source contents and publication records, corrected AdaTS's parameter-free method and workshop venue using the PDF plus coauthor announcement, cited the exact Meta model card, and explicitly described ASSET's on-policy GRPO variant with a REINFORCE citation. Also corrected the unused Llama 3 author prefix and recorded publisher-metadata normalization exceptions.

**Prevention.** Check the exact claim against the paper and the exact model against its release card. Verify conference/workshop status beyond the PDF template. Compare algorithm citations with local code and preserve source-date/version evidence. See drafts/icassp/research/citation_audit_2026-09-09.{md,json} and ADR-046.

**2026-09-10 terminology follow-up.** ADR-048 supersedes the GRPO wording above: the author requests GRPO consistently, with the omitted reference-policy KL regularizer stated as beta = 0. The paper retains the fresh-sample, one-update policy-gradient expression and removes the REINFORCE label/citation.


## 2026-09-10 — Boundary control must share the encoder time lattice

- **Issue:** An exactly count-matched continuous uniform grid scored 43.3% phone F1 while learned cuts occupied the 20 ms lattice and MFA times occupied a 10 ms lattice. The control timing convention artificially enlarged the measured gain at an inclusive ±20 ms tolerance.
- **Fix:** Round uniform cuts to valid encoder frames while preserving their exact per-utterance count. Final control F1 is 52.6%, versus learned 67.8%. Add a lattice/count invariant test and report the +12.5 ms timestamp sensitivity. Raw counts and the corrected comparison are in the LS960 pilot artifacts.
- **Prevention:** Match temporal quantization, obligatory endpoints, counts, and tolerance conventions before interpreting a boundary metric. Do not compare the older quantized 100h F1 directly with the continuous-time pilot.


## 2026-09-10 — Historical phone audits use different utterance-start conventions

- **Finding:** `audit_transformer_ar_phone_agreement.py` explicitly includes the Transformer's implicit frame-zero start. `audit_timit_phone_agreement.py` scores nonzero raw learned cuts without inserting that start, while native TIMIT references and fixed k=5 include frame zero. Consequently the TIMIT learned cut-frequency field also omits one pooled segment per utterance.
- **Reporting:** Recompute the historical F1 values from their counts, document the convention, avoid interpreting the cross-corpus score difference, and do not call TIMIT cut frequency the token rate. Adding frame zero can alter near-start one-to-one assignments, so its exact F1 effect requires per-utterance predictions; do not assert that the omission is necessarily conservative. No historical artifact was overwritten.
- **Next audit:** Standardize exclusion of obligatory endpoints across all systems and corpora; use per-utterance count-matched controls and explicit time quantization. See the phone-boundary revision evidence note and the fuller segment-interpretation plan.


## 2026-09-10 — TIMIT annotation end is not always waveform duration

- Phone-pattern audit job 1790791 stopped after retaining 381 utterances because one TIMIT waveform extends 112 ms beyond its final PHN endpoint. Header audit of all 1,344 TEST files found 21 unannotated tails longer than 20 ms (maximum 1.632 s).
- Read actual duration from the native NIST header, retain annotation_end separately, and exclude cuts in unannotated tails from boundary scoring. Use actual waveform duration for token rate. Preserve the original selection manifest and migrate only its duration bookkeeping/hash for already computed cuts; waveforms, selected utterances, checkpoints, and predictions are unchanged.
- Evidence: artifacts/segmenter/ls960_phone_patterns_2026-09-10/timit_header_audit.json. Pending dependent summary 1790798 cancelled; inference resumes from existing cuts.


## 2026-09-11 — Tectonic cache outside the writable workspace

- A manuscript rebuild failed with `Read-only file system (os error 30)` while using `/export/jsalt26/omnienc/users/cxiao/cache/tectonic` under the workspace sandbox.
- Copying the existing 35 MB cache into `/tmp/icassp-ar-notation-cache-20260911` and setting `TECTONIC_CACHE_DIR` to that copy completed the build without escalation or downloads. Use a writable cache copy for future sandboxed builds; preserve the shared cache.


## 2026-09-11 — New figure-label fonts fail in the offline Tectonic build

- Direct sans-serif typesetting requested uncached `lmsans10-regular.otf`, which could not download under restricted networking. A subsequent system-family lookup for DejaVu Sans/B triggered a Rust panic in Tectonic's font manager despite the installed font files.
- Render the title from its native draw.io XML cell with the existing Matplotlib environment and include the resulting embedded-font PDF. This builds without font downloads or system-family lookup and preserves the manual PNG beneath the title. See scripts/make_overview_title.py and figures/JSALT26.tex under drafts/icassp.


### 2026-09-12 — Non-ASR TEST phase metadata can describe the end of training rather than the selected checkpoint
- The selected speaker-count checkpoints (steps 1560/1300/1040) and intent seed 3408 (step 450) report phase=unfreeze in task_metrics.jsonl, although none has a final-block update. The step 1560 count checkpoint is exactly at the phase boundary; the first unfreeze update would follow. Direct comparison shows zero changed backbone tensors and changed FiLM parameters.
- Reporting fix in this audit: derive actual updates from the selected checkpoint step and schedule, and verify tensors. Preserve original records and expose phase_record_is_stale in checkpoint_audit.json. No training/evaluation code or old results were modified. Future logger repair should derive reported phase from loaded checkpoint state rather than a stale active-phase cache.

### 2026-09-12 — Boundary-audit endpoint assumption and Expresso dictionary coverage
- Initial inference job 1791550 asserted AR boundary[0]==1. Production AR fixes this action to 0 and pooling supplies the implicit initial segment. Corrected the assertion; successful 1791551/52/53 exclude frame 0 from matching and count its segment exactly once. No predictions from the failed job entered the analysis.
- MFA initially replaced nine out-of-dictionary Expresso words with spn (49 utterances). Resolved pronunciations using six CMUdict entries, a documented WebMD composition, and two MFA G2P entries; retained original references and ran an independent original-dictionary-covered 371-utterance sensitivity. Do not silently score unknown-word spans as normal phones.

## 2026-09-13: Small BF16 autoregressive cut differences across inference batches

The controlled Expresso probe reproduced351/360 original cached policy tracks exactly. Nine tracks across six recordings differed by1–4cut locations, despite per-utterance frozen WavLM feature extraction and deterministic boundary decisions. A batch-size1 sensitivity run over all48 acoustic variants from those recordings reproduced276/288 tracks of the new batch-size8 run. Substituting its tracks changes any60-recording condition mean F1 by at most0.107percentage points. This is consistent with numerical perturbations around AR decision thresholds, not proven bitwise batch invariance. Retain within-run controls and report the sensitivity. Evidence: artifacts/segmenter/expresso_boundary_patterns_2026-09-13/batch_sensitivity_summary.json and probes/original_reproduction.json. No training/model code was changed.
