# Reproducing the ASSET experiments

This branch (`asset-core-repro`) is a curated, code-only snapshot: everything
needed to re-run the experiments, and nothing else. It contains no datasets,
features, boundary targets, checkpoints, results, SLURM logs, generated
reports, manuscript sources, or session notes.

All experiment code lives in `recipes/LibriSpeech/ASR/transformer/` (called
`$R` below). `speechbrain/` is the toolkit the recipes import (SpeechBrain at
commit `da704dd9`, plus local changes such as `speechbrain/dataio/sampler.py`).

## Setup

```bash
pip install -r requirements.txt
pip install --editable .
pip install accelerate h5py
```

See the top-level `README.md` for the tested Python/PyTorch versions and the
base SpeechLLM commands. Some analysis scripts additionally use `matplotlib`,
`pyarrow`, `sacrebleu`, and `praat-parselmouth`.

Sanity check (CPU, ~30 s):

```bash
python -m pytest -q tests/unittests/test_rate_terms_and_bands.py \
    tests/unittests/test_bilevel.py tests/unittests/test_multitask_data.py \
    recipes/LibriSpeech/ASR/transformer/test_segment_pooling.py
```

## Paths and cluster identity

Launchers, SLURM wrappers and hparams carry absolute paths under
`/export/jsalt26/omnienc/users/cxiao/...` and CLSP SLURM flags
(`--account=highprio --partition=gpu-a100`). On another cluster, rewrite them
first; `tools/rehome_paths_clsp.sh` and `tools/rehome_slurm_clsp.sh` show the
complete set of patterns that need to change, and
`tools/preflight_clsp_assets.py` is a CPU check that the flagship system's
inputs (audio, boundary targets, offline HF caches for WavLM-Large and
Llama-3.2-1B-Instruct) resolve before a GPU job is spent on it.

## Scale convention

"Scale-up" means LibriSpeech-960h (`train-clean-100` + `train-clean-360` +
`train-other-500`); those launchers carry `ls960` in their names. Everything
else is LibriSpeech-100h (`train-clean-100`).

## Map of the code

Training entry points

- `train_speechllm.py` — base SpeechLLM; also the fixed-rate, native-rate and
  precomputed-boundary baselines (`hparams/speechllm_e2e.yaml`,
  `speechllm_ssl_feats.yaml`, `speechllm_fixed_pooling.yaml`)
- `train_speechllm_with_segmenter.py` — learned segmenter, char cold-start then
  joint RL (`hparams/speechllm_segmenter.yaml`); the main entry point
- `train_speechllm_multitask.py` — task-conditioned multi-task runs
  (`hparams/speechllm_multitask*.yaml`, `speechllm_singletask_nonasr.yaml`)
- `train_speechllm_adaptive.py` — LS960 continuation with a held rate penalty
  (`hparams/adaptive_frequency_ls960*.yaml`)
- `wavlm_phone_ctc.py` — phone-CTC head on frozen WavLM-Large

Model and training components

- `segmenter.py`, `segment_pooling.py`, `bilevel.py`, `blank_modules.py`,
  `candidate_scoring.py`, `multitask_data.py`, `multitask_modules.py`,
  `adaptive_frequency.py`, `batch_invariant_decode.py`,
  `transformer_ar_experimental_runtime.py`, `rtf_batch_selection.py`

Data and boundary-target preparation

- `librispeech_prepare.py`, `extract_ssl_feats.py`, `ctc_boundary_align.py`,
  `gen_word_ctc_sep.sh`
- Transcript-free boundary streams: `ctc_posterior_collapse.py`,
  `bpe_ctc_posterior_collapse.py`, `refine_ctc_boundaries.py`, with the
  `generate_*.slurm` wrappers
- Non-ASR and ST tasks: `cremad_prepare.py`, `fsc_prepare.py`,
  `speaker_count_prepare.py`, `speaker_count_mixing.py`,
  `libricount_prepare.py`, `covost2_prepare.py`, `download_covost2.py`,
  `mustc_prepare.py`, `prepare_expresso_boundary_audit.py`

Launchers, by experiment family (each script's header states its protocol)

- Learned segmenter, 100h: `submit_pilot.sh`, `submit_abl*.sh`,
  `submit_joint_AB.sh`, `submit_cer_opt.sh`, `submit_wavlm_multiseed_sweep.sh`,
  `submit_wavlm_transformer_ar*.sh`, `submit_wavlm_cnn_ar_bigru_*.sh`,
  `submit_w2v2*.sh`, `submit_compact_position_key_retraining.sh`
- Baselines, 100h: `submit_wavlm_baselines.sh`,
  `submit_wavlm_fixed_rate_multiseed.sh`, `submit_no_downsampling_multiseed.sh`,
  `submit_wavlm_char_alignment_multiseed.sh`,
  `submit_wavlm_oracle_alignment_multiseed.sh`, `submit_wavlm_oracleclose_rl.sh`,
  `submit_wavlm_ctc_posterior_bigru_100h.sh`,
  `submit_wavlm_bpe_ctc_state_runs_bigru_100h.sh`,
  `submit_wavlm_phone_ctc*_100h.sh`, `submit_wavlm_attribution_100h.sh`,
  `submit_wavlm_fixed_k5_mean_char_init_100h.sh`
- LS960 scale-up: `submit_ls960_scaleup.sh`, `submit_ls960_step24k_corrected.sh`,
  `run_segmenter_ls960_joint.slurm`,
  `submit_wavlm_phone_ctc_longsplit8_mean_ls960*.sh`,
  `submit_adaptive_frequency_ls960.py`, `submit_adaptive_frequency_test_ls960.py`
- Bilevel optimization: `submit_bilevel_pilot.sh`, `submit_bilevel_full.sh`,
  `submit_bilevel_memory_bench.sh`
- Multi-task and non-ASR: `submit_multitask_flagship*.sh`,
  `submit_multitask_nonasr.sh`, `submit_nonasr_singletask_baselines.sh`,
  `submit_libricount_eval.sh`
- Inference cost (RTF): `submit_memory_limited_rtf_benchmarks.sh`,
  `submit_rtf_common_batch16.sh`, `submit_rtf_batch32_retries.sh`,
  `submit_wavlm_inference_benchmark_2x2.sh`,
  `submit_wavlm_baseline_inference_benchmarks.sh`,
  `submit_batch_invariant_wavlm_*_eval.sh`
- Job shapes used by the submitters: `run_*.slurm`, `eval_*.slurm`
- Gates and smoke tests: `smoke_*.slurm`, `verify_*.slurm`, `verify_*.py`,
  `probe_a100_health.slurm`

Evaluation, audits and analysis (read saved run outputs; no training)

- Boundary quality: `audit_*.py`, `measure_oracle_rates.py`,
  `eval_wavlm_boundary_swap.py`, `score_asr_boundary_confidence.py`
- Non-ASR / Expresso boundary studies: `analyze_*.py`, `summarize_*.py`,
  `extend_expresso_patterns.py`, `probe_expresso_acoustics.py`,
  `validate_expresso_patterns.py`, `extract_nonasr_prosody.py`
- External references: `eval_seamless_covost.py`, `countnet_reference.py`,
  `probe_frozen_encoder_domains.py`
- Kernel and rollout benchmarks: `benchmark_*.py`
- Figures and tables from saved results: `plot_*.py`,
  `make_results_frontier_figure.py`, `make_qualitative_boundary_figure.py`,
  `make_table_nonasr_audit.py`, `report_*.py`

Tests: `tests/unittests/` (SpeechBrain's suite plus the project tests) and
`$R/test_segment_pooling.py`.

## Deliberately not on this branch

- Outputs: `results/`, `artifacts/`, `slurm_logs/`, feature caches, boundary
  targets, checkpoints (`*.ckpt`, `*.pt`, `*.safetensors`)
- HTML report / site / proposal builders (`make_*_report.py`,
  `make_*_proposal.py`, `make_training_report.py`, ...); they depend on an
  external `htmlkit` package and present results rather than produce them
- Manuscript sources, plans, surveys, research notes, project memory
- Unrelated upstream SpeechBrain recipes, templates, docs and tutorials; take
  them from SpeechBrain at the commit above if needed
