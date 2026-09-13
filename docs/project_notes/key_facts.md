# Key Facts

Project configuration, constants, and frequently-needed non-sensitive information. `HF_TOKEN`/credentials are never stored here — see `.env`/cluster environment.

## Report writing preferences

- **2026-09-13 — Natural text flow:** The author explicitly requests, “Do NOT do manual line breaks in the report, let the text roll out.” Use normal paragraphs and let the browser wrap prose, headings and captions within the report column. Avoid forced `<br>` breaks, preserved newlines, balanced headings and narrower text-only width limits that artificially determine line endings. Apply this preference to future report updates.

## Research Objective

- The learned-segmenter project is explicitly an **RL improvement project**: learn
  boundary policies that lower ASR WER and decoder-side audio-token cost together.
- Oracle and fixed segmentations provide initialization, upper bounds, and controls;
  they do not replace the goal of improving RL.
- A weak or drifting run is a prompt to improve initialization, reward/curriculum,
  and structured policy design—not a stop condition. See ADR-006.
- For the current local-history Transformer-AR study, the GRPO rollout group is fixed at
  `K=4`, including speed optimizations and local-attention comparisons. Do not reduce
  it to obtain a faster runtime unless this experimental control is explicitly reopened.
- The current local-history control is fixed at 64 frames (about 1.28 s at 50 Hz).
  Do not vary the window during speed comparisons unless this control is reopened.
- The optimized LS960 schedule uses one full-corpus pass. Cold starts use a 400 s
  dynamic-batch budget with accumulation 1. Learned systems use 300 s with
  accumulation 1; the first 20% of optimizer updates adapt the decoder and the
  remaining 80% run joint RL. Compressed decoder baselines use 500 s with
  accumulation 2; native 50 Hz decoding retains its validated 50 s / accumulation-20
  setting. All use BF16, four persistent data workers, pinned/prefetched loading,
  and shared verified LS960 CSV manifests.
- For the local-64 Transformer segmenter, the validated production kernel is
  PyTorch automatic SDPA with preallocated KV and combined K=4 rollouts. Forced
  Flash Attention cannot express the non-null local mask in this environment;
  the FlexAttention prototype was slower in the full training-loop benchmark.
  The Llama decoder separately uses `attn_implementation: sdpa`.

## Bilevel Segmenter–Decoder Interpretation

- The current joint-RL loop is simultaneous co-training, not a bilevel optimizer:
  sampled boundaries are scored with the current decoder, the reward is detached,
  and decoder CE plus segmenter GRPO are updated on the same training stream.
- A useful bilevel formulation treats temporary decoder adaptation as the inner
  problem and held-out ASR plus the audio-frequency penalty as the outer segmenter
  objective. Hard boundaries and GRPO can remain unchanged; the first prospective
  test should use a support/query split and a one-step decoder lookahead reward.
- A retrospective pilot over six matched WavLM runs found 0/6 agreement between the
  minimum-training-CE epoch and the best dev-clean-WER epoch. Selecting the former
  costs 0.42±0.22 WER points on average while the within-run dev audio-frequency span is
  only 0.8±0.5 Hz. This diagnoses an inner/outer mismatch, but does not prove that
  bilevel training fixes it.
- On-policy prefix distillation is complementary: CE+KD can be the inner decoder
  adaptation loss, while held-out ASR remains the outer segmenter score.

## Reporting Language

- In this project, **scale-up means training on the complete LibriSpeech-960h
  corpus**: `train-clean-100`, `train-clean-360`, and `train-other-500`. A run
  that uses only `train-clean-100` must be called a 100h replication or audit,
  never a scale-up. Before submission, verify that the launcher, manifest, and
  resolved hyperparameters all list the three splits and that `ls960` appears in
  the job and output names.
- Use plain, concrete language in reports and figures. Prefer descriptions such as
  "WER versus compression," "results," and "what causes the WER gap?" over phrases
  such as "quality-cost frontier," "benchmark matrix," and "controlled attribution."
- Call the boundary-prediction model the **segmenter** everywhere. Do not call it a
  detector or boundary detector. Use "segmenter input" for the encoder features fed
  to it, and "features pooled for the decoder" for the separate feature stream that
  is averaged between its predicted boundaries.
- In comparison tables, write the condition name next to its numbers. Do not use an
  unlabeled arrow or rely on column order to communicate direction. Example:
  "No downsampling: 7.05 clean, 10.10 other. Fixed k=5: 6.16 clean, 10.17 other."
- State that lower WER is better when a comparison could otherwise be unclear. Define
  audio-token frequency when it first appears: `f_audio = 50 Hz × ρ`, where the
  frozen speech encoder emits 50 frames/s and ρ remains the internal frame-retention
  fraction. In every reader-facing report, plot, table, caption, and proposal, report
  `f_audio` in Hz instead of ρ. Lower Hz means fewer decoder-side audio tokens and
  stronger compression. Use one decimal place by default, scale uncertainty by the
  same factor of 50, and use extra precision only for near-zero collapse diagnostics.
  Internal config keys, logs, and checkpoint fields may retain `rho` for compatibility.

## Cluster Migration: skipjack (JSALT) -> CLSP (2026-08-30/31)

The project now runs on the **JHU CLSP cluster** (`control1`/`ctl2`), not
bluecrab/skipjack. The 2026-08-24..30 rsync port (`tools/port_jointllm_to_clsp.sh`,
`reports/clsp_asset_port_2026-08-24.md`) copied bytes only; every path and Slurm
flag in the recipe still named skipjack. The adaptation below was done on
2026-08-30/31 and is scripted so it can be replayed:

- `tools/rehome_paths_clsp.sh` -- rewrites skipjack absolute paths (111 files
  now carry CLSP paths).
- `tools/rehome_slurm_clsp.sh` -- rewrites Slurm identity flags (49 files now
  carry `--account=highprio --partition=gpu-a100`). A follow-up pass removed
  the guards and echo strings the flag deletion left behind.
- `tools/rehome_manifests_clsp.sh` -- rewrites absolute `wav` paths inside the
  CoVoST/MuST-C manifests (the LibriSpeech CSVs need no pass; they use
  SpeechBrain's `$data_root/` placeholder).
- `tools/preflight_clsp_assets.py` -- CPU check that every flagship ingredient
  resolves and loads. All three rehome scripts are idempotent; re-run them
  after restoring any file from pre-port history.

**Path map** (old -> new):

| skipjack | CLSP |
|---|---|
| `/weka/.../omnienc/users/cxiao/jointllm` | `/export/jsalt26/omnienc/users/cxiao/skipjack/jointllm` |
| `~/cxiao/envs/jointllm` (`$HOME`-relative too) | `/export/jsalt26/omnienc/users/cxiao/envs/jointllm` |
| `.../omnienc/hf/hub`, `.../users/cxiao/ssl_cache` | `/export/jsalt26/omnienc/users/cxiao/hf/hub` |
| `.../omnienc/datasets` | `/export/jsalt26/omnienc/users/cxiao/datasets` |
| `.../users/cxiao/boundary_targets` | `/export/jsalt26/omnienc/users/cxiao/boundary_targets` (symlink into `skipjack/cxiao-backup-2026-08-30/`) |
| `.../users/cxiao/jsalt26-downsampling` | `/export/jsalt26/omnienc/users/cxiao/jsalt26-downsampling` (re-cloned from GitHub) |

**Slurm identity map.** The 2026-08-04 user instruction -- GPU work runs only
on A100 nodes, under an explicit account/partition pair, never a default
account -- carries over; only the names change.
`--account=jsalt2026-lgarci27` -> `--account=highprio`;
`--partition=a100` -> `--partition=gpu-a100` (A100 80GB, nodes `e02`-`e05`;
`e01` down); `--partition=med` -> `--partition=cpu`. The `"JSALT 2026"`
reservation and `--exclude=ga129/ga132` are dropped -- neither exists here.
`--comment=accept_cost` is retained but inert. All CLSP partitions have an
infinite time limit, so the 3-day flagship walltime is unchanged.

**Python environment.** CLSP has no `/usr/bin/python3.11`, so the copied venv at
`_portability/envs/jointllm` was not runnable as shipped (this is called out in
`reports/clsp_asset_port_2026-08-24.md`). Rather than rebuild from `uv.lock` and
risk floating the pinned Torch/Triton stack, the venv was **rehomed**: a managed
CPython 3.11.15 was installed with the ported `uv` into
`/export/jsalt26/omnienc/users/cxiao/uv-python`, the venv copied to
`/export/jsalt26/omnienc/users/cxiao/envs/jointllm`, and its `pyvenv.cfg` `home`,
`bin/python` symlink, 41 console-script shebangs, and the editable-speechbrain
finder MAPPING rewritten to CLSP paths. Verified: `torch 2.7.1+cu118`,
`torchaudio 2.7.1+cu118`, `transformers 5.14.1`, `speechbrain 1.1.0` (editable ->
the ported repo), `numpy 2.4.6`, on Python 3.11.15. The pristine
`_portability/envs/jointllm` copy is left untouched as recovery material.
Driver on `e03` is 580.159.03, well above the cu118 minimum.

**Data restored on CLSP:**
- LibriSpeech audio was *not* transferred (correctly -- it is public). All seven
  splits are on CLSP at `/export/corpora5/LibriSpeech`; `datasets/LibriSpeech`
  symlinks there. (`/export/fs06/cxiao7/LibriSpeech` is an equivalent copy but
  sits on a 99%-full pool.)
- `datasets/wavlm_boundaries/{char,phone,syllable,word,word_ctc}` extracted from
  `_portability/data/wavlm_boundaries/*.tgz` -- 292,367 `.pt` per level, matching
  the skipjack count exactly.
- `datasets/librispeech_manifests` populated from the verified LS960 manifests in
  `results/speechllm_ls960_scaleup_v2_optimized/manifests`. These CSVs use the
  `$data_root/` placeholder, so they are cluster-portable as-is.
- `boundary_targets/` (phone-CTC, BPE-CTC, word-CTC trees) were already expanded
  under `skipjack/cxiao-backup-2026-08-30/` and are exposed by symlink.

### Port verification (2026-08-31, A100 80GB on `e03`, driver 580.159.03)

Scripts: `recipes/.../verify_clsp_port_inference.slurm`,
`recipes/.../verify_clsp_port_training.slurm`,
`tools/preflight_clsp_assets.py`. Outputs under
`recipes/.../results/clsp_port_verification/`.

**Inference reproduces skipjack.** Re-decoding LibriSpeech test-clean with the
completed flagship checkpoint (`fullprefix_transformer_ar_local64_bigru_best_char_decoder/nll_mt/3407/save/CKPT+2026-08-15+03-13-00+00`,
batch 8, batch-invariant duration-cap protocol) against the matching corrected
skipjack evaluation of 2026-08-21:

| Metric | skipjack | CLSP run 1 | CLSP run 2 |
|---|---|---|---|
| test loss | 1.63e-01 | 1.63e-01 | 1.63e-01 |
| rho_mean / rho_std | 0.213 / 0.0306 | 0.213 / 0.0306 | 0.213 / 0.0306 |
| test-clean %WER | 4.73 (2486 err) | 4.70 (2471 err) | 4.70 (2469 err) |
| test-clean CER | 2.12 | 2.10 | 2.10 |

Teacher-forced loss and the segmentation rate match to every reported digit, so
the encoder, segmenter, pooler, projection and LoRA weights all load and compute
identically. The residual WER gap is greedy-decode jitter, not a port defect: two
identical CLSP runs of the same checkpoint themselves differ by 2 word errors
(2471 vs 2469), and the cross-cluster gap is 15 errors out of 52,576 (0.03 WER
absolute). Stage time 766 s vs skipjack's 736 s on a contended node.

The `train_speechllm.py` path reproduces at the same fidelity: oracle-char fixed
pooling, seed 3408, batch 4, test-clean **3.47** on CLSP vs **3.48** on skipjack.

**Training runs end to end.** The flagship joint-RL configuration warm-started
from the local-64 char cold start plus the best char-alignment decoder, run for
12 optimizer steps (4 decoder-CE warmup + 8 joint RL), completed in 1:17:52:

- The cold-start loader reports exactly the ten BiGRU pooler tensors as
  `new_parameters`, i.e. the intended zero-residual mean-pooling warm start.
- Warmup step 4: train dec_ce 2.35e-01, rho 0.289; valid WER 6.15.
- Joint RL step 6: non-zero `pg` -9.43e-04, `reward` -6.19e-02,
  `sampled_rate_penalty` 1.91e-03; valid WER 6.01.
- Joint RL step 12: `pg` -1.47e-03, `reward` -1.42e-01; valid WER 6.34.
- Checkpoint save, best-checkpoint recovery, and a final test-clean decode
  (%WER 6.10, rho 0.284) all succeeded.

So the GRPO rollouts, the band rate objective, the optimizer, checkpointing and
evaluation are all live. The WER here is a wiring result, not a quality result:
12 steps on dev-clean cannot say anything about the 4.70 the fully trained
checkpoint reaches. Peak GPU 6.4-13.3 GB of 79.25 GB.

**Other checks.** SpeechBrain unit tests: 828 passed, 5 skipped. Every launcher
that supports `DRY_RUN=1` now reaches its own anti-clobber guard rather than a
missing-asset error. `librispeech_prepare.py` run against the CLSP corpus
produces manifests byte-identical to the ported skipjack ones.

**Also restored in the same pass:**
- CoVoST 2 En-De (34 GB) and MuST-C v1 En-De (30 GB) extracted to
  `datasets/{covost2_en_de,mustc_en_de_prepared}`. Unlike the LibriSpeech CSVs,
  the ST manifests store one absolute `wav` path per utterance, so they needed
  `tools/rehome_manifests_clsp.sh` (14 manifests, all sampled paths resolve).
- `datasets/LibriSpeech_alignments` (MFA TextGrids) extracted; per-split counts
  28539/104014/148688/2703/2864/2620/2939 exactly match the documented corpus.
- `datasets/librispeech_manifests/dev-other.csv` was missing from the ported
  LS960 manifest set and blocked both multi-task launchers; regenerated with
  `librispeech_prepare.py` (test-other regenerated alongside it verified
  identical to the ported copy).
- `.cache/torch/hub/checkpoints/wav2vec2_fairseq_base_ls960_asr_ls960.pth`
  re-downloaded from download.pytorch.org for the CTC-posterior boundary path.

**Still outstanding:**
- **Reporting toolchain restored and made lossless (2026-08-31).** The
  `html-report` and `training-report` skills are installed at
  `~/.claude-scale/skills/`, so `htmlkit` imports and
  `make_training_report.py` regenerates cleanly (~2 min, login node, CPU only).
  The drift it exposed is fixed in commit `d0be3b7ec`: the regenerated page is
  byte-identical to the live one (2,085,302 bytes) apart from the timestamp, and
  every TIMIT number now derives from `audit_timit_phone_agreement.py`'s output
  rather than the rendered table. See `bugs.md` (2026-08-31).
  **Pending:** the source-branch push and the Pages re-sync were blocked by the
  Claude Code permission classifier and still need to be run. The live site is
  already correct -- the only pending change there is the generation timestamp.
- **`boundary_targets/word_ctc_sep` is a stalled transfer, not a missing asset.**
  7,435 of the expected 292,367 `.pt` files, stopping alphabetically at
  `1168-134958-0042` with an orphaned zero-byte rsync temp file
  (`.1168-134958-0042.pt.nnM5Cx`, 2026-08-31 01:36). It is `separator_target_dir`
  for the `blank_mode != keep` ablation only, so nothing on the flagship path is
  affected; resume the transfer before running that ablation. `word_ctc` in the
  same tree did complete (292,367).
- **Still absent from skipjack:** `hf/seamless` (SeamlessM4T comparison
  baseline), `boundary_targets/ctc_bpe500_state_runs_pilot` (a pilot artifact),
  and `datasets/mustc_en_de` (the raw MuST-C release -- only the prepared tree
  came over, which is enough unless the corpus is re-prepared).
- `build_flash_attention_2.sh` references skipjack's CUDA module at
  `/apps/software/extern/cuda/11.8.0`; only needed to rebuild flash-attn, which
  the validated production kernel does not use (see the SDPA note above).

## Cluster / Environment (HISTORICAL -- skipjack; superseded by the section above)

**Cluster:** bluecrab (JHU MARCC), SLURM, account `jsalt2026-lgarci27`
- **Hard GPU authorization pair (explicit user instruction, 2026-08-04): every GPU command/job must use both `--account=jsalt2026-lgarci27` and `--partition=a100`. Treat this account/partition pair as inseparable: never use the JSALT account with `b200`, `b300`, `h200`, or another GPU partition, and never use another account or an implicit/default account for GPU work. Do not submit, run, alter, reserve, or fall back to any non-A100 GPU node. CPU-only `med` with the JSALT account remains permitted when appropriate.**
- Partition: `a100` (only one this account can use, plus CPU-only `med`)
- Reservation: `"JSALT 2026"` on 8 nodes (`ga[129,132-134,138-141]`), still requires `--partition=a100` explicitly
- Known-bad node: `ga129` (NVLink/peer-GPU-memory hardware fault) — exclude with `--exclude=ga129`
- Always pass `--account=jsalt2026-lgarci27 --comment=accept_cost` on every `srun`/`sbatch`

**Python environment:**
- Dedicated venv: `~/cxiao/envs/jointllm` (uv-managed, Python 3.11)
- Key pinned versions: `torch==2.7.1+cu118`, `torchaudio==2.7.1+cu118`, `triton==3.3.1` (see `bugs.md` — do not let this float to latest, it breaks Triton JIT compilation on this cluster)
- `speechbrain` installed editable from this repo root (`pip install -e .`)

## Data Paths (HISTORICAL -- skipjack layout; see the CLSP path map above)

All under `/home/jhu/jsalt2026-ext-cxiao7/scratch_jsalt2026-lgarci27/omnienc/datasets/`:
- `LibriSpeech/` — standard raw corpus (all splits: train-clean-100/360, train-other-500, dev-clean/other, test-clean/other)
- `LibriSpeech_alignments/` — MFA word-level TextGrids, one per utterance; per-split file counts verified to exactly match the standard corpus (28539/104014/148688/2703/2864/2620/2939 for train-clean-100/-360/train-other-500/dev-clean/dev-other/test-clean/test-other respectively)
- `wavlm_boundaries/{word,phone,syllable,char,word_ctc}/` — precomputed `boundary_target_dir` tensors (owned by teammate ytseng21), 292367 `.pt` files per level (= full corpus, one per utterance). `word`/`phone`/`syllable` derive from the MFA TextGrids above; `char`/`word_ctc` derive from CTC forced alignment. Format: `torch.uint8`, values `{0,1}`, one label per SSL-encoder frame (50 Hz / 20 ms hop; 1 = frame starts a new segment).

**Shared HF model cache:** `/home/jhu/jsalt2026-ext-cxiao7/scratch_jsalt2026-lgarci27/omnienc/hf/hub` (multi-user; e.g. `meta-llama/Llama-3.2-1B-Instruct` already downloaded there by ytseng21). This account's own HF token is NOT authorized for gated repos — use this shared cache instead (see `decisions.md` ADR-002 for the exact env vars needed: `HF_HUB_CACHE` + `HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1`, plus pointing the relevant yaml `*_save_path` at the same directory).

## Multi-task ASR + Speech Translation (CoVoST 2 En-De)

- **Corpus**: CoVoST 2 En-De, not MuST-C (see ADR-024 -- MuST-C needs an FBK
  registration form and no HF mirror carries audio). Same English audio carries
  the English transcript and the German translation, which is the property the
  same-audio boundary analysis needs.
- **Raw parquet**: `.../users/cxiao/datasets/covost2_raw/en_de/` (24 shards,
  13.8 GB, from `fixie-ai/covost2`; `download_covost2.py`).
- **Prepared**: `.../users/cxiao/datasets/covost2_en_de/` -- 16 kHz mono FLAC
  under `audio/<split>/`, manifests under `manifests/`. Two views per split:
  `covost_asr_<split>.csv` and `covost_st_<split>.csv`, sharing `utt_key` and
  the audio path.
- **Sizes**: validation 15531 pairs / 26.1 h; test 15529 pairs / 24.6 h; train
  ~289k pairs (~430 h). FLAC costs roughly 57 MB per audio hour.
- **Text conventions**: English is folded to LibriSpeech style (upper case, no
  punctuation, apostrophes kept) so ASR WER stays comparable across the two ASR
  sources. German is verbatim -- sacreBLEU is case-sensitive detokenized.
- **LibriSpeech manifests** for replay are staged at
  `.../users/cxiao/datasets/librispeech_manifests/` (the LS960 set plus the
  `dev-other.csv` missing from the scaleup manifests dir; all `$data_root` paths
  verified).
- **Warm start**: `results/speechllm_ls960_step24k_corrected/transformer_ar_local64_bigru/nll_mt/3408/save/CKPT+2026-08-26+02-54-19+00`
  (dev-clean WER 2.879, best of the three-seed cohort; ADR-026).
- **Task ids** are fixed in `multitask_data.TASK_SPECS`: `asr=0`,
  `st_en_de=1`. Never reorder them -- checkpoints index FiLM embeddings and
  routed LoRA slots by these integers.
- **Decode cap check (2026-08-29)**: German targets run 3.14 tokens/s on average
  (p95 4.81, max 8.57) against the recipe's 12.0 tokens/s cap plus a 16-token
  margin. Zero truncations in 4000 dev utterances for either task, so no decode
  budget change is needed for German.
- **Equivalence gate (2026-08-29, job 171358)**: with the seed-3408 warm start,
  `task_id=asr` reproduces the single-task model -- boundaries exact, policy
  logits bit-identical, f_audio 10.58 Hz in both. The `projected` delta
  (2.4e-4) sits inside the same-model run-to-run GPU noise floor (1.9e-4), so it
  is float nondeterminism, not a functional difference.

- **Flagship run (2026-08-29)**: job **171461**, seed 3408, output
  `results/speechllm_multitask_ls960_covost_en_de/3408`. 15000 optimizer steps:
  bridge 0-3000 (frozen policy, decoder adapters + shared BiGRU), FiLM-only GRPO
  3000-9000, last-segmenter-block unfreezing 9000-15000. Validation every 1500
  steps over a fixed 3x1000-utterance mixture (CoVoST ST, CoVoST ASR,
  LibriSpeech dev-other). Preceded by smoke jobs 171389/171448 and the
  equivalence gate 171358.
- **Reading `task_metrics.jsonl`**: on TEST rows, `optimizer_step` is the step of
  the *recovered checkpoint* being evaluated, not the end of training; on VALID
  rows it is the current training step. `phase` always reflects the training
  phase reached.

- **ASR-only regression (2026-08-29, job 171453)**: the unchanged ASR segmenter
  recipe, run on the modified `segmenter.py`, reaches dev-clean WER 2.94 / 3.05
  at rho 0.211-0.208 (10.55/10.40 Hz) after a handful of steps from the
  seed-3408 warm start, i.e. the warm-started model's own operating point
  (2.879 WER, 10.58 Hz). The job ran to completion (COMPLETED, exit 0:0) with a
  final dev-clean test WER of 2.98 / CER 1.18 at rho 0.211 (10.55 Hz). Together
  with the bit-exact equivalence gate this establishes that the multi-task work
  did not disturb the ASR path.
- **Flagship first validation (job 171461, step 1500, bridge phase)**:
  WER_asr 8.50, sacreBLEU 8.73, chrF++ 32.47, f_audio 9.27 Hz (ASR mixture) /
  8.12 Hz (ST). Translation is non-degenerate well before policy RL begins, so
  the plan's "translation degenerate after the frozen-policy bridge" stop
  condition does not fire. Note the two rates are measured on different audio
  mixtures (the ASR view includes LibriSpeech dev-other), so they are NOT a
  task-conditioning comparison; only `analyze_multitask_boundaries.py` on paired
  CoVoST audio can make that claim.
- **Flagship bridge phase completed (job 171461)**: step 1500 BLEU 8.73 /
  chrF++ 32.47 / WER 8.50; step 3000 BLEU 12.59 / chrF++ 36.73 / WER 8.17. The
  frozen-policy bridge therefore produced a non-degenerate translation model
  before any policy update, and ASR improved rather than degraded. FiLM-only
  GRPO (Phase B) began at step 3000 with 4096/4096 FiLM parameters trainable and
  the boundary backbone still frozen. Bridge cost roughly 86 minutes.
- **Flagship FiLM phase (job 171461)**: BLEU 12.59 -> 16.46 -> 17.89 -> 18.79 at
  steps 3000/4500/6000/7500, with WER_asr improving 8.17 -> 7.88 and rates
  stable (ASR 9.27 -> 9.64 Hz, ST 8.12 -> 8.21 Hz; no collapse). During this
  phase the boundary backbone is frozen and the pooler does not affect
  placement, so FiLM is the only thing that can move the rate -- but the ASR and
  ST rates are measured on DIFFERENT audio mixtures and must never be compared
  to each other as evidence of conditioning.
- **Monitoring limitation**: validation `WER_asr` pools CoVoST ASR and
  LibriSpeech replay, because both carry `task_name="asr"`. Gate criterion 2
  (LibriSpeech dev-other within 0.2 absolute) therefore cannot be read from the
  validation rows; it is only available from the final test evaluation, which
  scores `librispeech_dev_other` as its own split. If mid-run retention
  monitoring is wanted in future runs, split the ASR metric by `source_key`
  rather than by `task_name`.
- **Flagship FiLM phase completed at step 9000**: BLEU 19.56, WER_asr 7.74.
  Net FiLM-phase movement was +6.97 BLEU and -0.43 WER at an essentially
  unchanged ST rate. **This is NOT yet a task-conditioning result**: the decoder
  adapters and shared pooler trained throughout the same phase, so the gain
  mixes continued decoder adaptation with any conditioning effect. Isolating
  conditioning requires the deferred frozen task-agnostic segmenter ST control
  and the paired same-audio mask analysis. Do not report the raw delta as
  evidence for the thesis.
- Phase C (last segmenter block unfrozen) began at step 9000 and runs to 15000.

## Non-ASR Multi-task Corpora (CLSP, verified 2026-09-01)

All three already live on shared read-only cluster storage; nothing to download,
and no TFDS dependency (`envs/jointllm` has no `tensorflow`/`tensorflow-datasets`,
and also no `datasets`, `peft`, or `librosa`).

- **CREMA-D** — `/export/corpora6/CREMA-D`. 7,442 WAV, 16 kHz mono PCM_16,
  91 speakers `1001`-`1091`, 5.26 h, mean clip 2.57 s. Crowd voice labels in
  `processedResults/summaryTable.csv` (`VoiceVote`); acted label is the third
  filename field. All 12 sentence transcripts are in `docs/README.md`.
  Voice-vote prior: neutral 57.3%, angry 14.5%, fearful 9.5%, disgust 8.0%,
  sad 5.4%, happy 5.2%; 644 clips (8.7%) are ties. Labelling and split decided
  in ADR-031.
- **Fluent Speech Commands** — `/export/corpora6/fluent_speech_commands_dataset`.
  16 kHz mono, official split present as `data/{train,valid,test}_data.csv`:
  23,132 / 3,118 / 3,793 rows, 14.72 / 1.95 / 2.58 h (19.25 h total), 31 intents,
  verified 0 speaker overlap between train and dev/test. Rows carry both
  `transcription` and the (action, object, location) triple.
- **LibriSpeech** — `/export/jsalt26/.../datasets/LibriSpeech` ->
  `/export/corpora5/LibriSpeech`; all three LS960 train splits plus dev/test
  present.
- **musan** — `/export/corpora6/musan` (noise augmentation).
- **Libri2Mix** — `/export/corpora5/Libri2Mix/wav16k` exists but is deliberately
  unused: a fixed two-speaker corpus cannot supply the 0-4 count range or the
  per-source activity intervals the boundary-alignment analysis needs.

**External test data (checked 2026-09-02):** the full Dynamic-SUPERB task-pack
tree is already staged in the shared evals area at
`/export/jsalt26/omnienc/evals/dynamic-superb/data/` (downloaded 2026-07-07),
including `HEARSpeakerCountIdentification_LibriCount-Fold1..5` (5 folds × 200
examples, labels 0–10, 48 kHz, ~0.83 h per fold; label 6 is rare, 5–6 per fold)
and `SpeakerCounting_LibriTTS-TestClean` (200 examples, labels one–five, 24 kHz,
variable 2–13 s durations). Measured zero-class level in the LibriCount folds is
~−48 dBFS (silence, not an audible floor) and RMS is flat across counts 1–10
(no energy confound); the LibriTTS pack has a monotonic RMS-vs-count confound.
Stress-17K/StressTest remain unstaged. Compute nodes do have outbound network
access (verified from `c25`, 2026-09-01), but SLURM jobs run with
`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, so stage from the login node
into `/export/jsalt26/...` rather than relaxing that inside a job.

**Prepared manifests** (2026-09-01) —
`/export/jsalt26/omnienc/users/cxiao/datasets/nonasr_manifests`, written by
`cremad_prepare.py` / `fsc_prepare.py`, both in the `MULTITASK_FIELDS` schema:

- `cremad_emotion_{train,valid,test}.csv` — voice labels, ties dropped:
  4,702 / 665 / 1,431 clips (3.31 / 0.47 / 1.01 h). Majority share 55.3% /
  62.3% / 61.7% (all neutral).
- `cremad_emotion_acted_{train,valid,test}.csv` — acted labels:
  5,147 / 738 / 1,557 clips, majority 17.1%. Within 4 clips of the TFDS split
  (5,144/738/1,556), which confirms the 63/9/19 speaker cut reproduces TFDS's
  speaker partition and that TFDS uses the acted label.
- `cremad_asr{,_acted}_{split}.csv` and `fsc_asr_{split}.csv` — ASR views over
  exactly the clips their task view keeps, sharing `utt_key` for the same-audio
  boundary comparison.
- `fsc_intent_{train,valid,test}.csv` — 23,132 / 3,118 / 3,793 clips,
  14.72 / 1.95 / 2.58 h, all 31 intents present in every split.
- `cremad_prep.json` / `fsc_prep.json` — speaker lists, per-split stats,
  provenance.

**Speaker-count mixtures** (2026-09-01; modes added 2026-09-02) — same
directory, from `speaker_count_prepare.py`:

- `speaker_count_{train,valid,test}.csv` — 20,000 / 2,000 / 2,000 five-second
  mixtures, balanced over counts 0–4, ~50% short-peak, from 2,338 train speakers
  and 40 each in dev-clean/test-clean.
- `speaker_count_{split}_recipes.jsonl` (8.4 MB total) — the mixtures are stored
  as recipes, never as audio. The manifest `wav` column holds a
  `mix:<recipes>#<utt_key>` pointer that `multitask_data.audio_pipeline`
  dispatches to `speaker_count_mixing.render_mixture`, which reconstructs the
  waveform deterministically from LibriSpeech audio already on disk.
- **Placement modes** (`--placement`, recipe field `placement`): `peak`
  (default) designs the common peak window as before; `random_offset` is
  LibriCount-style — independent random onsets, each source spanning at least
  3 s of the 5 s clip, accepted by rejection sampling only when the realized
  concurrency reaches the requested count (label stays "max concurrent
  speakers"), with the realized full-overlap interval stored in
  `peak_interval` (`full_overlap_interval` in `speaker_count_mixing`).
- **Zero-class styles** (`--zero-style`, recipe field `zero_style`): `noise`
  (−35 dBFS audible floor), `silence` (−48 dBFS, matching the measured
  LibriCount zero level), or `mixed` (either, equally likely; the default).
- Per-source activity intervals and the peak window are stored, so the
  boundary-alignment labels cannot drift from the audio.

**Two traps this task hit, both now guarded:**

- The LibriSpeech manifests' `spk_id` is a speaker-CHAPTER key (`2277-149874`):
  585 values in train-clean-100 for 251 real speakers, confirmed against
  `SPEAKERS.TXT`. Mixing two chapters of one reader would label a one-speaker
  mixture as two. `speaker_count_prepare.true_speaker` takes the leading field.
- The short-peak constraint alone did not remove the energy confound. On the
  first generated set a single RMS feature predicted the count at 48.3% (chance
  20%); 43.7% vs 25% on counts 1–4, Spearman +0.595. The mixer now normalizes
  the speech sum to a jittered −20 dBFS before adding the noise floor, giving
  29.0% on counts 1–4 (Spearman +0.137). Speech *density* still predicts at
  43.3% (Spearman +0.578) and is left alone — a real cue, not an artifact.

**Batching constraint (non-obvious).** SpeechBrain's `DynamicBatchSampler`
derives its bucket boundaries from `max_batch_length` and `num_buckets` **only**,
never from the corpus's durations (`_get_boundaries_through_warping`,
`speechbrain/dataio/sampler.py:548`). CREMA-D and FSC clips (~2.6 s) are far
shorter than the batch budget, so lowering `num_bucket` collapses them into
bucket 0 and pins the batch at `int(max_batch_length / boundary[0])` — at
`num_bucket: 10` that is 14 clips (~35 audio-s) instead of ~58, silently. At the
configured `num_bucket: 200` the measured fill is CREMA-D 43.7 clips / 110.8
audio-s and FSC 51.6 / 117.9. Do not reduce `num_bucket` for these corpora.

**Slurm note:** `sinfo` on 2026-09-01 shows `e01`-`e05` all up in `gpu-a100`;
the "e01 down" note in the Cluster Migration section above is stale.

## Repo / Branches

- Remote: `git@github.com:JSALT2026-OmniEnc/speechllm.git`
- Branches: `main`, `dsai`, `forced_alignment` (unmerged — has `prepare_boundary_targets.py`, the script that generated `wavlm_boundaries/`), `multitask`, `segmenter`
- Working branch for this work: `blank_analysis` (off `main`)

## Key Scripts

- `recipes/LibriSpeech/ASR/transformer/train_speechllm.py` — main SpeechLLM ASR training entry point (supports `boundary_source`: `none` / `fixed_rate` / `alignment`)
- `recipes/LibriSpeech/ASR/transformer/hparams/speechllm_fixed_pooling.yaml` — fixed/alignment-pooling config; fixed up 2026-07-22 with real cluster paths (previously full of placeholder paths from a different cluster and a gated LLM)
- `recipes/LibriSpeech/ASR/transformer/launch_fixed_pooling_ctc.sh` — `sbatch` launch script for the above (job-shape `#SBATCH` lines only; pass `--account`/`--partition`/`--comment`/`--reservation`/`--exclude` on the `sbatch` command line, not embedded in the script)

## Experimental Results

Group experiments by their **research purpose or question**. Each group gets its own named subsection with a one-line goal statement and a results table. Append new rows to the appropriate group; never edit existing rows.

---

#### Group: Boundary-source ablation for fixed-pooling SpeechLLM ASR

*Goal: compare mean-pooling boundary sources (fixed-rate vs. alignment-derived, and MFA vs. CTC alignment levels) for the frozen-SSL + frozen-LLM + LoRA SpeechLLM ASR setup on LibriSpeech.*

| ID | Date | Model / Config | Dataset / Split | WER | Cost / Throughput | Notes | Output |
|---|---|---|---|---|---|---|---|
| exp-001 | 2026-07-22 | `speechllm_fixed_pooling.yaml`, `boundary_source: alignment`, `boundary_target_dir: wavlm_boundaries/word_ctc` (CTC word boundaries), wav2vec2-base-960h (frozen) + Llama-3.2-1B-Instruct (frozen) + LoRA r16, 1 epoch, full 960h train | LibriSpeech dev-clean / dev-other / test-clean / test-other | 6.36% / 9.83% / 6.48% / 9.88% | 5h54m wall-clock (1x A100), ~1.4 it/s | CER 4.05%/6.19%/4.15%/6.25%. Train loss 6.0→0.72, valid loss 0.18. First real (non-debug) run on this pipeline — sane, non-garbage numbers, confirms CTC-word-boundary alignment pooling works for real training, not just mechanically. | `recipes/LibriSpeech/ASR/transformer/results/speechllm_fixed_pooling/alignment/3407/` |

---

## Analysis Notes

#### 2026-07-22 - First real alignment-pooling run validates the approach
- exp-001's WER (6.4-9.9% across splits) is a sane, non-degenerate result for 1 epoch with both backbones frozen and only LoRA + a projection layer trained — confirms `boundary_source: alignment` with CTC-derived word boundaries is a viable pooling strategy, not just mechanically functional.
- No comparison point yet against `word`-level MFA boundaries or the `fixed_rate` baseline on this same config/data — needed before concluding CTC boundaries are actually *better*, only that they work.

## Tips

- Keep entries current; never edit old experiment rows, only append
- Add an Analysis Notes entry whenever a pattern becomes clear across 2+ experiments

---

#### Group: Blank/separator-token removal ("remove blank") ablation

*Goal: quantify and explain why removing CTC separator/"blank" audio tokens (blank_mode:drop) hurts WER in the word_ctc fixed-pooling SpeechLLM. "Drop" = pool separator frames then discard the embedding before the decoder (word tokens untouched). All at 960h, 1 epoch, frozen SSL+LLM+LoRA, seed 3407.*

| ID | Date | Config | dev-clean | dev-other | test-clean | test-other | Notes |
|---|---|---|---|---|---|---|---|
| blk-keep | 2026-07-24 | blank_mode=keep, bs=1, max_decode_ratio=1 | 6.11 | 9.73 | 6.26 | 9.80 | Reproduces exp-001 (code re-validation). |
| blk-drop | 2026-07-24 | blank_mode=drop, bs=1, max_decode_ratio=1 | 21.77 | 22.80 | 22.26 | 23.26 | **Confounded**: deletion-dominated (8105 vs 124 del on dev-clean); del scales with utt length (2.5% @≤15w -> 26.9% @>40w). Cause: max_decode_steps = ratio*(audio prefix); drop halves audio tokens -> halves decode budget -> truncates long utts. |
| blk-keep-r3 | 2026-07-24 | blank_mode=keep, bs=16, max_decode_ratio=3 | 6.33 | 10.03 | 6.32 | 10.09 | Matched budget/batch baseline. |
| blk-drop-r3 | 2026-07-24 | blank_mode=drop, bs=16, max_decode_ratio=3 | 7.91 | 11.43 | 7.87 | 12.21 | Budget fixed (del 162, back to ~keep). **Residual gap +1.4-2.1% abs, now substitution-dominated.** |

#### 2026-07-24 - "Remove blank" is ~90% decode-budget artifact + a real ~1.5-2% residual
- Naive drop@960h looked catastrophic (dev-clean 6.11->21.77) but ~14 of the ~15.7pt gap was a **max_decode_ratio confound**: the greedy searcher's decode-step budget is `max_decode_ratio * audio-prefix-length`, and drop halves the audio-token count, starving long utterances (pure deletions, scaling with length). Re-decoding drop with max_decode_ratio=3 collapses deletions (8105->162) and recovers dev-clean to 7.91.
- **Genuine residual effect of removing blank tokens ≈ +1.5-2% WER** (matched bs=16/r=3: keep 6.33 vs drop 7.91 dev-clean), now **substitution-dominated** (+924 subs on dev-clean, not deletions). So the separator tokens do carry some useful signal, but the real effect is ~1.5-2%, an order of magnitude smaller than the naive number.
- Residual still needs disambiguation (content H5 vs structural-slot/count H2/H4/H6) via the content ablations (zero/shuffle/const), which hold token count/positions fixed at 2N and destroy only separator content. Compare to keep-r3 baseline at bs=16/max_decode_ratio=3.
- ACTION: ask co-worker what decode limit their original drop finding used — if tied to audio length, it carries the same confound and their effect size is likely inflated too.

#### Content ablations (2026-07-24): the residual is real GAP CONTENT (H5), not a register/slot

All 960h/1ep, eval bs=16 max_decode_ratio=3, vs anchors keep@r3 and drop@r3. (dev-clean/dev-other/test-clean/test-other)

| Condition | dev-clean | dev-other | test-clean | test-other | Δ keep (dev-clean) |
|---|---|---|---|---|---|
| keep (with blank) | 6.33 | 10.03 | 6.32 | 10.09 | 0 |
| shuffle (real sep content, permuted) | 6.72 | 11.05 | 6.83 | 10.67 | +0.39 |
| const (single LEARNED vector) | 7.60 | 11.02 | 8.21 | 12.30 | +1.27 |
| zero (content-free) | 7.76 | 10.97 | 7.55 | 11.73 | +1.43 |
| drop (remove sep) | 7.91 | 11.43 | 7.87 | 12.21 | +1.58 |
| blank_only (keep ONLY sep, drop words) | 38.51 | 39.45 | 38.78 | 40.09 | +32 |

- **const ≈ zero ≈ drop** (all ~7.6-7.9 dev-clean): a content-free slot -- even a LEARNED one -- recovers ~nothing. **Rejects H2 (register/[CLS]/sink), H4 (delimiter), H6 (token count):** extra content-free 2N slots don't help; a learned register token doesn't help; count per se doesn't help (const/zero keep 2N tokens, drop has N, all ~equal).
- **shuffle ≈ keep** (6.72 vs 6.33): keeping the REAL gap embeddings (even permuted among sep slots) recovers ~60% of the drop gap. So it's the *presence of real gap acoustic content*, not precise per-position alignment.
- **=> H5 confirmed: the separator/"blank" tokens carry real acoustic content** (co-articulation/transition cues) the LLM uses. Consistent with Phase-1 probes (sep emb -> pause duration R^2=0.90).
- **blank_only extreme**: training on ONLY the gap tokens (no word tokens) still reaches ~38% WER (62% words correct) -- the inter-word gaps alone are richly informative about the words. Strong H5 corroboration.
- Original user hypotheses resolved: "[CLS]/register token" -> REJECTED (const/zero); "boundary inaccuracy buffer" -> not the mechanism (Drop leaves word tokens unchanged; effect is gap content). Net: blanks carry real acoustic info, and the dramatic naive effect was ~90% a max_decode_ratio artifact.

---

#### Group: RL dynamic-downsampling segmenter — reward / multi-task ablation

*Goal: for the learned per-frame boundary policy (extends REBORN; see `plans/dynamic_segmenter_rl_v1.md`), compare (a) frozen vs multi-task (CE-co-trained) decoder during joint RL, and (b) NLL vs CER reward for the GRPO segmenter update. All: CNN backbone, char-CTC cold-start, warmup 6ep (decoder-CE only, segmenter frozen) -> joint 4ep (epochs 7-10), band rate objective [0.1,0.3], entropy off, train-clean-100, seed 3407.*

| ID | Date | Config | test-clean | test-other | dev-other | ρ (test) | Wall-clock | Notes |
|---|---|---|---|---|---|---|---|---|
| nll_frozen | 2026-08-02 | reward=nll, decoder frozen in joint (single-task) | 7.55 | 11.86 | — | 0.29 | 7h40m | job 42695. Baseline: RL only ever touches the segmenter. |
| nll_mt | 2026-08-02 | reward=nll, decoder co-trained in joint (multi-task) | 6.93 | 11.11 | — | 0.26 | 7h17m | job 42696. Beats nll_frozen on every split; valid WER improved monotonically through joint epochs (6.53→6.29→6.03) — RL genuinely refined boundaries. |
| cer_mt_opt | 2026-08-04 | reward=cer (free-running, KV-cached generate), decoder co-trained, off-policy GRPO (num_iterations=3, clip 0.2) | 7.53 | 11.81 | 11.68 | 0.297 | 15h38m | job 43226 (resumed epoch-6 warmup ckpt from cancelled 42697, ran only joint epochs 7-10). **Reported numbers are effectively the pre-RL warmup checkpoint, not a joint-RL result** — see analysis note below. |
| nll_frozen_shared_3seed | 2026-08-04 | reward=nll, decoder frozen in joint, corrected on-policy accumulation; seeds 3407/3408/3409 all recover exact epoch-6 `CKPT+2026-08-02+23-35-23+00` | 7.18±0.09 | 11.86±0.11 | 11.34±0.03 | 0.288±0.002 | 3h04m/run mean | jobs 45030/45033/45036. Output: `results/speechllm_segmenter/multiseed_shared_warmup_onpolicy/nll_frozen/cnn/`. |
| nll_mt_shared_3seed | 2026-08-04 | reward=nll, decoder co-trained, corrected on-policy accumulation; same exact warmup and seeds | **6.19±0.58** | **11.12±0.44** | **10.84±0.23** | 0.265±0.018 | 3h35m/run mean (1.17× frozen) | jobs 45031/45034/45037. Improves both test splits for all 3 seeds; raw SD is inflated by rare non-EOS insertion tails. Output: `results/speechllm_segmenter/multiseed_shared_warmup_onpolicy/nll_mt/cnn/`. |
| cer_mt_shared_3seed | 2026-08-05 | reward=cer, decoder co-trained, corrected on-policy (`batched_on_policy`, lr_segmenter=5e-5, pg_weight=0.5, max_decode_ratio=2); same exact CNN warmup/seeds as the nll rows above | 6.67±0.41 | 11.31±0.39 | 11.26±0.18 | 0.249±0.007 | ~11h10m/run mean (3.1× nll_mt) | jobs 45032/45035/45038, all COMPLETED. Per-seed valid WER improved monotonically through joint epochs 7-10 (e.g. seed 3407: 5.76→5.80→5.58→5.22) — unlike the earlier off-policy `cer_mt_opt`, this run genuinely trained, not just reproduced warmup. Output: `results/speechllm_segmenter/multiseed_shared_warmup_onpolicy/cer_mt/cnn/`. |

#### 2026-08-05 - Corrected on-policy CER reward now genuinely trains, but doesn't beat NLL multi-task and costs ~3x more

- **Resolves the 2026-08-04 open question**: the earlier `cer_mt_opt` off-policy run destabilized because of the algorithmic bug (3 optimizer steps/minibatch bypassing accumulation), not because CER reward is inherently unstable — with the same corrected accumulation-aware on-policy update used for `nll_mt_shared_3seed`, `cer_mt_shared_3seed`'s valid WER improves monotonically every joint epoch (7→10), for all 3 seeds.
- **But CER reward still underperforms NLL multi-task on every split**: cer_mt (6.67/11.31/11.26 clean/other/dev-other) sits between nll_frozen (7.18/11.86/11.34) and nll_mt (6.19/11.12/10.84) — beats the frozen baseline but doesn't match nll_mt's mean, though per-seed ranges overlap given nll_mt's higher seed variance.
- **Compute cost is the real differentiator**: cer_mt averages ~11h10m/run vs nll_mt's ~3h35m/run (3.1x), because the CER reward requires a free-running KV-cached `generate()` per training step to score hypotheses, vs. NLL reward's teacher-forced loss. For this task, NLL multi-task reward is the better reward choice on both quality and cost.

#### 2026-08-04 - CER-reward joint RL destabilized instead of improving (contrast with nll_mt)

- Unlike nll_mt, **cer_mt_opt's joint RL phase (epochs 7-10) never beat its own warmup starting point**: valid loss/WER at epoch 6 (end of warmup, segmenter frozen) = 1.98e-1 / 6.34%; every joint epoch 7-10 was worse on both (valid loss 2.79e-1/2.59e-1/2.28e-1/2.08e-1, valid WER 10.17/10.98/7.50/10.30 — noisy, non-monotonic; epoch 9's dip to 7.50 didn't hold at epoch 10). The checkpointer (min valid-loss) correctly kept epoch 6, so the "final" test numbers above are just the frozen-warmup checkpoint re-evaluated — i.e. **this run did not actually answer the NLL-vs-CER reward question**, it just reproduced the pre-RL baseline.
- Train-side signal during joint epochs: `train rho_std` (0.078-0.083) is visibly higher than the healthy nll_mt trace, consistent with the free-running CER reward being noisier per-step than teacher-forced NLL (K sampled segmentations get genuinely different decoded hypotheses, not just different teacher-forced losses).
- Open question / suggested next step: rerun with more joint epochs (the 4-epoch budget may simply be too short for the noisier CER reward to settle, unlike NLL which improved within the same 4), or check whether off-policy reuse (μ=3) is amplifying variance from stale rollouts. Not yet investigated.

#### 2026-08-04 - Shared-warm-up replication confirms co-training; evaluation has a rare insertion tail

- **Evidence-backed:** nll_mt beats nll_frozen in every paired seed on test-clean (−1.39/−0.23/−1.36 pp) and test-other (−0.26/−0.95/−0.99 pp); pooled gains are primarily fewer deletions (−1,372 clean, −892 other).
- **Evidence-backed:** rare >200%-WER hypotheses dominate nll_mt's seed SD (one 367-insertion test-clean output contributes 0.71 WER points); untrimmed WER remains the headline metric.
- **Inferred / priority:** the batch-global decode cap (`enc_states.shape[1]`, i.e. padded maximum) likely enables non-EOS items to consume a long batchmate's budget. Fix per-utterance evaluation length control and re-evaluate the six saved NLL checkpoints before interpreting variance.

---

#### Group: WavLM WER and compression controls (three seeds)

*Goal: establish whether compression and boundary placement improve WavLM-to-Llama ASR independently of learned RL.*

| ID | Date | Model / Config | Dataset / Split | WER | Cost / Throughput | Notes | Output |
|---|---|---|---|---|---|---|---|
| wl-base-k5 | 2026-08-10 | WavLM-Large, fixed mean pooling k=5, seeds 3407/08/09 | test-clean / test-other | 6.16±0.12 / 10.17±0.23 | 5h20m/run mean, 1× A100 (0.30× no-downsampling wall time) | Best fixed clean point at ρ=0.20; improves no-downsampling clean WER by 0.89. | `results/speechllm_fixed_pooling_wavlm/fixed_rate_k5_tc100/` |
| wl-base-native | 2026-08-10 | WavLM-Large, no downsampling, seeds 3407/08/09 | test-clean / test-other | 7.05±0.52 / 10.10±0.40 | 18h02m/run mean, 1× A100 (3.38× k=5) | Other is tied with k=5; clean is worse despite 5× more audio tokens. | `results/speechllm_fixed_pooling_wavlm/no_downsampling_tc100/` |
| wl-oracle-phone | 2026-08-10 | WavLM-Large, MFA phone boundaries, seeds 3407/08/09 | test-clean / test-other | 5.49±0.23 / 9.21±0.33 | 5h37m/run mean, 1× A100; ρ=0.210 | Closest oracle to k=5 in rate; establishes placement value at matched cost. | `results/speechllm_fixed_pooling_wavlm/phone_alignment_tc100/` |
| wl-oracle-char | 2026-08-10 | WavLM-Large, char-CTC boundaries, seeds 3407/08/09 | test-clean / test-other | 5.03±0.42 / 8.07±0.43 | 4h58m/run mean, 1× A100; ρ=0.293 | Strongest three-seed alignment baseline. | `results/speechllm_fixed_pooling_wavlm/char_alignment_tc100/` |

#### Group: WavLM learned-policy replication (three seeds)

*Goal: replicate the wav2vec2 reward/decoder/backbone grid with WavLM-Large features.*

| ID | Date | Model / Config | Dataset / Split | WER | Cost / Throughput | Notes | Output |
|---|---|---|---|---|---|---|---|
| wl-cnn-nll-frozen | 2026-08-10 | CNN, NLL reward, decoder frozen in RL | test-clean / test-other | 7.34±0.11 / 11.68±0.32 | 2h51m/run mean, 1× A100 | Stable but weaker than co-training on clean. | `results/speechllm_segmenter_wavlm/multiseed_shared_warmup_onpolicy/nll_frozen/cnn/` |
| wl-cnn-nll-mt | 2026-08-10 | CNN, NLL reward, decoder co-trained | test-clean / test-other | 6.93±0.47 / 12.01±0.42 | 3h55m/run mean, 1× A100 (1.37× CNN frozen) | WavLM is worse than matched wav2vec2 NLL-MT on both splits. | `results/speechllm_segmenter_wavlm/multiseed_shared_warmup_onpolicy/nll_mt/cnn/` |
| wl-cnn-cer-mt | 2026-08-10 | CNN, CER reward, decoder co-trained | test-clean / test-other | 7.07±0.59 / 10.95±0.20 | 12h50m/run mean, 1× A100 (3.28× CNN NLL-MT) | Best WavLM CNN test-other; CER remains expensive. | `results/speechllm_segmenter_wavlm/multiseed_shared_warmup_onpolicy/cer_mt/cnn/` |
| wl-tf-nll-frozen | 2026-08-10 | Transformer, NLL reward, decoder frozen in RL | test-clean / test-other | 8.53±0.77 / 11.88±0.63 | 2h54m/run mean, 1× A100 | Weak clean arm. | `results/speechllm_segmenter_wavlm/multiseed_shared_warmup_onpolicy/nll_frozen/transformer/` |
| wl-tf-nll-mt | 2026-08-10 | Transformer, NLL reward, decoder co-trained | test-clean / test-other | 6.99±0.70 / 10.88±0.50 | 3h08m/run mean, 1× A100 (1.08× TF frozen) | Strongest broad-sweep WavLM Transformer arm; substantially improves wav2vec2 Transformer test-other. | `results/speechllm_segmenter_wavlm/multiseed_shared_warmup_onpolicy/nll_mt/transformer/` |
| wl-tf-cer-mt | 2026-08-10 | Transformer, CER reward, decoder co-trained | test-clean / test-other | 7.36±0.23 / 11.29±0.44 | 12h17m/run mean, 1× A100 (3.93× TF NLL-MT) | No quality advantage over Transformer NLL-MT. | `results/speechllm_segmenter_wavlm/multiseed_shared_warmup_onpolicy/cer_mt/transformer/` |

#### Group: RL tests starting from the best char models

*Goal: measure the effects of the segmenter input features, decoder starting point, and use of previous boundary labels while keeping RL as the target method.*

| ID | Date | Model / Config | Dataset / Split | WER | Cost / Throughput | Notes | Output |
|---|---|---|---|---|---|---|---|
| hyb-oracle-init | 2026-08-10 | CNN segmenter on wav2vec2 → pool WavLM; best char decoder + 2-epoch bridge + 10 Bernoulli RL; 3 seeds | test-clean / test-other | 5.16±0.32 / 9.44±0.48 | 8h01m/run mean, 1× A100 (0.71× scratch wall time) | Fixed segmenter F1=0.8945; ρ≈0.234. Nearly closes char-oracle clean gap. | `results/speechllm_segmenter_wavlm/hybrid_w2v2_segmenter_controlled/oracle_char_decoder/` |
| hyb-scratch | 2026-08-10 | same segmenter/pooling/RL, fresh decoder + 6-epoch warmup; 3 seeds | test-clean / test-other | 6.48±0.14 / 10.04±0.15 | 11h18m/run mean, 1× A100 | Best char decoder initialization gains 1.32 clean / 0.61 other with the segmenter held fixed. | `results/speechllm_segmenter_wavlm/hybrid_w2v2_segmenter_controlled/scratch_decoder/` |
| oc-bern-1s | 2026-08-10 | CNN segmenter on WavLM, best char segmenter+decoder, 2 bridge + 10 Bernoulli RL, seed 3407 | test-clean / test-other | 5.03 / 9.45 | 7h42m, 1× A100 | ρ=0.229; matched control for first-order policy. | `results/speechllm_segmenter_wavlm/oracleclose_bestinit/bernoulli/3407/` |
| oc-ar1-1s | 2026-08-10 | same start, first-order label-autoregressive transition | test-clean / test-other | 4.86 / 9.39 | 13h07m, 1× A100 (1.70× Bernoulli) | Directional +0.17/+0.06 gain; single seed, not yet a replicated claim. | `results/speechllm_segmenter_wavlm/oracleclose_bestinit/autoregressive/3407/` |
| oc-bern-3s | 2026-08-11 | CNN segmenter on WavLM, best char segmenter+decoder, 2 bridge + 10 Bernoulli RL; seeds 3407/08/09 | test-clean / test-other | 5.62±0.52 / 9.73±0.66 | 9h25m/run mean, 1× A100 | Three-seed matched control; ρ≈0.234. | `results/speechllm_segmenter_wavlm/oracleclose_bestinit/bernoulli/` |
| oc-ar1-3s | 2026-08-11 | same start and schedule, first-order label-autoregressive transition; seeds 3407/08/09 | test-clean / test-other | 5.06±0.18 / 9.55±0.18 | 9h20m/run mean, 1× A100 | Beats Bernoulli on clean in all three seeds (−0.56 mean); other improves by 0.18 on average but not in every seed; ρ≈0.234. | `results/speechllm_segmenter_wavlm/oracleclose_bestinit/autoregressive/` |
| w2v-tf-ar-long-3s | 2026-08-11 | wav2vec2 Transformer segmenter, shared best NLL-MT checkpoint, first-order AR, 30 RL epochs; seeds 3407/08/09 | test-clean / test-other | 6.35±0.28 / 11.67±0.16 | 21h59m/run mean, 1× A100 | Best dev WER 5.43±0.06 at epochs 1/2/4; epoch-30 dev WER rebounds to 6.05±0.10, so the extra epochs overfit. | `results/speechllm_segmenter/w2v2_transformer_autoregressive_long/nll_mt/` |

#### 2026-08-10 - Cumulative interpretation after oracle and fixed-segmenter controls
- Compression is beneficial but rate alone is insufficient: k=5 beats native input on clean, while phone alignment beats k=5 at nearly identical ρ.
- With the same decoder, oracle, cold-learned, and post-RL boundaries give 4.27, 9.14, and 17.37 dev-clean WER. Together with the fixed-segmenter decoder-start test, this shows that boundary errors and decoder training—not a missing checkpoint or frame-grid error—explain the current gap.
- The long first-order wav2vec2 runs improve dev WER early (mean best 5.43 vs 5.85 initialization) but overfit after epochs 1–4; use early best checkpoints and improve structure/scheduling rather than assuming a larger epoch budget yields convergence.

#### 2026-08-11 - First-order boundary history helps clean speech, but longer training does not
- In the matched WavLM study, first-order AR beats Bernoulli on test-clean in every seed (5.06±0.18 vs. 5.62±0.52); the smaller test-other change (9.55±0.18 vs. 9.73±0.66) is mixed by seed.
- The 30-epoch wav2vec2 Transformer runs select epochs 1/2/4 and deteriorate afterward while training loss falls. Future structured-policy work should use early stopping or decay rather than extending the same constant-learning-rate schedule.

---

#### Group: Boundary agreement with char-CTC

*Goal: compare each Figure-2 boundary stream against saved char-CTC targets on the full dev-clean split. Learned systems use the exact WER-selected checkpoint for each seed. Exact F1 requires the same encoder frame; ±1 F1 permits a one-frame offset.*

| ID | Date | Model / Config | Dataset / Split | Boundary F1 (exact / ±1 frame) | Cost / Throughput | Notes | Output |
|---|---|---|---|---|---|---|---|
| controlled-boundary-audit | 2026-08-11 | Figure-2 systems; 3 seeds for learned segmenters, deterministic baselines once | LibriSpeech dev-clean, 2,703 utterances | wav2vec2→WavLM char decoder: 72.5±1.8 / 83.7±0.5; wav2vec2→WavLM scratch decoder: 73.6±1.0 / 85.3±1.7; WavLM Bernoulli: 55.8±0.4 / 80.0±0.6; WavLM first-order AR: 57.2±1.1 / 79.1±0.8; WavLM Transformer: 47.4±6.4 / 73.7±4.6 | 7m19s, 1× A100 | Char/phone/k=5/native deterministic F1: 100.0/100.0, 32.2/63.9, 23.9/52.1, 45.9/46.3. Raw frame accuracy is not reported because non-boundary frames dominate it. | `artifacts/segmenter/controlled_boundary_accuracy.json` |

#### 2026-08-11 - Char-boundary F1 diagnoses the segmenter but does not determine WER
- The scratch-decoder arm is slightly closer to char-CTC than the char-decoder arm (73.6/85.3 vs. 72.5/83.7 exact/±1 F1), yet its WER is substantially worse (6.48/10.04 vs. 5.16/9.44 clean/other). Decoder initialization is therefore an independent factor.
- First-order AR improves exact F1 over Bernoulli by 1.4 points but lowers ±1 F1 by 0.8 points, while improving clean WER by 0.56 points. Use char-F1 as a segmenter diagnostic, not as a stand-alone optimization target.
- The WavLM Transformer segmenter has the lowest learned mean and the largest seed spread (47.4±6.4 exact; 73.7±4.6 within ±1), consistent with its observed training instability.

#### 2026-08-11 - The best learned policy reads WavLM; wav2vec2 supplies only its cold-start labels
- The best replicated learned system is the WavLM-input CNN segmenter with first-order AR and the best char decoder (5.06±0.18 / 9.55±0.18 test-clean/other). It does not read wav2vec2 features at inference or during RL; wav2vec2 enters only through the saved char-CTC target labels used for supervised cold start.
- In the cleaner Bernoulli comparison, changing the segmenter input from WavLM to wav2vec2 improves WER from 5.62±0.52 / 9.73±0.66 to 5.16±0.32 / 9.44±0.48 and raises exact/±1 char-boundary F1 from 55.8/80.0 to 72.5/83.7. Encoder/target matching contributes, but the modest WER change relative to the large exact-F1 change shows it is not the whole performance gap.
- First-order boundary history recovers the WavLM clean result without wav2vec2 inputs (5.06 vs. 5.62 Bernoulli), even though its char-F1 changes little. The remaining gap to oracle char boundaries is essentially zero on clean (5.06 vs. 5.03) but about 1.48 WER points on other (9.55 vs. 8.07), focusing later analysis on boundary robustness and decoder adaptation for harder speech rather than on changing the runtime encoder alone.

#### 2026-08-23 - One-week research review and current corrected anchors

- The corrected main-report anchors are: WavLM character oracle 3.96 +/- 0.41 /
  7.86 +/- 0.71 WER at 14.6 Hz; WavLM local-history Transformer AR + BiGRU
  4.80 +/- 0.09 / 9.24 +/- 0.36 at 10.4 Hz; and WavLM CNN first-order AR mean
  4.86 +/- 0.13 / 9.37 +/- 0.17 at 11.6 Hz. These supersede older intermediate
  decoding summaries when discussing the current paper.
- The strongest learned system is 0.84 clean and 1.38 other WER points behind the
  character oracle while using about 29% fewer decoder-side audio tokens. The
  immediate experiment is decoder prefix-rate adaptation (OPD), followed only on
  success by constrained frequency continuation.
- The current system is not end-to-end streaming: WavLM uses future context and
  Llama receives a complete audio prefix before text generation. The AR segmenter
  is causal only with respect to boundary-label history. A streaming claim requires
  a causal/bounded-lookahead encoder and an interleaved read/write or transducer-like
  decoder.
- STAR (IWSLT 2025) is the closest prior work and already covers dynamic compression
  for streaming ASR and speech translation. In this project, streaming is a
  secondary contribution of the main RL segmentation pitch rather than a separate
  pitch. The one-week target is a bounded-lookahead learned-versus-fixed comparison;
  an efficient interleaved Llama decoder remains follow-up engineering. See
  `reports/one_week_research_strategy_2026-08-23.md` and ADR-014.

---

#### Group: Transformer-AR fixed-window-64 speed pilots

*Goal: reduce full-prefix Transformer-AR joint-RL time without changing `K=4`; all local-history tests fix the window at 64 frames.*

| ID | Date | Model / Config | Dataset / Split | Metric | Cost / Throughput | Notes | Output |
|---|---|---|---|---|---|---|---|
| tar-attn64-kernels | 2026-08-12 | A100 BF16 local causal attention, window=64; math SDPA vs fused masked SDPA vs compiled Flex | synthetic B=16, H=4, D=64; T=256/512/1024 | Flex vs fused SDPA speed: 0.53× / 1.42× / 3.11×; max BF16 output/gradient error ≤0.03125 | job 70087, 1× A100; Flex memory 16–24% lower than fused SDPA | Forced Flash is unsupported with a non-null local mask. Fused masked SDPA is already 6.7–10.2× faster than math SDPA. | `results/speechllm_segmenter_wavlm/transformer_ar_attention_bench/kernels_70087.json` |
| tar-train64-reward-batch | 2026-08-12 | preallocated KV, window=64, K=4; four NLL rewards in one decoder forward, greedy/sample rollouts separate | LibriSpeech real joint-RL debug, 30 train batches | finite matched training; rho≈0.28 | 91 s train loop / 226 s full mock run, 1.11× train-loop speedup vs 101 s reference | Reward batching alone gives a modest but consistent gain. Jobs 70091/70093. | `results/speechllm_segmenter_wavlm/transformer_ar_flex_training_bench/` |
| tar-train64-group-batch | 2026-08-12 | preallocated KV, window=64, K=4; greedy+four sampled rollouts share one sequential loop and four rewards share one decoder forward | LibriSpeech real joint-RL debug, 30 train batches | finite matched training; rho≈0.28 | 52 s train loop / 186 s full mock run, **1.94× train-loop speedup** vs reference | Strongest speed result. Direct unit check gives zero greedy-boundary mismatches and 2.4e-7 max FP32 logit drift. Jobs 70091/70094. | `results/speechllm_segmenter_wavlm/transformer_ar_flex_training_bench/` |
| tar-train64-flex | 2026-08-12 | compiled FlexAttention, preallocated KV, window=64, K=4; reference/reward/group batching arms | LibriSpeech real joint-RL debug, 30 train batches | all arms finite; steady-state batch rate approximately tied with fused SDPA | 110/98/62 s train loops for reference/reward/group; compile adds ~7–10 s | Flex helps isolated T≥512 attention, but not end-to-end at the observed dynamic-batch length mix; retain fused SDPA for now. Jobs 70092/70095/70096. | `results/speechllm_segmenter_wavlm/transformer_ar_flex_training_bench/` |
| tar-combined-production-smoke | 2026-08-12 | production `combined_on_policy`; preallocated KV, window=64, K=4; grouped greedy/sample rollout + one NLL reward batch | LibriSpeech real joint-RL debug; 4 train, 4 validation, 4 test batches | finite train/valid/test; checkpoint save+reload passed; test WER 8.11 is debug-only | job 70130, 36 s wall, 8.3/10.2/3.1 s train/valid/test, 1× A100 | Confirms the non-monkeypatched production path, gradient accumulation, checkpointing, and evaluation complete. | `results/speechllm_segmenter_wavlm/transformer_ar_combined_rollout_validation/smoke_combined_70130/` |
| tar-combined-full-loop | 2026-08-12 | matched reference vs production `combined_on_policy`; same seed/checkpoints, WavLM, K=4, window=64, preallocated KV | full LibriSpeech-100h one-epoch train → dev-clean → test-clean | running | jobs 70131/70132, 1× A100 each on ga129 | Full speed/stability gate; separate GPUs, no effect on production jobs on ga132. | `results/speechllm_segmenter_wavlm/transformer_ar_combined_rollout_validation/` |

#### 2026-08-12 - Batch rollouts and rewards; do not switch the segmenter to Flex yet

- The local masked-SDPA path is already fused. Compiled Flex is slower at T=256, faster at T=512/1024, but its gain disappears after projections, FFNs, WavLM, decoder work, and dynamic batch lengths are included.
- Combining greedy plus four sampled trajectories removes one complete 50 Hz sequential loop; batching the four NLL rewards removes three decoder launches. Together they cut the measured 30-batch training loop from 101 s to 52 s while keeping `K=4` and the window at 64.
- The combined path is now a disabled-by-default production mode named `combined_on_policy`. The short production smoke passed; matched full-loop jobs 70131/70132 are the remaining gate before using it for new training. Flex remains optional until longer-sequence batches dominate or compile overhead is amortized.

---

#### Group: LibriSpeech-960h scale-up controls

*Goal: establish full-data oracle, fixed-rate, and segmenter-initialization controls before interpreting the LS960 learned-policy runs.*

| ID | Date | Model / Config | Dataset / Split | WER / F1 | Cost / Throughput | Notes | Output |
|---|---|---|---|---|---|---|---|
| ls960-oracle-char-3s | 2026-08-24 | WavLM-Large, character-CTC oracle mean pooling, best char decoder; seeds 3407/08/09 | LS960 train; test-clean / test-other | WER **2.87±0.08 / 5.66±0.15** | 2h25m/run mean, 1× A100; 3,584 optimizer steps; ~14.6 Hz | Dev-clean 2.92±0.13. Completed jobs 105296/105297/105298. | `results/speechllm_ls960_scaleup_v2_optimized/oracle_char_decoder/` |
| ls960-fixed-k5-3s | 2026-08-24 | WavLM-Large, fixed mean pooling at k=5; seeds 3407/08/09 | LS960 train; test-clean / test-other | WER **4.37±0.10 / 7.72±0.18** | 1h55m/run mean, 1× A100; 3,584 optimizer steps; 10.0 Hz | Dev-clean 4.51±0.22. Completed jobs 105307/105309/105311. | `results/speechllm_ls960_scaleup_v2_optimized/fixed_k5/` |
| ls960-segmenter-coldstarts | 2026-08-24 | One LS960 pass of char-CTC supervision; local-64 Transformer AR or CNN | dev-clean boundary prediction | exact F1: Transformer **75.2%**, CNN **79.6%** | 55m Transformer / 49m CNN, 1× A100 each | CNN is 4.4 F1 points stronger after the same one-pass cold start. Completed jobs 105299/105300. | `results/speechllm_ls960_scaleup_v2_optimized/coldstart/` |
| ls960-native-3s | 2026-08-24 | WavLM-Large, no downsampling at native 50 Hz; seeds 3407/08/09 | LS960 train; test-clean / test-other | WER **4.34±0.47 / 7.08±0.19** | 5h46m/run mean, 1× A100; 3,936 optimizer steps; 50.0 Hz | Completed jobs 105308/105310/105312. Native is 0.03 better clean and 0.64 better other than fixed k=5, but uses 5× the decoder-side audio tokens. | `results/speechllm_ls960_scaleup_v2_optimized/no_downsampling/` |
| ls960-cnn-ar-bigru-nowarm-2s | 2026-08-24 | WavLM CNN segmenter, first-order AR policy, BiGRU pooling, oracle-char decoder initialization; seeds 3407/08 complete | LS960 train; test-clean / test-other | provisional WER **3.30±0.20 / 6.23±0.05** | 4h03m/run mean, 1× A100; ~11.5 / 11.3 Hz on clean / other | Jobs 105302/105304 complete; seed 3409 job 105306 pending. Invalid for the intended warmup study because joint RL began at step 0; retain only as a no-warmup diagnostic. | `results/speechllm_ls960_scaleup_v2_optimized/cnn_first_order_ar_bigru/nll_mt/` |

#### 2026-08-24 - LS960 controls improve strongly, but learned runs do not implement the intended warmup

- Relative to the 100h controls, LS960 reaches 2.87/5.66 WER with oracle character boundaries and 4.37/7.72 with fixed k=5 on test-clean/test-other (lower is better).
- The running learned jobs 105301–105306 are not the intended “20% decoder warmup + 80% joint RL” condition: they log `Decoder phase=joint-RL optimizer_step=0`. Treat their eventual scores as no-warmup LS960 runs unless they are rerun after the bug in `bugs.md` is fixed.
- The first two CNN first-order AR + BiGRU no-warmup seeds average 3.30/6.23 WER at about 11.5/11.3 Hz. This is 1.07/1.49 points better than LS960 fixed k=5 and 1.04/0.86 better than native 50 Hz, while remaining 0.43/0.57 behind the oracle-char control; wait for seed 3409 before treating the variance as final.

#### 2026-08-24 - Corrected 24k-step LS960 convergence queue

- Jobs 106155–106160 cover CNN first-order AR + BiGRU and local-64 Transformer AR + BiGRU, each with seeds 3407/08/09, under `results/speechllm_ls960_step24k_corrected/`.
- Each job stops at 24,000 optimizer steps, validates at warmup end (step 2,395) and steps 4k/8k/12k/16k/20k/24k, and selects minimum dev-clean WER. All jobs use the three LS960 training splits, K=4, 64-frame Transformer history, BF16, one A100, 12 CPUs, 64 GB RAM, and exclude ga129.
- This queue fixes the fractional-warmup propagation and effective-accumulation bugs; the earlier jobs 105301–105306 remain no-warmup diagnostics.

#### 2026-08-24 - Final no-warmup diagnostics and first corrected-run validation

- All six earlier no-warmup learned-policy jobs completed successfully. The CNN
  first-order AR + BiGRU arm (105302/105304/105306) reaches **3.23±0.18 / 6.17±0.11**
  test-clean/test-other WER across seeds 3407/08/09 at about 11.6/11.3 Hz. The
  local-64 Transformer AR + BiGRU arm (105301/105303/105305) reaches
  **3.29±0.12 / 6.39±0.07** at about 11.6/11.2 Hz. These are useful diagnostic
  anchors but do not test the intended decoder-warmup schedule.
- Corrected job 106155 (Transformer seed 3407) logs the intended transition:
  decoder-only warmup through optimizer step 2,395, then joint RL. Dev-clean WER
  improved from **4.45 at step 2,395** to **4.17 at step 4,000**; step-8,000
  validation was in progress at the 14:54 EDT status check. The other five
  corrected jobs were pending on `MaxGRESPerAccount`.
- At 18:28 EDT, pending jobs 106156–106160 were briefly restricted to the
  non-reserved A100 nodes `ga130,ga131,ga137,ga142,ga143` while retaining their
  original job IDs and queue age. None allocated: SLURM reported
  `MaxGRESPerAccount`. Broad eligibility was then restored (with ga129 still
  excluded), after which the reason returned to `Priority`. The account has only
  the `jsalt_2026` QOS, whose account-wide GPU cap is 64, so there is no separate
  authorized general-A100 QOS that bypasses the cap.

#### 2026-08-25 - First corrected 24k-step LS960 results and cluster audit

- Corrected Transformer AR + BiGRU seed 3407 (job 106155) completed normally at
  24,000 optimizer steps: **2.78 test-clean / 5.49 test-other WER** at about
  **11.1 / 10.7 Hz**. Dev-clean WER improved monotonically overall from 4.45 at
  warmup end to 2.94 at step 24k.
- Corrected CNN first-order AR + BiGRU seed 3407 (job 106156) also completed
  normally: **2.79 / 5.56 WER** at about **11.7 / 11.4 Hz**. Its best dev-clean
  WER was 2.98 at step 24k.
- At the 13:28 EDT cluster audit, Transformer seed 3408 (106157) was running on
  ga137 with zero restarts and a live log at batch 11,143/11,977 of the first
  pass (about 11.1k/24k optimizer steps). Jobs 106158–106160 remained pending on
  priority. No current corrected job had failed, and no traceback, OOM, killed
  step, I/O error, or transport error was present, so nothing was resubmitted.
- Later on 2026-08-25, job 106157 incurred `NODE_FAIL` after 10h29m on ga137 and
  was automatically requeued under the same job ID. The latest intact checkpoint
  is optimizer step 13,867; the log ended near step 14,342, so recovery loses only
  about 475 steps. ga137 subsequently rebooted and returned to service. At the
  17:39 EDT check, jobs 106157–106160 were pending on `MaxGRESPerAccount`; no
  manual duplicate submission was needed.
- Queue strategy was changed later that day: jobs 106157–106160 were attached in
  place to the active `JSALT 2026` reservation, preserving IDs/checkpoints and the
  ga129 exclusion. The account was exactly at its 64-GPU QOS cap, so the jobs
  remained pending, but five healthy reserved GPUs were idle and become immediately
  usable once an account allocation ends. Future LS960 launchers now default to the
  reservation, with an explicit empty override available after it expires. The
  pending jobs and launcher defaults use a two-day limit rather than three days,
  providing additional safety beyond the measured 18h16m Transformer runtime.
- By 21:51 EDT, the reservation strategy had placed three jobs: Transformer seed
  3408 (106157) on ga138, CNN seed 3408 (106158) on ga133, and CNN seed 3409
  (106160) on ga138. Their latest checkpoints were steps 16,983, 8,453, and
  11,641 of 24,000, respectively, with live progress beyond each checkpoint.
  Transformer 3408 reached 3.19 dev-clean WER at step 16k; the CNN seeds reached
  3.51/3.68 at step 8k. Logs were clean after recovery. Transformer seed 3409
  (106159) remained pending on `MaxGRESPerAccount`; its failed 22-minute ga137
  allocation had not reached the first periodic checkpoint, so it will restart
  from the intended initialization.
- At 01:09 EDT on 2026-08-26, all four remaining jobs were running in the JSALT
  reservation: 106157/106160 on ga138, 106158 on ga133, and 106159 on ga132.
  Live progress was approximately 21.9k, 18.6k, 3.4k, and 21.2k of 24k steps for
  Transformer-3408, CNN-3408, Transformer-3409, and CNN-3409, respectively. The
  latest dev-clean WERs were 3.04 at 20k, 2.98 at 16k, 4.36 at warmup end, and
  3.15 at 20k. All post-recovery logs were clean.
- The cumulative HTML report now leads with an LS960 table and convergence plot.
  Three-seed oracle/fixed/native controls are separated from the provisional
  corrected learned-policy results (`n=1` per setup), and the report explicitly
  notes that the 24k-step learned runs are not training-cost matched to the one-pass
  controls. The Pages publication commit is `4dfa5b7`.

#### 2026-08-26 - Publication experiment priority audit

- Do not expand to a second task before resolving the current ASR confounds. The
  24k-step learned LS960 systems have a longer optimization budget than the
  one-pass oracle/fixed/native controls, and the headline comparison also changes
  segment placement, BiGRU pooling, and joint RL together.
- After the running three-seed LS960 study, the highest-priority experiments are:
  (1) fixed + BiGRU and frozen-learned-segmenter + BiGRU controls with a matched
  decoder initialization/training budget, (2) transcript-free CTC posterior
  compression, (3) an identical end-to-end efficiency protocol, and (4) a small
  learned-frequency sweep plus per-utterance adaptivity analysis.
- A second task is gated on those controls. If later needed, begin with a frozen
  segmenter transfer pilot rather than another full joint-RL sweep. Full rationale:
  `reports/publication_experiment_priorities_review_2026-08-26.md`.

#### 2026-08-26 - Matched 100h Transformer-AR + BiGRU attribution study

- Jobs 113183–113191 are the three-seed, train-clean-100 attribution matrix under
  `results/speechllm_segmenter_wavlm/attribution_100h_transformer_ar_bigru/`:
  fixed k=5 + BiGRU (113183/113186/113189), frozen supervised Transformer-AR
  segmenter + BiGRU (113184/113187/113190), and joint Transformer-AR RL + BiGRU
  (113185/113188/113191), for seeds 3407/3408/3409.
- Every arm uses the same WavLM features, local history 64, best character-decoder
  initialization, supervised Transformer-AR segmenter initialization, zero-residual
  BiGRU initialization, seven decoder/pooler CE epochs, optimizer settings, data,
  and batch-invariant decoding. The joint arm uses two decoder/pooler-only epochs
  followed by five NLL-GRPO epochs; K remains fixed at 4.
- `freeze_boundary_policy` explicitly freezes FiLM/policy parameters while retaining
  trainable BiGRU parameters. `fixed_rate_k=5` bypasses the policy with the same
  fixed-boundary implementation as the established baseline. Nineteen focused unit
  tests pass, including nonzero BiGRU gradients with no policy gradients and the
  batch-invariant decoder suite.
- All jobs request one A100, 12 CPUs, 60 GB RAM, a two-day limit, the JSALT
  reservation, the required account/comment, and exclude ga129. They were initially
  accepted as pending at 01:48 EDT.

#### 2026-08-26 - Attribution and corrected LS960 status

- The first matched 100h control, fixed k=5 + BiGRU seed 3407 (113183), completed
  normally in 3h04m. The selected epoch-4 checkpoint reaches **5.19 test-clean /
  9.67 test-other WER** at about **10.1 Hz**. This is one seed; matched conclusions
  wait for all three arms and seeds.
- Frozen learned Transformer-AR segmenter + BiGRU seed 3407 (113184) is healthy in
  epoch 1 at batch 1,778/4,006 (44%). Its log confirms learned argmax boundaries,
  frozen boundary-policy parameters, trainable BiGRU parameters, and the intended
  best character-decoder initialization. Jobs 113185--113191 remain pending on the
  account-wide `MaxGRESPerAccount` limit.
- Five of six corrected 24k-step LS960 jobs are complete. The CNN first-order AR +
  BiGRU arm is complete across three seeds at **2.75±0.05 / 5.67±0.10 WER**
  (test-clean/test-other), averaging about **11.9 / 11.6 Hz**. The two completed
  Transformer-AR + BiGRU seeds give a provisional **2.79±0.03 / 5.60±0.15**
  at about **10.8 / 10.5 Hz**. These values use corpus error counts before
  aggregation rather than the rounded WER displayed in individual train logs.
- The final LS960 run, Transformer seed 3409 (106159), is healthy in epoch 2 at
  roughly optimizer step 13.7k/24k (57%). Its dev-clean WER improved from 4.36 at
  warmup end to 4.18 at step 4k, 3.68 at 8k, and 3.43 at 12k. No current job shows
  a traceback, OOM, or decoding failure.
- The HTML training report now derives the corrected LS960 summaries directly from
  completed artifacts. The CNN row is complete (3/3), the Transformer row is marked
  provisional (2/3), and the unequal training-budget caveat remains explicit. The
  validated Pages update is commit `d3f6453`.

#### 2026-08-27 - Corrected LS960 study complete; 100h attribution progressing

- Corrected Transformer AR + BiGRU seed 3409 (106159) completed successfully at
  **2.79 test-clean / 5.68 test-other WER**, about **11.3 / 11.0 Hz**. The complete
  three-seed Transformer aggregate is **2.79±0.02 / 5.63±0.12 WER** at about
  **10.9±0.4 / 10.6±0.4 Hz**. Together with the completed CNN result
  (**2.75±0.05 / 5.67±0.10** at 11.9/11.6 Hz), the two policy families are tied
  within seed variation; Transformer uses roughly one fewer audio token per second.
- The matched 100h fixed-k=5 + BiGRU arm is complete across three seeds at
  **5.17±0.05 / 9.70±0.12 WER** and about 10.1 Hz.
- Frozen learned-segmenter seed 3407 finished seven training epochs and selected
  epoch 3 (best dev-clean 4.95); test-clean/test-other are **4.79 / 9.28** at about
  14.4/13.9 Hz, with its final dev-other evaluation still running at the status
  check. The other two frozen seeds and all three joint-RL seeds remain active.
- Joint-RL seed 3407 improved to 5.10 dev-clean at epoch 4 (about 11.7 Hz) but
  worsened to 6.01 at epoch 5 (about 12.6 Hz); it is in epoch 6. This makes selected
  checkpoint behavior more important than the last epoch and is an early warning of
  joint-stage drift, not yet a three-seed conclusion. No active attribution job has
  an OOM, traceback, or node failure.
- The training report was regenerated and published at Pages commit `9f90628`. It
  now treats both corrected LS960 learned systems as complete three-seed results and
  adds a matched 100h attribution section with a labeled test table, test-WER plot,
  and separate joint-RL dev-progress table. Pending joint test cells remain blank;
  dev WER is not substituted for final test evidence.

#### 2026-08-27 - Matched 100h joint-RL arm complete

- All three matched Transformer AR + BiGRU joint-RL jobs completed successfully.
  The aggregate is **4.95±0.02 test-clean / 9.11±0.18 test-other WER** at about
  **11.2±0.1 / 10.8±0.1 Hz**. Individual selected epochs are 7/7/5 for seeds
  3407/3408/3409, showing that checkpoint selection recovered from temporary dev
  regressions rather than selecting the last checkpoint blindly.
- Against the matched fixed-k=5 + BiGRU control (**5.17±0.05 / 9.70±0.12** at
  10.1 Hz), joint RL improves by **0.22 clean / 0.59 other WER points** while using
  about 1.1/0.7 more audio tokens per second on clean/other.
- Two frozen-segmenter seeds are complete at a provisional **4.90±0.15 /
  9.37±0.13 WER** and 14.4/13.9 Hz. Joint RL is therefore essentially tied on
  clean (+0.05) and 0.26 points better on other while using about three fewer audio
  tokens per second. Frozen seed 3409 is in final evaluation; all other attribution
  jobs completed with exit code zero.

#### 2026-08-27 - RTF benchmark status

- The earlier preselected-operating-point suite was complete across three seeds:
  no downsampling was **0.082±0.006 RTF** (batch 2), fixed k=5 was
  **0.022±0.000** (batch 8), CNN first-order AR mean was **0.026±0.001** (batch
  8), and local-history Transformer AR + BiGRU was **0.037±0.001** (batch 8).
  These values are superseded in the report by the final memory-limited protocol
  summarized below.
- The stricter memory-limited follow-up completed all 12 batch selectors and all
  batch-1 passes, but only 24/36 evaluation jobs completed end to end. Selected
  batches were 16 for native 50 Hz, 32 for both Transformer arms, and 64 for
  fixed k=5 / both CNN arms. The fixed-k=5 and CNN-mean selected-batch passes
  failed 3/3 with CUDA OOM; CNN+BiGRU failed 1/3. There are no active RTF jobs.
- Complete follow-up values include native 50 Hz at **0.0667±0.0031 RTF**
  (batch 16) and Transformer AR + BiGRU at **0.0246±0.0007** (batch 32), while
  their batch-1 RTFs are **0.1132±0.0158** and **0.1889±0.0042**. Batch-1 versus
  selected-batch hypotheses differ on roughly 17--23% of utterances, although
  corpus WER shifts are much smaller; keep these partial results out of the HTML
  until the memory protocol and numerical batch sensitivity are resolved.
- Batch-32 fallback jobs 118878–118892 cover fixed k=5, fixed k=6, character
  oracle, CNN AR mean, and CNN AR + BiGRU, with all three seeds repeated at the
  same operating point. They are pending for priority. CNN AR + BiGRU now matches
  the completed Transformer AR batch size. Fresh artifacts live under
  `inference_eval_memory_limited_batch_v1/retry_batch32_v1/`.
- All 15 batch-32 fallback jobs completed successfully with exit code zero and no
  OOM/traceback. Three-seed RTFs are fixed k=5 **0.0182±0.0003**, fixed k=6
  **0.0184±0.0006**, character oracle **0.0299±0.0007**, CNN AR mean
  **0.0239±0.0011**, and CNN AR + BiGRU **0.0236±0.0009**. At the same batch 32,
  the existing Transformer AR mean and +BiGRU values are **0.0267±0.0012** and
  **0.0246±0.0007**. CNN+BiGRU and Transformer+BiGRU therefore have nearly equal
  throughput, with CNN about 4% lower RTF in this pipeline.
- Peak allocated memory at batch 32 is 25.3 GB for fixed k=5, 26.5 GB for CNN
  mean, 26.7 GB for CNN+BiGRU, 27.5 GB for Transformer mean, and 26.1 GB for
  Transformer+BiGRU. The conservative retry removed the OOM problem. Numerical
  batch sensitivity remains: batch-1 versus batch-32 hypotheses differ on
  17--21% of utterances depending on system, so these timings are valid operating
  points but their batched WER should not replace the official WER table.
- At each system's stable throughput operating point, no downsampling is
  **0.0667±0.0031 RTF** at batch 16, versus phone oracle **0.0229±0.0003**, CNN
  AR + BiGRU **0.0236±0.0009**, and Transformer AR + BiGRU **0.0246±0.0007** at
  batch 32. These correspond to 2.9×, 2.8×, and 2.7× lower RTF than native 50 Hz,
  but the unequal batches make them system-level throughput comparisons.
- A fair common-batch-16 RTF study was submitted as jobs 119834–119851 for fixed
  k=5, phone oracle, and all four CNN/Transformer × mean/BiGRU systems, with three
  seeds each. The existing no-downsampling batch-16 artifacts are the reference.
  Four jobs started immediately; the rest were accepted pending priority.
- The primary RTF claim should retain each system's stable throughput batch because
  fitting a larger batch is one effect of downsampling. For no downsampling batch
  16, test-split peak allocated memory is **36.05±1.79 GB**, but the long prewarm
  reaches **41.85±0.32 GB allocated / 78.66±0.04 GB reserved**; the batch-32
  selector candidate OOMed. The 14 not-yet-started common-batch jobs were placed on
  user hold; four already-running short jobs were left intact.

#### 2026-08-27 - Training report refreshed with final attribution and efficiency evidence

- The matched 100h attribution study is now complete across all three seeds. Fixed
  k=5 + BiGRU reaches **5.17±0.04 / 9.70±0.12 WER** at about 10.1 Hz; the frozen
  Transformer-AR segmenter + BiGRU reaches **5.09±0.34 / 9.52±0.27** at about
  14.4/13.9 Hz; joint RL reaches **4.95±0.02 / 9.11±0.18** at about 11.2/10.8 Hz.
  Thus joint RL improves on frozen boundaries by 0.14/0.41 WER while emitting about
  three fewer audio tokens/s, and improves on fixed k=5 by 0.22/0.59 WER at a
  modestly higher frequency.
- The HTML report now uses the completed memory-limited efficiency results rather
  than the earlier preselected-batch RTF table. It reports each system's largest
  reliable full-pass batch, forward RTF, utterances/s, measured frequency, and peak
  test allocation for all three seeds. Headline values are no downsampling at batch
  16 (**0.0667±0.0031 RTF, 2.16±0.10 utterances/s, 36.0±1.8 GB**) versus CNN AR +
  BiGRU at batch 32 (**0.0236±0.0009, 6.09±0.24, 26.7±0.7 GB**) and Transformer AR
  + BiGRU at batch 32 (**0.0246±0.0007, 5.85±0.16, 26.1±0.6 GB**).
- The report documents the measurement protocol: one A100 80 GB in BF16; seed 3407
  calibrates power-of-two candidate batches on the longest dev-other utterances;
  the selected batch must survive a complete evaluation, otherwise all seeds use
  the batch-32 fallback; ten longest-utterance dev-other batches warm CUDA and the
  allocator in the same process but are excluded from timing; test-clean and
  test-other are duration-sorted and evaluated completely with fixed utterance-count
  batches and no random sampling. HTML report changes are now publish-by-default;
  keep a report local only when the user explicitly requests a draft.
- This report revision was published from source commit `1819f4348` on
  `asset-dev/bilevel-optimization` and Pages commit `0e64845` on `origin/main`.
  Live verification confirmed HTTP 200, the new iframe cache key, and 3/3 seeds for
  both frozen and joint Transformer-AR rows in Table 3.

#### 2026-08-27 - Publication progress review after completing attribution and RTF

- The obsolete joint-RL progress card was removed from the training report because
  all three matched joint runs have final test evidence. Table 3 and Figure 2 remain
  as the concise completed comparison.
- Existing per-utterance inference records already support the “adaptive” claim:
  after regressing segment count and reference word count on duration, their mean
  Spearman correlation across seeds is **0.69 for CNN AR + BiGRU** and **0.76 for
  Transformer AR + BiGRU**. The fastest speaking-rate quartile receives about three
  more audio tokens/s than the slowest quartile. Plot this before launching a new
  adaptivity-training experiment.
- Remaining publication-critical experiments, in order, are: (1) calibrate the
  existing joint checkpoints to the fixed-control frequency and run paired bootstrap
  statistics; (2) add a transcript-free CTC posterior-collapse baseline; (3) run one
  three-seed LS960 fixed-k=5 + BiGRU control with the same best-character decoder and
  24k-step budget; (4) complete a KV-cached latency/memory benchmark; then (5) add one
  noise/reverberation robustness evaluation. More segmenter architectures, bilevel
  optimization, a full LS960 sweep, and a new downstream task remain off the critical
  path. Full review: `reports/publication_progress_review_2026-08-27.md`.
- The report cleanup/review is source commit `528346aba`; the Pages publication is
  `d3802ae`. Live verification confirmed HTTP 200, cache key
  `20260827-complete-attribution`, and absence of the joint-training progress card.

#### 2026-08-27 - Character-initialized fixed-k=5 mean-pooling control submitted

- Jobs **120870–120872** run seeds 3407/3408/3409 for the matched 100h fixed-rate
  control: WavLM features, fixed `k=5` boundaries, parameter-free mean pooling,
  the best character-decoder initialization, and seven decoder-CE epochs. This is
  a `train-clean-100` attribution experiment, not an LS960 scale-up.
- Outputs are isolated under
  `results/speechllm_segmenter_wavlm/attribution_100h_transformer_ar_bigru/fixed_k5_mean/`.
  Each job requests one A100, 12 CPUs, 60 GB RAM, two days, the JSALT reservation,
  and excludes `ga129`. All three reached `RUNNING` on `ga133`/`ga138` and began
  loading the frozen WavLM/Llama models successfully.
- The joint trainer now permits an empty segmenter optimizer group only for a
  frozen-policy joint run. This supports fixed boundaries plus parameter-free mean
  pooling while preserving the existing decoder parameter-group/LR behavior. The
  focused schedule/control test suite passes 16/16 tests.

#### 2026-08-28 - Transcript-free 100h CTC posterior-collapse baseline submitted

- Added an audio-only CTC compressor that runs the cached torchaudio
  `WAV2VEC2_ASR_BASE_960H` model, takes the framewise argmax path, collapses
  repetitions, and opens one segment per nonblank run. It enumerates FLAC files
  directly and never opens a transcript or forced-alignment target. Blank frames
  are merged into neighboring emitted-symbol segments rather than emitted as
  separate decoder tokens.
- The CTC acoustic model requires no new training. The matched downstream baseline
  still trains the best-character-initialized residual BiGRU, projection, and Llama
  LoRA for seven epochs on `train-clean-100`, using seeds 3407/3408/3409. The joint
  trainer now accepts validated external alignment-format boundaries while the
  boundary policy is frozen; the focused CTC/schedule tests pass 22/22.
- Array job **128765** generates boundaries for train-clean-100 plus the dev/test
  splits with eight one-A100 shards. Jobs **128766–128768** depend on successful
  completion of the entire array and train the three downstream seeds. Every job
  uses account `jsalt2026-lgarci27`, partition `a100`, comment `accept_cost`, the
  `JSALT 2026` reservation, and excludes `ga129`.
- Boundary tensors are written under
  `/weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/boundary_targets/ctc_posterior_argmax_nonblank/`;
  training outputs are isolated under
  `results/speechllm_segmenter_wavlm/attribution_100h_transformer_ar_bigru/ctc_posterior_argmax_nonblank_bigru/`.
  The first three running preprocessing shards report approximately 14.6–15.0 Hz.
  This is close to the existing character-scale operating point but above the
  learned joint system's roughly 11 Hz; retain it as the standard CTC row and defer
  blank-rule rate tuning unless it becomes publication-critical.
- Training/evaluation consumes precomputed boundaries. A later deployable RTF row
  must run and time the CTC encoder live rather than treating boundary lookup as the
  compressor cost.
- Per ADR-020, the next experimental work is MuST-C En–De preparation and direct
  one-seed training of the LS960-initialized task-conditioned Transformer-AR +
  BiGRU system. Broad ST baselines are deferred until that flagship pilot passes.

#### 2026-08-28 - Phone-boundary agreement interpreted at matched frequency

- On all 2,703 dev-clean utterances, the three WER-selected 100h WavLM local-64
  Transformer-AR + BiGRU checkpoints emit **10.58±0.65 audio tokens/s**, close to
  the phone reference's **10.74 boundaries/s**. Their harsh ordered one-to-one
  phone-boundary F1 is **22.7±2.6 exact**, **63.1±1.2 within ±20 ms**, and
  **81.1±0.9 within ±40 ms**; ± is sample SD over three seeds.
- The ±20 ms value is not raw frame accuracy. It corresponds to **63.7±2.8%
  precision / 62.7±1.8% recall**. A content-blind fixed `k=5` grid at **10.06
  Hz** already reaches **58.9% precision / 55.1% recall / 56.9% F1**, partly
  because a ±1-frame window around one cut every five frames covers roughly 60%
  of frame positions before one-to-one constraints.
- The rate-controlled evidence is the learned system's **+6.2 F1-point** gain
  over fixed `k=5` within ±20 ms (+6.3 within ±40 ms), not the raw 63% score.
  Fixed `k=4` reaches a similar **62.9%** F1 but needs **12.55 Hz**, about 19%
  more decoder-side audio tokens. The exact-frame gain over fixed `k=5` is only
  0.8 point, so learned placement is mainly closer within one or two frames; this
  supports adaptive placement efficiency but not the claim that the RL policy is
  a pure phone segmenter.
- The training report now includes the exact fixed-grid table, a rate-versus-F1
  plot, metric interpretation, and the matched-rate conclusion. Evidence remains
  in `artifacts/segmenter/{fixed_rate_phone_agreement_dev_clean.json,
  transformer_ar_bigru_phone_agreement_dev_clean.json}`.

#### 2026-08-28 - Rate-closer BPE-CTC baseline submitted

- The completed character-CTC posterior-collapse baseline reaches
  **4.65±0.09 / 9.19±0.20 WER** on test-clean/test-other, but emits about
  **14.6/14.3 audio tokens/s**. Its tokenizer makes it a character-scale CTC
  comparison, not a close frequency match to the joint Transformer-AR + BiGRU
  system at roughly 11.2/10.8 Hz.
- A released WavLM checkpoint with a suitably small BPE vocabulary was not found.
  The available ESPnet WavLM-Large model uses a 5,000-unit unigram vocabulary and
  is too coarse for this purpose. The selected deployable checkpoint is Icefall's
  LibriSpeech Conformer-CTC with a 500-piece SentencePiece vocabulary. It is frozen
  and used for audio-only inference; no transcript or forced alignment is read.
- A pre-WER pilot on 32 dev-clean utterances (235.3 s) measured **4.98 Hz** for
  ordinary nonblank BPE collapse and **9.06 Hz** when every CTC argmax state run is
  retained, including blank runs. Four disjoint pilot shards ranged 8.88–9.24 Hz
  for the blank-aware rule. The state-run rule was locked before downstream WER
  was observed because it is materially closer to the learned system and preserves
  the gap acoustics already shown useful by the blank-content ablation.
- Array job **135788** generates state-run boundaries for `train-clean-100` plus
  dev/test splits with eight one-A100 shards. Three five-epoch downstream
  residual-BiGRU/projection/LoRA jobs, **135789–135791**, depend on the full array
  and use seeds 3407/3408/3409. The five-epoch budget is a deadline-driven rapid
  comparison; extend the selected condition to seven epochs only if exact budget
  matching becomes necessary.
- All eight generation shards completed with exit code zero. The validated output
  contains exactly **39,665** tensors and eight statistics files. The full-corpus
  state-run rate is **9.678 Hz** (nonblank-only diagnostic: **5.348 Hz**);
  train-clean-100 is 9.672 Hz, dev-clean/dev-other are 9.996/9.601 Hz, and
  test-clean/test-other are 9.775/9.443 Hz. The aggregate is saved at
  `artifacts/segmenter/ctc_bpe500_state_run_rates.json`.
- Jobs 135789 and 135790 entered epoch 1 successfully on `ga132`, loading the
  intended character-decoder initialization, freezing the boundary policy, and
  training the residual BiGRU plus decoder parameters. Job 135791 is accepted and
  waiting only on `MaxGRESPerAccount`.
- All three five-epoch runs completed with exit code zero in 2h25m–2h30m. The
  three-seed result is **5.05±0.27 / 9.93±0.25 WER** on test-clean/test-other and
  **9.15±0.31** on dev-other, at **9.77/9.44 Hz** on the two test splits. Seeds
  3407/3408/3409 selected epochs 5/5/3 respectively. Exact edit counts and
  per-seed results are stored in
  `artifacts/segmenter/ctc_bpe500_state_run_5ep_results.json`.
- At similar frequency, the BPE-CTC row is 0.12 WER better than fixed k=5 + BiGRU
  on test-clean (5.05 versus 5.17) but 0.23 worse on test-other (9.93 versus 9.70).
  Against the established corrected Transformer-AR + BiGRU headline result
  (**4.80±0.09 / 9.24±0.36** at 10.38/10.16 Hz), BPE-CTC trails by 0.25/0.69 WER
  while using 0.61/0.71 fewer audio tokens/s. The higher-rate character-CTC row
  remains stronger at 4.65/9.19, confirming that tokenizer granularity and
  frequency materially affect the original comparison.
- Boundaries are written to
  `/weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/boundary_targets/ctc_bpe500_state_runs/`;
  training outputs are isolated under
  `results/speechllm_segmenter_wavlm/attribution_100h_transformer_ar_bigru/ctc_bpe500_state_runs_bigru_5ep/`.
  Precomputed-boundary evaluation still excludes live CTC cost, which must be
  included in any deployable RTF comparison.

#### 2026-08-28 - Self-trained WavLM phone-CTC rapid baseline implemented

- The clean replacement baseline freezes `microsoft/wavlm-large` and trains a
  small phone-CTC head on `train-clean-100` for three epochs. Supervision uses only
  ordered, stress-stripped MFA phone labels; interval times are discarded. All
  three epoch checkpoints are retained and ranked by dev-clean PER.
- Boundary inference opens segments at ordinary nonblank CTC runs and reads audio
  only. The best checkpoint must pass PER ≤40% and a 7–14 Hz rate gate before full
  boundary generation and downstream training proceed.
- The dependent downstream run trains residual BiGRU pooling plus the decoder for
  four epochs, retains the three best validation-WER checkpoints, and evaluates
  all three. This is one rapid 100h run (seed 3407), not a three-seed estimate.
- The training report's Joint Transformer AR + BiGRU row and summary now both use
  the corrected headline **4.80±0.09 / 9.24±0.36** family; epoch-count commentary
  was removed from that 100h comparison.
- The submitted dependency chain is: phone-CTC training **170350**, three-checkpoint
  dev-clean inference audit **170351**, best-checkpoint boundary generation
  **170352**, four-epoch BiGRU+decoder training **170353**, and three-checkpoint
  downstream inference **170354**. At the last submission check, 170350 was
  running on A100 node `ga138`; the remaining jobs were pending on their intended
  `afterok` dependencies.
- CTC training/audit/generation jobs 170350–170352 completed successfully. Original
  downstream job 170353 failed on a duplicate HyperPyYAML override, making 170354
  unsatisfiable. The corrected downstream/evaluation pair is **171034/171035**;
  existing CTC checkpoints and the complete 39,665-tensor boundary set are reused.
- Corrected job 171034 exposed a separate stale-`test_csv` configuration error and
  was replaced, together with its unsatisfiable evaluation array, by **171203/171204**.
  Both training and evaluation override mappings were validated for duplicate keys
  and paired split/CSV settings before submission. Job 171203 entered epoch 1 on
  `ga138`; job 171204 is waiting on its normal `afterok` dependency.
- The downstream study was expanded to three optimization seeds without retraining
  CTC: seed 3407 is job **171203** (evaluation **171204**), seed 3408 is **171226**
  (evaluation **171227**), and seed 3409 is **171228** (evaluation **171229**).
  All three training jobs entered epoch 1 successfully; every evaluation array runs
  checkpoint ranks 0–2 sequentially.
- All three downstream training jobs and all nine retained-checkpoint evaluations
  completed with exit code zero. Selecting rank 0 independently by dev-clean WER
  gives seed 3407/3408/3409 test-clean WER **4.95/5.26/5.09** and test-other WER
  **9.56/9.84/10.01**, for **5.10±0.16 / 9.81±0.23** (sample SD). Dev-other is
  **9.32±0.33**. The shared audio-only phone-CTC boundary rates are 9.76 Hz on
  test-clean and 9.39 Hz on test-other.
- The other retained ranks were also evaluated rather than selected on test data:
  rank 1 aggregates to 5.12±0.17 / 9.74±0.25 and rank 2 to
  5.04±0.16 / 9.84±0.28 test-clean/test-other WER. Rank 0 remains the reportable
  result because checkpoint ranking was fixed by dev-clean WER.
- The user subsequently selected the last checkpoint consistently across seeds.
  Epoch 4 is rank 1 for seed 3407 and rank 0 for seeds 3408/3409. The updated
  three-seed result is **5.09±0.17 / 9.80±0.24** test-clean/test-other WER and
  **9.42±0.27** dev-other WER. This final-epoch selection supersedes rank 0 as the
  headline result.
- A rate-raised comparison now preserves all phone-CTC boundaries and splits each
  predicted segment of at least 8 frames once at its midpoint. The threshold was
  locked from dev-clean frequency, without downstream WER or transcripts. Rates are
  10.77/11.06/10.70/10.90/10.49 Hz on train-clean-100/dev-clean/dev-other/
  test-clean/test-other. All three-epoch downstream jobs 202998/203000/203002 and
  all nine checkpoint evaluations in arrays 202999/203001/203003 completed with
  exit code zero.
- For the rate-raised comparison, the epoch 1/2/3 aggregate WERs are respectively
  **5.40±0.12 / 10.02±0.22**, **5.07±0.15 / 9.38±0.11**, and
  **5.02±0.12 / 9.34±0.26** on test-clean/test-other (three-seed sample SD).
  Dev-other improves in the same order: **9.38±0.08**, **8.90±0.15**, and
  **8.80±0.13**. Thus the requested last checkpoint, epoch 3, is also the
  strongest aggregate checkpoint among the three epochs.
- Section 1 of the training report now includes the epoch-3 rate-raised phone-CTC
  system in the overview figure, system inventory, baseline table, and interpretation.
  The row states the split-specific 10.90/10.49 Hz rate and the remaining
  0.22/0.10 clean/other WER gap to Transformer AR + BiGRU. Pages commit
  **ac8eb45** deployed successfully with cache key
  `20260829-phone-ctc-longsplit8`; live wrapper and standalone checks returned
  HTTP 200 and contained the new 5.02±0.12 / 9.34±0.26 row. The corresponding
  source commit is **c51bb4e34** locally; pushing it to the personal `asset-dev`
  GitHub remote requires explicit user authorization in the current environment.
- A one-pass LS960 follow-up keeps the 100h-trained phone-CTC head frozen, applies
  the same eight-frame long-segment split during audio-only inference over all
  seven LibriSpeech splits, mean-pools the resulting segments, and trains only the
  decoder-side projection + LoRA adapters with CE. Boundary array **204330** feeds
  CPU completeness/rate gate **204331**, then seeds 3407/3408/3409 train as jobs
  **204332/204333/204334**. This is a true LS960 run over train-clean-100,
  train-clean-360, and train-other-500; unlike the 100h row, it deliberately uses
  parameter-free mean pooling rather than training the BiGRU.
- The LS960 follow-up completed successfully. All 16 tasks in boundary array 204330,
  gate 204331, and training jobs 204332/204333/204334 exited 0:0. The gate verified
  exactly 292,367 tensors. Final three-seed WER is **4.05±0.08 test-clean**,
  **7.61±0.06 test-other**, and **7.04±0.26 dev-other** (sample SD). Per-seed
  clean/other WER is 4.10/7.60, 4.09/7.56, and 3.95/7.67 for seeds
  3407/3408/3409. Boundary rates are 10.76/10.80/10.51 Hz on the three LS960
  training splits and 11.06/10.70/10.90/10.49 Hz on dev-clean/dev-other/
  test-clean/test-other.
- A checkpoint-resumed continuation of the same three LS960 decoder-CE seeds is
  running as jobs **205552/205553/205554**. Each job loaded its seed's retained
  checkpoint and entered the requested additional training epoch on A100 node
  `ga138`; outputs remain in the original seed directories so SpeechBrain's
  best-checkpoint policy can preserve the stronger validation state.
- The training report treats the completed LS960 phone-CTC result as a reference
  baseline: it remains in the exact-value table and three-seed comparison figure but
  is absent from the headline tiles and has no highlighted table styling. The main
  comparison states that Transformer AR + BiGRU reaches **2.79±0.02 / 5.63±0.12**
  versus **4.05±0.08 / 7.61±0.06** for phone-CTC at essentially identical rates,
  an exact aggregate advantage of **1.25 / 1.99 WER points**. The report contains
  no continuation status and states no decoder-training epoch count.
- Source commits are `bc2c7293d` (initial LS960 report), `d5b7951b2`
  (continuation launcher), and `3f7c629e0` (baseline reframing). The configured
  personal `asset-dev` push remains blocked pending explicit user authorization,
  so these source commits are local.
- The report is publicly deployed from Pages commit **`a2c4261`** with cache key
  `20260830-ls960-phone-ctc`. GitHub Actions run `33324617750` completed
  successfully, and live wrapper/standalone checks returned the new cache key and
  the **4.05±0.08 / 7.61±0.06** LS960 phone-CTC row.
- The baseline-reframed revision is deployed from Pages commit **`e426322`** with
  cache key `20260830-ctc-baseline-reframe`; Actions run `33325542808` succeeded.
  Live verification confirms that the CTC headline tile is absent, the neutral
  baseline row remains, and the 1.25/1.99-point learned-system advantage is present.

#### 2026-08-28 - Batch-one and throughput RTF views separated

- The inference-efficiency analysis now appears immediately after the matched
  attribution section (section 0c), before the detailed experiment inventory.
  Its primary figure places batch-one processing on the left and each system's
  largest stable tested throughput batch on the right; these answer different
  deployment questions and must not be interpreted as the same speed metric.
- At batch one, no downsampling is **0.113±0.016 RTF**, CNN first-order AR +
  BiGRU is **0.100±0.001**, and local-64 Transformer AR + BiGRU is
  **0.189±0.004**. The Transformer's sequential policy overhead therefore makes
  single-utterance processing slower despite its compression.
- At the selected stable throughput batches, no downsampling is **0.067±0.003
  RTF at batch 16**, fixed `k=5` is **0.018±0.000 at batch 32**, CNN AR + BiGRU
  is **0.024±0.001 at batch 32**, and Transformer AR + BiGRU is **0.025±0.001 at
  batch 32**. The Transformer's efficiency case is batched throughput and a
  larger feasible batch, not a batch-one latency reduction.
- The phone-boundary agreement audit is now its own section, **0c.1**, immediately
  after the matched 100h attribution and before the inference-efficiency section
  0d. This keeps the boundary-placement evidence beside the headline controlled
  comparison instead of burying it in the later component-diagnostic inventory.

---

#### Group: Task-conditioned multi-task ASR + En-De speech translation

*Goal: test whether one shared hard segmenter, conditioned on the requested
task, can serve English ASR and En-De speech translation at a controlled
audio-token frequency without sacrificing LibriSpeech ASR.*

| ID | Date | Model / Config | Dataset / Split | Metric | Rate | Notes | Output |
|---|---|---|---|---|---|---|---|
| MT-1 | 2026-08-29 | LS960 Transformer-AR local-64 + BiGRU warm start (seed 3408), task FiLM + task-routed LoRA, 15k steps (bridge 3k / FiLM 6k / unfreeze 6k) | CoVoST 2 En-De test (15527/15529) | sacreBLEU **18.88**, chrF++ 42.25 | 7.39 Hz | Job 171461 train + 172481 eval; checkpoint step 13500 | `results/speechllm_multitask_ls960_covost_en_de/3408` |
| MT-1 | 2026-08-29 | same | CoVoST 2 En ASR test | WER **15.22**, CER 7.95 | 7.96 Hz | Same audio as the ST row | same |
| MT-1 | 2026-08-29 | same | LibriSpeech dev-clean | WER 3.14 | 10.86 Hz | vs 2.879 warm start (+0.26) | same |
| MT-1 | 2026-08-29 | same | LibriSpeech dev-other | WER **5.56** | 10.74 Hz | vs 5.63+-0.12 cohort: no degradation | same |
| MT-1 | 2026-08-29 | same | LibriSpeech test-clean | WER 3.26 | 10.76 Hz | | same |
| MT-1 | 2026-08-29 | same | LibriSpeech test-other | WER 5.68 | 10.45 Hz | | same |
| NA-1 | 2026-09-02 | LS960 Transformer-AR local-64 + BiGRU warm start (seed 3408), 5-task table, task FiLM + task-routed LoRA, 15k steps (bridge 3k / FiLM 6k / unfreeze 6k), mixture 0.40 LS960 ASR / 0.20 emotion / 0.20 speaker_count / 0.20 intent, `rate_channel: auto` (reward only), per-task bands active in Phase C | FSC intent test (3793) | acc **0.984**, macro-F1 0.988 | 7.55 Hz | Job 1780836, 8h41m on 1 A100. **Reported checkpoint is step 1500 (pre-policy)** -- see analysis | `results/speechllm_multitask_ls960_nonasr/3408` |
| NA-1 | 2026-09-02 | same | CREMA-D emotion test (1431, voice-vote labels) | acc 0.609, macro-F1 **0.307** | 8.56 Hz | test majority baseline 0.617, so accuracy alone is uninformative; macro-F1 is the number | same |
| NA-1 | 2026-09-02 | same | Synthetic speaker-count test (2000, counts 0-4) | acc 0.549, macro-F1 0.514 | 5.25 Hz | chance 0.20. Validation swung 0.485-0.688 across 9 points -- not a usable measurement | same |
| NA-1 | 2026-09-02 | same | LibriSpeech dev-clean | WER 3.143, CER 1.317 | 10.56 Hz | vs 2.879 warm start (+0.264); MT-1 got 3.14 | same |
| NA-1 | 2026-09-02 | same | LibriSpeech dev-other | WER **5.551**, CER 2.515 | 10.46 Hz | vs 5.63+-0.12 cohort: no degradation, gate criterion 1 passes; MT-1 got 5.56 | same |
| NA-1 | 2026-09-02 | same | LibriSpeech test-clean | WER 2.866, CER 1.053 | 10.45 Hz | MT-1 got 3.26 | same |
| NA-1 | 2026-09-02 | same | LibriSpeech test-other | WER 5.760, CER 2.608 | 10.18 Hz | MT-1 got 5.68 | same |
| NA-2 | 2026-09-04 | LS960 Transformer-AR local-64 + BiGRU warm start, 5-task table, task FiLM + task-routed LoRA, 15k steps (bridge 3k / FiLM 6k / unfreeze 6k), mixture 0.40 LS960 ASR / 0.20 emotion / 0.20 speaker_count / 0.20 intent, **`rate_channel: aux`** (differentiable expected-rate term, not the reward), **LibriCount-matched** speaker-count data (`nonasr_manifests_lcfull`), 3 seeds 3407-3409 | FSC intent test (3793) | acc **0.994**, macro-F1 0.994 (seed 3408; seeds 0.983-0.994) | 7.55 Hz | Jobs 1784263/64/65, ~8h30m each on 1 A100. **Reported checkpoints are pre-policy**: 3408 step 3000, 3407/3409 step 1500 | `results/speechllm_multitask_ls960_nonasr_lc_aux/{3407,3408,3409}` |
| NA-2 | 2026-09-04 | same | Synthetic speaker-count test (2000, counts 0-4, LibriCount-matched) | acc **0.730**, macro-F1 0.720 (seed 3408); seeds 0.535 / 0.730 / 0.623 | 10.57 Hz | Reference ceiling on this same split is **0.871** (validated CountNet CRNN port). Chance 0.20 | same |
| NA-2 | 2026-09-04 | same | CREMA-D emotion test (1431, voice-vote labels) | acc 0.614, macro-F1 **0.350** (seed 3408); seeds 0.285 / 0.350 / 0.285 | 8.56 Hz | Test majority baseline 0.617; macro-F1 is the number. Gate is 0.55 | same |
| NA-2 | 2026-09-04 | same | LibriSpeech dev-clean | WER 3.101, CER 1.221 (seed 3408; seeds 2.919-3.101) | 10.56 Hz | vs 2.879 warm start (+0.222) | same |
| NA-2 | 2026-09-04 | same | LibriSpeech dev-other | WER **5.525**, CER 2.522 (seed 3408; seeds 5.517-5.557) | 10.46 Hz | vs 5.63+-0.12 cohort: no degradation, gate criterion 1 passes | same |
| NA-2 | 2026-09-04 | same | LibriSpeech test-clean | WER 2.982, CER 1.089 (seed 3408; seeds 2.933-3.083) | 10.45 Hz | | same |
| NA-2 | 2026-09-04 | same | LibriSpeech test-other | WER 6.058, CER 2.755 (seed 3408; seeds 5.814-6.114) | 10.18 Hz | | same |

| SB-1 | 2026-09-04 | Single-task emotion baselines: LS960 Transformer-AR + BiGRU warm start, frozen boundary policy, decoder LoRA + shared pooler only, 1200 steps (~30 data epochs), matched 300 s effective batch, selection on `macro_f1_emotion` | CREMA-D test (1431, voice-vote) | macro-F1 **0.404 +- 0.060** | 50.00 Hz | no downsampling (k=1); jobs 1784461-63 | `results/speechllm_singletask_nonasr/emotion/nods_50hz` |
| SB-1 | 2026-09-04 | same, fixed grid k=2 | CREMA-D test | macro-F1 0.422 +- 0.023 | 25.11 Hz | | `.../emotion/fixed_25hz` |
| SB-1 | 2026-09-04 | same, fixed grid k=5 | CREMA-D test | macro-F1 **0.434 +- 0.033** | 10.19 Hz | best of the fixed arms, but within seed noise of all of them | `.../emotion/fixed_10hz` |
| SB-1 | 2026-09-04 | same, fixed grid k=10 | CREMA-D test | macro-F1 0.419 +- 0.033 | 5.20 Hz | | `.../emotion/fixed_5hz` |
| SB-1 | 2026-09-04 | same, learned Transformer-AR policy held frozen at argmax | CREMA-D test | macro-F1 0.431 +- **0.004** | 8.56 Hz | matches the multi-task bridge rate exactly (same frozen policy) | `.../emotion/learned_frozen` |
| SB-1 | 2026-09-05 | same, learned policy TRAINED with K=4 GRPO (bridge 20/FiLM 40/unfreeze 40%) | CREMA-D test | macro-F1 0.438 +- 0.044 | **4.15 Hz** | GRPO halves the rate at unchanged quality; seed spread rises 0.007 -> 0.088 | `.../emotion/learned_grpo` |
| SB-2 | 2026-09-05 | Single-task speaker-count baselines, 2600 steps (~7.8 data epochs), selection on `acc_speaker_count`, seeded (non-aliasing) valid subset | Synthetic speaker-count test (2000, 5-way) | acc 0.724 +- 0.069 | 50.00 Hz | no downsampling | `.../speaker_count/nods_50hz` |
| SB-2 | 2026-09-05 | same, fixed grid k=2 | same | acc **0.773 +- 0.059** | 25.10 Hz | best arm, but within ~1 SD of every other | `.../speaker_count/fixed_25hz` |
| SB-2 | 2026-09-05 | same, fixed grid k=5 | same | acc 0.735 +- 0.021 | 10.04 Hz | | `.../speaker_count/fixed_10hz` |
| SB-2 | 2026-09-05 | same, fixed grid k=10 | same | acc 0.736 +- 0.034 | 5.02 Hz | 10x compression vs no-downsampling costs nothing | `.../speaker_count/fixed_5hz` |
| SB-2 | 2026-09-05 | same, learned policy frozen | same | acc 0.729 +- 0.011 | 10.57 Hz | | `.../speaker_count/learned_frozen` |
| SB-2 | 2026-09-05 | same, learned policy + K=4 GRPO | same | acc 0.724 +- 0.012 | 9.85 Hz | GRPO compresses only 7% here vs 51% on emotion | `.../speaker_count/learned_grpo` |
| SB-3 | 2026-09-05 | Single-task intent baselines, 900 steps (~5.1 data epochs), selection on `acc_intent` | FSC intent test (3793) | acc 0.994 +- 0.002 | 50.00 Hz | | `.../intent/nods_50hz` |
| SB-3 | 2026-09-05 | same, fixed grids k=2 / k=5 / k=10 | same | acc 0.995 / 0.995 / 0.995 | 25.10 / 10.18 / 5.19 Hz | saturated at every rate | `.../intent/fixed_*` |
| SB-3 | 2026-09-05 | same, learned frozen / learned GRPO | same | acc 0.994 +- 0.001 / 0.993 +- 0.003 | 7.55 / 6.85 Hz | | `.../intent/learned_*` |

Analysis notes:

- **SB-1: for emotion the rate is not the bottleneck, and neither is
  placement.** Across a 10x range (50 -> 5 Hz) macro-F1 spans 0.404-0.434, well
  inside the 0.02-0.06 per-arm seed SDs; there is no detectable rate effect at
  all. The learned policy at 8.56 Hz (0.431) is indistinguishable from a uniform
  grid at 10.19 Hz (0.434), so learned boundary *placement* buys no accuracy
  either. Compression is free here rather than a trade-off.
- **SB-1: what the learned policy does buy is stability.** Its seed spread is
  0.007 against 0.043-0.105 for the fixed arms. Read cautiously -- a spread from
  3 seeds is itself noisy -- but a ~10x difference is consistent with the frozen
  policy placing boundaries identically regardless of decoder init.
- **SB-1: multi-task interference is real and large for emotion.** Every
  single-task arm beats the multi-task cohort's 0.307 +- 0.038 by 0.10-0.13
  macro-F1. Compare against the cohort's three seeds (0.285/0.350/0.285), not
  the single published 0.350 checkpoint, which would flatter the multi-task side.
- **SB-1: and yet emotion still misses its 0.55 gate at 50 Hz with no other task
  competing.** Rate, placement and mixture are each excluded as the explanation,
  which leaves the pooled-prefix interface or CREMA-D's crowd-vote labels. That
  is a more useful negative than the multi-task runs could produce.
- **SB-1/2/3 headline: the audio-token rate is not what costs us anything.** On
  all three tasks a 10x compression (50 -> 5 Hz) is free: emotion 0.404 -> 0.419
  macro-F1, speaker count 0.724 -> 0.736 accuracy, intent 0.994 -> 0.995, every
  span inside the per-arm seed SD. On both non-saturated tasks the 50 Hz arm is
  not even the best one. Whatever limits these tasks, compression is not it --
  which also means the segmenter's compression is not being bought with quality.
- **Learned placement buys no accuracy over a uniform grid, on any task.** At
  comparable rates: emotion 0.431 (8.56 Hz learned) vs 0.434 (10.19 Hz grid);
  speaker count 0.729 (10.57 Hz learned) vs 0.735 (10.04 Hz grid); intent 0.994
  vs 0.995. What the frozen learned policy does buy is *stability* -- seed
  spread 0.007-0.022 against 0.042-0.136 for the fixed arms.
- **GRPO does what it was designed to do, but unevenly.** It is the first
  measurement of a trained policy in this project (every multi-task checkpoint
  ever reported was pre-policy). On emotion it halves the rate at unchanged
  quality (8.56 -> 4.15 Hz, 0.431 -> 0.438). On intent it trims 7.55 -> 6.85 Hz.
  On speaker count it barely moves, 10.57 -> 9.85 Hz -- even though the
  fixed-rate arms prove that task runs fine at 5.02 Hz. So the rate pressure is
  too weak, or the quality reward too noisy, exactly where compression is most
  affordable. That is the concrete next thing to fix, and the fixed-rate sweep
  now supplies the per-task target the plan asked for.
- **Multi-task interference is real where there is headroom.** Single-task beats
  the multi-task cohort by ~0.12 macro-F1 on emotion (0.43 vs 0.307) and ~0.13
  accuracy on speaker count (0.73 vs 0.595), and by nothing on intent (0.995 vs
  0.992), which was already saturated.
- **The residual gap is the interface.** Speaker count reaches 0.773 at best
  against CountNet's 0.871 on the same test split, and only 0.724 at 50 Hz with
  no competing task and no compression. Rate, placement and mixture are each
  excluded by measurement, which leaves the pooled-prefix interface itself (or
  the decoder's use of it) as the remaining explanation.
- **Speaker-count baselines were re-run.** The first 15 (archived under
  `speaker_count_aliased_valid600`) selected checkpoints against a validation
  subset that contained only 3 of 5 classes -- see `bugs.md`, subset aliasing.
  Their TEST numbers were measured correctly on the full 5-class set but on a
  badly-selected checkpoint, so they understate every arm; do not quote them.
  An external reference (CountNet) scores 0.860 valid / 0.871 test, confirming
  the two splits are equally hard.


- **NA-2 settles the speaker-count data question against my earlier reading: the
  instability is architectural, not a property of the mixtures.** The generator
  was rebuilt to match LibriCount (equal-power sources, no mixture-level
  normalization, VAD labels, 93.7% voice duty vs LibriCount's 93.5%), and a
  validated external reference scores **0.871** on the new test split vs 0.927 on
  the real corpus -- i.e. the data admits far more than we extract. Yet
  within-run validation spread stayed just as wide (0.274 / 0.232 / 0.101 across
  seeds 3407/3408/3409) and test accuracy spans 0.535-0.730 across seeds. Data
  matching removed the excuse without removing the variance. The pooled-prefix
  interface is the remaining suspect and the fixed-rate sweep is the control.
- **All three NA-2 checkpoints are pre-policy, and the rates prove it.** Seeds
  3407 and 3409 selected step 1500, seed 3408 step 3000 -- all inside the frozen
  bridge. The reported f_audio is therefore *byte-identical* across all three
  seeds (8.56 emotion / 10.57 speaker_count / 7.55 intent Hz): it is the warm
  start's rate, carrying no information about the run. NA-1 hit this too; two
  independent configurations now confirm ASR-WER-only selection cannot report a
  compression result. This is the highest-value open fix.
- **`rate_channel: aux` vs `reward` remains unmeasured, not measured-as-null.**
  Both arms were run to compare the channels, but since selection lands on a
  pre-policy checkpoint in every arm, the two are compared in the exact regime
  where neither channel is active. The published report no longer presents that
  comparison; it needs a selection fix first, not more seeds.
- **Phase C is where the rate moves, again unreported.** In seed 3408, unfreezing
  the last segmenter block at step 9k took the non-ASR rates 8.1-9.6 -> 5.4-5.6 Hz
  and ASR 10.8 -> 8.3 Hz, at a validation-WER cost of 4.25 -> 5.04. Emotion's own
  best validation macro-F1 (0.463) also occurs there, at step 10.5k, well above
  the 0.350 the selected checkpoint reports.
- **Per-phase trainability, exact (from the run's own phase-transition logs).**
  Phase A bridge: FiLM 0/10,240, pooler 1,149,440/1,149,440, last block
  0/789,760 -> 1,149,440 trainable segmenter params. Phase B film: FiLM
  10,240 -> 1,159,680. Phase C unfreeze: + last block 789,760 -> 1,949,440.
  Decoder (proj + routed LoRA, 66,785,920 across 5 task slots) trains in every
  phase at lr 2e-4; `lr_decoder_warmup == lr_decoder` in these runs, so the
  decoder LR does *not* change at the bridge boundary. Segmenter group lr 5e-5.
  Two consequences: (1) "FiLM-only" describes the *policy* -- the shared pooler
  trains in every phase including the bridge because its trainability comes from
  `train_shared_pooler`, not the phase, so Phase A freezes the boundary policy
  and not the segmenter, and FiLM is only 0.9% of Phase B's trainable segmenter
  params; (2) Phase C multiplies the policy-gradient-reachable parameters ~78x
  (10,240 -> 800,000), which is exactly when the rate starts moving.
- **What task FiLM can and cannot do.** `TaskFiLM` is `gamma_task * x + beta_task`
  with `gamma/beta = nn.Embedding(5, 1024)` init ones/zeros (exact identity at
  the start of Phase B), 10,240 params = 2,048 per task. It is **time-invariant**
  (one scale/shift per channel broadcast over frames), so it cannot place a
  boundary directly -- only reweight channels the frozen backbone reads. And it
  is applied *only* on the path into the boundary backbone: `pool()` is called on
  the **raw** features, so the decoder never sees a modulated frame and FiLM's
  sole route to the loss is the discrete boundary sample. That is the mechanism
  behind the Phase-B degeneracy, not a tuning problem.
- **GRPO is degenerate until the segmenter unfreezes.** `zero_variance_share`
  sits at 0.285-0.291 through the entire FiLM-only phase (29% of utterances have
  all K=4 rollouts score identically, contributing zero gradient) with
  within-group reward SD 2.2e-3, then drops to 0.025-0.055 with reward SD
  ~1.2-1.7e-2 once the block is trainable. FiLM alone cannot move the boundary
  distribution enough to produce a learning signal -- consistent with ADR-005's
  absorbing-state concern.

- **NA-1's reported checkpoint is pre-policy, and this is a design flaw, not an
  accident.** Checkpoint selection is ASR-WER-only. ASR WER rose across the run
  (3.174 at step 1500 -> 4.950 at 15000), so the argmin is the earliest
  validation point: step 1500, mid-bridge, boundary policy frozen. Every Phase-C
  compression result therefore sits in a checkpoint that is never evaluated on
  test or reported. As long as selection is ASR-only and the anchor degrades,
  this recurs in every run of the matrix. Fix options: report the final
  checkpoint (precedent: ADR-024), or select on a non-ASR task aggregate subject
  to an ASR-within-tolerance constraint, which is what the plan's "best
  checkpoint per task aggregate" implies and what a rate-quality frontier needs.
- **Phase C did work, and the results are worth keeping even though unreported.**
  Between steps 9000 and 12000 the per-task bands pulled every task down:
  intent 8.57 -> 5.24 Hz with accuracy 0.975 -> 0.976 (a 39% rate cut for
  nothing), emotion 8.34 -> 5.03 Hz with macro-F1 0.446 -> 0.454 (its best),
  speaker_count 6.45 -> 5.78 Hz. ASR paid: 10.59 -> 7.55 Hz cost +0.78 WER, but
  non-monotonically (+0.75 for the first 2.1 Hz, +0.03 for the next 0.9 Hz),
  which is as consistent with decoder adaptation lag as with a genuine rate
  cost. The fixed-rate sweep is the control that separates them.
- **The task mixture, not the extension machinery, is what hurts ASR.** NA-1
  reproduces MT-1's LibriSpeech numbers to three significant figures
  (dev-clean 3.143 vs 3.14, dev-other 5.551 vs 5.56), so the decoder and eval
  path are unperturbed. But MT-1's ASR-selected checkpoint was step **13500**
  and NA-1's is step **1500** -- same criterion, same warm start, same schedule.
  Three non-ASR tasks at 60% of batches drag selection from the end of training
  to the start; ST at 50% did not. The shared BiGRU pooler
  (`train_shared_pooler: True`) and the always-trainable `proj` see every task
  and are the prime suspects; the plan specifies freezing both in Phase A.
- **GRPO's signal is very weak in this configuration.** At step 4500,
  `zero_variance_share` = 0.290: 29% of utterances scored identically across all
  K=4 rollouts and contributed exactly zero policy gradient, with a mean
  within-group spread of 3.14e-03 against a mean reward of -0.154. ASR rate moved
  10.494 -> 10.490 Hz over 1500 GRPO steps. Only speaker_count, which sat below
  the global band floor and so had a live penalty gradient, moved steadily
  (5.51 -> 6.45 Hz). Inside the band there is no rate gradient at all, and the
  Phase-C ASR rate visibly wanders (8.48 -> 7.55 -> 8.32 -> 8.86 Hz) rather than
  converging -- so a "minimum-sufficient frequency" read off a band run is an
  artifact of where the policy happened to be. `captax` is the better Phase-C
  mode.
- **Speaker count is not yet a measurement.** Validation accuracy spanned
  0.485-0.688 over nine points with no trend, against ~+-0.03 sampling error on
  1000 balanced windows. Candidate causes: the max-concurrent label on synthetic
  mixtures, the 20% near-silent count-0 windows (which also generate the
  zero-variance rollouts), or checkpoint sensitivity. `evals/ami/balanced`
  (26860/3673/3406 windows, real AMI SDM meeting audio, exactly balanced 0-4) is
  the natural replacement, with two caveats: its label is distinct-speakers-
  active rather than max-concurrent, and it carries no per-source activity
  intervals, so it cannot support the boundary-alignment analysis.

- **Gate criterion 2 passes cleanly.** LibriSpeech dev-other is 5.56 against the
  5.63+-0.12 cohort baseline, i.e. no degradation while learning translation.
  dev-clean regressed 0.26 (2.88 -> 3.14), which the gate does not cover but
  which should be reported.
- **The bottleneck is upstream of translation.** CoVoST ASR WER is 15.22 on the
  *same audio* that yields 18.88 BLEU, versus 3.14/5.56 on LibriSpeech.
  Translation quality is bounded by source comprehension, so the dominant
  limitation is acoustic/domain transfer from a LibriSpeech-trained stack to
  Common Voice, not the translation head.
- **The learned rate is not corpus invariant.** The same policy emits 10.7 Hz on
  LibriSpeech but only 7.4-8.0 Hz on CoVoST -- below the rate band's own floor
  (rho_lo 0.15 = 7.5 Hz) and well under the 10.5-11 Hz the plan targeted. The
  ST decoder therefore receives a representation that is both degraded and
  over-compressed relative to design.
- **More training buys little.** Dev BLEU increments per 1500 steps were +3.9,
  +1.4, +0.9, +0.8, +0.6, -0.8, +1.8, +0.1. Clearly decelerating; doubling the
  budget plausibly returns +2-3 BLEU.
- **Benchmark calibration.** The commonly cited ">40 BLEU En-De" is a text-MT or
  MuST-C-family number, not CoVoST 2. CoVoST 2 En-De is a harder, lower-scoring
  benchmark; a large well-pretrained system sits in roughly the high 20s to low
  30s. 18.88 is genuinely low but the gap is ~10-13 BLEU, not 20+.
- **Cheapest decisive next test**: evaluate CoVoST ASR with the fixed k=5
  segmenter at matched rate. Flat WER exonerates the segmenter and points at the
  encoder/decoder domain gap; a sharp drop indicts the transferred rate policy.
- The absolute BLEU does not invalidate the relative task-conditioning claim,
  which is measured against controls at matched rate -- but at 15% source WER it
  is hard to argue the effect would hold in a competitive system, so fixing the
  domain gap belongs on the critical path rather than in polish.

---

#### Group: SeamlessM4T-v2 topline on our own CoVoST/LibriSpeech test sets

*Goal: measure how far the multi-task system is from a strong purpose-built
speech-translation model, on identical utterances with identical metric code,
rather than against published numbers computed on different subsets.*

Setup: `facebook/seamless-m4t-v2-large` speech-to-text path (1.5B parameters,
fully trained), beam 5, scored with the same SpeechBrain `ErrorRateStats` and
the same `BLEU()`/`CHRF(word_order=2)` objects, on the exact manifests our run
was evaluated on (jobs 191161 CoVoST, 193213 LibriSpeech).

| Split | Metric | SeamlessM4T-v2 | Ours (MT-1) | Gap |
|---|---|---:|---:|---:|
| CoVoST En-De test | sacreBLEU | **37.43** | 18.88 | -18.55 |
| CoVoST En-De test | chrF++ | 60.75 | 42.25 | -18.50 |
| CoVoST En ASR test | WER | **5.62** | 15.22 | +9.60 |
| LibriSpeech test-clean | WER | 2.63 | 3.26 | +0.63 |
| LibriSpeech test-other | WER | 4.94 | 5.68 | +0.74 |
| LibriSpeech dev-other | WER | 5.27 | 5.55 | +0.28 |

Analysis notes:

- **The domain-transfer diagnosis is confirmed and quantified.** Seamless is
  essentially domain flat (LibriSpeech test-other 4.94 -> CoVoST 5.62, +0.68).
  Ours degrades 5.68 -> 15.22, +9.54. The CoVoST audio is therefore not
  intrinsically hard, and our model is not generally weak -- it is within
  0.3-0.7 WER of a fully-trained 1.5B model on LibriSpeech while using a frozen
  encoder, a frozen 1B LLM, ~13M trainable parameters per task, and ~10.7 Hz
  compression. The failure is specific to Common Voice.
- **BLEU is a downstream symptom.** Translation cannot exceed source
  comprehension, and the source is recognized at 15.2% WER against Seamless's
  5.6%.
- **Rate is the prime suspect and is now measured**: the same policy emits
  10.7 Hz on LibriSpeech but 7.96 Hz on CoVoST ASR (7.39 Hz on ST) -- about 26%
  fewer audio tokens per second on speech of comparable density.
- **Correction to an earlier estimate in this file's analysis**: a large
  well-pretrained system on CoVoST 2 En-De reaches **37.4 BLEU**, not the "high
  20s to low 30s" previously written from recollection. The real gap is ~18.6
  BLEU, roughly a factor of two, which is larger than earlier framing implied.
- **Scoring caveat that matters**: SeamlessM4T emits cased, punctuated text,
  while our references follow the LibriSpeech convention. Scoring raw output
  gives WER ~99-101% on every split. All comparisons above apply the same
  normalizer to both systems; both raw and normalized numbers are stored in
  `artifacts/benchmarks/seamless_m4t_v2_*.json`.
- **Next decisive test**: CoVoST ASR with a fixed ~10 Hz stride. A sharp drop
  toward 6% indicts the transferred rate policy; a flat result exonerates the
  segmenter and points at encoder/decoder domain adaptation.

---

#### Group: Frozen-encoder domain probe (why CoVoST fails)

*Goal: isolate whether the frozen WavLM encoder, the one component that gets no
adaptation, is responsible for the ~2.7x CoVoST degradation.*

Setup: frozen `microsoft/wavlm-large` + a ~0.8M-parameter character-CTC head.
No segmenter, no pooling, no LLM, full 50 Hz frames. One probe trained per
corpus on **100 matched hours** (not matched utterances -- mean durations differ
2.4x), 12000 steps each, then evaluated on both. Jobs 204321/204322.

| | eval LibriSpeech | eval CoVoST |
|---|---|---|
| train LibriSpeech | **5.98** / 21.11 | 17.65 / 45.02 |
| train CoVoST | 8.29 / 31.21 | **14.69** / 41.71 |

(cells: CER / WER; greedy CTC with no language model, so absolute WER is high by
construction -- the ratio is the object of interest, not the level.)

Analysis notes:

- **The encoder is confirmed as the primary cause.** In-domain, matched-hours
  CER is 5.98 on LibriSpeech vs 14.69 on CoVoST, a **2.46x** ratio, against a
  pre-registered threshold of 1.8x for "encoder implicated". The full system's
  domain ratio is 2.68x (5.68 -> 15.22 WER), so the frozen features alone
  reproduce nearly the entire gap.
- **In-domain training does not rescue CoVoST.** Training the head on CoVoST
  improves CoVoST CER only from 17.65 to 14.69 (17% relative) and it remains
  2.46x worse than LibriSpeech. If the problem were adaptation or head capacity,
  in-domain training would close the gap; it does not, because the information
  is already degraded in the features.
- **The asymmetry is diagnostic.** LibriSpeech features are robust enough that
  even a CoVoST-trained head reaches 8.29 CER on them, while CoVoST features
  cap a CoVoST-trained head at 14.69.
- **This unifies the earlier observations.** The segmenter's rate drop on CoVoST
  (10.7 -> 7.96 Hz) is most likely a *symptom* of weaker features -- a boundary
  policy reading less distinct structure emits fewer confident boundaries --
  rather than an independent cause. The encoder sits upstream of the segmenter,
  the pooler, and the decoder alike.
- **Ruled out by this and the preceding analyses**: audio-token budget (WER flat
  then worsening with more tokens), text normalization (0% digits, 0.54% word
  drift), audio difficulty and reference noise (our WER is 10.16 even on the
  73.4% of utterances SeamlessM4T transcribes perfectly), speaker outliers
  (worst decile carries only 30.5% of errors), and raw LoRA rank (the same rank
  is within 0.6 WER of a 1.5B model on LibriSpeech).
- **Lever**: adapt the encoder -- unfreeze upper WavLM layers during the bridge,
  or substitute an encoder pretrained on broader-domain audio. Freezing the
  encoder is an architectural choice of this project, so this is a real design
  tension to state rather than a bug to fix silently.

---

#### 2026-09-09 - ICASSP experiments, generated results table, and conclusion drafted

- `drafts/icassp/main.tex` now contains a compact Experiments section covering the LibriSpeech-960h and matched train-clean-100 settings, frozen WavLM-Large and Llama-3.2-1B-Instruct bases, trainable projection/LoRA/policy/pooler components, policy and BiGRU sizes, optimization schedule, baselines, three-seed protocol, A100 hardware, WER/audio-token-rate/RTF metrics, recognition results, controlled attribution, adaptive allocation, inference efficiency, and a scoped conclusion.
- The primary table has two panels: the LS960 system comparison and the matched 100h attribution. The caption explicitly states that the LS960 reference rows have shorter adaptation than ASSET, so causal language is reserved for the matched panel. Efficiency text distinguishes selected batched-throughput operating points from batch-one latency and makes no streaming claim.
- Table source: saved per-seed WER files and evaluation logs under `recipes/LibriSpeech/ASR/transformer/results/`, including `speechllm_ls960_scaleup_v2_optimized`, `speechllm_ls960_step24k_corrected`, `speechllm_ls960_phone_ctc_longsplit8_mean_ce_1ep`, and `speechllm_segmenter_wavlm/attribution_100h_transformer_ar_bigru`.
- Generator: `drafts/icassp/scripts/make_table_results.py`; output: `drafts/icassp/tables/main_results.tex`. Regenerate with `make -C drafts/icassp tables` or run `/export/jsalt26/omnienc/users/cxiao/envs/jointllm/bin/python drafts/icassp/scripts/make_table_results.py`.
- The generator recomputes three-seed WER aggregates from corpus edit counts, recovers learned token rates from evaluation logs, validates every rounded paper value, and deliberately reads the locked one-pass phone-CTC result from epoch-1 log entries because later continuation jobs reused that output directory (ADR-028/029).
- The compiled manuscript is five pages: all technical content ends on page 4 and page 5 contains references only. The generated table uses 9-point text and booktabs rules; the LaTeX log has no overfull boxes, unresolved citations, or unresolved references.

#### 2026-09-09 - Expanded ICASSP 100h evidence and concise net-efficiency result

- Figure 2 now presents ten three-seed 100h configurations on both test splits: fixed strides 3/4/5/6/8, CNN/Transformer with mean/BiGRU pooling, and final-epoch audio-only phone-CTC. `drafts/icassp/scripts/make_figure_100h.py` reads corrected evaluation records, validates the protocol and complete splits, and writes `figures/results_100h.{pdf,png,json}` with full per-seed provenance.
- The matched table now includes fixed-mean **5.42±0.23 / 9.71±0.30** WER. Its fixed-BiGRU comparator is **5.17±0.05 / 9.70±0.12**; the clean SD is **0.047143**, superseding the earlier manually forced 0.04. Likewise, exact-edit-count LS960 character-oracle clean SD is **0.074858 → 0.07**, superseding 0.08. Means are unchanged.
- The longer 100h Transformer/BiGRU sweep remains **4.80/9.24**, distinct from the seven-epoch joint control **4.95/9.11**. The new figure's corrected fixed-k=5 WER is **5.82±0.17 / 9.85±0.27**, not the old right-padded 6.16/10.17 values.
- Independently reproduced 100h full-path RTF means/sample SDs: native batch 16 **0.066699/0.003071**, CNN/BiGRU batch 32 **0.023634/0.000912**, Transformer/BiGRU batch 32 **0.024561/0.000690**. The paper states the 2.8×/2.7× net batched gain after all model costs; full timing and batch-one context remain in the revision note.
- The LS960 runtime logs confirm accumulation **1**, despite the stored YAML default of 4. The manuscript now describes 300-second dynamic batches per update. Metrics were folded into their first uses, repeated prose was trimmed, and all technical content remains on pages 1–4 with references only on page 5.
- Rebuild: `make -C drafts/icassp all check preview`. Rationale/provenance: `drafts/icassp/research/revision_2026-09-09.md`; editorial decision: ADR-043.

#### 2026-09-09 - ICASSP oracle rates, Transformer rate emphasis, and boundary diagnostic

- Character-oracle test rates are **14.6073915 / 14.3371380 Hz → 14.6 / 14.3**, measured in all three corrected oracle benchmarks. The table generator verifies that the benchmark and 960h oracle share the same precomputed char boundary source; 960h WER remains unchanged.
- Transformer/BiGRU emits **8–9% fewer tokens than CNN/BiGRU at 960h** at comparable WER. In the longer 100h sweep, the reductions are **12.6% / 12.5%** on test-clean/other. Table rates are bold within the ASSET comparison, and Figure 2 now pairs fixed-stride overviews with expanded policy/pooler views.
- Added dev-clean phone-boundary evidence from `transformer_ar_bigru_phone_agreement_dev_clean.json` and `fixed_rate_phone_agreement_dev_clean.json` under `artifacts/segmenter/`. On 2,703 utterances with one-to-one matching within ±20 ms, the learned policy scores **63.1191±1.2175% F1 at 10.5838±0.6542 Hz**. Fixed k=5 scores **56.9432% at 10.0557 Hz**; k=4 scores **62.9481% at 12.5513 Hz**. ASSET uses **15.6760% fewer tokens than k=4** at similar F1.
- The figure generator recomputes F1 from saved match counts, validates common references/splits/seeds, and includes the diagnostic in `drafts/icassp/figures/results_100h.json`.
- Abstract now introduces an adaptive segmenter and retains the headline WER/rate pair. The longer, number-free conclusion identifies additional tasks and task-specific policies as future work. Setup and captions are shorter; Metrics stays removed and efficiency remains centered on net batched savings.
- The rebuilt PDF is five pages, with the complete conclusion on page 4 and only references on page 5; all five page previews were inspected. All fonts are embedded/subset, and there are no overfull boxes or unresolved citations/references. See ADR-044 and `drafts/icassp/research/revision_2026-09-09_round2.md`.

#### 2026-09-09 - Table 1 comparability audit: completed stronger phone-CTC reference

- Verified the second LS960 phone-CTC + mean-pooling pass from current WER edit counts, epoch-2 evaluation logs, and checkpoint metadata: **3.322809±0.092731 / 6.487337±0.060485 WER → 3.32±0.09 / 6.49±0.06**, with unchanged 10.90/10.49 Hz test rates. All three seeds improve dev-clean WER. The path still ends in `mean_ce_1ep` but current configs specify two epochs.
- The paper still uses the earlier **4.05/7.61** row because its generator deliberately enforces ADR-028/029's old reporting lock. This audit recommends the stronger verified reference for the next revision; it does not silently change the manuscript or separately published HTML report.
- Existing 100h phone-CTC + BiGRU **5.02/9.34** matches pooler architecture and decoder initialization but uses three downstream CE epochs versus seven in matched controls or twelve in the broader learned sweep. It addresses the absence-of-BiGRU concern only partially; training and checkpoint selection still differ.
- Recommended first added experiment: phone-CTC + BiGRU under the existing matched seven-epoch 100h protocol. 960h matched controls are needed for a segmentation-specific 960h superiority claim, not an absolute prerequisite for the narrower adaptive-policy claim. No jobs were launched.
- Full assessment and proposed caption/prose: `drafts/icassp/research/table1_comparability_2026-09-09.md`; recomputed evidence: `drafts/icassp/research/table1_comparability_evidence_2026-09-09.json`.


#### 2026-09-09 - Fresh 100h phone/char BiGRU control plan

- Plan: `plans/icassp_bigru_controls_100h_2026-09-09.md`; ADR-045. Both arms start fresh for seven epochs, seeds 3407/3408/3409, on another cluster. No jobs launched.
- “Fresh” preserves the shared char-decoder initialization used by the matched controls, while resetting pooler, optimizer, and counters. Old phone-run and policy cold-start checkpoints are unnecessary; external alignment bypasses the frozen policy.
- Code audit: BiGRU belongs to `modules.segmenter`, so its AdamW group uses `lr_segmenter=5e-5`; projection/LoRA use 2e-4. With `freeze_boundary_policy=True`, `_is_warmup` stays true and all seven epochs use CE, with accumulation 4.
- Explicitly set `train_csv=train-clean-100.csv`: the shared `train.csv` is merged 960h, and `skip_prep=True` makes `train_splits` alone insufficient.
- Command template passes Bash syntax and config-key checks. Destination GPU smoke checks and complete training/evaluation remain to be run; compute estimates are derived from existing A100 logs, not destination measurements.


#### 2026-09-09 - ICASSP citation audit and attribution corrections

- Audited all 24 original bibliography entries (16 cited plus eight unused). All nine existing DOIs and seven explicit arXiv identifiers identify the correct titles. Saved 18 publisher-deposited Crossref records and ten arXiv metadata records, including complete source author lists.
- Corrected AdaTS from generic ICLR main-conference attribution to the ICLR 2026 Multimodal Intelligence workshop and from a learned selector to parameter-free cosine-similarity grouping. Venue resolved through André Martins's explicit workshop announcement and the official ICLR workshop program; the live OpenReview forum/API was inaccessible.
- Replaced the Llama 3 citation for Llama-3.2-1B-Instruct with Meta's exact model card. Updated the unused Llama 3 entry's author prefix to match current arXiv v3 (Aaron Grattafiori first).
- Qualified ASSET's objective as an on-policy GRPO variant and added REINFORCE credit; added CTC/AdamW citations and activated existing Bhati/Jiang references for fixed pooling and earlier RL frame-rate control. The bibliography now contains 28 entries, 21 cited.
- Training prose now correctly places the pooler with the policy at 5e-5, with projection/LoRA at 2e-4; verified against the AdamW parameter groups. No code or experimental results changed.
- Rebuilt and inspected the five-page PDF and five PNG previews. Page 5 contains only references; all fonts are embedded/subset, all 21 reference-title URLs match the cited BibTeX entries, and no overfull boxes or unresolved references remain. Initials and abbreviated IEEE venue names preserve the page budget at the existing font sizes.
- Audit and evidence: drafts/icassp/research/citation_audit_2026-09-09.{md,json}. Build: make -C drafts/icassp all check preview. Durable choices: ADR-046.


#### 2026-09-09 - Editable Figure 1 beta preview

- Created `drafts/icassp/figures/JSALT26_beta.drawio.xml` with PDF/SVG/PNG previews and `JSALT26_beta.layout.json`. The diagram has 17 editable module/card groups; the original draw.io source hash is verified.
- The beta distinguishes frozen bases, continuous trainable components, boundary decisions, and scalar feedback; it makes projection/LoRA, deterministic-vs-sampled boundaries, and the zero-initialized residual projection explicit.
- Standalone PDF: 178 × 90 mm, all fonts embedded/subset, minimum body label 9.25 pt, no overflowing labels. The existing manuscript Figure 1 is still the active asset. Render manual edits with `make -C drafts/icassp figure-beta`; see ADR-047.

#### 2026-09-10 - GRPO terminology correction

- The paper consistently uses GRPO and states that reference-policy KL regularization is omitted (beta = 0). The joint CE/GRPO mixing coefficient is now alpha, corresponding to `pg_weight`; the policy-gradient equation and training implementation are unchanged. This supersedes the REINFORCE/variant wording in the 2026-09-09 citation audit (ADR-048).
- Rebuilt `drafts/icassp/main.pdf` and all five page previews. Technical content ends on page 4; page 5 contains references only. There are 20 cited references, all fonts are embedded/subset, and no overfull boxes, undefined references, or missing characters occur. All pages were visually inspected.
- The author rejected the Figure 1 beta; its files remain an unused archive. The original manual Figure 1 remains active.


#### 2026-09-10 — LS960 segment-unit and inter-word-pause pilot

- Job 1790396 completed on one A100/e03 in 4m56s. Three min-dev-WER Transformer-AR/BiGRU checkpoints, all step 24000; 24 preselected dev-clean utterances (188.31 s), no decoder or training. Two separate 12-speaker cohorts: random utterances and pause-enriched utterances.
- Random cohort: 11.2±0.4 Hz, phone F1 at ±20 ms **67.8±3.5%**, versus **52.6±1.5%** for exactly count-matched uniform cuts rounded to the same 20 ms frame lattice. Continuous-time references exclude utterance endpoints; this differs from the older quantized 100h metric. Paired utterance-bootstrap F1 delta +18.3 pp, 95% interval [14.4,22.4]; seeds averaged within each utterance. Pause cohort: 64.3±2.4% versus 49.7±1.2%.
- Across the same 31 aligned gaps ≥250 ms, seeds 3407/3408/3409 leave **28/31/26 gaps without an internal cut**. Only **1/0/1 gaps** contain a segment wholly inside the pause; ≥80%-silent tokens occur in **11/5/10**. Median onset errors 110/140/110 ms versus offset errors 20/30/20 ms. Pauses typically share a token with adjacent speech; no general dedicated-silence-token claim is supported.
- Segment-duration median is 60 ms, with mean seed-specific 10th/90th percentiles 40/147 ms. Roughly half of speech-dominated segments have ≥80% overlap with one phone. Broadband RMS-valley matching does not beat the equal-count grid; harmonic energy/demi-syllables remain untested. Automatic MFA reference errors and small-sample limits remain explicit.
- Artifacts: `artifacts/segmenter/ls960_boundary_audit_2026-09-10/{README.md,selection.json,predictions.json,summary.json,bootstrap.json,utterance_panels.pdf,pause_examples.pdf}`. Scripts: `audit_segment_units.py`, `report_segment_units.py`, `run_segment_units_audit.slurm`. Pre-audit paper checkpoint: `e312a2677`.


#### 2026-09-10 — Pause initialization clue and paper rollback

- On the same 31 pilot gaps, **character-CTC supervision targets** have 28 gaps without internal cuts, 1 with a wholly silent segment, and 2 whose two edges match one segment within ±40 ms. Median signed nearest-boundary onset/offset errors are −110/+30 ms; learned seeds give −110 to −140/+20 to +30 ms. This suggests inherited segmentation conventions but is not an evaluation of the actual cold policy or causal evidence about RL.
- A more tolerant same-segment criterion yields 1/2/4 isolated gaps at ±40 ms for learned seeds 3407/08/09 (6/2/8 at ±80 ms), so the strict wholly-contained criterion alone is insufficient. Raw follow-up: `artifacts/segmenter/ls960_boundary_audit_2026-09-10/pause_followup.json`; script `audit_pause_followup.py`.
- User rejected the manuscript changes; source and README restored to `e312a2677` and PDF rebuilt. Exploratory results remain separate. Comprehensive plan uses all 5559 test utterances (10.74 h), stage tracing, and controlled decoder interventions; not yet run.


#### 2026-09-10 — Phone-boundary review candidate

- Recomputed full-split 100h Transformer/BiGRU F1 from stored integer match counts: LibriSpeech dev-clean (2,703 utterances) 63.1191±1.2175%; TIMIT TEST excluding SA prompts (1,344 utterances) 66.9274±2.5474%. Fixed-k5: 56.9432% / 57.7994%. Three checkpoint SHA-256 identities match across datasets; SD is sample SD across seeds.
- New abstract sentence: “Phone-boundary analyses of the 100h model on LibriSpeech and TIMIT suggest approximately phone-like segmentation.” Body adds the TIMIT result and explicitly limits interpretation to partial boundary correspondence. Existing 960h WER, method, figures, and table remain unchanged.
- Artifacts are historical frame-quantized diagnostics; the TIMIT learned predictions omit the implicit frame-zero start while references/fixed grid include it. This differs from the LibriSpeech protocol, which adds that start. No controlled cross-corpus F1 comparison is justified; TIMIT stored learned cut Hz is not pooled-token Hz.
- Exact review diff and provenance: `drafts/icassp/research/phone_boundary_revision_2026-09-10.{diff,md,json}`; ADR-051. No new model job or remote push.
- Validation: revised PDF remains five pages, all technical content/conclusion on page 4 and references only on page 5; 140-word abstract, 21 cited references, all fonts embedded/subset, no overfull boxes or unresolved citations. All five page previews inspected. Changes remain a local review candidate.


#### 2026-09-10 — Structural phone-pattern audit and revised wording

- Completed 1,709 utterances / 241 speakers: 200 test-clean + 165 test-other (five per speaker), 1,344 native TIMIT TEST minus SA. Three 100h checkpoints on both corpora; three 960h checkpoints on LibriSpeech. After the native-duration invariant stopped job 1790791, resume 1790805 completed in 2m11s; CPU summary 1790807 completed in 25s. All 381 prior predictions reused; no selected utterance removed. Artifacts ~37 MB in artifacts/segmenter/ls960_phone_patterns_2026-09-10/.
- On 26,655 LibriSpeech speech phones, 960h deep-interior (>40 ms from both edges) split percentages are 11.375/9.642/10.850%; all-token multi-phone-center percentages 14.795/16.148/14.023%. Means round to 11% / 15%, now in the manuscript. +12.5-ms timestamp sensitivity gives 12.33% / 12.74%, retaining the pattern.
- Native TIMIT 100h seeds: 12.58–16.41% deep-interior phone splits, 19.28–23.12% all-token multi-phone-center spans. Every corpus/model seed beats an exactly count-matched frame-lattice grid in continuous-time boundary F1; paired speaker-bootstrap intervals are positive. These metrics describe flexible granularity, not superior phone identity or causal WER effects.
- Within-phone encoder-feature-change ratios against uniform positions in the same phone are small/inconsistent (roughly 0.976–1.040). Do not claim extra cuts capture greater acoustic detail. Longer phones are subdivided by the uniform control too.
- Replaying the historical TIMIT grid protocol gives 67.008% mean F1, close to the stored 66.927%, and exactly reproduces fixed-k5 counts. Continuous-time ±20-ms F1 is lower; reference quantization before ±1-frame matching broadens timing acceptance. The paper now explicitly states frame-grid F1.
- Abstract: “Analyses on LibriSpeech and TIMIT show sensitivity to phonetic transitions, with flexible token allocation within and across phones.” Only this sentence and the boundary paragraph change relative to the preceding review candidate. Final PDF: five pages, 144-word abstract, complete technical content on pages 1–4, embedded fonts, no unresolved/overfull warnings. No push.


#### 2026-09-11 — Autoregressive boundary sampling terminology

- Transformer AR uses conditional Bernoulli draws: `segmenter.py` calls `torch.bernoulli(sigmoid(logit))` in both `rollout` and `rollout_group`, then feeds the sampled action into the next step. The CNN first-order AR sampler implements the same binary conditional distribution through uniform-threshold draws and prefix composition.
- Autoregression describes dependence on boundary history; Bernoulli describes each binary conditional distribution. The configuration label `bernoulli` denotes the independent variant, but `transformer_ar` overrides that strategy internally. LS960 Transformer configuration uses local history 64 and `combined_on_policy`.
- Clarified only Section 3.1 with explicit history-conditioned logits/sampling and greedy inference. Review diff: drafts/icassp/research/ar_bernoulli_clarification_2026-09-11.diff.


#### 2026-09-12 — Single-task non-ASR and shared-audio boundary audit

- Completed 2216 -utterance boundary inference: 1431 CREMA-D voice-vote test / 420 Expresso matched short reads / 365 held-out LibriSpeech. Fifteen checkpoint identities; per-utterance SSL normalization, deterministic AR, nine repeat/batch/frozen-parent checks all identical. Independent matcher verified against DP on 9747 cases. Artifacts: artifacts/segmenter/nonasr_boundary_audit_2026-09-12/; review.md and paper_revision_proposal.md are the entry points.
- Audited all 54 saved single-task TEST results. Emotion macro-F1: native 40.4±6.0, fixed 10 Hz 43.4±3.3, fixed 5 Hz 41.9±3.3, frozen ASR policy 43.1±0.4 at 8.5606 Hz, GRPO 43.8±4.4 at 4.1485 Hz (three seeds). The latter is 51.54% fewer tokens at similar means; accuracy gain and equivalence are NOT established. Test labels are crowd voice votes (883/1431 neutral), so report macro-F1 and identify the protocol. Earlier notes that causally exclude rate/placement as bottlenecks are too strong for this evidence.
- Same-audio mean rates: ASR/emotion seed mean = 8.55/4.17 Hz CREMA-D, 10.14/5.16 Hz Expresso, 10.36/5.57 Hz LibriSpeech. These fresh diagnostic rates are separate from the original classification TEST rates. About 94% of emotion cuts are within 20 ms of a parent ASR cut on CREMA-D/Expresso (roughly 62–66% exact); circular-shift nulls are about 44%/53%. Highest-confidence ASR subsets at the exact emotion cut counts reproduce only 48–59% boundary F1.
- Emotion cuts favor vowel onsets and automatic voicing changes over both random and confidence ASR thinning. On Expresso, vowel-onset F1 is 42.0–47.1% vs 23.0–26.9% for confidence thinning; voicing-change F1 is 35.0–37.0% vs 23.5–25.3%. Frozen-WavLM feature changes at emotion cuts average~15–16% above parent ASR cuts and~16–18% above confidence subsets. These are descriptive acoustic preferences, not proof of emotional/prosodic information preservation.
- Expresso emotion segment medians 140–160 ms vs 60 ms for parent ASR. Some 34–43% of emotion tokens span 2+ phone centers, so avoid unit-identity claims. Word-boundary agreement is inconsistent. Of 80 aligned Expresso internal gaps≥250 ms, strict single-token isolation within 40 ms is 2.5–6.25% for emotion vs 7.5% ASR; LibriSpeech has 470 gaps with 0.43–0.85% emotion vs 1.06% ASR.
- MFA3.3.9 English ARPA aligns 1425/1431 CREMA-D and 420/420 Expresso; six CREMA-D failures stay in audio-only analysis. Nine missing dictionary words initially affected 49 Expresso clips; CMUdict/G2P augmentation resolves unknown phones. Results persist on 371 clips fully covered by the original dictionary, after excluding the lowest likelihood decile within style, and under +12.5 ms timestamp sensitivity.
- All emotion selected checkpoints are step 1200 after 480 last-block updates. Count selected steps 1560/1300/1040 have zero last-block updates, as does intent 3408 step 450; their TEST phase metadata is stale. Direct tensor comparisons confirm this. On Expresso seed 3408, selected intent/count remain 99.9/96.5% F1 against ASR; last-step controls move to 7.11/3.14 Hz. Do not pair these latter rates with selected-checkpoint task scores.
- Successful jobs:1791551/52/53 boundary inference,1791557 prosody,1791562 confidence,1791563 corrected alignment,1791564 final summary. All completed 0:0. Initial 1791550 stopped on the analysis's mistaken frame 0 assertion and was corrected; it supplied no scored predictions. Total artifact footprint about 9.2 GB including isolated MFA environment and 5.8 GB Expresso Parquet inputs.
- Generated tables: recipes/LibriSpeech/ASR/transformer/make_table_nonasr_audit.py -> artifact emotion_results.tex and (--compact) emotion_paper_proposal.tex. Inputs checkpoint_audit.json, outputs task_results.csv/task_results_summary.json. Six PDF/PNG figures and four documented examples are included. All 54 manuscript snapshots unchanged; no HTML update or remote publication.

- Gap-energy follow-up after visual review: aligned blanks can contain vocal activity. With ≥80% of interior RMS windows at least 20 dB below utterance p90 RMS, quiet-gap counts are 2 CREMA-D, 54 Expresso and 427 LibriSpeech. Emotion strict single-token isolation is 0–3.70% on quiet Expresso and 0.23–0.94% on quiet LibriSpeech; ASR is 1.85%/1.17%. The illustrative laughing gap has ~19% Praat voicing and fails this low-energy criterion. Keep word-gap and silence terminology distinct; gap_energy_summary.json retains 15/25-dB sensitivity.


#### 2026-09-13 — Non-ASR analysis HTML publication

- Live Analysis 03: https://borrisonxiao.github.io/jsalt26-downsampling/research/nonasr-task-boundaries.html. Six figures, 16 numbered tables, corpus filter, detailed checkpoints/sensitivity in expandable sections, six PDF downloads, task CSV and numerical-evidence ZIP. The page uses the completed September 12 audit, without new training or paper edits.
- Generators: recipes/LibriSpeech/ASR/transformer/make_nonasr_boundary_report.py and stage_nonasr_boundary_site.py; local HTML in artifacts/segmenter/nonasr_boundary_audit_2026-09-12/html/. The source publication contains only selected reproducibility inputs and figures, not the large environment/datasets/checkpoints.
- Published source commits 35862b756/c0f7407a5 on asset-dev/bilevel-optimization; local report checkpoint 5810262fb; Pages bef0f59 on origin/main. The source push used an isolated history to avoid publishing local manuscript commits. The main local branch still contains those unpublished paper commits; do not blindly push that history to asset-dev.
- Pages deployment 34775608028 succeeded. All 13 live files (collection, wrappers, standalone reports, PDFs, ZIP, CSV) are HTTP 200 and byte-identical to validated copies. Browser checks passed at 1280/390 px in light/dark mode; wrapper navigation checked at both widths. Evidence: html_checks/live_validation.json, browser_validation.json, wrapper_validation.json, content_validation.json.
- Corrected overstrong older non-ASR prose in the cumulative training report; its 35 numerical tables are unchanged. All 54 manuscript snapshot hashes still match. Cache version: 20260913-nonasr-boundaries.

## Expresso boundary patterns — 2026-09-13

- Follow-up requested by author prioritizes where policies agree/differ, expressive same-text variation, and acoustic explanations over recognition scores. Evidence: artifacts/segmenter/expresso_boundary_patterns_2026-09-13/. All54 manuscript snapshot files remain identical.
- All420 natural utterances (60 speaker×text groups, four speakers, seven renditions) compared with ASRparent3408, emotion3407/08/09, intent3408last, count3408last.420 same-waveform comparisons;360 default-vs-other pairs. The main affective contrasts use happy/sad/confused; remaining styles are separate.
- Directional ASR-anchor retention at20ms: consonant→vowel versus vowel→consonant is emotion3408 75.2%/36.5%, intentlast85.7%/56.8%, countlast74.8%/2.1%. Count's most-rising/most-falling envelope-slope quartile retention64.7%/4.1%. Direction persists at+12.5ms and across emotion seeds. This is an onset preference, not a syllable or speaker-change result.
- Count shares82.7% of its matched parent anchors with emotion (independent same-size subsets46.0%);17.3% [15.5,19.1] absent emotion. Mostly shared but not strictly nested;17.1% when count must be sparser within utterance. Task causality remains confounded by rate bands, shared initialization and checkpoint stage.
- Same-text emotion3408 F1 after local phone warping: default vs happy75.2%, sad72.4%, confused67.7%; corresponding within-phone random-placement controls36.4/36.8/35.7%. Raw F1 is11.1points below ASR, but margin above same-phone null is5.3points larger, so raw cross-policy agreement alone cannot establish extra emotion sensitivity.
- All360 word sequences match;2306/2418 paired words have identical stress-stripped phones;268/360 whole non-gap phone sequences match. Main-policy eligible-cut coverage94.4–95.0%. No both-empty main-policy comparisons. CIs bootstrap60 speaker×text blocks and are conditional on only four speakers.
- Lengthened corresponding phones (≥25%, both≥40ms) gain0.218 emotion cuts on average versus0.000 at similar durations.20.0% have more cuts>20ms inside the phone,2.5% fewer; that internal-cut statistic can include edge-distance reclassification, not just newly added cuts.
-480 controlled inputs:60 defaults × original/sham/±3semitones/duration0.8/1.25/flat pitch/envelope compression. One A100 inference job1792793. Emotion token ratio1.170 for duration1.25,0.868 for0.8; pitch shifts preserve~92–94% F1 versus sham with count nearly unchanged. ASR/intent/count token ratios for duration1.25 are1.115/1.179/1.079. These fall between fixed-landmark-count and fixed-clock-rate reference behaviors.
- Manipulation QC: duration preserved to ratio1.000 for pitch shifts; measured medians−3.003/+2.995semitones; envelope correlations0.998/0.995. Duration factors produce median F0 shifts+0.003/−0.003semitones. These are approximate controls, not perceptually identical emotion manipulations.
-351/360 original policy tracks exactly reproduce prior cached cuts;9 differ by1–4locations. Batch-size1 check job1792844 on48 inputs from the six affected recordings reproduces276/288 tracks; replacing those recordings changes full-panel F1 by at most0.107points. Do not claim universal bitwise batch invariance.
- Main illustration ex03_sad_00134: emotion retains0.90s near the end of stars, count retains0.98s into the vowel of in. Same-text ex01_*_00264: stable for/the vowel entries, variable finer cuts. The face vowel lasts200/190ms in default/happy; default-only internal cut for two of three emotion seeds is a counterexample to a duration-only rule.
- Report proposal: compact non-ASR subsection on directional transition retention, a same-text case and controlled temporal flexibility. Keep ASR abstract pending author review; no unit-discovery, emotion-token or causal task-superiority claim.

- Published revised analysis: https://borrisonxiao.github.io/jsalt26-downsampling/research/nonasr-task-boundaries.html?v=20260913-expresso-patterns . Eight figures including two interactive explorers, eight tables, all420 utterances/60 groups selectable, earlier audit retained. Source66ae664c1; Pages45ceb4e; successful workflow34781281482. Desktop/mobile light/dark and all60 text-group selections pass; the whole12-page site validates.
- Live verification completed: all21 wrapper, standalone, archive and download files returnHTTP200 and match local SHA256. The training standalone remains byte-identical; its wrapper cache version is updated.
