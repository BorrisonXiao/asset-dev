# Mandarin ASR errors in the Table 3 replication

[Read the published analysis](https://borrisonxiao.github.io/jsalt26-downsampling/research/table3-asr-errors.html).

The analysis covers four systems and three seeds on the same 7,128 AISHELL-1 test utterances. It uses the November 2026 auxiliary-loss replication, with published [Table 3](https://arxiv.org/pdf/2610.14309#page=4) values alongside the new scores. The published ASSET run applied the rate penalty to the reward. The new ASSET runs apply it as an auxiliary loss.

All 12 prediction files match the filtered test manifest and the saved collector checksums. Independent scoring reproduced every archived CER numerator and denominator. Both the underscore-inclusive scorer and character-only diagnostics are present in `analysis.json`. The deterministic alignment tie rule can change the mix of edit types. The saved scorer's counts are preserved separately.

ASSET has the lowest replication CER. Its character-only substitution rate is 7.10%, compared with 8.10% for the character oracle, 8.55% for the fixed grid and 12.27% for the phone oracle. The character oracle makes fewer insertions and deletions than ASSET. Six seed-3407 transcript comparisons illustrate substitutions, omissions, repetitions and ASSET regressions. The analysis reports observed edits and descriptive paired comparisons.

Run the scorer from the repository root with the AISHELL recovery artifacts available.

```bash
python3 recipes/LibriSpeech/ASR/transformer/analyze_aishell_table3_errors.py \
  --replication-root artifacts/reproductions/aishell_table3_20261112_recovery/asr_20261113
```

Run `recipes/LibriSpeech/ASR/transformer/make_aishell_table3_error_report.py` using a Python environment with the HTML report skill and NumPy available. This generates the standalone HTML from `analysis.json` and the regenerated `utterance_candidates.json`. The scoring and rendering steps use 0 GPUs.
