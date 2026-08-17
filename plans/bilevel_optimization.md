# Plan: support/query bilevel training for ASSET

**Branch:** `bilevel-optimization`

**Status:** implemented; calibrated short full-loop pilots completed
**Primary system:** WavLM features, local-history Transformer-AR segmenter,
BiGRU residual pooling, best char-decoder initialization, K=4

## Question

The existing joint-RL loop scores boundaries with the decoder as it currently
stands. Does the segmenter make better choices if a boundary rollout is judged
by a decoder *after* that decoder has adapted to those boundaries?

## First implementation

Each dynamic batch is split into disjoint support and query utterances.

1. Sample K=4 hard Transformer-AR boundary histories.
2. For each rollout lane, make one temporary SGD step on support transcript CE.
3. Evaluate query NLL with the temporarily adapted parameters.
4. Restore every temporary parameter exactly; do not touch optimizer state.
5. Use query reward and query frequency cost for query actions. For lookahead
   modes, give support actions the mean query quality after adaptation minus each
   support utterance's own frequency cost.
6. Perform the real update: support decoder CE plus the outer GRPO loss.

No hypergradient crosses the hard boundary or the inner update. This is a
first-order score-function bilevel method, not a DARTS-style relaxation.

## Controlled arms

| Arm | Segmenter reward | Purpose |
|---|---|---|
| `off` | Current decoder NLL on the same batch | Established implementation control |
| `heldout` | Current decoder NLL on disjoint query speech | Isolate support/query separation |
| `decoder_lookahead` | Query NLL after temporary projection+LoRA adaptation | Test post-adaptation boundary value |
| `decoder_pooler_lookahead` | Query NLL after temporary decoder+BiGRU adaptation | Test whether the order-aware pooler belongs in the inner problem |

## Validation gates

The short pilots must satisfy all of these before full training:

- temporary parameters are restored exactly, including exception paths;
- accumulated outer gradients and optimizer state are not modified by lookahead;
- support CE decreases after the temporary step;
- gradients and rewards remain finite;
- adapted rewards change at least some K-rollout rankings;
- `bilevel_mode=off` continues to use the unchanged combined on-policy path;
- runtime and peak memory are measured before selecting the full pilot arm.

## Pilot sequence

1. CPU unit and Transformer-AR policy tests.
2. One two-batch `decoder_lookahead` GPU smoke run.
3. Four matched eight-batch full-loop debug runs from the same completed seed
   3407 Transformer-AR+BiGRU checkpoint.
4. If the lookahead signal is non-trivial and stable, run a one-seed full-data
   comparison of `off`, `heldout`, and the better lookahead scope.
5. Only after the one-seed result, schedule the three-seed confirmation.

Debug-run WER is a software check only; it is not a corpus-level result.

## Completed pilot result (2026-08-17)

Jobs 93094--93097 ran eight train batches per arm from the same seed-3407
checkpoint. All jobs completed with exit code zero. A separate zero-step control
confirmed exact equality for decoder-only paired rewards; the decoder-plus-pooler
control had only 9e-6 mean numerical drift, rank correlation 1.0, and no top-rollout
flips. The temporary SGD learning rate was calibrated to 0.01.

| Lookahead scope | Support CE reduction | Mean absolute query-reward change | Top-rollout flip | Rank correlation | Train time vs `off` |
|---|---:|---:|---:|---:|---:|
| projection + LoRA | 13.19% | 4.80e-4 | 31.0% | 0.706 | 1.12x |
| projection + LoRA + BiGRU | 13.53% | 4.83e-4 | 35.3% | 0.674 | 1.32x |

Both implementations pass the mechanics gates: all selected tensors receive finite,
nonzero gradients, the temporary parameters are restored, support CE falls, and
lookahead changes rollout rankings. Mean query reward improved slightly in both arms,
but an eight-batch run is not evidence of better WER.

The decoder-only scope is the next full-data candidate: it gives a clear ranking
signal at lower cost, while temporarily adapting the BiGRU does not show a stronger
signal in this pilot. Run one full seed of `off`, `heldout`, and
`decoder_lookahead` before any three-seed confirmation. Do not use the debug WER
numbers as ASR results.

Detailed measurements are in
`research/bilevel_segmenter_decoder/PILOT_RESULTS.md`.
