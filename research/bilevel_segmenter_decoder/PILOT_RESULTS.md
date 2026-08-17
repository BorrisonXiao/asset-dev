# Bilevel implementation pilot

**Date:** 2026-08-17

**Branch:** `bilevel-optimization`

**Status:** implementation checks passed; full-data ASR comparison not yet run

## What was tested

The pilot starts from the seed-3407 WavLM local-history Transformer-AR segmenter,
BiGRU residual pooling, and best char-decoder checkpoint. K is fixed at four. Each
training batch is divided into support and query utterances.

For every rollout lane, the lookahead arms:

1. take one temporary SGD step on support transcript CE;
2. score the same lane on query speech;
3. restore the changed parameters exactly;
4. use the query score as the segmenter reward; and
5. perform the real support-CE plus GRPO update.

Query boundaries receive their own query NLL and frequency penalty. Support
boundaries receive mean query NLL after adaptation plus their own frequency
penalty. This keeps the cost of a support action attached to that action.

## Controls

Unit tests cover disjoint support/query slicing, exact parameter restoration,
exception-safe restoration, unchanged accumulated gradients, RNG replay, and
rollout-ranking diagnostics. A zero-inner-step GPU control gave exactly equal
current and adapted rewards for decoder-only lookahead. The decoder-plus-pooler
control had only `9e-6` mean numerical drift, rank correlation 1.0, and no
preferred-rollout changes.

## Matched short runs

Jobs 93094--93097 ran eight training batches per arm, followed by the recipe's
debug validation and test loops. All completed with exit code zero. These runs
test the training mechanics and reward signal; their WER is not a corpus-level
result.

| Arm | Support CE before → after | Reduction | Mean signed query-reward change | Mean absolute change | Preferred rollout changed | Rank correlation | Train time | Time vs. existing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Existing update (`off`) | — | — | — | — | — | — | 16.48 s | 1.00× |
| Held-out reward, no lookahead | — | — | 0 | 0 | 0% | 1.000 | 12.49 s | 0.76× |
| Decoder lookahead | 0.03465 → 0.03008 | 13.19% | +1.56e-4 | 4.80e-4 | 31.0% | 0.706 | 18.38 s | 1.12× |
| Decoder + BiGRU lookahead | 0.03481 → 0.03010 | 13.53% | +1.35e-4 | 4.83e-4 | 35.3% | 0.674 | 21.70 s | 1.32× |

Every selected decoder tensor received a finite nonzero gradient: 230/230 for
decoder-only lookahead and 240/240 when the BiGRU was included.

## Conclusion

The bilevel signal is real enough to test at full scale. One support update lowers
the support loss and changes rollout rankings well above the numerical-control
floor. In this short run it also improves mean query reward slightly, but eight
batches are too few to claim a generalization or WER gain.

Decoder-only lookahead is the better next arm. It gives almost the same reward
change as decoder-plus-BiGRU at lower cost. The next experiment should be a
one-seed, full-data comparison of the existing update, held-out reward without
adaptation, and decoder-only lookahead. Run three seeds only if that comparison
improves dev WER or held-out NLL at the same output frequency.
