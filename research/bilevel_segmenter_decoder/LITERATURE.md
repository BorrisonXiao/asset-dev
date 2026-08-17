# Literature Review — Bilevel Optimization for a Learned Speech Segmenter and Decoder

**Instruction**: Determine whether bilevel optimization applies to the joint
segmenter–decoder problem and identify practical ideas that can be tested without
abandoning discrete RL.

**Date**: 2026-08-11

**Window**: all years; emphasis on practical large-scale and discrete methods

**Depth**: quick

## Scope & sub-questions

The core question is whether the segmenter should be optimized for the performance of
a decoder **after that decoder adapts to its boundaries**, rather than only for the
loss of the current decoder.

1. What is the exact inner/outer formulation for this project?
2. Can a discrete boundary policy use bilevel ideas without differentiating through
   hard mean pooling?
3. Which approximations are practical when the inner model is a LoRA-adapted LLM?
4. What failure modes make a direct DARTS-style relaxation risky?
5. What existing project evidence says the extra complexity may be worthwhile?

Target setting: frozen WavLM features, the local-history Transformer-AR segmenter,
BiGRU residual pooling, and the best char-initialized decoder-only LLM. The inner
parameters are the projection and LoRA adapters, with a separate arm that also
temporarily adapts the BiGRU pooler. K remains fixed at four.

## Implemented pilot on the bilevel branch

The first implementation keeps the deployed hard-boundary system unchanged. A dynamic
batch is split into disjoint support and query utterances. Each rollout lane receives
one reversible SGD update on support transcript CE, then a query NLL reward. The
temporary parameters are restored before the real support-CE plus GRPO update, and no
optimizer state is changed. Query actions receive per-utterance query advantages.
Support actions receive mean query quality for their lane, because their quality
effect passes through decoder adaptation, minus their own per-utterance frequency
cost.

The controlled arms are the established same-batch reward, a held-out query reward
without adaptation, decoder-only lookahead, and decoder-plus-pooler lookahead. The
short full-loop pilots passed the restoration and gradient checks. Decoder-only
lookahead reduced support CE by 13.19%, changed the preferred rollout for 31.0% of
query utterances, and took 1.12× the existing update time. Temporarily adapting the
BiGRU changed the preferred rollout for 35.3% but raised time to 1.32×. These are
mechanics and reward-signal results, not corpus-level WER results. See
`PILOT_RESULTS.md` for the complete measurements.

## Landscape summary

Bilevel optimization is a natural description of the scientific objective. The
segmenter parameters are upper-level variables. For any segmenter, the decoder is an
inner learner that adapts to the resulting acoustic-prefix distribution. The outer
objective asks whether that adapted decoder generalizes on held-out speech at the
desired kept ratio. This differs from the current single-loop implementation: sampled
boundaries are scored by the *current* decoder, the reward is detached, and decoder CE
and segmenter GRPO are then updated together on the same training stream.

The general bilevel literature offers two gradient families. Unrolled differentiation
backpropagates through a small number of inner optimizer steps; truncated variants can
be much cheaper than exact unrolling. Implicit differentiation avoids storing the
trajectory but requires Hessian-vector/inverse-Hessian approximations and assumptions
that are uncomfortable for a non-convex LoRA-adapted LLM. Neither is the cleanest first
step here because boundary actions and hard pooling remain discrete.

The closest algorithmic analogy is probabilistic bilevel coreset selection. It samples
a discrete subset, trains an inner model on that subset, evaluates the adapted model,
and uses policy gradient for the outer selection probabilities. That suggests a
practical speech version: sample support/query boundaries, make one or a few temporary
projection+LoRA updates on the support batch, score the temporary decoder on a query
batch, and use that post-adaptation score as the GRPO reward. No gradient must pass
through the hard boundaries or the temporary decoder update.

DARTS supplies a useful warning rather than the default implementation. A continuous
boundary relaxation would enable ordinary hypergradients, but differentiable
architecture search is known to suffer discretization and collapse pathologies. In
this project, a soft acoustic prefix would also change the pooling operator being
studied. A DARTS/CIF-style relaxation is therefore a separate later arm, not the first
bilevel pilot.

## Paper table

| Paper | Venue | Method | Relevance here | Status |
|---|---|---|---|---|
| Franceschi et al., “Bilevel Programming for Hyperparameter Optimization and Meta-Learning” (2018) | ICML | Treats the inner optimization dynamics as part of an approximate bilevel problem | Formal basis for optimizing the segmenter against an adapted decoder | ✅ verified (PMLR) |
| Shaban et al., “Truncated Back-propagation for Bilevel Optimization” (2019) | AISTATS | Backpropagates through only a few inner steps; reports similar performance to exact gradients at lower cost | Supports one/few-step decoder lookahead instead of full decoder convergence | ✅ verified (PMLR) |
| Lorraine, Vicol, Duvenaud, “Optimizing Millions of Hyperparameters by Implicit Differentiation” (2020) | AISTATS | Uses the implicit function theorem and inverse-Hessian approximations at large scale | Possible later route, but second-order approximations over our decoder are high-risk and costly | ✅ verified (PMLR) |
| Liu, Simonyan, Yang, “DARTS” (2019) | ICLR | Continuous relaxation of discrete architecture choices with a bilevel train/validation split | Template for soft boundaries, but it changes hard pooling and is not the safest first pilot | ✅ verified (arXiv/ICLR) |
| Zela et al., “Understanding and Robustifying Differentiable Architecture Search” (2020) | ICLR | Identifies degenerate high-curvature DARTS solutions and proposes robustification | Warns that continuous relaxations can optimize a proxy yet discretize poorly | ✅ verified (OpenReview/ICLR) |
| Ren et al., “Learning to Reweight Examples for Robust Deep Learning” (2018) | ICML | Uses a clean validation loss after a one-step model update to meta-learn training weights | Direct precedent for a held-out query batch controlling an inner training decision | ✅ verified (PMLR) |
| Zhou et al., “Probabilistic Bilevel Coreset Selection” (2023) | arXiv preprint | Samples discrete subsets and uses policy gradient on post-training model performance | Closest mechanical analogy: discrete frame/boundary selection can retain GRPO while changing its reward timing | ✅ verified (arXiv) |
| Yang, Gao, Yuan, “Bilevel Reinforcement Learning via the Development of Hyper-gradient without Lower-Level Convexity” (2025) | AISTATS | Develops first-order hypergradients for a structured bilevel RL problem without lower-level convexity | Confirms that RL and bilevel structure can coexist, though its MDP fixed-point assumptions do not directly match our decoder | ✅ verified (PMLR) |

## Themes and consensus

### The inner/outer data split matters

Bilevel methods normally avoid optimizing both levels on the same examples. The inner
decoder should adapt on a support batch; the segmenter should be judged on a separate
query batch. Otherwise the lookahead reward can favor boundaries that are easy to fit
immediately rather than boundaries that generalize.

### Short unrolls are diagnostics, not exact solutions

One decoder step asks “does this boundary distribution enable useful immediate
adaptation?” It does not approximate a fully converged decoder perfectly. Increasing
the inner horizon improves fidelity but raises cost and meta-overfitting risk. The
horizon must therefore be ablated explicitly.

### Discrete policy gradient is compatible with the bilevel objective

The outer variable need not be differentiable through the inner learner. A sampled
boundary sequence can receive a scalar reward computed after temporary decoder
adaptation. The score-function estimator then updates the policy, just as probabilistic
bilevel subset selection updates discrete sampling probabilities.

### Continuous relaxation has a train–deployment gap

Soft boundaries make hypergradients convenient, but the deployed model uses hard
segments. DARTS failure analyses show that optimizing a continuous proxy can favor
solutions that do not survive discretization. Any soft-pooling arm must evaluate this
gap rather than assume it away.

## Open gaps and opportunities

1. **Post-adaptation boundary value is unmeasured.** Current rewards score boundaries
   with the current decoder, not with a decoder after even one update on those
   boundaries.
2. **The same-batch update may reward memorization.** A support/query split would test
   whether a boundary choice produces transferable decoder gradients.
3. **OPD supplies a stronger inner objective.** The temporary decoder step can use
   CE+oracle-char distillation, while the outer query reward remains ASR NLL/CER plus
   the kept-ratio penalty.
4. **Hard-boundary bilevel RL avoids a proxy mismatch.** It preserves the exact
   first-order AR policy and mean-pooling deployment path.
5. **Compute can be bounded.** Clone only the roughly 19M trainable projection+LoRA
   parameters, use one inner step, two rollouts, and sequential fast weights.
6. **The first claim should be narrow.** Demonstrate that post-adaptation rewards rank
   rollouts differently and improve held-out NLL before attempting a full WER sweep.

## References

1. Franceschi, L., Frasconi, P., Salzo, S., Grazzi, R., Pontil, M. (2018). [Bilevel Programming for Hyperparameter Optimization and Meta-Learning](https://proceedings.mlr.press/v80/franceschi18a.html). ICML.
2. Shaban, A., Cheng, C.-A., Hatch, N., Boots, B. (2019). [Truncated Back-propagation for Bilevel Optimization](https://proceedings.mlr.press/v89/shaban19a.html). AISTATS.
3. Lorraine, J., Vicol, P., Duvenaud, D. (2020). [Optimizing Millions of Hyperparameters by Implicit Differentiation](https://proceedings.mlr.press/v108/lorraine20a.html). AISTATS.
4. Liu, H., Simonyan, K., Yang, Y. (2019). [DARTS: Differentiable Architecture Search](https://arxiv.org/abs/1806.09055). ICLR.
5. Zela, A., Elsken, T., Saikia, T., Marrakchi, Y., Brox, T., Hutter, F. (2020). [Understanding and Robustifying Differentiable Architecture Search](https://openreview.net/forum?id=H1gDNyrKDS). ICLR.
6. Ren, M., Zeng, W., Yang, B., Urtasun, R. (2018). [Learning to Reweight Examples for Robust Deep Learning](https://proceedings.mlr.press/v80/ren18a.html). ICML.
7. Zhou, X., Pi, R., Zhang, W., Lin, Y., Zhang, T. (2023). [Probabilistic Bilevel Coreset Selection](https://arxiv.org/abs/2301.09880). arXiv:2301.09880.
8. Yang, Y., Gao, B., Yuan, Y.-x. (2025). [Bilevel Reinforcement Learning via the Development of Hyper-gradient without Lower-Level Convexity](https://proceedings.mlr.press/v258/yang25g.html). AISTATS.
