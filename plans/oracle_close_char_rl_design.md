# Oracle-close char initialization followed by constrained RL

**Date:** 2026-08-09  
**Goal:** begin from behavior as close as possible to the WavLM char-oracle
system, then test whether learned merging improves WER/compression without
destroying char structure.

**Research commitment:** RL is the method being improved, not an optional arm to
drop when the current implementation underperforms. Diagnostic gates below decide
which part of RL training to revise; they are not stop conditions for the project.

## Active controlled dual-encoder study (launched 2026-08-10)

Before the fully independent replication, isolate the effect of the segmenter's
input representation with exactly six runs:

- fix the best wav2vec2 CNN char segmenter (seed 3407, epoch 4, dev exact F1
  0.89454) in every run;
- predict boundary indices from frozen wav2vec2-base features, apply those indices
  to frozen WavLM-Large features, then feed the pooled WavLM sequence to the
  projection + Llama decoder;
- compare three RL seeds (3407/3408/3409) from the best WavLM oracle-char decoder
  after two predicted-boundary adaptation epochs against three seeds from a fresh
  projection/LoRA after six warmup epochs;
- give both arms the same subsequent 10 epochs of Bernoulli NLL-GRPO, decoder and
  segmenter learning rates, rate band [0.15, 0.25], and checkpoint selection;
- report decoder-prefix kept ratio and total dual-encoder RTF separately.

Jobs 60818–60820 are the oracle-decoder arm; 60821–60823 are the scratch-decoder
arm. Fully independent detector/decoder replication is explicitly deferred until
this controlled attribution study is interpreted.

## Core design choice

Separate the system into two decisions:

1. a **char-boundary detector**, trained to imitate the oracle and kept stable;
2. an **RL merge controller**, initialized to keep every detected char boundary
   and allowed to remove boundaries conservatively.

Do not resume unconstrained Bernoulli sampling over every WavLM frame. That
parameterization can delete, add, and relocate boundaries immediately, so a
char initialization is not behaviorally preserved.

The preferred merge controller is the planned autoregressive Transformer
decoder operating over the candidate char-boundary sequence. Its previous
keep/merge decisions are explicit inputs, which models dependencies without
requiring it to rediscover char timing from scratch.

## Stage 0 — boundary/decoder intervention

Run the same oracle-trained decoder with:

- oracle char boundaries;
- cold learned boundaries;
- selected post-RL boundaries.

Run this separately for CNN and Transformer policies with identical decoding
settings. Then run the reverse swap (learned decoder + oracle/cold/post-RL
boundaries) only if the first intervention shows a meaningful gap.

**Gate:** quantify separately the cold-imitation cost and post-RL drift cost, then
use that attribution to revise initialization, warmup, reward, or policy structure.

## Stage 1 — make the WavLM char detector oracle-close

Use dev-clean only for model selection and screen one seed before replication.

1. **Calibrate the decision threshold.** Sweep the boundary-logit threshold on
   dev-clean instead of assuming logit > 0 is F1-optimal. Report exact, ±1, and
   ±2-frame F1 plus true segment count.
2. **Probe temporally local WavLM representations.** The targets were generated
   by a wav2vec2 CTC model, while the current segmenter consumes WavLM's final
   contextual layer. Compare a small set of lower/middle/final layers or a
   learned layer mixture. Avoid caching many full layers simultaneously: four
   fp16 100-hour WavLM layers can exceed 100 GB.
3. **Compare the current CNN with the autoregressive boundary Transformer.**
   Train the latter with teacher forcing on char labels; validate free-running
   decoding separately to detect exposure bias.
4. **Use a two-part imitation objective.** Keep exact-frame BCE as the primary
   loss so the detector matches the decoder's training distribution, and add a
   local ±1-frame matching term so acoustically equivalent offsets do not
   dominate optimization.
5. **Select by functional behavior, not F1 alone.** Evaluate every candidate
   under the frozen oracle decoder.

Suggested gate before RL:

- ±1-frame F1 ≥ 0.95 (aspirational target ≥ 0.97);
- exact F1 improves over the current 0.799 CNN / 0.777 Transformer;
- oracle-decoder WER with detected boundaries is within 0.5 absolute points of
  oracle-boundary WER on dev-clean.

If no detector passes the functional WER gate, keep the RL track active but make
the mismatch explicit: initialize from the best available detector and oracle
decoder, bridge briefly on predicted boundaries, and constrain the policy class so
RL cannot silently turn detector error into arbitrary framewise drift.

## Stage 2 — bridge the oracle decoder to predicted char boundaries

Initialize projection and LoRA weights from the selected oracle-char decoder.

- If the Stage-1 cross-swap is already within 0.5 WER points, use the oracle
  decoder directly.
- Otherwise freeze the detector and run a short decoder bridge phase. Begin
  with mostly oracle boundaries and anneal toward predicted boundaries
  (100/0 → 50/50 → 0/100). Add teacher-logit distillation from the frozen
  oracle-boundary forward pass.

Select the bridge checkpoint on dev-clean predicted-boundary WER, subject to no
material regression on the oracle-boundary control.

This phase addresses decoder/segmenter distribution mismatch without allowing
the boundary detector to drift.

## Stage 3 — constrained residual RL

### Action space

- Candidate actions exist only at detected char boundaries.
- Initial policy keeps every candidate (`P(keep)` near 1).
- First experiments allow **merge/delete only**. Optional ±1-frame shifts and
  new-boundary insertion are separate later ablations.
- The autoregressive controller conditions each keep/merge decision on WavLM
  context, the candidate char segment, and previous decisions.

### Objective and curriculum

Use a paired improvement reward relative to the all-keep segmentation for the
same utterance, rather than raw NLL alone. Include the all-keep action in every
GRPO group as a fixed control variate.

Recommended phases:

1. **Frozen decoder:** residual merge controller only; NLL reward; rate pinned
   near the starting char rate.
2. **Rate curriculum:** gradually move the target from approximately 0.30 to
   0.25, then 0.20 only after dev WER remains stable.
3. **Careful co-training:** unfreeze projection/LoRA at a lower decoder LR after
   the policy passes the frozen-decoder gate.
4. **Reward confirmation:** compare NLL with CER reward only for configurations
   that remain stable.

Keep either a KL penalty to the all-keep/cold policy or an auxiliary char loss.
Checkpoint by dev WER subject to a boundary-drift constraint; do not select on
training reward.

Suggested RL gate:

- dev-clean WER improves by ≥0.3 absolute or achieves a meaningful compression
  gain with ≤0.2 WER regression;
- ±1-frame agreement with retained char candidates stays ≥0.90;
- gains survive three independent end-to-end seeds.

## Required controls

For every claimed RL improvement, compare against:

- oracle boundaries + oracle decoder;
- detected char boundaries + oracle/bridged decoder;
- all-keep residual policy;
- random deletion of char boundaries at matched kept ratio;
- fixed-rate pooling at matched kept ratio;
- deterministic confidence- or duration-based char merging;
- learned RL merging with decoder frozen and decoder co-trained.

The random and deterministic char-merge controls are essential: they establish
whether RL learns *which* char boundaries to remove rather than benefiting only
from a shorter prefix.

## Reporting protocol

- Lock hyperparameters on dev-clean before touching test-clean/test-other.
- Report WER, CER, true kept ratio, exact/±1/±2 boundary F1, merge count, and
  selected epoch.
- Use three fully independent seeds, including detector and decoder bridge;
  shared warmup seeds do not measure end-to-end variance.
- Show paired per-utterance WER changes for oracle, cold, and RL boundaries.

## Recommended immediate sequence

1. Finish the oracle-decoder boundary swap.
2. Launch the best-segmenter + best-oracle-decoder RL warm start, with a short
   decoder bridge and a longer joint phase.
3. Run a matched structured autoregressive-policy arm to replace independent
   Bernoulli decisions.
4. Threshold-calibrate and probe WavLM layers in parallel to improve the next
   detector initialization.
5. Use the intervention and RL curves to refine reward/rate scheduling.
6. Replicate the winning design with three independent seeds.
