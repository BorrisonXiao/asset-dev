# Can we backprop the ASR loss (CE) into the segmenter with a straight-through estimator?

**Status:** feasibility analysis, 2026-08-05. No production code changed yet.
Probe code: `/weka/scratch/.../users/cxiao/ste_probe/`.
Related: `plans/dynamic_segmenter_rl_v1.md` (§6 Gumbel-ST arm), `docs/project_notes/issues.md`
(2026-08-02, "Gumbel-ST dropped").

---

## TL;DR

- The 2026-08-02 note ("can't backprop through hard mean-pool") is **correct for the current
  code**, and dropping Gumbel-ST on that basis was right at the time. Confirmed empirically:
  handing a relaxed boundary to `mean_pool_segments` produces **no error and no gradient**.
- It is **not** a mathematical dead end. Rewriting pooling as a matrix product makes an STE
  work: exact hard forward, live gradients into the boundary logits, composes with the
  existing `rate_loss` untouched, and ~2x cheaper per step than the GRPO arm.
- **But the gradient is weak.** Against a *degenerate* cold start every arm looks good, and a
  fully-soft (CIF-style) forward clearly beats the hard STE. Against a **realistic** cold start —
  already sensible, systematically 3 frames misaligned — **no arm corrects the misalignment**:
  mean |offset| goes 3.34 -> 3.14 (soft-index) / 3.25 (GRPO) / 3.90 (STE-index, *worse*), while
  direct supervision reaches 0.00 on the same task. Only the decoder-side content objective moves.
- **Recommendation:** worth running as a cheap baseline if the 2x2 axis matters, using the
  **soft-index** formulation — not hard STE, and not the survival-product form, which collapses
  reproducibly. Expect a flat/negative result. The higher-value experiment this surfaced is a
  different control entirely — see §7.

---

## 1. Why CE cannot reach the segmenter today

`segment_pooling.mean_pool_segments` consumes the boundary tensor **only as an integer index**.

Worked example — 1 utterance, 6 frames, 2 feature dims, last frame padded,
`boundary = [0, 0, 1, 0, 1, 0]`:

```
is_first_valid             [[1, 0, 0, 0, 0, 0]]   frame 0 always opens segment 0
(boundary == 1)            [[0, 0, 1, 0, 1, 0]]
starts_new  (OR, & valid)  [[1, 0, 1, 0, 1, 0]]   <- a (B,T) VECTOR of 0/1
cumsum(starts_new, dim=1)  [[1, 1, 2, 2, 3, 3]]   <- running count ALONG TIME
minus 1                    [[0, 0, 1, 1, 2, 2]]
masked_fill(pad, -1)       [[0, 0, 1, 1, 2, -1]]  <- seg_ids
```

`starts_new` is a per-frame 0/1 **vector**, and the cumsum runs along time, so at frame `t` it
is "how many segments have started at or before `t`". The `- 1` makes it 0-indexed (frame 0 is
already inside segment 0, so count 1 -> id 0). The `| is_first_valid` term is the frame-0 guard:
frame 0 opens segment 0 whether or not a boundary was predicted there, so a predicted boundary
at frame 0 cannot open an empty phantom segment.

Then the pooling itself flattens `(utterance b, segment j)` into one integer address so a
single 1-D scatter handles the whole batch:

```
batch_offset = b * T_new          [[0]]          T_new = 3 = max segments in batch
gidx = seg_ids + batch_offset     [[0, 0, 1, 1, 2, 0]]
flat_gidx = gidx[valid]           [0, 0, 1, 1, 2]      one slot id per VALID frame
flat_feats = features[valid]      [[1,10],[3,30],[5,50],[7,70],[9,90]]   one row per valid frame
```

(the padded frame reads `0` only because `-1` was clamped; `[valid]` drops it immediately.)

`sums` is a scratch buffer with one row per slot, `zeros(B*T_new, C) = zeros(3, 2)`, and
`sums.index_add_(0, flat_gidx, flat_feats)` means *"for every i: `sums[flat_gidx[i]] += flat_feats[i]`"*:

```
sums[0] += [1,10] ;  sums[0] += [3,30]
sums[1] += [5,50] ;  sums[1] += [7,70]
sums[2] += [9,90]
-> sums   [[4,40], [12,120], [9,90]]
```

The same trick with a vector of ones gives segment lengths, `counts = [2, 2, 1]`. Divide:

```
sums / counts -> [[2,20], [6,60], [9,90]]
```

i.e. segment 0 = `mean(frame0, frame1) = [(1+3)/2, (10+30)/2] = [2, 20]`. Verified identical to
the real `mean_pool_segments`.

---

## 2. "But mean pooling is differentiable"

It is — **in `X`**. Pooling has two inputs:

```
S = meanpool( X , b )
              ^   ^
              |   segmentation  <- this is what the segmenter outputs
              wav2vec2 features
```

`dS/dX` exists and works (measured `[[0.5,0.5],[0.5,0.5],[0.5,0.5],[0.5,0.5],[1,1],[0,0]]` for
`S.sum()`), which is why CE already reaches `proj`. But the segmenter does not produce `X`; it
produces `b`. Training it with CE needs **`dS/db`**, and that is the one that does not exist:

```
b_2 = 1.00 -> S = [[2,20], [6,60], [9,90]]      (unchanged)
b_2 = 1.30 -> S = [[2,20], [6,60], [9,90]]      (unchanged)
b_2 = 1.49 -> S = [[2,20], [6,60], [9,90]]      (unchanged)
flip b_3 0->1 -> S = [[2,20], [5,50], [7,70], [9,90]]
                     ^ jumped AND grew from 3 segments to 4
```

`b` is binary, so `S` is **flat everywhere** in `b` and then jumps discontinuously — and the jump
changes the output **length**. The derivative is 0 almost everywhere and undefined at the jumps;
there is not even a fixed-dimensional function to differentiate. This is a property of the maths,
not a shortcoming of the implementation. Same shape as `y = x[i]`: differentiable in `x`, no
gradient to `i`. No rewrite fixes it, because "40% of a boundary" has no meaning until you
*define* one.

So pooling is differentiable, and still useless for training the segmenter.

Concretely, in the code the boundary reaches the computation only as `flat_gidx`, an `int64`
index with `requires_grad=False`. `index_add_` is differentiable w.r.t. the **values** it
scatters, never w.r.t. the **index**. Every step from `boundary` to `flat_gidx` destroys the
graph anyway: `== 1` gives a bool, `.long()` an integer, `cumsum` on integers stays integer.

**Measured:** passing `b_st = b_hard + p - p.detach()` (textbook STE) into `mean_pool_segments`
runs fine, gradient reaches `features` (abs-sum 4.2e-1), and `logits.grad` is `None`.

---

## 3. The fix: write pooling as `S = M @ X`

Same example, 5 valid frames, `b = [0,0,1,0,1]` -> segments `{0,1} {2,3} {4}`. Express membership
as a matrix `M` of shape `(T'=3 segments x T=5 frames)`:

```
M_hard =  [[0.5, 0.5, 0. , 0. , 0. ]      segment 0 = mean(frame0, frame1)
           [0. , 0. , 0.5, 0.5, 0. ]      segment 1 = mean(frame2, frame3)
           [0. , 0. , 0. , 0. , 1. ]]     segment 2 = frame4

X       =  [[1,10], [3,30], [5,50], [7,70], [9,90]]

M @ X   =  [[2,20], [6,60], [9,90]]       identical to mean_pool_segments
```

Row `j` reads "segment `j` is this weighted combination of frames"; the `1/n_j` entries do the
averaging.

**`M` holds exactly the same information as `flat_gidx` + `counts`** — which frame goes to which
segment, and the segment lengths. The only change is *where that information sits*: floats in the
**value** operand of a matmul instead of integers in the **index** operand of `index_add_`.
Autograd differentiates matmuls w.r.t. both operands. That is the whole reason for the rewrite.

### A soft `M`

Make `M` a function of `p = sigmoid(logits)` instead of `b`. With `p = [-, 0.1, 0.9, 0.2, 0.8]`
(whose hard threshold reproduces the segmentation above):

```
M_soft =  [[0.482, 0.433, 0.043, 0.035, 0.007]
           [0.   , 0.   , 0.510, 0.408, 0.082]
           [0.   , 0.   , 0.   , 0.   , 1.   ]]

M_hard =  [[0.5  , 0.5  , 0.   , 0.   , 0.   ]
           [0.   , 0.   , 0.5  , 0.5  , 0.   ]
           [0.   , 0.   , 0.   , 0.   , 1.   ]]
```

Each entry is `P(no boundary fired between segment j's start and frame t)`. The boundary at
frame 2 is only 90% certain, so there is a 10% chance segment 0 really extends to frame 2 —
hence the `0.043` leak. Blurred, but the same object, and **smooth in `p`**.

### The straight-through step

```
M_ste = M_hard + ( M_soft - M_soft.detach() )
```

- **Forward:** the bracket is numerically zero, so `M_ste` is bit-identical to `M_hard`
  (max diff exactly `0.0` on the toy; `1.2e-7` end-to-end vs `mean_pool_segments`, which is
  float32 accumulation-order roundoff between `bmm` and `index_add_`). The LLM sees the exact
  hard-pooled sequence — no approximation, no train/test mismatch.
- **Backward:** `M_hard` and `detach(M_soft)` are constants, so `dM_ste/dp = dM_soft/dp`.
  Gradient flows as though the soft matrix had been used.

```
dS/d(logits) = [0.0, -0.69, -3.22, -1.72, -2.45]     <- nonzero
```

That is the *estimator* in "straight-through estimator": you deliberately return the gradient of
a **different function** than the one you evaluated, because the evaluated one has no useful
gradient. It is biased by construction — which is exactly why it can fail, and why this needed
testing rather than just building.

---

## 4. Why Gumbel-ST specifically was a dead end

Gumbel-softmax ST is not a different mechanism from the above. For a binary variable it is
Gumbel-sigmoid: sample with noise, hard forward, soft backward. The "ST" half is *identical* to
§3; the Gumbel half only adds exploration noise to how the hard `b` is drawn.

So **the blocker was never the sampler — it was everything downstream of it.** Whether `b` comes
from an argmax or a Gumbel draw, it still enters a hard mean-pool that consumes it as an integer
index. Annealing the temperature `tau` would have changed nothing. The v1 plan's `tau` schedule
was solving the wrong problem.

Secondary point: Gumbel noise could in principle help jump the discrete barrier described in §5.
But stochastic exploration over discrete structure, with an unbiased gradient, is precisely what
GRPO already does — and GRPO additionally handles the non-differentiable CER/WER reward that is
the v2 headline.

---

## 5. Does the gradient point anywhere useful?

Two natural ways to build `M_soft`, both implemented and tested.

### Formulation A — relax the segment index (CIF-style)

Segment id is a running total, `c_t = cumsum(b)_t`; relax to `c_t = cumsum(p)_t`, then score
against slot `j` with a Gaussian kernel of width `sigma` (`sigma` is the analogue of Gumbel's `tau`).

**The indexing problem:** `d c_t / d p_s = 1 for every s <= t`. Nudging one frame's boundary
probability shifts the segment index of *every downstream frame*:

```
now:    [0..5][6......15][16.....25][26.....35] ...
want:   [0..9][10.....19][20.....29][30.....39] ...

raise p_10  ->  every frame after 10 gets index +1
                slot 1 now holds what slot 0 held, slot 2 what slot 1 held, ...
                the entire pooled sequence renumbers
```

The gradient scores that as catastrophic — slot `j` now contains something else entirely — even
though the resulting *sequence* is nearly identical. It can never say "hold everything fixed and
move this one boundary". The global renumbering term drowns the local placement signal.

### Formulation B — relax membership only (survival product)

Anchor each slot to its hard start `s_j` (detached, so nothing can renumber) and relax only
membership: `w[j,t] = prod_{u=s_j+1..t} (1 - p_u)`. Raising `p` inside a slot splits it; lowering
`p` at a slot's start lets the previous slot swallow it. Both local, no renumbering.

### Both fail the single-utterance test

Synthetic task: piecewise-constant signal, true change points every 10 frames, init boundaries
every one 4 frames early, target = the pooled sequence under the true boundaries. The answer is
one 4-frame shift away.

| formulation | mean boundary error | loss |
|---|---|---|
| A (index) | 4.00 -> 4.43 frames | 0.99 -> 2.17 (**rose**) |
| B (local) | 4.00 -> 4.43 frames | 1.10 -> 3.37 (**rose**) |

Original boundaries `[6,16,26,...]` never moved; junk boundaries appeared at the tail. One
`sigma` setting collapsed to zero boundaries; one Formulation-B config NaN'd.

**Why (structural, not a bug):** the improving move is a *coordinated discrete flip*. Going from
boundary-at-6 to boundary-at-10 requires deleting at 6 **and** inserting at 10 simultaneously.
Each single step alone is uphill — delete-6 alone makes slot 0 span a true change point; insert-10
alone gives 12 boundaries where the target has 11. Gradient descent follows single-coordinate
directions, so it sits in the barrier.

> Discarded en route: a segment-mean **reconstruction** loss is degenerate as a test —
> `dL/d(pooled)` is identically zero, since the segment mean is already its own least-squares fit.
> Any arm scores zero gradient on it. Do not reuse that objective.

### The amortised test changes the verdict

The above optimises free per-frame logits on one utterance — 11 independent coordinates that must
all flip together, the hardest possible case. The real system never does that: `logits = CNN(features)`,
one shared function trained over millions of frames. It does not need to solve any single
utterance's coordinated flip; it needs to learn a *statistical rule* ("boundaries belong at
acoustic change points"), with each utterance casting a noisy vote.

Real `CNNBoundaryPredictor`, 768 synthetic utterances, real `rate_loss`, 1500 steps, identical
data/init/LR across arms; evaluation is hard pooling everywhere:

| arm | content-loss | bnd-F1 (+/-1) | prec | rec | rho |
|---|---|---|---|---|---|
| `init` (biased cold start) | 3.080 | 0.228 | 0.376 | 0.163 | 0.034 |
| **`ste_index`** (Form. A) | **2.906** | **0.343** | 0.349 | 0.337 | 0.065 |
| **`ste_local`** (Form. B) | **2.893** | 0.297 | 0.398 | 0.237 | 0.043 |
| `grpo` (same reward, K=4) | 3.466 | 0.127 | 0.383 | 0.076 | 0.020 |
| `bce_true` (supervised ceiling) | 0.514 | 0.987 | 0.995 | 0.979 | 0.066 |

Both STE arms improve on `init` in loss *and* boundary-F1. **Signal, not solution:** they recover
~7% of the available content-loss headroom and ~15% of the F1 headroom.

### Soft forward beats hard STE

Adding fully-soft (CIF-style) forward arms to the *same* harness — the decoder consumes the
soft-pooled sequence during training, evaluation is hard pooling in every arm (job 47302):

| arm | content-loss | bnd-F1 (+/-1) | headroom recovered (loss / F1) |
|---|---|---|---|
| `init` | 3.080 | 0.228 | — |
| `ste_local` | 2.893 | 0.297 | 7% / 9% |
| `ste_index` | 2.906 | 0.343 | 7% / 15% |
| `soft_local` | 2.805 | 0.266 | 11% / 5% |
| **`soft_index`** | **2.682** | **0.502** | **16% / 36%** |
| `bce_true` | 0.514 | 0.987 | ceiling |

Removing the hard forward — i.e. removing the piecewise-constant landscape — roughly doubles what
the relaxation recovers. **If only one differentiable arm is run, it should be soft-index.**

### ...but on a realistic cold start, nothing works

The grid cold start above is a weak floor: its positions are not a function of local features, so
the CNN provably cannot fit it (rho collapsed to 0.034 vs the true ~0.083), and much of every
arm's "gain" is just recovery from a degenerate start.

Corrected setup (jobs 47302/47310): cold start = BCE on the true boundaries **shifted +3 frames** —
perfectly learnable from features, but systematically misaligned. The honest analogue of char-CTC
boundaries being sensible-but-not-decoder-optimal. Metric is mean |offset to nearest true boundary|,
which unlike a tolerance-F1 can see *partial* correction. `lambda_cap` raised 1 -> 20 after two arms
collapsed at 1.0.

| arm | content-loss | mean abs offset | F1@0 | F1@2 | F1@3 | rho |
|---|---|---|---|---|---|---|
| `init` | 1.948 | **3.34** | 0.000 | 0.110 | 0.912 | 0.063 |
| `ste_index` | 2.257 | 3.90 *(worse)* | 0.001 | 0.144 | 0.900 | 0.075 |
| `ste_local` | 3.668 | — *collapsed* | 0.000 | 0.000 | 0.000 | 0.008 |
| `soft_index` | **1.872** | 3.14 | 0.003 | 0.148 | 0.903 | 0.067 |
| `soft_local` | 3.668 | — *collapsed* | 0.000 | 0.000 | 0.000 | 0.008 |
| `grpo` | 1.821 | 3.25 | 0.000 | 0.121 | 0.917 | 0.070 |
| `bce_true` | 0.598 | **0.00** | 0.985 | 0.985 | 0.985 | 0.065 |

`init`'s mean offset of 3.34 and F1@3 of 0.912 confirm the setup is exactly as designed, and
`bce_true` reaching 0.00 confirms the task is solvable and the metric is sensitive. Against that:

- **No gradient-based arm corrects the misalignment.** Best is `soft_index` at 3.14 (-0.20 frames);
  GRPO manages -0.09; `ste_index` goes *backwards* to 3.90. Only direct supervision recovers it.
- **The survival-product formulation collapses reproducibly** — rho 0.008, F1 0.000, identical
  content-loss (3.6675) for both its hard and soft variants, and raising `lambda_cap` 1 -> 20 did
  not save it. This is the T'=1 trap: with a single segment the relaxation's gradient is *exactly*
  zero (measured), so collapse is an absorbing state and only the rate loss can escape — and in
  ratio space that penalty is `20*(0.05-0.008)^2 = 0.035`, negligible against a content loss of 3.7.
  **Formulation A (index) is the only robust one.**
- **Drop the earlier "STE beats GRPO" reading.** It was an artifact of the degenerate grid cold
  start, exactly as suspected. Here GRPO is competitive with the best differentiable arm on the
  content objective.

**Interpretation.** In a controlled setting where the correct answer is a uniform 3-frame shift and
supervision recovers it perfectly, no CE-shaped gradient — delivered via STE, soft pooling, *or*
GRPO — recovers it. The signal that reaches the segmenter through the decoder is weak for
*refining* an already-reasonable segmentation, regardless of the estimator used to deliver it.

---

## 6. Cost

Per step, the STE arm is **1 forward + 1 backward** and drops the K=4 no-grad reward rollouts
entirely. The extra backward through the LLM down to `inputs_embeds` is already paid today
(LoRA needs it). The `(B, T', T)` matrix is small at real shapes:

| B | T | T' | matrix | fwd+bwd (CPU) |
|---|---|---|---|---|
| 16 | 500 | 69 | 2.2 MB | 36 ms |
| 16 | 1000 | 131 | 8.4 MB | 162 ms |
| 8 | 1600 | 194 | 9.9 MB | 196 ms |

Negligible next to the 1.24B LLM forward. Extrapolating from the existing multi-seed runs
(`nll_mt` 3h35m/run, `cer_mt` ~11h10m/run for 4 joint epochs on train-clean-100), an STE arm
should land around **1.5-2 h/run** — the cheapest arm in the sweep. Gradient scale is bounded
(RMS 3.7e-4 at T=100 down to 1.8e-4 at T=1600, no cumsum blow-up), though clipping is still
advisable.

Formulation B's matrix is effectively banded, so a windowed implementation could cut memory
further if long utterances ever become a problem. Not needed for v1.

---

## 7. Options

**(a) STE with the rewritten pooling.** Working code exists; exact hard forward, live gradients,
cheapest arm. Fills the differentiable half of the bias/variance bake-off the v1 plan wanted.
Evidence says: expect a flat or negative result. Still publishable as "the differentiable arm does
not beat RL", which is a stated acceptable outcome in `dynamic_segmenter_rl_v1.md` §11.

**(b) Soft-forward (CIF-style).** No hard forward during training; the decoder consumes the
soft-pooled sequence, hard at eval, optionally annealed soft->hard. Removes the piecewise-constant
landscape. **Measured: ~2x better than hard STE**, and the only differentiable arm that improved
anything on the realistic cold start. This is what the 2026-08-02 note pointed at, and it was right.

**(c) Keep GRPO regardless.** It is the only arm that handles a non-differentiable reward, which
is the v2 CER/WER headline — and on the realistic setup it was competitive with the best
differentiable arm anyway.

### Recommendation, in priority order

**1. Run the missing control first — it is cheaper and worth more than any of the above.**

The finding that the CE-shaped signal is weak for boundary refinement *regardless of estimator*
raises a direct question about the existing results. `nll_mt` beats `nll_frozen` on every seed
(6.19 vs 7.18 test-clean), and the two differ by **decoder co-training during the joint phase** —
but both also run RL on the segmenter, and rho barely moves (0.288 -> 0.265). So the gain may be
mostly "four more epochs of decoder training", not "RL found better boundaries".

The control that separates them is **absent from the sweep**: decoder co-trained, **segmenter
frozen**, epochs 7-10, from the same shared epoch-6 checkpoint. No rollouts, no segmenter updates —
so it costs about a warm-up epoch, cheaper than every existing arm. If it matches `nll_mt`, the
learned-segmentation story needs rethinking before more estimator work is worthwhile. The report
already hints at this: `nll_mt`'s pooled gains are "primarily fewer deletions", which is a
decoder-side signature.

**2. Only then, if the 2x2 axis is still wanted:** implement **one** pooling module with a
`forward_mode: hard_ste | soft` flag so (a) and (b) share code. Use the **index** formulation
(Formulation A) for both — the survival-product form collapses reproducibly and should not ship.
Roughly 60-80 lines plus a yaml switch. Integration points, all localised:

- `compute_forward`: stop detaching `logits` on the decoder path; pass them to the new pooling.
- `_run_decoder`: swap `mean_pool_segments` for the new pooling (needs `logits` as an argument).
- `_obj_joint`: drop the GRPO pg term for the STE arm (CE now carries the gradient); keep
  `rate_loss` — it still supplies the "how many segments" signal that no relaxation can give,
  since neither gradient can create a segment that does not already exist (with only one segment,
  the STE gradient is exactly zero).
- yaml: `learn_method`, `forward_mode`, `sigma`.

It slots into the existing shared-warm-up multi-seed design as an additional variant, directly
comparable to `nll_frozen` / `nll_mt` / `cer_mt` at seeds 3407/3408/3409 off the same epoch-6
checkpoint.

---

## 8. Open items

- [x] Soft-forward (CIF) arms vs STE arms on the same harness — job 47302. Soft-index wins ~2x.
- [x] Corrected cold-start run (+3-frame misalignment) — jobs 47302 / 47310. No arm corrects it.
- [ ] **Run the decoder-co-trained / segmenter-frozen control** (epochs 7-10 from the shared
      epoch-6 checkpoint). Highest value per GPU-hour of anything here.
- [ ] Decide whether the 2x2 differentiable axis is still worth the compute given the above.
- [ ] If yes: implement Formulation A behind `learn_method` + `forward_mode` flags.

### Caveats on the evidence

These are **synthetic** probes: piecewise-constant signals, an MSE-to-target-pooled-sequence
objective, a 1.5k-step budget, one seed, and a small CNN. They are a screening test for whether to
spend A100-hours, not a substitute for a real run. In particular the target objective demands
*exact* recovery of the true pooled sequence, which is harsher than the LLM's CE. A real STE arm
could plausibly behave better than the toy suggests — but nothing here gives a reason to expect it
to, and three independent estimators failing the same controlled test is a consistent signal.

**Note on infra:** the first two SLURM submissions (47252 and an earlier `srun`) failed instantly
with no output because the scripts lived on the login node's local `/tmp`, which compute nodes
cannot see. Probe code now lives on weka at `/weka/scratch/.../users/cxiao/ste_probe/`.
