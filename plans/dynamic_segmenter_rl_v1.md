# Plan (v1): Learned Dynamic Downsampling via RL Boundary Segmentation

> Pilot to learn a **task-conditionable** segmentation policy that emits per-frame
> binary boundaries over frozen-SSL features; frames between boundaries are
> mean-pooled into one audio token; the shorter token sequence is projected into
> a (frozen, LoRA-adapted) LLM. The policy is trained with **GRPO**. Framed as an
> extension of **REBORN** (Tseng et al., NeurIPS 2024) from unsupervised ASR to a
> supervised speech-LLM (v1 optimizes NLL with RL *and* a differentiable Gumbel-ST arm; the
> verifiable WER/RLVR reward, where RL's edge is decisive, is the v2 headline).
>
> Status: DRAFT for review. Author: cxiao + Claude. Date: 2026-07-31.

## 0. Design choices — status board (living)

Legend: ✅ locked · 🟡 leaning · 🔴 open / under discussion. Update this table as
decisions land; §-refs point to the rationale.

| Axis | Status | Choice / leading option | Notes & alternatives |
|---|---|---|---|
| Boundary output | ✅ | per-frame binary boundary (1=segment start); mean-pool 0-runs | given |
| Policy backbone | ✅ | **CNN + Transformer**, contrastive (§4.1) | Conformer = v2 |
| Policy factorization | 🟡 | **A/B independent Bernoulli vs autoregressive Transformer decoder** | AR implementation is the highest-priority TODO; see `plans/autoregressive_transformer_boundary_policy.md` |
| Aggregator | ✅ | mean-pool, out_dim=768 (§4.2) | LSTM / attn-pool later |
| SSL encoder | ✅ | wav2vec2-base, **frozen** (cacheable) | WavLM later |
| LLM decoder | ✅ | Llama-3.2-1B + LoRA r16 | reuse exp-001 |
| RL algorithm | ✅ | **GRPO** (group baseline, K/utt) (§5) | REINFORCE rejected (§5.2); PPO-clip optional |
| gamma | ✅ | **1** (bandit) | delete discounted returns |
| pg reduction | ✅ | **`.mean()`** over valid frames | v1 used `.sum()` |
| Objective | ✅ | **teacher-forced NLL** only; WER→v2 (§6) | dense/cheap; WER needs decoding — deferred |
| Learning method | ✅ | **2×2 axis: GRPO-RL vs Gumbel-ST** (both on NLL) (§6,§8) | RL extends to WER in v2; Gumbel = diff. method, in v1 |
| Entropy schedule | ✅ | **annealed down** (β_H≈0.01→~0 RL; τ high→low Gumbel) | explore early, commit late |
| Anti-all-1's / rate reg. | ✅ | **one-sided cap at ρ* + mild token-tax** (rate-distortion) + entropy (§6.1) | rewards fewer tokens; Lagrangian = fixed-avg alt |
| Budget cap ρ* (ceiling) | 🟡 | cap ≈0.2 (≤5×); model may use fewer | a ceiling, not a pin; λ_press mild |
| Training scheme | 🟡 | **joint, NO separate pretrain**; in-run **decoder-warmup → joint** (§7) | fully-simultaneous riskier; REBORN-alternating fallback |
| Decoder init | ✅ | no separate base — decoder trained in-run (warmup) on the char-CTC cold-start segmentation | warmup byproduct = B1 alignment baseline |
| Cold-start (RL init) | ✅ | supervised BC; **target = char-level CTC bnd** (`wavlm_boundaries/char`) | finest level → RL learns to *merge/coarsen*; CTC-derived (like word_ctc) |
| KL-to-cold-start | ✅ | **dropped** — cold-start = init only, no leash | we WANT RL to leave the cold-start boundaries |
| ρ* curriculum | ✅ | **off** (v1) | hook exists; revisit only if training unstable |
| Decoder trains on | 🟡 | argmax segmentation | best-of-K / all-K augmentation |
| Task-conditioning | 🟡 | FiLM + per-task ρ*; hook only, ASR-only v1 (§4.3) | 2nd task (spk-count) later |
| Data scale | ✅ | train-clean-100 → 960h if it wins | |
| Baselines | ✅ | fixed-rate k-sweep + word_ctc oracle (§8) | Pareto frontier |
| CIF baseline | ⛔ | **deferred (not v1)** | Gumbel-ST already covers the differentiable comparison |

## 1. TL;DR

- **Goal**: does a *learned, content-adaptive* segmentation beat *uniform fixed-rate*
  pooling on the **WER-vs-token-count Pareto frontier** — lower WER at a given budget,
  and/or fewer tokens at equal WER (budget is a *ceiling* ρ*, model may use less). That is
  the whole scientific question of v1.
- **Method**: frozen SSL (wav2vec2-base, 50 Hz) → boundary policy → mean-pool →
  `proj` → LoRA-Llama-3.2-1B. The policy is trained by **GRPO** (K samples/utt,
  group-relative baseline) **jointly with the decoder** (proj+LoRA, supervised;
  encoder frozen) so the reward model never drifts off-distribution (§7). Boundaries are
  a **one-step joint action** (a bandit), so `gamma=1` and every frame gets the same
  utterance-level advantage.
- **v1 experiment grid = 2×2 = 4 runs**: backbone {**CNN**, **Transformer**} ×
  learning method {**GRPO-RL**, **Gumbel-ST**}, single objective **NLL** (WER→v2). All
  char-cold-started, all against the same
  frozen segmentation-robust reward model.
- **Anti-degeneracy** (the "all-1's maximizes reward but learns nothing" worry):
  three independent defenses — cold-start init at target rate, a **compression
  regularizer** on the kept-ratio, and an entropy bonus.
- **Compute**: the policy is ~1–8 M params and ≤~2–6 % of the LLM's audio-prefix
  forward — negligible; the cost driver is the RL arm's K-sample LLM forwards (the
  Gumbel-ST arm is cheaper — no K). Whole pilot fits in ~1–2 wall-clock days on the JSALT
  reservation.

## 2. Background — what is already established

- **The stack** (`train_speechllm.py`, `speechllm_fixed_pooling.yaml`): frozen SSL
  → boundary-driven mean-pool (`segment_pooling.py`) → `proj` (VanillaNN) → frozen
  Llama-3.2-1B + LoRA r16; CE on text tokens. `exp-001` (960 h, word_ctc alignment
  boundaries) = WER 6.4/9.8 dev-clean/other — a sane reference point.
- **A v1 segmenter already exists but was never validated** — `origin/segmenter`
  branch (single commit "learnable segmenter: v1; testing TBD"): `segmenter.py`
  (`BoundaryPredictor` CNN + `Segmenter`), `train_speechllm_with_segmenter.py`
  (cold-start + RL modes), `hparams/speechllm_segmenter.yaml`. It uses vanilla
  batch-REINFORCE with a discounted return. We **port and rework** it (§10), not
  restart. Its boundary/pool conventions already match blank_analysis's
  `segment_pooling.py` (`1`=boundary, `0`=continuation, `-1`=pad; frame-0 guard;
  `build_aggregator`, `mean_pool_segments`, etc.).
- **Positioning** (literature):
  - **REBORN** (arXiv 2402.03988) — 1-D CNN boundary policy + REINFORCE, behavior-
    cloning cold-start, *alternating* segmenter/downstream training, for *unsupervised*
    ASR. Our direct antecedent; we extend it to a supervised speech-LLM and a real
    reward. RL alone moved PER 19.3→12.9 there — evidence learned boundaries beat
    fixed ones.
  - **CIF** (arXiv 1905.11235) and **LLM-ASR dynamic downsampling with CIF**
    (arXiv 2606.10439) — the *differentiable* incumbent for this setting (deferred as a v1
    method; **Gumbel-ST** stands in as the differentiable arm). v1 optimizes **NLL** with
    *both* GRPO-RL and Gumbel-ST — a bias/variance bake-off on identical objective +
    architecture. RL's decisive edge (optimizing a **non-differentiable WER/RLVR reward**,
    which Gumbel/CIF cannot backprop through) is the **v2** headline, deferred here because
    WER needs per-sample decoding (costly, unreliable while the decoder trains).
  - **RLOO / GRPO** (arXiv 2402.14740; Shao et al. 2024) — the group-relative baseline
    we adopt; for a one-step bandit RLOO and GRPO coincide up to std-normalization.

## 3. Problem framing

### 3.1 It is a bandit, not a sequential MDP
The encoder features are fixed before any boundary is chosen; all per-frame boundary
decisions are sampled *simultaneously and independently* and jointly produce one
segmentation that earns one utterance-level reward. There is no state transition
between frames. Consequences, baked into the design:
- **`gamma = 1`.** Every frame's log-prob receives the *same* utterance-level
  advantage. (The v1 `discounted_returns` with `gamma=0.99` gives frame 0 of a 10 s
  utterance ~0.7 % of the credit — a latent bug; we delete it.)
- **Group baselines are cheap and natural**: sampling K segmentations of one utterance
  is a K-fold repeat of a forward, no rollout.

### 3.2 Degenerate optima we must design against
- **All-1's** (every frame a boundary → mean-pool is identity → no compression):
  maximizes information to the LLM, trivially good quality, *defeats the purpose*.
- **All-0's** (one segment → one token): maximal compression, destroys quality.
Defenses (all three, belt-and-suspenders): (1) **cold-start** initializes the policy
near the target rate; (2) a **compression regularizer** makes reward non-monotonic in
the boundary count; (3) an **entropy bonus** keeps the policy stochastic so it can
escape either collapsed corner. We monitor kept-ratio ρ and policy entropy every
epoch as the primary degeneracy alarms.

### 3.3 The comparison that defines success
Plot **WER vs average kept-token ratio ρ**. The learned segmenter wins only if it
sits *below-left* of the fixed-rate curve `{k=3,4,5,6,8}` at its operating ρ (≤ cap ρ*), with the
word_ctc alignment point as an "oracle-boundary" upper reference. Secondary: does the
learned segmenter *rediscover* word boundaries (boundary-F1 vs word_ctc)?

### 3.4 Initialization bias (cold-start)
The cold-start sets the RL starting basin, and local optimization tends to stay near it.
Two components: **rate bias** toward the init's ρ is *desired* (we pin ρ anyway), but
**placement bias** toward the init's *pattern* is a real risk — cold-starting from a
**uniform** grid (fixed-rate k) can trap the policy at "basically uniform," so RL buys
nothing over the fixed-rate baseline. Mitigations: (i) **cold-start from char-level CTC
boundaries** (non-uniform, content-correlated, and the *finest* level — so RL explores by
*merging/coarsening* toward the budget, a gentler problem than splitting); (ii)
entropy bonus for exploration; (iii) **no KL leash to the init** (cold-start is init only —
we *want* the policy to leave the cold-start boundaries); (iv) **verify the policy moves** — track boundary-F1 drift and per-frame entropy
vs the init; if it doesn't move, RL is inert. The init biases the basin, not a hard
constraint: the gradient *will* leave it if non-uniform placements score higher; the risk
is slow escape / local optima, which (i)+(ii) address.

## 4. Architecture

```
wav (16k) → [frozen] SSL wav2vec2-base → feats (B,T,768) @50Hz
          → BoundaryPolicy(feats, task_id?) → per-frame logits (B,T,2)
              ├ train: sample K boundaries ~ Bernoulli(σ(logit))   (GRPO)
              └ eval : argmax
          → mean_pool_segments(feats, boundary) → (B,T',768)
          → proj (VanillaNN 768→2048) → audio tokens
          → [frozen, LoRA] Llama-3.2-1B :  <soa> audio <eoa> prompt <bos> text <eos>
          → reward = −NLL (RL arm) ／ NLL backprops through relaxed boundary (Gumbel arm)
```

### 4.1 Boundary policy — two contrastive backbones (config `segmenter_backbone`)
Both emit per-frame 2-class logits; the **action factorization is identical**
(independent per-frame Bernoulli), so the backbone changes only *logit quality*, not
RL variance.
- **`cnn`** (v1 baseline): `conv 768→H k7 → H→H k3 → H→2 k1`, `H=256`
  (down from v1's 512; ~1.0 M params). Receptive field 9 frames ≈ 180 ms — enough
  for *local* boundary cues, blind to global speaking-rate.
- **`transformer`**: input proj 768→d, **2–4 pre-norm Transformer-encoder layers**
  (d=256, 4 heads, FFN 1024, rel/rotary pos), linear head →2. ~2–8 M params. Rationale
  (your locality worry): lets a frame attend back to the previous boundary
  (segment-length / rate awareness) and model boundary interactions ("don't cluster").
  Note the SSL input is *already* globally contextualized, so we expect the gain to be
  mostly in **controllable rate** and future task-conditioning, not raw local cues —
  which is exactly why we measure both. (Conformer = conv+attn is the natural v2 middle
  rung; keep the switch open.)

### 4.2 Aggregator
Mean-pool (parameter-free, `out_dim = 768`) for v1 so `proj` input stays 768 and the
existing `proj`/decoder contract (and the fixed-rate baselines) are unchanged.
(`RecurrentAggregator` exists but changes the downstream contract — defer.)

### 4.3 Task-conditioning hook (built, not exercised in v1)
Add an **optional** `task_id` → small embedding table → injected as **FiLM**
(per-task γ,β on the backbone's hidden activations) plus a **per-task target ratio**
`ρ*[task]`. In v1 there is one task (ASR): the hook is a no-op bias, but the interface
is in place so speaker-counting (a much coarser ρ*) drops in later without an arch
change. This satisfies "design for task-conditioning from day 1" without adding
pilot scope.

### 4.4 Parameter / compute budget (answers "is the segmenter worth it?")
Per 10 s utterance (~500 frames, T′≈100 tokens at ρ*=0.2):

| Component | Params | Fwd FLOP/utt | vs LLM audio-prefix |
|---|---|---|---|
| SSL (wav2vec2-base, frozen) | 95 M | ~95 GFLOP | (fixed, shared) |
| **Policy CNN (H=256)** | **~1.0 M** | **~1 GFLOP** | **~0.4 %** |
| **Policy Transformer (4×d256)** | **~5 M** | **~2–6 GFLOP** | **~1–2 %** |
| Llama-3.2-1B on audio prefix | 1.24 B | ~250 GFLOP | 100 % |

The policy's *marginal* cost over free fixed-rate pooling is ≤~6 GFLOP — recovered the
instant it saves one audio token on average, and shorter prefixes also shrink the
autoregressive decode + KV cache. The "no point if the segmenter costs more than the
decoder" bar is met with ~40–250× headroom. It would only bind if the policy were a
*large* transformer over the full sequence — which is why we cap it small.

## 5. RL algorithm — GRPO (and why not REINFORCE)

### 5.1 Estimator
For utterance *u* draw **K segmentations** `b^{(1..K)} ~ π_θ(·|feats_u[, task])`, get
rewards `R^{(k)}` (§6). Group-relative advantage:

```
Â^(k) = (R^(k) − mean_k R) / (std_k R + ε)          # GRPO; drop the /std → RLOO
loss  = − (1/ΣK) Σ_{u,k} Σ_t m_{u,t} · Â^(k) · logπ_θ(b^{(k)}_{u,t})
        − β_H · Ĥ(π)                                  # entropy bonus
        (no KL-to-cold-start in v1: cold-start is init only, policy free to leave it)
```
`m` masks padding; the per-frame log-prob sum uses **gamma=1** (same Â to every frame);
the loss is a **`.mean()`** over valid frames (not v1's `.sum()`, which made the
effective LR track dynamic-batch token count). Optional PPO-style ratio clip (ε=0.2)
enables >1 grad step per K-sample batch (sample-efficiency, since sampling *is* the
LLM-forward cost); off by default in v1.

### 5.2 Why vanilla REINFORCE is a poor fit here (recorded per request)
The reward variance is dominated by **between-utterance difficulty**, not by the
**between-segmentation quality** we want to learn. A baseline averaged over *different*
utterances (batch-mean REINFORCE) has the wrong conditioning, so the advantage sign is
frequently set by *which* utterances happened to share the batch. Worked example
(batch of 2, one sample each): easy utt A (R≈−0.5) sampled its *bad* seg (−0.55), hard
utt B (R≈−2.0) sampled its *good* seg (−1.80); batch mean −1.175 ⇒ A's bad seg gets
**+0.63** (pushed toward it) and B's good seg gets **−0.63** (pushed away) — both
signals backwards, purely from the difficulty gap (1.35) swamping the quality gap
(0.1). Plus an O(1/B) bias from the batch mean including its own sample. GRPO's
same-utterance group baseline makes difficulty **cancel exactly**, isolating "is this
segmentation better than typical segmentations *of this utterance*." Because the
problem is a one-step bandit, the K extra samples are just K forwards — cheap.

### 5.3 Hyperparameters (starting points, to be tuned on a smoke run)
Keep `K=4` fixed for the current study, with `gamma=1` and
`entropy_coeff β_H≈0.01` **annealed →~0** over the
joint phase (no KL-to-cold-start), `lr=1e-4` AdamW, `max_grad_norm=1.0`, cold-start 5 ep,
RL 3–5 ep, data **train-clean-100** first (REBORN-scale; scale to 960 h only if v1 wins).

## 6. Objective (NLL) and the two learning methods

**One objective: teacher-forced NLL.** `quality = −(1/|y|) Σ_t CE(LLM_logit_t, y_t)` — mean
per-token negative log-likelihood of the reference under the (jointly-trained) decoder; one
forward, **no decoding**. Dense, low-variance, cheap. Full segmenter objective =
`−quality + L_rate + (−β_H·entropy)` (rate term in §6.1).

**Two learning methods for the boundary policy — the v1 2×2 axis (× backbone):**
- **GRPO-RL** (§5): sample K segmentations, group-relative advantage on the *detached*
  reward `−NLL`. Unbiased, higher-variance; the machinery that extends to
  non-differentiable rewards (WER) in v2.
- **Gumbel-softmax straight-through**: the *same* binary-boundary policy, but the NLL is
  backpropped through a relaxed (hard-forward / soft-backward) boundary straight into the
  policy — fully differentiable, biased, lower-variance, cheaper (no K-sampling, no
  detached-reward step). Its exploration knob is the Gumbel **temperature τ**, annealed
  high→low (the analog of the RL entropy anneal).

Both optimize the *same* NLL under the *same* rate objective (§6.1) and pipeline (char
cold-start → warmup → joint), so it's a clean **bias/variance bake-off** on identical
objective + architecture: does RL's unbiased-but-noisy gradient beat Gumbel-ST's
biased-but-cheap one? (Honest expectation: on NLL alone they may be close, or Gumbel may
edge it — RL's real win needs the non-differentiable WER reward, which is v2.)

**WER / RLVR reward — deferred to v2.** It needs *decoding* each sampled segmentation
(expensive, and unreliable while the decoder is still training) → too much complexity for
v1. It is exactly where RL beats Gumbel-ST (you can't backprop through WER), so it's the
natural v2 headline.

### 6.1 Anti-collapse / rate regularizers (the all-1's fix)

**Notation.** `T` = #frames in an utterance (50 Hz; 10 s → 500). `p_t = σ(logit_t) ∈ [0,1]`
= policy's *probability* frame `t` is a boundary (differentiable). `b_t ∈ {0,1}` = the
*sampled* decision (non-differentiable coin flip). #tokens ≈ #boundaries = `Σ_t b_t`; its
expectation `E[Σ_t b_t] = Σ_t p_t` **is** differentiable — the whole trick. `ρ = #tokens/T`
= compression ratio (1 = no compression / all-1's; 0.2 = 5×). `ρ*` = *target* ratio (a
chosen dial, e.g. 0.2); `N* = ρ*·T` = *target* #tokens (per-utterance; scales with length).

**`ρ*` is a dial, not a contradiction with "dynamic".** *Something* must set the operating
point on the quality-vs-compression curve (unconstrained, the model always wants more
tokens → all-1's), so we fix one knob — a target rate `ρ*` (or, v2, a per-token price `λ`).
Everything content-adaptive stays **learned**: (a) **placement** *within* an utterance (RL
decides *where* the `N*` boundaries go — long segments over silence, short over dense
speech — vs fixed-rate's uniform grid); (b) **per-utterance rate**, because the quantity
rate is set by a *soft tax-vs-quality balance under a ceiling* (§6.1 recipe), not a hard
per-utterance quota — each utterance's count floats (hard utt spends more, up to the cap;
easy utt fewer), so the rate is genuinely dynamic and `ρ*` is a *ceiling/typical* level;
(c) **per-task rate** via `ρ*[task]`.

**Framing.** All-1's is a *rate* failure, best controlled **differentiably on the expected
count `Σ_t p_t`** — split the jobs: a differentiable auxiliary loss sets *how many* tokens;
the RL reward sets *where*. This keeps rate control out of the high-variance RL budget and
kills all-1's at its mechanism (`logit→+∞` ⇒ `Σp = T ≫ N*` ⇒ large penalty). Candidates
(by how rigidly they pin the rate):

1. **Quantity loss (CIF-style)** — `L_q = (Σ_t p_t − N*)²`. Soft penalty centering the
   count at `N*` [arXiv:1905.11235]. Two-sided — penalizes going *under* ρ* too; v1 uses a
   **one-sided cap** variant instead (see recipe below).
2. **Ponder / expected-L0** — `|Σ_t p_t − N*|`; ACT ponder-cost / hard-concrete L0
   [Graves 2016; Louizos et al. 2018, arXiv:1712.01312]. (Sigmoid head already gives the
   expected count; hard-concrete only for exact zeros.) Dropped.
3. **Rate-KL** — `KL(Bern(ρ̄)‖Bern(ρ*))`, ρ̄=mean_t p_t. Dropped.
4. **Lagrangian / constrained** — pin only the *dataset-average* `E[ρ]=ρ*` via a PI-tuned
   dual variable [arXiv:2406.04558]; individual utterances fully free (max per-utterance
   dynamism), average exact, no `λ_q` hand-tuning. **Upgrade knob (same run)** if the soft
   average drifts or we want maximal dynamism.
5. **Entropy bonus** (§5): keeps `p_t` off the 0/1 rails — necessary anti-saturation.
   **Annealed down** (`β_H≈0.01→~0`; Gumbel arm: temperature `τ` high→low) so training
   explores early and commits to confident boundaries late.
6. **RL compression reward** on the *sampled* ρ: high-variance, needs a λ sweep. Dropped.

**v1 rate objective — one-sided cap + mild token tax (rate-distortion).** A two-sided
`(Σp−N*)²` *pins* ρ at ρ*, but there's no reason to force *more* tokens when the model
already does well with fewer. So make it asymmetric:

    L_rate = λ_cap · max(0, Σ_t p_t − N*)²      # hard: never exceed budget N* (=ρ*·T)
           + λ_press · (Σ_t p_t / T)            # gentle: always prefer fewer tokens

\+ always-on entropy (5). Quality pulls ρ *up*, the token tax pulls *down*, the cap forbids
ρ>ρ*; per-utterance equilibrium = where marginal-quality-per-token = `λ_press`, capped at
ρ* ⇒ **easy utts settle BELOW ρ*, hard utts ride the cap** (the "use fewer when you can"
behavior). Note the cap *alone* leaves ρ at ρ* (quality mildly prefers more), so the mild
tax is what produces sub-ρ* usage — keep `λ_press` small (quality dominates when tokens
truly help; dev-WER is the arbiter). This is the price / rate-distortion form (once flagged
as v2), pulled into v1. Drop (2)/(3)/(6); (4) Lagrangian = alternative if we instead want a
*fixed* dataset-average ρ. Report the **whole per-utterance ρ distribution** + entropy each
epoch (spread = dynamism; ρ→1 ⇒ cap/tax too weak, ρ→0 ⇒ tax too strong / quality ignored).

## 7. Training scheme: joint segmenter + decoder, encoder frozen

> **Status (user, 2026-07-31): no separate pretrained ASR — one joint run with an in-run
> decoder-warmup front phase.** Leading scheme; REBORN-alternating kept as a fallback.

**The bootstrapping tension.** A decoder that has never learned ASR gives no usable signal
— everything decodes to garbage and every segmentation scores equally badly, so the policy
gradient is noise. Two ways out: pre-train a separate base, or **train the decoder inside
the joint run**. Also: a decoder trained at its *native* rate (no downsampling) learns to
want *all* frames → reward would push the policy to **all-1's**; so whatever we do, the
decoder's "familiar rate" must be the *compressed* target and the rate guardrail (§6.1)
must be on.

**Resolution: one joint run, warmup first (no separate pretrained ASR).**
1. **Decoder-warmup (~1 epoch)** — freeze the segmenter at its char-level CTC cold-start
   (deterministic boundaries) and train *only* the decoder (proj+LoRA, CE). The decoder
   learns ASR on a *fixed, compressed, sensible* segmentation — exactly the exp-001
   alignment-pooling setup, known to work — so by the time RL starts the reward is
   meaningful. This byproduct decoder **is the B1 char-CTC-alignment baseline** (free).
2. **Joint** — unfreeze the segmenter; train segmenter (GRPO on K samples) + decoder (CE on
   the argmax segmentation) together, encoder frozen. The decoder keeps adapting to the
   segmenter, so the reward model stays **on-distribution** — no separate base, no stale
   rate. This folds the old "Phase-0 base" into the front of the run (§8).

*Fully-simultaneous joint-from-scratch (no warmup) is viable but riskier: the early RL
signal is uninformative (untrained decoder → flat reward) and can nudge the segmenter off
its cold-start on noise. If going that route: low/ramped segmenter LR and freeze the
segmenter until inter-sample reward variance clears a threshold. The
~1-epoch warmup removes this risk for negligible cost, so it is the default.*

**Precise mechanics (one training step, joint phase).** SSL frozen (features deterministic
→ cacheable). Init: decoder = the warmup-trained decoder (step 1); segmenter =
**cold-started to char-level CTC boundaries** (`wavlm_boundaries/char`) — a *non-uniform*
init reduces the placement bias toward the uniform grid (§3.4). The warmup trains the
decoder on the same cold-start segmentation, so rates are matched by construction. (Char is
the finest level, ρ≈0.3 > a ~0.2 cap, so RL coarsens down to budget — see §3.4.)
- **Decoder path** — pool with the **argmax (deterministic)** boundaries, `proj` + LLM
  *with grad*, CE on text → gradient to proj+LoRA only. (Argmax = the segmentation we
  actually use at eval; keeps the decoder's target clean rather than chasing exploratory
  samples. Alternative: best-of-K or all-K as built-in augmentation.)
- **Segmenter path** — draw **K sampled** boundaries; each pooled+proj+LLM forward runs
  *under `no_grad`* to produce a **detached** reward (−NLL); the policy log-prob
  carries the grad. GRPO group-baseline over the K samples (§5).
- One combined backward `loss = CE_decoder + grpo_loss + L_rate` (disjoint parameter
  dependencies; proj+LoRA get grad from CE, segmenter from GRPO+L_rate). Cost = 1 grad LLM
  forward (like exp-001) + K `no_grad` forwards. Separate param groups / LRs for decoder vs
  segmenter.
- **Gumbel-ST arm (2nd learning method):** replace the segmenter path — instead of K
  sampled rewards + GRPO, sample **one** boundary via Gumbel-softmax straight-through and
  backprop the decoder's NLL *through* the relaxed boundary into the policy (fully
  differentiable, no `no_grad`, no K). Everything else (char cold-start, warmup, `L_rate`,
  annealed τ, frozen encoder) is identical — that's what makes it a clean bake-off.

**The new caveat.** Joint training removes the off-distribution problem but makes the
**compression regularizer the *sole* guardrail against all-1's** (the co-adapting decoder
is always happy to accept more tokens, so quality reward alone still favors ρ→1). So the
**quantity-loss weight (§6.1)** carries the load — treat **ρ and entropy as first-class alarms**;
a **ρ\* curriculum** (anneal 1→ρ*) is the **off-by-default** (§0) fallback if aggressive
compression from step one proves unstable. **Held-out dev WER is the arbiter** against co-adaptation that
games the train reward.

**Fallbacks (documented, not v1 default):** (a) **frozen + segmentation-robust base**
(base trained at `k∼U{3..8}` / boundary-jitter — cheaper, fully parallel, no decoder
backward; for fast iteration or if joint is unstable); (b) **REBORN-style alternating**
(RL segmenter → retrain decoder → repeat — same effect, staged). GRPO's *within-batch*
group baseline is computed against the *current* decoder, so it tolerates the moving
reward model — which is why simultaneous joint is our default over alternating.

## 8. Experiment matrix & phases

**Baselines / prerequisites**
- **B0 fixed-rate sweep** `k∈{3,4,5,6,8}` (reuse `train_speechllm.py`,
  `boundary_source=fixed_rate`) on train-clean-100 → the WER-vs-ρ frontier (x-axis).
- **B1 char-CTC-alignment oracle** = the **decoder-warmup byproduct** (§7 step 1): the
  decoder trained on the fixed cold-start boundaries, evaluated with them. Free — no
  separate run. Upper reference. (exp-001's `word_ctc` @960 h is a loose external reference.)

**Phase 1 — segmenter cold-start (supervised BC, no LLM)**: BCE of boundary logits vs
**char-level CTC** boundary targets (`wavlm_boundaries/char`), for **CNN** and
**Transformer**. Gate: boundary-F1 high (the policy reproduces the char boundaries before
RL perturbs/coarsens them).

**Phase 2 — joint run (the 2×2, 4 parallel runs)**, each run = **(2a) decoder-warmup**
(segmenter frozen at cold-start, ~1 ep, decoder CE — produces B1) **→ (2b) joint GRPO RL +
decoder** (segmenter GRPO on K samples, decoder CE on the argmax segmentation), encoder
frozen (§7):

| run | backbone | learning method | objective |
|---|---|---|---|
| CNN-RL | cnn | GRPO-RL | NLL |
| CNN-Gumbel | cnn | Gumbel-ST | NLL |
| TF-RL | transformer | GRPO-RL | NLL |
| TF-Gumbel | transformer | Gumbel-ST | NLL |

Gate per run: NLL ↓, ρ under the cap / sensible per-utt spread (no degeneracy), and dev WER
≤ the warmup (B1) WER.

**Phase 3 — analysis**: WER-vs-ρ Pareto (all runs + B0 + B1); boundary-F1 vs word_ctc;
qualitative boundary plots; degeneracy trace (ρ distribution, entropy vs step);
**RL-vs-Gumbel** and **CNN-vs-Transformer** comparisons. **Success = any run below-left of the
fixed-rate frontier (lower WER at its operating ρ, and/or fewer tokens at equal WER).**

## 9. Compute & schedule

**Unit**: exp-001 = 960 h/1 ep ≈ 5 h54 on 1 A100 ⇒ train-clean-100 ≈ **~37 min/ep** for a
single-forward LoRA run. Joint RL keeps the LoRA backward (1 grad forward, like exp-001)
and adds K `no_grad` forwards for the NLL reward; the Gumbel-ST arm has no K (1 fwd+bwd).
SSL is frozen ⇒
**cache SSL feats once**
(`extract_ssl_feats.py`, `use_feats=True`) for train-clean-100 + dev/test (~20–45 GB
HDF5) and share across *all* runs — removes the SSL forward everywhere.

| Item | GPUs | est. wall-clock | note |
|---|---|---|---|
| SSL feature cache | 1 | ~1–2 h | once; reused by every run |
| B0 fixed-rate ×5 (baselines) | 1–6 | ~3–4 h | parallel; each ≈ exp-001/10 |
| Phase 1 cold-start ×2 (cnn, tf) | 2 | ~1–3 h each | no LLM; cheap |
| RL ×2 (cnn, tf) | 2 | ~5–7 h each | incl. ~1 ep warmup; then 1 grad + K=4 no_grad fwd; ~2.5× base/step |
| Gumbel ×2 (cnn, tf) | 2 | ~3–5 h each | incl. warmup; 1 grad fwd+bwd (no K-sampling); ~1.3× base/step |

All estimates **to be validated by a 2-batch `--debug` smoke test + a 200-step profiling
run before committing multi-hour jobs** (per house practice). With the JSALT reservation
(8×A100 nodes), the four Phase-2 runs go truly in parallel on one node; **end-to-end
pilot ≈ 1–2 wall-clock days**, well inside the 3-day walltime per job. SLURM: always
`--account=jsalt2026-lgarci27 --comment=accept_cost --partition=a100
--reservation="JSALT 2026" --exclude=ga129`; identity flags on the `sbatch` command
line, `#SBATCH` for job-shape only (as in `launch_fixed_pooling_ctc.sh`).

## 10. Implementation work items

1. **Port** `segmenter.py`, `train_speechllm_with_segmenter.py`, `speechllm_segmenter.yaml`
   from `origin/segmenter` onto `blank_analysis`; confirm imports resolve against the
   current `segment_pooling.py` (they do: `build_aggregator`,
   `lengths_to_padding_mask`, `mean_pool_segments`, `padding_mask_to_lengths`,
   `fixed_rate_boundary_targets`). Fix yaml paths → Llama-3.2-1B + shared HF cache + real
   LibriSpeech / boundary dirs (mirror `speechllm_fixed_pooling.yaml`).
2. **`segmenter.py`**: (a) delete `discounted_returns`; make returns = advantage broadcast
   to valid frames (`gamma=1`). (b) `reinforce_loss` → GRPO loss, **`.mean()`** over valid
   frames, group-relative advantage + optional std-norm + optional ratio-clip (no
   KL-to-cold-start). (c) add `TransformerBoundaryPredictor` and a `segmenter_backbone` switch.
   (d) add optional `task_id`→FiLM hook (no-op default). (e) add the **rate objective**:
   one-sided cap `λ_cap·max(0,Σσ(logit)−N*)²` + token-tax `λ_press·(Σσ(logit)/T)` +
   **annealed entropy** (differentiable, on the logits); keep `boundary_prf`,
   `boundary_ce_loss`. (f) add a **Gumbel-ST head option** (hard-forward/soft-backward
   sample + annealed temperature `τ`).
3. **`train_speechllm_with_segmenter.py`**: (a) **warmup sub-phase** — segmenter frozen at
   the cold-start, decoder CE only (no sampling), ~1 ep, then switch to joint. (b) joint
   step, **two methods via `learn_method` flag**: **GRPO-RL** — decoder CE on the **argmax**
   segmentation (grad → proj+LoRA) + **K sampled** segmentations under `no_grad` for detached
   `−NLL` rewards + GRPO group baseline (grad → segmenter); vs **Gumbel-ST** — one relaxed
   sample, decoder NLL backprops *through* it into the policy (no K, no detached reward).
   One combined backward `CE + (grpo|gumbel) + L_rate`; separate param groups/LRs. Keep
   `cold-start` (BC) mode. Log sampled+expected ρ distribution, entropy, loss components;
   `ρ*` curriculum hook (off by default). **No WER/decoding path in v1.**
4. **(removed — no separate pretrained base)**; the decoder-warmup in (3a) replaces it and
   yields the B1 alignment baseline.
5. **YAML configs**: one config + CLI overrides for `segmenter_backbone` × `learn_method`
   (the 2×2), plus the GRPO block (K, clip, annealed entropy; no KL) / Gumbel block (annealed
   `τ`), `λ_cap`/`λ_press` + ρ*, warmup length. B0 fixed-rate baselines reuse
   `speechllm_fixed_pooling.yaml`.
6. **Launch scripts**: `sbatch` wrappers for B0 sweep, cold-start ×2, joint ×4 (RL/Gumbel × cnn/tf).
7. **Smoke tests**: extend the `__main__` self-tests in `segmenter.py` (GRPO shapes,
   gamma=1 credit, degeneracy guards); `--debug` end-to-end on dev-clean for each of the
   4 configs before full runs.

## 11. Risks & mitigations

- **Degeneracy (all-1's / all-0's)** → three defenses (§3.2) + ρ/entropy alarms; if it
  still collapses, strengthen the regularizer or switch to a hard token cap.
- **Reward-model off-distribution / hacking** → largely handled by construction (joint
  training keeps the decoder on-distribution, §7); residual co-adaptation hacking caught by
  the held-out **dev-WER arbiter** (train NLL ↓ but dev WER not ↓ ⇒ hacking).
- **Learned underperforms fixed-rate** → a *valid, publishable* result; the RL-vs-Gumbel and
  CNN-vs-TF axes localize *why*, and the fixed-rate + char-CTC-oracle points bound it.
- **RL shows no edge over Gumbel-ST** → *expected* on the NLL objective (bias/variance
  wash) — not a failure. RL's payoff is the **v2 WER/RLVR reward** Gumbel can't backprop
  through; v1 validates the RL machinery + the learned-vs-fixed question.
- **Gumbel-ST instability** (hard-forward/soft-backward bias) → slower τ anneal / higher τ
  floor; it's the biased-gradient cost we're explicitly measuring against RL.
- **Port drift** (segmenter branch is 7 commits behind main) → port files, don't merge
  the branch; re-run `python segment_pooling.py` / `segmenter.py` self-tests after port.

## 12. References
- REBORN — Reinforcement-Learned Boundary Segmentation for Unsupervised ASR, NeurIPS 2024. arXiv:2402.03988
- CIF — Continuous Integrate-and-Fire, ICASSP 2020. arXiv:1905.11235
- LLM-based ASR with MoE + CIF dynamic downsampling, 2026. arXiv:2606.10439
- Back to Basics: Revisiting REINFORCE (RLOO), 2024. arXiv:2402.14740
- GRPO — DeepSeekMath, Shao et al. 2024. arXiv:2402.03300
- (context) `docs/project_notes/key_facts.md` exp-001; `origin/segmenter` v1.

## 13. Decision log
- 2026-07-31 — Loss = **GRPO** (group baseline, gamma=1, pg `.mean()`); vanilla REINFORCE
  rejected (difficulty-confounded baseline, §5.2).
- 2026-07-31 — v1 backbones = **CNN + Transformer**, contrastive; mean-pool aggregator.
- 2026-07-31 — v1 objective = **NLL only**; 2×2 grid axis flipped to **backbone {CNN,TF} ×
  learning method {GRPO-RL, Gumbel-ST}** = 4 runs (user). **WER/RLVR deferred to v2** (needs
  per-sample decoding — costly + unreliable while decoder trains). Gumbel-ST promoted from
  deferred baseline to a v1 arm ⇒ clean RL-vs-differentiable bake-off on identical
  objective+arch (honest: RL may not beat Gumbel on NLL; RL's edge is the v2 WER reward).
- 2026-07-31 — **Entropy annealed** (β_H≈0.01→~0; Gumbel τ high→low): explore early, commit
  late (user requirement).
- 2026-07-31 — Rate objective (revised, user) = **one-sided budget CAP `λ_cap·max(0,Σσ(logit)−N*)²`
  + mild token-tax `λ_press·(Σσ(logit)/T)`** + always-on entropy; differentiable on the
  logits, split OUT of the RL reward. Asymmetric on purpose: **no penalty for using fewer
  tokens than ρ*** — quality vs token-tax balance lets easy utts drop below the cap ("use
  fewer when doing well"; rate-distortion view). Cap-alone would sit at ρ*, so the tax is
  what yields sub-ρ* usage; keep `λ_press` mild. Two-sided pin, ponder/L0, rate-KL, RL
  compression term dropped; Lagrangian (fixed dataset-avg ρ) is the alternative (§6.1).
- 2026-07-31 — **`ρ*` is a DIAL (avg-rate target), not a per-utterance straitjacket** — no
  contradiction with "dynamic": placement (within-utt), per-utterance rate (soft penalty ⇒
  counts float around `N*`), and per-task rate (`ρ*[task]`) are all learned; `ρ*` only sets
  the operating point on the quality-vs-compression curve. Monitor the *distribution* of
  per-utterance ρ (spread = dynamism), not just the mean. v2: replace `ρ*` with a per-token
  price `λ` for fully-learned rate.
- 2026-07-31 — **KL-to-cold-start DROPPED** (user): cold-start is init only, no leash — we
  want RL to leave the cold-start (char) boundaries; also removes a knob and reduces init bias.
- 2026-07-31 — Cold-start (renamed from warmstart) init target = **char-level CTC boundaries**
  (`wavlm_boundaries/char`), DECIDED. CTC-derived (like the validated word_ctc); *finest*
  level ⇒ RL explores by merging/coarsening toward budget (gentler than splitting); neutral
  about linguistic units (no word-level lock-in). Warmup decoder trains on it (= B1).
- 2026-07-31 — ρ*-curriculum = **OFF** for v1 (hook exists; revisit only if unstable).
  CIF/Gumbel differentiable baseline = **deferred, not v1** (proxy-only control).
- 2026-07-31 — Training scheme = **joint, NO separate pretrained ASR** (user): one run =
  **decoder-warmup (~1 ep, segmenter frozen at char-CTC cold-start; = B1 baseline) →
  joint GRPO + decoder**, encoder frozen. Warmup removes the "RL on a garbage/untrained
  decoder" risk of fully-simultaneous joint-from-scratch. Dev WER is the arbiter;
  fully-simultaneous and REBORN-alternating are documented alternatives (§7).
- 2026-07-31 — Task-conditioning: **build the hook, run ASR-only** in v1.
- _open_: `λ_press`/`λ_cap` magnitudes (tune mild, no sweep) + exact ρ* ceiling value;
  CNN-vs-Transformer is an experiment, not a pre-decision. (Curriculum=off, cold-start=char,
  CIF/Gumbel=deferred are now settled.)

## 14. Open TODOs (priority-ordered, living)

1. **[TOP PRIORITY, 2026-08-12] Speed up full-prefix Transformer-AR before
   scheduling another large sweep.** The current implementation takes about 1 h 49 m
   per decoder-warm-up epoch and 3 h 22 m per joint-RL epoch, versus about 27 m and
   37 m for CNN first-order AR. Preserve the active jobs as the unoptimized control;
   do not cancel or mutate them.
   - **Implementation status (2026-08-12):** preallocated KV, window 64, and the
     opt-in production `combined_on_policy` mode are implemented. The latter batches
     greedy + `K=4` sampled policy lanes and all four NLL rewards while retaining one
     on-policy update. CPU equivalence tests and production smoke 70130 pass; full
     one-epoch reference/combined jobs 70131/70132 are running. The 30-batch pilot was
     1.94× faster in the training loop. Do not schedule a large sweep until the full
     stage-time, finite-loss, checkpoint, WER, and kept-ratio gate is reviewed.
   - **Measure first:** add synchronized timers around WavLM, greedy boundary rollout,
     `K`-sample rollout, decoder reward forwards, parallel policy re-score, backward,
     and validation generation. Benchmark the same 100 real minibatches and report
     wall time, utterances/s, frames/s, and peak VRAM.
   - **Exact in-process changes first:** replace per-step `torch.cat` KV growth with a
     preallocated cache and indexed writes; keep the cache API backend-neutral so a
     future vLLM/FlashInfer adapter remains possible. Batch greedy plus `K` sampled
     policy trajectories in one `(K+1)·B` step loop where practical. Extend the
     existing batched reward decoder to full-prefix AR, with configurable rollout
     microbatches if `K·B` exceeds memory.
   - **Local-attention arm:** fix the causal history window at `64` frames (about
     1.28 s at 50 Hz) for the current speed study. Apply the identical window in
     cached rollout and parallel teacher-forced re-scoring, and retain full history
     as the control. This is a model ablation, not a transparent implementation
     optimization, even though it uses the same parameters/checkpoint format. Do not
     vary the window until this control is explicitly reopened.
   - **Rollout group is fixed:** keep `grpo_k=4` for profiling, optimized runs,
     local-attention arms, and final comparisons. Do not reduce `K` as a speed
     optimization. Add rollout-diversity telemetry—unique boundary sequences per
     group, mean pairwise boundary disagreement/F1, within-group kept-ratio standard
     deviation, reward standard deviation, and zero-variance-group rate—so convergence
     within the four samples is directly visible.
   - **vLLM decision gate:** do not build an async rollout server now. vLLM is not a
     drop-in sampler for this policy: each generated boundary step must ingest the
     corresponding continuous WavLM frame, so it needs an out-of-tree vLLM model,
     custom per-request audio-feature state/input handling, weight synchronization,
     and likely a dedicated colocated or external rollout process. Reconsider this
     only if (a) the in-process exact changes plus local attention still leave rollout
     over 2× slower than CNN, and (b) full-prefix/local-history AR provides enough WER
     or boundary benefit to justify the engineering and extra GPU.
   - **Correctness gates:** cache logits and sampled actions match the reference path;
     rollout and parallel-score log probabilities agree under the same window;
     padding/frame-0 behavior is unchanged; fixed-seed rewards and gradients agree;
     existing CNN and independent-Transformer tests remain unchanged.

2. **[RUNNING, implemented 2026-08-11] A/B an autoregressive Transformer decoder
   boundary policy.** The current CNN and Transformer produce acoustically
   contextual logits, but for a fixed utterance they still sample every frame from an
   independent Bernoulli and therefore do not condition `b_t` on sampled labels
   `b_<t`. Add a new `transformer_ar` policy rather than changing `transformer`, so the
   current runs remain the matched independent control. The AR policy consumes the
   current WavLM frame plus the previous boundary embedding, samples with a causal KV
   cache, and re-scores completed samples in a parallel teacher-forced pass for GRPO.
   - Full implementation plan, API, rate-objective change, tests, launch gates, and
     risks: **`plans/autoregressive_transformer_boundary_policy.md`**.
   - Passed gates: causality + cache/score equivalence, CPU gradient/checkpoint tests,
     A100 cold-start smoke (64593), and `K=4` NLL joint-RL smoke (64594).
   - Active production path: fresh AR cold-start 64597 -> shared six-epoch warmup
     64598 -> three-seed `nll_mt` jobs 64599/64600/64601.
   - Preserve the running WavLM independent-Transformer jobs; do not cancel or
     overwrite them, because they answer the factorization A/B.

3. **[HIGHEST PRIORITY, 2026-08-03] Stop re-running decoder-warmup per RL ablation —
   share it across ablations that share a backbone.** Every `joint` run currently
   executes its own `warmup_epochs` of decoder-only training from scratch, but warmup
   is provably invariant to every RL-stage knob: in `compute_forward`
   (`train_speechllm_with_segmenter.py:313, :319-321`) the decoder path is built from
   `logits.detach()` (no gradient reaches the segmenter), and `_obj_joint`'s warmup
   branch (`:394-399`) just returns `dec_ce` — it never reads `segmenter_reward`,
   `rate_mode`/`rho_*`/`lambda_*`, `entropy_coeff_*`, `freeze_decoder_in_joint`,
   `grpo_k`, `num_iterations`, `pg_weight`, or `grpo_normalize_std`. Warmup's result
   depends only on `{segmenter_backbone (via coldstart_ckpt_dir), warmup_epochs, seed,
   lr_decoder, data/batching config}`. We've already paid for this redundancy at least
   twice: the A/B/C rate-objective sweep (2 backbones × 3 rate modes — each backbone's
   warmup redone 3×) and the nll_frozen / nll_mt / cer_mt reward ablation (3
   independent 6-epoch warmups on the same cnn backbone + seed). Note
   `submit_cer_opt.sh` only reused *cer_mt's own* crashed-run checkpoint (ordinary
   SpeechBrain crash recovery) — it did not, and currently structurally could not,
   reuse *nll_mt's or nll_frozen's* already-completed warmup.
   - Fix direction: an explicit warmup hand-off, analogous to `coldstart_ckpt_dir`.
     Either (a) a `warmup_ckpt_dir` hparam — after loading the cold-start segmenter,
     also load `proj`/`llm` (LoRA) state from a shared warmup checkpoint and start the
     epoch counter at `warmup_epochs+1`; or (b) an explicit third stage/script
     (`coldstart` → `warmup` → `joint_rl`), mirroring the existing two-stage pattern
     one level deeper — probably more robust since it avoids epoch-counter surgery.
   - Key the shared checkpoint by `(backbone, warmup_epochs, seed, lr_decoder,
     data-config)` so it's never loaded into an incompatible run.
   - Run once per backbone before launching the *next* ablation sweep (transformer
     backbone, future reward/rate variants) — the waste compounds with every batch of
     ablations until this lands.

4. **[Proposed, needs an A/B before adopting] Pooled independent rollouts (no reuse)
   vs. the current off-policy rollout-reuse for the GRPO group.** Current
   `_fit_batch_rl` (`:466-551`) draws **one** rollout of `K=grpo_k` sampled
   segmentations, then takes `mu=num_iterations` gradient *steps* reusing it via the
   PPO-clipped importance ratio (`grpo_clipped_pg_loss`), which corrects for steps
   2..mu being off-policy (θ has moved since those samples were drawn under θ_old).
   User-proposed alternative (recalled from ms-swift's GRPO — flagging that I haven't
   independently confirmed this is literally ms-swift's default mechanism, so treat
   the attribution loosely and judge the scheme on its own merits): draw `mu`
   *independent* rollouts of K samples each — shape `(mu, B, K)`, all still under the
   *same* θ since no step has been taken yet — pool them into one size-`(mu·K)` group
   per utterance and take a **single**, fully on-policy gradient step (plain
   `grpo_pg_loss`, no importance ratio/clip needed at all since θ never moved between
   samples).
   - Why it's plausible: the GRPO baseline (per-utterance mean/std) is a Monte Carlo
     estimate — a 12-sample group is a materially lower-variance baseline than a
     4-sample group reused 3×, and it removes clip-induced staleness/bias entirely.
     May matter more for the `cer`/`wer` reward (sparse, discrete, high-variance) than
     it did for `nll`.
   - Cost trade-off (why this isn't a free win): it **triples** the expensive
     free-running-decode cost per parameter update instead of amortizing it — current
     design gets 3 updates per 1 batched decode call; this gets 1 update per 3 batched
     decode calls. That directly reverses the point of the KV-cache + batched-K-decode
     work that specifically targeted the CER-reward decode as the project's
     bottleneck (`docs/project_notes/bugs.md`, 2026-08-03). Needs an explicit
     wall-clock + WER/CER comparison, not an assumed improvement.
   - If pursued, gate it behind a new flag rather than repurposing `num_iterations`
     (its meaning — off-policy reuse count — is incompatible with "independent
     pooled rollouts, single step") so both schemes stay available to compare.
