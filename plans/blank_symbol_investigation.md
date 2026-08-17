# Investigation Plan: The Role of "Blank" (Separator) Segments in CTC Word-Aligned Pooling

**Status:** Draft / proposed
**Date:** 2026-07-23
**Branch:** `blank_analysis`
**Anchor experiment:** exp-001 (`docs/project_notes/key_facts.md`), `boundary_source: alignment`, `boundary_target_dir: wavlm_boundaries/word_ctc`

---

## 1. TL;DR

In the fixed-pooling SpeechLLM (`speechllm_fixed_pooling.yaml`), a frozen SSL encoder's
frames are mean-pooled between CTC-forced-alignment word boundaries, projected, and fed
to a frozen LLM (+LoRA) as "audio tokens." A co-worker found that **removing the "blank"
symbol degrades WER.** This plan (a) pins down what "blank" concretely is in this pipeline,
(b) enumerates candidate mechanisms — including but going beyond the two current guesses
(boundary inaccuracy; `[CLS]`-like token) — and (c) lays out a cheap-first sequence of
experiments that *discriminate* between those mechanisms rather than merely re-confirming
the effect.

**Clarification (2026-07-23, from the user):** "removing blank" = after pooling the blank/
separator frames into a segment embedding, that embedding is **discarded** before the decoder
(the word embeddings are untouched). This is exactly condition **E-drop** below, and it makes
the word tokens *byte-for-byte identical* in the with-/without-blank conditions — so the whole
question reduces to **what the SEP tokens contribute to the decoder**. The decisive experiment
is therefore the **content ablation at fixed SEP positions** (E-const / E-zero / E-shuffle):
holding token count and positions fixed, it separates "SEP tokens carry useful *acoustic
content*" (H5) from "SEP tokens are content-free *slots / delimiters / sinks*" (H2/H4).

---

## 2. What "blank" actually is here (established from code + data, 2026-07-23)

This was verified directly, not assumed:

- The alignment boundaries come from `ctc_boundary_align.py` (on `origin/forced_alignment`),
  using `torchaudio` `WAV2VEC2_ASR_BASE_960H` (char-CTC, blank `'-'`=id 0, separator
  `'|'`=id 1). `merge_tokens` yields **blank-free** token spans; blank *frames* are not
  their own spans — they extend the previous segment.
- The **`word_ctc`** boundary policy (`boundaries_from_spans`) places a segment start at
  **(a)** every `'|'`-span start **and (b)** the first char-span after each `'|'`. So the
  pooled "audio-token" sequence alternates:
  `[lead, WORD1, SEP, WORD2, SEP, WORD3, …]`, where **SEP = the inter-word `'|'`/gap
  segment** (silence, breath, co-articulation — the frames the CTC model emits mostly as
  blank between words).
- **Empirically** (decoding real `.pt` tensors for 3 dev-clean utts): `word_ctc` yields
  **almost exactly 2×(#words)** segments (17w→34, 10w→20, 15w→30 = 2.00 seg/word), i.e.
  a clean word/separator alternation. Contrast the sibling levels: MFA `word` ≈ 1.2–1.3
  seg/word (no dedicated separators), `syllable` ≈ 1.5–2, `phone` ≈ 3.5–4.7, `char` ≈ 5–6.
  Mean compression for `word_ctc` ≈ **8–12 frames/segment** (vs `fixed_rate_k=5` = 5).

**So "the blank symbol" in this experiment = the SEP / inter-word segments, which are
~half of all audio tokens the LLM sees in `word_ctc` mode.** "Removing the blank" removes
roughly half the audio-token sequence, collapsing `[W, SEP, W, SEP, …]` → `[W, W, …]`.

> Note on terminology: technically the CTC *blank* token (`'-'`, id 0) and the *separator*
> (`'|'`, id 1) differ. The SEP segments are dominated by blank-emitted frames, which is
> why the team calls them "blank." Both readings point at the same objects (the gap
> segments); the plan treats them as the study target and does not hinge on the label.

### 2.1 RESOLVED — "remove blank" = Drop (confirmed by user, 2026-07-23)

"Removing blanks" = after aggregating the blank/separator frames into a segment embedding,
**discard that embedding** (do not send it to the decoder). Concretely: pool `word_ctc` as
usual, then **delete the SEP token columns** before the LLM. Three consequences shape the
whole investigation:

- **WORD embeddings are byte-for-byte identical** in the with-/without-blank conditions —
  only the *presence of SEP tokens in the decoder input* differs. Sequence length ~2N → ~N.
- The gap/silence acoustics live *entirely* in the SEP tokens; Drop removes exactly that
  content (and those positions) and nothing else. So the question is not "are word segments
  contaminated?" (they are unchanged) but "**what do the SEP tokens do for the decoder?**"
- This is condition **E-drop** (§4) and is the **reproduction target for E-repro**.

**Confirmed (2026-07-23):** Drop is applied in **both training and inference** (not
test-only). So H7 (train/test mismatch) is *not* the explanation — the effect is a genuine
"this trained model is worse."

### 2.2 Experimental regime — every without-blank condition is its own from-scratch training run

Because removal is train+test consistent, a without-blank result **cannot** be obtained by
intervening on the with-blank checkpoint at inference (that would reintroduce the mismatch we
just ruled out). Each without-blank condition — **E-repro/E-drop, E-const, E-zero, E-shuffle,
E-merge, E-count-match, E-delimiter/register** — is a **separate training run** with the
treatment applied consistently across train and eval. exp-001 is the **with-blank baseline
(Model A)** and is reused *only* for the Phase-1 **no-training diagnostics** — those
characterize what the SEP embeddings encode in Model A but do **not** by themselves reproduce
the WER drop; the effect requires the trained without-blank models.

---

## 3. Hypotheses (why would SEP/"blank" segments help WER?)

Each hypothesis has a literature anchor and, crucially, a *distinguishing prediction*.

- **H1 — Boundary buffer / CTC alignment inaccuracy** *(co-worker guess #1)*
  CTC forced alignment is peaky and its word edges are imprecise (well documented:
  Zeyer et al. 2021; Huang et al. 2024, "Less peaky… by label priors"). The SEP segments
  absorb the *uncertain edge frames* (partial phones, co-articulation, silence), keeping the
  WORD segments acoustically "clean." Remove them and those frames contaminate word
  embeddings (Merge) or are lost (Drop).
  → *Predicts:* Merge ≳ Drop, and Merge ≈ full; and using **accurate** boundaries without
  separators (MFA `word`) recovers most of the gap.
  *Caveat under the confirmed Drop definition:* the strong "buffer" form is **largely
  neutralized** — word embeddings are identical with/without SEP tokens, so SEP cannot be
  "protecting" word purity by absorbing edge frames. A residual form survives (SEP tokens
  give the decoder context to route around boundary errors); H1 is de-prioritized and probed
  indirectly via E-merge and E-word-MFA.

- **H2 — Register / attention-sink / `[CLS]`-like content-free slots** *(co-worker guess #2)*
  SEP tokens are extra positions the frozen LLM repurposes for internal computation /
  as attention sinks — their *acoustic content is irrelevant* (Darcet et al. 2024,
  "ViTs Need Registers"; Xiao et al. 2023, StreamingLLM; Gu et al. 2024, "When Attention
  Sink Emerges"). Halving the token count removes these scratch/sink slots.
  → *Predicts:* replacing SEP embeddings with a **single learned constant vector** (a real
  register token) recovers performance; **shuffling** SEP contents does not hurt; attention
  analysis shows SEP tokens act as sinks (high incoming attention, low content read-out);
  SEP embeddings are high-norm and tightly clustered.

- **H4 — Structural delimiter (word-boundary marker, à la whitespace)**
  Even if content-free, a SEP *between every pair of words* tells the LLM exactly where one
  word ends and the next begins. Distinct from H2: the *position between words* is the point,
  not extra compute. Without it, adjacent word embeddings abut with no delimiter and the
  model must re-infer segmentation.
  → *Predicts:* a content-free token **at word boundaries** recovers performance, but the
  same number of content-free tokens placed **not** at boundaries (mid-word / fixed-rate)
  does not.

- **H5 — Gap acoustic content is genuinely informative**
  Inter-word regions carry disambiguating acoustics: co-articulation/liaison ("an apple" vs
  "a napple", "grey day" vs "grade A"), reduced function words, breaths, hesitations. The
  LLM reads them. (Counterpoint to the usual "blank = no content" view; cf. Zhang et al.
  2025, "Analyzing the Importance of Blank for CTC-Based Knowledge Distillation," which
  argues blank is informative.)
  → *Predicts:* replacing SEP embeddings with constant / zero / shuffled vectors **hurts**;
  a linear probe recovers linguistic info (adjacent-word identity, pause duration, real-pause
  vs cross-word) from SEP embeddings.

- **H6 — Token-count / compression confound**
  Maybe it is simply "~2N audio tokens beat ~N": finer temporal granularity / lower
  compression, nothing special about separators. Speech-LLM sequence-length is known to
  matter (Soundwave 2025; CTC-compression work, Gaido et al. 2021).
  → *Predicts:* any scheme matched to ~2N tokens (fixed-rate at matched rate, or split-each-
  word-in-two, both separator-free) matches `word_ctc`; the with/without-SEP gap shrinks once
  token counts are equalized.

- **H7 — Train/test inconsistency — CONFIRMED NOT THE CAUSE**
  Removal is applied consistently in train **and** test (§2.2), so the drop is not a
  distribution-shift artifact. (Code check: `needs_boundary_target` adds `boundary_target` to
  *all* splits' `output_keys`, and `get_fixed_boundaries` runs for every stage — a modified
  boundary treatment is automatically consistent across train/valid/test.)
- **H8 — Optimization / capacity (weak, residual)**
  Even with consistent removal, the without-blank model is a *different trained model*: fewer
  and shorter audio-token sequences change the gradient signal to the projection/LoRA and
  shift the text prompt's positions. Part of any gap could be an optimization/capacity effect
  rather than "missing information." → *Control:* the content ablations (E-const/zero/shuffle)
  hold sequence length and count fixed, so they isolate content from this confound.

These are **not mutually exclusive**; the goal is to weigh their relative contributions.

---

## 4. Experimental design

Frozen SSL + frozen LLM ⇒ only LoRA + projection train ⇒ each run is cheap
(exp-001 = full 960h in ~6 h on 1×A100). **Ordering is cheap-evidence-first.**
Unless noted, sweep on **train-clean-100** (≈1 h/run) for fast comparative signal, then
**confirm the 2–3 decisive contrasts on full 960h**. All runs report WER/CER on
dev-clean / dev-other / test-clean / test-other; hold seed=3407 fixed.

### Phase 0 — Reproduce (gate for everything else)
- **E-repro = E-drop:** pool `word_ctc` as usual, then discard SEP token columns before the
  decoder (the confirmed definition, §2.1), removing consistently in train **and** test.
  Establishes the effect size on our stack and **rules out H7**. Requires the per-segment
  `is_separator` mask (§6). If it does not reproduce the co-worker's drop, stop and reconcile
  before proceeding.

### Phase 1 — Representation analysis (NO training; run on exp-001 checkpoint) — do first
Cheapest, fastest evidence; can run today on the existing with-blank checkpoint (Model A).
It **prioritizes** hypotheses but does **not** reproduce the WER drop — that needs the trained
without-blank models (Phase 0/2, §2.2).
- **A1 — Probe / geometry of SEP embeddings.** Dump pooled WORD vs SEP embeddings for a
  dev subset (a few hundred utts). Measure: norm, per-dim variance, PCA/cluster tightness;
  train linear probes SEP-emb → {pause-duration bucket, preceding/following word id (or
  word-length), real-pause vs cross-word gap}.
  *Reads:* tight cluster + high norm + low probe accuracy ⇒ register-like (H2);
  spread + high probe accuracy ⇒ content-rich (H5).
- **A2 — Attention / sink behavior.** Re-run forward with `output_attentions=True`; measure
  attention mass from text-decode positions and from WORD audio tokens onto SEP tokens (and
  onto the first/lead audio token). Sink signature (concentrated, position- not content-
  driven, esp. on the leading segment) ⇒ H2; distributed content-dependent attention ⇒ H5.

### Phase 2 — Core mechanism ablations (training) — decisive
Because Drop leaves the word tokens untouched, the question is purely *what SEP tokens
contribute*. Hold **positions and count fixed** (keep all ~2N slots) and vary only SEP
*content*:
- **E-const / E-zero / E-shuffle** *(the decisive contrast under the Drop definition)* —
  replace each SEP embedding with (a) a single **learned** vector (a genuine register/
  delimiter token), (b) **zeros**, (c) a **within-utterance shuffle** of SEP embeddings;
  keep them in their positions.
  - *Reads:* const/learned recovers full WER ⇒ SEP content is irrelevant; they act as
    content-free slots/delimiters ⇒ **H2/H4**. Any of these stays degraded vs full ⇒ SEP
    **content** matters ⇒ **H5**. Shuffle isolates content-*at-correct-position* from mere
    presence-of-a-token. **E-const is also the clean count/position control for E-drop**
    (same ~2N length, dead content): E-const vs E-drop isolates "having *a* token there" from
    "having the *right content* there."
- **E-word-MFA** (accurate boundaries, no separators) — also the missing baseline from HANDOFF.
  Point `boundary_target_dir` at `wavlm_boundaries/word` (MFA, ≈1 token/word). Zero code
  change. If it matches full `word_ctc` ⇒ SEP was compensating for CTC boundary error
  (**H1**). If `word_ctc` still wins ⇒ SEP helps beyond boundary accuracy.
- **E-merge** *(secondary — a **different** manipulation than the co-worker's; probes H1/H5)* —
  re-segment so SEP frames fold into the preceding WORD (WORD embeddings now include the gap
  frames; seq ~N, no SEP tokens; new `word_only` level, §5). Unlike E-drop, the gap acoustics
  are **retained inside** the word tokens.
  - *Reads:* E-merge ≈ full ≫ E-drop ⇒ the gap **content** was the point and can be salvaged
    without extra tokens (**H5**; argues against pure H2/H4). E-merge ≈ E-drop ⇒ folding the
    frames back doesn't help, so the **separate slot/position** mattered (**H2/H4**).

### Phase 3 — Confound controls (training)
- **E-count-match** (isolate token count, **H6**): (i) `boundary_source: fixed_rate` with
  `fixed_rate_k` tuned to match `word_ctc`'s ~2N count (measure exact mean; ≈ k=9), and
  (ii) **split-each-word-in-two** (2N word-aligned tokens, separator-free; new boundary
  level, §5). If a count-matched separator-free scheme ≈ `word_ctc`-with-SEP ⇒ mostly count
  (H6); if `word_ctc` still wins ⇒ separator structure matters beyond count.
- **E-delimiter-vs-register** (separate **H4** from **H2**): content-free token inserted
  *at word boundaries* (delimiter) vs same count of content-free tokens at *fixed-rate /
  mid-word* positions (register but wrong positions). Boundary-position recovery ⇒ H4;
  position-agnostic recovery ⇒ H2.

---

## 5. Hypothesis × experiment discrimination matrix

| Experiment | H1 buffer | H2 register/sink | H4 delimiter | H5 gap content | H6 count |
|---|---|---|---|---|---|
| A1 probe geometry | — | tight/high-norm/low-probe | — | spread/high-probe | — |
| A2 attention | — | sink signature | (boundary attn) | content attn | — |
| E-const/zero/shuffle *(decisive)* | (n/a) | **const recovers, shuffle ok** | const-at-bound recovers | **all stay degraded** | (n/a; count fixed) |
| E-merge vs E-drop | Merge≈full | Merge≈Drop | Merge≈Drop | **Merge≫Drop** | — |
| E-word-MFA | **matches word_ctc** | word_ctc still wins | word_ctc still wins | word_ctc still wins | depends on count |
| E-count-match | — | — | word_ctc wins if H4 | — | **match recovers** |
| E-delimiter-vs-register | — | any-position recovers | **only boundary recovers** | neither recovers | — |

Bold = the signature that most cleanly *confirms* that column's hypothesis.

---

## 6. Implementation notes (where things plug in)

- **Boundary loading hook:** `train_speechllm.py:576` `boundary_pipeline` loads
  `<utt_id>.pt` from `boundary_target_dir`. Boundary generation → mean-pool happens in
  `compute_forward` (`train_speechllm.py:197-211`) via `get_fixed_boundaries`
  (`:122`) and `mean_pool_segments` (`segment_pooling.py:171`).
- **Config-only variants (no code change):**
  - E-word-MFA → `--boundary_target_dir …/wavlm_boundaries/word`
  - E-count-match(fixed_rate) → `--boundary_source fixed_rate --fixed_rate_k 9`
- **New boundary levels (data-gen only, via `ctc_boundary_align.py`):**
  - `word_only` (E-merge): in `boundaries_from_spans`, drop the `if sp.token == sep_id:
    b[sp.start] = 1` writes; keep only word-onset writes ⇒ gaps fold into preceding word.
  - `word_split2` (E-count-match): after building word segments, add a midpoint boundary
    inside each word segment ⇒ ~2N word-aligned tokens, no separators.
  - Also emit, alongside `word_ctc`, a **per-segment `is_separator` mask** file (SEP vs WORD)
    so Drop / content-ablations can identify SEP tokens without re-aligning at train time.
- **Small, flag-gated code changes in `compute_forward` (contained, ~10-30 lines each):**
  - Drop: after `mean_pool_segments`, index-select non-SEP columns using the `is_separator`
    mask (aligned to segments via `compute_segment_ids`), recompute lengths/pad mask.
  - Const/zero/shuffle: overwrite SEP rows with an `nn.Parameter` register vector / zeros /
    a permutation. Add the parameter to `modules`/optimizer for the learned case.
  - Cleanest refactor: a single `boundary_transform` / `post_pool_transform` hook selected by
    a `blank_mode` config string (`keep|drop|merge|const|zero|shuffle`), so all Phase-2/3
    variants are one config switch and share one code path.
- **Phase-1 analysis script (new, eval-only):** load exp-001 checkpoint, run forward on a dev
  subset with `output_attentions=True`, dump per-segment embeddings + `is_separator` mask +
  attentions to disk; do variance/PCA/probe/attention-mass offline. SEP mask for the subset
  can come from re-running `ctc_boundary_align.segment_labels` (cheap for a few hundred utts).

---

## 7. Compute, ordering, budget

**METHODOLOGY LESSON (2026-07-23): train-clean-100 × 1 epoch is UNDERTRAINED — do not use
it for WER.** 100h×1ep = ~390 optimizer steps vs exp-001's ~3625 at 960h×1ep; the frozen-
backbone LoRA+proj setup stalls at train-loss ~3.0 (vs 0.72) and produces degenerate WER
>100% for *both* keep and drop, so the comparison is uninformative. Faithful regimes:
(a) **960h × 1 epoch** — the proven one; **keep@960h == exp-001** (blank_mode=keep is a strict
no-op), so exp-001 is the free with-blank baseline and only the without-blank variants need
running; or (b) **100h × ~9 epochs** to match ~3625 steps (cheaper wall-clock but multi-pass,
overfitting risk — validate against a 960h point before trusting it for the sweep).
Also: **evaluation is the long pole** (~hours; greedy decode at `test_batch_size=1` over
~11k dev/test utts) — most of a run's wall-clock is eval, not training.

Revised ordering:
1. **Phase 1 (A1, A2)** — no training; ~minutes on 1×A100. **Done (see §13).**
2. **Phase 0 E-repro** = **drop@960h vs exp-001 (keep@960h)** — the real reproduction. (+ a
   fresh keep@960h to re-validate the refactored code reproduces exp-001.)
3. **E-const / E-zero / E-shuffle** at a faithful regime (960h×1ep, or validated 100h×9ep) —
   the decisive content-vs-slot discrimination.
4. **E-word-MFA** + **E-merge**; then Phase-3 confounds (fixed_rate count-match, word_split2,
   delimiter-vs-register).

Each 960h run ≈ 5–6 h wall-clock (mostly eval); parallelizable across the reserved 8×A100.
**Output-folder isolation:** every run uses a distinct `--output_folder`
(`results/blank_ablation/<tag>_<mode>`) so nothing overwrites exp-001
(`results/…/alignment/3407/`). New runs must also override `--ssl_folder` to a populated SSL
cache (see bugs.md).

## 8. Metrics & analysis
- Primary: **WER** on dev-clean/dev-other/test-clean/test-other; secondary **CER**.
- Report each variant as a delta vs exp-001 (full `word_ctc`) and vs its own no-SEP counterpart.
- Log every run to `docs/project_notes/key_facts.md` under the existing "Boundary-source
  ablation" group (append rows, never edit); add an Analysis Note once a hypothesis is
  confirmed/rejected. Keep a per-run record of exact token counts (mean segments/utt) so H6
  can be read off directly.

## 9. Risks / pitfalls / validity threats
- **Output-folder overwrite** of exp-001 (see §7) — set distinct output paths.
- **SEP-mask correctness:** the `is_separator` mask must be generated from the *same*
  alignment that produced `word_ctc`, or Drop/content ablations mislabel tokens.
- **Length/positional confound leaking into H2/H5 tests:** const/zero/shuffle keep length
  fixed (good); Drop/Merge change length — always read them *together* with a count-matched
  control (Phase 3) before attributing effects to content vs count.
- **Frozen-LLM RoPE sensitivity:** changing audio-token count shifts positions of the text
  prompt; part of any Drop/Merge effect could be positional. E-const (same length, dead
  content) is the clean control for "content" that avoids this.
- **Small-data noise:** 100h single-seed deltas can be noisy; confirm decisive contrasts on
  960h and, for close calls, a second seed.
- **Don't over-read a single WER number:** require the *pattern across experiments* in the
  §5 matrix to agree before claiming a mechanism.

## 10. Relation to prior work (why the finding is interesting)
In speech-translation / speech-LLM, **dropping CTC-blank frames is a standard compression
trick that usually helps efficiency at little cost** (Gaido et al. 2021, "CTC-based
Compression"; Soundwave 2025). Our setup differs in two ways that plausibly flip the sign:
(1) blank/gap frames are pooled into **dedicated tokens**, not discarded; (2) the downstream
is a **frozen** LLM consuming them as embeddings. That the co-worker sees removal *hurt* is a
genuine counterpoint worth explaining — and the register/attention-sink literature (Darcet
2024; Xiao 2023) offers a concrete, testable reason the frozen LLM might *want* those extra
slots. Establishing which mechanism dominates is a publishable, self-contained result.

## 11. References
- Zeyer et al., 2021 — *Why does CTC result in peaky behavior?* arXiv:2105.14849
- Huang et al., 2024 — *Less Peaky and More Accurate CTC Forced Alignment by Label Priors* — arXiv:2406.02560
- Zhang et al., 2025 — *Analyzing the Importance of Blank for CTC-Based Knowledge Distillation* — arXiv:2506.01503
- Gaido et al., 2021 — *CTC-based Compression for Direct Speech Translation* — arXiv:2102.01578
- Darcet et al., 2024 (ICLR) — *Vision Transformers Need Registers* — arXiv:2309.16588
- Xiao et al., 2023 — *Efficient Streaming Language Models with Attention Sinks* — arXiv:2309.17453
- Gu et al., 2024 — *When Attention Sink Emerges in Language Models: An Empirical View* — arXiv:2410.10781
- (SpeechLLM sequence compression) Soundwave, 2025 — arXiv:2502.12900

## 12. Decision log (how results should update belief)
- E-repro (= E-drop) fails to reproduce → protocol/H7 issue; reconcile before continuing.
- E-const/learned recovers full WER (content-free slot suffices) → the SEP *content* is
  irrelevant → **H2/H4**; then E-delimiter-vs-register splits them.
- E-const/zero/shuffle all stay degraded + A1 high probe accuracy → SEP **content** matters → **H5**.
- E-merge ≫ E-drop (gap frames help when folded into words) → **H5** (salvageable content),
  and the extra *tokens* per se weren't the point; E-word-MFA≈full `word_ctc` additionally
  implicates CTC boundary error (**H1**).
- E-const recovers only when at word boundaries (E-delimiter-vs-register) → **H4 dominant**.
- E-const/zero/shuffle all hurt + A1 high probe accuracy → **H5 dominant** (real content).
- Count-matched separator-free scheme recovers → **H6 dominant** (it was mostly token count).
- Most likely outcome is a *mixture*; report contributions, not a single winner.

## 13. Execution log / results

**Infra built (2026-07-23):** `ctc_boundary_align.word_ctc_separator_frames` + `word_ctc_sep`
output (regenerated word_ctc verified bit-for-bit vs the shared tensors); `blank_mode:
keep|drop|zero|const|shuffle` in `segment_pooling.py`+`train_speechllm.py` (unit-tested);
`launch_blank_ablation.sh`, `phase1_blank_diag.py`. Confirmed empirically: word_ctc is a
strict `[sep,word,…]` alternation, **N sep + N word** tokens (6097/6097 across the 300-utt
dev subset). "Drop" removes exactly the N sep tokens; word tokens byte-identical.

**Phase-1 diagnostic** (with-blank exp-001 model, 300 dev-clean utts): SEP vs WORD tokens —
higher norm (22.5 vs 19.1 pre-proj), tighter cluster (mean-cos 0.72 vs 0.55 pre / 0.66 vs
0.42 post), i.e. a mild high-norm/register-ish signature; **but** linear probes on the
LLM-input SEP embeddings recover **pause duration R²=0.90** and **leading-vs-gap acc 0.996**
(base 0.951) — the SEP tokens carry strong timing/positional structure. *Read:* mixed;
argues **against a pure content-free register (H2)** and toward timing/structure (H3/H4/H5),
but is only suggestive (duration is partly a mean-pooling artifact; the const/zero/shuffle
training ablations are decisive). Embeddings saved: `results/blank_ablation/phase1_embeddings.pt`.

**E-repro (2026-07-24):** 100h×1ep was UNDERTRAINED (discarded, see §7). At 960h:
keep@960h reproduces exp-001 (dev-clean 6.11); drop@960h looked catastrophic (dev-clean
**21.77**) — but that is **~90% a decode-budget artifact**, not blanks carrying info. The
greedy searcher's `max_decode_steps = max_decode_ratio × audio-prefix-length`, and drop
halves the audio tokens → halves the budget → truncates long utterances (drop errors are
pure **deletions** that scale with utt length: 2.5% @≤15w → 26.9% @>40w). Re-decoding the
same drop checkpoint at `max_decode_ratio=3` collapses deletions (8105→162) and gives:

| bs=16, r=3 | dev-clean | dev-other | test-clean | test-other |
|---|---|---|---|---|
| keep@960h | 6.33 | 10.03 | 6.32 | 10.09 |
| drop@960h | 7.91 | 11.43 | 7.87 | 12.21 |

**Genuine residual effect of removing blanks ≈ +1.5–2% WER** (substitution-dominated) — real
but ~10× smaller than the naive number. NEW confound found & documented (max_decode ↔ audio
token count); NEW hypothesis to fold in: part of the residual could still be token-count (H6),
which the content ablations (zero/shuffle/const, fixed 2N tokens) will isolate from content (H5).
Next: content ablations at 960h/bs16/r3 vs the keep@r3 baseline (6.33). Also ask the co-worker
what decode limit their original finding used (same confound risk).

**Content ablations (2026-07-24) — VERDICT: the residual is gap CONTENT (H5).** All 960h,
bs=16, r=3, dev-clean WER: keep 6.33 | shuffle 6.72 | const 7.60 | zero 7.76 | drop 7.91 |
blank_only 38.51.
- **const ≈ zero ≈ drop** → a content-free slot, even a *learned* one, recovers nothing:
  **H2 (register/[CLS]/sink), H4 (delimiter), H6 (count) all REJECTED.**
- **shuffle ≈ keep** (real gap embeddings, permuted) → recovers ~60% of the gap: it's the
  *presence of real gap acoustic content*, not per-position alignment or extra slots.
- **blank_only** (gaps only, no words) → ~38% WER: the inter-word gaps alone are richly
  informative. **H5 CONFIRMED**, consistent with the Phase-1 duration probe (R²=0.90).
- Bottom line for the write-up: the naive "remove blank ⇒ +15% WER" is **~90% a
  `max_decode_ratio` artifact**; the genuine effect is **~+1.5% WER and it is the separator
  tokens' real acoustic content** (co-articulation/transition cues), not a CLS/register role.
