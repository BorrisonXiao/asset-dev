#!/usr/bin/env python3
"""Build the standalone cross-lingual 24k-step replication proposal.

Usage:
    MPLCONFIGDIR=/tmp/jointllm-mpl-cache \
      ~/cxiao/envs/jointllm/bin/python \
      make_xlingual_replication_proposal.py \
      --out artifacts/segmenter/xlingual_24k_replication_proposal.html
"""

import argparse
import datetime
import os
import sys

sys.path.insert(0, os.path.expanduser("~/.claude-scale/skills/html-report"))
import htmlkit as hk  # noqa: E402


EXTRA_CSS = """
.wrap{max-width:1100px}
a{color:var(--series-sed);text-decoration-thickness:1px;text-underline-offset:2px}
.note{font-size:13px;color:var(--text-secondary)}
.left-table th,.left-table td{text-align:left;vertical-align:top}
.left-table th:first-child,.left-table td:first-child{white-space:nowrap}
.matrix th,.matrix td{text-align:left;vertical-align:top}
.matrix tr.recommended{background:var(--band)}
.phase{display:grid;grid-template-columns:42px 1fr;gap:12px;align-items:start;margin:12px 0}
.phase-num{width:36px;height:36px;border-radius:50%;border:1px solid var(--border-strong);display:flex;align-items:center;justify-content:center;font-weight:600;background:var(--surface-1)}
.phase h3{margin:1px 0 4px;font-size:16px}.phase p{margin:0;color:var(--text-secondary)}
code{font-family:ui-monospace,"SF Mono","Cascadia Code",Consolas,monospace;font-size:.92em}
.gate{display:inline-block;border:1px solid var(--border-strong);border-radius:8px;padding:1px 8px;font-size:12px;background:var(--band);margin-left:6px}
@media(max-width:760px){.wrap{padding:0 18px}.card{overflow-x:auto}}
"""


def table(headers, rows, cls="left-table"):
    head = "".join("<th>%s</th>" % h for h in headers)
    body = []
    for row, row_cls in rows:
        body.append(
            "<tr%s>%s</tr>"
            % (
                (' class="%s"' % row_cls) if row_cls else "",
                "".join("<td>%s</td>" % cell for cell in row),
            )
        )
    return '<table class="%s"><thead><tr>%s</tr></thead><tbody>%s</tbody></table>' % (
        cls,
        head,
        "".join(body),
    )


def phase(num, title, text):
    return (
        '<div class="phase"><div class="phase-num">%s</div>'
        "<div><h3>%s</h3><p>%s</p></div></div>" % (num, title, text)
    )


def build(args):
    body = "<style>%s</style>" % EXTRA_CSS

    body += hk.hero(
        "Proposal · RL dynamic downsampling",
        "Cross-lingual 24k-step replication: Mandarin and Japanese",
        "Replicate the LS960 24k-step learned-downsampling study in two languages "
        "whose natural acoustic-unit rates differ from English, with the minimal "
        "baseline set (fixed k=5, character/mora oracle, phone oracle). Nothing in "
        "this proposal has been trained yet; six design choices below need sign-off "
        "before jobs are submitted.",
    )

    # ------------------------------------------------------------------ 0
    body += hk.section(
        "0 · Decision summary",
        body=hk.tiles(
            [
                ("Languages", "zh + ja", "Mandarin (syllable-timed), Japanese (mora-timed); English study was stress-timed"),
                ("Datasets", "AISHELL-1 · ReazonSpeech", "150 h read Mandarin; 100 h Japanese TV audio (encoder-in-domain)"),
                ("Encoders", "HuBERT-Large ×2", "chinese-hubert-large (10k h WenetSpeech) · japanese-hubert-large (19k h ReazonSpeech)"),
                ("Budget", "24k steps/seed", "Learned arm; 3 seeds per arm; 4 arms per language"),
            ]
        )
        + hk.finding(
            "<b>Status: proposal only.</b> No training jobs are submitted. Section 9 "
            "lists the design choices that need confirmation; each has a "
            "recommendation and a fallback gate.",
        ),
    )

    # ------------------------------------------------------------------ 1
    body += hk.section(
        "1 · Question and rationale",
        lead="Does learned boundary placement transfer outside English — and does it "
        "still pay off when the language's natural unit rate sits on the other side "
        "of the fixed-rate control?",
        body=(
            "<p>The completed LS960 study showed that a learned Transformer-AR "
            "segmenter with BiGRU pooling (2.79±0.02 / 5.63±0.12 WER at ≈10.9 Hz) "
            "matches the character-oracle system and clearly beats fixed k=5. All of "
            "that evidence is in one language. English is <i>stress-timed</i>, and its "
            "character-CTC oracle runs at ≈14.6 Hz (kept ratio \\(\\rho\\approx0.29\\), "
            "where \\(\\rho\\) is emitted boundaries per 50 Hz encoder frame; lower "
            "means more compression) — <b>above</b> the 10 Hz fixed k=5 grid.</p>"
            "<p>The two added languages invert or shift that relationship, which is "
            "exactly what makes the replication informative:</p>"
            "<ul>"
            "<li><b>Mandarin</b> is syllable-timed; one written character is one "
            "syllable, spoken at ≈5–6 /s, so the character oracle sits at roughly "
            "\\(\\rho\\approx0.11\\) — <b>below</b> the fixed k=5 grid. If placement "
            "still wins near 10 Hz, the gain is not an artifact of English character "
            "rate exceeding the control rate.</li>"
            "<li><b>Japanese</b> is mora-timed at ≈7–8 mora/s "
            "(\\(\\rho\\approx0.15\\)), between the two. Its phone rate is high "
            "(mostly CV moras), separating the phone oracle from the mora oracle more "
            "than in English.</li>"
            "</ul>"
            "<p>With English, the study then spans the three canonical rhythm classes "
            "(stress-, syllable-, and mora-timed), and each language contributes the "
            "same four-arm contrast: fixed grid vs. two alignment oracles vs. learned "
            "placement, all sharing one encoder and one decoder within the "
            "language.</p>"
        ),
    )

    # ------------------------------------------------------------------ 2
    body += hk.section(
        "2 · Languages, datasets, encoders",
        lead="Each language pairs the strongest public in-domain SSL encoder with a "
        "standard clean benchmark; both encoders expose the same 1024-d, 50 Hz "
        "interface as WavLM-Large, so the pipeline is unchanged.",
        body=hk.labeled_asset(
            table(
                ["Language", "Corpus", "Train", "Selection (dev)", "Test", "Notes"],
                [
                    (
                        [
                            "Mandarin (zh)",
                            "AISHELL-1 (openslr SLR33)",
                            "120,098 utts / ≈150 h / 340 spk",
                            "dev: 14,326 utts / 40 spk",
                            "test: 7,176 utts / 20 spk",
                            "16 kHz read speech, 11 topic domains; clean human transcripts; standard CER benchmark",
                        ],
                        None,
                    ),
                    (
                        [
                            "Japanese (ja)",
                            "ReazonSpeech v2 <code>small</code>",
                            "≈100 h TV broadcast audio (FLAC 16 kHz)",
                            "held-out ≈5 h, disjoint broadcast dates",
                            "held-out ≈5 h (in-domain) + JSUT basic5000 (clean read, out-of-domain)",
                            "Same corpus family the encoder was pre-trained on; transcripts are subtitle-aligned (v2 pipeline), so a clean out-of-domain test set is added",
                        ],
                        None,
                    ),
                ],
            ),
            "Table",
            1,
            "Datasets per language. The ja dev/test carve-out is made disjoint by broadcast "
            "date (the ReazonSpeech utterance name encodes date and program) to avoid "
            "near-duplicate leakage.",
            title="Datasets",
        )
        + hk.labeled_asset(
            table(
                ["Language", "Encoder", "Architecture", "Pre-training data", "In-domain?"],
                [
                    (
                        [
                            "en (done)",
                            "<code>microsoft/wavlm-large</code>",
                            "WavLM-Large, 24 L, 1024-d, 50 Hz",
                            "94k h (Libri-Light + GigaSpeech + VoxPopuli)",
                            "Read audiobooks ⊂ pre-training mix",
                        ],
                        None,
                    ),
                    (
                        [
                            "zh",
                            "<code>TencentGameMate/chinese-hubert-large</code>",
                            "HuBERT-Large, 24 L, 1024-d, 50 Hz",
                            "10k h WenetSpeech-L (Mandarin web audio)",
                            "Same language, broader domain than AISHELL-1 read speech",
                        ],
                        None,
                    ),
                    (
                        [
                            "ja",
                            "<code>rinna/japanese-hubert-large</code>",
                            "HuBERT-Large, 24 L, 1024-d, 50 Hz",
                            "19k h ReazonSpeech v1 (Japanese TV)",
                            "Pre-training and fine-tuning share the corpus family — the closest match available",
                        ],
                        None,
                    ),
                ],
            ),
            "Table",
            2,
            "Encoders. Both new encoders are HuBERT-Large — identical feature interface to "
            "WavLM-Large (swap via --ssl_hub / --ssl_feat_dims 1024), loaded through the same "
            "SpeechBrain HF wrapper. WavLM adds a denoising objective and gated relative "
            "position bias over HuBERT; this caveats cross-language absolute comparisons but "
            "not the within-language arm contrasts, which all share one encoder.",
            title="Encoders",
        )
        + "<p class='note'>Decoder: <code>meta-llama/Llama-3.2-1B-Instruct</code> "
        "(projection + LoRA), identical to the English study. A tokenizer check on "
        "AISHELL/ReazonSpeech-style text gives 0.83–0.96 tokens per Chinese character "
        "and 0.65–0.71 per Japanese character with exact round-trip, so sequence "
        "lengths are unproblematic. Chinese and Japanese are outside Llama-3.2's "
        "eight officially supported languages; absolute CER may trail a zh/ja-native "
        "LLM, but every arm within a language shares the decoder, so the contrasts "
        "stay valid. Section 9 (choice 1) defines the smoke-test gate for switching "
        "to Qwen if generation proves degenerate.</p>",
    )

    # ------------------------------------------------------------------ 3
    body += hk.section(
        "3 · Arms",
        lead="Four arms per language, three seeds each (3407/3408/3409): the minimal "
        "baseline set plus the learned system.",
        body=hk.labeled_asset(
            table(
                ["Arm", "Boundaries", "Pooling", "Trained parts", "Budget/seed"],
                [
                    (
                        [
                            "fixed k=5",
                            "Uniform grid, every 5th frame (10 Hz, \\(\\rho=0.20\\))",
                            "BiGRU residual",
                            "Projection + LoRA + BiGRU (CE)",
                            "10 epochs, dev-selected",
                        ],
                        None,
                    ),
                    (
                        [
                            "char / mora oracle",
                            "zh: character-CTC alignment (≈ syllables). ja: mora-CTC alignment (katakana units)",
                            "BiGRU residual",
                            "Projection + LoRA + BiGRU (CE)",
                            "10 epochs, dev-selected",
                        ],
                        None,
                    ),
                    (
                        [
                            "phone oracle",
                            "MFA forced alignment (<code>mandarin_mfa</code> / <code>japanese_mfa</code> v3)",
                            "BiGRU residual",
                            "Projection + LoRA + BiGRU (CE)",
                            "10 epochs, dev-selected",
                        ],
                        None,
                    ),
                    (
                        [
                            "learned (replication target)",
                            "Transformer-AR policy, 64-frame local history, GRPO",
                            "BiGRU residual",
                            "Policy + projection + LoRA + BiGRU",
                            "<b>24,000 optimizer steps</b> (warmup + joint RL)",
                        ],
                        "recommended",
                    ),
                ],
                cls="matrix",
            ),
            "Table",
            3,
            "Arm matrix per language. All arms share the language's encoder (frozen), the "
            "decoder initialization, and BiGRU residual pooling, so the only difference "
            "between arms is where boundaries fall. The LS960 study pooled its fixed/oracle "
            "baselines with parameter-free means and added BiGRU controls later; here BiGRU "
            "is used everywhere from the start (Section 9, choice 4).",
            title="Experiment matrix",
        ),
    )

    # ------------------------------------------------------------------ 4
    body += hk.section(
        "4 · Boundary targets",
        lead="Oracle targets reuse the LibriSpeech tooling "
        "(prepare_boundary_targets.py / ctc_boundary_align.py from the "
        "forced_alignment branch); the 50 Hz frame formula is identical for "
        "HuBERT-Large. Only the unit inventories are language-specific.",
        body=(
            "<ul>"
            "<li><b>Phone oracle.</b> MFA v3 with <code>mandarin_mfa</code> and "
            "<code>japanese_mfa</code> acoustic models + dictionaries. AISHELL-1 "
            "transcripts are unsegmented character strings, so zh runs "
            "<code>jieba</code> word segmentation before MFA lookup; ja uses MFA's "
            "own tokenization. Runs on the CPU partition.</li>"
            "<li><b>zh character oracle.</b> Character-CTC forced alignment on frozen "
            "chinese-hubert-large features (AISHELL-1 train has a ≈4.2k character "
            "inventory — routine for Mandarin CTC). One character ≈ one syllable, so "
            "this is simultaneously the syllable oracle.</li>"
            "<li><b>ja mora oracle.</b> Raw Japanese orthography mixes kanji and kana: "
            "a 3k+ character CTC over 100 h would align rare kanji unreliably, and one "
            "kanji spans several moras. Instead transcripts are converted to katakana "
            "readings (<code>fugashi</code> + <code>unidic-lite</code>) and aligned "
            "with a ≈150-unit mora-CTC. Mora is Japanese's natural sub-word rhythm "
            "unit, exactly parallel to zh character = syllable (Section 9, "
            "choice 3).</li>"
            "<li><b>Acceptance stats.</b> After generation: coverage ≥ 99.5% of "
            "utterances per split, and measured \\(\\rho_{\\text{char}}\\), "
            "\\(\\rho_{\\text{phone}}\\) reported next to the predictions in "
            "Table 4 — a large deviation would flag an alignment bug before any GPU "
            "training starts.</li>"
            "</ul>"
        )
        + hk.labeled_asset(
            table(
                ["Language", "char/mora rate (pred.)", "\\(\\rho_{\\text{char}}\\) (pred.)", "phone rate (pred.)", "\\(\\rho_{\\text{phone}}\\) (pred.)", "fixed k=5"],
                [
                    (["en (measured)", "≈14.6 Hz (char-CTC)", "0.29", "≈10.5 Hz (MFA)", "0.21", "10 Hz"], None),
                    (["zh", "≈5–6 Hz (chars = syllables)", "≈0.11", "≈11–13 Hz", "≈0.22–0.26", "10 Hz"], None),
                    (["ja", "≈7–8 Hz (moras)", "≈0.15", "≈13–15 Hz", "≈0.26–0.30", "10 Hz"], None),
                ],
            ),
            "Table",
            4,
            "Predicted boundary rates (to be replaced by measured values after alignment). "
            "en values are measured from the completed study. The zh character oracle falls "
            "below the fixed control; the en character oracle sits above it — the reversal "
            "the study is designed around.",
            title="Expected unit rates",
        ),
    )

    # ------------------------------------------------------------------ 5
    body += hk.section(
        "5 · Curriculum and budget",
        lead="The learned arm replicates the LS960 step24k curriculum stage for "
        "stage; only corpus-size-dependent knobs are re-derived, and each mapping is "
        "stated explicitly.",
        body=phase(
            "0",
            "Cold start (per language, seed 3407, shared)",
            "Supervised char/mora-CTC cold start of the Transformer-AR policy "
            "(BCE against oracle boundaries), 6 epochs at 400 s batches "
            "(≈8.1k steps on AISHELL — matching the ≈8.6k-step single-pass LS960 "
            "cold start), select best dev boundary F1. All three RL seeds share it, "
            "as on LS960.",
        )
        + phase(
            "1",
            "Decoder initialization + in-run warmup",
            "Each learned seed initializes its decoder from the char/mora-oracle "
            "arm's dev-selected checkpoint of the same seed (LS960 used the "
            "oracle-char decoder the same way), then trains decoder-only CE on "
            "argmax boundaries for the first 2,400 optimizer steps — 10% of the "
            "budget, the same share as LS960's 2,395-step warmup. Because 2,400 "
            "steps exceeds one epoch on these corpora, the existing "
            "fraction-of-first-epoch knob cannot express it; a small "
            "--warmup_optimizer_steps extension (absolute steps) is added with a "
            "unit test beside test_segmenter_step_schedule.py.",
        )
        + phase(
            "2",
            "Joint GRPO to 24,000 steps",
            "Identical to LS960 step24k: NLL reward, K=4 rollouts, rate band "
            "\\(\\rho\\in[0.15,0.25]\\) with \\(\\lambda_{\\text{cap}}=1.0\\), "
            "rate_channel auto, lr 2e-4 (decoder) / 5e-5 (policy), entropy 0, BF16, "
            "300 s batches, validation at warmup end and every 4k steps, selection "
            "by minimum dev CER.",
        )
        + phase(
            "3",
            "CE baselines",
            "fixed k=5, char/mora oracle, phone oracle: 10 epochs CE with "
            "dev-selection at 500 s batches × grad-accumulation 2 — the protocol of "
            "the matched 100 h controls. At these corpus sizes 10 CE epochs "
            "(≈10 passes) is comparable data exposure to the learned arm's 24k "
            "steps (≈13 passes at 300 s), and CE converges well within it. "
            "Section 9 (choice 6) records the strict-parity alternative.",
        )
        + hk.finding(
            "<b>Budget parity is better than the original.</b> On LS960 the "
            "baselines were single-pass and the report carried an explicit "
            "unequal-budget caveat. Here every arm is dev-selected inside a "
            "generous budget, so no arm is read at an arbitrary stopping point.",
            ok=True,
        ),
    )

    # ------------------------------------------------------------------ 6
    body += hk.section(
        "6 · Hyperparameters",
        lead="Everything not listed as language-dependent is carried over from the "
        "LS960 step24k configuration unchanged.",
        body=hk.labeled_asset(
            table(
                ["Group", "Setting", "Value", "Same as LS960?"],
                [
                    (["Policy", "Transformer-AR", "4 layers, 256-d, 4 heads, FFN 1024, dropout 0, history window 64, max positions 4096, preallocated cache", "yes"], None),
                    (["Pooling", "BiGRU residual", "hidden 128, 1 layer, dropout 0", "yes"], None),
                    (["RL", "GRPO", "K=4, NLL reward, pg_weight 1.0, entropy 0, rl_update_mode combined_on_policy", "yes"], None),
                    (["Rate", "band", "\\(\\rho_{lo}=0.15,\\ \\rho_{hi}=0.25\\), \\(\\lambda_{\\text{cap}}=1.0\\), rate_channel auto", "yes (choice 5)"], None),
                    (["Optimization", "LRs", "decoder 2e-4 (warmup and joint), policy 5e-5; BF16; grad-accum 1 (learned) / 2 (CE)", "yes"], None),
                    (["Batching", "duration budget", "300 s learned, 400 s cold start, 500 s CE arms; min 2 utts; train cap 25 s/utt", "yes"], None),
                    (["Schedule", "steps", "24,000 optimizer steps; warmup 2,400 absolute steps (new knob); validate every 4,000 + warmup end", "24k/4k yes; warmup knob new"], None),
                    (["Selection", "checkpoint", "min dev CER (learned + CE); max dev boundary F1 (cold start)", "yes (CER for WER)"], None),
                    (["Encoder", "frozen SSL", "ssl_hub per language, ssl_feat_dims 1024, output_norm, frozen", "swap only"], None),
                    (["Decoder", "LLM", "Llama-3.2-1B-Instruct + projection + LoRA (choice 1)", "yes"], None),
                    (["Seeds", "3 per arm", "3407 / 3408 / 3409", "yes"], None),
                ],
            ),
            "Table",
            5,
            "Hyperparameters. \"Same as LS960?\" marks deliberate deviations; each deviation "
            "is one of the numbered design choices in Section 9.",
            title="Configuration vs. the English study",
        ),
    )

    # ------------------------------------------------------------------ 7
    body += hk.section(
        "7 · Evaluation",
        lead="CER is primary in both languages; boundary statistics are reported "
        "beside recognition quality so rate and placement are never conflated.",
        body=hk.labeled_asset(
            table(
                ["What", "zh", "ja"],
                [
                    (["Primary metric", "CER on AISHELL-1 test (dev for selection)", "CER on held-out ReazonSpeech test (in-domain) and JSUT basic5000 (clean, out-of-domain)"], None),
                    (["Text normalization", "Strip punctuation/whitespace (AISHELL transcripts are already clean)", "NFKC width normalization, strip punctuation; score raw orthography (no kana conversion)"], None),
                    (["Boundary metrics", "Realized rate (Hz) and \\(\\rho\\); boundary F1 vs. MFA phone and char references at ±20/±40 ms", "Same, vs. MFA phone and mora references"], None),
                    (["Efficiency", "Decoder-side audio tokens/s for every arm (rate alone explains part of any CER gap)", "same"], None),
                ],
            ),
            "Table",
            6,
            "Evaluation protocol. CER uses the existing ErrorRateStats character splitting; "
            "±20/±40 ms tolerances match the completed boundary audits.",
            title="Metrics",
        )
        + "<p class='note'>Deliverables: per-language four-arm CER tables with "
        "three-seed mean ± sample SD, a CER-vs-rate frontier figure per language in "
        "the style of the LS960 report, and a cross-language section relating each "
        "language's learned operating rate to its oracle unit rates.</p>",
    )

    # ------------------------------------------------------------------ 8
    body += hk.section(
        "8 · Compute and schedule",
        body=hk.tiles(
            [
                ("GPU budget", "≈420 A100-h", "≈210 per language, dominated by 3 learned 24k-step runs (~45 h each)"),
                ("Wall time", "≈6–7 days", "At the account's 3-GPU concurrency cap; CPU alignment runs in parallel on med"),
                ("Storage", "≈35 GB data + ≈25 GB results", "AISHELL-1 15 GB, ReazonSpeech small 6 GB, JSUT 2.7 GB, boundary targets, checkpoints"),
                ("New code", "small, tested", "2 prep scripts, mora conversion, --warmup_optimizer_steps knob, 2 launchers"),
            ]
        )
        + phase("S0", "Data + manifests (1–2 days, CPU)", "Download AISHELL-1 / ReazonSpeech small / JSUT; prep scripts with manifest unit tests (utterance counts, durations, date-disjoint ja splits). No GPU spent before these pass.")
        + phase("S1", "Alignment + targets (2–3 days, CPU + few GPU-h)", "MFA runs; char/mora CTC aligner training; generate per-utterance .pt targets; publish measured rate table (updates Table 4). <span class='gate'>Gate: coverage ≥99.5%, rates plausible</span>")
        + phase("S2", "Smoke (½ day, 2 GPU-h)", "One char/mora-oracle seed per language, 2k steps: verifies decoder path, CER scoring, text normalization end to end. <span class='gate'>Gate: decoder keeps Llama vs. switches to Qwen (choice 1)</span>")
        + phase("S3", "CE baselines (2 days)", "3 arms × 3 seeds × 2 languages, 10-epoch CE, dev-selected.")
        + phase("S4", "Cold starts + learned arm (3–4 days)", "1 cold start per language, then 3 × 24k-step joint runs per language.")
        + phase("S5", "Analysis + report", "Tables, frontier figures, boundary audits; publish as a new report page; log results in project notes."),
    )

    # ------------------------------------------------------------------ 9
    body += hk.section(
        "9 · Key design choices needing sign-off",
        lead="Each row is a real fork. The recommended option is what the stages "
        "above assume; saying \"go\" adopts all six recommendations.",
        body=hk.labeled_asset(
            table(
                ["#", "Choice", "Options", "Recommendation and why", "Gate / fallback"],
                [
                    (
                        [
                            "1",
                            "Decoder LLM for zh/ja",
                            "(a) Llama-3.2-1B-Instruct everywhere · (b) Qwen3-1.7B (zh/ja-native)",
                            "<b>(a)</b> — identical decoder to the English study removes a confound; tokenizer already efficient on zh/ja (0.65–0.96 tok/char, exact round-trip); within-language contrasts don't need a SOTA decoder",
                            "S2 smoke: if oracle-arm dev CER is degenerate (>30%) or output drifts scripts, switch both languages to (b)",
                        ],
                        "recommended",
                    ),
                    (
                        [
                            "2",
                            "Dataset scale",
                            "(a) Clean ≈100–150 h benchmarks · (b) scale-matched ≈1,000 h (WenetSpeech-M, ReazonSpeech medium)",
                            "<b>(a)</b> — oracle arms need trustworthy alignments and eval needs trusted transcripts; (b) costs ≈500 GB download, noisier labels weaken the oracles, and 24k steps stays the budget either way",
                            "If (a) shows learned ≫ oracles only because oracles are weak, (b) becomes the follow-up",
                        ],
                        "recommended",
                    ),
                    (
                        [
                            "3",
                            "Japanese sub-word oracle unit",
                            "(a) mora (katakana) CTC · (b) raw orthography char CTC",
                            "<b>(a)</b> — 150-unit inventory aligns reliably on 100 h; parallels zh char=syllable as the language's rhythm unit; (b) has 3k+ sparse units and kanji spanning several moras",
                            "Report both unit rates; if fugashi reading coverage <99.5%, revisit",
                        ],
                        "recommended",
                    ),
                    (
                        [
                            "4",
                            "Baseline pooling",
                            "(a) BiGRU residual everywhere · (b) parameter-free mean (strict LS960 parity)",
                            "<b>(a)</b> — matched pooling isolates boundary placement; the project already added BiGRU controls to fix this on English",
                            "None needed; (b) only if compute must shrink",
                        ],
                        "recommended",
                    ),
                    (
                        [
                            "5",
                            "Rate band for the learned arm",
                            "(a) shared [0.15, 0.25] · (b) per-language band anchored at the char/mora rate",
                            "<b>(a)</b> — the band brackets the fixed k=5 control (ρ=0.20) in every language, keeping the matched-rate comparison; note zh's syllable rate (≈0.11) lies below the band floor by design",
                            "Per-language bands are the natural follow-up sweep, not part of the minimal replication",
                        ],
                        "recommended",
                    ),
                    (
                        [
                            "6",
                            "CE baseline budget",
                            "(a) 10 epochs, dev-selected · (b) full 24k steps for CE arms too",
                            "<b>(a)</b> — CE converges long before 24k steps (≈44 passes at CE batching); dev-selection makes the endpoint principled and saves ≈90 GPU-h",
                            "If any CE arm's dev curve is still improving at epoch 10, extend that arm",
                        ],
                        "recommended",
                    ),
                ],
                cls="matrix",
            ),
            "Table",
            7,
            "Design choices. Every recommendation is the option the schedule and budget in "
            "Section 8 assume.",
            title="Sign-off list",
        ),
    )

    # ------------------------------------------------------------------ 10
    body += hk.section(
        "10 · Risks",
        body=(
            "<ul>"
            "<li><b>ReazonSpeech transcript noise.</b> Subtitle-derived labels inflate "
            "absolute CER and blur the mora oracle. Mitigations: encoder is in-domain "
            "by construction, JSUT provides a clean secondary test, and alignment "
            "coverage stats are a hard gate before training.</li>"
            "<li><b>Llama-3.2 weakness in zh/ja.</b> Handled by design choice 1's "
            "smoke gate. Note the failure mode to watch is script drift (e.g., pinyin "
            "or romaji output), not just high CER.</li>"
            "<li><b>Domain gap zh.</b> chinese-hubert-large is pre-trained on web "
            "audio, AISHELL-1 is clean read speech — an easier direction than the "
            "reverse, but it makes zh the weaker in-domain claim of the two "
            "languages.</li>"
            "<li><b>More passes under RL.</b> 24k steps is ≈13 passes here vs. ≈2 on "
            "LS960; the policy re-samples the same utterances more often. Dev-CER "
            "selection every 4k steps plus the existing joint-drift monitoring "
            "(selected-epoch bookkeeping) covers it; if drift appears, densify "
            "validation to 2k steps — no new machinery.</li>"
            "<li><b>MFA dictionary coverage.</b> OOV entries (AISHELL topic domains, "
            "ja proper nouns) get G2P fallback via the MFA models' own G2P; OOV "
            "counts are reported with the alignment stats.</li>"
            "</ul>"
        ),
    )

    # ------------------------------------------------------------------ 11
    body += hk.section(
        "11 · English anchors",
        lead="Completed LS960 results the replication is read against (three-seed "
        "mean ± sample SD; WER, lower is better).",
        body=hk.labeled_asset(
            table(
                ["System", "test-clean", "test-other", "Rate", "Budget"],
                [
                    (["learned Transformer-AR local64 + BiGRU", "2.79 ± 0.02", "5.63 ± 0.12", "≈10.9 Hz", "24k steps"], None),
                    (["learned CNN first-order AR + BiGRU", "2.75 ± 0.05", "5.67 ± 0.10", "≈11.9 Hz", "24k steps"], None),
                    (["character oracle (mean pooling)", "2.87 ± 0.08", "5.66 ± 0.15", "≈14.6 Hz", "1 pass"], None),
                    (["fixed k=5 (mean pooling)", "4.37 ± 0.10", "7.72 ± 0.18", "10 Hz", "1 pass"], None),
                    (["no downsampling", "4.34 ± 0.47", "7.08 ± 0.19", "50 Hz", "1 pass"], None),
                ],
            ),
            "Table",
            8,
            "LS960 anchors from the completed study. The unequal budgets in the last column "
            "are the caveat this replication's protocol removes (Section 5).",
            title="Completed English results",
        ),
    )

    generated = datetime.date.today().isoformat()
    body += (
        "<p class='note'>Generated %s by "
        "<code>recipes/LibriSpeech/ASR/transformer/make_xlingual_replication_proposal.py</code>. "
        "Encoder cards: <a href='https://huggingface.co/TencentGameMate/chinese-hubert-large'>chinese-hubert-large</a>, "
        "<a href='https://huggingface.co/rinna/japanese-hubert-large'>japanese-hubert-large</a>; "
        "data: <a href='https://www.openslr.org/33/'>AISHELL-1</a>, "
        "<a href='https://huggingface.co/datasets/reazon-research/reazonspeech'>ReazonSpeech</a>, "
        "<a href='https://sites.google.com/site/shinnosuketakamichi/publication/jsut'>JSUT</a>.</p>"
        % generated
    )

    full, _ = hk.document(
        "Cross-lingual 24k-step replication · Proposal", body, mathjax=True
    )
    return full


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="artifacts/segmenter/xlingual_24k_replication_proposal.html",
    )
    args = parser.parse_args()
    html = build(args)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(html)
    print("wrote", args.out, len(html), "bytes")


if __name__ == "__main__":
    main()
