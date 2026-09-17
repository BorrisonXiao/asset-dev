#!/usr/bin/env python3
"""Build the adaptive-frequency continuation proposal page."""

from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

sys.path.insert(0, "/home/cxiao7/.codex/skills/html-report")
import htmlkit as hk  # noqa: E402


SITE = "https://borrisonxiao.github.io/jsalt26-downsampling"
SOURCE = "https://github.com/BorrisonXiao/asset-dev/blob/bilevel-optimization"
EXTRA_CSS = """
.wrap{max-width:1100px}
h1,h2,h3{text-wrap:wrap}
a{color:var(--series-sed);text-decoration-thickness:1px;text-underline-offset:2px}
.note{font-size:13px;color:var(--text-secondary)}
.matrix th,.matrix td{text-align:left;vertical-align:top}
.flow{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:14px 0}
.flow-box{border:1px solid var(--border);border-radius:12px;padding:13px 14px;background:var(--surface-1)}
.flow-box strong{display:block;font-size:14px;margin-bottom:4px}
.flow-box span{display:block;color:var(--text-secondary);font-size:13px}
.jump-links{display:flex;flex-wrap:wrap;gap:8px 20px;margin:16px 0;font-size:14px}
.table-scroll{overflow-x:auto}
code{font-family:ui-monospace,"SF Mono","Cascadia Code",Consolas,monospace;font-size:.92em;overflow-wrap:anywhere}
section{scroll-margin-top:12px}
@media(max-width:900px){.flow{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(max-width:600px){.wrap{padding:0 16px}.flow{grid-template-columns:1fr}.matrix{min-width:620px}}
"""


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def link(url: str, label: str) -> str:
    return f'<a href="{esc(url)}" target="_blank" rel="noopener">{esc(label)}</a>'


def table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f'<th scope="col">{esc(item)}</th>' for item in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return (
        '<div class="table-scroll"><table class="matrix">'
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"
    )


def caption(kind: str, number: int, text: str) -> str:
    return f'<p class="cap"><strong>{kind} {number}.</strong> {text}</p>'


def section(anchor: str, title: str, body: str, lead: str = "") -> str:
    return hk.section(title, body=body, lead=lead).replace(
        "<section>", f'<section id="{anchor}">', 1
    )


def build(args: argparse.Namespace) -> None:
    body = f"<style>{EXTRA_CSS}</style>"
    body += hk.hero(
        "Proposal · Revised 17 September 2026",
        "Adaptive frequency curriculum",
        "Continue learning from the trained LS960 checkpoint: consolidate task performance, "
        "reduce the audio-token rate, give the decoder time to adapt, and keep the reduction "
        "only when validation quality recovers.",
    )
    body += '<nav class="jump-links" aria-label="On this page">'
    for anchor, label in [
        ("ls960", "Continue after LS960"),
        ("ema", "What EMA means"),
        ("bilevel", "Connection to bilevel training"),
        ("controller", "The proposed loop"),
        ("experiment", "How to test it"),
    ]:
        body += f'<a href="#{anchor}">{label}</a>'
    body += "</nav>"

    body += section(
        "summary", "0 · Proposal in brief",
        hk.tiles([
            ("Status", "Proposed", "The adaptive controller has not been implemented or run."),
            ("Starting point", "LS960", "Continue from the validation-selected Transformer-AR + BiGRU checkpoint."),
            ("Quality signal", "EMA", "Exponential moving average: a smoothed history of validation WER."),
            ("Endpoint", "Last accepted", "The lowest tested rate that passes after a bounded recovery stage."),
        ])
        + hk.finding(
            "<b>Yes, a continuation after LS960 makes sense.</b> The segmenter and decoder "
            "already work together at a useful operating point. This stage can test whether "
            "they can adapt to fewer audio tokens while preserving recognition. The expected "
            "benefit is a better starting point for that search; further improvement remains "
            "an experimental question.",
            ok=True,
        ),
    )

    body += section(
        "ls960", "1 · Insert an adaptive continuation after LS960",
        '<p>The corrected LS960 Transformer-AR local-64 + BiGRU study trained through '
        '24,000 optimizer updates. Its selected checkpoints give '
        '<b>2.79 ± 0.02% / 5.63 ± 0.12% WER</b> on test-clean/test-other at '
        '<b>10.9 ± 0.4 / 10.6 ± 0.4 Hz</b>. These are existing results '
        '(mean ± sample SD across seeds 3407–3409), and establish the starting system. '
        'Audio-token frequency counts decoder-side audio tokens per second; lower Hz means '
        'stronger compression. Lower word error rate (WER) means better recognition. '
        + link(f"{SITE}/reports/segmenter-training.html", "Existing training report")
        + '.</p>'
        '<p>For each seed, restore its matched segmenter, BiGRU pooler, projection and LoRA '
        'weights from the selected LS960 checkpoint. Keep WavLM and the Llama base frozen. '
        'Measure that checkpoint’s dev-clean/dev-other WER and actual audio-token rate '
        'before updating anything. This per-checkpoint validation measurement defines the '
        'quality reference and starting target; the historical test averages above do not '
        'set controller thresholds.</p>'
        + hk.card(
            table(
                ["Stage", "What learns", "Purpose and transition"],
                [
                    ["A · LS960 training (completed)", "Existing decoder adaptation and joint GRPO training.",
                     "Supplies a trained segmenter and recognizer. Preserve this checkpoint as the starting reference."],
                    ["B · Consolidate performance (proposed)", "Brief decoder/projection/LoRA and BiGRU continuation with the boundary policy fixed.",
                     "Verify the restored model and stabilize performance at its current rate. Apply no additional adaptive rate pressure."],
                    ["C · Compress, recover, validate (proposed)", "Joint updates during compression; decoder and pooler updates with the segmenter frozen during recovery.",
                     "Reduce the target a little, let the recognizer catch up, then accept or roll back. Repeat while the quality gate passes."],
                ],
            )
            + caption("Table", 1, "Placement of the new stage. Only Stage A has completed results; B and C are proposed continuation work."),
        )
        + '<p><b>Start the rate schedule where the checkpoint is.</b> The earlier '
        '14.5 → 12.5 → 11.0 → 10.0 Hz ladder assumed character-based initialization. '
        'For this continuation, start at the measured validation rate. If it is 11.0 Hz, '
        'an illustrative search is 11.0 → 10.5 → 10.0 → 9.5 Hz. A 0.5 Hz step is a '
        'pilot setting to calibrate, not a measured optimum. The search must also respect '
        'a declared lower safety bound and segment-duration limits.</p>'
        '<p>Use a small continuation learning rate with an explicit new update budget and '
        'scheduler. The implementation must handle the saved 24,000-step counter and '
        'saved scheduler state deliberately, and verify all restored components. '
        'Continue on all three LS960 training splits: <code>train-clean-100</code>, '
        '<code>train-clean-360</code> and <code>train-other-500</code>. This proposal does '
        'not assume a return to character supervision or require OPD before continuing.</p>',
    )

    body += section(
        "ema", "2 · What EMA means",
        '<p><b>EMA stands for exponential moving average.</b> It combines the newest '
        'validation score with the previous smoothed score. Recent evaluations receive '
        'more weight; older evaluations fade gradually. Here it smooths the WER '
        'measurement used by the rate controller.</p>'
        + hk.equation(r"""
\begin{aligned}
m_0 &= w_0, \\
m_k &= \alpha w_k + (1-\alpha)m_{k-1}.
\end{aligned}
""")
        + r'<p>Here \(w_k\) is the current dev-other WER at evaluation \(k\), '
        r'\(m_k\) is its EMA, and \(w_0\) is the starting checkpoint’s WER. '
        r'The smoothing weight \(0 &lt; \alpha \le 1\) controls responsiveness: a larger value follows '
        'the latest score more closely, while a smaller value smooths more and reacts '
        'more slowly.</p>'
        + hk.card(
            r'<p>With \(\alpha=0.2\), a previous EMA of <b>5.0%</b> and a new WER '
            'of <b>5.5%</b> give a new EMA of <b>5.1%</b>: 20% of the new score plus '
            '80% of the previous average. If the following WER is 5.0%, the EMA becomes '
            '<b>5.08%</b>. These are illustrative numbers.</p>'
            '<p>This dampens one evaluation’s movement before changing rate pressure. '
            'It also shows the delay: the raw 5.5% score is worse than the smoothed 5.1%. '
            'EMA averages validation scores here; it does not average model weights, '
            'and it is not a confidence interval.</p>',
            title="A small numerical example",
        )
        + '<p><b>Use both the raw score and the EMA.</b> Require two consecutive checks '
        'at distinct training checkpoints inside the quality limit before accepting a reduction. Keep a separate raw-WER '
        'stop for a large deterioration, so smoothing cannot hide a harmful step. '
        'A candidate acceptance tolerance is <b>0.1 absolute WER percentage points</b>; '
        'for example, a 5.0% reference permits at most 5.1%. A 0.3-point raw deterioration '
        'is an illustrative emergency stop. These defaults require pilot calibration.</p>'
        '<p>Anchor the limit to the initial checkpoint’s validation score and tighten it '
        'only after a confirmed improvement. Do not grant a fresh degradation allowance '
        'at every rate step. Check dev-clean separately as a secondary guard. After '
        'rollback, restore the saved EMA and controller history from the accepted state. '
        'Use a fixed validation set and evaluation protocol throughout.</p>',
    )

    body += section(
        "bilevel", "3 · How this connects to the bilevel experiments",
        '<p><b>Both ideas let the recognizer adapt before judging compression.</b> A '
        'decoder trained at one rate can initially score a new segmentation poorly '
        'because its input distribution changed. Bilevel lookahead tests adaptation '
        'within each training batch. The proposed curriculum gives the actual model '
        'time to adapt between successive rate decisions.</p>'
        + hk.card(
            table(
                ["Question", "Earlier support/query bilevel method", "Adaptive continuation proposed here"],
                [
                    ["What is being chosen?", "Which sampled boundary rollout works after decoder adaptation?", "When to lower the target rate, increase pressure, recover or stop?"],
                    ["Inner work", "For each of K=4 rollouts, take a temporary decoder SGD step on support utterances.", "Run real training updates, including a recovery block at the proposed lower rate."],
                    ["Outer feedback", "Query NLL from disjoint training utterances becomes the segmenter’s GRPO reward; temporary weights are restored.", "Dev WER and its EMA change the target and auxiliary rate-loss weight between blocks."],
                    ["Time scale", "One temporary adaptation step per rollout and training batch.", "Many optimizer updates between controller decisions."],
                    ["Gradient path", "First-order score-function update; no hypergradient through the inner step or hard boundaries.", "Validation-driven feedback schedule; no differentiation through dev WER or EMA."],
                ],
            )
            + caption("Table", 2, "The two methods control different parts of learning. The bilevel query split comes from training data; the curriculum controller uses a separate validation set."),
        )
        + '<p>The August 17 bilevel pilots ran <b>eight training batches per arm</b>. '
        'Decoder-only lookahead changed the preferred rollout for <b>31.0%</b> of cases '
        'at <b>1.12×</b> the control’s training time; including BiGRU changed '
        '<b>35.3%</b> at <b>1.32×</b>. The temporary updates and parameter restoration '
        'passed their checks. This established a usable ranking signal, but did not '
        'establish a full-data WER improvement. '
        + link(f"{SOURCE}/research/bilevel_segmenter_decoder/PILOT_RESULTS.md", "Bilevel pilot record")
        + ' · '
        + link(f"{SITE}/research/bilevel-segmenter-decoder.html", "Bilevel analysis")
        + '.</p>'
        '<p>The adaptive schedule has an outer feedback loop, so the resemblance is real. '
        'It does not by itself implement the earlier support/query bilevel objective: '
        'its outer decision changes training hyperparameters rather than assigning '
        'post-adaptation rewards to individual rollouts.</p>'
        '<p><b>They can be combined.</b> The curriculum can choose the target rate and '
        'rate pressure, while decoder lookahead chooses boundary placements inside '
        'each compression block. First test the controller with the existing joint '
        'GRPO update. Then add decoder-only lookahead as a separate arm if the '
        'controller is stable; the earlier pilot favors that scope for its lower cost. '
        'The recovery block is a practical connection to bilevel reasoning, not evidence '
        'that the two algorithms are equivalent.</p>',
    )

    flow = """
    <div class="flow">
      <div class="flow-box"><strong>1 · Establish quality</strong><span>Load the LS960 state, evaluate it, and consolidate performance at its measured rate.</span></div>
      <div class="flow-box"><strong>2 · Reduce the target</strong><span>Unfreeze the segmenter. Lower the target by one small step and apply bounded rate pressure.</span></div>
      <div class="flow-box"><strong>3 · Recover performance</strong><span>Hold the candidate boundary policy fixed and adapt the decoder and BiGRU to its shorter prefixes.</span></div>
      <div class="flow-box"><strong>4 · Validate and decide</strong><span>Check raw WER, EMA and realized Hz. Save an accepted state and repeat from step 2, or restore the previous state.</span></div>
    </div>
    """
    body += section(
        "controller", "4 · The proposed compression–recovery loop",
        hk.card(
            flow + caption("Figure", 1, "A proposed continuation loop after LS960. Recovery gives each candidate rate a bounded adaptation opportunity before acceptance. This is a design schematic."),
        )
        + '<p>Keep K=4, the 64-frame Transformer history and hard boundary sampling. '
        'During compression, use joint decoder CE and segmenter GRPO plus an adaptive '
        'auxiliary rate loss. During recovery, freeze the candidate segmenter and '
        'continue decoder/pooler learning, holding the target fixed. Give both blocks '
        'predeclared update budgets so failed targets cannot consume unlimited training.</p>'
        + hk.equation(r"""
\begin{aligned}
\mathcal{L}_{\mathrm{joint}}
 &= \mathcal{L}_{\mathrm{CE}} + \mathcal{L}_{\mathrm{GRPO}}
 \\
 &\quad + \lambda_k^{\mathrm{aux}} P_f(\widehat f,f_k)
 \\
 &\quad + \beta D_{\mathrm{boundary}}(\pi,\pi_0), \\
P_f(f,f_k)
 &= [\max(0,f-f_k)]^2
 \\
 &\quad + \eta[\max(0,f_{\min}-f)]^2.
\end{aligned}
""")
        + r'<p class="note">This is a proposed objective, not implemented controller code. '
        r'\(\mathcal{L}_{\mathrm{CE}}\) adapts the recognizer; '
        r'\(\mathcal{L}_{\mathrm{GRPO}}\) is the existing policy loss. '
        r'\(f_k\) is the target ceiling in Hz, \(f_{\min}\) the safety floor, '
        r'and \(\eta\) its relative penalty. \(\widehat f\) is the auxiliary estimate '
        r'of frequency. \(\lambda_k^{\mathrm{aux}}\) sets its pressure. '
        r'\(D_{\mathrm{boundary}}\) is an optional KL or imitation anchor to the '
        r'loaded LS960 policy \(\pi_0\), weighted by \(\beta\). Choose and log that '
        'anchor explicitly. Check long-segment duration separately; a safe mean Hz '
        'does not rule out pathological gaps.</p>'
        + hk.card(
            table(
                ["Observation", "Controller action"],
                [
                    ["Quality holds, but the realized rate remains above the new target", "Within the compression budget, increase the auxiliary weight modestly up to a declared cap. Initialize it to a small positive value when compression starts."],
                    ["The lower rate is reached, or quality worsens modestly", "Hold the target and enter recovery. Freeze the segmenter so the decoder/pooler can adapt without further boundary drift."],
                    ["After recovery, raw WER and EMA pass twice and the lower target is reached", "Accept only if realized Hz also fell from the previous accepted state, allowing a predeclared target tolerance. Save model, optimizer/scheduler and controller history; then propose the next lower target."],
                    ["Quality still fails after recovery, or an emergency WER/duration/diversity limit is crossed", "Roll back the complete training state; emergency stops act immediately. Permit at most one predefined retry with a smaller target step or gentler pressure; stop the search if it fails."],
                    ["Quality holds, but the rate never fell within the budget", "Record the target as unreached. A quality pass alone cannot count as a compression success."],
                ],
            )
            + caption("Table", 3, "Acceptance combines quality, realized compression and safety. Thresholds and retry budgets are fixed before the run."),
        )
        + hk.card(
            table(
                ["Signal path", "How the adaptive stage uses it"],
                [
                    ["GRPO reward", "Keep the established task-quality reward and declared realized-rate cost. With std normalization and flat quality across rollouts, scaling only the rate reward largely cancels in normalized advantages."],
                    ["Auxiliary rate loss", "Put the adaptive multiplier here so its weight directly scales a differentiable gradient. For an autoregressive policy, the conditional probabilities along sampled histories form a surrogate; they are not exact marginal expected rates."],
                    ["Validation controller", "Raw WER and EMA choose phase, target and pressure between blocks. Dev examples do not supply training gradients; the bilevel query split, if used later, remains inside the training stream."],
                ],
            )
            + caption("Table", 4, "Placement of rate control. Keep the existing GRPO normalization fixed in the first controller comparison and verify that the auxiliary gradient responds to its multiplier."),
        ),
    )

    body += section(
        "experiment", "5 · Test the continuation against equal extra training",
        '<p>First calibrate one seed from its selected LS960 state. Hold data exposure, '
        'total real optimizer updates, starting weights and validation cadence fixed '
        'across the continuation arms in Table 5. Use all three LS960 training splits. '
        'The static rate-loss arm helps distinguish the value of the adaptive schedule '
        'from the value of simply adding compression pressure.</p>'
        + hk.card(
            table(
                ["Arm", "What changes after the common LS960 checkpoint", "What it tests"],
                [
                    ["Starting checkpoint", "No further training.", "The original quality and rate reference."],
                    ["Ordinary continuation", "Continue the existing objective for the same extra update budget.", "Whether extra training alone improves WER or rate."],
                    ["Fixed-pressure continuation", "Use the same auxiliary rate-loss path and a predeclared lower target, with a fixed multiplier and no adaptive schedule.", "Whether feedback and recovery improve on a static compression recipe."],
                    ["Adaptive continuation", "Use the proposed quality → compression → recovery loop.", "Whether the controller finds a better quality/rate operating point."],
                    ["Adaptive + decoder lookahead (follow-up)", "Add the earlier support/query decoder-only lookahead within compression blocks.", "Whether post-adaptation rollout rewards add value once rate control is established."],
                ],
            )
            + caption("Table", 5, "Proposed comparisons. The last arm follows a stable controller pilot. Record wall time and GPU memory as well as update counts, since lookahead adds compute."),
        )
        + '<p>Keep a fixed dev-other controller set and dev-clean guard. Repeated '
        'controller decisions use validation information, so reserve test-clean/test-other '
        'for the final selected checkpoints after the recipe is fixed. A successful '
        'one-seed pilot should be repeated with all three original seeds.</p>'
        '<p><b>Log each decision:</b> checkpoint and phase, target and realized Hz, '
        'raw WER and EMA, the quality reference, auxiliary weight and gradient, '
        '95th-percentile segment duration, rollout reward spread, '
        '<code>zero_variance_share</code>, accept/reject status and rollback count. '
        'Save the lowest accepted checkpoint and the first failed or unreached target '
        'so selection cannot silently report only the starting model.</p>'
        + hk.finding(
            "<b>Success needs a matched comparison.</b> As an initial replication gate, "
            "seek either at least 15% lower realized Hz with dev-other WER within "
            "0.1 point of ordinary continuation, or at least 0.3 point better WER "
            "at comparable Hz, while retaining the starting quality limit and dev-clean "
            "guard. These are proposed decision thresholds, not observed gains.",
            ok=True,
        )
        + '<p><b>Interpret stopping locally.</b> If a lower target fails even after the '
        'allowed recovery and retry, retain the previous accepted state. This identifies '
        'the lowest successful rate for the tested checkpoint, step sizes and adaptation '
        'budget. A different optimizer, longer adaptation or better boundary placements '
        'might still do better; a rejected step cannot establish that all further '
        'downsampling must hurt performance.</p>',
    )

    body += section(
        "evidence", "6 · Earlier evidence behind the safeguards",
        hk.card(
            table(
                ["Recorded observation", "Consequence for this proposal"],
                [
                    ["An unbounded one-sided rate objective collapsed to a zero-variance state.", "Retain a rate floor, duration checks and rollout-diversity diagnostics."],
                    ["A 100× rate-weight sweep was ineffective in flat-quality, std-normalized GRPO groups.", "Drive the adaptive weight through the auxiliary loss and check its actual gradient."],
                    ["Phase-C rates moved, but ASR rate wandered and multi-task checkpoint selection often chose a pre-policy state.", "Track each accepted rate and evaluate the matching checkpoint, including the starting model."],
                    ["Across six retrospective trajectories, minimum training CE and best dev-clean WER selected the same epoch in 0/6 cases.", "Use validation quality to control compression. This mismatch motivates the approach but does not establish that it will improve WER."],
                ],
            )
            + caption("Table", 6, "Measured precursors from the project records. None is a result of the adaptive continuation proposed on this page."),
        )
        + '<p class="note">Provenance: '
        + link(f"{SOURCE}/plans/bilevel_optimization.md", "support/query bilevel plan")
        + ' · '
        + link(f"{SOURCE}/research/bilevel_segmenter_decoder/PILOT_RESULTS.md", "eight-batch pilot")
        + ' · '
        + link(f"{SITE}/research/bilevel-segmenter-decoder.html", "retrospective inner/outer analysis")
        + ' · '
        + link(f"{SITE}/reports/segmenter-training.html", "LS960 and Phase-C report")
        + '. The earlier rate ladder and deferred EMA idea are recorded locally in '
        '<code>reports/one_week_research_strategy_2026-08-23.md</code> and '
        '<code>survey/2026-08-31/projects/non_asr_multitask_plan.md</code> '
        '(with the normalization constraint in ADR-037).</p>',
    )
    body += (
        '<footer>Adaptive frequency curriculum · Proposal revised 17 September 2026. '
        'LS960 and the short bilevel pilots are completed evidence; the EMA controller '
        'and the proposed adaptive continuation have not been run.</footer>'
    )
    full, _ = hk.document(
        "Adaptive frequency curriculum · EMA, bilevel training and LS960 continuation",
        body, mathjax=True,
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(full, encoding="utf-8")
    print(f"wrote {output} ({len(full) // 1024} KB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/segmenter/adaptive_frequency_curriculum.html")
    build(parser.parse_args())
