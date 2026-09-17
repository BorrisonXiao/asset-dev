#!/usr/bin/env python3
"""Build the adaptive-frequency curriculum proposal page.

Usage:
    python make_adaptive_frequency_curriculum_proposal.py \
        --out artifacts/segmenter/adaptive_frequency_curriculum.html
"""

from __future__ import annotations

import argparse
import html
import os
import sys
from pathlib import Path

sys.path.insert(0, "/home/cxiao7/.codex/skills/html-report")
import htmlkit as hk  # noqa: E402


EXTRA_CSS = """
.wrap{max-width:1100px}
a{color:var(--series-sed);text-decoration-thickness:1px;text-underline-offset:2px}
.note{font-size:13px;color:var(--text-secondary)}
.left-table th,.left-table td{text-align:left;vertical-align:top}
.left-table th:first-child,.left-table td:first-child{width:20%}
.matrix th,.matrix td{text-align:left;vertical-align:top}
.matrix tr.recommended{background:var(--band)}
.flow{display:grid;grid-template-columns:1fr 44px 1fr 44px 1fr;gap:8px;align-items:stretch;margin:14px 0}
.flow-box{border:1px solid var(--border);border-radius:12px;padding:13px 14px;background:var(--surface-1)}
.flow-box strong{display:block;font-size:14px;margin-bottom:4px}
.flow-box span{display:block;color:var(--text-secondary);font-size:13px}
.flow-arrow{display:flex;align-items:center;justify-content:center;color:var(--text-muted);font-size:23px}
.phase{display:grid;grid-template-columns:42px 1fr;gap:12px;align-items:start;margin:12px 0}
.phase-num{width:36px;height:36px;border-radius:50%;border:1px solid var(--border-strong);display:flex;align-items:center;justify-content:center;font-weight:600;background:var(--surface-1)}
.phase h3{margin:1px 0 4px;font-size:16px}.phase p{margin:0;color:var(--text-secondary)}
.accept{color:var(--ok);font-weight:650}.reject{color:var(--bad);font-weight:650}
code{font-family:ui-monospace,"SF Mono","Cascadia Code",Consolas,monospace;font-size:.92em}
@media(max-width:900px){.flow{grid-template-columns:1fr}.flow-arrow{height:24px;transform:rotate(90deg)}}
@media(max-width:760px){.wrap{padding:0 18px}.card{overflow-x:auto}}
"""


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def table(headers: list[str], rows: list[list[str]], cls: str = "left-table") -> str:
    head = "".join(f"<th>{esc(item)}</th>" for item in headers)
    body = "".join(
        "<tr%s>%s</tr>"
        % (
            ' class="recommended"' if row and row[0] == "Recommended" else "",
            "".join(f"<td>{cell}</td>" for cell in row),
        )
        for row in rows
    )
    return (
        f'<table class="{cls}"><thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table>"
    )


def caption(kind: str, number: int, text: str) -> str:
    return f'<p class="cap"><strong>{kind} {number}.</strong> {text}</p>'


def build(args: argparse.Namespace) -> None:
    body = f"<style>{EXTRA_CSS}</style>"
    body += hk.hero(
        "Proposal · RL dynamic downsampling",
        "Adaptive frequency curriculum",
        "Let task quality lead, increase compression pressure only while held-out "
        "performance remains inside a quality floor, and stop at the lowest accepted "
        "audio-token frequency rather than guessing one fixed penalty in advance.",
    )

    body += hk.section(
        "0 · Decision summary",
        body=hk.tiles(
            [
                (
                    "Status",
                    "Proposal",
                    "The EMA controller itself has not been run.",
                ),
                (
                    "Starting point",
                    "Quality first",
                    "Character-initialized decoder and segmenter, optionally after OPD.",
                ),
                (
                    "Controller",
                    "Held-out EMA",
                    "Increase pressure when quality holds; back off when it does not.",
                ),
                (
                    "Endpoint",
                    "Last accepted rate",
                    "Stop when the next lower-frequency target fails the quality gate.",
                ),
            ]
        )
        + hk.finding(
            "<b>The proposal is a controller around the existing hard-boundary GRPO loop.</b> "
            "It does not differentiate through the discrete segmentation or replace the "
            "decoder-quality reward. The controller changes the target rate and rate "
            "pressure between short training blocks, using held-out quality to decide "
            "whether to continue.",
            ok=True,
        ),
    )

    body += hk.section(
        "1 · Why a fixed penalty is not enough",
        lead=(
            "The project already has evidence that a single rate weight is not a reliable "
            "proxy for the quality–cost frontier. The page distinguishes measured "
            "precursors from the unrun adaptive controller."
        ),
        body=hk.card(
            table(
                ["Evidence", "What it says", "Design consequence"],
                [
                    [
                        "Initial RL collapse",
                        "An unbounded one-sided rate objective reached a zero-variance absorbing state.",
                        "Keep a floor, bounded target, and collapse diagnostics.",
                    ],
                    [
                        "100× lambda sweep",
                        "With std-normalized GRPO and flat rollout quality, reward-side lambda is inert.",
                        "Adaptive pressure must reach the auxiliary rate-loss path, or normalization must change.",
                    ],
                    [
                        "Coarse Phase C",
                        "Rates moved, but ASR frequency wandered and checkpoint selection landed before policy training.",
                        "Do not call a band run a minimum-sufficient frequency frontier.",
                    ],
                    [
                        "Inner/outer pilot",
                        "Minimum training CE and best held-out WER agreed in 0/6 trajectories.",
                        "Use held-out quality for controller decisions, not training reward alone.",
                    ],
                ],
                cls="matrix",
            )
            + caption(
                "Table",
                1,
                "Prior evidence motivating the proposal. The entries are observations from earlier studies, not results of this controller.",
            ),
            title="Observed failure modes and safeguards",
        )
        + hk.finding(
            "<b>Controller signal:</b> validation WER/NLL is deliberately separated from "
            "the training-side GRPO reward. Earlier notes show that an undertrained decoder "
            "can make the reward improve as the rate falls, even when held-out recognition is "
            "not improving.",
        ),
    )

    flow = """
    <div class="flow">
      <div class="flow-box"><strong>1 · Adapt for quality</strong><span>Keep rate pressure small. Train the decoder on the current prefixes and let the segmenter improve task quality.</span></div>
      <div class="flow-arrow">→</div>
      <div class="flow-box"><strong>2 · Propose less frequency</strong><span>Lower the target by one step and increase pressure with a bounded dual update.</span></div>
      <div class="flow-arrow">→</div>
      <div class="flow-box"><strong>3 · Validate, accept or roll back</strong><span>Use held-out WER/NLL and rate stability. Accept the target or restore the last accepted checkpoint.</span></div>
    </div>
    """
    body += hk.section(
        "2 · Proposed controller",
        lead=(
            "The training loop remains on-policy GRPO with K=4 hard rollouts. A short "
            "controller interval surrounds it: train, validate, then update the rate target "
            "only after the quality gate is evaluated."
        ),
        body=hk.card(
            flow
            + caption(
                "Figure",
                1,
                "Proposed block-level control loop. This is a design schematic, not a measured training trajectory.",
            ),
            title="Performance-first, then controlled compression",
        )
        + hk.equation(
            r"""
\begin{aligned}
R_{\mathrm{train}}
  &= Q_{\mathrm{train}}
   - \lambda_k\,P_f(f_{\mathrm{audio}}, f_k)
   - \mu\,\max(0,d_{95}-d_{\max})
   - \alpha_k D_{\mathrm{boundary}}(\pi,\pi_{\mathrm{char}}), \\
f_{\mathrm{audio}} &= \rho f_{\mathrm{enc}}, \\
P_f(f,f_k) &= \bigl(\max(0,f-f_k)\bigr)^2
              + \eta\bigl(\max(0,f_k-f)\bigr)^2.
\end{aligned}
"""
        )
        + '<p class="note"><code>Q_train</code> is the existing detached task-quality signal (for example, negative transcript NLL). '
        'The controller target <code>f_k</code> is in audio-token Hz; lower Hz means stronger compression. '
        '<code>d_95</code> protects against a small number of excessively long segments. '
        '<code>\u03b1_k</code> can decay so the policy is allowed to move gradually away from the character initialization.</p>'
        + hk.equation(
            r"""
\begin{aligned}
\mathrm{accept}_k
  &= \mathbf{1}\!\left[\operatorname{EMA}(\mathrm{WER}_{\mathrm{dev\mbox{-}other}})_k
      \le \mathrm{WER}_{\mathrm{best}}+\delta\right], \\
\mathrm{accept}_k=1 &: \quad
  f_{k+1}=f_k-\Delta f,\quad
  \lambda_{k+1}=\operatorname{clip}(\gamma_{\uparrow}\lambda_k,0,\lambda_{\max}), \\
\mathrm{accept}_k=0 &: \quad
  \theta\leftarrow\theta_{\mathrm{last\ accepted}},\quad
  \lambda_{k+1}=\gamma_{\downarrow}\lambda_k.
\end{aligned}
"""
        )
        + '<p class="note">Here <code>\u03b4</code> is the allowed WER tolerance, <code>\u0394f</code> is one target-rate step, '
        '<code>\u03b3\u2191 &gt; 1</code> increases pressure after an accepted step, and <code>0 &lt; \u03b3\u2193 &lt; 1</code> eases pressure after a rejected step. '
        'The held-out score controls the schedule; it is not back-propagated through the hard boundary sample.</p>',
    )

    body += hk.section(
        "3 · Initial schedule and acceptance rules",
        lead=(
            "The ladder below is an initial search path, not a claim that these are the "
            "task-optimal rates. The controller may stop early or retain an intermediate target."
        ),
        body=hk.card(
            table(
                ["Block", "Nominal target", "Main purpose", "Advance condition"],
                [
                    [
                        "0 · Quality bridge",
                        "14.5 Hz",
                        "Adapt the decoder to the current or OPD-sampled prefixes while the boundary policy stays anchored.",
                        "Decoder and dev WER are finite and stable.",
                    ],
                    [
                        "1 · First reduction",
                        "12.5 Hz",
                        "Turn on bounded auxiliary rate pressure and retain the best accepted checkpoint.",
                        "Two held-out evaluations inside the quality floor.",
                    ],
                    [
                        "2 · Controlled reduction",
                        "11.0 Hz",
                        "Continue the same joint objective; monitor rate variance and long-segment tails.",
                        "Dev-other WER stable; no collapse signal.",
                    ],
                    [
                        "3 · Lower-rate probe",
                        "10.0 Hz",
                        "Test whether additional compression still preserves task performance.",
                        "Accept only if the predeclared gate passes.",
                    ],
                ],
                cls="matrix",
            )
            + caption(
                "Table",
                2,
                "Illustrative target ladder from the earlier proposal. Frequencies are reader-facing audio-token rates; lower is more compressed.",
            )
        )
        + hk.card(
            table(
                ["Decision", "Rule", "Action"],
                [
                    [
                        "Quality holds",
                        "Dev-other WER EMA remains within \u03b4 of the best accepted value for two checks, with dev-clean as a secondary guard.",
                        '<span class="accept">Accept</span> the target, save checkpoint, lower the next target, and increase pressure modestly.',
                    ],
                    [
                        "Quality degrades",
                        "Dev-other WER exceeds the quality floor or the rate/tail constraint becomes unstable.",
                        '<span class="reject">Rollback</span> to the last accepted checkpoint, ease pressure once, then stop if the retry fails.',
                    ],
                    [
                        "No useful reduction",
                        "The lower target fails while the previous target passes.",
                        "Report the previous target as the last accepted operating point; do not force more downsampling.",
                    ],
                ],
                cls="matrix",
            )
            + caption(
                "Table",
                3,
                "Predeclared controller decisions. A rejected lower target is evidence about the quality–rate frontier, not a reason to continue increasing pressure.",
            ),
            title="Quality gate",
        ),
    )

    body += hk.section(
        "4 · Where the rate signal must enter",
        lead=(
            "The adaptive part should not rely on changing only the scalar multiplier in the "
            "realized-reward channel. That channel is the wrong place to express pressure when "
            "the rollout group has flat quality."
        ),
        body=hk.card(
            table(
                ["Path", "Role", "Recommendation"],
                [
                    [
                        "Reward",
                        "Subtract the realized per-rollout rate penalty before group-relative advantage estimation.",
                        "Keep for auditable realized cost, but do not expect lambda alone to control the gradient in flat groups.",
                    ],
                    [
                        "Auxiliary rate loss",
                        "Apply differentiable pressure to the conditional boundary probabilities along sampled histories.",
                        "Use this path for the adaptive multiplier; it is the path where the weight scales the gradient.",
                    ],
                    [
                        "Held-out controller",
                        "Update target and multiplier between training blocks from dev WER/NLL and measured frequency.",
                        "Use validation only for control decisions, never as a differentiable reward shortcut.",
                    ],
                ],
                cls="matrix",
            )
            + caption(
                "Table",
                4,
                "Recommended separation of optimization and control signals. The auxiliary path is required for an effective adaptive rate weight under std-normalized GRPO.",
            )
            + hk.finding(
                "<b>Implementation constraint:</b> for the full-history Transformer policy, the auxiliary term is a conditional-expectation surrogate along sampled histories, not an exact marginal expected rate. Report it that way and compare it with realized frequency.",
                ok=True,
            ),
        ),
    )

    body += hk.section(
        "5 · Experiment boundary and logging",
        body=hk.card(
            '<div class="phase"><div class="phase-num">1</div><div><h3>One-seed calibration</h3><p>Start from the best character-initialized segmenter and decoder. If OPD is used, finish the frozen-segmenter decoder screen first. Calibrate <code>\u0394f</code>, <code>\u03b4</code>, <code>\u03b3\u2191</code>, and <code>\u03b3\u2193</code> on a short run.</p></div></div>'
            '<div class="phase"><div class="phase-num">2</div><div><h3>One-seed full-data gate</h3><p>Run the staged controller with K=4, hard boundaries, the existing decoder update, and the auxiliary rate channel. Save every accepted and rejected checkpoint.</p></div></div>'
            '<div class="phase"><div class="phase-num">3</div><div><h3>Replication only after success</h3><p>Use the existing gate: improve dev-other by at least 0.3 WER at similar frequency, or keep WER within 0.1 while reducing frequency by at least 15%.</p></div></div>'
            '<div class="phase"><div class="phase-num">4</div><div><h3>Report the frontier honestly</h3><p>Report the last accepted target, the first rejected target, realized Hz, WER, <code>d_95</code>, <code>zero_variance_share</code>, rollback count, and whether selection used a pre- or post-policy checkpoint.</p></div></div>',
            title="Minimal execution plan",
        )
        + hk.card(
            table(
                ["Log every controller interval", "Why it matters"],
                [
                    ["target Hz, realized Hz, rate penalty, auxiliary rate loss", "Separates requested pressure from what the policy actually emitted."],
                    ["dev-clean/dev-other WER and EMA state", "Makes each accept/rollback decision auditable."],
                    ["lambda, target step, accepted/rejected state", "Reconstructs the controller trajectory."],
                    ["d95 segment duration, rho std, zero-variance share", "Catches pathological long segments and GRPO collapse."],
                    ["checkpoint id and trainable phase", "Prevents the Phase-C selection mistake from hiding the policy result."],
                ],
                cls="left-table",
            )
            + caption(
                "Table",
                5,
                "Minimum audit record for a dynamic rate objective. These values are proposed logging requirements, not collected results.",
            )
        ),
    )

    body += hk.section(
        "6 · What would count as success",
        body=hk.finding(
            "<b>Success is not merely a lower rate.</b> The controller succeeds if it finds a "
            "lower accepted audio-token frequency without crossing the held-out quality floor, "
            "and if the first rejected target is reproducible enough to make the stopping point "
            "meaningful across seeds.",
            ok=True,
        )
        + hk.card(
            table(
                ["Outcome", "Interpretation"],
                [
                    ["Quality holds as target decreases", "The current decoder/policy can exploit a lower-rate prefix; continue to the next target."],
                    ["Quality fails immediately after a stable target", "The previous accepted rate is the minimum sufficient point for that seed; retain it."],
                    ["Rate changes but quality is flat and lambda has no effect", "The reward path is saturated; inspect auxiliary loss and rollout diversity before interpreting the rate."],
                    ["Different seeds stop at different targets", "The frontier is noisy; report a distribution or revisit initialization/controller step size."],
                ],
                cls="left-table",
            )
            + caption(
                "Table",
                6,
                "Possible outcomes and the corresponding claim. None should be described as a completed result until the controller is run and audited.",
            )
        ),
    )

    body += (
        '<footer>Proposal generated from the project’s rate-curriculum, lambda-invariance, '
        'and Phase-C analysis records. The adaptive controller is unrun; measured precursor '
        'results are labeled as such.</footer>'
    )
    full, _ = hk.document("Adaptive frequency curriculum proposal", body, mathjax=True)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(full, encoding="utf-8")
    print(f"wrote {output} ({len(full) // 1024} KB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="artifacts/segmenter/adaptive_frequency_curriculum.html",
    )
    build(parser.parse_args())
