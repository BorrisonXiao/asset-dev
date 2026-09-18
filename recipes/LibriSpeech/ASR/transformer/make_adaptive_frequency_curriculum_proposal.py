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
        "reduce the audio-token rate, let both the segmenter and decoder recover task performance, and keep the reduction "
        "only when validation quality recovers.",
    )
    body += '<nav class="jump-links" aria-label="On this page">'
    for anchor, label in [
        ("ls960", "Continue after LS960"),
        ("ema", "What EMA means"),
        ("bilevel", "Connection to bilevel training"),
        ("objective", "Every objective component"),
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
                    ["B · Consolidate performance (proposed)", "Boundary policy, BiGRU, projection and LoRA all continue learning from task quality.",
                     "Verify the restored model and use a short quality-only block with all rate penalties disabled. Remeasure WER and Hz before compression."],
                    ["C · Compress, recover, validate (proposed)", "The segmenter and recognizer remain trainable in both phases. Compression enables rate pressure; recovery removes it.",
                     "Reduce the target a little, let boundary placement and recognition improve together, then accept or roll back using the rate after recovery."],
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
                    ["Inner work", "For each of K=4 rollouts, take a temporary decoder SGD step on support utterances.", "Run real joint updates; during recovery the segmenter also learns, with rate penalties removed."],
                    ["Outer feedback", "Query NLL from disjoint training utterances becomes the segmenter’s GRPO reward; temporary weights are restored.", "Dev WER and its EMA change the target, phase and rate pressure between blocks."],
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
        'In the revised recovery stage, both the boundary policy and recognizer change. '
        'This is joint quality refinement; the earlier bilevel inner step instead held '
        'sampled boundaries fixed while temporarily adapting the decoder.</p>',
    )

    body += section(
        "objective", "4 · The joint objective, component by component",
        '<p><b>The LS960 Transformer-AR recipe already includes the rate penalty inside '
        'the GRPO reward.</b> The separate auxiliary rate loss is zero. The current local '
        'code reproduces this behavior with <code>rate_channel: auto</code>, which resolves '
        'to <code>reward</code> for this policy. '
        'The previous draft introduced an auxiliary route and a boundary anchor without '
        'clearly separating those optional changes from the trained objective.</p>'
        r'<p>Let \(\phi\) denote boundary-policy parameters and \(\theta\) the trainable '
        'BiGRU pooler, projection and decoder LoRA parameters. WavLM and the Llama base '
        r'remain frozen. A batch has utterances \(i\); each has \(K=4\) sampled boundary '
        r'sequences \(b_{ik}\). The controller block is \(s\).</p>'
        + hk.equation(r"""
\begin{aligned}
\mathcal{L}_{\mathrm{total}}
 &= \mathcal{L}_{\mathrm{CE}}
    + \omega\,\mathcal{L}_{\mathrm{GRPO}}(r) \\
 &\quad + \lambda_s^{A}\mathcal{L}_{\mathrm{rate,aux}}
    - \tau H(\pi_\phi).
\end{aligned}
""")
        + hk.card(
            table(
                ["Component", "What it does / where gradients go", "LS960 Transformer-AR setting"],
                [
                    [r"\(\mathcal{L}_{\mathrm{CE}}\)", "Transcript prediction on the greedy hard segmentation. Updates BiGRU, projection and LoRA; does not differentiate through boundary decisions.", "Active, weight 1."],
                    [r"\(\omega\mathcal{L}_{\mathrm{GRPO}}(r)\)", "Uses detached rollout advantages to update the boundary policy. Its reward already combines task quality and any enabled realized-rate penalty.", r"Active; \(\omega=1\) (<code>pg_weight</code>)."],
                    [r"\(\lambda_s^A\mathcal{L}_{\mathrm{rate,aux}}\)", "A separate differentiable rate surrogate acting on boundary probabilities. This is an additional rate-gradient route, not an expansion of the GRPO term.", r"Off: \(\lambda_s^A=0\). Only enable in an explicitly declared alternative arm."],
                    [r"\(-\tau H(\pi_\phi)\)", "An entropy bonus encouraging less certain boundary decisions; its minus sign reflects loss minimization.", r"Off: \(\tau=0\). Keep it off in the first continuation comparison."],
                ],
            )
            + caption("Table", 3, "Components of the implemented joint update, with proposal notation for the rate channels. The boundary policy and pooler are separated by their gradient roles even though both live in the segmenter module."),
        )
        + '<h3>Decoder learning and the task-quality reward</h3>'
        r'<p>Write \(\ell_\theta(i,b)\) for mean teacher-forced negative log-likelihood '
        '(NLL) of utterance i’s valid transcript tokens under segmentation b. Audio-prefix '
        r'positions and text padding are excluded. If \(N_i\) is the number of valid '
        r'target tokens and \(b_i^{g}\) the greedy segmentation, decoder training uses:</p>'
        + hk.equation(r"""
\mathcal{L}_{\mathrm{CE}}
  = \frac{\sum_i N_i\,\ell_\theta(i,b_i^{g})}
         {\sum_i N_i}.
""")
        + '<p>This is a mean over all valid text tokens in the batch. For each sampled '
        'segmentation, the task reward is the negative per-utterance NLL. That sampled '
        'NLL is evaluated without gradients: it scores the segmenter’s action. The '
        'separate CE pass above trains the recognizer.</p>'
        + '<h3>Rate cost inside the GRPO reward</h3>'
        + hk.equation(r"""
\begin{aligned}
r_{ik}
 &= -\ell_\theta(i,b_{ik})
    - \lambda_s^{R} c_s(f_{ik}), \\
c_s(f)
 &= \left[\max\left(0,\frac{f-f_s}{f_{\mathrm{enc}}}\right)\right]^2 \\
 &\quad + \left[\max\left(0,\frac{f_{\min}-f}{f_{\mathrm{enc}}}\right)\right]^2.
\end{aligned}
""")
        + r'<p>\(f_{ik}\) is realized audio-token Hz for a sampled segmentation; '
        r'\(f_s\) is the upper target and \(f_{\min}\) the lower edge of the rate band. '
        r'\(f_{\mathrm{enc}}=50\) Hz is the encoder frame rate. The band cost \(c_s\) '
        'is zero inside the band. This formula uses Hz for readability but divides '
        'by the encoder frequency before squaring, matching the code’s kept-ratio units. '
        'The saved LS960 band is 7.5–12.5 Hz; compression lowers the upper target.</p>'
        r'<p>\(\lambda_s^R\) is the reward-side rate weight. In the saved band-mode '
        'recipe it corresponds to <code>lambda_cap=1.0</code>; '
        '<code>lambda_press</code> and <code>lambda_floor</code> belong to another '
        'rate mode and are inactive here. The implicit first segment is counted once, '
        'plus later boundary events, when computing the token rate.</p>'
        + '<h3>How the reward becomes a boundary-policy loss</h3>'
        + hk.equation(r"""
\begin{aligned}
A_{ik}
 &= \frac{r_{ik}-\bar r_i}{\sigma_i+\epsilon}, \\
\mathcal{L}_{\mathrm{GRPO}}
 &= -\frac{1}{Z}\sum_{(i,k,t)\in V}
       \operatorname{sg}(A_{ik}) \\
 &\qquad\qquad\cdot
       \log \pi_\phi(b_{ikt}\mid h_{ikt}).
\end{aligned}
""")
        + r'<p>\(\bar r_i\) and \(\sigma_i\) are the mean and sample standard deviation '
        r'across the four rewards for the same utterance; \(\epsilon=10^{-6}\) prevents '
        r'division by zero. \(\operatorname{sg}\) means stop-gradient. '
        r'\(h_{ikt}\) contains acoustic features and sampled boundary history; '
        r'\(b_{ikt}\) is the binary boundary action at frame \(t\). '
        r'\(V\) contains valid stochastic decisions across the batch and rollouts, '
        r'and \(Z=|V|\); padding and the forced first-frame action are excluded. '
        'Thus the policy loss averages over valid decisions, as the implementation does.</p>'
        '<p>A rollout’s rate cost changes its reward, then its normalized advantage, '
        'then the gradient on its boundary decisions. No separate additive rate term '
        'is needed for that route. This recipe makes one on-policy score-function update; '
        'there is no reference-policy KL or clipped likelihood-ratio term in this default.</p>'
        + hk.card(
            table(
                ["Rate-channel choice", "Reward / separate rate loss", "Use in this proposal"],
                [
                    [r"<code>reward</code>: \(\lambda_s^R&gt;0,\ \lambda_s^A=0\)", "Reward is negative NLL minus the realized band cost. No auxiliary rate loss.", "Primary continuation: retain the existing LS960 route and adapt the target / compression–recovery schedule."],
                    [r"<code>aux</code>: \(\lambda_s^R=0,\ \lambda_s^A&gt;0\)", "GRPO reward is task quality alone. Add a rate surrogate directly to the loss.", "Separate compression arm if direct control of rate-gradient strength is useful."],
                    [r"<code>both</code>: \(\lambda_s^R&gt;0,\ \lambda_s^A&gt;0\)", "Rate affects normalized policy advantages and a separate differentiable gradient.", "An explicit combined objective; not the default continuation and not silently enabled."],
                    [r"Quality / recovery: \(\lambda_s^R=\lambda_s^A=0\)", "Reward is negative NLL; every rate-loss contribution is zero.", "Segmenter, BiGRU, projection and LoRA keep learning from task quality."],
                ],
            )
            + caption("Table", 4, "Exactly where rate enters. The R/A weights are explanatory notation: the code selects routes with rate_channel and uses the band helper’s lambda_cap; it does not already provide an independent two-weight adaptive scheduler."),
        )
        + r'<p>For the auxiliary option, \(\mathcal{L}_{\mathrm{rate,aux}}\) averages '
        r'\(c_s(\widehat f_{ik})\) across utterances and rollouts. '
        r'\(\widehat f_{ik}\) replaces later hard boundary events with their conditional '
        'probabilities along the sampled history, includes the implicit first segment, '
        'and converts the resulting count to Hz. This is a differentiable surrogate '
        'for the autoregressive policy, not an exact marginal expected rate.</p>'
        '<p><b>Why the earlier draft mentioned the auxiliary route:</b> when task quality '
        'is identical across a rollout group, multiplying all rate costs by a positive '
        'constant largely cancels during standard-deviation normalization (apart from '
        'the numerical epsilon). Changing the band target can still change rollout '
        'preferences, and the multiplier matters when quality varies. Use the reward '
        'route as the matched baseline; test the auxiliary route separately rather than '
        'assuming a reward-weight ramp always controls gradient strength.</p>'
        + hk.card(
            r'<p><b>\(D_{\mathrm{boundary}}\) meant an optional boundary-policy anchor.</b> '
            'One possible definition is the mean KL divergence between the current '
            'and fixed LS960 Bernoulli boundary distributions, evaluated on the same '
            'features and sampled histories. It penalizes changing the old policy’s '
            'boundary probabilities; it does not penalize token count directly.</p>'
            '<p>That was an extra proposal, not a component of the trained LS960 '
            'objective or an inherent part of this GRPO implementation. Its exact '
            'estimator was not specified in the earlier draft. The revised primary '
            'objective omits it, equivalently giving it weight zero, so the segmenter '
            'can refine boundary allocation freely. Any future reference-policy '
            'regularizer should be a separately defined ablation.</p>',
            title="What was D_boundary?",
        )
        + '<p class="note">Implementation evidence: '
        + link(f"{SOURCE}/recipes/LibriSpeech/ASR/transformer/train_speechllm_with_segmenter.py",
               "published LS960 joint update and decoder losses")
        + ' · '
        + link(f"{SOURCE}/recipes/LibriSpeech/ASR/transformer/segmenter.py",
               "band penalties, group advantages and policy-gradient reduction")
        + '. The configurable route selector and full-AR auxiliary option are in the '
        'current local working code; the published training source records the earlier '
        'reward-only full-AR path. The adaptive controller remains a proposal.</p>',
    )

    flow = """
    <div class="flow">
      <div class="flow-box"><strong>1 · Establish quality</strong><span>Load LS960, evaluate it, and refine the segmenter and recognizer with task quality alone.</span></div>
      <div class="flow-box"><strong>2 · Reduce the target</strong><span>Lower the target by one small step. Enable the declared rate-penalty route while both models learn.</span></div>
      <div class="flow-box"><strong>3 · Recover performance</strong><span>Remove all rate penalties. Keep the segmenter trainable so boundary placement and recognition improve together.</span></div>
      <div class="flow-box"><strong>4 · Validate and decide</strong><span>Measure WER and actual Hz after recovery. Accept a retained reduction and repeat from step 2, or retry / restore.</span></div>
    </div>
    """
    body += section(
        "controller", "5 · Recovery keeps the segmenter learning",
        hk.card(
            flow + caption("Figure", 1, "The proposed continuation alternates compression pressure with joint quality refinement. The boundary policy stays trainable throughout; recovery does not hold the rate fixed."),
        )
        + '<p><b>Recovery removes the rate objective, not the segmenter’s task gradient.</b> '
        'Keep K=4 and the 64-frame history. The recognizer still learns from CE, and '
        'the segmenter still learns from the relative task quality of its sampled '
        'boundaries. With both rate coefficients and the existing entropy coefficient '
        'set to zero, the recovery objective is:</p>'
        + hk.equation(r"""
\begin{aligned}
r_{ik}^{\mathrm{recovery}} &= -\ell_\theta(i,b_{ik}), \\
\mathcal{L}_{\mathrm{recovery}}
  &= \mathcal{L}_{\mathrm{CE}}
   + \omega\mathcal{L}_{\mathrm{GRPO}}
       (r^{\mathrm{recovery}}).
\end{aligned}
""")
        + '<p>The segmenter can move, add or remove boundaries in this phase. Task '
        'quality may improve by reallocating the same number of boundaries, or by '
        'restoring more audio tokens. Consequently, removing rate pressure does not '
        'guarantee fixed-frequency allocation. Keep the target as an evaluation '
        'criterion and measure the resulting rate after recovery. Accept only a '
        'quality-preserving reduction that survives that measurement.</p>'
        '<p>Give compression and recovery finite update budgets. In band mode, setting '
        'the active rate weight to zero removes both upper- and lower-band penalties; '
        'if an auxiliary route is enabled, disable it too. Rate/duration/diversity '
        'emergency checks remain external stop criteria. There is no hidden rate loss '
        'or boundary anchor left in the quality-only phase. If all K task rewards '
        'are identical, the GRPO task gradient is zero; an unfrozen policy needs '
        'informative rollout differences to learn useful allocation.</p>'
        + hk.card(
            table(
                ["Observation", "Controller action"],
                [
                    ["The new target is not yet reached and quality is acceptable", "Continue the bounded compression block using the selected rate route. Change the upper band target explicitly; do not rely solely on a reward-weight ramp."],
                    ["The lower rate is reached, or quality worsens modestly", "Enter joint recovery: remove every rate penalty and continue updating the segmenter and recognizer."],
                    ["After recovery, raw WER and EMA pass twice and actual Hz remains within the lower target tolerance", "Accept the checkpoint only if its rate is lower than the previous accepted point and the original LS960 reference. Save model, optimizer/scheduler and controller history."],
                    ["Quality recovers but Hz rises beyond the target", "Record a rate rebound, not successful compression. Allow one predeclared gentler compression–recovery retry, or restore the last accepted state."],
                    ["Quality still fails, or an emergency WER/duration/diversity limit is crossed", "Restore the complete accepted state; emergency stops act immediately. Stop this search if the allowed retry also fails."],
                ],
            )
            + caption("Table", 5, "The acceptance rate is measured after quality-only recovery. The target remains a decision criterion while its training penalty is disabled."),
        ),
    )

    body += section(
        "experiment", "6 · Test the continuation against equal extra training",
        '<p>First calibrate one seed from its selected LS960 state. Hold data exposure, '
        'total real optimizer updates, starting weights and validation cadence fixed '
        'across the continuation arms in Table 6. Use all three LS960 training splits. '
        'The static reward-penalty arm helps distinguish the value of the adaptive schedule '
        'from the value of simply adding compression pressure.</p>'
        + hk.card(
            table(
                ["Arm", "What changes after the common LS960 checkpoint", "What it tests"],
                [
                    ["Starting checkpoint", "No further training.", "The original quality and rate reference."],
                    ["Ordinary continuation", "Continue the existing objective for the same extra update budget.", "Whether extra training alone improves WER or rate."],
                    ["Fixed-pressure continuation", "Keep the historical reward-side rate route, with a predeclared lower target and fixed multiplier throughout.", "Whether feedback and recovery improve on a static compression recipe."],
                    ["Adaptive continuation · reward route", "Use the same reward-side rate route during compression; remove it during joint quality recovery.", "Whether scheduling compression and recovery improves the attained quality/rate point."],
                    ["Adaptive continuation · auxiliary route (separate arm)", "During compression use task-only GRPO plus an auxiliary rate loss; remove that loss during joint recovery.", "Whether direct rate-gradient control helps relative to the reward-route controller."],
                    ["Adaptive + decoder lookahead (follow-up)", "Add the earlier support/query decoder-only lookahead within compression blocks.", "Whether post-adaptation rollout rewards add value once rate control is established."],
                ],
            )
            + caption("Table", 6, "Proposed comparisons. Keep the rate route matched in the first static/adaptive pair. Test the auxiliary route separately; add lookahead after a stable controller pilot. Record wall time and GPU memory as well as update counts."),
        )
        + '<p>Keep a fixed dev-other controller set and dev-clean guard. Repeated '
        'controller decisions use validation information, so reserve test-clean/test-other '
        'for the final selected checkpoints after the recipe is fixed. A successful '
        'one-seed pilot should be repeated with all three original seeds.</p>'
        '<p><b>Log each decision:</b> checkpoint and phase, target and realized Hz, '
        'raw WER and EMA, the quality reference, rate channel and every active rate weight, '
        'reward-side and auxiliary costs separately, '
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
        "evidence", "7 · Earlier evidence behind the safeguards",
        hk.card(
            table(
                ["Recorded observation", "Consequence for this proposal"],
                [
                    ["An unbounded one-sided rate objective collapsed to a zero-variance state.", "Use a bounded compression band and keep external duration/diversity checks active during quality-only recovery."],
                    ["A 100× rate-weight sweep was ineffective in flat-quality, std-normalized GRPO groups.", "Do not equate a reward-weight ramp with stronger gradients. Move the target band explicitly and compare an auxiliary-only route separately."],
                    ["Phase-C rates moved, but ASR rate wandered and multi-task checkpoint selection often chose a pre-policy state.", "Track each accepted rate and evaluate the matching checkpoint, including the starting model."],
                    ["Across six retrospective trajectories, minimum training CE and best dev-clean WER selected the same epoch in 0/6 cases.", "Use validation quality to control compression. This mismatch motivates the approach but does not establish that it will improve WER."],
                ],
            )
            + caption("Table", 7, "Measured precursors from the project records. None is a result of the adaptive continuation proposed on this page."),
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
