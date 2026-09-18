#!/usr/bin/env python3
"""Build the adaptive-frequency continuation proposal page."""

from __future__ import annotations

import argparse
import html
import json
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
        "Proposal · Revised 18 September 2026",
        "Adaptive frequency curriculum",
        "Continue learning from the trained LS960 checkpoint. Lower the rate target when validation quality permits; "
        "otherwise hold the target and keep learning with the same rate penalty. End at the maximum update count.",
    )
    body += '<nav class="jump-links" aria-label="On this page">'
    for anchor, label in [
        ("ls960", "Continue after LS960"),
        ("ema", "What EMA means"),
        ("bilevel", "Connection to bilevel training"),
        ("objective", "Every objective component"),
        ("controller", "The proposed loop"),
        ("schedule", "Concrete experiment plan"),
        ("experiment", "How to test it"),
    ]:
        body += f'<a href="#{anchor}">{label}</a>'
    body += "</nav>"

    body += section(
        "summary", "0 · Proposal in brief",
        hk.tiles([
            ("Status", "Implemented", "Seed-3407 pilot; no full-run outcome is available yet."),
            ("Starting point", "LS960", "Continue from the validation-selected Transformer-AR + BiGRU checkpoint."),
            ("Quality signal", "EMA", "Exponential moving average: a smoothed history of validation WER."),
            ("Stop", "8,000 updates", "Select the lowest measured rate that passes the quality checks; keep LS960 as fallback."),
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
    if args.launch_record:
        jobs = json.loads(Path(args.launch_record).read_text())
        items = "".join(
            f'<li><code>{esc(job["arm"])}</code>: Slurm <code>{esc(job["job_id"])}</code>, '
            f'seed {esc(job["seed"])}, full LS960.</li>' for job in jobs
        )
        body += hk.card(
            '<p><b>Submitted 18 September 2026.</b> Three 8,000-update runs, one A100 80 GB each. '
            'These are queued/running experiments, not completed results.</p>'
            f'<ul>{items}</ul>'
            '<p>58 scoped CPU tests passed. GPU check <code>1799880</code> completed four '
            'real updates, both dev checks, a process restart at update 2 and the maximum-step stop. '
            'It used only 16 examples per dev set for integration testing; its WER is not a corpus result. '
            'Production uses both full dev sets. Policy, BiGRU, projection and LoRA tensors all changed '
            'in the check; the matched optimizer moments and reduced LRs were retained.</p>'
            '<p>Outputs: <code>results/speechllm_ls960_adaptive_frequency_20260918/</code> '
            'under the Transformer recipe. Each arm saves <code>frequency_events.jsonl</code>, '
            '<code>selected_checkpoint.json</code> and all validation checkpoints. '
            'The submission root contains the job manifest, logs and frozen runtime sources.</p>',
            title="Pilot launch record",
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
                    ["B · Measure the starting checkpoint", "No updates during this validation.",
                     "Measure full dev-clean and dev-other. Fix the WER limits and initialize EMA from these measurements."],
                    ["C · Continue learning", "Boundary policy, BiGRU, projection and LoRA all learn. The rate penalty stays on.",
                     "Lower the target only after quality and measured rate pass twice. Otherwise hold it and keep training until 8,000 new updates."],
                ],
            )
            + caption("Table", 1, "Placement of the continuation. Stage A supplies the matched model and optimizer; the pilot does not yet establish a quality or rate gain."),
        )
        + '<p><b>Start the rate schedule where the checkpoint is.</b> The earlier '
        '14.5 → 12.5 → 11.0 → 10.0 Hz ladder assumed character-based initialization. '
        'For this continuation, start at the measured validation rate. If it is 11.0 Hz, '
        'an illustrative search is 11.0 → 10.5 → 10.0 → 9.5 Hz. A 0.5 Hz step is a '
        'pilot setting to calibrate, not a measured optimum. Section 6 gives the exact '
        'implemented schedule, including the lowest target and stopping rule.</p>'
        '<p>Use a small continuation learning rate with an explicit new update budget and '
        'counter. Restore the matched optimizer moments, then set the smaller constant '
        'learning rates. Record the source step 24,000 separately and count new updates from zero. '
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
        'at distinct training checkpoints inside the quality limit before lowering the target. '
        'The measured rate must also reach the current target at both checks. '
        'The pilot allowance is <b>0.1 absolute WER percentage points</b>; '
        'for example, a 5.0% reference permits at most 5.1%. A failing raw score or EMA '
        'holds the target; it does not end training. These defaults require pilot calibration.</p>'
        '<p>For the first pilot, keep the limit fixed at the initial checkpoint’s '
        'validation score plus the allowance. Do not grant a fresh degradation allowance '
        'at every rate step. Tightening the limit after confirmed improvements can be '
        'tested later. Check dev-clean separately as a secondary guard. There is no '
        'rollback during learning. Save EMA and controller history for interrupted-run recovery. '
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
                    ["What is being chosen?", "Which sampled boundary rollout works after decoder adaptation?", "Whether to lower the current rate target or hold it."],
                    ["Inner work", "For each of K=4 rollouts, take a temporary decoder SGD step on support utterances.", "Run real joint updates, including during recovery. The current rate penalty remains on."],
                    ["Outer feedback", "Query NLL from disjoint training utterances becomes the segmenter’s GRPO reward; temporary weights are restored.", "Dev WER, its EMA and measured Hz decide when to lower the target."],
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
        'hold/lower decision, while decoder lookahead evaluates boundary placements inside '
        'each training batch. First test the controller with the existing joint '
        'GRPO update. Then add decoder-only lookahead as a separate arm if the '
        'controller is stable; the earlier pilot favors that scope for its lower cost. '
        'In the revised recovery stage, both the boundary policy and recognizer change. '
        'This is joint learning at the held rate target; the earlier bilevel inner step instead held '
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
                    [r"\(\omega\mathcal{L}_{\mathrm{GRPO}}(r)\)", "Multiply each sampled boundary decision’s log probability by its detached, normalized rollout reward. The explicit Bernoulli loss and its logit derivative appear below.", r"Active; \(\omega=1\) (<code>pg_weight</code>)."],
                    [r"\(\lambda_s^A\mathcal{L}_{\mathrm{rate,aux}}\)", r"Compute soft token count \(1+\sum_t p_{ikt}\), convert it to Hz, and apply the band penalty. Backpropagate that penalty through the probabilities \(p_{ikt}\); the sampled 0/1 cuts stay fixed.", r"Off: \(\lambda_s^A=0\). The equations below define this optional additional objective."],
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
\begin{aligned}
q_{\theta,in}(b)
 &= P_\theta(y_{in}\mid y_{i,<n},x_i,b), \\
\ell_\theta(i,b)
 &= -\frac{1}{N_i}\sum_{n=1}^{N_i}
      \log q_{\theta,in}(b), \\
\mathcal{L}_{\mathrm{CE}}
 &= -\frac{1}{\sum_i N_i}
      \sum_i\sum_{n=1}^{N_i}\log q_{\theta,in}(b_i^g).
\end{aligned}
""")
        + r'<p>\(x_i\) is the speech input, \(y_{in}\) the nth target token, '
        r'and \(y_{i,&lt;n}\) the preceding reference tokens. '
        r'\(q_{\theta,in}(b)\) is the recognizer’s probability of the correct token '
        'given that segmentation.</p>'
        + '<p>This is a mean over all valid text tokens in the batch. For each sampled '
        'segmentation, the task reward is the negative per-utterance NLL. That sampled '
        'NLL is evaluated without gradients: it scores the segmenter’s action. The '
        'separate CE pass above trains the recognizer.</p>'
        + '<h3>Rate cost inside the GRPO reward</h3>'
        + hk.equation(r"""
\begin{aligned}
p_{ikt}
 &= \pi_\phi(b_{ikt}=1\mid x_i,b_{ik,<t}) \\
 &= \sigma(z_{ikt})=\frac{1}{1+\exp(-z_{ikt})}, \\
f_{ik}
 &= \frac{f_{\mathrm{enc}}}{T_i}
       \left(1+\sum_{t=2}^{T_i}b_{ikt}\right), \\
r_{ik}
 &= -\ell_\theta(i,b_{ik})
    - \lambda_s^{R} c_s(f_{ik}), \\
c_s(f)
 &= \left[\max\left(0,\frac{f-f_s}{f_{\mathrm{enc}}}\right)\right]^2 \\
 &\quad + \left[\max\left(0,\frac{f_{\min}-f}{f_{\mathrm{enc}}}\right)\right]^2.
\end{aligned}
""")
        + r'<p>\(z_{ikt}\) is the boundary logit from the current policy given the '
        r'sampled history. \(p_{ikt}\) is its probability of opening a new segment, '
        r'and \(b_{ikt}\in\{0,1\}\) is the sampled decision. '
        r'\(T_i\) counts all valid encoder frames, including the first; frame 1 '
        'already opens the implicit initial segment, so only frames 2 onward '
        r'contribute new boundary decisions. \(f_{ik}\) is realized audio-token Hz; '
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
        + '<h3>Expanded GRPO: group statistics, Bernoulli loss and gradient</h3>'
        + hk.equation(r"""
\begin{aligned}
\bar r_i &= \frac{1}{K}\sum_{k=1}^K r_{ik}, \\
\sigma_i
 &= \sqrt{\frac{1}{K-1}\sum_{k=1}^K(r_{ik}-\bar r_i)^2}, \\
A_{ik}
 &= \frac{r_{ik}-\bar r_i}{\sigma_i+\epsilon}.
\end{aligned}
""")
        + r'<p>The mean and sample standard deviation use the four rewards of the '
        r'same utterance. \(\epsilon=10^{-6}\). A positive advantage means this '
        'sampled segmentation scored better than the others in its group.</p>'
        + hk.equation(r"""
\begin{aligned}
&\mathcal{L}_{\mathrm{GRPO}}
 = -\frac{1}{Z}\sum_{(i,k,t)\in V}
       \operatorname{sg}(A_{ik}) \\
 &\quad\cdot\bigl[b_{ikt}\log p_{ikt} \\
 &\qquad +(1-b_{ikt})\log(1-p_{ikt})\bigr], \\
&Z = K\sum_i(T_i-1).
\end{aligned}
""")
        + r'<p>\(\operatorname{sg}\) means stop-gradient. \(V\) contains every '
        'valid stochastic boundary decision across utterances and the K rollouts; '
        'padding and the forced first-frame action are excluded. The bracket '
        'is the log probability of a Bernoulli action: it selects log p for a cut '
        'and log(1−p) for no cut. The actual code computes the negative of this '
        'bracket with binary cross-entropy with logits, multiplies by the detached '
        'advantage, and averages over valid decisions.</p>'
        + hk.equation(r"""
\begin{aligned}
&\frac{\partial\,[\omega\mathcal{L}_{\mathrm{GRPO}}]}
     {\partial z_{ikt}}
 \\
&= \frac{\omega}{Z}\operatorname{sg}(A_{ik})
     (p_{ikt}-b_{ikt}).
\end{aligned}
""")
        + '<p>For a positive advantage, gradient descent raises the probability '
        'of the action that was sampled; for a negative advantage, it lowers it. '
        'The same rollout advantage weights every valid decision in that rollout. '
        'The reward, its mean and standard deviation, and the sampled history '
        'are all held fixed during this backward pass.</p>'
        + '<p><b>This is how rate already acts through GRPO:</b> the realized '
        'count changes the band cost, which changes the reward and advantage, '
        'which changes the gradient above. There is no direct derivative through '
        'the hard count. This is the recipe’s one-update on-policy loss; it has '
        'no clipped likelihood ratio or reference-policy KL term.</p>'
        + '<h3>Expanded auxiliary loss: penalize a soft count directly</h3>'
        + hk.equation(r"""
\begin{aligned}
\widehat f_{ik}
 &= \frac{f_{\mathrm{enc}}}{T_i}
       \left(1+\sum_{t=2}^{T_i}p_{ikt}\right), \\
\mathcal{L}_{\mathrm{rate,aux}}
 &= \frac{1}{BK}\sum_{i=1}^B\sum_{k=1}^K
       c_s(\widehat f_{ik}).
\end{aligned}
""")
        + r'<p>\(B\) is batch size. Compare \(\widehat f_{ik}\) with the realized '
        r'rate \(f_{ik}\): each hard cut \(b_{ikt}\) has been replaced by its '
        r'probability \(p_{ikt}\), while the initial segment still contributes 1. '
        r'The band function \(c_s\) is exactly the squared out-of-band function '
        'defined above. Thus this loss averages a penalty on soft counts; it '
        'does not use the rollout advantage.</p>'
        + hk.equation(r"""
\begin{aligned}
&\lbrack a\rbrack_+ := \max(0,a), \\
&c_s'(f)
 = \frac{2}{f_{\mathrm{enc}}^2} \\
&\quad\cdot\left([f-f_s]_+-[f_{\min}-f]_+\right), \\
&\frac{\partial\,(\lambda_s^A\mathcal{L}_{\mathrm{rate,aux}})}
     {\partial z_{ikt}}
 \\
&= \frac{\lambda_s^A}{BK}\,
      c_s'(\widehat f_{ik})\,
      \frac{f_{\mathrm{enc}}}{T_i}\,
      p_{ikt}(1-p_{ikt}).
\end{aligned}
""")
        + '<p>If the soft rate is above the upper target, this derivative is '
        'positive, so gradient descent reduces boundary logits and their cut '
        'probabilities. Below the lower bound it has the opposite sign; inside '
        'the band it is zero. This pressure depends on the soft rate and sigmoid '
        'slope, rather than on whether a particular sampled segmentation improved '
        'recognition. Its coefficient directly scales the gradient.</p>'
        '<p><b>Why call it a surrogate?</b> In an autoregressive policy, each '
        'probability depends on earlier sampled cuts. The auxiliary backward pass '
        'holds those histories fixed; it does not include how changing the policy '
        'changes the distribution of histories. Also, a nonlinear penalty on a '
        'soft count is generally different from the expected penalty on a sampled '
        'hard count. It is a useful additional objective to test, not an algebraic '
        'expansion of the GRPO estimator.</p>'
        '<details><summary>Implementation caveat for the optional auxiliary arm</summary>'
        r'<p>The equations use all \(T_i\) acoustic frames for duration, just as '
        'the realized-rate reward does. The current local '
        '<code>_full_ar_rate_loss</code> instead passes the action-valid mask to '
        '<code>rate_loss</code>. That mask excludes the first frame; the helper '
        r'therefore divides by \(\max(T_i-1,1)\), not \(T_i\). Its current soft '
        r'rate and derivative use that denominator in place of \(T_i\) above.</p>'
        '<p>A CPU check with 100 frames and probability 0.25 at every later frame '
        'gives 12.875 Hz with the acoustic mask, versus 13.005 Hz through the local '
        'auxiliary wrapper. With an upper edge of 10 Hz and weight 1, the penalties '
        'are 0.00330625 and 0.00361213, respectively. Correct the duration mask '
        'and add a regression test before running this optional arm. This report '
        'does not change training code. The historical LS960 reward-only update '
        'and the proposed primary continuation are unaffected because their '
        'auxiliary coefficient is zero.</p></details>'
        + '<h3>Entropy, also written out</h3>'
        + hk.equation(r"""
H(\pi_\phi)
 = -\frac{1}{Z}\sum_{(i,k,t)\in V}
     \left[p_{ikt}\log p_{ikt}
          +(1-p_{ikt})\log(1-p_{ikt})\right].
""")
        + r'<p>The joint loss subtracts \(\tau H\), so a positive coefficient '
        'would encourage uncertain boundary probabilities. The saved LS960 '
        r'recipe and this proposal keep \(\tau=0\).</p>'
        + '<h3>Putting the policy gradients together</h3>'
        + hk.equation(r"""
\begin{aligned}
\frac{\partial\mathcal{L}_{\mathrm{total}}}{\partial z_{ikt}}
 &= \frac{\omega}{Z}\operatorname{sg}(A_{ik})(p_{ikt}-b_{ikt}) \\
 &\quad + \frac{\lambda_s^A}{BK}\,
      c_s'(\widehat f_{ik})\,
      \frac{f_{\mathrm{enc}}}{T_i}\,
      p_{ikt}(1-p_{ikt}).
\end{aligned}
""")
        + '<p>This uses the configured zero entropy coefficient. CE supplies '
        'no derivative through the hard boundary choices; it updates the '
        'recognizer separately. The first term is the existing GRPO update, '
        'whose advantage already includes the reward-side rate penalty. The '
        'second term exists only if the auxiliary option is enabled. Shared '
        'policy parameters receive the sum of these local logit derivatives '
        'backpropagated through the boundary network.</p>'
        + hk.card(
            table(
                ["Rate-channel choice", "Reward / separate rate loss", "Use in this proposal"],
                [
                    [r"<code>reward</code>: \(\lambda_s^R&gt;0,\ \lambda_s^A=0\)", "Reward is negative NLL minus the realized band cost. No auxiliary rate loss.", "Used throughout this continuation, including recovery. Only the upper rate target can change."],
                    [r"<code>aux</code>: \(\lambda_s^R=0,\ \lambda_s^A&gt;0\)", "GRPO reward is task quality alone. Add a rate surrogate directly to the loss.", "Separate compression arm if direct control of rate-gradient strength is useful."],
                    [r"<code>both</code>: \(\lambda_s^R&gt;0,\ \lambda_s^A&gt;0\)", "Rate affects normalized policy advantages and a separate differentiable gradient.", "An explicit combined objective; not the default continuation and not silently enabled."],
                    [r"Recovery in this pilot: \(\lambda_s^R=1,\ \lambda_s^A=0\)", "Keep the same reward-side band penalty at the current lowered target.", "All trainable components keep learning; a failing quality check pauses target reductions only."],
                ],
            )
            + caption("Table", 4, "Exactly where rate enters. The R/A weights are explanatory notation: the code selects routes with rate_channel and uses the band helper’s lambda_cap; it does not already provide an independent two-weight adaptive scheduler."),
        )
        + '<h3>Why a reward-weight ramp may not strengthen GRPO</h3>'
        r'<p>If task quality is identical across a rollout group, let '
        r'\(c_{ik}=c_s(f_{ik})\), with group mean \(\bar c_i\) and sample '
        r'standard deviation \(\sigma_{c,i}\). For \(\lambda_s^R&gt;0\), '
        'substituting the reward into the advantage gives:</p>'
        + hk.equation(r"""
\begin{aligned}
A_{ik}
 &= \frac{-\lambda_s^R(c_{ik}-\bar c_i)}
          {\lambda_s^R\sigma_{c,i}+\epsilon} \\
 &= -\frac{c_{ik}-\bar c_i}
           {\sigma_{c,i}+\epsilon/\lambda_s^R}.
\end{aligned}
""")
        + '<p>The multiplier largely cancels, apart from the numerical epsilon. '
        'That is why increasing the reward-side rate weight is not generally '
        'equivalent to multiplying a rate gradient by that weight. Changing '
        'the band target can still change rollout preferences, and the multiplier '
        'matters when quality varies. Use the reward route as the matched baseline; '
        'test the auxiliary route separately rather than assuming a reward-weight '
        'ramp always controls gradient strength.</p>'
        + hk.card(
            r'<p><b>\(D_{\mathrm{boundary}}\) meant an optional boundary-policy anchor.</b> '
            'To make that suggestion explicit, one possible definition is the mean '
            'conditional Bernoulli KL from the current policy to a fixed LS960 policy:</p>'
            + hk.equation(r"""
\begin{aligned}
p^0_{ikt}
 &= \pi_{\phi_0}(b_{ikt}=1\mid x_i,b_{ik,<t}), \\
D_{\mathrm{boundary}}
 &= \frac{1}{Z}\sum_{(i,k,t)\in V}
       \Bigg[p_{ikt}\log\frac{p_{ikt}}{p^0_{ikt}} \\
 &\qquad +(1-p_{ikt})\log\frac{1-p_{ikt}}{1-p^0_{ikt}}\Bigg].
\end{aligned}
""")
            + r'<p>\(\phi_0\) is the frozen LS960 policy; both policies see the same '
            'features and sampled histories. This estimator holds the histories and '
            'reference probabilities fixed during differentiation. It penalizes '
            'changing the old policy’s boundary probabilities, not token count directly.</p>'
            '<p>That was an extra proposal, not a component of the trained LS960 '
            'objective or an inherent part of this GRPO implementation. Its exact '
            'estimator was not specified in the earlier draft; the formula here '
            'defines a possible future ablation, not an existing implementation. The revised primary '
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
        'reward-only full-AR path. The continuation controller is implemented locally '
        'in <code>adaptive_frequency.py</code> and <code>train_speechllm_adaptive.py</code>; '
        'submitted jobs use frozen source snapshots.</p>',
    )

    flow = """
    <div class="flow">
      <div class="flow-box"><strong>1 · Measure LS960</strong><span>Fix the starting WER limits. Initialize EMA and the current rate target.</span></div>
      <div class="flow-box"><strong>2 · Keep learning</strong><span>Train the segmenter and recognizer with CE and GRPO. Keep the current rate penalty on.</span></div>
      <div class="flow-box"><strong>3 · Check every 500 updates</strong><span>If quality and actual rate pass twice, lower the target by 0.5 Hz. Otherwise hold it.</span></div>
      <div class="flow-box"><strong>4 · Repeat until 8,000</strong><span>Go back to learning. At the maximum update count, finish and select a quality-qualified checkpoint.</span></div>
    </div>
    """
    body += section(
        "controller", "5 · Recovery keeps the segmenter learning",
        hk.card(
            flow + caption("Figure", 1, "One training loop with a target that only decreases. Recovery holds the target, not the boundary locations or the actual measured rate."),
        )
        + '<p><b>Recovery keeps the rate penalty at the current target.</b> '
        'Keep K=4 and the 64-frame history. The recognizer learns from CE, and '
        'the segmenter learns from task quality and the band penalty inside GRPO. '
        'The objective is the same while reducing and while holding:</p>'
        + hk.equation(r"""
\begin{aligned}
r_{ik} &= -\ell_\theta(i,b_{ik}) - c_s(f_{ik}), \\
\mathcal{L}_{\mathrm{hold}}
  &= \mathcal{L}_{\mathrm{CE}}
   + \mathcal{L}_{\mathrm{GRPO}}(r), \\
\lambda_s^R &= 1,\qquad \lambda_s^A=0,\qquad \tau=0.
\end{aligned}
""")
        + '<p>The band cost is defined in Section 4. During a hold, its upper target '
        'does not move. The segmenter can still move, add or remove boundaries, '
        'but rates outside the band still lower the reward. This discourages a '
        'rebound to a higher rate; it is a soft penalty, not a hard token-count constraint.</p>'
        '<p>There is no penalty-off phase, phase timeout, smaller-step retry or '
        'training rollback. WER decides whether to lower the target, never whether '
        'to terminate training. The 8,000-update limit prevents an endless run. '
        'If all K combined rewards are identical, the GRPO gradient is zero; '
        'leaving the policy trainable alone does not guarantee useful changes.</p>'
        + hk.card(
            table(
                ["Observation", "Controller action"],
                [
                    ["Quality and actual rate pass twice", "Lower the upper target by 0.5 Hz, down to a minimum of 7.5 Hz. Keep all training losses active."],
                    ["Raw WER or EMA is outside its limit", "Hold the current target and continue joint task + rate training. Reset the target-advance streak."],
                    ["Quality passes but measured Hz is above the target", "Hold the target. Let the current rate penalty keep working; do not lower the target again yet."],
                    ["The target has reached 7.5 Hz", "Hold it and keep training to the same maximum update count."],
                    ["8,000 new optimizer updates are complete", "End training. Select the lowest measured dev-other Hz among quality-qualified checkpoints, or retain the starting LS960 model."],
                ],
            )
            + caption("Table", 5, "The distinction is lower target versus hold target. Both use the same rate-penalized joint objective."),
        ),
    )

    body += section(
        "schedule", "6 · Concrete experiment plan",
        '<p><b>First run: seed 3407, starting from its selected LS960 checkpoint.</b> '
        'Keep task learning and the existing reward-side rate penalty on throughout. '
        'Do not add the auxiliary rate loss, '
        'boundary KL, entropy bonus or bilevel lookahead in this first run.</p>'
        '<p>The controller and continuation entry point are implemented. Table 6 '
        'lists settings we already used. Table 7 makes the new pilot choices explicit; '
        'their values are not known to be optimal. Full-run results are pending.</p>'
        + hk.card(
            table(
                ["Keep from LS960", "Setting"],
                [
                    ["Training data", "All three splits: <code>train-clean-100</code>, <code>train-clean-360</code>, <code>train-other-500</code>."],
                    ["Model", "Transformer-AR segmenter: 4 layers, width 256, 4 heads, 64-frame history. BiGRU pooler: 1 layer, hidden size 128. Keep the saved projection and LoRA."],
                    ["What learns", "Boundary policy, BiGRU, projection and LoRA in every phase. WavLM and the Llama base stay frozen."],
                    ["Task losses", "Transcript CE plus on-policy GRPO with negative transcript NLL as the task reward; <code>grpo_k=4</code>, <code>grpo_normalize_std=True</code>, <code>pg_weight=1</code>."],
                    ["Rate penalty", "Reward only; select <code>rate_channel=reward</code>, <code>rate_mode=band</code>. The existing enabled weight is <code>lambda_cap=1</code>. Keep the lower band edge at 7.5 Hz (<code>rho_lo=0.15</code>)."],
                    ["Optimizer and batches", "AdamW, weight decay 0; BF16; 300-second dynamic training batches; gradient accumulation 1. One A100 per run."],
                ],
            )
            + caption("Table", 6, "Existing LS960 settings retained for the pilot. Accumulation 1 is confirmed by the runtime log; the saved YAML's value of 4 was overridden by the launcher."),
        )
        + hk.card(
            table(
                ["Pilot setting", "Implemented value and meaning"],
                [
                    ["Learning rates", "Boundary policy + BiGRU: <code>5e-6</code>. Projection + LoRA: <code>2e-5</code>. These are one tenth of the LS960 rates. Keep them constant in every phase; no new LR warmup or decay."],
                    ["Validation", "Evaluate full dev-other and dev-clean before training, then every 500 new optimizer updates. Use greedy boundaries and text decoding, with batch size 8 throughout. Save a checkpoint at each check."],
                    ["EMA", "Use 20% of the new dev-other WER and 80% of the previous EMA (<code>alpha=0.2</code>). Start the EMA at the measured LS960 dev-other WER."],
                    ["Quality limits", "Dev-other raw WER and its EMA must both be at most initial dev-other WER + 0.1 percentage point. Raw dev-clean WER must be at most its initial value + 0.1 point."],
                    ["Rate step", "Lower the upper target by 0.5 Hz; <code>rho_hi</code> = target Hz / 50. Lower band edge and minimum upper target are both 7.5 Hz. A final reduction smaller than 0.5 Hz is clamped to 7.5 Hz."],
                    ["Starting target", "Adaptive: initial measured dev-other Hz, limited to 7.5–12.5 Hz. Original control: 12.5 Hz. Fixed-lower control: adaptive starting target minus 0.5 Hz, no lower than 7.5 Hz."],
                    ["When to lower", "Two consecutive validation checks must meet all quality limits and have measured dev-other Hz at or below the current target. Reset this count after a target change or failed check."],
                    ["Total length", "8,000 additional optimizer updates in every arm. No WER early stop, recovery timeout or retry. If quality never recovers, hold the target and continue to update 8,000."],
                ],
            )
            + caption("Table", 7, "Pilot settings, not measured optima. Maximum updates is the sole normal training stop; non-finite values or runtime failures still require inspection."),
        )
        + '<h3>Exactly how EMA is used</h3>'
        '<p>After each validation, compute the new dev-other EMA by adding '
        '<b>20% of the new WER and 80% of the previous EMA</b>. '
        'The raw dev-clean WER is a separate check; it is not mixed into that average. '
        'Use the initial LS960 WER values for the limits throughout the pilot. '
        'Training loss and training reward do not decide when to change the target.</p>'
        '<p>A <b>passing quality check</b> means all three quality limits in Table 7 '
        'are met. Keep EMA running across all target changes. A separate count '
        'tracks consecutive checks where both quality and rate pass. Reset that '
        'count when the target changes or either check fails; do not reset EMA. '
        'For checkpoint selection, track consecutive quality passes independently, '
        'so changing the target does not disqualify a good measured result. '
        'EMA is a decision check, not a training loss or a model-weight average.</p>'
        + '<h3>Training order and target changes</h3>'
        + hk.card(
            table(
                ["Step", "What runs", "What happens next"],
                [
                    ["0 · Measure the source", "Restore matched segmenter, BiGRU, projection, LoRA, normalization and optimizer moments. Apply the smaller LRs; validate both full dev sets; save step zero.", "Set fixed WER references, EMA and the arm's initial target. The baseline measurement does not count as one of the two passing checks."],
                    ["1 · Train 500 more updates", "CE and reward-only GRPO; rate weight = 1. Boundary policy and recognizer both learn.", "Validate both dev sets. Update EMA. Log raw WER, actual Hz, target and counters."],
                    ["2 · Lower or hold", "For the adaptive arm, lower only after two checks pass quality and rate. Otherwise leave the target unchanged. Fixed controls never change their targets.", "Return to training. The earliest adaptive reduction is at update 1,000. At 8,000, finish without setting a new unused target."],
                    ["3 · Select a checkpoint", "At every check, a checkpoint is eligible after two consecutive quality passes. Save it as selected if its measured dev-other Hz is lower than the previous selection.", "Keep all validation checkpoints and a selection manifest. The initial LS960 model remains the fallback if no qualified rate improvement occurs."],
                ],
            )
            + caption("Table", 8, "The concrete pilot schedule. Hz means 50 times the mean per-utterance kept ratio on dev-other, using greedy boundaries, matching the existing rate reporting. Also log dev-clean Hz, but use dev-other Hz for the rate decisions."),
        )
        + '<h3>“Hold the target” is not “stop training”</h3>'
        '<p><b>If WER leaves the acceptable range, stop lowering the target.</b> '
        'Continue optimizing task quality with the current band penalty still active. '
        'If quality and measured rate later pass twice, lower the target again. '
        'If they never pass, keep training at that target until update 8,000.</p>'
        '<p><b>The whole run normally ends only at 8,000 new updates.</b> '
        'Reaching the 7.5 Hz floor also means hold, not terminate. A technical '
        'failure is not a curriculum decision and must be inspected. There is '
        'no automatic training extension after the update budget is exhausted.</p>'
        + hk.card(
            '<p>Suppose the initial dev-other WER is 5.0%: the quality limit is '
            '5.1%. If the target is 9.5 Hz and WER becomes 5.2%, keep the '
            '9.5 Hz penalty and continue learning. Even 5.35% does not trigger '
            'an automatic WER stop in this pilot. If raw WER later returns to '
            '5.05% but EMA is still 5.12%, keep holding. Once all quality checks '
            'and actual Hz pass twice, try 9.0 Hz. These are examples, '
            'not LS960 measurements.</p>',
            title="Example: recover without removing the rate penalty",
        )
        + '<p>This plan does not wait for task performance to stop improving. '
        'It uses a fixed acceptable quality range. Task learning never switches '
        'off; only the decision to request a lower rate is paused.</p>'
        + '<h3>Run order and checks before launch</h3>'
        '<p>First verify checkpoint loading, both validation sets, target decisions '
        'and restart from a saved checkpoint. Then run the seed-3407 adaptive '
        'pilot and the ordinary/fixed-pressure comparisons in Section 7, using '
        'the same proposed LRs and validation cadence. The fixed-pressure run '
        'keeps the adaptive starting target minus 0.5 Hz for its whole run. '
        'Only after this works should the same settings be repeated with seeds '
        '3408 and 3409. Keep auxiliary rate loss and bilevel lookahead for later tests.</p>'
        '<p>The new entry point is <code>train_speechllm_adaptive.py</code>, with '
        'controller settings in <code>hparams/adaptive_frequency_ls960.yaml</code>. '
        'The launcher restores the matched source optimizer and reapplies the '
        'smaller LRs. <code>warmup_epochs=0</code>, '
        '<code>warmup_optimizer_steps=0</code> and no fractional warmup let the '
        'boundary policy learn from the first update. Each submitted job reads '
        'a frozen source snapshot. Checkpoints save the training-data position, '
        'optimizer, RNG, EMA, target and selection history.</p>',
    )

    body += section(
        "experiment", "7 · Compare against the same extra training budget",
        '<p>Use the same starting weights, learning rates, training-data order and '
        'validation cadence across the continuation arms in Table 9, each with '
        '8,000 new optimizer updates. This matches update counts, not necessarily '
        'wall time: rates can change the compute per update. Use all three LS960 training splits. '
        'The static reward-penalty arm helps distinguish the value of the adaptive schedule '
        'from the value of simply adding compression pressure.</p>'
        + hk.card(
            table(
                ["Arm", "What changes after the common LS960 checkpoint", "What it tests"],
                [
                    ["Starting checkpoint", "No further training.", "The original quality and rate reference."],
                    ["Original fixed band · <code>original</code>", "Keep 7.5–12.5 Hz and reward-side rate weight 1 for all 8,000 updates.", "Whether extra training alone improves WER or rate."],
                    ["Fixed lower target · <code>fixed_lower</code>", "Use the adaptive starting target minus 0.5 Hz (bounded at 7.5 Hz), with weight 1 throughout.", "Whether the adaptive schedule improves on a static lower target."],
                    ["Adaptive target · <code>adaptive</code>", "Start at the measured rate (limited to 7.5–12.5 Hz). Keep weight 1, and lower only when quality and actual rate pass twice.", "Whether feedback improves the attained quality/rate point."],
                ],
            )
            + caption("Table", 9, "Three seed-3407 continuation arms plus the common starting checkpoint. Auxiliary losses and bilevel lookahead are not in this pilot. Record wall time and GPU memory as well as update counts."),
        )
        + '<p>The two fixed-objective controls never change targets. '
        'They share the same maximum update count and do not stop for WER. For each '
        'control, select the lowest-Hz checkpoint whose quality passed at two '
        'consecutive checks within the comparison budget; if none passes, keep '
        'the original LS960 checkpoint. Use validation only for this selection.</p>'
        '<p>Keep a fixed dev-other controller set and dev-clean guard. Repeated '
        'controller decisions use validation information, so reserve test-clean/test-other '
        'for the final selected checkpoints after the recipe is fixed. A successful '
        'one-seed pilot should be repeated with all three original seeds.</p>'
        '<p><b>Log each decision:</b> checkpoint and update count, target and measured Hz, '
        'raw WER and EMA, the quality reference, rate channel and every active rate weight, '
        'reward-side and auxiliary costs separately, '
        'rollout reward spread and <code>zero_variance_share</code>. '
        'The event log records quality passes, target-advance passes and checkpoint selection. '
        'Save every validation checkpoint and its current target '
        'so selection cannot silently report only the starting model.</p>'
        + hk.finding(
            "<b>Success needs a matched comparison.</b> As an initial replication gate, "
            "seek either at least 15% lower realized Hz with dev-other WER within "
            "0.1 point of ordinary continuation, or at least 0.3 point better WER "
            "at comparable Hz, while retaining the starting quality limit and dev-clean "
            "guard. These are proposed decision thresholds, not observed gains.",
            ok=True,
        )
        + '<p><b>Interpret the result locally.</b> If a lower target never passes '
        'before the update budget ends, retain the lowest qualified checkpoint. This identifies '
        'the lowest successful rate for the tested checkpoint, step sizes and adaptation '
        'budget. A different optimizer, longer adaptation or better boundary placements '
        'might still do better; a rejected step cannot establish that all further '
        'downsampling must hurt performance.</p>',
    )

    body += section(
        "evidence", "8 · Earlier evidence behind the safeguards",
        hk.card(
            table(
                ["Recorded observation", "Consequence for this proposal"],
                [
                    ["An unbounded one-sided rate objective collapsed to a zero-variance state.", "Use a rate band, monitor rollout diversity, and keep a qualified-checkpoint fallback. WER failures hold the target rather than ending training."],
                    ["A 100× rate-weight sweep was ineffective in flat-quality, std-normalized GRPO groups.", "Do not equate a reward-weight ramp with stronger gradients. Move the target band explicitly and compare an auxiliary-only route separately."],
                    ["Phase-C rates moved, but ASR rate wandered and multi-task checkpoint selection often chose a pre-policy state.", "Track each accepted rate and evaluate the matching checkpoint, including the starting model."],
                    ["Across six retrospective trajectories, minimum training CE and best dev-clean WER selected the same epoch in 0/6 cases.", "Use validation quality to control compression. This mismatch motivates the approach but does not establish that it will improve WER."],
                ],
            )
            + caption("Table", 10, "Measured precursors from the project records. None is a result of the adaptive continuation proposed on this page."),
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
        '<footer>Adaptive frequency curriculum · Proposal revised 18 September 2026. '
        'LS960 and the short bilevel pilots are completed evidence. The continuation '
        'controller is implemented; full-run quality/rate outcomes are pending.</footer>'
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
    parser.add_argument("--launch-record", default=str(Path(__file__).resolve().parents[4] / "artifacts/segmenter/adaptive_frequency_launch_20260918.json"))
    build(parser.parse_args())
