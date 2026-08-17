#!/usr/bin/env python3
"""Build the site-native bilevel segmenter--decoder research analysis.

The empirical values come from ``pilot_bilevel_dynamics.py``.  This generator
keeps the interpretation and the report coupled to the machine-readable pilot
output instead of copying aggregate values by hand.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def equation(tex: str) -> str:
    """Return a MathJax display block without leaking TeX '<' into HTML."""
    return '<div class="equation">\\[\n%s\n\\]</div>' % html.escape(tex.strip())


def table(
    number: int, caption: str, headers: list[str], rows: list[list[str]]
) -> str:
    head = "".join(f"<th>{cell}</th>" for cell in headers)
    body = "".join(
        "<tr>%s</tr>" % "".join(f"<td>{cell}</td>" for cell in row)
        for row in rows
    )
    return f"""
    <div class="table-wrap">
      <table>
        <caption><strong>Table {number}.</strong> {caption}</caption>
        <thead><tr>{head}</tr></thead>
        <tbody>{body}</tbody>
      </table>
    </div>
    """.strip()


def pm(
    value: dict, digits: int = 2, scale: float = 1.0, suffix: str = ""
) -> str:
    return (
        f"{value['mean'] * scale:.{digits}f} ± "
        f"{value['sample_sd'] * scale:.{digits}f}{suffix}"
    )


def build(result: dict) -> str:
    aggregate = result["aggregate"]["all"]
    per_run_rows = []
    policy_names = {
        "bernoulli": "Bernoulli",
        "autoregressive": "First-order AR",
    }
    for run in result["runs"]:
        per_run_rows.append(
            [
                policy_names[run["policy"]],
                html.escape(run["seed"]),
                str(run["inner_ce_epoch"]),
                str(run["outer_wer_epoch"]),
                f"{run['train_ce_reduction_pct']:.1f}%",
                f"+{run['wer_regret_if_inner_selected']:.2f}",
                f"{50.0 * run['rho_span']:.1f} Hz",
            ]
        )

    comparison_table = table(
        2,
        "The current loop and the proposed bilevel view. S is a support batch used "
        "for temporary decoder adaptation; Q is a separate query batch used to judge "
        "the adapted decoder and update the segmenter.",
        ["Part", "Current joint RL", "Bilevel version"],
        [
            [
                "Decoder",
                "Updated by transcript CE on the training stream",
                "Temporarily adapted on support speech for each sampled boundary policy",
            ],
            [
                "Segmenter score",
                "NLL from the decoder before it adapts to the sampled boundaries",
                "Query loss after the temporary decoder adaptation",
            ],
            [
                "Data split",
                "Decoder update and segmenter reward come from the same training stream",
                "The decoder adapts on S; the segmenter is judged on held-out Q",
            ],
            [
                "Discrete boundaries",
                "GRPO score-function update",
                "Keep GRPO; only change when and where its scalar reward is measured",
            ],
        ],
    )

    current_reference = table(
        1,
        "Current WavLM reference systems. WER values are mean ± sample SD over "
        "seeds 3407/3408/3409; lower is better. The local-history Transformer-AR "
        "+ BiGRU system has the lowest replicated test-clean mean, while its "
        "test-other variance remains large.",
        [
            "System",
            "Audio-token frequency",
            "Test-clean WER",
            "Test-other WER",
            "Role in the bilevel study",
        ],
        [
            [
                "Oracle char boundaries + char decoder",
                "14.6 Hz",
                "5.03 ± 0.42",
                "8.07 ± 0.43",
                "Reference for boundary quality; not a learned policy arm",
            ],
            [
                "CNN segmenter · first-order AR · mean pooling · char decoder",
                "11.7 Hz",
                "5.06 ± 0.18",
                "9.55 ± 0.17",
                "Faster, more stable control",
            ],
            [
                "Local-history Transformer AR · mean pooling · char decoder",
                "11.3 Hz",
                "5.67 ± 1.00",
                "9.95 ± 0.65",
                "Matched pooling control",
            ],
            [
                "Local-history Transformer AR · BiGRU residual pooling · char decoder",
                "10.7 Hz",
                "4.92 ± 0.13",
                "9.66 ± 1.18",
                "Primary clean-speech configuration; robustness is not established",
            ],
        ],
    )

    pilot_setup = table(
        3,
        "Retrospective pilot setup. This reuses completed trajectories and does not "
        "train a bilevel optimizer. It is historical evidence from the earlier CNN "
        "study, not a re-analysis of the Transformer-AR + BiGRU runs.",
        ["Item", "Choice"],
        [
            ["Model family", "WavLM CNN segmenter with the best char decoder"],
            ["Policies", "Bernoulli and first-order autoregressive"],
            ["Seeds", "3407, 3408, 3409 for each policy"],
            ["Window", "The 10 joint-RL epochs, epochs 3–12"],
            [
                "Inner signals",
                "Training decoder CE and recorded training reward",
            ],
            ["Outer signals", "Dev-clean WER and dev-clean loss"],
            ["Question", "Do the inner-best and held-out-best epochs agree?"],
        ],
    )

    per_run_table = table(
        4,
        "Run-level checkpoint comparison. WER regret is dev-clean WER at the "
        "minimum-training-CE epoch minus the best dev-clean WER; lower is better.",
        [
            "Policy",
            "Seed",
            "Min train CE epoch",
            "Best dev WER epoch",
            "Train CE drop",
            "WER regret",
            "Dev audio-frequency span",
        ],
        per_run_rows,
    )

    method_table = table(
        5,
        "Bilevel ideas that transfer to this project and ideas that should wait.",
        ["Idea", "Use now?", "Reason"],
        [
            [
                "Support/query split",
                "Yes",
                "Separates boundaries that fit one batch from boundaries that help the decoder generalize",
            ],
            [
                "One-step decoder lookahead",
                "Yes, as a pilot",
                "Tests the post-adaptation question while bounding memory and runtime",
            ],
            [
                "Policy-gradient outer update",
                "Yes",
                "Works with the actual hard boundary sequence and existing GRPO machinery",
            ],
            [
                "CE + OPD distillation inner loss",
                "After the CE pilot",
                "OPD can make the temporary decoder step more stable without changing the outer definition",
            ],
            [
                "Full unrolling or implicit Hessians",
                "Not first",
                "Costly for a LoRA-adapted LLM and unnecessary for testing whether lookahead changes rollout rankings",
            ],
            [
                "DARTS-style soft boundaries",
                "Separate later study",
                "Changes hard mean pooling and adds a soft-to-hard deployment gap",
            ],
        ],
    )

    experiment_table = table(
        6,
        "Revised prospective comparison. All arms reuse the same K=4 rollouts, "
        "initialization, audio-frequency penalty, data batches, and update count. "
        "Only the reward measurement or temporary adaptation scope changes.",
        ["Arm", "Segmenter reward", "What it isolates"],
        [
            [
                "A · Current",
                "Current-decoder NLL on the same training batch",
                "Existing joint-RL procedure",
            ],
            [
                "B · Held-out",
                "Current-decoder NLL on a separate query batch",
                "Effect of the support/query split alone",
            ],
            [
                "C · Decoder lookahead",
                "Query NLL after one temporary decoder step on support speech",
                "Whether decoder adaptation changes which boundary rollout is preferred",
            ],
            [
                "D · Decoder + pooler lookahead",
                "Query NLL after temporarily adapting the decoder and BiGRU residual branch on support speech",
                "Whether the new order-aware pooler must adapt jointly with the decoder",
            ],
        ],
    )

    current_eq = equation(r"""
R_{\mathrm{current}}(a;\theta)
  =-\mathcal{L}_{\mathrm{NLL}}(\theta;a)
   -\lambda_{f}\mathcal{P}_{f}(a),
\qquad f(a)=50N(a)/T\;\mathrm{Hz}.
""")
    bilevel_eq = equation(r"""
\begin{aligned}
\theta^{\star}(\phi)
  &\in \arg\min_{\theta}\;
    \mathbb{E}_{a_S\sim\pi_{\phi}}
    \left[\mathcal{L}^{S}_{\mathrm{inner}}(\theta,a_S)\right], \\
\min_{\phi}\quad
  &\mathbb{E}_{a_Q\sim\pi_{\phi}}
    \left[
      \mathcal{L}^{Q}_{\mathrm{outer}}(\theta^{\star}(\phi),a_Q)
      +\lambda_{f}\mathcal{P}_{f}(a_Q)
    \right].
\end{aligned}
""")
    lookahead_eq = equation(r"""
\begin{aligned}
\theta'_k
  &=\theta-\alpha\nabla_{\theta}
    \mathcal{L}^{S}_{\mathrm{inner}}(\theta,a_{S,k}), \\
R^{\mathrm{lookahead}}_k
  &=-\mathcal{L}^{Q}_{\mathrm{outer}}(\theta'_k,a_{Q,k})
    -\lambda_{f}\mathcal{P}_{f}(a_{Q,k}).
\end{aligned}
""")

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="Research analysis and retrospective pilot on bilevel optimization for a learned speech segmenter and decoder.">
  <title>Bilevel optimization for the segmenter and decoder · JSALT 2026</title>
  <link rel="stylesheet" href="../assets/site.css">
  <script>
    window.MathJax = {{
      tex: {{ inlineMath: [['\\\\(', '\\\\)']], displayMath: [['\\\\[', '\\\\]']] }},
      svg: {{ fontCache: 'global' }}
    }};
  </script>
  <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
</head>
<body>
  <nav class="site-nav" aria-label="Primary navigation">
    <div class="nav-inner">
      <a class="brand" href="../index.html">JSALT 2026 · Downsampling</a>
      <div class="nav-links">
        <a href="../index.html">Home</a>
        <a href="../reports/segmenter-training.html">Training report</a>
        <a href="./" aria-current="page">Analyses</a>
        <a href="../proposals/">Proposals</a>
      </div>
    </div>
  </nav>

  <main>
    <header class="hero">
      <div class="container">
        <p class="eyebrow">Research analysis · updated 17 August 2026</p>
        <h1>Does bilevel optimization fit our segmenter–decoder problem?</h1>
        <p class="lede">Yes as a description of the goal: choose boundaries that help a decoder after it adapts, not only boundaries that score well under its current parameters. The earlier six-run pilot still shows a training-versus-held-out mismatch. The prospective test should now use the WavLM local-history Transformer-AR + BiGRU configuration, while keeping the CNN first-order system as a stable control.</p>
        <div class="button-row">
          <a class="button primary" href="#next-test">Jump to the proposed test</a>
          <a class="button" href="./">← All analyses</a>
          <a class="button" href="../reports/segmenter-training.html">Open training report</a>
        </div>
      </div>
    </header>

    <section class="section">
      <div class="container split">
        <aside class="toc" aria-label="Table of contents">
          <strong>On this page</strong>
          <a href="#verdict">Verdict</a>
          <a href="#current-system">Current system</a>
          <a href="#mapping">Problem mapping</a>
          <a href="#pilot">Retrospective pilot</a>
          <a href="#borrow">What to borrow</a>
          <a href="#opd">Connection to OPD</a>
          <a href="#next-test">Next test</a>
          <a href="#limits">Limits</a>
          <a href="#references">References</a>
        </aside>

        <article class="prose">
          <section id="verdict">
            <p class="eyebrow">Bottom line</p>
            <h2>The formulation applies; the expensive machinery may not be needed</h2>
            <div class="callout">
              <strong>Recommended interpretation</strong>
              <p>Treat decoder adaptation on support speech as the inner problem and query ASR plus the audio-frequency penalty as the outer problem. Keep hard boundary sampling, K=4, and GRPO. For the first test, change the reward to measure query loss after one temporary decoder update.</p>
            </div>
            <p>The important distinction is temporal. The current segmenter is rewarded for boundaries that the <em>current</em> decoder likes. The scientific goal is closer to boundaries that permit a decoder to learn and generalize well. Those rankings can differ.</p>
            <p>This does <strong>not</strong> imply that a full second-order bilevel optimizer is the next move. A support/query split plus a one-step decoder lookahead tests the central idea while preserving the current discrete policy and inference path.</p>
          </section>

          <section id="current-system" class="section">
            <p class="eyebrow">Current empirical reference</p>
            <h2>The clean-speech reference has changed; the robustness conclusion has not</h2>
            {current_reference}
            <div class="callout amber">
              <strong>How this changes the bilevel pilot</strong>
              <p>Use the local-history Transformer-AR + BiGRU system for the main diagnostic because it has the lowest replicated test-clean mean. Do not treat it as an unqualified overall winner: its 1.18-point test-other sample SD is much larger than the CNN first-order control’s 0.17. The pilot should therefore retain the CNN system as a cheaper stability control and report dev-clean and dev-other separately.</p>
            </div>
          </section>

          <section id="mapping" class="section">
            <p class="eyebrow">Problem mapping</p>
            <h2>Our current loop is joint training, not bilevel training</h2>
            <p>Let \\(\\phi\\) be segmenter parameters, \\(\\pi_\\phi\\) the boundary policy, \\(a\\) a hard boundary sequence, and \\(\theta\\) the trainable decoder projection and LoRA parameters. The current sampled reward has the form:</p>
            {current_eq}
            <p>Here \\(f(a)\\) is the decoder-side audio-token frequency: the 50 Hz encoder frame rate multiplied by the fraction of frames retained as pooled tokens. Lower Hz means stronger compression.</p>
            <p>The reward is detached; the segmenter receives a GRPO update, while decoder CE updates \\(\theta\\). Both happen in the same loop. A bilevel statement instead asks how well the decoder performs after adapting to the boundary distribution:</p>
            {bilevel_eq}
            <p><strong>Support and query describe two roles within training.</strong> The support batch is the speech used for the temporary decoder update. The query batch is different training speech used to score that adapted decoder and update the segmenter. They can be two disjoint parts of a minibatch; neither is LibriSpeech test-clean or test-other.</p>
            {comparison_table}
          </section>

          <section id="pilot" class="section">
            <p class="eyebrow">Pilot on completed runs</p>
            <h2>The inner and held-out checkpoints disagree in all six runs</h2>
            {pilot_setup}
            <div class="stat-grid">
              <div class="stat"><strong>0 / 6</strong><span>minimum train-CE epoch also has the best dev-clean WER</span></div>
              <div class="stat"><strong>{pm(aggregate["train_ce_reduction_pct"], 1, suffix="%")}</strong><span>training decoder CE reduction during joint RL</span></div>
              <div class="stat"><strong>+{pm(aggregate["wer_regret_if_inner_selected"], 2)}</strong><span>dev-clean WER points lost by selecting minimum train CE</span></div>
              <div class="stat"><strong>{pm(aggregate["rho_span"], 1, scale=50.0, suffix=" Hz")}</strong><span>within-run dev audio-frequency span</span></div>
            </div>

            <figure class="analysis-figure">
              <img src="../assets/bilevel-inner-outer-dynamics.png" alt="Six joint-RL trajectories showing steadily decreasing training decoder cross entropy but earlier, non-monotonic dev-clean WER minima.">
              <figcaption><strong>Figure 1.</strong> Inner training behavior and held-out behavior across the six matched WavLM runs. White rings mark each run’s best dev-clean WER. The policy is Bernoulli in the top row and first-order autoregressive in the bottom row.</figcaption>
            </figure>

            {per_run_table}
            <div class="callout amber">
              <strong>What the pilot does—and does not—show</strong>
              <p>It shows a stable checkpoint-selection mismatch: training CE and the recorded reward are best at epoch 12 in all six runs, while dev-clean WER is best at epochs 7–9. Selecting by the inner signal costs {pm(aggregate["wer_regret_if_inner_selected"], 2)} WER points on average. Dev audio-token frequency moves by only {pm(aggregate["rho_span"], 1, scale=50.0, suffix=" Hz")}, so simple frequency drift is unlikely to explain the pattern. The epoch-wise rank correlation is weak and noisy ({pm(aggregate["spearman_train_ce_vs_valid_wer"], 2)}), so the stronger claim is disagreement of optima—not systematic anticorrelation.</p>
            </div>
          </section>

          <section id="borrow" class="section">
            <p class="eyebrow">Practical lessons from bilevel work</p>
            <h2>Borrow the objective structure before the Hessians</h2>
            {method_table}
            <p>The closest mechanical precedent is probabilistic bilevel coreset selection: sample a discrete subset, adapt an inner model, evaluate post-adaptation performance, and update selection probabilities with policy gradient. Here the discrete subset becomes a boundary sequence and GRPO remains the outer estimator.</p>
            <p>Short unrolling is an approximation, not a claim that the decoder has converged. Its immediate value is diagnostic: if current-decoder and post-update rewards rank boundary rollouts differently, then adaptation is changing what the segmenter should prefer.</p>
          </section>

          <section id="opd" class="section">
            <p class="eyebrow">Connection to the OPD proposal</p>
            <h2>OPD and bilevel training solve different parts of the loop</h2>
            <p>On-policy prefix distillation changes <em>how the decoder adapts</em> to sampled lower-rate prefixes. Bilevel training changes <em>how the segmenter is judged</em>: by the performance of that adapted decoder on held-out speech. They are complementary rather than competing ideas.</p>
            <p>In a later arm, the inner support loss can be transcript CE plus oracle-char distillation. The outer query loss should remain ASR NLL or CER plus the audio-frequency penalty, so the segmenter is not rewarded merely for making the teacher easy to imitate.</p>
          </section>

          <section id="next-test" class="section">
            <p class="eyebrow">Recommended prospective pilot</p>
            <h2>First isolate held-out scoring from decoder lookahead</h2>
            {experiment_table}
            <p>For each sampled rollout \\(k\\), clone only the trainable decoder projection and LoRA weights, take one support step, evaluate a query batch, then discard the temporary weights:</p>
            {lookahead_eq}
            <ol class="steps">
              <li><strong>Use the current clean-speech reference</strong><p>Start from WavLM features, the local-history Transformer-AR segmenter with a 64-frame causal window, BiGRU residual pooling, and the best char decoder. Reuse the combined K=4 rollout path and the existing audio-frequency band.</p></li>
              <li><strong>Keep K fixed; shorten the run instead</strong><p>Use K=4 in every arm, one temporary adaptation step, and a small matched number of batches before any full three-seed WER study. The first output is rollout-rank agreement between current-decoder and lookahead rewards.</p></li>
              <li><strong>Adapt the decoder before the BiGRU</strong><p>Arm C temporarily updates only the decoder projection and LoRA weights while the BiGRU is fixed. This isolates decoder prefix adaptation. Run Arm D only if Arm C changes rollout rankings; D adds the BiGRU residual branch to the temporary update.</p></li>
              <li><strong>Retain a stable control and harder-speech guardrail</strong><p>Repeat the short diagnostic with the CNN first-order system. Report dev-clean and dev-other separately, and do not advance a setting that improves clean speech by sacrificing dev-other stability.</p></li>
              <li><strong>Advance only on a measurable signal</strong><p>Proceed to three seeds if lookahead changes rollout rankings consistently and improves query NLL without raising audio-token frequency. Use held-out WER only for checkpoint selection and final evaluation.</p></li>
            </ol>
          </section>

          <section id="limits" class="section">
            <p class="eyebrow">Limits</p>
            <h2>This pilot diagnoses a mismatch; it does not establish a cure</h2>
            <ul>
              <li>The comparison is retrospective and follows trajectories produced by ordinary joint RL.</li>
              <li>Epoch summaries mix decoder and segmenter changes, so they cannot identify which component caused each dev change.</li>
              <li>Dev-clean WER is useful for selection but is not the differentiable outer loss used by a prospective method.</li>
              <li>One-step lookahead favors boundaries that support fast adaptation; more steps may produce different rankings.</li>
              <li>A support/query split increases variance and compute, so the gain must be compared at matched updates and reported with runtime.</li>
              <li>The 4.92 test-clean result does not settle harder-speech behavior: the corresponding test-other mean is 9.66 with 1.18 sample SD.</li>
            </ul>
          </section>

          <section id="references" class="section">
            <p class="eyebrow">Primary sources</p>
            <h2>References</h2>
            <ol class="references">
              <li>Franceschi et al. (2018). <a href="https://proceedings.mlr.press/v80/franceschi18a.html">Bilevel Programming for Hyperparameter Optimization and Meta-Learning</a>. ICML.</li>
              <li>Shaban et al. (2019). <a href="https://proceedings.mlr.press/v89/shaban19a.html">Truncated Back-propagation for Bilevel Optimization</a>. AISTATS.</li>
              <li>Lorraine, Vicol, and Duvenaud (2020). <a href="https://proceedings.mlr.press/v108/lorraine20a.html">Optimizing Millions of Hyperparameters by Implicit Differentiation</a>. AISTATS.</li>
              <li>Liu, Simonyan, and Yang (2019). <a href="https://arxiv.org/abs/1806.09055">DARTS: Differentiable Architecture Search</a>. ICLR.</li>
              <li>Zela et al. (2020). <a href="https://openreview.net/forum?id=H1gDNyrKDS">Understanding and Robustifying Differentiable Architecture Search</a>. ICLR.</li>
              <li>Ren et al. (2018). <a href="https://proceedings.mlr.press/v80/ren18a.html">Learning to Reweight Examples for Robust Deep Learning</a>. ICML.</li>
              <li>Zhou et al. (2023). <a href="https://arxiv.org/abs/2301.09880">Probabilistic Bilevel Coreset Selection</a>. arXiv:2301.09880.</li>
              <li>Yang, Gao, and Yuan (2025). <a href="https://proceedings.mlr.press/v258/yang25g.html">Bilevel Reinforcement Learning via the Development of Hyper-gradient without Lower-Level Convexity</a>. AISTATS.</li>
            </ol>
          </section>
        </article>
      </div>
    </section>
  </main>

  <footer class="site-footer">
    <div class="container">JSALT 2026 · Learned dynamic downsampling · Research analysis</div>
  </footer>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("artifacts/segmenter/bilevel_pilot/pilot_results.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "artifacts/segmenter/bilevel_pilot/bilevel-segmenter-decoder.html"
        ),
    )
    args = parser.parse_args()
    result = json.loads(args.results.read_text(encoding="utf-8"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build(result), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
