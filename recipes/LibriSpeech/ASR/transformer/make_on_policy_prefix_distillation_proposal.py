#!/usr/bin/env python3
"""Build the standalone on-policy prefix-distillation proposal.

Usage:
    MPLCONFIGDIR=/tmp/jointllm-mpl-cache \
      /weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/envs/jointllm/bin/python \
      make_on_policy_prefix_distillation_proposal.py \
      --out artifacts/segmenter/on_policy_prefix_distillation.html
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
.flow{display:grid;grid-template-columns:1fr 44px 1fr;gap:10px;align-items:stretch;margin:12px 0}
.flow-col{display:flex;flex-direction:column;gap:10px}
.flow-box{border:1px solid var(--border);border-radius:12px;padding:12px 14px;background:var(--surface-1)}
.flow-box strong{display:block;font-size:14px;margin-bottom:3px}
.flow-box span{display:block;color:var(--text-secondary);font-size:13px}
.flow-arrow{display:flex;align-items:center;justify-content:center;color:var(--text-muted);font-size:24px}
.merge{border:1px solid var(--border-strong);border-radius:12px;padding:14px;background:var(--band);margin-top:10px;text-align:center}
.note{font-size:13px;color:var(--text-secondary)}
.left-table th,.left-table td{text-align:left;vertical-align:top}
.left-table th:first-child,.left-table td:first-child{width:18%}
.matrix th,.matrix td{text-align:left;vertical-align:top}
.matrix tr.recommended{background:var(--band)}
.phase{display:grid;grid-template-columns:42px 1fr;gap:12px;align-items:start;margin:12px 0}
.phase-num{width:36px;height:36px;border-radius:50%;border:1px solid var(--border-strong);display:flex;align-items:center;justify-content:center;font-weight:600;background:var(--surface-1)}
.phase h3{margin:1px 0 4px;font-size:16px}.phase p{margin:0;color:var(--text-secondary)}
code{font-family:ui-monospace,"SF Mono","Cascadia Code",Consolas,monospace;font-size:.92em}
@media(max-width:760px){.flow{grid-template-columns:1fr}.flow-arrow{transform:rotate(90deg);height:26px}.wrap{padding:0 18px}.card{overflow-x:auto}}
"""


def table(headers, rows, cls="left-table"):
    head = "".join("<th>%s</th>" % h for h in headers)
    body = []
    for row, row_cls in rows:
        body.append(
            '<tr%s>%s</tr>'
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


def build(args):
    body = "<style>%s</style>" % EXTRA_CSS
    body += hk.hero(
        "Proposal · RL dynamic downsampling",
        "On-policy prefix distillation",
        "Adapt the decoder to the shorter, variable audio prefixes produced by the "
        "current segmenter, while a frozen oracle-char decoder anchors the text "
        "distribution. The first study freezes the segmenter so the decoder effect is "
        "measured without boundary drift.",
    )

    body += hk.section(
        "0 · Decision summary",
        body=hk.tiles(
            [
                ("Question", "Prefix bias?", "Does the decoder struggle when its audio prefix rate changes?"),
                ("Teacher", "Oracle char", "Frozen WavLM char-aligned decoder"),
                ("Student input", "On-policy", "Sampled WavLM CNN first-order AR prefixes"),
                ("First test", "2 × 2", "Greedy/sample prefixes × CE/CE+KD"),
            ]
        )
        + hk.finding(
            '<b>The proposal makes sense, but its scope should be precise.</b> It can train '
            'the projection and LoRA decoder to handle lower-rate prefixes. It does not '
            'backpropagate through hard segmentation, so GRPO remains responsible for '
            'updating the segmenter.',
            ok=True,
        ),
    )

    evidence = table(
        ["system", "prefix source", "audio-token frequency", "test-clean WER", "test-other WER"],
        [
            (["Oracle-char reference", "WavLM char alignment", "14.7 Hz", "5.03±0.42", "8.07±0.43"], ""),
            (["Highlighted learned configuration", "WavLM CNN first-order AR", "11.7 Hz", "5.06±0.18", "9.55±0.17"], "recommended"),
        ],
        "matrix",
    )
    body += hk.section(
        "1 · Motivation",
        "The working hypothesis is a decoder input-distribution mismatch, not simply a "
        "lack of decoder capacity.",
        hk.card(
            evidence
            + '<p class="cap">Table 1. The learned system is nearly tied with the oracle-char '
              'reference on test-clean at a lower audio-token frequency, but its larger test-other gap '
              'leaves room for better robustness to shorter and noisier prefixes.</p>',
            title="Observed audio-frequency and WER gap",
        )
        + hk.finding(
            '<b>Current training only partly covers the inference distribution.</b> Decoder CE '
            'uses one deterministic segmentation. GRPO explores sampled segmentations, but '
            'their decoder forwards are detached and become rewards rather than decoder '
            'training examples. The decoder therefore does not learn directly from the full '
            'prefix distribution induced by the segmenter.'
        )
        + '<p class="note">This follows the general motivation of '
          '<a href="https://arxiv.org/abs/2306.13649">on-policy language-model distillation</a> '
          'and <a href="https://proceedings.mlr.press/v15/ross11a.html">DAgger-style training '
          'under model-induced inputs</a>: train under the distribution the system itself '
          'creates, instead of only under a fixed reference distribution.</p>',
    )

    flow = """
    <div class="flow">
      <div class="flow-col">
        <div class="flow-box"><strong>Frozen teacher</strong><span>Best WavLM char-level decoder</span></div>
        <div class="flow-box"><strong>Oracle-char boundaries</strong><span>14.7 Hz · stable, information-rich prefix</span></div>
        <div class="flow-box"><strong>Teacher-forced transcript</strong><span>Detached teacher logits at text positions</span></div>
      </div>
      <div class="flow-arrow">→</div>
      <div class="flow-col">
        <div class="flow-box"><strong>Trainable student</strong><span>Same initialization; projection + LoRA updated</span></div>
        <div class="flow-box"><strong>Current segmenter boundaries</strong><span>Greedy or sampled WavLM CNN first-order AR prefix</span></div>
        <div class="flow-box"><strong>Teacher-forced transcript</strong><span>CE to gold text + KL to teacher distribution</span></div>
      </div>
    </div>
    <div class="merge"><strong>Align at transcript-token positions</strong><br>
    Teacher and student audio prefixes may have different lengths; compare the final text logits, not audio positions.</div>
    """
    body += hk.section(
        "2 · High-level design",
        "The teacher sees the oracle-char prefix. The student sees the shorter prefix "
        "produced by the segmenter. Both use the same gold transcript history.",
        hk.card(flow + '<p class="cap">Figure 1. Oracle-prefix teacher and learned-prefix student forwards.</p>', title="Two forwards, one text target")
        + '<p>For transcript positions <span class="mono">M</span>, the student loss is:</p>'
        + hk.equation(r"""
\begin{aligned}
\mathcal{L}_{\mathrm{decoder}}
  &= \mathcal{L}_{\mathrm{CE}} + \lambda_{\mathrm{KD}}\mathcal{L}_{\mathrm{KD}}, \\
\mathcal{L}_{\mathrm{CE}}
  &= -\sum_{t\in\mathcal{M}} \log p_S\!\left(y_t\mid z_{\mathrm{sample}},y_{<t}\right), \\
\mathcal{L}_{\mathrm{KD}}
  &= T^2\!\sum_{t\in\mathcal{M}} \operatorname{KL}\!\left(
       p_T^{(T)}\!\left(\cdot\mid z_{\mathrm{char}},y_{<t}\right)
       \,\middle\|\,
       p_S^{(T)}\!\left(\cdot\mid z_{\mathrm{sample}},y_{<t}\right)
     \right).
\end{aligned}
""")
        + '<p class="note"><span class="mono">M</span> is the set of unmasked transcript-token positions; '
          '<span class="mono">z_char</span> and <span class="mono">z_sample</span> are the oracle-character '
          'and sampled learned acoustic prefixes; <span class="mono">T</span> is the distillation temperature. '
          'The teacher distribution is the first argument of the forward KL.</p>'
        + table(
            ["component", "initial choice", "reason"],
            [
                (["Teacher", "Frozen oracle-char decoder", "Preserves the strongest char-prefix behavior as the target"], ""),
                (["Student prefix", "One closed-loop sample per utterance", "Covers the segmenter-induced prefix distribution without K-fold decoder cost"], ""),
                (["Loss", "CE + forward KL", "CE anchors correctness; KD transfers the teacher's uncertainty"], "recommended"),
                (["Temperature", "T=2", "Softens an overconfident teacher; tune only after the controlled study"], ""),
                (["KD weight", "0.25 or 0.5", "Keep transcript CE dominant initially"], ""),
            ],
        )
        + '<p class="cap">Table 2. Initial design choices. KD-only is a diagnostic, not the main proposal.</p>',
    )

    matrix = table(
        ["arm", "decoder training prefix", "loss", "what it isolates"],
        [
            (["zero-shot", "none", "none", "Initial char-decoder mismatch on learned boundaries"], ""),
            (["det-ce", "Greedy", "CE", "Current deterministic adaptation control"], ""),
            (["det-cekd", "Greedy", "CE + KD", "Teacher value without on-policy sampling"], ""),
            (["sample-ce", "Sampled", "CE", "On-policy exposure without distillation"], ""),
            (["sample-cekd", "Sampled", "CE + KD", "Full proposal"], "recommended"),
        ],
        "matrix",
    )
    body += hk.section(
        "3 · Controlled experiment",
        "Freeze the segmenter and initialize every student from the same char decoder. "
        "Only the decoder prefix distribution and decoder loss change.",
        hk.card(
            matrix
            + '<p class="cap">Table 3. The four trained conditions form a 2×2 comparison; '
              'zero-shot is evaluated without further updates.</p>',
            title="Phase I arms",
        )
        + hk.card(
            '<div class="phase"><div class="phase-num">1</div><div><h3>Fix the boundary sets</h3>'
            '<p>Calibrate the frozen segmenter to approximately {14.5, 11.5, 10.0} Hz on dev-clean. '
            'Reuse identical boundaries for every decoder arm.</p></div></div>'
            '<div class="phase"><div class="phase-num">2</div><div><h3>Train three seeds</h3>'
            '<p>Use seeds 3407/3408/3409, matched update counts, and the same checkpoint rule.</p></div></div>'
            '<div class="phase"><div class="phase-num">3</div><div><h3>Test the interaction</h3>'
            '<p>Compare sample vs. greedy, KD vs. no KD, and whether the KD gain grows as the prefix gets shorter.</p></div></div>',
            title="Evaluation sequence",
        )
        + hk.finding(
            '<b>Most diagnostic result:</b> sampled CE+KD holds WER better than sampled CE as '
            'audio-token frequency moves from about 14.5 toward 10.0 Hz, while the exact same boundaries are used. '
            'That would directly support decoder prefix-rate robustness rather than improved '
            'boundary placement.',
            ok=True,
        ),
    )

    body += hk.section(
        "4 · Joint-RL extension",
        "Only after the frozen-segmenter study identifies the decoder objective should both "
        "components be trained together.",
        hk.card(
            hk.equation(r"""
\begin{aligned}
\mathcal{L}_{\mathrm{decoder}}
  &= \mathcal{L}_{\mathrm{CE}} + \lambda_{\mathrm{KD}}\mathcal{L}_{\mathrm{KD}}, \\
R_{\mathrm{segmenter}}
  &= -\operatorname{NLL}_{S} - \lambda_{f}\mathcal{P}_{f},
  \qquad f=50N/T\;\mathrm{Hz}.
\end{aligned}
""")
            + '<p>Keep these roles separate first. The decoder learns from sampled prefixes; '
            'the segmenter receives the existing GRPO update using '
            '<span class="mono">R_segmenter</span>. Here <span class="mono">NLL_S</span> is '
            'the detached student transcript loss and <span class="mono">P_f</span> penalizes '
            'deviation from the current audio-token frequency target. Use a '
            'gradual frequency schedule: <code>14.5 → 12.5 → 11.5 → 10.0 Hz</code>.</p>',
            title="Recommended first joint objective",
        )
        + hk.finding(
            '<b>Optional later ablation:</b> use detached teacher–student KL as part of the '
            'GRPO reward. Do not include it initially: the segmenter could learn prefixes that '
            'are easy to imitate rather than prefixes that retain the information needed for ASR.'
        ),
    )

    risks = table(
        ["risk", "control"],
        [
            (["Gain is only extra decoder training", "Match update counts and retain det-ce"], ""),
            (["Gain is only KD", "Retain both deterministic and sampled CE+KD arms"], ""),
            (["Teacher is overconfident", "Use T=2, retain CE, monitor entropy"], ""),
            (["Teacher doubles memory", "Try named PEFT adapters and sequential forwards; smoke-test peak memory"], ""),
            (["Different boundaries confound WER", "Precompute and reuse the same evaluation boundary sets"], ""),
            (["KD falls but WER does not", "Select by WER; treat KL as a diagnostic only"], ""),
            (["Joint training drifts", "Establish the decoder-only effect first, then lower audio-token frequency gradually"], ""),
        ],
    )
    body += hk.section(
        "5 · Interpretation and guardrails",
        body=hk.card(risks + '<p class="cap">Table 4. Main confounds and safeguards.</p>', title="What could mislead us")
        + hk.finding(
            '<b>A null result would also be useful.</b> If sampled CE and sampled CE+KD do not '
            'beat deterministic CE at matched boundaries, the remaining gap is less likely to '
            'be decoder prefix-rate bias. Boundary information loss or boundary placement then '
            'becomes the stronger explanation.'
        ),
    )

    body += hk.section(
        "6 · Implementation boundary",
        body=table(
            ["change", "location", "high-level requirement"],
            [
                (["Configuration", "speechllm_segmenter.yaml", "Prefix mode, loss mode, teacher checkpoint, temperature, KD weight"], ""),
                (["Teacher forward", "train_speechllm_with_segmenter.py", "Frozen oracle-boundary projection + LoRA forward under inference mode"], ""),
                (["Text-logit alignment", "decoder helper", "Slice aligned text positions despite unequal audio-prefix lengths"], ""),
                (["KD loss", "decoder objective", "Masked fp32 forward KL; gradients to student decoder only"], ""),
                (["Launcher", "new submit script", "Same checkpoints, updates, and seeds across all four trained arms"], ""),
                (["Tests", "CPU/GPU smoke", "Masking, zero-KL identity, gradient isolation, checkpoint recovery, memory"], ""),
            ],
        ) + '<p class="cap">Table 5. The detailed execution plan is in <code>plans/on_policy_prefix_distillation.md</code>.</p>',
    )

    generated = datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    body += (
        '<footer>Generated %s · Proposal only; no experiment has been launched from this document · '
        'Project evidence from the cumulative segmenter report.</footer>' % generated
    )
    full, _ = hk.document(
        "On-policy prefix distillation proposal", body, mathjax=True
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(full)
    print("wrote", args.out, "(%d KB)" % (len(full) // 1024))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="artifacts/segmenter/on_policy_prefix_distillation.html",
    )
    build(parser.parse_args())
