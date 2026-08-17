#!/usr/bin/env python3
"""Build the implementation note for combined Transformer-AR rollout batching."""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

SKILL_DIR = Path.home() / ".codex" / "skills" / "html-report"
sys.path.insert(0, str(SKILL_DIR))
import htmlkit as hk  # noqa: E402


RECIPE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RECIPE_DIR.parents[3]
RESULT_ROOT = (
    RECIPE_DIR
    / "results/speechllm_segmenter_wavlm/transformer_ar_combined_rollout_validation"
)


def table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{html.escape(cell)}</th>" for cell in headers)
    body = "".join(
        "<tr>%s</tr>" % "".join(f"<td>{cell}</td>" for cell in row)
        for row in rows
    )
    return f'<div class="table-scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def pipeline_svg() -> str:
    return r"""
    <svg class="chart pipeline" viewBox="0 0 1020 350" role="img"
         aria-label="Reference and combined Transformer AR rollout flow">
      <defs>
        <marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 z" fill="var(--text-muted)"/>
        </marker>
      </defs>
      <text x="18" y="43" class="row-title">Reference</text>
      <text x="18" y="214" class="row-title">Combined</text>

      <g class="box"><rect x="130" y="18" width="195" height="62" rx="10"/>
        <text x="228" y="43">Greedy policy loop</text><text x="228" y="63" class="small">B lanes · T sequential steps</text></g>
      <g class="box"><rect x="380" y="18" width="195" height="62" rx="10"/>
        <text x="478" y="43">Sample policy loop</text><text x="478" y="63" class="small">K×B lanes · T sequential steps</text></g>
      <g class="box"><rect x="630" y="18" width="176" height="62" rx="10"/>
        <text x="718" y="43">Reward decoder</text><text x="718" y="63" class="small">K separate calls</text></g>
      <g class="box"><rect x="850" y="18" width="150" height="62" rx="10"/>
        <text x="925" y="43">GRPO update</text><text x="925" y="63" class="small">one backward</text></g>
      <path class="arrow" d="M325 49 H380"/><path class="arrow" d="M575 49 H630"/><path class="arrow" d="M806 49 H850"/>
      <text x="500" y="119" class="cost">2 sequential policy loops + K reward calls</text>

      <g class="box good"><rect x="130" y="188" width="310" height="68" rx="10"/>
        <text x="285" y="215">Greedy + K sampled policy lanes</text>
        <text x="285" y="237" class="small">(K+1)×B lanes · one T-step loop</text></g>
      <g class="box good"><rect x="545" y="188" width="260" height="68" rx="10"/>
        <text x="675" y="215">Batched reward decoder</text>
        <text x="675" y="237" class="small">one call on K×B items</text></g>
      <g class="box"><rect x="850" y="188" width="150" height="68" rx="10"/>
        <text x="925" y="215">GRPO update</text><text x="925" y="237" class="small">same backward</text></g>
      <path class="arrow" d="M440 222 H545"/><path class="arrow" d="M805 222 H850"/>
      <text x="500" y="302" class="cost good-text">1 sequential policy loop + 1 reward call</text>
      <text x="500" y="332" class="small center">The frame loop remains sequential; independent trajectories move to the batch dimension.</text>
    </svg>
    """


def speed_svg() -> str:
    baseline, combined = 101.0, 52.0
    maximum = 110.0
    width = lambda value: 610 * value / maximum
    return f"""
    <svg class="chart bars" viewBox="0 0 900 220" role="img"
         aria-label="Thirty batch training loop time baseline 101 seconds, combined 52 seconds">
      <text x="25" y="58" class="bar-label">Reference</text>
      <rect x="180" y="32" width="{width(baseline):.1f}" height="42" rx="7" class="bar base"/>
      <text x="{195 + width(baseline):.1f}" y="59" class="bar-value">101 s</text>
      <text x="25" y="133" class="bar-label">Combined</text>
      <rect x="180" y="107" width="{width(combined):.1f}" height="42" rx="7" class="bar fast"/>
      <text x="{195 + width(combined):.1f}" y="134" class="bar-value">52 s</text>
      <line x1="180" y1="176" x2="790" y2="176" class="axis"/>
      <text x="180" y="198" class="tick">0</text><text x="457" y="198" class="tick">50</text><text x="735" y="198" class="tick">100 seconds</text>
      <text x="450" y="18" class="speed-title">Real LibriSpeech batches · WavLM + 1B decoder · K=4 · window 64</text>
    </svg>
    """


def read_completed_runs() -> list[list[str]]:
    rows: list[list[str]] = []
    for variant in ("baseline", "combined"):
        summaries = sorted(RESULT_ROOT.glob(f"full_{variant}_*/benchmark_summary.json"))
        if not summaries:
            rows.append([variant.title(), "Queued", "—", "—", "—", "—"])
            continue
        summary_path = summaries[-1]
        summary = json.loads(summary_path.read_text())
        output = summary_path.parent
        timing = {}
        timing_file = output / "stage_timing.jsonl"
        if timing_file.exists():
            for line in timing_file.read_text().splitlines():
                item = json.loads(line)
                timing[item["stage"]] = timing.get(item["stage"], 0.0) + item["seconds"]
        train_log = output / "train_log.txt"
        final = train_log.read_text().splitlines()[-1] if train_log.exists() else ""

        def field(label: str) -> str:
            marker = f"{label}: "
            if marker not in final:
                return "—"
            return final.split(marker, 1)[1].split(",", 1)[0].split(" - ", 1)[0]

        def frequency_field(label: str) -> str:
            value = field(label)
            if value == "—":
                return value
            return f"{50.0 * float(value):.1f} Hz"

        rows.append(
            [
                variant.title(),
                "Passed" if summary["exit_status"] == 0 else "Failed",
                f"{timing.get('TRAIN', 0) / 60:.1f} min" if timing else "—",
                f"{summary['wall_seconds'] / 60:.1f} min",
                field("test WER"),
                frequency_field("test rho_mean"),
            ]
        )
    return rows


def build() -> str:
    full_rows = read_completed_runs()
    full_complete = all(row[1] == "Passed" for row in full_rows)
    status_value = "Complete" if full_complete else "Running"
    body = (
        '<style>.wrap{max-width:1100px}.table-scroll{overflow-x:auto}'
        'th,td{text-align:left;vertical-align:top}.pipeline text,.bars text{font-family:-apple-system,'
        'BlinkMacSystemFont,"Segoe UI",sans-serif;fill:var(--text-primary)}'
        '.pipeline .box rect{fill:var(--surface-2);stroke:var(--border-strong);stroke-width:1.2}'
        '.pipeline .box.good rect{fill:color-mix(in srgb,var(--ok) 10%,var(--surface-1));stroke:var(--ok)}'
        '.pipeline .box text{text-anchor:middle;font-size:15px}.pipeline .small{font-size:12.5px;fill:var(--text-secondary)}'
        '.pipeline .row-title{font-weight:700;font-size:16px}.pipeline .arrow{stroke:var(--text-muted);stroke-width:2;marker-end:url(#arrow)}'
        '.pipeline .cost{text-anchor:middle;font-size:15px;font-weight:650}.pipeline .good-text{fill:var(--ok)}.pipeline .center{text-anchor:middle}'
        '.bars .bar.base{fill:var(--w-muted)}.bars .bar.fast{fill:var(--series-asr)}.bars .bar-label{font-size:15px;font-weight:600}'
        '.bars .bar-value{font-size:16px;font-weight:700}.bars .axis{stroke:var(--border-strong)}.bars .tick{font-size:12px;fill:var(--text-muted)}'
        '.bars .speed-title{text-anchor:middle;font-size:13px;fill:var(--text-secondary)}'
        '.code{font-family:ui-monospace,"SF Mono",Consolas,monospace;font-size:13px;white-space:pre-wrap;background:var(--surface-2);padding:14px 16px;border-radius:9px;overflow-x:auto}'
        '.callout{border-left:3px solid var(--series-sed);background:var(--band);padding:11px 14px;border-radius:0 8px 8px 0}'
        '@media(max-width:720px){.pipeline{min-width:820px}.pipeline-wrap{overflow-x:auto}}'
        '</style>'
        + hk.hero(
            "Implementation note · 12 August 2026",
            "Faster Transformer-AR training by combining rollouts",
            "One frame-by-frame segmenter pass now produces the greedy decoder input and all four GRPO samples. The four detached rewards are evaluated in one decoder call. The change is opt-in and preserves the original on-policy objective.",
        )
        + hk.tiles(
            [
                ("30-batch pilot", "1.94×", "training loop: 101 s → 52 s"),
                ("GRPO samples", "K = 4", "unchanged"),
                ("History window", "64 frames", "unchanged"),
                ("Full-loop check", status_value, "jobs 70131 / 70132"),
            ]
        )
        + hk.section(
            "1 · What changed",
            "The optimization removes repeated launches; it does not shorten the sampled histories or reuse them across updates.",
            hk.finding(
                "<b>Main idea.</b> Greedy and sampled boundary histories are conditionally independent once placed in separate batch lanes. They can therefore advance through the same Transformer call at every frame.",
                ok=True,
            )
            + hk.labeled_asset(
                '<div class="pipeline-wrap">' + pipeline_svg() + "</div>",
                "Figure",
                1,
                "Reference and combined on-policy computation. The combined path widens the batch but executes only one sequential frame loop and one reward-decoder call.",
                title="Two sources of repeated work are merged",
            )
            + hk.labeled_asset(
                table(
                    ["Operation", "Reference", "Combined", "Preserved behavior"],
                    [
                        ["Greedy boundaries", "Standalone B-lane rollout", "First B lanes of grouped rollout", "Exact argmax history"],
                        ["GRPO histories", "One K×B rollout", "Remaining K×B grouped lanes", "Exact matched-seed samples and RNG state"],
                        ["NLL rewards", "K decoder calls", "One K×B decoder call", "Same per-utterance rewards"],
                        ["Policy gradient", "One parallel re-score", "Same parallel re-score", "Same on-policy GRPO objective"],
                        ["Optimizer", "SpeechBrain accumulation", "Same accumulation", "One update; no rollout reuse"],
                    ],
                ),
                "Table",
                1,
                "Operations changed by the optimization and the invariants retained by construction.",
                title="Implementation map",
            ),
        )
        + hk.section(
            "2 · Concrete example",
            "For a minibatch with B=2 utterances and K=4 GRPO samples, the policy runs on 10 lanes: 2 greedy lanes followed by four groups of 2 sampled lanes.",
            hk.card(
                '<div class="code">lanes 0–1   greedy histories → decoder CE input\nlanes 2–3   sample 1         ↘\nlanes 4–5   sample 2          ↘ one 8-item NLL reward batch\nlanes 6–7   sample 3          ↗\nlanes 8–9   sample 4         ↗\n\nall 10 lanes advance together for t = 0 … T−1\nonly the 8 sampled lanes consume random draws</div>',
                title="Lane layout for B=2, K=4",
            )
            + '<p class="callout"><b>Why the random samples still match.</b> The grouped implementation draws Bernoulli actions only for sampled lanes. With the same seed, it produces bit-identical K×B histories and leaves the generator in the same state as the reference sample rollout.</p>',
        )
        + hk.section(
            "3 · How to use it",
            "Existing jobs remain unchanged because the default is still on_policy. The faster path is selected explicitly for the local-history Transformer segmenter.",
            hk.card(
                '<div class="code">--segmenter_backbone transformer_ar \\\n+--segmenter_ar_history_window 64 \\\n+--segmenter_ar_cache_mode preallocated \\\n+--grpo_k 4 \\\n+--rl_update_mode combined_on_policy</div>',
                title="Training override",
            )
            + hk.labeled_asset(
                table(
                    ["File", "Implementation"],
                    [
                        ['<a href="../../recipes/LibriSpeech/ASR/transformer/segmenter.py">segmenter.py</a>', "Grouped cached rollout and exactness self-test"],
                        ['<a href="../../recipes/LibriSpeech/ASR/transformer/train_speechllm_with_segmenter.py">train_speechllm_with_segmenter.py</a>', "Grouped rollout, batched rewards, and unchanged GRPO re-score/backward"],
                        ['<a href="../../recipes/LibriSpeech/ASR/transformer/hparams/speechllm_segmenter.yaml">speechllm_segmenter.yaml</a>', "Disabled-by-default update-mode switch"],
                    ],
                ),
                "Table",
                2,
                "Production files touched by the opt-in implementation.",
                title="Code locations",
            ),
        )
        + hk.section(
            "4 · Measured speed and validation",
            "The completed pilot uses real LibriSpeech batches and the full WavLM/segmenter/decoder training step. The new jobs extend this to a complete 4,006-batch epoch with validation, checkpoint recovery, and test decoding.",
            hk.labeled_asset(
                speed_svg(),
                "Figure",
                2,
                "Earlier matched 30-batch pilot. Combined rollout plus reward batching reduced the measured training loop from 101 seconds to 52 seconds (1.94×). Full debug wall time, including validation and test, fell from 265 to 186 seconds (1.42×).",
                title="Short real-batch pilot",
            )
            + hk.labeled_asset(
                table(
                    ["Path", "Status", "Train stage", "End-to-end wall", "Test-clean WER", "Test audio-token frequency"],
                    full_rows,
                ),
                "Table",
                3,
                "Matched one-epoch production-code validation. Both arms use seed 3407, WavLM, the same segmenter and char-decoder initialization, K=4, window 64, preallocated KV, and one A100. Regenerate this page after the jobs finish to fill the measurements.",
                title="Full training-loop comparison",
            ),
        )
        + hk.section(
            "5 · Stability checks",
            "The checks focus on equivalence and training health, not only elapsed time.",
            hk.labeled_asset(
                table(
                    ["Check", "Result", "Why it matters"],
                    [
                        ["Greedy boundaries", "0 mismatches", "Decoder CE receives the same segmentation"],
                        ["Greedy logits", "Within cached-rollout tolerance", "No policy-function change"],
                        ["Matched-seed samples", "Bit-identical", "Same on-policy trajectories in a controlled run"],
                        ["RNG state", "Exact match", "No hidden change to later stochastic operations"],
                        ["Frame 0 and padding", "All assertions pass", "Pooling semantics remain valid"],
                        ["Original segmenter tests", "All pass", "Causality, scoring, gradient direction, rate loss, and checkpoint recovery remain intact"],
                        ["Full-loop finite loss / checkpoint / test", "Pending jobs 70131 / 70132", "Longer-run implementation stability gate"],
                    ],
                ),
                "Table",
                4,
                "Correctness and stability gates. The last row will be resolved by the full-loop comparison.",
                title="Validation checklist",
            )
            + hk.finding(
                "<b>Current recommendation.</b> Use <span class='mono'>combined_on_policy</span> for new local-history Transformer-AR NLL training after the one-epoch gate completes. Keep fused masked SDPA; FlexAttention did not improve the end-to-end loop on the observed batch-length mix.",
                ok=True,
            ),
        )
        + "<footer>Generated from the implementation and benchmark artifacts. The report is self-contained and can be regenerated after SLURM completion.</footer>"
    )
    document, _ = hk.document(
        "Combined Transformer-AR rollout implementation", body
    )
    return document


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts/segmenter/transformer_ar_combined_rollout.html",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build(), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
