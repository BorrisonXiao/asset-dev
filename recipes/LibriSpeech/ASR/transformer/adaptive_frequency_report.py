#!/usr/bin/env python3
"""Collect audited continuation evidence and render its training-report section.

This does not train, evaluate a model, alter scores, or relax the legacy LaTeX
collector. Both existing whitespace conventions are explicitly re-scored.
Run --collect to refresh the small, publishable evidence snapshot from local runs.
"""

import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import re
import statistics

REPO = Path(__file__).resolve().parents[4]
RESULTS = REPO / "recipes/LibriSpeech/ASR/transformer/results"
PILOT = RESULTS / "speechllm_ls960_adaptive_frequency_20260918"
REPLICATION = RESULTS / "speechllm_ls960_adaptive_frequency_seeds3408_3409_20260918"
SNAPSHOT = REPO / "artifacts/segmenter/adaptive_frequency_results_20260919.json"
ARMS = {"original": "Original fixed target", "fixed_lower": "Fixed lower target",
        "adaptive": "Adaptive target"}
SPLITS = {"test-clean": 2620, "test-other": 2939}
CONFIG = dict(max_steps=8000, validation_interval=500, ema_alpha=0.2,
              wer_tolerance_pp=0.1, consecutive_checks=2, rate_step_hz=0.5,
              min_target_hz=7.5, original_target_hz=12.5, frame_hz=50.0)
WER_RE = re.compile(r"^%WER\s+[0-9.]+\s+\[\s*(\d+)\s*/\s*(\d+)")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def distance(hypothesis, reference):
    previous = list(range(len(reference) + 1))
    for i, token in enumerate(hypothesis, 1):
        current = [i]
        for j, target in enumerate(reference, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (token != target)))
        previous = current
    return previous[-1]


def audit_transcripts(rows, errors, words):
    """Require BOTH existing loggers to reproduce, retaining any differences."""
    literal_total = normalized_total = literal_words = normalized_words = 0
    differences = []
    for row in rows:
        hypothesis, reference = row["hypothesis"], row["reference"]
        normal_hyp, normal_ref = hypothesis.split(), reference.split()
        literal_hyp, literal_ref = hypothesis.split(" "), reference.split(" ")
        normalized = distance(normal_hyp, normal_ref)
        literal = (normalized if (normal_hyp, normal_ref) == (literal_hyp, literal_ref)
                   else distance(literal_hyp, literal_ref))
        require(normalized == row["word_errors"], f"Diagnostic error mismatch: {row['id']}")
        require(len(normal_ref) == row["reference_words"], f"Diagnostic word mismatch: {row['id']}")
        literal_total += literal
        normalized_total += normalized
        literal_words += len(literal_ref)
        normalized_words += len(normal_ref)
        if literal != normalized or len(literal_ref) != len(normal_ref):
            differences.append(dict(id=row["id"], main_errors=literal,
                                    normalized_errors=normalized))
    require(literal_total == errors and literal_words == words,
            "Main WER report does not reproduce from literal-space scoring")
    return dict(main_errors=errors, main_words=words,
                normalized_errors=normalized_total, normalized_words=normalized_words,
                normalized_WER=100 * normalized_total / normalized_words,
                whitespace_differences=differences)


def collect():
    evidence = {}

    def remember(path):
        evidence[str(path.relative_to(REPO))] = sha(path)

    runs = []
    for seed in (3407, 3408, 3409):
        train_root = PILOT if seed == 3407 else REPLICATION / f"seed{seed}"
        eval_root = (REPO / "artifacts/segmenter/ls960_adaptive_frequency_test_eval_20260918"
                     if seed == 3407 else train_root / "test_eval")
        manifest_path = eval_root / "evaluation_manifest.json"
        remember(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        require(int(manifest["seed"]) == seed, "Wrong manifest seed")
        expected_ids = {}
        for split, count in SPLITS.items():
            info = manifest["splits"][split]
            path = Path(info["path"])
            require(sha(path) == info["sha256"], "Split manifest changed")
            with path.open(newline="") as stream:
                ids = [row["ID"] for row in csv.DictReader(stream)]
            require(len(ids) == len(set(ids)) == count, "Incomplete split manifest")
            expected_ids[split] = set(ids)
        for arm in ARMS:
            train = train_root / arm
            directory = eval_root / arm
            config_path = train / "continuation_config.json"
            config = json.loads(config_path.read_text())
            require(config["arm"] == arm, "Wrong training arm")
            require(all(config[k] == v for k, v in CONFIG.items()), "Changed curriculum")
            events_path = train / "frequency_events.jsonl"
            events = jsonl(events_path)
            require([e["step"] for e in events] == list(range(0, 8001, 500)), "Missing dev checks")
            require(events[-1]["training_done"], "Training not finished")
            for event in events:
                require(event["source_optimizer_step"] == 24000, "Wrong source update count")
                require(event["splits"]["dev-clean"]["utterances"] == 2703
                        and event["splits"]["dev-other"]["utterances"] == 2864,
                        "Incomplete dev validation")
            tests = {}
            metric_path = directory / "test_metrics.jsonl"
            metrics = jsonl(metric_path)
            require(len(metrics) == 2 and {m["split"] for m in metrics} == set(SPLITS),
                    "Missing or repeated test split")
            benchmarks = jsonl(directory / "benchmark.jsonl")
            require(len(benchmarks) == 2 and {b["split"] for b in benchmarks} == set(SPLITS),
                    "Missing or repeated benchmark split")
            benchmark_by_split = {b["split"]: b for b in benchmarks}
            for metric in metrics:
                split = metric["split"]
                require(int(metric.get("seed", seed)) == seed, "Wrong test seed")
                require(metric["continuation_optimizer_step"] == 8000, "Wrong test step")
                require(metric["checkpoint"] == manifest["arms"][arm]["checkpoint"],
                        "Wrong final checkpoint")
                bench = benchmark_by_split[split]
                require(bench["batch_size"] == 8 and bench["decoding_protocol"] ==
                        "batch_invariant_left_packed_duration_cap_v1", "Wrong decode protocol")
                path = directory / f"benchmark_{split}_utterances.jsonl"
                utterances = jsonl(path)
                require(len(utterances) == SPLITS[split] == metric["utterances"] == bench["utterances"]
                        and {u["id"] for u in utterances} == expected_ids[split], "Incomplete test coverage")
                wer_path = directory / "wer_results" / f"wer_{split}.txt"
                match = WER_RE.match(wer_path.read_text().splitlines()[0])
                require(match is not None, "Invalid WER report")
                errors, words = map(int, match.groups())
                require(math.isclose(metric["WER"], 100 * errors / words, abs_tol=1e-10)
                        and math.isclose(metric["WER"], bench["WER"], abs_tol=1e-10), "WER mismatch")
                tests[split] = {**audit_transcripts(utterances, errors, words),
                               "WER": metric["WER"], "CER": metric["CER"],
                               "Hz": metric["rate_hz_50rho"], "utterances": len(utterances)}
                for item in (path, wer_path):
                    remember(item)
            for item in (config_path, events_path, metric_path, directory / "benchmark.jsonl"):
                remember(item)
            selection = json.loads((train / "selected_checkpoint.json").read_text())
            runs.append(dict(seed=seed, arm=arm, source_checkpoint=config["source_checkpoint"],
                             test_checkpoint=manifest["arms"][arm]["checkpoint"],
                             test_job=str(manifest["arms"][arm]["job_id"]),
                             rate_selected_step=selection["selected_new_step"],
                             tests=tests, events=events))
    return dict(schema_version=1, updated="2026-09-19", config=CONFIG, source_updates=24000,
                score_convention="Logged main WER; literal-space tokens, not whitespace-normalized WER",
                rate_definition="50 times the mean per-utterance kept-frame ratio",
                runs=runs, input_sha256=evidence)


def summary(data, arm, split, key):
    values = [r["tests"][split][key] for r in data["runs"] if r["arm"] == arm]
    require(len(values) == 3, "Require three seeds per result")
    return statistics.mean(values), statistics.stdev(values)


def table(headers, rows, caption, table_id):
    cells = lambda row, tag: "".join(f"<{tag}>{cell}</{tag}>" for cell in row)
    return (f'<table id="{table_id}"><thead><tr>{cells(headers, "th")}</tr></thead><tbody>'
            + "".join(f'<tr>{cells(row, "td")}</tr>' for row in rows)
            + f'</tbody></table><p class="cap">{caption}</p>')


def training_runs(data):
    """Require complete, aligned validation histories before drawing curves."""
    require(data["config"] == CONFIG, "Unexpected curve configuration")
    runs = {(r["seed"], r["arm"]): r for r in data["runs"]}
    require(len(runs) == len(data["runs"]) == 9 and set(runs) ==
            {(seed, arm) for seed in (3407, 3408, 3409) for arm in ARMS},
            "Require all nine unique seed/arm histories")
    steps = list(range(0, CONFIG["max_steps"] + 1, CONFIG["validation_interval"]))
    for run in runs.values():
        events = run["events"]
        require([e["step"] for e in events] == steps, "Missing or misaligned validation steps")
        require(events[-1]["training_done"], "Incomplete training curve")
        previous = events[0]["other_wer"]
        for event in events:
            for key in ("other_hz", "other_wer", "ema_other_wer", "target_hz"):
                require(math.isfinite(event[key]), "Non-finite curve value")
            if event["step"]:
                previous = (CONFIG["ema_alpha"] * event["other_wer"]
                            + (1 - CONFIG["ema_alpha"]) * previous)
            require(math.isclose(event["ema_other_wer"], previous, abs_tol=1e-10),
                    "Logged EMA does not reproduce")
    return runs


def curve_summary(runs, arm, key):
    """Per-validation mean and sample SD; never pool test or minibatch metrics."""
    histories = [runs[(seed, arm)]["events"] for seed in (3407, 3408, 3409)]
    return [(group[0]["step"], statistics.mean(e[key] for e in group),
             statistics.stdev(e[key] for e in group)) for group in zip(*histories)]


def step_points(points):
    """A value at x takes effect AFTER that check (a post-step staircase)."""
    require(bool(points), "Empty step series")
    output = [points[0]]
    for (previous_x, previous_y), (x, y) in zip(points, points[1:]):
        require(x > previous_x, "Step coordinates must increase")
        output.extend(((x, previous_y), (x, y)))
    return output


def curve_svg(chart_id, title, description, series, ylim, yticks, ylabel, markers=()):
    """Small theme-aware SVG; exact logged points, optional SD bands/step lines.

    Unlike the toolkit's epoch chart, the update axis needs only five ticks.
    Each panel lives inside a numbered HTML figure, so it is not numbered twice.
    """
    width, height = 420, 280
    left, right, top, bottom = 52, 18, 15, 50
    plot_bottom = height - bottom

    def px(x):
        return left + x / CONFIG["max_steps"] * (width - left - right)

    def py(y):
        return plot_bottom - (y - ylim[0]) / (ylim[1] - ylim[0]) * (plot_bottom - top)

    def coordinates(points):
        require(all(0 <= x <= CONFIG["max_steps"] and ylim[0] <= y <= ylim[1]
                    for x, y in points), f"Curve outside axes: {chart_id}")
        return " ".join(f"{px(x):.3f},{py(y):.3f}" for x, y in points)

    parts = [f'<svg class="adaptive-chart" id="{chart_id}" viewBox="0 0 {width} {height}" '
             f'role="img" aria-labelledby="{chart_id}-title {chart_id}-desc">',
             f'<title id="{chart_id}-title">{html.escape(title)}</title>',
             f'<desc id="{chart_id}-desc">{html.escape(description)}</desc>']
    for value in yticks:
        y = py(value)
        parts += [f'<line x1="{left}" x2="{width-right}" y1="{y:.3f}" y2="{y:.3f}" class="grid"/>',
                  f'<text x="{left-8}" y="{y+4:.3f}" text-anchor="end">{value:g}</text>']
    for step in range(0, CONFIG["max_steps"] + 1, 2000):
        label = f"{step // 1000}k" if step else "0"
        parts.append(f'<text x="{px(step):.3f}" y="{plot_bottom+22}" text-anchor="middle">{label}</text>')
    parts += [f'<text x="{(left+width-right)/2}" y="{height-6}" text-anchor="middle">Additional optimizer updates</text>',
              f'<text transform="translate(15,{(top+plot_bottom)/2}) rotate(-90)" text-anchor="middle">{html.escape(ylabel)}</text>']
    # Bands first, so an overlapping band's fill cannot dim another mean line.
    for item in series:
        if "band" in item:
            band = item["band"]
            outline = [(x, mean-sd) for x, mean, sd in band]
            outline += [(x, mean+sd) for x, mean, sd in reversed(band)]
            parts.append(f'<polygon data-series="{item["key"]}-sd" points="{coordinates(outline)}" '
                         f'fill="{item["color"]}" fill-opacity="0.10" stroke="none"/>')
    for item in series:
        points = step_points(item["points"]) if item.get("step") else item["points"]
        dash = f' stroke-dasharray="{item["dash"]}"' if item.get("dash") else ""
        parts.append(f'<polyline data-series="{item["key"]}" points="{coordinates(points)}" '
                     f'fill="none" stroke="{item["color"]}" stroke-width="2.5"{dash} '
                     'stroke-linejoin="round"/>')
        if item.get("dots"):
            for x, y in item["points"]:
                parts.append(f'<circle cx="{px(x):.3f}" cy="{py(y):.3f}" r="2.4" '
                             f'fill="{item["color"]}"><title>{html.escape(item["key"])}; '
                             f'update {x}: {y:.6f}</title></circle>')
    for kind, step, value, color, label in markers:
        x, y = px(step), py(value)
        if kind == "lower_target":
            parts.append(f'<rect data-event="{kind}" data-step="{step}" x="{x-3.5:.3f}" '
                         f'y="{y-3.5:.3f}" width="7" height="7" fill="{color}">')
        else:
            parts.append(f'<circle data-event="{kind}" data-step="{step}" cx="{x:.3f}" '
                         f'cy="{y:.3f}" r="6" fill="none" stroke="{color}" stroke-width="2">'
                         f'<title>{html.escape(label)} at update {step}: {value:.6f}</title></circle>')
            continue
        parts.append(f'<title>{html.escape(label)}</title></rect>')
    return "".join(parts) + '</svg>'


def curve_legend(items):
    return '<div class="legend adaptive-legend">' + ''.join(
        f'<span class="lg"><span class="adaptive-line" style="border-color:{color};'
        f'border-top-style:{style}" aria-hidden="true"></span>{html.escape(label)}</span>'
        for label, color, style in items) + '</div>'


def training_figures(hk, data):
    runs = training_runs(data)
    colors = {"original": hk.C["blue"], "fixed_lower": hk.C["orange"], "adaptive": hk.C["green"]}
    dashes = {"original": "", "fixed_lower": "8 4", "adaptive": "3 3"}
    style = '''<style>
    .adaptive-figure{margin:14px 0}
    .adaptive-panels{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px 24px;margin:14px 0}
    .adaptive-panel{min-width:0}
    .adaptive-panel h4{font-size:15px;margin:0 0 8px}
    .adaptive-chart{width:100%;height:auto;display:block}
    .adaptive-chart text{font-size:13px;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
    .adaptive-line{width:23px;height:0;display:inline-block;border-top-width:3px;border-top-style:solid}
    .adaptive-legend{gap:7px 16px;margin-top:8px}
    @media(max-width:720px){.adaptive-panels{grid-template-columns:1fr;gap:24px}.adaptive-chart text{font-size:15px}}
    </style>'''
    comparison = ''
    for key, label, ylim, ticks in (
        ("other_hz", "Measured rate (Hz)", (8.5, 12), (9, 10, 11, 12)),
        ("other_wer", "Word error rate (%)", (4.6, 6), (4.6, 5, 5.4, 5.8)),
    ):
        series = []
        for arm in ARMS:
            band = curve_summary(runs, arm, key)
            series.append(dict(key=arm, color=colors[arm], dash=dashes[arm],
                               points=[(x, mean) for x, mean, _ in band], band=band))
        comparison += '<div class="adaptive-panel"><h4>Dev-other · ' + label + '</h4>'
        comparison += curve_svg(f"adaptive-comparison-{key}", f"Dev-other {label}",
            "Three-seed mean curves and plus/minus one sample standard deviation for original fixed, "
            "fixed lower, and adaptive targets. Checks run from zero to 8,000 additional updates.",
            series, ylim, ticks, label) + '</div>'
    figures = style + '<figure class="card adaptive-figure" id="adaptive-training-curves">'
    figures += '<h3>Training curves — adaptive and fixed-target continuations</h3>'
    figures += curve_legend([(ARMS[arm], colors[arm], style) for arm, style in
                            (("original", "solid"), ("fixed_lower", "dashed"), ("adaptive", "dotted"))])
    figures += '<div class="adaptive-panels">' + comparison + '</div>'
    figures += ('<figcaption class="cap">Figure 1. Full dev-other validation, not test results: lines are the mean of '
                'seeds 3407, 3408 and 3409; shading is ± one sample SD, not a confidence interval. '
                'All 17 logged checks are used, without smoothing. Lower WER and lower rate are desirable. '
                'Step 0 is the LS960 24,000-update checkpoint; step 8,000 is the end of every continuation '
                '(32,000 cumulative updates). Rate is 50 × mean per-utterance kept-frame ratio.</figcaption></figure>')
    figures += ('<p>Figure 1 shows the rate falling further under the adaptive target while WER generally improves '
                'from initialization in all three approaches. Figure 2 shows the individual adaptive decisions; '
                'an average curve alone would hide the one failed quality check.</p>')
    figures += '<figure class="card adaptive-figure" id="adaptive-controller-curves"><h3>Adaptive controller — each seed</h3><div class="adaptive-panels">'
    for seed in (3407, 3408, 3409):
        events = runs[(seed, "adaptive")]["events"]
        points = lambda key: [(e["step"], e[key]) for e in events]
        measured, target, limit = hk.C["blue"], hk.C["green"], hk.C["red"]
        guard = events[0]["other_wer"] + CONFIG["wer_tolerance_pp"]
        rate_series = [dict(key="measured-rate", color=measured, points=points("other_hz"), dots=True),
                       dict(key="target-after-check", color=target, points=points("target_hz"), step=True, dash="7 4")]
        rate_markers = [("lower_target", e["step"], e["target_hz"], target,
                         f'Target lowered after update {e["step"]}: {e["target_hz"]:.6f} Hz')
                        for e in events if e["action"] == "lower_target"]
        wer_series = [dict(key="wer-limit", color=limit, points=[(0, guard), (CONFIG["max_steps"], guard)], dash="2 4"),
                      dict(key="raw-wer", color=measured, points=points("other_wer"), dots=True),
                      dict(key="ema-wer", color=target, points=points("ema_other_wer"), dash="7 4")]
        wer_markers = [("quality_failed", e["step"], e["other_wer"], limit, "Quality check failed")
                       for e in events if not e["quality_ok"]]
        for kind, label, series, markers, ylim, ticks, legend, description in (
            ("rate", "Rate (Hz)", rate_series, rate_markers, (8.5, 12), (9, 10, 11, 12),
             [("Measured rate", measured, "solid"), ("Upper target", target, "dashed")],
             "Measured dev-other rate and the upper target, held constant between checks. Squares mark target reductions."),
            ("wer", "WER (%)", wer_series, wer_markers, (4.6, 6), (4.6, 5, 5.4, 5.8),
             [("Raw WER", measured, "solid"), ("EMA", target, "dashed"), ("Limit", limit, "dotted")],
             f"Raw dev-other WER and its exponential moving average (alpha 0.2). The fixed limit is {guard:.6f} percent. "
             "A red ring marks a failed quality check; seed 3409 fails at update 1,000 and lowers its target again at 2,000."),
        ):
            figures += f'<div class="adaptive-panel"><h4>Seed {seed} · {label}</h4>'
            figures += curve_svg(f"adaptive-seed-{seed}-{kind}", f"Seed {seed}: dev-other {label}",
                                 description, series, ylim, ticks, label, markers)
            figures += curve_legend(legend) + '</div>'
    figures += ('</div><figcaption class="cap">Figure 2. All adaptive validation checks. Blue points are observations; '
                'the green rate staircase is the target <b>after</b> each check, applied to subsequent training. '
                'Squares mark its 0.5-Hz reductions. At a reduction step, the measured rate was obtained under the '
                'previous target, so it can lie above the new one. In the WER panels, the green dashed line is the logged '
                'EMA (exponential moving average, with weight 0.2 on the latest WER); the red dotted limit is each seed’s initial dev-other WER + 0.1 point. '
                'Both raw WER and EMA must pass, along with the separate dev-clean guard (not plotted; it passed at every '
                'adaptive check). The red ring is seed 3409’s raw-WER failure at step 1,000. The target stays fixed, '
                'the rate penalty stays on, and two passing quality-and-rate checks at 1,500 and 2,000 allow lowering to resume. '
                'All curves stop at the 8,000-update budget, not at a WER threshold. The target floor is 7.5 Hz; none reached it. '
                'Exact target-change steps and final values are in Table 8; the downloadable evidence below contains every plotted value.</figcaption></figure>')
    return figures


def build_section(hk, data=None):
    data = data or json.loads(SNAPSHOT.read_text())
    require(data["config"] == CONFIG and len(data["runs"]) == 9, "Unexpected report snapshot")
    original_rate = summary(data, "original", "test-other", "Hz")[0]
    adaptive_rate = summary(data, "adaptive", "test-other", "Hz")[0]
    delta = summary(data, "adaptive", "test-other", "WER")[0] - summary(data, "original", "test-other", "WER")[0]
    reduction = 100 * (1 - adaptive_rate / original_rate)
    setup = [
        ("Training data and seeds", "Full LibriSpeech-960h: train-clean-100, train-clean-360, train-other-500; seeds 3407, 3408, 3409."),
        ("Starting checkpoint", "Each seed's own LS960 <b>24,000-update</b> Transformer-AR local64 + BiGRU checkpoint. Restore the boundary policy, pooler, projection, LoRA weights and AdamW moments."),
        ("Maximum steps", "<b>8,000 additional optimizer updates</b> per arm: 32,000 cumulative. One budget covers all target reductions and recovery. No new warmup and no quality-based early stop."),
        ("Trainable components", "Boundary policy, residual BiGRU pooler, projection and LoRA throughout. WavLM-Large and the Llama-3.2-1B-Instruct base weights remain frozen."),
        ("Policy and pooling", "Transformer AR: 4 layers, width 256, 4 heads, FFN 1,024, history window 64. Residual BiGRU: hidden size 128, 1 layer. LoRA rank 16."),
        ("Optimizer", "AdamW; constant learning rates 5e−6 for policy/pooler and 2e−5 for projection/LoRA; weight decay 0; gradient norm cap 1.0."),
        ("Batching and precision", "One A100 per run; dynamic batches up to 300 seconds of audio; accumulation 1; bf16."),
        ("Joint objective", "Decoder cross-entropy on the greedy segmentation plus on-policy GRPO for the boundary policy. Four sampled segmentations per utterance; negative NLL reward; within-group sample-SD normalization. Rate-penalty weight 1.0 and GRPO weight 1.0."),
        ("Validation and testing", "Full dev-clean (2,703) and dev-other (2,864) before training, then every 500 updates. Full test-clean (2,620) and test-other (2,939) only after training; batch 8 and the same greedy, batch-invariant duration-cap decoder."),
    ]
    body = hk.card(table(["Setting", "Implemented value"], setup,
                         "Table 4 summarizes the shared setup. All nine runs completed the same update budget.",
                         "adaptive-setup"), title="Experimental setup")
    body += hk.card(table(["Continuation", "Upper rate target", "What stays the same"], [
        ("Original fixed target", "12.5 Hz for the whole run", "Lower band edge 7.5 Hz; joint objective and 8,000-update budget."),
        ("Fixed lower target", "Initial measured dev-other rate minus 0.5 Hz, then fixed: 10.514 / 9.959 / 10.812 Hz for seeds 3407 / 3408 / 3409.", "Same lower edge, reward weight and budget."),
        ("Adaptive target", "Start at the initial measured dev-other rate: 11.014 / 10.459 / 11.312 Hz. Reduce the target by 0.5 Hz only after the checks below.", "Same lower edge, reward weight and budget; upper target cannot go below 7.5 Hz."),
    ], "Table 5. Matched controls. Initial targets are clipped to 7.5–12.5 Hz; none of these seeds needed clipping.",
       "adaptive-arms"), title="Three matched continuations")
    body += hk.card(
        '<p><b>EMA means exponential moving average.</b> Here it smooths the dev-other word error rate (WER). '
        'It is not an average of model weights. Start the EMA at the measured starting-checkpoint WER, then update it after each validation:</p>'
        + hk.equation(r'E_j = 0.2\,W_j + 0.8\,E_{j-1},\qquad E_0=W_0')
        + r'<p>Here \(j\) counts validation checks, \(W\) is dev-other WER in percent, and \(E\) is its EMA. '
        'The reference WERs are measured once before training and never moved.</p>'
        '<ol><li><b>Check quality:</b> both raw dev-other WER and its EMA must be at most the initial dev-other WER + 0.1 percentage point. '
        'Raw dev-clean WER must also be at most its own initial value + 0.1 point.</li>'
        '<li><b>Check the achieved rate:</b> measured dev-other rate must be at or below the current upper target. '
        'Good WER alone does not allow another reduction.</li>'
        '<li><b>Lower the target by 0.5 Hz after two consecutive checks pass both tests.</b> '
        'Reset that two-check counter after a failure or target change; do not reset the EMA. The step-0 check does not count.</li>'
        '<li><b>Otherwise, hold the current target and keep training.</b> The segmenter stays trainable and the rate penalty stays on. '
        'When both tests pass twice again, reductions resume. At step 8,000, save the final validation/checkpoint and end training, '
        'even if recovery is incomplete. At the 7.5-Hz target floor, hold until the same step cap.</li></ol>'
        '<p>The band is a <b>soft penalty, not a hard cap</b>. Its target never rises, but the measured rate can fluctuate or remain above it. '
        'There is no penalty-off phase, target rollback, separate task-only optimizer, or extra recovery budget.</p>',
        title="How the target changes, pauses and stops")
    objective = (
        '<p>The decoder learns the transcript from the greedy segmentation. The policy samples four alternative segmentations, '
        'scores each with the current decoder, and learns from their relative rewards:</p>'
        + hk.equation(r'\mathcal{L}=\mathcal{L}_{\mathrm{CE,greedy}}+\mathcal{L}_{\mathrm{GRPO}}')
        + hk.equation(r'\begin{aligned}R_{ik}&=-\operatorname{NLL}_{ik}-P(\rho_{ik};u),\\'
                      r'P(\rho;u)&=\max(0,0.15-\rho)^2+\max(0,\rho-u/50)^2,\\'
                      r'A_{ik}&=(R_{ik}-\overline{R}_i)/(s_i+10^{-6}),\\'
                      r'\mathcal{L}_{\mathrm{GRPO}}&=-\operatorname{mean}_{(i,k,t)\in\mathcal{V}}'
                      r'\left[\operatorname{stopgrad}(A_{ik})\log\pi_\theta(b_{ikt}\mid x_i,b_{ik,&lt;t})\right].\end{aligned}'.replace('&lt;', '<'))
        + r'<p>\(i\) identifies an utterance, \(k\) one of four samples, and \(t\) an acoustic frame. '
        'NLL is the mean transcript-token negative log likelihood. The kept ratio '
        r'\(\rho\) is pooled-token count divided by acoustic-frame count; \(u\) is the upper target in Hz. '
        r'The lower edge is 0.15 × 50 = 7.5 Hz. The bar and \(s\) are the four rewards’ mean and sample SD. '
        r'\(\mathcal{V}\) contains valid, non-padding boundary decisions and excludes the forced first-frame action. '
        r'\(b\) is a sampled boundary bit, \(x\) is the acoustic input, and \(\pi_\theta\) is the boundary policy. '
        'Rewards and advantages are detached when updating the policy.</p>'
        '<p>The rate penalty is already inside the GRPO reward. There is <b>no extra differentiable rate loss</b>, '
        'entropy bonus, KL term, clipping ratio, or bilevel lookahead in these runs. Unlike the earlier bilevel experiment, '
        'this controller changes a rate target after validation; it does not differentiate through an inner decoder update.</p>'
        '<p>Details of the earlier proposal: <a href="https://borrisonxiao.github.io/jsalt26-downsampling/proposals/adaptive-frequency-curriculum.html">adaptive-frequency curriculum</a>.</p>'
    )
    body += '<details class="card"><summary>Exact joint objective and connection to bilevel optimization</summary>' + objective + '</details>'
    body += training_figures(hk, data)
    result_rows = []
    for arm, label in ARMS.items():
        values = [summary(data, arm, split, key) for key in ("WER", "Hz") for split in SPLITS]
        result_rows.append([label] + [f'{mean:.3f} ± {sd:.3f}' for mean, sd in values])
    body += hk.card(table(["Continuation", "Test-clean WER (%)", "Test-other WER (%)", "Clean rate (Hz)", "Other rate (Hz)"],
        result_rows, "Table 6. Final step-8,000 checkpoints; three-seed mean ± sample SD, not standard error. "
        "Lower WER and lower rate are desirable. WER is the logged main-scorer value; see the scoring note below.",
        "adaptive-results"), title="Completed test results — same training budget")
    body += '<p>Table 6 shows a rate–quality trade-off, not an accuracy win. Adaptive uses '
    body += f'{reduction:.1f}% less test-other rate than original, with mean WER +{delta:.3f} percentage point. '
    body += ('Compared with fixed-lower, it uses about 7% less test-other rate but has 0.088 point higher WER. '
             'The quality checks allow this because they compare against the initial checkpoint, not the fixed-target control. '
             'These descriptive three-seed results do not establish statistical significance or a minimum feasible rate.</p>')
    seed_rows = []
    for run in data["runs"]:
        seed_rows.append([str(run["seed"]), ARMS[run["arm"]]] + [f'{run["tests"][split][key]:.3f}'
                         for key in ("WER", "Hz") for split in SPLITS] + [run["test_job"]])
    body += '<details class="card"><summary>Per-seed test results and evaluation job IDs</summary>' + table(
        ["Seed", "Continuation", "Clean WER (%)", "Other WER (%)", "Clean Hz", "Other Hz", "Test job"],
        seed_rows, "Table 7. All nine final checkpoints, each evaluated on both full test sets. Rates are 50 × mean per-utterance kept ratio, not total tokens divided by total audio duration.",
        "adaptive-per-seed") + '</details>'
    controller_rows = []
    for run in data["runs"]:
        if run["arm"] != "adaptive":
            continue
        first, last = run["events"][0], run["events"][-1]
        changes = [f'{e["step"]:,}' for e in run["events"] if e["action"] == "lower_target"]
        controller_rows.append([str(run["seed"]), f'{first["target_hz"]:.3f}', '; '.join(changes),
            f'{last["target_hz"]:.3f} / {last["other_hz"]:.3f}',
            f'{last["ema_other_wer"]:.3f} / {first["other_wer"] + 0.1:.3f}'])
    body += hk.card(table(["Seed", "Initial target (Hz)", "Updates that lowered the target", "Final target / actual dev-other Hz", "Final EMA / allowed other WER (%)"],
        controller_rows, "Table 8. Observed adaptive-controller decisions, from all 17 paired dev checks per run. All runs ended at 8,000 additional updates; none reached the 7.5-Hz target floor.",
        "adaptive-decisions")
        + '<p><b>A real recovery example:</b> seed 3409 failed the raw dev-other quality check at step 1,000 '
        '(5.619% versus a 5.486% limit; its EMA still passed). The target stayed at 11.312 Hz and both policy and decoder continued learning with the penalty active. '
        'Checks at 1,500 and 2,000 passed, so the target fell to 10.812 Hz at 2,000. Seeds 3407 and 3408 had no failed quality checks; '
        'their holds came from rate attainment or the two-check requirement.</p>', title="What actually happened during training")
    sources = ''.join(f'<li>Seed {r["seed"]}: <code>{html.escape(r["source_checkpoint"])}</code></li>'
                      for r in data["runs"] if r["arm"] == "adaptive")
    body += hk.card(
        '<p><b>Scoring note:</b> the main scorer splits on a literal space, while per-utterance diagnostics collapse whitespace. '
        'A leading space therefore adds one main-scorer error in eight seed-3408/3409 split/run results (about 0.002 WER percentage point each). '
        'Saved predictions reproduce both conventions exactly. The tables retain the logged scores; no predictions or scoring code were changed. '
        'The evidence snapshot also records normalized WER and the affected utterance IDs. The earlier strict LaTeX collector remains unchanged.</p>'
        '<p><b>Checkpoint choice:</b> tests use final step 8,000 in every arm, fixed before testing. The separate lowest-rate, quality-qualified dev selector '
        'can choose an earlier checkpoint, including step 0 for an original-target control; that selector is not used for these tables. '
        'Test scores never controlled the EMA or target schedule.</p>'
        '<details><summary>Exact initialization checkpoints and reproducible evidence</summary><ul>' + sources + '</ul>'
        '<p>Source family: <code>speechllm_ls960_step24k_corrected/transformer_ar_local64_bigru/nll_mt</code>. '
        'The encoder is WavLM-Large, not the multilingual HuBERT encoders in the preceding section.</p>'
        '<p><a href="https://borrisonxiao.github.io/jsalt26-downsampling/reports/adaptive-frequency-results-20260919.json">Download the audited JSON snapshot</a>: '
        'full precision, both scoring conventions, dev checks, checkpoint/job provenance and input SHA256 values. '
        'Regenerate it with <code>python recipes/LibriSpeech/ASR/transformer/adaptive_frequency_report.py --collect</code>, then run the training-report generator. '
        'No new training or inference is needed.</p></details>', title="Scoring and provenance")
    return hk.section("Adaptive frequency continuation — completed three-seed results",
        lead=f"After a good LS960 starting point, adaptive training lowers the test-other rate by {reduction:.1f}% versus original-target continuation, with a {delta:.3f}-point increase in mean WER. Completed September 19, 2026; this is a measured result, not the earlier proposal.",
        body=body).replace('<section>', '<section id="adaptive-frequency-results">', 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect", action="store_true", help="Audit saved outputs and refresh the JSON input")
    parser.add_argument("--output", type=Path, default=SNAPSHOT)
    args = parser.parse_args()
    if not args.collect:
        parser.error("Choose --collect; the main training-report generator renders this section")
    data = collect()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    print(f"Audited {len(data['runs'])} runs; wrote {args.output}")
