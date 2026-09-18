#!/usr/bin/env python3
"""Verify six test evaluations; generate JSON, Markdown, and a booktabs table.

Outputs stay with the evaluation artifacts; this does not edit the manuscript.
Re-run with --results EVAL_ROOT --output EVAL_ROOT/results.tex.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re

REPO = Path(__file__).resolve().parents[4]
DEFAULT_RESULTS = REPO / "artifacts/segmenter/ls960_adaptive_frequency_test_eval_20260918"
ARMS = {"original": "Original fixed target", "fixed_lower": "Fixed lower target", "adaptive": "Adaptive target"}
SPLITS = {"test-clean": 2620, "test-other": 2939}
METRIC_DIRECTION = {"test-clean_WER": False, "test-other_WER": False,
                    "test-clean_Hz": False, "test-other_Hz": False}
WER_RE = re.compile(r"^%WER\s+([0-9.]+)\s+\[\s*(\d+)\s*/\s*(\d+)")


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def indexed(rows):
    result = {row["split"]: row for row in rows}
    if len(result) != len(rows) or set(result) != set(SPLITS):
        raise ValueError("Require exactly one completed result per test split")
    return result


def close(actual, expected):
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-9):
        raise ValueError(f"Metric mismatch: {actual} vs {expected}")


def fmt(value):
    try:
        number = float(value)
        return f"{number:.2f}" if math.isfinite(number) else "--"
    except (ValueError, TypeError):
        return "--"


def load_results(root):
    manifest = json.loads((root / "evaluation_manifest.json").read_text())
    results = []
    expected_ids = {}
    for split, info in manifest["splits"].items():
        path = Path(info["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != info["sha256"]:
            raise ValueError(f"Manifest changed: {path}")
        with path.open(newline="") as stream:
            expected_ids[split] = {row["ID"] for row in csv.DictReader(stream)}
    for arm, label in ARMS.items():
        directory = root / arm
        metrics = indexed(jsonl(directory / "test_metrics.jsonl"))
        benchmarks = indexed(jsonl(directory / "benchmark.jsonl"))
        arm_info = manifest["arms"][arm]
        row = {"arm": arm, "system": label, "job_id": arm_info["job_id"],
               "checkpoint": arm_info["checkpoint"], "splits": {}}
        for split, count in SPLITS.items():
            metric, bench = metrics[split], benchmarks[split]
            if metric["checkpoint"] != arm_info["checkpoint"] or metric["continuation_optimizer_step"] != 8000:
                raise ValueError(f"Wrong checkpoint: {arm}/{split}")
            if bench["decoding_protocol"] != "batch_invariant_left_packed_duration_cap_v1" or bench["batch_size"] != 8:
                raise ValueError(f"Wrong decoding protocol: {arm}/{split}")
            utterances = jsonl(directory / f"benchmark_{split}_utterances.jsonl")
            if len(utterances) != count or {u["id"] for u in utterances} != expected_ids[split]:
                raise ValueError(f"Incomplete or duplicate utterances: {arm}/{split}")
            if metric["utterances"] != count or bench["utterances"] != count:
                raise ValueError(f"Incorrect scoring count: {arm}/{split}")
            wer_path = directory / "wer_results" / f"wer_{split}.txt"
            match = WER_RE.match(wer_path.read_text().splitlines()[0])
            if match is None:
                raise ValueError(f"Invalid WER report: {wer_path}")
            errors, words = int(match[2]), int(match[3])
            close(metric["WER"], 100.0 * errors / words)
            close(bench["WER"], metric["WER"])
            if sum(u["word_errors"] for u in utterances) != errors or sum(u["reference_words"] for u in utterances) != words:
                raise ValueError(f"Per-utterance and corpus word counts differ: {arm}/{split}")
            row[f"{split}_WER"] = metric["WER"]
            row[f"{split}_Hz"] = metric["rate_hz_50rho"]
            row["splits"][split] = {**metric, "word_errors": errors, "reference_words": words,
                                     "duration_based_token_hz": bench["token_frequency_hz"],
                                     "stage_seconds": bench["stage_seconds_including_data_and_metrics"],
                                     "forward_rtf": bench["forward_rtf"],
                                     "utterances_per_second": bench["utterances_per_second"],
                                     "hardware": bench["hardware"]}
        results.append(row)
    return results


def make_table(rows):
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"System & Clean WER (\%) & Other WER (\%) & Clean Hz & Other Hz \\",
             r"\midrule"]
    for row in rows:
        lines.append(" & ".join([row["system"]] + [fmt(row.get(key)) for key in METRIC_DIRECTION]) + r" \\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", ""])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = load_results(args.results)
    output = args.output or args.results / "results.tex"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(make_table(rows))
    summary = {"continuation_updates": 8000, "source_updates": 24000, "seed": 3407,
               "selection": "Final checkpoints, matched continuation budget; not test-selected",
               "rate_definition": "50 times mean per-utterance kept-frame ratio, as in training",
               "verified_full_test_sets": True, "results": rows}
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["# LS960 continuation: final-checkpoint test evaluation", "",
             "All runs: seed 3407; 8,000 additional updates after the 24,000-update checkpoint.",
             "Final checkpoints chosen before test evaluation; no test-set tuning.", "",
             "| System | Test-clean WER (%) | Test-other WER (%) | Clean Hz | Other Hz |",
             "|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append("| " + " | ".join([row["system"]] + [fmt(row.get(key)) for key in METRIC_DIRECTION]) + " |")
    lines += ["", "Hz uses 50 × the mean per-utterance kept-frame ratio, matching the curriculum logs.",
              "All six results cover the complete splits (2,620 clean / 2,939 other utterances).",
              "Corpus WER is verified against the WER reports and summed per-utterance word errors.",
              "Single seed: these differences do not establish statistical significance.", "",
              "## Evidence", ""]
    for row in rows:
        lines += [f"- {row['system']}: job {row['job_id']}; checkpoint `{row['checkpoint']}`.",
                  f"  Metrics: `{row['arm']}/test_metrics.jsonl`; predictions and timing: `{row['arm']}/benchmark*`; word alignments: `{row['arm']}/wer_results/`."]
    lines += ["", "## Regenerate", "",
              "Run `make tables` from this directory. `results.tex` is a booktabs snippet; no macros are emitted.",
              r"Optional LaTeX use: `\usepackage{booktabs}`, then `\input{path/to/results.tex}`.", ""]
    output.with_suffix(".md").write_text("\n".join(lines))
    print("\n".join(lines[:13]))
    print(f"Wrote {output}, {output.with_suffix('.json')}, {output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
