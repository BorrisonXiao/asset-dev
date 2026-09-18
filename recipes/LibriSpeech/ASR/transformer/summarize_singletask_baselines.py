#!/usr/bin/env python3
"""Collect the single-task non-ASR baselines into a rate-quality table.

One row per (task, arm), aggregated over seeds, reading each run's own
``task_metrics.jsonl``. The question the table answers is which of three things
limits the non-ASR tasks:

* if quality is flat from 50 Hz down to 5 Hz, the **rate** is not the
  bottleneck and the pooled-prefix interface is;
* if ``learned_frozen`` beats a fixed grid at a comparable rate, boundary
  **placement** is doing real work;
* if ``learned_frozen`` here beats the multi-task cohort, the task **mixture**
  was interfering;
* if ``learned_grpo`` beats ``learned_frozen`` at the same step budget, training
  the boundary policy buys something -- a comparison the multi-task runs could
  not make, since their ASR-selected checkpoint was always pre-policy.

Both the selected-checkpoint TEST number and the best VALID number are shown:
TEST is the result, and best-VALID says whether selection found the run's own
peak or a mid-training point.

Usage::

    python summarize_singletask_baselines.py
    python summarize_singletask_baselines.py --root results/speechllm_singletask_nonasr
    python summarize_singletask_baselines.py --csv baselines.csv
"""

import argparse
import csv
import json
import os
import statistics
import sys

# Arm order is the reported order: descending rate, then the learned policy.
ARMS = ("nods_50hz", "fixed_25hz", "fixed_10hz", "fixed_5hz",
        "learned_frozen", "learned_grpo")
TASKS = ("emotion", "speaker_count", "intent")

# The metric each task is judged on, and the rate key that travels with it.
JUDGED = {
    "emotion": "macro_f1_emotion",
    "speaker_count": "acc_speaker_count",
    "intent": "acc_intent",
}
# Reference points to compare against, so a number is never read in isolation.
# These are the multi-task LC-aux cohort's THREE seeds (3407/3408/3409) on the
# same test splits, not the single published checkpoint: comparing a 3-seed mean
# against one run would flatter whichever side happened to draw a good seed.
# Published single checkpoint, for cross-reference: emotion 0.350,
# speaker_count 0.720, intent 0.994 (all seed 3408, step 3000).
MULTITASK_SEEDS = {
    "emotion": (0.285, 0.350, 0.285),
    "speaker_count": (0.493, 0.720, 0.573),
    "intent": (0.993, 0.994, 0.988),
}
REFERENCE = {
    task: ("multi-task n=3", statistics.fmean(v), statistics.stdev(v))
    for task, v in MULTITASK_SEEDS.items()
}
CEILING = {
    # Validated CountNet CRNN port on this same test split.
    "speaker_count": ("CountNet ref", 0.871),
}


def read_rows(path):
    if not os.path.isfile(path):
        return []
    out = []
    for line in open(path, encoding="utf-8"):
        if '"stage"' not in line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def run_summary(run_dir, task):
    """(test_metric, test_hz, best_valid, n_valid) for one run, or None."""
    rows = read_rows(os.path.join(run_dir, "task_metrics.jsonl"))
    if not rows:
        return None
    key = JUDGED[task]
    hz_key = "f_audio_hz_" + task
    test = [r for r in rows if r["stage"] == "TEST" and key in r]
    valid = [r for r in rows if r["stage"] == "VALID" and key in r]
    if not test:
        return None
    return {
        "test": float(test[0][key]),
        "hz": float(test[0].get(hz_key, float("nan"))),
        "best_valid": max(float(r[key]) for r in valid) if valid else None,
        "n_valid": len(valid),
        "step": test[0].get("optimizer_step"),
    }


def agg(values):
    """mean, sample SD (None when a single seed), and the raw list."""
    if not values:
        return None, None
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else None
    return mean, sd


def fmt(mean, sd, places=3):
    if mean is None:
        return "-"
    if sd is None:
        return f"{mean:.{places}f}"
    return f"{mean:.{places}f}+-{sd:.{places}f}"


DEFAULT_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results",
    "speechllm_singletask_nonasr",
)


def collect(root=DEFAULT_ROOT):
    """One aggregated row per (task, arm). Shared with the training report.

    Kept as a function rather than inlined in ``main`` so the report and the CLI
    cannot drift into two different definitions of the same numbers.
    """
    table = []
    for task in TASKS:
        for arm in ARMS:
            arm_dir = os.path.join(root, task, arm)
            if not os.path.isdir(arm_dir):
                continue
            seeds = sorted(
                d for d in os.listdir(arm_dir)
                if os.path.isdir(os.path.join(arm_dir, d))
            )
            runs = []
            for seed in seeds:
                got = run_summary(os.path.join(arm_dir, seed), task)
                if got:
                    got["seed"] = seed
                    runs.append(got)
            if not runs:
                # Distinguish "not finished" from "no such arm": both are
                # legitimate mid-sweep states and neither is a result.
                table.append(
                    {"task": task, "arm": arm, "n": 0, "pending": len(seeds)}
                )
                continue
            t_mean, t_sd = agg([r["test"] for r in runs])
            h_mean, _ = agg([r["hz"] for r in runs])
            v_mean, v_sd = agg(
                [r["best_valid"] for r in runs if r["best_valid"] is not None]
            )
            table.append({
                "task": task, "arm": arm, "n": len(runs),
                "pending": len(seeds) - len(runs),
                "test_mean": t_mean, "test_sd": t_sd,
                "hz": h_mean,
                "valid_mean": v_mean, "valid_sd": v_sd,
                "spread": max(r["test"] for r in runs)
                          - min(r["test"] for r in runs),
                "seeds": [(r["seed"], r["test"], r["step"]) for r in runs],
            })
    return table


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--csv", default=None, help="also write the rows as CSV")
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        sys.exit(f"no such results root: {args.root}")
    table = collect(args.root)

    width = 92
    for task in TASKS:
        rows = [r for r in table if r["task"] == task]
        if not rows:
            continue
        metric = JUDGED[task]
        print("=" * width)
        print(f"{task}   (judged on {metric}; lower f_audio is better)")
        refs = []
        if task in REFERENCE:
            name, val, sd = REFERENCE[task]
            refs.append(f"{name} {val:.3f}+-{sd:.3f}")
        if task in CEILING:
            name, val = CEILING[task]
            refs.append(f"{name} ceiling {val:.3f}")
        if refs:
            print("   reference: " + " | ".join(refs))
        print("-" * width)
        print(f"{'arm':<16}{'f_audio':>9}{'TEST':>18}{'best VALID':>18}"
              f"{'seed spread':>13}{'n':>5}")
        print("-" * width)
        for r in rows:
            if r["n"] == 0:
                print(f"{r['arm']:<16}{'-':>9}{'(pending)':>18}"
                      f"{'-':>18}{'-':>13}{r['pending']:>5}")
                continue
            note = "" if not r["pending"] else f" (+{r['pending']} pending)"
            print(f"{r['arm']:<16}{r['hz']:>8.2f} "
                  f"{fmt(r['test_mean'], r['test_sd']):>17}"
                  f"{fmt(r['valid_mean'], r['valid_sd']):>18}"
                  f"{r['spread']:>13.3f}{r['n']:>5}{note}")
        print()

    if args.csv:
        cols = ["task", "arm", "n", "pending", "hz", "test_mean", "test_sd",
                "valid_mean", "valid_sd", "spread"]
        with open(args.csv, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=cols, extrasaction="ignore"
            )
            writer.writeheader()
            for r in table:
                writer.writerow(r)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
