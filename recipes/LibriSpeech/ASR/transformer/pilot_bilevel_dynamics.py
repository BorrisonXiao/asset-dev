#!/usr/bin/env python3
"""Retrospective bilevel diagnostic on the six matched WavLM RL trajectories.

The current joint loop improves decoder CE and the segmenter reward on the training
batch, while model selection uses held-out WER.  This script checks whether those
rankings agree during the ten RL epochs (epochs 3--12) for Bernoulli and first-order
AR policies, seeds 3407/3408/3409.

It does not claim to test a bilevel optimizer.  It tests the prerequisite: whether
an inner-training optimum and an outer held-out optimum measurably diverge.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EPOCH_RE = re.compile(r"^epoch: (?P<epoch>\d+),")
FIELD_RE = re.compile(
    r"([A-Za-z_]+): ([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)"
)


def ranks(values):
    """Average ranks with ties, matching scipy.stats.rankdata(method='average')."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    out = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while (
            stop < len(values) and values[order[stop]] == values[order[start]]
        ):
            stop += 1
        out[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return out


def spearman(x, y):
    rx, ry = ranks(x), ranks(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def parse_log(path):
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = EPOCH_RE.match(line)
        if not match or "train reward:" not in line:
            continue
        train_part, valid_part = line.split(" - valid ", maxsplit=1)
        train = {
            key: float(value) for key, value in FIELD_RE.findall(train_part)
        }
        valid = {
            key: float(value) for key, value in FIELD_RE.findall(valid_part)
        }
        records.append(
            {
                "epoch": int(match.group("epoch")),
                "train_dec_ce": train["dec_ce"],
                "train_reward": train["reward"],
                "train_rho": train["rho_mean"],
                "valid_loss": valid["loss"],
                "valid_wer": valid["WER"],
                "valid_rho": valid["rho_mean"],
            }
        )
    if len(records) != 10:
        raise ValueError(
            f"expected 10 RL epochs in {path}, found {len(records)}"
        )
    return records


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "sample_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
    }


def summarize_run(policy, seed, path, records):
    inner = min(records, key=lambda row: row["train_dec_ce"])
    reward = max(records, key=lambda row: row["train_reward"])
    outer_wer = min(records, key=lambda row: row["valid_wer"])
    outer_loss = min(records, key=lambda row: row["valid_loss"])
    first = records[0]
    return {
        "policy": policy,
        "seed": seed,
        "source": str(path),
        "records": records,
        "first_rl_epoch": first["epoch"],
        "inner_ce_epoch": inner["epoch"],
        "reward_epoch": reward["epoch"],
        "outer_wer_epoch": outer_wer["epoch"],
        "outer_loss_epoch": outer_loss["epoch"],
        "train_ce_reduction_pct": 100.0
        * (first["train_dec_ce"] - inner["train_dec_ce"])
        / first["train_dec_ce"],
        "wer_regret_if_inner_selected": inner["valid_wer"]
        - outer_wer["valid_wer"],
        "valid_loss_regret_if_inner_selected": inner["valid_loss"]
        - outer_loss["valid_loss"],
        "rho_span": max(row["valid_rho"] for row in records)
        - min(row["valid_rho"] for row in records),
        "spearman_train_ce_vs_valid_wer": spearman(
            [row["train_dec_ce"] for row in records],
            [row["valid_wer"] for row in records],
        ),
        "spearman_train_reward_vs_valid_wer": spearman(
            [row["train_reward"] for row in records],
            [row["valid_wer"] for row in records],
        ),
        "inner_and_outer_wer_agree": inner["epoch"] == outer_wer["epoch"],
        "reward_and_outer_wer_agree": reward["epoch"] == outer_wer["epoch"],
    }


def aggregate(runs):
    return {
        "runs": len(runs),
        "inner_outer_epoch_agreement": sum(
            run["inner_and_outer_wer_agree"] for run in runs
        ),
        "reward_outer_epoch_agreement": sum(
            run["reward_and_outer_wer_agree"] for run in runs
        ),
        "train_ce_reduction_pct": mean_sd(
            [run["train_ce_reduction_pct"] for run in runs]
        ),
        "wer_regret_if_inner_selected": mean_sd(
            [run["wer_regret_if_inner_selected"] for run in runs]
        ),
        "valid_loss_regret_if_inner_selected": mean_sd(
            [run["valid_loss_regret_if_inner_selected"] for run in runs]
        ),
        "rho_span": mean_sd([run["rho_span"] for run in runs]),
        "spearman_train_ce_vs_valid_wer": mean_sd(
            [run["spearman_train_ce_vs_valid_wer"] for run in runs]
        ),
        "spearman_train_reward_vs_valid_wer": mean_sd(
            [run["spearman_train_reward_vs_valid_wer"] for run in runs]
        ),
    }


def plot_runs(runs, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.8), sharex=True)
    colors = {"3407": "#245d91", "3408": "#176b4d", "3409": "#9a5b13"}
    policy_labels = {
        "bernoulli": "Bernoulli",
        "autoregressive": "First-order AR",
    }
    for row_index, policy in enumerate(("bernoulli", "autoregressive")):
        policy_runs = [run for run in runs if run["policy"] == policy]
        for run in policy_runs:
            epochs = [record["epoch"] for record in run["records"]]
            first_ce = run["records"][0]["train_dec_ce"]
            axes[row_index, 0].plot(
                epochs,
                [
                    record["train_dec_ce"] / first_ce
                    for record in run["records"]
                ],
                marker="o",
                ms=3.8,
                lw=1.8,
                color=colors[run["seed"]],
                label=f"seed {run['seed']}",
            )
            axes[row_index, 1].plot(
                epochs,
                [record["valid_wer"] for record in run["records"]],
                marker="o",
                ms=3.8,
                lw=1.8,
                color=colors[run["seed"]],
                label=f"seed {run['seed']}",
            )
            best = min(run["records"], key=lambda record: record["valid_wer"])
            axes[row_index, 1].scatter(
                [best["epoch"]],
                [best["valid_wer"]],
                s=70,
                facecolors="white",
                edgecolors=colors[run["seed"]],
                linewidths=2.0,
                zorder=4,
            )
        axes[row_index, 0].set_ylabel(
            f"{policy_labels[policy]}\nrelative train decoder CE"
        )
        axes[row_index, 0].axhline(1.0, color="#a9aca7", lw=1.0, ls="--")
        axes[row_index, 1].set_ylabel("dev-clean WER (%)")
        for axis in axes[row_index]:
            axis.grid(axis="y", color="#e3e2dc", lw=0.9)
            axis.spines[["top", "right"]].set_visible(False)
            axis.margins(x=0.04, y=0.12)
    axes[0, 0].set_title("Inner training signal keeps improving", pad=14)
    axes[0, 1].set_title("Held-out ASR has an early optimum", pad=14)
    axes[1, 0].set_xlabel("training epoch")
    axes[1, 1].set_xlabel("training epoch")
    axes[0, 0].legend(frameon=False, ncol=3, fontsize=9, loc="upper right")
    fig.suptitle(
        "Matched WavLM joint-RL trajectories: inner versus held-out behavior",
        fontsize=15,
        y=0.985,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955), h_pad=2.2, w_pad=2.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_path,
        dpi=180,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.12,
    )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("recipes/LibriSpeech/ASR/transformer/results")
        / "speechllm_segmenter_wavlm/oracleclose_bestinit",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/segmenter/bilevel_pilot"),
    )
    args = parser.parse_args()

    runs = []
    for policy in ("bernoulli", "autoregressive"):
        for seed in ("3407", "3408", "3409"):
            path = args.root / policy / seed / "train_log.txt"
            runs.append(summarize_run(policy, seed, path, parse_log(path)))

    result = {
        "study": "Retrospective inner/outer alignment diagnostic",
        "scope": "Six matched WavLM joint-RL runs; 10 RL epochs per run",
        "limitations": [
            "This compares recorded training and held-out trajectories; it does not train a bilevel optimizer.",
            "Epoch-level statistics cannot isolate decoder and segmenter changes within an epoch.",
            "Dev-clean WER is a selection metric, not a differentiable outer loss.",
        ],
        "runs": runs,
        "aggregate": {
            "all": aggregate(runs),
            "bernoulli": aggregate(
                [run for run in runs if run["policy"] == "bernoulli"]
            ),
            "autoregressive": aggregate(
                [run for run in runs if run["policy"] == "autoregressive"]
            ),
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "pilot_results.json"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    plot_runs(runs, args.out_dir / "inner_outer_dynamics.png")
    print(json.dumps(result["aggregate"], indent=2))
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
