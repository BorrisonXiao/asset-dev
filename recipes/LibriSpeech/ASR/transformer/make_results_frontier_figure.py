#!/usr/bin/env python3
"""Generate separate high-resolution WER/compression frontiers per SSL encoder.

The plots deliberately contain no point-value annotations. Exact values live in the
HTML tables; k labels appear only in the isolated fixed-baseline row so the learned-
model clusters around 12.5 Hz remain readable.
"""

import argparse
import math
import re
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


RECIPE = Path(__file__).resolve().parent
ROOT = RECIPE.parents[3]
RESULTS = RECIPE / "results"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts/segmenter/diagrams_highres"
SEEDS = (3407, 3408, 3409)

WIDTH_PX, HEIGHT_PX, DPI = 3233, 2550, 300
TEXT = "#111923"
GRID = "#d3d9dd"
FIXED = "#7b858c"
NO_DOWN = "#9a5b73"
FRAME_HZ = 50.0

MODEL_SPECS = (
    ("CNN · NLL frozen", "cnn", "nll_frozen", "#c9782e", "s"),
    ("CNN · NLL co-trained", "cnn", "nll_mt", "#178a17", "D"),
    ("CNN · CER co-trained", "cnn", "cer_mt", "#2a78d6", "^"),
    ("Transformer · NLL frozen", "transformer", "nll_frozen", "#b07aa1", "s"),
    ("Transformer · NLL co-trained", "transformer", "nll_mt", "#6741a5", "D"),
    ("Transformer · CER co-trained", "transformer", "cer_mt", "#d14f7b", "^"),
)


def mean_sd(values):
    if not values:
        return float("nan"), float("nan")
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance)


def read_wer(path):
    if not path.exists():
        return None
    match = re.search(r"^%WER ([0-9.]+) \[ (\d+) / (\d+),", path.read_text(), re.M)
    if not match:
        return None
    errors, words = int(match.group(2)), int(match.group(3))
    return 100.0 * errors / max(words, 1)


def read_test_clean_rho(path):
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        match = re.match(r"Epoch loaded: \d+ - test .*?rho_mean: ([-+0-9.eE]+)", line)
        if match:
            return float(match.group(1))
    return None


def collect_fixed(root):
    rows = []
    for k in (3, 4, 5, 6, 8):
        clean_values, other_values = [], []
        for seed in SEEDS:
            run = root / f"fixed_rate_k{k}_tc100/{seed}/wer_results"
            clean = read_wer(run / "wer_test-clean.txt")
            other = read_wer(run / "wer_test-other.txt")
            if clean is not None and other is not None:
                clean_values.append(clean)
                other_values.append(other)
        clean, clean_sd = mean_sd(clean_values)
        other, other_sd = mean_sd(other_values)
        rows.append((k, 1.0 / k, clean, clean_sd, other, other_sd, len(clean_values)))
    return rows


def collect_no_down(root):
    values = {}
    for split in ("test-clean", "test-other"):
        split_values = []
        for seed in SEEDS:
            value = read_wer(
                root / f"no_downsampling_tc100/{seed}/wer_results/wer_{split}.txt"
            )
            if value is not None:
                split_values.append(value)
        values[split] = mean_sd(split_values)
    return (
        values["test-clean"][0], values["test-clean"][1],
        values["test-other"][0], values["test-other"][1],
    )


def collect_models(root):
    rows = []
    sweep = root / "multiseed_shared_warmup_onpolicy"
    for label, backbone, variant, color, marker in MODEL_SPECS:
        rhos, clean_values, other_values = [], [], []
        for seed in SEEDS:
            run = sweep / f"{variant}/{backbone}/{seed}"
            rho = read_test_clean_rho(run / "train_log.txt")
            clean = read_wer(run / "wer_results/wer_test-clean.txt")
            other = read_wer(run / "wer_results/wer_test-other.txt")
            if rho is not None and clean is not None and other is not None:
                rhos.append(rho)
                clean_values.append(clean)
                other_values.append(other)
        rho, _ = mean_sd(rhos)
        clean, clean_sd = mean_sd(clean_values)
        other, other_sd = mean_sd(other_values)
        rows.append((label, rho, clean, clean_sd, other, other_sd, color, marker))
    return rows


def style_axis(ax, split, row_label, row_idx, metric_idx):
    if row_idx == 0:
        ax.set_title(split, fontsize=18, fontweight="bold", color=TEXT, pad=6)
    ax.set_xlim(5.5, 17.5)
    ax.set_xticks((7.5, 10.0, 12.5, 15.0, 17.5))
    if row_idx == 2:
        ax.set_xlabel("Audio-token frequency (Hz)  (← more compression)", fontsize=12.5)
    if metric_idx == 0:
        ax.set_ylabel(f"{row_label}\nWord error rate (%)", fontsize=11.5)
    ax.tick_params(axis="both", labelsize=10.5, width=0.9, length=5)
    ax.grid(axis="y", color=GRID, linewidth=0.85)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_frontier(encoder_name, fixed_rows, models, no_down, output):
    cnn_models = [model for model in models
                  if not model[0].lower().startswith("transformer")]
    transformer_models = [model for model in models
                          if model[0].lower().startswith("transformer")]
    row_specs = (
        ("Fixed baselines", []),
        ("CNN policy", cnn_models),
        ("Transformer policy", transformer_models),
    )
    fig, axes = plt.subplots(
        3, 2, figsize=(WIDTH_PX / DPI, HEIGHT_PX / DPI), dpi=DPI,
        facecolor="white", sharex=True
    )
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.125, top=0.91,
                        wspace=0.18, hspace=0.30)

    ordered = sorted(fixed_rows, key=lambda row: row[1])
    for row_idx, (row_label, row_models) in enumerate(row_specs):
        for metric_idx, split in enumerate(("test-clean", "test-other")):
            ax = axes[row_idx, metric_idx]
            value_idx = 2 if metric_idx == 0 else 4
            sd_idx = value_idx + 1
            frequencies = [FRAME_HZ * row[1] for row in ordered]
            values = [row[value_idx] for row in ordered]
            sds = [row[sd_idx] for row in ordered]
            ax.plot(frequencies, values, color=FIXED, marker="o", markersize=5.6,
                    linewidth=1.75, zorder=3)
            if any(sd > 0 for sd in sds):
                ax.errorbar(frequencies, values, yerr=sds, fmt="none", ecolor=FIXED,
                            elinewidth=1.1, capsize=3, alpha=0.8, zorder=2)
            if row_idx == 0:
                for fixed in ordered:
                    ax.annotate(f"k={fixed[0]}",
                                (FRAME_HZ * fixed[1], fixed[value_idx]),
                                xytext=(0, 8), textcoords="offset points",
                                ha="center", va="bottom", fontsize=8.5,
                                color="#66727b")

            for label, rho, clean, clean_sd, other, other_sd, color, marker in row_models:
                value, sd = (clean, clean_sd) if metric_idx == 0 else (other, other_sd)
                ax.errorbar(FRAME_HZ * rho, value, yerr=sd, fmt=marker, markersize=7.8,
                            markerfacecolor=color, markeredgecolor="white",
                            markeredgewidth=0.8, ecolor=color, elinewidth=1.4,
                            capsize=4, zorder=5)

            nd_value, nd_sd = (no_down[0], no_down[1]) if metric_idx == 0 else (
                no_down[2], no_down[3]
            )
            ax.axhline(nd_value, color=NO_DOWN, linestyle=(0, (5, 3)),
                       linewidth=1.35, zorder=1)
            if nd_sd > 0:
                ax.axhspan(nd_value - nd_sd, nd_value + nd_sd,
                           color=NO_DOWN, alpha=0.08, lw=0, zorder=0)
            style_axis(ax, split, row_label, row_idx, metric_idx)

    handles = [
        Line2D([0], [0], color=FIXED, marker="o", markersize=5.6,
               linewidth=1.75, label="Fixed-rate pooling"),
        Line2D([0], [0], color=NO_DOWN, linestyle=(0, (5, 3)),
               linewidth=1.35, label="No downsampling"),
    ]
    handles.extend(
        Line2D([0], [0], color=color, marker=marker, markerfacecolor=color,
               markeredgecolor="white", markersize=7.8, linewidth=0, label=label)
        for label, _, _, _, _, _, color, marker in models
    )
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.012),
               ncol=4, frameon=False, fontsize=8.7, handlelength=2.0,
               columnspacing=1.8, handletextpad=0.55)
    fig.suptitle(f"{encoder_name} · trained on LibriSpeech-100h",
                 fontsize=19, fontweight="bold", color=TEXT, y=0.985)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"wrote {output} ({WIDTH_PX}x{HEIGHT_PX})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    configs = (
        ("wav2vec2-base-960h", "speechllm_fixed_pooling", "speechllm_segmenter",
         "05_results_frontier_wav2vec2.png"),
        ("WavLM-Large", "speechllm_fixed_pooling_wavlm", "speechllm_segmenter_wavlm",
         "05_results_frontier_wavlm.png"),
    )
    for encoder, fixed_dir, learned_dir, filename in configs:
        fixed_root = RESULTS / fixed_dir
        learned_root = RESULTS / learned_dir
        output = args.output_dir / filename
        plot_frontier(
            encoder,
            collect_fixed(fixed_root),
            collect_models(learned_root),
            collect_no_down(fixed_root),
            output,
        )
        if encoder == "WavLM-Large":
            legacy_output = args.output_dir / "05_results_frontier.png"
            shutil.copyfile(output, legacy_output)
            print(f"updated compatibility path {legacy_output}")


if __name__ == "__main__":
    main()
