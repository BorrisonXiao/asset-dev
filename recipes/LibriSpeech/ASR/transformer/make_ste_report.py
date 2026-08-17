#!/usr/bin/env python3
"""Build the standalone HTML report on straight-through gradients for the learned segmenter.

Output: artifacts/segmenter/ste_analysis/report.html   (deliberately a different subdirectory
from artifacts/segmenter/report.html, which is the training-dynamics report.)

Run with the project venv (needs numpy + matplotlib + torch):
    ~/cxiao/envs/jointllm/bin/python recipes/LibriSpeech/ASR/transformer/make_ste_report.py
"""

import argparse
import itertools
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.expanduser("~/.claude-scale/skills/html-report"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import htmlkit as hk  # noqa: E402

from segment_pooling import compute_segment_ids, mean_pool_segments  # noqa: E402

# Palette chosen to read on the opaque white figure background in both page themes.
INK = "#1c1e21"
MUTED = "#6b7280"
BLUE = "#2f6fb2"
GREEN = "#3f7d4e"
PINK = "#b4477d"
RED = "#b23b3b"
ORANGE = "#c07a2c"
GRID = "#d8d6d0"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
})


def _png(fig):
    return hk.mpl_png(fig, facecolor="white")


def label_report_assets(body):
    """Insert sequential visible labels before every table and rendered figure."""
    counts = {"table": 0, "figure": 0}
    pattern = re.compile(r'<table\b|<img\s+class=["\']fig["\']')

    def add_label(match):
        kind = "table" if match.group(0).startswith("<table") else "figure"
        counts[kind] += 1
        return '<div class="asset-label">%s %d</div>%s' % (
            kind.title(), counts[kind], match.group(0)
        )

    return pattern.sub(add_label, body)


# ---------------------------------------------------------------------------
# shared toy problem (used by several figures so the numbers agree)
# ---------------------------------------------------------------------------
X5 = torch.tensor([[1.0, 10.0], [3.0, 30.0], [5.0, 50.0], [7.0, 70.0], [9.0, 90.0]])
PAD5 = torch.zeros(1, 5, dtype=torch.bool)
B5 = torch.tensor([[0, 0, 1, 0, 1]])          # frames 0 1 | 2 3 | 4
P5 = torch.tensor([0.00, 0.10, 0.90, 0.20, 0.80])


def membership_hard(boundary, pad):
    seg_ids, num_seg, Tn = compute_segment_ids(boundary, pad)
    T = boundary.shape[1]
    M = torch.zeros(Tn, T)
    for t in range(T):
        if seg_ids[0, t] >= 0:
            M[seg_ids[0, t], t] = 1.0
    return M / M.sum(1, keepdim=True).clamp(min=1.0)


def membership_soft_survival(p, starts, T, Tn):
    """Row j = P(no boundary fired between segment j's start and frame t), normalised."""
    M = torch.zeros(Tn, T)
    for j, s in enumerate(starts):
        w = 1.0
        for t in range(s, T):
            if t > s:
                w = w * (1 - float(p[t]))
            M[j, t] = w
    return M / M.sum(1, keepdim=True).clamp(min=1e-8)


# ---------------------------------------------------------------------------
# Figure 1 — the loss is piecewise constant in the logits
# ---------------------------------------------------------------------------
def fig_piecewise():
    """Real content loss over a grid of two boundary logits (3-frame utterance)."""
    X = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    pad = torch.zeros(1, 3, dtype=torch.bool)
    target, _ = mean_pool_segments(X.unsqueeze(0), pad, torch.tensor([[0, 1, 0]]))

    n = 220
    ax_lin = np.linspace(-3, 3, n)
    Z = np.zeros((n, n))
    for i, l2 in enumerate(ax_lin):
        for j, l1 in enumerate(ax_lin):
            b = torch.tensor([[0, int(l1 > 0), int(l2 > 0)]])
            o, _ = mean_pool_segments(X.unsqueeze(0), pad, b)
            k = min(o.shape[1], target.shape[1])
            Z[i, j] = float(((o[:, :k] - target[:, :k]) ** 2).mean())

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 3.9))
    im = a1.pcolormesh(ax_lin, ax_lin, Z, cmap="RdYlBu_r", shading="auto")
    a1.axhline(0, color=INK, lw=1.1)
    a1.axvline(0, color=INK, lw=1.1)
    for (xx, yy, lab) in [(-1.6, -1.6, "1 segment"), (1.6, -1.6, "2 segments"),
                          (-1.6, 1.6, "2 segments"), (1.6, 1.6, "3 segments")]:
        a1.text(xx, yy, lab, ha="center", va="center", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=GRID, alpha=0.9))
    a1.set_xlabel("boundary logit, frame 1")
    a1.set_ylabel("boundary logit, frame 2")
    a1.set_title("loss over the two logits", fontsize=10)
    fig.colorbar(im, ax=a1, fraction=0.046, pad=0.03).set_label("content loss", fontsize=8)

    a2.plot(ax_lin, Z[n // 2 + 30, :], color=BLUE, lw=2)
    a2.set_xlabel("boundary logit, frame 1  (frame 2 held fixed)")
    a2.set_ylabel("content loss")
    a2.set_title("a slice: flat, then a step", fontsize=10)
    a2.axvline(0, color=MUTED, ls="--", lw=1)
    a2.annotate("gradient is exactly 0\neverywhere it exists", xy=(-2.0, Z[n // 2 + 30, 20]),
                xytext=(-2.9, Z[n // 2 + 30].max() * 0.72), fontsize=8.5, color=MUTED,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
    a2.annotate("undefined here", xy=(0, Z[n // 2 + 30].mean()), xytext=(0.35, Z[n // 2 + 30].max() * 0.9),
                fontsize=8.5, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1))
    a2.grid(alpha=0.25, color=GRID)
    fig.tight_layout()
    return _png(fig)


# ---------------------------------------------------------------------------
# Figure 2 — the feasible set is finite
# ---------------------------------------------------------------------------
def fig_partitions(T=5):
    parts = list(itertools.product([0, 1], repeat=T - 1))
    cols, rows = 4, (len(parts) + 3) // 4
    fig, axes = plt.subplots(rows, cols, figsize=(9.4, 0.62 * rows + 0.7))
    shades = [BLUE, GREEN, PINK, ORANGE, "#7d6bb0"]
    for idx, ax in enumerate(axes.ravel()):
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        if idx >= len(parts):
            continue
        b = (0,) + parts[idx]
        seg, sid = 0, []
        for t in range(T):
            if t > 0 and b[t] == 1:
                seg += 1
            sid.append(seg)
        for t in range(T):
            ax.add_patch(mpatches.Rectangle((t, 0), 0.94, 1, fc=shades[sid[t] % len(shades)],
                                            ec="white", lw=1.4, alpha=0.85))
        ax.set_xlim(-0.15, T); ax.set_ylim(-0.15, 1.15)
        ax.set_title(f"{seg + 1} seg", fontsize=7.5, color=MUTED, pad=2)
    fig.suptitle(f"Every possible segmentation of a {T}-frame utterance: "
                 f"{2 ** (T - 1)} of them, and nothing in between",
                 fontsize=10, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return _png(fig)


# ---------------------------------------------------------------------------
# Figure 3 — the simplex: categorical vs segment pooling
# ---------------------------------------------------------------------------
def _bary(w):
    V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
    return np.asarray(w) @ V


def fig_simplex():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 5.0))
    V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])

    for ax, title in ((a1, "Categorical / Gumbel-softmax"), (a2, "Segment pooling (our case)")):
        ax.add_patch(mpatches.Polygon(V, closed=True, fc="#eceae4", ec=MUTED, lw=1.3))
        ax.set_xlim(-0.24, 1.24); ax.set_ylim(-0.30, 1.06)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(title, fontsize=10.5, pad=6)
        for v, lab, off in zip(V, ["frame 1", "frame 2", "frame 3"],
                               [(-0.11, -0.075), (0.11, -0.075), (0.0, 0.055)]):
            ax.text(v[0] + off[0], v[1] + off[1], lab, fontsize=8.2,
                    color=MUTED, ha="center")

    # --- left: the discrete points ARE the vertices; the hull IS the simplex
    for v in V:
        a1.plot(*v, "o", ms=11, color=GREEN, zorder=5)
    a1.add_patch(mpatches.Polygon(V, closed=True, fc=GREEN, ec="none", alpha=0.16))
    mid = _bary([0.45, 0.35, 0.20])
    a1.plot(*mid, "o", ms=7, color=BLUE, zorder=5)
    a1.text(mid[0], mid[1] - 0.10, "any relaxed point\nis a valid input", fontsize=8,
            color=BLUE, ha="center")
    a1.text(0.5, -0.235, "reachable set = the 3 corners\nconvex hull = the whole triangle",
            fontsize=8.8, color=INK, ha="center")

    # --- right: reachable rows are uniform over a CONTIGUOUS frame set
    reach = [("(1,0,0)", [1, 0, 0], (26, 10)), ("(0,1,0)", [0, 1, 0], (-26, 10)),
             ("(0,0,1)", [0, 0, 1], (0, -17)), ("(½,½,0)", [.5, .5, 0], (0, 12)),
             ("(0,½,½)", [0, .5, .5], (26, 4)), ("(⅓,⅓,⅓)", [1 / 3, 1 / 3, 1 / 3], (0, 12))]
    for lab, w, off in reach:
        pt = _bary(w)
        a2.plot(*pt, "o", ms=10, color=PINK, zorder=6)
        a2.annotate(lab, pt, textcoords="offset points", xytext=off,
                    fontsize=7.8, ha="center", color=INK, zorder=7)
    bad = _bary([0.5, 0, 0.5])
    a2.plot(*bad, "x", ms=11, mew=2.4, color=RED, zorder=6)
    a2.annotate("(½,0,½)\nnot contiguous", bad, textcoords="offset points",
                xytext=(-13, 2), fontsize=7.8, color=RED, ha="right")
    p1, p2 = _bary([.5, .5, 0]), _bary([0, .5, .5])
    a2.plot([p1[0], p2[0]], [p1[1], p2[1]], ls="--", lw=1.5, color=MUTED, zorder=4)
    midp = (p1 + p2) / 2
    a2.plot(*midp, "o", ms=6, mfc="white", mec=MUTED, mew=1.6, zorder=6)
    a2.annotate("midpoint (¼,½,¼)\nis not a segmentation", midp, textcoords="offset points",
                xytext=(13, -16), fontsize=7.8, color=MUTED, ha="left")
    a2.text(0.5, -0.235, "reachable set = 6 isolated interior points\n"
                         "convex hull ≠ reachable set",
            fontsize=8.8, color=INK, ha="center")
    fig.tight_layout()
    return _png(fig)


# ---------------------------------------------------------------------------
# Figure 4 — M_hard, M_soft, and the straight-through combination
# ---------------------------------------------------------------------------
def fig_matrices():
    Mh = membership_hard(B5, PAD5)
    Ms = membership_soft_survival(P5, starts=[0, 2, 4], T=5, Tn=3)
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 2.75))
    for ax, M, title in ((axes[0], Mh, "M   (hard, what the model uses)"),
                         (axes[1], Ms, "M̃   (soft, used only for the gradient)"),
                         (axes[2], Ms - Mh, "M̃ − M   (the discrepancy)")):
        vm = float(max(abs(M).max(), 1e-6))
        im = ax.imshow(M.numpy(), cmap="RdBu_r" if "−" in title else "Blues",
                       vmin=-vm if "−" in title else 0, vmax=vm)
        for j in range(M.shape[0]):
            for t in range(M.shape[1]):
                v = float(M[j, t])
                ax.text(t, j, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                        color="white" if abs(v) > 0.6 * vm else INK)
        ax.set_xticks(range(5)); ax.set_xticklabels([f"f{t}" for t in range(5)], fontsize=8)
        ax.set_yticks(range(3)); ax.set_yticklabels([f"seg {j}" for j in range(3)], fontsize=8)
        ax.set_title(title, fontsize=9)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.tight_layout()
    return _png(fig)


# ---------------------------------------------------------------------------
# Figure 5 — what the gradient actually says
# ---------------------------------------------------------------------------
def fig_gradient_field():
    """dL/dM[j,t]: a value for every (segment, frame) pair, never for a partition."""
    # Features with genuine structure, so the gradient pattern is informative rather
    # than an artefact of colinear frames. Current grouping {0,1}{2,3}{4}; the target
    # is the pooled sequence of a DIFFERENT 3-segment grouping, {0}{1,2}{3,4}.
    X = torch.tensor([[2.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 3.0]])
    target, _ = mean_pool_segments(X.unsqueeze(0), PAD5, torch.tensor([[0, 1, 0, 1, 0]]))
    target = target.squeeze(0).detach()
    Mh = membership_hard(B5, PAD5).clone().requires_grad_(True)
    ((Mh @ X - target) ** 2).mean().backward()
    # Each row of M must sum to 1, so only the part of the gradient that redistributes
    # weight within a row is actionable. Centre each row (project onto that constraint).
    G = Mh.grad.numpy()
    G = G - G.mean(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6.4, 2.5))
    vm = float(np.abs(G).max())
    im = ax.imshow(G, cmap="RdBu_r", vmin=-vm, vmax=vm)
    seg_ids, _, _ = compute_segment_ids(B5, PAD5)
    for t in range(5):
        j = int(seg_ids[0, t])
        ax.add_patch(mpatches.Rectangle((t - 0.5, j - 0.5), 1, 1, fill=False,
                                        ec=INK, lw=2.2))
    for j in range(3):
        for t in range(5):
            ax.text(t, j, f"{G[j, t]:+.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if abs(G[j, t]) > 0.62 * vm else INK)
    ax.set_xticks(range(5)); ax.set_xticklabels([f"frame {t}" for t in range(5)], fontsize=8)
    ax.set_yticks(range(3)); ax.set_yticklabels([f"seg {j}" for j in range(3)], fontsize=8)
    ax.set_title("∂L/∂M : “segment j should take more (blue) or less (red) of frame t”\n"
                 "black outline = the frames the hard segmentation actually assigns", fontsize=9)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    return _png(fig)


# ---------------------------------------------------------------------------
# Figure 6 — discrepancy and signal vanish together
# ---------------------------------------------------------------------------
def fig_bias_signal():
    """Gaussian index relaxation, fixed data: ||M̃-M|| and |dL/dlogits| against sigma."""
    torch.manual_seed(0)
    B, T, C = 4, 200, 32
    feats = torch.randn(B, T, C)
    pad = torch.zeros(B, T, dtype=torch.bool)
    base_logits = torch.randn(B, T) - 1.1
    tgt = torch.randn(B, 60, C) * 0.3

    sigmas = np.linspace(0.15, 2.4, 24)
    disc, sig = [], []
    for s in sigmas:
        lg = base_logits.clone().requires_grad_(True)
        b = (lg.detach() > 0).long()
        seg_ids, num_seg, J = compute_segment_ids(b, pad)
        valid = (~pad).float()
        p = torch.sigmoid(lg) * valid
        p = torch.cat([torch.zeros_like(p[:, :1]), p[:, 1:]], 1)
        c_soft = torch.cumsum(p, 1)
        c_hard = seg_ids.clamp(min=0).float()
        c_st = c_hard + (c_soft - c_soft.detach())
        j = torch.arange(J, dtype=torch.float32)
        d = c_st.unsqueeze(1) - j.view(1, J, 1)
        segpad = torch.arange(J).unsqueeze(0) >= num_seg.unsqueeze(1)
        cell = valid.unsqueeze(1) * (~segpad).float().unsqueeze(2)
        A = torch.exp(-0.5 * d ** 2 / s ** 2) * cell
        Ms = A / A.sum(2, keepdim=True).clamp(min=1e-8)
        Ah = (d.detach().abs() < 1e-6).float() * cell
        Mh = Ah / Ah.sum(2, keepdim=True).clamp(min=1.0)
        disc.append(float((Ms.detach() - Mh).pow(2).sum(dim=(1, 2)).sqrt().mean()))
        M = Mh + (Ms - Ms.detach())
        out = torch.bmm(M, feats)
        k = min(out.shape[1], tgt.shape[1])
        ((out[:, :k] - tgt[:, :k]) ** 2).mean().backward()
        sig.append(float(lg.grad.pow(2).mean().sqrt()))

    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    ax.plot(sigmas, np.array(disc) / max(disc), "-o", ms=3.5, color=RED,
            label="discrepancy ‖M̃ − M‖  (bias)")
    ax.plot(sigmas, np.array(sig) / max(sig), "-o", ms=3.5, color=BLUE,
            label="gradient magnitude  (signal)")
    ax.set_xlabel("relaxation width σ")
    ax.set_ylabel("relative to each curve's maximum")
    ax.grid(alpha=0.25, color=GRID)
    ax.legend(frameon=False, fontsize=8.5, loc="center right")
    ax.set_title("Making the relaxation tighter removes the bias and the signal together",
                 fontsize=9.5)
    ax.annotate("σ → 0:  M̃ → M,\nbut no gradient left", xy=(0.19, 0.02), xytext=(0.60, 0.09),
                fontsize=8.2, color=MUTED, ha="center",
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
    fig.tight_layout()
    return _png(fig)


# ---------------------------------------------------------------------------
# Figure 7 — experimental summary
# ---------------------------------------------------------------------------
def fig_results():
    arms = ["cold start", "straight-through", "soft pooling", "GRPO", "supervised"]
    off = [3.34, 3.90, 3.14, 3.25, 0.00]
    cols = [MUTED, RED, BLUE, GREEN, INK]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 3.5))
    bars = a1.bar(range(5), off, color=cols, alpha=0.88, width=0.62)
    a1.axhline(3.34, color=MUTED, ls="--", lw=1.2)
    a1.text(4.45, 3.42, "no better than\nthe starting point", fontsize=7.8, color=MUTED, ha="right")
    for b, v in zip(bars, off):
        a1.text(b.get_x() + b.get_width() / 2, v + 0.08, f"{v:.2f}", ha="center", fontsize=8.5)
    a1.set_xticks(range(5))
    a1.set_xticklabels(arms, fontsize=8, rotation=16, ha="right")
    a1.set_ylabel("mean distance to the correct boundary\n(frames; lower is better)")
    a1.set_ylim(0, 4.4)
    a1.set_title("Boundary placement, realistic cold start", fontsize=9.5)
    a1.grid(axis="y", alpha=0.22, color=GRID)

    arms2 = ["cold start", "straight-through (survival)", "straight-through (index)",
             "soft pooling (survival)", "soft pooling (index)", "supervised"]
    f1 = [0.228, 0.297, 0.343, 0.266, 0.502, 0.987]
    cols2 = [MUTED, "#d08a8a", RED, "#8fb4d8", BLUE, INK]
    bars = a2.bar(range(6), f1, color=cols2, alpha=0.88, width=0.62)
    for b, v in zip(bars, f1):
        a2.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center", fontsize=8.5)
    a2.set_xticks(range(6))
    a2.set_xticklabels(arms2, fontsize=7.4, rotation=20, ha="right")
    a2.set_ylabel("boundary F1 (higher is better)")
    a2.set_ylim(0, 1.1)
    a2.set_title("Boundary recovery, degenerate cold start", fontsize=9.5)
    a2.grid(axis="y", alpha=0.22, color=GRID)
    fig.tight_layout()
    return _png(fig)


# ---------------------------------------------------------------------------
# document
# ---------------------------------------------------------------------------
EXTRA_CSS = """
<style>
.eq{margin:1.05rem auto;padding:.72rem .9rem;background:var(--surface-1);
    border:1px solid var(--rule);border-radius:8px;text-align:center;
    font-family:'Palatino Linotype',Palatino,Georgia,serif;font-size:1.02rem;overflow-x:auto}
.eq .n{float:right;color:var(--muted);font-size:.82rem;font-family:system-ui,sans-serif}
.defs{margin:.8rem 0 .2rem;padding-left:0;list-style:none}
.defs li{margin:.3rem 0;padding-left:1.1rem;text-indent:-1.1rem;font-size:.95rem}
.defs code{font-size:.92em}
.small{font-size:.9rem;color:var(--muted)}
.scroll{overflow-x:auto}
</style>
"""


def build():
    body = EXTRA_CSS
    body += hk.hero(
        "Learned segmenter · gradient estimation",
        "Why straight-through does not train a segmenter",
        "The ASR cross-entropy cannot reach the boundary policy through hard mean-pooling. "
        "This report shows exactly where the gradient is lost, what a straight-through "
        "estimator does and does not recover, and why the obstruction is structural.",
    )

    # ---------------------------------------------------------------- 0
    body += hk.section(
        "0 · Context",
        lead="What the segmenter is, and what question this report answers.",
        body=(
            "<p>The speech-LLM pipeline compresses audio before the language model sees it. "
            "A frozen wav2vec2 encoder produces one feature vector per 20&nbsp;ms frame. A small "
            "<b>segmenter</b> reads those features and emits one number per frame — a "
            "<b>boundary logit</b>, high where it thinks a new segment should start. Frames "
            "between boundaries are averaged into a single vector, and that shorter sequence is "
            "projected and fed to the LLM, which is trained with cross-entropy on the "
            "transcript.</p>"
            "<p>Averaging frames between boundaries is the step that destroys the gradient. The "
            "segmenter's output is a set of yes/no decisions, and those decisions are used to "
            "<i>look up</i> which frames belong to which group. Because of that, the "
            "cross-entropy loss cannot be differentiated with respect to the segmenter, and the "
            "policy is currently trained with reinforcement learning (GRPO) instead.</p>"
            "<p>A <b>straight-through estimator</b> is the standard trick for pushing gradients "
            "through discrete decisions, and it is what makes VQ-VAE and quantised networks "
            "trainable. The obvious question is whether it would work here — replacing an "
            "RL objective with a cheaper, lower-variance gradient. This report answers that "
            "question: <b>it can be implemented, and it does not do what we need</b>, for a "
            "reason that is geometric rather than a matter of tuning.</p>"
            + hk.finding(
                "<b>Summary.</b> A straight-through estimator can be made to produce a nonzero "
                "gradient here, and the implementation is small. But that gradient only ever "
                "expresses <i>how much of each frame to mix into a segment</i>, never "
                "<i>which segmentation to use</i>. Against a realistic starting point it does "
                "not improve boundary placement at all.")
        ),
    )

    # ---------------------------------------------------------------- 1
    body += hk.section(
        "1 · Where the gradient is lost",
        lead="Mean-pooling is differentiable — but not with respect to the thing we need.",
        body=(
            "<p>Pooling takes two inputs:</p>"
            '<div class="eq">S = pool(<b>X</b>, <b>b</b>)</div>'
            '<ul class="defs">'
            "<li><code>X</code> — the frame features from wav2vec2, a <code>T×C</code> matrix "
            "(<code>T</code> frames, <code>C</code> channels).</li>"
            "<li><code>b</code> — the segmentation, a vector of <code>T</code> zeros and ones. "
            "<code>b<sub>t</sub>=1</code> means frame <code>t</code> starts a new segment.</li>"
            "<li><code>S</code> — the pooled sequence, one row per segment.</li>"
            "</ul>"
            "<p>The derivative <b>∂S/∂X</b> exists and behaves normally; that is why "
            "cross-entropy already trains the projection layer. The derivative we actually need "
            "is <b>∂S/∂b</b>, because <code>b</code> is what the segmenter produces — and that "
            "one does not exist. <code>b</code> is binary, so the number of segments is an "
            "integer. There is no state of the model in which there are 2.3&nbsp;segments.</p>"
            "<p>Concretely, with five frames and <code>b = [0,0,1,0,1]</code>:</p>"
            '<div class="scroll"><table><thead><tr><th>change to b</th><th>resulting S</th>'
            "<th>effect</th></tr></thead><tbody>"
            "<tr><td>none</td><td><code>[[2,20], [6,60], [9,90]]</code></td><td>—</td></tr>"
            "<tr><td>b<sub>2</sub>: 1.00 → 1.30</td><td><code>[[2,20], [6,60], [9,90]]</code></td>"
            "<td>nothing</td></tr>"
            "<tr><td>b<sub>2</sub>: 1.00 → 1.49</td><td><code>[[2,20], [6,60], [9,90]]</code></td>"
            "<td>nothing</td></tr>"
            "<tr><td>b<sub>3</sub>: 0 → 1</td><td><code>[[2,20], [5,50], [7,70], [9,90]]</code></td>"
            "<td>jumps, and gets <b>longer</b></td></tr>"
            "</tbody></table></div>"
            "<p>Flat, flat, flat, then a discontinuous jump that also changes the length of the "
            "output. Figure&nbsp;1 shows the same thing as a loss surface over two boundary "
            "logits: four flat plateaus separated by cliffs. The gradient is exactly zero "
            "wherever it is defined.</p>"
            + hk.card(fig_piecewise(),
                      title="The loss as a function of the boundary logits")
            + '<p class="cap">Left: content loss over two boundary logits of a three-frame '
              "utterance; the segmentation only changes when a logit crosses zero. Right: a "
              "horizontal slice. Every point in the interior has zero gradient, and the only "
              "informative locations are the cliffs, where the derivative does not exist.</p>"
            "<p>In code this appears as the boundary being consumed only as an integer array "
            "index: <code>(boundary == 1)</code> gives a boolean, <code>cumsum(...).long()</code> "
            "gives an integer, and that integer is used to index into an accumulation buffer. "
            "Passing a relaxed boundary in produces no error and no gradient — measured, "
            "<code>logits.grad</code> is <code>None</code> while the gradient to <code>X</code> "
            "flows normally.</p>"
        ),
    )

    # ---------------------------------------------------------------- 2
    body += hk.section(
        "2 · Pooling written as a matrix product",
        lead="This is what makes a straight-through estimator possible at all.",
        body=(
            "<p>Averaging frames within segments can be written as a single matrix product:</p>"
            '<div class="eq">S = M X <span class="n">(1)</span></div>'
            '<ul class="defs">'
            "<li><code>M</code> — the <b>membership matrix</b>, with one row per segment and one "
            "column per frame. Entry <code>M<sub>jt</sub></code> is the weight frame "
            "<code>t</code> carries in segment <code>j</code>. For plain averaging it is "
            "<code>1/(segment length)</code> for the frames inside segment <code>j</code> and "
            "zero elsewhere, so every row sums to one.</li></ul>"
            "<p>For <code>b = [0,0,1,0,1]</code> this is exactly the left panel of Figure&nbsp;2, "
            "and multiplying by <code>X</code> reproduces the pooled output to floating-point "
            "precision. Nothing has changed except <i>where the information sits</i>: the "
            "grouping is now a set of <b>numbers being multiplied</b> rather than <b>positions "
            "being looked up</b>. Matrix products are differentiable in both arguments.</p>"
            "<p>Now define a second, smooth matrix. Write "
            "<code>p<sub>t</sub> = sigmoid(logit<sub>t</sub>)</code> for the probability that "
            "frame <code>t</code> starts a segment, and let</p>"
            '<div class="eq">M̃<sub>jt</sub> ∝ P(no boundary fired between segment j\'s start '
            'and frame t) <span class="n">(2)</span></div>'
            "<p>which is a product of <code>(1 − p)</code> terms, then normalised so each row "
            "sums to one. This is a genuine soft segmentation: in Figure&nbsp;2 frame&nbsp;2 "
            "leaks 4% into segment&nbsp;0 because the boundary there is only 90% certain. Unlike "
            "<code>M</code>, it is smooth in <code>p</code>.</p>"
            + hk.card(fig_matrices(), title="Hard, soft, and the difference")
            + '<p class="cap">The hard membership matrix M reproduces mean-pooling exactly. '
              "The soft matrix M̃ is a blurred version of it that is differentiable in the "
              "boundary probabilities. The right panel is what the straight-through estimator "
              "silently substitutes.</p>"
            "<p>The straight-through estimator combines them:</p>"
            '<div class="eq">M<sub>ST</sub> = M + (M̃ − stopgrad(M̃)) <span class="n">(3)</span></div>'
            "<p>The bracket is numerically zero, so the forward pass is bit-identical to hard "
            "pooling — the LLM sees exactly the sequence it would have seen anyway. In the "
            "backward pass the constants drop out and the gradient is taken as if <code>M̃</code> "
            "had been used. This works: measured end-to-end, the forward output matches "
            "<code>mean_pool_segments</code> to <code>1.2e-7</code> (floating-point rounding) and "
            "the boundary logits receive a nonzero gradient.</p>"
            + hk.finding(
                "<b>So the plumbing is not the obstacle.</b> Equations (1)–(3) give a working "
                "estimator in roughly sixty lines, costing one forward and one backward pass — "
                "about half the cost of the current GRPO arm, which needs four extra rollouts "
                "per step. Everything that follows is about whether the resulting gradient is "
                "worth having.")
        ),
    )

    # ---------------------------------------------------------------- 3
    body += hk.section(
        "3 · When straight-through is justified",
        lead="It is an approximation with a precise validity condition. VQ-VAE meets it; this does not.",
        body=(
            "<p>Straight-through is usually presented as a trick, but it is a first-order "
            "approximation with a stateable condition. Let <code>a</code> be the continuous "
            "quantity being optimised, <code>Q</code> the hard operation, and <code>L</code> the "
            "downstream loss. The estimator replaces the true derivative of <code>Q</code> with "
            "the identity. That substitution is justified when:</p>"
            '<ul class="defs">'
            "<li><b>(i) Common space.</b> The discrete outputs and <code>a</code> live in the "
            "<i>same</i> vector space, so <code>Q(a) − a</code> is a meaningful difference.</li>"
            "<li><b>(ii) Proximity.</b> <code>‖Q(a) − a‖ ≤ ε</code> — the hard operation barely "
            "moves its input.</li>"
            "<li><b>(iii) Smoothness.</b> <code>L</code> has a Lipschitz gradient, with "
            "constant <code>Λ</code>.</li>"
            "</ul>"
            "<p>Then the error in the estimated gradient is bounded:</p>"
            '<div class="eq">‖∇L(Q(a)) − ∇L(a)‖ ≤ Λ·ε <span class="n">(4)</span></div>'
            "<p><b>VQ-VAE satisfies all three.</b> The encoder output is a vector in "
            "<code>R<sup>D</sup></code>; the codebook is a finite set of points in the same "
            "<code>R<sup>D</sup></code>; quantisation replaces a vector with its nearest "
            "neighbour, so <code>ε</code> is the quantisation error and is small when the "
            "codebook covers the data. Bound (4) then genuinely applies, and the same argument "
            "covers quantised networks and neural audio codecs.</p>"
            "<p><b>Our case fails (i) outright.</b> The quantity being optimised is a vector of "
            "probabilities; the output of the hard operation is a <i>segmentation</i>. These are "
            "different kinds of object, and <code>Q(a) − a</code> is not an expression one can "
            "write. There is no <code>ε</code>, so bound (4) says nothing.</p>"
            "<p>Rewriting pooling as <code>S = M X</code> repairs (i) — <code>M</code> and "
            "<code>M̃</code> are both matrices of the same shape. Condition (ii) still fails, and "
            "Section&nbsp;4 explains why it cannot be repaired.</p>"
            + hk.card(
                '<div class="scroll"><table><thead><tr><th>setting</th>'
                "<th>what is discretised</th><th>output shape</th><th>(i)</th><th>(ii)</th>"
                "</tr></thead><tbody>"
                "<tr><td>Quantised / binarised networks</td><td>weights → few bits</td>"
                "<td>fixed</td><td>yes</td><td>yes</td></tr>"
                "<tr><td>VQ-VAE, neural audio codecs</td><td>latent vector → codebook entry</td>"
                "<td>fixed</td><td>yes</td><td>yes</td></tr>"
                "<tr><td>Categorical / Gumbel-softmax</td><td>choice among K options</td>"
                "<td>fixed</td><td>yes</td><td>yes</td></tr>"
                "<tr><td>Mixture-of-experts routing</td><td>which experts to use</td>"
                "<td>fixed</td><td>partly</td><td>needs a balancing loss</td></tr>"
                "<tr><td><b>Segment pooling (this work)</b></td><td>how frames are grouped</td>"
                "<td><b>varies</b></td><td><b>no</b></td><td><b>no</b></td></tr>"
                "</tbody></table></div>"
                '<p class="cap">The pattern: straight-through is a tool for quantising a '
                "continuous quantity. Every setting where it works leaves the shape of the "
                "computation untouched and only changes a value.</p>",
                title="Where straight-through is and is not justified")
        ),
    )

    # ---------------------------------------------------------------- 4
    body += hk.section(
        "4 · The geometry: why proximity cannot be repaired",
        lead="The set of segmentations is finite and scattered. There is no direction to move in.",
        body=(
            "<p>An optimiser needs a direction. A gradient is a direction in the space of "
            "things you are allowed to choose. So the question is what that space looks like "
            "here.</p>"
            "<p>For a <code>T</code>-frame utterance the segmenter is choosing among "
            "<code>2<sup>T−1</sup></code> segmentations — one for each way of deciding, at each "
            "of the <code>T−1</code> interior frames, whether a new segment starts. That set is "
            "<b>finite</b>. Figure&nbsp;3 draws all sixteen of them for five frames.</p>"
            + hk.card(fig_partitions(), title="The entire space of choices, T = 5")
            + '<p class="cap">Colour indicates which segment each frame belongs to. These '
              "sixteen are the complete set of options; there is nothing between any two of "
              "them. For a ten-second utterance at 50&nbsp;Hz the count is 2<sup>499</sup>, but "
              "it is still a finite scatter of isolated points.</p>"
            "<p>This is the crux. A finite set of isolated points has <b>no tangent "
            "directions</b> — you cannot take a small step from one segmentation and arrive at "
            "another. A gradient is precisely an element of a tangent space, so on this set "
            "there is nothing for a gradient to be.</p>"
            "<p>It is worth being careful about what does and does not go wrong, because the "
            "logit space itself is perfectly ordinary. The logits live in "
            "<code>R<sup>T</sup></code>, which is as well-behaved as any space. The problem is "
            "that the map from logits to segmentations is <b>piecewise constant</b>: the "
            "hyperplanes <code>logit<sub>t</sub> = 0</code> cut <code>R<sup>T</sup></code> into "
            "<code>2<sup>T</sup></code> cells, and the loss is constant on each. That is exactly "
            "Figure&nbsp;1.</p>"
            "<p>Figure&nbsp;4 shows why this is different from the categorical case, where "
            "straight-through and Gumbel-softmax do work. Consider one row of the membership "
            "matrix over three frames; because it is non-negative and sums to one, it lives in a "
            "triangle.</p>"
            + hk.card(fig_simplex(), title="The reachable set inside the triangle")
            + '<p class="cap">Left: for a categorical choice the reachable points are the three '
              "corners, and the triangle is exactly their convex hull — the relaxation fills in "
              "the space between the discrete options, and every intermediate point is a valid "
              "input. Right: for segment pooling the reachable rows are uniform over a "
              "<i>contiguous</i> run of frames. There are six of them, they sit in the "
              "<i>interior</i> rather than at the corners, and their convex hull is not the "
              "reachable set: the midpoint of two reachable points is not a segmentation, and "
              "(½,0,½) is excluded because frames 1 and 3 are not adjacent.</p>"
            "<p>This is the precise sense in which the two cases differ. For a categorical "
            "variable the relaxation is a genuine <b>convexification</b>: the simplex <i>is</i> "
            "the convex hull of the one-hot vectors, so moving inside it interpolates between "
            "real options. For segment pooling the reachable set is a scatter of interior "
            "points whose convex hull contains matrices that correspond to no segmentation at "
            "all. Moving continuously between two valid segmentations leaves the set of "
            "segmentations immediately.</p>"
            + hk.finding(
                "<b>Consequence.</b> Condition (ii) cannot be repaired by choosing a better "
                "soft matrix. The reachable set stays zero-dimensional whatever <code>M̃</code> "
                "is, so there is never an admissible direction for a first-order method to "
                "return. This is not an argument that the estimator is badly tuned; it is an "
                "argument that the object it computes is not the object we want.")
        ),
    )

    # ---------------------------------------------------------------- 5
    body += hk.section(
        "5 · What the estimator does compute",
        lead="A mixing weight for every frame, not a choice of segmentation.",
        body=(
            "<p>The estimator does return a nonzero number for each logit, so it is worth being "
            "exact about what that number means. Differentiating through equation (1):</p>"
            '<div class="eq">∂L/∂p<sub>u</sub> = Σ<sub>j,t</sub> '
            "⟨ ∂L/∂S<sub>j</sub> , X<sub>t</sub> ⟩ · ∂M̃<sub>jt</sub>/∂p<sub>u</sub>"
            '<span class="n">(5)</span></div>'
            "<p>The inner product is the interesting factor. It measures how much segment "
            "<code>j</code>'s contribution to the loss would improve if it absorbed more of "
            "frame <code>t</code>. So the gradient is a table of values over "
            "<b>(segment, frame)</b> pairs — Figure&nbsp;5.</p>"
            + hk.card(fig_gradient_field(), title="The gradient, drawn in full")
            + '<p class="cap">Each cell is one term of equation (5): blue means "this segment '
              "should take more of this frame\", red means less. The black outlines mark the "
              "assignment the hard segmentation actually made. Because every row of M has to "
              "sum to one, only the part of the gradient that redistributes weight <i>within</i> "
              "a row is actionable, so each row is shown centred on its own mean. Note that the "
              "gradient still assigns a value to every pair, including frames far outside a "
              "segment — it is a reweighting field, and it has no way to express \"use a "
              "different partition\".</p>"
            "<p>Put plainly: the estimator answers <i>how much of each frame should be mixed "
            "into each segment</i>. It never answers <i>where the boundaries should be</i>. "
            "Those are different questions, and only the second one is what the segmenter is "
            "for.</p>"
            "<p>This also explains a pattern in the measurements. Summing a reweighting field "
            "over frames produces something that behaves like a change in the effective number "
            "of segments, so the estimator does move the <b>audio-token frequency</b> "
            "(50 times the number of segments divided by the number of frames). It moved from "
            "3.15 to 3.75 Hz in the "
            "experiments below. But that is the one quantity we can already differentiate "
            "exactly: <code>Σ sigmoid(logit)</code> is smooth in the logits with no "
            "approximation at all, which is what the existing audio-frequency penalty uses. And the "
            "direction is unhelpful — cross-entropy always prefers more tokens, since keeping "
            "every frame loses no information, so this component pushes toward no compression "
            "and has to be actively suppressed.</p>"
            + hk.finding(
                "<b>The two halves of the signal.</b> The strong, reliable component of the "
                "estimated gradient controls <i>how many</i> segments — redundant with an exact "
                "penalty we already have, and pointing the wrong way. The component that "
                "controls <i>where</i> boundaries go is the weak one, and it is the only reason "
                "we wanted the estimator.")
        ),
    )

    # ---------------------------------------------------------------- 6
    body += hk.section(
        "6 · Bias and signal vanish together",
        lead="The obvious fix — tighten the relaxation — removes the gradient along with the error.",
        body=(
            "<p>A natural response to the discrepancy in Figure&nbsp;2 is to make <code>M̃</code> "
            "closer to <code>M</code>. The relaxation has a width parameter <code>σ</code> that "
            "does exactly that. It does not help, because both quantities are controlled by the "
            "same parameter.</p>"
            + hk.card(fig_bias_signal(), title="The trade-off has no good end")
            + '<p class="cap">Measured on fixed data, both curves normalised to their own '
              "maximum. As σ shrinks the soft matrix converges to the hard one, so the "
              "approximation error goes to zero — and so does the gradient. As σ grows the "
              "gradient decays too, while the error keeps rising. There is a middle setting "
              "that maximises signal, and it is also where the approximation is worst.</p>"
            "<p>The degenerate case makes the mechanism obvious. In the relaxation used here, "
            "the gradient reaching a frame comes entirely from <i>neighbouring</i> segments — "
            "the term for a frame's own segment is exactly zero. When a whole utterance collapses "
            "to a single segment there are no neighbours, and the gradient is exactly zero. That "
            "is an absorbing state: once collapsed, only the rate penalty can escape it, and in "
            "practice it did not. Two of the six configurations tested collapsed to an "
            "audio-token frequency of 0.40 Hz and stayed there, even after the frequency penalty was strengthened "
            "twenty-fold.</p>"
        ),
    )

    # ---------------------------------------------------------------- 7
    body += hk.section(
        "7 · Experiments",
        lead="Small controlled problems where the correct segmentation is known.",
        body=(
            "<p>These are deliberately small: synthetic signals that are constant within a "
            "segment and jump between segments, so the correct boundaries are known exactly and "
            "boundary placement can be measured directly, which WER cannot do. The segmenter is "
            "the real convolutional boundary predictor from the codebase, trained across 768 "
            "utterances with the real rate penalty. Every arm gets identical data, initialisation, "
            "learning rate and step budget, and every arm is <i>evaluated</i> with hard "
            "pooling.</p>"
            + hk.card(fig_results(), title="Both experiments")
            + '<p class="cap">Left: the decisive test. Right: the earlier, easier test.</p>'
            "<p><b>The easier test</b> starts the segmenter from a deliberately bad "
            "initialisation and asks whether each method can recover. Every method improves, and "
            "the fully-soft variant improves most — removing the hard forward pass roughly "
            "doubles what is recovered. That is consistent with the analysis: making the forward "
            "pass match the backward pass removes the approximation error of equation (3).</p>"
            "<p><b>The decisive test</b> is more honest. The starting point is a competent "
            "segmenter whose boundaries are systematically three frames early — the analogue of "
            "a cold start that is sensible but not aligned to what the decoder wants. The "
            "question is whether any method can correct a known, uniform, small error.</p>"
            + hk.card(
                '<div class="scroll"><table><thead><tr><th>method</th>'
                "<th>mean distance to correct boundary</th><th>content loss</th>"
                "<th>audio-token frequency</th></tr></thead><tbody>"
                "<tr><td>cold start (no further training)</td><td>3.34</td><td>1.948</td>"
                "<td>3.15 Hz</td></tr>"
                "<tr><td>straight-through (index relaxation)</td><td><b>3.90</b> — worse</td>"
                "<td>2.257</td><td>3.75 Hz</td></tr>"
                "<tr><td>straight-through (survival relaxation)</td><td>collapsed</td>"
                "<td>3.668</td><td>0.40 Hz</td></tr>"
                "<tr><td>soft pooling (index relaxation)</td><td>3.14</td><td><b>1.872</b></td>"
                "<td>3.35 Hz</td></tr>"
                "<tr><td>soft pooling (survival relaxation)</td><td>collapsed</td>"
                "<td>3.668</td><td>0.40 Hz</td></tr>"
                "<tr><td>GRPO (reinforcement learning)</td><td>3.25</td><td>1.821</td>"
                "<td>3.50 Hz</td></tr>"
                "<tr><td>direct supervision on the true boundaries</td><td><b>0.00</b></td>"
                "<td>0.598</td><td>3.25 Hz</td></tr>"
                "</tbody></table></div>"
                '<p class="cap">The cold start measuring 3.34 confirms it is three frames off '
                "as designed, and supervision reaching 0.00 confirms the task is solvable and "
                "the metric is sensitive. Between those two anchors, nothing works: the best "
                "gradient-based method recovers 0.20 of the 3.34 frames.</p>",
                title="Correcting a known three-frame misalignment")
            + hk.finding(
                "<b>Result.</b> No gradient-based method corrects the misalignment. "
                "Straight-through makes it worse; soft pooling and GRPO barely move it. Only "
                "direct supervision on boundaries recovers it. Since the same failure appears "
                "across three different estimators, the limitation is not in the estimator — "
                "the cross-entropy loss simply carries very little information about where "
                "boundaries belong.")
            + '<p class="small">Caveats. These are synthetic signals with an objective that '
              "demands exact recovery of the target pooled sequence, which is harsher than the "
              "LLM's cross-entropy; one seed; 1500 steps; a small convolutional segmenter. They "
              "are a screening test to decide whether to spend GPU-days, not a substitute for a "
              "full run. A real straight-through arm could behave better than this suggests — "
              "but nothing here gives a reason to expect it to.</p>"
        ),
    )

    return hk.document(
        "Why straight-through does not train a segmenter",
        label_report_assets(body),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/segmenter/ste_analysis/report.html")
    args = ap.parse_args()
    full, _ = build()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(full)
    print(f"wrote {args.out}  ({len(full)/1e6:.2f} MB)")
