#!/usr/bin/env python3
"""Pipeline figure for the learned-segmenter SpeechLLM — emits .drawio AND .png.

ONE layout spec (`build()` appends primitive dicts to `P`) feeds two emitters, so the
editable draw.io file and the rendered image can never drift apart. Primitives are kept
to what both back-ends render identically: rect, ellipse, straight line/arrow, text.

    python make_pipeline_figure.py --outdir ../../../../artifacts/segmenter/figures

Design conventions (from method figures at recent NeurIPS/audio-LLM venues):
  * left-to-right dataflow spine, tensor shape / frame-rate annotated under each block;
  * the physical quantities that matter in audio-token compression are called out
    explicitly (Hz, dims, token count) — that's what makes the figure
    illustrative rather than decorative;
  * frozen vs trainable is written explicitly, never communicated by colour alone;
  * three restrained accents have one role each: green=segmentation, blue=decoder
    adaptation, coral=RL objective; the frozen and standard path stays neutral;
  * a zoom-in panel carries the actual contribution (learned vs fixed-rate boundaries).

Coordinates are authored in draw.io space (y grows DOWN, origin top-left); the PNG
emitter flips y. Palette is the report's own (htmlkit THEME_CSS) so the figure sits
naturally next to the charts in report.html.
"""
import argparse
import os
import xml.etree.ElementTree as ET

# --------------------------------------------------------------------------
# Palette — report THEME_CSS (light surfaces, warm paper)
# --------------------------------------------------------------------------
INK, INK2, MUTED = "#17212b", "#3f4954", "#626a73"
BORDER, BORDER_S, BAND, CARD = "#d9ddd9", "#bfc5c1", "#f1f2ee", "#fdfdfc"

# Semantic palette. Segmentation uses draw.io's default green family so the
# exported PNGs and editable diagram share the familiar built-in styling.
SEG, SEG_D, SEG_F = "#82b366", "#2d7600", "#d5e8d4"
ADAPT, ADAPT_D, ADAPT_F = "#2f6fb6", "#24588f", "#e6eef8"
RL, RL_D, RL_F = "#c64b57", "#a63843", "#fae9eb"
GREY_F = "#efefe9"

W, H = 1640, 812

# Sizes are authored as draw.io px. Matplotlib wants points, and the PNG axes are set
# to 100 data-units per inch, so 1 pt = 100/72 units -> scale by 72/100 to match.
PT = 0.72

P = []  # primitive list, appended back-to-front (z-order)


# --------------------------------------------------------------------------
# Primitive constructors
# --------------------------------------------------------------------------
def box(x, y, w, h, label="", fill=CARD, stroke=BORDER_S, dashed=False, tw=2,
        fs=14, bold=True, fc=INK, r=10):
    P.append(dict(k="box", x=x, y=y, w=w, h=h, label=label, fill=fill, stroke=stroke,
                  dashed=dashed, tw=tw, fs=fs, bold=bold, fc=fc, r=r))
    return P[-1]


def ell(x, y, w, h, label="", fill=CARD, stroke=BORDER_S, tw=2, fs=16, fc=INK):
    P.append(dict(k="ell", x=x, y=y, w=w, h=h, label=label, fill=fill, stroke=stroke,
                  tw=tw, fs=fs, fc=fc))
    return P[-1]


def or_gate(x, y, w, h, fill=CARD, stroke=INK2, tw=2):
    """Native draw.io OR junction; PNG uses the same integrated circle-plus geometry."""
    P.append(dict(k="gate", x=x, y=y, w=w, h=h, fill=fill, stroke=stroke, tw=tw))
    return P[-1]


def txt(x, y, s, fs=11, fc=MUTED, anchor="center", bold=False, italic=False, w=260):
    """Anchor point (x, y) is the text's centre-line; `anchor` sets horizontal align."""
    P.append(dict(k="txt", x=x, y=y, s=s, fs=fs, fc=fc, anchor=anchor, bold=bold,
                  italic=italic, w=w))
    return P[-1]


def line(x1, y1, x2, y2, color=INK2, tw=2, dashed=False, arrow=False):
    P.append(dict(k="line", x1=x1, y1=y1, x2=x2, y2=y2, color=color, tw=tw,
                  dashed=dashed, arrow=arrow))
    return P[-1]


def arrow(x1, y1, x2, y2, color=INK2, tw=2, dashed=False):
    return line(x1, y1, x2, y2, color, tw, dashed, arrow=True)


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------
# learned segmentation used throughout the zoom panel: irregular, 7 segs / 28 frames
NF = 28
LEARNED = [0, 3, 8, 12, 17, 19, 24]        # segment start indices
FIXED = [0, 4, 8, 12, 16, 20, 24]          # 7 uniform k=4 segments
LEARNED_FEWER = [0, 8, 12, 19, 24]          # 5 irregular segments on learned peaks


def _prob(i):
    """p_t = sigma(logit_t). Derived from LEARNED so peaks can't drift off the ticks."""
    if i in LEARNED:
        return 0.90                            # equally confident boundaries
    return 0.08                                # confidently not a boundary


PROB = [_prob(i) for i in range(NF)]


def spine():
    """Band 1 — the left-to-right forward pass."""
    y, h = 96, 78
    cy = y + h / 2

    txt(57, 30, "Learned acoustic segmentation for a speech LLM", fs=21, fc=INK,
        anchor="left", bold=True, w=900)
    txt(57, 58, "A boundary policy turns 50 Hz speech frames into variable-length audio "
        "tokens before the LLM.", fs=12.5, fc=MUTED,
        anchor="left", w=900)

    # --- waveform card (vertical bars read as "audio" in both renderers) ---
    b = box(57, y, 110, h, "", fill=CARD, stroke=BORDER_S)
    amp = [.30, .55, .80, .45, .95, .62, .35, .70, .50, .25]
    for i, a in enumerate(amp):
        bh = a * 34
        box(69 + i * 9.6, cy - 8 - bh / 2, 5, bh, "", fill=ADAPT, stroke=ADAPT,
            tw=0, r=2)
    txt(112, cy + 22, "waveform", fs=11, fc=INK2)
    txt(112, y + h + 18, "16 kHz", fs=10.5, fc=MUTED)

    # --- frozen SSL encoder ---
    box(207, y, 185, h, "", fill=BAND, stroke=BORDER_S, dashed=True, fs=13.5)
    txt(299, cy - 8, "wav2vec2", fs=14, fc=INK, bold=True, w=150)
    txt(299, cy + 13, "frozen", fs=10.5, fc=MUTED, w=100)
    txt(299, y + h + 18, "T × 768   @ 50 Hz", fs=10.5, fc=MUTED)

    # --- the segmenter: the only genuinely new module ---
    box(432, y, 185, h, "", fill=SEG_F, stroke=SEG, fs=14)
    txt(524, cy - 9, "Segmenter", fs=14.2, fc=SEG_D, bold=True, w=145)
    txt(524, cy + 13, "boundary policy · trainable", fs=10.2, fc=SEG_D, w=155)
    txt(524, y + h + 18, "1 boundary logit / frame", fs=10.5, fc=MUTED)

    # --- pooling ---
    box(657, y, 185, h, "Mean pool\nwithin segments", fill=CARD, stroke=BORDER_S,
        fs=12.5, fc=INK)
    txt(749, y + h + 18, "T → N · 768   (f_audio = 50·N/T Hz)", fs=10.5, fc=MUTED)

    # --- projector ---
    box(882, y, 140, h, "", fill=ADAPT_F, stroke=ADAPT, fs=14)
    txt(952, cy - 8, "Projection", fs=13.5, fc=ADAPT_D, bold=True, w=120)
    txt(952, cy + 13, "trainable", fs=10.5, fc=ADAPT_D, w=90)
    txt(952, y + h + 18, "768 → 2048", fs=10.5, fc=MUTED)

    # --- concatenate audio tokens with the text prompt ---
    gate_x, gate_y, gate_d, prompt_w = 1080, cy, 48, 230
    box(gate_x - prompt_w / 2, 16, prompt_w, 44, "“Transcribe speech to text.”", fill=CARD,
        stroke=BORDER, fs=11.5, bold=False, fc=INK2, r=8)
    txt(gate_x, 74, "prompt embeddings", fs=10, fc=MUTED)
    or_gate(gate_x - gate_d / 2, gate_y - gate_d / 2, gate_d, gate_d,
            fill=CARD, stroke=INK2, tw=2.1)
    arrow(gate_x, 86, gate_x, gate_y - gate_d / 2, color=MUTED, tw=1.6)

    # --- decoder: neutral frozen block with a nested blue adapter tag ---
    box(1142, y, 215, h, "", fill=BAND, stroke=BORDER_S, dashed=True, fs=11.8)
    txt(1249, cy - 13, "Llama 3.2 (1B)", fs=13.2, fc=INK, bold=True, w=185)
    txt(1205, cy + 13, "frozen", fs=10.2, fc=MUTED, w=70)
    box(1232, cy + 2, 108, 24, "LoRA · trainable", fill=ADAPT_F, stroke=ADAPT,
        fs=9.2, bold=True, fc=ADAPT_D, r=6)
    txt(1249, y + h + 18, "LoRA rank 16 · ~1.5% trainable", fs=10.5, fc=MUTED,
        w=260)

    # --- output ---
    box(1397, y, 185, h, "", fill=CARD, stroke=BORDER, r=8)
    txt(1489, cy - 10, "“the walls of the", fs=11.5, fc=INK, italic=True)
    txt(1489, cy + 8, "rooms were…”", fs=11.5, fc=INK, italic=True)
    txt(1489, y + h + 18, "transcript", fs=10.5, fc=MUTED)

    for x1, x2 in ((167, 205), (392, 430), (617, 655), (842, 880),
                   (1022, gate_x - gate_d / 2),
                   (gate_x + gate_d / 2, 1140), (1357, 1395)):
        arrow(x1, cy, x2, cy, color=INK2, tw=2)
    return cy


def zoom_panel():
    """Band 2 — the contribution: learned, variable-length segments vs a fixed clock."""
    py, ph = 250, 300
    box(57, py, 1526, ph, "", fill=CARD, stroke=BORDER, r=12)
    line(838, py + 46, 838, py + ph - 22, color=BORDER, tw=1.5)

    txt(77, py + 26, "How the boundary policy compresses speech",
        fs=13.5, fc=INK, anchor="left", bold=True, w=560)

    # ---------------- left: probabilities -> boundaries -> fewer tokens -------
    x0, pitch, cw = 88, 22, 19
    base, bmax = 396, 46                      # bar baseline / max height
    fy, fh = 402, 26                          # frame-cell row
    txt(x0, 338, "per-frame boundary probability", fs=10, fc=INK2, anchor="left",
        italic=True, w=300)

    thr = base - 0.5 * bmax
    line(x0, thr, x0 + NF * pitch - 3, thr, color=MUTED, tw=1.2, dashed=True)
    txt(x0 + NF * pitch + 6, thr, "p = 0.5", fs=9.5, fc=MUTED, anchor="left", w=60)

    for i, p in enumerate(PROB):
        bh = max(p * bmax, 1.5)
        on = i in LEARNED
        box(x0 + i * pitch + 4, base - bh, cw - 8, bh, "",
            fill=SEG if on else BORDER_S, stroke=SEG if on else BORDER_S, tw=0, r=2)

    for i in range(NF):
        box(x0 + i * pitch, fy, cw, fh, "", fill=SEG_F, stroke=BORDER, tw=1, r=3)
    for i in LEARNED:                          # boundary ticks sit on the frame's
        line(x0 + i * pitch - 1.5, fy - 7, x0 + i * pitch - 1.5, fy + fh + 3,
             color=SEG, tw=2.2)                # leading edge: "a segment starts here"

    xr = x0 + NF * pitch + 8
    txt(xr, fy + fh / 2, "T = 28 frames", fs=10.5, fc=INK2, anchor="left", w=120)
    txt(xr, fy + fh / 2 + 15, "@ 50 Hz", fs=10.5, fc=MUTED, anchor="left", w=120)

    # pooled tokens: visibly fewer boxes == the compression
    ns = len(LEARNED)
    tw_, tg = 48, 10
    tx0 = x0 + (NF * pitch - (ns * tw_ + (ns - 1) * tg)) / 2
    ty = 472
    for j in range(ns):
        tx = tx0 + j * (tw_ + tg)
        box(tx, ty, tw_, 30, "", fill=SEG_F, stroke=SEG, tw=1.5, r=4)
        a = LEARNED[j]
        b = LEARNED[j + 1] if j + 1 < ns else NF
        src = x0 + (a + b) / 2 * pitch - pitch / 2 + cw / 2
        line(src, fy + fh + 10, tx + tw_ / 2, ty - 2, color=SEG, tw=1.1)
    txt(x0 + NF * pitch / 2, ty + 50, "N = 7 audio tokens  →  "
        "12.5 Hz   (4× fewer)", fs=11.5, fc=INK2, bold=False, w=460)

    # ---------------- right: fewer learned tokens, placed adaptively ----------
    txt(868, py + 26, "Fewer tokens, smarter placement", fs=13.5, fc=INK,
        anchor="left", bold=True, w=420)

    rx0, rp, rw = 876, 23, 19
    for row, (starts, name, col, sub) in enumerate((
            (FIXED, "Fixed-rate: 7 tokens (k=4)", MUTED,
             "seven uniformly spaced segments"),
            (LEARNED_FEWER, "Learned: 5 tokens", SEG_D,
             "five segments placed on acoustic events"))):
        ry = 368 + row * 84
        txt(rx0, ry - 16, name, fs=11.5, fc=col, anchor="left", bold=True, w=220)
        for i in range(NF):
            light = col == MUTED
            f = GREY_F if light else SEG_F
            box(rx0 + i * rp, ry, rw, 24, "", fill=f, stroke=BORDER, tw=1, r=3)
        for i in starts:
            line(rx0 + i * rp - 1.5, ry - 5, rx0 + i * rp - 1.5, ry + 29,
                 color=col, tw=2)
        txt(rx0, ry + 42, sub, fs=10.5, fc=MUTED, anchor="left", w=460)

    txt(868, py + ph - 26, "The learned policy uses 5 tokens instead of 7—and places "
        "them where the audio changes.", fs=11, fc=INK2, anchor="left",
        italic=True, w=700)


def curriculum():
    """Band 3 — a continuous stage rail with objectives and updated components."""
    heading_y, rail_y = 576, 615
    xs = (300, 820, 1340)
    stages = (
        ("Cold start", "Boundary BCE · char-CTC targets", SEG, SEG_D, SEG_F),
        ("Decoder warmup", "Transcript CE · hard segmentation", ADAPT, ADAPT_D, ADAPT_F),
        ("Joint RL (GRPO)", "K samples · −NLL / −CER + rate band", RL, RL_D, RL_F),
    )

    txt(57, heading_y, "Training curriculum", fs=16, fc=INK, anchor="left",
        bold=True, w=260)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(a + 25, rail_y, b - 25, rail_y, color=BORDER_S, tw=2.4)

    for i, (x, (name, objective, col, dark, fill)) in enumerate(zip(xs, stages)):
        ell(x - 21, rail_y - 21, 42, 42, str(i + 1), fill=col, stroke=col,
            tw=1.2, fs=14, fc="#ffffff")
        txt(x, 648, name, fs=15, fc=INK, bold=True, w=330)
        txt(x, 671, objective, fs=11.5, fc=INK2, w=410)

        if i == 0:
            box(x - 90, 683, 180, 24, "updates · Segmenter", fill=fill,
                stroke=col, tw=1.1, fs=10.3, bold=True, fc=dark, r=6)
        elif i == 1:
            box(x - 120, 683, 240, 24, "updates · Projection + LoRA", fill=fill,
                stroke=col, tw=1.1, fs=10.3, bold=True, fc=dark, r=6)
        else:
            box(x - 150, 683, 168, 24, "updates · Segmenter", fill=SEG_F,
                stroke=SEG, tw=1.1, fs=10.1, bold=True, fc=SEG_D, r=6)
            box(x + 28, 683, 142, 24, "decoder optional", fill=CARD,
                stroke=ADAPT, dashed=True, tw=1.1, fs=10, bold=True,
                fc=ADAPT_D, r=6)


def legend():
    y = 748
    line(57, y - 22, 1583, y - 22, color=BORDER, tw=1)
    items = (("dashed = frozen", MUTED, None),
             ("segmentation", SEG_D, SEG_F),
             ("decoder adaptation", ADAPT_D, ADAPT_F),
             ("RL objective", RL_D, RL_F))
    x = 57
    for label, col, sw in items:
        if sw:
            box(x, y - 9, 16, 16, "", fill=sw, stroke=col, tw=1.6, r=3)
            x += 24
        txt(x, y, label, fs=11, fc=col, anchor="left", w=200)
        x += 5.9 * len(label) + 34
    txt(1583, y, "audio-token frequency = 50·N/T Hz", fs=11, fc=MUTED,
        anchor="right", w=300)


def build():
    """(Re)build the primitive list. Idempotent — safe to call more than once."""
    del P[:]
    spine()
    zoom_panel()
    curriculum()
    legend()
    return P


# --------------------------------------------------------------------------
# Emitter 1 — draw.io (uncompressed mxGraphModel, so the file stays diff-able)
# --------------------------------------------------------------------------
def _style(d):
    if d["k"] == "gate":
        return ("shape=or;html=1;whiteSpace=wrap;aspect=fixed;fillColor=%s;"
                "strokeColor=%s;strokeWidth=%s;" % (d["fill"], d["stroke"], d["tw"]))
    if d["k"] in ("box", "ell"):
        base = "ellipse;" if d["k"] == "ell" else "rounded=1;arcSize=%d;" % d.get("r", 10)
        return (base + "whiteSpace=wrap;html=1;fillColor=%s;strokeColor=%s;"
                "strokeWidth=%s;dashed=%d;fontColor=%s;fontSize=%s;fontStyle=%d;"
                "align=center;verticalAlign=middle;"
                % (d["fill"], d["stroke"] if d["tw"] else "none", d["tw"],
                   1 if d.get("dashed") else 0, d.get("fc", INK), d["fs"],
                   1 if d.get("bold") else 0))
    align = {"center": "center", "left": "left", "right": "right"}[d["anchor"]]
    st = 2 if d.get("italic") else 0
    st |= 1 if d.get("bold") else 0
    return ("text;html=1;strokeColor=none;fillColor=none;align=%s;verticalAlign=middle;"
            "whiteSpace=wrap;rounded=0;fontSize=%s;fontColor=%s;fontStyle=%d;"
            % (align, d["fs"], d["fc"], st))


def emit_drawio(path):
    mx = ET.Element("mxfile", host="Claude", agent="make_pipeline_figure.py", type="device")
    dia = ET.SubElement(mx, "diagram", id="pipeline", name="Pipeline")
    mod = ET.SubElement(dia, "mxGraphModel", dx="1600", dy="900", grid="0", gridSize="10",
                        guides="1", tooltips="1", connect="1", arrows="1", fold="1",
                        page="1", pageScale="1", pageWidth=str(W), pageHeight=str(H),
                        math="0", shadow="0", background="#ffffff")
    root = ET.SubElement(mod, "root")
    ET.SubElement(root, "mxCell", id="0")
    ET.SubElement(root, "mxCell", id="1", parent="0")

    for n, d in enumerate(P):
        i = "n%d" % n
        if d["k"] in ("box", "ell", "gate"):
            c = ET.SubElement(root, "mxCell", id=i, value=d.get("label", ""), style=_style(d),
                              vertex="1", parent="1")
            ET.SubElement(c, "mxGeometry", {"x": "%.1f" % d["x"], "y": "%.1f" % d["y"],
                          "width": "%.1f" % d["w"], "height": "%.1f" % d["h"],
                          "as": "geometry"})
        elif d["k"] == "txt":
            w, hh = d["w"], 20.0
            x = {"center": d["x"] - w / 2, "left": d["x"], "right": d["x"] - w}[d["anchor"]]
            c = ET.SubElement(root, "mxCell", id=i, value=d["s"], style=_style(d),
                              vertex="1", parent="1")
            ET.SubElement(c, "mxGeometry", {"x": "%.1f" % x,
                          "y": "%.1f" % (d["y"] - hh / 2), "width": "%.1f" % w,
                          "height": "%.1f" % hh, "as": "geometry"})
        else:
            st = ("edgeStyle=none;html=1;rounded=0;strokeColor=%s;strokeWidth=%s;"
                  "dashed=%d;endArrow=%s;endFill=1;endSize=6;"
                  % (d["color"], d["tw"], 1 if d["dashed"] else 0,
                     "blockThin" if d["arrow"] else "none"))
            c = ET.SubElement(root, "mxCell", id=i, style=st, edge="1", parent="1")
            g = ET.SubElement(c, "mxGeometry", {"relative": "1", "as": "geometry"})
            ET.SubElement(g, "mxPoint", {"x": "%.1f" % d["x1"], "y": "%.1f" % d["y1"],
                                         "as": "sourcePoint"})
            ET.SubElement(g, "mxPoint", {"x": "%.1f" % d["x2"], "y": "%.1f" % d["y2"],
                                         "as": "targetPoint"})

    xml = ET.tostring(mx, encoding="unicode")
    with open(path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n' + xml + "\n")
    return path


# --------------------------------------------------------------------------
# Emitter 2 — PNG (matplotlib; y flipped from draw.io space)
# --------------------------------------------------------------------------
def emit_png(path, dpi=200):
    """`path` may be a filename or any file-like object (used by `img_tag`)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Ellipse

    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=dpi)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)

    def fy(y):
        return H - y

    for d in P:
        if d["k"] == "box":
            r = min(d.get("r", 10), d["w"] / 2, d["h"] / 2)
            ax.add_patch(FancyBboxPatch(
                (d["x"], fy(d["y"] + d["h"])), d["w"], d["h"],
                boxstyle="round,pad=0,rounding_size=%f" % r, facecolor=d["fill"],
                edgecolor=d["stroke"] if d["tw"] else "none",
                linewidth=d["tw"], linestyle=(0, (5, 3)) if d.get("dashed") else "-",
                mutation_aspect=1, zorder=2))
            if d["label"]:
                ax.text(d["x"] + d["w"] / 2, fy(d["y"] + d["h"] / 2), d["label"],
                        ha="center", va="center", fontsize=d["fs"] * PT,
                        color=d.get("fc", INK),
                        fontweight="bold" if d.get("bold") else "normal", zorder=4)
        elif d["k"] == "ell":
            ax.add_patch(Ellipse((d["x"] + d["w"] / 2, fy(d["y"] + d["h"] / 2)),
                                 d["w"], d["h"], facecolor=d["fill"],
                                 edgecolor=d["stroke"], linewidth=d["tw"], zorder=2))
            ax.text(d["x"] + d["w"] / 2, fy(d["y"] + d["h"] / 2), d["label"],
                    ha="center", va="center", fontsize=d["fs"] * PT,
                    color=d.get("fc", INK), zorder=4)
        elif d["k"] == "gate":
            cx, cy = d["x"] + d["w"] / 2, fy(d["y"] + d["h"] / 2)
            ax.add_patch(Ellipse((cx, cy), d["w"], d["h"], facecolor=d["fill"],
                                 edgecolor=d["stroke"], linewidth=d["tw"], zorder=2))
            arm = min(d["w"], d["h"]) * 0.20
            ax.plot([cx - arm, cx + arm], [cy, cy], color=d["stroke"],
                    linewidth=d["tw"], solid_capstyle="round", zorder=4)
            ax.plot([cx, cx], [cy - arm, cy + arm], color=d["stroke"],
                    linewidth=d["tw"], solid_capstyle="round", zorder=4)
        elif d["k"] == "txt":
            ax.text(d["x"], fy(d["y"]), d["s"],
                    ha={"center": "center", "left": "left", "right": "right"}[d["anchor"]],
                    va="center", fontsize=d["fs"] * PT, color=d["fc"],
                    fontweight="bold" if d.get("bold") else "normal",
                    style="italic" if d.get("italic") else "normal", zorder=4)
        else:
            ls = (0, (5, 3)) if d["dashed"] else "-"
            if d["arrow"]:
                ax.add_patch(FancyArrowPatch(
                    (d["x1"], fy(d["y1"])), (d["x2"], fy(d["y2"])),
                    arrowstyle="-|>", mutation_scale=13, color=d["color"],
                    linewidth=d["tw"], linestyle=ls, shrinkA=0, shrinkB=0, zorder=3))
            else:
                ax.plot([d["x1"], d["x2"]], [fy(d["y1"]), fy(d["y2"])],
                        color=d["color"], linewidth=d["tw"], linestyle=ls,
                        solid_capstyle="round", zorder=3)

    fig.savefig(path, format="png", dpi=dpi, facecolor="white",
                bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return path


def img_tag(cls="fig", dpi=170):
    """Self-contained base64 <img> of this figure, for embedding in report.html.

    Called by make_training_report.py so the report always carries the *current*
    figure — there is no stale copy to forget to regenerate.
    """
    import base64
    import io
    build()
    buf = io.BytesIO()
    emit_png(buf, dpi=dpi)
    return '<img class="%s" src="data:image/png;base64,%s"/>' % (
        cls, base64.b64encode(buf.getvalue()).decode())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="artifacts/segmenter/figures")
    ap.add_argument("--name", default="pipeline")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    build()
    d = emit_drawio(os.path.join(a.outdir, a.name + ".drawio"))
    p = emit_png(os.path.join(a.outdir, a.name + ".png"))
    print("wrote %s (%d primitives)" % (d, len(P)))
    print("wrote %s" % p)
