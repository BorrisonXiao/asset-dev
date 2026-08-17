#!/usr/bin/env python3
"""Generate the qualitative CTC-char versus learned-boundary figure.

The WER values are for LibriSpeech dev-clean utterance 2277-149874-0017:
the CTC-character-boundary SpeechLLM baseline makes one substitution (1/10),
while the learned-boundary model shown here transcribes all ten words exactly.

Pass ``--show-char-intervals`` to make an alignment-diagnostic version. It
labels every interval opened by a CTC token span (``␣`` denotes the CTC word
separator) and labels the synthetic frame-0 interval as leading silence.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import torch
import torchaudio

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RECIPE = Path(__file__).resolve().parent
ROOT = RECIPE.parents[3]
DATA = Path(
    "/home/jhu/jsalt2026-ext-cxiao7/"
    "scratch_jsalt2026-lgarci27/omnienc/datasets"
)
SSL_CACHE = Path(
    "/weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/ssl_cache"
)
SSL_SNAPSHOT = (
    SSL_CACHE
    / "models--facebook--wav2vec2-base-960h/snapshots"
    / "22aad52d435eb6dbaf354bdad9b0da84ce7d6156"
)

DEFAULT_UID = "2277-149874-0017"
DEFAULT_WAV = (
    DATA
    / "LibriSpeech/dev-clean/2277/149874"
    / f"{DEFAULT_UID}.flac"
)
DEFAULT_CHAR_BOUNDARIES = (
    DATA / "wavlm_boundaries/char" / f"{DEFAULT_UID}.pt"
)
DEFAULT_SEGMENTER = (
    RECIPE
    / "results/speechllm_segmenter/abl/nll_mt/cnn/3407/save"
    / "CKPT+2026-08-03+01-13-02+00/segmenter.ckpt"
)
DEFAULT_OUTPUT = (
    ROOT
    / "artifacts/segmenter/diagrams_highres"
    / "07_spectrogram_boundaries.png"
)
TRANSCRIPT = "a shop girl was the destiny prefigured for the newcomer"

# Exact 300-dpi canvas and axes geometry used by the presentation figure.
WIDTH_PX, HEIGHT_PX, DPI = 2904, 1234, 300
PLOT_LEFT, PLOT_RIGHT = 84, 2875
AXES_TOP_BOTTOM = {
    "spectrogram": (28, 643),
    "ctc": (682, 820),
    "learned": (860, 998),
    "transcript": (1038, 1186),
}

CTC_FILL = "#e5f3fa"
CTC_COLOR = "#469ac4"
CTC_TEXT = "#174f78"
LEARNED_FILL = "#fdebea"
LEARNED_COLOR = "#cf4e4e"
LEARNED_TEXT = "#9b2f2f"


def _axes_rect(top: int, bottom: int):
    """Convert pixel coordinates measured from the top into figure fractions."""
    return (
        PLOT_LEFT / WIDTH_PX,
        (HEIGHT_PX - bottom) / HEIGHT_PX,
        (PLOT_RIGHT - PLOT_LEFT) / WIDTH_PX,
        (bottom - top) / HEIGHT_PX,
    )


def _load_boundaries(wav_path, char_path, segmenter_path):
    sys.path.insert(0, str(RECIPE))
    from segmenter import Segmenter

    from speechbrain.integrations.huggingface.wav2vec2 import Wav2Vec2
    from speechbrain.processing.features import InputNormalization

    ssl_source = (
        str(SSL_SNAPSHOT)
        if SSL_SNAPSHOT.is_dir()
        else "facebook/wav2vec2-base-960h"
    )
    ssl = Wav2Vec2(
        source=ssl_source,
        output_norm=True,
        freeze=True,
        save_path=str(SSL_CACHE),
        device_map="cpu",
    )
    normalization = InputNormalization(norm_type="sentence")
    segmenter = Segmenter(input_dim=768, backbone="cnn", hidden_dim=256)
    segmenter.load_state_dict(torch.load(segmenter_path, map_location="cpu"))
    segmenter.eval()
    ssl.eval()

    wav, sample_rate = torchaudio.load(wav_path)
    wav = wav.mean(0, keepdim=True)
    if sample_rate != 16000:
        wav = torchaudio.functional.resample(wav, sample_rate, 16000)

    with torch.no_grad():
        wav_lens = torch.ones(1)
        features = ssl(normalization(wav, wav_lens), wav_lens)
        padding_mask = torch.zeros(
            1, features.size(1), dtype=torch.bool
        )
        logits = segmenter.boundary_logits(features, padding_mask)

    learned = [
        boundary
        for boundary in (logits[0] > 0).nonzero().flatten().tolist()
        if boundary > 0
    ]
    char_target = torch.load(char_path, map_location="cpu").view(-1)
    if char_target.numel() != features.size(1):
        raise ValueError(
            f"{char_path}: {char_target.numel()} target frames, but the "
            f"encoder produced {features.size(1)} frames"
        )
    values = set(torch.unique(char_target).tolist())
    if values > {0, 1}:
        raise ValueError(f"{char_path}: expected binary labels, got {values}")
    if char_target[0].item() != 1:
        raise ValueError(f"{char_path}: frame 0 must open the first segment")
    ctc = [
        boundary
        for boundary in (char_target == 1).nonzero().flatten().tolist()
        if boundary > 0
    ]
    logmel = torch.log(
        torchaudio.transforms.MelSpectrogram(
            16000, n_fft=400, hop_length=320, n_mels=80
        )(wav)[0]
        + 1e-6
    ).numpy()

    mel_frames = logmel.shape[1]
    scale = mel_frames / max(features.size(1), 1)
    # The historical presentation export placed boundary centers one pixel to
    # the left of Matplotlib's default rasterization. Preserve that geometry.
    one_pixel = mel_frames / (PLOT_RIGHT - PLOT_LEFT)
    ctc_x = [boundary * scale - one_pixel for boundary in ctc]
    learned_x = [boundary * scale - one_pixel for boundary in learned]
    return logmel, ctc_x, learned_x, len(ctc), len(learned)


def _char_intervals(boundary_positions, mel_frames, transcript):
    """Pair ordered CTC token starts with their displayed intervals.

    ``ctc_boundary_align.py`` writes one nonzero boundary for every merged CTC
    target token, in transcript order, plus a synthetic boundary at frame 0.
    The count assertion below prevents a plausible-looking but shifted label
    overlay if the transcript and saved target ever stop matching.
    """
    tokens = list(transcript.upper().replace(" ", "|"))
    if len(tokens) != len(boundary_positions):
        raise ValueError(
            f"Transcript has {len(tokens)} CTC tokens, but the target has "
            f"{len(boundary_positions)} nonzero token boundaries"
        )

    intervals = [(0.0, boundary_positions[0], "sil")]
    for index, (start, token) in enumerate(zip(boundary_positions, tokens)):
        end = (
            boundary_positions[index + 1]
            if index + 1 < len(boundary_positions)
            else mel_frames
        )
        intervals.append((start, end, "␣" if token == "|" else token))
    return intervals


def generate(args):
    logmel, ctc_x, learned_x, n_ctc, n_learned = _load_boundaries(
        args.wav, args.char_boundaries, args.segmenter
    )
    mel_frames = logmel.shape[1]
    px_in_data = mel_frames / (PLOT_RIGHT - PLOT_LEFT)
    char_intervals = None
    if args.show_char_intervals:
        char_intervals = _char_intervals(ctc_x, mel_frames, args.transcript)

    fig = plt.figure(
        figsize=(WIDTH_PX / DPI, HEIGHT_PX / DPI),
        dpi=DPI,
        facecolor="white",
    )
    axes = {
        name: fig.add_axes(_axes_rect(*bounds))
        for name, bounds in AXES_TOP_BOTTOM.items()
    }

    spectrogram = axes["spectrogram"]
    spectrogram.imshow(
        logmel,
        origin="lower",
        aspect="auto",
        cmap="magma",
        extent=[0, mel_frames, 0, 80],
        interpolation="nearest",
    )
    # A one-pixel offset keeps coincident blue/red guides distinguishable.
    for boundary in ctc_x:
        spectrogram.axvline(
            boundary - px_in_data,
            color=CTC_COLOR,
            linewidth=1.45,
            alpha=0.65,
        )
    for boundary in learned_x:
        spectrogram.axvline(
            boundary + px_in_data,
            color=LEARNED_COLOR,
            linewidth=1.45,
            alpha=0.69,
        )
    spectrogram.set_xlim(0, mel_frames)
    spectrogram.set_ylim(0, 80)

    def boundary_lane(
        ax,
        positions,
        fill,
        color,
        annotation,
        text_color,
        intervals=None,
    ):
        ax.set_facecolor(fill)
        if intervals:
            for index, (start, end, label) in enumerate(intervals):
                if index % 2 == 0:
                    ax.axvspan(
                        start,
                        end,
                        facecolor="white",
                        alpha=0.30,
                        linewidth=0,
                    )
                ax.text(
                    (start + end) / 2,
                    0.50,
                    label,
                    ha="center",
                    va="center",
                    fontsize=4.2,
                    fontfamily="DejaVu Sans Mono",
                    fontweight="bold",
                    color=text_color,
                    clip_on=True,
                    zorder=4,
                )
        for boundary in positions:
            ax.axvline(boundary, color=color, linewidth=1.70, alpha=1.0)
        ax.set_xlim(0, mel_frames)
        ax.set_ylim(0, 1)
        if annotation:
            ax.text(
                0.986,
                0.50,
                annotation,
                transform=ax.transAxes,
                ha="right",
                va="center",
                fontsize=7.0,
                fontweight="normal",
                color=text_color,
                zorder=5,
                bbox={
                    "boxstyle": "round,pad=0.34",
                    "facecolor": "white",
                    "edgecolor": color,
                    "linewidth": 0.65,
                    "alpha": 0.92,
                },
            )

    boundary_lane(
        axes["ctc"],
        ctc_x,
        CTC_FILL,
        CTC_COLOR,
        None
        if args.show_char_intervals
        else f"WER {args.ctc_wer:.1f}% (1 sub)",
        CTC_TEXT,
        intervals=char_intervals,
    )
    boundary_lane(
        axes["learned"],
        learned_x,
        LEARNED_FILL,
        LEARNED_COLOR,
        None
        if args.show_char_intervals
        else f"WER {args.learned_wer:.1f}% (exact)",
        LEARNED_TEXT,
    )

    transcript = axes["transcript"]
    transcript.set_facecolor("white")
    transcript.set_xlim(0, mel_frames)
    transcript.set_ylim(0, 1)
    transcript.text(
        0.5,
        0.5,
        args.transcript,
        transform=transcript.transAxes,
        ha="center",
        va="center",
        fontsize=8.2,
        color="black",
    )
    for ax in axes.values():
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(0.8)

    label_x = 28 / WIDTH_PX
    for label, key in (
        ("spectrogram", "spectrogram"),
        ("CTC-char", "ctc"),
        ("learned", "learned"),
        ("transcript", "transcript"),
    ):
        top, bottom = AXES_TOP_BOTTOM[key]
        center_y = 1.0 - ((top + bottom) / 2) / HEIGHT_PX
        fig.text(
            label_x,
            center_y,
            label,
            rotation=90,
            ha="center",
            va="center",
            fontsize=7.2,
            color="black",
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(
        f"wrote {args.output} ({WIDTH_PX}x{HEIGHT_PX}); "
        f"CTC={n_ctc}, learned={n_learned}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", type=Path, default=DEFAULT_WAV)
    parser.add_argument(
        "--char-boundaries", type=Path, default=DEFAULT_CHAR_BOUNDARIES
    )
    parser.add_argument("--segmenter", type=Path, default=DEFAULT_SEGMENTER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ctc-wer", type=float, default=10.0)
    parser.add_argument("--learned-wer", type=float, default=0.0)
    parser.add_argument("--transcript", default=TRANSCRIPT)
    parser.add_argument(
        "--show-char-intervals",
        action="store_true",
        help="Label each CTC-char interval; ␣ marks a word separator.",
    )
    generate(parser.parse_args())


if __name__ == "__main__":
    main()
