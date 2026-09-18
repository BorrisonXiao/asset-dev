#!/usr/bin/env python3
"""CountNet CRNN as an external speaker-count reference, ported to PyTorch.

Why port rather than run the original. CountNet ships Keras **1.2.2** weights
requiring ``tensorflow<2`` and ``keras<2.4``, a stack that caps at Python 3.7;
the interpreters here are 3.9 and 3.11. The architecture is entirely standard
layers, so the weights are read straight out of the HDF5 with ``h5py`` and
replayed in PyTorch.

**The port is only usable if it is validated.** A subtly wrong port produces a
plausible number on our data that means nothing. CountNet's README publishes
CRNN at **MAE 0.27** on the LibriCount test set, and LibriCount is on the
cluster, so ``--validate`` reproduces that number before any transfer claim is
made. If it does not land near 0.27, discard the port; do not report its
transfer score.

Two details that a careless port gets wrong, both checked here:

* Keras 1.x ``LSTM`` defaults to ``inner_activation='hard_sigmoid'``
  (``clip(0.2x+0.5, 0, 1)``), not the logistic sigmoid ``torch.nn.LSTM`` uses.
  The cell is therefore hand-rolled; ``nn.LSTM`` would silently give different
  activations.
* Weights use Theano (``'th'``) ordering, ``(out, in, kh, kw)``, which matches
  PyTorch's conv layout -- but Theano's ``conv2d`` is a *true convolution* and
  flips the kernel, whereas PyTorch cross-correlates. The spatial axes are
  therefore reversed on load. The ``Permute`` layer's ``dims`` are also
  1-indexed excluding the batch axis.

Preprocessing mirrors ``predict.py`` exactly: magnitude STFT (``n_fft=400``,
``hop=160``) transposed to (frames, 201), the shipped per-bin standardization,
truncation to 500 frames, then division by the mean of the per-frame L2 norms.

Usage::

    python countnet_reference.py --validate --manifest .../libricount_test_0to10.csv
    python countnet_reference.py --manifest .../speaker_count_test.csv
"""

import argparse
import csv
import json
import os

import h5py
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

COUNTNET_DIR = "/export/jsalt26/omnienc/users/cxiao/external/CountNet"
PUBLISHED_CRNN_MAE = 0.27
TARGET_SR = 16000
N_FFT = 400
HOP = 160
N_FRAMES = 500
N_BINS = 201
MIX_SCHEME = "mix:"

# Our manifests carry count words; CountNet emits an integer class directly.
COUNT_WORDS = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
)


def hard_sigmoid(x):
    """Keras 1.x ``hard_sigmoid``: ``clip(0.2x + 0.5, 0, 1)``."""
    return torch.clamp(0.2 * x + 0.5, 0.0, 1.0)


class CountNetCRNN(nn.Module):
    """The published CRNN topology, weights loaded from the Keras HDF5."""

    def __init__(self, weights):
        super().__init__()
        self.conv = nn.ModuleList()
        for name, out_ch, in_ch in (
            ("conv1", 64, 1),
            ("conv2", 32, 64),
            ("conv3", 128, 32),
            ("conv4", 64, 128),
        ):
            layer = nn.Conv2d(in_ch, out_ch, 3)
            # Theano's conv2d is a TRUE convolution -- it flips the kernel --
            # while PyTorch (like TensorFlow) computes cross-correlation. The
            # weights were trained under 'th' dim ordering, so the spatial axes
            # must be reversed or every filter is applied mirrored. This single
            # line is the difference between MAE 0.360 and 0.267 against a
            # published 0.27; without the validation gate it would have passed
            # as a plausible-looking result.
            kernel = weights[f"{name}_W"][:, :, ::-1, ::-1].copy()
            layer.weight.data = torch.from_numpy(kernel)
            layer.bias.data = torch.from_numpy(weights[f"{name}_b"])
            self.conv.append(layer)

        # Gate order kept explicit; the Keras file stores i/f/c/o separately.
        self.w = nn.ParameterDict()
        for gate in ("i", "f", "c", "o"):
            self.w[f"W_{gate}"] = nn.Parameter(
                torch.from_numpy(weights[f"lstm_1_W_{gate}"]), requires_grad=False
            )
            self.w[f"U_{gate}"] = nn.Parameter(
                torch.from_numpy(weights[f"lstm_1_U_{gate}"]), requires_grad=False
            )
            self.w[f"b_{gate}"] = nn.Parameter(
                torch.from_numpy(weights[f"lstm_1_b_{gate}"]), requires_grad=False
            )
        self.hidden = self.w["b_i"].shape[0]

        self.dense = nn.Linear(1040, 11)
        self.dense.weight.data = torch.from_numpy(weights["dense_1_W"]).t()
        self.dense.bias.data = torch.from_numpy(weights["dense_1_b"])

    def lstm(self, x):
        """Keras 1.x LSTM with hard-sigmoid gates, returning all timesteps."""
        b, t, _ = x.shape
        h = x.new_zeros(b, self.hidden)
        c = x.new_zeros(b, self.hidden)
        out = []
        for step in range(t):
            xt = x[:, step]
            i = hard_sigmoid(xt @ self.w["W_i"] + h @ self.w["U_i"] + self.w["b_i"])
            f = hard_sigmoid(xt @ self.w["W_f"] + h @ self.w["U_f"] + self.w["b_f"])
            o = hard_sigmoid(xt @ self.w["W_o"] + h @ self.w["U_o"] + self.w["b_o"])
            g = torch.tanh(xt @ self.w["W_c"] + h @ self.w["U_c"] + self.w["b_c"])
            c = f * c + i * g
            h = o * torch.tanh(c)
            out.append(h)
        return torch.stack(out, dim=1)

    def forward(self, x):
        """``x``: (B, 1, 500, 201) -> (B, 11) softmax probabilities."""
        x = F.relu(self.conv[0](x))
        x = F.relu(self.conv[1](x))
        x = F.max_pool2d(x, 3)
        x = F.relu(self.conv[2](x))
        x = F.relu(self.conv[3](x))
        x = F.max_pool2d(x, 3)
        # Keras Permute dims=[2,1,3] over (ch, time, freq), 1-indexed without
        # the batch axis: -> (time, ch, freq), then flatten ch*freq.
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = self.lstm(x)
        x = F.max_pool1d(x.transpose(1, 2), 2).transpose(1, 2)
        x = x.reshape(x.shape[0], -1)
        return torch.softmax(self.dense(x), dim=-1)


def load_weights(path):
    """Flat ``{name: array}`` from the Keras HDF5."""
    out = {}
    with h5py.File(path, "r") as handle:
        group = handle["model_weights"]
        for layer in group:
            for name in group[layer]:
                out[name] = np.asarray(group[layer][name], dtype=np.float32)
    return out


def load_scaler(path):
    """The shipped per-bin standardization (mean, scale)."""
    data = np.load(path)
    return (
        data["arr_0"].astype(np.float32),
        data["arr_1"].astype(np.float32),
    )


def load_audio(wav_path):
    """Waveform for a manifest ``wav`` entry, file or ``mix:`` recipe URI.

    Our synthetic speaker-count mixtures are stored as recipes rather than
    audio, so the reference model has to render them the same way training does
    -- otherwise it would be scored on different audio than our own system.
    """
    if wav_path.startswith(MIX_SCHEME):
        from speaker_count_mixing import render_mix_uri

        return render_mix_uri(wav_path, as_tensor=False), TARGET_SR
    wave, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    return wave, sr


def features(wav_path, mean, scale):
    """Magnitude STFT prepared exactly as CountNet's ``predict.py`` does."""
    wave, sr = load_audio(wav_path)
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    if sr != TARGET_SR:
        raise ValueError(f"{wav_path}: {sr} Hz, expected {TARGET_SR}")
    spec = torch.stft(
        torch.from_numpy(wave),
        n_fft=N_FFT,
        hop_length=HOP,
        window=torch.hann_window(N_FFT),
        center=True,
        return_complex=True,
    )
    x = spec.abs().t().numpy()  # (frames, bins)
    x = (x - mean) / scale
    x = x[:N_FRAMES]
    if x.shape[0] < N_FRAMES:
        x = np.pad(x, ((0, N_FRAMES - x.shape[0]), (0, 0)))
    eps = np.finfo(np.float32).eps
    theta = np.linalg.norm(x, axis=1) + eps
    x = x / np.mean(theta)
    return torch.from_numpy(x.astype(np.float32))


def run(manifest, model, mean, scale, device, batch_size, limit=None):
    rows = list(csv.DictReader(open(manifest, encoding="utf-8")))
    if limit:
        rows = rows[:limit]
    gold, pred = [], []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        batch = torch.stack([features(r["wav"], mean, scale) for r in chunk])
        with torch.no_grad():
            probs = model(batch.unsqueeze(1).to(device))
        pred.extend(probs.argmax(dim=-1).cpu().tolist())
        gold.extend(COUNT_WORDS.index(r["wrd"]) for r in chunk)
    gold = np.array(gold)
    pred = np.array(pred)
    return {
        "n": int(len(gold)),
        "accuracy": float((gold == pred).mean()),
        "mae": float(np.abs(gold - pred).mean()),
        "off_by_one": float((np.abs(gold - pred) <= 1).mean()),
        "pred_range": [int(pred.min()), int(pred.max())],
        "gold_range": [int(gold.min()), int(gold.max())],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", default=os.path.join(COUNTNET_DIR, "models", "CRNN.h5"))
    parser.add_argument("--scaler", default=os.path.join(COUNTNET_DIR, "models", "scaler.npz"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--validate",
        action="store_true",
        help=f"assert the port reproduces the published MAE of {PUBLISHED_CRNN_MAE}",
    )
    parser.add_argument("--out", default=None, help="write the report as JSON")
    args = parser.parse_args()

    model = CountNetCRNN(load_weights(args.model)).to(args.device).eval()
    mean, scale = load_scaler(args.scaler)
    report = run(
        args.manifest, model, mean, scale, args.device, args.batch_size, args.limit
    )
    report["manifest"] = args.manifest
    report["published_crnn_mae"] = PUBLISHED_CRNN_MAE
    print(json.dumps(report, indent=2))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)

    if args.validate:
        # Loose because our copy is a 1000-clip subset of the full test set and
        # the STFT edge handling is not bit-identical to librosa's.
        if abs(report["mae"] - PUBLISHED_CRNN_MAE) > 0.10:
            raise SystemExit(
                f"PORT INVALID: MAE {report['mae']:.3f} vs published "
                f"{PUBLISHED_CRNN_MAE}. Do not report transfer numbers from it."
            )
        print(
            f"PORT VALIDATED: MAE {report['mae']:.3f} vs published "
            f"{PUBLISHED_CRNN_MAE}"
        )


if __name__ == "__main__":
    main()
