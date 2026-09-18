#!/usr/bin/env python
"""CPU preflight for the CLSP port: are every ingredient of the flagship
WavLM Transformer-AR + BiGRU system present and loadable on this cluster?

Checks, in order of what a training/inference job touches:
  1. interpreter + torch/speechbrain/transformers import cleanly
  2. LibriSpeech audio decodes (via the `data_folder` the hparams now name)
  3. char boundary targets exist and parse for real utterance ids
  4. HF caches resolve offline for WavLM-Large and Llama-3.2-1B-Instruct
  5. the flagship cold-start / decoder / trained checkpoints load as tensors

Exit status is 0 only if every check passes. Run with the project venv:
  /export/jsalt26/omnienc/users/cxiao/envs/jointllm/bin/python \
      tools/preflight_clsp_assets.py
"""

from __future__ import annotations

import csv
import os
import sys
import traceback
from pathlib import Path

USER_ROOT = Path("/export/jsalt26/omnienc/users/cxiao")
REPO = USER_ROOT / "skipjack/jointllm"
RECIPE = REPO / "recipes/LibriSpeech/ASR/transformer"
HF_HUB = USER_ROOT / "hf/hub"
DATA_FOLDER = USER_ROOT / "datasets/LibriSpeech"
BOUNDARY_DIR = USER_ROOT / "datasets/wavlm_boundaries/char"
MANIFESTS = USER_ROOT / "datasets/librispeech_manifests"

FLAGSHIP = (
    RECIPE
    / "results/speechllm_segmenter_wavlm"
    / "fullprefix_transformer_ar_local64_bigru_best_char_decoder"
)
COLD_DIR = (
    RECIPE
    / "results/speechllm_segmenter_wavlm"
    / "fullprefix_transformer_ar_local64_best_char_decoder/coldstart/3407/save"
)
DECODER_CKPT = (
    RECIPE
    / "results/speechllm_fixed_pooling_wavlm/char_alignment_tc100/3408/save"
    / "CKPT+2026-08-09+17-11-52+00"
)
TRAINED_CKPT = FLAGSHIP / "nll_mt/3407/save/CKPT+2026-08-15+03-13-00+00"

results: list[tuple[str, bool, str]] = []


def check(name):
    def wrap(fn):
        try:
            detail = fn()
            results.append((name, True, detail or "ok"))
        except Exception as exc:  # noqa: BLE001 - preflight reports, never raises
            results.append((name, False, f"{type(exc).__name__}: {exc}"))
            if os.environ.get("PREFLIGHT_TRACEBACK"):
                traceback.print_exc()
        return fn

    return wrap


@check("interpreter + core packages")
def _packages():
    import speechbrain
    import torch
    import transformers

    sb_path = Path(speechbrain.__file__).resolve()
    if REPO not in sb_path.parents:
        raise RuntimeError(
            f"speechbrain resolves to {sb_path}, not the ported repo {REPO}"
        )
    return (
        f"python {sys.version.split()[0]}, torch {torch.__version__}, "
        f"transformers {transformers.__version__}, "
        f"speechbrain {speechbrain.__version__} (editable -> {REPO})"
    )


@check("LibriSpeech audio")
def _audio():
    import torchaudio

    splits = ["train-clean-100", "train-clean-360", "train-other-500",
              "dev-clean", "dev-other", "test-clean", "test-other"]
    missing = [s for s in splits if not (DATA_FOLDER / s).is_dir()]
    if missing:
        raise FileNotFoundError(f"missing splits: {missing}")
    row = next(csv.DictReader((MANIFESTS / "test-clean.csv").open()))
    wav = row["wav"].replace("$data_root/", str(DATA_FOLDER) + "/")
    sig, sr = torchaudio.load(wav)
    return (
        f"{len(splits)} splits under {DATA_FOLDER} "
        f"(-> {DATA_FOLDER.resolve()}); decoded {row['ID']} "
        f"{sig.shape[-1] / sr:.2f}s @ {sr} Hz"
    )


@check("char boundary targets")
def _boundaries():
    import torch

    if not BOUNDARY_DIR.is_dir():
        raise FileNotFoundError(BOUNDARY_DIR)
    ids = [
        next(csv.DictReader((MANIFESTS / f"{s}.csv").open()))["ID"]
        for s in ("train-clean-100", "dev-clean", "test-clean", "test-other")
    ]
    for utt in ids:
        target = torch.load(BOUNDARY_DIR / f"{utt}.pt").view(-1)
        if target.numel() == 0:
            raise ValueError(f"empty boundary target for {utt}")
    n = sum(1 for _ in BOUNDARY_DIR.iterdir())
    return f"{n} files under {BOUNDARY_DIR}; sampled {len(ids)} ids across splits"


@check("HF caches (offline)")
def _hf():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_CACHE"] = str(HF_HUB)
    from transformers import AutoConfig, AutoTokenizer

    wavlm = AutoConfig.from_pretrained(
        "microsoft/wavlm-large", cache_dir=str(HF_HUB), local_files_only=True
    )
    llama = AutoConfig.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct",
        cache_dir=str(HF_HUB),
        local_files_only=True,
    )
    tok = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct",
        cache_dir=str(HF_HUB),
        local_files_only=True,
    )
    weights = sorted(HF_HUB.glob("models--*/snapshots/*/*.safetensors"))
    weights += sorted(HF_HUB.glob("models--*/snapshots/*/*.bin"))
    if not weights:
        raise FileNotFoundError(f"no model weight files under {HF_HUB}")
    return (
        f"wavlm hidden={wavlm.hidden_size}, llama hidden={llama.hidden_size}, "
        f"tokenizer vocab={len(tok)}, {len(weights)} weight file(s)"
    )


@check("flagship checkpoints")
def _checkpoints():
    import torch

    cold = sorted(COLD_DIR.glob("CKPT*/segmenter.ckpt"))
    if not cold:
        raise FileNotFoundError(f"no cold-start segmenter under {COLD_DIR}")
    lines = []
    to_check = [
        ("cold-start segmenter", cold[-1]),
        ("decoder llm (LoRA)", DECODER_CKPT / "llm.ckpt"),
        ("decoder proj", DECODER_CKPT / "proj.ckpt"),
        ("decoder normalize", DECODER_CKPT / "normalize.ckpt"),
        ("trained segmenter (seed 3407)", TRAINED_CKPT / "segmenter.ckpt"),
        ("trained llm (seed 3407)", TRAINED_CKPT / "llm.ckpt"),
        ("trained proj (seed 3407)", TRAINED_CKPT / "proj.ckpt"),
    ]
    for label, path in to_check:
        state = torch.load(path, map_location="cpu", weights_only=False)
        if hasattr(state, "state_dict"):
            state = state.state_dict()
        n = len(state) if hasattr(state, "__len__") else 0
        if n == 0:
            raise ValueError(f"{label}: empty state dict at {path}")
        lines.append(f"{label}={n} tensors")
    has_bigru = any(
        "pool" in k.lower() and ("gru" in k.lower() or "rnn" in k.lower())
        for k in torch.load(
            TRAINED_CKPT / "segmenter.ckpt", map_location="cpu"
        )
    )
    lines.append(f"bigru pooler params present={has_bigru}")
    return "; ".join(lines)


def main() -> int:
    width = max(len(name) for name, _, _ in results)
    ok = True
    print("CLSP port preflight -- flagship WavLM Transformer-AR + BiGRU\n")
    for name, passed, detail in results:
        print(f"[{'PASS' if passed else 'FAIL'}] {name.ljust(width)}  {detail}")
        ok &= passed
    print()
    print("All ingredients present." if ok else "MISSING INGREDIENTS -- see FAIL above.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
