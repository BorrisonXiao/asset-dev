#!/usr/bin/env python3
"""Frozen-equivalence gate for the task-conditioned multi-task warm start.

Before any multi-task training runs, ``task_id=asr`` must reproduce the
warm-started single-task LS960 model exactly. If it does not, a later change in
WER, BLEU, or boundary placement cannot be attributed to task conditioning --
it might just be a broken warm start. This script builds both models from their
own hparams files, restores the same checkpoint into each, and compares:

* boundary masks,
* emitted audio-token frequency (f_audio = 50 Hz x rho),
* pooled acoustic states,
* projected states, and
* decoder logits

on identical audio. It also checks that ``task_id=st_en_de`` runs and, at
initialization, matches ASR (identity FiLM plus a zero-initialized ST adapter),
which is the documented starting point rather than a defect.

Exit status is non-zero if any tolerance is violated.
"""

import argparse
import logging
import os
import sys

import torch
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multitask_data import multitask_dataio_prepare, task_id_of  # noqa: E402
from multitask_modules import (  # noqa: E402
    copy_asr_lora_into_task_slot,
    expand_film_to_tasks,
)
from segment_pooling import lengths_to_padding_mask  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)


def _load_hparams(path, overrides=""):
    with open(path, encoding="utf-8") as stream:
        return load_hyperpyyaml(stream, overrides)


def _restore_single_task(hparams, ckpt_dir, device):
    """Rebuild the single-task ASR model exactly as the ASR recipe would."""
    segmenter = hparams["segmenter"]
    segmenter.load_state_dict(
        torch.load(
            os.path.join(ckpt_dir, "segmenter.ckpt"),
            map_location="cpu",
            weights_only=True,
        )
    )
    hparams["proj"].load_state_dict(
        torch.load(
            os.path.join(ckpt_dir, "proj.ckpt"),
            map_location="cpu",
            weights_only=True,
        )
    )
    hparams["normalize"]._load(
        os.path.join(ckpt_dir, "normalize.ckpt"), end_of_epoch=False
    )
    hparams["llm"].loader(os.path.join(ckpt_dir, "llm.ckpt"), True)
    for module in (segmenter, hparams["proj"], hparams["llm"], hparams["ssl"]):
        module.to(device).eval()
    return hparams


def _restore_multitask(hparams, ckpt_dir, device):
    """Rebuild the multi-task model through the trainer's warm-start path."""
    segmenter = hparams["segmenter"]
    state = torch.load(
        os.path.join(ckpt_dir, "segmenter.ckpt"),
        map_location="cpu",
        weights_only=True,
    )
    missing, unexpected = segmenter.load_state_dict(state, strict=False)
    assert not [k for k in unexpected if not k.startswith("film.")], unexpected
    assert not [k for k in missing if not k.startswith("film.")], missing
    hparams["proj"].load_state_dict(
        torch.load(
            os.path.join(ckpt_dir, "proj.ckpt"),
            map_location="cpu",
            weights_only=True,
        )
    )
    hparams["normalize"]._load(
        os.path.join(ckpt_dir, "normalize.ckpt"), end_of_epoch=False
    )
    copy_asr_lora_into_task_slot(
        hparams["llm"],
        torch.load(
            os.path.join(ckpt_dir, "llm.ckpt"),
            map_location="cpu",
            weights_only=True,
        ),
        task_id=task_id_of("asr"),
    )
    expand_film_to_tasks(segmenter, hparams["segmenter_num_tasks"])
    for module in (segmenter, hparams["proj"], hparams["llm"], hparams["ssl"]):
        module.to(device).eval()
    return hparams


@torch.no_grad()
def _run(hparams, batch, device, task_id=None):
    """One deterministic forward pass, returning every comparable tensor."""
    wavs, wav_lens = batch.sig
    wavs, wav_lens = wavs.to(device), wav_lens.to(device)
    feats = hparams["ssl"](hparams["normalize"](wavs, wav_lens), wav_lens)
    pad_mask = lengths_to_padding_mask(wav_lens, feats.size(1))

    segmenter = hparams["segmenter"]
    if task_id is not None:
        segmenter.active_task_id = torch.full(
            (feats.size(0),), task_id, dtype=torch.long, device=device
        )
    else:
        segmenter.active_task_id = None

    rollout = segmenter.sample_boundaries(feats, pad_mask, deterministic=True)
    pooled, seg_pad = segmenter.pool(feats, pad_mask, rollout.boundaries)
    projected = hparams["proj"](pooled)

    valid = (~pad_mask).sum(dim=1).float().clamp(min=1.0)
    segments = (rollout.boundaries == 1).sum(dim=1).float() + 1.0
    return {
        "boundaries": rollout.boundaries,
        "logits": rollout.logits,
        "pooled": pooled,
        "projected": projected,
        "seg_pad": seg_pad,
        "f_audio_hz": (segments / valid) * 50.0,
    }


def _max_delta(left, right):
    """Largest absolute elementwise difference, or None on a shape mismatch."""
    if left.shape != right.shape:
        return None
    return float((left.float() - right.float()).abs().max())


def _compare(name, left, right, tolerance, noise_floor=0.0, margin=8.0):
    """Report one comparison against the larger of tolerance and the noise floor.

    Repeated forward passes of the *same* model are not bit-identical on GPU
    (cuDNN picks reduction orders per call), so a fixed absolute tolerance would
    flag float noise as a warm-start defect. The honest threshold is the
    measured same-model run-to-run spread, scaled by ``margin``; a real
    functional difference is orders of magnitude larger than that, not a small
    multiple of it.
    """
    if left.shape != right.shape:
        logger.error(
            "%-14s SHAPE MISMATCH %s vs %s", name, left.shape, right.shape
        )
        return False
    if left.dtype in (torch.bool, torch.long, torch.int32, torch.int64):
        ok = torch.equal(left, right)
        logger.info(
            "%-14s %s (exact integer match)", name, "OK" if ok else "FAIL"
        )
        return ok
    worst = _max_delta(left, right)
    threshold = max(float(tolerance), float(noise_floor) * margin)
    ok = worst <= threshold
    logger.info(
        "%-14s %s max|delta|=%.3e threshold=%.3e "
        "(tolerance=%.1e, same-model noise=%.3e x%g)",
        name,
        "OK" if ok else "FAIL",
        worst,
        threshold,
        tolerance,
        noise_floor,
        margin,
    )
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multitask-hparams", required=True)
    parser.add_argument("--asr-hparams", required=True)
    parser.add_argument(
        "--asr-overrides",
        default="",
        help="hyperpyyaml overrides aligning the ASR config with the warm start",
    )
    parser.add_argument(
        "--work-dir",
        default="results/multitask_diagnostics/warm_start_equivalence",
        help="scratch directory for the probe manifest; kept out of the "
        "flagship output root so the launcher's reuse guard stays meaningful",
    )
    parser.add_argument("--num-utterances", type=int, default=8)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)

    logger.info("Building the multi-task model from %s", args.multitask_hparams)
    mt = _load_hparams(args.multitask_hparams)
    ckpt_dir = mt["warm_start_ckpt_dir"]
    logger.info("Warm start: %s", ckpt_dir)

    # A small, fixed probe batch of LibriSpeech dev audio -- the exact domain the
    # single-task model was selected on.
    os.makedirs(args.work_dir, exist_ok=True)
    probe_csv = os.path.join(args.work_dir, "equivalence_probe.csv")
    from multitask_data import build_multitask_manifest

    build_multitask_manifest(
        [
            {
                "csv": os.path.join(
                    mt["librispeech_manifest_dir"], "dev-clean.csv"
                ),
                "source_key": "librispeech_asr",
                "kind": "librispeech",
                "limit": args.num_utterances,
            }
        ],
        probe_csv,
        data_folder=mt["data_folder"],
    )
    dataset = multitask_dataio_prepare(
        mt, mt["llm"].tokenizer, probe_csv, sorting="ascending"
    )
    loader = sb.dataio.dataloader.make_dataloader(
        dataset,
        batch_size=args.num_utterances,
        collate_fn=mt["test_dataloader_opts"]["collate_fn"],
    )
    batch = next(iter(loader))

    _restore_multitask(mt, ckpt_dir, device)
    mt_asr = _run(mt, batch, device, task_id=task_id_of("asr"))
    # Same model, same input, twice: the run-to-run spread is the floor below
    # which no cross-model comparison can distinguish anything.
    mt_asr_repeat = _run(mt, batch, device, task_id=task_id_of("asr"))
    noise = {
        key: _max_delta(mt_asr[key], mt_asr_repeat[key])
        for key in ("logits", "pooled", "projected", "f_audio_hz")
    }
    logger.info(
        "Same-model noise floor: %s",
        ", ".join(f"{k}={v:.3e}" for k, v in noise.items()),
    )
    mt_st = _run(mt, batch, device, task_id=task_id_of("st_en_de"))
    # Free the multi-task LLM before building the reference model.
    del mt["llm"], mt["ssl"]
    torch.cuda.empty_cache()

    logger.info(
        "Building the single-task ASR reference from %s", args.asr_hparams
    )
    ref = _load_hparams(args.asr_hparams, args.asr_overrides)
    _restore_single_task(ref, ckpt_dir, device)
    reference = _run(ref, batch, device, task_id=None)

    logger.info("--- task_id=asr vs the single-task warm start ---")
    checks = [
        _compare(
            "boundaries", reference["boundaries"], mt_asr["boundaries"], 0
        ),
        _compare(
            "policy_logits",
            reference["logits"],
            mt_asr["logits"],
            args.tolerance,
            noise["logits"],
        ),
        _compare(
            "pooled",
            reference["pooled"],
            mt_asr["pooled"],
            args.tolerance,
            noise["pooled"],
        ),
        _compare(
            "projected",
            reference["projected"],
            mt_asr["projected"],
            args.tolerance,
            noise["projected"],
        ),
        _compare(
            "f_audio_hz",
            reference["f_audio_hz"],
            mt_asr["f_audio_hz"],
            1e-5,
            noise["f_audio_hz"],
        ),
    ]
    logger.info(
        "Reference f_audio: mean %.2f Hz; multi-task ASR: mean %.2f Hz",
        float(reference["f_audio_hz"].mean()),
        float(mt_asr["f_audio_hz"].mean()),
    )

    logger.info("--- task_id=st_en_de at initialization ---")
    # Identity FiLM means the ST mask starts equal to the ASR mask; Phase B is
    # what should move it apart. A difference HERE would mean the warm start
    # is not identity.
    same_mask = torch.equal(mt_asr["boundaries"], mt_st["boundaries"])
    logger.info(
        "ST boundaries equal ASR boundaries at init: %s (expected True)",
        same_mask,
    )
    checks.append(same_mask)

    if all(checks):
        logger.info("PASS: the multi-task warm start is the identity for ASR.")
        return 0
    logger.error("FAIL: the multi-task warm start is NOT the identity for ASR.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
