#!/usr/bin/env python3
"""Same-audio ASR/ST boundary analysis for a trained multi-task checkpoint.

This produces the evidence the plan's scientific gate needs beyond the metric
comparison: proof that switching the task actually changes *which frames are
retained* for the same utterance, by more than deterministic noise.

For every paired CoVoST utterance it runs the segmenter twice -- once per task
id -- on identical audio and reports

* boundary-mask Jaccard and tolerance-F1 between the two tasks,
* per-task emitted frequency (f_audio = 50 Hz x rho) and their difference,
* segment-duration distributions,

against two null distributions:

* a **repeat null**: the same task run twice, which captures GPU
  nondeterminism, and
* a **permutation null**: task labels shuffled across utterances, which
  captures how different two masks look for unrelated reasons.

A task-conditioning effect is only credible when the ASR/ST divergence exceeds
both nulls. It also reports the cosine similarity between per-task GRPO policy
gradients on the shared task-FiLM parameters -- the objective-conflict
diagnostic the plan asks for.
"""

import argparse
import json
import logging
import os
import sys

import torch
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multitask_data import (  # noqa: E402
    build_multitask_manifest,
    multitask_dataio_prepare,
    task_id_of,
)
from multitask_modules import film_parameters  # noqa: E402
from segment_pooling import lengths_to_padding_mask  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

FRAME_SECONDS = 0.02  # 50 Hz encoder grid


def _boundary_positions(mask, valid):
    """Frame indices marked as segment starts within the valid region."""
    return torch.nonzero(((mask == 1) & valid), as_tuple=False).view(-1)


def _jaccard(left, right, valid):
    """Intersection over union of two boundary sets on one utterance."""
    a = (left == 1) & valid
    b = (right == 1) & valid
    union = (a | b).sum().item()
    if union == 0:
        return 1.0
    return (a & b).sum().item() / union


def _tolerance_f1(left, right, valid, tolerance):
    """Boundary F1 allowing a +/- ``tolerance`` frame match window."""
    a = _boundary_positions(left, valid)
    b = _boundary_positions(right, valid)
    if a.numel() == 0 and b.numel() == 0:
        return 1.0
    if a.numel() == 0 or b.numel() == 0:
        return 0.0
    distance = (a.view(-1, 1) - b.view(1, -1)).abs()
    hits_a = (distance.min(dim=1).values <= tolerance).float().mean().item()
    hits_b = (distance.min(dim=0).values <= tolerance).float().mean().item()
    if hits_a + hits_b == 0:
        return 0.0
    return 2 * hits_a * hits_b / (hits_a + hits_b)


def _segment_durations(mask, valid):
    """Segment lengths in seconds implied by a boundary mask."""
    positions = _boundary_positions(mask, valid).tolist()
    length = int(valid.sum().item())
    edges = [0] + [p for p in positions if p > 0] + [length]
    return [
        (edges[i + 1] - edges[i]) * FRAME_SECONDS
        for i in range(len(edges) - 1)
        if edges[i + 1] > edges[i]
    ]


@torch.no_grad()
def _masks_for_task(hparams, batch, device, task_id):
    """Deterministic boundary masks and rates for one task id."""
    wavs, wav_lens = batch.sig
    wavs, wav_lens = wavs.to(device), wav_lens.to(device)
    feats = hparams["ssl"](hparams["normalize"](wavs, wav_lens), wav_lens)
    pad_mask = lengths_to_padding_mask(wav_lens, feats.size(1))
    segmenter = hparams["segmenter"]
    segmenter.active_task_id = torch.full(
        (feats.size(0),), task_id, dtype=torch.long, device=device
    )
    rollout = segmenter.sample_boundaries(feats, pad_mask, deterministic=True)
    valid = ~pad_mask
    segments = (rollout.boundaries == 1).sum(dim=1).float() + 1.0
    frames = valid.sum(dim=1).float().clamp(min=1.0)
    return rollout.boundaries, valid, (segments / frames) * 50.0


def _pairwise_stats(left, right, valid, tolerance):
    """Per-utterance divergence between two boundary mask sets."""
    jaccard, f1 = [], []
    for i in range(left.size(0)):
        jaccard.append(_jaccard(left[i], right[i], valid[i]))
        f1.append(_tolerance_f1(left[i], right[i], valid[i], tolerance))
    return jaccard, f1


def _summary(values):
    if not values:
        return {"n": 0}
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "n": len(values),
        "mean": float(tensor.mean()),
        "std": float(tensor.std()) if tensor.numel() > 1 else 0.0,
        # quantile(0.5), not torch.median: median() returns the lower of the
        # two middle values on an even count, which would be inconsistent with
        # the interpolated p10/p90 reported alongside it.
        "median": float(tensor.quantile(0.5)),
        "p10": float(tensor.quantile(0.10)),
        "p90": float(tensor.quantile(0.90)),
    }


def _film_gradient_cosine(hparams, batches, device):
    """Cosine similarity of per-task GRPO gradients on shared FiLM parameters.

    A strongly negative value means the two tasks pull the shared conditioning
    in opposing directions, which is the signal that would justify testing a
    gradient-projection remedy. Near zero means the tasks are largely
    independent; strongly positive means conditioning is not being used to
    separate them.
    """
    segmenter = hparams["segmenter"]
    params = film_parameters(segmenter)
    if not params:
        return None
    for param in params:
        param.requires_grad_(True)

    task_gradients = {}
    for task_name, batch in batches.items():
        for param in params:
            param.grad = None
        wavs, wav_lens = batch.sig
        wavs, wav_lens = wavs.to(device), wav_lens.to(device)
        with torch.no_grad():
            feats = hparams["ssl"](
                hparams["normalize"](wavs, wav_lens), wav_lens
            )
        pad_mask = lengths_to_padding_mask(wav_lens, feats.size(1))
        segmenter.active_task_id = torch.full(
            (feats.size(0),),
            task_id_of(task_name),
            dtype=torch.long,
            device=device,
        )
        with torch.no_grad():
            rollout = segmenter.sample_boundaries(
                feats, pad_mask, deterministic=False
            )
        # Score the realized actions differentiably; the surrogate below has the
        # same gradient direction as the policy-gradient term for a unit
        # advantage, which is all a cosine comparison needs.
        scored = segmenter.score_boundaries(feats, pad_mask, rollout.boundaries)
        logits = scored.logits if hasattr(scored, "logits") else scored
        actions = rollout.boundaries.clamp(min=0).float()
        valid = (~pad_mask).float()
        surrogate = (
            torch.nn.functional.binary_cross_entropy_with_logits(
                logits, actions, reduction="none"
            )
            * valid
        ).sum() / valid.sum().clamp(min=1.0)
        surrogate.backward()
        task_gradients[task_name] = torch.cat(
            [
                (p.grad if p.grad is not None else torch.zeros_like(p))
                .detach()
                .reshape(-1)
                for p in params
            ]
        )
        for param in params:
            param.grad = None

    names = sorted(task_gradients)
    if len(names) < 2:
        return None
    a, b = task_gradients[names[0]], task_gradients[names[1]]
    denominator = (a.norm() * b.norm()).clamp(min=1e-12)
    return {
        "tasks": names,
        "cosine": float((a @ b) / denominator),
        "norm_" + names[0]: float(a.norm()),
        "norm_" + names[1]: float(b.norm()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hparams", default="hparams/speechllm_multitask.yaml")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="a CKPT* directory from the multi-task run",
    )
    parser.add_argument("--num-utterances", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--tolerance-frames",
        type=int,
        default=1,
        help="match window for tolerance-F1 (1 frame = 20 ms)",
    )
    parser.add_argument("--out", required=True, help="JSON report path")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    with open(args.hparams, encoding="utf-8") as stream:
        hparams = load_hyperpyyaml(stream)

    segmenter = hparams["segmenter"]
    segmenter.load_state_dict(
        torch.load(
            os.path.join(args.checkpoint, "segmenter.ckpt"),
            map_location="cpu",
            weights_only=True,
        )
    )
    hparams["normalize"]._load(
        os.path.join(args.checkpoint, "normalize.ckpt"), end_of_epoch=False
    )
    segmenter.to(device).eval()
    hparams["ssl"].to(device).eval()

    # The ST view supplies the audio; both task ids are then applied to it, so
    # every comparison below is strictly same-audio.
    probe_csv = os.path.join(
        os.path.dirname(os.path.abspath(args.out)), "boundary_probe.csv"
    )
    build_multitask_manifest(
        [
            {
                "csv": os.path.join(
                    hparams["covost_manifest_dir"], "covost_st_test.csv"
                ),
                "source_key": "covost_st",
                "limit": args.num_utterances,
            }
        ],
        probe_csv,
    )
    dataset = multitask_dataio_prepare(
        hparams, hparams["llm"].tokenizer, probe_csv, sorting="ascending"
    )
    loader = sb.dataio.dataloader.make_dataloader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=hparams["test_dataloader_opts"]["collate_fn"],
    )

    observed = {"jaccard": [], "f1": []}
    repeat_null = {"jaccard": [], "f1": []}
    permutation_null = {"jaccard": [], "f1": []}
    rates = {"asr": [], "st_en_de": [], "delta": []}
    durations = {"asr": [], "st_en_de": []}
    first_batches = {}

    for index, batch in enumerate(loader):
        asr_mask, valid, asr_rate = _masks_for_task(
            hparams, batch, device, task_id_of("asr")
        )
        st_mask, _, st_rate = _masks_for_task(
            hparams, batch, device, task_id_of("st_en_de")
        )
        asr_repeat, _, _ = _masks_for_task(
            hparams, batch, device, task_id_of("asr")
        )

        jaccard, f1 = _pairwise_stats(
            asr_mask, st_mask, valid, args.tolerance_frames
        )
        observed["jaccard"].extend(jaccard)
        observed["f1"].extend(f1)

        jaccard, f1 = _pairwise_stats(
            asr_mask, asr_repeat, valid, args.tolerance_frames
        )
        repeat_null["jaccard"].extend(jaccard)
        repeat_null["f1"].extend(f1)

        if st_mask.size(0) > 1:
            # Shift ST masks by one utterance: same masks, wrong pairing. This
            # is how different two masks look for reasons unrelated to the task.
            shifted = torch.roll(st_mask, shifts=1, dims=0)
            jaccard, f1 = _pairwise_stats(
                asr_mask, shifted, valid, args.tolerance_frames
            )
            permutation_null["jaccard"].extend(jaccard)
            permutation_null["f1"].extend(f1)

        rates["asr"].extend(asr_rate.tolist())
        rates["st_en_de"].extend(st_rate.tolist())
        rates["delta"].extend((st_rate - asr_rate).tolist())
        for i in range(asr_mask.size(0)):
            durations["asr"].extend(_segment_durations(asr_mask[i], valid[i]))
            durations["st_en_de"].extend(
                _segment_durations(st_mask[i], valid[i])
            )

        if index == 0:
            first_batches = {"asr": batch, "st_en_de": batch}
        if len(observed["jaccard"]) >= args.num_utterances:
            break

    report = {
        "checkpoint": args.checkpoint,
        "utterances": len(observed["jaccard"]),
        "tolerance_frames": args.tolerance_frames,
        "observed_asr_vs_st": {
            "jaccard": _summary(observed["jaccard"]),
            "tolerance_f1": _summary(observed["f1"]),
        },
        "repeat_null_asr_vs_asr": {
            "jaccard": _summary(repeat_null["jaccard"]),
            "tolerance_f1": _summary(repeat_null["f1"]),
        },
        "permutation_null_shifted_pairing": {
            "jaccard": _summary(permutation_null["jaccard"]),
            "tolerance_f1": _summary(permutation_null["f1"]),
        },
        "f_audio_hz": {
            "asr": _summary(rates["asr"]),
            "st_en_de": _summary(rates["st_en_de"]),
            "st_minus_asr": _summary(rates["delta"]),
        },
        "segment_duration_seconds": {
            "asr": _summary(durations["asr"]),
            "st_en_de": _summary(durations["st_en_de"]),
        },
    }
    if first_batches:
        report["film_gradient_cosine"] = _film_gradient_cosine(
            hparams, first_batches, device
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)

    observed_j = report["observed_asr_vs_st"]["jaccard"]["mean"]
    repeat_j = report["repeat_null_asr_vs_asr"]["jaccard"]["mean"]
    logger.info("Wrote %s", args.out)
    logger.info(
        "ASR-vs-ST boundary Jaccard %.4f; repeat null %.4f; ASR %.2f Hz, "
        "ST %.2f Hz (delta %.2f Hz)",
        observed_j,
        repeat_j,
        report["f_audio_hz"]["asr"]["mean"],
        report["f_audio_hz"]["st_en_de"]["mean"],
        report["f_audio_hz"]["st_minus_asr"]["mean"],
    )
    if observed_j >= repeat_j - 1e-9:
        logger.warning(
            "Task conditioning did NOT move the boundaries beyond the "
            "same-task repeat null: the masks are effectively identical."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
