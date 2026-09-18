#!/usr/bin/env python3
"""Validate and optionally submit the three final-checkpoint test evaluations."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import yaml

RECIPE = Path(__file__).resolve().parent
REPO = RECIPE.parents[3]
ARMS = ("adaptive", "original", "fixed_lower")
MANIFESTS = Path("/export/jsalt26/omnienc/users/cxiao/datasets/librispeech_manifests")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, default=RECIPE / "results/speechllm_ls960_adaptive_frequency_20260918")
    parser.add_argument("--output", type=Path, default=REPO / "artifacts/segmenter/ls960_adaptive_frequency_test_eval_20260918")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    training = args.training_root.resolve(strict=True)
    output = args.output.resolve()
    source = training / "source"
    hashes = json.loads((source / "source_manifest.json").read_text())
    for name, expected in hashes.items():
        if sha256(source / name) != expected:
            raise ValueError(f"Frozen source changed: {name}")
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "source": str(source), "verified_source_files": len(hashes),
                "source_manifest_sha256": sha256(source / "source_manifest.json"),
                "continuation_updates": 8000, "source_updates": 24000,
                "selection": "final matched-budget checkpoint; not rate-first selection",
                "seed": 3407, "splits": {}, "arms": {}}
    for split, expected in (("test-clean", 2620), ("test-other", 2939)):
        path = MANIFESTS / f"{split}.csv"
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != expected or len({row["ID"] for row in rows}) != expected:
            raise ValueError(f"Incorrect test manifest coverage: {split}")
        manifest["splits"][split] = {"path": str(path), "utterances": expected,
                                     "sha256": sha256(path)}
    for arm in ARMS:
        checkpoint = training / arm / "save/CKPT+continuation-step-008000"
        metadata = yaml.safe_load((checkpoint / "CKPT.yaml").read_text())
        if metadata["optimizer_step"] != 8000 or not metadata["frequency"]["training_done"]:
            raise ValueError(f"Not a completed final checkpoint: {arm}")
        for name in ("segmenter", "proj", "llm", "normalize", "model_optimizer", "brain", "counter"):
            if not (checkpoint / f"{name}.ckpt").is_file():
                raise FileNotFoundError(checkpoint / f"{name}.ckpt")
        manifest["arms"][arm] = {"checkpoint": str(checkpoint),
                                  "checkpoint_sha256": {p.name: sha256(p) for p in sorted(checkpoint.iterdir()) if p.is_file()}}
    print(json.dumps({"preflight": "passed", "source_files": len(hashes),
                      "arms": list(ARMS), "test_counts": [2620, 2939],
                      "output": str(output)}, indent=2), flush=True)
    if not args.submit:
        return
    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    runner_dir = output / "runner"
    runner_dir.mkdir()
    for name in ("eval_adaptive_frequency_ls960.py", "eval_adaptive_frequency_ls960.slurm"):
        shutil.copy2(RECIPE / name, runner_dir / name)
    manifest["runner_sha256"] = {p.name: sha256(p) for p in runner_dir.iterdir()}
    for arm in ARMS:
        arm_output = output / arm
        view = arm_output / "checkpoint_view"
        view.mkdir(parents=True)
        (view / "CKPT+continuation-step-008000").symlink_to(manifest["arms"][arm]["checkpoint"], target_is_directory=True)
    manifest_path = output / "evaluation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    for arm in ARMS:
        command = ["sbatch", "--parsable", "--account=highprio", "--partition=gpu-a100",
                   "--exclude=e01", "--nodes=1", "--ntasks=1", "--gres=gpu:1",
                   "--cpus-per-task=8", "--mem=60G", "--time=02:00:00",
                   f"--job-name=ls960_adapt_test_{arm}",
                   f"--chdir={source / 'recipes/LibriSpeech/ASR/transformer'}",
                   f"--output={output / 'logs' / (arm + '-%j.out')}",
                   f"--error={output / 'logs' / (arm + '-%j.err')}",
                   str(runner_dir / "eval_adaptive_frequency_ls960.slurm"),
                   str(training), str(output / arm), str(runner_dir / "eval_adaptive_frequency_ls960.py")]
        job = subprocess.check_output(command, text=True).strip().split(";")[0]
        manifest["arms"][arm].update(job_id=job, submit_command=command)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"{arm}: submitted {job}", flush=True)


if __name__ == "__main__":
    main()
