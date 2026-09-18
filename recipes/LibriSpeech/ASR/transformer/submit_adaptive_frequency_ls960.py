"""Freeze sources and submit one smoke job or the three full-LS960 pilot arms.

Dry run unless --submit is supplied. Production requires a successful smoke job.
Each submission gets an immutable source directory; never reads changing recipe
code once queued. Existing output directories are rejected (no silent overwrite).
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess

import yaml

RECIPE = Path(__file__).resolve().parent
REPO = RECIPE.parents[3]
MANIFESTS = Path("/export/jsalt26/omnienc/users/cxiao/datasets/librispeech_manifests")


def preflight():
    counts = {"train-clean-100": 28539, "train-clean-360": 104014, "train-other-500": 148688,
              "train": 281241, "dev-clean": 2703, "dev-other": 2864}
    for split, expected in counts.items():
        with (MANIFESTS / f"{split}.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != expected:
            raise ValueError(f"{split}: {len(rows)} rows, expected {expected}")
        if split == "train":
            found = {next((s for s in counts if s.startswith("train-") and s in row["wav"]), "missing") for row in rows}
            if found != {"train-clean-100", "train-clean-360", "train-other-500"}:
                raise ValueError(f"Merged manifest does not contain exactly LS960: {found}")
        for row in (rows[0], rows[-1]):
            audio = Path(row["wav"].replace("$data_root", "/export/jsalt26/omnienc/users/cxiao/datasets/LibriSpeech"))
            if not audio.is_file():
                raise FileNotFoundError(audio)
    cfg = yaml.safe_load((RECIPE / "hparams/adaptive_frequency_ls960.yaml").read_text())
    source = Path(cfg["source_checkpoint"])
    for name in ("segmenter", "llm", "proj", "normalize", "model_optimizer"):
        if not (source / f"{name}.ckpt").is_file():
            raise FileNotFoundError(source / f"{name}.ckpt")
    meta = yaml.safe_load((source / "CKPT.yaml").read_text())
    if meta["optimizer_step"] != 24000:
        raise ValueError("Expected selected seed-3407 LS960 step-24000 checkpoint")
    print("PREFLIGHT: full LS960 manifests, sampled audio paths, matched step-24000 checkpoint OK")


def snapshot(destination):
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copytree(REPO / "speechbrain", destination / "speechbrain", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    recipe_copy = destination / "recipes/LibriSpeech/ASR/transformer"
    recipe_copy.mkdir(parents=True)
    for path in RECIPE.glob("*.py"):
        shutil.copy2(path, recipe_copy / path.name, follow_symlinks=True)
    shutil.copy2(RECIPE / "run_adaptive_frequency_ls960.sh", recipe_copy)
    (recipe_copy / "hparams").mkdir()
    for name in ("speechllm_segmenter.yaml", "adaptive_frequency_ls960.yaml", "adaptive_frequency_ls960_smoke.yaml"):
        shutil.copy2(RECIPE / "hparams" / name, recipe_copy / "hparams" / name)
    hashes = {str(path.relative_to(destination)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(destination.rglob("*")) if path.is_file()}
    (destination / "source_manifest.json").write_text(json.dumps(hashes, indent=2) + "\n")
    (destination / "base_commit.txt").write_text(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True))
    return recipe_copy / "run_adaptive_frequency_ls960.sh"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "production"), required=True)
    parser.add_argument("--smoke-output", type=Path)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    root = args.output_root.resolve()
    if "ls960" not in str(root) or not str(root).startswith("/export/jsalt26/"):
        raise ValueError("Output root must be LS960 on /export/jsalt26")
    if root.exists():
        raise FileExistsError(f"Refusing existing submission output root: {root}")
    preflight()
    if args.mode == "production":
        if args.smoke_output is None:
            raise ValueError("Production requires --smoke-output from the passed integration job")
        subprocess.run(["/export/jsalt26/omnienc/users/cxiao/envs/jointllm/bin/python",
                        str(RECIPE / "verify_adaptive_frequency_smoke.py"), str(args.smoke_output)], check=True)
        # Require canary sources to match the files about to be submitted.
        frozen = args.smoke_output.parent / "source"
        manifest = json.loads((frozen / "source_manifest.json").read_text())
        for relative, sha in manifest.items():
            current = REPO / relative
            if current.exists() and hashlib.sha256(current.read_bytes()).hexdigest() != sha:
                raise ValueError(f"Source changed since successful smoke: {relative}; rerun smoke")
    runner = root / "source/recipes/LibriSpeech/ASR/transformer/run_adaptive_frequency_ls960.sh"
    if args.submit:
        root.mkdir(parents=True)
        (root / "logs").mkdir()
        runner = snapshot(root / "source")
    arms = ["adaptive"] if args.mode == "smoke" else ["adaptive", "original", "fixed_lower"]
    print("Training splits: train-clean-100, train-clean-360, train-other-500 (960h)")
    print(f"Output: {root}; estimated total <=20 GB; each job: 1 A100 80GB, 12 CPU, 64 GB RAM")
    records = []
    for arm in arms:
        output = root / arm
        command = ["sbatch", "--parsable", "--account=highprio", "--partition=gpu-a100",
                   "--exclude=e01", "--gpus=1", "--cpus-per-task=12", "--mem=64G",
                   "--time=02:00:00" if args.mode == "smoke" else "--time=2-00:00:00",
                   f"--job-name=ls960_freq_{args.mode}_{arm}", f"--output={root}/logs/%x-%j.out",
                   f"--chdir={runner.parent}", str(runner), arm, str(output), args.mode]
        print(shlex.join(command))
        if args.submit:
            job = subprocess.check_output(command, text=True).strip().split(";")[0]
            if not job.isdigit():
                raise RuntimeError(f"Unexpected Slurm job id: {job}")
            records.append(dict(arm=arm, seed=3407, job_id=job, output=str(output), mode=args.mode))
            (root / "submitted_jobs.json").write_text(json.dumps(records, indent=2) + "\n")
            print(f"QUEUED {arm}: {job}")


if __name__ == "__main__":
    main()
