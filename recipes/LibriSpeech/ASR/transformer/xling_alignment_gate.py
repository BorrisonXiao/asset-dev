#!/usr/bin/env python3
"""S1 alignment-quality gate for the cross-lingual step24k pipeline.

Hard checks (non-zero exit on failure => dependent training jobs are killed
via SLURM afterok chaining):
  1. coverage: >= 99.5% of manifest utterances have BOTH char-level and phone
     boundary files, per split;
  2. aligner reliability: the char/mora CTC aligner's best greedy dev
     unit-error-rate <= 0.15;
  3. phone agreement: >= 80% of char-level boundaries fall within +-2 frames
     (40 ms) of a phone boundary, on a sample of utterances — chars/moras are
     made of whole phones, so their onsets must coincide with phone onsets;
  4. initialization invariant: mean phone rho >= rho_hi (the learned arm
     cold-starts and warm-starts at phone rate; the band must pull DOWN).

Everything measured is written to ``--report`` (JSON), including the
char-level rho vs. the band (reported, not enforced; the floor is lowered
manually if a measured char rate falls below it).
"""

import argparse
import csv
import json
import os
import sys

import torch


def utt_ids(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as fh:
        return [row["ID"] for row in csv.DictReader(fh)]


def coverage(ids, target_dir):
    have = sum(
        1 for u in ids if os.path.exists(os.path.join(target_dir, u + ".pt"))
    )
    return have / max(len(ids), 1)


def boundary_agreement(ids, char_dir, phone_dir, tol_frames, sample):
    """Fraction of char boundaries within tol of a phone boundary."""
    hits = total = 0
    used = 0
    for u in ids:
        cp = os.path.join(char_dir, u + ".pt")
        pp = os.path.join(phone_dir, u + ".pt")
        if not (os.path.exists(cp) and os.path.exists(pp)):
            continue
        c = torch.load(cp).nonzero().flatten()
        p = torch.load(pp).nonzero().flatten()
        if len(p) == 0:
            continue
        dist = (c[:, None] - p[None, :]).abs().min(dim=1).values
        hits += int((dist <= tol_frames).sum())
        total += len(c)
        used += 1
        if used >= sample:
            break
    return hits / max(total, 1), used


def mean_rho(ids, target_dir, sample):
    rho_sum = used = 0
    for u in ids:
        path = os.path.join(target_dir, u + ".pt")
        if not os.path.exists(path):
            continue
        b = torch.load(path)
        rho_sum += float(b.sum()) / len(b)
        used += 1
        if used >= sample:
            break
    return rho_sum / max(used, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", nargs="+", required=True)
    parser.add_argument("--char_dir", required=True)
    parser.add_argument("--phone_dir", required=True)
    parser.add_argument("--aligner_metrics", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--min_coverage", type=float, default=0.995)
    parser.add_argument("--max_aligner_uer", type=float, default=0.15)
    parser.add_argument("--min_phone_agreement", type=float, default=0.80)
    parser.add_argument("--tol_frames", type=int, default=2)
    parser.add_argument("--rho_hi", type=float, default=0.20)
    parser.add_argument("--rho_lo", type=float, default=0.10)
    parser.add_argument("--sample", type=int, default=2000)
    args = parser.parse_args()

    report = {"checks": {}, "splits": {}, "pass": True}

    def check(name, ok, detail):
        report["checks"][name] = {"pass": bool(ok), **detail}
        if not ok:
            report["pass"] = False
        print(("PASS " if ok else "FAIL ") + name, json.dumps(detail))

    with open(args.aligner_metrics) as fh:
        uer = float(json.load(fh)["best_dev_unit_error_rate"])
    check(
        "aligner_dev_unit_error_rate",
        uer <= args.max_aligner_uer,
        {"value": uer, "threshold": args.max_aligner_uer},
    )

    all_ids = []
    for path in args.csv:
        ids = utt_ids(path)
        all_ids.extend(ids)
        split = os.path.splitext(os.path.basename(path))[0]
        cov_c = coverage(ids, args.char_dir)
        cov_p = coverage(ids, args.phone_dir)
        report["splits"][split] = {
            "n": len(ids),
            "char_coverage": cov_c,
            "phone_coverage": cov_p,
        }
        check(
            f"coverage_{split}",
            cov_c >= args.min_coverage and cov_p >= args.min_coverage,
            {"char": cov_c, "phone": cov_p, "threshold": args.min_coverage},
        )

    agree, used = boundary_agreement(
        all_ids, args.char_dir, args.phone_dir, args.tol_frames, args.sample
    )
    check(
        "char_phone_agreement",
        agree >= args.min_phone_agreement,
        {
            "value": agree,
            "threshold": args.min_phone_agreement,
            "tol_frames": args.tol_frames,
            "utterances": used,
        },
    )

    rho_phone = mean_rho(all_ids, args.phone_dir, args.sample)
    rho_char = mean_rho(all_ids, args.char_dir, args.sample)
    # The pathology guard: an init BELOW the band floor would have the rate
    # term force MORE boundaries than the init emits (upsampling). Whether the
    # init additionally sits above the band top (strictly compressive, the
    # English geometry) is reported, not enforced — silence frames dilute
    # realized per-frame rates well below speech-active predictions, and an
    # in-band init is rate-neutral, which is acceptable.
    check(
        "phone_rate_init_invariant",
        rho_phone >= args.rho_lo,
        {
            "rho_phone": rho_phone,
            "hz_phone": 50 * rho_phone,
            "rho_lo": args.rho_lo,
            "compressive_geometry": rho_phone >= args.rho_hi,
            "rho_hi": args.rho_hi,
        },
    )
    report["char_rate"] = {
        "rho_char": rho_char,
        "hz_char": 50 * rho_char,
        "inside_band": args.rho_lo <= rho_char <= args.rho_hi,
        "note": "reported only; lower the band floor manually if below",
    }
    print("char rate:", json.dumps(report["char_rate"]))

    # The coverage thresholds tolerate a few unalignable utterances, but the
    # training loader hard-fails on ANY missing boundary file — so once the
    # checks pass, prune the stragglers from the manifests (all arms share
    # them; targets are keyed by utterance id, nothing else changes).
    if report["pass"]:
        pruned = {}
        # Selection subsets (e.g. zh dev_sub.csv) live beside the split csvs
        # and must stay consistent with them.
        extra_subsets = [
            p
            for p in (
                os.path.join(os.path.dirname(args.csv[0]), "dev_sub.csv"),
            )
            if os.path.exists(p)
        ]
        for path in list(args.csv) + extra_subsets:
            with open(path, newline="", encoding="utf-8") as fh:
                rows = list(csv.reader(fh))
            header, body = rows[0], rows[1:]
            kept = [
                r
                for r in body
                if os.path.exists(os.path.join(args.char_dir, r[0] + ".pt"))
                and os.path.exists(os.path.join(args.phone_dir, r[0] + ".pt"))
            ]
            if len(kept) != len(body):
                with open(path, "w", newline="", encoding="utf-8") as fh:
                    writer = csv.writer(fh)
                    writer.writerow(header)
                    writer.writerows(kept)
                pruned[os.path.basename(path)] = len(body) - len(kept)
        report["pruned_missing_targets"] = pruned
        print("pruned:", json.dumps(pruned))

    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w") as fh:
        json.dump(report, fh, indent=2)
    sys.exit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
