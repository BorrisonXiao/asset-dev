#!/usr/bin/env python3
"""Drop eval utterances whose oracle units fall outside the train inventory.

The char/mora CTC aligner can only align units it was trained on. Train
splits contain no OOV units by construction; dev/test splits can (rare hanzi
in AISHELL dev/test, one expressive mora in JSUT). Such utterances get no
oracle boundary file, which would crash the oracle arms, so they are removed
from the manifests — every arm shares the same filtered eval sets, keeping
comparisons internally fair. Counts are printed and must be reported.

Usage:
    python xling_filter_oov_evals.py --manifest_dir m/ --level char \
        --splits dev test [--resample_sub dev:dev_sub.csv:7]
"""

import argparse
import csv
import os


def load_units(path):
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            utt, _, units = line.rstrip("\n").partition("\t")
            out[utt] = units.split()
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest_dir", required=True)
    parser.add_argument("--level", required=True)
    parser.add_argument("--splits", nargs="+", required=True)
    parser.add_argument(
        "--resample_sub",
        default=None,
        help="src_split:sub_name.csv:stride — rebuild a selection subset "
        "from the filtered source split",
    )
    args = parser.parse_args()

    inventory = set()
    train_units = os.path.join(
        args.manifest_dir, f"train_units_{args.level}.tsv"
    )
    for units in load_units(train_units).values():
        inventory.update(units)
    print(f"train inventory: {len(inventory)} units")

    for split in args.splits:
        csv_path = os.path.join(args.manifest_dir, f"{split}.csv")
        units_path = os.path.join(
            args.manifest_dir, f"{split}_units_{args.level}.tsv"
        )
        units_by_utt = load_units(units_path)
        keep = {
            u
            for u, units in units_by_utt.items()
            if all(x in inventory for x in units)
        }
        with open(csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            rows = list(reader)
        header, body = rows[0], rows[1:]
        kept_rows = [r for r in body if r[0] in keep]
        dropped = len(body) - len(kept_rows)
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            writer.writerows(kept_rows)
        with open(units_path, "w", encoding="utf-8") as fh:
            for r in kept_rows:
                fh.write(f"{r[0]}\t{' '.join(units_by_utt[r[0]])}\n")
        print(f"{split}: kept {len(kept_rows)}, dropped {dropped} OOV-unit utts")

    if args.resample_sub:
        src, sub_name, stride = args.resample_sub.split(":")
        src_path = os.path.join(args.manifest_dir, f"{src}.csv")
        sub_path = os.path.join(args.manifest_dir, sub_name)
        with open(src_path, encoding="utf-8") as fh:
            lines = fh.readlines()
        sub = [lines[0]] + lines[1 :: int(stride)]
        with open(sub_path, "w", encoding="utf-8") as fh:
            fh.writelines(sub)
        print(f"{sub_name}: {len(sub) - 1} utts (stride {stride} of {src})")


if __name__ == "__main__":
    main()
