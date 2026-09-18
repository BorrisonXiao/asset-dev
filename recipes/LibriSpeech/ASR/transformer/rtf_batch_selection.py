#!/usr/bin/env python3
"""Helpers for reproducible, memory-limited RTF batch selection.

The selector uses only development-set duration and GPU memory.  It never reads
RTF when choosing a batch size.  Final test measurements are produced by the
ordinary corrected batch-invariant evaluation recipes.
"""

import argparse
import csv
import json
from pathlib import Path


PROTOCOL = "batch_invariant_left_packed_duration_cap_v1"


def prepare(args):
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    if output.exists() or output.with_suffix(output.suffix + ".meta.json").exists():
        raise FileExistsError(f"Refusing to overwrite calibration output: {output}")
    with source.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "duration" not in reader.fieldnames:
            raise ValueError(f"CSV has no duration column: {source}")
        rows = list(reader)
        fieldnames = list(reader.fieldnames)
    if args.num_utterances < 1 or args.num_utterances > len(rows):
        raise ValueError(
            f"Requested {args.num_utterances} rows from a {len(rows)}-row manifest"
        )
    selected = sorted(
        rows,
        key=lambda row: (-float(row["duration"]), str(row.get("ID", ""))),
    )[: args.num_utterances]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)
    durations = [float(row["duration"]) for row in selected]
    metadata = {
        "source": str(source),
        "output": str(output),
        "selection": "longest_duration_first",
        "utterances": len(selected),
        "duration_seconds_min": min(durations),
        "duration_seconds_max": max(durations),
        "duration_seconds_total": sum(durations),
    }
    output.with_suffix(output.suffix + ".meta.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, sort_keys=True))


def inspect(args):
    benchmark = Path(args.benchmark).resolve()
    rows = []
    with benchmark.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows = [row for row in rows if row.get("decoding_protocol") == PROTOCOL]
    if len(rows) != 1:
        raise ValueError(
            f"Expected one corrected calibration record in {benchmark}; found {len(rows)}"
        )
    row = rows[0]
    if int(row["batch_size"]) != args.expected_batch:
        raise ValueError(
            f"Expected batch {args.expected_batch}, observed {row['batch_size']}"
        )
    capacity = float(row["gpu_capacity_gb"])
    peak_allocated = float(row["gpu_peak_allocated_gb"])
    fraction = peak_allocated / capacity
    passed = fraction <= args.max_memory_fraction
    summary = {
        "batch_size": args.expected_batch,
        "decoding_protocol": PROTOCOL,
        "gpu_capacity_gb": capacity,
        "gpu_peak_allocated_gb": peak_allocated,
        "memory_fraction": fraction,
        "max_memory_fraction": args.max_memory_fraction,
        "measured_utterances": int(row["measured_utterances"]),
        "passed_memory_rule": passed,
    }
    destination = Path(args.output).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite candidate summary: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0 if passed else 2


def compare(args):
    def load(path):
        records = {}
        with Path(path).open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                records[str(row["id"])] = row
        return records

    batch1 = load(args.batch1)
    selected = load(args.selected)
    common = sorted(set(batch1) & set(selected))
    missing_batch1 = sorted(set(selected) - set(batch1))
    missing_selected = sorted(set(batch1) - set(selected))
    hypothesis_mismatches = [
        utterance_id for utterance_id in common
        if batch1[utterance_id]["hypothesis"]
        != selected[utterance_id]["hypothesis"]
    ]
    summary = {
        "batch1_records": len(batch1),
        "selected_records": len(selected),
        "common_records": len(common),
        "missing_from_batch1": len(missing_batch1),
        "missing_from_selected": len(missing_selected),
        "hypothesis_mismatches": len(hypothesis_mismatches),
        "mismatch_examples": hypothesis_mismatches[:10],
        "identical": not (
            missing_batch1 or missing_selected or hypothesis_mismatches
        ),
    }
    destination = Path(args.output).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite comparison: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["identical"] or not args.require_identical else 2


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--source", required=True)
    prepare_parser.add_argument("--output", required=True)
    prepare_parser.add_argument("--num-utterances", type=int, required=True)
    prepare_parser.set_defaults(func=prepare)

    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("--benchmark", required=True)
    inspect_parser.add_argument("--expected-batch", type=int, required=True)
    inspect_parser.add_argument("--max-memory-fraction", type=float, default=0.90)
    inspect_parser.add_argument("--output", required=True)
    inspect_parser.set_defaults(func=inspect)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--batch1", required=True)
    compare_parser.add_argument("--selected", required=True)
    compare_parser.add_argument("--output", required=True)
    compare_parser.add_argument("--require-identical", action="store_true")
    compare_parser.set_defaults(func=compare)

    args = parser.parse_args()
    result = args.func(args)
    raise SystemExit(0 if result is None else result)


if __name__ == "__main__":
    main()
