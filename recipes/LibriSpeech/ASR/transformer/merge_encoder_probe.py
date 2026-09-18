#!/usr/bin/env python3
"""Merge the per-domain frozen-encoder probe reports into one 2x2 and read it.

Each probe job trains one domain and evaluates both, so each contributes one
row. This joins them and prints the comparison that decides whether the frozen
encoder is the CoVoST bottleneck.
"""

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    grid, meta = {}, {}
    for path in args.reports:
        with open(path, encoding="utf-8") as stream:
            report = json.load(stream)
        grid.update(report["grid"])
        meta.setdefault("wavlm", report["wavlm"])
        meta.setdefault("steps", report["steps"])
        meta.setdefault("train_hours_requested", report["train_hours_requested"])
        meta.setdefault("corpora", {}).update(report.get("corpora", {}))

    required = [
        f"train_{t}__eval_{e}"
        for t in ("librispeech", "covost")
        for e in ("librispeech", "covost")
    ]
    missing = [k for k in required if k not in grid]
    if missing:
        print(f"Incomplete grid, missing: {missing}", file=sys.stderr)
        return 1

    print(f"Frozen encoder: {meta['wavlm']}")
    print(
        f"Matched budget: {meta['train_hours_requested']} h per corpus, "
        f"{meta['steps']} steps per probe"
    )
    for name, info in meta["corpora"].items():
        print(
            f"  {name}: {info['train_utts']} train utts / "
            f"{info['train_hours']:.1f} h"
        )

    print(f"\n{'':16s}{'eval LibriSpeech':>20s}{'eval CoVoST':>16s}")
    for train in ("librispeech", "covost"):
        cells = []
        for ev in ("librispeech", "covost"):
            s = grid[f"train_{train}__eval_{ev}"]
            cells.append(f"{s['CER']:.2f} / {s['WER']:.2f}")
        print(f"train {train:10s}{cells[0]:>20s}{cells[1]:>16s}")
    print("(each cell: CER / WER)")

    in_ls = grid["train_librispeech__eval_librispeech"]
    in_cv = grid["train_covost__eval_covost"]
    gap_cer = in_cv["CER"] - in_ls["CER"]
    gap_wer = in_cv["WER"] - in_ls["WER"]
    ratio = in_cv["CER"] / in_ls["CER"] if in_ls["CER"] else float("inf")

    print(
        f"\nDECISIVE (both in-domain, matched hours):"
        f"\n  LibriSpeech CER {in_ls['CER']:.2f}  WER {in_ls['WER']:.2f}"
        f"\n  CoVoST      CER {in_cv['CER']:.2f}  WER {in_cv['WER']:.2f}"
        f"\n  gap {gap_cer:+.2f} CER ({ratio:.2f}x), {gap_wer:+.2f} WER"
    )
    # The full system is ~2.7x worse on CoVoST (5.68 -> 15.22 WER). If the
    # frozen features alone reproduce a comparable ratio, the encoder carries
    # the failure; if the probe is near domain-flat, it does not.
    if ratio >= 1.8:
        verdict = (
            "Frozen WavLM features are substantially worse for Common Voice. "
            "The encoder is implicated; adapting it (unfreezing upper layers "
            "or substituting a broader-domain encoder) is the lever."
        )
    elif ratio <= 1.25:
        verdict = (
            "Frozen WavLM features are close to domain flat. The encoder is "
            "largely exonerated, so the failure sits downstream -- boundary "
            "placement and adapter capacity become the live hypotheses."
        )
    else:
        verdict = (
            "Partial effect: the encoder contributes but does not by itself "
            "account for the full-system gap. Expect a second cause downstream."
        )
    print(f"\nReading: {verdict}")

    merged = {**meta, "grid": grid,
              "in_domain_gap": {"CER": gap_cer, "WER": gap_wer, "ratio": ratio},
              "verdict": verdict}
    with open(args.out, "w", encoding="utf-8") as stream:
        json.dump(merged, stream, indent=2)
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
