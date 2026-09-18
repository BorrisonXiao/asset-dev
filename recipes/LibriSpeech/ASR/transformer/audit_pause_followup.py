#!/usr/bin/env python3
"""Reanalyze saved pilot cuts and char-CTC targets; no audio/model inference."""
import json
import statistics

from audit_segment_units import DATA, OUT, SEEDS, gap_stats


def main():
    import torch

    rows = json.loads((OUT / "predictions.json").read_text())["utterances"]
    systems = {f"learned_{seed}": [] for seed in SEEDS}
    systems["char_ctc_targets"] = []
    for row in rows:
        path = DATA / "wavlm_boundaries/char" / f"{row['uid']}.pt"
        target = torch.load(path, map_location="cpu", weights_only=True).flatten()
        assert len(target) == row["frames"]
        cuts = {
            f"learned_{seed}": [i * .02 for i in row["predictions"][str(seed)]]
            for seed in SEEDS
        }
        cuts["char_ctc_targets"] = [
            i * .02 for i in (target == 1).nonzero().flatten().tolist() if i > 0
        ]
        for system, positions in cuts.items():
            edges = [0., *positions, row["duration"]]
            for gap in row["pauses"]:
                a, b, *_ = gap
                stats = gap_stats(positions, row["duration"], gap)
                stats["nearest_onset_signed_ms"] = 1000 * (
                    min(edges, key=lambda t: abs(t - a)) - a
                )
                stats["nearest_offset_signed_ms"] = 1000 * (
                    min(edges, key=lambda t: abs(t - b)) - b
                )
                intervals = list(zip(edges, edges[1:]))
                stats["edge_enclosing_segment"] = {
                    str(ms): any(
                        abs(x - a) <= ms / 1000 + 1e-8
                        and abs(y - b) <= ms / 1000 + 1e-8
                        for x, y in intervals
                    )
                    for ms in (20, 40, 60, 80)
                }
                systems[system].append(
                    dict(uid=row["uid"], cohort=row["cohort"], gap=gap, **stats)
                )
    summary = {}
    for system, gaps in systems.items():
        assert len(gaps) == 31
        summary[system] = dict(
            n=31,
            without_internal_cut=sum(g["internal_cuts"] == 0 for g in gaps),
            with_pure_token=sum(g["pure_tokens"] > 0 for g in gaps),
            with_80pct_silent_token=sum(g["mostly_silent_tokens"] > 0 for g in gaps),
            onset_signed_median_ms=statistics.median(
                g["nearest_onset_signed_ms"] for g in gaps
            ),
            offset_signed_median_ms=statistics.median(
                g["nearest_offset_signed_ms"] for g in gaps
            ),
            isolated_segment_with_edge_tolerance={
                str(ms): sum(g["edge_enclosing_segment"][str(ms)] for g in gaps)
                for ms in (20, 40, 60, 80)
            },
        )
    output = dict(
        note="Same 31 pilot gaps, repeated across seeds. Char targets are supervision, not cold-policy predictions. This comparison cannot attribute changes specifically to RL. Edge-tolerant isolation requires both edges of the SAME segment to match the pause edges.",
        summary=summary,
        gaps=systems,
    )
    (OUT / "pause_followup.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
