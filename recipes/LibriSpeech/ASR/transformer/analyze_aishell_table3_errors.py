#!/usr/bin/env python3
"""Audit AISHELL-1 predictions without loading a model.

python analyze_aishell_table3_errors.py --runs-root PATH/speechllm_xling_step24k

Use --replication-root for the new auxiliary-loss runs and their JSONL outputs,
or --runs-root for the historical SpeechBrain word-alignment reports.
Both inputs require all four arms and all three seeds.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics

SEEDS = (3407, 3408, 3409)
ARMS = {
    "fixed": "fixed_k5",
    "phone": "phone_oracle",
    "character": "char_oracle",
    "asset": "transformer_ar_local64_bigru",
}
PUBLISHED_SEEDS = {
    "fixed": [9.31, 9.26, 8.94],
    "phone": [12.80, 13.06, 12.34],
    "character": [8.57, 8.70, 8.61],
    "asset": [7.45, 7.57, 7.57],
}
RECIPE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = RECIPE.parents[3] / "artifacts/segmenter/aishell_table3_error_analysis"
UTT_HEADER = re.compile(
    r"^([^,\n]+), %WER ([0-9.]+) \[ (\d+) / (\d+), "
    r"(\d+) ins, (\d+) del, (\d+) sub \]$", re.M
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_predictions(path):
    """Keep token boundaries so both historical and plain CER can be checked."""
    text = path.read_text(encoding="utf-8")
    rows = {}
    for block in re.split(r"^={10,}\s*$", text, flags=re.M):
        lines = block.strip("\n").splitlines()
        if not lines or not UTT_HEADER.fullmatch(lines[0]):
            continue
        if len(lines) != 4:
            raise ValueError(f"Malformed alignment block in {path}: {lines[0]}")
        match = UTT_HEADER.fullmatch(lines[0])
        uid, _, errors, tokens, ins, dele, sub = match.groups()
        if uid in rows:
            raise ValueError(f"Duplicate ID {uid} in {path}")
        errors, tokens, ins, dele, sub = map(int, (errors, tokens, ins, dele, sub))
        if errors != ins + dele + sub:
            raise ValueError(f"Inconsistent saved word counts: {path}, {uid}")
        def tokens_from(line):
            return [x.strip() for x in line.split(" ; ") if x.strip() != "<eps>"]
        rows[uid] = {
            "ref_words": tokens_from(lines[1]),
            "hyp_words": tokens_from(lines[3]),
        }
    if len(rows) != len(UTT_HEADER.findall(text)) or not rows:
        raise ValueError(f"Incomplete/unrecognized prediction report: {path}")
    return rows


def align(ref, hyp):
    """Unit-cost Levenshtein; diagonal, deletion, insertion priority on ties."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1,
                           dp[i-1][j-1] + (ref[i-1] != hyp[j-1]))
    ops = []
    i, j = n, m
    while i or j:
        if i and j and dp[i][j] == dp[i-1][j-1] + (ref[i-1] != hyp[j-1]):
            ops.append(("eq" if ref[i-1] == hyp[j-1] else "sub", i-1, ref[i-1], hyp[j-1]))
            i, j = i-1, j-1
        elif i and dp[i][j] == dp[i-1][j] + 1:
            ops.append(("del", i-1, ref[i-1], ""))
            i -= 1
        else:
            ops.append(("ins", i, "", hyp[j-1]))
            j -= 1
    ops.reverse()
    return ops


def score_run(rows, joiner):
    per_utt = {}
    confusion = {k: Counter() for k in ("sub", "del", "ins")}
    for uid, row in rows.items():
        ref = joiner.join(row["ref_words"])
        hyp = joiner.join(row["hyp_words"])
        if not ref:
            raise ValueError(f"Empty reference for {uid}")
        ops = align(ref, hyp)
        counts = Counter(op[0] for op in ops if op[0] != "eq")
        for kind, _, r, h in ops:
            if kind != "eq":
                confusion[kind][(r, h)] += 1
        per_utt[uid] = dict(ref=ref, hyp=hyp, n=len(ref), errors=sum(counts.values()),
                            sub=counts["sub"], dele=counts["del"], ins=counts["ins"], ops=ops)
    n = sum(r["n"] for r in per_utt.values())
    totals = {k: sum(r[k] for r in per_utt.values()) for k in ("errors", "sub", "dele", "ins")}
    result = {"utterances": len(rows), "ref_characters": n, **totals}
    for k in ("errors", "sub", "dele", "ins"):
        result["cer" if k == "errors" else k + "_rate"] = 100 * totals[k] / n
    result["exact_match_pct"] = 100 * sum(r["errors"] == 0 for r in per_utt.values()) / len(rows)
    result["over_50pct_cer_pct"] = 100 * sum(r["errors"] / r["n"] > .5 for r in per_utt.values()) / len(rows)
    result["empty_hypotheses"] = sum(not r["hyp"] for r in per_utt.values())
    result["top_confusions"] = {
        kind: [{"reference": r, "hypothesis": h, "count": c} for (r, h), c in counter.most_common(20)]
        for kind, counter in confusion.items()
    }
    return result, per_utt


def paired_comparison(arms, baseline):
    """Match by seed and utterance; count improved, regressed and tied cases."""
    outcomes = []
    for seed in SEEDS:
        a, b = arms["asset"][seed], arms[baseline][seed]
        counts = Counter()
        error_delta = Counter()
        for uid in sorted(a):
            learned, base = a[uid], b[uid]
            delta = base["errors"] - learned["errors"]
            counts["improved" if delta > 0 else "regressed" if delta < 0 else "tied"] += 1
            counts["both_exact" if not base["errors"] and not learned["errors"] else
                   "asset_only_exact" if not learned["errors"] else
                   "baseline_only_exact" if not base["errors"] else "both_have_errors"] += 1
            error_delta["errors_removed"] += max(0, delta)
            error_delta["errors_added"] += max(0, -delta)
        outcomes.append({"seed": seed, **counts, **error_delta,
                         "net_errors_removed": sum(b[u]["errors"] - a[u]["errors"] for u in a)})
    return outcomes


def collect(root, output):
    paths = {(arm, seed): root / "zh" / directory / str(seed) / "wer_results/wer_test.txt"
             for arm, directory in ARMS.items() for seed in SEEDS}
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("All 12 Table 3 prediction reports are required. Missing:\n" + "\n".join(missing))
    raw = {key: parse_predictions(path) for key, path in paths.items()}
    canonical = raw[("fixed", SEEDS[0])]
    if len(canonical) != 7176:
        raise ValueError(f"Expected AISHELL-1 test's 7176 utterances; found {len(canonical)}")
    analyze(raw, output, paths, "historical", {})


def collect_replication(root, output):
    collected = json.loads((root / "replication_results.json").read_text())
    manifest = root / "runs/zh/manifests/test.csv"
    if collected["rate_channel"] != "aux" or sha256(manifest) != collected["test_manifest_sha256"]:
        raise ValueError("Replication channel or test manifest mismatch")
    with manifest.open() as handle:
        references = {}
        for row in csv.DictReader(handle):
            if row["ID"] in references:
                raise ValueError(f"Duplicate test manifest ID: {row['ID']}")
            references[row["ID"]] = row["wrd"].split()
    if len(references) != collected["test_utterances"]:
        raise ValueError("Collected test count disagrees with the manifest")
    records = {(row["arm"], row["seed"]): row for row in collected["runs"]}
    expected = {(directory, seed) for directory in ARMS.values() for seed in SEEDS}
    if set(records) != expected or len(collected["runs"]) != len(expected):
        raise ValueError("All 12 unique replication runs are required")
    paths, raw, saved = {}, {}, {}
    for arm, directory in ARMS.items():
        for seed in SEEDS:
            key = arm, seed
            path = root / "runs/zh" / directory / str(seed) / "predictions/test.jsonl"
            evidence = records[directory, seed]
            if sha256(path) != evidence["prediction_sha256"] or evidence["rate_channel"] != "aux":
                raise ValueError(f"Prediction checksum or channel mismatch: {path}")
            rows = {}
            with path.open() as handle:
                for line in handle:
                    row = json.loads(line)
                    uid = row["id"]
                    if (uid in rows or references.get(uid) != row["reference_words"]
                            or (row["split"], row["seed"], row["rate_channel"]) != ("test", seed, "aux")):
                        raise ValueError(f"Prediction identity, reference, or metadata mismatch: {path}, {uid}")
                    rows[uid] = {"ref_words": row["reference_words"], "hyp_words": row["hypothesis_words"]}
            if rows.keys() != references.keys() or len(rows) != evidence["utterances"]:
                raise ValueError(f"Test coverage mismatch: {path}")
            raw[key], paths[key], saved[key] = rows, path, evidence
    analyze(raw, output, paths, "aux", saved, test_manifest_sha256=sha256(manifest))


def analyze(raw, output, paths, channel, saved, **metadata):
    canonical = raw[("fixed", SEEDS[0])]
    for key, rows in raw.items():
        if rows.keys() != canonical.keys():
            raise ValueError(f"Utterance coverage differs for {key}")
        if any(rows[u]["ref_words"] != canonical[u]["ref_words"] for u in rows):
            raise ValueError(f"References differ for {key}")
    result = {"status": "complete", "dataset": "AISHELL-1 test", "seeds": list(SEEDS),
              "rate_channel": channel, "test_utterances": len(canonical), **metadata,
              "alignment_tie_order": ["diagonal", "deletion", "insertion"],
              "source_files": [{"arm": a, "seed": s, "path": str(p.relative_to(p.parents[5]) if saved else p), "sha256": sha256(p)}
                               for (a, s), p in paths.items()], "conventions": {}}
    all_per_utt = {}
    for name, joiner in (("characters_without_spaces", ""), ("archived_recipe_word_boundaries", "_")):
        stats, scored = {}, {a: {} for a in ARMS}
        for (arm, seed), rows in raw.items():
            summary, details = score_run(rows, joiner)
            if saved and joiner:
                evidence = saved[arm, seed]
                if (summary["errors"], summary["ref_characters"]) != (evidence["errors"], evidence["ref_tokens"]):
                    raise ValueError(f"Independent CER disagrees with the saved scorer: {arm} {seed}")
                summary["saved_scorer_counts"] = {
                    key: evidence[key] for key in ("insertions", "deletions", "substitutions")
                }
            stats.setdefault(arm, []).append({"seed": seed, **summary})
            scored[arm][seed] = details
        comparison = {a: paired_comparison(scored, a) for a in ARMS if a != "asset"}
        matched = all(f"{r['cer']:.2f}" == f"{PUBLISHED_SEEDS[a][i]:.2f}"
                      for a, rows in stats.items() for i, r in enumerate(rows))
        result["conventions"][name] = {"per_seed": stats, "matches_all_published_seed_cers": matched,
                                          "paired_against_asset": comparison}
        if saved and joiner:
            result["conventions"][name]["matches_collected_error_totals"] = True
        all_per_utt[name] = scored
    # Use plain characters for linguistic diagnostics; never relabel this as the
    # published metric unless the reproduction check above succeeds.
    scored = all_per_utt["characters_without_spaces"]
    candidates = []
    seed = SEEDS[0]
    for uid in sorted(canonical):
        a = scored["asset"][seed][uid]
        if not any(scored[arm][seed][uid]["errors"] for arm in ARMS):
            continue
        case = {"id": uid, "seed": seed, "reference": a["ref"],
                "predictions": {arm: scored[arm][seed][uid] for arm in ARMS}}
        candidates.append(case)
    output.mkdir(parents=True, exist_ok=True)
    (output / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (output / "utterance_candidates.json").write_text(json.dumps(candidates, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "utterances_per_run": len(canonical),
                      "matched_conventions": [k for k, v in result["conventions"].items()
                                              if v["matches_all_published_seed_cers"]]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--runs-root", type=Path)
    inputs.add_argument("--replication-root", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.replication_root:
        collect_replication(args.replication_root, args.output)
    else:
        collect(args.runs_root or RECIPE / "results/speechllm_xling_step24k", args.output)
