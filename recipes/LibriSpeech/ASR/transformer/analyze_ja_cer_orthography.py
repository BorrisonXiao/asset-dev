#!/usr/bin/env python3
"""Decompose the ja learned arm's CER into orthography vs recognition.

Japanese CER scores the written form, which bundles TWO tasks: recognizing
the sounds and choosing the orthography (kana vs kanji spellings; homophone
kanji selection — an IME-style conversion problem). Re-scoring both
hypothesis and reference in katakana space (fugashi + unidic-lite readings)
removes the orthography layer, leaving a phonetic error rate.

Reads the per-utterance alignments the trainer already wrote
(wer_results/wer_{test,jsut_test}.txt) for every seed of the learned arm and
writes artifacts/segmenter/ja_cer_orthography.json, which the training
report's multilingual section consumes — values in the report therefore
cannot drift from the artifacts.
"""

import json
import os
import re
import statistics
import unicodedata

import fugashi

RUN_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results",
    "speechllm_xling_step24k",
    "ja",
    "transformer_ar_local64_bigru",
)
OUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "artifacts",
    "segmenter",
    "ja_cer_orthography.json",
)
SEEDS = (3407, 3408, 3409)
SPLITS = ("test", "jsut_test")

_KATA = re.compile(r"^[ァ-ヴー]+$")
_H2K = {chr(c): chr(c + 0x60) for c in range(0x3041, 0x3097)}

tagger = fugashi.Tagger()


def to_kana(text):
    out = []
    for w in tagger(text):
        kana = w.feature.kana or w.feature.pron
        if not kana:
            s = "".join(_H2K.get(c, c) for c in w.surface)
            kana = s if _KATA.match(s) else w.surface
        out.append(unicodedata.normalize("NFKC", kana))
    return "".join(out)


def edit_ops(ref, hyp):
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]),
            )
    i, j, S, D, I = n, m, 0, 0, 0
    while i > 0 or j > 0:
        if (
            i > 0
            and j > 0
            and dp[i][j] == dp[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1])
        ):
            S += ref[i - 1] != hyp[j - 1]
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            D += 1
            i -= 1
        else:
            I += 1
            j -= 1
    return dp[n][m], S, D, I


def parse_wer(path):
    blocks = open(path, encoding="utf-8").read().split("=" * 80)
    pairs = []
    for b in blocks:
        lines = [ln for ln in b.strip().split("\n") if ln.strip()]
        if len(lines) >= 4 and ", %WER" in lines[0]:
            ref = "".join(
                t for t in lines[1].split(" ; ") if t.strip() != "<eps>"
            ).replace(" ", "")
            hyp = "".join(
                t for t in lines[3].split(" ; ") if t.strip() != "<eps>"
            ).replace(" ", "")
            pairs.append((lines[0].split(",")[0], ref, hyp))
    return pairs


def main():
    report = {"seeds": list(SEEDS), "splits": {}}
    for split in SPLITS:
        per_seed = []
        examples_ortho, examples_bad = [], []
        for seed in SEEDS:
            path = os.path.join(
                RUN_ROOT, str(seed), "wer_results", f"wer_{split}.txt"
            )
            pairs = parse_wer(path)
            err = S = D = I = reflen = kerr = kreflen = cat = 0
            for _, ref, hyp in pairs:
                e, s, d, ins = edit_ops(ref, hyp)
                err += e
                S += s
                D += d
                I += ins
                reflen += len(ref)
                kr, kh = to_kana(ref), to_kana(hyp)
                ke, *_ = edit_ops(kr, kh)
                kerr += ke
                kreflen += len(kr)
                if e / max(len(ref), 1) > 0.5:
                    cat += 1
                if seed == SEEDS[0]:
                    ucer = e / max(len(ref), 1)
                    kcer = ke / max(len(kr), 1)
                    if ucer >= 0.25 and kcer <= 0.05 and len(examples_ortho) < 3:
                        examples_ortho.append({"ref": ref, "hyp": hyp})
                    if ucer > 0.6 and kcer > 0.4 and len(examples_bad) < 3:
                        examples_bad.append({"ref": ref, "hyp": hyp})
            per_seed.append(
                {
                    "seed": seed,
                    "n": len(pairs),
                    "char_cer": 100 * err / reflen,
                    "sub": 100 * S / reflen,
                    "del": 100 * D / reflen,
                    "ins": 100 * I / reflen,
                    "kana_er": 100 * kerr / kreflen,
                    "catastrophic_pct": 100 * cat / len(pairs),
                }
            )
        agg = {
            k: (
                statistics.mean(s[k] for s in per_seed),
                statistics.stdev(s[k] for s in per_seed),
            )
            for k in ("char_cer", "sub", "del", "ins", "kana_er",
                      "catastrophic_pct")
        }
        report["splits"][split] = {
            "per_seed": per_seed,
            "mean_sd": agg,
            "examples_orthographic": examples_ortho,
            "examples_genuine": examples_bad,
        }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    for split, data in report["splits"].items():
        m = data["mean_sd"]
        print(
            f"{split}: char CER {m['char_cer'][0]:.2f}±{m['char_cer'][1]:.2f} "
            f"-> kana ER {m['kana_er'][0]:.2f}±{m['kana_er'][1]:.2f} "
            f"(S {m['sub'][0]:.1f} / D {m['del'][0]:.1f} / I {m['ins'][0]:.1f})"
        )
    print("wrote", OUT)


if __name__ == "__main__":
    main()
