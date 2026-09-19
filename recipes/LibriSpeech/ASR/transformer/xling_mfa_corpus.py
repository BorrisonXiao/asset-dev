#!/usr/bin/env python3
"""Build an MFA corpus directory (audio symlinks + .lab files) from a manifest.

MFA aligns ``<utt>.wav|.flac`` against ``<utt>.lab``. The lab text must be
RAW and UNSPACED for zh/ja: MFA 3.4 tokenizes internally (pkuseg / sudachipy)
into exactly the word conventions its dictionaries use, and it treats input
spaces as symbols to preserve — pre-segmented text becomes underscore-joined
compound OOVs (ja: the whole utterance aligned as one ``spn``; zh: spaces
reported as OOV tokens). Measured 2026-09-19 on the real tokenizers.

Everything is placed flat in one directory (utt ids are globally unique), so
the MFA output TextGrids land flat as well.

Usage:
    python xling_mfa_corpus.py --csv m/train.csv m/dev.csv m/test.csv \
        --language zh --corpus_dir /path/to/mfa_corpus
"""

import argparse
import csv
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", nargs="+", required=True)
    parser.add_argument("--language", choices=("zh", "ja"), required=True)
    parser.add_argument("--corpus_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.corpus_dir, exist_ok=True)
    n = 0
    for path in args.csv:
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                utt, wav, text = row["ID"], row["wav"], row["wrd"]
                ext = os.path.splitext(wav)[1]
                link = os.path.join(args.corpus_dir, utt + ext)
                if not os.path.lexists(link):
                    os.symlink(wav, link)
                text = text.replace(" ", "")
                with open(
                    os.path.join(args.corpus_dir, utt + ".lab"),
                    "w",
                    encoding="utf-8",
                ) as lh:
                    lh.write(text + "\n")
                n += 1
    print(f"corpus ready: {n} utterances in {args.corpus_dir}")


if __name__ == "__main__":
    main()
