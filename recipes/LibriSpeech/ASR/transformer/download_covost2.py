"""Download the CoVoST 2 En->De parquet shards (audio + transcript + translation).

The FBK MuST-C distribution requires a manual registration form, so the paired
ASR/ST corpus for the task-conditioned study is CoVoST 2 En->De, which carries
the same essential property: one English utterance with both an English
transcript and a German translation.
"""

import argparse
import os
import sys

from huggingface_hub import snapshot_download

REPO_ID = "fixie-ai/covost2"
CONFIG = "en_de"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="local snapshot directory")
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    # HF_HUB_OFFLINE is set globally for the gated LLM; this download needs the network.
    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ.pop(var, None)

    path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        allow_patterns=[f"{args.config}/*.parquet", "README.md"],
        local_dir=args.out,
        max_workers=args.workers,
    )
    print(f"snapshot at {path}", file=sys.stderr)

    shard_dir = os.path.join(args.out, args.config)
    shards = sorted(f for f in os.listdir(shard_dir) if f.endswith(".parquet"))
    total = sum(os.path.getsize(os.path.join(shard_dir, f)) for f in shards)
    print(f"{len(shards)} shards, {total / 1e9:.2f} GB")
    for f in shards:
        print(
            f"  {f} {os.path.getsize(os.path.join(shard_dir, f)) / 1e6:.1f} MB"
        )


if __name__ == "__main__":
    main()
