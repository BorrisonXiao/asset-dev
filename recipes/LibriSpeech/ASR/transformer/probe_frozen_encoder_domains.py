#!/usr/bin/env python3
"""Does frozen WavLM represent Common Voice as well as it represents LibriSpeech?

The multi-task system is ~2x worse on CoVoST than on LibriSpeech even on audio a
strong model transcribes perfectly, and that gap is not explained by the audio
token budget, the text normalization, or a handful of difficult speakers. The
one component that received no adaptation at all is the frozen WavLM encoder,
so this probe tests it directly and in isolation.

Design: frozen WavLM-large -> a small trainable character-CTC head. No
segmenter, no pooling, no LLM, full 50 Hz frames. Trained separately on
matched *hours* of each corpus and evaluated on both, giving a 2x2:

                      eval LibriSpeech   eval CoVoST
    train LibriSpeech        A                B
    train CoVoST             C                D

* A vs D is the decisive comparison: both in-domain, matched budget. If D is
  much worse than A, the frozen features are genuinely poorer for Common Voice
  and the encoder is the bottleneck.
* B and C measure transferability, separating "features are worse" from
  "features are fine but our training was LibriSpeech-shaped".

Character CTC is used rather than phones because it needs only transcripts,
which both corpora have, and it keeps the two domains exactly comparable.
"""

import argparse
import json
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wavlm_phone_ctc import (  # noqa: E402
    edit_distance,
    forward_features,
    load_wavlm,
    output_lengths,
)

BLANK_ID = 0
# The manifests' English text is already folded to this alphabet.
ALPHABET = " 'ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHAR_TO_ID = {c: i + 1 for i, c in enumerate(ALPHABET)}
VOCAB_SIZE = len(ALPHABET) + 1


def encode_text(text):
    return [CHAR_TO_ID[c] for c in text if c in CHAR_TO_ID]


def decode_ids(ids):
    inverse = {i: c for c, i in CHAR_TO_ID.items()}
    return "".join(inverse.get(i, "") for i in ids)


class ManifestDataset(Dataset):
    """Utterances from a multi-task CSV manifest, capped to a total duration."""

    def __init__(self, csv_path, max_hours=None, max_seconds=20.0, seed=0):
        import csv as csv_module

        with open(csv_path, encoding="utf-8") as stream:
            rows = [
                r
                for r in csv_module.DictReader(stream)
                if 1.0 <= float(r["duration"]) <= max_seconds
            ]
        # Deterministic shuffle, then take utterances until the hour budget is
        # met. Matching hours rather than utterance counts is what makes the
        # two corpora comparable -- their mean durations differ by over 2x.
        random.Random(seed).shuffle(rows)
        if max_hours is not None:
            kept, total = [], 0.0
            for row in rows:
                if total >= max_hours * 3600.0:
                    break
                kept.append(row)
                total += float(row["duration"])
            rows = kept
        self.rows = rows
        self.hours = sum(float(r["duration"]) for r in rows) / 3600.0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import soundfile as sf

        row = self.rows[index]
        wave, _ = sf.read(row["wav"], dtype="float32")
        if wave.ndim > 1:
            wave = wave.mean(axis=1)
        return {
            "wave": torch.from_numpy(wave),
            "text": row["wrd"],
            "target": torch.tensor(encode_text(row["wrd"]), dtype=torch.long),
        }


def collate(batch):
    batch = [b for b in batch if b["target"].numel() > 0]
    lengths = torch.tensor([b["wave"].numel() for b in batch])
    padded = torch.zeros(len(batch), int(lengths.max()))
    for i, item in enumerate(batch):
        padded[i, : item["wave"].numel()] = item["wave"]
    mask = torch.arange(padded.size(1))[None, :] < lengths[:, None]
    return {
        "waveforms": padded,
        "attention_mask": mask.long(),
        "sample_lengths": lengths,
        "targets": [b["target"] for b in batch],
        "texts": [b["text"] for b in batch],
    }


class CharCTCHead(nn.Module):
    """Deliberately small: this measures the features, not the head."""

    def __init__(self, input_dim, hidden_dim, vocab_size, dropout):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size),
        )

    def forward(self, features):
        return self.network(features)


def greedy_decode(logits, lengths):
    ids = logits.argmax(dim=-1)
    out = []
    for row, length in zip(ids, lengths):
        previous, collapsed = None, []
        for token in row[:length].tolist():
            if token != previous and token != BLANK_ID:
                collapsed.append(token)
            previous = token
        out.append(decode_ids(collapsed))
    return out


@torch.inference_mode()
def evaluate(encoder, head, loader, device, max_batches=None):
    """Character and word error rate of the probe on one corpus."""
    head.eval()
    char_edits = char_total = word_edits = word_total = 0
    for index, batch in enumerate(loader):
        if max_batches and index >= max_batches:
            break
        waveforms = batch["waveforms"].to(device)
        mask = batch["attention_mask"].to(device)
        feats = forward_features(encoder, waveforms, mask)
        frame_lengths = output_lengths(
            encoder, batch["sample_lengths"].to(device)
        )
        logits = head(feats.float())
        hyps = greedy_decode(logits, frame_lengths.tolist())
        for hyp, ref in zip(hyps, batch["texts"]):
            char_edits += edit_distance(list(ref), list(hyp))
            char_total += len(ref)
            word_edits += edit_distance(ref.split(), hyp.split())
            word_total += len(ref.split())
    head.train()
    return {
        "CER": 100.0 * char_edits / max(char_total, 1),
        "WER": 100.0 * word_edits / max(word_total, 1),
    }


def train_probe(encoder, train_loader, device, args):
    head = CharCTCHead(1024, args.hidden_dim, VOCAB_SIZE, args.dropout).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr)
    step = 0
    started = time.time()
    head.train()
    while step < args.steps:
        for batch in train_loader:
            if step >= args.steps:
                break
            waveforms = batch["waveforms"].to(device)
            mask = batch["attention_mask"].to(device)
            feats = forward_features(encoder, waveforms, mask)
            frame_lengths = output_lengths(
                encoder, batch["sample_lengths"].to(device)
            )
            logits = head(feats.float())
            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
            targets = torch.cat(batch["targets"]).to(device)
            target_lengths = torch.tensor(
                [t.numel() for t in batch["targets"]], device=device
            )
            loss = F.ctc_loss(
                log_probs,
                targets,
                frame_lengths,
                target_lengths,
                blank=BLANK_ID,
                zero_infinity=True,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
            optimizer.step()
            step += 1
            if step % args.log_interval == 0:
                print(
                    f"  step {step}/{args.steps} loss {loss.item():.4f} "
                    f"({step / max(time.time() - started, 1e-6):.2f} step/s)",
                    flush=True,
                )
    return head


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--librispeech-train", required=True)
    parser.add_argument("--covost-train", required=True)
    parser.add_argument("--librispeech-eval", required=True)
    parser.add_argument("--covost-eval", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--wavlm", default="microsoft/wavlm-large")
    parser.add_argument(
        "--train-hours",
        type=float,
        default=100.0,
        help="matched budget per corpus; hours, not utterances",
    )
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batches", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--train-domains",
        nargs="+",
        default=["librispeech", "covost"],
        choices=["librispeech", "covost"],
        help="which probe(s) to train here. The two trainings are independent "
        "-- they share only the read-only frozen encoder -- so running one per "
        "job halves wall clock. Each job still evaluates on BOTH corpora, so a "
        "row of the 2x2 is always complete and self-contained.",
    )
    parser.add_argument("--log-interval", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)

    print(f"Loading frozen {args.wavlm}", flush=True)
    encoder = load_wavlm(args.wavlm, args.cache_dir, device)

    corpora = {}
    for name, train_csv, eval_csv in (
        ("librispeech", args.librispeech_train, args.librispeech_eval),
        ("covost", args.covost_train, args.covost_eval),
    ):
        train_set = ManifestDataset(
            train_csv, max_hours=args.train_hours, seed=args.seed
        )
        eval_set = ManifestDataset(eval_csv, max_hours=None, seed=args.seed)
        print(
            f"{name}: train {len(train_set)} utts / {train_set.hours:.1f} h; "
            f"eval {len(eval_set)} utts / {eval_set.hours:.1f} h",
            flush=True,
        )
        corpora[name] = {
            "train": DataLoader(
                train_set,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                collate_fn=collate,
                drop_last=True,
            ),
            "eval": DataLoader(
                eval_set,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=collate,
            ),
            "train_hours": train_set.hours,
            "train_utts": len(train_set),
        }

    report = {
        "wavlm": args.wavlm,
        "train_hours_requested": args.train_hours,
        "steps": args.steps,
        "corpora": {
            k: {"train_hours": v["train_hours"], "train_utts": v["train_utts"]}
            for k, v in corpora.items()
        },
        "grid": {},
    }

    for train_domain in args.train_domains:
        print(f"\n=== training probe on {train_domain} ===", flush=True)
        head = train_probe(encoder, corpora[train_domain]["train"], device, args)
        for eval_domain in ("librispeech", "covost"):
            scores = evaluate(
                encoder,
                head,
                corpora[eval_domain]["eval"],
                device,
                max_batches=args.eval_batches,
            )
            report["grid"][f"train_{train_domain}__eval_{eval_domain}"] = scores
            print(
                f"  train={train_domain:11s} eval={eval_domain:11s} "
                f"CER={scores['CER']:.2f} WER={scores['WER']:.2f}",
                flush=True,
            )

    in_ls = report["grid"].get("train_librispeech__eval_librispeech")
    in_cv = report["grid"].get("train_covost__eval_covost")
    if in_ls and in_cv:
        report["in_domain_gap"] = {
            "CER": in_cv["CER"] - in_ls["CER"],
            "WER": in_cv["WER"] - in_ls["WER"],
        }
        print(
            "\nDecisive comparison (both in-domain, matched hours): "
            f"LibriSpeech CER {in_ls['CER']:.2f} vs CoVoST CER "
            f"{in_cv['CER']:.2f} (gap {report['in_domain_gap']['CER']:+.2f})",
            flush=True,
        )
    else:
        print(
            "\nPartial grid (one training domain in this job); merge the "
            "per-domain reports for the decisive in-domain comparison.",
            flush=True,
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(f"Wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
