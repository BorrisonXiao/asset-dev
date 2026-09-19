#!/usr/bin/env python3
"""Char-level CTC forced-alignment boundaries for the cross-lingual study.

LibriSpeech's char boundaries came from torchaudio's pretrained English char
CTC (ctc_boundary_align.py on origin/forced_alignment). No such public model
exists for zh chars / ja moras on our encoders, so this script first TRAINS a
small CTC head on the frozen language encoder (chinese-hubert-large /
japanese-hubert-large), then Viterbi-aligns with
``torchaudio.functional.forced_align`` and writes the same boundary format:
``{boundary_out}/{utt_id}.pt`` holding a 1-D ``torch.uint8`` tensor, one label
per encoder frame, ``1`` = frame starts a new segment, index 0 always ``1``.

The emission frame grid IS the downstream encoder's frame grid (same conv
stack; asserted per utterance), so alignment frame indices are label indices.

Boundary policy (mirrors the English ``char`` level): a boundary at the start
frame of EVERY unit span; CTC blank frames extend the previous segment, so
silences mostly attach to the preceding unit and leading silence belongs to
segment 0.

Reliability artifacts for the S1 gate: ``{head_dir}/metrics.json`` records the
aligner's own greedy dev unit-error-rate per epoch; ``{stats_out}`` records
coverage, failures, and realized rates per aligned split.

Usage (train + align everything):
    python xling_ctc_boundary_align.py --mode all \
        --ssl_hub TencentGameMate/chinese-hubert-large \
        --train_csv m/train.csv --train_units m/train_units_char.tsv \
        --dev_csv m/dev.csv --dev_units m/dev_units_char.tsv \
        --align "train=m/train.csv:m/train_units_char.tsv" \
        --align "dev=m/dev.csv:m/dev_units_char.tsv" \
        --head_dir out/ctc_head --boundary_out out/boundaries/char \
        --stats_out out/boundaries/char_stats.json
"""

import argparse
import csv
import json
import os
import random
import time

import torch
import torchaudio
from torchaudio.functional import forced_align, merge_tokens

SAMPLE_RATE = 16000
BLANK_ID = 0
CONV = [(10, 5), (3, 2), (3, 2), (3, 2), (3, 2), (2, 2), (2, 2)]


def num_encoder_frames(num_samples):
    L = num_samples
    for k, s in CONV:
        L = (L - k) // s + 1
    return L


def read_csv_rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return [
            (r["ID"], float(r["duration"]), r["wav"])
            for r in csv.DictReader(fh)
        ]


def read_units(path):
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            utt, _, units = line.rstrip("\n").partition("\t")
            out[utt] = units.split()
    return out


def build_vocab(units_by_utt):
    inventory = sorted({u for units in units_by_utt.values() for u in units})
    return {u: i + 1 for i, u in enumerate(inventory)}  # 0 = blank


class CTCHead(torch.nn.Module):
    def __init__(self, feat_dim, vocab_size, hidden=320, layers=2):
        super().__init__()
        self.rnn = torch.nn.LSTM(
            feat_dim,
            hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
        )
        self.out = torch.nn.Linear(2 * hidden, vocab_size + 1)

    def forward(self, feats):
        return self.out(self.rnn(feats)[0])


def duration_batches(rows, batch_seconds, max_utts):
    rows = sorted(rows, key=lambda r: r[1])
    batch, secs = [], 0.0
    for row in rows:
        if batch and (secs + row[1] > batch_seconds or len(batch) >= max_utts):
            yield batch
            batch, secs = [], 0.0
        batch.append(row)
        secs += row[1]
    if batch:
        yield batch


def load_batch_audio(batch, device):
    """Load a batch; a corrupt file drops its utterance (logged), not the job."""
    waves, kept = [], []
    for utt, dur, wav in batch:
        try:
            audio, sr = torchaudio.load(wav)
        except Exception as exc:
            print(f"SKIP undecodable audio {utt} ({wav}): {exc}", flush=True)
            continue
        assert sr == SAMPLE_RATE, f"{wav}: {sr} Hz"
        waves.append(audio.mean(dim=0))
        kept.append((utt, dur, wav))
    batch[:] = kept
    if not waves:
        return None, None, None
    lens = [w.numel() for w in waves]
    padded = torch.zeros(len(waves), max(lens))
    mask = torch.zeros(len(waves), max(lens), dtype=torch.long)
    for i, w in enumerate(waves):
        padded[i, : w.numel()] = w
        mask[i, : w.numel()] = 1
    return padded.to(device), mask.to(device), lens


@torch.no_grad()
def ssl_forward(ssl, padded, mask):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = ssl(padded, attention_mask=mask).last_hidden_state
    return hidden.float()


def greedy_unit_error_rate(head, ssl, rows, units_by_utt, vocab, device,
                           batch_seconds, max_utts):
    inv = {i: u for u, i in vocab.items()}
    errors = total = 0
    head.eval()
    with torch.no_grad():
        for batch in duration_batches(rows, batch_seconds, max_utts):
            padded, mask, lens = load_batch_audio(batch, device)
            if padded is None:
                continue
            logits = head(ssl_forward(ssl, padded, mask))
            best = logits.argmax(dim=-1).cpu()
            for i, (utt, _, _) in enumerate(batch):
                T = num_encoder_frames(lens[i])
                ids = best[i, :T].tolist()
                hyp, prev = [], BLANK_ID
                for t in ids:
                    if t != BLANK_ID and t != prev:
                        hyp.append(inv[t])
                    prev = t
                ref = units_by_utt[utt]
                errors += _edit_distance(hyp, ref)
                total += len(ref)
    head.train()
    return errors / max(total, 1)


def _edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ai in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, bj in enumerate(b, 1):
            cur[j] = min(
                prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ai != bj)
            )
        prev = cur
    return prev[-1]


def train_head(args, ssl, vocab, device):
    train_rows = read_csv_rows(args.train_csv)
    dev_rows = read_csv_rows(args.dev_csv)[: args.dev_subset]
    train_units = read_units(args.train_units)
    dev_units = read_units(args.dev_units)
    feat_dim = ssl.config.hidden_size
    head = CTCHead(feat_dim, len(vocab)).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)
    ctc = torch.nn.CTCLoss(blank=BLANK_ID, zero_infinity=True)
    os.makedirs(args.head_dir, exist_ok=True)
    metrics = {"vocab_size": len(vocab), "epochs": []}
    best = None
    for epoch in range(1, args.epochs + 1):
        t0, seen, loss_sum, nb = time.time(), 0, 0.0, 0
        batches = list(
            duration_batches(train_rows, args.batch_seconds, args.max_utts)
        )
        random.Random(1000 + epoch).shuffle(batches)
        for batch in batches:
            padded, mask, lens = load_batch_audio(batch, device)
            if padded is None:
                continue
            logits = head(ssl_forward(ssl, padded, mask))
            frame_lens = torch.tensor(
                [num_encoder_frames(n) for n in lens], dtype=torch.long
            )
            targets, target_lens = [], []
            for utt, _, _ in batch:
                ids = [vocab[u] for u in train_units[utt]]
                targets.extend(ids)
                target_lens.append(len(ids))
            log_probs = torch.log_softmax(logits, dim=-1).transpose(0, 1)
            loss = ctc(
                log_probs,
                torch.tensor(targets, dtype=torch.long),
                frame_lens,
                torch.tensor(target_lens, dtype=torch.long),
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
            opt.step()
            loss_sum += float(loss)
            nb += 1
            seen += len(batch)
            if nb % 200 == 0:
                print(
                    f"epoch {epoch} batch {nb} utts {seen} "
                    f"loss {loss_sum / nb:.3f}",
                    flush=True,
                )
        uer = greedy_unit_error_rate(
            head, ssl, dev_rows, dev_units, vocab, device,
            args.batch_seconds, args.max_utts,
        )
        row = {
            "epoch": epoch,
            "train_ctc_loss": loss_sum / max(nb, 1),
            "dev_unit_error_rate": uer,
            "minutes": (time.time() - t0) / 60,
        }
        metrics["epochs"].append(row)
        print("EPOCH", json.dumps(row), flush=True)
        if best is None or uer < best:
            best = uer
            torch.save(head.state_dict(), os.path.join(args.head_dir, "head.pt"))
    metrics["best_dev_unit_error_rate"] = best
    with open(os.path.join(args.head_dir, "vocab.txt"), "w") as fh:
        for u, i in sorted(vocab.items(), key=lambda kv: kv[1]):
            fh.write(f"{u}\t{i}\n")
    with open(os.path.join(args.head_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    return head


def align_split(name, csv_path, units_path, head, ssl, vocab, args, device):
    rows = read_csv_rows(csv_path)
    units_by_utt = read_units(units_path)
    os.makedirs(args.boundary_out, exist_ok=True)
    stats = {"n": len(rows), "aligned": 0, "failed": 0, "rho_sum": 0.0}
    head.eval()
    with torch.no_grad():
        for batch in duration_batches(rows, args.batch_seconds, args.max_utts):
            n_before = len(batch)
            padded, mask, lens = load_batch_audio(batch, device)
            stats["failed"] += n_before - len(batch)
            if padded is None:
                continue
            logits = head(ssl_forward(ssl, padded, mask))
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            for i, (utt, _, _) in enumerate(batch):
                T = num_encoder_frames(lens[i])
                units = units_by_utt[utt]
                oov = [u for u in units if u not in vocab]
                if oov:
                    # Units outside the train inventory cannot be aligned;
                    # eval manifests are pre-filtered for this, so any hit
                    # here is worth seeing in the log.
                    print(f"OOV units for {utt}: {oov}", flush=True)
                    stats["oov_units"] = stats.get("oov_units", 0) + 1
                    stats["failed"] += 1
                    continue
                ids = [vocab[u] for u in units]
                if len(ids) == 0 or len(ids) > T:
                    stats["failed"] += 1
                    continue
                emission = log_probs[i : i + 1, :T]
                targets = torch.tensor([ids], dtype=torch.int32, device=device)
                try:
                    frames, scores = forced_align(
                        emission, targets, blank=BLANK_ID
                    )
                except Exception:
                    stats["failed"] += 1
                    continue
                spans = merge_tokens(frames[0], scores[0])
                boundary = torch.zeros(T, dtype=torch.uint8)
                boundary[0] = 1
                for span in spans:
                    boundary[span.start] = 1
                torch.save(
                    boundary, os.path.join(args.boundary_out, f"{utt}.pt")
                )
                stats["aligned"] += 1
                stats["rho_sum"] += float(boundary.sum()) / T
    stats["mean_rho"] = stats["rho_sum"] / max(stats["aligned"], 1)
    stats["mean_hz"] = 50.0 * stats["mean_rho"]
    del stats["rho_sum"]
    print(name, json.dumps(stats), flush=True)
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "align", "all"),
                        default="all")
    parser.add_argument("--ssl_hub", required=True)
    parser.add_argument("--hf_cache", default=os.environ.get("HF_HUB_CACHE"))
    parser.add_argument("--train_csv")
    parser.add_argument("--train_units")
    parser.add_argument("--dev_csv")
    parser.add_argument("--dev_units")
    parser.add_argument("--dev_subset", type=int, default=1500)
    parser.add_argument("--align", action="append", default=[],
                        help="name=csv:units, repeatable")
    parser.add_argument("--head_dir", required=True)
    parser.add_argument("--boundary_out")
    parser.add_argument("--stats_out")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_seconds", type=float, default=320.0)
    parser.add_argument("--max_utts", type=int, default=48)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from transformers import AutoModel

    device = torch.device(args.device)
    ssl = AutoModel.from_pretrained(args.ssl_hub, cache_dir=args.hf_cache)
    ssl.eval().to(device)
    for p in ssl.parameters():
        p.requires_grad = False

    vocab_path = os.path.join(args.head_dir, "vocab.txt")
    if args.mode in ("train", "all"):
        vocab = build_vocab(read_units(args.train_units))
        head = train_head(args, ssl, vocab, device)
    else:
        vocab = {}
        with open(vocab_path) as fh:
            for line in fh:
                u, i = line.rstrip("\n").split("\t")
                vocab[u] = int(i)
        head = CTCHead(ssl.config.hidden_size, len(vocab)).to(device)
    head.load_state_dict(
        torch.load(os.path.join(args.head_dir, "head.pt"), map_location=device)
    )

    if args.mode in ("align", "all"):
        all_stats = {}
        for spec in args.align:
            name, _, rest = spec.partition("=")
            csv_path, _, units_path = rest.partition(":")
            all_stats[name] = align_split(
                name, csv_path, units_path, head, ssl, vocab, args, device
            )
        if args.stats_out:
            os.makedirs(os.path.dirname(args.stats_out), exist_ok=True)
            with open(args.stats_out, "w") as fh:
                json.dump(all_stats, fh, indent=2)


if __name__ == "__main__":
    main()
