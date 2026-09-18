#!/usr/bin/env python3
"""Score SeamlessM4T-v2 on the exact CoVoST test sets our multi-task model used.

This exists to answer "how far off is our number, really?" with a topline
measured on the same audio, the same utterance list, and the same metric
implementations -- which is worth more than any published figure computed on a
different subset with a different tokenizer.

Comparability rules enforced here:

* the utterance list is read from the run's own evaluated manifests, so both
  systems score the identical 15527 utterances;
* ASR hypotheses are folded through the same LibriSpeech-style normalizer the
  manifests' English references already went through, because our model is
  trained to emit upper-case unpunctuated text and SeamlessM4T is not --
  scoring raw output against normalized references would penalize it for a
  formatting convention rather than for recognition;
* both raw and normalized ASR WER are reported so the effect of that choice is
  visible rather than hidden;
* translation uses the same sacreBLEU/chrF++ objects and therefore the same
  signatures.
"""

import argparse
import csv
import json
import logging
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from covost2_prepare import normalize_english  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)


def read_manifest(path):
    """Utterance rows in manifest order."""
    with open(path, encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def load_audio(path, target_sr=16000):
    """Read one utterance as a mono waveform at ``target_sr``."""
    import soundfile as sf

    wave, sample_rate = sf.read(path, dtype="float32")
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    if sample_rate != target_sr:
        import torchaudio

        wave = torchaudio.functional.resample(
            torch.from_numpy(wave), sample_rate, target_sr
        ).numpy()
    return wave


def batched(rows, size):
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


@torch.no_grad()
def generate(model, processor, rows, tgt_lang, device, batch_size, max_new_tokens):
    """Decode one manifest with SeamlessM4T-v2, longest-first for packing."""
    order = sorted(range(len(rows)), key=lambda i: -float(rows[i]["duration"]))
    hypotheses = [None] * len(rows)
    done = 0
    started = time.time()

    for chunk in batched(order, batch_size):
        waves = [load_audio(rows[i]["wav"]) for i in chunk]
        inputs = processor(
            # transformers 5.x renamed this from `audios`.
            audio=waves,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        ).to(device)
        tokens = model.generate(
            **inputs,
            tgt_lang=tgt_lang,
            num_beams=5,
            max_new_tokens=max_new_tokens,
        )
        texts = processor.batch_decode(tokens, skip_special_tokens=True)
        for index, text in zip(chunk, texts):
            hypotheses[index] = text.strip()
        done += len(chunk)
        if done % (batch_size * 20) < batch_size:
            rate = done / max(time.time() - started, 1e-6)
            logger.info(
                "%s: %d/%d utterances (%.1f utt/s, eta %.0f min)",
                tgt_lang,
                done,
                len(rows),
                rate,
                (len(rows) - done) / max(rate, 1e-6) / 60.0,
            )
    return hypotheses


def score_asr(hypotheses, references):
    """WER/CER both raw and under the manifests' normalization convention."""
    from speechbrain.utils.metric_stats import ErrorRateStats

    results = {}
    for label, transform in (
        ("normalized", normalize_english),
        ("raw", lambda text: text),
    ):
        wer, cer = ErrorRateStats(), ErrorRateStats(split_tokens=True)
        ids = [str(i) for i in range(len(hypotheses))]
        hyp = [transform(h).split(" ") for h in hypotheses]
        ref = [transform(r).split(" ") for r in references]
        wer.append(ids, hyp, ref)
        cer.append(ids, hyp, ref)
        results[f"WER_{label}"] = wer.summarize("error_rate")
        results[f"CER_{label}"] = cer.summarize("error_rate")
    return results


def score_translation(hypotheses, references):
    """Case-sensitive detokenized sacreBLEU plus chrF++, with signatures."""
    from sacrebleu.metrics import BLEU, CHRF

    bleu, chrf = BLEU(), CHRF(word_order=2)
    return {
        "BLEU": bleu.corpus_score(hypotheses, [references]).score,
        "chrF2": chrf.corpus_score(hypotheses, [references]).score,
        "BLEU_signature": str(bleu.get_signature()),
        "chrF2_signature": str(chrf.get_signature()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="facebook/seamless-m4t-v2-large")
    parser.add_argument("--model-cache", required=True)
    parser.add_argument(
        "--asr-manifest",
        required=True,
        help="the run's own evaluated CoVoST ASR manifest",
    )
    parser.add_argument(
        "--st-manifest",
        default=None,
        help="the run's own evaluated ST manifest; omit for ASR-only corpora "
        "such as LibriSpeech, which is scored as a cross-domain control",
    )
    parser.add_argument("--out", required=True, help="JSON results path")
    parser.add_argument("--hypothesis-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText

    device = torch.device(args.device)
    logger.info("Loading %s", args.model)
    processor = AutoProcessor.from_pretrained(
        args.model, cache_dir=args.model_cache
    )
    model = (
        SeamlessM4Tv2ForSpeechToText.from_pretrained(
            args.model, cache_dir=args.model_cache, dtype=torch.float16
        )
        .to(device)
        .eval()
    )
    logger.info(
        "Parameters: %.2fB", sum(p.numel() for p in model.parameters()) / 1e9
    )

    asr_rows = read_manifest(args.asr_manifest)
    st_rows = read_manifest(args.st_manifest) if args.st_manifest else None
    if args.limit:
        asr_rows = asr_rows[: args.limit]
        if st_rows is not None:
            st_rows = st_rows[: args.limit]
    if st_rows is not None:
        # Two views of one audio list; if they ever diverge the paired
        # comparison is meaningless, so check rather than assume.
        assert [r["utt_key"] for r in asr_rows] == [
            r["utt_key"] for r in st_rows
        ], "ASR and ST manifests cover different utterances"
    logger.info(
        "Scoring %d utterances (%s)",
        len(asr_rows),
        "ASR + ST" if st_rows is not None else "ASR only",
    )

    report = {
        "model": args.model,
        "utterances": len(asr_rows),
        "asr_manifest": args.asr_manifest,
        "st_manifest": args.st_manifest,
    }


    started = time.time()
    asr_hyp = generate(
        model, processor, asr_rows, "eng", device, args.batch_size,
        args.max_new_tokens,
    )
    report["asr_seconds"] = time.time() - started
    report["asr"] = score_asr(asr_hyp, [r["wrd"] for r in asr_rows])
    logger.info("ASR: %s", report["asr"])

    st_hyp = None
    if st_rows is not None:
        started = time.time()
        st_hyp = generate(
            model, processor, st_rows, "deu", device, args.batch_size,
            args.max_new_tokens,
        )
        report["st_seconds"] = time.time() - started
        report["st"] = score_translation(st_hyp, [r["wrd"] for r in st_rows])
        logger.info("ST: %s", report["st"])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
    logger.info("Wrote %s", args.out)

    if args.hypothesis_dir:
        os.makedirs(args.hypothesis_dir, exist_ok=True)
        pairs = [("asr", asr_rows, asr_hyp)]
        if st_hyp is not None:
            pairs.append(("st", st_rows, st_hyp))
        for name, rows, hyps in pairs:
            path = os.path.join(args.hypothesis_dir, f"seamless_{name}.tsv")
            with open(path, "w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream, delimiter="\t")
                writer.writerow(["utt_key", "reference", "hypothesis"])
                for row, hyp in zip(rows, hyps):
                    writer.writerow([row["utt_key"], row["wrd"], hyp])
            logger.info("Wrote %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
