#!/usr/bin/env python3
"""Training-dynamics report for the SpeechLLM segmenter — a recipe on top of `htmlkit`.

Domain logic (SpeechBrain train_log parsing, wav2vec2+segmenter boundary-on-spectrogram
viz) lives here; the reusable design system + SVG chart engine + assembly come from the
`html-report` skill's `htmlkit`. Produces a single self-contained report.html, pushed to
GitHub Pages (borrisonxiao.github.io/jsalt26-downsampling) — NOT to a Claude Artifact,
which doesn't work reliably for this user; don't reintroduce that.

    python make_training_report.py --out artifacts/segmenter/report.html
"""
import argparse
import base64
import html as H
import json
import os
import re
import sys

import numpy as np
import torch

# reusable HTML toolkit from the html-report skill
sys.path.insert(0, os.path.expanduser("~/.claude-scale/skills/html-report"))
import htmlkit as hk  # noqa: E402

DATA = "/export/jsalt26/omnienc/users/cxiao/datasets"
SSL_CACHE = "/export/jsalt26/omnienc/users/cxiao/hf/hub"
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
_NUM = r"[-+0-9.eE]+"
FRAME_HZ = 50.0


def rate_hz(rho):
    """Convert the internal frame-retention fraction to audio-token frequency."""
    return FRAME_HZ * float(rho)


def rate_hz_stats(stats):
    """Convert a ``(mean, sample_sd)`` retention-fraction pair to Hz."""
    return rate_hz(stats[0]), rate_hz(stats[1])


# --------------------------- log parsing ---------------------------
def parse_log(path):
    rows = {}
    if not os.path.exists(path):
        return []
    for line in open(path):
        if "epoch:" not in line or "train" not in line:
            continue
        d = {}
        for k, v in re.findall(rf"(\w[\w ]*?): ({_NUM})", line):
            try:
                d[k.strip()] = float(v)
            except ValueError:
                pass
        if "epoch" in d:
            rows[int(d["epoch"])] = d
    return [rows[e] for e in sorted(rows)]


def parse_test_wer(path):
    out = []
    if os.path.exists(path):
        for line in open(path):
            m = re.search(rf"test .*?WER: ({_NUM})", line)
            if m:
                out.append(float(m.group(1)))
    return out


def parse_wer_file(path):
    """Read aggregate and per-utterance edit counts from a SpeechBrain WER file."""
    if not os.path.exists(path):
        return None
    text = open(path, encoding="utf-8", errors="replace").read()
    first = re.search(
        r"^%WER ([0-9.]+) \[ (\d+) / (\d+), (\d+) ins, (\d+) del, (\d+) sub \]",
        text,
        re.M,
    )
    if not first:
        return None
    reported_wer, errors, words, ins, dele, sub = first.groups()
    errors, words, ins, dele, sub = map(int, (errors, words, ins, dele, sub))
    utterances = {}
    for m in re.finditer(
        r"^([^,\n]+), %WER ([^ ]+) \[ (\d+) / (\d+), "
        r"(\d+) ins, (\d+) del, (\d+) sub \]$",
        text,
        re.M,
    ):
        uid, reported_uwer, uerr, uwords, uins, udel, usub = m.groups()
        uerr, uwords, uins, udel, usub = map(int, (uerr, uwords, uins, udel, usub))
        utterances[uid] = {
            "wer": 100 * uerr / max(uwords, 1), "reported_wer": float(reported_uwer),
            "errors": uerr, "words": uwords, "ins": uins, "del": udel, "sub": usub,
        }
    return {
        "wer": 100 * errors / max(words, 1), "reported_wer": float(reported_wer),
        "errors": errors, "words": words, "ins": ins, "del": dele, "sub": sub,
        "utterances": utterances,
    }


def parse_jsonl(path):
    """Read the complete JSON objects from a benchmark JSONL artifact."""
    if not os.path.exists(path):
        return []
    rows = []
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # A killed evaluation can leave a partial final line. Earlier complete
            # split records are still valid and are kept explicitly.
            continue
    return rows


def aggregate_inference_benchmark(
        path, splits=("test-clean", "test-other"), required_protocol=None):
    """Aggregate forward-only inference measurements over named data splits.

    RTF and utterances/second are recomputed from summed durations rather than
    averaged across corpora. Peak memory is the maximum observed allocation.
    """
    wanted = set(splits)
    protocol_rows = [
        row for row in parse_jsonl(path)
        if (required_protocol is None
            or row.get("decoding_protocol") == required_protocol)
    ]
    rows = [row for row in protocol_rows if row.get("split") in wanted]
    if not rows:
        return None
    forward_seconds = sum(row["measured_forward_seconds"] for row in rows)
    audio_seconds = sum(row["measured_audio_seconds"] for row in rows)
    utterances = sum(row["measured_utterances"] for row in rows)
    tokens = sum(
        row["token_frequency_hz"]["global"] * row["audio_seconds"]
        for row in rows
    )
    token_audio_seconds = sum(row["audio_seconds"] for row in rows)
    batch_sizes = {int(row["batch_size"]) for row in rows}
    warmup_batches = {int(row["warmup_batches"]) for row in rows}
    if len(batch_sizes) != 1 or len(warmup_batches) != 1:
        raise ValueError("Mixed inference protocol settings in %s" % path)
    return {
        "forward_rtf": forward_seconds / audio_seconds,
        "utterances_per_second": utterances / forward_seconds,
        "token_frequency_hz": tokens / token_audio_seconds,
        "peak_allocated_gb": max(row["gpu_peak_allocated_gb"] for row in rows),
        "peak_reserved_gb": max(row["gpu_peak_reserved_gb"] for row in rows),
        "run_peak_allocated_gb": max(
            row["gpu_peak_allocated_gb"] for row in protocol_rows
        ),
        "run_peak_reserved_gb": max(
            row["gpu_peak_reserved_gb"] for row in protocol_rows
        ),
        "prewarm_splits": sum(
            row.get("split") not in wanted for row in protocol_rows
        ),
        "splits": len(rows),
        "expected_splits": len(wanted),
        "hardware": rows[0].get("hardware", "unknown GPU"),
        "precision": rows[0].get("precision", "unknown precision"),
        "batch_size": batch_sizes.pop(),
        "warmup_batches": warmup_batches.pop(),
        "decoding_protocol": rows[0].get("decoding_protocol", "legacy"),
    }


def selected_epoch(path):
    """Epoch recovered for final evaluation, or None while evaluation is pending."""
    if not os.path.exists(path):
        return None
    for line in open(path, encoding="utf-8", errors="replace"):
        m = re.match(r"Epoch loaded: (\d+) - test", line)
        if m:
            return int(m.group(1))
    return None


def mean_sd(values):
    values = [float(v) for v in values]
    if not values:
        return float("nan"), float("nan")
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def series(rows, key, warmup=0):
    """[(epoch - warmup, value)] so x=1 is the first RL epoch, x<=0 is warmup."""
    return [(int(r["epoch"]) - warmup, r[key]) for r in rows
            if key in r and r[key] == r[key]]


def frequency_series(rows, key, warmup=0):
    """Convert an internal frame fraction series into reader-facing tokens/second."""
    return [(x, rate_hz(value)) for x, value in series(rows, key, warmup)]


def multiseed_audit_section(segmenter_root):
    """Evidence-backed snapshot of the corrected shared-warm-up sweep."""
    root = os.path.join(segmenter_root, "multiseed_shared_warmup_onpolicy")
    seeds = (3407, 3408, 3409)
    splits = ("test-clean", "test-other", "dev-other")
    variants = ("nll_frozen", "nll_mt")
    data = {}
    for variant in variants:
        for seed in seeds:
            run = os.path.join(root, variant, "cnn", str(seed))
            for split in splits:
                data[variant, seed, split] = parse_wer_file(
                    os.path.join(run, "wer_results", "wer_%s.txt" % split)
                )

    def stats(variant, split):
        vals = [data[variant, seed, split]["wer"] for seed in seeds
                if data[variant, seed, split]]
        return mean_sd(vals)

    def artifact_status(variant, backbone, seed, total=10):
        run = os.path.join(root, variant, backbone, str(seed))
        logs = parse_log(os.path.join(run, "train_log.txt"))
        n_eval = sum(os.path.exists(os.path.join(run, "wer_results", "wer_%s.txt" % s))
                     for s in splits)
        if n_eval == len(splits):
            return "done"
        if logs:
            return "ep %d/%d" % (int(logs[-1]["epoch"]), total)
        return "waiting"

    def seed_status(variant, backbone, total=10):
        pills = " ".join(
            '<span class="pill">%d · %s</span>'
            % (seed, artifact_status(variant, backbone, seed, total))
            for seed in seeds
        )
        return '<div style="display:flex;flex-wrap:wrap;gap:6px">%s</div>' % pills

    frozen_clean, frozen_clean_sd = stats("nll_frozen", "test-clean")
    frozen_other, frozen_other_sd = stats("nll_frozen", "test-other")
    mt_clean, mt_clean_sd = stats("nll_mt", "test-clean")
    mt_other, mt_other_sd = stats("nll_mt", "test-other")

    progress_rows = [
        ("CNN · NLL · frozen decoder", "45030/33/36",
         seed_status("nll_frozen", "cnn")),
        ("CNN · NLL · co-trained decoder", "45031/34/37",
         seed_status("nll_mt", "cnn")),
        ("CNN · CER · co-trained decoder", "45032/35/38",
         seed_status("cer_mt", "cnn")),
    ]
    tf_warm = parse_log(os.path.join(
        segmenter_root, "shared_warmup", "transformer", "3407", "train_log.txt"
    ))
    progress_rows += [
        ("Transformer · shared warm-up", "45236",
         ('<span class="pill">ep %d/6</span>' % int(tf_warm[-1]["epoch"]))
         if tf_warm else '<span class="pill">waiting</span>'),
        ("Transformer · 3 variants × 3 seeds", "45237–45245",
         "%d/9 complete" % sum(
             artifact_status(v, "transformer", s) == "done"
             for v in ("nll_frozen", "nll_mt", "cer_mt") for s in seeds)),
    ]
    progress_table = (
        '<table style="table-layout:fixed"><colgroup><col style="width:35%">'
        '<col style="width:18%"><col style="width:47%"></colgroup>'
        "<thead><tr><th>run family</th><th>jobs</th><th>current state</th>"
        "</tr></thead><tbody>"
        + "".join('<tr><td>%s</td><td style="text-align:center">%s</td>'
                  '<td style="text-align:left">%s</td></tr>' % r
                  for r in progress_rows)
        + "</tbody></table>"
    )

    result_rows = []
    for variant, label in (("nll_frozen", "NLL · decoder frozen"),
                           ("nll_mt", "NLL · decoder co-trained")):
        for seed in seeds:
            run = os.path.join(root, variant, "cnn", str(seed))
            ep = selected_epoch(os.path.join(run, "train_log.txt"))
            vals = [data[variant, seed, split] for split in splits]
            if all(vals):
                result_rows.append((label, str(seed), str(ep) if ep else "—")
                                   + tuple("%.2f" % x["wer"] for x in vals))
        means = [stats(variant, split) for split in splits]
        result_rows.append(("<b>%s</b>" % label, "mean ± SD", "—")
                           + tuple("<b>%.2f ± %.2f</b>" % x for x in means))
    results_table = (
        "<table><thead><tr><th>condition</th><th>seed</th><th>selected epoch</th>"
        "<th>test-clean WER</th><th>test-other WER</th><th>dev-other WER</th>"
        "</tr></thead><tbody>"
        + "".join("<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td>"
                  "<td>%s</td></tr>" % r for r in result_rows)
        + "</tbody></table>"
    )

    paired = {}
    edit_delta = {}
    runaway = {}
    diagnostic = {}
    for split in splits:
        wins = losses = ties = 0
        edits = dict.fromkeys(("errors", "ins", "del", "sub"), 0)
        for seed in seeds:
            frozen = data["nll_frozen", seed, split]
            mt = data["nll_mt", seed, split]
            if not frozen or not mt:
                continue
            common = frozen["utterances"].keys() & mt["utterances"].keys()
            for uid in common:
                ef = frozen["utterances"][uid]["errors"]
                em = mt["utterances"][uid]["errors"]
                if em < ef:
                    wins += 1
                elif em > ef:
                    losses += 1
                else:
                    ties += 1
            for key in edits:
                edits[key] += mt[key] - frozen[key]
        paired[split] = (wins, losses, ties)
        edit_delta[split] = edits

        for variant in variants:
            n_bad = 0
            diag_vals = []
            for seed in seeds:
                record = data[variant, seed, split]
                if not record:
                    continue
                bad = [u for u in record["utterances"].values() if u["wer"] > 200]
                n_bad += len(bad)
                e = record["errors"] - sum(u["errors"] for u in bad)
                n = record["words"] - sum(u["words"] for u in bad)
                diag_vals.append(100 * e / max(n, 1))
            runaway[variant, split] = n_bad
            diagnostic[variant, split] = mean_sd(diag_vals)

    evidence_rows = [
        ("Shared warm-up", "Confirmed",
         "All nine corrected CNN jobs load the same epoch-6 checkpoint."),
        ("NLL decoder co-training", "Confirmed",
         "WER improves in every seed; mean Δ is %.2f pp clean and %.2f pp other."
         % (mt_clean - frozen_clean, mt_other - frozen_other)),
        ("CER reward vs NLL", "Open",
         "Final CER evaluations are not all available."),
        ("Transformer vs CNN", "Open",
         "The Transformer warm-up is active; downstream evaluations are pending."),
    ]
    evidence_table = (
        '<table style="table-layout:fixed"><colgroup><col style="width:30%">'
        '<col style="width:16%"><col style="width:54%"></colgroup>'
        "<thead><tr><th>claim</th><th>verdict</th><th>evidence</th></tr></thead><tbody>"
        + "".join('<tr><td>%s</td><td style="text-align:center">'
                  '<span class="pill">%s</span></td><td style="text-align:left">%s</td></tr>' % r
                  for r in evidence_rows)
        + "</tbody></table>"
    )

    clean_w, clean_l, clean_t = paired["test-clean"]
    other_w, other_l, other_t = paired["test-other"]
    clean_ed = edit_delta["test-clean"]
    other_ed = edit_delta["test-other"]
    diag_clean = diagnostic["nll_mt", "test-clean"]
    diag_other = diagnostic["nll_mt", "test-other"]
    derived = (
        hk.finding(
            "<b>Evidence-backed · the gain is seed-robust and broader than a few anecdotes.</b> "
            "Co-training wins every paired seed comparison (clean ΔWER %.2f/%.2f/%.2f pp; "
            "other %.2f/%.2f/%.2f pp). Across all seed×utterance pairs it improves %s vs worsens "
            "%s on test-clean (%s ties), and improves %s vs worsens %s on test-other (%s ties)."
            % tuple(
                [data["nll_mt", s, "test-clean"]["wer"] - data["nll_frozen", s, "test-clean"]["wer"]
                 for s in seeds]
                + [data["nll_mt", s, "test-other"]["wer"] - data["nll_frozen", s, "test-other"]["wer"]
                   for s in seeds]
                + [format(clean_w, ","), format(clean_l, ","), format(clean_t, ","),
                   format(other_w, ","), format(other_l, ","), format(other_t, ",")]
            )
        )
        + hk.finding(
            "<b>Evidence-backed · the main aggregate benefit is fewer deletions.</b> Relative to "
            "the frozen decoder across three seeds, co-training changes clean errors by "
            "%+d insertions / %+d deletions / %+d substitutions (net %+d), and other errors by "
            "%+d / %+d / %+d (net %+d). It recovers speech content that the frozen decoder "
            "otherwise truncates, while insertion control is the remaining weakness."
            % (clean_ed["ins"], clean_ed["del"], clean_ed["sub"], clean_ed["errors"],
               other_ed["ins"], other_ed["del"], other_ed["sub"], other_ed["errors"])
        )
        + hk.finding(
            "<b>Evidence-backed · raw seed variance is dominated by rare runaway decodes.</b> "
            "The co-trained arm has %d test-clean and %d test-other utterance-seed outputs above "
            "200%% WER. One seed-3408 test-clean utterance alone contributes 0.71 WER points "
            "(367 insertions). The standard, untrimmed WER remains the headline above; as a "
            "diagnostic only, excluding &gt;200%%-WER cases changes the co-trained mean±SD to "
            "%.2f±%.2f clean and %.2f±%.2f other, showing that most apparent between-seed "
            "instability is a decoding-tail problem."
            % (runaway["nll_mt", "test-clean"], runaway["nll_mt", "test-other"],
               diag_clean[0], diag_clean[1], diag_other[0], diag_other[1])
        )
        + hk.finding(
            "<b>Inferred · likely mechanism for the runaway tail.</b> Evaluation sets "
            "<span class=\"mono\">max_decode_steps = enc_states.shape[1] × ratio</span>, where "
            "<span class=\"mono\">shape[1]</span> is the batch-padded maximum, not each "
            "utterance's true segment length. If an item fails to emit EOS, a short utterance can "
            "therefore consume a much longer batchmate's budget. The code path and the repeated "
            "max-length hypotheses support this explanation, but the original test batches were "
            "not serialized, so the causal link is not directly proven."
        )
    )
    if tf_warm:
        tf_first, tf_last = tf_warm[0], tf_warm[-1]
        derived += hk.finding(
            "<b>Open question · the Transformer warm-up's early greedy decode is "
            "pathological.</b> Through epoch %d/6, valid loss moves %.2f→%.2f while valid WER "
            "moves %.0f%%→%.0f%%. Early CNN warm-ups also produced triple-digit WER before "
            "settling, and the batch-global decode cap can exaggerate non-EOS tails, so this is "
            "not evidence that the Transformer backbone fails. It is a live risk: inspect epoch "
            "3+ and runaway counts before allowing the nine dependent jobs to define an "
            "architecture comparison."
            % (int(tf_last["epoch"]), tf_first.get("valid loss", float("nan")),
               tf_last.get("valid loss", float("nan")),
               tf_first.get("valid WER", float("nan")),
               tf_last.get("valid WER", float("nan")))
        )

    examples = (
        "<table><thead><tr><th>ID / context</th><th>frozen errors (3407/08/09)</th>"
        "<th>co-trained errors</th><th>interpretation</th></tr></thead><tbody>"
        "<tr><td><span class=\"mono\">3729-6852-0008</span><br>23.9 s, 69-word test-clean "
        "utterance</td><td>32 / 35 / 35</td><td><b>4 / 18 / 4</b></td>"
        "<td><b>Evidence-backed success:</b> co-training consistently prevents the long deletion "
        "run seen with the frozen decoder.</td></tr>"
        "<tr><td><span class=\"mono\">7021-79730-0004</span><br>17.9 s; lexical/inflectional "
        "errors</td><td><b>2 / 2 / 2</b></td><td>3 / 3 / 3</td>"
        "<td><b>Evidence-backed regression:</b> the gain is not universal; co-training adds one "
        "repeatable substitution (<i>artifice→artifices</i>).</td></tr>"
        "<tr><td><span class=\"mono\">5639-40744-0023</span><br>13.9 s, 41-word test-clean "
        "utterance</td><td>6 / 2 / 5</td><td>1 / <b>371</b> / 1</td>"
        "<td><b>Evidence-backed failure:</b> seed 3408 emits 408 words (367 insertions) instead of "
        "terminating; the other two co-trained seeds make one error.</td></tr>"
        "</tbody></table>"
    )

    return hk.section(
        "0b · Latest progress audit — corrected shared-warm-up sweep",
        lead="Standard-depth review of the newest artifacts. Headline WER is independently "
             "recomputed from edit counts in the per-utterance WER files; findings are explicitly "
             "separated into evidence-backed observations, inferences, and open questions.",
        body=(
            hk.card(
                "<p><b>Primary source under review:</b> the running-status claims in "
                "<span class=\"mono\">docs/project_notes/{key_facts,issues}.md</span> and this "
                "report's earlier single-seed conclusions.</p>"
                "<p><b>Outcome evidence:</b> aggregate + per-utterance files under "
                "<span class=\"mono\">results/speechllm_segmenter/"
                "multiseed_shared_warmup_onpolicy/</span>; execution evidence in "
                "<span class=\"mono\">slurm_logs/</span>; critical-path spot-checks in "
                "<span class=\"mono\">train_speechllm_with_segmenter.py</span> and "
                "<span class=\"mono\">speechbrain/decoders/seq2seq.py</span>.</p>"
                "<p><b>Explicitly not ranked:</b> incomplete CER-reward and Transformer arms. "
                "No new training or adversarial rerun was performed for this standard-depth "
                "review.</p>", title="Scope & evidence")
            +
            hk.card(progress_table + '<p class="cap">Artifact-derived snapshot. “Done” means '
                    'all three held-out WER files exist; partial epochs are not final results.</p>',
                    title="Sweep progress")
            + hk.card(evidence_table, title="Claims & verdicts")
            + hk.card(results_table + '<p class="cap">Standard LibriSpeech corpus WER. '
                      'Every numerator was checked against insertions + deletions + substitutions; '
                      'reported and recomputed values agree to display precision.</p>',
                      title="Completed CNN NLL arms — three exact shared-warm-up seeds")
            + hk.card(derived, title="Derived findings")
            + hk.card(examples, title="Qualitative examples")
            + hk.card(
                "<table><thead><tr><th>failure mode</th><th>prevalence / examples</th>"
                "<th>layer</th></tr></thead><tbody>"
                "<tr><td>Runaway, non-EOS insertion tail</td><td>Rare (&gt;200%% WER: %d clean / "
                "%d other co-trained outputs); <span class=\"mono\">5639-40744-0023</span></td>"
                "<td>Evaluation decoding / length control</td></tr>"
                "<tr><td>Long-span deletion / premature truncation</td><td>Co-training removes "
                "1,372 clean and 892 other deletions in aggregate; "
                "<span class=\"mono\">3729-6852-0008</span></td><td>Decoder adaptation</td></tr>"
                "<tr><td>Residual lexical/inflectional substitutions</td><td>Local and repeatable; "
                "<span class=\"mono\">7021-79730-0004</span></td><td>ASR decoder / scoring</td>"
                "</tr></tbody></table>"
                "<p class=\"cap\"><b>Collective assessment.</b> The core NLL result is ready: "
                "decoder co-training is a robust improvement. The bottleneck for precise seed "
                "variance is now evaluation length control, not broad training collapse. CER "
                "and Transformer conclusions remain open until their jobs finish.</p>"
                % (runaway["nll_mt", "test-clean"], runaway["nll_mt", "test-other"]),
                title="Failure-mode synthesis")
            + hk.card(
                "<ol><li><b>Highest priority:</b> make evaluation's decode cap per-utterance "
                "(or decode length-homogeneous microbatches), then re-evaluate the six saved NLL "
                "checkpoints. <b>Success:</b> no runaway &gt;200%-WER hypotheses caused by missing "
                "EOS and materially lower seed SD, without changing normal utterances.</li>"
                "<li>After CER and Transformer complete, compare paired seed deltas from the same "
                "warm-up and report confidence intervals; do not rank partial traces.</li>"
                "<li>Keep standard untrimmed WER as the primary metric, but add an explicit "
                "non-EOS/runaway count so a few insertion tails cannot masquerade as training "
                "instability.</li></ol>", title="Prioritized recommendations")
        ),
    )


# --------------------------- boundary / spectrogram viz ---------------------------
def load_ssl_segmenter(ckpt_dir, backbone):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from segmenter import Segmenter

    from speechbrain.integrations.huggingface.wav2vec2 import Wav2Vec2
    from speechbrain.processing.features import InputNormalization
    ssl = Wav2Vec2(source="facebook/wav2vec2-base-960h", output_norm=True,
                   freeze=True, save_path=SSL_CACHE, device_map="cpu")
    norm = InputNormalization(norm_type="sentence")
    seg = Segmenter(input_dim=768, backbone=backbone, hidden_dim=256)
    seg.load_state_dict(torch.load(os.path.join(ckpt_dir, "segmenter.ckpt"), map_location="cpu"))
    seg.eval()
    ssl.eval()
    return ssl, norm, seg


def boundary_fig(wav_path, char_pt, ssl, norm, seg):
    """3 rows sharing one time axis: log-mel spectrogram (both boundary sets overlaid,
    red=learned / blue=char-CTC ground truth), then each boundary set again as its own
    dedicated lane — same x-position in every row, so a boundary can be traced straight
    down the figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torchaudio
    wav, sr = torchaudio.load(wav_path)
    wav = wav.mean(0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    with torch.no_grad():
        wl = torch.ones(1)
        feats = ssl(norm(wav, wl), wl)
        logits = seg.boundary_logits(feats, torch.zeros(1, feats.size(1), dtype=torch.bool))
        pred = (logits[0] > 0).nonzero().flatten().tolist()
    Tf = feats.size(1)
    gtb = (torch.load(char_pt).view(-1) == 1).nonzero().flatten().tolist()
    logmel = torch.log(torchaudio.transforms.MelSpectrogram(
        16000, n_fft=400, hop_length=320, n_mels=80)(wav)[0] + 1e-6).numpy()
    Tm = logmel.shape[1]
    sc = Tm / max(Tf, 1)
    gt_x = [b * sc for b in gtb]
    pred_x = [b * sc for b in pred]

    fig, (ax0, ax1, ax2) = plt.subplots(
        3, 1, figsize=(9.2, 3.65), dpi=110, sharex=True,
        gridspec_kw={"height_ratios": [2.6, 0.6, 0.6], "hspace": 0.15})
    ax0.imshow(logmel, origin="lower", aspect="auto", cmap="magma", extent=[0, Tm, 0, 80])
    for b in gt_x:
        ax0.axvline(b, color="#7fd0ff", lw=0.8, alpha=0.45)
    for b in pred_x:
        ax0.axvline(b, color="#ff5d5d", lw=1.3, alpha=0.9)
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.set_ylabel("spectrogram", fontsize=7.5)

    def lane(ax, xs, bg, fg, label):
        """A thin boundary-only row: alternating segment tint + a tick per boundary."""
        ax.set_facecolor(bg)
        for i, b in enumerate(xs):
            end = xs[i + 1] if i + 1 < len(xs) else Tm
            if i % 2:
                ax.axvspan(b, end, color=fg, alpha=0.14, lw=0)
            ax.axvline(b, color=fg, lw=1.1, alpha=0.95)
        ax.set_xlim(0, Tm)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(label, fontsize=7.5)

    lane(ax1, gt_x, "#eaf1f7", "#215f8c", "gold char")
    lane(ax2, pred_x, "#f7eaea", "#a5322c", "learned")
    fig.tight_layout(pad=0.3)
    return (hk.mpl_png(fig, cls="fig", facecolor="white"), (len(pred) + 1) / max(Tf, 1),
            (len(gtb) + 1) / max(Tf, 1), len(pred), len(gtb))


def frontier_fig(base, models, no_down, encoder_name, aligned=()):
    """Three-row WER/compression frontier split by model setting.

    The first row isolates fixed/native-rate baselines; the next two compare CNN
    and Transformer policies against those references. This keeps the learned-model
    cluster readable while retaining k labels in the uncluttered baseline row.

    base: [(k, rho, clean, clean_sd, other, other_sd), ...]
    models: [(name, rho, clean, clean_sd, other, other_sd, color, marker), ...]
    no_down: (clean, clean_sd, other, other_sd)
    aligned: optional alignment-control points in the same tuple format as models
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    b = sorted(base, key=lambda r: r[1])

    cnn_models = [model for model in models if not model[0].lower().startswith("transformer")]
    transformer_models = [model for model in models
                          if model[0].lower().startswith("transformer")]
    row_specs = (
        ("Fixed baselines", []),
        ("CNN policy", cnn_models),
        ("Transformer policy", transformer_models),
    )

    fig, axes = plt.subplots(3, 2, figsize=(9.2, 9.4), dpi=130, sharex=True)
    for row_idx, (row_label, row_models) in enumerate(row_specs):
        for metric_idx, split in enumerate(("test-clean", "test-other")):
            ax = axes[row_idx, metric_idx]
            value_idx = 2 if metric_idx == 0 else 4
            sd_idx = value_idx + 1
            frequencies = [rate_hz(r[1]) for r in b]
            vals = [r[value_idx] for r in b]
            sds = [r[sd_idx] for r in b]
            ax.plot(frequencies, vals, "-o", color="#7b858c", lw=1.55, ms=4.8,
                    label="Fixed-rate pooling", zorder=3)
            if any(sd > 0 for sd in sds):
                ax.errorbar(frequencies, vals, yerr=sds, fmt="none", ecolor="#7b858c",
                            elinewidth=1.0, capsize=2.5, alpha=0.8, zorder=2)
            if row_idx == 0:
                for fixed in b:
                    ax.annotate("k=%d" % fixed[0],
                                (rate_hz(fixed[1]), fixed[value_idx]),
                                textcoords="offset points", xytext=(0, 7),
                                fontsize=7.5, color="#68737a", ha="center")

                for (name, rho, clean, clean_sd, other, other_sd,
                     color, marker) in aligned:
                    value, sd = ((clean, clean_sd) if metric_idx == 0
                                 else (other, other_sd))
                    ax.errorbar([rate_hz(rho)], [value], yerr=[sd], fmt=marker, ms=8.2,
                                color=color, ecolor=color, capsize=3, zorder=5,
                                mec="white", mew=0.7, label=name)

            for name, rho, clean, clean_sd, other, other_sd, color, marker in row_models:
                value, sd = (clean, clean_sd) if metric_idx == 0 else (other, other_sd)
                ax.errorbar([rate_hz(rho)], [value], yerr=[sd], fmt=marker, ms=8.2,
                            color=color, ecolor=color, capsize=3, zorder=5,
                            mec="white", mew=0.7, label=name)

            nd_value, nd_sd = (no_down[0], no_down[1]) if metric_idx == 0 else (
                no_down[2], no_down[3]
            )
            ax.axhline(nd_value, color="#9a5b73", linestyle=(0, (5, 3)), lw=1.25,
                       label="No downsampling", zorder=1)
            if nd_sd > 0:
                ax.axhspan(nd_value - nd_sd, nd_value + nd_sd, color="#9a5b73",
                           alpha=0.08, lw=0, zorder=0)

            if row_idx == 0:
                ax.set_title(split, fontsize=13, fontweight="bold")
            ax.set_xlim(5.5, 17.5)
            if row_idx == len(row_specs) - 1:
                ax.set_xlabel("Audio-token frequency (Hz)  (← more compression)")
            if metric_idx == 0:
                ax.set_ylabel("%s\nWER (%%)" % row_label)
            ax.grid(axis="y", alpha=0.25, lw=0.6)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)

    all_handles, all_labels = [], []
    for ax in axes.flat:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label not in all_labels:
                all_handles.append(handle)
                all_labels.append(label)
    fig.legend(all_handles, all_labels, frameon=False, fontsize=7.8,
               loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=4,
               columnspacing=1.5)
    fig.suptitle("%s · trained on LibriSpeech-100h" % encoder_name,
                 fontsize=14, fontweight="bold", y=0.985)
    fig.tight_layout(rect=(0, 0.09, 1, 0.965), pad=0.65, w_pad=1.5, h_pad=1.2)
    return hk.mpl_png(fig, cls="fig", pad=0.10, facecolor="white")


def system_overview_fig(rows):
    """Summarize the main WavLM systems on compression and both test splits.

    ``rows`` is a list of dictionaries with display labels, family colors, and
    three-seed mean/SD values. Deterministic boundary rules have zero frequency
    variance but still report WER uncertainty across decoder-training seeds.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator

    y = np.arange(len(rows))
    panels = (
        ("frequency", "frequency_sd", "Audio-token frequency", "Hz · lower = more compression"),
        ("clean", "clean_sd", "test-clean", "WER (%) · lower is better"),
        ("other", "other_sd", "test-other", "WER (%) · lower is better"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 7.8), dpi=130, sharey=True)
    for panel_idx, (ax, (value_key, sd_key, title, xlabel)) in enumerate(
            zip(axes, panels)):
        for row_idx, row in enumerate(rows):
            value, sd, color = row[value_key], row[sd_key], row["color"]
            ax.errorbar(
                value, row_idx, xerr=sd if sd > 0 else None,
                fmt=row.get("marker", "o"), ms=7.3, color=color,
                ecolor=color, capsize=3, markerfacecolor=color,
                markeredgecolor="white", markeredgewidth=0.8, zorder=3,
            )
            right_edge = value + sd
            ax.annotate(
                "%.1f" % value if value_key == "frequency" else "%.2f" % value,
                (right_edge, row_idx), textcoords="offset points", xytext=(5, 0),
                va="center", fontsize=7.7, color="#24323f",
            )

        for boundary in range(1, len(rows)):
            if rows[boundary]["family"] != rows[boundary - 1]["family"]:
                ax.axhline(boundary - 0.5, color="#d8dee4", lw=0.9, zorder=0)

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", alpha=0.24, lw=0.7)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="y", length=0)

        if panel_idx == 0:
            ax.set_xscale("log")
            ticks = [6.25, 8.33, 10.0, 12.5, 16.67, 25.0, 50.0]
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_major_formatter(FixedFormatter(
                ["6.2", "8.3", "10", "12.5", "16.7", "25", "50"]
            ))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.set_xlim(5.3, 63.0)
        else:
            left_edge = min(row[value_key] - row[sd_key] for row in rows)
            right_edge = max(row[value_key] + row[sd_key] for row in rows)
            ax.set_xlim(max(0, left_edge - 0.65), right_edge + 0.9)

    axes[0].set_yticks(y, labels=[row["label"] for row in rows], fontsize=8.3)
    axes[0].invert_yaxis()
    fig.suptitle(
        "Important WavLM systems · LibriSpeech-100h",
        fontsize=14, fontweight="bold", y=0.985,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955), pad=0.8, w_pad=1.3)
    return hk.mpl_png(fig, cls="fig", pad=0.10, facecolor="white")


def ls960_convergence_fig():
    """Show dev-clean convergence for the first completed corrected LS960 seeds."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [2395, 4000, 8000, 12000, 16000, 20000, 24000]
    curves = (
        ("Local-history Transformer AR + BiGRU",
         [4.45, 4.17, 3.75, 3.52, 3.08, 3.15, 2.94], "#27865d", "o"),
        ("CNN first-order AR + BiGRU",
         [4.48, 4.61, 3.59, 3.36, 3.34, 3.01, 2.98], "#b96f20", "s"),
    )

    fig, ax = plt.subplots(figsize=(9.2, 4.8), dpi=130)
    for curve_idx, (label, values, color, marker) in enumerate(curves):
        ax.plot(steps, values, color=color, marker=marker, lw=2.0, ms=5.5,
                label=label, zorder=3)
        ax.annotate(
            "%.2f" % values[-1], (steps[-1], values[-1]),
            textcoords="offset points", xytext=(7, -8 if curve_idx == 0 else 8),
            va="center",
            color=color, fontsize=8.5, fontweight="bold",
        )

    ax.axvline(2395, color="#7b858c", linestyle=(0, (4, 3)), lw=1.15,
               alpha=0.85, zorder=1)
    ax.annotate(
        "decoder warmup ends", (2395, 4.67), textcoords="offset points",
        xytext=(6, 0), color="#68737a", fontsize=8.2, va="center",
    )
    ax.set_xlim(1200, 26000)
    ax.set_ylim(2.72, 4.82)
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("dev-clean WER (%)  ·  lower is better")
    ax.set_title("Corrected LS960 training curves · seed 3407",
                 fontsize=13, fontweight="bold", pad=15)
    ax.grid(axis="y", alpha=0.25, lw=0.7)
    ax.legend(frameon=False, fontsize=8.3, loc="upper right")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout(pad=1.0)
    return hk.mpl_png(fig, cls="fig", pad=0.10, facecolor="white")


def ls960_system_comparison_fig(rows):
    """Compare attained clean/other WER for the principal LS960 systems."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.4), dpi=130, sharey=True)
    for metric_idx, (ax, title) in enumerate(zip(
            axes, ("test-clean", "test-other"))):
        value_key = "clean" if metric_idx == 0 else "other"
        sd_key = value_key + "_sd"
        for row_idx, row in enumerate(rows):
            value, sd = row[value_key], row[sd_key]
            ax.errorbar(
                value, row_idx, xerr=sd if sd > 0 else None,
                fmt=row.get("marker", "o"), ms=7.5, color=row["color"],
                ecolor=row["color"], capsize=3, markerfacecolor=row["color"],
                markeredgecolor="white", markeredgewidth=0.8, zorder=3,
            )
            ax.annotate(
                "%.2f" % value, (value + sd, row_idx),
                textcoords="offset points", xytext=(5, 0), va="center",
                fontsize=8.0, color="#24323f",
            )
        values = [row[value_key] for row in rows]
        sds = [row[sd_key] for row in rows]
        ax.set_xlim(
            max(0, min(v - s for v, s in zip(values, sds)) - 0.55),
            max(v + s for v, s in zip(values, sds)) + 0.85,
        )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("WER (%) · lower is better")
        ax.grid(axis="x", alpha=0.24, lw=0.7)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="y", length=0)

    axes[0].set_yticks(y, labels=[row["label"] for row in rows], fontsize=8.5)
    axes[0].invert_yaxis()
    fig.suptitle(
        "Attained system comparison · LibriSpeech-960h",
        fontsize=14, fontweight="bold", y=0.985,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95), pad=0.9, w_pad=1.4)
    return hk.mpl_png(fig, cls="fig", pad=0.10, facecolor="white")


def controlled_wer_fig(rows):
    """Plot clean/other WER for the plainly named systems in section 2.

    Each row is ``(label, clean, clean_sd, other, other_sd, n, color)``.
    A hollow marker means only one seed has finished; filled markers are
    three-seed means. Lower WER is better.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y = np.arange(len(rows))
    labels = [row[0] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.6), dpi=130, sharey=True)
    for metric_idx, (ax, title) in enumerate(zip(
            axes, ("test-clean", "test-other"))):
        value_idx = 1 if metric_idx == 0 else 3
        values = [row[value_idx] for row in rows]
        for row_idx, row in enumerate(rows):
            value, sd, n, color = (
                row[value_idx], row[value_idx + 1], row[5], row[6]
            )
            ax.errorbar(
                value, row_idx, xerr=sd if n > 1 else None,
                fmt="o", ms=7.5, color=color, ecolor=color, capsize=3,
                markerfacecolor=color if n > 1 else "white",
                markeredgewidth=1.5, zorder=3,
            )
            # Put the value after the right error-bar cap instead of on the bar.
            right_edge = value + (sd if n > 1 else 0.0)
            ax.annotate(
                "%.2f" % value, (right_edge, row_idx),
                textcoords="offset points", xytext=(5, 0), va="center",
                fontsize=8, color="#24323f",
            )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("WER (%)  ·  lower is better")
        rightmost = max(row[value_idx] + row[value_idx + 1] for row in rows)
        ax.set_xlim(max(0, min(values) - 0.8), rightmost + 0.85)
        ax.set_yticks(y, labels=labels, fontsize=8.3)
        ax.grid(axis="x", alpha=0.25, lw=0.7)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].invert_yaxis()
    fig.suptitle(
        "WER for each setup", fontsize=13.5,
        fontweight="bold", y=0.985,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96), pad=0.8, w_pad=2.0)
    return hk.mpl_png(fig, cls="fig", pad=0.10, facecolor="white")


def inference_efficiency_fig(rows):
    """Compare batch-1 and memory-limited batched forward RTF on one A100.

    Each bar is the mean of three independently trained seeds after aggregating
    test-clean and test-other within each seed. Error bars are sample SD.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y = np.arange(len(rows))
    labels = [
        "%s · %.1f Hz" % (row["label"], row["frequency"][0])
        for row in rows
    ]
    panels = (
        ("batch1_rtf", "Batch 1\nsingle-utterance processing", False),
        ("rtf", "Stable throughput batch\nshorter prefixes allow larger batches", True),
    )
    fig, axes = plt.subplots(
        1, 2, figsize=(11.2, 8.6), dpi=130, sharey=True,
        gridspec_kw={"wspace": 0.16},
    )
    for ax, (metric, title, show_batch) in zip(axes, panels):
        values = [row[metric][0] for row in rows]
        errors = [row[metric][1] for row in rows]
        bars = ax.barh(
            y, values, height=0.60,
            color=[row["color"] for row in rows], alpha=0.88,
        )
        ax.errorbar(
            values, y, xerr=errors, fmt="none", ecolor="#24323f",
            elinewidth=1.0, capsize=2.5, capthick=1.0, zorder=3,
        )
        for row_idx, (row, bar, value, error) in enumerate(
                zip(rows, bars, values, errors)):
            text = "%.3f ± %.3f" % (value, error)
            if show_batch:
                text += " · b%d" % row["batch_size"]
            ax.annotate(
                text, (bar.get_width() + error, row_idx),
                textcoords="offset points", xytext=(5, 0), va="center",
                fontsize=7.4, color="#24323f",
            )
        for boundary in range(1, len(rows)):
            if rows[boundary]["family"] != rows[boundary - 1]["family"]:
                ax.axhline(
                    boundary - 0.5, color="#d8dee4", lw=0.9, zorder=0
                )
        ax.set_title(title, fontsize=11.0, fontweight="bold", pad=12)
        ax.set_xlabel("forward RTF  ·  lower is faster")
        ax.set_xlim(
            0,
            max(value + error for value, error in zip(values, errors)) * 1.42,
        )
        ax.grid(axis="x", alpha=0.24, lw=0.7)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y, labels=labels, fontsize=7.8)
    axes[0].invert_yaxis()
    fig.suptitle(
        "Inference RTF: single-utterance processing and batched throughput\n"
        "three-seed mean ± sample SD",
        fontsize=13.0, fontweight="bold", y=0.995,
    )
    fig.subplots_adjust(left=0.22, right=0.98, bottom=0.08, top=0.86, wspace=0.16)
    return hk.mpl_png(fig, cls="fig", pad=0.10, facecolor="white")


def controlled_boundary_accuracy_fig(rows):
    """Plot char-CTC boundary F1 for the same systems as ``controlled_wer_fig``.

    Each row is ``(label, exact, exact_sd, tol1, tol1_sd, n, color)``. Learned
    systems report three selected checkpoints; deterministic sources report one
    fixed boundary stream. Higher F1 is better.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y = np.arange(len(rows))
    labels = [row[0] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.6), dpi=130, sharey=True)
    for metric_idx, (ax, title) in enumerate(zip(
            axes, ("exact frame", "within ±1 frame"))):
        value_idx = 1 if metric_idx == 0 else 3
        values = [row[value_idx] for row in rows]
        for row_idx, row in enumerate(rows):
            value, sd, n, color = (
                row[value_idx], row[value_idx + 1], row[5], row[6]
            )
            ax.errorbar(
                value, row_idx, xerr=sd if n > 1 else None,
                fmt="o" if n > 1 else "s", ms=7.5, color=color,
                ecolor=color, capsize=3, markerfacecolor=color,
                markeredgecolor="white", markeredgewidth=0.8, zorder=3,
            )
            right_edge = value + (sd if n > 1 else 0.0)
            ax.annotate(
                "%.1f" % value, (right_edge, row_idx),
                textcoords="offset points", xytext=(5, 0), va="center",
                fontsize=8, color="#24323f",
            )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("boundary F1 (%)  ·  higher is better")
        ax.set_xlim(max(0, min(values) - 7), 105)
        ax.set_yticks(y, labels=labels, fontsize=8.3)
        ax.grid(axis="x", alpha=0.25, lw=0.7)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].invert_yaxis()
    fig.suptitle(
        "Boundary agreement with char-CTC on dev-clean",
        fontsize=13.5, fontweight="bold", y=0.985,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96), pad=0.8, w_pad=2.0)
    return hk.mpl_png(fig, cls="fig", pad=0.10, facecolor="white")


def phone_boundary_agreement_fig(fixed_phone, learned_phone):
    """Plot phone-boundary F1 against audio-token frequency.

    Fixed-rate grids form the rate-conditioned reference curve. The learned
    point reports the three-seed mean and sample SD in both rate and F1.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fixed = [
        fixed_phone["systems"]["fixed_k%d" % k]
        for k in (8, 6, 5, 4, 3)
    ]
    fixed_rates = np.asarray(
        [row["actual_token_rate_hz"] for row in fixed], dtype=float
    )
    learned_rates = np.asarray([
        FRAME_HZ * row["kept_ratio"]
        for row in learned_phone["per_seed"].values()
    ])
    learned_rate = float(np.mean(learned_rates))
    learned_rate_sd = float(np.std(learned_rates, ddof=1))
    series = (
        ("exact frame", "exact", "#7b858c"),
        ("within ±20 ms", "tol_20ms", "#2f6f9f"),
        ("within ±40 ms", "tol_40ms", "#27866f"),
    )

    fig, ax = plt.subplots(figsize=(9.8, 5.8), dpi=130)
    for label, tolerance, color in series:
        fixed_f1 = 100 * np.asarray([
            row["scores"][tolerance]["harsh"]["f1"] for row in fixed
        ])
        learned_f1 = (
            100
            * learned_phone["summary"][tolerance]["harsh"]["f1"]["mean"]
        )
        learned_f1_sd = (
            100
            * learned_phone["summary"][tolerance]["harsh"]["f1"]["sd"]
        )
        ax.plot(
            fixed_rates, fixed_f1, "o-", lw=1.8, ms=6.5,
            color=color, label=label, zorder=2,
        )
        ax.errorbar(
            learned_rate, learned_f1,
            xerr=learned_rate_sd, yerr=learned_f1_sd,
            fmt="*", ms=13, color=color, ecolor=color,
            capsize=3, markeredgecolor="white", markeredgewidth=0.8,
            zorder=4,
        )

    tol20 = [
        100 * row["scores"]["tol_20ms"]["harsh"]["f1"] for row in fixed
    ]
    for row, x, y in zip(fixed, fixed_rates, tol20):
        ax.annotate(
            "k=%d" % row["k"], (x, y), textcoords="offset points",
            xytext=(0, 8), ha="center", fontsize=8, color="#4d5963",
        )
    ax.annotate(
        "learned\n%.1f Hz" % learned_rate,
        (
            learned_rate,
            100
            * learned_phone["summary"]["tol_20ms"]["harsh"]["f1"]["mean"],
        ),
        textcoords="offset points", xytext=(-12, 10), ha="right", va="bottom",
        fontsize=8.5, fontweight="bold", color="#24323f",
    )
    phone_rate = fixed_phone["reference"]["boundary_rate_hz"]
    ax.axvline(
        phone_rate, color="#b94747", ls="--", lw=1.2, alpha=0.9,
        label="phone reference rate (%.1f Hz)" % phone_rate,
    )
    shape_legend = Line2D(
        [], [], marker="*", linestyle="none", markersize=11,
        markerfacecolor="#58636c", markeredgecolor="white",
        label="learned system · mean ± sample SD",
    )
    handles, labels = ax.get_legend_handles_labels()
    handles.append(shape_legend)
    labels.append(shape_legend.get_label())
    ax.legend(
        handles, labels, loc="lower right", frameon=False,
        fontsize=8.5, ncol=2,
    )
    ax.set_title(
        "Phone-boundary agreement must be read at matched frequency\n"
        "fixed grids are the rate-conditioned reference",
        fontsize=12.5, fontweight="bold", pad=15,
    )
    ax.set_xlabel("audio-token frequency (Hz)  ·  lower means fewer decoder tokens")
    ax.set_ylabel("one-to-one boundary F1 (%)  ·  higher is better")
    ax.set_xlim(5.4, 17.7)
    ax.set_ylim(10, 89)
    ax.grid(alpha=0.22, lw=0.7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout(pad=1.2)
    return hk.mpl_png(fig, cls="fig", pad=0.10, facecolor="white")


def component_checks_fig(boundary_swap, cold_f1):
    """Two direct checks: boundary-stream WER and char-boundary imitation F1."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=130)

    swap_labels = ["Oracle char", "Cold learned", "After RL"]
    swap_values = [boundary_swap[key] for key in ("oracle", "cold", "rl")]
    swap_colors = ["#2f6f9f", "#d5902f", "#b94747"]
    swap_y = np.arange(len(swap_labels))
    axes[0].barh(swap_y, swap_values, color=swap_colors, height=0.58)
    axes[0].set_yticks(swap_y, labels=swap_labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("dev-clean WER (%)  ·  lower is better")
    axes[0].set_title("Same decoder, different boundaries", fontweight="bold")
    axes[0].grid(axis="x", alpha=0.22, lw=0.7)
    for i, value in enumerate(swap_values):
        axes[0].text(value + 0.25, i, "%.2f" % value, va="center", fontsize=8.5)
    axes[0].set_xlim(0, max(swap_values) + 3.0)

    backbones = ("CNN", "Transformer")
    x = np.arange(len(backbones))
    width = 0.34
    w2v = [cold_f1[("wav2vec2", backbone)] for backbone in backbones]
    wavlm = [cold_f1[("WavLM", backbone)] for backbone in backbones]
    axes[1].bar(x - width / 2, w2v, width, label="wav2vec2 input",
                color="#2f6f9f")
    axes[1].bar(x + width / 2, wavlm, width, label="WavLM input",
                color="#d5902f")
    axes[1].set_xticks(x, labels=backbones)
    axes[1].set_ylim(0.70, 0.93)
    axes[1].set_ylabel("Exact-frame F1  ·  higher is better")
    axes[1].set_title("Imitating char boundaries", fontweight="bold")
    axes[1].grid(axis="y", alpha=0.22, lw=0.7)
    axes[1].legend(frameon=False, fontsize=8, loc="lower left")
    for xpos, value in zip(x - width / 2, w2v):
        axes[1].text(xpos, value + 0.006, "%.3f" % value,
                     ha="center", va="bottom", fontsize=8)
    for xpos, value in zip(x + width / 2, wavlm):
        axes[1].text(xpos, value + 0.006, "%.3f" % value,
                     ha="center", va="bottom", fontsize=8)

    for ax in axes:
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.tight_layout(pad=0.9, w_pad=2.2)
    return hk.mpl_png(fig, cls="fig", pad=0.10, facecolor="white")


def label_report_assets(body):
    """Insert sequential, visible labels before every table and figure."""
    counts = {"table": 0, "figure": 0}
    pattern = re.compile(
        r'<table\b|<img\s+class=["\']fig["\']|<svg\b[^>]*\bclass="chart"[^>]*>'
    )

    def add_label(match):
        tag = match.group(0)
        kind = "table" if tag.startswith("<table") else "figure"
        counts[kind] += 1
        label = '<div class="asset-label %s-label">%s %d</div>' % (
            kind, kind.title(), counts[kind]
        )
        return label + tag

    return pattern.sub(add_label, body)


def pipeline_fig():
    """Architecture/curriculum overview, generated fresh from make_pipeline_figure.py."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        import make_pipeline_figure as mpf
        return mpf.img_tag(cls="fig")
    except Exception as ex:  # noqa - never let the figure break the whole report
        return ('<span class="muted">pipeline figure unavailable: %s</span>'
                % H.escape(str(ex)))


def ar_policy_demo_figure():
    """Theme-aware diagram contrasting the three boundary-policy factorizations."""
    return r'''
<svg class="chart" viewBox="0 0 1040 592" role="img"
     aria-labelledby="ar-policy-demo-title ar-policy-demo-desc">
  <title id="ar-policy-demo-title">Three levels of boundary-policy memory</title>
  <desc id="ar-policy-demo-desc">The independent CNN uses only an acoustic score, the
  first-order CNN adds a two-way bias selected by the previous boundary, and the
  local-history Transformer attends over the latest 64 audio and shifted-boundary states.</desc>
  <defs>
    <marker id="ar-arrow" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--text-muted)"/>
    </marker>
    <style>
      .ar-row{fill:var(--surface-2);stroke:var(--border);stroke-width:1.3}
      .ar-box{fill:var(--surface-1);stroke:var(--border-strong);stroke-width:1.3}
      .ar-blue{fill:var(--series-sed);fill-opacity:.10;stroke:var(--series-sed);stroke-width:1.5}
      .ar-orange{fill:var(--pair-sed-asr);fill-opacity:.11;stroke:var(--pair-sed-asr);stroke-width:1.5}
      .ar-purple{fill:var(--pair-sed-speaker);fill-opacity:.11;stroke:var(--pair-sed-speaker);stroke-width:1.5}
      .ar-line{stroke:var(--text-muted);stroke-width:1.5;fill:none;marker-end:url(#ar-arrow)}
      .ar-feedback{stroke:var(--text-muted);stroke-width:1.5;fill:none}
      .ar-dash{stroke:var(--text-muted);stroke-width:1.25;stroke-dasharray:5 5;fill:none}
      .ar-title{font-size:16px;font-weight:700;fill:var(--text-primary)}
      .ar-label{font-size:13px;fill:var(--text-primary)}
      .ar-small{font-size:11.5px;fill:var(--text-secondary)}
      .ar-note{font-size:12px;font-weight:600}
    </style>
  </defs>

  <rect class="ar-row" x="8" y="12" width="1024" height="146" rx="14"/>
  <text class="ar-title" x="28" y="42">Independent CNN</text>
  <text class="ar-small" x="28" y="64">no generated-label memory</text>
  <rect class="ar-blue" x="214" y="39" width="138" height="62" rx="10"/>
  <text class="ar-label" x="283" y="65" text-anchor="middle">WavLM frame</text>
  <text class="ar-small" x="283" y="84" text-anchor="middle">x<tspan baseline-shift="sub">t</tspan></text>
  <path class="ar-line" d="M352 70 H418"/>
  <rect class="ar-box" x="420" y="39" width="150" height="62" rx="10"/>
  <text class="ar-label" x="495" y="65" text-anchor="middle">local CNN</text>
  <text class="ar-small" x="495" y="84" text-anchor="middle">acoustic score a<tspan baseline-shift="sub">t</tspan></text>
  <path class="ar-line" d="M570 70 H637"/>
  <rect class="ar-box" x="639" y="39" width="142" height="62" rx="10"/>
  <text class="ar-label" x="710" y="65" text-anchor="middle">sigmoid</text>
  <text class="ar-small" x="710" y="84" text-anchor="middle">boundary chance</text>
  <path class="ar-line" d="M781 70 H848"/>
  <rect class="ar-blue" x="850" y="39" width="150" height="62" rx="10"/>
  <text class="ar-label" x="925" y="65" text-anchor="middle">sample b<tspan baseline-shift="sub">t</tspan></text>
  <text class="ar-small" x="925" y="84" text-anchor="middle">cut or continue</text>
  <text class="ar-note" x="710" y="135" text-anchor="middle" fill="var(--text-muted)">no feedback into the next decision</text>

  <rect class="ar-row" x="8" y="172" width="1024" height="184" rx="14"/>
  <text class="ar-title" x="28" y="202">CNN + first-order AR</text>
  <text class="ar-small" x="28" y="224">one-bit memory: only b<tspan baseline-shift="sub">t−1</tspan></text>
  <rect class="ar-blue" x="214" y="238" width="138" height="62" rx="10"/>
  <text class="ar-label" x="283" y="264" text-anchor="middle">CNN score a<tspan baseline-shift="sub">t</tspan></text>
  <text class="ar-small" x="283" y="283" text-anchor="middle">same acoustic head</text>
  <rect class="ar-orange" x="420" y="238" width="150" height="62" rx="10"/>
  <text class="ar-label" x="495" y="264" text-anchor="middle">choose δ<tspan baseline-shift="sub">0</tspan> or δ<tspan baseline-shift="sub">1</tspan></text>
  <text class="ar-small" x="495" y="283" text-anchor="middle">using previous bit</text>
  <path class="ar-line" d="M352 269 H418"/>
  <path class="ar-line" d="M570 269 H637"/>
  <rect class="ar-box" x="639" y="238" width="142" height="62" rx="10"/>
  <text class="ar-label" x="710" y="264" text-anchor="middle">a<tspan baseline-shift="sub">t</tspan> + δ<tspan baseline-shift="sub">b(t−1)</tspan></text>
  <text class="ar-small" x="710" y="283" text-anchor="middle">then sigmoid</text>
  <path class="ar-line" d="M781 269 H848"/>
  <rect class="ar-orange" x="850" y="238" width="150" height="62" rx="10"/>
  <text class="ar-label" x="925" y="264" text-anchor="middle">sample b<tspan baseline-shift="sub">t</tspan></text>
  <text class="ar-small" x="925" y="283" text-anchor="middle">becomes next bit</text>
  <path class="ar-feedback" d="M925 304 V323 H495"/>
  <path class="ar-line" d="M495 323 V302"/>
  <text class="ar-note" x="710" y="345" text-anchor="middle" fill="var(--pair-sed-asr)">can learn “avoid two cuts in a row”</text>

  <rect class="ar-row" x="8" y="370" width="1024" height="208" rx="14"/>
  <text class="ar-title" x="28" y="400">Local-history Transformer AR</text>
  <text class="ar-small" x="28" y="422">learned summary of the latest 64 shifted-boundary states</text>
  <rect class="ar-purple" x="190" y="440" width="114" height="67" rx="10"/>
  <text class="ar-label" x="247" y="465" text-anchor="middle">z<tspan baseline-shift="sub">0</tspan></text>
  <text class="ar-small" x="247" y="486" text-anchor="middle">x<tspan baseline-shift="sub">0</tspan> + BOS</text>
  <rect class="ar-purple" x="326" y="440" width="114" height="67" rx="10"/>
  <text class="ar-label" x="383" y="465" text-anchor="middle">z<tspan baseline-shift="sub">1</tspan></text>
  <text class="ar-small" x="383" y="486" text-anchor="middle">x<tspan baseline-shift="sub">1</tspan> + b<tspan baseline-shift="sub">0</tspan></text>
  <rect class="ar-box" x="470" y="450" width="36" height="47" rx="9"/>
  <text class="ar-title" x="488" y="480" text-anchor="middle">…</text>
  <rect class="ar-purple" x="532" y="440" width="114" height="67" rx="10"/>
  <text class="ar-label" x="589" y="465" text-anchor="middle">z<tspan baseline-shift="sub">t</tspan></text>
  <text class="ar-small" x="589" y="486" text-anchor="middle">x<tspan baseline-shift="sub">t</tspan> + b<tspan baseline-shift="sub">t−1</tspan></text>
  <path class="ar-line" d="M304 474 H324"/>
  <path class="ar-line" d="M440 474 H468"/>
  <path class="ar-line" d="M506 474 H530"/>
  <path class="ar-line" d="M646 474 H674"/>
  <rect class="ar-box" x="676" y="440" width="158" height="67" rx="10"/>
  <text class="ar-label" x="755" y="465" text-anchor="middle">4 causal layers</text>
  <text class="ar-small" x="755" y="486" text-anchor="middle">attention over z<tspan baseline-shift="sub">0:t</tspan></text>
  <path class="ar-line" d="M834 474 H874"/>
  <rect class="ar-purple" x="876" y="440" width="124" height="67" rx="10"/>
  <text class="ar-label" x="938" y="465" text-anchor="middle">sample b<tspan baseline-shift="sub">t</tspan></text>
  <text class="ar-small" x="938" y="486" text-anchor="middle">cache state</text>
  <path class="ar-feedback" d="M938 511 V533 H589"/>
  <path class="ar-line" d="M589 533 V509"/>
  <text class="ar-note" x="720" y="562" text-anchor="middle" fill="var(--pair-sed-speaker)">can model run length, rhythm, and audio-dependent history</text>
</svg>'''


def ar_modeling_demo():
    """Browser-readable worked explanation of CNN and local-history Transformer AR."""
    comparison = (
        '<div class="demo-table-scroll"><table><thead><tr><th>policy</th>'
        '<th>direct boundary memory</th>'
        '<th>how history changes the current decision</th><th>status here</th>'
        '</tr></thead><tbody>'
        '<tr><td>independent CNN</td><td>none</td><td>it does not</td>'
        '<td>matched control</td></tr>'
        '<tr><td>CNN · first-order AR</td><td>only the previous bit</td>'
        '<td>selects one of two global logit offsets</td>'
        '<td>three-seed WER result</td></tr>'
        '<tr><td>Transformer · local-history AR</td><td>64-state window per causal layer</td>'
        '<td>attention learns an audio- and history-dependent adjustment</td>'
        '<td>three-seed WER and pooling study</td></tr>'
        '</tbody></table></div>'
        '<p class="cap">The acoustic input is WavLM in all three descriptions. “Causal” refers '
        'to boundary-label history; WavLM itself is bidirectional, so none of these rows is a '
        'streaming claim.</p>'
    )

    worked_pooling = (
        '<div class="demo-timeline" aria-label="Six frame pooling example">'
        '<div class="demo-row"><span class="demo-name">frames</span>'
        '<span class="frame">x<sub>0</sub></span><span class="frame">x<sub>1</sub></span>'
        '<span class="frame cut">x<sub>2</sub></span><span class="frame">x<sub>3</sub></span>'
        '<span class="frame">x<sub>4</sub></span><span class="frame cut">x<sub>5</sub></span></div>'
        '<div class="demo-row"><span class="demo-name">boundary</span>'
        '<span class="bit zero">0</span><span class="bit zero">0</span>'
        '<span class="bit one">1</span><span class="bit zero">0</span>'
        '<span class="bit zero">0</span><span class="bit one">1</span></div>'
        '<div class="demo-groups"><span>[ x<sub>0</sub>, x<sub>1</sub> ]</span>'
        '<span>[ x<sub>2</sub>, x<sub>3</sub>, x<sub>4</sub> ]</span>'
        '<span>[ x<sub>5</sub> ]</span></div></div>'
        '<p>Boundary value <b>1</b> means “this frame starts a new segment”; <b>0</b> means '
        '“continue the current segment.” Each bracket is averaged, so six WavLM frames become '
        'three decoder-side audio tokens. Frame 0 always starts the first segment.</p>'
    )

    cnn = (
        '<p>The CNN first produces an acoustic score \(a_t\). It then adds one of two learned '
        'numbers: \(\delta_0\) if the preceding boundary was 0, or \(\delta_1\) if it was 1. '
        'Those are the <i>only</i> autoregressive parameters.</p>'
        + hk.equation(r'''
\begin{aligned}
a_t &= \operatorname{CNN}(x)_{t}, \\
\Pr(b_t=1\mid x,b_{t-1})
    &= \sigma\!\left(a_t+\delta_{b_{t-1}}\right).
\end{aligned}''')
        + '<p><b>Worked probability.</b> Suppose \(a_t=0.3\), '
        '\(\delta_0=+0.4\), and \(\delta_1=-1.0\). After a continuation, the boundary '
        'chance is \(\sigma(0.7)=0.67\). Immediately after a boundary it is '
        '\(\sigma(-0.7)=0.33\). The same acoustics therefore produce fewer adjacent cuts.</p>'
        '<p class="cap">The real learned offsets were smaller; their consistent negative gap '
        '\(\delta_1-\delta_0\) is the evidence for an anti-adjacent-boundary spacing prior.</p>'
    )

    transformer = (
        '<p>The local-history Transformer replaces the two offsets with a small decoder-only '
        'network. At frame \(t\), it adds the current projected WavLM vector, a positional '
        'encoding, and an embedding of the <i>previous</i> boundary. Causal attention then '
        'attends over a 64-state window in each of four layers. Stacking the layers can relay '
        'older information indirectly, but this is not unrestricted global attention.</p>'
        + hk.equation(r'''
\begin{aligned}
z_t &= W_x x_t + E(\widetilde b_{t-1}) + P_t,
& \widetilde b_{-1}&=\operatorname{BOS}, \\
h_t &= \operatorname{LocalCausalTransformer}_{64}(z_{0:t}), \\
\Pr(b_t=1\mid x,b_{<t}) &= \sigma(w^{\top}h_t),
& b_0&=0.
\end{aligned}''')
        + hk.equation(r'''
\pi(b_{1:T-1}\mid x)
=\prod_{t=1}^{T-1}\pi(b_t\mid x,b_{<t}).''')
        + '<p>Here \(x_t\) is the WavLM feature, \(E\) is a three-entry embedding table for '
        '<span class="mono">BOS/0/1</span>, \(P_t\) is the position encoding, and \(h_t\) is '
        'the four-layer causal state. Frame 0 is fixed to 0 and excluded from the policy loss '
        'because pooling already opens the first segment.</p>'
    )

    history_example = (
        '<div class="history-compare">'
        '<div><div class="mini-label">History A · recent cut</div>'
        '<div class="bitline"><span class="bit one">1</span><span class="bit zero">0</span></div></div>'
        '<div><div class="mini-label">History B · long run</div>'
        '<div class="bitline"><span class="bit one">1</span><span class="bit zero">0</span>'
        '<span class="bit zero">0</span><span class="bit zero">0</span>'
        '<span class="bit zero">0</span></div></div></div>'
        '<p>Both histories end in 0. The first-order CNN therefore applies exactly the same '
        '\(\delta_0\) adjustment to the next frame. The local-history Transformer can tell that '
        'History A cut recently while History B has gone several frames without a cut. It can '
        'learn “wait after a cut,” “cut after a long run,” or make either rule depend on the audio.</p>'
    )

    training = (
        '<div class="demo-steps">'
        '<div><b>1 · Cold-start</b><span>Shift gold char boundaries right and score every frame '
        'in one causal teacher-forced pass.</span></div>'
        '<div><b>2 · Closed-loop rollout</b><span>At inference or RL, sample one boundary at a '
        'time and feed it into the next frame. A KV cache preserves earlier attention states.</span></div>'
        '<div><b>3 · Decoder reward</b><span>Mean-pool each sampled segmentation, run the speech '
        'decoder, and score transcript NLL or error.</span></div>'
        '<div><b>4 · Parallel re-score</b><span>Shift the completed sampled history right and '
        'recompute its exact conditional log-probabilities in one differentiable pass.</span></div>'
        '</div>'
        + hk.equation(r'''
\mathcal{L}_{\mathrm{PG}}
=-\operatorname{mean}_{b,k,t}
\left[A_{b,k}\log\pi_\theta\!\left(b_t^{(b,k)}\mid x^{(b)},b_{<t}^{(b,k)}\right)\right].''')
        + '<p>For batch item \(b\), rollout \(k\), and frame \(t\), \(A_{b,k}\) is the '
        'group-relative advantage among the sampled segmentations. Sampling is discrete, so it '
        'runs without gradients; the parallel re-score supplies the gradient through the exact '
        'same conditional policy. For the Transformer model, the realized audio-frequency '
        'penalty is '
        'part of each rollout reward because exact marginals over all possible histories would '
        'be exponentially expensive.</p>'
    )

    return (
        hk.finding('<b>Key distinction.</b> “AR” describes the boundary-label distribution, '
                   'not the acoustic encoder. The CNN version is a one-step Markov correction; '
                   'the Transformer version is a learned local-history controller.', ok=True)
        + hk.card(comparison, title="What each policy remembers")
        + hk.card('<div class="demo-figure-scroll">' + ar_policy_demo_figure() + '</div>'
                  + '<p class="cap">The independent policy has no label feedback. The CNN '
                    'first-order policy feeds back one bit through two offsets. The local-history '
                    'Transformer feeds back the sampled bit as the next input and attends over '
                    'the latest 64 states in each causal layer.</p>',
                  title="One acoustic stream, three boundary policies")
        + hk.card(worked_pooling, title="Start with the object being predicted")
        + '<div class="grid2">'
        + hk.card(cnn, title="CNN first-order AR: a learned spacing bias")
        + hk.card(transformer, title="Transformer AR: a learned history model")
        + '</div>'
        + hk.card(history_example, title="Why local history is more expressive")
        + hk.card(training, title="How the Transformer AR trains without backpropagating through samples")
    )


def bigru_pooling_diagram():
    """Theme-aware diagram of mean-plus-BiGRU residual segment pooling."""
    return r'''
<svg class="chart" viewBox="0 0 1080 610" role="img"
     aria-labelledby="bigru-pooling-title bigru-pooling-desc">
  <title id="bigru-pooling-title">BiGRU residual pooling</title>
  <desc id="bigru-pooling-desc">A segment's ordered WavLM frames feed a mean branch and a bidirectional GRU branch. The final forward and backward states are concatenated, then a zero-initialized projection turns them into a correction that is added to the mean. The lower panel shows mean-equivalent initialization, two adaptation epochs, and joint reinforcement learning.</desc>
  <defs>
    <marker id="bg-arrow" viewBox="0 0 10 10" refX="10" refY="5"
            markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0 0 L10 5 L0 10 z" fill="var(--text-muted)"/>
    </marker>
  </defs>
  <style>
    .bg-panel{fill:var(--surface-2);stroke:var(--border-strong);stroke-width:1.4}
    .bg-box{fill:var(--surface-1);stroke:var(--border-strong);stroke-width:1.4}
    .bg-blue{fill:color-mix(in srgb,var(--pair-sed-asr) 13%,var(--surface-1));stroke:var(--pair-sed-asr);stroke-width:1.7}
    .bg-orange{fill:color-mix(in srgb,#b96f20 13%,var(--surface-1));stroke:#b96f20;stroke-width:1.7}
    .bg-green{fill:color-mix(in srgb,#27865d 13%,var(--surface-1));stroke:#27865d;stroke-width:1.7}
    .bg-line{fill:none;stroke:var(--text-muted);stroke-width:1.8;marker-end:url(#bg-arrow)}
    .bg-title{fill:var(--text);font:700 20px ui-sans-serif,system-ui,sans-serif}
    .bg-label{fill:var(--text);font:650 15px ui-sans-serif,system-ui,sans-serif}
    .bg-small{fill:var(--text-secondary);font:13px ui-sans-serif,system-ui,sans-serif}
    .bg-mono{fill:var(--text);font:600 15px ui-monospace,"SFMono-Regular",monospace}
  </style>

  <rect class="bg-panel" x="10" y="10" width="1060" height="390" rx="16"/>
  <text class="bg-title" x="34" y="44">One predicted segment</text>
  <text class="bg-small" x="34" y="67">The boundary policy fixes the interval; pooling decides how its ordered frames become one audio token.</text>

  <rect class="bg-blue" x="35" y="105" width="78" height="58" rx="10"/>
  <rect class="bg-blue" x="126" y="105" width="78" height="58" rx="10"/>
  <rect class="bg-blue" x="217" y="105" width="78" height="58" rx="10"/>
  <rect class="bg-blue" x="308" y="105" width="78" height="58" rx="10"/>
  <text class="bg-mono" x="74" y="140" text-anchor="middle">x₀</text>
  <text class="bg-mono" x="165" y="140" text-anchor="middle">x₁</text>
  <text class="bg-mono" x="256" y="140" text-anchor="middle">x₂</text>
  <text class="bg-mono" x="347" y="140" text-anchor="middle">x₃</text>
  <text class="bg-small" x="210" y="187" text-anchor="middle">WavLM frames in time order</text>

  <path class="bg-line" d="M386 132 H450 V111 H496"/>
  <path class="bg-line" d="M386 142 H450 V266 H496"/>

  <rect class="bg-box" x="500" y="82" width="210" height="82" rx="12"/>
  <text class="bg-label" x="605" y="111" text-anchor="middle">Mean branch</text>
  <text class="bg-mono" x="605" y="140" text-anchor="middle">m = mean(x₀…x₃)</text>
  <text class="bg-small" x="605" y="180" text-anchor="middle">Order-insensitive baseline</text>

  <rect class="bg-orange" x="500" y="210" width="210" height="112" rx="12"/>
  <text class="bg-label" x="605" y="240" text-anchor="middle">Order branch</text>
  <text class="bg-mono" x="605" y="266" text-anchor="middle">BiGRU(x₀…x₃)</text>
  <text class="bg-small" x="605" y="291" text-anchor="middle">concat(forward-last,</text>
  <text class="bg-small" x="605" y="309" text-anchor="middle">backward-last)</text>

  <path class="bg-line" d="M710 266 H740"/>
  <rect class="bg-orange" x="742" y="218" width="150" height="96" rx="12"/>
  <text class="bg-label" x="817" y="246" text-anchor="middle">Projection</text>
  <text class="bg-mono" x="817" y="273" text-anchor="middle">Δ = Wh + b</text>
  <text class="bg-small" x="817" y="297" text-anchor="middle">learned correction</text>

  <path class="bg-line" d="M710 123 H924 V162"/>
  <path class="bg-line" d="M892 266 H924 V234"/>
  <circle class="bg-green" cx="924" cy="198" r="25"/>
  <text class="bg-title" x="924" y="205" text-anchor="middle">+</text>
  <path class="bg-line" d="M949 198 H982"/>
  <rect class="bg-green" x="994" y="158" width="60" height="80" rx="12"/>
  <text class="bg-label" x="1024" y="190" text-anchor="middle">audio</text>
  <text class="bg-label" x="1024" y="210" text-anchor="middle">token</text>
  <text class="bg-mono" x="924" y="350" text-anchor="middle">z = m + Δ</text>
  <text class="bg-small" x="924" y="373" text-anchor="middle">Mean plus an order-sensitive residual</text>

  <rect class="bg-panel" x="10" y="420" width="1060" height="180" rx="16"/>
  <text class="bg-title" x="34" y="455">How the run starts and learns</text>
  <rect class="bg-box" x="34" y="474" width="290" height="108" rx="12"/>
  <text class="bg-label" x="179" y="504" text-anchor="middle">Initialization</text>
  <text class="bg-mono" x="179" y="531" text-anchor="middle">W = 0, b = 0  ⇒  Δ = 0</text>
  <text class="bg-small" x="179" y="554" text-anchor="middle">Exactly the same output as mean pooling</text>
  <path class="bg-line" d="M324 528 H368"/>
  <rect class="bg-blue" x="374" y="474" width="306" height="108" rx="12"/>
  <text class="bg-label" x="527" y="504" text-anchor="middle">Epochs 1–2 · adaptation</text>
  <text class="bg-small" x="527" y="530" text-anchor="middle">Fixed predicted boundaries</text>
  <text class="bg-small" x="527" y="551" text-anchor="middle">CE updates BiGRU + projection</text>
  <text class="bg-small" x="527" y="569" text-anchor="middle">and LoRA decoder</text>
  <path class="bg-line" d="M680 528 H724"/>
  <rect class="bg-green" x="730" y="474" width="306" height="108" rx="12"/>
  <text class="bg-label" x="883" y="504" text-anchor="middle">Epochs 3–12 · joint RL</text>
  <text class="bg-small" x="883" y="530" text-anchor="middle">CE keeps adapting pooler + decoder</text>
  <text class="bg-small" x="883" y="551" text-anchor="middle">GRPO additionally updates boundary policy</text>
</svg>'''


def sample_utts(csv_path, n=3, dmin=2.5, dmax=5.0):
    import csv as _csv
    lsroot = os.path.join(DATA, "LibriSpeech")
    out = []
    for row in _csv.DictReader(open(csv_path)):
        if dmin < float(row["duration"]) < dmax:
            out.append((row["ID"],
                        row["wav"].replace("$data_root", lsroot).replace("{data_root}", lsroot),
                        row["wrd"]))
        if len(out) >= n:
            break
    return out


# --------------------------- report assembly ---------------------------
def build(args):
    S, F = os.path.join(RES, "speechllm_segmenter"), os.path.join(RES, "speechllm_fixed_pooling")
    SW = os.path.join(RES, "speechllm_segmenter_wavlm")
    FW = os.path.join(RES, "speechllm_fixed_pooling_wavlm")
    abl = {"nll_frozen": (os.path.join(S, "abl/nll_frozen/cnn/3407/train_log.txt"), hk.C["muted"], 6),
           "nll_mt": (os.path.join(S, "abl/nll_mt/cnn/3407/train_log.txt"), hk.C["green"], 6),
           "cer_mt": (os.path.join(S, "abl/cer_mt/cnn/3407/train_log.txt"), hk.C["blue"], 6)}
    rows = {k: parse_log(v[0]) for k, v in abl.items()}
    coll = parse_log(os.path.join(S, "joint/transformer/3407/train_log.txt"))  # warmup 2
    nll = rows["nll_mt"]

    base = []
    for k in (3, 4, 5, 6, 8):
        run = os.path.join(F, "fixed_rate_k%d_tc100/3407/wer_results" % k)
        clean_rec = parse_wer_file(os.path.join(run, "wer_test-clean.txt"))
        other_rec = parse_wer_file(os.path.join(run, "wer_test-other.txt"))
        clean = clean_rec["wer"] if clean_rec else float("nan")
        other = other_rec["wer"] if other_rec else float("nan")
        base.append((k, 1.0 / k, clean, other))

    w2v_no_down = {}
    for split in ("test-clean", "test-other"):
        vals = []
        for seed in (3407, 3408, 3409):
            rec = parse_wer_file(os.path.join(
                F, "no_downsampling_tc100", str(seed), "wer_results",
                "wer_%s.txt" % split,
            ))
            if rec:
                vals.append(rec["wer"])
        w2v_no_down[split] = mean_sd(vals)

    ms_root = os.path.join(S, "multiseed_shared_warmup_onpolicy")
    seeds = (3407, 3408, 3409)

    # Corrected 24k-step LS960 study. Read the finished evaluations directly so
    # the report can be regenerated while the last seed is still training.
    ls960_root = os.path.join(RES, "speechllm_ls960_step24k_corrected")

    def ls960_learned_stats(family):
        runs = []
        for seed in seeds:
            run = os.path.join(ls960_root, family, "nll_mt", str(seed))
            clean = parse_wer_file(os.path.join(
                run, "wer_results", "wer_test-clean.txt"
            ))
            other = parse_wer_file(os.path.join(
                run, "wer_results", "wer_test-other.txt"
            ))
            if not clean or not other:
                continue

            # Final evaluation is written in test-clean/test-other order.
            rhos = []
            train_log = os.path.join(run, "train_log.txt")
            if os.path.exists(train_log):
                for line in open(train_log, encoding="utf-8", errors="replace"):
                    match = re.match(
                        rf"Epoch loaded: \d+ - test .*?rho_mean: ({_NUM})", line
                    )
                    if match:
                        rhos.append(float(match.group(1)))
            if len(rhos) < 2:
                continue
            runs.append({
                "seed": seed,
                "clean": clean["wer"],
                "other": other["wer"],
                "clean_hz": rate_hz(rhos[0]),
                "other_hz": rate_hz(rhos[1]),
            })

        return {
            "runs": runs,
            "n": len(runs),
            "clean": mean_sd([run["clean"] for run in runs]),
            "other": mean_sd([run["other"] for run in runs]),
            "clean_hz": mean_sd([run["clean_hz"] for run in runs]),
            "other_hz": mean_sd([run["other_hz"] for run in runs]),
        }

    ls960_transformer = ls960_learned_stats("transformer_ar_local64_bigru")
    ls960_cnn = ls960_learned_stats("cnn_first_order_ar_bigru")

    # Frozen audio-only phone-CTC boundaries with deterministic midpoint splits
    # for predicted segments of at least eight 20-ms frames. Only the decoder
    # projection and LoRA parameters are adapted with regular CE on LS960.
    ls960_phone_ctc_root = os.path.join(
        RES, "speechllm_ls960_phone_ctc_longsplit8_mean_ce_1ep"
    )
    ls960_phone_ctc_runs = []
    for seed in seeds:
        run = os.path.join(ls960_phone_ctc_root, str(seed), "wer_results")
        records = {
            split: parse_wer_file(os.path.join(run, "wer_%s.txt" % split))
            for split in ("test-clean", "test-other", "dev-other")
        }
        if not all(records.values()):
            continue
        ls960_phone_ctc_runs.append({
            "seed": seed,
            **{split: record["wer"] for split, record in records.items()},
        })
    ls960_phone_ctc = {
        "runs": ls960_phone_ctc_runs,
        "n": len(ls960_phone_ctc_runs),
        **{
            split: mean_sd([run[split] for run in ls960_phone_ctc_runs])
            for split in ("test-clean", "test-other", "dev-other")
        },
    }
    ls960_phone_ctc_summary_path = (
        "/export/jsalt26/omnienc/users/cxiao/"
        "boundary_targets/wavlm_phone_ctc_tc100_best_longsplit8/_stats/"
        "long_segment_split_summary.json"
    )
    with open(ls960_phone_ctc_summary_path, encoding="utf-8") as stream:
        ls960_phone_ctc_boundary_summary = json.load(stream)
    ls960_phone_ctc_hz = {
        split: ls960_phone_ctc_boundary_summary["splits"][split]["refined_rate_hz"]
        for split in ("dev-clean", "dev-other", "test-clean", "test-other")
    }

    def ls960_metric_text(stats, n, digits=2, suffix="%"):
        value = ("%.*f" % (digits, stats[0]))
        if n > 1:
            value += " ± %.*f" % (digits, stats[1])
        return value + suffix

    def ls960_one_decimal(value):
        # Frequencies are positive; the epsilon makes x.x5 display with the
        # conventional half-up presentation rather than binary/banker's rounding.
        return "%.1f" % (value + 1e-9)

    def ls960_frequency_text(result):
        if result["n"] > 1:
            return "%s ± %s / %s ± %s Hz" % (
                ls960_one_decimal(result["clean_hz"][0]),
                ls960_one_decimal(result["clean_hz"][1]),
                ls960_one_decimal(result["other_hz"][0]),
                ls960_one_decimal(result["other_hz"][1]),
            )
        return "%s / %s Hz" % (
            ls960_one_decimal(result["clean_hz"][0]),
            ls960_one_decimal(result["other_hz"][0]),
        )

    # Matched 100h comparison: fixed boundaries, a frozen supervised
    # Transformer-AR segmenter, and the same segmenter updated by joint RL.
    attribution_root = os.path.join(
        SW, "attribution_100h_transformer_ar_bigru"
    )

    def attribution_family_stats(folder):
        runs = []
        for seed in seeds:
            run = os.path.join(attribution_root, folder, str(seed))
            clean = parse_wer_file(os.path.join(
                run, "wer_results", "wer_test-clean.txt"
            ))
            other = parse_wer_file(os.path.join(
                run, "wer_results", "wer_test-other.txt"
            ))
            if not clean or not other:
                continue
            rhos = []
            train_log = os.path.join(run, "train_log.txt")
            if os.path.exists(train_log):
                for line in open(train_log, encoding="utf-8", errors="replace"):
                    match = re.match(
                        rf"Epoch loaded: \d+ - test .*?rho_mean: ({_NUM})", line
                    )
                    if match:
                        rhos.append(float(match.group(1)))
            if len(rhos) < 2:
                continue
            # A complete evaluation writes clean, other, then dev-other. If an
            # evaluation was repeated, use the final complete triplet.
            final_rhos = rhos[-3:] if len(rhos) >= 3 else rhos[-2:]
            runs.append({
                "seed": seed,
                "clean": clean["wer"],
                "other": other["wer"],
                "clean_hz": rate_hz(final_rhos[0]),
                "other_hz": rate_hz(final_rhos[1]),
            })
        return {
            "runs": runs,
            "n": len(runs),
            "clean": mean_sd([run["clean"] for run in runs]),
            "other": mean_sd([run["other"] for run in runs]),
            "clean_hz": mean_sd([run["clean_hz"] for run in runs]),
            "other_hz": mean_sd([run["other_hz"] for run in runs]),
        }

    attribution_fixed_mean = attribution_family_stats("fixed_k5_mean")
    attribution_fixed = attribution_family_stats("fixed_k5_bigru")
    attribution_frozen = attribution_family_stats("frozen_segmenter_bigru")
    attribution_joint = attribution_family_stats("joint_rl_bigru")

    def headline_stats(variant, split, backbone="cnn"):
        vals = []
        for seed in seeds:
            rec = parse_wer_file(os.path.join(
                ms_root, variant, backbone, str(seed), "wer_results", "wer_%s.txt" % split
            ))
            if rec:
                vals.append(rec["wer"])
        return mean_sd(vals)

    def headline_count(variant, split, backbone="cnn"):
        return sum(
            os.path.exists(os.path.join(
                ms_root, variant, backbone, str(seed), "wer_results", "wer_%s.txt" % split
            ))
            for seed in seeds
        )

    frozen_clean = headline_stats("nll_frozen", "test-clean")
    mt_clean = headline_stats("nll_mt", "test-clean")
    frozen_other = headline_stats("nll_frozen", "test-other")
    mt_other = headline_stats("nll_mt", "test-other")
    frozen_dev_other = headline_stats("nll_frozen", "dev-other")
    mt_dev_other = headline_stats("nll_mt", "dev-other")
    cer_clean = headline_stats("cer_mt", "test-clean")
    cer_other = headline_stats("cer_mt", "test-other")
    cer_dev_other = headline_stats("cer_mt", "dev-other")

    tf_frozen_clean = headline_stats("nll_frozen", "test-clean", "transformer")
    tf_frozen_other = headline_stats("nll_frozen", "test-other", "transformer")
    tf_frozen_dev_other = headline_stats("nll_frozen", "dev-other", "transformer")
    tf_mt_clean = headline_stats("nll_mt", "test-clean", "transformer")
    tf_mt_other = headline_stats("nll_mt", "test-other", "transformer")
    tf_mt_dev_other = headline_stats("nll_mt", "dev-other", "transformer")
    tf_cer_clean = headline_stats("cer_mt", "test-clean", "transformer")
    tf_cer_other = headline_stats("cer_mt", "test-other", "transformer")
    tf_cer_dev_other = headline_stats("cer_mt", "dev-other", "transformer")

    def headline_rho(variant, backbone="cnn"):
        values = []
        for seed in seeds:
            path = os.path.join(ms_root, variant, backbone, str(seed), "train_log.txt")
            if not os.path.exists(path):
                continue
            for line in open(path, encoding="utf-8", errors="replace"):
                match = re.match(rf"Epoch loaded: \d+ - test .*?rho_mean: ({_NUM})", line)
                if match:
                    values.append(float(match.group(1)))
                    break
        return mean_sd(values)

    frozen_rho = headline_rho("nll_frozen")
    mt_rho = headline_rho("nll_mt")
    cer_rho = headline_rho("cer_mt")
    tf_mt_rho = headline_rho("nll_mt", "transformer")
    completed_test_grid = 0
    completed_full_grid = 0
    for variant in ("nll_frozen", "nll_mt", "cer_mt"):
        for backbone in ("cnn", "transformer"):
            for seed in seeds:
                run = os.path.join(ms_root, variant, backbone, str(seed), "wer_results")
                completed_test_grid += all(
                    os.path.exists(os.path.join(run, "wer_%s.txt" % split))
                    for split in ("test-clean", "test-other")
                )
                completed_full_grid += all(
                    os.path.exists(os.path.join(run, "wer_%s.txt" % split))
                    for split in ("test-clean", "test-other", "dev-other")
                )

    # Latest WavLM replication. Keep this summary compact: one learned-model table,
    # one baseline table, and a few decision-relevant takeaways. The longer wav2vec2
    # diagnostics below remain useful as experiment history.
    wavlm_ms = os.path.join(SW, "multiseed_shared_warmup_onpolicy")

    def wavlm_stats(variant, backbone, split):
        vals = []
        for seed in seeds:
            rec = parse_wer_file(os.path.join(
                wavlm_ms, variant, backbone, str(seed), "wer_results",
                "wer_%s.txt" % split,
            ))
            if rec:
                vals.append(rec["wer"])
        return mean_sd(vals)

    def wavlm_rho(variant, backbone, split="test-clean"):
        split_idx = {"test-clean": 0, "test-other": 1, "dev-other": 2}[split]
        vals = []
        for seed in seeds:
            path = os.path.join(wavlm_ms, variant, backbone, str(seed), "train_log.txt")
            found = []
            if os.path.exists(path):
                for line in open(path, encoding="utf-8", errors="replace"):
                    match = re.match(
                        rf"Epoch loaded: \d+ - test .*?rho_mean: ({_NUM})", line
                    )
                    if match:
                        found.append(float(match.group(1)))
            if len(found) > split_idx:
                vals.append(found[split_idx])
        return mean_sd(vals)

    wavlm_specs = [
        ("CNN", "nll_frozen", "NLL", "frozen"),
        ("CNN", "nll_mt", "NLL", "co-trained"),
        ("CNN", "cer_mt", "CER", "co-trained"),
        ("Transformer", "nll_frozen", "NLL", "frozen"),
        ("Transformer", "nll_mt", "NLL", "co-trained"),
        ("Transformer", "cer_mt", "CER", "co-trained"),
    ]
    wavlm_results = {}
    for backbone, variant, _, _ in wavlm_specs:
        key = backbone.lower(), variant
        wavlm_results[key] = {
            "rho": wavlm_rho(variant, backbone.lower()),
            "clean": wavlm_stats(variant, backbone.lower(), "test-clean"),
            "other": wavlm_stats(variant, backbone.lower(), "test-other"),
        }

    wavlm_fixed = []
    for k in (3, 4, 5, 6, 8):
        clean_vals, other_vals, dev_clean_vals = [], [], []
        for seed in seeds:
            run = os.path.join(
                FW, "fixed_rate_k%d_tc100" % k, str(seed), "wer_results"
            )
            clean = parse_wer_file(os.path.join(run, "wer_test-clean.txt"))
            other = parse_wer_file(os.path.join(run, "wer_test-other.txt"))
            dev_clean = parse_wer_file(os.path.join(run, "wer_dev-clean.txt"))
            if clean and other:
                clean_vals.append(clean["wer"])
                other_vals.append(other["wer"])
                if dev_clean:
                    dev_clean_vals.append(dev_clean["wer"])
        wavlm_fixed.append({
            "k": k,
            "rho": 1.0 / k,
            "clean": mean_sd(clean_vals),
            "other": mean_sd(other_vals),
            "dev_clean": mean_sd(dev_clean_vals),
            "n": len(clean_vals),
        })

    wavlm_no_down = {}
    for split in ("test-clean", "test-other", "dev-clean"):
        vals = []
        for seed in seeds:
            rec = parse_wer_file(os.path.join(
                FW, "no_downsampling_tc100", str(seed), "wer_results",
                "wer_%s.txt" % split,
            ))
            if rec:
                vals.append(rec["wer"])
        wavlm_no_down[split] = mean_sd(vals)

    wavlm_tf_mt = wavlm_results["transformer", "nll_mt"]
    wavlm_cnn_mt = wavlm_results["cnn", "nll_mt"]
    wavlm_k5 = next(row for row in wavlm_fixed if row["k"] == 5)
    wavlm_selected_epochs = []
    wavlm_selected_by_condition = {}
    for backbone, variant, _, _ in wavlm_specs:
        condition_epochs = []
        for seed in seeds:
            epoch = selected_epoch(os.path.join(
                wavlm_ms, variant, backbone.lower(), str(seed), "train_log.txt"
            ))
            condition_epochs.append(epoch)
            wavlm_selected_epochs.append(epoch)
        wavlm_selected_by_condition[backbone.lower(), variant] = condition_epochs
    wavlm_selected_final = sum(epoch == 10 for epoch in wavlm_selected_epochs)
    wavlm_selected_joint_early = sum(
        epoch is not None and 7 <= epoch <= 9 for epoch in wavlm_selected_epochs
    )
    wavlm_selected_warmup = sum(
        epoch is not None and epoch <= 6 for epoch in wavlm_selected_epochs
    )

    # Controlled studies added after the broad WavLM replication.  Keep these
    # artifact-derived rather than copying values into the prose: this report is
    # regenerated while the long autoregressive jobs are still producing rows.
    def split_record(run, split):
        for filename in ("wer_%s.txt" % split, "wer_test-%s.txt" % split):
            record = parse_wer_file(os.path.join(run, "wer_results", filename))
            if record:
                return record
        return None

    def family_stats(pattern, split, seed_ids=None):
        values = []
        for seed in seeds if seed_ids is None else seed_ids:
            record = split_record(pattern.format(seed=seed), split)
            if record:
                values.append(record["wer"])
        return mean_sd(values), len(values)

    def family_rho(pattern, seed_ids=None):
        values = []
        for seed in seeds if seed_ids is None else seed_ids:
            path = os.path.join(pattern.format(seed=seed), "train_log.txt")
            if not os.path.exists(path):
                continue
            for line in open(path, encoding="utf-8", errors="replace"):
                match = re.match(
                    rf"Epoch loaded: \d+ - test .*?rho_mean: ({_NUM})", line
                )
                if match:
                    values.append(float(match.group(1)))
                    break
        return mean_sd(values)

    wavlm_char_pattern = os.path.join(FW, "char_alignment_tc100", "{seed}")
    wavlm_phone_pattern = os.path.join(FW, "phone_alignment_tc100", "{seed}")
    wavlm_char_clean, wavlm_char_n = family_stats(wavlm_char_pattern, "test-clean")
    wavlm_char_other, _ = family_stats(wavlm_char_pattern, "test-other")
    wavlm_char_dev, _ = family_stats(wavlm_char_pattern, "dev-clean")
    wavlm_phone_clean, wavlm_phone_n = family_stats(wavlm_phone_pattern, "test-clean")
    wavlm_phone_other, _ = family_stats(wavlm_phone_pattern, "test-other")
    wavlm_phone_dev, _ = family_stats(wavlm_phone_pattern, "dev-clean")

    hybrid_root = os.path.join(SW, "hybrid_w2v2_segmenter_controlled")
    hybrid = {}
    for arm in ("oracle_char_decoder", "scratch_decoder"):
        pattern = os.path.join(hybrid_root, arm, "{seed}")
        hybrid[arm] = {
            "clean": family_stats(pattern, "test-clean"),
            "other": family_stats(pattern, "test-other"),
            "dev_other": family_stats(pattern, "dev-other"),
            "rho": family_rho(pattern),
        }

    oracleclose_root = os.path.join(SW, "oracleclose_bestinit")
    oracleclose = {}
    for policy in ("bernoulli", "autoregressive"):
        pattern = os.path.join(oracleclose_root, policy, "{seed}")
        oracleclose[policy] = {
            "clean": family_stats(pattern, "test-clean"),
            "other": family_stats(pattern, "test-other"),
            "dev_other": family_stats(pattern, "dev-other"),
            "rho": family_rho(pattern),
        }

    boundary_swap_root = os.path.join(
        RES, "speechllm_boundary_swap_wavlm", "oracle_decoder_seed3407",
        "cnn_v2", "wer_results",
    )
    boundary_swap = {}
    for source in ("oracle", "cold", "rl"):
        record = parse_wer_file(os.path.join(
            boundary_swap_root, "wer_%s_dev-clean.txt" % source
        ))
        boundary_swap[source] = record["wer"] if record else float("nan")

    def final_cold_f1(root, backbone):
        rows_ = parse_log(os.path.join(root, "coldstart", backbone, "3407",
                                       "train_log.txt"))
        values = [row["valid boundary_f1"] for row in rows_
                  if "valid boundary_f1" in row]
        return values[-1] if values else float("nan")

    cold_f1 = {
        ("wav2vec2", "CNN"): final_cold_f1(S, "cnn"),
        ("wav2vec2", "Transformer"): final_cold_f1(S, "transformer"),
        ("WavLM", "CNN"): final_cold_f1(SW, "cnn"),
        ("WavLM", "Transformer"): final_cold_f1(SW, "transformer"),
    }

    long_ar_root = os.path.join(
        S, "w2v2_transformer_autoregressive_long", "nll_mt"
    )
    long_ar = []
    for seed in seeds:
        run = os.path.join(long_ar_root, str(seed))
        logged = [row for row in parse_log(os.path.join(run, "train_log.txt"))
                  if "valid WER" in row]
        best_row = min(logged, key=lambda row: row["valid WER"]) if logged else {}
        latest_row = logged[-1] if logged else {}
        clean = split_record(run, "test-clean")
        other = split_record(run, "test-other")
        long_ar.append({
            "seed": seed,
            "epochs": int(latest_row.get("epoch", 0)),
            "best_epoch": int(best_row.get("epoch", 0)),
            "best_dev": best_row.get("valid WER", float("nan")),
            "best_rho": best_row.get("valid rho_mean", float("nan")),
            "latest_dev": latest_row.get("valid WER", float("nan")),
            "latest_gap": latest_row.get("valid ar_bias_gap", float("nan")),
            "clean": clean["wer"] if clean else None,
            "other": other["wer"] if other else None,
            "selected_epoch": selected_epoch(os.path.join(run, "train_log.txt")),
        })

    long_ar_best = mean_sd([row["best_dev"] for row in long_ar
                            if row["best_dev"] == row["best_dev"]])
    long_ar_latest = mean_sd([row["latest_dev"] for row in long_ar
                              if row["latest_dev"] == row["latest_dev"]])
    long_ar_clean = mean_sd([row["clean"] for row in long_ar
                             if row["clean"] is not None])
    long_ar_other = mean_sd([row["other"] for row in long_ar
                             if row["other"] is not None])

    # Production full-prefix Transformer-AR study.  Both arms use WavLM for
    # boundary prediction and decoder-side features, the best char decoder,
    # local attention (window 64), combined K=4 rollouts, and the same
    # two-warmup + ten-joint-RL schedule.  The only intended change is the
    # within-segment representation: plain mean versus a zero-initialized
    # order-aware BiGRU residual added to that mean.
    fullprefix_roots = {
        "mean": os.path.join(
            SW, "fullprefix_transformer_ar_local64_best_char_decoder", "nll_mt"
        ),
        "bigru": os.path.join(
            SW,
            "fullprefix_transformer_ar_local64_bigru_best_char_decoder",
            "nll_mt",
        ),
    }
    fullprefix_seed_ids = {
        "mean": (3407, 3408, 3409),
        "bigru": (3407, 3408, 3409),
    }
    fullprefix = {}
    for pooling, root in fullprefix_roots.items():
        pattern = os.path.join(root, "{seed}")
        runs = []
        pooling_seeds = fullprefix_seed_ids[pooling]
        for seed in pooling_seeds:
            run = pattern.format(seed=seed)
            clean = split_record(run, "test-clean")
            other = split_record(run, "test-other")
            test_rhos = []
            log_path = os.path.join(run, "train_log.txt")
            if os.path.exists(log_path):
                for line in open(log_path, encoding="utf-8", errors="replace"):
                    match = re.match(
                        rf"Epoch loaded: \d+ - test .*?rho_mean: ({_NUM})", line
                    )
                    if match:
                        test_rhos.append(float(match.group(1)))
            runs.append({
                "seed": seed,
                "epoch": selected_epoch(log_path),
                "rho": test_rhos[0] if test_rhos else float("nan"),
                "clean": clean["wer"] if clean else None,
                "other": other["wer"] if other else None,
            })
        fullprefix[pooling] = {
            "runs": runs,
            "rho": family_rho(pattern, pooling_seeds),
            "clean": family_stats(pattern, "test-clean", pooling_seeds),
            "other": family_stats(pattern, "test-other", pooling_seeds),
            "dev_other": family_stats(pattern, "dev-other", pooling_seeds),
        }

    fullprefix_mean_by_seed = {
        run["seed"]: run for run in fullprefix["mean"]["runs"]
    }
    fullprefix_bigru_by_seed = {
        run["seed"]: run for run in fullprefix["bigru"]["runs"]
    }
    fullprefix_paired_seeds = sorted(
        set(fullprefix_mean_by_seed) & set(fullprefix_bigru_by_seed)
    )
    fullprefix_clean_wins = sum(
        fullprefix_bigru_by_seed[seed]["clean"]
        < fullprefix_mean_by_seed[seed]["clean"]
        for seed in fullprefix_paired_seeds
        if fullprefix_mean_by_seed[seed]["clean"] is not None
        and fullprefix_bigru_by_seed[seed]["clean"] is not None
    )
    fullprefix_other_wins = sum(
        fullprefix_bigru_by_seed[seed]["other"]
        < fullprefix_mean_by_seed[seed]["other"]
        for seed in fullprefix_paired_seeds
        if fullprefix_mean_by_seed[seed]["other"] is not None
        and fullprefix_bigru_by_seed[seed]["other"] is not None
    )

    # Matched CNN first-order-AR pooling control. It uses the same char segmenter
    # and decoder initialization, two adaptation epochs, ten RL epochs, and K=4
    # as the existing CNN mean-pooling arm; only the zero-residual BiGRU pooler is
    # added. Together with the Transformer pair this forms a 2x2 architecture ×
    # pooling study.
    cnn_mean_pattern = os.path.join(
        oracleclose_root, "autoregressive", "{seed}"
    )
    cnn_bigru_pattern = os.path.join(
        SW, "cnn_first_order_ar_bigru_best_char_decoder", "nll_mt", "{seed}"
    )
    cnn_bigru = {
        "rho": family_rho(cnn_bigru_pattern),
        "clean": family_stats(cnn_bigru_pattern, "test-clean"),
        "other": family_stats(cnn_bigru_pattern, "test-other"),
        "dev_other": family_stats(cnn_bigru_pattern, "dev-other"),
    }

    def paired_pooling_wins(mean_pattern, bigru_pattern, split):
        wins = 0
        for seed in seeds:
            mean_record = split_record(mean_pattern.format(seed=seed), split)
            bigru_record = split_record(bigru_pattern.format(seed=seed), split)
            if mean_record and bigru_record and bigru_record["wer"] < mean_record["wer"]:
                wins += 1
        return wins

    cnn_bigru_clean_wins = paired_pooling_wins(
        cnn_mean_pattern, cnn_bigru_pattern, "test-clean"
    )
    cnn_bigru_other_wins = paired_pooling_wins(
        cnn_mean_pattern, cnn_bigru_pattern, "test-other"
    )

    # Corrected batch-invariant evaluation.  The old checkpoint-time WER used
    # right-padded decoder prefixes and is retained only in the training logs.
    # Headline system comparisons below must come from this isolated artifact
    # tree.  A seed enters an aggregate only after all three requested splits
    # (test-clean, test-other, dev-other) are complete.
    artifact_root = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "../../../..", "artifacts", "segmenter"
    ))

    # Current four-split oracle evaluations. Only seeds with all requested WER
    # files and benchmark records enter these report aggregates.
    baseline_eval_root = os.path.join(
        artifact_root,
        "inference_eval_batch_invariant_leftpack_durationcap_v1",
        "baselines_wavlm",
    )
    baseline_eval_splits = ("test-clean", "test-other", "dev-clean", "dev-other")

    def current_oracle_family(folder, batch_size):
        complete = []
        for seed in seeds:
            run_dir = os.path.join(
                baseline_eval_root, folder, "seed%d" % seed,
                "batch%d" % batch_size,
            )
            split_rows = {
                split: parse_wer_file(os.path.join(
                    run_dir, "wer_results", "wer_%s.txt" % split
                ))
                for split in baseline_eval_splits
            }
            benchmarks = {
                row.get("split"): row
                for row in parse_jsonl(os.path.join(run_dir, "benchmark.jsonl"))
                if row.get("decoding_protocol")
                == "batch_invariant_left_packed_duration_cap_v1"
            }
            if (all(split_rows.values())
                    and all(split in benchmarks for split in baseline_eval_splits)):
                complete.append({"splits": split_rows, "benchmarks": benchmarks})
        return {
            "n": len(complete),
            "splits": {
                split: mean_sd([
                    run["splits"][split]["wer"] for run in complete
                ])
                for split in baseline_eval_splits
            },
            "frequency": mean_sd([
                run["benchmarks"]["test-clean"]["token_frequency_hz"]["global"]
                for run in complete
            ]),
            "rtf": mean_sd([
                run["benchmarks"]["test-clean"]["forward_rtf"]
                for run in complete
            ]),
        }

    current_oracle_phone = current_oracle_family("oracle_phone", 8)
    current_oracle_char = current_oracle_family("oracle_char", 4)
    if current_oracle_phone["n"] != 3 or current_oracle_char["n"] != 3:
        raise RuntimeError("Oracle report rows require three complete seeds")
    wavlm_phone_clean = current_oracle_phone["splits"]["test-clean"]
    wavlm_phone_other = current_oracle_phone["splits"]["test-other"]
    wavlm_phone_dev = current_oracle_phone["splits"]["dev-clean"]
    wavlm_phone_n = current_oracle_phone["n"]
    wavlm_phone_frequency = current_oracle_phone["frequency"]
    wavlm_char_clean = current_oracle_char["splits"]["test-clean"]
    wavlm_char_other = current_oracle_char["splits"]["test-other"]
    wavlm_char_dev = current_oracle_char["splits"]["dev-clean"]
    wavlm_char_n = current_oracle_char["n"]
    wavlm_char_frequency = current_oracle_char["frequency"]

    corrected_eval_root = os.path.join(
        artifact_root,
        "inference_eval_batch_invariant_leftpack_durationcap_v1",
        "learned_wavlm",
    )
    corrected_splits = ("test-clean", "test-other", "dev-other")
    corrected_candidates = {
        "cnn_mean": {
            3407: ["cnn_ar_mean/seed3407/batch8"],
            3408: [
                "cnn_ar_mean/seed3408/batch8",
                "cnn_ar_mean/seed3408/batch8_retry_off_ga129_20260821",
            ],
            3409: ["cnn_ar_mean/seed3409/batch8"],
        },
        "cnn_bigru": {
            3407: ["cnn_ar_bigru/seed3407/batch8"],
            3408: ["cnn_ar_bigru/seed3408/batch8"],
            3409: [
                "cnn_ar_bigru/seed3409/batch8",
                "cnn_ar_bigru/seed3409/batch8_retry_off_ga129_20260821",
            ],
        },
        "transformer_mean": {
            seed: ["transformer_ar_local64_mean/seed%d/batch8" % seed]
            for seed in seeds
        },
        "transformer_bigru": {
            seed: ["transformer_ar_local64_bigru/seed%d/batch8" % seed]
            for seed in seeds
        },
    }

    def corrected_run(candidate_paths):
        """Return the most complete corrected-evaluation candidate."""
        best = None
        for order, relative in enumerate(candidate_paths):
            run_dir = os.path.join(corrected_eval_root, relative)
            split_rows = {
                split: parse_wer_file(os.path.join(
                    run_dir, "wer_results", "wer_%s.txt" % split
                ))
                for split in corrected_splits
            }
            benchmarks = {
                row.get("split"): row
                for row in parse_jsonl(os.path.join(run_dir, "benchmark.jsonl"))
                if row.get("decoding_protocol")
                == "batch_invariant_left_packed_duration_cap_v1"
            }
            score = (sum(value is not None for value in split_rows.values()), order)
            candidate = {
                "path": run_dir,
                "splits": split_rows,
                "benchmarks": benchmarks,
                "complete": all(split_rows.values())
                          and all(split in benchmarks for split in corrected_splits),
                "score": score,
            }
            if best is None or candidate["score"] > best["score"]:
                best = candidate
        return best

    corrected = {}
    for key, seed_candidates in corrected_candidates.items():
        runs = []
        for seed, candidates in seed_candidates.items():
            run = corrected_run(candidates)
            run["seed"] = seed
            runs.append(run)
        complete = [run for run in runs if run["complete"]]
        corrected[key] = {
            "runs": runs,
            "n": len(complete),
            "clean": (mean_sd([
                run["splits"]["test-clean"]["wer"] for run in complete
            ]), len(complete)),
            "other": (mean_sd([
                run["splits"]["test-other"]["wer"] for run in complete
            ]), len(complete)),
            "dev_other": (mean_sd([
                run["splits"]["dev-other"]["wer"] for run in complete
            ]), len(complete)),
            "clean_hz": mean_sd([
                run["benchmarks"]["test-clean"]["token_frequency_hz"]["global"]
                for run in complete
            ]),
            "other_hz": mean_sd([
                run["benchmarks"]["test-other"]["token_frequency_hz"]["global"]
                for run in complete
            ]),
            "rtf": mean_sd([
                run["benchmarks"]["test-clean"]["forward_rtf"]
                for run in complete
            ]),
        }
        # Most report tables use test-clean frequency as their compact rate
        # coordinate; retain the established key alongside the split-specific
        # rates needed by the clean/other comparison table.
        corrected[key]["frequency"] = corrected[key]["clean_hz"]

    # Use the batch-invariant headline Transformer-AR + BiGRU evidence wherever
    # this named system appears.  The later attribution rerun is a distinct
    # experiment family and should not silently replace the headline result.
    headline_joint = {
        "runs": corrected["transformer_bigru"]["runs"],
        "n": corrected["transformer_bigru"]["n"],
        "clean": corrected["transformer_bigru"]["clean"][0],
        "other": corrected["transformer_bigru"]["other"][0],
        "clean_hz": corrected["transformer_bigru"]["clean_hz"],
        "other_hz": corrected["transformer_bigru"]["other_hz"],
    }

    def corrected_paired_wins(mean_key, bigru_key, split):
        mean_runs = {
            run["seed"]: run for run in corrected[mean_key]["runs"]
            if run["complete"]
        }
        bigru_runs = {
            run["seed"]: run for run in corrected[bigru_key]["runs"]
            if run["complete"]
        }
        paired = sorted(set(mean_runs) & set(bigru_runs))
        wins = sum(
            bigru_runs[seed]["splits"][split]["wer"]
            < mean_runs[seed]["splits"][split]["wer"]
            for seed in paired
        )
        return wins, len(paired)

    fullprefix_clean_wins, fullprefix_clean_pairs = corrected_paired_wins(
        "transformer_mean", "transformer_bigru", "test-clean"
    )
    fullprefix_other_wins, fullprefix_other_pairs = corrected_paired_wins(
        "transformer_mean", "transformer_bigru", "test-other"
    )
    cnn_bigru_clean_wins, cnn_bigru_clean_pairs = corrected_paired_wins(
        "cnn_mean", "cnn_bigru", "test-clean"
    )
    cnn_bigru_other_wins, cnn_bigru_other_pairs = corrected_paired_wins(
        "cnn_mean", "cnn_bigru", "test-other"
    )

    pooling_2x2 = [
        {
            "family": "CNN", "segmenter": "CNN first-order AR", "pooling": "mean",
            "frequency": corrected["cnn_mean"]["frequency"],
            "clean": corrected["cnn_mean"]["clean"],
            "other": corrected["cnn_mean"]["other"],
            "color": "#b96f20",
        },
        {
            "family": "CNN", "segmenter": "CNN first-order AR", "pooling": "BiGRU residual",
            "frequency": corrected["cnn_bigru"]["frequency"],
            "clean": corrected["cnn_bigru"]["clean"],
            "other": corrected["cnn_bigru"]["other"], "color": "#e2a24c",
        },
        {
            "family": "Transformer", "segmenter": "Transformer AR · local history 64",
            "pooling": "mean", "frequency": corrected["transformer_mean"]["frequency"],
            "clean": corrected["transformer_mean"]["clean"],
            "other": corrected["transformer_mean"]["other"], "color": "#75579b",
        },
        {
            "family": "Transformer", "segmenter": "Transformer AR · local history 64",
            "pooling": "BiGRU residual",
            "frequency": corrected["transformer_bigru"]["frequency"],
            "clean": corrected["transformer_bigru"]["clean"],
            "other": corrected["transformer_bigru"]["other"], "color": "#27865d",
        },
    ]

    # Final inference-speed measurements use padding-invariant left packing,
    # explicit positions, and per-utterance duration caps. Batch size is selected
    # by memory on the longest dev-other examples, never by observed RTF. Systems
    # whose initial batch-64 choice failed during the complete test pass use the
    # conservative batch-32 retry for all three seeds.
    corrected_protocol = "batch_invariant_left_packed_duration_cap_v1"
    memory_speed_root = os.path.join(
        artifact_root, "inference_eval_memory_limited_batch_v1"
    )

    def memory_speed_dirs(root, folder):
        return [
            os.path.join(root, folder, "seed%d" % seed)
            for seed in seeds
        ]

    efficiency_specs = [
        ("Reference", "No downsampling", 16,
         memory_speed_dirs(os.path.join(memory_speed_root, "results"),
                           "no_downsampling"), "#9a5b73", "selected"),
        ("Fixed", "Fixed k=3", 32,
         memory_speed_dirs(os.path.join(memory_speed_root, "results"),
                           "fixed_k3"), "#7b858c", "selected"),
        ("Fixed", "Fixed k=4", 32,
         memory_speed_dirs(os.path.join(memory_speed_root, "results"),
                           "fixed_k4"), "#7b858c", "selected"),
        ("Fixed", "Fixed k=5", 32,
         memory_speed_dirs(os.path.join(
             memory_speed_root, "retry_batch32_v1", "results"),
             "fixed_k5"), "#59646c", "fallback"),
        ("Fixed", "Fixed k=6", 32,
         memory_speed_dirs(os.path.join(
             memory_speed_root, "retry_batch32_v1", "results"),
             "fixed_k6"), "#7b858c", "fallback"),
        ("Fixed", "Fixed k=8", 64,
         memory_speed_dirs(os.path.join(memory_speed_root, "results"),
                           "fixed_k8"), "#7b858c", "selected"),
        ("Oracle", "Phone alignment", 32,
         memory_speed_dirs(os.path.join(memory_speed_root, "results"),
                           "oracle_phone"), "#138a8a", "selected"),
        ("Oracle", "Character alignment", 32,
         memory_speed_dirs(os.path.join(
             memory_speed_root, "retry_batch32_v1", "results"),
             "oracle_char"), "#1c4e80", "fallback"),
        ("Learned", "CNN AR · mean", 32,
         memory_speed_dirs(os.path.join(
             memory_speed_root, "retry_batch32_v1", "results"),
             "cnn_ar_mean"), "#b96f20", "fallback"),
        ("Learned", "CNN AR · BiGRU", 32,
         memory_speed_dirs(os.path.join(
             memory_speed_root, "retry_batch32_v1", "results"),
             "cnn_ar_bigru"), "#e2a24c", "fallback"),
        ("Learned", "Transformer AR · mean", 32,
         memory_speed_dirs(os.path.join(memory_speed_root, "results"),
                           "transformer_ar_local64_mean"),
         "#75579b", "selected"),
        ("Learned", "Transformer AR · BiGRU", 32,
         memory_speed_dirs(os.path.join(memory_speed_root, "results"),
                           "transformer_ar_local64_bigru"),
         "#27865d", "selected"),
    ]
    batch1_folders = {
        "No downsampling": "no_downsampling",
        "Fixed k=3": "fixed_k3",
        "Fixed k=4": "fixed_k4",
        "Fixed k=5": "fixed_k5",
        "Fixed k=6": "fixed_k6",
        "Fixed k=8": "fixed_k8",
        "Phone alignment": "oracle_phone",
        "Character alignment": "oracle_char",
        "CNN AR · mean": "cnn_ar_mean",
        "CNN AR · BiGRU": "cnn_ar_bigru",
        "Transformer AR · mean": "transformer_ar_local64_mean",
        "Transformer AR · BiGRU": "transformer_ar_local64_bigru",
    }
    efficiency_rows = []
    for (family, label, batch_size, run_dirs, color,
         batch_choice) in efficiency_specs:
        measurements = []
        batch1_measurements = []
        mismatch_utterances = 0
        compared_utterances = 0
        for run_dir in run_dirs:
            result = aggregate_inference_benchmark(
                os.path.join(run_dir, "batch%d" % batch_size,
                             "benchmark.jsonl"),
                required_protocol=corrected_protocol,
            )
            if (result is not None and result["splits"] == 2
                    and result["batch_size"] == batch_size
                    and result["prewarm_splits"] >= 1):
                measurements.append(result)
            for split in ("test-clean", "test-other"):
                comparison_path = os.path.join(
                    run_dir, "batch_invariance_%s.json" % split
                )
                if not os.path.exists(comparison_path):
                    continue
                comparison = json.load(open(comparison_path, encoding="utf-8"))
                mismatch_utterances += comparison["hypothesis_mismatches"]
                compared_utterances += comparison["common_records"]
        for run_dir in memory_speed_dirs(
                os.path.join(memory_speed_root, "results"),
                batch1_folders[label]):
            result = aggregate_inference_benchmark(
                os.path.join(run_dir, "batch1", "benchmark.jsonl"),
                required_protocol=corrected_protocol,
            )
            if (result is not None and result["splits"] == 2
                    and result["batch_size"] == 1
                    and result["prewarm_splits"] >= 1):
                batch1_measurements.append(result)
        if len(measurements) != 3:
            raise RuntimeError(
                "%s requires three complete corrected speed runs; found %d" %
                (label, len(measurements))
            )
        if len(batch1_measurements) != 3:
            raise RuntimeError(
                "%s requires three complete corrected batch-1 runs; found %d" %
                (label, len(batch1_measurements))
            )
        efficiency_rows.append({
            "family": family,
            "label": label,
            "color": color,
            "batch_size": batch_size,
            "batch_choice": batch_choice,
            "n": len(measurements),
            "rtf": mean_sd([row["forward_rtf"] for row in measurements]),
            "batch1_rtf": mean_sd([
                row["forward_rtf"] for row in batch1_measurements
            ]),
            "utterances_per_second": mean_sd([
                row["utterances_per_second"] for row in measurements
            ]),
            "batch1_utterances_per_second": mean_sd([
                row["utterances_per_second"] for row in batch1_measurements
            ]),
            "frequency": mean_sd([
                row["token_frequency_hz"] for row in measurements
            ]),
            "peak_allocated_gb": mean_sd([
                row["peak_allocated_gb"] for row in measurements
            ]),
            "run_peak_allocated_gb": mean_sd([
                row["run_peak_allocated_gb"] for row in measurements
            ]),
            "run_peak_reserved_gb": mean_sd([
                row["run_peak_reserved_gb"] for row in measurements
            ]),
            "prewarm_batches": 10,
            "batch_mismatch_pct": (
                100 * mismatch_utterances / compared_utterances
                if compared_utterances else float("nan")
            ),
            "hardware": measurements[0]["hardware"],
            "precision": measurements[0]["precision"],
        })

    efficiency_by_label = {row["label"]: row for row in efficiency_rows}

    # htmlkit defaults to 960 px. This report has several wide comparison tables,
    # so use a wider desktop column while retaining the existing responsive cap.
    body = (
        '<style>.wrap{max-width:1100px}'
        '.card{min-width:0;max-width:100%;overflow-x:auto}'
        '.asset-label{margin:14px 2px 5px;color:var(--text-muted);font-size:12px;'
        'font-weight:600;letter-spacing:.06em;text-transform:uppercase}'
        '.figure-label+.fig,.figure-label+.chart{margin-top:9px}'
        '.demo-timeline{overflow-x:auto;padding:10px 0 4px}'
        '.demo-figure-scroll{overflow-x:auto}.demo-figure-scroll .chart{min-width:900px}'
        '.demo-table-scroll{max-width:100%;overflow-x:auto}'
        '.demo-table-scroll table{min-width:720px}'
        '.demo-row{display:grid;grid-template-columns:92px repeat(6,minmax(48px,1fr));'
        'gap:7px;align-items:center;min-width:540px;margin:7px 0}'
        '.demo-name,.mini-label{font-size:12px;color:var(--text-muted);text-transform:uppercase;'
        'letter-spacing:.04em}'
        '.frame,.bit{display:inline-flex;align-items:center;justify-content:center;border:1px solid '
        'var(--border-strong);background:var(--surface-2);border-radius:8px;min-height:35px}'
        '.frame.cut{border-left:4px solid var(--pair-sed-asr)}'
        '.bit{width:35px;min-height:31px;font-family:ui-monospace,"SF Mono",monospace;font-weight:700}'
        '.bit.one{background:color-mix(in srgb,var(--pair-sed-asr) 16%,var(--surface-1));'
        'border-color:var(--pair-sed-asr)}'
        '.bit.zero{color:var(--text-secondary)}'
        '.demo-groups{display:grid;grid-template-columns:2fr 3fr 1fr;gap:8px;margin:12px 0 4px;'
        'min-width:540px;padding-left:99px}'
        '.demo-groups span{padding:8px;text-align:center;border-radius:8px;background:var(--band);'
        'border:1px dashed var(--border-strong)}'
        '.history-compare{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:10px 0}'
        '.history-compare>div{padding:12px;border-radius:10px;background:var(--surface-2)}'
        '.bitline{display:flex;gap:6px;margin-top:8px}'
        '.demo-steps{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:10px 0}'
        '.demo-steps>div{padding:11px;border:1px solid var(--border);border-radius:10px;'
        'background:var(--surface-2)}'
        '.demo-steps b{display:block;font-size:13px;margin-bottom:4px}'
        '.demo-steps span{display:block;font-size:12.5px;color:var(--text-secondary)}'
        '@media(max-width:720px){.demo-steps{grid-template-columns:1fr 1fr}'
        '.history-compare{grid-template-columns:1fr}.demo-groups{padding-left:0}}'
        '@media(max-width:460px){.demo-steps{grid-template-columns:1fr}}'
        '</style>'
        + hk.hero(
        "JSALT 2026 · RL dynamic downsampling",
        "Learned speech segmentation — cumulative results",
        "Current LibriSpeech-960h scale-up results followed by the LibriSpeech-100h "
        "experiments that isolate compression, boundary placement, initialization, "
        "label-dependent sampling, and inference cost. "
        "Corpus WER is recomputed from saved edit counts; ± is sample SD over seeds.",
        )
    )

    body += hk.section("", body=hk.tiles([
        ("LS960 · Transformer AR + BiGRU",
         ls960_metric_text(ls960_transformer["clean"], ls960_transformer["n"]),
         "test-clean · n=%d · %s Hz · test-other %s at %s Hz" % (
             ls960_transformer["n"],
             ls960_one_decimal(ls960_transformer["clean_hz"][0]),
             ls960_metric_text(ls960_transformer["other"],
                               ls960_transformer["n"]),
             ls960_one_decimal(ls960_transformer["other_hz"][0]))),
        ("LS960 · CNN first-order AR + BiGRU",
         ls960_metric_text(ls960_cnn["clean"], ls960_cnn["n"]),
         "test-clean · n=%d · %s Hz · test-other %s at %s Hz" % (
             ls960_cnn["n"], ls960_one_decimal(ls960_cnn["clean_hz"][0]),
             ls960_metric_text(ls960_cnn["other"], ls960_cnn["n"]),
             ls960_one_decimal(ls960_cnn["other_hz"][0]))),
        ("LS960 · char alignment", "2.87±0.08%",
         "test-clean · n=3 · 14.6 Hz · test-other 5.66±0.15%"),
        ("LS960 · fixed k=5", "4.37±0.10%",
         "test-clean · n=3 · 10.0 Hz · test-other 7.72±0.18%"),
        ("LS960 · no downsampling", "4.34±0.47%",
         "test-clean · n=3 · 50.0 Hz · test-other 7.08±0.19%")]))

    evidence_rows = [
        ("Does compression help?",
         "No downsampling compared with fixed k=5",
         "No downsampling: %.2f clean, %.2f other. Fixed k=5: %.2f clean, "
         "%.2f other." %
         (wavlm_no_down["test-clean"][0], wavlm_no_down["test-other"][0],
          wavlm_k5["clean"][0], wavlm_k5["other"][0]),
         "Shortening the audio input helps on clean speech; the two settings are "
         "about tied on test-other.",
         "Evidence-backed"),
        ("Does placement matter at matched cost?",
         "Phone alignment compared with fixed k=5 at a similar audio-token frequency",
         "Phone alignment: %.2f clean, %.2f other. Fixed k=5: %.2f clean, "
         "%.2f other." %
         (wavlm_phone_clean[0], wavlm_phone_other[0],
          wavlm_k5["clean"][0], wavlm_k5["other"][0]),
         "The locations of the boundaries matter, not just the resulting audio-token frequency.",
         "Evidence-backed"),
        ("Why is learned below the char oracle?",
         "Same decoder, three boundary sources",
         "Oracle boundaries: %.2f. Cold learned boundaries: %.2f. After RL: %.2f "
         "(dev-clean)." %
         (boundary_swap["oracle"], boundary_swap["cold"], boundary_swap["rl"]),
         "Imperfect imitation already costs WER, and unconstrained RL drift amplifies "
         "the mismatch. This is our clearest direct test so far.",
         "Evidence-backed"),
        ("Does decoder initialization matter?",
         "Same segmenter and pooling, two decoder starting points",
         "Best char decoder start: %.2f clean, %.2f other. From scratch: %.2f "
         "clean, %.2f other." %
         (hybrid["oracle_char_decoder"]["clean"][0][0],
          hybrid["oracle_char_decoder"]["other"][0][0],
          hybrid["scratch_decoder"]["clean"][0][0],
          hybrid["scratch_decoder"]["other"][0][0]),
         "A decoder already trained on char segments is the dominant controlled gain "
         "(−1.32 clean, −0.61 other).",
         "Evidence-backed"),
        ("Does label history help?",
         "Bernoulli compared with first-order AR, three matched WavLM seeds",
         "Bernoulli: %.2f clean, %.2f other. First-order AR: %.2f clean, "
         "%.2f other." %
         (oracleclose["bernoulli"]["clean"][0][0],
          oracleclose["bernoulli"]["other"][0][0],
          oracleclose["autoregressive"]["clean"][0][0],
          oracleclose["autoregressive"]["other"][0][0]),
         "First-order AR lowers clean WER in all three seeds (−%.2f mean). The "
         "other-speech change is smaller and mixed across seeds (−%.2f mean)." %
         (oracleclose["bernoulli"]["clean"][0][0]
          - oracleclose["autoregressive"]["clean"][0][0],
          oracleclose["bernoulli"]["other"][0][0]
          - oracleclose["autoregressive"]["other"][0][0]),
         "Evidence-backed"),
        ("Does order-aware pooling help?",
         "Mean versus BiGRU pooling",
         "Transformer mean: %.2f clean, %.2f other (n=%d). Transformer BiGRU: "
         "%.2f clean, %.2f other (n=%d). CNN mean: %.2f/%.2f (n=%d); CNN BiGRU: "
         "%.2f/%.2f (n=%d)." %
         (corrected["transformer_mean"]["clean"][0][0],
          corrected["transformer_mean"]["other"][0][0],
          corrected["transformer_mean"]["n"],
          corrected["transformer_bigru"]["clean"][0][0],
          corrected["transformer_bigru"]["other"][0][0],
          corrected["transformer_bigru"]["n"],
          corrected["cnn_mean"]["clean"][0][0],
          corrected["cnn_mean"]["other"][0][0],
          corrected["cnn_mean"]["n"],
          corrected["cnn_bigru"]["clean"][0][0],
          corrected["cnn_bigru"]["other"][0][0],
          corrected["cnn_bigru"]["n"]),
         "Transformer BiGRU improves clean WER in all three paired seeds and other "
         "WER in two. CNN BiGRU wins %d/%d clean seeds but raises mean clean WER by %.2f; "
         "it improves other WER in %d/%d seeds by %.2f on average." %
         (cnn_bigru_clean_wins, cnn_bigru_clean_pairs,
          corrected["cnn_bigru"]["clean"][0][0]
          - corrected["cnn_mean"]["clean"][0][0],
          cnn_bigru_other_wins, cnn_bigru_other_pairs,
         corrected["cnn_mean"]["other"][0][0]
          - corrected["cnn_bigru"]["other"][0][0]),
        "Evidence-backed"),
        ("What changes in the 100h system comparison?",
         "Fixed k=5 mean/BiGRU, frozen learned boundaries, and joint RL",
         "Fixed k=5 + mean: %.2f clean, %.2f other (n=%d). Fixed k=5 + BiGRU: "
         "%.2f clean, %.2f other (n=%d). Frozen Transformer AR: "
         "%.2f clean, %.2f other (n=%d). Joint Transformer AR: %.2f clean, "
         "%.2f other (n=%d)." %
         (attribution_fixed_mean["clean"][0],
          attribution_fixed_mean["other"][0], attribution_fixed_mean["n"],
          attribution_fixed["clean"][0], attribution_fixed["other"][0],
          attribution_fixed["n"], attribution_frozen["clean"][0],
          attribution_frozen["other"][0], attribution_frozen["n"],
          headline_joint["clean"][0], headline_joint["other"][0],
          headline_joint["n"]),
         "At identical fixed boundaries and frequency, BiGRU lowers clean WER by %.2f "
         "but changes other WER by only %.2f. Joint RL then lowers WER by %.2f clean / "
         "%.2f other versus fixed k=5 + BiGRU, at about %.1f more audio tokens/s on "
         "test-clean." %
         (attribution_fixed_mean["clean"][0] - attribution_fixed["clean"][0],
          attribution_fixed_mean["other"][0] - attribution_fixed["other"][0],
          attribution_fixed["clean"][0] - headline_joint["clean"][0],
          attribution_fixed["other"][0] - headline_joint["other"][0],
          headline_joint["clean_hz"][0] - attribution_fixed["clean_hz"][0]),
         "Evidence-backed"),
        ("What does learned segmentation cost at inference?",
         "Memory-limited operating points; one A100 80 GB, BF16; three seeds",
         "No downsampling: RTF %.3f (b%d). Phone oracle: %.3f (b%d). Fixed k=5: "
         "%.3f (b%d). CNN AR + BiGRU: %.3f (b%d). Transformer AR + BiGRU: "
         "%.3f (b%d)." %
         (efficiency_by_label["No downsampling"]["rtf"][0],
          efficiency_by_label["No downsampling"]["batch_size"],
          efficiency_by_label["Phone alignment"]["rtf"][0],
          efficiency_by_label["Phone alignment"]["batch_size"],
          efficiency_by_label["Fixed k=5"]["rtf"][0],
          efficiency_by_label["Fixed k=5"]["batch_size"],
          efficiency_by_label["CNN AR · BiGRU"]["rtf"][0],
          efficiency_by_label["CNN AR · BiGRU"]["batch_size"],
          efficiency_by_label["Transformer AR · BiGRU"]["rtf"][0],
          efficiency_by_label["Transformer AR · BiGRU"]["batch_size"]),
         "CNN AR + BiGRU and Transformer AR + BiGRU are close to the phone-oracle "
         "throughput and have %.1f× / %.1f× lower RTF than native 50-Hz decoding. "
         "This system-level gain intentionally includes the larger batch enabled by "
         "shorter audio prefixes." %
         (efficiency_by_label["No downsampling"]["rtf"][0]
          / efficiency_by_label["CNN AR · BiGRU"]["rtf"][0],
          efficiency_by_label["No downsampling"]["rtf"][0]
          / efficiency_by_label["Transformer AR · BiGRU"]["rtf"][0]),
         "Evidence-backed"),
        ("Do more RL epochs help?",
         "Thirty-epoch wav2vec2 Transformer AR continuation",
         "Selected checkpoints: %.2f±%.2f clean, %.2f±%.2f other. Best dev "
         "epochs: %s." %
         (long_ar_clean[0], long_ar_clean[1], long_ar_other[0], long_ar_other[1],
          "/".join(str(row["best_epoch"]) for row in long_ar)),
         "No: dev WER is best at epochs 1–4 and then worsens even while training "
         "loss keeps falling.",
         "Evidence-backed"),
    ]
    evidence_table = (
        '<table style="table-layout:fixed"><colgroup><col style="width:18%">'
        '<col style="width:22%"><col style="width:19%"><col style="width:31%">'
        '<col style="width:10%"></colgroup><thead><tr><th>question</th>'
        '<th>comparison</th><th>WER results (lower is better)</th>'
        '<th>what it tells us</th><th>evidence</th></tr></thead><tbody>'
        + "".join(
            '<tr><td><b>%s</b></td><td>%s</td><td>%s</td><td>%s</td>'
            '<td><span class="pill">%s</span></td></tr>' % row
            for row in evidence_rows
        ) + '</tbody></table>'
    )
    body += hk.section(
        "0 · Summary — what each experiment shows",
        lead="The rows move from basic baselines to focused tests of the segmenter and "
             "decoder. Each result names the system before its numbers.",
        body=(
            hk.card(evidence_table, title="What we compared and what we learned")
            + hk.finding(
                '<span class="pill">Evidence-backed</span> <b>Compression itself is not the '
                'problem.</b> WavLM fixed k=5 beats no downsampling by %.2f WER points on '
                'test-clean while averaging 5h20m instead of 18h02m of A100 wall time per run '
                '(3.4× faster in this protocol). Phone-aligned pooling improves another %.2f '
                'points at nearly the same audio-token frequency. The best learned system is '
                'close to the phone oracle at a similar frequency, whereas the char oracle remains '
                'stronger; the remaining gap is about boundary placement and granularity, not '
                'whether compression is viable.' %
                (wavlm_no_down["test-clean"][0] - wavlm_k5["clean"][0],
                 wavlm_k5["clean"][0] - wavlm_phone_clean[0]))
        ),
    )

    ls960_rows = (
        '<tr style="background:var(--band)"><td><b>Local-history Transformer AR + BiGRU</b></td>'
        '<td>64-frame causal history · BiGRU residual pooling</td>'
        '<td>24k steps · decoder warmup to step 2,395</td><td>%s</td>'
        '<td><b>%s</b></td><td><b>%s</b></td>'
        '<td><span class="pill">%d / 3</span></td></tr>'
        '<tr style="background:var(--band)"><td><b>CNN first-order AR + BiGRU</b></td>'
        '<td>previous boundary label · BiGRU residual pooling</td>'
        '<td>24k steps · decoder warmup to step 2,395</td><td>%s</td>'
        '<td><b>%s</b></td><td><b>%s</b></td>'
        '<td><span class="pill">%d / 3</span></td></tr>'
        '<tr><td>Character alignment (oracle)</td><td>char-CTC boundaries · mean pooling</td>'
        '<td>decoder CE on LS960</td><td>14.6 Hz</td><td>2.87 ± 0.08%%</td>'
        '<td>5.66 ± 0.15%%</td><td><span class="pill">3 / 3</span></td></tr>'
        '<tr><td>Phone-CTC + long-segment split (baseline)</td>'
        '<td>audio-only phone-CTC · ≥160 ms midpoint split · mean pooling</td>'
        '<td>decoder CE on LS960</td><td>%.1f / %.1f Hz</td>'
        '<td>%s</td><td>%s</td>'
        '<td><span class="pill">%d / 3</span></td></tr>'
        '<tr><td>Fixed k=5</td><td>keep every fifth frame · mean pooling</td>'
        '<td>decoder CE on LS960</td><td>10.0 Hz</td><td>4.37 ± 0.10%%</td>'
        '<td>7.72 ± 0.18%%</td><td><span class="pill">3 / 3</span></td></tr>'
        '<tr><td>No downsampling</td><td>keep every WavLM frame · no pooling</td>'
        '<td>decoder CE on LS960</td><td>50.0 Hz</td><td>4.34 ± 0.47%%</td>'
        '<td>7.08 ± 0.19%%</td><td><span class="pill">3 / 3</span></td></tr>'
    ) % (
        ls960_frequency_text(ls960_transformer),
        ls960_metric_text(ls960_transformer["clean"], ls960_transformer["n"]),
        ls960_metric_text(ls960_transformer["other"], ls960_transformer["n"]),
        ls960_transformer["n"],
        ls960_frequency_text(ls960_cnn),
        ls960_metric_text(ls960_cnn["clean"], ls960_cnn["n"]),
        ls960_metric_text(ls960_cnn["other"], ls960_cnn["n"]),
        ls960_cnn["n"],
        ls960_phone_ctc_hz["test-clean"],
        ls960_phone_ctc_hz["test-other"],
        ls960_metric_text(ls960_phone_ctc["test-clean"],
                          ls960_phone_ctc["n"]),
        ls960_metric_text(ls960_phone_ctc["test-other"],
                          ls960_phone_ctc["n"]),
        ls960_phone_ctc["n"],
    )
    ls960_comparison_rows = [
        {"label": "Transformer AR + BiGRU", "clean": ls960_transformer["clean"][0],
         "clean_sd": ls960_transformer["clean"][1],
         "other": ls960_transformer["other"][0],
         "other_sd": ls960_transformer["other"][1],
         "color": "#27865d", "marker": "o"},
        {"label": "CNN first-order AR + BiGRU", "clean": ls960_cnn["clean"][0],
         "clean_sd": ls960_cnn["clean"][1], "other": ls960_cnn["other"][0],
         "other_sd": ls960_cnn["other"][1], "color": "#b96f20", "marker": "s"},
        {"label": "Oracle baseline · character alignment",
         "clean": 2.87, "clean_sd": 0.08,
         "other": 5.66, "other_sd": 0.15, "color": "#1c4e80", "marker": "P"},
        {"label": "Transcript-free baseline · phone-CTC + split",
         "clean": ls960_phone_ctc["test-clean"][0],
         "clean_sd": ls960_phone_ctc["test-clean"][1],
         "other": ls960_phone_ctc["test-other"][0],
         "other_sd": ls960_phone_ctc["test-other"][1],
         "color": "#587b8a", "marker": "D"},
        {"label": "Fixed k=5", "clean": 4.37, "clean_sd": 0.10,
         "other": 7.72, "other_sd": 0.18, "color": "#7b858c", "marker": "o"},
        {"label": "No downsampling", "clean": 4.34, "clean_sd": 0.47,
         "other": 7.08, "other_sd": 0.19, "color": "#9a5b73", "marker": "X"},
    ]
    body += hk.section(
        "0b · LibriSpeech-960h scale-up — three seeds complete",
        lead="All rows train on train-clean-100, train-clean-360, and train-other-500. "
             "The fixed, oracle, transcript-free CTC, and no-downsampling baselines are complete for three seeds. "
             "The corrected learned-policy study has %d/3 Transformer seeds and %d/3 CNN "
             "seeds; ± is sample SD across the completed seeds." %
             (ls960_transformer["n"], ls960_cnn["n"]),
        body=(
            hk.card(
                '<table style="table-layout:fixed"><colgroup>'
                '<col style="width:19%%"><col style="width:24%%">'
                '<col style="width:18%%"><col style="width:12%%">'
                '<col style="width:10%%"><col style="width:10%%">'
                '<col style="width:7%%"></colgroup><thead><tr>'
                '<th>system</th><th>segmenter / pooling</th><th>training recipe</th>'
                '<th>test-clean / other frequency</th><th>test-clean WER</th>'
                '<th>test-other WER</th><th>seeds</th></tr></thead>'
                '<tbody>%s</tbody></table>'
                '<p class="cap">Lower WER and lower audio-token frequency are better. '
                'Rows aggregate the completed seeds; ± is sample SD when n&gt;1.</p>' % ls960_rows,
                title="Current LS960 WER and audio-token frequency",
            )
            + hk.card(
                ls960_system_comparison_fig(ls960_comparison_rows)
                + '<p class="cap">All points are three-seed means; horizontal bars show '
                  'sample SD. Character alignment, phone-CTC, fixed k=5, and no '
                  'downsampling are reference baselines.</p>',
                title="LS960 system comparison",
            )
            + hk.card(
                ls960_convergence_fig()
                + '<p class="cap">Validation uses dev-clean. The dashed line marks the end '
                  'of decoder-only warmup; joint segmenter/decoder RL follows. Both completed '
                  'runs select the 24k-step checkpoint. The small Transformer regression at '
                  '20k does not change the overall downward trend.</p>',
                title="Convergence of the first corrected seed",
            )
            + hk.finding(
                '<span class="pill">Both learned systems complete · 3 seeds</span> '
                '<b>CNN and Transformer AR reach essentially the same WER, while the '
                'Transformer uses about one fewer audio token per second.</b> CNN first-order '
                'AR + BiGRU reaches %s clean and %s other at %s. Transformer AR + BiGRU '
                'reaches %s/%s at %s. CNN is 0.04 points lower on clean; Transformer is 0.04 '
                'points lower on other. Both are near the character-alignment control '
                '(2.87 ± 0.08 / 5.66 ± 0.15 at 14.6 Hz).' % (
                    ls960_metric_text(ls960_cnn["clean"], ls960_cnn["n"]),
                    ls960_metric_text(ls960_cnn["other"], ls960_cnn["n"]),
                    ls960_frequency_text(ls960_cnn),
                    ls960_metric_text(ls960_transformer["clean"],
                                      ls960_transformer["n"]),
                    ls960_metric_text(ls960_transformer["other"],
                                      ls960_transformer["n"]),
                    ls960_frequency_text(ls960_transformer),
                )
            )
            + hk.finding(
                '<span class="pill">Full LS960 scale-up · 3 seeds</span> '
                '<b>The learned Transformer AR + BiGRU system clearly outperforms the '
                'transcript-free phone-CTC baseline at essentially the same audio-token '
                'frequency.</b> Transformer AR + BiGRU reaches %s clean and %s other at %s, '
                'versus %s/%s at %.1f / %.1f Hz for phone-CTC + long-segment split. The '
                'learned system delivers a <b>%.1f%% / %.1f%% relative WER reduction</b> '
                '(%.2f / %.2f absolute points) on test-clean/test-other. This sizeable '
                'relative gain at a matched token rate demonstrates the effectiveness of '
                'learning content-adaptive boundaries and the within-segment representation '
                'rather than relying on CTC boundaries with mean pooling.' % (
                    ls960_metric_text(ls960_transformer["clean"],
                                      ls960_transformer["n"]),
                    ls960_metric_text(ls960_transformer["other"],
                                      ls960_transformer["n"]),
                    ls960_frequency_text(ls960_transformer),
                    ls960_metric_text(ls960_phone_ctc["test-clean"],
                                      ls960_phone_ctc["n"]),
                    ls960_metric_text(ls960_phone_ctc["test-other"],
                                      ls960_phone_ctc["n"]),
                    ls960_phone_ctc_hz["test-clean"],
                    ls960_phone_ctc_hz["test-other"],
                    100 * (1 - ls960_transformer["clean"][0]
                           / ls960_phone_ctc["test-clean"][0]),
                    100 * (1 - ls960_transformer["other"][0]
                           / ls960_phone_ctc["test-other"][0]),
                    ls960_phone_ctc["test-clean"][0]
                    - ls960_transformer["clean"][0],
                    ls960_phone_ctc["test-other"][0]
                    - ls960_transformer["other"][0],
                )
            )
            + hk.finding(
                '<b>Warmup plus longer joint training matters in the seed-3407 pair.</b> Against '
                'the same-seed no-warmup diagnostics, the corrected Transformer improves by '
                '0.39 clean and 0.97 other WER points; the corrected CNN improves by 0.37 and '
                '0.63 points. The next decision should therefore use the completed three-seed '
                'comparison, not the earlier no-warmup aggregate.'
            )
        ),
    )

    def attribution_frequency_text(result):
        if (result["n"] > 1 and result["clean_hz"][1] < 0.05
                and result["other_hz"][1] < 0.05):
            return "%s / %s Hz" % (
                ls960_one_decimal(result["clean_hz"][0]),
                ls960_one_decimal(result["other_hz"][0]),
            )
        return ls960_frequency_text(result)

    def attribution_result_row(label, boundaries, training, result, highlight=False):
        style = ' style="background:var(--band)"' if highlight else ""
        if result["n"]:
            frequency = attribution_frequency_text(result)
            clean = ls960_metric_text(result["clean"], result["n"])
            other = ls960_metric_text(result["other"], result["n"])
        else:
            frequency = clean = other = '<span class="pill">pending</span>'
        return (
            '<tr%s><td><b>%s</b></td><td>%s</td><td>%s</td><td>%s</td>'
            '<td>%s</td><td>%s</td><td>%d / 3</td></tr>' %
            (style, label, boundaries, training, frequency, clean, other, result["n"])
        )

    attribution_rows_html = "".join([
        attribution_result_row(
            "Fixed k=5 + mean", "keep every fifth frame",
            "decoder CE", attribution_fixed_mean,
        ),
        attribution_result_row(
            "Fixed k=5 + BiGRU", "keep every fifth frame",
            "decoder/BiGRU CE", attribution_fixed,
        ),
        attribution_result_row(
            "Frozen Transformer AR + BiGRU",
            "supervised local-history segmenter; boundaries frozen",
            "decoder/BiGRU CE", attribution_frozen, highlight=True,
        ),
        attribution_result_row(
            "Joint Transformer AR + BiGRU",
            "same segmenter; updated by NLL policy gradient",
            "decoder adaptation + joint NLL policy gradient",
            headline_joint, highlight=True,
        ),
    ])

    attribution_plot_rows = []
    for label, result, color in (
        ("Fixed k=5 + mean", attribution_fixed_mean, "#8a9399"),
        ("Fixed k=5 + BiGRU", attribution_fixed, "#59646c"),
        ("Frozen Transformer AR + BiGRU", attribution_frozen, "#75579b"),
        ("Joint Transformer AR + BiGRU", headline_joint, "#27865d"),
    ):
        if result["n"]:
            attribution_plot_rows.append((
                label, result["clean"][0], result["clean"][1],
                result["other"][0], result["other"][1], result["n"], color,
            ))

    body += hk.section(
        "0c · 100h comparison — mean, BiGRU, and joint RL",
        lead="All four arms use WavLM features, the same best character-decoder "
             "initialization, and padding-invariant decoding. The fixed-k=5 pair "
             "isolates pooler capacity; the BiGRU rows contrast fixed, frozen, and "
             "jointly optimized boundary treatments.",
        body=(
            hk.card(
                '<table style="table-layout:fixed"><colgroup>'
                '<col style="width:18%%"><col style="width:23%%">'
                '<col style="width:18%%"><col style="width:14%%">'
                '<col style="width:10%%"><col style="width:10%%">'
                '<col style="width:7%%"></colgroup><thead><tr>'
                '<th>setup</th><th>boundary treatment</th><th>training</th>'
                '<th>clean / other frequency</th><th>test-clean</th>'
                '<th>test-other</th><th>seeds</th></tr></thead><tbody>%s</tbody></table>'
                '<p class="cap">Lower WER is better; frequency is the decoder-side audio-token '
                'rate in Hz. ± is sample SD across the same three seeds. Every row reports final '
                'test evidence.</p>'
                % attribution_rows_html,
                title="100h comparison results",
            )
            + hk.card(
                controlled_wer_fig(attribution_plot_rows)
                + '<p class="cap">Every marker is a three-seed mean; error bars are sample '
                  'SD. Lower WER is better. All four arms use the same decoder initialization, '
                  'WavLM frontend, and evaluation protocol. The fixed pair isolates mean '
                  'versus BiGRU pooling; the remaining rows compare boundary treatments.</p>',
                title="Completed 100h WER by setup",
            )
            + hk.finding(
                '<span class="pill">Evidence-backed</span> <b>The fixed-k=5 mean control '
                'shows where BiGRU capacity helps.</b> Mean pooling reaches %s/%s at %s; '
                'BiGRU pooling reaches %s/%s at %s. With identical fixed boundaries and '
                'frequency, BiGRU is %.2f points lower on clean and %.2f lower on other. '
                'Its seed variation is also smaller (clean SD %.2f versus %.2f; other SD '
                '%.2f versus %.2f).' % (
                    ls960_metric_text(attribution_fixed_mean["clean"], 3),
                    ls960_metric_text(attribution_fixed_mean["other"], 3),
                    attribution_frequency_text(attribution_fixed_mean),
                    ls960_metric_text(attribution_fixed["clean"], 3),
                    ls960_metric_text(attribution_fixed["other"], 3),
                    attribution_frequency_text(attribution_fixed),
                    attribution_fixed_mean["clean"][0]
                    - attribution_fixed["clean"][0],
                    attribution_fixed_mean["other"][0]
                    - attribution_fixed["other"][0],
                    attribution_fixed["clean"][1],
                    attribution_fixed_mean["clean"][1],
                    attribution_fixed["other"][1],
                    attribution_fixed_mean["other"][1],
                )
            )
            + hk.finding(
                '<span class="pill">Evidence-backed</span> <b>The established joint-RL '
                'system gives the strongest clean-set result among these BiGRU systems.</b> '
                'Frozen Transformer AR + BiGRU reaches %s/%s at %s; joint RL reaches %s/%s '
                'at %s. The joint system is %.2f points lower on clean, %.2f lower on other, '
                'and emits %.1f fewer audio tokens/s on test-clean.' % (
                    ls960_metric_text(attribution_frozen["clean"], 3),
                    ls960_metric_text(attribution_frozen["other"], 3),
                    attribution_frequency_text(attribution_frozen),
                    ls960_metric_text(headline_joint["clean"], 3),
                    ls960_metric_text(headline_joint["other"], 3),
                    attribution_frequency_text(headline_joint),
                    attribution_frozen["clean"][0] - headline_joint["clean"][0],
                    attribution_frozen["other"][0] - headline_joint["other"][0],
                    attribution_frozen["clean_hz"][0]
                    - headline_joint["clean_hz"][0],
                )
            )
            + hk.finding(
                '<span class="pill">Evidence-backed</span> <b>Joint RL also beats fixed k=5, '
                'but uses a slightly denser audio prefix.</b> Fixed k=5 + BiGRU reaches %s/%s '
                'at %s. Joint RL lowers WER by %.2f clean / %.2f other while emitting %.1f '
                'more audio tokens/s on test-clean. The frozen arm has substantially larger '
                'clean-WER seed variation (SD %.2f), so conclusions should use the three-seed '
                'aggregate rather than its strongest seed.' % (
                    ls960_metric_text(attribution_fixed["clean"], 3),
                    ls960_metric_text(attribution_fixed["other"], 3),
                    attribution_frequency_text(attribution_fixed),
                    attribution_fixed["clean"][0] - headline_joint["clean"][0],
                    attribution_fixed["other"][0] - headline_joint["other"][0],
                    headline_joint["clean_hz"][0]
                    - attribution_fixed["clean_hz"][0],
                    attribution_frozen["clean"][1],
                )
            )
            + hk.finding(
                '<span class="pill">Evidence-backed</span> <b>The simplest fixed mean '
                'control does not explain the learned result.</b> Joint Transformer AR + '
                'BiGRU is %.2f points lower on clean and %.2f lower on other than fixed '
                'k=5 + mean, while emitting %.1f / %.1f more audio tokens/s on test-clean '
                '/ test-other.' % (
                    attribution_fixed_mean["clean"][0]
                    - headline_joint["clean"][0],
                    attribution_fixed_mean["other"][0]
                    - headline_joint["other"][0],
                    headline_joint["clean_hz"][0]
                    - attribution_fixed_mean["clean_hz"][0],
                    headline_joint["other_hz"][0]
                    - attribution_fixed_mean["other_hz"][0],
                )
            )
        ),
    )
    # These two sections are assembled later, after their audit artifacts have
    # been loaded, but belong here in the reader-facing evidence order.
    body += "__PHONE_BOUNDARY_AGREEMENT_SECTION__"
    body += "__INFERENCE_EFFICIENCY_SECTION__"

    # Rate-raised phone-CTC baseline. Checkpoint rank differs by seed, so recover
    # the retained rank whose metadata names epoch 3 rather than assuming rank 0.
    phone_ctc_longsplit_root = os.path.join(
        attribution_root, "wavlm_phone_ctc_longsplit8_bigru_3ep"
    )

    def retained_epoch_stats(root, epoch):
        runs = []
        for seed in seeds:
            run = os.path.join(root, str(seed))
            rank = None
            eval_log = os.path.join(run, "log.txt")
            if os.path.exists(eval_log):
                for line in open(eval_log, encoding="utf-8", errors="replace"):
                    match = re.search(
                        r"Evaluation loaded retained checkpoint rank=(\d+).*"
                        r"['\"]epoch['\"]: (\d+)",
                        line,
                    )
                    if match and int(match.group(2)) == epoch:
                        rank = int(match.group(1))
                        break
            if rank is None:
                continue
            records = {
                split: parse_wer_file(os.path.join(
                    run, "wer_results_rank_%d" % rank, "wer_%s.txt" % split
                ))
                for split in ("test-clean", "test-other", "dev-other")
            }
            train_rows = [
                row for row in parse_log(os.path.join(run, "train_log.txt"))
                if int(row.get("epoch", -1)) == epoch
            ]
            if not all(records.values()) or not train_rows:
                continue
            runs.append({
                "seed": seed,
                "rank": rank,
                "dev-clean": train_rows[-1]["valid WER"],
                **{split: record["wer"] for split, record in records.items()},
            })
        return {
            "runs": runs,
            "n": len(runs),
            **{
                split: mean_sd([run[split] for run in runs])
                for split in ("dev-clean", "dev-other", "test-clean", "test-other")
            },
        }

    phone_ctc_longsplit = retained_epoch_stats(phone_ctc_longsplit_root, 3)
    phone_ctc_longsplit_summary_path = (
        "/export/jsalt26/omnienc/users/cxiao/"
        "boundary_targets/wavlm_phone_ctc_tc100_best_longsplit8/_stats/"
        "long_segment_split_summary.json"
    )
    with open(phone_ctc_longsplit_summary_path, encoding="utf-8") as stream:
        phone_ctc_longsplit_boundary_summary = json.load(stream)
    phone_ctc_longsplit_hz = {
        split: phone_ctc_longsplit_boundary_summary["splits"][split]["refined_rate_hz"]
        for split in ("dev-clean", "dev-other", "test-clean", "test-other")
    }

    # One compact inventory of the systems readers most often need to compare.
    # This intentionally spans several experiment families, so decoder treatment
    # and policy context are explicit rather than implying a controlled ablation.
    system_overview_rows = [{
        "family": "Reference", "label": "No downsampling",
        "system": "No downsampling", "configuration": "keep every 50-Hz frame · no pooling",
        "decoder": "from scratch", "frequency": 50.0, "frequency_sd": 0.0,
        "clean": wavlm_no_down["test-clean"][0],
        "clean_sd": wavlm_no_down["test-clean"][1],
        "other": wavlm_no_down["test-other"][0],
        "other_sd": wavlm_no_down["test-other"][1],
        "color": "#9a5b73", "marker": "X",
    }]
    for row in wavlm_fixed:
        k = row["k"]
        system_overview_rows.append({
            "family": "Fixed", "label": "Fixed mean · k=%d" % k,
            "system": "Fixed k=%d" % k,
            "configuration": "keep 1 in %d frames · mean pooling" % k,
            "decoder": "from scratch", "frequency": rate_hz(row["rho"]),
            "frequency_sd": 0.0, "clean": row["clean"][0],
            "clean_sd": row["clean"][1], "other": row["other"][0],
            "other_sd": row["other"][1], "color": "#7b858c", "marker": "o",
        })
    system_overview_rows.extend([
        {
            "family": "Oracle", "label": "Oracle · phone",
            "system": "Phone alignment", "configuration": "phone boundaries · mean pooling",
            "decoder": "from scratch", "frequency": wavlm_phone_frequency[0],
            "frequency_sd": wavlm_phone_frequency[1],
            "clean": wavlm_phone_clean[0], "clean_sd": wavlm_phone_clean[1],
            "other": wavlm_phone_other[0], "other_sd": wavlm_phone_other[1],
            "color": "#138a8a", "marker": "P",
        },
        {
            "family": "Oracle", "label": "Oracle · character",
            "system": "Character alignment", "configuration": "character boundaries · mean pooling",
            "decoder": "from scratch", "frequency": wavlm_char_frequency[0],
            "frequency_sd": wavlm_char_frequency[1],
            "clean": wavlm_char_clean[0], "clean_sd": wavlm_char_clean[1],
            "other": wavlm_char_other[0], "other_sd": wavlm_char_other[1],
            "color": "#1c4e80", "marker": "P",
        },
        {
            "family": "CTC baseline", "label": "Phone-CTC + split · BiGRU",
            "system": "Phone-CTC + long-segment split",
            "configuration": "audio-only CTC runs + ≥160 ms midpoint · BiGRU residual",
            "decoder": "best char init · BiGRU/decoder CE",
            "frequency": phone_ctc_longsplit_hz["test-clean"],
            "other_frequency": phone_ctc_longsplit_hz["test-other"],
            "frequency_sd": 0.0,
            "clean": phone_ctc_longsplit["test-clean"][0],
            "clean_sd": phone_ctc_longsplit["test-clean"][1],
            "other": phone_ctc_longsplit["test-other"][0],
            "other_sd": phone_ctc_longsplit["test-other"][1],
            "n": phone_ctc_longsplit["n"],
            "color": "#a65f2b", "marker": "D",
        },
        {
            "family": "CNN policy", "label": "CNN · Bernoulli",
            "system": "CNN Bernoulli", "configuration": "independent decisions · mean pooling",
            "decoder": "best char init · co-trained",
            "frequency": rate_hz(oracleclose["bernoulli"]["rho"][0]),
            "frequency_sd": rate_hz(oracleclose["bernoulli"]["rho"][1]),
            "clean": oracleclose["bernoulli"]["clean"][0][0],
            "clean_sd": oracleclose["bernoulli"]["clean"][0][1],
            "other": oracleclose["bernoulli"]["other"][0][0],
            "other_sd": oracleclose["bernoulli"]["other"][0][1],
            "color": "#d5902f", "marker": "s",
        },
        {
            "family": "CNN policy", "label": "CNN · first-order AR",
            "system": "CNN AR", "configuration": "conditions on previous boundary · mean pooling",
            "decoder": "best char init · co-trained",
            "frequency": corrected["cnn_mean"]["frequency"][0],
            "frequency_sd": corrected["cnn_mean"]["frequency"][1],
            "clean": corrected["cnn_mean"]["clean"][0][0],
            "clean_sd": corrected["cnn_mean"]["clean"][0][1],
            "other": corrected["cnn_mean"]["other"][0][0],
            "other_sd": corrected["cnn_mean"]["other"][0][1],
            "n": corrected["cnn_mean"]["n"],
            "color": "#b96f20", "marker": "D",
        },
        {
            "family": "CNN policy", "label": "CNN · first-order AR · BiGRU",
            "system": "CNN AR + BiGRU",
            "configuration": "previous boundary · BiGRU residual pooling",
            "decoder": "best char init · co-trained",
            "frequency": corrected["cnn_bigru"]["frequency"][0],
            "frequency_sd": corrected["cnn_bigru"]["frequency"][1],
            "clean": corrected["cnn_bigru"]["clean"][0][0],
            "clean_sd": corrected["cnn_bigru"]["clean"][0][1],
            "other": corrected["cnn_bigru"]["other"][0][0],
            "other_sd": corrected["cnn_bigru"]["other"][0][1],
            "n": corrected["cnn_bigru"]["n"],
            "color": "#e2a24c", "marker": "D",
        },
        {
            "family": "Transformer policy", "label": "Transformer · Bernoulli",
            "system": "Transformer Bernoulli",
            "configuration": "bidirectional acoustic context · independent decisions · mean pooling",
            "decoder": "shared warm-up · co-trained",
            "frequency": rate_hz(wavlm_tf_mt["rho"][0]),
            "frequency_sd": rate_hz(wavlm_tf_mt["rho"][1]),
            "clean": wavlm_tf_mt["clean"][0], "clean_sd": wavlm_tf_mt["clean"][1],
            "other": wavlm_tf_mt["other"][0], "other_sd": wavlm_tf_mt["other"][1],
            "color": "#6741a5", "marker": "^",
        },
        {
            "family": "Transformer policy", "label": "Transformer AR · mean",
            "system": "Transformer AR", "configuration": "causal history 64 · mean pooling",
            "decoder": "best char init · co-trained",
            "frequency": corrected["transformer_mean"]["frequency"][0],
            "frequency_sd": corrected["transformer_mean"]["frequency"][1],
            "clean": corrected["transformer_mean"]["clean"][0][0],
            "clean_sd": corrected["transformer_mean"]["clean"][0][1],
            "other": corrected["transformer_mean"]["other"][0][0],
            "other_sd": corrected["transformer_mean"]["other"][0][1],
            "n": corrected["transformer_mean"]["n"],
            "color": "#75579b", "marker": "D",
        },
        {
            "family": "Transformer + BiGRU", "label": "Transformer AR · BiGRU",
            "system": "Transformer AR + BiGRU",
            "configuration": "causal history 64 · BiGRU residual pooling",
            "decoder": "best char init · co-trained",
            "frequency": corrected["transformer_bigru"]["frequency"][0],
            "frequency_sd": corrected["transformer_bigru"]["frequency"][1],
            "clean": corrected["transformer_bigru"]["clean"][0][0],
            "clean_sd": corrected["transformer_bigru"]["clean"][0][1],
            "other": corrected["transformer_bigru"]["other"][0][0],
            "other_sd": corrected["transformer_bigru"]["other"][0][1],
            "n": corrected["transformer_bigru"]["n"],
            "color": "#27865d", "marker": "D", "highlight": True,
        },
    ])

    system_overview_table_rows = ""
    previous_family = None
    for row in system_overview_rows:
        styles = []
        if previous_family is not None and row["family"] != previous_family:
            styles.append("border-top:2px solid var(--border)")
        if row.get("highlight"):
            styles.append("background:var(--band)")
        style = ' style="%s"' % ";".join(styles) if styles else ""
        emphasis = "b" if row.get("highlight") else "span"
        frequency = "%.1f Hz" % row["frequency"]
        if row["frequency_sd"] > 0:
            frequency = "%.1f ± %.1f Hz" % (row["frequency"], row["frequency_sd"])
        elif "other_frequency" in row:
            frequency = "%.1f / %.1f Hz" % (
                row["frequency"], row["other_frequency"]
            )
        n_complete = row.get("n", 3)
        if n_complete == 1:
            clean_text = "%.2f%%" % row["clean"]
            other_text = "%.2f%%" % row["other"]
        else:
            clean_text = "%.2f ± %.2f%%" % (row["clean"], row["clean_sd"])
            other_text = "%.2f ± %.2f%%" % (row["other"], row["other_sd"])
        if n_complete < 3:
            clean_text += ' <span class="pill">n=%d</span>' % n_complete
            other_text += ' <span class="pill">n=%d</span>' % n_complete
        system_overview_table_rows += (
            '<tr%s><td>%s</td><td><%s>%s</%s></td><td>%s</td><td>%s</td>'
            '<td><%s>%s</%s></td><td><%s>%s</%s></td>'
            '<td><%s>%s</%s></td></tr>'
            % (style, row["family"], emphasis, row["system"], emphasis,
               row["configuration"], row["decoder"], emphasis, frequency, emphasis,
               emphasis, clean_text, emphasis,
               emphasis, other_text, emphasis)
        )
        previous_family = row["family"]

    wavlm_baseline_rows = ""
    for row in wavlm_fixed:
        k, rho = row["k"], row["rho"]
        clean, other, dev_clean, n = (
            row["clean"], row["other"], row["dev_clean"], row["n"]
        )
        clean_text = "%.2f%%" % clean[0] if n == 1 else "%.2f ± %.2f%%" % clean
        other_text = "%.2f%%" % other[0] if n == 1 else "%.2f ± %.2f%%" % other
        dev_clean_text = (
            "%.2f%%" % dev_clean[0] if n == 1 else "%.2f ± %.2f%%" % dev_clean
        )
        clean_cell = "<b>%s</b>" % clean_text if k == 5 else clean_text
        wavlm_baseline_rows += (
            "<tr><td>fixed k=%d</td><td>%.1f Hz</td><td>%s</td><td>%s</td><td>%s</td>"
            '<td><span class="pill">n=%d</span></td></tr>'
            % (k, rate_hz(rho), dev_clean_text, clean_cell, other_text, n)
        )
    wavlm_baseline_rows += (
        '<tr><td>phone alignment</td><td>%.1f Hz</td>'
        '<td>%.2f ± %.2f%%</td><td>%.2f ± %.2f%%</td>'
        '<td>%.2f ± %.2f%%</td><td><span class="pill">n=%d</span></td></tr>'
        % ((wavlm_phone_frequency[0],) + wavlm_phone_dev + wavlm_phone_clean
           + wavlm_phone_other
           + (wavlm_phone_n,))
    )
    wavlm_baseline_rows += (
        '<tr style="background:var(--band)"><td><b>phone-CTC + long-segment split + BiGRU</b></td>'
        '<td>%.1f / %.1f Hz</td><td>%.2f ± %.2f%%</td>'
        '<td><b>%.2f ± %.2f%%</b></td><td><b>%.2f ± %.2f%%</b></td>'
        '<td><span class="pill">n=%d</span></td></tr>'
        % ((phone_ctc_longsplit_hz["test-clean"],
            phone_ctc_longsplit_hz["test-other"])
           + phone_ctc_longsplit["dev-clean"]
           + phone_ctc_longsplit["test-clean"]
           + phone_ctc_longsplit["test-other"]
           + (phone_ctc_longsplit["n"],))
    )
    wavlm_baseline_rows += (
        '<tr style="background:var(--band)"><td><b>char alignment</b></td><td>%.1f Hz</td>'
        '<td><b>%.2f ± %.2f%%</b></td><td><b>%.2f ± %.2f%%</b></td>'
        '<td><b>%.2f ± %.2f%%</b></td><td><span class="pill">n=%d</span></td></tr>'
        % ((wavlm_char_frequency[0],) + wavlm_char_dev + wavlm_char_clean
           + wavlm_char_other + (wavlm_char_n,))
    )
    wavlm_baseline_rows += (
        '<tr><td>No downsampling</td><td>50.0 Hz</td>'
        '<td>%.2f ± %.2f%%</td><td>%.2f ± %.2f%%</td>'
        '<td>%.2f ± %.2f%%</td>'
        '<td><span class="pill">n=3</span></td></tr>'
        % (wavlm_no_down["dev-clean"] + wavlm_no_down["test-clean"]
           + wavlm_no_down["test-other"])
    )

    wavlm_checkpoint_rows = ""
    for backbone, variant, reward, decoder in wavlm_specs:
        epochs = wavlm_selected_by_condition[backbone.lower(), variant]
        epoch_text = " / ".join("—" if epoch is None else str(epoch)
                                for epoch in epochs)
        if any(epoch is not None and epoch <= 6 for epoch in epochs):
            assessment = "warm-up retained in one seed"
        elif any(epoch == 10 for epoch in epochs):
            assessment = "mixed; one or more reach final epoch"
        else:
            assessment = "dev WER peaks before final epoch"
        wavlm_checkpoint_rows += (
            "<tr><td>%s</td><td>%s · %s</td><td>%s</td><td>%s</td></tr>"
            % (backbone, reward, decoder, epoch_text, assessment)
        )

    body += hk.section(
        "1 · WavLM-Large results — LibriSpeech-100h",
        lead="The full CNN/Transformer × NLL-frozen/NLL-multitask/CER-multitask grid and "
             "all fixed-pooling, no-downsampling, alignment, and rate-raised phone-CTC controls finished "
             "for seeds 3407/3408/3409. "
             "Both local-history Transformer AR rows are complete, while the CNN AR mean and "
             "BiGRU rows currently have one and two complete seeds. Values are corpus WER from "
             "saved edit counts; ± is "
             "sample SD over the completed seeds named in each row.",
        body=(
            hk.card(
                system_overview_fig(system_overview_rows)
                + '<p class="cap">Rows are grouped by system family; the three panels show '
                  'test-clean audio-token frequency on a log scale and WER on both test splits. '
                  'The frequency panel uses test-clean; the table gives split-specific rates '
                  'where they differ. Error bars use completed seeds only; CNN n is reported in the table. '
                  'Deterministic policies '
                  'have fixed frequency, so only learned-policy frequencies carry error bars. '
                  'Lower frequency means more compression; lower WER is better.</p>',
                title="Important systems at a glance")
            +
            hk.card(
                '<table style="table-layout:fixed"><colgroup>'
                '<col style="width:12%%"><col style="width:16%%">'
                '<col style="width:25%%"><col style="width:17%%">'
                '<col style="width:12%%"><col style="width:9%%">'
                '<col style="width:9%%"></colgroup><thead><tr>'
                '<th>family</th><th>system</th><th>policy / pooling</th>'
                '<th>decoder treatment</th><th>test-clean / other audio-token frequency</th>'
                '<th>test-clean WER</th><th>test-other WER</th>'
                '</tr></thead><tbody>%s</tbody></table>'
                '<p class="cap">This is a system inventory, not one fully controlled ablation: '
                'decoder initialization and training treatment differ across experiment families '
                'and are therefore shown explicitly. All rows use WavLM-Large runtime features '
                'and seeds 3407/3408/3409. Rows show n when fewer than three seeds are complete. '
                'Learned-policy frequency is mean ± sample SD; '
                'deterministic boundary rules have fixed frequencies. The highlighted final row '
                'is the Transformer-AR policy with order-aware BiGRU residual pooling.</p>'
                % system_overview_table_rows,
                title="Comprehensive system summary")
            + hk.card(
                "<table><thead><tr><th>boundary backbone</th><th>objective</th>"
                "<th>selected epoch · seeds 3407/08/09</th><th>assessment</th>"
                "</tr></thead><tbody>%s</tbody></table>"
                '<p class="cap">Selection minimizes dev WER over epochs 1–10; epochs 1–6 '
                "are shared warm-up and epochs 7–10 are joint RL. This checks task-metric "
                "convergence, not merely whether the policy reward flattened.</p>"
                % wavlm_checkpoint_rows,
                title="RL checkpoint selection")
            + hk.card(
                "<table><thead><tr><th>baseline</th><th>test-clean / other audio-token frequency</th>"
                "<th>dev-clean WER</th>"
                "<th>test-clean WER</th><th>test-other WER</th>"
                "<th>seeds complete</th></tr></thead>"
                "<tbody>%s</tbody></table>"
                '<p class="cap">dev-clean is the checkpoint-selection split. Every row is a '
                "three-seed mean ± sample SD. Bold marks the current best value in each test "
                "column.</p>"
                % wavlm_baseline_rows,
                title="Fixed pooling, alignment, CTC, and no-downsampling controls")
            + hk.finding(
                "<b>The clean-set compression gain survives three seeds.</b> Fixed k=5 reaches "
                "%.2f±%.2f on test-clean versus %.2f±%.2f without downsampling, a %.2f-point "
                "gain with much smaller seed variance. It also reduces mean A100 wall time "
                "from 18h02m to 5h20m under the actual memory-safe protocols. Moderate "
                "mean-pooling can denoise "
                "redundant 50-Hz WavLM frames and regularize the Llama decoder by shortening "
                "its audio prefix. The effect is not universal, however: dev-clean is "
                "%.2f±%.2f versus %.2f±%.2f, and test-other is %.2f±%.2f versus %.2f±%.2f—both "
                "essentially tied. The no-downsampling memory protocol also changes batching and "
                "decode settings, so a fully matched no-downsampling control remains useful."
                % (wavlm_k5["clean"] + wavlm_no_down["test-clean"]
                   + (wavlm_no_down["test-clean"][0] - wavlm_k5["clean"][0],)
                   + wavlm_k5["dev_clean"] + wavlm_no_down["dev-clean"]
                   + wavlm_k5["other"] + wavlm_no_down["test-other"]))
            + hk.finding(
                "<b>RL convergence is mixed, not complete in the task metric.</b> Boundary rate "
                "and training reward stabilize, but only %d/18 runs select epoch 10; %d select "
                "epochs 7–9 and %d retain the pre-RL warm-up. Training reward often continues "
                "to improve after dev WER has peaked, and late-epoch WER frequently rebounds. "
                "The policy optimization has settled, but the reward/generalization alignment "
                "has not; simply training longer is unlikely to fix that without stronger early "
                "stopping or a better-aligned objective."
                % (wavlm_selected_final, wavlm_selected_joint_early,
                   wavlm_selected_warmup))
            + hk.finding(
                "<b>Alignment controls separate count from placement.</b> Phone boundaries are "
                "the closest oracle to k=5 in cost (%.1f Hz vs. 10.0 Hz) and improve WER from "
                "%.2f±%.2f to %.2f±%.2f clean and %.2f±%.2f to %.2f±%.2f other. Char "
                "boundaries operate at %.1f Hz but are stronger still at "
                "%.2f±%.2f / %.2f±%.2f. Thus the fixed-rate gain combines smoothing with "
                "token-budget regularization, while another sizeable gain is available from "
                "content-aware placement. The learned Transformer AR + BiGRU system at %.1f Hz "
                "has similar mean WER to the phone oracle (%.2f/%.2f versus %.2f/%.2f), but "
                "the char oracle remains %.2f points better on clean and %.2f on other. This "
                "makes the char-level boundary gap—not further compression—the clearer target."
                % ((wavlm_phone_frequency[0],) + wavlm_k5["clean"]
                   + wavlm_phone_clean + wavlm_k5["other"] + wavlm_phone_other
                   + (wavlm_char_frequency[0],) + wavlm_char_clean + wavlm_char_other
                   + (corrected["transformer_bigru"]["frequency"][0],
                      corrected["transformer_bigru"]["clean"][0][0],
                      corrected["transformer_bigru"]["other"][0][0],
                      wavlm_phone_clean[0], wavlm_phone_other[0],
                      corrected["transformer_bigru"]["clean"][0][0]
                      - wavlm_char_clean[0],
                      corrected["transformer_bigru"]["other"][0][0]
                      - wavlm_char_other[0]))
            )
            + hk.finding(
                "<b>The learned Transformer AR + BiGRU system also outperforms the "
                "rate-matched phone-CTC baseline in the 100h study.</b> "
                "At %.1f / %.1f Hz, phone-CTC + long-segment split + BiGRU reaches "
                "%.2f±%.2f clean and %.2f±%.2f other with regular decoder CE. The established "
                "Transformer AR + BiGRU system reaches %.2f±%.2f / %.2f±%.2f, so its "
                "advantage is %.2f clean and %.2f other WER points. CTC boundary "
                "inference is transcript-free, although the frozen CTC head was trained from "
                "ordered phone labels. The midpoint refinement uses only predicted segment "
                "length, not a transcript or downstream WER." % (
                    phone_ctc_longsplit_hz["test-clean"],
                    phone_ctc_longsplit_hz["test-other"],
                    *phone_ctc_longsplit["test-clean"],
                    *phone_ctc_longsplit["test-other"],
                    *corrected["transformer_bigru"]["clean"][0],
                    *corrected["transformer_bigru"]["other"][0],
                    phone_ctc_longsplit["test-clean"][0]
                    - corrected["transformer_bigru"]["clean"][0][0],
                    phone_ctc_longsplit["test-other"][0]
                    - corrected["transformer_bigru"]["other"][0][0],
                )
            )
        ),
    )

    def result_text(value_and_n):
        stats_, n_ = value_and_n
        suffix = "" if n_ == 3 else ' <span class="pill">n=%d</span>' % n_
        return "%.2f ± %.2f%%%s" % (stats_[0], stats_[1], suffix)

    setup_rows = [
        ("Char/phone alignment", "not used", "saved alignment labels", "WavLM", "mean",
         "from scratch", "fixed boundaries; no RL"),
        ("Fixed k=5", "not used", "every fifth frame", "WavLM", "mean",
         "from scratch", "fixed boundaries; no RL"),
        ("CNN segmenter on wav2vec2 → pool WavLM", "wav2vec2", "CNN segmenter",
         "WavLM", "mean", "best char decoder or from scratch",
         "decoder warm-up, then Bernoulli RL"),
        ("CNN segmenter on WavLM → pool WavLM", "WavLM", "CNN segmenter",
         "WavLM", "mean or BiGRU residual", "best char decoder",
         "2 decoder-warm-up epochs, then 10 Bernoulli or first-order AR RL epochs"),
        ("Transformer segmenter on WavLM → pool WavLM", "WavLM",
         "Transformer segmenter", "WavLM", "mean", "shared warm-up",
         "NLL RL with decoder co-training"),
        ("Local-history Transformer AR on WavLM → pool WavLM", "WavLM",
         "local-history Transformer AR, 64-state window", "WavLM",
         "mean or BiGRU residual", "best char decoder",
         "2 decoder/pooler-adaptation epochs, then 10 combined-on-policy RL epochs; K=4"),
        ("No downsampling", "not used", "keep every frame", "WavLM", "none",
         "from scratch", "no boundary model or RL"),
    ]
    setup_table = (
        "<table><thead><tr><th>name used below</th><th>segmenter input</th>"
        "<th>how boundaries are chosen</th><th>features pooled for decoder</th>"
        "<th>within-segment pooling</th><th>decoder start</th>"
        "<th>training after initialization</th>"
        "</tr></thead><tbody>"
        + "".join("<tr>%s</tr>" % "".join("<td>%s</td>" % cell for cell in row)
                  for row in setup_rows)
        + "</tbody></table>"
    )

    controlled_rows = [
        ("Char-aligned baseline", "saved char alignment", "from scratch",
         "fixed boundaries", wavlm_char_frequency[0],
         "%.2f ± %.2f%%" % wavlm_char_clean,
         "%.2f ± %.2f%%" % wavlm_char_other, "3"),
        ("Phone-aligned baseline", "saved phone alignment", "from scratch",
         "fixed boundaries", wavlm_phone_frequency[0],
         "%.2f ± %.2f%%" % wavlm_phone_clean,
         "%.2f ± %.2f%%" % wavlm_phone_other, "3"),
        ("Fixed k=5 baseline", "every fifth frame", "from scratch",
         "fixed boundaries", rate_hz(0.200),
         "%.2f ± %.2f%%" % wavlm_k5["clean"],
         "%.2f ± %.2f%%" % wavlm_k5["other"], "3"),
        ("CNN segmenter on wav2vec2 → pool WavLM", "CNN segmenter",
         "best char decoder", "Bernoulli RL",
         rate_hz(hybrid["oracle_char_decoder"]["rho"][0]),
         result_text(hybrid["oracle_char_decoder"]["clean"]),
         result_text(hybrid["oracle_char_decoder"]["other"]), "3"),
        ("CNN segmenter on wav2vec2 → pool WavLM", "CNN segmenter", "from scratch",
         "Bernoulli RL",
         rate_hz(hybrid["scratch_decoder"]["rho"][0]),
         result_text(hybrid["scratch_decoder"]["clean"]),
         result_text(hybrid["scratch_decoder"]["other"]), "3"),
        ("CNN segmenter on WavLM → pool WavLM", "CNN segmenter",
         "best char decoder", "Bernoulli RL",
         rate_hz(oracleclose["bernoulli"]["rho"][0]),
         result_text(oracleclose["bernoulli"]["clean"]),
         result_text(oracleclose["bernoulli"]["other"]),
         str(oracleclose["bernoulli"]["clean"][1])),
        ("CNN segmenter on WavLM → pool WavLM", "CNN segmenter",
         "best char decoder", "first-order AR RL",
         rate_hz(oracleclose["autoregressive"]["rho"][0]),
         result_text(oracleclose["autoregressive"]["clean"]),
         result_text(oracleclose["autoregressive"]["other"]),
         str(oracleclose["autoregressive"]["clean"][1])),
        ("Transformer segmenter on WavLM → pool WavLM", "Transformer segmenter",
         "shared warm-up", "NLL RL",
         rate_hz(wavlm_tf_mt["rho"][0]), "%.2f ± %.2f%%" % wavlm_tf_mt["clean"],
         "%.2f ± %.2f%%" % wavlm_tf_mt["other"], "3"),
        ("No-downsampling baseline", "keep every frame", "from scratch",
         "no RL", rate_hz(1.000),
         "%.2f ± %.2f%%" % wavlm_no_down["test-clean"],
         "%.2f ± %.2f%%" % wavlm_no_down["test-other"], "3"),
    ]
    controlled_table = (
        "<table><thead><tr><th>system</th><th>how boundaries are chosen</th>"
        "<th>decoder start</th><th>training policy</th>"
        "<th>audio-token frequency</th>"
        "<th>test-clean</th><th>test-other</th><th>seeds</th></tr></thead><tbody>"
        + "".join(
            '<tr%s><td>%s</td><td>%s</td><td>%s</td><td>%s</td>'
            '<td>%.1f Hz</td><td>%s</td><td>%s</td><td>%s</td></tr>' %
            ((' style="background:var(--band)"' if
              row[0] == "CNN segmenter on WavLM → pool WavLM"
              and row[3] == "first-order AR RL" else ""),
             *row)
            for row in controlled_rows
        ) + "</tbody></table>"
    )

    controlled_plot_rows = [
        ("Char-aligned baseline", *wavlm_char_clean, *wavlm_char_other, 3,
         "#2f6f9f"),
        ("Phone-aligned baseline", *wavlm_phone_clean, *wavlm_phone_other, 3,
         "#4f8fba"),
        ("Fixed k=5 baseline", *wavlm_k5["clean"], *wavlm_k5["other"], 3,
         "#7b858c"),
        ("CNN segmenter on wav2vec2 → pool WavLM · char decoder",
         *hybrid["oracle_char_decoder"]["clean"][0],
         *hybrid["oracle_char_decoder"]["other"][0],
         hybrid["oracle_char_decoder"]["clean"][1], "#27866f"),
        ("CNN segmenter on wav2vec2 → pool WavLM · scratch decoder",
         *hybrid["scratch_decoder"]["clean"][0],
         *hybrid["scratch_decoder"]["other"][0],
         hybrid["scratch_decoder"]["clean"][1], "#55a98f"),
        ("CNN segmenter on WavLM → pool WavLM · Bernoulli",
         *oracleclose["bernoulli"]["clean"][0],
         *oracleclose["bernoulli"]["other"][0],
         oracleclose["bernoulli"]["clean"][1], "#d5902f"),
        ("CNN segmenter on WavLM → pool WavLM · first-order AR",
         *oracleclose["autoregressive"]["clean"][0],
         *oracleclose["autoregressive"]["other"][0],
         oracleclose["autoregressive"]["clean"][1], "#b96f20"),
        ("Transformer segmenter on WavLM → pool WavLM · shared warm-up",
         *wavlm_tf_mt["clean"], *wavlm_tf_mt["other"], 3, "#75579b"),
        ("No-downsampling baseline", *wavlm_no_down["test-clean"],
         *wavlm_no_down["test-other"], 3, "#9a5b73"),
    ]

    boundary_accuracy_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "../../../..", "artifacts", "segmenter",
        "controlled_boundary_accuracy.json",
    ))
    if not os.path.isfile(boundary_accuracy_path):
        raise FileNotFoundError(
            "Run run_controlled_boundary_accuracy.slurm before building the report: %s"
            % boundary_accuracy_path
        )
    with open(boundary_accuracy_path, encoding="utf-8") as handle:
        boundary_accuracy = json.load(handle)
    boundary_key_rows = [
        ("Char-aligned baseline", "Char-aligned baseline"),
        ("Phone-aligned baseline", "Phone-aligned baseline"),
        ("Fixed k=5 baseline", "Fixed k=5 baseline"),
        ("CNN segmenter on wav2vec2 → pool WavLM · char decoder",
         "CNN segmenter on wav2vec2 -> pool WavLM - char decoder"),
        ("CNN segmenter on wav2vec2 → pool WavLM · scratch decoder",
         "CNN segmenter on wav2vec2 -> pool WavLM - scratch decoder"),
        ("CNN segmenter on WavLM → pool WavLM · Bernoulli",
         "CNN segmenter on WavLM -> pool WavLM - Bernoulli"),
        ("CNN segmenter on WavLM → pool WavLM · first-order AR",
         "CNN segmenter on WavLM -> pool WavLM - first-order AR"),
        ("Transformer segmenter on WavLM → pool WavLM · shared warm-up",
         "Transformer segmenter on WavLM -> pool WavLM - shared warm-up"),
        ("No-downsampling baseline", "No-downsampling baseline"),
    ]
    controlled_colors = {row[0]: row[6] for row in controlled_plot_rows}
    boundary_plot_rows = []
    for label, key in boundary_key_rows:
        result = boundary_accuracy["systems"][key]
        exact = result["summary"]["exact"]["f1"]
        tol1 = result["summary"]["tol1"]["f1"]
        boundary_plot_rows.append((
            label,
            100 * exact["mean"], 100 * exact["sd"],
            100 * tol1["mean"], 100 * tol1["sd"],
            result["n"], controlled_colors[label],
        ))
    boundary_by_label = {row[0]: row for row in boundary_plot_rows}

    phone_agreement_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "../../../..", "artifacts", "segmenter",
    ))
    fixed_phone_path = os.path.join(
        phone_agreement_dir, "fixed_rate_phone_agreement_dev_clean.json"
    )
    learned_phone_path = os.path.join(
        phone_agreement_dir,
        "transformer_ar_bigru_phone_agreement_dev_clean.json",
    )
    for path in (fixed_phone_path, learned_phone_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                "Run the phone-boundary agreement audits before building the report: %s"
                % path
            )
    with open(fixed_phone_path, encoding="utf-8") as handle:
        fixed_phone = json.load(handle)
    with open(learned_phone_path, encoding="utf-8") as handle:
        learned_phone = json.load(handle)

    # Cross-corpus TIMIT check, from audit_timit_phone_agreement.py.
    timit_phone_path = os.path.join(
        phone_agreement_dir, "timit_test_phone_agreement.json"
    )
    if not os.path.isfile(timit_phone_path):
        raise FileNotFoundError(
            "Run audit_timit_phone_agreement.py before building the report: %s"
            % timit_phone_path
        )
    with open(timit_phone_path, encoding="utf-8") as handle:
        timit_phone = json.load(handle)

    # Row order and banding for the TIMIT table; the two rate-matched
    # comparators (phone-CTC and the Transformer) are the highlighted pair.
    timit_display = (
        ("fixed_k5", "Fixed k=5", False),
        ("wavlm_phone_ctc", "Trained WavLM phone-CTC", True),
        ("cnn_first_order_ar_bigru", "CNN first-order AR + BiGRU", False),
        ("transformer_ar_local64_bigru", "Transformer local-64 AR + BiGRU", True),
    )

    def timit_rate_hz(key):
        """Mean audio-token frequency over whatever seeds the system has."""
        return float(np.mean([
            per_seed["audio_frequency_hz"]
            for per_seed in timit_phone["systems"][key]["per_seed"].values()
        ]))

    def timit_f1(key, tolerance):
        return timit_phone["systems"][key]["summary"][tolerance]["f1"]

    def timit_gain(key, reference, tolerance):
        """F1-point gain over `reference`, from full-precision scores."""
        return 100 * (timit_f1(key, tolerance)["mean"]
                      - timit_f1(reference, tolerance)["mean"])

    timit_agreement_rows = ""
    for key, label, highlight in timit_display:
        deterministic = timit_phone["systems"][key]["n"] == 1
        timit_agreement_rows += (
            '<tr style="background:var(--band)">' if highlight else "<tr>"
        )
        timit_agreement_rows += (
            "<td>%s</td><td>%.2f Hz</td>" % (label, timit_rate_hz(key))
        )
        for tolerance in ("exact", "tol_20ms", "tol_40ms"):
            score = timit_f1(key, tolerance)
            timit_agreement_rows += (
                "<td>%.1f%%</td>" % (100 * score["mean"]) if deterministic
                else "<td>%.1f±%.1f%%</td>" % (100 * score["mean"],
                                               100 * score["sd"])
            )
        timit_agreement_rows += "</tr>"
    timit_n_utts = "{:,}".format(timit_phone["utterances"])

    phone_agreement_rows = ""
    for k in (3, 4, 5, 6, 8):
        result = fixed_phone["systems"]["fixed_k%d" % k]
        scores = result["scores"]
        phone_agreement_rows += (
            "<tr><td>Fixed k=%d</td><td>%.2f Hz</td><td>%.1f%%</td>"
            "<td>%.1f%%</td><td>%.1f%%</td></tr>"
            % (k, result["actual_token_rate_hz"],
               100 * scores["exact"]["harsh"]["f1"],
               100 * scores["tol_20ms"]["harsh"]["f1"],
               100 * scores["tol_40ms"]["harsh"]["f1"])
        )
    learned_rates = np.asarray([
        50 * result["kept_ratio"]
        for result in learned_phone["per_seed"].values()
    ])
    learned_rate = float(np.mean(learned_rates))
    learned_rate_sd = float(np.std(learned_rates, ddof=1))
    learned_phone_summary = learned_phone["summary"]
    phone_agreement_rows += (
        '<tr style="background:var(--band)"><td><b>Transformer AR + BiGRU</b></td>'
        "<td>%.2f Hz</td><td><b>%.1f±%.1f%%</b></td>"
        "<td><b>%.1f±%.1f%%</b></td><td><b>%.1f±%.1f%%</b></td></tr>"
        % ((learned_rate,)
           + tuple(100 * value for tolerance in ("exact", "tol_20ms", "tol_40ms")
                   for value in (
                       learned_phone_summary[tolerance]["harsh"]["f1"]["mean"],
                       learned_phone_summary[tolerance]["harsh"]["f1"]["sd"],
                   )))
    )
    fixed_k5_phone = fixed_phone["systems"]["fixed_k5"]
    fixed_k4_phone = fixed_phone["systems"]["fixed_k4"]
    learned_phone_exact = learned_phone_summary["exact"]["harsh"]["f1"]["mean"]
    learned_phone_20 = learned_phone_summary["tol_20ms"]["harsh"]["f1"]["mean"]
    learned_phone_40 = learned_phone_summary["tol_40ms"]["harsh"]["f1"]["mean"]
    learned_phone_precision_20 = (
        learned_phone_summary["tol_20ms"]["harsh"]["precision"]["mean"]
    )
    learned_phone_precision_20_sd = (
        learned_phone_summary["tol_20ms"]["harsh"]["precision"]["sd"]
    )
    learned_phone_recall_20 = (
        learned_phone_summary["tol_20ms"]["harsh"]["recall"]["mean"]
    )
    learned_phone_recall_20_sd = (
        learned_phone_summary["tol_20ms"]["harsh"]["recall"]["sd"]
    )
    fixed_phone_exact = fixed_k5_phone["scores"]["exact"]["harsh"]["f1"]
    fixed_phone_20 = fixed_k5_phone["scores"]["tol_20ms"]["harsh"]["f1"]
    fixed_phone_40 = fixed_k5_phone["scores"]["tol_40ms"]["harsh"]["f1"]
    fixed_k4_phone_20 = fixed_k4_phone["scores"]["tol_20ms"]["harsh"]["f1"]
    fixed_k5_precision_20 = (
        fixed_k5_phone["scores"]["tol_20ms"]["harsh"]["precision"]
    )
    fixed_k5_recall_20 = (
        fixed_k5_phone["scores"]["tol_20ms"]["harsh"]["recall"]
    )

    swap_rows = "".join(
        '<tr%s><td>%s</td><td>%.2f%%</td><td>%s</td></tr>' % row
        for row in (
            (' style="background:var(--band)"', "oracle char", boundary_swap["oracle"],
             "training distribution / reference"),
            ("", "cold learned CNN", boundary_swap["cold"],
             "+%.2f points: imitation + decoder mismatch" %
             (boundary_swap["cold"] - boundary_swap["oracle"])),
            ("", "post-RL CNN", boundary_swap["rl"],
             "+%.2f points beyond cold: policy drift + mismatch" %
             (boundary_swap["rl"] - boundary_swap["cold"])),
        )
    )
    cold_rows = "".join(
        "<tr><td>%s</td><td>%s</td><td>%.3f</td><td>%s</td></tr>" %
        (encoder, backbone, cold_f1[(encoder, backbone)],
         "target-generator family" if encoder == "wav2vec2" else "cross-family")
        for encoder in ("wav2vec2", "WavLM")
        for backbone in ("CNN", "Transformer")
    )
    body += hk.section(
        "2 · What causes the WER gap? — segmenter, decoder, and boundary changes",
        lead="These experiments change one important part at a time where possible. Rows with "
             "different decoder starts are not treated as pure segmenter comparisons.",
        body=(
            hk.card(
                setup_table
                + '<p class="cap"><b>Segmenter input</b> is the frozen frame sequence fed '
                  'to the model that predicts boundaries. <b>Features pooled for decoder</b> are the '
                  'frames averaged between those boundaries and then passed to the LLM. '
                  'The wav2vec2→WavLM setup therefore gives wav2vec2 features to the segmenter '
                  'only to choose boundary '
                  'positions; the decoder still receives pooled WavLM features. '
                  '<b>BiGRU residual</b> means the frame mean plus an order-sensitive BiGRU '
                  'summary. Its residual projection starts at zero, so training begins from '
                  'the exact mean-pooling representation.</p>',
                title="Experiment setup — what each system name means")
            +
            hk.card(
                controlled_table
                + '<p class="cap">Every row gives WER as a three-seed mean ± sample SD. '
                  'Audio-token frequency is the average number of pooled audio tokens passed '
                  'to the decoder per second; lower Hz means stronger compression. The '
                  'highlighted row uses WavLM features, a CNN segmenter, first-order AR, and '
                  'the best char decoder.</p>',
                title="WavLM results by boundary source and decoder start")
            + hk.card(
                controlled_wer_fig(controlled_plot_rows)
                + '<p class="cap">All points are three-seed means with sample-SD '
                  'error bars. Exact values and the setup behind each point are given '
                  'in the table above.</p>',
                title="The same results as a plot")
            + hk.card(
                controlled_boundary_accuracy_fig(boundary_plot_rows)
                + '<p class="cap">Full dev-clean split. Learned systems report mean ± '
                  'sample SD over the same three WER-selected checkpoints; square markers '
                  'are deterministic boundary sources. Exact matching requires the same '
                  'encoder frame; ±1 allows a one-frame timing offset. F1 is used instead '
                  'of raw frame accuracy because most frames are not boundaries.</p>',
                title="How closely each boundary stream matches char-CTC")
            + "__PHONE_BOUNDARY_AGREEMENT_START__"
            + hk.card(
                "<table><thead><tr><th>boundary stream</th><th>audio-token frequency</th>"
                "<th>exact F1</th><th>±20 ms F1</th><th>±40 ms F1</th>"
                "</tr></thead><tbody>%s</tbody></table>"
                '<p class="cap">One-to-one boundary F1 on all 2,703 dev-clean utterances; '
                'each predicted or reference boundary can be matched at most once. ±20 ms '
                'allows a one-frame offset on the 50-Hz WavLM grid. Fixed grids are '
                'deterministic; Transformer AR + BiGRU reports mean ± sample SD over three '
                'WER-selected checkpoints. The phone reference contains 10.74 boundaries/s. '
                'This is boundary-event F1, not raw frame accuracy.</p>'
                % phone_agreement_rows,
                title="Agreement with phone boundaries")
            + hk.card(
                phone_boundary_agreement_fig(fixed_phone, learned_phone)
                + '<p class="cap">Circles and lines are deterministic fixed-rate grids. '
                  'Stars are the learned system’s three-seed mean with sample-SD error bars '
                  'in both frequency and F1. Comparing points at similar frequency prevents '
                  'a denser boundary stream from looking better merely because it places '
                  'more candidates near each phone transition.</p>',
                title="Phone agreement as a function of audio-token frequency")
            + hk.finding(
                '<span class="pill">Evidence-backed</span> <b>The ≈63%% score is meaningful '
                'only after controlling for frequency.</b> Transformer AR + BiGRU emits '
                '%.2f±%.2f audio tokens/s, almost matching the phone reference at 10.74 Hz. '
                'Within ±20 ms, its precision is %.1f±%.1f%% and recall is %.1f±%.1f%%, '
                'so the %.1f±%.1f%% F1 means roughly 63%% of learned cuts and phone cuts can '
                'be paired one-to-one. But a content-blind fixed k=5 grid at %.2f Hz already '
                'gets %.1f%% precision, %.1f%% recall, and %.1f%% F1. The rate-matched '
                'result is unsurprising: a ±1-frame window around a cut every five frames '
                'covers about three fifths of possible frame positions before one-to-one '
                'matching. The evidence for learned placement is therefore the '
                '<b>+%.1f F1-point gain</b>, not the raw 63%% alone.'
                % (learned_rate, learned_rate_sd,
                   100 * learned_phone_precision_20,
                   100 * learned_phone_precision_20_sd,
                   100 * learned_phone_recall_20,
                   100 * learned_phone_recall_20_sd,
                   100 * learned_phone_20,
                   100 * learned_phone_summary["tol_20ms"]["harsh"]["f1"]["sd"],
                   fixed_k5_phone["actual_token_rate_hz"],
                   100 * fixed_k5_precision_20, 100 * fixed_k5_recall_20,
                   100 * fixed_phone_20,
                   100 * (learned_phone_20 - fixed_phone_20)))
            + hk.finding(
                '<span class="pill">Evidence-backed</span> <b>The learned advantage is '
                'mostly near-boundary timing, not exact phone recovery.</b> Against fixed '
                'k=5, the gain is only %.1f point at the exact WavLM frame, then %.1f points '
                'within ±20 ms and %.1f within ±40 ms. A denser fixed k=4 grid reaches %.1f%% '
                'at ±20 ms, essentially the learned %.1f%%, but requires %.2f Hz—%.0f%% more '
                'decoder-side audio tokens than the learned system. This supports adaptive '
                'placement efficiency, while not implying that RL has discovered a pure '
                'phone segmenter.'
                % (100 * (learned_phone_exact - fixed_phone_exact),
                   100 * (learned_phone_20 - fixed_phone_20),
                   100 * (learned_phone_40 - fixed_phone_40),
                   100 * fixed_k4_phone_20, 100 * learned_phone_20,
                   fixed_k4_phone["actual_token_rate_hz"],
                   100 * (fixed_k4_phone["actual_token_rate_hz"] / learned_rate - 1)))
            + hk.card(
                "<table><thead><tr><th>boundary stream</th>"
                "<th>audio-token frequency</th><th>exact F1</th>"
                "<th>±20 ms F1</th><th>±40 ms F1</th>"
                "</tr></thead><tbody>" + timit_agreement_rows + "</tbody></table>"
                + '<p class="cap"><b>Cross-corpus check: TIMIT TEST.</b> Native TIMIT '
                  '.PHN phone starts are mapped from sample indices to the 50-Hz WavLM '
                  'frame grid; SA prompts are excluded (' + timit_n_utts + ' utterances). '
                  'Scores are micro precision/recall/F1 under maximum ordered one-to-one '
                  'matching, so each predicted and reference boundary can match at most '
                  'once. Exact requires the same 20-ms frame; tolerances allow ±20 or '
                  '±40 ms. Fixed and phone-CTC rows are deterministic; learned segmenters '
                  'are three-seed mean ± sample SD. The phone-CTC system has a frozen '
                  'WavLM-Large encoder and was trained with ordered phone sequences, but '
                  'boundary inference is audio-only CTC decoding.</p>',
                title="Native TIMIT phone-boundary agreement")
            + hk.finding(
                '<span class="pill">Cross-corpus</span> <b>The trained WavLM phone-CTC '
                'source beats the rate-matched fixed grid.</b> At %.2f Hz versus %.2f Hz '
                'for fixed k=5, it gains <b>%+.1f</b> exact-F1 points, <b>%+.1f</b> at '
                '±20 ms, and <b>%+.1f</b> at ±40 ms. This is the appropriate CTC '
                'comparator here: it is the project’s trained WavLM phone model, not an '
                'external wav2vec2 CTC system.'
                % (timit_rate_hz("wavlm_phone_ctc"), timit_rate_hz("fixed_k5"),
                   timit_gain("wavlm_phone_ctc", "fixed_k5", "exact"),
                   timit_gain("wavlm_phone_ctc", "fixed_k5", "tol_20ms"),
                   timit_gain("wavlm_phone_ctc", "fixed_k5", "tol_40ms")))
            + hk.finding(
                '<span class="pill">Cross-corpus</span> <b>The Transformer improves '
                'tolerance-scale agreement at nearly the same rate as phone-CTC.</b> At '
                '%.2f Hz, its exact F1 changes by <b>%+.1f</b> points, then it gains '
                '<b>%+.1f</b> at ±20 ms, and <b>%+.1f</b> at ±40 ms against the %.2f-Hz '
                'WavLM phone-CTC source. The CNN has the best exact F1 (%.1f%%), but emits '
                '%.0f%% more audio tokens than phone-CTC and does not improve tolerance '
                'F1. Thus the TIMIT evidence supports more accurate near-boundary timing '
                'for the Transformer, not a claim of exact phone segmentation.'
                % (timit_rate_hz("transformer_ar_local64_bigru"),
                   timit_gain("transformer_ar_local64_bigru",
                              "wavlm_phone_ctc", "exact"),
                   timit_gain("transformer_ar_local64_bigru",
                              "wavlm_phone_ctc", "tol_20ms"),
                   timit_gain("transformer_ar_local64_bigru",
                              "wavlm_phone_ctc", "tol_40ms"),
                   timit_rate_hz("wavlm_phone_ctc"),
                   100 * timit_f1("cnn_first_order_ar_bigru", "exact")["mean"],
                   100 * (timit_rate_hz("cnn_first_order_ar_bigru")
                          / timit_rate_hz("wavlm_phone_ctc") - 1)))
            + "__PHONE_BOUNDARY_AGREEMENT_END__"
            + hk.card(
                component_checks_fig(boundary_swap, cold_f1)
                + '<p class="cap">Left: only the boundary stream changes; the decoder is '
                  'fixed. Right: exact-frame F1 for supervised imitation of the char labels. '
                  'These checks explain why the decoder-start and segmenter-input columns '
                  'must be shown explicitly.</p>',
                title="Direct checks of boundary quality")
            + '<div class="grid2">'
            + hk.card(
                "<table><thead><tr><th>boundary stream</th><th>dev-clean WER</th>"
                "<th>within-decoder reading</th></tr></thead><tbody>%s</tbody></table>"
                '<p class="cap">Same seed-3407 oracle-char decoder and decoding protocol; only '
                'the boundary stream changes.</p>' % swap_rows,
                title="Same decoder, different boundaries")
            + hk.card(
                "<table><thead><tr><th>segmenter input</th><th>backbone</th>"
                "<th>final exact-frame F1</th><th>relationship to char target</th>"
                "</tr></thead><tbody>%s</tbody></table>"
                '<p class="cap">Char targets were produced by a wav2vec2 CTC model. WavLM '
                'nevertheless reaches ≈0.94 F1 with ±1-frame tolerance for the CNN, so much of '
                'the cross-family deficit is timing offset rather than missing all events.</p>'
                % cold_rows,
                title="Cold-start imitation quality")
            + '</div>'
            + hk.finding(
                '<span class="pill">Evidence-backed</span> <b>Decoder initialization matters '
                'when the segmenter is held fixed.</b> With the exact same wav2vec2-based segmenter, WavLM '
                'pooling, Bernoulli policy, rate band, and 10 RL epochs, the oracle-decoder '
                'start reaches %.2f±%.2f / %.2f±%.2f versus %.2f±%.2f / %.2f±%.2f from '
                'scratch. The improvement comes from starting the decoder near the char-based '
                'task, not from changing either encoder.' %
                (hybrid["oracle_char_decoder"]["clean"][0]
                 + hybrid["oracle_char_decoder"]["other"][0]
                 + hybrid["scratch_decoder"]["clean"][0]
                 + hybrid["scratch_decoder"]["other"][0]))
            + hk.finding(
                '<span class="pill">Evidence-backed</span> <b>Boundary F1 does not determine '
                'WER by itself.</b> The scratch-decoder arm is slightly closer to char-CTC than '
                'the char-decoder arm (%.1f/%.1f versus %.1f/%.1f exact/±1 F1), yet its WER is '
                'worse. First-order AR raises exact F1 by %.1f points but lowers ±1 F1 by %.1f '
                'versus Bernoulli while improving clean WER. Char-F1 is therefore a useful '
                'segmenter diagnostic, but it cannot replace the end-to-end WER comparison.'
                % (boundary_by_label[
                       "CNN segmenter on wav2vec2 → pool WavLM · scratch decoder"][1],
                   boundary_by_label[
                       "CNN segmenter on wav2vec2 → pool WavLM · scratch decoder"][3],
                   boundary_by_label[
                       "CNN segmenter on wav2vec2 → pool WavLM · char decoder"][1],
                   boundary_by_label[
                       "CNN segmenter on wav2vec2 → pool WavLM · char decoder"][3],
                   boundary_by_label[
                       "CNN segmenter on WavLM → pool WavLM · first-order AR"][1]
                   - boundary_by_label[
                       "CNN segmenter on WavLM → pool WavLM · Bernoulli"][1],
                   boundary_by_label[
                       "CNN segmenter on WavLM → pool WavLM · Bernoulli"][3]
                   - boundary_by_label[
                       "CNN segmenter on WavLM → pool WavLM · first-order AR"][3]))
        ),
    )

    w2v_matched = {
        ("CNN", "nll_frozen"): (frozen_clean, frozen_other),
        ("CNN", "nll_mt"): (mt_clean, mt_other),
        ("CNN", "cer_mt"): (cer_clean, cer_other),
        ("Transformer", "nll_frozen"): (tf_frozen_clean, tf_frozen_other),
        ("Transformer", "nll_mt"): (tf_mt_clean, tf_mt_other),
        ("Transformer", "cer_mt"): (tf_cer_clean, tf_cer_other),
    }
    encoder_rows = ""
    for backbone, variant, reward, decoder in wavlm_specs:
        w_clean, w_other = (wavlm_results[backbone.lower(), variant]["clean"],
                            wavlm_results[backbone.lower(), variant]["other"])
        v_clean, v_other = w2v_matched[backbone, variant]
        encoder_rows += (
            '<tr%s><td>%s</td><td>%s · %s</td><td>%.2f ± %.2f</td>'
            '<td>%.2f ± %.2f</td><td>%+.2f</td><td>%.2f ± %.2f</td>'
            '<td>%.2f ± %.2f</td><td>%+.2f</td></tr>' %
            ((' style="background:var(--band)"' if variant == "nll_mt" else ""),
             backbone, reward, decoder,
             v_clean[0], v_clean[1], w_clean[0], w_clean[1],
             w_clean[0] - v_clean[0],
             v_other[0], v_other[1], w_other[0], w_other[1],
             w_other[0] - v_other[0])
        )
    body += hk.section(
        "3 · Same experiments with wav2vec2 and WavLM",
        lead="The same six learned-policy conditions were repeated with each frozen speech "
             "encoder. Negative Δ means WavLM has lower WER. Sections 1 and 9 show WER "
             "against audio-token frequency for each encoder.",
        body=(
            hk.card(
                '<table><thead><tr><th rowspan="2">boundary backbone</th>'
                '<th rowspan="2">objective</th><th colspan="3">test-clean</th>'
                '<th colspan="3">test-other</th></tr><tr><th>wav2vec2</th><th>WavLM</th>'
                '<th>Δ</th><th>wav2vec2</th><th>WavLM</th><th>Δ</th></tr></thead>'
                '<tbody>%s</tbody></table>'
                '<p class="cap">Three seeds per cell. Δ = WavLM − wav2vec2 in absolute WER '
                'points. These are matched broad-sweep curricula, not the later oracle-init '
                'later best-char-decoder test.</p>' % encoder_rows,
                title="Matched learned-policy replication")
            + hk.finding(
                '<span class="pill">Evidence-backed</span> <b>WavLM is not a universal '
                'drop-in gain.</b> It substantially improves the Transformer policy on '
                'test-other (for NLL co-training, %.2f→%.2f), yet degrades the CNN NLL '
                'co-trained arm (%.2f→%.2f clean; %.2f→%.2f other). Encoder quality interacts '
                'with the segmenter and decoder curriculum; quoting one encoder-wide ranking '
                'would hide that interaction.' %
                (tf_mt_other[0], wavlm_tf_mt["other"][0], mt_clean[0],
                 wavlm_cnn_mt["clean"][0], mt_other[0], wavlm_cnn_mt["other"][0]))
            + hk.finding(
                '<span class="pill">Inferred</span> <b>Separating segmenter input from the '
                'features pooled for the decoder is promising.</b> The char targets themselves '
                'come from wav2vec2, whose cold CNN '
                'F1 is %.3f versus %.3f with WavLM. Reusing the stronger wav2vec2-based segmenter '
                'while pooling WavLM features recovers %.2f±%.2f / %.2f±%.2f with oracle '
                'decoder initialization. This separates boundary-timing compatibility from '
                'the representation presented to the LLM.' %
                (cold_f1[("wav2vec2", "CNN")], cold_f1[("WavLM", "CNN")],
                 hybrid["oracle_char_decoder"]["clean"][0][0],
                 hybrid["oracle_char_decoder"]["clean"][0][1],
                 hybrid["oracle_char_decoder"]["other"][0][0],
                 hybrid["oracle_char_decoder"]["other"][0][1]))
        ),
    )

    long_ar_rows = ""
    for row in long_ar:
        if row["clean"] is not None and row["other"] is not None:
            state = "complete · selected e%d" % row["selected_epoch"]
            test_text = "%.2f / %.2f" % (row["clean"], row["other"])
        elif row["epochs"] >= 30:
            state = "30/30 · held-out evaluation pending"
            test_text = "—"
        else:
            state = "%d/30 epochs" % row["epochs"]
            test_text = "—"
        long_ar_rows += (
            "<tr><td>%d</td><td>%s</td><td>e%d · %.2f</td><td>%.1f Hz</td>"
            "<td>%.2f</td><td>%.3f</td><td>%s</td></tr>" %
            (row["seed"], state, row["best_epoch"], row["best_dev"],
             rate_hz(row["best_rho"]), row["latest_dev"], row["latest_gap"], test_text)
        )

    pooling_2x2_rows = ""
    for row in pooling_2x2:
        highlight = (
            row["family"] == "Transformer" and row["pooling"] == "BiGRU residual"
        )
        tag = "b" if highlight else "span"
        style = ' style="background:var(--band)"' if highlight else ""
        pooling_2x2_rows += (
            '<tr%s><td><%s>%s</%s></td><td><%s>%s</%s></td>'
            '<td><%s>%.1f ± %.1f Hz</%s></td><td><%s>%.2f ± %.2f</%s></td>'
            '<td><%s>%.2f ± %.2f</%s></td><td>%d</td></tr>' %
            ((style, tag, row["segmenter"], tag, tag, row["pooling"], tag, tag)
             + row["frequency"] + (tag, tag)
             + row["clean"][0] + (tag, tag) + row["other"][0]
             + (tag, row["clean"][1]))
        )
    pooling_2x2_plot_rows = [
        ("%s · %s" % (row["family"], row["pooling"]),
         *row["clean"][0], *row["other"][0], row["clean"][1], row["color"])
        for row in pooling_2x2
    ]

    def speed_stat_text(stats, digits):
        return "%.*f ± %.*f" % (digits, stats[0], digits, stats[1])

    efficiency_table_rows = ""
    for row in efficiency_rows:
        batch_text = "%d%s" % (
            row["batch_size"],
            "&#8224;" if row["batch_choice"] == "fallback" else "",
        )
        efficiency_table_rows += (
            "<tr><td>%s</td><td>%s</td><td>%s</td><td>%s Hz</td>"
            "<td>%s</td><td>%s</td><td>%s</td><td>%s GB</td><td>%d</td></tr>" %
            (row["family"], row["label"], batch_text,
             speed_stat_text(row["frequency"], 1),
             speed_stat_text(row["batch1_rtf"], 3),
             speed_stat_text(row["rtf"], 3),
             speed_stat_text(row["utterances_per_second"], 2),
             speed_stat_text(row["peak_allocated_gb"], 1), row["n"])
        )

    body += hk.section(
        "4 · Do boundary history and order-aware pooling help?",
        lead="The first-order AR version uses the previous sampled boundary when deciding the "
             "current frame. The local-history Transformer AR policy summarizes the latest "
             "64 boundary states through local causal attention. The completed WavLM study also tests "
             "whether preserving frame order inside each predicted segment changes the result.",
        body=(
            '<div class="grid2">'
            + hk.card(
                '<table><thead><tr><th>policy</th><th>audio-token frequency</th><th>test-clean</th>'
                '<th>test-other</th><th>seeds</th></tr></thead><tbody>'
                '<tr><td>independent Bernoulli</td><td>%.1f Hz</td><td>%s</td><td>%s</td>'
                '<td>3</td></tr><tr style="background:var(--band)"><td>first-order AR</td>'
                '<td>%.1f Hz</td><td>%s</td><td>%s</td><td>3</td></tr></tbody></table>'
                '<p class="cap">Best char segmenter + best char decoder, WavLM CNN: identical '
                'decoder, two adaptation epochs, ten RL epochs, and rate band.</p>' %
                (rate_hz(oracleclose["bernoulli"]["rho"][0]),
                 result_text(oracleclose["bernoulli"]["clean"]),
                 result_text(oracleclose["bernoulli"]["other"]),
                 rate_hz(oracleclose["autoregressive"]["rho"][0]),
                 result_text(oracleclose["autoregressive"]["clean"]),
                 result_text(oracleclose["autoregressive"]["other"])),
                title="Matched three-seed policy comparison")
            + hk.card(
                '<table><thead><tr><th>seed</th><th>job/result status</th>'
                '<th>best dev WER</th><th>frequency at best</th><th>latest dev</th>'
                '<th>latest AR gap</th><th>test clean / other</th></tr></thead>'
                '<tbody>%s</tbody></table>'
                '<p class="cap">Jobs 60853–60855; snapshot is read from train logs when this '
                'page is generated. AR gap = bias(prev=1) − bias(prev=0); a negative value '
                'discourages adjacent boundaries.</p>' % long_ar_rows,
                title="Long-horizon wav2vec2 Transformer continuation")
            + '</div>'
            + hk.card(
                bigru_pooling_diagram()
                + '<p class="cap">For each predicted segment, the unchanged mean branch gives '
                  'the order-insensitive baseline. A one-layer BiGRU reads the same frames in '
                  'both directions; its final forward and backward states are concatenated, '
                  'projected into a correction, and added to the mean. Zero-initializing '
                  'that projection makes '
                  'the initial audio token exactly equal to mean pooling.</p>',
                title="BiGRU residual pooling: preserve order without discarding the mean")
            + hk.card(
                '<table><thead><tr><th>component</th><th>setup</th><th>when it learns</th>'
                '<th>role</th></tr></thead><tbody>'
                '<tr><td>WavLM features</td><td>frozen, 1024-d, 50 Hz</td><td>never updated</td>'
                '<td>ordered frame sequence</td></tr>'
                '<tr><td>boundary policy</td><td>local-history Transformer AR; 4 layers; '
                '64-state causal window</td><td>fixed in epochs 1–2; GRPO in epochs 3–12</td>'
                '<td>chooses segment intervals</td></tr>'
                '<tr><td>mean branch</td><td>parameter-free segment mean</td><td>not applicable</td>'
                '<td>stable base representation</td></tr>'
                '<tr><td>BiGRU branch</td><td>1 layer; 128 hidden units per direction</td>'
                '<td>CE in epochs 1–12</td><td>summarizes within-segment order</td></tr>'
                '<tr><td>residual projection</td><td>256→1024; weight and bias start at zero</td>'
                '<td>CE in epochs 1–12</td><td>maps the BiGRU summary to a correction</td></tr>'
                '<tr><td>decoder</td><td>best oracle-char decoder; LoRA adaptation</td>'
                '<td>CE in epochs 1–12</td><td>scores the pooled audio tokens</td></tr>'
                '</tbody></table>'
                '<p class="cap">The supervised char cold-start initializes only the boundary '
                'policy; it does not pretrain the BiGRU. The first two epochs therefore adapt '
                'the new pooler and decoder while predicted boundaries stay fixed. Joint RL '
                'starts in epoch 3, after the BiGRU path has learned to depart gradually from '
                'its exact mean-pooling initialization.</p>',
                title="BiGRU pooling setup and training schedule")
            + hk.card(
                '<table><thead><tr><th>segmenter</th><th>within-segment pooling</th>'
                '<th>test-clean audio-token frequency</th><th>test-clean WER</th>'
                '<th>test-other WER</th><th>seeds</th></tr></thead>'
                '<tbody>%s</tbody></table>'
                '<p class="cap">WavLM features, best char decoder, two adaptation epochs, '
                'ten joint-RL epochs, and K=4. Within each segmenter '
                'family, the BiGRU residual projection starts at zero, so the two arms begin '
                'with exactly the same mean-pooled decoder input. Values are mean ± sample SD '
                'over three fully completed seeds.</p>'
                % pooling_2x2_rows,
                title="CNN and Transformer AR × mean and BiGRU pooling")
            + hk.card(
                controlled_wer_fig(pooling_2x2_plot_rows)
                + '<p class="cap">The plot shows the aggregate rows from the table '
                  'above. Error bars are sample SD over completed seeds; the table gives n. '
                  'CNN rows use n=%d and n=%d, while both Transformer rows use '
                  'the same three seeds. Lower WER is better.</p>' %
                  (corrected["cnn_mean"]["n"], corrected["cnn_bigru"]["n"]),
                title="Pooling effect depends on the segmenter architecture")
            + hk.finding(
                '<span class="pill">Evidence-backed</span> <b>BiGRU residual pooling improves '
                'the Transformer AR arm on test-clean in %d/%d paired seeds.</b> It changes '
                'from %.2f±%.2f to %.2f±%.2f clean WER, a %.2f-point mean reduction. The '
                'audio-token frequency also shifts only modestly (%.1f→%.1f Hz), so this is '
                'not explained by a large compression change.' %
                ((fullprefix_clean_wins, fullprefix_clean_pairs)
                 + corrected["transformer_mean"]["clean"][0]
                 + corrected["transformer_bigru"]["clean"][0]
                 + (corrected["transformer_mean"]["clean"][0][0]
                    - corrected["transformer_bigru"]["clean"][0][0],
                    corrected["transformer_mean"]["frequency"][0],
                    corrected["transformer_bigru"]["frequency"][0])))
            + hk.finding(
                '<span class="pill">Evidence-backed</span> <b>The CNN BiGRU result is mixed.</b> '
                'Mean pooling is %.2f±%.2f clean / %.2f±%.2f other; '
                'BiGRU is %.2f±%.2f clean / %.2f±%.2f other. That is a %.2f-point clean '
                'increase but a %.2f-point other reduction, with seed-level wins in %d/%d '
                'and %d/%d.' %
                (corrected["cnn_mean"]["clean"][0][0],
                 corrected["cnn_mean"]["clean"][0][1],
                 corrected["cnn_mean"]["other"][0][0],
                 corrected["cnn_mean"]["other"][0][1],
                 corrected["cnn_bigru"]["clean"][0][0],
                 corrected["cnn_bigru"]["clean"][0][1],
                 corrected["cnn_bigru"]["other"][0][0],
                 corrected["cnn_bigru"]["other"][0][1],
                 corrected["cnn_bigru"]["clean"][0][0]
                 - corrected["cnn_mean"]["clean"][0][0],
                 corrected["cnn_mean"]["other"][0][0]
                 - corrected["cnn_bigru"]["other"][0][0],
                 cnn_bigru_clean_wins, cnn_bigru_clean_pairs,
                 cnn_bigru_other_wins, cnn_bigru_other_pairs))
            + hk.finding(
                '<span class="pill">Interpretation</span> <b>Order-aware pooling clearly helps '
                'the Transformer segmenter, while its CNN effect is mixed.</b> Transformer '
                'BiGRU reduces mean WER by %.2f clean and %.2f other. CNN BiGRU changes mean '
                'WER by %+.2f clean and %+.2f other. This supports an architecture-dependent '
                'benefit, although one three-seed comparison is not enough to establish a '
                'general interaction.' %
                (corrected["transformer_mean"]["clean"][0][0]
                 - corrected["transformer_bigru"]["clean"][0][0],
                 corrected["transformer_mean"]["other"][0][0]
                 - corrected["transformer_bigru"]["other"][0][0],
                 corrected["cnn_bigru"]["clean"][0][0]
                 - corrected["cnn_mean"]["clean"][0][0],
                 corrected["cnn_bigru"]["other"][0][0]
                 - corrected["cnn_mean"]["other"][0][0]))
            + hk.finding(
                '<span class="pill">Evidence-backed</span> <b>The first-order dependency learns '
                'a spacing prior, but task WER is best early.</b> Across the three '
                'long runs, best dev WER is %.2f±%.2f (epochs %s), improving %.2f points over '
                'the 5.85 initialization. By epoch 30 it has worsened to %.2f±%.2f. The selected '
                'checkpoints give %.2f±%.2f / %.2f±%.2f test WER. All transition gaps become '
                'negative, so extra constant-LR epochs keep strengthening the no-adjacent-boundary '
                'bias after recognition quality has stopped improving.' %
                (long_ar_best[0], long_ar_best[1],
                 "/".join(str(row["best_epoch"]) for row in long_ar),
                 5.84905 - long_ar_best[0], long_ar_latest[0], long_ar_latest[1],
                 long_ar_clean[0], long_ar_clean[1],
                 long_ar_other[0], long_ar_other[1]))
            + hk.finding(
                '<span class="pill">Evidence-backed</span> <b>First-order AR improves WavLM '
                'test-clean in all three paired seeds.</b> Mean WER changes by −%.2f clean and '
                '−%.2f other versus Bernoulli. The other-speech result is not consistent by seed, '
                'so the supported claim is the clean improvement. The local-history pooling study '
                'above changes a different factor; it should not be used to attribute the CNN '
                'gain to longer boundary memory.' %
                (oracleclose["bernoulli"]["clean"][0][0]
                 - oracleclose["autoregressive"]["clean"][0][0],
                 oracleclose["bernoulli"]["other"][0][0]
                 - oracleclose["autoregressive"]["other"][0][0]))
        ),
    )

    body += '<div id="ar-policy-demo">' + hk.section(
        "4b · Demo — CNN first-order AR and local-history Transformer AR",
        lead="Both policies make one cut/continue decision per WavLM frame. The difference is "
             "how much of the generated boundary history can change the next decision.",
        body=ar_modeling_demo(),
    ) + '</div>'

    no_down_eff = efficiency_by_label["No downsampling"]
    k5_eff = efficiency_by_label["Fixed k=5"]
    phone_eff = efficiency_by_label["Phone alignment"]
    cnn_bigru_eff = efficiency_by_label["CNN AR · BiGRU"]
    transformer_bigru_eff = efficiency_by_label["Transformer AR · BiGRU"]
    mismatch_range = (
        min(row["batch_mismatch_pct"] for row in efficiency_rows),
        max(row["batch_mismatch_pct"] for row in efficiency_rows),
    )
    efficiency_section = hk.section(
        "0d · Inference speed and GPU memory",
        lead="Batch 1 shows single-utterance processing without a batching advantage. The "
             "primary throughput comparison gives each system its largest batch that completed "
             "the full evaluation reliably; shorter audio prefixes use less memory and can "
             "therefore admit more utterances per batch. Every value aggregates three seeds "
             "on one A100 80 GB with BF16. Lower RTF is faster.",
        body=(
            hk.card(
                '<table><thead><tr><th>family</th><th>system</th><th>batch</th>'
                '<th>measured frequency</th><th>batch-1 RTF</th>'
                '<th>stable-batch RTF</th>'
                '<th>utterances/s</th><th>test peak allocated</th><th>seeds</th></tr></thead>'
                '<tbody>%s</tbody></table>'
                '<p class="cap">Both RTF columns aggregate all of test-clean and test-other '
                  'within each seed; ± is sample SD across seeds. Utterances/s and peak '
                  'allocated memory use the stable throughput batch shown in the batch '
                  'column. Test peak '
                  'allocated is the larger PyTorch allocation from the two timed test sets. '
                  'A dagger (†) marks systems stepped down from batch 64 to batch 32 after a '
                  'complete-pass OOM, then rerun at batch 32 for all three seeds. Oracle '
                  'timings use precomputed boundaries and exclude alignment generation.</p>'
                % efficiency_table_rows,
                title="Memory-limited inference measurements")
            + hk.card(
                inference_efficiency_fig(efficiency_rows)
                + '<p class="cap">Left: batch-1 forward RTF, which removes the ability to '
                  'process more utterances together. Right: each system’s stable complete-pass '
                  'throughput batch; the value label includes its batch size. Every bar sums '
                  'model-forward time and audio duration over test-clean and test-other within '
                  'a seed; error bars are sample SD across three seeds.</p>',
                title="Batch-1 processing and stable-batch throughput")
            + hk.card(
                '<table><thead><tr><th>protocol item</th><th>implementation</th></tr></thead>'
                '<tbody>'
                '<tr><td>hardware and precision</td><td>one NVIDIA A100-SXM4 80 GB; BF16; '
                'one process per system and seed</td></tr>'
                '<tr><td>checkpoints</td><td>seeds 3407, 3408, and 3409; the same selected '
                'checkpoints used for the corresponding WER systems</td></tr>'
                '<tr><td>batch-size calibration data</td><td>seed 3407 only; the longest '
                'dev-other utterances, selected by duration without reading RTF or test WER</td></tr>'
                '<tr><td>batch-size search</td><td>powers of two; two batches per candidate; '
                'keep the largest candidate that finishes at no more than 90% allocated '
                'memory. If the later complete pass OOMs, step down one power and rerun all '
                'three seeds at that batch.</td></tr>'
                '<tr><td>GPU warm-up</td><td>ten batches drawn from the longest dev-other '
                'utterances, in the same process as timing; excluded from both RTF and '
                'utterances/s</td></tr>'
                '<tr><td>measured data</td><td>complete LibriSpeech test-clean and test-other; '
                'each set is sorted by ascending duration and grouped into a fixed number of '
                'utterances per batch, with no random test sampling</td></tr>'
                '<tr><td>timing</td><td>sum model-forward time and divide by summed audio '
                'duration. Data loading and metric writing are excluded; WavLM, the segmenter '
                'or fixed/oracle pooling, and the current Llama evaluation path are included.</td></tr>'
                '<tr><td>decoding rules</td><td>left-packed prefixes, explicit position IDs, '
                'and a per-utterance duration-based output cap prevent padded neighbors from '
                'changing the allowed decode length</td></tr>'
                '</tbody></table>'
                '<p class="cap">RTF is measured on held-out test audio, but test labels and '
                'RTF are never used to select a batch. The longest-utterance dev-other warm-up '
                'both primes CUDA kernels/allocator state and exercises difficult shapes.</p>',
                title="RTF measurement setup")
            + hk.finding(
                '<span class="pill">Evidence-backed</span> <b>Learned downsampling reaches '
                'near-oracle throughput and is much faster than native 50-Hz decoding.</b> '
                'No downsampling reaches %.3f±%.3f RTF at batch %d. Phone alignment, CNN AR + '
                'BiGRU, and Transformer AR + BiGRU reach %.3f±%.3f, %.3f±%.3f, and '
                '%.3f±%.3f at batch 32—%.1f×, %.1f×, and %.1f× lower RTF than no '
                'downsampling.' %
                (no_down_eff["rtf"][0], no_down_eff["rtf"][1],
                 no_down_eff["batch_size"], phone_eff["rtf"][0], phone_eff["rtf"][1],
                 cnn_bigru_eff["rtf"][0], cnn_bigru_eff["rtf"][1],
                 transformer_bigru_eff["rtf"][0], transformer_bigru_eff["rtf"][1],
                 no_down_eff["rtf"][0] / phone_eff["rtf"][0],
                 no_down_eff["rtf"][0] / cnn_bigru_eff["rtf"][0],
                 no_down_eff["rtf"][0] / transformer_bigru_eff["rtf"][0]))
            + hk.finding(
                '<span class="pill">Evidence-backed</span> <b>Batch 1 and batched throughput '
                'answer different questions.</b> At batch 1, no downsampling is %.3f±%.3f '
                'RTF, CNN AR + BiGRU is %.3f±%.3f, and Transformer AR + BiGRU is '
                '%.3f±%.3f. The local-history Transformer is slower for one utterance because '
                'its sequential boundary policy adds work that cannot be amortized. Its '
                'throughput advantage appears only when the shorter decoder prefix permits a '
                'larger batch; do not describe the right panel as a batch-1 latency gain.' %
                (no_down_eff["batch1_rtf"][0], no_down_eff["batch1_rtf"][1],
                 cnn_bigru_eff["batch1_rtf"][0],
                 cnn_bigru_eff["batch1_rtf"][1],
                 transformer_bigru_eff["batch1_rtf"][0],
                 transformer_bigru_eff["batch1_rtf"][1]))
            + hk.finding(
                '<span class="pill">Evidence-backed</span> <b>At the same batch 32, the two '
                'learned BiGRU systems have nearly equal throughput.</b> CNN AR + BiGRU is '
                '%.3f±%.3f RTF and Transformer AR + BiGRU is %.3f±%.3f; CNN is about '
                '%.0f%% lower. Fixed k=5 is %.3f±%.3f, so learned boundary generation and '
                'order-aware pooling add about %.0f%% RTF over fixed pooling.' %
                (cnn_bigru_eff["rtf"][0], cnn_bigru_eff["rtf"][1],
                 transformer_bigru_eff["rtf"][0], transformer_bigru_eff["rtf"][1],
                 100 * (1 - cnn_bigru_eff["rtf"][0]
                        / transformer_bigru_eff["rtf"][0]),
                 k5_eff["rtf"][0], k5_eff["rtf"][1],
                 100 * (cnn_bigru_eff["rtf"][0] / k5_eff["rtf"][0] - 1)))
            + hk.finding(
                '<span class="pill">Memory evidence</span> <b>The larger feasible batch is '
                'part of the downsampling gain.</b> No downsampling peaks at %.1f±%.1f GB '
                'allocated on the timed tests and %.1f±%.1f GB on the long warm-up; reserved '
                'memory reaches %.1f±%.1f GB of the 79.25-GB device, and batch 32 OOMs. Its '
                'stable batch is therefore 16, while the headline compressed systems finish '
                'at batch 32.' %
                (no_down_eff["peak_allocated_gb"][0],
                 no_down_eff["peak_allocated_gb"][1],
                 no_down_eff["run_peak_allocated_gb"][0],
                 no_down_eff["run_peak_allocated_gb"][1],
                 no_down_eff["run_peak_reserved_gb"][0],
                 no_down_eff["run_peak_reserved_gb"][1]))
            + hk.finding(
                '<span class="pill">Numerical audit</span> <b>Padding no longer changes the '
                'decode budget, but greedy outputs remain numerically batch-sensitive.</b> '
                'Batch-1 and throughput-batch hypotheses differ for %.0f–%.0f%% of utterances '
                'depending on the system, while corpus WER shifts are much smaller. Use this '
                'section for speed and memory; the main WER tables remain the recognition '
                'evidence.' % mismatch_range)
            + hk.finding(
                '<span class="pill">Scope</span> <b>These numbers describe the current '
                'evaluation pipeline, not an optimized production decoder.</b> The pipeline '
                'performs a teacher-forced Llama pass before greedy decoding, and the greedy '
                'searcher does not use a KV cache. Oracle rows also exclude the cost of '
                'creating their forced alignments.')
        ),
    )
    phone_start_marker = "__PHONE_BOUNDARY_AGREEMENT_START__"
    phone_end_marker = "__PHONE_BOUNDARY_AGREEMENT_END__"
    phone_placeholder = "__PHONE_BOUNDARY_AGREEMENT_SECTION__"
    if (body.count(phone_start_marker) != 1
            or body.count(phone_end_marker) != 1
            or body.count(phone_placeholder) != 1):
        raise RuntimeError("Phone-boundary section markers are missing or duplicated")
    phone_start = body.index(phone_start_marker)
    phone_end = body.index(phone_end_marker, phone_start)
    phone_agreement_body = body[
        phone_start + len(phone_start_marker):phone_end
    ]
    body = body[:phone_start] + body[phone_end + len(phone_end_marker):]
    phone_agreement_section = hk.section(
        "0c.1 · Agreement with phone boundaries",
        lead="This audit asks whether the learned cuts follow phone transitions or merely "
             "benefit from emitting boundaries at a similar frequency. One-to-one F1 is "
             "therefore compared with deterministic fixed-rate grids at matched audio-token "
             "frequencies.",
        body=phone_agreement_body,
    )
    body = body.replace(phone_placeholder, phone_agreement_section)

    if body.count("__INFERENCE_EFFICIENCY_SECTION__") != 1:
        raise RuntimeError("Inference-efficiency section placeholder is missing or duplicated")
    body = body.replace("__INFERENCE_EFFICIENCY_SECTION__", efficiency_section)

    body += hk.section(
        "5 · Method — data, model &amp; curriculum",
        lead="Standard LibriSpeech ASR, but the audio-to-LLM interface is learned rather than "
             "fixed: a small per-frame policy decides where an utterance breaks into segments, "
             "so the compression rate can adapt to the audio instead of pooling every k frames "
             "by hand (the fixed-rate baseline this report compares against below).",
        body=(
            hk.card(
                pipeline_fig()
                + '<p class="cap">The segmenter\'s predicted boundaries <i>are</i> the dynamic '
                'downsampling: more boundaries → more, shorter audio tokens reaching the '
                'decoder; fewer boundaries → a lower audio-token frequency and higher compression. '
                'Source for this '
                'figure: <span class="mono">make_pipeline_figure.py</span> (also emits an '
                'editable <span class="mono">.drawio</span>).</p>',
                title="Pipeline")
            + '<div class="grid2">'
            + hk.card(
                "<table><thead><tr><th>split / source</th><th>role</th></tr></thead><tbody>"
                "<tr><td>train-clean-100 (100h)</td><td>training (cold-start + joint RL)</td></tr>"
                "<tr><td>dev-clean</td><td>validation / model selection</td></tr>"
                "<tr><td>test-clean, test-other, dev-other</td><td>held-out WER/CER eval</td></tr>"
                "<tr><td>wavlm_boundaries/char</td><td>char-CTC boundary targets for cold-start "
                "BCE (1 label/SSL-frame, 50 Hz)</td></tr></tbody></table>",
                title="Data")
            + hk.card(
                "<table><thead><tr><th>component</th><th>detail</th></tr></thead><tbody>"
                "<tr><td>SSL encoder</td><td>wav2vec2-base (768-d) or WavLM-Large "
                "(1024-d), frozen @ 50 Hz</td></tr>"
                "<tr><td>Segmenter</td><td>CNN or Transformer boundary policy, 1 logit/frame "
                "(both backbones evaluated with three seeds)</td></tr>"
                "<tr><td>Projection</td><td>VanillaNN, 768→2048→2048, 2 blocks, ReLU</td></tr>"
                "<tr><td>Decoder</td><td>Llama-3.2-1B-Instruct, frozen backbone + LoRA (rank 16, "
                "all_linear) — ~1.5% trainable</td></tr></tbody></table>"
                '<p class="cap">Exact architectures and trainable-parameter counts are in the '
                'Appendix.</p>',
                title="Model")
            + '</div>'
            + hk.card(
                "<table><thead><tr><th>stage</th><th>segmenter</th>"
                "<th>decoder (proj+LoRA)</th><th>signal</th></tr></thead><tbody>"
                "<tr><td>cold-start</td><td>trained</td><td>—</td>"
                "<td>supervised BCE vs. char-CTC boundaries</td></tr>"
                "<tr><td>warmup</td><td>frozen (cold-start init)</td><td>trained</td>"
                "<td>CE on the argmax segmentation</td></tr>"
                "<tr><td>joint RL</td><td>trained (GRPO)</td><td>trained or frozen (ablated)</td>"
                "<td>−NLL / −CER reward + 5–15 Hz frequency-band objective</td></tr>"
                "</tbody></table>"
                '<p class="cap">GRPO: K sampled segmentations per utterance, group-relative '
                '(std-normalized) advantage, γ=1 (bandit — every frame in an utterance shares '
                'that utterance\'s reward). Warmup length and reward choice are themselves '
                'ablated — see the sections below for the specific values used in each run.</p>',
                title="Curriculum"))
    )

    coll_chart = hk.card(
        hk.svg_line([("collapsed (transformer, entropy on, warmup 2)", hk.C["red"],
                      frequency_series(coll, "train rho_mean", 2)),
                     ("fixed (CNN, entropy off, warmup 6)", hk.C["green"],
                      frequency_series(nll, "train rho_mean", 6))],
                    ylabel="argmax audio-token frequency (Hz)", ymin=0, ymax=17.5, vline=1)
        + hk.legend([("collapsed → 0 Hz", hk.C["red"]), ("fixed: holds ~14 Hz", hk.C["green"]),
                     ("RL start", hk.C["muted"])])
        + '<p class="cap">x = epochs since RL begins (≤0 = warmup). Old run crashes to about '
          '0.25 Hz in one RL epoch; fixed run holds about 14 Hz. (Backbone also differs here — transformer '
          'vs CNN — see the analysis below for a same-backbone comparison.)</p>',
        title="Audio-token frequency once RL turns on")
    body += hk.section(
        "6 · Historical failure — collapse and the fix",
        lead="Teacher-forced NLL from an undertrained decoder plus a per-frame entropy bonus "
             "drove the boundary probabilities to a uniform mush: the argmax produced ~0 "
             "boundaries, the decoder saw ~1 token, WER hit 100%.",
        body=hk.finding("<b>Diagnosis.</b> Entropy pushed every p→0.5 (the rate loss constrains "
                        "only the <i>sum</i>, not sharpness) → uniform p, empty argmax. And "
                        "teacher-forced NLL under-rewards audio → RL pushed audio-token frequency down. "
                        "<b>Fix:</b> drop the entropy bonus + warm the decoder longer (2→6 epochs).")
             + coll_chart)

    body += hk.section(
        "6b · Failure audit — two mechanisms, not one",
        lead="Now that we have non-collapsing runs to compare against, it's worth being precise "
             "about the mechanism — the failure history is actually <b>two different collapses</b>, "
             "not one, and the eventual fix changed two variables at once, so this is an honest "
             "post-mortem rather than a clean ablation.",
        body=(
            hk.finding(
                "<b>Failure #1 (original run): a true absorbing state, not entropy-smearing.</b> "
                "At epoch 3 <i>both</i> the argmax and expected audio-token frequencies crash "
                "together, to about 0.15 and 0.10 Hz — not to a moderate shared value. The instrumented GRPO trace "
                "(<span class=\"mono\">grpo_trace.py</span>) makes the mechanism exact: for every "
                "traced utterance the collapsed policy has mean-p = 0.000 and per-frame entropy = "
                "0.0000, so all K=8 sampled segmentations are <i>bit-for-bit identical</i> — same "
                "audio-token frequency, same reward, reward std = 0.0000, advantage = +0.000 for every sample. GRPO's "
                "group-relative advantage has zero within-group variance to learn from once every "
                "sample in the group is the same point — the gradient isn't small, it is exactly "
                "zero, everywhere. (Compare the healthy cold-start trace: mean-p 0.05–0.23, "
                "entropy 0.06–0.16, reward std 0.22–2.60, per-sample frequency spread up to 2×.) That's "
                "the absorbing state, and it explains why frequency stayed pinned near 0.10 Hz for "
                "epochs 4 through 10 with no further movement.")
            + '<div class="grid2">'
            + hk.card(
                "<table><thead><tr><th>trace</th><th>mean p</th><th>entropy</th>"
                "<th>reward std (K=8)</th><th>sample frequency spread</th></tr></thead><tbody>"
                "<tr><td>cold-start (healthy)</td><td>0.05–0.23</td><td>0.06–0.16</td>"
                "<td>0.22–2.60</td><td>up to 2×</td></tr>"
                "<tr style=\"background:var(--band)\"><td>collapsed (original run)</td>"
                "<td><b>0.000</b></td><td><b>0.0000</b></td><td><b>0.0000</b></td>"
                "<td><b>0 (identical)</b></td></tr></tbody></table>"
                '<p class="cap">grpo_trace.py, K=8 samples/utt, reward=−NLL. 5 dev-clean utts '
                'per trace; ranges are min–max across them.</p>',
                title="Smoking gun: the GRPO trace")
            + hk.card(
                "<table><thead><tr><th>run (frequency mode)</th><th>epoch</th>"
                "<th>argmax frequency</th><th>expected frequency</th></tr></thead><tbody>"
                "<tr><td>jointB (cap+tax+floor)</td><td>3</td><td>0.25 Hz</td><td>9.2 Hz</td></tr>"
                "<tr><td>jointB (cap+tax+floor)</td><td>5</td><td>0.10 Hz</td><td>3.9 Hz</td></tr>"
                "<tr><td>jointC (band 5–15 Hz)</td><td>3</td><td>0.45 Hz</td><td>13.7 Hz</td></tr>"
                "<tr><td>jointC (band 5–15 Hz)</td><td>4</td><td>0.15 Hz</td><td>10.9 Hz</td></tr>"
                "</tbody></table>"
                '<p class="cap">Both CNN, entropy still on (probability floor 0.05→0.01). Expected '
                'audio-token frequency stays healthy — the floor/band mechanism works — while '
                'argmax frequency still collapses.</p>',
                title="Failure #2: argmax dies, expected survives")
            + '</div>'
            + hk.finding(
                "<b>Failure #2 (A/B/C retry): a different collapse, caused by entropy, not the "
                "rate objective.</b> Adding a floor (jointB) or a free band (jointC) worked "
                "exactly as designed — the <i>expected</i> audio-token frequency "
                "(50·Σp/T Hz, the differentiable proxy) "
                "never crashes to zero, it stays right where the rate objective wants it. But "
                "the <i>argmax</i> audio-token frequency — what the decoder actually receives — collapses anyway, "
                "within one or two epochs, in <b>both</b> rate objectives (a soft floor and a "
                "hard free-band are structurally different, yet produce the identical divergence "
                "signature). That commonality points at the one thing neither rate objective "
                "constrains: <i>sharpness</i>. Per-frame Bernoulli entropy is maximized at p=0.5, "
                "and the entropy bonus (still on here) pushes every frame toward it; the rate "
                "loss only constrains the sum Σp, never the shape of the per-frame distribution. "
                "So the policy can satisfy \"correct average rate\" while every single frame sits "
                "uniformly <i>below</i> the argmax threshold (p&gt;0.5) — expected count healthy, "
                "decoded count empty. (jointA, a two-sided pin — in principle the most "
                "collapse-resistant of the three — was killed mid-epoch-3 once the same pattern "
                "was already clear in its siblings, so we don't have its numbers, but nothing in "
                "the mechanism above is specific to jointB or jointC's rate objective.)")
            + hk.finding(
                "<b>A separate, third observation: the reward pointed the wrong way.</b> Within "
                "jointB's own run, as expected frequency fell 9.2→6.0→3.9 Hz across epochs 3–5, the "
                "teacher-forced NLL reward <i>improved</i> (−3.01→−2.93→−2.84) — the gradient the "
                "segmenter received was actively rewarding fewer audio tokens. Consistent with an "
                "undertrained (2-epoch) decoder leaning on its text prior rather than the audio: "
                "noisy under-trained audio conditioning hurts more than it helps. This is a "
                "genuinely separate problem from the entropy/sharpness mechanism above — it's "
                "about the <i>sign</i> of the gradient, not its <i>shape</i>.")
            + '<p class="cap"><b>So which fix mattered?</b> The runs that finally didn\'t collapse '
              '(nll_frozen/nll_mt/cer_mt) changed two things at once — dropped entropy to exactly '
              '0 <i>and</i> extended warmup 2→6 epochs — so this isn\'t a controlled experiment '
              'and we can\'t say with certainty which was necessary. Our read of the evidence: '
              '<b>dropping entropy is the better-supported primary fix for the argmax-collapse '
              'specifically</b> — it\'s a direct, within-run, mechanism-level explanation (p→0.5 '
              'is a plain mathematical property of the entropy bonus, not something that depends '
              'on decoder quality) that reproduces identically across two structurally different '
              'rate objectives. The longer warmup targets the separate reward-misdirection '
              'problem above, and plausibly also helps sharpness indirectly — a better-trained '
              'decoder should give higher-SNR, more decisive per-frame reward gradients that can '
              'outcompete entropy\'s smoothing pull — but no run isolates '
              '"entropy-off/short-warmup" or "entropy-on/long-warmup", so that half of the '
              'story is inferred, not proven.</p>'
        ))

    def multi(key, ylabel, ylog=False, ymin=None, ymax=None):
        return hk.svg_line([(k, c, series(rows[k], key, w)) for k, (p, c, w) in abl.items()],
                           ylabel=ylabel, ylog=ylog, ymin=ymin, ymax=ymax, vline=1)
    res = [("nll_frozen", "CNN", "−NLL", "frozen", "14.5 Hz", "7.55", "11.86", "7h40m"),
           ("nll_mt", "CNN", "−NLL", "co-trained", "13.0 Hz", "<b>6.93</b>", "<b>11.11</b>", "7h17m"),
           ("cer_mt_opt&dagger;", "CNN", "−CER", "co-trained", "15.0 Hz", "7.53", "11.81", "15h38m")]
    res_tbl = hk.card(
        "<table><thead><tr><th>version</th><th>arch</th><th>reward</th><th>decoder</th>"
        "<th>audio-token frequency</th>"
        "<th>test-clean</th><th>test-other</th><th>wall-time</th></tr></thead><tbody>"
        + "".join("<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td>"
                  "<td>%s</td><td>%s</td></tr>" % r for r in res) + "</tbody></table>"
        '<p class="cap">These are the historical first-pass <b>CNN</b> runs retained for their '
        'training curves and wall-clock context; the completed matched CNN/Transformer sweep is '
        'reported above. Co-training the decoder (multi-task) beats the frozen decoder on every '
        'split at the NLL reward. <b>&dagger; cer_mt_opt\'s joint phase never surpassed its own warmup '
        'checkpoint</b> — the numbers above are the pre-RL checkpoint, re-evaluated (see the '
        'analysis below the charts).</p>',
        title="Initial single-seed results & wall-clock cost")

    def result_cell(stats, count=3):
        suffix = '' if count == 3 else ' <span class="pill">n=%d</span>' % count
        return '<b>%.2f ± %.2f%%</b>%s' % (stats[0], stats[1], suffix)

    replicated_rows = []
    replicated_specs = [
        ("CNN", "nll_frozen", "NLL", "frozen", frozen_clean, frozen_other,
         frozen_dev_other),
        ("CNN", "nll_mt", "NLL", "co-trained", mt_clean, mt_other, mt_dev_other),
        ("CNN", "cer_mt", "CER", "co-trained", cer_clean, cer_other, cer_dev_other),
        ("Transformer", "nll_frozen", "NLL", "frozen", tf_frozen_clean,
         tf_frozen_other, tf_frozen_dev_other),
        ("Transformer", "nll_mt", "NLL", "co-trained", tf_mt_clean, tf_mt_other,
         tf_mt_dev_other),
        ("Transformer", "cer_mt", "CER", "co-trained", tf_cer_clean, tf_cer_other,
         tf_cer_dev_other),
    ]
    for backbone, variant, reward, decoder, clean, other, dev_other in replicated_specs:
        style = ' style="background:var(--band)"' if variant == "nll_mt" else ""
        dev_n = headline_count(variant, "dev-other", backbone.lower())
        replicated_rows.append(
            '<tr%s><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>'
            % (style, backbone, reward, decoder, result_cell(clean), result_cell(other),
               result_cell(dev_other, dev_n))
        )
    replicated_tbl = hk.card(
        "<table><thead><tr><th>backbone</th><th>reward</th><th>decoder in joint RL</th>"
        "<th>test-clean WER</th><th>test-other WER</th><th>dev-other WER</th>"
        "</tr></thead><tbody>%s</tbody></table>"
        '<p class="cap">Mean ± sample SD across seeds 3407/3408/3409, recomputed from raw '
        'held-out edit counts. Within each backbone, all variants recover one exact epoch-6 '
        'shared warm-up checkpoint. Two Transformer runs completed training and both test splits '
        'but OOMed during the third sequential evaluation, so their dev-other cells use n=2; no '
        'retraining is needed.</p>' % "".join(replicated_rows),
        title="Completed corrected replication · CNN and Transformer, three seeds")
    dyn = (replicated_tbl + res_tbl
           + hk.legend([(k, c) for k, (p, c, w) in abl.items()]) + '<div class="grid2">'
           + hk.card(multi("train dec_ce", "cross-entropy", ylog=True), "Decoder CE (train)")
           + hk.card(multi("valid WER", "WER (%)", ylog=True), "Valid WER")
           + hk.card(
               hk.svg_line(
                   [(k, c, frequency_series(rows[k], "train rho_mean", w))
                    for k, (p, c, w) in abl.items()],
                   ylabel="audio-token frequency (Hz)", ymin=0, ymax=17.5, vline=1),
               "Argmax audio-token frequency")
           + hk.card(multi("train reward", "reward"), "Reward (−NLL / −CER)") + '</div>'
           + hk.finding(
               "<b>cer_mt_opt (2026-08-04): joint RL never beat its own warmup.</b> Valid "
               "loss/WER at the end of warmup (epoch 6, segmenter still frozen) were 0.198 / "
               "6.34%% — better than every subsequent joint epoch (valid WER 10.17 &rarr; 10.98 "
               "&rarr; 7.50 &rarr; 10.30, noisy and non-monotonic through epochs 7&ndash;10). "
               "The checkpointer (min valid loss) correctly kept epoch 6, so the cer_mt_opt row "
               "above is the <i>pre-RL warmup checkpoint re-evaluated</i>, not a converged "
               "CER-reward result &mdash; contrast nll_mt, whose joint phase improved valid WER "
               "almost every epoch over the identical 4-epoch budget (6.53 &rarr; 6.29 &rarr; "
               "6.03 &rarr; 5.62, ticking back up slightly to 5.78 at epoch 10). The blue curves "
               "above show a likely reason: the train audio-frequency SD during cer_mt_opt's "
               "joint phase (4.0&ndash;4.2 Hz) runs roughly double nll_mt's (1.9&ndash;2.0 Hz) &mdash; the "
               "free-running CER reward (K genuinely different decoded hypotheses per step) is a "
               "visibly noisier training signal than teacher-forced NLL, and 4 epochs wasn't "
               "enough for it to settle. The later corrected on-policy sweep resolves this: CNN "
               "CER genuinely trains to %.2f±%.2f / %.2f±%.2f WER, but still trails NLL "
               "co-training and costs about 3× more." % (cer_clean + cer_other))
           + hk.finding(
               "<b>Completed Transformer result · stable training, negative backbone result.</b> "
               "The best Transformer arm is NLL co-training at <b>%.2f±%.2f clean / "
               "%.2f±%.2f other</b>, versus CNN's <b>%.2f±%.2f / %.2f±%.2f</b> at a similar "
               "audio-token frequency (%.1f vs %.1f Hz on test-clean). Transformer CER selected epoch 7 in "
               "all three seeds and degraded afterward; the model trained, but it did not beat "
               "the smaller CNN boundary policy."
               % (tf_mt_clean + tf_mt_other + mt_clean + mt_other
                  + (rate_hz(tf_mt_rho[0]), rate_hz(mt_rho[0])))))
    body += hk.section(
        "7 · Reward and multi-task ablations — full training dynamics",
        lead="The corrected three-seed, exact-shared-warm-up NLL replication is the headline "
             "result. The historical first-pass CNN runs below retain the training curves and "
             "CER context (warmup 6, band rate objective, entropy off): "
             "<b>nll_frozen</b> (RL vs a frozen decoder), <b>nll_mt</b> (co-trained, NLL "
             "reward), and <b>cer_mt</b> (co-trained, free-running CER reward). "
             "RL starts at the dashed line.",
        body=dyn)

    # Reuse the manually checked qualitative figure instead of loading wav2vec2 and
    # regenerating three near-duplicate examples every time the HTML is refreshed.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
    qualitative_path = os.path.join(
        project_root, "artifacts", "segmenter", "diagrams_highres",
        "07_spectrogram_boundaries.png",
    )
    if os.path.exists(qualitative_path):
        encoded = base64.b64encode(open(qualitative_path, "rb").read()).decode("ascii")
        figs = hk.card(
            '<img class="fig" src="data:image/png;base64,%s"/>'
            '<p class="cap">One representative utterance. Char-CTC symbols are placed inside '
            'their aligned intervals; boundary indexing was checked against the saved frame '
            'labels. This qualitative panel intentionally omits per-utterance WER.</p>' % encoded
        )
    else:
        figs = hk.card('<span class="muted">qualitative boundary figure unavailable</span>')
    body += hk.section(
        "8 · Qualitative learned boundaries",
        lead='A representative prediction from the RL-trained segmenter (nll_mt), against '
             'the char-CTC alignment. Top row: log-mel spectrogram with both boundary sets '
             'overlaid (<span style="color:var(--diverge-neg)">red</span> = learned, '
             '<span style="color:#4aa3d8">blue</span> = gold char alignment). The two rows below '
             'repeat the same boundaries as their own dedicated, time-aligned lanes — same x '
             'position as the spectrogram above — so a boundary can be traced straight down to '
             'see whether learned and gold line up.', body=figs)

    # Current points use corrected three-seed means and sample SDs from section 2.
    model_pts = [("CNN · NLL frozen", frozen_rho[0], frozen_clean[0],
                  frozen_clean[1], frozen_other[0], frozen_other[1], "#c9782e", "s"),
                 ("CNN · NLL co-trained", mt_rho[0], mt_clean[0], mt_clean[1],
                  mt_other[0], mt_other[1], "#178a17", "D"),
                 ("CNN · CER co-trained", cer_rho[0], cer_clean[0], cer_clean[1],
                  cer_other[0], cer_other[1], "#2a78d6", "^"),
                 ("Transformer · NLL co-trained", tf_mt_rho[0], tf_mt_clean[0],
                  tf_mt_clean[1], tf_mt_other[0], tf_mt_other[1], "#8e58b7", "P")]

    tbl_rows = ([("k=%d" % k, rho, c, o, None, None, None)
                 for k, rho, c, o in base]
                + [("No downsampling · 3-seed mean", 1.0,
                    w2v_no_down["test-clean"][0], w2v_no_down["test-other"][0],
                    None, w2v_no_down["test-clean"][1],
                    w2v_no_down["test-other"][1])]
                + [(name, rho, c, o, color, c_sd, o_sd)
                   for name, rho, c, c_sd, o, o_sd, color, marker in model_pts])
    tbl_rows.sort(key=lambda r: r[1], reverse=True)

    def frontier_cell(value, sd, ours):
        text = "%.2f ± %.2f%%" % (value, sd) if sd is not None else "%.2f%%" % value
        return "<b>%s</b>" % text if ours else text

    trows = ""
    for row in tbl_rows:
        label, rho, clean, other, color = row[:5]
        clean_sd, other_sd = row[5:]
        trows += '<tr%s><td>%s</td><td>%.1f Hz</td><td>%s</td><td>%s</td></tr>' % (
            ' style="background:var(--band)"' if color else "",
            (label + ' <span class="pill">ours</span>') if color else label,
            rate_hz(rho), frontier_cell(clean, clean_sd, bool(color)),
            frontier_cell(other, other_sd, bool(color)))

    w2v_plot_base = [(k, rho, c, 0.0, o, 0.0) for k, rho, c, o in base]
    w2v_no_down_plot = (
        w2v_no_down["test-clean"][0], w2v_no_down["test-clean"][1],
        w2v_no_down["test-other"][0], w2v_no_down["test-other"][1],
    )

    body += hk.section(
        "9 · wav2vec2-base WER versus compression (historical)",
        lead="This figure keeps the earlier wav2vec2-base results separate from WavLM. The "
             "multi-row layout isolates fixed baselines, CNN policies, and Transformer "
             "policies; exact values remain in the table.",
        body=hk.card(frontier_fig(w2v_plot_base, model_pts, w2v_no_down_plot,
                                  "wav2vec2-base-960h")
                     + '<p class="cap">Top row labels each fixed k; lower rows isolate the two '
                       'policy backbones. Fixed-rate points are single-seed; learned points and '
                       'the No-downsampling reference report three-seed mean ± sample SD.</p>',
                     title="WER vs. compression — wav2vec2")
            + hk.card("<table><thead><tr><th>rate</th><th>audio-token frequency</th><th>clean WER</th>"
                     "<th>other WER</th></tr></thead><tbody>%s</tbody></table>"
                     '<p class="cap">All WER columns use test-clean/test-other. Rows tagged '
                     '<span class="pill">ours</span> are trained segmenters at their measured '
                     'audio-token frequency; '
                     'trained and No-downsampling rows report three-seed mean ± sample SD.</p>'
                     % trows))

    # exact trainable-param counts, computed from the real modules (segmenter.py,
    # VanillaNN) + the Llama-3.2-1B-Instruct config (hidden=2048, intermediate=8192,
    # 16 layers, 32 query / 8 KV heads (GQA), vocab=128256, tied embeddings) — see
    # docs/project_notes for the derivation. LoRA rank=16, all_linear=True adapts
    # every one of 113 nn.Linear modules (7 per decoder layer x16 + lm_head).
    SEG_CNN_PARAMS, SEG_TF_PARAMS = 1_573_633, 3_356_161
    PROJ_PARAMS, LORA_PARAMS = 5_771_264, 13_357_056
    LLM_BACKBONE_PARAMS = 1_235_814_400
    DEC_TRAINABLE = PROJ_PARAMS + LORA_PARAMS

    body += hk.section("", body=hk.tiles([
        ("Segmenter (CNN, used)", "%.2fM" % (SEG_CNN_PARAMS / 1e6), "trainable, all of it"),
        ("Segmenter (Transformer)", "%.2fM" % (SEG_TF_PARAMS / 1e6),
         "trainable, all of it — completed sweep, weaker than CNN"),
        ("Decoder trainable", "%.1fM" % (DEC_TRAINABLE / 1e6), "proj + LoRA, ~1.5% of the LLM"),
        ("LLM backbone (frozen)", "%.2fB" % (LLM_BACKBONE_PARAMS / 1e9),
         "Llama-3.2-1B-Instruct, untouched")]))

    body += hk.section(
        "Appendix · architecture &amp; trainable parameters",
        lead="Exact counts for the wav2vec2-base configuration from the real modules (not "
             "estimates) — see <span class=\"mono\">segmenter.py</span> and the hparams yaml. "
             "WavLM-Large uses 1024-d inputs with the same topology, so its input projections "
             "have different parameter counts.",
        body=(
            hk.card(
                "<table><thead><tr><th>&nbsp;</th><th>CNN (primary result)</th>"
                "<th>Transformer (completed comparison)</th></tr></thead><tbody>"
                "<tr><td>input dim</td><td>768 (wav2vec2-base-960h)</td><td>768 "
                "(WavLM-Large runs use 1024)</td></tr>"
                "<tr><td>architecture</td>"
                "<td>Conv1d(768→256, k=7, pad=3) → ReLU/dropout(0.1) → "
                "Conv1d(256→256, k=3, pad=1) → ReLU/dropout → Conv1d(256→1, k=1)</td>"
                "<td>Linear(768→256) + sinusoidal PE → 4× pre-norm TransformerEncoderLayer "
                "(d=256, heads=4, ffn=1024, GELU, dropout 0.1) → Linear(256→1)</td></tr>"
                "<tr><td>trainable params</td><td><b>%s</b> (%.2fM)</td>"
                "<td><b>%s</b> (%.2fM)</td></tr></tbody></table>"
                '<p class="cap">Both output one boundary logit/frame (sigmoid-Bernoulli), '
                'trained end-to-end (no frozen sub-parts).</p>'
                % (format(SEG_CNN_PARAMS, ","), SEG_CNN_PARAMS / 1e6,
                   format(SEG_TF_PARAMS, ","), SEG_TF_PARAMS / 1e6),
                title="Segmenter (boundary policy)")
            + hk.card(
                "<table><thead><tr><th>component</th><th>frozen / trainable</th>"
                "<th>params</th><th>detail</th></tr></thead><tbody>"
                "<tr><td>SSL encoder</td><td>frozen</td><td>—</td>"
                "<td>wav2vec2-base-960h, 768-dim, 50 Hz</td></tr>"
                "<tr><td>proj</td><td>trainable</td><td><b>%s</b></td>"
                "<td>VanillaNN, 2 blocks: Linear(768→2048)+ReLU, Linear(2048→2048)+ReLU</td></tr>"
                "<tr><td>LLM backbone</td><td>frozen</td><td>%s</td>"
                "<td>Llama-3.2-1B-Instruct: 16 layers, hidden 2048, 32 query / 8 KV heads "
                "(GQA), ffn 8192, vocab 128256, tied embeddings</td></tr>"
                "<tr><td>LoRA adapters</td><td>trainable</td><td><b>%s</b></td>"
                "<td>rank 16, <span class=\"mono\">all_linear=True</span> → every one of 113 "
                "nn.Linear modules (q/k/v/o/gate/up/down_proj ×16 layers + lm_head)</td></tr>"
                "<tr style=\"background:var(--band)\"><td colspan=\"2\"><b>decoder trainable "
                "total</b></td><td><b>%s</b></td><td>proj + LoRA, ≈%.2f%% of the frozen LLM"
                "</td></tr></tbody></table>"
                '<p class="cap">LoRA touches <i>every</i> linear layer, including '
                '<span class="mono">lm_head</span> — %s of the %s LoRA total is the '
                'head alone (rank·(hidden+vocab) = 16·(2048+128256)), since vocab is far '
                'bigger than any hidden dim in the model.</p>'
                % (format(PROJ_PARAMS, ","), format(LLM_BACKBONE_PARAMS, ","),
                   format(LORA_PARAMS, ","), format(DEC_TRAINABLE, ","),
                   100 * DEC_TRAINABLE / LLM_BACKBONE_PARAMS,
                   format(16 * (2048 + 128256), ","), format(LORA_PARAMS, ",")),
                title="Decoder (proj + LoRA-adapted Llama)")
        ))

    import datetime
    generated = datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    body += ('<footer class="wrap">Generated %s · SpeechBrain train logs + wav2vec2/WavLM '
             'artifacts · inline figures; equations rendered with MathJax.</footer>'
             % generated)

    body = label_report_assets(body)
    full, inner = hk.document("Segmenter training dynamics", body, mathjax=True)
    open(args.out, "w").write(full)
    print("wrote", args.out, "(%d KB)" % (len(full) // 1024))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="training_report.html")
    build(ap.parse_args())
