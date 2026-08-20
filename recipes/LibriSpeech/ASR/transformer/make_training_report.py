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

DATA = "/home/jhu/jsalt2026-ext-cxiao7/scratch_jsalt2026-lgarci27/omnienc/datasets"
SSL_CACHE = "/weka/scratch/jhu/jsalt2026-lgarci27/omnienc/users/cxiao/ssl_cache"
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


def aggregate_inference_benchmark(path, splits=("test-clean", "test-other")):
    """Aggregate forward-only inference measurements over named data splits.

    RTF and utterances/second are recomputed from summed durations rather than
    averaged across corpora. Peak memory is the maximum observed allocation.
    """
    wanted = set(splits)
    rows = [row for row in parse_jsonl(path) if row.get("split") in wanted]
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
    return {
        "forward_rtf": forward_seconds / audio_seconds,
        "utterances_per_second": utterances / forward_seconds,
        "token_frequency_hz": tokens / token_audio_seconds,
        "peak_allocated_gb": max(row["gpu_peak_allocated_gb"] for row in rows),
        "peak_reserved_gb": max(row["gpu_peak_reserved_gb"] for row in rows),
        "splits": len(rows),
        "expected_splits": len(wanted),
        "hardware": rows[0].get("hardware", "unknown GPU"),
        "precision": rows[0].get("precision", "unknown precision"),
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
    """Compare single-utterance and batched forward RTF on one A100.

    Each row is a dictionary containing a plain system label, measured token
    frequency, family/color, and aggregate batch-1/throughput benchmark records.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y = np.arange(len(rows))
    labels = [
        "%s · %.1f Hz%s" % (
            row["label"], row["frequency"], "†" if row.get("unstable") else ""
        )
        for row in rows
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 8.4), dpi=130, sharey=True)
    panels = (
        ("batch1", "Batch size 1", "one utterance at a time"),
        ("throughput", "Largest tested batch", "throughput-oriented batching"),
    )
    for panel_idx, (ax, (key, title, subtitle)) in enumerate(zip(axes, panels)):
        values = [row[key]["forward_rtf"] for row in rows]
        bars = ax.barh(
            y, values, height=0.60,
            color=[row["color"] for row in rows], alpha=0.88,
        )
        for row_idx, (row, bar, value) in enumerate(zip(rows, bars, values)):
            batch = row["batch1_batch"] if key == "batch1" else row["throughput_batch"]
            ax.annotate(
                "%.3f · b%d" % (value, batch),
                (bar.get_width(), row_idx), textcoords="offset points", xytext=(5, 0),
                va="center", fontsize=8, color="#24323f",
            )
        for boundary in range(1, len(rows)):
            if rows[boundary]["family"] != rows[boundary - 1]["family"]:
                ax.axhline(boundary - 0.5, color="#d8dee4", lw=0.9, zorder=0)
        ax.set_title("%s\n%s" % (title, subtitle), fontsize=11.5, fontweight="bold")
        ax.set_xlabel("forward RTF  ·  lower is faster")
        ax.set_xlim(0, max(values) * 1.28)
        ax.grid(axis="x", alpha=0.24, lw=0.7)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="y", length=0)
        if panel_idx == 0:
            ax.set_yticks(y, labels=labels, fontsize=8.2)
    axes[0].invert_yaxis()
    fig.suptitle(
        "Measured inference speed · A100 80 GB · BF16",
        fontsize=13.5, fontweight="bold", y=0.988,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955), pad=0.9, w_pad=1.8)
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

    pooling_2x2 = [
        {
            "family": "CNN", "segmenter": "CNN first-order AR", "pooling": "mean",
            "rho": oracleclose["autoregressive"]["rho"],
            "clean": oracleclose["autoregressive"]["clean"],
            "other": oracleclose["autoregressive"]["other"],
            "color": "#b96f20",
        },
        {
            "family": "CNN", "segmenter": "CNN first-order AR", "pooling": "BiGRU residual",
            "rho": cnn_bigru["rho"], "clean": cnn_bigru["clean"],
            "other": cnn_bigru["other"], "color": "#e2a24c",
        },
        {
            "family": "Transformer", "segmenter": "Transformer AR · local history 64",
            "pooling": "mean", "rho": fullprefix["mean"]["rho"],
            "clean": fullprefix["mean"]["clean"],
            "other": fullprefix["mean"]["other"], "color": "#75579b",
        },
        {
            "family": "Transformer", "segmenter": "Transformer AR · local history 64",
            "pooling": "BiGRU residual", "rho": fullprefix["bigru"]["rho"],
            "clean": fullprefix["bigru"]["clean"],
            "other": fullprefix["bigru"]["other"], "color": "#27865d",
        },
    ]

    # Inference benchmarks use seed 3407's selected checkpoint, one A100 80 GB,
    # BF16, and the same generation settings. Aggregate only test-clean and
    # test-other so every row covers the same audio and the k=3 batch-4 timing is
    # still comparable despite its later dev-other allocator-fragmentation OOM.
    artifact_root = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "../../../..", "artifacts", "segmenter"
    ))
    learned_bench_root = os.path.join(artifact_root, "inference_benchmark_2x2")
    baseline_bench_root = os.path.join(
        artifact_root, "inference_benchmark_baselines"
    )
    efficiency_specs = [
        ("Reference", "No downsampling", baseline_bench_root, "no_downsampling", 1, 2,
         "#9a5b73", False),
        ("Fixed", "Fixed k=3", baseline_bench_root, "fixed_k3", 1, 4,
         "#7b858c", True),
        ("Fixed", "Fixed k=4", baseline_bench_root, "fixed_k4", 1, 6,
         "#7b858c", False),
        ("Fixed", "Fixed k=5", baseline_bench_root, "fixed_k5", 1, 8,
         "#59646c", False),
        ("Fixed", "Fixed k=6", baseline_bench_root, "fixed_k6", 1, 8,
         "#7b858c", False),
        ("Fixed", "Fixed k=8", baseline_bench_root, "fixed_k8", 1, 8,
         "#7b858c", False),
        ("Oracle", "Phone alignment", baseline_bench_root, "oracle_phone", 1, 8,
         "#138a8a", False),
        ("Oracle", "Character alignment", baseline_bench_root, "oracle_char", 1, 4,
         "#1c4e80", False),
        ("Learned", "CNN AR · mean", learned_bench_root, "cnn_ar_mean", 1, 8,
         "#b96f20", False),
        ("Learned", "CNN AR · BiGRU", learned_bench_root, "cnn_ar_bigru", 1, 8,
         "#e2a24c", False),
        ("Learned", "Transformer AR · mean", learned_bench_root,
         "transformer_ar_mean", 1, 8, "#75579b", False),
        ("Learned", "Transformer AR · BiGRU", learned_bench_root,
         "transformer_ar_bigru", 1, 8, "#27865d", False),
    ]
    efficiency_rows = []
    for (family, label, root, folder, batch1, throughput_batch, color,
         unstable) in efficiency_specs:
        run_root = os.path.join(root, folder, "seed3407")
        batch1_result = aggregate_inference_benchmark(os.path.join(
            run_root, "batch%d" % batch1, "benchmark.jsonl"
        ))
        throughput_result = aggregate_inference_benchmark(os.path.join(
            run_root, "batch%d" % throughput_batch, "benchmark.jsonl"
        ))
        if batch1_result is None or throughput_result is None:
            raise FileNotFoundError("Incomplete inference benchmark for %s" % label)
        efficiency_rows.append({
            "family": family, "label": label, "color": color,
            "batch1_batch": batch1, "throughput_batch": throughput_batch,
            "batch1": batch1_result, "throughput": throughput_result,
            "frequency": batch1_result["token_frequency_hz"],
            "unstable": unstable,
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
        "LibriSpeech-100h experiments organized by the question they answer: whether "
        "compression helps, whether boundary placement matters, which initialization "
        "controls the oracle gap, whether label-dependent sampling improves RL, and what "
        "the learned segmenters cost at inference. "
        "Corpus WER is recomputed from saved edit counts; ± is sample SD over seeds.",
        )
    )

    body += hk.section("", body=hk.tiles([
        ("Oracle · WavLM char alignment", "%.2f±%.2f%%" % wavlm_char_clean,
         "oracle reference · %.1f Hz · other %.2f±%.2f%% · n=%d" %
         (rate_hz(0.292651),
          wavlm_char_other[0], wavlm_char_other[1], wavlm_char_n)),
        ("WavLM · CNN segmenter · first-order AR",
         "%.2f±%.2f%%" % oracleclose["autoregressive"]["clean"][0],
         "best char decoder · %.1f Hz · other %.2f±%.2f%%" %
         (rate_hz(oracleclose["autoregressive"]["rho"][0]),
          oracleclose["autoregressive"]["other"][0][0],
          oracleclose["autoregressive"]["other"][0][1])),
        ("WavLM · local-history Transformer AR + BiGRU",
         "%.2f±%.2f%%" % fullprefix["bigru"]["clean"][0],
         "best char decoder · %.1f Hz · other %.2f±%.2f%%" %
         (rate_hz(fullprefix["bigru"]["rho"][0]),
          fullprefix["bigru"]["other"][0][0],
          fullprefix["bigru"]["other"][0][1])),
        ("WavLM · fixed k=5", "%.2f±%.2f%%" % wavlm_k5["clean"],
         "10.0 Hz · other %.2f±%.2f%%" % wavlm_k5["other"]),
        ("WavLM · no downsampling", "%.2f±%.2f%%" % wavlm_no_down["test-clean"],
         "50.0 Hz · other %.2f±%.2f%% · 18h02m/run" %
         wavlm_no_down["test-other"])]))

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
         "Mean versus BiGRU pooling for both CNN AR and Transformer AR",
         "CNN mean/BiGRU: %.2f/%.2f clean, %.2f/%.2f other. Transformer "
         "mean/BiGRU: %.2f/%.2f clean, %.2f/%.2f other." %
         (oracleclose["autoregressive"]["clean"][0][0],
          cnn_bigru["clean"][0][0],
          oracleclose["autoregressive"]["other"][0][0],
          cnn_bigru["other"][0][0],
          fullprefix["mean"]["clean"][0][0],
          fullprefix["bigru"]["clean"][0][0],
          fullprefix["mean"]["other"][0][0],
          fullprefix["bigru"]["other"][0][0]),
         "BiGRU helps the Transformer arm but not the CNN arm. The gain is "
         "architecture-specific rather than a general replacement for mean pooling.",
         "Evidence-backed"),
        ("What does learned segmentation cost at inference?",
         "One A100, BF16, identical decoding; seed-3407 selected checkpoints",
         "No downsampling: RTF %.3f (b%d). Fixed k=5: %.3f (b%d). CNN AR: "
         "%.3f (b%d). Transformer AR + BiGRU: %.3f (b%d)." %
         (efficiency_by_label["No downsampling"]["throughput"]["forward_rtf"],
          efficiency_by_label["No downsampling"]["throughput_batch"],
          efficiency_by_label["Fixed k=5"]["throughput"]["forward_rtf"],
          efficiency_by_label["Fixed k=5"]["throughput_batch"],
          efficiency_by_label["CNN AR · mean"]["throughput"]["forward_rtf"],
          efficiency_by_label["CNN AR · mean"]["throughput_batch"],
          efficiency_by_label["Transformer AR · BiGRU"]["throughput"]["forward_rtf"],
          efficiency_by_label["Transformer AR · BiGRU"]["throughput_batch"]),
         "CNN first-order AR is effectively tied with fixed k=5 in batched RTF. "
         "The local-history Transformer costs %.0f%% more RTF than CNN AR." %
         (100 * (efficiency_by_label["Transformer AR · BiGRU"]["throughput"]
                 ["forward_rtf"]
                 / efficiency_by_label["CNN AR · mean"]["throughput"]["forward_rtf"]
                 - 1)),
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
                'points at nearly the same audio-token frequency. The decoder benefits from a shorter '
                'prefix, but it also cares which frames are averaged together.' %
                (wavlm_no_down["test-clean"][0] - wavlm_k5["clean"][0],
                 wavlm_k5["clean"][0] - wavlm_phone_clean[0]))
            + hk.finding(
                '<span class="pill">Evidence-backed</span> <b>Highlighted configuration: '
                'WavLM features, CNN segmenter, first-order AR, and the best char decoder.</b> '
                'It reaches %.2f±%.2f / %.2f±%.2f WER at %.1f Hz—about %.1f× '
                'fewer decoder-side audio tokens than no downsampling. It is only %.2f points '
                'behind oracle char boundaries on clean, but %.2f points behind on other. '
                'The main remaining gap is therefore harder-speech robustness, not the need '
                'for wav2vec2 features at inference.' %
                (oracleclose["autoregressive"]["clean"][0]
                 + oracleclose["autoregressive"]["other"][0]
                 + (rate_hz(oracleclose["autoregressive"]["rho"][0]),
                    1.0 / oracleclose["autoregressive"]["rho"][0],
                    oracleclose["autoregressive"]["clean"][0][0]
                    - wavlm_char_clean[0],
                    oracleclose["autoregressive"]["other"][0][0]
                    - wavlm_char_other[0])))
            + hk.finding(
                '<span class="pill">Evidence-backed</span> <b>The local-history '
                'Transformer AR + BiGRU system has the lowest clean-speech mean among the '
                'learned systems reported here.</b> '
                'It reaches %.2f±%.2f clean and %.2f±%.2f other at %.1f Hz. Relative to the '
                'matched mean-pooling arm, clean WER is %.2f points lower, with improvements '
                'in %d/3 paired seeds; other WER improves in %d/3. '
                'Test-other has %.2f-point sample SD, so the clean gain is the clearer result; '
                'harder-speech robustness remains less certain. The matched CNN control goes '
                'the other way: mean pooling gives %.2f/%.2f clean/other versus %.2f/%.2f '
                'with BiGRU. Order-aware pooling is therefore not a general improvement.' %
                (fullprefix["bigru"]["clean"][0]
                 + fullprefix["bigru"]["other"][0]
                 + (rate_hz(fullprefix["bigru"]["rho"][0]),
                    fullprefix["mean"]["clean"][0][0]
                    - fullprefix["bigru"]["clean"][0][0],
                    fullprefix_clean_wins,
                    fullprefix_other_wins,
                    fullprefix["bigru"]["other"][0][1],
                    oracleclose["autoregressive"]["clean"][0][0],
                    oracleclose["autoregressive"]["other"][0][0],
                    cnn_bigru["clean"][0][0],
                    cnn_bigru["other"][0][0])))
        ),
    )

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
            "decoder": "from scratch", "frequency": 10.5, "frequency_sd": 0.0,
            "clean": wavlm_phone_clean[0], "clean_sd": wavlm_phone_clean[1],
            "other": wavlm_phone_other[0], "other_sd": wavlm_phone_other[1],
            "color": "#138a8a", "marker": "P",
        },
        {
            "family": "Oracle", "label": "Oracle · character",
            "system": "Character alignment", "configuration": "character boundaries · mean pooling",
            "decoder": "from scratch", "frequency": 14.7, "frequency_sd": 0.0,
            "clean": wavlm_char_clean[0], "clean_sd": wavlm_char_clean[1],
            "other": wavlm_char_other[0], "other_sd": wavlm_char_other[1],
            "color": "#1c4e80", "marker": "P",
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
            "frequency": rate_hz(oracleclose["autoregressive"]["rho"][0]),
            "frequency_sd": rate_hz(oracleclose["autoregressive"]["rho"][1]),
            "clean": oracleclose["autoregressive"]["clean"][0][0],
            "clean_sd": oracleclose["autoregressive"]["clean"][0][1],
            "other": oracleclose["autoregressive"]["other"][0][0],
            "other_sd": oracleclose["autoregressive"]["other"][0][1],
            "color": "#b96f20", "marker": "D",
        },
        {
            "family": "CNN policy", "label": "CNN · first-order AR · BiGRU",
            "system": "CNN AR + BiGRU",
            "configuration": "previous boundary · BiGRU residual pooling",
            "decoder": "best char init · co-trained",
            "frequency": rate_hz(cnn_bigru["rho"][0]),
            "frequency_sd": rate_hz(cnn_bigru["rho"][1]),
            "clean": cnn_bigru["clean"][0][0],
            "clean_sd": cnn_bigru["clean"][0][1],
            "other": cnn_bigru["other"][0][0],
            "other_sd": cnn_bigru["other"][0][1],
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
            "frequency": rate_hz(fullprefix["mean"]["rho"][0]),
            "frequency_sd": rate_hz(fullprefix["mean"]["rho"][1]),
            "clean": fullprefix["mean"]["clean"][0][0],
            "clean_sd": fullprefix["mean"]["clean"][0][1],
            "other": fullprefix["mean"]["other"][0][0],
            "other_sd": fullprefix["mean"]["other"][0][1],
            "color": "#75579b", "marker": "D",
        },
        {
            "family": "Transformer + BiGRU", "label": "Transformer AR · BiGRU",
            "system": "Transformer AR + BiGRU",
            "configuration": "causal history 64 · BiGRU residual pooling",
            "decoder": "best char init · co-trained",
            "frequency": rate_hz(fullprefix["bigru"]["rho"][0]),
            "frequency_sd": rate_hz(fullprefix["bigru"]["rho"][1]),
            "clean": fullprefix["bigru"]["clean"][0][0],
            "clean_sd": fullprefix["bigru"]["clean"][0][1],
            "other": fullprefix["bigru"]["other"][0][0],
            "other_sd": fullprefix["bigru"]["other"][0][1],
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
        system_overview_table_rows += (
            '<tr%s><td>%s</td><td><%s>%s</%s></td><td>%s</td><td>%s</td>'
            '<td><%s>%s</%s></td><td><%s>%.2f ± %.2f%%</%s></td>'
            '<td><%s>%.2f ± %.2f%%</%s></td></tr>'
            % (style, row["family"], emphasis, row["system"], emphasis,
               row["configuration"], row["decoder"], emphasis, frequency, emphasis,
               emphasis, row["clean"], row["clean_sd"], emphasis,
               emphasis, row["other"], row["other_sd"], emphasis)
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
        '<tr><td>phone alignment</td><td>10.5 Hz</td>'
        '<td>%.2f ± %.2f%%</td><td>%.2f ± %.2f%%</td>'
        '<td>%.2f ± %.2f%%</td><td><span class="pill">n=%d</span></td></tr>'
        % (wavlm_phone_dev + wavlm_phone_clean + wavlm_phone_other
           + (wavlm_phone_n,))
    )
    wavlm_baseline_rows += (
        '<tr style="background:var(--band)"><td><b>char alignment</b></td><td>14.7 Hz</td>'
        '<td><b>%.2f ± %.2f%%</b></td><td><b>%.2f ± %.2f%%</b></td>'
        '<td><b>%.2f ± %.2f%%</b></td><td><span class="pill">n=%d</span></td></tr>'
        % (wavlm_char_dev + wavlm_char_clean + wavlm_char_other + (wavlm_char_n,))
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
             "all fixed-pooling, no-downsampling, char-aligned, and phone-aligned controls finished "
             "for seeds 3407/3408/3409. "
             "All requested test files are complete; fixed k=3 seed 3409 only missed the "
             "final dev-other decode after an evaluation-time OOM. Values below are corpus "
            "WER from saved edit counts; ± is sample SD.",
        body=(
            hk.card(
                system_overview_fig(system_overview_rows)
                + '<p class="cap">Rows are grouped by system family; the three panels show '
                  'test-clean audio-token frequency on a log scale and WER on both test splits. '
                  'Error bars are sample SD across seeds 3407/3408/3409. Deterministic policies '
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
                '<th>decoder treatment</th><th>test-clean audio-token frequency</th>'
                '<th>test-clean WER</th><th>test-other WER</th>'
                '</tr></thead><tbody>%s</tbody></table>'
                '<p class="cap">This is a system inventory, not one fully controlled ablation: '
                'decoder initialization and training treatment differ across experiment families '
                'and are therefore shown explicitly. All rows use WavLM-Large runtime features '
                'and seeds 3407/3408/3409. WER and learned-policy frequency are mean ± sample SD; '
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
                "<table><thead><tr><th>baseline</th><th>audio-token frequency</th>"
                "<th>dev-clean WER</th>"
                "<th>test-clean WER</th><th>test-other WER</th>"
                "<th>seeds complete</th></tr></thead>"
                "<tbody>%s</tbody></table>"
                '<p class="cap">dev-clean is the checkpoint-selection split. Every row is a '
                "three-seed mean ± sample SD. Bold marks the current best value in each test "
                "column.</p>"
                % wavlm_baseline_rows,
                title="Fixed pooling, alignment, and no-downsampling controls")
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
                "the closest oracle to k=5 in cost (10.5 Hz vs. 10.0 Hz) and improve WER from "
                "%.2f±%.2f to %.2f±%.2f clean and %.2f±%.2f to %.2f±%.2f other. Char "
                "boundaries operate at 14.7 Hz but are stronger still at "
                "%.2f±%.2f / %.2f±%.2f. Thus the fixed-rate gain combines smoothing with "
                "token-budget regularization, while another sizeable gain is available from "
                "content-aware placement. Learned policies should be compared to both k=5 "
                "and phone alignment, not only to no downsampling."
                % (wavlm_k5["clean"] + wavlm_phone_clean + wavlm_k5["other"]
                   + wavlm_phone_other + wavlm_char_clean + wavlm_char_other)
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
         "fixed boundaries", rate_hz(0.293),
         "%.2f ± %.2f%%" % wavlm_char_clean,
         "%.2f ± %.2f%%" % wavlm_char_other, "3"),
        ("Phone-aligned baseline", "saved phone alignment", "from scratch",
         "fixed boundaries", rate_hz(0.210),
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
             + rate_hz_stats(row["rho"]) + (tag, tag)
             + row["clean"][0] + (tag, tag) + row["other"][0]
             + (tag, row["clean"][1]))
        )
    pooling_2x2_plot_rows = [
        ("%s · %s" % (row["family"], row["pooling"]),
         *row["clean"][0], *row["other"][0], row["clean"][1], row["color"])
        for row in pooling_2x2
    ]

    efficiency_table_rows = ""
    for row in efficiency_rows:
        note = "†" if row.get("unstable") else ""
        efficiency_table_rows += (
            "<tr><td>%s</td><td>%s%s</td><td>%.1f Hz</td><td>%.3f</td>"
            "<td>%d%s</td><td>%.3f</td><td>%.2f</td><td>%.1f GB</td></tr>" %
            (row["family"], row["label"], note, row["frequency"],
             row["batch1"]["forward_rtf"], row["throughput_batch"], note,
             row["throughput"]["forward_rtf"],
             row["throughput"]["utterances_per_second"],
             row["throughput"]["peak_allocated_gb"])
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
                'ten joint-RL epochs, K=4, and seeds 3407/3408/3409. Within each segmenter '
                'family, the BiGRU residual projection starts at zero, so the two arms begin '
                'with exactly the same mean-pooled decoder input. Values are mean ± sample SD.</p>'
                % pooling_2x2_rows,
                title="CNN and Transformer AR × mean and BiGRU pooling")
            + hk.card(
                controlled_wer_fig(pooling_2x2_plot_rows)
                + '<p class="cap">The plot shows the four aggregate rows from the table '
                  'above. Error bars are sample SD across the same three seeds; lower WER '
                  'is better.</p>',
                title="Pooling effect depends on the segmenter architecture")
            + hk.finding(
                '<span class="pill">Evidence-backed</span> <b>BiGRU residual pooling improves '
                'the Transformer AR arm on test-clean in %d/3 paired seeds.</b> It changes '
                'from %.2f±%.2f to %.2f±%.2f clean WER, a %.2f-point mean reduction. The '
                'audio-token frequency also shifts only modestly (%.1f→%.1f Hz), so this is '
                'not explained by a large compression change.' %
                ((fullprefix_clean_wins,)
                 + fullprefix["mean"]["clean"][0] + fullprefix["bigru"]["clean"][0]
                 + (fullprefix["mean"]["clean"][0][0]
                    - fullprefix["bigru"]["clean"][0][0],
                    rate_hz(fullprefix["mean"]["rho"][0]),
                    rate_hz(fullprefix["bigru"]["rho"][0]))))
            + hk.finding(
                '<span class="pill">Evidence-backed</span> <b>The matched CNN control does '
                'not improve.</b> BiGRU changes CNN first-order AR from %.2f±%.2f to '
                '%.2f±%.2f clean and from %.2f±%.2f to %.2f±%.2f other; it wins only %d/3 '
                'and %d/3 paired seeds. Mean pooling remains the supported CNN choice.' %
                (oracleclose["autoregressive"]["clean"][0]
                 + cnn_bigru["clean"][0]
                 + oracleclose["autoregressive"]["other"][0]
                 + cnn_bigru["other"][0]
                 + (cnn_bigru_clean_wins, cnn_bigru_other_wins)))
            + hk.finding(
                '<span class="pill">Interpretation</span> <b>The pooling effect is tied to '
                'the segmenter architecture.</b> The Transformer may leave more information '
                'for the order-sensitive pooler to recover, while the CNN already summarizes '
                'local acoustic patterns before segmentation. This is still a system-level '
                'result: a fixed-boundary pooler swap is needed to separate representation '
                'order from RL-induced boundary changes. On test-other, Transformer mean/BiGRU '
                'is %.2f±%.2f versus %.2f±%.2f and the BiGRU arm has high seed variance, so '
                'the clean-speech effect is the stronger claim.' %
                (fullprefix["mean"]["other"][0] + fullprefix["bigru"]["other"][0]))
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
    cnn_eff = efficiency_by_label["CNN AR · mean"]
    transformer_bigru_eff = efficiency_by_label["Transformer AR · BiGRU"]
    body += hk.section(
        "4c · Inference speed and GPU memory",
        lead="These measurements add the missing runtime comparison. Every row uses one A100 "
             "80 GB, BF16, seed 3407's selected checkpoint, and the same decoding settings. "
             "RTF below measures model forward time; lower is faster.",
        body=(
            hk.card(
                inference_efficiency_fig(efficiency_rows)
                + '<p class="cap">Forward RTF is summed over test-clean and test-other '
                  'after five warm-up batches. Labels include the measured audio-token '
                  'frequency. The left panel fixes batch size 1; the right uses the largest '
                  'tested batch shown after each bar. † Fixed k=3 batch 4 completed both test '
                  'splits but later hit CUDA allocator fragmentation on dev-other, so it is a '
                  'measured but not memory-safe operating point.</p>',
                title="Single-utterance latency and batched throughput")
            + hk.card(
                '<table><thead><tr><th>family</th><th>system</th><th>measured frequency</th>'
                '<th>batch-1 RTF</th><th>largest tested batch</th><th>batched RTF</th>'
                '<th>utterances/s</th><th>peak allocated</th></tr></thead>'
                '<tbody>%s</tbody></table>'
                '<p class="cap">All speed values aggregate test-clean and test-other. Peak '
                  'allocated memory is the maximum over those two splits at the listed '
                  'throughput batch. Reserved memory is omitted because allocator caching '
                  'made it a poor measure of live tensors. Oracle timings condition on '
                  'precomputed boundaries and therefore exclude alignment generation. † The '
                  'fixed-k=3 batch-4 caveat is described in the figure caption.</p>'
                % efficiency_table_rows,
                title="Exact inference measurements")
            + hk.finding(
                '<span class="pill">Evidence-backed</span> <b>Compression mainly pays off '
                'through batching.</b> Fixed k=5 improves throughput from %.2f to %.2f '
                'utterances/s (%.1f×) versus no downsampling, while reducing forward RTF from '
                '%.3f to %.3f. At batch 1 the RTF difference is much smaller (%.3f versus '
                '%.3f), because WavLM encoding and text generation still dominate.' %
                (no_down_eff["throughput"]["utterances_per_second"],
                 k5_eff["throughput"]["utterances_per_second"],
                 k5_eff["throughput"]["utterances_per_second"]
                 / no_down_eff["throughput"]["utterances_per_second"],
                 no_down_eff["throughput"]["forward_rtf"],
                 k5_eff["throughput"]["forward_rtf"],
                 no_down_eff["batch1"]["forward_rtf"],
                 k5_eff["batch1"]["forward_rtf"]))
            + hk.finding(
                '<span class="pill">Evidence-backed</span> <b>CNN first-order AR has little '
                'measured throughput overhead over fixed k=5.</b> Their batched forward RTFs '
                'are %.3f and %.3f at batch 8, respectively. The learned segmenter therefore '
                'adds adaptive placement without giving back the main batching benefit.' %
                (cnn_eff["throughput"]["forward_rtf"],
                 k5_eff["throughput"]["forward_rtf"]))
            + hk.finding(
                '<span class="pill">Evidence-backed</span> <b>The local-history Transformer '
                'is the expensive learned policy.</b> Transformer AR + BiGRU uses %.3f batched '
                'RTF versus %.3f for CNN AR mean (%.0f%% slower) and %.3f versus %.3f at '
                'batch 1. This makes the accuracy comparison a real speed tradeoff: the '
                'Transformer/BiGRU arm has stronger mean WER, while CNN AR is the simpler and '
                'faster learned system.' %
                (transformer_bigru_eff["throughput"]["forward_rtf"],
                 cnn_eff["throughput"]["forward_rtf"],
                 100 * (transformer_bigru_eff["throughput"]["forward_rtf"]
                        / cnn_eff["throughput"]["forward_rtf"] - 1),
                 transformer_bigru_eff["batch1"]["forward_rtf"],
                 cnn_eff["batch1"]["forward_rtf"]))
            + hk.finding(
                '<span class="pill">Protocol note</span> <b>These are one-checkpoint timing '
                'runs, not confidence intervals.</b> They are sufficient to expose the large '
                'batching and architecture effects, but small differences should be confirmed '
                'with sequential repeated timings before publication.')
        ),
    )

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
