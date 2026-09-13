#!/usr/bin/env python3
"""Build the published non-ASR boundary audit from retained numerical evidence.

No inference, training, or manuscript editing. The HTML displays offline; the
adjacent assets directory holds downloadable PDFs and a portable evidence ZIP.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import io
import json
import os
from pathlib import Path
import shutil
import statistics
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[4]
DEFAULT = ROOT / "artifacts/segmenter/nonasr_boundary_audit_2026-09-12"
SKILL = Path(os.environ.get("HTML_REPORT_SKILL_DIR", Path.home() / ".codex/skills/html-report"))
sys.path.insert(0, str(SKILL))
import htmlkit as hk  # noqa: E402

SEEDS = (3407, 3408, 3409)
CORPORA = {"cremad": "CREMA-D", "expresso": "Expresso", "librispeech": "LibriSpeech"}
ARMS = {"nods_50hz": "Native frames", "fixed_25hz": "Fixed stride 2",
        "fixed_10hz": "Fixed stride 5", "fixed_5hz": "Fixed stride 10",
        "learned_frozen": "Frozen ASR policy", "learned_grpo": "Task GRPO"}
POLICIES = {"asr_3408": "ASR parent", "emotion_3408": "Emotion · selected",
            "intent_3408": "Intent · selected", "speaker_count_3408": "Speaker count · selected",
            "intent_3408_last": "Intent · last step", "speaker_count_3408_last": "Speaker count · last step"}
SITE = "https://borrisonxiao.github.io/jsalt26-downsampling"
VERSION = "20260913-nonasr-boundaries"
ASSETS = "nonasr-task-boundaries-assets"

CSS = """
.wrap{max-width:1080px}body{font-family:ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
h1{font-size:clamp(29px,4vw,43px);max-width:850px;letter-spacing:-.035em}
h2{font-size:25px;letter-spacing:-.025em;margin-bottom:12px}h3{font-size:18px}
p{max-width:88ch}a{color:var(--series-asr);text-underline-offset:3px}
.kicker{color:var(--series-asr);font-size:12px}.sub{font-size:18px;max-width:83ch}
.jump{display:flex;gap:6px;flex-wrap:wrap;padding:10px 0 18px;border-bottom:1px solid var(--border)}
.jump a,.button,.filter button{display:inline-block;padding:7px 12px;border:1px solid var(--border);border-radius:8px;background:var(--surface-1);color:var(--text-primary);font:inherit;font-size:13px;text-decoration:none;cursor:pointer}
.jump a:hover,.filter button[aria-pressed=true]{border-color:var(--series-asr);background:var(--surface-2)}
.section{scroll-margin-top:18px;padding:32px 0}.section:focus{outline:none}
.table-scroll,.figure-scroll{max-width:100%;overflow-x:auto;margin:18px 0;border:1px solid var(--border);border-radius:10px;background:var(--surface-1)}
table{margin:0;font-variant-numeric:tabular-nums;line-height:1.5}caption{text-align:left;padding:14px 16px;color:var(--text-secondary);font-size:13px;border-bottom:1px solid var(--border)}
thead th{text-transform:none;font-weight:650;letter-spacing:0;color:var(--text-secondary);padding:10px 14px;white-space:normal}
td{padding:9px 14px}th,td{vertical-align:top}td:first-child{min-width:140px}tr.emphasis{background:var(--band)}
table.wide{min-width:780px}table.matrix{min-width:700px}table td:not(:first-child){white-space:nowrap}
.label{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--text-muted)}
.finding{font-size:16px;line-height:1.6}.note{border-left:3px solid var(--pair-sed-asr);background:var(--band);padding:14px 18px;border-radius:0 10px 10px 0;margin:18px 0}.note p{margin:5px 0}
.compact{font-size:14px;color:var(--text-secondary)}.filter{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0}.filter[hidden]{display:none}
figure{margin:22px 0;padding:14px;border:1px solid var(--border);border-radius:12px;background:var(--surface-1)}
figure img{display:block;width:100%;height:auto;background:white;border-radius:6px}figcaption{font-size:13px;color:var(--text-secondary);margin-top:12px;line-height:1.6}
figcaption a{margin-left:8px}.figure-scroll{border:0;margin:0;background:white}.figure-scroll img{min-width:700px}
details{border:1px solid var(--border);border-radius:10px;padding:12px 16px;margin:18px 0;background:var(--surface-1)}summary{font-weight:650;cursor:pointer}details .table-scroll{margin-bottom:0}
blockquote{margin:16px 0;padding:14px 20px;border-left:3px solid var(--series-sed);background:var(--band);line-height:1.7}
.grid2{gap:18px}.grid2 .card{margin:0}.stat-row{margin:24px 0}.tile .l{font-weight:650}.tile .s{line-height:1.5}.tile .v{font-size:30px}
footer{font-size:13px;color:var(--text-secondary);padding:25px 0 40px;border-top:1px solid var(--border)}
code{font-size:.86em;overflow-wrap:anywhere}li{margin:7px 0}.sources a{overflow-wrap:anywhere}[hidden]{display:none!important}
@media(max-width:620px){.wrap{padding:0 16px}.section{padding:25px 0}.stat-row{grid-template-columns:1fr 1fr}.tile{padding:13px}.tile .v{font-size:25px}.grid2{grid-template-columns:1fr}td,thead th{padding:8px 10px}figure{padding:8px}}
@media print{details>*{display:block!important}.filter{display:none!important}[hidden]{display:block!important}.wrap{max-width:none}.table-scroll{overflow:visible}figure,table{break-inside:avoid}.jump{display:none}}
"""

JS = """
document.querySelectorAll('[data-filter]').forEach(group=>{
 group.hidden=false;
 group.querySelectorAll('button').forEach(button=>button.addEventListener('click',()=>{
  group.querySelectorAll('button').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));
  document.querySelectorAll(group.dataset.filter).forEach(row=>{row.hidden=button.dataset.value!=='all'&&row.dataset.corpus!==button.dataset.value;});
 }));
});
"""


def esc(value):
    return html.escape(str(value))


def avg(values):
    return statistics.mean(values)


def pct(value, digits=1):
    if value is None:
        return '—'
    return f"{100 * value:.{digits}f}"


def ci(value, scale=1):
    lo, hi = value["ci95"]
    return f"{scale * value['mean']:.3f} [{scale * lo:.3f}, {scale * hi:.3f}]"


class Report:
    def __init__(self, source, output):
        self.source, self.output = source, output
        self.assets = output.parent / ASSETS
        self.assets.mkdir(parents=True, exist_ok=True)
        self.tables = self.figures = 0

    def read(self, name):
        return json.loads((self.source / name).read_text())

    def table(self, heads, rows, caption, *, wide=False, groups=None):
        self.tables += 1
        body = []
        for i, row in enumerate(rows):
            attrs = f' data-corpus="{groups[i]}"' if groups else ""
            body.append(f"<tr{attrs}>" + "".join(f"<td>{esc(x)}</td>" for x in row) + "</tr>")
        return ('<div class="table-scroll"><table' + (' class="wide"' if wide else '') + '>'
                f'<caption><strong>Table {self.tables}.</strong> {caption}</caption><thead><tr>'
                + "".join(f'<th scope="col">{esc(x)}</th>' for x in heads)
                + '</tr></thead><tbody>' + "".join(body) + '</tbody></table></div>')

    def figure(self, stem, caption, alt):
        self.figures += 1
        png = base64.b64encode((self.source / "figures" / f"{stem}.png").read_bytes()).decode()
        shutil.copyfile(self.source / "figures" / f"{stem}.pdf", self.assets / f"{stem}.pdf")
        return (f'<figure id="figure-{self.figures}"><div class="figure-scroll">'
                f'<img width="1400" loading="lazy" src="data:image/png;base64,{png}" alt="{esc(alt)}"></div>'
                f'<figcaption><strong>Figure {self.figures}.</strong> {caption} '
                f'<a href="{ASSETS}/{stem}.pdf">Download PDF</a></figcaption></figure>')

    @staticmethod
    def section(anchor, title, body):
        return f'<section class="section" id="{anchor}" tabindex="-1"><h2>{title}</h2>{body}</section>'

    @staticmethod
    def detail(title, body):
        return f'<details><summary>{title}</summary>{body}</details>'

    def package(self):
        names = ["task_results_summary.json", "task_results.csv", "checkpoint_audit.json",
                 "boundary_summary.json", "control_sensitivity.json", "gap_energy_summary.json",
                 "parameter_changes.json", "qualitative_examples.json", "metric_validation.json",
                 "emotion_label_distribution.json", "confidence_protocol.json", "prosody_protocol.json",
                 "data/expresso_source.json", "data/dictionary_augmentation.json"]
        for corpus in CORPORA:
            names.extend(f"{corpus}/{n}" for n in ("alignment_qc.json", "inference_metadata.json", "determinism_checks.json"))
        hashes = {n: hashlib.sha256((self.source / n).read_bytes()).hexdigest() for n in names}

        def portable(obj):
            if isinstance(obj, dict):
                return {portable(k): portable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [portable(x) for x in obj]
            if isinstance(obj, str):
                return obj.replace(str(ROOT), "$PROJECT").replace("/export/jsalt26/omnienc/users/cxiao", "$WORKSPACE")
            return obj

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as z:
            files = {}
            for name in names:
                raw = (self.source / name).read_text()
                files[name] = json.dumps(portable(json.loads(raw)), indent=2) + "\n" if name.endswith(".json") else raw
            files["source_sha256.json"] = json.dumps(hashes, indent=2) + "\n"
            files["README.txt"] = ("Non-ASR boundary audit, 12 September 2026; HTML publication 13 September 2026.\n"
                "Numerical evidence is unchanged. Absolute cluster paths in JSON have portable $PROJECT/$WORKSPACE prefixes.\n"
                "source_sha256.json hashes original local evidence before path substitution.\n"
                "Classification quality is from saved TEST records. Boundary inference is a separate evaluation.\n"
                "Selected-checkpoint quality must not be paired with last-step boundary rates.\n"
                "No datasets, audio, checkpoints, or manuscript files are included.\n")
            for name, content in sorted(files.items()):
                info = zipfile.ZipInfo(name, date_time=(2026, 9, 13, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                z.writestr(info, content.encode())
        (self.assets / "numerical-evidence.zip").write_bytes(buffer.getvalue())
        shutil.copyfile(self.source / "task_results.csv", self.assets / "task_results.csv")
        return hashes


def build(source, output):
    r = Report(source, output)
    tasks = r.read("task_results_summary.json")
    summary_doc = r.read("boundary_summary.json")
    summaries = summary_doc["corpora"]
    controls = r.read("control_sensitivity.json")
    gaps = r.read("gap_energy_summary.json")["corpora"]
    audit = r.read("checkpoint_audit.json")
    params = {x["system"]: x for x in r.read("parameter_changes.json")}
    examples = r.read("qualitative_examples.json")
    labels = r.read("emotion_label_distribution.json")
    validation = r.read("metric_validation.json")
    emo = tasks["emotion"]
    reduction = 100 * (1 - emo["learned_grpo"]["hz_mean"] / emo["learned_frozen"]["hz_mean"])
    sources = r.package()

    body = hk.hero("Analysis 03 · experiments 12 September 2026 · published 13 September 2026",
        "How task adaptation changes speech boundaries",
        "Emotion adaptation roughly halves the token rate at similar mean macro-F1. "
        "Its boundaries form coarser groups around largely shared acoustic transitions, with selective retention and local shifts.")
    body += '<nav class="jump" aria-label="On this page">' + "".join(
        f'<a href="#{anchor}">{title}</a>' for anchor, title in [
            ("task-quality", "Task quality"), ("shared-audio", "Shared audio"), ("retention", "Boundary patterns"),
            ("silences", "Silences"), ("checkpoints", "Checkpoints"), ("examples", "Examples"),
            ("methods", "Methods & controls"), ("paper", "Paper proposal"), ("downloads", "Downloads")]) + '</nav>'
    body += hk.tiles([("Emotion token reduction", f"{reduction:.1f}%", "4.15 vs 8.56 tokens/s · saved TEST"),
        ("Expresso boundary containment", "93.7%", "Emotion cuts within 20 ms of ASR"),
        ("New boundary inference", "2,216", "CREMA-D · Expresso · LibriSpeech"),
        ("Existing task records audited", "54", "3 tasks × 6 arms × 3 adaptation seeds")])
    body += ('<div class="note"><strong>What this evidence establishes.</strong><p>The rates and boundary locations change after adaptation. '
        'Three seeds do not establish an emotion accuracy gain or equivalence. Prescribed rate bands prevent interpreting the resulting '
        'frequencies as unconstrained task optima. Phone, vowel, and voicing comparisons describe patterns; they do not establish token identity or prosody preservation.</p></div>')
    body += ('<p class="compact">The analysis uses existing trained checkpoints. Classification quality below comes from saved TEST records; '
        'the new experiments extract and compare boundaries. The manuscript remains unchanged. The final section is a revision proposal.</p>')

    part = ('<p>CREMA-D uses six crowd voice-vote emotion labels, with 1,431 test utterances from 19 held-out speakers. '
        'Checkpoints were selected by validation macro-F1. All arms start from the same LibriSpeech-960h ASR seed-3408 model; '
        'seeds 3407–3409 vary adaptation, rather than the ASR initialization.</p>')
    rows = [(ARMS[a].replace("Task GRPO", "Emotion GRPO"), f"{x['hz_mean']:.2f}", f"{pct(x['metric_mean'])} ± {pct(x['metric_sd'])}") for a, x in emo.items()]
    part += r.table(["Segmentation", "Mean tokens/s ↓", "Macro-F1 (%) ↑"], rows,
        "Emotion recognition on CREMA-D. Quality is mean ± sample SD over three adaptation seeds; the displayed rate is the mean from the original classification evaluation.")
    part += hk.finding(f'<b>{reduction:.1f}% fewer tokens at similar mean macro-F1.</b> Frozen ASR: 43.1 ± 0.4%; '
        'emotion GRPO: 43.8 ± 4.4%. The larger GRPO spread matters. Fixed 5-Hz and 10-Hz baselines remain competitive.', ok=True)
    part += ('<p>Neutral accounts for 883 of 1,431 labels (61.7%), so accuracy alone would be a misleading headline. '
        'The classifier selects the label with highest mean per-token log-likelihood. GRPO rewards the gold-label score minus the strongest competing label score.</p>')
    extra_rows = []
    for arm in ARMS:
        count, intent = tasks['speaker_count'][arm], tasks['intent'][arm]
        extra_rows.append((ARMS[arm], f"{count['hz_mean']:.2f}", f"{pct(count['metric_mean'],2)} ± {pct(count['metric_sd'],2)}",
                           f"{intent['hz_mean']:.2f}", f"{pct(intent['metric_mean'],2)} ± {pct(intent['metric_sd'],2)}"))
    part += r.table(["Segmentation", "Count tokens/s", "Count accuracy (%) ↑", "Intent tokens/s", "Intent accuracy (%) ↑"], extra_rows,
        "The other audited single-task results: speaker count and intent. Mean ± sample SD over three seeds; selected checkpoints only. "
        "The checkpoint audit below explains why several selected policies have no final-block updates.", wide=True)
    part += ('<p>Intent is near ceiling; speaker-count fixed-rate baselines remain strong. These results do not identify boundary placement as the cause of a quality difference. '
        'Table 2 must not inherit the lower rates of the last-step sensitivity checkpoints shown later.</p>')
    part += r.detail('Emotion label counts and all 54 seed records', '<p>' + '; '.join(f'{esc(k)}: {v}' for k, v in labels['labels'].items())
        + f'.</p><p><a href="{ASSETS}/numerical-evidence.zip">Download the evidence ZIP</a> for individual TEST scores, validation selection, checkpoint hashes, and configuration provenance.</p>')
    body += r.section('task-quality', '1 · Quantitative task results', part)

    rows = []
    for corpus, name in CORPORA.items():
        s = summaries[corpus]
        rows.append((name, s['utterances'], s['speakers'], s['aligned_utterances'], '342 for extra policies' if corpus == 'cremad' else 'Full selected set'))
    part = r.table(['Corpus', 'Utterances', 'Speakers', 'Phone/word aligned', 'Auxiliary-policy panel'], rows,
        'New boundary inference covers 2,216 utterances. Six CREMA-D alignment failures remain in audio-only analyses. All comparisons use identical utterances within a reported panel.')
    part += ('<p><b>Expresso:</b> four speakers × seven speaking styles × 15 texts per speaker. Texts are matched across styles within a speaker; '
        'there is no text shared across every speaker/style cell. Styles are default, confused, enunciated, happy, laughing, sad, and whisper. '
        'This is an out-of-domain segmentation probe, with no Expresso emotion-classification score. LibriSpeech uses the prior selection of five test utterances per speaker, 365 total.</p>')
    rows = []
    for corpus, name in CORPORA.items():
        s = summaries[corpus]
        pairs = [s['pairs'][f'asr_3408/emotion_{seed}'] for seed in SEEDS]
        rows.append((name, f"{s['systems']['asr_3408']['mean_utterance_hz']:.2f}",
            f"{avg(s['systems'][f'emotion_{seed}']['mean_utterance_hz'] for seed in SEEDS):.2f}",
            pct(avg(p['b_in_a_20ms'] for p in pairs)), pct(avg(p['shift_null_b_in_a_20ms'] for p in pairs)),
            pct(avg(controls[corpus]['controls'][f'emotion_{seed}']['confidence_cut_f1_20ms'] for seed in SEEDS))))
    part += r.table(['Corpus', 'ASR tokens/s', 'Emotion tokens/s', 'Emotion near ASR (%)', 'Shift-null (%)', 'Emotion vs confident subset F1 (%)'], rows,
        'Same-audio boundary comparison at ±20 ms. Emotion columns average three seeds. Containment asks whether an emotion cut has any nearby ASR cut; '
        'F1 uses one-to-one matching. They are different metrics. Confidence subsets have exactly the emotion cut count per utterance.', wide=True)
    part += ('<p>Table 4 uses fresh, independent per-utterance feature extraction; its rates are distinct from the saved classification TEST rates in Tables 1–2. '
        'About 94% of emotion cuts stay near parent ASR cuts. Exact-frame containment is lower, roughly 62–66% on CREMA-D/Expresso: '
        'the policy can move a boundary locally. A circular shift preserves cut count and spacing but has much lower containment.</p>')
    part += ('<p>Keeping the same number of highest-confidence ASR cuts reproduces only about 48–59% boundary F1 across individual seeds and corpora. '
        'This simple confidence rule does not reproduce the adapted policy. Shared ASR initialization remains an explanation for shared locations; '
        'the study does not establish universal acoustic landmarks.</p>')
    part += r.figure('boundary_overview', 'Same-audio token rates, containment versus the circular-shift null, count-matched event agreement, '
        'and segment-duration distributions. Individual emotion seeds are shown where appropriate. See Tables 4–6 for exact values. '
        'All control cuts occupy the encoder frame grid.', 'Four panels comparing token rates, shared boundary locations, reference events, and segment durations.')
    body += r.section('shared-audio', '2 · Coarser grouping around shared locations', part)

    part = ('<p>Table 5 compares reference-event agreement at the same number of cuts on each utterance. Emotion boundaries show a consistent preference '
        'for vowel onsets and automatic voicing changes. Word-edge agreement is inconsistent. Vowel onset is a phone-class diagnostic, not syllable ground truth.</p>')
    part += '<div class="filter" hidden data-filter="#retention tr[data-corpus]" aria-label="Filter boundary scores by corpus">' + ''.join(
        f'<button type="button" data-value="{k}" aria-pressed="{str(k=="all").lower()}">{v}</button>' for k, v in {'all':'All corpora', **CORPORA}.items()) + '</div>'
    rows, row_groups = [], []
    for corpus, name in CORPORA.items():
        systems = [summaries[corpus]['systems'][f'emotion_{seed}'] for seed in SEEDS]
        for metric, label in [('phone_20ms','Phone edge'),('vowel_onset_20ms','Vowel onset'),('voicing_20ms','Voicing change'),('word_20ms','Word edge')]:
            rows.append((name, label, *[pct(avg(s[control][metric]['f1'] for s in systems)) for control in ['scores','grid_scores','asr_thin_scores','confidence_thin_scores']]))
            row_groups.append(corpus)
    part += r.table(['Corpus', 'Reference event', 'Emotion F1', 'Uniform grid', 'Random ASR subset', 'Confident ASR subset'], rows,
        'Boundary F1 (%) at ±20 ms, micro-averaged within corpus/seed and then averaged across three emotion seeds. '
        'Uniform grids, random ASR thinning, and confidence thinning exactly match each utterance’s emotion cut count; random thinning averages 32 trials.', wide=True, groups=row_groups)
    rows = []
    for corpus, name in CORPORA.items():
        for seed in SEEDS:
            key = f'emotion_{seed}'; s = summaries[corpus]['systems'][key]
            shape = s['phone_shape']; total = sum(shape[k] for k in ['tokens_phone_centers0','tokens_phone_centers1','tokens_phone_centers2plus'])
            rows.append((name, seed, f"{s['segment_ms_p10_p50_p90'][1]:.0f}", pct(shape['tokens_phone_centers2plus']/total),
                ci(s['feature_change_ratio_to_asr']), ci(controls[corpus]['controls'][key]['feature_change_ratio_to_confidence'])))
    part += r.table(['Corpus', 'Emotion seed', 'Median token (ms)', 'Tokens spanning ≥2 phones (%)', 'Feature change / ASR [95% CI]', 'Feature change / confident subset [95% CI]'], rows,
        'Token structure and adjacent-frame changes in frozen WavLM features. A multi-phone token contains at least two phone centers. '
        'Feature ratios are means of per-utterance ratios with clustered bootstrap 95% intervals; 1 means equal average cosine change.', wide=True)
    part += ('<p>The ASR median segment on Expresso is 60 ms, compared with 140–160 ms for emotion; 34–43% of emotion tokens contain two or more phone centers. '
        'Emotion cuts fall at larger frozen-feature changes than both all ASR cuts and confidence subsets. These are descriptive acoustic preferences, '
        'rather than a measure of perceptual salience or retained emotion information.</p>')
    rows = []
    for style, value in controls['expresso']['style_token_ratio'].items():
        rows.append((style, value['n'], f"{summaries['expresso']['systems']['asr_3408']['styles'][style]['mean_hz']:.2f}",
            f"{avg(summaries['expresso']['systems'][f'emotion_{s}']['styles'][style]['mean_hz'] for s in SEEDS):.2f}", pct(value['mean_token_ratio_to_asr'])))
    part += r.detail('Speaking-style breakdown on Expresso', r.table(['Style', 'Utterances', 'ASR tokens/s', 'Emotion tokens/s', 'Emotion / ASR tokens (%)'], rows,
        'Matched text across styles within each of four speakers. Emotion rate averages seeds; the token ratio first averages seed token counts for each utterance and then utterance ratios. '
        'These descriptive style differences are not tests of emotion recognition.'))
    body += r.section('retention', '3 · Which boundaries are retained?', part)

    part = ('<p>We first identify automatically aligned internal inter-word gaps lasting at least 250 ms. '
        '<b>Strict single-token isolation</b> requires the same segment to match both gap edges within ±40 ms. '
        'An untranscribed gap may contain laughter or other vocal activity, so we separately require low acoustic energy.</p>')
    part += ('<p>A gap is “quiet” when at least 80% of its interior 25-ms RMS windows lie at least 20 dB below the utterance’s 90th-percentile RMS, '
        'excluding 20 ms at each edge. This is a relative-energy diagnostic, not manual silence annotation.</p>')
    rows = []
    for corpus, name in CORPORA.items():
        q = gaps[corpus]['thresholds']['20']
        er = [q['systems'][f'emotion_{s}']['isolation40_fraction'] for s in SEEDS]
        rows.append((name, q['quiet_unique_gaps'], pct(q['systems']['asr_3408']['isolation40_fraction'],2), f'{pct(min(er),2)}–{pct(max(er),2)}'))
    part += r.table(['Corpus', 'Quiet gaps ≥250 ms', 'ASR isolation (%)', 'Emotion isolation, seed range (%)'], rows,
        'Rare single-token isolation also holds after filtering aligned gaps by acoustic energy. CREMA-D has only two qualifying gaps, too few for generalization.')
    part += hk.finding('<b>Quiet intervals rarely get their own bracketed token.</b> On Expresso, only 0–3.7% of the 54 quiet gaps are isolated by an emotion policy; '
        'the ASR parent isolates 1.9%. Cuts can attach a quiet interval to surrounding speech. This behavior also appears on LibriSpeech.', ok=True)
    part += ('<p><b>Possible mechanism:</b> utterance-level label reward and the specified rate objective offer no explicit reward for placing a cut at each silence edge. '
        'Attachment inherited from ASR initialization is another possibility. These explanations remain hypotheses: this audit did not isolate the training mechanism or test silence-boundary interventions.</p>')
    rows = []
    for corpus, name in CORPORA.items():
        for key in ['asr_3408', *[f'emotion_{s}' for s in SEEDS]]:
            p = summaries[corpus]['systems'][key]['pauses']
            rows.append((name, key.replace('_',' '), p['n'], f"{p['internal_cuts']:.2f}", pct(p['isolated_single_token40'],2)))
    part += r.detail('All aligned gaps, including gaps with vocal activity', r.table(['Corpus','Policy','Gaps','Mean interior cuts','Strict isolation (%)'], rows,
        'All internal word-aligned gaps ≥250 ms, before the quiet-gap filter. Emotion generally reduces interior cuts without systematically bracketing both edges.'))
    rows = []
    for corpus, name in CORPORA.items():
        for threshold, q in gaps[corpus]['thresholds'].items():
            er = [q['systems'][f'emotion_{s}']['isolation40_fraction'] for s in SEEDS]
            er = [x for x in er if x is not None]
            emotion_range = f'{pct(min(er),2)}–{pct(max(er),2)}' if er else '— (no qualifying gaps)'
            rows.append((name, f'{threshold} dB', q['quiet_unique_gaps'], pct(q['systems']['asr_3408']['isolation40_fraction'],2), emotion_range))
    part += r.detail('Quiet-gap sensitivity at 15, 20, and 25 dB', r.table(['Corpus','Below reference RMS','Quiet gaps','ASR isolation (%)','Emotion isolation (%)'], rows,
        'The required fraction of low-energy windows stays at 80%; only the relative RMS threshold changes. Emotion values give the three-seed range.'))
    body += r.section('silences', '4 · Word gaps and acoustically quiet intervals', part)

    part = ('<div class="note"><strong>Selected checkpoint ≠ last training step.</strong><p>The saved TEST phase field is stale for the selected speaker-count runs '
        'and intent seed 3408: it says “unfreeze,” but those checkpoints have no final-block updates. We verified the actual selected step and compared tensors directly.</p></div>')
    rows = []
    for key, cp in audit['checkpoints'].items():
        if not key.startswith(('emotion_', 'intent_', 'speaker_count_')):
            continue
        rows.append((key.replace('_',' '), cp['step'], cp['final_block_updates'], len(params[key]['changed_boundary_tensors']),
            'FiLM only' if cp['final_block_updates']==0 else 'FiLM + final block'))
    part += r.table(['Checkpoint','Selected / retained step','Final-block updates','Changed backbone tensors','Policy adaptation'], rows,
        'Actual checkpoint state from the schedule and parameter comparison. Pooler, projection, and LoRA also adapt. '
        '“Last” denotes a sensitivity checkpoint, not the checkpoint selected for reported task quality.', wide=True)
    part += ('<p>Late-stage bands were prescribed: <b>emotion 2–5 Hz, speaker count 3–6 Hz, intent 4–7 Hz</b>. '
        'Earlier phases used 7.5–12.5 Hz. The outcome combines the task reward, rate prior, shared initialization, and checkpoint selection. '
        'It does not reveal a task’s natural or minimum-sufficient token frequency.</p>')
    rows = [(name, *[f"{controls[c]['agreement']['mean_hz'][key]:.2f}" for c in CORPORA]) for key,name in POLICIES.items()]
    part += r.table(['Checkpoint · seed 3408','CREMA-D (342)','Expresso (420)','LibriSpeech (365)'], rows,
        'Mean tokens/s on identical audio across six checkpoints. The smaller CREMA-D panel was fixed before predictions; do not compare its rates with the full 1,431-utterance panel as a model effect.')
    part += ('<p>On Expresso, selected intent and speaker-count checkpoints retain 99.9% and 96.5% boundary F1 against ASR. '
        'Last-step controls differ much more: 79.8% and 39.9%. Emotion provides the cleaner paper extension because all three selected checkpoints include final-block adaptation.</p>')
    part += r.figure('cross_task_agreement', 'One-to-one boundary F1 (%) at ±20 ms on the same 420 Expresso utterances, seed 3408. '
        'Selected and last-step checkpoints are distinguished. Sparse-versus-dense F1 is lower than directional containment because F1 includes unmatched cuts from both policies.',
        'Six by six matrix comparing ASR, emotion, selected intent and speaker-count policies, and last-step controls.')
    matrix = controls['expresso']['agreement']
    part += r.detail('Exact matrix values', r.table(['Checkpoint', *[POLICIES[k] for k in matrix['keys']]],
        [(POLICIES[k], *[pct(x) for x in matrix['f1_20ms'][i]]) for i,k in enumerate(matrix['keys'])], 'Exact values underlying Figure 2, rounded to one decimal percent.', wide=True))
    rows = []
    for corpus,name in CORPORA.items():
        for key,s in summaries[corpus]['systems'].items():
            rows.append((name,key.replace('_',' '),s['utterances'],f"{s['mean_utterance_hz']:.3f}",pct(s['scores']['phone_20ms']['f1']),pct(s['scores']['word_20ms']['f1'])))
    part += r.detail('All 14 recorded policies: counts, rates, phone and word F1', r.table(['Corpus','Policy','Audio n','Tokens/s','Phone F1 (%)','Word F1 (%)'], rows,
        'All recorded policies, including the other ASR and task seeds. Additional policies use 342 CREMA-D utterances; core policies use 1,431. '
        'Reference-based scores exclude failed alignments. The fifteenth checkpoint identity is the frozen identity control, used for determinism checks.', wide=True))
    body += r.section('checkpoints', '5 · Cross-task agreement and checkpoint selection', part)

    part = ('<p>Figures 3–5 share the text “Out of nowhere you get blindsided?” from one speaker. The default utterance was selected by median duration, '
        'then paired with its whisper and laughing renditions. Figure 6 was selected by the median longest aligned gap among qualifying utterances. '
        'Selection did not optimize model agreement.</p><p>ASR emits 26/31/32 tokens for default/whisper/laughing; emotion seed 3408 emits 14/15/12. '
        'Both policies respond to acoustic rendition with lexical content held fixed. These examples illustrate variation, not a population-level style effect.</p>')
    for x in examples:
        stem = Path(x['figure']).stem
        token_text = ', '.join(f'{s}: {x["tokens"][f"emotion_{s}"]}' for s in SEEDS)
        caption = f'<b>{esc(x["uid"])}.</b> {esc(x["transcript"].replace("*", ""))} ASR: {x["tokens"]["asr_3408"]} total tokens; emotion seeds {token_text} total tokens. '
        if x['number']==4:
            caption += 'The plot zooms into an inter-word gap excerpt; token counts refer to the full utterance. Yellow shading marks the aligned 400-ms gap, not verified silence.'
        else:
            caption += 'Phone and word tiers are automatic MFA alignments. Boundary rows show the ASR parent and all three emotion seeds.'
        part += r.figure(stem, caption, f'{x["style"]} Expresso spectrogram with aligned phone and word tiers and ASR/emotion boundary rows.')
    part += ('<p>In Figure 6, the aligned gap between “contrasting” and “the” runs from 2.77 to 3.17 s. None of the four policies cuts inside it or brackets both edges within 40 ms. '
        'The gap still contains vocal energy: Praat marks about 19% voiced, and only 67% of interior RMS windows meet the 20-dB threshold. '
        'It fails the 80% quiet criterion and illustrates why an aligned blank cannot automatically be called silence. The aggregate quiet-gap results are in Table 8.</p>')
    body += r.section('examples', '6 · Qualitative boundary examples', part)

    part = ('<ul><li><b>Inference:</b> features extracted and normalized separately for each utterance; only boundary policies are batched. '
        'BF16, greedy autoregressive decisions, 64-frame local history. The implicit first segment is counted once, and compulsory endpoints are excluded from scoring.</li>'
        '<li><b>Boundary metrics:</b> one-to-one matching at ±20 ms, with ±40 ms retained in the downloads. Timestamps are frame index ×20 ms; '
        '+12.5 ms is checked separately. Exact-count grids share the encoder time lattice.</li>'
        '<li><b>Controls:</b> 32 random ASR thinnings, 32 circular shifts, an equal-count uniform grid, and the highest-confidence ASR subset at each utterance’s emotion cut count. '
        'Confidence ranks conditional logits along the original ASR trajectory, scored by parallel teacher forcing. Tiny BF16 differences affect a few near-zero logits; this is a descriptive control.</li>'
        '<li><b>Alignment:</b> MFA 3.3.9 English ARPA models; 1,425/1,431 CREMA-D and 420/420 Expresso align successfully. '
        'Nine missing dictionary words affected 49 Expresso clips. Six CMUdict entries, a documented WebMD composition, and two G2P entries remove unknown-phone spans. '
        'Pronunciations and alignment remain automatic.</li><li><b>Voicing:</b> Praat autocorrelation estimates at 10-ms intervals, with a 75–600-Hz pitch range and smoothed voicing decisions. '
        'Whisper is not assumed to have zero voicing. Neither these estimates nor vowel onsets supply syllable or emotional-information ground truth.</li>'
        '<li><b>Uncertainty:</b> 2,000 paired bootstrap samples, clustered by speaker on CREMA-D/LibriSpeech and by speaker×text on Expresso. '
        'Expresso intervals are conditional on four speakers. F1 tables pool event counts; bootstrap deltas average per-utterance differences, so their point estimates need not equal table differences.</li></ul>')
    part += hk.finding(f'<b>Validation:</b> {validation["one_to_one_cases"]:,} independent dynamic-programming matching checks, '
        f'{validation["grid_count_checks"]} grid/count checks, and {validation["determinism_checks"]} repeat/batch/frozen-parent checks passed.', ok=True)
    rows = []
    for cohort,label in [('all_aligned','All aligned'),('original_dictionary_covered','Original dictionary covered'),
                         ('without_lowest_likelihood_decile_per_style','Remove bottom likelihood decile / style')]:
        vals = [controls['expresso']['controls'][f'emotion_{s}'][cohort] for s in SEEDS]
        rows.append((label, vals[0]['n'], *[pct(avg(v['scores'][m][method]['f1'] for v in vals)) for m in ['phone_20ms','vowel_onset_20ms','voicing_20ms'] for method in ['scores','confidence_thin_scores']]))
    vals = [controls['expresso']['controls'][f'emotion_{s}']['all_aligned'] for s in SEEDS]
    rows.append(('All aligned · +12.5-ms timestamps',420,*[pct(avg(v['scores'][m][method]['f1'] for v in vals)) for m in ['phone_20ms','vowel_onset_20ms','voicing_20ms'] for method in ['offset12_5_scores','confidence_thin_offset12_5_scores']]))
    part += r.table(['Expresso subset / timing','n','Phone: emotion','Phone: confident','Vowel: emotion','Vowel: confident','Voicing: emotion','Voicing: confident'],rows,
        'Alignment and timestamp sensitivity: F1 (%) at ±20 ms, averaged across three emotion seeds. Reference-event preferences persist; absolute scores depend on timing convention.', wide=True)
    rows=[]
    for corpus,name in CORPORA.items():
        for seed in SEEDS:
            d=summaries[corpus]['systems'][f'emotion_{seed}']['confidence_thin_scores_deltas']
            rows.append((name,seed,*[ci(d[m],100) for m in ['phone_20ms','vowel_onset_20ms','voicing_20ms','word_20ms']]))
    part += r.detail('Paired confidence-control differences and 95% intervals', r.table(['Corpus','Seed','Phone Δ F1 pp','Vowel Δ F1 pp','Voicing Δ F1 pp','Word Δ F1 pp'],rows,
        'Mean per-utterance F1 differences (emotion minus confident ASR subset), in percentage points, with clustered bootstrap 95% intervals. These are exploratory comparisons without a multiple-testing adjustment.',wide=True))
    part += ('<p>The original phone analyses in the training report and manuscript used a historical frame-quantized ±1-frame convention. '
        'Their LibriSpeech/TIMIT scores should not be directly compared with this audit’s continuous-time F1. This report adds evidence from non-ASR adaptation without replacing those historical results.</p>')
    body += r.section('methods','7 · Methods, sensitivity, and limits',part)

    part = ('<div class="note"><strong>Proposal only — no manuscript edits.</strong><p>The emotion result is the most defensible compact extension. '
        'Keep the current abstract’s ASR/oracle headline and “approximately 25% fewer” wording.</p></div>')
    part += ('<p>Add a short <b>Emotion recognition and task adaptation</b> paragraph near the end of §4.2, with five rows from Table 1: '
        'native frames, fixed 10/5-Hz controls, frozen ASR policy, and emotion GRPO. Omit the 25-Hz row to save space.</p>')
    proposal = (source/'paper_revision_proposal.md').read_text()
    setup_quote = proposal.split('Proposed setup addition')[1].split('\n> ',1)[1].split('\n\n',1)[0]
    result_quote = proposal.split('Proposed results paragraph:')[1].split('\n> ',1)[1].split('\n\n',1)[0]
    part += '<h3>Proposed setup</h3><blockquote>'+esc(setup_quote)+'</blockquote><h3>Proposed results paragraph</h3><blockquote>'+esc(result_quote).replace('**Emotion recognition and task adaptation.**','<strong>Emotion recognition and task adaptation.</strong>')+'</blockquote>'
    part += ('<p>Clarify the restricted adaptation: task FiLM, the final policy block, pooler/projection/LoRA, and a classification-margin reward. '
        'The non-ASR setting is not unchanged ASR NLL training. If space is tight, retain the 94% containment result and leave the finer vowel/voicing interpretation in this report.</p>')
    part += ('<p>Recover roughly 105–140 words by removing repeated 100h pooling numbers, condensing the input-dependent allocation discussion, and shortening efficiency/protocol detail. '
        'Keep the TIMIT and LibriSpeech evidence. Update the conclusion to describe emotion recognition as an initial extension. Page fit still needs compilation after the author accepts a revision.</p>')
    part += ('<h3>Experiments that would strengthen the interpretation</h3><ol>'
        '<li><b>Test placement at matched counts.</b> Evaluate learned emotion cuts, random ASR thinning, and confidence thinning on the same held-out audio with the emotion decoder fixed; then allow equal-budget decoder adaptation. This tests whether the retained locations help recognition.</li>'
        '<li><b>Separate the task from the rate prior.</b> Train paired ASR/emotion adaptations with the same rate objective and compare rate–quality curves. This is needed before attributing frequency differences to the task itself.</li>'
        '<li><b>Validate pause and expressive-event references.</b> Manually check a stratified subset of quiet gaps, laughter, and whisper; assess whether adding or removing pause-edge cuts changes task quality.</li></ol>')
    part += ('<p>These follow-ups are proposed, not completed. Current evidence supports selective grouping around acoustic transitions; '
        'claims of phoneme/word discovery, syllable identity, preserved prosody, or an optimal emotion frequency remain unsupported.</p>')
    body += r.section('paper','8 · Proposed paper changes and next experiments',part)

    part = (f'<p><a class="button" href="{ASSETS}/numerical-evidence.zip">Download numerical evidence ZIP</a> '
        f'<a class="button" href="{ASSETS}/task_results.csv">Download task results CSV</a></p>')
    part += ('<p>The ZIP includes all 54 audited TEST records, exact checkpoint identifiers and hashes, unrounded task results, all aggregate boundary/reference scores, '
        'controls, uncertainty estimates, alignment QC, quiet-gap sensitivity, and original evidence hashes. Cluster paths use portable prefixes. '
        'PDFs are linked under each figure. No audio, model weights, or manuscript files are included.</p>')
    part += ('<p>Generator: <code>recipes/LibriSpeech/ASR/transformer/make_nonasr_boundary_report.py</code>. '
        'Inputs: <code>artifacts/segmenter/nonasr_boundary_audit_2026-09-12/</code>. '
        'The generator reads the retained summaries; it does not rerun inference or change the manuscript.</p>')
    part += ('<ul class="sources"><li>Expresso: <a href="https://speechbot.github.io/expresso/">official dataset page</a>, '
        '<a href="https://github.com/facebookresearch/textlesslib/tree/main/examples/expresso/dataset">dataset documentation</a>, '
        '<a href="https://arxiv.org/abs/2308.05725">dataset paper</a>. Acquisition used the '
        '<a href="https://huggingface.co/datasets/ylacombe/expresso">ylacombe read-speech mirror</a>, pinned by commit and file hashes. Expresso is CC BY-NC 4.0.</li>'
        '<li>Alignment: <a href="https://montreal-forced-aligner.readthedocs.io/en/latest/">Montreal Forced Aligner</a> 3.3.9 and English ARPA models. '
        'Pronunciation additions use <a href="https://github.com/cmusphinx/cmudict">CMUdict</a> and documented G2P outputs.</li></ul>')
    body += r.section('downloads','9 · Evidence and sources',part)
    body += (f'<footer>JSALT 2026 · Analysis 03 · {VERSION}<br><a href="{SITE}/research/" target="_top">All analyses</a> · '
        f'<a href="{SITE}/reports/segmenter-training.html" target="_top">Training report</a></footer><script>{JS}</script>')
    full,_=hk.document('Non-ASR task adaptation and speech boundaries · JSALT 2026',body,
        head_html='<meta name="description" content="Audited single-task results and shared-audio boundary analysis on CREMA-D, Expresso and LibriSpeech, with controls and proposed manuscript changes.">')
    full=full.replace('</style>',CSS+'</style>',1)
    output.write_text(full)
    meta={'version':VERSION,'tables':r.tables,'figures':r.figures,'source_sha256':sources,
          'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'html_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
          'html_bytes':output.stat().st_size,'manuscript_edited':False}
    (output.parent/'build_manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps({k:v for k,v in meta.items() if k!='source_sha256'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=DEFAULT)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    build(args.source,args.output or args.source/'html/nonasr-task-boundaries-standalone.html')
