#!/usr/bin/env python3
"""Build a Table 3 analysis tab; explicitly label missing transcript evidence.

Use --aggregate-only to prepare a draft before the saved predictions are available.
This draft is not a completed ASR error analysis and must not be represented as one.
"""

from __future__ import annotations

import argparse
import hashlib
import html
from html.parser import HTMLParser
import json
from pathlib import Path
import statistics
import sys

from analyze_aishell_table3_errors import ARMS, DEFAULT_OUTPUT, PUBLISHED_SEEDS, SEEDS

SKILL_DIR = Path("/home/cxiao7/.codex/skills/html-report")
sys.path.insert(0, str(SKILL_DIR))
import htmlkit as hk  # noqa: E402

LABELS = {"fixed": "Fixed grid (k = 5)", "phone": "Phone oracle†",
          "character": "Character oracle†", "asset": "ASSET (Transformer)"}
PAPER = {"fixed": (9.17, .20, 10.1), "phone": (12.73, .36, 7.1),
         "character": (8.63, .07, 3.2), "asset": (7.53, .07, 7.7)}
SOURCE_URL = "https://borrisonxiao.github.io/jsalt26-downsampling/research/xlingual-24k-results-standalone.html"
PAPER_URL = "https://arxiv.org/pdf/2610.14309"
REPLICATION_TABS = (
    ("quantitative", "Quantitative"), ("examples", "Examples"),
    ("linguistics", "Linguistic patterns"),
    ("interpretation", "Interpretation"), ("evidence", "Evidence"),
)
TAB_SCRIPT = """<script>
const tabs = [...document.querySelectorAll('[role=tab]')];
function selectTab(tab) {
  for (const other of tabs) {
    const active = other === tab;
    other.setAttribute('aria-selected', String(active));
    other.tabIndex = active ? 0 : -1;
    document.getElementById(other.getAttribute('aria-controls')).hidden = !active;
  }
}
tabs.forEach((tab, index) => {
  tab.addEventListener('click', () => selectTab(tab));
  tab.addEventListener('keydown', event => {
    const target = {ArrowRight: (index + 1) % tabs.length,
      ArrowLeft: (index + tabs.length - 1) % tabs.length, Home: 0, End: tabs.length - 1}[event.key];
    if (target === undefined) return;
    event.preventDefault();
    selectTab(tabs[target]);
    tabs[target].focus();
  });
});
</script>"""


class TextParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
    def handle_data(self, data):
        self.parts.append(data)


def table(number, headers, rows, caption):
    contents = "<table><thead><tr>" + "".join(f"<th>{h}</th>" for h in headers)
    contents += "</tr></thead><tbody>"
    contents += "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    contents += "</tbody></table>"
    return hk.labeled_asset('<div class="table-scroll">' + contents + '</div>', "Table", number, caption)


def linguistic_panel(data):
    def average(values):
        return f"{statistics.mean(values):.2f} ± {statistics.stdev(values):.2f}"

    systems = data["per_seed"]
    same = "same_syllable_and_tone"
    tone = "same_syllable_different_tone"
    different = "different_syllable"

    def pronunciation_mean(arm, category):
        return statistics.mean(run["pronunciation_rates"][category] for run in systems[arm])

    body = (
        f'<p>ASSET reduces different-syllable substitutions from {pronunciation_mean("phone", different):.2f} to '
        f'{pronunciation_mean("asset", different):.2f} per 100 reference characters compared with the phone oracle. '
        f'The phone oracle has a slightly lower same-syllable, same-tone rate ({pronunciation_mean("phone", same):.2f} versus '
        f'{pronunciation_mean("asset", same):.2f}). Compared with the character oracle, ASSET’s largest substitution reduction is among same-syllable, same-tone pairs.</p>'
    )
    rows = [[LABELS[arm], *(average([run["pronunciation_rates"][key] for run in runs])
                           for key in (same, tone, different, "unscored")),
             average([run["novel_repetition_utterance_rate"] for run in runs])]
            for arm, runs in systems.items()]
    body += table(4, ["System", "Same syllable and tone", "Same syllable, different tone",
                     "Different syllable", "Unscored substitutions", "New repeats (% of utterances)"], rows,
                  "Substitution counts per 100 reference characters and repeated-text utterances per 100 utterances. Mean ± sample SD across three seeds. Pinyin categories use dictionary readings in context.")
    pairs = [pair for pair in data["top_pronunciation_pairs"]["asset"] if pair["class"] == same][:3]
    pair_text = ", ".join(f'{html.escape(pair["reference"])} → {html.escape(pair["hypothesis"])} '
                          f'({html.escape(pair["reference_pinyin"])}, {pair["count"]} edits)' for pair in pairs)
    body += (f'<p>Frequent same-tone ASSET substitutions across the three seeds include {pair_text}. '
             'Repeated sequences are rare. This measure looks for a new adjacent repetition of a two- to eight-character unit in the hypothesis.</p>')

    body += '<h3>Numbers and named expressions</h3>'
    span_names = {"number": "Numeric expression", "person": "Person name", "place": "Place name", "organization": "Organization name"}
    rows = [[label, f'{data["reference_spans_per_run"][key]:,}',
             *(average([run["span_error_rates"][key] for run in systems[arm]]) for arm in ARMS)]
            for key, label in span_names.items()]
    body += table(5, ["Reference span", "Spans per run", *(LABELS[arm] for arm in ARMS)], rows,
                  "Percentage of tagged reference spans with at least one character edit. Mean ± sample SD across seeds. Numbers use text rules and names use a curated reference lexicon.")
    body += ('<p>The character oracle has the lowest error rate on tagged numbers, places and organizations. '
             'ASSET has the lowest rate on the curated person-name set. That set is small, and the score covers exact name spelling in the listed spans.</p>')

    body += '<h3>Utterance length and speaker</h3>'
    groups = data["speaker_groups"]
    wins = sum(group["systems"]["asset"]["mean"] < group["systems"]["character"]["mean"] for group in groups)
    body += (f'<p>ASSET has the lowest CER in every length band and beats the character oracle for {wins} of '
             f'{len(groups)} test speakers. The 11–15 character group has the lowest CER for all four systems. '
             'These groups count reference characters and can differ in text content.</p>')

    def group_rows(groups):
        return [[group["group"], f'{group["utterances"]:,}',
                 *(f'{s["mean"]:.2f} ± {s["sample_sd"]:.2f}' for s in (group["systems"][arm] for arm in ARMS))]
                for group in groups]

    group_headers = ["Utterances", *(LABELS[arm] for arm in ARMS)]
    body += table(6, ["Reference length", *group_headers], group_rows(data["length_groups"]),
                  "Character-only CER (%) by number of reference characters. Mean ± sample SD across seeds.")
    body += '<details><summary>Show CER for all test speakers</summary>'
    body += table(7, ["Speaker", *group_headers], group_rows(groups),
                  "Character-only CER (%) on the same utterances per speaker. Mean ± sample SD across seeds.")
    body += '</details><h3>Examples from the tagged errors</h3><p>Seven seed-3407 examples show how the four hypotheses differ on the same reference.</p>'
    for case in data["illustrative_examples"]:
        body += (f'<article class="example"><h4>{html.escape(case["title"])}</h4>'
                 f'<p class="badge">{html.escape(case["id"])} · seed 3407</p><p>{html.escape(case["comment"])}</p>'
                 f'<p class="zh"><b>Reference</b> <span lang="zh">{html.escape(case["reference"])}</span></p>')
        for arm, (hypothesis, edits) in case["hypotheses"].items():
            body += (f'<p class="zh"><b>{LABELS[arm]}</b> <span lang="zh">{html.escape(hypothesis)}</span> '
                     f'<span class="badge">({edits} edits)</span></p>')
        body += '</article>'
    body += ('<h3>How the tags are assigned</h3><p><a href="https://pypi.org/project/pypinyin/0.55.0/">Pypinyin 0.55.0</a> '
             'assigns a syllable and tone in the context of each transcript. Polyphonic characters can have different readings in the audio. '
             'The tags describe transcript edits. The span analysis marks substitutions or deletions inside the reference span, '
             'insertions between its characters, and adjacent numeral insertions for numeric spans. '
             'The number rules and name lexicon cover a subset of expressions in the test data. '
             'Rates are descriptive across training seeds, and the illustrative cases were selected by hand.</p>')
    return body


def build_replication(output, analysis, css):
    """Render verified auxiliary-loss scores, paired counts and audited examples."""
    archived = analysis["conventions"]["archived_recipe_word_boundaries"]
    plain = analysis["conventions"]["characters_without_spaces"]
    if not archived["matches_collected_error_totals"]:
        raise ValueError("Replication CER must match all saved scorer totals")
    count = analysis["test_utterances"]
    pairs = count * len(SEEDS)

    def summary(runs, metric):
        values = [run[metric] for run in runs]
        return f"{statistics.mean(values):.2f} ± {statistics.stdev(values):.2f}"

    body = css + hk.hero(
        "AISHELL-1 Mandarin · Auxiliary-loss replication",
        "What errors do the four Mandarin ASR systems make?",
        f"ASSET has the lowest character error rate (CER). Its largest error reduction is in substitutions. "
        f"The analysis covers {count:,} test utterances in each of 12 runs across three seeds.",
    )
    linguistic_path = output / "linguistic_analysis.json"
    linguistic = json.loads(linguistic_path.read_text()) if linguistic_path.is_file() else None
    if linguistic and (linguistic["analysis_sha256"] != hashlib.sha256((output / "analysis.json").read_bytes()).hexdigest()
                       or linguistic["rate_channel"] != "aux" or linguistic["status"] != "complete"):
        raise ValueError("Linguistic evidence does not match this replication audit")
    body += '<div class="tabs" role="tablist" aria-label="Analysis views">'
    tabs = [(key, label) for key, label in REPLICATION_TABS if linguistic or key != "linguistics"]
    for index, (key, label) in enumerate(tabs):
        selected = 'true' if index == 0 else 'false'
        body += (f'<button type="button" id="tab-{key}" role="tab" aria-selected="{selected}" '
                 f'aria-controls="panel-{key}" tabindex="{0 if index == 0 else -1}">{label}</button>')
    body += '</div>'

    def panel(key, title, content, hidden=True):
        return (f'<section id="panel-{key}" role="tabpanel" aria-labelledby="tab-{key}"'
                f'{" hidden" if hidden else ""}><h2>{title}</h2>{content}</section>')

    rows = []
    for arm, (mean, sd, _) in PAPER.items():
        runs = archived["per_seed"][arm]
        rows.append([LABELS[arm], f"{mean:.2f} ± {sd:.2f}", summary(runs, "cer"),
                     *[f"{run['cer']:.2f}" for run in runs]])
    quantitative = table(1, ["System", "Paper CER (%)", "Replication CER (%)", *map(str, SEEDS)], rows,
                         "Published Table 3 and the new replication using the archived scorer. Replication seed columns show test CER. ± is sample SD across three seeds. † uses test transcripts to place boundaries.")
    rows = []
    for arm, runs in plain["per_seed"].items():
        rows.append([LABELS[arm], *(summary(runs, key) for key in (
            "cer", "sub_rate", "dele_rate", "ins_rate", "exact_match_pct", "over_50pct_cer_pct"))])
    quantitative += table(2, ["System", "CER (%)", "Substitution (%)", "Deletion (%)", "Insertion (%)",
                              "Exact utterances (%)", "Utterances above 50% CER (%)"], rows,
                          "Replication CER after removing word separators. Edit rates use reference characters as the denominator. Utterance rates use test utterances. Values are mean ± sample SD across seeds.")
    rows = []
    for arm, comparisons in plain["paired_against_asset"].items():
        totals = [sum(r[key] for r in comparisons) for key in ("improved", "regressed", "tied", "net_errors_removed")]
        rows.append([LABELS[arm], *(f"{value:,}" for value in totals)])
    quantitative += table(3, ["Baseline", "ASSET fewer errors", "ASSET more errors", "Tied", "Net edits removed"], rows,
                          f"Comparison on {pairs:,} seed–utterance pairs per baseline. The same utterance appears once per seed. Counts are descriptive and use character-only scoring.")
    body += panel("quantitative", "Error rates and paired comparisons", quantitative, hidden=False)

    cases = {case["id"]: case for case in json.loads((output / "utterance_candidates.json").read_text())}
    examples = (
        ("BAC009S0764W0132", "Similar-sounding substitutions", "The fixed grid changes three characters, and the character oracle writes 界线 in place of 界限. ASSET matches the reference."),
        ("BAC009S0765W0315", "Dropped phrase", "The phone oracle omits 之日起, the phrase that specifies when the buyback period begins."),
        ("BAC009S0764W0231", "Repeated phrase", "The phone oracle ends with 概念概念 in place of 概念股. Both ASSET and the character oracle preserve the financial term."),
        ("BAC009S0764W0356", "Changed age expression", "The fixed grid changes 二十多岁 to 二十七岁. The phone and character oracles omit part of the age expression. ASSET preserves it."),
        ("BAC009S0764W0133", "ASSET substitution", "ASSET writes 宋 for 宗 in the land-parcel expression. The three baselines match the reference."),
        ("BAC009S0764W0176", "ASSET insertion", "ASSET adds 速逐 to 骤跌. The character oracle matches the reference."),
    )
    example_html = '<p>These seed-3407 examples illustrate recurring error types and cases where ASSET loses to a baseline. Frequencies are in Tables 2 and 3.</p>'
    for uid, title, comment in examples:
        case = cases[uid]
        example_html += (f'<article class="example"><h3>{title}</h3><p class="badge">{uid} · seed {case["seed"]}</p>'
                         f'<p>{comment}</p><p class="zh"><b>Reference</b> <span lang="zh">{html.escape(case["reference"])}</span></p>')
        for arm, row in case["predictions"].items():
            example_html += (f'<p class="zh"><b>{LABELS[arm]}</b> <span lang="zh">{html.escape(row["hyp"])}</span> '
                             f'<span class="badge">({row["errors"]} edits)</span></p>')
        example_html += '</article>'
    body += panel("examples", "What the models wrote", example_html)
    if linguistic:
        body += panel("linguistics", "Pronunciation, repeated text, names and numbers", linguistic_panel(linguistic))

    subst = {arm: statistics.mean(run["sub_rate"] for run in runs) for arm, runs in plain["per_seed"].items()}
    exact = {arm: statistics.mean(run["exact_match_pct"] for run in runs) for arm, runs in plain["per_seed"].items()}
    interpretation = (
        f'<p><b>Substitutions account for the largest gain.</b> ASSET averages {subst["asset"]:.2f} substitutions per 100 reference characters, '
        f'versus {subst["character"]:.2f} for the character oracle, {subst["fixed"]:.2f} for the fixed grid and {subst["phone"]:.2f} for the phone oracle. '
        'Table 2 also shows lower insertion and deletion rates for the character oracle than for ASSET.</p>'
        f'<p><b>Residual errors still matter.</b> ASSET transcribes {exact["asset"]:.1f}% of utterances exactly on average. '
        'Frequent substitutions include 他 ↔ 她 and 城 → 成. These pairs share a Mandarin pronunciation in the relevant readings. '
        'The phone oracle also has the highest deletion and insertion rates, with examples of dropped phrases and repetition.</p>'
        '<p><b>The comparison covers complete systems.</b> The oracles use transcript-assisted boundaries, the baselines train for ten CE epochs, and ASSET uses '
        '24,000 updates including a 2,400-update warmup with the seed-matched phone decoder. '
        'The results combine differences in boundary placement, token budget and training. Transcript examples establish edit patterns and help identify cases for acoustic review.</p>'
    )
    body += panel("interpretation", "What explains the CER gap", interpretation)
    evidence = (
        f'<p>The replication used seeds {", ".join(map(str, SEEDS))} and a shared filtered test manifest. Each prediction file covers all {count:,} manifest IDs '
        'with matching reference tokens. Independent scoring reproduces the saved CER numerator and denominator for every run.</p>'
        '<p>The archived scorer inserts underscores between words before computing character edits. The character-only analysis removes these separators and computes '
        'minimum-edit alignments with diagonal, deletion, then insertion tie priority. Equally optimal alignments can redistribute edits across types.</p>'
        '<p>The new ASSET runs apply the rate penalty as an auxiliary loss. Historical Table 3 used a reward-side rate penalty. '
        'The test manifest excludes 48 utterances from the full AISHELL-1 test split. All runs use development CER for checkpoint selection.</p>'
        f'<p>Background sources include the <a href="{PAPER_URL}#page=4">ASSET paper, Table 3</a> and the '
        f'<a href="{SOURCE_URL}">original multilingual results report</a>.</p>'
    )
    body += panel("evidence", "Data and scoring", evidence)
    full, _ = hk.document("Table 3 Mandarin ASR error analysis", body + TAB_SCRIPT)
    (output / "table3-asr-errors-standalone.html").write_text(full, encoding="utf-8")


def build(args):
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.published_source.read_bytes()
    parsed = TextParser(); parsed.feed(source.decode())
    source_text = " ".join(" ".join(parsed.parts).split())
    for arm, (mean, sd, rate) in PAPER.items():
        expected = f"{mean:.2f} ± {sd:.2f}"
        seeds = " / ".join(f"{v:.2f}" for v in PUBLISHED_SEEDS[arm])
        if expected not in source_text or seeds not in source_text:
            raise ValueError(f"Published evidence changed: {arm}: {expected}, {seeds}")
    analysis_path = args.output / "analysis.json"
    if not analysis_path.exists() and not args.aggregate_only:
        raise FileNotFoundError("Collect all 12 prediction reports first, or use --aggregate-only for a clearly marked draft")
    analysis = json.loads(analysis_path.read_text()) if analysis_path.exists() else None
    complete = analysis is not None and analysis["status"] == "complete"
    data = {
        "status": ("aux_replication_complete" if complete and analysis.get("rate_channel") == "aux"
                   else "transcripts_available" if complete else "aggregate_only_draft"),
        "paper": PAPER_URL, "published_source": SOURCE_URL,
        "published_source_file": args.published_source.name,
        "published_source_sha256": hashlib.sha256(source).hexdigest(),
        "arms": {a: {"label": LABELS[a], "cer_mean": v[0], "cer_sample_sd": v[1],
                     "tokens_per_second": v[2], "seed_cers_rounded": PUBLISHED_SEEDS[a]}
                 for a, v in PAPER.items()},
        "seeds": list(SEEDS), "gpu_count": 0,
    }
    (args.output / "published_evidence.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    css = """<style>
    h1,h2{text-wrap:wrap}.wrap{max-width:1100px}.table-scroll{overflow-x:auto}
    .tabs{display:flex;gap:8px;flex-wrap:wrap;margin:24px 0 18px}
    .tabs button{font:inherit;color:var(--text-primary);background:var(--surface-1);border:1px solid var(--border-strong);padding:8px 14px;border-radius:8px;cursor:pointer}
    .tabs button[aria-selected=true]{background:var(--surface-2);border-color:var(--series-sed)}
    th,td{vertical-align:top}td:first-child{min-width:160px}
    .zh{font-family:system-ui,sans-serif;line-height:1.9;overflow-wrap:anywhere}
    .example{border-top:1px solid var(--border);padding:16px 0}.example p{margin:5px 0}
    code{overflow-wrap:anywhere}.badge{font:13px system-ui,sans-serif;color:var(--text-secondary)}
    .intro{font-size:17px}.evidence-list li{margin:8px 0}
    </style>"""
    if complete and analysis.get("rate_channel") == "aux":
        build_replication(args.output, analysis, css)
        print(json.dumps({"status": "replication_analysis_complete", "output": str(args.output)}))
        return
    body = css + hk.hero("ASR errors · Paper Table 3 · AISHELL-1 Mandarin",
                         "What errors do the four Mandarin systems make?",
                         "A comparison of fixed-grid pooling, transcript-assisted phone and character oracles, and ASSET across seeds 3407, 3408 and 3409. Character error rate (CER) is lower when recognition is better.")
    if not complete:
        body += hk.finding("<b>Draft: transcript-level analysis is blocked.</b> The published scores are available, but all twelve saved prediction reports are absent from this checkout and the original cluster rejected SSH authentication. Quantitative comparisons below use published aggregate scores. Actual substitutions, deletions, insertions and utterance examples remain unmeasured.")
    body += '<div class="tabs" role="tablist" aria-label="Analysis views">'
    for i, (key, title) in enumerate((("quantitative", "Quantitative"), ("examples", "Examples"), ("interpretation", "Interpretation"), ("evidence", "Evidence"))):
        body += f'<button type="button" id="tab-{key}" role="tab" aria-selected="{"true" if i == 0 else "false"}" aria-controls="panel-{key}" tabindex="{0 if i == 0 else -1}">{title}</button>'
    body += '</div><div id="panel-quantitative" role="tabpanel" aria-labelledby="tab-quantitative">'
    body += '<section><h2>ASSET has the lowest CER in every reported seed</h2><p class="intro">Table 1 reproduces the four systems in the paper. ASSET reduces mean CER by 1.64 percentage points versus the fixed grid (17.9% relative), 1.10 points versus the character oracle (12.7%), and 5.20 points versus the phone oracle (40.8%). These comparisons establish a reduction in total errors; they do not identify which kinds of errors disappear.</p>'
    rows = []
    for arm, (mean, sd, rate) in PAPER.items():
        rows.append([LABELS[arm], f"{rate:.1f}", f"{mean:.2f} ± {sd:.2f}", *[f"{v:.2f}" for v in PUBLISHED_SEEDS[arm]]])
    body += table(1, ["System", "Tokens/s", "CER (%)", "3407", "3408", "3409"], rows,
                  "Published AISHELL-1 test results. ± is sample SD across three seeds; seed scores are published to two decimals and rates to one decimal. † uses test transcripts to place boundaries. Values are transcribed from the linked published results report and checked at generation.")
    rows = []
    for arm in ("fixed", "phone", "character"):
        mean, _, rate = PAPER[arm]
        rows.append([LABELS[arm], f"{mean - 7.53:.2f}", f"{100*(mean-7.53)/mean:.1f}%",
                     *[f"{b-a:.2f}" for b, a in zip(PUBLISHED_SEEDS[arm], PUBLISHED_SEEDS["asset"])],
                     f"{7.7/rate:.2f}×"])
    body += table(2, ["Baseline", "CER reduction (pp)", "Relative reduction", "3407 gap", "3408 gap", "3409 gap", "ASSET/baseline token rate"], rows,
                  "Positive gaps favor ASSET. Mean gaps and relative reductions use rounded published means; paired seed gaps use rounded published seed scores. Rate ratios use one-decimal rates. These are descriptive comparisons, without a paired utterance confidence interval.")
    body += '<p>ASSET uses approximately 24% fewer audio tokens than the fixed grid, 8% more than the phone oracle, and 2.41 times as many as the character oracle. The oracle comparison therefore includes a change in both boundary placement and token budget. The learned system also has a different initialization and training budget.</p>'
    if complete:
        plain = analysis["conventions"]["characters_without_spaces"]
        rows = []
        for arm, runs in plain["per_seed"].items():
            def fmt(k):
                return f"{statistics.mean(r[k] for r in runs):.2f} ± {statistics.stdev(r[k] for r in runs):.2f}"
            rows.append([LABELS[arm], fmt("cer"), fmt("sub_rate"), fmt("dele_rate"), fmt("ins_rate"), fmt("exact_match_pct"), fmt("over_50pct_cer_pct")])
        body += table(3, ["System", "CER", "Substitution %", "Deletion %", "Insertion %", "Exact utterance %", ">50% CER utterance %"], rows,
                      "Recomputed on all 7,176 utterances per run, removing word spaces. Each value is mean ± sample SD across seeds. S/D/I use a deterministic minimum-edit alignment; ties may differ from historical alignments. This diagnostic CER is distinct from the published CER until the scoring convention is verified.")
    else:
        body += '<p><b>Not yet measured:</b> substitution/deletion/insertion rates, sentence accuracy, error concentration by speaker or utterance length, homophone confusions, repeated output, and paired improvements/regressions. A single aggregate CER cannot recover any of these quantities.</p>'
    body += '</section></div><div id="panel-examples" role="tabpanel" aria-labelledby="tab-examples" hidden><section><h2>Reference and hypothesis examples</h2>'
    if complete:
        cases = json.loads((args.output / "utterance_candidates.json").read_text())
        selected = []
        selectors = [
            ("Largest ASSET improvement over the fixed grid", lambda c: c["predictions"]["fixed"]["errors"] - c["predictions"]["asset"]["errors"]),
            ("Largest ASSET regression from the fixed grid", lambda c: c["predictions"]["asset"]["errors"] - c["predictions"]["fixed"]["errors"]),
            ("Residual ASSET error burden", lambda c: c["predictions"]["asset"]["errors"]),
        ]
        for title, key in selectors:
            case = max((c for c in cases if c["id"] not in selected), key=key)
            selected.append(case["id"])
            body += f'<article class="example"><h3>{title}</h3><p class="badge">{html.escape(case["id"])} · seed {case["seed"]} · selected by absolute character error count</p><p class="zh"><b>Reference:</b> {html.escape(case["reference"])}</p>'
            for arm, row in case["predictions"].items():
                body += f'<p class="zh"><b>{LABELS[arm]}:</b> {html.escape(row["hyp"])} <span class="badge">({row["errors"]}/{row["n"]} character errors)</span></p>'
            body += '</article>'
        body += '<p>These intentionally selected cases illustrate extremes and do not estimate prevalence. Listening and linguistic review are required before assigning causes such as homophony or reference ambiguity.</p>'
    else:
        body += '<p><b>No model transcripts are available.</b> No invented reference/hypothesis pairs are shown. The intended comparison will show identical utterance IDs across all four systems, including ASSET corrections, ASSET regressions, shared failures, and representative frequent character confusions.</p><p>A classification such as homophone substitution needs the actual reference and hypothesis, and preferably the recording. The aggregate CER and token rate alone cannot establish it.</p>'
    body += '</section></div><div id="panel-interpretation" role="tabpanel" aria-labelledby="tab-interpretation" hidden><section><h2>What the evidence establishes</h2>'
    body += '<p><b>Evidence-backed:</b> The ranking is the same in all three reported seeds: ASSET, character oracle, fixed grid, phone oracle. Every reported ASSET seed is below every seed from each baseline. This supports consistency across these training seeds, while uncertainty over test speakers and utterances remains unmeasured.</p><p><b>Evidence-backed:</b> Finer phone-oracle segmentation has worse CER than the character oracle despite more tokens (7.1 versus 3.2 tokens/s). Retaining more tokens does not guarantee lower total CER in these trained systems.</p><p><b>Inferred:</b> Units near the written character or syllable scale may be easier for this decoder to use than these MFA phone segments. This is a hypothesis. Alignment quality, the residual pooler, decoder adaptation, and training schedule are alternative explanations.</p><p><b>Open question:</b> Does ASSET mainly reduce character substitutions, dropped content, or added content? Does it fix different utterances from the oracles? Are residual errors concentrated in names, numbers, or phonetically similar characters? Saved predictions are necessary to answer those questions.</p><p>The CE baselines trained for ten epochs with dev-CER selection. ASSET followed a 24,000-update curriculum, including a 2,400-update decoder warmup initialized from its seed-matched phone-oracle checkpoint. These results compare complete trained systems; they do not isolate a causal effect of boundary placement alone.</p>'
    body += '</section></div><div id="panel-evidence" role="tabpanel" aria-labelledby="tab-evidence" hidden><section><h2>Inputs and scoring checks</h2>'
    body += f'<p>Sources: <a href="{PAPER_URL}">ASSET paper, Table 3</a> and the <a href="{SOURCE_URL}">published multilingual results report</a>. This analysis uses Mandarin AISHELL-1, rather than the English systems in Table 1.</p>'
    body += '<p>The archived preparation script preserves word spaces. Its CER metric declares <code>split_tokens: True</code>; the archived SpeechBrain implementation joins words with an underscore by default before splitting characters. Whether the executed runs overrode that setting must be checked against saved predictions and run configurations. A source-code observation alone does not prove the published scores used that convention.</p>'
    body += '<p>The collector scores both whitespace-free characters and the archived underscore convention, checks all twelve prediction reports cover exactly the same 7,176 IDs and references, and reports which convention reproduces every published seed CER. Substitution, deletion and insertion counts use a documented minimum-edit tie rule. Word error counts in <code>wer_test.txt</code> are not treated as character counts.</p>'
    body += '<p>Required input layout, relative to <code>speechllm_xling_step24k/</code>:</p><ul class="evidence-list">'
    for arm, directory in ARMS.items():
        body += f'<li>{LABELS[arm]}: <code>zh/{directory}/{{3407,3408,3409}}/wer_results/wer_test.txt</code></li>'
    body += '</ul><p>Also retain each run’s <code>train_log.txt</code> and resolved configuration to confirm checkpoint selection and the exact scoring setup. This analysis needs 0 GPUs and does not require retraining or new decoding.</p>'
    body += '</section></div><script>const tabs=[...document.querySelectorAll("[role=tab]")];function selectTab(tab){tabs.forEach(t=>{const on=t===tab;t.setAttribute("aria-selected",String(on));t.tabIndex=on?0:-1;document.getElementById(t.getAttribute("aria-controls")).hidden=!on;});}tabs.forEach((t,i)=>{t.addEventListener("click",()=>selectTab(t));t.addEventListener("keydown",e=>{let j;if(e.key==="ArrowRight")j=(i+1)%tabs.length;if(e.key==="ArrowLeft")j=(i+tabs.length-1)%tabs.length;if(e.key==="Home")j=0;if(e.key==="End")j=tabs.length-1;if(j!==undefined){e.preventDefault();selectTab(tabs[j]);tabs[j].focus();}});});</script>'
    full, _ = hk.document("Table 3 Mandarin ASR error analysis", body)
    (args.output / "table3-asr-errors-standalone.html").write_text(full, encoding="utf-8")
    print(json.dumps({"status": data["status"], "output": str(args.output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--published-source", type=Path,
                        default=Path(__file__).with_name("report_sections") / "multilingual_20260919.html")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--aggregate-only", action="store_true")
    build(parser.parse_args())
