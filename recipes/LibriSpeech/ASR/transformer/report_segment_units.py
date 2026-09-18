#!/usr/bin/env python3
"""Generate the reviewable Markdown audit and annotated PDF from saved results."""
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from audit_segment_units import OUT, SEEDS, count_matched_grid, match_count, reference_edges


def ms(values, scale=1):
    values = [v*scale for v in values]
    return f'{statistics.mean(values):.1f} ± {statistics.stdev(values):.1f}'


def main():
    result = json.loads((OUT / 'summary.json').read_text())
    data = json.loads((OUT / 'predictions.json').read_text())
    selection = json.loads((OUT / 'selection.json').read_text())
    rows = data['utterances']
    text = ['# What do the approximately 11 Hz segments capture?', '',
            'Inference-only pilot, 2026-09-10. Transformer-AR local-64 / BiGRU, trained on all three LibriSpeech-960h splits; three dev-WER-selected checkpoints. No decoder, retraining, or reference-conditioned segmentation.', '',
            '## Sampling and interpretation', '',
            f"Selection was frozen before predictions (`selection.json`, RNG seed {selection['selection_seed']}). Twelve distinct speakers were sampled uniformly and one utterance sampled per speaker. Twelve additional utterances from distinct pause-eligible speakers were sampled, excluding the first 12 utterance IDs. These cohorts are analyzed separately. They total {sum(r['duration'] for r in rows):.2f} seconds and {sum(len(r['pauses']) for r in rows)} inter-word gaps ≥250 ms. Speakers may overlap between cohorts. The alignment scan found {selection['population_pause_utterances']}/{selection['population_utterances']} dev-clean utterances with an eligible gap.", '',
            'Reference word/phone intervals come from the existing MFA TextGrids. These are automatic forced alignments, not manually checked phonetic truth. Phone and word references include speech/silence transitions and exclude utterance endpoints. Scores use maximum ordered one-to-one matching on continuous timestamps. Primary cut time is frame index ×20 ms; a +12.5 ms sensitivity check uses the encoder receptive-field center. This protocol differs from the older quantized full-dev-clean 100h diagnostic.', '',
            'The equal-count grid has exactly the same token count as the learned policy for each utterance and seed, with uniform cuts rounded to the same 20 ms encoder-frame lattice. Fixed k=5 is a secondary control. A deterministic utterance-specific circular shift preserves learned cut count and circular spacing. Neither control is trained. A high phone-boundary score alone does not identify phonemes or demi-syllables, especially with dense boundaries and a nonzero tolerance.', '',
            '## Boundary agreement', '',
            'Values are mean ± sample SD over three model seeds. F1 is computed from pooled within-cohort counts per seed; seeds are repeated measurements of the same utterances, not independent speech samples.', '']
    for cohort, cv in result['summary'].items():
        text += [f'### {cohort} (12 utterances)', '', '| Placement | Tokens/s | Phone F1 ±20 ms | Phone F1 ±40 ms | Word F1 ±20 ms | Broadband valley F1 ±20 ms |', '|---|---:|---:|---:|---:|---:|']
        for system in ('learned','count_matched_grid','fixed_k5','circular_shift'):
            vals = [cv[str(seed)][system] for seed in SEEDS]
            text.append('| ' + system + ' | ' + ' | '.join([ms([v['token_rate_hz'] for v in vals]), *[ms([v['scores'][key]['f1'] for v in vals],100) for key in ('phone_20ms_offset0ms','phone_40ms_offset0ms','word_20ms_offset0ms','energy_valley_20ms_offset0ms')]]) + ' |')
        vals = [cv[str(seed)]['learned'] for seed in SEEDS]
        percentiles = ' / '.join(str(round(np.mean([v['segment_ms_percentiles'][i] for v in vals]))) for i in range(3))
        text += ['', f"Learned segment-duration percentiles (10/50/90; mean of seed-specific percentiles): {percentiles} ms. Among segments with ≥80% speech overlap, {ms([v['fraction_speech_segments_80pct_one_phone'] for v in vals],100)}% have ≥80% of their full duration within one reference phone. This is a span-overlap criterion, not token semantics.", '',
                 f"At +12.5 ms offset, learned/equal-count-grid phone F1 within ±20 ms is {ms([cv[str(s)]['learned']['scores']['phone_20ms_offset12.5ms']['f1'] for s in SEEDS],100)} / {ms([cv[str(s)]['count_matched_grid']['scores']['phone_20ms_offset12.5ms']['f1'] for s in SEEDS],100)}%.", '']
    cv = result['summary']['speaker_sample']
    learned_f1 = ms([cv[str(s)]['learned']['scores']['phone_20ms_offset0ms']['f1'] for s in SEEDS], 100)
    grid_f1 = ms([cv[str(s)]['count_matched_grid']['scores']['phone_20ms_offset0ms']['f1'] for s in SEEDS], 100)
    grouped_gaps = [[g for g in result['pause_details'] if g['seed']==str(seed)] for seed in SEEDS]
    uncut = [sum(g['internal_cuts']==0 for g in gs) for gs in grouped_gaps]
    pure = [sum(g['pure_tokens']>0 for g in gs) for gs in grouped_gaps]
    findings = ['## Measured findings', '',
        f'On the 12-utterance speaker sample, learned phone-boundary F1 is {learned_f1}%, versus {grid_f1}% for exactly count-matched grids on the same encoder-frame lattice. This supports a preference for phone transitions, not a one-to-one phoneme-token interpretation.', '',
        f'Of the 31 aligned inter-word gaps, {min(uncut)}–{max(uncut)} per model seed have no internal cut, and only {min(pure)}–{max(pure)} contain a segment wholly inside the gap. The model usually pools the pause with neighboring speech instead of assigning a separate silence-only token. Median gap-onset / gap-offset boundary error is ' + '; '.join(f"{seed}: {np.median([g['onset_error_ms'] for g in gs]):.0f} / {np.median([g['offset_error_ms'] for g in gs]):.0f} ms" for seed, gs in zip(SEEDS,grouped_gaps)) + '. The asymmetry is consistent with extending a preceding segment through the pause until speech resumes.', '',
        'Broadband energy-valley agreement does not improve over the equal-count grid in either cohort. That result does not test harmonic energy. No claim about demi-syllable identity is warranted.', '',
        'The unrounded continuous-grid control was initially 43.3% phone F1 on the speaker sample. Rounding it to the model’s 20 ms lattice raises it to 52.6%; the final tables use the fairer frame-lattice comparison.', '']
    text[4:4] = findings
    text[2:2] = ['**Manuscript status (2026-09-10):** The author rejected the proposed text changes; the original paper has been restored. This report remains exploratory evidence. The [follow-up plan](../../../plans/segment_interpretation_followup_2026-09-10.md) records the initialization comparison and stronger tests.', '']

    # Paired bootstrap across utterances, averaging model seeds within utterance.
    bootstrap = {}
    rng = np.random.default_rng(20260910)
    for cohort in ('speaker_sample','pause_sample'):
        subset = [r for r in rows if r['cohort']==cohort]
        differences = []
        per_utterance = []
        for row in subset:
            ref = reference_edges(row['tiers']['phones'],row['duration'])
            delta = []
            for seed in SEEDS:
                cuts = np.array(row['predictions'][str(seed)])*.02
                grid = count_matched_grid(row['frames'], len(cuts))
                delta.append(200*(match_count(cuts,ref,.02)-match_count(grid,ref,.02))/(len(cuts)+len(ref)))
            differences.append(np.mean(delta))
            per_utterance.append(dict(uid=row['uid'],phone_f1_delta_pp=float(np.mean(delta))))
        boot = np.mean(rng.choice(differences,size=(10000,len(differences)),replace=True),axis=1)
        ci = np.percentile(boot,[2.5,97.5]).tolist()
        bootstrap[cohort] = dict(mean_delta_pp=float(np.mean(differences)),percentile95_ci=ci,utterances=per_utterance)
        text += [f"Paired exploratory bootstrap, {cohort}: learned minus equal-count-grid phone F1 averages {np.mean(differences):+.1f} points per utterance (95% percentile interval [{ci[0]:+.1f}, {ci[1]:+.1f}], 10,000 utterance resamples, seeds averaged within each utterance). This macro difference differs from the pooled F1 in the table.", '']
    (OUT / 'bootstrap.json').write_text(json.dumps(bootstrap,indent=2)+'\n')
    text += ['## Inter-word pauses', '',
             'A **pure silent token** is a segment fully contained in the aligned gap. A **mostly silent token** has ≥80% of its duration inside the gap. These labels describe the pooled acoustic span; contextual WavLM features may still encode neighboring speech. Boundary errors use gap onset/offset separately. A gap can receive multiple tokens, so “gets its own token” is tested separately from “becomes exactly one token.”', '',
             '| Cohort / seed / placement | Gaps | Without internal cut | With pure silent token | With ≥80% silent token | Both edges within 40 ms | Median internal cuts |', '|---|---:|---:|---:|---:|---:|---:|']
    for cohort, cv in result['summary'].items():
        for seed in SEEDS:
            for system in ('learned','count_matched_grid'):
                p=cv[str(seed)][system]['pauses']
                text.append(f"| {cohort} / {seed} / {system} | {p['n']} | {p['without_internal_cut']} | {p['with_pure_token']} | {p['with_mostly_silent_token']} | {p['both_edges_within40ms']} | {p['median_internal_cuts']:.0f} |")
    text += ['', 'The same gaps recur across seeds; do not count them as independent observations. A uniform grid also generates pure silent tokens for long enough gaps. Evidence of a dedicated pause unit additionally needs both edges aligned and little internal fragmentation.', '',
             '## Acoustic/syllabic limitations', '',
             'Broadband RMS uses a 25 ms window and 5 ms hop, 5 ms Gaussian smoothing, and minima ≥3 dB prominent and ≥40 ms apart. These parameters were fixed before predictions. RMS does not isolate harmonic energy; this audit cannot claim harmonic-energy valley detection. Existing derived syllable targets are retained as an auxiliary score in `summary.json`, but their construction was not independently audited. Demi-syllable identity is not tested.', '',
             '## Individual utterances', '',
             '`utterance_panels.pdf` shows every selected utterance (no example filtering): broadband energy, forced word and phone intervals, and the three learned partitions. `pause_examples.pdf` zooms the first gap of the first three pause-cohort utterances in the frozen selection. The seed-3407 grid is shown in the zooms as a matched-count control.', '',
             '| Cohort | Utterance | Duration (s) | Gaps ≥250 ms | Transcript |', '|---|---|---:|---:|---|']
    for r in rows:
        text.append(f"| {r['cohort']} | {r['uid']} | {r['duration']:.2f} | {len(r['pauses'])} | {r['transcript']} |")
    text += ['', '## Reproduction and provenance', '',
             'Run `audit_segment_units.py prepare`, submit `run_segment_units_audit.slurm`, then run `report_segment_units.py`. `selection.json` stores every TextGrid SHA-256 and all reference intervals. `predictions.json` stores checkpoint paths, hashes, metadata, frame counts, cut positions, and energy. `summary.json` stores exact match counts and each gap/seed outcome; `bootstrap.json` stores per-utterance paired differences. Checkpoints are selected by minimum saved dev-clean WER, matching trainer `min_key=WER`.', '',
             'The pilot is descriptive and small. It cannot establish universal linguistic units, causality of RL changes, or that detected pauses improve recognition. No ASR scores were recomputed.']
    (OUT / 'README.md').write_text('\n'.join(text)+'\n')
    plot_all(rows)


def plot_row(row, zoom=None):
    fig, axes = plt.subplots(6,1,figsize=(12,7),sharex=True,gridspec_kw={'height_ratios':[2,1,1,1,1,1]})
    left,right = zoom or (0,row['duration'])
    axes[0].plot(row['energy']['times'],row['energy']['db'],color='#39465e',lw=.8)
    axes[0].set_ylabel('RMS (dB)')
    for ax in axes:
        for a,b,*_ in row['pauses']:
            ax.axvspan(a,b,color='#e2ba42',alpha=.25)
        ax.spines[['top','right']].set_visible(False)
    for ax,tier in zip(axes[1:3],('words','phones')):
        for i,(a,b,label) in enumerate(row['tiers'][tier]):
            if b<left or a>right:
                continue
            ax.barh(.5,b-a,left=a,height=.8,color=['#dce9f4','#edf3f8'][i%2],edgecolor='#8da6bd',lw=.5)
            if label:
                ax.text((max(a,left)+min(b,right))/2,.5,label,ha='center',va='center',fontsize=7 if zoom else 5,rotation=0 if zoom else 45)
        ax.set_ylabel(tier)
        ax.set_ylim(0,1)
        ax.set_yticks([])
    for ax,seed in zip(axes[3:],SEEDS):
        cuts=np.array(row['predictions'][str(seed)])*.02
        edges=[0,*cuts,row['duration']]
        for i,(a,b) in enumerate(zip(edges,edges[1:])):
            ax.barh(.5,b-a,left=a,height=.8,color=['#bddecf','#e1eee9'][i%2],edgecolor='#38745a',lw=.5)
        if zoom and seed==3407:
            grid=count_matched_grid(row['frames'], len(cuts))
            ax.vlines(grid,0,1,colors='#a24e51',linestyles='dashed',lw=.8)
        ax.set_ylabel(str(seed))
        ax.set_ylim(0,1)
        ax.set_yticks([])
    axes[-1].set_xlabel('Time (s); shaded: aligned inter-word gap ≥250 ms')
    axes[-1].set_xlim(left,right)
    fig.suptitle(f"{row['uid']} — {row['cohort']}\n{row['transcript']}",fontsize=9,wrap=True)
    fig.tight_layout(rect=(0,0,1,.93))
    return fig


def plot_all(rows):
    with PdfPages(OUT / 'utterance_panels.pdf') as pdf:
        for row in rows:
            fig=plot_row(row)
            pdf.savefig(fig)
            plt.close(fig)
    with PdfPages(OUT / 'pause_examples.pdf') as pdf:
        for index,row in enumerate([r for r in rows if r['cohort']=='pause_sample'][:3]):
            a,b,*_=row['pauses'][0]
            fig=plot_row(row,(max(0,a-.5),min(row['duration'],b+.5)))
            pdf.savefig(fig)
            fig.savefig(OUT / f'pause_example_{index+1}.png',dpi=140)
            plt.close(fig)


if __name__ == '__main__':
    main()
