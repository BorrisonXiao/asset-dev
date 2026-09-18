#!/usr/bin/env python3
"""Create review figures and a concise evidence note from the completed audit."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from audit_phone_patterns import OUT
from summarize_phone_patterns import SILENCE, label


def main():
    x=json.loads((OUT/'summary.json').read_text())
    assert x['completed']==x['expected']==1709
    groups=[('test-clean','100h'),('test-clean','960h'),('test-other','100h'),('test-other','960h'),('timit-test','100h')]
    labels=['LS clean\n100h','LS clean\n960h','LS other\n100h','LS other\n960h','TIMIT\n100h']
    seeds=['3407','3408','3409']
    def rows(c,s,system='learned'):
        return [x['results'][c][f'{s}_{seed}'][system] for seed in seeds]
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    positions=np.arange(len(groups))
    for system,shift,color,name in [('learned',-.18,'#186a8c','ASSET'),('count_matched_grid',.18,'#a2a2a2','Same-count grid')]:
        vals=np.array([[100*r['scores']['phone_20ms']['f1'] for r in rows(c,s,system)] for c,s in groups])
        axes[0].bar(positions+shift,vals.mean(axis=1),width=.34,yerr=vals.std(axis=1,ddof=1),color=color,label=name,capsize=3)
    axes[0].set(xticks=positions,xticklabels=labels,ylabel='Phone-boundary F1 (%) at ±20 ms',ylim=(0,85),title='(a) Phone transitions, with token count controlled')
    axes[0].legend(frameon=False,fontsize=9)
    bins=['0_60ms','60_120ms','120_200ms','200ms_plus']
    for corpus,scale,color,style in [('test-clean','960h','#186a8c','-'),('test-other','960h','#a2552e','-'),('timit-test','100h','#75559b','-')]:
        for system,ls in [('learned',style),('count_matched_grid','--')]:
            vals=np.array([[100*r['phone_bins'][b][2]/max(r['phone_bins'][b][0],1) for b in bins] for r in rows(corpus,scale,system)])
            axes[1].plot(range(4),vals.mean(axis=0),marker='o',ls=ls,color=color,label=f'{corpus}, {scale}' if system=='learned' else None)
    axes[1].set(xticks=range(4),xticklabels=['<60','60–120','120–200','≥200'],xlabel='Annotated phone duration (ms)',ylabel='Phones with a cut >40 ms from both edges (%)',ylim=(-2,102),title='(b) Subdivision depends strongly on duration')
    axes[1].legend(frameon=False,fontsize=9)
    axes[1].text(.97,.05,'Solid: ASSET\nDashed: same-count grid',transform=axes[1].transAxes,
                 ha='right',va='bottom',fontsize=9,bbox=dict(facecolor='white',edgecolor='none',alpha=.9))
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.18)
        ax.set_axisbelow(True)
    fig.savefig(OUT/'patterns.pdf')
    fig.savefig(OUT/'patterns.png',dpi=160)
    plt.close(fig)
    selected=json.loads((OUT/'selection.json').read_text())
    ref={r['uid']:r for r in selected['utterances']}
    pred={r['uid']:r for r in (json.loads(line) for line in (OUT/'predictions.jsonl').read_text().splitlines())}
    paper_evidence={}
    for timing_offset in (0.,.0125):
        evidence=[]
        for seed in seeds:
            nphone=nsplit=ntoken=nmulti=0
            for uid,row in ref.items():
                if row['corpus']=='timit-test':
                    continue
                cuts=np.asarray(pred[uid]['predictions'][f'960h_{seed}'])*.02+timing_offset
                cuts=cuts[cuts<row['duration']]
                phones=np.array([[a,b] for a,b,lab in row['phones'] if label(lab) not in SILENCE])
                deep=(cuts[:,None]>phones[None,:,0]+.040000001)&(cuts[:,None]<phones[None,:,1]-.040000001)
                mids=phones.mean(axis=1)
                edges=np.r_[0.,cuts,row['duration']]
                midpoint_counts=np.diff(np.searchsorted(mids,edges,side='left'))
                nphone+=len(phones);nsplit+=int(deep.any(axis=0).sum())
                ntoken+=len(edges)-1;nmulti+=int((midpoint_counts>=2).sum())
            evidence.append(dict(seed=seed,speech_phones=nphone,tokens=ntoken,phones_with_deep_cut=nsplit,
                                 tokens_spanning_multiple_phone_centers=nmulti,
                                 phones_with_cut_over40ms_inside_pct=100*nsplit/nphone,
                                 tokens_spanning_multiple_phone_centers_pct=100*nmulti/ntoken))
        paper_evidence[f'offset_{1000*timing_offset:g}ms']=dict(per_seed=evidence,
            mean={key:float(np.mean([v[key] for v in evidence])) for key in
                  ('phones_with_cut_over40ms_inside_pct','tokens_spanning_multiple_phone_centers_pct')})
    paper_evidence['scope']='365 LibriSpeech test utterances, 73 speakers, three 960h checkpoints. Paper uses frame-index*20ms timestamps; +12.5ms is a sensitivity check.'
    (OUT/'paper_evidence.json').write_text(json.dumps(paper_evidence,indent=2)+'\n')
    candidates=json.loads((OUT/'split_examples.json').read_text())
    examples=[]
    for corpus,scale in [('test-clean','960h'),('timit-test','100h')]:
        eligible=sorted((r for r in candidates if r['corpus']==corpus and r['system']==f'{scale}_3407' and r['end']-r['start']>=.12),key=lambda r:(r['end']-r['start'],r['uid'],r['start']))
        examples.append(eligible[len(eligible)//2])
    fig,axes=plt.subplots(2,1,figsize=(11,4.8),layout='constrained')
    for ax,ex in zip(axes,examples):
        row=ref[ex['uid']];p=pred[ex['uid']];scale=ex['system'].split('_')[0]
        left=max(0.,ex['start']-.25);right=min(row['duration'],ex['end']+.25)
        for a,b,lab in row['phones']:
            if b<=left or a>=right:continue
            a1,b1=max(a,left),min(b,right)
            ax.broken_barh([(a1,b1-a1)],(3.2,.55),facecolors='#dde7ec',edgecolors='white')
            if b1-a1>.025:
                ax.text((a1+b1)/2,3.47,lab,ha='center',va='center',fontsize=9)
            ax.axvline(a,color='#b5b5b5',lw=.6,zorder=0)
        for i,seed in enumerate(seeds):
            cuts=np.array(p['predictions'][f'{scale}_{seed}'])*.02
            cuts=cuts[(cuts>=left)&(cuts<=right)]
            ax.vlines(cuts,2.4-i*.8,2.9-i*.8,color='#186a8c',lw=1.5)
        ax.axvspan(ex['start']+.04,ex['end']-.04,color='#e5b54e',alpha=.25,zorder=0)
        ax.set(xlim=(left,right),ylim=(.3,4),yticks=[.9,1.7,2.5,3.47],yticklabels=['seed 3409','seed 3408','seed 3407','phones'],xlabel='Time in utterance (s)',title=f"{ex['corpus']} · {ex['uid']} · {scale}-trained policy")
        ax.spines[['top','right','left']].set_visible(False)
    fig.savefig(OUT/'examples.pdf')
    fig.savefig(OUT/'examples.png',dpi=160)
    plt.close(fig)
    dump_examples=dict(selection='For each corpus, select the median-duration phone among seed-3407 phones lasting at least 120 ms with a cut more than 40 ms from both ends. These are illustrations, not prevalence estimates.',examples=examples)
    (OUT/'illustration_selection.json').write_text(json.dumps(dump_examples,indent=2)+'\n')
    lines=['# Segmentation structure beyond phone-boundary F1','',
           'Completed inference-only audit: **1,709 utterances from 241 speakers**. LibriSpeech: five utterances per speaker, 200 test-clean + 165 test-other; TIMIT: all 1,344 TEST utterances excluding SA prompts. Three 100h checkpoints on both corpora; three 960h checkpoints on LibriSpeech. All statistics use stored cuts, not a new trained model.','',
           '## Descriptive results','',
           'Percentages below are means across the three seeds; F1 includes sample SD. Endpoints are excluded, references use continuous annotation times, and the grid matches the learned number of cuts in each utterance. Historical frame-quantized F1 values in the paper are a different protocol.','',
           '| Corpus / training | F1 ±20 ms | Same-count grid F1 | Phones split >40 ms inside | Speech tokens spanning ≥2 phone centers | Whole-phone speech tokens ±20 ms |','|---|---:|---:|---:|---:|---:|']
    for c,s in groups:
        vals=rows(c,s);grid=rows(c,s,'count_matched_grid')
        f=np.array([100*r['scores']['phone_20ms']['f1'] for r in vals])
        mean=lambda key:100*np.mean([r[key] for r in vals])
        lines.append(f"| {c} / {s} | {f.mean():.1f} ± {f.std(ddof=1):.1f} | {np.mean([100*r['scores']['phone_20ms']['f1'] for r in grid]):.1f} | {mean('speech_phone_split40_fraction'):.1f} | {mean('multiple_phone_midpoints_fraction'):.1f} | {mean('whole_phone_speech_segment20_fraction'):.1f} |")
    lines+=['','## What the measurements mean','',
            '- Agreement with phones is higher than the same-count grid. That supports sensitivity to phonetic transitions, not merely a phone-like average token rate.',
            '- Some phones are subdivided more than 40 ms from both annotated edges, and some tokens span multiple phone centers. These definitions are more informative than calling every unmatched boundary a segmentation error. Both remain descriptive overlap diagnostics.',
            '- Longer phones are more often subdivided, including under the uniform control. Duration dependence by itself is not evidence of a learned advantage.',
            '- A flexible number of tokens within and across phones is an observed property. Optimizing that allocation for recognition is the training design. The current audit does not show that a particular non-phone cut causes better WER or that ASSET beats an exact-phone oracle.',
            '- Local encoder-feature changes are included in summary.json. Comparing within-phone cuts against other eligible positions in the same phone controls for phone identity, duration, and cut allocation; do not claim extra acoustic detail from a small or inconsistent conditional effect.','',
            'The paper reports the 960h LibriSpeech overlap statistics: approximately **11% of phones** have a cut more than 40 ms from both edges, and **15% of all tokens** span at least two phone centers (mean of three seed-specific corpus percentages). Exact counts and the +12.5-ms sensitivity are generated in `paper_evidence.json`. These observations support flexible granularity; they do not prove a WER benefit of the departures from phones.','',
            'The historical TIMIT frame-grid result is independently replayed in `legacy_timit_timing_recheck.json`: the new cuts give about 67.0% under that convention, close to the stored 66.9%, while the fixed-grid integer counts match exactly. Continuous-time ±20-ms scores are lower because allowing ±1 frame after reference quantization accepts a wider set of continuous reference times. Do not interchange the two protocols.','',
            'Native NIST headers revealed 21 TIMIT files with more than 20 ms of unannotated tail (maximum 1.632 s). The audit uses actual audio duration for rate and excludes unannotated tails from boundary scoring. See `timit_header_audit.json`.','',
            '## Figures and reproducibility','',
            '- `patterns.pdf` / `.png`: same-count F1 and duration-dependent phone subdivision. Bars show three-seed means with sample SD; solid lines are ASSET and dashed lines are same-count grids.',
            '- `examples.pdf` / `.png`: annotation/cut timelines from one LibriSpeech and one native TIMIT example. Gold shading excludes 40 ms at each edge of the selected phone. All three seeds are displayed; selection is recorded in illustration_selection.json.',
            '- `selection.json`, `predictions.jsonl`, `inference_metadata.json`, `summary.json`, and `per_utterance_scores.json` retain source references, checkpoint hashes, cuts, counts, and per-seed speaker-bootstrap intervals.',
            '- Entry points: `audit_phone_patterns.py prepare`, Slurm `run_phone_patterns_audit.slurm`, `summarize_phone_patterns.py`, and `report_phone_patterns.py` in the LibriSpeech Transformer recipe.','']
    (OUT/'README.md').write_text('\n'.join(lines))


if __name__=='__main__':
    main()
