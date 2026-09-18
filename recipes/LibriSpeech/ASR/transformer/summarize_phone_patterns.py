#!/usr/bin/env python3
"""Describe phone overlap without interpreting F1 as token identity."""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from audit_segment_units import count_matched_grid, dump, match_count
from audit_phone_patterns import OUT

SILENCE = {'', 'sil', 'sp', 'spn', '<eps>', 'h#', 'pau', 'epi'}
VOWELS = set('aa ae ah ao aw ay eh er ey ih iy ow oy uh uw ax ax-h axr ix ux'.split())
CLASS = {
    **dict.fromkeys(VOWELS, 'vowel'),
    **dict.fromkeys('b d g p t k dx q'.split(), 'stop_or_flap'),
    **dict.fromkeys('bcl dcl gcl pcl tcl kcl'.split(), 'stop_closure'),
    **dict.fromkeys('f v s z sh zh th dh hh hv'.split(), 'fricative'),
    **dict.fromkeys('ch jh'.split(), 'affricate'),
    **dict.fromkeys('m n ng em en eng nx'.split(), 'nasal'),
    **dict.fromkeys('l r w y el'.split(), 'approximant'),
}


def label(x):
    return re.sub(r'\d+$', '', x.lower())


def nearest_distance(points, refs):
    points, refs = np.asarray(points), np.asarray(refs)
    if not len(refs):
        return np.full(len(points), np.inf)
    idx = np.searchsorted(refs, points)
    left = refs[np.clip(idx-1, 0, len(refs)-1)]
    right = refs[np.clip(idx, 0, len(refs)-1)]
    return np.minimum(np.abs(points-left), np.abs(points-right))


def f1(counts):
    h,p,r=map(int,counts)
    return dict(hits=h,predicted=p,reference=r,precision=h/p if p else 0.,recall=h/r if r else 0.,f1=2*h/(p+r) if p+r else 0.)


def phone_bin(duration):
    return ('0_60ms' if duration<.060-1e-8 else '60_120ms' if duration<.120-1e-8
            else '120_200ms' if duration<.200-1e-8 else '200ms_plus')


def describe(row, frames, frame_cuts, changes, offset=0.):
    cuts = np.asarray(frame_cuts, dtype=float)*.02+offset
    cuts = cuts[(cuts>0)&(cuts<row['duration'])]
    phones = row['phones']
    starts = np.array([p[0] for p in phones])
    ends = np.array([p[1] for p in phones])
    labs = [label(p[2]) for p in phones]
    speech = np.array([p not in SILENCE for p in labs])
    refs = sorted({p[0] for p in phones if 0<p[0]<row['duration']})
    annotated_cuts=cuts[cuts<row.get('annotation_end',row['duration'])]
    metrics = {}
    for tol in (.02,.04):
        metrics[f'phone_{int(tol*1000)}ms'] = [match_count(annotated_cuts,refs,tol),len(annotated_cuts),len(refs)]
        speech_refs = [phones[i][0] for i in range(1,len(phones)) if speech[i] and speech[i-1]]
        # Restrict candidate cuts to annotated speech regions, not pause interiors.
        phone_at = np.clip(np.searchsorted(starts,annotated_cuts,side='right')-1,0,len(phones)-1)
        speech_cuts = annotated_cuts[speech[phone_at]]
        metrics[f'within_speech_{int(tol*1000)}ms'] = [match_count(speech_cuts,speech_refs,tol),len(speech_cuts),len(speech_refs)]
    interior = {}
    for margin in (.02,.04):
        lo=np.searchsorted(cuts,starts+margin+1e-8,side='right')
        hi=np.searchsorted(cuts,ends-margin-1e-8,side='left')
        interior[int(margin*1000)] = np.maximum(0,hi-lo)
    bins,classes={},{}
    for key in ('0_60ms','60_120ms','120_200ms','200ms_plus'):
        ix=np.array([speech[i] and phone_bin(ends[i]-starts[i])==key for i in range(len(phones))])
        bins[key]=[int(ix.sum()),int((interior[20][ix]>0).sum()),int((interior[40][ix]>0).sum()),int(interior[20][ix].sum())]
    for key in sorted(set(CLASS.values())|{'other'}):
        ix=np.array([speech[i] and CLASS.get(labs[i],'other')==key for i in range(len(phones))])
        classes[key]=[int(ix.sum()),int((interior[20][ix]>0).sum()),int((interior[40][ix]>0).sum()),int(interior[20][ix].sum()),float((ends-starts)[ix].sum())]
    segment_starts=np.r_[0.,cuts]
    segment_ends=np.r_[cuts,row['duration']]
    durations=segment_ends-segment_starts
    overlap=np.maximum(0.,np.minimum(segment_ends[:,None],ends[None,:])-np.maximum(segment_starts[:,None],starts[None,:]))
    speech_overlap=overlap[:,speech].sum(axis=1)
    eligible=speech_overlap>=.8*durations-1e-8
    midpoints=((starts+ends)/2)[speech]
    n_midpoints=(np.searchsorted(midpoints,segment_ends,side='left')-np.searchsorted(midpoints,segment_starts,side='left'))
    n_mid_counts=[int((eligible&(n_midpoints==k)).sum()) for k in (0,1)]
    n_mid_counts.append(int((eligible&(n_midpoints>=2)).sum()))
    # A strict whole-phone candidate must match both edges of the same phone;
    # ordinary F1 does not measure this property.
    edge_match=(np.abs(segment_starts[:,None]-starts[None,:])<=.020000001)&(np.abs(segment_ends[:,None]-ends[None,:])<=.020000001)&speech[None,:]
    whole_phone=int((eligible&edge_match.any(axis=1)).sum())
    speech_cuts_total=int(speech[np.clip(np.searchsorted(starts,cuts,side='right')-1,0,len(phones)-1)].sum())
    speech_interior_cuts20=int(interior[20][speech].sum())
    indices=np.asarray(frame_cuts,dtype=int)
    feat_change=np.asarray(changes)
    at_cuts=feat_change[indices]
    # Mid-phone novelty is checked separately, to avoid transition/pause confounds.
    far=nearest_distance(cuts, np.r_[starts,ends[-1]])>.020000001
    at_speech=np.asarray([speech[min(max(np.searchsorted(starts,t,side='right')-1,0),len(speech)-1)] for t in cuts])
    conditional_expected=0.
    frame_times=np.arange(frames)*.02
    for i in range(len(phones)):
        if not speech[i] or not interior[20][i]:
            continue
        candidates=(frame_times>starts[i]+.020000001)&(frame_times<ends[i]-.020000001)
        assert candidates.any()
        conditional_expected+=interior[20][i]*float(feat_change[candidates].mean())
    return dict(scores=metrics,phone_bins=bins,phone_classes=classes,
                tokens=len(cuts)+1,seconds=row['duration'],n_speech_phones=int(speech.sum()),
                split_phones20=int((interior[20][speech]>0).sum()),split_phones40=int((interior[40][speech]>0).sum()),
                speech_internal_cuts=speech_cuts_total,deep_within_phone_cuts20=speech_interior_cuts20,
                speech_segments=int(eligible.sum()),phone_midpoints_per_speech_segment=n_mid_counts,
                all_multiple_phone_midpoint_segments=int((n_midpoints>=2).sum()),
                whole_phone_segments20=whole_phone,segment_ms=(1000*durations).tolist(),
                novelty_cut_sum=float(at_cuts.sum()),novelty_cut_n=len(at_cuts),
                novelty_midphone_sum=float(at_cuts[far&at_speech].sum()),novelty_midphone_n=int((far&at_speech).sum()),
                novelty_midphone_conditional_expected_sum=conditional_expected)


def aggregate(rows):
    simple=('tokens','seconds','n_speech_phones','split_phones20','split_phones40','speech_internal_cuts','deep_within_phone_cuts20','speech_segments','whole_phone_segments20','novelty_cut_sum','novelty_cut_n','novelty_midphone_sum','novelty_midphone_n','novelty_midphone_conditional_expected_sum','all_multiple_phone_midpoint_segments')
    out={k:sum(r[k] for r in rows) for k in simple}
    out['utterances']=len(rows)
    out['token_rate_hz']=out['tokens']/out['seconds']
    out['speech_phone_split20_fraction']=out['split_phones20']/out['n_speech_phones']
    out['speech_phone_split40_fraction']=out['split_phones40']/out['n_speech_phones']
    out['deep_within_phone_cuts20_fraction']=out['deep_within_phone_cuts20']/max(out['speech_internal_cuts'],1)
    out['whole_phone_speech_segment20_fraction']=out['whole_phone_segments20']/max(out['speech_segments'],1)
    out['phone_midpoints_per_speech_segment']=np.sum([r['phone_midpoints_per_speech_segment'] for r in rows],axis=0).tolist()
    out['multiple_phone_midpoints_fraction']=out['phone_midpoints_per_speech_segment'][2]/max(out['speech_segments'],1)
    out['all_token_multiple_phone_midpoints_fraction']=out['all_multiple_phone_midpoint_segments']/out['tokens']
    out['segment_ms_p10_p50_p90']=np.percentile([d for r in rows for d in r['segment_ms']],[10,50,90]).tolist()
    out['scores']={k:f1(np.sum([r['scores'][k] for r in rows],axis=0)) for k in rows[0]['scores']}
    for name in ('phone_bins','phone_classes'):
        out[name]={k:np.sum([r[name][k] for r in rows],axis=0).tolist() for k in rows[0][name]}
    out['mean_novelty_at_cut']=out['novelty_cut_sum']/max(out['novelty_cut_n'],1)
    out['midphone_novelty_ratio_vs_same_phone_uniform']=out['novelty_midphone_sum']/max(out['novelty_midphone_conditional_expected_sum'],1e-12)
    return out


def summarize(out,partial=False):
    selected=json.loads((out/'selection.json').read_text())
    refs={r['uid']:r for r in selected['utterances']}
    predictions=[json.loads(line) for line in (out/'predictions.jsonl').read_text().splitlines()]
    assert len({r['uid'] for r in predictions})==len(predictions)
    if not partial:
        assert len(predictions)==len(refs),(len(predictions),len(refs))
    collections=defaultdict(list)
    paired=defaultdict(list)
    examples=[]
    for i,pred in enumerate(predictions):
        row=refs[pred['uid']]
        for key,frames in pred['predictions'].items():
            grid=[round(t/.02) for t in count_matched_grid(pred['frames'],len(frames))]
            learned=describe(row,pred['frames'],frames,pred['feature_change'])
            control=describe(row,pred['frames'],grid,pred['feature_change'])
            collections[(row['corpus'],key,'learned')].append(learned)
            collections[(row['corpus'],key,'count_matched_grid')].append(control)
            shifted_cuts=[t*.02+.0125 for t in frames if t*.02+.0125<row.get('annotation_end',row['duration'])]
            reference=sorted({p[0] for p in row['phones'] if 0<p[0]<row['duration']})
            learned['scores']['phone_20ms_offset12.5ms']=[match_count(shifted_cuts,reference,.02),len(shifted_cuts),len(reference)]
            grid_shift=[t*.02+.0125 for t in grid if t*.02+.0125<row.get('annotation_end',row['duration'])]
            control['scores']['phone_20ms_offset12.5ms']=[match_count(grid_shift,reference,.02),len(grid_shift),len(reference)]
            paired[(row['corpus'],key)].append(dict(uid=row['uid'],speaker=row['speaker'],
                learned=learned['scores']['phone_20ms'],grid=control['scores']['phone_20ms'],
                novelty_learned=[learned['novelty_cut_sum'],learned['novelty_cut_n']],novelty_grid=[control['novelty_cut_sum'],control['novelty_cut_n']]))
            if key.endswith('3407'):
                cuts=np.array(frames)*.02
                for a,b,lab in row['phones']:
                    if label(lab) in SILENCE:
                        continue
                    internal=cuts[(cuts>a+.04)&(cuts<b-.04)].tolist()
                    if internal:
                        examples.append(dict(corpus=row['corpus'],system=key,uid=row['uid'],phone=lab,start=a,end=b,interior_cuts=internal,transcript=row['transcript']))
        if (i+1)%300==0:
            print(f'summarized {i+1}/{len(predictions)}',flush=True)
    result={}
    for (corpus,key,system),rows in collections.items():
        result.setdefault(corpus,{}).setdefault(key,{})[system]=aggregate(rows)
    rng=np.random.default_rng(202609102)
    cis={}
    for (corpus,key),rows in paired.items():
        speaker_data=[]
        for sp in sorted({r['speaker'] for r in rows}):
            rs=[r for r in rows if r['speaker']==sp]
            speaker_data.append(np.sum([r['learned']+r['grid']+r['novelty_learned']+r['novelty_grid'] for r in rs],axis=0))
        arr=np.array(speaker_data)
        totals=arr[rng.integers(0,len(arr),size=(2000,len(arr)))].sum(axis=1)
        delta=200*totals[:,0]/(totals[:,1]+totals[:,2])-200*totals[:,3]/(totals[:,4]+totals[:,5])
        novelty=totals[:,6]/totals[:,7]-totals[:,8]/totals[:,9]
        cis.setdefault(corpus,{})[key]=dict(speakers=len(arr),phone_f1_gain_pp_ci95=np.percentile(delta,[2.5,97.5]).tolist(),mean_novelty_gain_ci95=np.percentile(novelty,[2.5,97.5]).tolist())
    suffix='.partial' if partial else ''
    dump(dict(protocol=selected['protocol'],completed=len(predictions),expected=len(refs),
              time_protocol='Continuous annotated phone starts, internal cuts only, 20-ms frame timestamps; +12.5-ms sensitivity. Not numerically interchangeable with historical quantized audits.',
              class_mapping=CLASS,results=result,speaker_bootstrap=cis),out/f'summary{suffix}.json')
    dump(examples,out/f'split_examples{suffix}.json')
    dump({f'{c}/{k}':v for (c,k),v in paired.items()},out/f'per_utterance_scores{suffix}.json')
    for corpus,systems in result.items():
        print(corpus)
        for key,vals in systems.items():
            s=vals['learned'];g=vals['count_matched_grid']
            print(key,dict(hz=round(s['token_rate_hz'],2),F1=round(100*s['scores']['phone_20ms']['f1'],1),grid=round(100*g['scores']['phone_20ms']['f1'],1),whole_phone=round(100*s['whole_phone_speech_segment20_fraction'],1),split20=round(100*s['speech_phone_split20_fraction'],1),split40=round(100*s['speech_phone_split40_fraction'],1),multi_midpoint=round(100*s['multiple_phone_midpoints_fraction'],1),novelty_ratio=round(s['mean_novelty_at_cut']/g['mean_novelty_at_cut'],2)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--partial',action='store_true')
    args=parser.parse_args()
    summarize(args.output,args.partial)
