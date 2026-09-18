#!/usr/bin/env python3
"""Summarize same-audio boundary placement with exact-count controls.

References are automatic MFA intervals and Praat voicing estimates, never
treated as gold prosodic labels. Per-utterance sufficient statistics are saved
for auditable aggregation and speaker/block bootstrap uncertainty.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from audit_nonasr_boundaries import OUT, SEEDS, dump, sha
from audit_segment_units import read_grid, match_count, count_matched_grid, pauses, gap_stats

SIL = {'','sil','sp','<eps>'}
VOWELS = set('aa ae ah ao aw ay eh er ey ih iy ow oy uh uw ax ix'.split())


def lab(x): return re.sub(r'\d+$','',x.lower())


def nearest(points, refs):
    points,refs=np.asarray(points),np.asarray(refs)
    if not len(refs): return np.full(len(points),np.inf)
    idx=np.searchsorted(refs,points)
    return np.minimum(abs(points-refs[np.clip(idx-1,0,len(refs)-1)]),abs(points-refs[np.clip(idx,0,len(refs)-1)]))


def metrics(counts):
    h,p,r=map(float,counts)
    return dict(hits=h,predicted=p,reference=r,precision=h/p if p else 0.,
                recall=h/r if r else 0.,f1=2*h/(p+r) if p+r else 0.)


def reference_sets(row, pred, prosody):
    duration=row['duration']
    refs={}
    if 'phones' in row:
        phones=row['phones']; words=row['words']
        refs['phone']=sorted({round(t,8) for a,b,l in phones if lab(l) not in SIL for t in (a,b) if 0<t<duration})
        refs['word']=sorted({round(t,8) for a,b,l in words if lab(l) not in SIL for t in (a,b) if 0<t<duration})
        refs['word_start']=sorted({a for a,b,l in words if lab(l) not in SIL and 0<a<duration})
        refs['vowel_center']=sorted((a+b)/2 for a,b,l in phones if lab(l) in VOWELS)
        refs['vowel_onset']=sorted(a for a,b,l in phones if lab(l) in VOWELS and a>0)
    db=20*np.log10(np.maximum(pred['acoustic']['rms'],1e-6))
    smooth=gaussian_filter1d(db,.75)
    valleys,_=find_peaks(-smooth,prominence=3.,distance=3)
    refs['rms_valley']=(valleys*.02+.0125).tolist()
    refs['voicing']=prosody['voicing_transitions']
    return refs


def score(cuts, refs, offset=0.):
    times=np.asarray(cuts)*.02+offset
    return {f'{name}_{int(tol*1000)}ms':[match_count(times,edges,tol),len(times),len(edges)]
            for name,edges in refs.items() for tol in (.02,.04)}


def phone_shape(cuts,row):
    if 'phones' not in row:return {}
    times=np.asarray(cuts)*.02
    edges=np.r_[0.,times,row['duration']]
    p=np.asarray([[a,b] for a,b,l in row['phones'] if lab(l) not in SIL])
    mid=p.mean(axis=1)
    n=np.searchsorted(mid,edges[1:],side='left')-np.searchsorted(mid,edges[:-1],side='left')
    ns=np.searchsorted(times,p[:,1]-.04000001)-np.searchsorted(times,p[:,0]+.04000001)
    return dict(phones=len(p),split_phones40=int(np.sum(ns>0)),
                tokens_phone_centers0=int(np.sum(n==0)),tokens_phone_centers1=int(np.sum(n==1)),
                tokens_phone_centers2plus=int(np.sum(n>=2)),tokens_phone_centers3plus=int(np.sum(n>=3)))


def cut_acoustics(cuts,pred,prosody):
    if not len(cuts):return dict(feature_sum=0.,feature_n=0,pitch_delta_sum=0.,pitch_delta_n=0)
    change=np.asarray(pred['feature_change'])[cuts]
    t=np.asarray(prosody['times']); idx=np.searchsorted(t,np.asarray(cuts)*.02+.0125)
    idx=np.clip(idx,0,len(t)-1)
    valid=np.asarray(prosody['f0_delta_valid'])[idx]
    delta=np.asarray(prosody['abs_f0_delta_octaves'])[idx]
    return dict(feature_sum=float(change.sum()),feature_n=len(cuts),
                pitch_delta_sum=float(delta[valid].sum()),pitch_delta_n=int(valid.sum()))


def bootstrap_mean(values, blocks, repetitions=2000):
    groups=sorted(set(blocks)); sums=np.zeros(len(groups)); ns=np.zeros(len(groups))
    for i,g in enumerate(groups):
        v=[v for v,b in zip(values,blocks) if b==g]
        sums[i]=sum(v); ns[i]=len(v)
    rng=np.random.default_rng(20260912)
    indices=rng.integers(0,len(groups),(repetitions,len(groups)))
    b=sums[indices].sum(1)/ns[indices].sum(1)
    return dict(mean=float(np.mean(values)),ci95=np.percentile(b,[2.5,97.5]).tolist(),blocks=len(groups))


def pair_stats(a,b,frames,rng,trials):
    a=np.asarray(a);b=np.asarray(b)
    out={}
    for tol in (0,1,2):
        h=match_count(a,b,tol)
        out[f'f1_{tol*20}ms']=[h,len(a),len(b)]
        out[f'b_in_a_{tol*20}ms']=[int(np.sum(nearest(b,a)<=tol)),len(b)]
        out[f'a_in_b_{tol*20}ms']=[int(np.sum(nearest(a,b)<=tol)),len(a)]
    # Rotate the sparse policy over internal frame positions. Exact count and
    # inter-boundary spacing are preserved on this circular null domain.
    null=[]
    for _ in range(trials):
        shift=int(rng.integers(1,frames-1)) if frames>2 else 0
        rotated=np.sort((b-1+shift)%(frames-1)+1)
        null.append([match_count(a,rotated,1),int(np.sum(nearest(rotated,a)<=1))])
    out['shift_null_20ms']=np.mean(null,axis=0).tolist()+[len(a),len(b)]
    out['exact_shared']=len(np.intersect1d(a,b))
    return out


def summarize_corpus(out,corpus,trials):
    base=out/corpus
    sel=json.loads((base/'selection.json').read_text())
    rows={r['uid']:dict(r) for r in sel['utterances']}
    predictions=[json.loads(l) for l in (base/'predictions.jsonl').read_text().splitlines()]
    prosody={r['uid']:r for r in (json.loads(l) for l in (base/'prosody.jsonl').read_text().splitlines())}
    confidence={r['uid']:r for r in json.loads((base/'asr_boundary_confidence.json').read_text())}
    assert len(predictions)==len(rows)==len({p['uid'] for p in predictions})
    assert set(prosody)==set(rows)
    assert all(p['selection_sha256']==sha(base/'selection.json') for p in predictions)
    quality=[]
    if corpus!='librispeech':
        index={r['uid']:r for r in json.loads((out/'alignment_index.json').read_text()) if r['corpus']==corpus}
        for uid,row in rows.items():
            path=Path(index[uid]['alignment_file'])
            if not path.exists():
                quality.append(dict(uid=uid,style=row['style'],speaker=row['speaker'],status='missing_alignment'))
                continue
            tiers=read_grid(path)
            assert abs(tiers['words'][-1][1]-row['duration'])<.021
            row.update(phones=tiers['phones'],words=tiers['words'],reference=str(path),reference_sha256=sha(path))
            unknown=[p for p in row['phones'] if lab(p[2]) in {'spn','<unk>'}]
            quality.append(dict(uid=uid,style=row['style'],speaker=row['speaker'],status='aligned',
                                unknown_phones=len(unknown),
                                phone_duration_p90=float(np.percentile([b-a for a,b,l in row['phones'] if lab(l) not in SIL],90))))
    dump(quality,base/'alignment_qc.json')
    dump(list(rows.values()),base/'references.json')
    per_system=defaultdict(list); per_pair=defaultdict(list); pause_rows=[]
    for i,pred in enumerate(predictions):
        row=rows[pred['uid']];pro=prosody[row['uid']];refs=reference_sets(row,pred,pro)
        frames=pred['frames'];asr=pred['predictions']['asr_3408']
        block=row['speaker'] if corpus!='expresso' else row['speaker']+'::'+row['text_group']
        tags=dict(uid=row['uid'],speaker=row['speaker'],style=row['style'],block=block,extra_policies=row['extra_policies'])
        for key,cuts in pred['predictions'].items():
            rng=np.random.default_rng(int(hashlib.sha256((row['uid']+'/'+key).encode()).hexdigest()[:16],16))
            grid=np.rint(np.asarray(count_matched_grid(frames,len(cuts)))/.02).astype(int).tolist()
            own_score=score(cuts,refs)
            ctrl=score(grid,refs)
            shifted=score(cuts,refs,.0125)
            # Exact expected acoustic mean under uniformly thinning ASR cuts.
            acoustic=cut_acoustics(cuts,pred,pro)
            asr_acoustic=cut_acoustics(asr,pred,pro)
            record=dict(tags,tokens=len(cuts)+1,frames=frames,seconds=row['duration'],
                mean_utterance_hz=50*(len(cuts)+1)/frames,
                segment_ms=(1000*np.diff(np.r_[0.,np.asarray(cuts)*.02,row['duration']])).tolist(),
                scores=own_score,grid_scores=ctrl,offset12_5_scores=shifted,
                acoustics=acoustic,asr_acoustics=asr_acoustic,phone_shape=phone_shape(cuts,row),
                voiced_fraction=float(np.mean(pro['voiced_smoothed'])))
            if key.startswith('emotion_') and len(cuts)<=len(asr):
                conf=confidence[row['uid']]
                assert conf['asr_cuts']==asr
                ranked=sorted(zip(conf['logits_at_cuts'],asr),key=lambda pair:(-pair[0],pair[1]))
                top=sorted(t for _,t in ranked[:len(cuts)])
                record['confidence_thin_scores']=score(top,refs)
                record['confidence_thin_offset12_5_scores']=score(top,refs,.0125)
                record['confidence_thin_acoustics']=cut_acoustics(top,pred,pro)
                record['confidence_thin_agreement20']=[match_count(cuts,top,1),len(cuts),len(top)]
                record['confidence_thin_cuts']=top
                thins=[]
                for _ in range(trials):
                    thin=sorted(rng.choice(asr,size=len(cuts),replace=False).tolist())
                    thins.append(score(thin,refs))
                record['asr_thin_scores']={k:np.mean([x[k] for x in thins],axis=0).tolist() for k in own_score}
                # Compare retained versus discarded parent cuts by local feature
                # change quartile, calculated within each utterance.
                retained=nearest(asr,cuts)<=1
                change=np.asarray(pred['feature_change'])[asr]
                if len(asr)>=4:
                    low,high=np.percentile(change,[25,75])
                    record['asr_retention_feature_quartiles']=[int(np.sum(retained&(change<=low))),int(np.sum(change<=low)),
                                                               int(np.sum(retained&(change>=high))),int(np.sum(change>=high))]
            per_system[key].append(record)
            if 'words' in row:
                for gap in pauses(row['words']):
                    g=gap_stats((np.asarray(cuts)*.02).tolist(),row['duration'],gap)
                    # Strict single-token isolation means the SAME segment's
                    # two edges both match the pause edges within 40 ms.
                    edges=np.r_[0.,np.asarray(cuts)*.02,row['duration']]
                    g['isolated_single_token40']=bool(np.any((abs(edges[:-1]-gap[0])<=.040000001)&(abs(edges[1:]-gap[1])<=.040000001)))
                    pause_rows.append(dict(tags,system=key,start=gap[0],end=gap[1],left_word=gap[2],right_word=gap[3],**g))
        keys=list(pred['predictions'])
        pairs=[('asr_3408',k) for k in keys if k!='asr_3408']
        pairs += [(f'emotion_{a}',f'emotion_{b}') for a,b in [(3407,3408),(3407,3409),(3408,3409)]]
        for a,b in pairs:
            rng=np.random.default_rng(int(hashlib.sha256((row['uid']+'/'+a+'/'+b).encode()).hexdigest()[:16],16))
            per_pair[a+'/'+b].append(dict(tags,**pair_stats(pred['predictions'][a],pred['predictions'][b],frames,rng,trials)))
        if (i+1)%300==0:print(corpus,'summarized',i+1,'/',len(predictions),flush=True)
    dump(dict(per_system),base/'per_utterance_statistics.json')
    dump(dict(per_pair),base/'per_utterance_pairs.json')
    dump(pause_rows,base/'pause_statistics.json')
    result=dict(corpus=corpus,utterances=len(rows),aligned_utterances=sum('phones' in r for r in rows.values()),
                speakers=len({r['speaker'] for r in rows.values()}),systems={},pairs={})
    for key,rs in per_system.items():
        summary=dict(utterances=len(rs),tokens=sum(r['tokens'] for r in rs),
                     mean_utterance_hz=float(np.mean([r['mean_utterance_hz'] for r in rs])),
                     pooled_tokens_per_second=sum(r['tokens'] for r in rs)/sum(r['seconds'] for r in rs),
                     segment_ms_p10_p50_p90=np.percentile([d for r in rs for d in r['segment_ms']],[10,50,90]).tolist())
        for kind in ['scores','grid_scores','offset12_5_scores','asr_thin_scores','confidence_thin_scores','confidence_thin_offset12_5_scores']:
            names=sorted({k for r in rs for k in r.get(kind,{})})
            summary[kind]={k:metrics(np.sum([r[kind][k] for r in rs if k in r.get(kind,{})],axis=0)) for k in names}
        shapes=[r['phone_shape'] for r in rs if r['phone_shape']]
        summary['phone_shape']={k:sum(r[k] for r in shapes) for k in shapes[0]} if shapes else {}
        for control in ['grid_scores','asr_thin_scores','confidence_thin_scores']:
            summary[control+'_deltas']={}
            for metric in ['phone_20ms','word_20ms','word_start_20ms','voicing_20ms','vowel_onset_20ms']:
                eligible=[r for r in rs if metric in r.get(control,{})]
                if eligible:
                    delta=[metrics(r['scores'][metric])['f1']-metrics(r[control][metric])['f1'] for r in eligible]
                    summary[control+'_deltas'][metric]=bootstrap_mean(delta,[r['block'] for r in eligible])
        usable=[r for r in rs if r['acoustics']['feature_n'] and r['asr_acoustics']['feature_n']]
        ratios=[(r['acoustics']['feature_sum']/r['acoustics']['feature_n'])/(r['asr_acoustics']['feature_sum']/r['asr_acoustics']['feature_n']) for r in usable]
        summary['feature_change_ratio_to_asr']=bootstrap_mean(ratios,[r['block'] for r in usable])
        q=[r['asr_retention_feature_quartiles'] for r in rs if 'asr_retention_feature_quartiles' in r]
        if q:
            z=np.sum(q,axis=0);summary['asr_retention_feature_quartiles']=dict(low=float(z[0]/z[1]),high=float(z[2]/z[3]))
        summary['styles']={}
        for style in sorted({r['style'] for r in rs}):
            group=[r for r in rs if r['style']==style]
            summary['styles'][style]=dict(n=len(group),mean_hz=float(np.mean([r['mean_utterance_hz'] for r in group])),
                                          voiced_fraction=float(np.mean([r['voiced_fraction'] for r in group])))
        pauses_k=[r for r in pause_rows if r['system']==key]
        summary['pauses']=dict(n=len(pauses_k))
        if pauses_k:
            summary['pauses'].update({k:float(np.mean([r[k] for r in pauses_k])) for k in
                ['internal_cuts','pure_tokens','mostly_silent_tokens','isolated_single_token40','onset_error_ms','offset_error_ms']})
        result['systems'][key]=summary
    for key,rs in per_pair.items():
        item=dict(utterances=len(rs))
        for tol in (0,20,40):
            item[f'f1_{tol}ms']=metrics(np.sum([r[f'f1_{tol}ms'] for r in rs],axis=0))
            for direction in ('b_in_a','a_in_b'):
                z=np.sum([r[f'{direction}_{tol}ms'] for r in rs],axis=0)
                item[f'{direction}_{tol}ms']=float(z[0]/z[1]) if z[1] else 0.
        null=np.sum([r['shift_null_20ms'] for r in rs],axis=0)
        item['shift_null_f1_20ms']=float(2*null[0]/(null[2]+null[3]))
        item['shift_null_b_in_a_20ms']=float(null[1]/null[3])
        eligible=[r for r in rs if r['b_in_a_20ms'][1]]
        item['containment_minus_shift_null']=bootstrap_mean(
            [(r['b_in_a_20ms'][0]-r['shift_null_20ms'][1])/r['b_in_a_20ms'][1] for r in eligible],
            [r['block'] for r in eligible])
        result['pairs'][key]=item
    dump(result,base/'summary.json')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--corpora',nargs='+',default=['cremad','expresso','librispeech'])
    p.add_argument('--null-trials',type=int,default=32)
    a=p.parse_args()
    results={c:summarize_corpus(a.output,c,a.null_trials) for c in a.corpora}
    dump(dict(protocol=dict(null_trials=a.null_trials,tolerances_ms=[20,40],mandatory_endpoints_excluded=True,
                           timestamp_offset_sensitivity_ms=12.5,randomization_seed='sha256(uid/system)',
                           bootstrap='2000 speaker clusters; Expresso speaker-by-text blocks, conditional on these four speakers'),
              corpora=results),a.output/'boundary_summary.json')
    for c,r in results.items():
        print(c,r['aligned_utterances'],'aligned of',r['utterances'])
        for k in ['asr_3408','emotion_3407','emotion_3408','emotion_3409']:
            s=r['systems'][k]
            print(k,round(s['mean_utterance_hz'],2),'Hz; phone P/F1',
                  [round(100*s['scores']['phone_20ms'][m],1) for m in ['precision','f1']],
                  'word F1',round(100*s['scores']['word_20ms']['f1'],1),flush=True)


if __name__=='__main__':main()
