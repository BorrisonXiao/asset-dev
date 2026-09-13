#!/usr/bin/env python3
"""Timing sensitivity, within-phone allocation and concrete examples for the audit."""
import json,csv
from collections import defaultdict
import numpy as np
from analyze_expresso_patterns import OLD,OUT,PRIMARY,EMOTIONS,STYLES,lab,cls,block,cuts,closest_matching,estimate,write_csv,dump,word_phones,probes,features

def main():
    refs=json.loads((OLD/'expresso/references.json').read_text());byid={r['uid']:r for r in refs}
    preds={r['uid']:r for r in map(json.loads,(OLD/'expresso/predictions.jsonl').read_text().splitlines())}
    pro={r['uid']:r for r in map(json.loads,(OLD/'expresso/prosody.jsonl').read_text().splitlines())}
    category=[];allseeds=[];lagrows=[]
    for r in refs:
        p=preds[r['uid']];base=cuts(p,'asr_3408');f=features(r,p,pro[r['uid']]);cons={'fricative','sonorant','stop/affricate'}
        maps={k:{i for i,j in closest_matching(base,cuts(p,k))} for k in PRIMARY[1:]+EMOTIONS[:1]+EMOTIONS[2:]}
        for off in [0.,.0125]:
            for tol in [.02,.04]:
                for i,t in enumerate(base):
                    edge=min(f['phoneedges'],key=lambda e:abs(e[0]-(t+off)))
                    left,right=cls(edge[2]),cls(edge[3]);cat='other/interior'
                    if abs(edge[0]-(t+off))<=tol+1e-8:
                        if left in cons and right=='vowel':cat='consonant → vowel'
                        elif left=='vowel' and right in cons:cat='vowel → consonant'
                        elif left=='gap' and right=='vowel':cat='gap → vowel'
                    row=dict(uid=r['uid'],block=block(r),offset_ms=off*1000,tol_ms=tol*1000,category=cat)
                    for k,m in maps.items():row[k]=int(i in m)
                    category.append(row)
        # Waveform-envelope profile around retained and dropped common ASR cuts.
        for k,ids in maps.items():
            for state in [0,1]:
                indices=np.asarray([i for i in range(len(base)) if int(i in ids)==state],int)
                for lag in range(-5,6):
                    fi=np.rint(base[indices]/.02).astype(int)+lag
                    fi=fi[(fi>=0)&(fi<len(f['env']))]
                    lagrows.append(dict(uid=r['uid'],block=block(r),policy=k,retained=state,lag_ms=lag*20,
                       envelope_sum=float(f['env'][fi].sum()),n=len(fi)))
    write_csv(category,OUT/'tables/transition_sensitivity.csv');write_csv(lagrows,OUT/'tables/event_triggered_envelope.csv')
    cs={str(off):{str(tol):{c:{k:estimate([x for x in category if x['offset_ms']==off and x['tol_ms']==tol and x['category']==c],k) for k in PRIMARY[1:]+EMOTIONS[:1]+EMOTIONS[2:]} for c in ['consonant → vowel','vowel → consonant','gap → vowel','other/interior']} for tol in [20,40]} for off in [0.,12.5]}
    lags={k:{str(state):{str(lag):estimate([x for x in lagrows if x['policy']==k and x['retained']==state and x['lag_ms']==lag],'envelope_sum','n') for lag in range(-100,101,20)} for state in [0,1]} for k in PRIMARY[1:]}
    ph=list(csv.DictReader((OUT/'tables/phone_paired_changes.csv').open()))
    paired={}
    for k in PRIMARY+EMOTIONS[:1]+EMOTIONS[2:]:
        tmp=[]
        for r in ph:
            if r['style'] not in ['happy','sad','confused']:continue
            a,b=int(r['default_'+k]),int(r['style_'+k]);di,si=int(r['default_internal_'+k]),int(r['style_internal_'+k])
            dd=np.log(float(r['style_duration'])/float(r['default_duration']))
            tmp.append(dict(block=r['block'],same_allocation=int(a==b),same_present=int(a>0 and b>0),
                twice_min=2*min(a,b),total=a+b,has_extra_internal=int(si>di),has_lost_internal=int(si<di),
                delta=b-a,internal_delta=si-di,cls=r['phone_class'],duration_bin='longer ≥25%' if dd>=np.log(1.25) else 'shorter ≥20%' if dd<=np.log(.8) else 'similar duration',
                min_duration=min(float(r['default_duration']),float(r['style_duration']))))
        paired[k]=dict(same_allocation=estimate(tmp,'same_allocation'),phone_allocation_dice=estimate(tmp,'twice_min','total'),
          internal_change={d:{m:estimate([r for r in tmp if r['duration_bin']==d and r['min_duration']>=.04],m) for m in ['delta','internal_delta','has_extra_internal','has_lost_internal']} for d in ['longer ≥25%','similar duration','shorter ≥20%']})
    # Select a short near-typical-retention same-audio case, and a near-mean same-text case.
    chosen=['ex03_sad_00134','ex01_default_00264']
    # A contrast with a substantial word gap; use existing acoustic quietness audit to qualify it.
    candidates=[]
    for r in refs:
        if r['style']!='default':continue
        words=word_phones(r)
        for a,b in zip(words,words[1:]):
            gap=b['start']-a['end']
            if gap>=.25:
                rms=np.asarray(preds[r['uid']]['acoustic']['rms']);tm=np.arange(len(rms))*.02+.0125
                inside=rms[(tm>=a['end']+.04)&(tm<=b['start']-.04)]
                if len(inside):
                    db=20*np.log10(max(float(np.mean(inside)),1e-8)/max(float(np.percentile(rms,75)),1e-8))
                    if db< -20:candidates.append((abs(gap-.5),r['uid'],a['label'],b['label'],a['end'],b['start'],db))
    quiet=sorted(candidates)[:1]
    if quiet:chosen.append(quiet[0][1])
    examples=[]
    for uid in chosen:
        r=byid[uid];p=preds[uid];ft=features(r,p,pro[uid]);ev=[]
        for t in cuts(p,'asr_3408'):
            edge=min(ft['phoneedges'],key=lambda e:abs(e[0]-t));word=next((x[2] for x in r['words'] if x[0]<=t<x[1] and x[2]),'[gap]')
            ev.append(dict(t=float(t),word=word,nearest_phone_transition=edge[2]+' → '+edge[3],edge_distance_ms=abs(edge[0]-t)*1000,
                decisions={k:[float(x) for x in cuts(p,k) if abs(x-t)<=.0200001] for k in PRIMARY},rms_slope=float(np.interp(t,ft['t'],ft['slope']))))
        examples.append(dict(uid=uid,transcript=r['transcript'],duration=r['duration'],events=ev,words=r['words'],phones=r['phones'],predictions=p['predictions']))
    # Same-text word-by-word token count and phonetic placement for all seven renditions.
    group='ex01::FOR THE FRESH FACE CONTEST';gr=[r for r in refs if block(r)==group]
    wordrows=[]
    for r in sorted(gr,key=lambda r:STYLES.index(r['style'])):
        for wi,w in enumerate(word_phones(r)):
            ct=cuts(preds[r['uid']],'emotion_3408');inside=ct[(ct>=w['start'])&(ct<w['end'])]
            wordrows.append(dict(uid=r['uid'],style=r['style'],word_index=wi,word=w['label'],duration=w['end']-w['start'],cuts=inside.tolist(),phones=w['phones']))
    dump(dict(examples=examples,same_text_group=group,same_text_words=wordrows,quiet_gap=quiet,
      selection='Same-audio: short utterance with parent retention near panel median and at least one non-nested count cut. Same-text: illustrative group with mean emotion phone-warp agreement near the panel mean (0.72). Quiet gap: default rendition with word-gap duration nearest 0.5s, gap RMS at least20dB below utterance75th-percentile RMS.'),OUT/'selected_examples.json')
    dump(dict(transitions=cs,envelope_profiles=lags,phone_allocation=paired),OUT/'extended_summary.json')
    probe=probes()
    if probe is not None:dump(probe,OUT/'probe_summary.json')
    print('Extended analysis complete')
if __name__=='__main__':main()
