#!/usr/bin/env python3
"""Where task boundaries agree and move on Expresso, from saved per-frame cuts.

No recognition scores. All summaries retain utterance/block denominators and
same-text controls. Forced alignments define correspondences, not gold units.
"""
from __future__ import annotations
import csv,json,re
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata,spearmanr
ROOT=Path(__file__).resolve().parents[4]
OLD=ROOT/'artifacts/segmenter/nonasr_boundary_audit_2026-09-12'
OUT=ROOT/'artifacts/segmenter/expresso_boundary_patterns_2026-09-13'
PRIMARY=['asr_3408','emotion_3408','intent_3408_last','speaker_count_3408_last']
EMOTIONS=['emotion_3407','emotion_3408','emotion_3409']
STYLES=['default','happy','sad','confused','whisper','laughing','enunciated']
VOWELS=set('AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW AX IX'.split())

def dump(x,p):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def lab(x):return re.sub(r'\d+$','',x.upper())
def cls(x):
    x=lab(x)
    if x in VOWELS:return 'vowel'
    if x in ('M','N','NG','L','R','W','Y'):return 'sonorant'
    if x in ('F','V','TH','DH','S','Z','SH','ZH','HH'):return 'fricative'
    if x in ('P','B','T','D','K','G','CH','JH'):return 'stop/affricate'
    return 'gap' if x in ('','SIL','SP','<EPS>') else 'other'
def block(r):return r['speaker']+'::'+r['text_group']
def cuts(p,k):return np.asarray(p['predictions'][k],float)*.02

def matches(a,b,tol):
    a=np.sort(a);b=np.sort(b);i=j=hit=0
    while i<len(a) and j<len(b):
        if abs(a[i]-b[j])<=tol+1e-8:hit+=1;i+=1;j+=1
        elif a[i]<b[j]:i+=1
        else:j+=1
    return hit

def closest_matching(a,b,tol=.02000001):
    if not len(a) or not len(b):return []
    d=np.abs(np.asarray(a)[:,None]-np.asarray(b)[None,:]);cost=np.where(d<=tol,d-100.,1.)
    ai,bi=linear_sum_assignment(cost)
    return [(int(i),int(j)) for i,j in zip(ai,bi) if d[i,j]<=tol]

def f1(a,b,tol=.02):return 2*matches(a,b,tol)/(len(a)+len(b)) if len(a)+len(b) else 1.
def nearest(a,b):
    if not len(b):return np.full(len(a),np.inf)
    return np.min(np.abs(np.asarray(a)[:,None]-np.asarray(b)[None,:]),axis=1)

def estimate(rows,numerator='value',denominator=None,reps=2000):
    buckets=defaultdict(lambda:np.zeros(2))
    for r in rows:
        v=r[numerator]
        if v is None:continue
        buckets[r['block']]+=np.asarray([v,r[denominator] if denominator else 1.])
    if not buckets:return None
    a=np.asarray(list(buckets.values()));rng=np.random.default_rng(20260913)
    idx=rng.integers(len(a),size=(reps,len(a)));s=a[idx].sum(1)
    vals=s[:,0]/np.maximum(s[:,1],1e-12)
    return dict(mean=float(a[:,0].sum()/a[:,1].sum()),ci95=np.percentile(vals,[2.5,97.5]).tolist(),blocks=len(a),n=len(rows))

def write_csv(rows,path):
    if not rows:return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def word_phones(r):
    words=[]
    for a,b,l in r['words']:
        if not l:continue
        ps=[p for p in r['phones'] if p[2] and a-1e-5<=(p[0]+p[1])/2<b+1e-5]
        words.append(dict(start=a,end=b,label=l,phones=ps))
    return words

def features(r,p,pr):
    t=np.arange(p['frames'])*.02
    rms=np.asarray(p['acoustic']['rms']);assert len(rms)==len(t)
    env=gaussian_filter1d(rms,1.)
    env=env/max(np.percentile(env,95),1e-8)
    slope=np.gradient(env,.02)
    fchange=np.asarray(p['feature_change'])
    pp=np.asarray(pr['times']);f0=np.asarray(pr['f0_hz']);voiced=np.interp(t+.0125,pp,(f0>0).astype(float))>.5
    slope_rank=rankdata(slope)/len(t);change_rank=rankdata(fchange)/len(t)
    landmarks={}
    for name,v,prom in [('rise',slope,1.),('fall',-slope,1.),('peak',env,.12),('valley',-env,.12)]:
        ids,_=find_peaks(v,distance=4,prominence=prom)
        if name in ['rise','fall']:ids=ids[v[ids]>0]
        landmarks[name]=t[ids]+.0125
    phoneedges=[]
    for i in range(1,len(r['phones'])):
        a,b,label=r['phones'][i];prev=r['phones'][i-1][2]
        if cls(prev)=='gap' and cls(label)=='gap':continue
        phoneedges.append((a,cls(prev)+' → '+cls(label),prev,label))
    return dict(t=t,env=env,slope=slope,slope_rank=slope_rank,change_rank=change_rank,
      fchange=fchange,voiced=voiced,landmarks=landmarks,phoneedges=phoneedges)

def same_audio(refs,preds,prosodies):
    pairs=[];anchors=[];landmarks=[];utts=[];nest=[]
    for u,r in enumerate(refs):
        p=preds[r['uid']];pr=prosodies[r['uid']];ft=features(r,p,pr);base=cuts(p,'asr_3408')
        for k in PRIMARY+EMOTIONS[:1]+EMOTIONS[2:]+['intent_3408','speaker_count_3408']:
            a=cuts(p,k);utts.append(dict(uid=r['uid'],block=block(r),style=r['style'],policy=k,cuts=len(a),hz=(len(a)+1)/r['duration']))
        for i,k in enumerate(PRIMARY):
            a=cuts(p,k)
            for l in PRIMARY[i+1:]:
                b=cuts(p,l)
                for tol in [0.,.02,.04]:
                    h=matches(a,b,tol)
                    pairs.append(dict(uid=r['uid'],block=block(r),a=k,b=l,tol_ms=int(tol*1000),hits=h,den=len(a)+len(b),twice_hits=2*h,a_n=len(a),b_n=len(b),b_covered=int((nearest(b,a)<=tol+1e-8).sum())))
        maps={k:{i:j for i,j in closest_matching(base,cuts(p,k))} for k in PRIMARY[1:]+EMOTIONS[:1]+EMOTIONS[2:]}
        ce=set(maps['speaker_count_3408_last']);ee=set(maps['emotion_3408']);ie=set(maps['intent_3408_last'])
        nest.append(dict(uid=r['uid'],block=block(r),style=r['style'],count_anchors=len(ce),emotion_anchors=len(ee),intent_anchors=len(ie),parent_anchors=len(base),
            count_in_emotion=len(ce&ee),count_not_emotion=len(ce-ee),count_in_intent=len(ce&ie),emotion_in_intent=len(ee&ie),
            random_expected_count_in_emotion=len(ce)*len(ee)/len(base) if len(base) else 0.,
            random_expected_emotion_in_intent=len(ee)*len(ie)/len(base) if len(base) else 0.,count_is_sparser=len(cuts(p,'speaker_count_3408_last'))<=len(cuts(p,'emotion_3408'))))
        words=word_phones(r);phoneedges=ft['phoneedges'];edget=np.asarray([x[0] for x in phoneedges])
        for ai,t in enumerate(base):
            fi=int(round(t/.02));ei=int(np.argmin(abs(edget-t))) if len(edget) else None
            edge=phoneedges[ei] if ei is not None else (0,'none','','')
            w=next((w for w in words if w['start']<=t<w['end']),None)
            ph=next((ph for ph in r['phones'] if ph[0]<=t<ph[1]),r['phones'][-1])
            distance=abs(edge[0]-t)
            row=dict(uid=r['uid'],block=block(r),style=r['style'],t=round(float(t),5),word=w['label'] if w else '[gap]',phone=ph[2],phone_class=cls(ph[2]),
              nearest_transition=edge[1] if distance<=.02000001 else 'interior (>20 ms)',edge_distance_ms=round(distance*1000,3),
              right_phone=edge[3] if distance<=.02000001 else '',word_start=bool(w and abs(t-w['start'])<=.02),
              stress=('primary' if edge[3].endswith('1') else 'unstressed' if edge[3].endswith('0') else 'secondary') if cls(edge[3])=='vowel' and distance<=.02000001 else 'not vowel onset',
              phone_duration=ph[1]-ph[0],word_fraction=(t-w['start'])/(w['end']-w['start']) if w else None,
              envelope=float(ft['env'][fi]),slope=float(ft['slope'][fi]),slope_quartile=int(min(3,ft['slope_rank'][fi]*4)),
              change=float(ft['fchange'][fi]),change_quartile=int(min(3,ft['change_rank'][fi]*4)),voiced=bool(ft['voiced'][fi]))
            for k,m in maps.items():row[k]=int(ai in m)
            anchors.append(row)
        for k in PRIMARY+EMOTIONS[:1]+EMOTIONS[2:]:
            c=cuts(p,k)
            for event,ts in ft['landmarks'].items():
                for offset in [0.,.0125]:
                    for tol in [.02,.04]:
                        landmarks.append(dict(uid=r['uid'],block=block(r),policy=k,event=event,offset_ms=offset*1000,tol_ms=tol*1000,
                            hits=matches(c+offset,ts,tol),cuts=len(c),refs=len(ts),den=len(c)+len(ts),twice_hits=2*matches(c+offset,ts,tol)))
        if (u+1)%100==0:print('same audio',u+1,flush=True)
    write_csv(pairs,OUT/'tables/same_audio_pairs.csv');write_csv(anchors,OUT/'tables/parent_anchor_decisions.csv');write_csv(nest,OUT/'tables/nesting.csv')
    write_csv(landmarks,OUT/'tables/envelope_landmarks.csv');write_csv(utts,OUT/'tables/utterance_rates.csv')
    summary={}
    for i,k in enumerate(PRIMARY):
        for l in PRIMARY[i+1:]:
            summary[k+'|'+l]={str(t):estimate([x for x in pairs if x['a']==k and x['b']==l and x['tol_ms']==t],'twice_hits','den') for t in [0,20,40]}
    retain={}
    for field in ['nearest_transition','stress','slope_quartile','change_quartile','phone_class','voiced']:
        retain[field]={}
        for v in sorted({x[field] for x in anchors},key=str):
            rows=[x for x in anchors if x[field]==v]
            retain[field][str(v)]={k:estimate(rows,k) for k in PRIMARY[1:]+EMOTIONS[:1]+EMOTIONS[2:]}
    nesting={metric:estimate(nest,metric,den) for metric,den in [('count_in_emotion','count_anchors'),('count_not_emotion','count_anchors'),('count_in_intent','count_anchors'),('emotion_in_intent','emotion_anchors'),('random_expected_count_in_emotion','count_anchors'),('random_expected_emotion_in_intent','emotion_anchors')]}
    nesting['sparser_subset']=estimate([r for r in nest if r['count_is_sparser']],'count_not_emotion','count_anchors')
    rates={k:{s:estimate([x for x in utts if x['policy']==k and x['style']==s],'hz') for s in STYLES} for k in PRIMARY+EMOTIONS[:1]+EMOTIONS[2:]}
    envelope={k:{ev:{str(off):{str(t):estimate([x for x in landmarks if x['policy']==k and x['event']==ev and x['offset_ms']==off and x['tol_ms']==t],'twice_hits','den') for t in [20,40]} for off in [0.,12.5]} for ev in ['rise','fall','peak','valley']} for k in PRIMARY}
    return dict(pair_f1=summary,parent_retention=retain,nesting=nesting,rates=rates,envelope_f1=envelope),anchors


def regions(w1,w2,mode):
    out=[]
    for wi,(a,b) in enumerate(zip(w1,w2)):
        assert a['label']==b['label']
        if mode=='word': out.append((b['start'],b['end'],a['start'],a['end'],wi,-1,''));continue
        if [lab(p[2]) for p in a['phones']]!=[lab(p[2]) for p in b['phones']]:continue
        for pi,(pa,pb) in enumerate(zip(a['phones'],b['phones'])):
            if pa[1]<=pa[0] or pb[1]<=pb[0]:continue
            out.append((pb[0],pb[1],pa[0],pa[1],wi,pi,pa[2]))
    return out

def warp(times,regs,reverse=False):
    output=[];assignment=[]
    for t in times:
        for ri,rr in enumerate(regs):
            a,b,c,d=rr[:4]
            if reverse:a,b,c,d=c,d,a,b
            if a-1e-8<=t<b-1e-8:
                output.append(c+(t-a)/(b-a)*(d-c));assignment.append(ri);break
    return np.asarray(output),assignment

def same_text(refs,preds,prosodies):
    groups=defaultdict(dict)
    for r in refs:groups[block(r)][r['style']]=r
    rows=[];phone_rows=[];qcs=[];all_examples=[];rng=np.random.default_rng(20260913)
    policies=PRIMARY+EMOTIONS[:1]+EMOTIONS[2:]
    for gi,(bl,styles) in enumerate(sorted(groups.items())):
        d=styles['default'];wd=word_phones(d);pd=preds[d['uid']]
        for style in STYLES[1:]:
            s=styles[style];ws=word_phones(s);ps=preds[s['uid']]
            assert [w['label'] for w in wd]==[w['label'] for w in ws]
            rp=regions(wd,ws,'phone');rw=regions(wd,ws,'word')
            nidentical=sum([lab(p[2]) for p in a['phones']]==[lab(p[2]) for p in b['phones']] for a,b in zip(wd,ws))
            qcs.append(dict(block=bl,style=style,words=len(wd),matching_phone_words=nidentical,
                all_phones_match=[lab(p[2]) for p in d['phones'] if p[2]]==[lab(p[2]) for p in s['phones'] if p[2]],matched_phones=len(rp)))
            # One row for each phone corresponds exactly in the two renditions.
            for ri,(a,b,c,e,wi,pi,phone) in enumerate(rp):
                stat=dict(block=bl,style=style,default_uid=d['uid'],style_uid=s['uid'],word_index=wi,word=wd[wi]['label'],phone_index=pi,phone=phone,
                    default_start=c,default_end=e,style_start=a,style_end=b,default_duration=e-c,style_duration=b-a,
                    log_duration_ratio=float(np.log((b-a)/(e-c))),phone_class=cls(phone))
                for label,r,p,pr,left,right in [('default',d,pd,prosodies[d['uid']],c,e),('style',s,ps,prosodies[s['uid']],a,b)]:
                    rms=np.asarray(p['acoustic']['rms']);tm=np.arange(len(rms))*.02+.0125
                    vals=rms[(tm>=left)&(tm<right)];stat[label+'_relative_db']=float(20*np.log10(max(float(np.mean(vals)) if len(vals) else float(np.interp((left+right)/2,tm,rms)),1e-8)/max(float(np.percentile(rms,75)),1e-8)))
                    f0=np.asarray(pr['f0_hz']);pt=np.asarray(pr['times']);valid=(pt>=left)&(pt<right)&(f0>0)
                    stat[label+'_f0']=float(np.median(f0[valid])) if valid.any() else None
                    for k in policies:
                        ct=cuts(p,k)
                        stat[label+'_'+k]=int(np.sum((ct>=left-1e-8)&(ct<right-1e-8)))
                        stat[label+'_internal_'+k]=int(np.sum((ct>left+.02000001)&(ct<right-.02000001)))
                phone_rows.append(stat)
            ex=dict(block=bl,style=style,default_uid=d['uid'],style_uid=s['uid'],regions=[list(x) for x in rp],policies={})
            for k in policies:
                dc=cuts(pd,k);sc=cuts(ps,k)
                # Retain only default cuts within the exact same matched phone domains.
                _,di=warp(dc,rp,True);db,_=warp(warp(dc,rp,True)[0],rp)
                mapped,si=warp(sc,rp)
                # Within-phone randomization preserves the exact phone allocation/count.
                null=[]
                for _ in range(40):
                    rc=np.sort([rng.uniform(rp[i][2],rp[i][3]) for i in si])
                    null.append(f1(db,rc))
                # A clock grid with the original complete utterance cut count, then identical exclusion/map.
                grid=[]
                for _ in range(20):
                    if len(sc):
                        step=s['duration']/(len(sc)+1);gc=(np.arange(len(sc))+rng.uniform(.2,1.8))*step
                        gm,_=warp(gc,rp);grid.append(f1(db,gm))
                    else:grid.append(f1(db,[]))
                dword,_=warp(warp(dc,rw,True)[0],rw);sword,_=warp(sc,rw)
                for offset in [0.,.0125]:
                    dbase,_=warp(warp(dc+offset,rp,True)[0],rp);smap,_=warp(sc+offset,rp)
                    for tol in [.02,.04]:
                        hit=matches(dbase,smap,tol)
                        rows.append(dict(block=bl,style=style,policy=k,default_uid=d['uid'],style_uid=s['uid'],offset_ms=offset*1000,tol_ms=tol*1000,
                          f1=f1(dbase,smap,tol),hits=hit,twice_hits=2*hit,den=len(dbase)+len(smap),default_cuts=len(dbase),style_cuts=len(smap),
                          default_total=len(dc),style_total=len(sc),phone_null_f1=float(np.mean(null)) if offset==0 and tol==.02 else None,
                          grid_f1=float(np.mean(grid)) if offset==0 and tol==.02 else None,
                          word_warp_f1=f1(dword,sword,tol) if offset==0 else None,
                          duration_warp_f1=f1(dc,sc*d['duration']/s['duration'],tol) if offset==0 else None,
                          strict_phone_sequence=bool(qcs[-1]['all_phones_match'])))
                ex['policies'][k]=dict(default=db.tolist(),mapped_style=mapped.tolist(),default_regions=di,style_regions=si,
                    f1=f1(db,mapped),phone_null_f1=float(np.mean(null)))
            all_examples.append(ex)
        print('same text',gi+1,'/',len(groups),flush=True)
    write_csv(rows,OUT/'tables/same_text_pairs.csv');write_csv(phone_rows,OUT/'tables/phone_paired_changes.csv');dump(qcs,OUT/'alignment_correspondence_qc.json')
    dump(all_examples,OUT/'same_text_mapped_cuts.json')
    main=[x for x in rows if x['offset_ms']==0 and x['tol_ms']==20]
    sm={k:{s:{m:estimate([x for x in main if x['policy']==k and x['style']==s],m) for m in ['f1','phone_null_f1','grid_f1','word_warp_f1','duration_warp_f1']} for s in STYLES[1:]} for k in policies}
    qc=dict(groups=len(groups),pairs=len(qcs),words=sum(x['words'] for x in qcs),matching_phone_words=sum(x['matching_phone_words'] for x in qcs),whole_phone_sequences=sum(x['all_phones_match'] for x in qcs))
    deltas=[];lookup={(x['block'],x['style'],x['policy']):x for x in main}
    for k in EMOTIONS:
        for bl,style in {(x['block'],x['style']) for x in main}:
            e=lookup[bl,style,k];a=lookup[bl,style,'asr_3408']
            deltas.append(dict(block=bl,style=style,policy=k,value=e['f1']-a['f1'],above_null=e['f1']-e['phone_null_f1'],adjusted_delta=(e['f1']-e['phone_null_f1'])-(a['f1']-a['phone_null_f1'])))
    sm['emotion_vs_asr']={k:{field:estimate([x for x in deltas if x['policy']==k and x['style'] in ['happy','sad','confused']],field) for field in ['value','above_null','adjusted_delta']} for k in EMOTIONS}
    sm['sensitivity']={k:{str(off):{str(t):estimate([x for x in rows if x['policy']==k and x['style'] in ['happy','sad','confused'] and x['offset_ms']==off and x['tol_ms']==t],'f1') for t in [20,40]} for off in [0.,12.5]} for k in policies}
    sm['strict_sequences']={k:estimate([x for x in main if x['policy']==k and x['strict_phone_sequence'] and x['style'] in ['happy','sad','confused']],'f1') for k in policies}
    # Paired acoustic differences; collapse by block for resampling, not frame/phone iid.
    slice_results={}
    for k in PRIMARY+EMOTIONS[:1]+EMOTIONS[2:]:
        slices=[]
        for r in phone_rows:
            if r['style'] not in ['happy','sad','confused']:continue
            if min(r['default_duration'],r['style_duration'])<.04:continue
            change=r['style_'+k]-r['default_'+k]
            dd=r['log_duration_ratio'];de=r['style_relative_db']-r['default_relative_db']
            slices.append(dict(block=r['block'],duration_bin='longer ≥25%' if dd>=np.log(1.25) else 'shorter ≥20%' if dd<=np.log(.8) else 'similar duration',
              energy_bin='louder ≥3dB' if de>=3 else 'quieter ≥3dB' if de<=-3 else 'similar level',value=change,
              added=int(change>0),deleted=int(change<0),same=int(change==0),duration_change=dd,energy_change=de,
              pitch_change=12*np.log2(r['style_f0']/r['default_f0']) if r['style_f0'] and r['default_f0'] else None))
        slice_results[k]={db:{eb:{m:estimate([r for r in slices if r['duration_bin']==db and r['energy_bin']==eb],m) for m in ['value','added','deleted','same']} for eb in ['louder ≥3dB','similar level','quieter ≥3dB']} for db in ['longer ≥25%','similar duration','shorter ≥20%']}
    return dict(qc=qc,agreement=sm,paired_allocation=slice_results)


def probes():
    selection=OUT/'probes/selection.json';pp=OUT/'probes/predictions.jsonl'
    if not pp.exists():return None
    ss=json.loads(selection.read_text());meta={r['uid']:r for r in ss['utterances']}
    pred={r['uid']:r for r in map(json.loads,pp.read_text().splitlines())}
    if len(pred)!=len(meta):return dict(incomplete=True,completed=len(pred),expected=len(meta))
    old={r['uid']:r for r in map(json.loads,(OLD/'expresso/predictions.jsonl').read_text().splitlines())}
    out=[];identity=[]
    for uid,r in meta.items():
        c=r['condition'];base='original' if c in ['sham','envelope_compress','original'] else 'sham'
        b=pred[r['source_uid']+'__'+base];a=pred[uid]
        for k in ss['policies']:
            ac=cuts(a,k)/r['duration_scale'];bc=cuts(b,k)
            if c=='original':
                ref=cuts(old[r['source_uid']],k)
                identity.append(dict(uid=r['source_uid'],policy=k,exact_same=bool(np.array_equal(ac,ref)),differing_cuts=len(set(ac)^set(ref))))
            for tol in [.02,.04]:
                out.append(dict(uid=uid,block=r['speaker']+'::'+r['text_group'],source_uid=r['source_uid'],policy=k,condition=c,comparison=base,
                    tol_ms=tol*1000,f1=f1(ac,bc,tol),boundary_retention=matches(ac,bc,tol)/len(bc) if len(bc) else 1.,
                    token_ratio=(len(ac)+1)/(len(bc)+1),token_delta=len(ac)-len(bc),hz=(len(ac)+1)/r['duration']))
    write_csv(out,OUT/'tables/probe_comparisons.csv');dump(identity,OUT/'probes/original_reproduction.json')
    summary={k:{c:{str(t):{m:estimate([r for r in out if r['policy']==k and r['condition']==c and r['tol_ms']==t],m) for m in ['f1','token_ratio','token_delta']} for t in [20,40]} for c in ss['conditions']} for k in ss['policies']}
    qc=json.loads((OUT/'probes/acoustic_qc.json').read_text())
    qsum={c:{f:float(np.median([r[f] for r in qc if r['condition']==c and r[f] is not None])) for f in ['duration_ratio','f0_median_shift_semitones','f0_shift_iqr_semitones','f0_log_sd','voicing_agreement','envelope_correlation','envelope_cv']} for c in ss['conditions']}
    return dict(comparisons=summary,acoustic_qc=qsum,reproduction=dict(records=len(identity),exact=sum(r['exact_same'] for r in identity),changed=[r for r in identity if not r['exact_same']]))

if __name__=='__main__':
    refs=json.loads((OLD/'expresso/references.json').read_text())
    preds={r['uid']:r for r in map(json.loads,(OLD/'expresso/predictions.jsonl').read_text().splitlines())}
    prosodies={r['uid']:r for r in map(json.loads,(OLD/'expresso/prosody.jsonl').read_text().splitlines())}
    same,anchors=same_audio(refs,preds,prosodies);dump(same,OUT/'same_audio_summary.json')
    text=same_text(refs,preds,prosodies);dump(text,OUT/'same_text_summary.json')
    p=probes()
    if p is not None:dump(p,OUT/'probe_summary.json')
    print('Complete',flush=True)
