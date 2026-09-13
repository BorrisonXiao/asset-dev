#!/usr/bin/env python3
"""Evidence checks independent of the report prose; no training or publication."""
import csv,json,hashlib
from collections import defaultdict
import numpy as np
from analyze_expresso_patterns import OLD,OUT,PRIMARY,EMOTIONS,cuts,block,f1,matches,closest_matching,warp,dump

def main():
    selection=json.loads((OUT/'probes/selection.json').read_text());meta={r['uid']:r for r in selection['utterances']}
    full={r['uid']:r for r in map(json.loads,(OUT/'probes/predictions.jsonl').read_text().splitlines())}
    single={r['uid']:r for r in map(json.loads,(OUT/'probes/batch1_sensitivity.jsonl').read_text().splitlines())}
    assert len(full)==480 and len(single)==48
    exact=0;diff=[]
    for uid,p in single.items():
        for k,a in p['predictions'].items():
            b=full[uid]['predictions'][k];same=a==b;exact+=same
            if not same:diff.append(dict(uid=uid,policy=k,symmetric_cut_difference=len(set(a)^set(b)),f1=f1(np.array(a)*.02,np.array(b)*.02)))
    changes=[]
    sources={r['source_uid'] for uid,r in meta.items() if uid in single}
    for k in selection['policies']:
        for c in selection['conditions']:
            if c=='original':continue
            base='original' if c in ['sham','envelope_compress'] else 'sham'
            full_values=[];sub_values=[]
            for uid,r in meta.items():
                if r['condition']!=c:continue
                buid=r['source_uid']+'__'+base
                a,b=full[uid],full[buid];value=f1(cuts(a,k)/r['duration_scale'],cuts(b,k));full_values.append(value)
                aa,bb=single.get(uid,a),single.get(buid,b);sub_values.append(f1(cuts(aa,k)/r['duration_scale'],cuts(bb,k)))
            changes.append(dict(policy=k,condition=c,original_mean=float(np.mean(full_values)),substituted_mean=float(np.mean(sub_values)),change=float(np.mean(sub_values)-np.mean(full_values))))
    batch=dict(tracks=len(single)*len(selection['policies']),exact_tracks=exact,changed=diff,records=len(single),source_utterances=len(sources),
      max_full_panel_f1_change=max(abs(r['change']) for r in changes),aggregate_changes=changes)
    dump(batch,OUT/'batch_sensitivity_summary.json')
    refs=json.loads((OLD/'expresso/references.json').read_text());old={r['uid']:r for r in map(json.loads,(OLD/'expresso/predictions.jsonl').read_text().splitlines())}
    assert len(refs)==420 and len(old)==420
    # Cut arrays are ordered, unique, internal; provenance hashes match fixed protocol.
    for pred in list(old.values())+list(full.values())+list(single.values()):
        for cuts_ in pred['predictions'].values():assert cuts_==sorted(set(cuts_)) and all(0<t<pred['frames'] for t in cuts_)
    digests=json.loads((OUT/'input_sha256.json').read_text())
    for name,expected in digests.items():
        path=OLD/name
        if not path.exists():path=Path(name)
        assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,(name,'input changed')
    # Matching cardinality must agree with independent assignment on retained real pairs.
    for r in refs[::7]:
        p=old[r['uid']]
        for k in PRIMARY[1:]:
            a,b=cuts(p,'asr_3408'),cuts(p,k)
            assert matches(a,b,.02)==len(closest_matching(a,b))
    mapping=json.loads((OUT/'same_text_mapped_cuts.json').read_text());assert len(mapping)==360
    for m in mapping:
        for k,v in m['policies'].items():
            assert abs(f1(v['default'],v['mapped_style'])-v['f1'])<1e-12
            reg=m['regions'];a=warp(v['default'],reg,True)[0];b=warp(a,reg)[0];assert np.allclose(b,v['default'])
    pairrows=list(csv.DictReader((OUT/'tables/same_text_pairs.csv').open()));main=[r for r in pairrows if r['offset_ms']=='0.0' and r['tol_ms']=='20.0']
    cover={}
    for k in PRIMARY:
        rr=[r for r in main if r['policy']==k]
        num=sum(int(r['default_cuts'])+int(r['style_cuts']) for r in rr);den=sum(int(r['default_total'])+int(r['style_total']) for r in rr)
        cover[k]=dict(eligible_cuts=num,total_cuts=den,fraction=num/den,both_empty=sum(int(r['den'])==0 for r in rr))
    before=json.loads((OLD/'paper_before_sha256.json').read_text())
    # The stored snapshot is a mapping of relative manuscript paths to hashes.
    paper=before.get('files',before) if isinstance(before,dict) else before
    if isinstance(paper,dict):
        checked=0
        for name,digest in paper.items():
            path=Path(name) if name.startswith('/') else ROOT/name
            if not path.exists():continue
            expected=digest.get('sha256') if isinstance(digest,dict) else digest
            if isinstance(expected,str) and len(expected)==64:
                assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,(name,'manuscript changed');checked+=1
    else:checked=0
    dump(dict(natural_utterances=420,paired_comparisons=360,controlled_inputs=480,batch_check_inputs=48,
      valid_ordered_internal_cuts=True,input_hashes_verified=len(digests),matching_cross_check=True,time_warp_inverses=True,
      same_text_coverage=cover,paper_snapshot_files_checked=checked,max_batch_aggregate_change=batch['max_full_panel_f1_change']),OUT/'validation.json')
    print(json.dumps(dict(batch=batch['max_full_panel_f1_change'],exact=exact,coverage=cover,paper=checked),indent=2))
if __name__=='__main__':
    from pathlib import Path
    from analyze_expresso_patterns import ROOT
    main()
