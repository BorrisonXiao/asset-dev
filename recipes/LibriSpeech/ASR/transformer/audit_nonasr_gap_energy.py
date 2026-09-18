#!/usr/bin/env python3
"""Separate aligned inter-word blanks from acoustically quiet gaps.

A diagnostic low-energy gap has >=80% of interior RMS windows at least 20 dB
below the utterance's 90th-percentile RMS. Also report 15/25 dB sensitivity.
This is an explicit relative-energy definition, not manual silence ground truth.
"""
import json
import numpy as np
from audit_nonasr_boundaries import OUT,dump


def main():
    summaries={}
    for corpus in ['cremad','expresso','librispeech']:
        base=OUT/corpus
        pred={r['uid']:r for r in (json.loads(l) for l in (base/'predictions.jsonl').read_text().splitlines())}
        prosody={r['uid']:r for r in (json.loads(l) for l in (base/'prosody.jsonl').read_text().splitlines())}
        gaps=json.loads((base/'pause_statistics.json').read_text())
        evidence={}
        for r in gaps:
            key=(r['uid'],r['start'],r['end'])
            if key in evidence:continue
            p=pred[r['uid']];pr=prosody[r['uid']]
            rms=np.asarray(p['acoustic']['rms']);times=np.arange(len(rms))*.02+.0125
            inside=(times>=r['start']+.02)&(times<=r['end']-.02)
            assert inside.any()
            db=20*np.log10(np.maximum(rms,1e-8)/max(np.percentile(rms,90),1e-8))
            t=np.asarray(pr['times']);ix=(t>=r['start']+.02)&(t<=r['end']-.02)
            evidence[key]=dict(uid=r['uid'],start=r['start'],end=r['end'],
                median_rms_db_relative_to_utterance_p90=float(np.median(db[inside])),
                low_energy_fraction={str(th):float(np.mean(db[inside]<-th)) for th in [15,20,25]},
                praat_voiced_fraction=float(np.mean(np.asarray(pr['voiced_smoothed'])[ix])) if ix.any() else None)
        output=[]
        for r in gaps:
            e=evidence[r['uid'],r['start'],r['end']]
            output.append(dict(r,energy=e))
        dump(output,base/'gap_energy_statistics.json')
        summary=dict(total_unique_gaps=len(evidence),thresholds={})
        for th in [15,20,25]:
            kept={k for k,v in evidence.items() if v['low_energy_fraction'][str(th)]>=.8}
            ss={}
            for key in ['asr_3408','emotion_3407','emotion_3408','emotion_3409']:
                rows=[r for r in output if r['system']==key and (r['uid'],r['start'],r['end']) in kept]
                ss[key]=dict(n=len(rows),isolation40_fraction=float(np.mean([r['isolated_single_token40'] for r in rows])) if rows else None,
                             mean_internal_cuts=float(np.mean([r['internal_cuts'] for r in rows])) if rows else None)
            summary['thresholds'][str(th)]=dict(quiet_unique_gaps=len(kept),systems=ss)
        summaries[corpus]=summary
        print(corpus,summary,flush=True)
    dump(dict(protocol=dict(window_ms=25,hop_ms=20,edge_exclusion_ms=20,
                           reference='utterance RMS 90th percentile',threshold_db=[15,20,25],required_fraction=.8),
              corpora=summaries),OUT/'gap_energy_summary.json')


if __name__=='__main__':main()
