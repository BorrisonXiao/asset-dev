#!/usr/bin/env python3
"""Descriptive Praat pitch/periodicity features on the fixed audit selection.

Automatic voicing and F0 are diagnostic measurements, not prosodic ground truth.
Whisper/laughter can challenge pitch tracking; report their coverage explicitly.
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import parselmouth
from scipy.ndimage import median_filter

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'artifacts/segmenter/nonasr_boundary_audit_2026-09-12'


def extract(row):
    sound=parselmouth.Sound(row['wav'])
    pitch=sound.to_pitch_ac(time_step=.01,pitch_floor=75.,pitch_ceiling=600.,
                            silence_threshold=.03,voicing_threshold=.45)
    f0=pitch.selected_array['frequency']
    times=pitch.xs()
    voiced=median_filter((f0>0).astype(np.uint8),size=5)>0
    transitions=(times[:-1]+times[1:])[np.diff(voiced.astype(int))!=0]/2
    valid=(f0[:-1]>0)&(f0[1:]>0)
    delta=np.zeros(len(f0))
    delta[1:][valid]=np.abs(np.log2(f0[1:][valid]/f0[:-1][valid]))
    return dict(uid=row['uid'],times=times.tolist(),f0_hz=f0.tolist(),
                strength=pitch.selected_array['strength'].tolist(),
                voiced_smoothed=voiced.tolist(),voicing_transitions=transitions.tolist(),
                abs_f0_delta_octaves=delta.tolist(),f0_delta_valid=np.r_[False,valid].tolist())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    for corpus in ['cremad','expresso','librispeech']:
        rows=json.loads((a.output/corpus/'selection.json').read_text())['utterances']
        dest=a.output/corpus/'prosody.jsonl'
        completed=set()
        if dest.exists():
            completed={json.loads(l)['uid'] for l in dest.read_text().splitlines()}
        rows=[r for r in rows if r['uid'] not in completed]
        with ProcessPoolExecutor(max_workers=a.workers) as pool, dest.open('a') as f:
            for i,r in enumerate(pool.map(extract,rows,chunksize=8)):
                f.write(json.dumps(r)+'\n')
                if (i+1)%200==0:print(corpus,i+1,'/',len(rows),flush=True)
        print('Completed',corpus,len(rows),'new records',flush=True)
    (a.output/'prosody_protocol.json').write_text(json.dumps(dict(
        tool='praat-parselmouth',version=parselmouth.__version__,algorithm='Praat autocorrelation pitch',
        time_step=.01,pitch_floor=75,pitch_ceiling=600,silence_threshold=.03,voicing_threshold=.45,
        voicing_smoothing='five-frame (50-ms) median filter',
        interpretation='Automatic acoustic diagnostics; not manual voicing or prosodic annotations'),indent=2)+'\n')
