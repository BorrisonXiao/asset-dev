#!/usr/bin/env python3
"""Stage already-selected audio/transcripts for independent MFA references."""
import argparse
import json
import math
import re
from pathlib import Path
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/'artifacts/segmenter/nonasr_boundary_audit_2026-09-12'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--corpora',nargs='+',default=['cremad','expresso'])
    p.add_argument('--alignment-dir',type=Path,default=None)
    a=p.parse_args()
    alignment_dir=a.alignment_dir or a.output/'alignments'
    index=[]
    for corpus in a.corpora:
        selection=json.loads((a.output/corpus/'selection.json').read_text())
        for r in selection['utterances']:
            target=a.output/'data/mfa_corpus'/f"{corpus}_{r['speaker']}"/r['uid']
            target.parent.mkdir(parents=True,exist_ok=True)
            wav=Path(r['wav'])
            info=sf.info(wav)
            assert info.channels==1
            audio_target=target.with_suffix('.wav')
            if not audio_target.exists():
                if info.samplerate==16000:
                    audio_target.symlink_to(wav.resolve())
                else:
                    x,sr=sf.read(wav,dtype='float32')
                    factor=math.gcd(sr,16000)
                    sf.write(audio_target,resample_poly(x,16000//factor,sr//factor),16000,subtype='PCM_16')
            text=re.sub(r"[^a-zA-Z'\s]",' ',r['transcript'])
            text=' '.join(text.lower().split())
            assert text
            target.with_suffix('.lab').write_text(text+'\n')
            index.append(dict(uid=r['uid'],corpus=corpus,speaker=r['speaker'],
                              transcript_original=r['transcript'],transcript_normalized=text,
                              alignment_file=str(alignment_dir/target.parent.name/(target.name+'.TextGrid'))))
    (a.output/'alignment_index.json').write_text(json.dumps(index,indent=2)+'\n')
    print('Staged',len(index),'utterances for phone/word alignment',flush=True)


if __name__=='__main__': main()
