#!/usr/bin/env python3
"""Score the saved parent ASR trajectory for a stronger thinning control.

Use the original greedy boundary history, then retain its N highest-confidence
internal cuts, where N is the emotion policy's exact count on that utterance.
This is a descriptive, post-hoc control; no downstream quality claim is made.
"""
import json
import time
from pathlib import Path
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from audit_nonasr_boundaries import OUT, build_model, dump
from speechbrain.integrations.huggingface.wav2vec2 import Wav2Vec2
from speechbrain.processing.features import InputNormalization


def main():
    torch.set_num_threads(4)
    ck=json.loads((OUT/'checkpoint_audit.json').read_text())['checkpoints']['asr_3408']
    model=build_model(ck)
    ssl=Wav2Vec2(source='microsoft/wavlm-large',output_norm=True,freeze=True,
        save_path='/export/jsalt26/omnienc/users/cxiao/hf/hub',device_map='cuda').eval()
    norm=InputNormalization(norm_type='sentence').eval().cuda()
    start=time.monotonic()
    for corpus in ['cremad','expresso','librispeech']:
        base=OUT/corpus
        rows=json.loads((base/'selection.json').read_text())['utterances']
        rows=sorted(rows,key=lambda r:(r['duration'],r['uid']))
        preds={r['uid']:r for r in (json.loads(l) for l in (base/'predictions.jsonl').read_text().splitlines())}
        records=[]
        for offset in range(0,len(rows),24):
            batch=rows[offset:offset+24]
            features=[]
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                for r in batch:
                    wav,sr=torchaudio.load(r['wav'])
                    if sr!=16000:wav=torchaudio.functional.resample(wav,sr,16000)
                    length=torch.ones(1,device='cuda')
                    f=ssl(norm(wav.cuda(),length),length)[0]
                    assert len(f)==preds[r['uid']]['frames']
                    features.append(f)
                padded=pad_sequence(features,batch_first=True)
                n=torch.tensor([len(f) for f in features],device='cuda')
                mask=torch.arange(padded.size(1),device='cuda')[None]>=n[:,None]
                actions=torch.zeros(mask.shape,device='cuda',dtype=torch.long)
                for i,r in enumerate(batch):actions[i,preds[r['uid']]['predictions']['asr_3408']]=1
                logits=model.teacher_forced_logits(padded,mask,actions).float().cpu()
            for i,r in enumerate(batch):
                cuts=preds[r['uid']]['predictions']['asr_3408']
                records.append(dict(uid=r['uid'],asr_cuts=cuts,logits_at_cuts=logits[i,cuts].tolist()))
            if (offset//24)%10==0:print(corpus,len(records),'/',len(rows),flush=True)
        dump(records,base/'asr_boundary_confidence.json')
    dump(dict(method='Parallel teacher-forced conditional logits along saved parent ASR greedy trajectories',
              purpose='Exact-count confidence-thinning null',checkpoint=ck,
              precision='BF16 autocast',seconds=time.monotonic()-start),OUT/'confidence_protocol.json')


if __name__=='__main__':main()
