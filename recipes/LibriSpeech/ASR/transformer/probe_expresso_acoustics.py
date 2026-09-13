#!/usr/bin/env python3
"""Controlled pitch/duration/envelope probes; no training or recognition decoding."""
from __future__ import annotations
import argparse, hashlib, json, os, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
ROOT=Path(__file__).resolve().parents[4]
OLD=ROOT/'artifacts/segmenter/nonasr_boundary_audit_2026-09-12'
OUT=ROOT/'artifacts/segmenter/expresso_boundary_patterns_2026-09-13'
POLICIES=['asr_3408','emotion_3407','emotion_3408','emotion_3409','intent_3408_last','speaker_count_3408_last']
CONDITIONS=['original','sham','pitch_down3','pitch_up3','duration_080','duration_125','pitch_flat','envelope_compress']

def dump(x,p):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def acoustic(y,sr):
    import numpy as np, parselmouth
    from scipy.ndimage import gaussian_filter1d
    sound=parselmouth.Sound(y,sampling_frequency=sr)
    pitch=sound.to_pitch_ac(time_step=.01,pitch_floor=75.,pitch_ceiling=600.,silence_threshold=.03,voicing_threshold=.45)
    rms=np.sqrt(gaussian_filter1d(y*y,sigma=.01*sr))[::160]
    return dict(times=pitch.xs().tolist(),f0=pitch.selected_array['frequency'].tolist(),rms=rms.tolist(),rms_times=(np.arange(len(rms))*.01).tolist())

def prepare_one(row):
    import numpy as np, soundfile as sf, parselmouth
    from parselmouth.praat import call
    from scipy.ndimage import gaussian_filter1d
    y,sr=sf.read(row['wav']);assert sr==16000 and y.ndim==1
    snd=parselmouth.Sound(y,sampling_frequency=sr)
    records=[];features={};generated={}
    for condition in CONDITIONS:
        scale={'duration_080':.8,'duration_125':1.25}.get(condition,1.)
        if condition=='original': z=y.copy()
        elif condition=='envelope_compress':
            env=np.sqrt(gaussian_filter1d(y*y,sigma=.025*sr)+1e-10)
            ref=np.percentile(env,75)
            gain=np.clip((ref/np.maximum(env,.05*ref))**.5,.5,2.)
            z=y*gain
        else:
            m=call(snd,'To Manipulation',.01,75.,600.)
            if condition.startswith('pitch_'):
                tier=call(m,'Extract pitch tier')
                if condition=='pitch_flat':
                    n=call(tier,'Get number of points')
                    vals=[call(tier,'Get value at index',i) for i in range(1,n+1)]
                    if vals:
                        med=float(np.median(vals));call(tier,'Remove points between',0,snd.xmax)
                        call(tier,'Add point',0.,med);call(tier,'Add point',snd.xmax,med)
                else: call(tier,'Multiply frequencies',0,snd.xmax,2**((3 if condition=='pitch_up3' else -3)/12))
                call([tier,m],'Replace pitch tier')
            if condition.startswith('duration_'):
                tier=call(m,'Extract duration tier')
                call(tier,'Add point',0.,scale);call(tier,'Add point',snd.xmax,scale)
                call([tier,m],'Replace duration tier')
            z=call(m,'Get resynthesis (overlap-add)').values[0]
        # Apply only a global peak reduction if needed; sentence normalization removes global gain.
        peak=float(np.max(np.abs(z)));gain_out=min(1.,.98/max(peak,1e-10))
        z=z*gain_out
        path=OUT/'probes/wav'/f"{row['uid']}__{condition}.wav"
        if condition=='original': path=Path(row['wav'])
        else:
            path.parent.mkdir(parents=True,exist_ok=True);sf.write(path,z,sr,subtype='PCM_16')
            z,_=sf.read(path)
        generated[condition]=z;features[condition]=acoustic(z,sr)
        records.append(dict(uid=f"{row['uid']}__{condition}",source_uid=row['uid'],speaker=row['speaker'],
            text_group=row['text_group'],condition=condition,wav=str(path),wav_sha256=sha(path),
            duration=len(z)/sr,duration_scale=scale,peak_scale=gain_out))
    q=[]
    for r in records:
        c=r['condition'];base='original' if c in ['original','sham','envelope_compress'] else 'sham'
        a,b=features[c],features[base]
        t=np.asarray(b['times']);fa=np.interp(t*r['duration_scale'],a['times'],a['f0']);fb=np.asarray(b['f0'])
        both=(fa>0)&(fb>0)
        semi=12*np.log2(fa[both]/fb[both])
        env_a=np.interp(np.asarray(b['rms_times'])*r['duration_scale'],a['rms_times'],a['rms'])
        q.append(dict(uid=r['uid'],condition=c,comparison=base,
          duration_ratio=r['duration']/(len(y)/sr),f0_median_shift_semitones=float(np.median(semi)) if len(semi) else None,
          f0_shift_iqr_semitones=float(np.ptp(np.percentile(semi,[25,75]))) if len(semi) else None,
          f0_log_sd=float(np.std(np.log2(np.asarray(a['f0'])[np.asarray(a['f0'])>0]))) if any(np.asarray(a['f0'])>0) else None,
          jointly_voiced_frames=int(both.sum()),voicing_agreement=float(np.mean((fa>0)==(fb>0))),
          envelope_correlation=float(np.corrcoef(env_a,b['rms'])[0,1]),
          envelope_cv=float(np.std(a['rms'])/np.mean(a['rms']))))
    return records,q

def prepare(workers,limit):
    refs=json.loads((OLD/'expresso/references.json').read_text())
    rows=sorted([r for r in refs if r['style']=='default'],key=lambda r:r['uid'])
    if limit: rows=rows[:limit]
    records=[];qc=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i,(rs,qs) in enumerate(pool.map(prepare_one,rows,chunksize=1)):
            records.extend(rs);qc.extend(qs);print('prepared',i+1,'/',len(rows),flush=True)
    dump(dict(conditions=CONDITIONS,policies=POLICIES,utterances=records,source_sha256=sha(OLD/'expresso/references.json')),OUT/'probes/selection.json')
    dump(qc,OUT/'probes/acoustic_qc.json')
    import parselmouth
    dump(dict(parselmouth=parselmouth.__version__,resynthesis='Praat overlap-add; pitch floor75Hz ceiling600Hz;10ms analysis',
      pitch_semitones=[-3,3],duration_factors=[.8,1.25],flatten='median pitch-tier frequency',
      envelope='25ms Gaussian RMS; gain=(75th percentile RMS / RMS)^0.5 clipped to[0.5,2]',
      controls='pitch/duration compared with sham; envelope compared with original; no claim of perceptual emotion constancy'),OUT/'probes/protocol.json')

def infer(batch_size,prediction_name="predictions.jsonl",source_uids=None):
    import torch,torchaudio
    from torch.nn.utils.rnn import pad_sequence
    from speechbrain.integrations.huggingface.wav2vec2 import Wav2Vec2
    from speechbrain.processing.features import InputNormalization
    from audit_nonasr_boundaries import build_model
    from ctc_boundary_align import num_encoder_frames
    torch.set_num_threads(4)
    selection=OUT/'probes/selection.json';data=json.loads(selection.read_text());digest=sha(selection)
    ck=json.loads((OLD/'checkpoint_audit.json').read_text())['checkpoints']
    for k in POLICIES: assert sha(ck[k]['path'])==ck[k]['sha256']
    models={k:build_model(ck[k]) for k in POLICIES}
    ssl=Wav2Vec2(source='microsoft/wavlm-large',output_norm=True,freeze=True,save_path='/export/jsalt26/omnienc/users/cxiao/hf/hub',device_map='cuda').eval()
    norm=InputNormalization(norm_type='sentence').cuda().eval()
    dest=OUT/'probes'/prediction_name;completed=set()
    if dest.exists():
        for l in dest.read_text().splitlines():
            r=json.loads(l);assert r['selection_sha256']==digest;completed.add(r['uid'])
    rows=sorted([r for r in data['utterances'] if r['uid'] not in completed and (not source_uids or r['source_uid'] in source_uids)],key=lambda r:(r['duration'],r['uid']))
    checks=[];start=time.monotonic()
    for off in range(0,len(rows),batch_size):
        batch=rows[off:off+batch_size];feats=[]
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            for r in batch:
                assert sha(r['wav'])==r['wav_sha256']
                wav,sr=torchaudio.load(r['wav']);assert sr==16000 and wav.shape[0]==1
                lengths=torch.ones(1,device='cuda');f=ssl(norm(wav.cuda(),lengths),lengths)[0]
                assert len(f)==num_encoder_frames(wav.shape[1]);feats.append(f)
            padded=pad_sequence(feats,batch_first=True);lens=torch.tensor([len(f) for f in feats],device='cuda')
            mask=torch.arange(padded.shape[1],device='cuda')[None,:]>=lens[:,None]
            predictions={}
            for k,m in models.items():
                m.active_task_id=torch.full((len(batch),),ck[k]['task_id'],dtype=torch.long,device='cuda')
                predictions[k]=m.sample_boundaries(padded,mask,deterministic=True).boundaries.cpu()
            if off==0:
                for k,m in models.items():
                    m.active_task_id=torch.tensor([ck[k]['task_id']],device='cuda')
                    sm=torch.zeros(1,len(feats[0]),dtype=torch.bool,device='cuda')
                    z=m.sample_boundaries(feats[0][None],sm,deterministic=True).boundaries[0].cpu()
                    checks.append(dict(uid=batch[0]['uid'],policy=k,different_frames=int((z!=predictions[k][0,:len(z)]).sum())))
        with dest.open('a') as f:
            for i,r in enumerate(batch):
                cuts={k:[int(t) for t in (z[i,:len(feats[i])]==1).nonzero().flatten() if t>0] for k,z in predictions.items()}
                assert all(int(z[i,0])==0 for z in predictions.values())
                f.write(json.dumps(dict(uid=r['uid'],frames=len(feats[i]),predictions=cuts,selection_sha256=digest))+'\n')
        print('inferred',off+len(batch),'/',len(rows),'seconds',round(time.monotonic()-start,1),flush=True)
    dump(dict(device=torch.cuda.get_device_name(0),torch=torch.__version__,batch_size=batch_size,
        seconds=time.monotonic()-start,selection_sha256=digest,checkpoints={k:ck[k]['sha256'] for k in POLICIES},
        batch_checks=checks),OUT/'probes'/('inference_metadata.json' if prediction_name=='predictions.jsonl' else prediction_name.replace('.jsonl','_metadata.json')))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','infer'])
    p.add_argument('--workers',type=int,default=4);p.add_argument('--limit',type=int);p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--prediction-name',default='predictions.jsonl');p.add_argument('--source-uids')
    a=p.parse_args();prepare(a.workers,a.limit) if a.stage=='prepare' else infer(a.batch_size,a.prediction_name,a.source_uids.split(',') if a.source_uids else None)
