#!/usr/bin/env python3
"""Retain per-utterance cuts for a descriptive phone split/merge audit.

Selection reads annotations only. Inference runs under Slurm, extracts WavLM
features separately per utterance (avoiding padded normalization), then batches
the autoregressive policy. No decoder, training, or feature cache is used.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
from pathlib import Path

from audit_segment_units import DATA, ROOT, SEEDS, dump, read_grid, sha

OUT = ROOT / 'artifacts/segmenter/ls960_phone_patterns_2026-09-10'
TIMIT = Path('/export/corpora5/LDC/LDC93S1/timit/TIMIT')


def sphere_duration(path):
    with path.open('rb') as f:
        header=f.read(1024)
    assert header.startswith(b'NIST_1A'), path
    count=int(re.search(rb'sample_count -i (\d+)',header).group(1))
    rate=int(re.search(rb'sample_rate -i (\d+)',header).group(1))
    assert rate==16000
    return count/rate


def prepare(out):
    rng = random.Random(202609101)
    rows = []
    for split in ('test-clean', 'test-other'):
        candidates = list(csv.DictReader((DATA / f'librispeech_manifests/{split}.csv').open()))
        speakers = sorted({r['ID'].split('-')[0] for r in candidates})
        for speaker in speakers:
            group = sorted((r for r in candidates if r['ID'].split('-')[0] == speaker), key=lambda r: r['ID'])
            for r in rng.sample(group, min(5, len(group))):
                uid = r['ID']
                chapter = uid.split('-')[1]
                grid = DATA / f'LibriSpeech_alignments/{split}/{speaker}/{chapter}/{uid}.TextGrid'
                tiers = read_grid(grid)
                duration = float(r['duration'])
                assert abs(tiers['phones'][-1][1] - duration) < .021
                rows.append(dict(uid=uid, corpus=split, speaker=speaker, duration=duration,
                                 wav=r['wav'].replace('$data_root', str(DATA / 'LibriSpeech')),
                                 transcript=r['wrd'], phones=tiers['phones'], words=tiers['words'],
                                 reference=str(grid), reference_sha256=sha(grid)))
    for p in sorted((TIMIT / 'TEST').rglob('*.PHN')):
        if p.stem.lower() in ('sa1', 'sa2'):
            continue
        phones = [[int(a)/16000, int(b)/16000, label] for a,b,label in (line.split() for line in p.read_text().splitlines())]
        words = [[int(a)/16000, int(b)/16000, label] for a,b,label in (line.split() for line in p.with_suffix('.WRD').read_text().splitlines())]
        rows.append(dict(uid=p.relative_to(TIMIT).with_suffix('').as_posix(), corpus='timit-test',
                         speaker=p.parent.name, duration=sphere_duration(p.with_suffix('.WAV')),
                         annotation_end=phones[-1][1], wav=str(p.with_suffix('.WAV')),
                         transcript=' '.join(w[2] for w in words), phones=phones, words=words,
                         reference=str(p), reference_sha256=sha(p)))
    assert sum(r['corpus']=='timit-test' for r in rows) == 1344
    sources = {
        '100h': json.loads((ROOT / 'artifacts/segmenter/transformer_ar_bigru_phone_agreement_dev_clean.json').read_text())['model']['selected_checkpoints'],
        '960h': json.loads((ROOT / 'artifacts/segmenter/ls960_boundary_audit_2026-09-10/predictions.json').read_text())['checkpoints'],
    }
    checkpoints = {}
    for scale, source in sources.items():
        for seed in map(str, SEEDS):
            item = source[seed]
            p = Path(item['path'])
            if not p.is_absolute():
                p = Path(__file__).resolve().parent / p
            assert p.is_file(), p
            assert sha(p) == item['sha256'], p
            checkpoints[f'{scale}_{seed}'] = dict(path=str(p), sha256=item['sha256'])
    counts = {c:dict(utterances=sum(r['corpus']==c for r in rows),
                      speakers=len({r['speaker'] for r in rows if r['corpus']==c}),
                      seconds=sum(r['duration'] for r in rows if r['corpus']==c))
              for c in sorted({r['corpus'] for r in rows})}
    dump(dict(selection_seed=202609101, selection='Five random utterances per speaker on each LibriSpeech test split; complete TIMIT TEST excluding SA1/SA2. Selection uses annotations, no predictions.',
              counts=counts, checkpoints=checkpoints, utterances=rows,
              protocol='Descriptive split/merge and duration analysis; compulsory endpoints excluded from boundary scoring; 100h models on both corpora, 960h models on LibriSpeech. No causal interpretation of boundary departures.'), out / 'selection.json')
    print(json.dumps(counts, indent=2))


def infer(out, batch_size, max_utts):
    import torch
    import torchaudio
    from torch.nn.utils.rnn import pad_sequence
    from audit_transformer_ar_phone_agreement import _build_segmenter
    from ctc_boundary_align import num_encoder_frames
    from speechbrain.integrations.huggingface.wav2vec2 import Wav2Vec2
    from speechbrain.processing.features import InputNormalization

    torch.set_num_threads(4)
    selected = json.loads((out / 'selection.json').read_text())
    models = {key:_build_segmenter(Path(v['path']), torch.device('cuda')) for key,v in selected['checkpoints'].items()}
    ssl = Wav2Vec2(source='microsoft/wavlm-large', output_norm=True, freeze=True,
                  save_path='/export/jsalt26/omnienc/users/cxiao/hf/hub', device_map='cuda').eval()
    norm = InputNormalization(norm_type='sentence').cuda().eval()
    rows = sorted(selected['utterances'], key=lambda r:(r['corpus'], r['duration'], r['uid']))
    if max_utts:
        rows = rows[:max_utts]
    completed = {}
    path = out / 'predictions.jsonl'
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            assert r['selection_sha256'] == sha(out / 'selection.json')
            completed[r['uid']] = r
    rows = [r for r in rows if r['uid'] not in completed]
    selection_sha = sha(out / 'selection.json')
    start = time.monotonic()
    n_done = 0
    for corpus in sorted({r['corpus'] for r in rows}):
        subset = [r for r in rows if r['corpus']==corpus]
        keys = [k for k in models if k.startswith('100h') or corpus!='timit-test']
        for offset in range(0, len(subset), batch_size):
            batch = subset[offset:offset+batch_size]
            features = []
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                for row in batch:
                    wav, sr = torchaudio.load(row['wav'])
                    assert sr==16000 and wav.size(0)==1
                    assert abs(wav.size(1)/sr-row['duration']) < .021
                    lengths = torch.ones(1, device='cuda')
                    f = ssl(norm(wav.cuda(), lengths), lengths)[0]
                    assert f.size(0)==num_encoder_frames(wav.size(1))
                    features.append(f)
                padded = pad_sequence(features, batch_first=True)
                frame_lens = torch.tensor([f.size(0) for f in features], device='cuda')
                mask = torch.arange(padded.size(1), device='cuda')[None,:] >= frame_lens[:,None]
                predictions = {key:models[key].sample_boundaries(padded, mask, deterministic=True).boundaries.cpu() for key in keys}
                # Report a batched-policy sensitivity check on the first item.
                if n_done==0:
                    single = models[keys[0]].sample_boundaries(features[0][None], torch.zeros(1,features[0].size(0),dtype=torch.bool,device='cuda'), deterministic=True).boundaries[0].cpu()
                    batched = predictions[keys[0]][0,:single.numel()]
                    check = dict(system=keys[0], uid=batch[0]['uid'], frames=single.numel(), differing_frames=int((single!=batched).sum()), single_cuts=int(single.sum()), batched_cuts=int(batched.sum()))
                    dump(check, out/'policy_batch_check.json')
                    print('Batch sensitivity:',json.dumps(check),flush=True)
            with path.open('a') as handle:
                for i,row in enumerate(batch):
                    frames = features[i].size(0)
                    cuts = {key:[j for j in (b[i,:frames]==1).nonzero().flatten().tolist() if j>0] for key,b in predictions.items()}
                    for v in cuts.values():
                        assert v==sorted(set(v)) and all(0<t<frames for t in v)
                    unit = torch.nn.functional.normalize(features[i].float(), dim=-1)
                    change = torch.cat([unit.new_zeros(1), 1-(unit[:-1]*unit[1:]).sum(-1)]).cpu().tolist()
                    record = dict(uid=row['uid'], corpus=corpus, frames=frames, predictions=cuts,
                                  feature_change=change, selection_sha256=selection_sha)
                    handle.write(json.dumps(record)+'\n')
            n_done += len(batch)
            print(f'{n_done}/{len(rows)} new utterances; {corpus}; elapsed={time.monotonic()-start:.1f}s',flush=True)
    dump(dict(checkpoints=selected['checkpoints'], selection_sha256=selection_sha, precision='BF16',
              ssl_batch_size=1, policy_batch_size=batch_size, seconds=time.monotonic()-start,
              newly_completed=n_done, previous_completed=len(completed)), out/'inference_metadata.json')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','infer'])
    p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--max-utts',type=int,default=0)
    args=p.parse_args()
    if args.stage=='prepare':
        prepare(args.output)
    else:
        infer(args.output,args.batch_size,args.max_utts)
