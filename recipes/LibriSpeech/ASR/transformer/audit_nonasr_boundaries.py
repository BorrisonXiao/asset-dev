#!/usr/bin/env python3
"""Checkpoint-grounded, same-audio audit of ASR and non-ASR boundary policies.

Prepare selects data without looking at predictions. Infer is GPU-only and
loads no decoder. All artifacts live outside the manuscript directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[4]
RECIPE = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts/segmenter/nonasr_boundary_audit_2026-09-12'
DATA = Path('/export/jsalt26/omnienc/users/cxiao/datasets')
SEEDS = (3407, 3408, 3409)
TASK_IDS = {'asr': 0, 'emotion': 2, 'speaker_count': 3, 'intent': 4}
ARMS = ('nods_50hz', 'fixed_25hz', 'fixed_10hz', 'fixed_5hz', 'learned_frozen', 'learned_grpo')
METRICS = {'emotion': 'macro_f1_emotion', 'speaker_count': 'acc_speaker_count', 'intent': 'acc_intent'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def dump(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load_hparams(path):
    # Read scalars without instantiating HyperPyYAML objects or downloading models.
    return yaml.load(path.read_text(), Loader=yaml.BaseLoader)


def audit_runs(out):
    records, checkpoints = [], {}
    historical = json.loads((ROOT / 'artifacts/segmenter/ls960_phone_patterns_2026-09-10/selection.json').read_text())
    for seed in SEEDS:
        ck = dict(historical['checkpoints'][f'960h_{seed}'])
        assert sha(ck['path']) == ck['sha256']
        checkpoints[f'asr_{seed}'] = dict(ck, task='asr', task_id=0, seed=seed,
                                          num_tasks=1, role='common_parent' if seed == 3408 else 'asr_seed_control')
    for task, metric in METRICS.items():
        for arm in ARMS:
            for seed in SEEDS:
                run = RECIPE / f'results/speechllm_singletask_nonasr/{task}/{arm}/{seed}'
                hp = load_hparams(run / 'hyperparams.yaml')
                rows = [json.loads(l) for l in (run / 'task_metrics.jsonl').read_text().splitlines() if l.startswith('{')]
                tests = [r for r in rows if r.get('stage') == 'TEST' and metric in r]
                assert len(tests) == 1, (run, tests)
                test = tests[0]
                step = test['optimizer_step']
                bridge, film = int(hp['bridge_optimizer_steps']), int(hp['film_optimizer_steps'])
                # Checkpoint after exactly bridge+film updates has had NO final-block updates.
                updates_c = max(0, step - bridge - film)
                phase = 'unfreeze' if updates_c else ('film' if step > bridge else 'bridge')
                candidates = [(p, yaml.safe_load((p / 'CKPT.yaml').read_text())) for p in sorted((run / 'save').glob('CKPT*'))]
                selected = [(p, m) for p, m in candidates if m.get('optimizer_step') == step]
                assert len(selected) == 1, (run, selected)
                ckdir, meta = selected[0]
                validation = [r for r in rows if r.get('stage') == 'VALID' and r.get('optimizer_step') == step]
                assert validation and abs(meta['WER'] - 100 * (1 - validation[-1][metric])) < 1e-5
                record = dict(task=task, arm=arm, seed=seed, metric=metric, test=test,
                              selected_step=step, actual_training_phase=phase,
                              final_block_updates=updates_c, bridge_steps=bridge, film_steps=film,
                              phase_recorded_in_test=test.get('phase'),
                              phase_record_is_stale=test.get('phase') != phase,
                              selected_checkpoint=str(ckdir), checkpoint_metadata=meta,
                              selection_validation=validation[-1],
                              rate_channel=hp['rate_channel'], rate_bands=hp['task_rate_bands'],
                              global_rate_band=[float(hp['rho_lo']), float(hp['rho_hi'])],
                              warm_start=hp['warm_start_ckpt_dir'],
                              result_source=str(run / 'task_metrics.jsonl'),
                              result_sha256=sha(run / 'task_metrics.jsonl'),
                              hparams_sha256=sha(run / 'hyperparams.yaml'))
                records.append(record)
                if arm == 'learned_grpo' or (task == 'emotion' and arm == 'learned_frozen' and seed == 3408):
                    key = f'{task}_{seed}' if arm == 'learned_grpo' else 'frozen_emotion_3408'
                    p = ckdir / 'segmenter.ckpt'
                    checkpoints[key] = dict(path=str(p), sha256=sha(p), task=task,
                                            task_id=TASK_IDS[task], seed=seed, num_tasks=5,
                                            role='selected' if arm == 'learned_grpo' else 'identity_control',
                                            step=step, training_phase=phase, final_block_updates=updates_c)
                if arm == 'learned_grpo' and seed == 3408 and task in ('intent', 'speaker_count'):
                    final_dir, final_meta = max(candidates, key=lambda x: x[1]['optimizer_step'])
                    p = final_dir / 'segmenter.ckpt'
                    checkpoints[f'{task}_{seed}_last'] = dict(path=str(p), sha256=sha(p), task=task,
                        task_id=TASK_IDS[task], seed=seed, num_tasks=5, role='last_step_sensitivity',
                        step=final_meta['optimizer_step'], training_phase='unfreeze',
                        final_block_updates=final_meta['optimizer_step']-bridge-film)
    dump(dict(runs=records, checkpoints=checkpoints), out / 'checkpoint_audit.json')
    print('Audited', len(records), 'run metrics;', len(checkpoints), 'policy checkpoints', flush=True)
    return checkpoints


def prepare(out, corpus):
    checkpoint_path = out / 'checkpoint_audit.json'
    checkpoints = (json.loads(checkpoint_path.read_text())['checkpoints'] if checkpoint_path.exists() else audit_runs(out))
    rng = random.Random(20260912)
    if corpus == 'cremad':
        emo = {r['utt_key']:r for r in csv.DictReader((DATA / 'nonasr_manifests/cremad_emotion_test.csv').open())}
        rows = []
        for r in csv.DictReader((DATA / 'nonasr_manifests/cremad_asr_test.csv').open()):
            e = emo[r['utt_key']]
            rows.append(dict(uid=r['utt_key'], corpus=corpus, speaker=r['spk_id'],
                             duration=float(r['duration']), wav=r['wav'], transcript=r['wrd'],
                             emotion=e['wrd'], style=r['utt_key'].split('_')[2]))
        assert len(rows) == 1431
        groups = defaultdict(list)
        for r in rows:
            groups[(r['speaker'], r['style'])].append(r)
        extra = {r['uid'] for group in groups.values() for r in rng.sample(group, min(3,len(group)))}
        for r in rows:
            r['extra_policies'] = r['uid'] in extra
    elif corpus == 'librispeech':
        old = json.loads((ROOT / 'artifacts/segmenter/ls960_phone_patterns_2026-09-10/selection.json').read_text())
        rows = [dict(r, corpus=corpus, split=r['corpus'], style=r['corpus'], extra_policies=True)
                for r in old['utterances'] if r['corpus'] != 'timit-test']
    else:
        rows = json.loads((out / 'data/expresso_selection.json').read_text())['utterances']
        for r in rows:
            r['extra_policies'] = True
    main = ['asr_3408'] + [f'emotion_{s}' for s in SEEDS]
    extras = [k for k in checkpoints if k not in main and k != 'frozen_emotion_3408']
    counts = dict(utterances=len(rows), speakers=len({r['speaker'] for r in rows}),
                  seconds=sum(r['duration'] for r in rows), extras=sum(r['extra_policies'] for r in rows))
    dest = out / corpus
    dump(dict(corpus=corpus, selection_seed=20260912, counts=counts, utterances=rows,
              main_policies=main, extra_policies=extras, checkpoints=checkpoints), dest / 'selection.json')
    print(corpus, json.dumps(counts), flush=True)


def build_model(ck):
    import torch
    from segmenter import Segmenter
    model = Segmenter(input_dim=1024, backbone='transformer_ar', hidden_dim=256,
        num_layers=4, nhead=4, ffn_dim=1024, dropout=0.0,
        ar_hidden_dim=256, ar_num_layers=4, ar_nhead=4, ar_ffn_dim=1024,
        ar_dropout=0.0, ar_max_positions=4096, ar_history_window=64, ar_cache_mode='preallocated',
        pooling='bigru_residual', pooling_input_dim=1024, pooling_hidden_dim=128,
        pooling_num_layers=1, pooling_dropout=0.0, num_tasks=ck['num_tasks'])
    model.load_state_dict(torch.load(ck['path'], map_location='cpu', weights_only=True), strict=True)
    return model.eval().cuda()


def infer(out, corpus, batch_size, max_utts):
    import torch
    import torchaudio
    from torch.nn.utils.rnn import pad_sequence
    from ctc_boundary_align import num_encoder_frames
    from speechbrain.integrations.huggingface.wav2vec2 import Wav2Vec2
    from speechbrain.processing.features import InputNormalization
    torch.set_num_threads(4)
    dest = out / corpus
    selection = json.loads((dest / 'selection.json').read_text())
    digest = sha(dest / 'selection.json')
    ckpts = selection['checkpoints']
    models = {k:build_model(v) for k,v in ckpts.items()}
    ssl = Wav2Vec2(source='microsoft/wavlm-large', output_norm=True, freeze=True,
                  save_path='/export/jsalt26/omnienc/users/cxiao/hf/hub', device_map='cuda').eval()
    norm = InputNormalization(norm_type='sentence').cuda().eval()
    # Sort by extra-policy group and length, so batches share the same model set.
    rows = sorted(selection['utterances'], key=lambda r:(r['extra_policies'],r['duration'],r['uid']))
    if max_utts:
        rows = rows[:max_utts]
    completed = set()
    pred_path = dest / 'predictions.jsonl'
    if pred_path.exists():
        for line in pred_path.read_text().splitlines():
            r = json.loads(line)
            assert r['selection_sha256'] == digest
            completed.add(r['uid'])
    rows = [r for r in rows if r['uid'] not in completed]
    start = time.monotonic()
    n_done, checks = 0, []
    for extra in (False,True):
        subset = [r for r in rows if r['extra_policies'] == extra]
        keys = selection['main_policies'] + (selection['extra_policies'] if extra else [])
        for offset in range(0,len(subset),batch_size):
            batch = subset[offset:offset+batch_size]
            features, acoustic = [], []
            with torch.inference_mode(), torch.autocast('cuda',dtype=torch.bfloat16):
                for row in batch:
                    wav, sr = torchaudio.load(row['wav'])
                    assert wav.size(0) == 1
                    if sr != 16000:
                        wav = torchaudio.functional.resample(wav,sr,16000)
                    assert abs(wav.size(1)/16000-row['duration']) < .021
                    lengths = torch.ones(1,device='cuda')
                    f = ssl(norm(wav.cuda(),lengths),lengths)[0]
                    assert f.size(0) == num_encoder_frames(wav.size(1))
                    features.append(f)
                    # 25-ms RMS windows centered at encoder receptive-field centers.
                    rms = wav[0].unfold(0,400,320).square().mean(-1).clamp(min=1e-12).sqrt()
                    acoustic.append(dict(rms=rms.tolist(), samples=wav.size(1)))
                padded = pad_sequence(features,batch_first=True)
                frame_lens = torch.tensor([f.size(0) for f in features],device='cuda')
                mask = torch.arange(padded.size(1),device='cuda')[None,:] >= frame_lens[:,None]
                preds = {}
                for key in keys:
                    models[key].active_task_id = torch.full((len(batch),),ckpts[key]['task_id'],dtype=torch.long,device='cuda')
                    preds[key] = models[key].sample_boundaries(padded,mask,deterministic=True).boundaries.cpu()
                if n_done == 0:
                    single_mask = torch.zeros(1,features[0].size(0),dtype=torch.bool,device='cuda')
                    for key in ('asr_3408','emotion_3408','frozen_emotion_3408'):
                        model = models[key]
                        model.active_task_id = torch.tensor([ckpts[key]['task_id']],device='cuda')
                        a = model.sample_boundaries(features[0][None],single_mask,deterministic=True).boundaries[0].cpu()
                        b = model.sample_boundaries(features[0][None],single_mask,deterministic=True).boundaries[0].cpu()
                        reference = preds['asr_3408' if key == 'frozen_emotion_3408' else key][0,:a.numel()]
                        checks.append(dict(uid=batch[0]['uid'],system=key,
                            repeat_differing_frames=int((a!=b).sum()),
                            batched_or_parent_differing_frames=int((a!=reference).sum())))
                    dump(checks,dest/'determinism_checks.json')
            with pred_path.open('a') as handle:
                for i,row in enumerate(batch):
                    frames = features[i].size(0)
                    cuts = {k:[t for t in (b[i,:frames]==1).nonzero().flatten().tolist() if t>0] for k,b in preds.items()}
                    # Full AR fixes the first action to 0; pooling supplies its
                    # implicit initial segment. All reported cuts are internal.
                    assert all(int(b[i,0]) == 0 for b in preds.values())
                    unit = torch.nn.functional.normalize(features[i].float(),dim=-1)
                    change = torch.cat([unit.new_zeros(1),1-(unit[:-1]*unit[1:]).sum(-1)]).cpu().tolist()
                    handle.write(json.dumps(dict(uid=row['uid'],corpus=corpus,frames=frames,
                        predictions=cuts,feature_change=change,acoustic=acoustic[i],selection_sha256=digest))+'\n')
            n_done += len(batch)
            print(f'{corpus} {n_done}/{len(rows)} new utterances; {len(keys)} models; elapsed={time.monotonic()-start:.1f}s',flush=True)
    dump(dict(checkpoints=ckpts,selection_sha256=digest,precision='BF16 autocast',
              ssl_batch_size=1,policy_batch_size=batch_size,newly_completed=n_done,
              previous_completed=len(completed),seconds=time.monotonic()-start),dest/'inference_metadata.json')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['audit','prepare','infer'])
    p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--corpus',choices=['cremad','expresso','librispeech'],default='cremad')
    p.add_argument('--batch-size',type=int,default=24)
    p.add_argument('--max-utts',type=int,default=0)
    args = p.parse_args()
    if args.stage == 'audit': audit_runs(args.output)
    elif args.stage == 'prepare': prepare(args.output,args.corpus)
    else: infer(args.output,args.corpus,args.batch_size,args.max_utts)


if __name__ == '__main__':
    main()
