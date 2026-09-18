#!/usr/bin/env python3
"""Small, preselected LS960 alignment/pause audit; no decoder or retraining.

prepare reads only alignment text. infer runs on Slurm and writes auditable
per-utterance predictions, references, and acoustic energy. summarize is CPU only.
All boundary scores exclude the two obligatory utterance endpoints.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DATA = Path('/export/jsalt26/omnienc/users/cxiao/datasets')
OUT = ROOT / 'artifacts/segmenter/ls960_boundary_audit_2026-09-10'
RUNS = Path(__file__).resolve().parent / 'results/speechllm_ls960_step24k_corrected/transformer_ar_local64_bigru/nll_mt'
SEEDS = (3407, 3408, 3409)
SIL = {'', 'sil', 'sp', 'spn', '<eps>'}


def dump(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read_grid(path):
    text = path.read_text()
    tiers = {}
    for block in re.split(r'item\s*\[\d+\]:', text)[1:]:
        name = re.search(r'name\s*=\s*"([^"]+)"', block).group(1)
        tiers[name] = [
            [float(a), float(b), label.replace('""', '"')]
            for a, b, label in re.findall(
                r'intervals\s*\[\d+\]:\s*xmin\s*=\s*([\d.eE+-]+)\s*xmax\s*=\s*([\d.eE+-]+)\s*text\s*=\s*"((?:[^"]|"")*)"', block
            )
        ]
    assert set(tiers) == {'words', 'phones'}, path
    for intervals in tiers.values():
        assert intervals and all(b > a for a, b, _ in intervals), path
        assert all(abs(intervals[i][1] - intervals[i+1][0]) < 1e-6 for i in range(len(intervals)-1)), path
    return tiers


def pauses(words):
    speech = [x for x in words if x[2].lower() not in SIL]
    return [[left[1], right[0], left[2], right[2]] for left, right in zip(speech, speech[1:]) if right[0] - left[1] >= .25 - 1e-8]


def prepare(out):
    rng = random.Random(20260910)
    rows = list(csv.DictReader((DATA / 'librispeech_manifests/dev-clean.csv').open()))
    candidates = []
    for row in sorted(rows, key=lambda r: r['ID']):
        uid = row['ID']
        speaker, chapter, _ = uid.split('-')
        grid = DATA / f'LibriSpeech_alignments/dev-clean/{speaker}/{chapter}/{uid}.TextGrid'
        tiers = read_grid(grid)
        assert abs(tiers['words'][-1][1] - float(row['duration'])) < .021
        candidates.append(dict(uid=uid, speaker=speaker, duration=float(row['duration']), transcript=row['wrd'], wav=row['wav'].replace('$data_root', str(DATA / 'LibriSpeech')), textgrid=str(grid), textgrid_sha256=sha(grid), tiers=tiers, pauses=pauses(tiers['words'])))
    speakers = sorted({r['speaker'] for r in candidates})
    selected = []
    for speaker in rng.sample(speakers, 12):
        row = dict(rng.choice([r for r in candidates if r['speaker'] == speaker]))
        row['cohort'] = 'speaker_sample'
        selected.append(row)
    selected_ids = {r['uid'] for r in selected}
    eligible = [r for r in candidates if r['pauses'] and r['uid'] not in selected_ids]
    pause_speakers = sorted({r['speaker'] for r in eligible})
    for speaker in rng.sample(pause_speakers, 12):
        row = dict(rng.choice([r for r in eligible if r['speaker'] == speaker]))
        row['cohort'] = 'pause_sample'
        selected.append(row)
    result = dict(selection_seed=20260910, split='dev-clean', selection='12 uniformly sampled speakers, one random utterance each; independently 12 speakers with an internal aligned gap >=250 ms, one random eligible utterance each; disjoint utterances; selected without model predictions', population_utterances=len(candidates), population_pause_utterances=sum(bool(r['pauses']) for r in candidates), utterances=selected)
    dump(result, out / 'selection.json')
    print(json.dumps({k:v for k,v in result.items() if k != 'utterances'}, indent=2))
    print('Selected', len(selected), 'utterances;', round(sum(r['duration'] for r in selected), 2), 's;', sum(len(r['pauses']) for r in selected), 'pauses')


def infer(out):
    import numpy as np
    import torch
    import torchaudio
    import yaml
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks
    from audit_transformer_ar_phone_agreement import _build_segmenter
    from speechbrain.integrations.huggingface.wav2vec2 import Wav2Vec2
    from speechbrain.processing.features import InputNormalization
    torch.set_num_threads(4)
    selected = json.loads((out / 'selection.json').read_text())
    checkpoints = {}
    for seed in SEEDS:
        run = RUNS / str(seed)
        candidates = []
        for p in (run / 'save').glob('CKPT*/CKPT.yaml'):
            meta = yaml.safe_load(p.read_text())
            if 'WER' in meta and (p.parent / 'segmenter.ckpt').exists():
                candidates.append((float(meta['WER']), p.parent / 'segmenter.ckpt', meta))
        wer, path, meta = min(candidates, key=lambda x: x[0])
        checkpoints[str(seed)] = dict(path=str(path), sha256=sha(path), metadata=meta, selection='minimum retained dev-clean WER, matching trainer min_key=WER; epoch counters alone are ambiguous for step-validation runs')
    print(json.dumps(checkpoints, indent=2), flush=True)
    models = {s: _build_segmenter(Path(v['path']), torch.device('cuda')) for s,v in checkpoints.items()}
    ssl = Wav2Vec2(source='microsoft/wavlm-large', output_norm=True, freeze=True, save_path='/export/jsalt26/omnienc/users/cxiao/hf/hub', device_map='cuda').eval()
    norm = InputNormalization(norm_type='sentence').cuda().eval()
    output = []
    for row in selected['utterances']:
        waveform, sr = torchaudio.load(row['wav'])
        assert sr == 16000 and waveform.size(0) == 1
        wav = waveform.cuda()
        assert abs(wav.size(1) / sr - row['duration']) < 1e-6
        lens = torch.ones(1, device='cuda')
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            feat = ssl(norm(wav, lens), lens)
            mask = torch.zeros(feat.shape[:2], dtype=torch.bool, device='cuda')
            predictions = {seed: model.sample_boundaries(feat, mask, deterministic=True).boundaries[0].cpu() for seed, model in models.items()}
        row['frames'] = feat.size(1)
        row['predictions'] = {s: [i for i in (b == 1).nonzero().flatten().tolist() if i > 0] for s,b in predictions.items()}
        # Broadband RMS: 25 ms windows, 5 ms hops, centered timestamps.
        arr = waveform[0].numpy()
        frames = np.lib.stride_tricks.sliding_window_view(arr, 400)[::80]
        energy = 10 * np.log10(np.maximum(np.mean(frames ** 2, axis=1), 1e-12))
        energy = gaussian_filter1d(energy, 1.0)
        times = (np.arange(len(energy)) * 80 + 200) / sr
        # >=3 dB local prominence and >=40 ms separation; fixed before inference.
        minima = find_peaks(-energy, prominence=3, distance=8)[0]
        row['energy'] = dict(times=times.tolist(), db=energy.tolist(), valleys=times[minima].tolist())
        # Keep the existing syllable reference as an auxiliary diagnostic only.
        path = DATA / f"wavlm_boundaries/syllable/{row['uid']}.pt"
        if path.exists():
            target = torch.load(path, map_location='cpu', weights_only=True).flatten()
            assert len(target) == feat.size(1)
            row['syllable_starts'] = [i * .02 for i in (target == 1).nonzero().flatten().tolist() if i > 0]
        output.append(row)
        print(f"{len(output)}/{len(selected['utterances'])} {row['uid']} frames={feat.size(1)} tokens=" + str({s:len(v)+1 for s,v in row['predictions'].items()}), flush=True)
        dump(dict(checkpoints=checkpoints, selection_sha256=sha(out / 'selection.json'), utterances=output), out / 'predictions.partial.json')
    dump(dict(checkpoints=checkpoints, selection_sha256=sha(out / 'selection.json'), precision='BF16; one utterance per batch; greedy boundaries', frame_time='frame index * 20 ms; +12.5 ms receptive-field-center sensitivity also scored', utterances=output), out / 'predictions.json')


def match_count(pred, ref, tol):
    i = j = hits = 0
    while i < len(pred) and j < len(ref):
        if abs(pred[i] - ref[j]) <= tol + 1e-9:
            hits += 1
            i += 1
            j += 1
        elif pred[i] < ref[j]:
            i += 1
        else:
            j += 1
    return hits


def reference_edges(intervals, duration):
    return sorted({round(t, 8) for a,b,label in intervals if label.lower() not in SIL for t in (a,b) if 0 < t < duration})


def overlap(a, b, c, d):
    return max(0., min(b,d) - max(a,c))


def count_matched_grid(frames, internal_cuts):
    """Uniform cuts rounded to the same 20 ms frame lattice as the policy."""
    positions = [round(i * frames / (internal_cuts + 1)) for i in range(1, internal_cuts + 1)]
    assert len(set(positions)) == internal_cuts
    assert all(0 < p < frames for p in positions)
    return [p * .02 for p in positions]


def gap_stats(cuts, duration, gap):
    a,b,*_ = gap
    intervals = list(zip([0., *cuts], [*cuts, duration]))
    fractions = [overlap(x,y,a,b) / (y-x) for x,y in intervals]
    return dict(internal_cuts=sum(a < t < b for t in cuts), overlapping_tokens=sum(v > 0 for v in fractions), pure_tokens=sum(v >= 1-1e-8 for v in fractions), mostly_silent_tokens=sum(v >= .8 for v in fractions), onset_error_ms=min(abs(t-a) for t in [0., *cuts, duration])*1000, offset_error_ms=min(abs(t-b) for t in [0., *cuts, duration])*1000, max_single_token_gap_coverage=max(overlap(x,y,a,b) for x,y in intervals)/(b-a))


def summarize(out):
    import numpy as np
    payload = json.loads((out / 'predictions.json').read_text())
    rows = payload['utterances']
    assert len(rows) == 24 and len({r['uid'] for r in rows}) == 24
    systems = ['learned', 'count_matched_grid', 'fixed_k5', 'circular_shift']
    summaries = {}
    gaps_out = []
    for cohort in ('speaker_sample', 'pause_sample'):
        subset = [r for r in rows if r['cohort'] == cohort]
        cohort_result = {}
        for seed in map(str, SEEDS):
            seed_result = {}
            for system in systems:
                score_counts = {}
                durations = []
                single_phone = []
                pause_results = []
                n_token = n_frames = 0
                acoustic_duration = 0.
                for r in subset:
                    learned = np.array(r['predictions'][seed], dtype=float) * .02
                    dur = r['duration']
                    if system == 'learned':
                        cuts = learned.tolist()
                    elif system == 'count_matched_grid':
                        cuts = count_matched_grid(r['frames'], len(learned))
                    elif system == 'fixed_k5':
                        cuts = (np.arange(5, r['frames'], 5) * .02).tolist()
                    else:
                        # Deterministic utterance-specific shift preserves cut count and spacing.
                        fraction = .2 + .6 * int(hashlib.sha256(r['uid'].encode()).hexdigest()[:8],16) / (2**32-1)
                        cuts = sorted(((learned + fraction*dur) % dur).tolist())
                    assert all(0 < x < dur for x in cuts) and all(x < y for x,y in zip(cuts,cuts[1:]))
                    n_token += len(cuts) + 1
                    n_frames += r['frames']
                    acoustic_duration += dur
                    refs = dict(phone=reference_edges(r['tiers']['phones'],dur), word=reference_edges(r['tiers']['words'],dur), energy_valley=r['energy']['valleys'])
                    if 'syllable_starts' in r:
                        refs['syllable_precomputed'] = r['syllable_starts']
                    for name, ref in refs.items():
                        for tol in (.02,.04):
                            for offset in (0., .0125):
                                label = f'{name}_{int(tol*1000)}ms_offset{offset*1000:g}ms'
                                counts = score_counts.setdefault(label,[0,0,0])
                                pred = [x+offset for x in cuts if x+offset < dur]
                                counts[0] += match_count(pred,ref,tol)
                                counts[1] += len(pred)
                                counts[2] += len(ref)
                    intervals = list(zip([0., *cuts], [*cuts,dur]))
                    durations.extend((b-a)*1000 for a,b in intervals)
                    phones = [p for p in r['tiers']['phones'] if p[2].lower() not in SIL]
                    for a,b in intervals:
                        speech_overlap = sum(overlap(a,b,c,d) for c,d,_ in phones)
                        if speech_overlap >= .8*(b-a):
                            single_phone.append(max(overlap(a,b,c,d) for c,d,_ in phones) >= .8*(b-a))
                    for gap in r['pauses']:
                        values = gap_stats(cuts,dur,gap)
                        pause_results.append(values)
                        if system == 'learned':
                            gaps_out.append(dict(uid=r['uid'],cohort=cohort,seed=seed,gap=gap,**values))
                scores = {key:dict(matches=v[0],predicted=v[1],reference=v[2],precision=v[0]/v[1] if v[1] else 0.,recall=v[0]/v[2] if v[2] else 0.,f1=2*v[0]/(v[1]+v[2]) if v[1]+v[2] else 0.) for key,v in score_counts.items()}
                seed_result[system] = dict(tokens=n_token,seconds=acoustic_duration,token_rate_hz=n_token/acoustic_duration,frame_based_rate_hz=50*n_token/n_frames,segment_ms_percentiles=np.percentile(durations,[10,50,90]).tolist(),speech_segments=len(single_phone),fraction_speech_segments_80pct_one_phone=float(np.mean(single_phone)),scores=scores,pauses=dict(n=len(pause_results),without_internal_cut=sum(g['internal_cuts']==0 for g in pause_results),with_pure_token=sum(g['pure_tokens']>0 for g in pause_results),with_mostly_silent_token=sum(g['mostly_silent_tokens']>0 for g in pause_results),both_edges_within40ms=sum(g['onset_error_ms']<=40+1e-8 and g['offset_error_ms']<=40+1e-8 for g in pause_results),median_internal_cuts=float(np.median([g['internal_cuts'] for g in pause_results]))) if pause_results else None)
            cohort_result[seed] = seed_result
        summaries[cohort] = cohort_result
    result = dict(protocol='Pilot; 12 speaker-sampled and 12 pause-enriched utterances analyzed separately, three LS960 seeds. Internal phone/word boundaries include speech/silence transitions, exclude utterance endpoints. Maximum ordered one-to-one matches in continuous time. Equal-count uniform grid uses exactly the learned token count for each utterance, rounded to the same 20 ms frame lattice. Energy valleys are broadband RMS minima, not harmonic energy. Syllable labels are existing derived targets, not independently audited.',summary=summaries,pause_details=gaps_out)
    dump(result,out / 'summary.json')
    print(json.dumps({c:{s:{k:dict(hz=round(v['token_rate_hz'],2),phone_f1=round(v['scores']['phone_20ms_offset0ms']['f1']*100,2),word_f1=round(v['scores']['word_20ms_offset0ms']['f1']*100,2),pauses=v['pauses']) for k,v in sv.items()} for s,sv in cv.items()} for c,cv in summaries.items()},indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['prepare','infer','summarize'])
    p.add_argument('--output', type=Path, default=OUT)
    args = p.parse_args()
    globals()[args.stage](args.output)
