#!/usr/bin/env python3
"""Acquire a pinned Expresso read-speech mirror and select matched-text styles.

Run using the analysis environment (pyarrow, requests, soundfile, scipy).
The public mirror retains the original Expresso utterance ids and transcripts.
No model prediction or alignment output enters selection.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import random
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
import requests
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'artifacts/segmenter/nonasr_boundary_audit_2026-09-12'
REPO = 'ylacombe/expresso'
STYLES = ('default','confused','enunciated','happy','laughing','sad','whisper')


def dump(value,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')


def metadata_text(text):
    return re.sub(r'[^A-Z\s\']','',text.upper()).strip()


def download(out):
    provenance = out/'data/expresso_source.json'
    if provenance.exists():
        source = json.loads(provenance.read_text())
    else:
        r = requests.get(f'https://huggingface.co/api/datasets/{REPO}',timeout=60)
        r.raise_for_status()
        commit = r.json()['sha']
        r = requests.get(f'https://huggingface.co/api/datasets/{REPO}/tree/{commit}?recursive=true',timeout=60)
        r.raise_for_status()
        files = [f for f in r.json() if f['path'].endswith('.parquet')]
        source = dict(repo=REPO,commit=commit,files=files,
            original='https://github.com/facebookresearch/textlesslib/tree/main/examples/expresso/dataset',
            license='CC BY-NC 4.0', acquisition='Hugging Face read-speech Parquet mirror; original ids retained')
        dump(source,provenance)
    dest = out/'downloads/expresso'
    dest.mkdir(parents=True,exist_ok=True)
    def fetch(info):
        path = dest/Path(info['path']).name
        if path.exists() and path.stat().st_size == info['size']:
            return path
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{source['commit']}/{info['path']}?download=true"
        temp = path.with_suffix('.partial')
        with requests.get(url,timeout=(30,120),stream=True) as r:
            r.raise_for_status()
            h = hashlib.sha256()
            with temp.open('wb') as f:
                for block in r.iter_content(1048576):
                    f.write(block); h.update(block)
        assert temp.stat().st_size == info['size'], (temp,temp.stat().st_size,info['size'])
        if info.get('lfs',{}).get('oid'):
            assert h.hexdigest() == info['lfs']['oid']
        temp.rename(path)
        print('Downloaded',path.name,path.stat().st_size,flush=True)
        return path
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(fetch,source['files']))
    rows = []
    for path in sorted(paths):
        pf = pq.ParquetFile(path)
        cols = [c for c in pf.schema_arrow.names if c != 'audio']
        for group in range(pf.num_row_groups):
            for index,row in enumerate(pf.read_row_group(group,columns=cols).to_pylist()):
                rows.append(dict(row,parquet=str(path),row_group=group,row_index=index))
    dump(dict(source=source,rows=rows),out/'data/expresso_metadata.json')
    print('Metadata rows',len(rows),flush=True)


def select(out, texts_per_speaker):
    meta = json.loads((out/'data/expresso_metadata.json').read_text())
    candidates = [r for r in meta['rows'] if r['style'] in STYLES
                  and re.fullmatch(r'ex\d+_[a-z]+_\d+',r['id'])]
    cells = defaultdict(dict)
    for r in candidates:
        cells[r['speaker_id'],r['style']].setdefault(metadata_text(r['text']),r)
    speakers = sorted({r['speaker_id'] for r in candidates})
    assert len(speakers) == 4
    common = set.intersection(*(set(cells[s,style]) for s in speakers for style in STYLES))
    print('Common texts over all 28 speaker/style cells:',len(common),flush=True)
    rng = random.Random(20260912)
    chosen = []
    if len(common) >= texts_per_speaker:
        texts = rng.sample(sorted(common),texts_per_speaker)
        for s in speakers:
            for style in STYLES:
                chosen.extend(cells[s,style][t] for t in texts)
        design = 'Same randomly selected texts in all four speakers and seven styles'
    else:
        for s in speakers:
            shared = set.intersection(*(set(cells[s,style]) for style in STYLES))
            assert len(shared) >= texts_per_speaker,(s,len(shared))
            texts = rng.sample(sorted(shared),texts_per_speaker)
            for style in STYLES:
                chosen.extend(cells[s,style][t] for t in texts)
        design = 'Same randomly selected texts across seven styles within each speaker'
    audio_cache = {}
    rows = []
    wavdir = out/'data/expresso_wav16k'
    wavdir.mkdir(parents=True,exist_ok=True)
    for r in sorted(chosen,key=lambda r:(r['parquet'],r['row_group'],r['row_index'])):
        key = (r['parquet'],r['row_group'])
        if key not in audio_cache:
            audio_cache.clear()  # One row group (about 50 MB) in memory.
            audio_cache[key] = pq.ParquetFile(r['parquet']).read_row_group(r['row_group'],columns=['audio']).to_pylist()
        audio = audio_cache[key][r['row_index']]['audio']
        wav,sr = sf.read(io.BytesIO(audio['bytes']),dtype='float32')
        assert wav.ndim == 1 and sr == 48000,(r['id'],wav.shape,sr)
        y = resample_poly(wav,1,3)
        path = wavdir/(r['id']+'.wav')
        sf.write(path,y,16000,subtype='PCM_16')
        duration = len(y)/16000
        assert 0.2 < duration < 45,(r['id'],duration)
        rows.append(dict(uid=r['id'],corpus='expresso',speaker=r['speaker_id'],style=r['style'],
                         transcript=r['text'],text_group=metadata_text(r['text']),
                         duration=duration,wav=str(path),original_audio_path=audio['path'],
                         original_audio_sha256=hashlib.sha256(audio['bytes']).hexdigest(),
                         wav_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    assert len(rows) == 4*len(STYLES)*texts_per_speaker
    dump(dict(design=design,selection_seed=20260912,texts_per_speaker=texts_per_speaker,
              global_common_texts=len(common),source=meta['source'],utterances=rows),out/'data/expresso_selection.json')
    print('Selected',len(rows),'utterances;',sum(r['duration'] for r in rows),'seconds;',design,flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['download','select'])
    p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--texts-per-speaker',type=int,default=15)
    a = p.parse_args()
    if a.stage == 'download': download(a.output)
    else: select(a.output,a.texts_per_speaker)
