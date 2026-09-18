#!/usr/bin/env python3
"""Recreate the audited dictionary augmentation from its retained provenance."""
import json
from audit_nonasr_boundaries import OUT

p=OUT/'data'
provenance=json.loads((p/'dictionary_augmentation.json').read_text())
source=OUT/'tools/mfa_home/pretrained_models/dictionary/english_us_arpa.dict'
original=source.read_text().rstrip().splitlines()
words={line.split()[0] for line in original}
for entry in provenance['entries']:
    assert entry['word'] not in words
    original.append('\t'.join([entry['word'],*[str(v) for v in provenance['new_entry_probabilities']],
                               ' '.join(entry['phones'])]))
dest=p/'english_us_arpa_expresso_augmented.dict'
dest.write_text('\n'.join(original)+'\n')
print('Wrote',dest)
