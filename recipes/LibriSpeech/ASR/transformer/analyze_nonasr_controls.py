#!/usr/bin/env python3
"""Check alignment-quality, timing and confidence-control sensitivity."""
import csv
import json
from pathlib import Path
import numpy as np
from audit_nonasr_boundaries import OUT,SEEDS,dump
from audit_segment_units import match_count
from summarize_nonasr_boundaries import metrics,bootstrap_mean


def main():
    result={}
    first_qc=json.loads((OUT/'alignment_v1_audit/expresso/alignment_qc.json').read_text())
    original_oov={r['uid'] for r in first_qc if r.get('unknown_phones',0)}
    alignment_quality={r['file']:r for r in csv.DictReader((OUT/'alignments_v2/alignment_analysis.csv').open())}
    for corpus in ['cremad','expresso','librispeech']:
        base=OUT/corpus
        stats=json.loads((base/'per_utterance_statistics.json').read_text())
        refs={r['uid']:r for r in json.loads((base/'references.json').read_text())}
        preds=[json.loads(l) for l in (base/'predictions.jsonl').read_text().splitlines()]
        group={}
        for uid,r in refs.items():
            if uid in alignment_quality and alignment_quality[uid]['speech_log_likelihood']:
                group.setdefault(r['style'],[]).append((float(alignment_quality[uid]['speech_log_likelihood']),uid))
        # Remove the lowest decile WITHIN each style, so the sensitivity check
        # does not simply delete all whisper or laughing speech.
        poor={uid for rows in group.values() for _,uid in sorted(rows)[:max(1,len(rows)//10)]}
        control_results={}
        for key in [f'emotion_{s}' for s in SEEDS]:
            rows=stats[key]
            control_results[key]={}
            for cohort in ['all_aligned','original_dictionary_covered','without_lowest_likelihood_decile_per_style']:
                rs=[r for r in rows if 'phone_20ms' in r['scores']
                    and (cohort!='original_dictionary_covered' or r['uid'] not in original_oov)
                    and (cohort!='without_lowest_likelihood_decile_per_style' or r['uid'] not in poor)]
                entry=dict(n=len(rs),scores={})
                for metric in ['phone_20ms','word_20ms','vowel_onset_20ms','voicing_20ms']:
                    entry['scores'][metric]={}
                    for system in ['scores','grid_scores','asr_thin_scores','confidence_thin_scores','offset12_5_scores','confidence_thin_offset12_5_scores']:
                        v=[r[system][metric] for r in rs if metric in r.get(system,{})]
                        entry['scores'][metric][system]=metrics(np.sum(v,axis=0))
                control_results[key][cohort]=entry
            eligible=[r for r in rows if 'confidence_thin_agreement20' in r]
            z=np.sum([r['confidence_thin_agreement20'] for r in eligible],axis=0)
            control_results[key]['confidence_cut_f1_20ms']=metrics(z)['f1']
            ratios=[(r['acoustics']['feature_sum']/max(1,r['acoustics']['feature_n']))/
                    (r['confidence_thin_acoustics']['feature_sum']/max(1,r['confidence_thin_acoustics']['feature_n']))
                    for r in eligible if r['acoustics']['feature_n']]
            blocks=[r['block'] for r in eligible if r['acoustics']['feature_n']]
            control_results[key]['feature_change_ratio_to_confidence']=bootstrap_mean(ratios,blocks)
        keys=['asr_3408','emotion_3408','intent_3408','speaker_count_3408','intent_3408_last','speaker_count_3408_last']
        common=[r for r in preds if all(k in r['predictions'] for k in keys)]
        matrix=np.zeros((len(keys),len(keys)))
        for i,a in enumerate(keys):
            for j,b in enumerate(keys):
                hit=sum(match_count(r['predictions'][a],r['predictions'][b],1) for r in common)
                total=sum(len(r['predictions'][a])+len(r['predictions'][b]) for r in common)
                matrix[i,j]=2*hit/total if total else 1.
        common_rates={k:float(np.mean([50*(len(r['predictions'][k])+1)/r['frames'] for r in common])) for k in keys}
        by_style={}
        for style in sorted({r['style'] for r in refs.values()}):
            style_rows=[r for r in stats['asr_3408'] if r['style']==style]
            ratio=[]
            for r in style_rows:
                vals=[next(x for x in stats[f'emotion_{s}'] if x['uid']==r['uid'])['tokens'] for s in SEEDS]
                ratio.append(float(np.mean(vals))/r['tokens'])
            by_style[style]=dict(n=len(style_rows),mean_token_ratio_to_asr=float(np.mean(ratio)))
        result[corpus]=dict(controls=control_results,agreement=dict(keys=keys,utterances=len(common),f1_20ms=matrix.tolist(),mean_hz=common_rates),
                            style_token_ratio=by_style,excluded_original_dictionary_oov=len(original_oov) if corpus=='expresso' else 0,
                            low_likelihood_excluded=len(poor))
    dump(result,OUT/'control_sensitivity.json')
    print('Saved alignment, timestamp, confidence and cross-task controls')


if __name__=='__main__':main()
