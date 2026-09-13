#!/usr/bin/env python3
"""Build an example-centered Expresso boundary study and its portable evidence."""
from __future__ import annotations
import argparse,base64,hashlib,html,io,json,shutil,statistics,sys,zipfile
from pathlib import Path
from analyze_expresso_patterns import ROOT,OLD,OUT,PRIMARY,EMOTIONS,STYLES,block,features
from make_nonasr_boundary_report import Report,CSS,esc,hk,ASSETS
VERSION='20260913-expresso-patterns'
LABELS={'asr_3408':'ASR parent','emotion_3408':'Emotion 3408','emotion_3407':'Emotion 3407','emotion_3409':'Emotion 3409','intent_3408_last':'Intent · last step','speaker_count_3408_last':'Count · last step'}
RECIPE=Path(__file__).resolve().parent
EXTRA_CSS='''
.controls{display:flex;flex-wrap:wrap;gap:12px;padding:14px;background:var(--surface-2);border-radius:8px;margin:12px 0}.controls label{display:flex;flex-direction:column;gap:5px;font-size:13px;color:var(--text-secondary)}.controls label.check{flex-direction:row;align-items:center}.controls select{max-width:min(620px,85vw);border:1px solid var(--border);background:var(--surface-1);color:var(--text-primary);padding:8px;border-radius:5px;font:inherit}.explorer svg{display:block;min-width:850px;width:100%;height:auto}.explorer .figure-scroll{background:var(--surface-1)}.svgt{fill:var(--text-primary);font:12px system-ui,sans-serif}.svgt.small{font-size:10px}.status{font-size:12px;letter-spacing:.05em;text-transform:uppercase;color:var(--text-muted)}.obs{border-left:3px solid var(--series-asr);padding:2px 0 2px 16px;margin:20px 0}.sources li{margin:13px 0}.stat-row .tile .v{font-size:26px}.explorer{padding:14px;border:1px solid var(--border);border-radius:10px;margin:18px 0}table td{white-space:normal!important}.evidence-note{font-size:13px;color:var(--text-secondary)}
'''

def pc(x,d=1):return f'{100*x:.{d}f}'
def est(x,scale=100,d=1):return f"{x['mean']*scale:.{d}f} [{x['ci95'][0]*scale:.{d}f}, {x['ci95'][1]*scale:.{d}f}]"
def read(name):return json.loads((OUT/name).read_text())
def paragraph(s):return '<p>'+s+'</p>'

def explorer_data():
    refs=json.loads((OLD/'expresso/references.json').read_text());pr={r['uid']:r for r in map(json.loads,(OLD/'expresso/predictions.jsonl').read_text().splitlines())}
    pro={r['uid']:r for r in map(json.loads,(OLD/'expresso/prosody.jsonl').read_text().splitlines())}
    out=[]
    for r in refs:
        f=features(r,pr[r['uid']],pro[r['uid']]);p=pr[r['uid']]
        out.append(dict(uid=r['uid'],block=block(r),style=r['style'],transcript=r['transcript'],duration=r['duration'],words=r['words'],phones=r['phones'],
            env=[round(float(x),4) for x in f['env']],cuts={k:p['predictions'][k] for k in PRIMARY+EMOTIONS[:1]+EMOTIONS[2:]}))
    return dict(utterances=out,mapped=read('same_text_mapped_cuts.json'),initial_audio_group='ex03::WHO STARS IN THE NUTTY PROFESSOR',initial_text_group='ex01::FOR THE FRESH FACE CONTEST')

def interactive(r,kind,title):
    r.figures+=1
    if kind=='audio':
        controls='<label>Speaker and text<select id="audio-group"></select></label><label>Rendition<select id="audio-style"></select></label><label>Zoom to word<select id="audio-window"></select></label><label class="check"><input type="checkbox" id="audio-seeds">Show other emotion seeds</label>'
        desc='All 420 utterances. Compare the same signal across tasks; use the word selector to inspect a local disagreement. Hover over a cut for its time. Counts include the implicit initial segment.'
    else:
        controls='<label>Speaker and text<select id="text-group"></select></label><label>Policy<select id="text-policy"></select></label><label class="check"><input type="checkbox" id="text-styles">Include whisper, laughing and enunciated</label>'
        desc='All 60 speaker–text groups. Cuts are mapped through matching phone sequences inside corresponding words. Green: one-to-one match to default within 20 ms; orange: unmatched; dotted: default. The 20 ms tolerance is measured on the default time axis.'
    return f'<div class="explorer" hidden><h3>{title}</h3><div class="controls">{controls}</div><p class="compact" id="{kind}-description"></p><figure><div class="figure-scroll"><svg id="{kind}-svg" role="img" aria-label="{esc(title)}"></svg></div><figcaption><strong>Figure {r.figures}.</strong> {desc}</figcaption></figure></div><noscript><p>Interactive exploration requires JavaScript; the static examples and tables remain available.</p></noscript>'

def package(r):
    # Preserve the earlier audit with its numerical downloads, clearly demoted to supporting evidence.
    oldhtml=OLD/'html/nonasr-task-boundaries-standalone.html'
    archive=oldhtml.read_text().replace('<body>','<body><div style="padding:14px;background:#e8eef4;color:#22364a">Supporting audit from 12 September. <a href="nonasr-task-boundaries-standalone.html">Open the current boundary-pattern study</a>.</div>',1)
    (r.output.parent/'nonasr-task-boundaries-performance-audit.html').write_text(archive)
    shutil.copytree(OLD/'html'/ASSETS,r.assets,dirs_exist_ok=True)
    files=[OUT/n for n in ['protocol.md','input_sha256.json','same_audio_summary.json','same_text_summary.json','extended_summary.json','probe_summary.json','selected_examples.json','alignment_correspondence_qc.json','same_text_mapped_cuts.json']]
    files+=sorted((OUT/'tables').glob('*.csv'))
    files += [p for p in (OUT/'probes').glob('*') if p.is_file() and p.suffix in ['.json','.jsonl']]
    files += [p for p in [OUT/'review.md',OUT/'validation.json',OUT/'batch_sensitivity_summary.json'] if p.exists()]
    files += [RECIPE/n for n in ['analyze_expresso_patterns.py','extend_expresso_patterns.py','probe_expresso_acoustics.py','plot_expresso_patterns.py','make_expresso_pattern_report.py','expresso_pattern_explorer.js','run_expresso_patterns.slurm','validate_expresso_patterns.py','audit_nonasr_boundaries.py']]
    files += [OLD/'expresso'/n for n in ['references.json','predictions.jsonl','prosody.jsonl']]
    digest={}
    with zipfile.ZipFile(r.assets/'pattern-evidence.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(set(files)):
            name=str(path.relative_to(ROOT));raw=path.read_bytes();digest[name]=hashlib.sha256(raw).hexdigest()
            text=raw.decode().replace(str(ROOT),'$PROJECT').replace('/export/jsalt26/omnienc/users/cxiao','$WORKSPACE')
            zi=zipfile.ZipInfo(name,(2026,9,13,0,0,0));zi.compress_type=zipfile.ZIP_DEFLATED;z.writestr(zi,text.encode())
        z.writestr('original_sha256.json',json.dumps(digest,indent=2))
        z.writestr('README.txt','Expresso boundary-pattern study, 13 September2026. All420 natural renditions,60 texts byspeaker,480 controlled probe inputs. No source audio or model checkpoint files are distributed. Absolute cluster paths use portable prefixes; original_sha256 hashes original files before substitution. Tables retain per-utterance/block evidence. The original audit remains separately linked. Scripts require the project modules/environment and licensed source audio for inference.\n')
    return digest

def build():
    dest=OUT/'html/nonasr-task-boundaries-standalone.html';dest.parent.mkdir(parents=True,exist_ok=True);r=Report(OUT,dest)
    s=read('same_audio_summary.json');st=read('same_text_summary.json');ext=read('extended_summary.json');p=read('probe_summary.json');ex=read('selected_examples.json')
    assert 'comparisons' in p
    em='emotion_3408';cnt='speaker_count_3408_last';intent='intent_3408_last';tr=ext['transitions']['0.0']['20']
    q=st['qc'];sur=p['comparisons'][em]
    body=hk.hero('Analysis 03 · boundary patterns · 13 September 2026','Shared acoustic landmarks, different boundary preferences',
      'On the same Expresso audio, task policies retain different kinds of transitions. Across expressive renditions, many locations persist, while extra cuts respond to temporal stretching. Pitch changes alone leave most cuts intact.')
    body+='<nav class="jump" aria-label="On this page">'+''.join(f'<a href="#{a}">{t}</a>' for a,t in [('same-audio','Same audio'),('same-text','Same text'),('patterns','Acoustic patterns'),('probes','Controlled probes'),('panel','Full panel'),('pauses','Pauses'),('paper','Paper proposal'),('methods','Methods'),('sources','Sources & downloads')])+'</nav>'
    body+=hk.tiles([('Natural renditions','420','60 speaker–text groups · 4 speakers · 7 renditions'),('Controlled inputs','480','60 originals × 8 conditions · six policies'),('Count’s onset preference',f"{pc(tr['consonant → vowel'][cnt]['mean'])}% / {pc(tr['vowel → consonant'][cnt]['mean'])}%",'ASR cuts retained at consonant→vowel / vowel→consonant transitions'),('Duration ×1.25',f"+{100*(sur['duration_125']['20']['token_ratio']['mean']-1):.1f}%",'Emotion tokens · pitch preserved approximately')])
    body+='<div class="note"><strong>Scope of the interpretation.</strong><p>These are boundary-policy measurements, not recognition scores. All task adaptations share one ASR parent and use different prescribed rate bands. The evidence identifies preferences in these checkpoints; it does not isolate task reward from rate constraints or establish phoneme, syllable, or emotion-token identities. Intent and count use explicitly labeled last-step checkpoints with final-block updates.</p></div>'
    part=paragraph('The policies share many cut locations, but their differences have a direction. In Figure 1, emotion retains a cut near the end of <em>stars</em>; count drops that cut and retains the nearby transition into the vowel of <em>in</em>. This is a local change in which transition is kept, not just a lower utterance-wide rate.')
    part+=r.figure('same_audio_example','Four policies on exactly the same waveform. Orange shading marks the <em>stars → in</em> neighborhood. Each tick is an internal cut; the number at right includes the initial segment. MFA phone and word labels are automatic.','Spectrogram, envelope and four boundary tracks for Who stars in the Nutty Professor, sad rendition.')
    sample=ex['examples'][0];events=[x for x in sample['events'] if .89<=x['t']<=1.02]
    part+=r.table(['ASR location (s)','Nearest MFA transition','Emotion cuts nearby','Intent cuts nearby','Count cuts nearby'],[
      (f"{x['t']:.2f}",x['nearest_phone_transition'],', '.join(f'{v:.2f}' for v in x['decisions'][em]) or 'none',', '.join(f'{v:.2f}' for v in x['decisions'][intent]) or 'none',', '.join(f'{v:.2f}' for v in x['decisions'][cnt]) or 'none') for x in events],
      'Exact local decisions in Figure 1. “Nearby” means within 20 ms of that ASR cut; raw times are shown so local shifts remain visible.')
    n=s['nesting'];part+=paragraph(f"Across all 420 utterances, {pc(n['count_in_emotion']['mean'])}% of count’s matched ASR anchors also belong to emotion, far above the {pc(n['random_expected_count_in_emotion']['mean'])}% expected from independently selecting the same numbers of parent anchors. Yet {est(n['count_not_emotion'])}% are absent from emotion’s set (estimate [95% block-bootstrap interval]). The policies are mostly, but not strictly, nested. This remains {pc(n['sparser_subset']['mean'])}% when count is required to be the sparser policy on that utterance.")
    part+=interactive(r,'audio','Inspect the same utterance across tasks')
    body+=r.section('same-audio','1 · Same recording: what is shared, and what changes?',part)

    part=paragraph('For the same speaker saying “For the fresh face contest,” the emotion checkpoint places cuts near the vowel entries in <em>for</em> and <em>the</em> across the four main renditions. Its finer treatment of <em>fresh</em>, <em>face</em> and <em>contest</em> varies. Figure 3 aligns corresponding phones to the default rendition, so speaking faster does not by itself look like a new boundary pattern.')
    part+=r.figure('same_text_example','The same emotion checkpoint on seven renditions. Gray dotted cuts are the eligible default reference; orange cuts come from each rendition after phone-wise time mapping. Actual duration and total token count are printed at left. Unmatched phone sequences and gaps are omitted from the comparison. Whisper, laughing and enunciated are speaking styles, separate from the main affective contrasts.','Phone-aligned emotion cuts for seven renditions of For the fresh face contest.')
    words=ex['same_text_words'];part+=r.table(['Rendition','for: cuts / word duration','the','fresh','face','contest'],[
      (style.title(),*[f"{len(next(x for x in words if x['style']==style and x['word']==w)['cuts'])} / {1000*next(x for x in words if x['style']==style and x['word']==w)['duration']:.0f}ms" for w in ['for','the','fresh','face','contest']]) for style in STYLES[:4]],
      'Cuts inside each aligned word for the main four renditions in Figure 3. These are cut counts, not word tokens; a pooled segment can cross a word edge.')
    part+=paragraph('A useful counterexample to a duration-only account is <em>face</em>: its vowel lasts 200 ms in default and 190 ms in happy, but the main emotion checkpoint has an interior cut at 1.64s only in default. The two vowel-entry cuts remain, at 1.54s and 1.08s in the respective recordings. The extra-cut difference appears in two of the three emotion seeds; seed 3409 also splits the happy vowel. This is a concrete policy difference, but it is not enough to identify its acoustic cause.')
    part+=paragraph(f"Over all main affective pairs, {pc(ext['phone_allocation'][em]['same_allocation']['mean'])}% of corresponding phones have the same cut count (including zero-cut phones). When a phone is at least 25% longer, the number of cuts lying more than 20 ms inside that phone increases in {est(ext['phone_allocation'][em]['internal_change']['longer ≥25%']['has_extra_internal'])}% of pairs; the reverse occurs in {pc(ext['phone_allocation'][em]['internal_change']['longer ≥25%']['has_lost_internal']['mean'])}%. This internal-cut measure can reflect new cuts or existing cuts becoming farther from a stretched edge. The total cut count also rises by 0.22 per lengthened phone, versus 0.00 for similar-duration phones. Duration helps explain the allocation changes, while the example above shows it does not explain every decision.")
    part+=interactive(r,'text','Inspect the same text across renditions')
    body+=r.section('same-text','2 · Same text: stable locations and flexible internal cuts',part)

    part=paragraph('The strongest aggregate pattern is directional: entries into vowels and rising amplitude envelopes survive compression more often than exits from vowels. This is especially pronounced for the count checkpoint. Emotion retains a wider mix of entries, exits and internal cuts. Figure 5 and Table 3 condition on available ASR cuts, so a policy’s lower overall rate cannot by itself explain the contrast between transition types.')
    part+=r.figure('transition_preferences','Fraction of common ASR candidate cuts retained within 20 ms. Left: cuts near consonant→vowel versus vowel→consonant transitions. Right: envelope-slope quartiles computed within each utterance’s frame sequence. Error bars:95% bootstrap intervals over 60 speaker–text blocks, conditional on these four speakers.','Transition retention by task: count retains many vowel entries and very few vowel exits; all policies prefer rising envelope slopes.')
    part+=r.table(['Policy','Consonant → vowel (%)','Vowel → consonant (%)','Most-rising slope quartile (%)','Most-falling slope quartile (%)'],[
      (LABELS[k],est(tr['consonant → vowel'][k]),est(tr['vowel → consonant'][k]),est(s['parent_retention']['slope_quartile']['3'][k]),est(s['parent_retention']['slope_quartile']['0'][k])) for k in PRIMARY[1:]],
      'Retention of available ASR cuts, with 95% block-bootstrap intervals. Consonants are broad stop/affricate, fricative and sonorant classes. No human prosodic labels are assumed.')
    part+=paragraph(f"The direction survives a +12.5 ms timestamp convention: count retains {pc(ext['transitions']['12.5']['20']['consonant → vowel'][cnt]['mean'])}% of entry candidates versus {pc(ext['transitions']['12.5']['20']['vowel → consonant'][cnt]['mean'])}% of exit candidates. Emotion’s corresponding ranges across three seeds are {min(ext['transitions']['0.0']['20']['consonant → vowel'][k]['mean'] for k in EMOTIONS)*100:.1f}–{max(ext['transitions']['0.0']['20']['consonant → vowel'][k]['mean'] for k in EMOTIONS)*100:.1f}% at entries and {min(ext['transitions']['0.0']['20']['vowel → consonant'][k]['mean'] for k in EMOTIONS)*100:.1f}–{max(ext['transitions']['0.0']['20']['vowel → consonant'][k]['mean'] for k in EMOTIONS)*100:.1f}% at exits.")
    part+=paragraph('This suggests <strong>preferential retention of acoustic onsets</strong>, with task-dependent amounts of intervening detail. “Valleys in harmonic energy” would be too specific: we measured the broadband amplitude envelope, and rising edges are a better descriptive match than its valleys. “Syllable segmentation” would also be premature: vowel-entry correspondence does not supply syllable annotations or one-to-one syllable units.')
    body+=r.section('patterns','3 · A phonetic and acoustic pattern, rather than a unit label',part)

    part=paragraph('Natural renditions change timing, pitch, articulation and voice quality together. We therefore reran six frozen policies on 60 default recordings under eight controlled conditions. Pitch and duration variants are compared with a Praat overlap-add resynthesis sham. The sham and envelope-compressed versions are compared with the original recording. Duration-variant cuts are mapped back to the original time scale before matching.')
    part+=r.figure('controlled_acoustic_probes','One-to-one boundary F1 at 20 ms and per-utterance token-count ratios. Each point averages60 paired recordings; error bars are95% speaker–text block-bootstrap intervals. The duration conditions are mapped back in time before scoring. Small pitch effects should be read alongside the sham effect.','Controlled pitch and duration probes: duration changes alter token allocation much more than pitch shifts.')
    part+=r.table(['Condition','Emotion boundary F1 (%)','Emotion token ratio','ASR token ratio','Count token ratio'],[
      (label,est(sur[c]['20']['f1']),est(sur[c]['20']['token_ratio'],1,3),est(p['comparisons']['asr_3408'][c]['20']['token_ratio'],1,3),est(p['comparisons'][cnt][c]['20']['token_ratio'],1,3)) for c,label in [('sham','Resynthesis sham'),('pitch_down3','Pitch −3 semitones'),('pitch_up3','Pitch +3 semitones'),('pitch_flat','Flattened pitch'),('envelope_compress','Envelope compression'),('duration_080','Duration ×0.8'),('duration_125','Duration ×1.25')]],
      'Controlled-probe results with 95% intervals. A token ratio of 1 means no count change relative to the named comparison. Boundary F1 measures location agreement, not recognition performance.')
    part+=paragraph(f"With duration multiplied by 1.25, token counts increase by {100*(p['comparisons']['asr_3408']['duration_125']['20']['token_ratio']['mean']-1):.1f}% for ASR, {100*(sur['duration_125']['20']['token_ratio']['mean']-1):.1f}% for emotion, {100*(p['comparisons'][intent]['duration_125']['20']['token_ratio']['mean']-1):.1f}% for intent and {100*(p['comparisons'][cnt]['duration_125']['20']['token_ratio']['mean']-1):.1f}% for count. A perfectly time-warped fixed set of landmarks would preserve token count; a fixed clock rate would add approximately 25%. These policies fall between those reference behaviors, with count closer to retaining a fixed set of onsets.")
    part+=paragraph('Pitch changes preserve most boundary locations and nearly preserve the number of tokens. Thus a plausible paper interpretation is sensitivity to temporal and articulatory realization around a stable set of acoustic transitions. The probes do not show that pitch is ignored by the encoder or pooled representation, and they do not establish unchanged human emotion perception.')
    body+=r.section('probes','4 · What changes when pitch and duration are manipulated separately?',part)

    part=paragraph('The examples are backed by the entire predefined panel. Local phone mapping increases agreement substantially over a whole-utterance duration warp. It also mechanically aligns phone edges, which is why Figure 7 includes a control that keeps the same number of cuts in each corresponding phone and randomizes only their within-phone positions.')
    part+=r.figure('style_agreement','Default versus each rendition on matching local phone sequences. Observed F1 is compared with 40 within-phone random placements preserving the rendition’s phone allocation and cut count. Each point averages60 pairs; intervals bootstrap the60 speaker–text blocks. All cuts are internal; no forced utterance endpoints are scored.','Same-text agreement across styles for four task policies, with within-phone random-placement controls.')
    part+=r.table(['Rendition vs default','Emotion F1 (%)','Same-phone random placement (%)','Word-only warp (%)','Whole-duration warp (%)'],[
      (s.title(),*[est(st['agreement'][em][s][m]) for m in ['f1','phone_null_f1','word_warp_f1','duration_warp_f1']]) for s in STYLES[1:]],
      'Mean per-pair 20 ms agreement with 95% block-bootstrap intervals. The controls differ deliberately: phone randomization tests sub-phone placement; word/whole-duration warps test whether coarse time normalization suffices.')
    diff=st['agreement']['emotion_vs_asr'][em]
    part+=paragraph(f"Emotion’s raw F1 on the three main affective contrasts is {abs(diff['value']['mean'])*100:.1f} points below ASR. However, its margin over the within-phone random-placement control is {diff['adjusted_delta']['mean']*100:.1f} points larger. Raw cross-policy F1 therefore does not establish that emotion adaptation makes the boundary policy more emotion-sensitive. Sparse cut sets change the agreement baseline. The stronger result is persistent phone-relative placement within each policy, together with selective additions and deletions.")
    body+=r.section('panel','5 · How often do the examples’ patterns occur?',part)

    gap=ex['quiet_gap'][0];ruid=gap[1];pred=next(z for z in map(json.loads,(OLD/'expresso/predictions.jsonl').read_text().splitlines()) if z['uid']==ruid)
    part=paragraph(f"The pause example contains a {1000*(gap[5]-gap[4]):.0f}ms gap between <em>{gap[2]}</em> and <em>{gap[3]}</em>, with interior RMS {abs(gap[6]):.1f}dB below the utterance’s75th-percentile frame RMS. Figure 8 shows whether cuts flank the quiet region, rather than counting silence frames as evidence of a sentence unit.")
    part+=r.figure('pause_example','Same waveform and four policies around an automatically aligned, acoustically quiet inter-word gap (green shading). This is a selected illustrative case. The supporting audit retains the larger gap statistics and their silence-quality checks.','Quiet pause after Street before but with ASR, emotion, intent and count boundaries.')
    prow=[]
    for k in PRIMARY:
        ct=[v*.02 for v in pred['predictions'][k]];before=max([x for x in ct if x<=gap[4]],default=0);after=min([x for x in ct if x>=gap[5]],default=999)
        inside=[x for x in ct if gap[4]<x<gap[5]]
        prow.append((LABELS[k],f'{before:.2f}',', '.join(f'{x:.2f}' for x in inside) or 'none',f'{after:.2f}',f'{after-before:.2f}' if not inside else 'multiple segments'))
    part+=r.table(['Policy','Last pre-gap cut (s)','Cuts inside gap','First post-gap cut (s)','Span between flanking cuts (s)'],prow,
      'Exact cuts around the 420 ms quiet gap in Figure 8. A lack of cuts inside the gap does not itself imply a separate silence token; flanking cut positions determine whether the pooled segment also includes speech.')
    part+=paragraph('The existing broader audit found very few strictly isolated quiet-gap tokens. These examples are more compatible with silence being absorbed into a neighboring acoustic span than with a special silence unit. The short-read panel has no manually annotated prosodic phrase boundaries and no speaker-change events, so it cannot establish sentence segmentation, phrasing accuracy, or a speaker-change mechanism.')
    body+=r.section('pauses','6 · Pauses and sentence structure: what the panel can tell us',part)

    part='<div class="status">Interpretation and manuscript proposal · not applied</div>'
    part+=paragraph('I would make the non-ASR addition about <strong>which transitions survive task adaptation</strong>, rather than about a new linguistic unit or a headline accuracy gain. A compact subsection near the end of the experiments could combine the transition-direction panel with a small same-text example. The controlled duration result gives the most useful complement: retained landmark locations are stable, but the number of intervening cuts can change with temporal realization.')
    part+='<blockquote>On matched Expresso utterances, adapted policies largely retain shared acoustic transitions but differ in their selection: the speaker-count checkpoint strongly favors consonant-to-vowel transitions, while emotion recognition retains a broader set of transitions and within-phone cuts. Across expressive renditions of the same text, many phone-relative locations persist, with additional cuts associated with temporal stretching; controlled duration changes alter token allocation more than pitch shifts.</blockquote>'
    part+=paragraph('This is proposed wording for the analysis paragraph, subject to space and author review. I would retain the current ASR abstract and oracle comparison until the non-ASR presentation is agreed. A task-causal claim would still require a matched-rate, matched-training-stage experiment. A syllable or prosodic-phrase claim would require corresponding annotations. No manuscript files were edited for this study.')
    body+=r.section('paper','7 · How I would use the evidence in the paper',part)

    part=paragraph(f"<strong>Panel and alignment.</strong> Four Expresso speakers, 15 texts per speaker, seven renditions each. All {q['pairs']} default–rendition pairs have the same word sequence. {q['matching_phone_words']}/{q['words']} corresponding words ({100*q['matching_phone_words']/q['words']:.2f}%) have matching stress-stripped phone sequences; only these words enter phone-wise mapping. {q['whole_phone_sequences']}/{q['pairs']} pairs match over the whole non-gap phone sequence. The eligible mappings retain 94.4–95.0% of the compared cuts, depending on policy; no comparison has two empty cut sets. Word/gap boundaries come from MFA and are automatic, especially uncertain on whisper or laughter.")
    part+=paragraph('<strong>Timing and matching.</strong> Internal cut time is frame index × 20 ms; compulsory endpoints are excluded. Matches use a one-to-one maximum-cardinality rule. ASR-anchor attribution additionally minimizes timing distance. Anchor matches use 20 ms; phone-class assignment is checked at 20/40 ms and with +12.5 ms cut timestamps. For same-text comparisons, each corresponding phone is mapped linearly onto its default-rendition interval. Gaps and nonmatching phone sequences are excluded.40 within-phone randomizations preserve phone occupancy;20 clock-grid phases provide an additional baseline. Both-empty cut sets receive agreement 1; denominators and coverage remain in the per-pair CSV.')
    part+=paragraph('<strong>Acoustics.</strong>25 ms RMS windows at 20 ms hops, smoothed by a Gaussian of 20 ms standard deviation; envelope amplitudes normalized by the utterance 95th percentile. Rising/falling landmarks are peaks of the signed envelope derivative, with 80 ms minimum separation and prominence 1 normalized amplitude/s; envelope peaks/valleys use prominence 0.12. These explicit diagnostics are inspired by envelope-landmark work, not an exact replication of its extraction method or a harmonic-energy measure. Lexical stress in MFA is not an annotation of expressive prominence. Praat pitch estimates sometimes show octave errors, so natural local pitch traces are treated cautiously.')
    ck=json.loads((OLD/'checkpoint_audit.json').read_text())['checkpoints']
    part+=r.table(['Primary policy','Training checkpoint','Final-block updates','Phase-C rate band','Boundary selection'],[
      (LABELS[k],str(ck[k].get('step','ASR parent')),str(ck[k].get('final_block_updates','—')),{'asr_3408':'ASR training','emotion_3408':'2–5 Hz','intent_3408_last':'4–7 Hz','speaker_count_3408_last':'3–6 Hz'}[k],ck[k]['role']) for k in PRIMARY],
      'Checkpoint scope. All task warm starts use the same ASR parent. Emotion seeds 3407/3409 are sensitivity controls. Previously selected FiLM-only intent/count checkpoints are retained in the supporting audit, not substituted for the final-block-adapted policies here.')
    ac=p['acoustic_qc'];part+=r.table(['Probe','Measured duration ratio','Median F0 shift (semitones)','Envelope correlation','Voicing agreement (%)'],[
      (c,f"{ac[c]['duration_ratio']:.3f}",f"{ac[c]['f0_median_shift_semitones']:.3f}",f"{ac[c]['envelope_correlation']:.3f}",f"{100*ac[c]['voicing_agreement']:.1f}") for c in ['sham','pitch_down3','pitch_up3','duration_080','duration_125','pitch_flat','envelope_compress']],
      'Median acoustic manipulation checks over 60 source utterances. Pitch/duration rows compare with sham; envelope and sham compare with original. Envelope/time measurements use the known duration map. Preservation is approximate, not an assertion that all non-target cues are identical.')
    part+=paragraph(f"<strong>Numerics and uncertainty.</strong> Frozen WavLM features are extracted per utterance; only policies are batched. BF16 autoregressive thresholds can amplify small numerical differences: {p['reproduction']['exact']}/{p['reproduction']['records']} original policy tracks reproduce the prior cached cuts exactly; the other 9 differ by 1–4 cut locations. All probe contrasts use originals recomputed in the same run, not the old cache. An additional batch-size 1 check covers all 48 conditions from the six affected source recordings. Confidence intervals resample 60 speaker–text blocks 2,000 times and are conditional on these four speakers. This is an exploratory analysis, not a population-level estimate of emotion behavior.")
    if (OUT/'batch_sensitivity_summary.json').exists():
        bs=read('batch_sensitivity_summary.json');part+=paragraph(f"The affected-recording batch check reproduces {bs['exact_tracks']}/{bs['tracks']} tracks exactly; the largest absolute change to a 60-recording aggregate boundary-F1 mean after substituting those six recordings is {100*bs['max_full_panel_f1_change']:.2f} percentage points. It does not change the pitch-versus-duration interpretation.")
    body+=r.section('methods','8 · Reproducibility and limits',part)

    part='<ol class="sources">'
    sourceitems=[('https://www.isca-archive.org/interspeech_2023/nguyen23_interspeech.pdf','Nguyen et al. (2023), Expresso','The source dataset motivates paired content/style comparisons. This panel uses short reads, not the full expressive or conversational coverage.'),
      ('https://pmc.ncbi.nlm.nih.gov/articles/PMC6957234/','Oganian & Chang (2019), A speech envelope landmark for syllable encoding in human superior temporal gyrus','Motivates testing rising envelope edges separately from peaks and valleys. The present measurements do not establish a neural analogue or syllable encoding.'),
      ('https://arxiv.org/abs/2110.02345','Bhati et al. (2021), Segmental Contrastive Predictive Coding','Motivates conditioning boundary agreement on phonetic transition classes instead of relying only on an aggregate F1.'),
      ('https://www.isca-archive.org/speechprosody_2008/mo08b_speechprosody.html','Mo (2008), Duration and intensity as perceptual cues','Motivates separating duration from intensity and avoiding a direct inference from lengthening to prominence or phrasing.'),
      ('https://parselmouth.readthedocs.io/en/stable/examples/pitch_manipulation.html','Parselmouth pitch-manipulation documentation','Primary documentation for Praat pitch-tier manipulation and overlap-add resynthesis; the code also changes duration tiers and retains a sham control.')]
    for url,title,desc in sourceitems:part+=f'<li><a href="{url}">{esc(title)}</a>. {desc}</li>'
    part+='</ol>'
    part+=f'<p><a class="button" href="{ASSETS}/pattern-evidence.zip">Download pattern evidence and analysis scripts</a> <a class="button" href="nonasr-task-boundaries-performance-audit.html">Open the earlier performance and aggregate audit</a></p>'
    part+=paragraph('The download includes all numerical summaries, per-utterance and per-phone tables, exact cuts/references, checkpoint identifiers, acoustic-probe QC, protocol and reproducible scripts. Audio and checkpoints are not redistributed. Each static figure has a separate PDF download. The earlier audit retains CREMA-D/LibriSpeech results, selected-checkpoint quality, quiet-gap controls and its original downloads.')
    body+=r.section('sources','9 · Primary sources and supporting evidence',part)
    data=explorer_data();body+='<script type="application/json" id="explorer-data">'+json.dumps(data,separators=(',',':')).replace('<','\\u003c')+'</script>'
    body+='<script>'+(RECIPE/'expresso_pattern_explorer.js').read_text()+'</script>'
    body+='<footer>Generated from retained evidence · manuscript unchanged · '+VERSION+'</footer>'
    full,_=hk.document('Expresso: shared acoustic landmarks and task-specific boundary preferences',body,head_html='<style>'+CSS+EXTRA_CSS+'</style>')
    # Add spaces around inline numerical prose without changing values or notation in code/data.
    import re
    head,tail=full.split('<script type="application/json" id="explorer-data">',1)
    dest.write_text(head+'<script type="application/json" id="explorer-data">'+tail)
    hashes=package(r)
    (OUT/'html/report_manifest.json').write_text(json.dumps(dict(version=VERSION,tables=r.tables,figures=r.figures,source_sha256=hashes,explorer_utterances=len(data['utterances']),explorer_groups=60),indent=2)+'\n')
    print(json.dumps(dict(output=str(dest),bytes=dest.stat().st_size,tables=r.tables,figures=r.figures)))
if __name__=='__main__':build()
