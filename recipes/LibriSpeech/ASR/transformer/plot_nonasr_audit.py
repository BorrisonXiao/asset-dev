#!/usr/bin/env python3
"""Generate standalone scientific figures; do not edit the paper."""
import json
import os
import textwrap
from pathlib import Path
import numpy as np
from audit_nonasr_boundaries import OUT,SEEDS,dump
os.environ.setdefault('MPLCONFIGDIR',str(OUT/'tools/matplotlib_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import spectrogram
from audit_segment_units import pauses

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,
                     'axes.spines.top':False,'axes.spines.right':False,'axes.titlesize':10})
ASR='#2671a8';EMOTION='#c75d18';GRAY='#747b86';CONF='#4c7f60'
FIG=OUT/'figures'


def save(fig,name):
    FIG.mkdir(parents=True,exist_ok=True)
    fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(FIG/(name+'.png'),dpi=180,bbox_inches='tight')
    plt.close(fig)


def overview(summary):
    corpora=['cremad','expresso','librispeech'];labels=['CREMA-D','Expresso','LibriSpeech']
    fig,axes=plt.subplots(2,2,figsize=(10,7.2),layout='constrained')
    x=np.arange(3)
    rates=[summary[c]['systems']['asr_3408']['mean_utterance_hz'] for c in corpora]
    emotion=np.array([[summary[c]['systems'][f'emotion_{s}']['mean_utterance_hz'] for s in SEEDS] for c in corpora])
    ax=axes[0,0];ax.bar(x-.18,rates,.34,color=ASR,label='ASR parent')
    ax.bar(x+.18,emotion.mean(1),.34,yerr=emotion.std(1,ddof=1),capsize=3,color=EMOTION,label='Emotion, three seeds')
    ax.set(xticks=x,xticklabels=labels,ylabel='Mean tokens/s',title='(a) Same audio, coarser emotion segmentation',ylim=(0,13))
    ax.legend(frameon=False,fontsize=8)
    ax=axes[0,1]
    learned=np.array([[summary[c]['pairs'][f'asr_3408/emotion_{s}']['b_in_a_20ms'] for s in SEEDS] for c in corpora])*100
    null=np.array([[summary[c]['pairs'][f'asr_3408/emotion_{s}']['shift_null_b_in_a_20ms'] for s in SEEDS] for c in corpora])*100
    ax.bar(x-.18,learned.mean(1),.34,color=EMOTION,label='Observed')
    ax.bar(x+.18,null.mean(1),.34,color=GRAY,label='Circular-shift null')
    ax.set(xticks=x,xticklabels=labels,ylabel='Emotion cuts near ASR cuts (%)',ylim=(0,115),title='(b) Shared boundary locations (±20 ms)')
    ax.legend(frameon=False,fontsize=8,loc='upper center',ncol=2)
    ax=axes[1,0];names=['phone_20ms','vowel_onset_20ms','voicing_20ms']
    systems=[('scores','Emotion',EMOTION),('asr_thin_scores','Random ASR thinning',GRAY),('confidence_thin_scores','Confident ASR thinning',CONF)]
    for j,(key,label,color) in enumerate(systems):
        vals=np.array([[100*summary['expresso']['systems'][f'emotion_{s}'][key][m]['f1'] for s in SEEDS] for m in names])
        ax.bar(np.arange(3)+(j-1)*.24,vals.mean(1),.23,yerr=vals.std(1,ddof=1),capsize=2,color=color,label=label)
    ax.set(xticks=np.arange(3),xticklabels=['Phone edge','Vowel onset','Voicing change'],ylabel='Boundary F1 (%)',ylim=(0,70),title='(c) Expresso: controls with identical cut counts')
    ax.legend(frameon=False,fontsize=7,loc='upper right')
    ax=axes[1,1]
    stats=json.loads((OUT/'expresso/per_utterance_statistics.json').read_text())
    for keys,label,color in [(['asr_3408'],'ASR parent',ASR),([f'emotion_{s}' for s in SEEDS],'Emotion, pooled seeds',EMOTION)]:
        dur=np.sort([d for k in keys for r in stats[k] for d in r['segment_ms']])
        ax.plot(dur,np.arange(1,len(dur)+1)/len(dur),color=color,label=label,lw=2)
    ax.set(xlabel='Segment duration (ms)',ylabel='Cumulative fraction',xlim=(0,800),ylim=(0,1),title='(d) Longer segments on Expresso')
    ax.legend(frameon=False,fontsize=8)
    fig.suptitle('Task adaptation changes which shared boundaries are retained',fontsize=13)
    save(fig,'boundary_overview')


def agreement(controls):
    d=controls['expresso']['agreement'];a=np.array(d['f1_20ms'])*100
    labels=['ASR\nparent','Emotion\nselected','Intent\nselected (FiLM)','Count\nselected (FiLM)','Intent\nlast step','Count\nlast step']
    fig,ax=plt.subplots(figsize=(7.4,6),layout='constrained')
    im=ax.imshow(a,vmin=0,vmax=100,cmap='Blues')
    for i in range(len(a)):
        for j in range(len(a)):
            ax.text(j,i,f'{a[i,j]:.1f}',ha='center',va='center',color='white' if a[i,j]>65 else '#182b3f',fontsize=9)
    ax.set_xticks(range(6),['ASR\nparent','Emotion\nselected','Intent\nFiLM','Count\nFiLM','Intent\nlast step','Count\nlast step'],fontsize=8)
    ax.set_yticks(range(6),[f"{lab.replace(chr(10),' ')}  ({d['mean_hz'][key]:.2f} Hz)" for lab,key in zip(labels,d['keys'])],fontsize=8)
    ax.set_title('Expresso: agreement of different task checkpoints\nOne-to-one boundary F1 at ±20 ms; seed 3408, 420 identical utterances',pad=12)
    fig.colorbar(im,ax=ax,shrink=.75,label='Boundary F1 (%)')
    save(fig,'cross_task_agreement')


def example(row,pred,stats,number,reason):
    duration=row['duration'];x,sr=sf.read(row['wav']);f,t,s=spectrogram(x,sr,nperseg=400,noverlap=240)
    db=10*np.log10(np.maximum(s,1e-12));db-=db.max()
    fig,axes=plt.subplots(3,1,figsize=(11,4.9),sharex=True,gridspec_kw={'height_ratios':[1.4,.8,1.2]})
    fig.subplots_adjust(left=.24,right=.99,bottom=.11,top=.83,hspace=.15)
    gaps=pauses(row['words'])
    view_start,view_end=0.,duration
    if number==4 and gaps:
        gap=max(gaps,key=lambda g:g[1]-g[0])
        view_start=max(0.,gap[0]-1.);view_end=min(duration,gap[1]+1.5)
    axes[0].pcolormesh(t,f/1000,db,shading='auto',cmap='Greys',vmin=-65,vmax=0)
    axes[0].set(ylim=(0,4),ylabel='kHz')
    axes[1].set(ylim=(-.2,1.9),yticks=[.3,1.3],yticklabels=['Phones','Words'])
    for tier,y in [('phones',.3),('words',1.3)]:
        for start,end,label in row[tier]:
            if not label or end<view_start or start>view_end:continue
            axes[1].plot([start,end],[y,y],lw=8,color='#dce6ef',solid_capstyle='butt')
            axes[1].plot([start,start],[y-.23,y+.23],color='#8b9aab',lw=.6)
            axes[1].text((start+end)/2,y,label,ha='center',va='center',fontsize=6 if tier=='phones' else 7,clip_on=True)
    tracks=[('ASR parent',pred['predictions']['asr_3408'],ASR)]
    tracks += [(f'Emotion {seed}',pred['predictions'][f'emotion_{seed}'],EMOTION) for seed in SEEDS]
    conf=stats['confidence_thin_cuts']
    tracks.append(('ASR confidence control',conf,CONF))
    for i,(label,cuts,color) in enumerate(tracks):
        y=len(tracks)-i
        axes[2].hlines(y,0,duration,color='#dddddd',lw=.7)
        axes[2].vlines(np.array(cuts)*.02,y-.3,y+.3,color=color,lw=1)
    count_label='total tokens' if number==4 else 'tokens'
    axes[2].set(yticks=np.arange(1,6),yticklabels=[f'{name} ({len(cuts)+1} {count_label})' for name,cuts,_ in tracks[::-1]],ylim=(.4,5.6),xlabel='Time (s)',xlim=(view_start,view_end))
    axes[2].tick_params(axis='y',labelsize=7)
    for gap in pauses(row['words']):
        for ax in axes:ax.axvspan(gap[0],gap[1],color='#e2bc5b',alpha=.2,zorder=2)
    for ax in axes:ax.spines['left'].set_visible(False)
    excerpt=' (inter-word gap excerpt)' if number==4 else ''
    fig.suptitle(f"Example {number}: {row['uid']} — {row['style']}{excerpt}\n"+'\n'.join(textwrap.wrap(row['transcript'],100)),fontsize=10)
    name=f'example_{number}_{row["uid"]}'
    save(fig,name)
    return dict(number=number,uid=row['uid'],corpus=row['corpus'],style=row['style'],reason=reason,
                transcript=row['transcript'],figure='figures/'+name+'.pdf',
                tokens={k:len(v)+1 for k,v in pred['predictions'].items()},
                words=row['words'],phones=row['phones'],pauses=pauses(row['words']))


def examples():
    rows=json.loads((OUT/'expresso/references.json').read_text())
    preds={r['uid']:r for r in (json.loads(l) for l in (OUT/'expresso/predictions.jsonl').read_text().splitlines())}
    stats={r['uid']:r for r in json.loads((OUT/'expresso/per_utterance_statistics.json').read_text())['emotion_3408']}
    candidates=sorted((r for r in rows if r['speaker']=='ex01' and r['style']=='default'),key=lambda r:r['duration'])
    base=candidates[len(candidates)//2]
    chosen=[next(r for r in rows if r['speaker']==base['speaker'] and r['text_group']==base['text_group'] and r['style']==style) for style in ['default','whisper','laughing']]
    pause_candidates=sorted((r for r in rows if pauses(r['words'])),key=lambda r:max(g[1]-g[0] for g in pauses(r['words'])))
    pause_row=pause_candidates[len(pause_candidates)//2]
    if pause_row['uid'] in {r['uid'] for r in chosen}:pause_row=pause_candidates[len(pause_candidates)//2+1]
    chosen.append(pause_row)
    output=[]
    for i,row in enumerate(chosen,1):
        reason='Median-duration ex01 default prompt and its matched whisper/laughing renditions' if i<=3 else 'Median longest-pause duration among Expresso utterances containing an aligned pause >=250 ms'
        output.append(example(row,preds[row['uid']],stats[row['uid']],i,reason))
    dump(output,OUT/'qualitative_examples.json')


if __name__=='__main__':
    summary=json.loads((OUT/'boundary_summary.json').read_text())['corpora']
    controls=json.loads((OUT/'control_sensitivity.json').read_text())
    overview(summary);agreement(controls);examples()
    print('Wrote six scientific figures (PDF and PNG), plus example provenance')
