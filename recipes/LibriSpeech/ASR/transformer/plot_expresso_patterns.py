#!/usr/bin/env python3
"""Publication-ready pattern figures; values loaded from recorded evidence."""
import os,json
from pathlib import Path
import numpy as np
from analyze_expresso_patterns import ROOT,OLD,OUT,PRIMARY,STYLES,block,cuts,word_phones,features,regions,warp
os.environ.setdefault('MPLCONFIGDIR',str(OUT/'tools/matplotlib_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import soundfile as sf
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.titlesize':11})
COLORS=['#2671a8','#c75d18','#5e5596','#24794f'];LABELS=['ASR parent','Emotion','Intent (last step)','Count (last step)']
FIG=OUT/'figures'
def save(fig,name):
    FIG.mkdir(parents=True,exist_ok=True);fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight');fig.savefig(FIG/(name+'.png'),dpi=170,bbox_inches='tight');plt.close(fig)

def tiers(ax,row,xlim=None):
    for tier,y in [('phones',.25),('words',1.15)]:
        for a,b,l in row[tier]:
            if not l:continue
            ax.plot([a,b],[y,y],lw=12 if tier=='words' else 10,color='#e1e9f1',solid_capstyle='butt')
            ax.plot([a,a],[y-.23,y+.23],color='#a3b2c0',lw=.6)
            ax.text((a+b)/2,y,l,ha='center',va='center',fontsize=6.5 if tier=='phones' else 9,rotation=90 if tier=='phones' and b-a<.065 else 0,clip_on=True)
    ax.set(ylim=(-.2,1.7),yticks=[.25,1.15],yticklabels=['Phones','Words']);ax.spines['bottom'].set_visible(False)

def same_audio(r,p,pr,filename,quiet=None):
    y,sr=sf.read(r['wav']);f,t,s=spectrogram(y,sr,nperseg=400,noverlap=240);db=10*np.log10(np.maximum(s,1e-12));db-=db.max()
    fig,axes=plt.subplots(4,1,figsize=(11,6.5),sharex=True,gridspec_kw={'height_ratios':[1.3,.7,1.1,2.]})
    fig.subplots_adjust(left=.17,right=.99,bottom=.09,top=.86,hspace=.15)
    axes[0].pcolormesh(t,f/1000,db,cmap='Greys',vmin=-65,vmax=0,shading='auto',rasterized=True);axes[0].set(ylim=(0,4),ylabel='kHz')
    feat=features(r,p,pr);axes[1].plot(feat['t']+.0125,feat['env'],color='#5e6977',lw=1.6);axes[1].set(ylabel='Envelope',yticks=[0,1])
    tiers(axes[2],r)
    for i,(k,label,col) in enumerate(zip(PRIMARY,LABELS,COLORS)):
        yy=3-i;ct=cuts(p,k)
        axes[3].hlines(yy,0,r['duration'],color='#e1e6ec',lw=.8);axes[3].vlines(ct,yy-.27,yy+.27,color=col,lw=1.7)
        axes[3].text(r['duration']+.01,yy,f'{len(ct)+1}',va='center',fontsize=8,color=col)
    axes[3].set(yticks=range(4),yticklabels=LABELS[::-1],xlabel='Time in the original recording (s)',ylim=(-.55,3.55),xlim=(0,r['duration']))
    if quiet:
        for ax in axes:ax.axvspan(quiet[0],quiet[1],color='#9ebcae',alpha=.22)
    if r['uid']=='ex03_sad_00134':
        for ax in axes:ax.axvspan(.89,1.02,color='#edc779',alpha=.22)
    fig.suptitle(f'“{r["transcript"]}”\n{r["uid"]} · same audio, four frozen policies',fontsize=13)
    save(fig,filename)

def retention(s,e):
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.1),layout='constrained')
    ax=axes[0];cats=['consonant → vowel','vowel → consonant']
    for i,k in enumerate(PRIMARY[1:]):
        vals=[e['transitions']['0.0']['20'][c][k] for c in cats];m=np.array([v['mean'] for v in vals])*100;ci=np.array([v['ci95'] for v in vals]).T*100
        ax.bar(np.arange(2)+(i-1)*.24,m,.23,color=COLORS[i+1],label=LABELS[i+1],yerr=[m-ci[0],ci[1]-m],capsize=3)
    ax.set(xticks=[0,1],xticklabels=['Consonant → vowel','Vowel → consonant'],ylim=(0,103),ylabel='Available ASR cuts retained (%)',title='(a) Direction of the phonetic transition')
    ax.legend(frameon=False,fontsize=8)
    ax=axes[1]
    for i,k in enumerate(PRIMARY[1:]):
        vals=[s['parent_retention']['slope_quartile'][str(q)][k] for q in range(4)];m=np.array([v['mean'] for v in vals])*100;ci=np.array([v['ci95'] for v in vals]).T*100
        ax.errorbar(range(4),m,yerr=[m-ci[0],ci[1]-m],color=COLORS[i+1],fmt='o-',capsize=3,label=LABELS[i+1])
    ax.set(xticks=range(4),xticklabels=['Most falling','Q2','Q3','Most rising'],ylim=(0,100),ylabel='Available ASR cuts retained (%)',title='(b) Local amplitude-envelope slope')
    fig.suptitle('Task policies retain different kinds of acoustic transitions',fontsize=13)
    save(fig,'transition_preferences')

def same_text(refs,preds,group):
    styles={r['style']:r for r in refs if block(r)==group};d=styles['default'];wd=word_phones(d)
    fig,axes=plt.subplots(8,1,figsize=(11,8.3),sharex=True,gridspec_kw={'height_ratios':[1.2]+[1]*7})
    fig.subplots_adjust(left=.22,right=.985,bottom=.075,top=.90,hspace=.16)
    tiers(axes[0],d)
    for i,s in enumerate(STYLES):
        r=styles[s];p=preds[r['uid']];ax=axes[i+1];regs=regions(wd,word_phones(r),'phone')
        ct,_=warp(cuts(p,'emotion_3408'),regs)
        # Baseline uses identical domain restriction to each rendition.
        dc,_=warp(warp(cuts(preds[d['uid']],'emotion_3408'),regs,True)[0],regs)
        ax.vlines(dc,0,1,color='#c3cad2',lw=1,linestyle='dotted')
        ax.vlines(ct,0,1,color=COLORS[1],lw=2)
        ax.hlines(.5,0,d['duration'],color='#e5e9ee',lw=.6,zorder=-1)
        ax.set(ylim=(-.12,1.12),yticks=[.5],yticklabels=[f'{s.title()}\n{r["duration"]:.2f}s · {len(cuts(p,"emotion_3408"))+1} tokens'])
        ax.tick_params(axis='y',length=0,labelsize=9)
        for a,b,c,e,*_ in regs:ax.axvspan(c,e,alpha=.035,color='black')
    axes[-1].set(xlabel='Default-rendition time after mapping corresponding phones (s)',xlim=(0,d['duration']))
    fig.suptitle(f'“{d["transcript"]}” · {d["speaker"]}\nEmotion policy on seven renditions of the same text',fontsize=13)
    save(fig,'same_text_example')

def style_stats(s):
    fig,axes=plt.subplots(2,2,figsize=(10.5,7),layout='constrained')
    styles=STYLES[1:]
    for ax,k,col,title in zip(axes.flat,PRIMARY,COLORS,LABELS):
        for metric,label,fmt,c in [('f1','Observed','o-',col),('phone_null_f1','Same-phone random placement','s--','#9098a2')]:
            vals=[s['agreement'][k][st][metric] for st in styles];m=np.array([v['mean'] for v in vals])*100;ci=np.array([v['ci95'] for v in vals]).T*100
            ax.errorbar(range(6),m,yerr=[m-ci[0],ci[1]-m],fmt=fmt,color=c,capsize=2,label=label)
        ax.axvline(2.5,color='#c8d0d8',lw=1,linestyle=':');ax.set(xticks=range(6),xticklabels=['Happy','Sad','Confused','Whisper','Laughing','Enunciated'],ylim=(0,100),ylabel='Same-text boundary F1 (%)',title=title)
        ax.tick_params(axis='x',rotation=25,labelsize=8)
    axes[0,0].legend(frameon=False,fontsize=8)
    fig.suptitle('Shared phone-relative locations persist across expressive renditions\nDefault versus each rendition; 60 speaker–text groups',fontsize=13)
    save(fig,'style_agreement')

def probe_plot(s):
    cs=['sham','pitch_down3','pitch_up3','pitch_flat','envelope_compress','duration_080','duration_125'];labs=['Sham','Pitch −3 st','Pitch +3 st','Flat pitch','Envelope\ncompression','Duration\n×0.8','Duration\n×1.25']
    fig,axes=plt.subplots(2,1,figsize=(10.5,7.2),sharex=True,layout='constrained')
    for k,label,col in zip(PRIMARY,LABELS,COLORS):
        for ax,metric,scale in [(axes[0],'f1',100),(axes[1],'token_ratio',1)]:
            vals=[s['comparisons'][k][c]['20'][metric] for c in cs];m=np.array([v['mean'] for v in vals])*scale;ci=np.array([v['ci95'] for v in vals]).T*scale
            ax.errorbar(np.arange(len(cs))+(PRIMARY.index(k)-1.5)*.07,m,yerr=[m-ci[0],ci[1]-m],fmt='o-',capsize=2,color=col,label=label)
    axes[0].set(ylabel='Boundary F1 (%)',ylim=(0,103));axes[0].legend(frameon=False,fontsize=9,ncol=2)
    axes[1].axhline(1,color='#8796a4',lw=.8,linestyle=':');axes[1].set(ylabel='Number of tokens / reference',xticks=range(len(cs)),xticklabels=labs)
    fig.suptitle('Controlled changes to the same 60 default recordings\nPitch/duration versus resynthesis sham; sham/envelope versus original',fontsize=13)
    save(fig,'controlled_acoustic_probes')

if __name__=='__main__':
    refs=json.loads((OLD/'expresso/references.json').read_text());byid={r['uid']:r for r in refs}
    preds={r['uid']:r for r in map(json.loads,(OLD/'expresso/predictions.jsonl').read_text().splitlines())}
    pro={r['uid']:r for r in map(json.loads,(OLD/'expresso/prosody.jsonl').read_text().splitlines())}
    s=json.loads((OUT/'same_audio_summary.json').read_text());st=json.loads((OUT/'same_text_summary.json').read_text());e=json.loads((OUT/'extended_summary.json').read_text());ex=json.loads((OUT/'selected_examples.json').read_text())
    r=byid['ex03_sad_00134'];same_audio(r,preds[r['uid']],pro[r['uid']],'same_audio_example')
    if ex['quiet_gap']:
        g=ex['quiet_gap'][0];r=byid[g[1]];same_audio(r,preds[r['uid']],pro[r['uid']],'pause_example',g[4:6])
    retention(s,e);same_text(refs,preds,ex['same_text_group']);style_stats(st)
    if (OUT/'probe_summary.json').exists():
        ps=json.loads((OUT/'probe_summary.json').read_text())
        if 'comparisons' in ps:probe_plot(ps)
    print('Figures complete')
