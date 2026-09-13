/* Self-contained boundary explorers. Input contains numerical evidence, no audio. */
(()=>{
const D=JSON.parse(document.getElementById('explorer-data').textContent);
const $=id=>document.getElementById(id), NS='http://www.w3.org/2000/svg';
const styles=['default','happy','sad','confused','whisper','laughing','enunciated'];
const keys=['asr_3408','emotion_3408','intent_3408_last','speaker_count_3408_last'];
const labels={'asr_3408':'ASR parent','emotion_3408':'Emotion 3408','emotion_3407':'Emotion 3407','emotion_3409':'Emotion 3409','intent_3408_last':'Intent · last step','speaker_count_3408_last':'Count · last step'};
const cols=['var(--series-sed)','var(--pair-sed-asr)','var(--pair-sed-speaker)','var(--series-asr)'];
const groups=[...new Set(D.utterances.map(r=>r.block))].sort();
const rows=new Map(D.utterances.map(r=>[r.block+'|'+r.style,r]));
const mapped=new Map(D.mapped.map(r=>[r.block+'|'+r.style,r]));
function el(tag,attrs={},text=''){let x=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(attrs))x.setAttribute(k,v);if(text)x.textContent=text;return x;}
function add(svg,tag,attrs,text){const n=el(tag,attrs,text);svg.appendChild(n);return n;}
function option(sel,value,label){const o=document.createElement('option');o.value=value;o.textContent=label;sel.appendChild(o);}
function initSvg(id,h){const svg=$(id);svg.replaceChildren();svg.setAttribute('viewBox',`0 0 1040 ${h}`);svg.setAttribute('height',h);return svg;}
function axis(svg,x,a,b,y){add(svg,'line',{x1:x(a),x2:x(b),y1:y,y2:y,stroke:'var(--text-muted)'});const step=(b-a)>3?.5:.2;for(let t=Math.ceil(a/step)*step;t<=b+.001;t+=step){add(svg,'line',{x1:x(t),x2:x(t),y1:y,y2:y+5,stroke:'var(--text-muted)'});add(svg,'text',{x:x(t),y:y+20,'text-anchor':'middle',class:'svgt'},t.toFixed(1));}}
function tier(svg,r,x,a,b){for(const[name,y]of [['words',23],['phones',59]]){add(svg,'text',{x:135,y:y+4,'text-anchor':'end',class:'svgt'},name==='words'?'Words':'MFA phones');for(const[s,e,l]of r[name]){if(!l||e<a||s>b)continue;const xx=x(Math.max(s,a)),ww=x(Math.min(e,b))-xx;add(svg,'rect',{x:xx,y:y-12,width:Math.max(0,ww-1),height:24,fill:'var(--surface-2)'});if(ww>l.length*5)add(svg,'text',{x:xx+ww/2,y:y+4,'text-anchor':'middle',class:'svgt small'},l);else{const rec=add(svg,'rect',{x:xx,y:y-12,width:Math.max(0,ww-1),height:24,fill:'transparent'});rec.appendChild(el('title',{},`${l} ${s.toFixed(3)}–${e.toFixed(3)} s`));}}}}
function cut(svg,t,x,y,col,tip,dashed=false){const l=add(svg,'line',{x1:x(t),x2:x(t),y1:y-14,y2:y+14,stroke:col,'stroke-width':dashed?1:2,'stroke-dasharray':dashed?'3 3':''});l.appendChild(el('title',{},tip));}
for(const id of ['audio-group','text-group']){for(const g of groups)option($(id),g,g.replace('::',' · '));}
for(const s of styles)option($('audio-style'),s,s[0].toUpperCase()+s.slice(1));
for(const k of [...keys,'emotion_3407','emotion_3409'])option($('text-policy'),k,labels[k]);
$('audio-group').value=D.initial_audio_group;$('audio-style').value='sad';$('text-group').value=D.initial_text_group;$('text-policy').value='emotion_3408';
function windows(){const r=rows.get($('audio-group').value+'|'+$('audio-style').value);$('audio-window').replaceChildren();option($('audio-window'),'all','Whole utterance');r.words.forEach((w,i)=>{if(w[2])option($('audio-window'),String(i),`${w[2]} · ${w[0].toFixed(2)}–${w[1].toFixed(2)} s`);});audio();}
function audio(){const r=rows.get($('audio-group').value+'|'+$('audio-style').value),win=$('audio-window').value;let a=0,b=r.duration;if(win!=='all'){const w=r.words[Number(win)];a=Math.max(0,w[0]-.15);b=Math.min(r.duration,w[1]+.15);}const kk=$('audio-seeds').checked?[...keys,'emotion_3407','emotion_3409']:keys;const h=180+kk.length*46,svg=initSvg('audio-svg',h),x=t=>160+(t-a)/(b-a)*850;tier(svg,r,x,a,b);
let points=[];r.env.forEach((v,i)=>{const t=i*.02+.0125;if(t>=a&&t<=b)points.push(`${x(t)},${115-v*35}`);});add(svg,'polyline',{points:points.join(' '),fill:'none',stroke:'var(--text-secondary)','stroke-width':1.5});add(svg,'text',{x:135,y:108,'text-anchor':'end',class:'svgt'},'Envelope');
kk.forEach((k,i)=>{const y=155+i*46;add(svg,'text',{x:135,y:y+4,'text-anchor':'end',class:'svgt'},labels[k]);add(svg,'line',{x1:160,x2:1010,y1:y,y2:y,stroke:'var(--border)'});for(const fr of r.cuts[k]){const t=fr*.02;if(t>=a&&t<=b){const near=r.phones.filter(p=>p[2]).sort((p,q)=>Math.abs(p[0]-t)-Math.abs(q[0]-t))[0];cut(svg,t,x,y,cols[i%4],`${labels[k]} · ${t.toFixed(3)} s · nearest phone onset ${near[2]} at ${near[0].toFixed(3)} s`);}}add(svg,'text',{x:1017,y:y+4,class:'svgt small'},String(r.cuts[k].length+1));});axis(svg,x,a,b,h-33);$('audio-description').textContent=`${r.uid} · ${r.transcript} · ${r.duration.toFixed(3)} s. Right-hand numbers count all tokens, including the initial segment.`;}
function textview(){const g=$('text-group').value,k=$('text-policy').value,d=rows.get(g+'|default'),ss=$('text-styles').checked?styles:styles.slice(0,4);const h=135+ss.length*60,svg=initSvg('text-svg',h),x=t=>160+t/d.duration*850;tier(svg,d,x,0,d.duration);
ss.forEach((s,i)=>{const r=rows.get(g+'|'+s),y=105+i*60;add(svg,'text',{x:135,y:y-1,'text-anchor':'end',class:'svgt'},s[0].toUpperCase()+s.slice(1));add(svg,'text',{x:135,y:y+15,'text-anchor':'end',class:'svgt small'},`${r.duration.toFixed(2)}s · ${r.cuts[k].length+1} tokens`);add(svg,'line',{x1:160,x2:1010,y1:y,y2:y,stroke:'var(--border)'});
if(s==='default'){for(const fr of d.cuts[k])cut(svg,fr*.02,x,y,'var(--text-secondary)',`Default ${fr*.02} s`);return;}
const pair=mapped.get(g+'|'+s),v=pair.policies[k];for(const region of pair.regions){add(svg,'rect',{x:x(region[2]),y:y-19,width:Math.max(0,x(region[3])-x(region[2])),height:38,fill:'var(--surface-2)',opacity:.35});}for(const t of v.default)cut(svg,t,x,y,'var(--text-muted)',`Warped default reference ${t.toFixed(3)} s`,true);
// Greedy matching of sorted positions gives a one-to-one maximum-cardinality match.
const matched=new Set();let ai=0,bi=0;while(ai<v.default.length&&bi<v.mapped_style.length){if(Math.abs(v.default[ai]-v.mapped_style[bi])<=.02000001){matched.add(bi);ai++;bi++;}else if(v.default[ai]<v.mapped_style[bi])ai++;else bi++;}
v.mapped_style.forEach((t,j)=>cut(svg,t,x,y,matched.has(j)?'var(--series-asr)':'var(--pair-sed-asr)',`${s} · canonical ${t.toFixed(3)} s · ${matched.has(j)?'matches default within 20 ms':'unmatched to default within 20 ms'}`));});axis(svg,x,0,d.duration,h-34);$('text-description').textContent=`${g.replace('::',' · ')}. Green cuts match the default within 20 ms after local phone warping; orange cuts differ; dotted lines mark default cuts. Unshaded regions lack an eligible exact phone correspondence and are not scored. Select the raw-audio explorer for their original timings.`;}
for(const id of ['audio-group','audio-style'])$(id).addEventListener('change',windows);for(const id of ['audio-window','audio-seeds'])$(id).addEventListener('change',audio);for(const id of ['text-group','text-policy','text-styles'])$(id).addEventListener('change',textview);
windows();textview();document.querySelectorAll('.explorer').forEach(e=>e.hidden=false);
})();
