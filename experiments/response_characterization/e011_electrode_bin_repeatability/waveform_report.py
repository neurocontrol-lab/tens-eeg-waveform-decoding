"""Local interactive canvas explorer: no averaging over waveforms/participants."""
import json
from pathlib import Path
import pandas as pd

TEMPLATE = r'''<!doctype html>
<html><head><meta charset="utf-8"><title>E011 individual-waveform explorer</title>
<style>
body{margin:0;background:#f5f7fa;color:#172437;font:15px system-ui}main{max-width:1680px;margin:auto;padding:24px}h1{font-size:27px;margin-bottom:8px}h2{font-size:20px}p{line-height:1.55}.panel{background:white;border:1px solid #dbe1e9;border-radius:9px;padding:18px;margin:16px 0}.controls{display:flex;gap:14px;flex-wrap:wrap;position:sticky;top:0;background:#f5f7fa;z-index:2;padding:12px 0}label{display:flex;flex-direction:column;font-size:13px;font-weight:600;gap:5px}select{font:15px system-ui;padding:8px;border:1px solid #adb8c7;border-radius:5px;max-width:380px}.columns{display:grid;grid-template-columns:1fr 1fr;gap:18px}canvas{width:100%;height:auto;background:white}table{border-collapse:collapse;width:100%;font-size:14px}td,th{text-align:left;padding:9px;border-bottom:1px solid #e5e9ee}.selected{background:#e8f2fc}.note{color:#526379;font-size:13px}.badge{font-weight:600;color:#0b5a96}details{margin:14px 0}summary{cursor:pointer;font-weight:600}.error{color:#b71c1c}@media(max-width:1000px){.columns{grid-template-columns:1fr}.controls{position:static}}
</style></head><body><main>
<h1>E011 — Is this waveform's response repeatable?</h1>
<p>Choose a participant and waveform. The main views keep <b>both individual</b>; no participant or waveform averaging is applied to these maps. First and second presentations remain separate.</p>
<details class="panel"><summary>What the two response views and statistics mean</summary>
<p><b>Waveform-specific deviation</b> is the context/time/position-adjusted waveform effect from E011, with the common waveform response removed. It asks whether a distinguishing waveform deviation repeats.</p>
<p><b>Total observed response</b> retains the entire response for that waveform, giving its observed comparison cells equal weight. A repeatable total response can be shared by every waveform, or reflect repeatable artifacts/context. It is not evidence of waveform identity information.</p>
<p><b>At a single electrode × bin</b>, each half supplies one number: a correlation is undefined. The heatmap instead shows that waveform's mean A×B product across 100 splits, signed full-data response, RMS half-to-half disagreement or fraction of splits with matching signs.</p>
<p><b>Temporal pattern r</b> compares that waveform's A/B time courses at one electrode over fixed [50,1050) ms, then takes the median across splits. <b>Joint pattern r</b> compares the flattened 14-electrode × one-second patterns. These do not correlate across the five waveform identities or across complementary split repetitions. Coarse 200-ms temporal correlations contain only five time bins; all sampled time points are correlated after filtering.</p>
<p>Paired splits, trial eligibility, 25-ms derivative estimator and all six bin widths exactly match the original E011. Pointwise products retain squared physical units. Large products can reflect large unstable responses; read them alongside disagreement and pattern correlations. The optional group table resamples participants, not splits. No binary repeatable/not-repeatable threshold, corrected significance test or feature selection is supplied.</p>
</details>
<div class="controls">
<label>Participant<select id="participant"></select></label>
<label>Waveform<select id="waveform"></select></label>
<label>Representation<select id="representation"></select></label>
<label>Bin width<select id="width"></select></label>
<label>Response view<select id="view"><option value="specific">Waveform-specific deviation</option><option value="total">Total observed response</option></select></label>
</div>
<div id="loading" class="note">Loading participant data…</div>
<section class="panel"><h2>All five waveforms — selected participant only</h2><p class="note">Joint pattern r: median split correlation across 14 electrodes × [50,1050) ms. Each row remains a separate waveform; rows are not ranked. Trial support is shown alongside.</p><div id="waveTable"></div></section>
<section class="panel"><h2 id="mapTitle">Individual-waveform electrode × time maps</h2>
<label>Map statistic<select id="metric"><option value="mean_split_product">Mean split product A × B</option><option value="full_response">Signed full-data response</option><option value="rms_half_disagreement">RMS half-to-half disagreement</option><option value="split_sign_agreement">Fraction of splits with matching signs</option></select></label>
<p id="scale" class="note"></p><div class="columns"><div><h3>First presentation</h3><canvas id="map0" width="900" height="440"></canvas></div><div><h3>Second presentation</h3><canvas id="map1" width="900" height="440"></canvas></div></div>
<p class="note">Color limits are shared across all five waveforms and both positions for this participant/representation/bin width/view. Display clips at the common 99th-percentile magnitude (no stored values are clipped). Grey tails contain no complete bin.</p></section>
<section class="panel"><h2>Inspect a waveform's two halves at one electrode</h2><label>Electrode<select id="channel"></select></label>
<p class="note">Orange/blue curves show the preselected <b>first split (split 0)</b>, not a best-looking split. Grey shows the full-data estimate. The displayed r summarizes all 100 splits and only [50,1050) ms. Grey tail after 1050 ms is outside that correlation summary.</p>
<div class="columns"><div><h3 id="curveTitle0"></h3><canvas id="curve0" width="900" height="400"></canvas></div><div><h3 id="curveTitle1"></h3><canvas id="curve1" width="900" height="400"></canvas></div></div>
<div id="channelTable"></div></section>
<details class="panel"><summary>Optional group summaries — still separate for each waveform</summary><p class="note">Equal-participant mean of each waveform's median joint-pattern r, with descriptive 95% participant-bootstrap intervals. These are not corrected significance tests. The individual-participant maps above remain the primary view.</p><div id="groupTable"></div></details>
<section class="panel"><h2>Limits and downloadable outputs</h2><p>Ten participants; 595 pairs / 1,190 events, identical to E011. Sparse waveform/position support can make estimates unstable. Split halves share comparison contexts; total responses also retain unadjusted nuisance activity. A repeatable response is not necessarily neural, waveform-distinct or decodable on held-out comparisons. RMS derivatives may reflect noise. 1-ms sample spacing does not imply independent 1-ms physiological resolution. All findings are exploratory.</p>
<p><a href="participant_waveform_summary.csv">Per-participant/waveform summaries</a> · <a href="group_waveform_summary.csv">Separate-waveform group intervals</a> · <a href="config.json">Methods/provenance</a> · <a href="../e011_repeatability_20261030/index.html">Preserved original E011 report</a></p></section>
</main><script>
const config=__CONFIG__;
const groups=__GROUPS__;
const el=id=>document.getElementById(id);
const labels={voltage_mean:'Mean voltage',slope_mean:'Signed mean slope',slope_rms:'RMS slope',curvature_mean:'Signed mean curvature',curvature_rms:'RMS curvature'};
for(const s of config.subjects)el('participant').add(new Option('Subject '+s,s));
for(let w=0;w<config.waves.length;w++)el('waveform').add(new Option(config.waves[w],w));
for(const r of config.representations)el('representation').add(new Option(labels[r],r));
for(const w of config.widths)el('width').add(new Option(w===1?'1 ms (native grid)':w+' ms',w));el('width').value='50';
for(const c of config.channels)el('channel').add(new Option(c,c));el('channel').value='C3';
let active=null,request=0;
const decoded=new Map();
function unpack(str){if(decoded.has(str))return decoded.get(str);const s=atob(str),b=new Uint8Array(s.length);for(let i=0;i<s.length;i++)b[i]=s.charCodeAt(i);const f=new Float32Array(b.buffer);decoded.set(str,f);return f;}
function fmt(x){return x===null||!Number.isFinite(x)?'undefined':Number(x).toFixed(3);}
function scientific(x){if(!Number.isFinite(x))return 'undefined';return Math.abs(x)<1e-3||Math.abs(x)>=1e4?x.toExponential(2):x.toFixed(3);}
function offset(p,w,c,n){return ((p*5+w)*14+c)*n;}
function valueArray(view,metric){return unpack(view[metric]);}
function table(headers,rows){return '<table><thead><tr>'+headers.map(h=>'<th>'+h+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr'+(r.selected?' class="selected"':'')+'>'+r.cells.map(c=>'<td>'+c+'</td>').join('')+'</tr>').join('')+'</tbody></table>';}
function color(v,limit,positive){let t=Math.max(-1,Math.min(1,v/limit));if(positive){const a=Math.max(0,t);return `rgb(${Math.round(247-190*a)},${Math.round(250-140*a)},${Math.round(252-70*a)})`;}return t>=0?`rgb(${Math.round(250-70*t)},${Math.round(250-205*t)},${Math.round(250-205*t)})`:`rgb(${Math.round(250+205*t)},${Math.round(250+135*t)},${Math.round(250+65*t)})`;}
function map(canvas,values,p,w,info,limit,positive){const ctx=canvas.getContext('2d'),L=65,T=15,W=810,H=350;ctx.clearRect(0,0,900,440);ctx.fillStyle='#e9edf1';ctx.fillRect(L,T,W,H);for(let c=0;c<14;c++){const off=offset(p,w,c,info.bins);for(let b=0;b<info.bins;b++){const x=L+W*(50+b*Number(el('width').value)-50)/1138;const wid=W*Number(el('width').value)/1138;const v=values[off+b];ctx.fillStyle=Number.isFinite(v)?color(v,limit,positive):'#ccc';ctx.fillRect(x,T+c*H/14,Math.max(wid,.6)+.2,H/14+.2);}ctx.fillStyle='#24374b';ctx.font='13px system-ui';ctx.textAlign='right';ctx.fillText(config.channels[c],L-8,T+(c+.68)*H/14);}axes(ctx,L,T,W,H);}
function axes(ctx,L,T,W,H){ctx.font='13px system-ui';ctx.fillStyle='#314356';ctx.textAlign='center';for(const t of [50,250,450,650,850,1050,1188]){const x=L+(t-50)/1138*W;ctx.fillText(t,x,T+H+24);}ctx.strokeStyle='#8493a4';ctx.strokeRect(L,T,W,H);ctx.setLineDash([4,4]);const x=L+1000/1138*W;ctx.beginPath();ctx.moveTo(x,T);ctx.lineTo(x,T+H);ctx.stroke();ctx.setLineDash([]);ctx.fillText('Time from recorded trigger (ms)',L+W/2,T+H+55);}
function curve(canvas,view,p,w,c,info){const ctx=canvas.getContext('2d'),L=80,T=25,W=795,H=290;ctx.clearRect(0,0,900,400);const a=valueArray(view,'example_half_a'),b=valueArray(view,'example_half_b'),full=valueArray(view,'full_response'),off=offset(p,w,c,info.bins);let limit=1e-9;for(let j=0;j<info.bins;j++)limit=Math.max(limit,Math.abs(a[off+j]),Math.abs(b[off+j]),Math.abs(full[off+j]));limit*=1.1;ctx.fillStyle='#f1f3f6';ctx.fillRect(L+1000/1138*W,T,W*(138/1138),H);ctx.strokeStyle='#bcc6d1';ctx.beginPath();ctx.moveTo(L,T+H/2);ctx.lineTo(L+W,T+H/2);ctx.stroke();for(const [array,col] of [[full,'#8a96a5'],[a,'#0072B2'],[b,'#D55E00']]){ctx.strokeStyle=col;ctx.lineWidth=1.5;ctx.beginPath();for(let j=0;j<info.bins;j++){const x=L+((j+.5)*Number(el('width').value))/1138*W,y=T+H/2-array[off+j]/limit*H/2;if(j===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);}ctx.stroke();}ctx.font='12px system-ui';ctx.fillStyle='#314356';ctx.textAlign='right';for(const v of [-limit,0,limit])ctx.fillText(scientific(v),L-7,T+H/2-v/limit*H/2+4);axes(ctx,L,T,W,H);ctx.textAlign='left';ctx.fillStyle='#0072B2';ctx.fillText('Half A',L+5,16);ctx.fillStyle='#D55E00';ctx.fillText('Half B',L+85,16);ctx.fillStyle='#8a96a5';ctx.fillText('Full data',L+165,16);}
function draw(){if(!active)return;const w=Number(el('waveform').value),width=el('width').value,viewName=el('view').value,info=active.widths[width],view=info.views[viewName],metric=el('metric').value;const values=valueArray(view,metric),positive=metric==='rms_half_disagreement'||metric==='split_sign_agreement';let limit;if(metric==='split_sign_agreement')limit=1;else{const all=Array.from(values,x=>Math.abs(x)).filter(Number.isFinite).sort((a,b)=>a-b);limit=Math.max(all[Math.floor(.99*(all.length-1))]||0,1e-12);}el('mapTitle').textContent=config.waves[w]+' — Subject '+active.subject+' — '+labels[active.representation];el('scale').textContent='Color scale: '+(positive?'0 to ':'±')+scientific(limit)+(metric==='split_sign_agreement'?' (dependent-split fraction)':metric==='mean_split_product'?' ('+active.unit+')²':' '+active.unit);for(let p=0;p<2;p++)map(el('map'+p),values,p,w,info,limit,positive);const rows=[];for(let wave=0;wave<5;wave++){const s=active.support.filter(s=>s.trigger===239+wave);rows.push({selected:wave===w,cells:[config.waves[wave],fmt(view.joint_profile_r[0][wave]),fmt(view.joint_profile_r[1][wave]),s[0].n_events+' / '+s[1].n_events,s[0].minimum_half_events+' / '+s[1].minimum_half_events]});}el('waveTable').innerHTML=table(['Waveform','P1 joint pattern r','P2 joint pattern r','Trials: P1 / P2','Minimum half trials: P1 / P2'],rows);const c=config.channels.indexOf(el('channel').value);for(let p=0;p<2;p++){el('curveTitle'+p).textContent='Presentation '+(p+1)+' — '+config.channels[c]+' temporal r = '+fmt(view.temporal_profile_r[p][w][c]);curve(el('curve'+p),view,p,w,c,info);}el('channelTable').innerHTML=table(['Electrode','P1 temporal pattern r','P2 temporal pattern r'],config.channels.map((ch,ci)=>({selected:ci===c,cells:[ch,fmt(view.temporal_profile_r[0][w][ci]),fmt(view.temporal_profile_r[1][w][ci])]})));const g=groups.filter(g=>g.representation===active.representation&&g.width_ms===Number(width)&&g.view===viewName);el('groupTable').innerHTML=table(['Waveform','Presentation','Mean joint pattern r','Descriptive 95% interval'],g.map(g=>({selected:g.trigger===239+w,cells:[g.waveform,g.position,fmt(g.mean_joint_pattern_r),'['+fmt(g.low)+', '+fmt(g.high)+']']})));}
async function load(){const ticket=++request,key=el('participant').value+':'+el('representation').value;el('loading').textContent='Loading selected participant/representation…';try{window.e011WaveData=window.e011WaveData||{};if(!window.e011WaveData[key])await new Promise((resolve,reject)=>{const script=document.createElement('script');script.src='data/sub-'+el('participant').value+'_'+el('representation').value+'.js';script.onload=resolve;script.onerror=()=>reject(new Error('Could not load local data file'));document.head.append(script);});if(ticket!==request)return;active=window.e011WaveData[key];decoded.clear();el('loading').textContent='Loaded Subject '+active.subject+'. Correlations summarize 100 paired splits; individual maps preserve each waveform.';draw();}catch(e){el('loading').textContent=e.message;el('loading').className='error';}}
el('participant').addEventListener('change',load);el('representation').addEventListener('change',load);for(const id of ['waveform','width','view','metric','channel'])el(id).addEventListener('change',draw);load();
</script></body></html>'''


def render(out):
    out = Path(out)
    config = json.loads((out/'config.json').read_text())
    groups = pd.read_csv(out/'group_waveform_summary.csv').to_dict('records')
    settings = dict(subjects=config['subjects'],representations=config['representations'],
        widths=config['widths_ms'],channels=['FC1','FC2','C3','C1','Cz','C2','C4','CP5','CP1','CP2','CP6','P3','Pz','P4'],
        waves=['Sine','Square','Triangle','LinDec','LinInc'])
    html = TEMPLATE.replace('__CONFIG__',json.dumps(settings)).replace('__GROUPS__',json.dumps(groups,allow_nan=False))
    (out/'index.html').write_text(html,encoding='utf-8')
    (out/'RESULTS.md').write_text('# E011 - Individual-waveform repeatability extension\n\n'
        'Interactive index.html defaults to one participant and waveform. Total observed responses and '
        'context-adjusted waveform deviations remain separate. Every width/representation/pair split '
        'matches original E011. Per-waveform split products average back to the preserved pooled products.\n\n'
        'Single electrode/bin cells have product, signed response and half disagreement, not a scalar correlation. '
        'Temporal pattern r spans the fixed [50,1050)-ms time course at one electrode. Joint pattern r spans '
        '14 electrodes x the same time bins. Both take the median over 100 paired splits. '
        'The display also retains individual participant values; group intervals are optional and waveform-specific.\n\n'
        'Total-response repeatability is not waveform distinctiveness; shared evoked responses and artifacts '
        'can repeat. A waveform-specific deviation is relative to the other waveform responses under the '
        'fixed nuisance model, rather than a perfectly isolated waveform mechanism. Different trial counts, '
        'coarse five-bin temporal correlations and dependent temporal samples limit interpretation. '
        'No significance threshold, feature selection or classifier was added. Original E011 outputs remain unchanged.\n',encoding='utf-8')
