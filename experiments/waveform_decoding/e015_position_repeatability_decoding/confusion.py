"""Count and participant-balanced views of saved held-out E015 predictions."""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

CODES = [239, 240, 241, 242, 243]
NAMES = ['Sine', 'Square', 'Triangle', 'LinDec', 'LinInc']


def row_normalize(counts):
    counts = np.asarray(counts, dtype=float)
    support = counts.sum(axis=1)
    return np.divide(counts, support[:, None], out=np.full_like(counts, np.nan), where=support[:, None] > 0)


def aggregate(matrices):
    """Pool counts separately from equal-participant row-normalized rates."""
    arrays = np.asarray(matrices)
    pooled = arrays.sum(axis=0)
    rates = np.array([row_normalize(m) for m in arrays])
    available = np.isfinite(rates).sum(axis=0)
    mean = np.divide(np.nansum(rates, axis=0), available,
                     out=np.full_like(rates[0], np.nan), where=available > 0)
    return pooled, row_normalize(pooled), mean


def prepare_confusions(sources, output):
    records = []
    for mode, root, models in sources:
        for model in models:
            if model['status'] != 'observed_complete':
                continue
            subject, position = model['subject'], model['position']
            file = root / f'sub-{subject}' / f'position-{position}' / 'predictions.csv'
            predictions = pd.read_csv(file)
            assert (predictions.position == position).all()
            assert set(predictions.trigger) <= set(CODES)
            for selector, saved in model['models'].items():
                guessed = predictions[selector + '_predicted']
                assert set(guessed) <= set(CODES)
                counts = confusion_matrix(predictions.trigger, guessed, labels=CODES)
                assert counts.tolist() == saved['confusion_matrix']
                rates = row_normalize(counts)
                np.testing.assert_allclose(np.diag(rates).mean(), saved['balanced_accuracy'], atol=1e-12)
                records.append(dict(mode=mode, subject=subject, position=position, selector=selector,
                                    counts=counts.tolist(), rates=rates.tolist()))
    class_rows, summary_rows, aggregates = [], [], []
    for record in records:
        counts = np.array(record['counts'])
        for i, (code, name) in enumerate(zip(CODES, NAMES)):
            class_rows.append({k: record[k] for k in ['mode', 'subject', 'position', 'selector']}
                              | dict(trigger=code, waveform=name, support=int(counts[i].sum()),
                                     correct=int(counts[i, i]), predicted_count=int(counts[:, i].sum()),
                                     recall=record['rates'][i][i]))
    for mode in ['strict', 'exploratory']:
        for position in [1, 2]:
            for selector in ['repeatability', 'anova']:
                group = [r for r in records if (r['mode'], r['position'], r['selector']) == (mode, position, selector)]
                counts, pooled, mean = aggregate([r['counts'] for r in group])
                aggregates.append(dict(mode=mode, position=position, selector=selector,
                                       subject='pooled', counts=counts.tolist(), rates=pooled.tolist()))
                aggregates.append(dict(mode=mode, position=position, selector=selector,
                                       subject='mean', counts=counts.tolist(), rates=mean.tolist()))
                for i, (code, name) in enumerate(zip(CODES, NAMES)):
                    summary_rows.append(dict(mode=mode, position=position, selector=selector,
                                             trigger=code, waveform=name, participants=len(group),
                                             support=int(counts[i].sum()), correct=int(counts[i, i]),
                                             pooled_recall=float(pooled[i, i]),
                                             mean_participant_recall=float(mean[i, i])))
    pd.DataFrame(class_rows).to_csv(output / 'class_recall.csv', index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / 'class_recall_summary.csv', index=False)
    (output / 'confusion_matrices.json').write_text(json.dumps(dict(labels=CODES, names=NAMES,
                                                    individual=records, aggregate=aggregates), indent=2) + '\n')
    section = '''<section id="confusion-section"><h2>Confusion matrices and class recall</h2>
<style>
#confusion-section select{padding:6px;margin:5px 12px 5px 3px}
.cm-legend{max-width:540px;margin:16px 0 20px}
.cm-colorbar{height:20px;border:1px solid #738394;background:linear-gradient(to right,#ffffff,rgba(0,114,178,.7))}
.cm-colorbar.cm-errors{background:linear-gradient(to right,#ffffff,rgba(213,94,0,.7))}
.cm-ticks{display:flex;justify-content:space-between;font-size:13px;color:#172437}
.cm-diagonal{font-weight:700}
</style>
<p>Rows are <b>delivered waveforms</b>; columns are <b>predicted waveforms</b>. Diagonal percentages are class recall: correct predictions divided by all held-out epochs of that true class. Off-diagonal cells show which other labels receive its errors.</p>
<div class="cm-legend" aria-label="Correct predictions: blue intensity shows row-normalized percentage from zero to one hundred percent">
<b>Correct predictions (diagonal) — percentage of true-class trials</b><div class="cm-colorbar"></div>
<div class="cm-ticks"><span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span></div>
</div>
<div class="cm-legend" aria-label="Misclassifications: orange intensity shows row-normalized percentage from zero to one hundred percent">
<b>Misclassifications (off-diagonal) — percentage of true-class trials</b><div class="cm-colorbar cm-errors"></div>
<div class="cm-ticks"><span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span></div>
</div>
<p><b>Blue diagonal cells</b> are correct predictions; <b>orange off-diagonal cells</b> are errors. Stronger color means a larger fraction of the true-class trials. Both legends use a fixed 0–100% scale across every view. Colors encode percentages, not raw counts or statistical significance. Gray cells indicate unavailable percentages.</p>
<label>Pipeline <select id="cm-mode"><option value="exploratory">Exploratory (unreliable)</option><option value="strict">Strict supported subset</option></select></label>
<label>Features <select id="cm-selector"><option value="repeatability">Repeatability top 100</option><option value="anova">ANOVA top 100</option></select></label>
<label>Presentation <select id="cm-position"><option value="1">First</option><option value="2">Second</option></select></label>
<label>Participant / summary <select id="cm-subject"><option value="mean">All: equal-participant recall</option><option value="pooled">All: pooled counts and recall</option>SUBJECTS</select></label>
<p id="cm-caption"></p><div style="overflow-x:auto" id="cm-matrix"></div><div style="overflow-x:auto" id="cm-recall"></div>
<p>Equal-participant percentages average row-normalized matrices; their diagonal mean equals the reported mean participant balanced accuracy. Pooled percentages weight participants by the number of epochs of each class. Counts always refer to the available held-out events. Strict unsupported cases display no matrix. Comparisons remain descriptive; class support, comparison context and sparse training can affect recall.</p>
<p><a href="class_recall.csv">Individual class recall CSV</a> · <a href="class_recall_summary.csv">Class recall summaries</a> · <a href="confusion_matrices.json">All confusion matrices</a></p>
</section><script>
const CM_DATA=PAYLOAD, CM_NAMES=NAMES;
function confusionColor(rate,diagonal){
if(rate==null||!Number.isFinite(rate))return {background:'#e5e7eb',foreground:'#000000'};
const alpha=Math.min(1,Math.max(0,rate))*.7;
return {background:'rgba('+(diagonal?'0,114,178':'213,94,0')+','+alpha+')',foreground:'#000000'};
}
function renderConfusion(){
const mode=document.getElementById('cm-mode').value, selector=document.getElementById('cm-selector').value,
position=Number(document.getElementById('cm-position').value), subject=document.getElementById('cm-subject').value;
const r=CM_DATA.find(x=>x.mode===mode&&x.selector===selector&&x.position===position&&String(x.subject)===subject);
if(!r){document.getElementById('cm-caption').textContent='Ineligible under the strict design: no held-out model is available for this participant and position.';document.getElementById('cm-matrix').innerHTML='';document.getElementById('cm-recall').innerHTML='';return;}
const mean=subject==='mean';
document.getElementById('cm-caption').textContent=(mean?'Equal-participant mean percentages':subject==='pooled'?'Pooled held-out counts and percentages':'Participant '+subject+' held-out counts and percentages')+' — '+(mode==='exploratory'?'all source cases flagged unreliable':'strict supported cases only')+'.';
let matrix='<table aria-label="Confusion matrix"><thead><tr><th>True ↓ / predicted →</th>'+CM_NAMES.map(n=>'<th>'+n+'</th>').join('')+'</tr></thead><tbody>';
for(let i=0;i<5;i++){
matrix+='<tr><th>'+CM_NAMES[i]+'</th>';
for(let j=0;j<5;j++){
const rate=r.rates[i][j], pct=rate==null?'—':(100*rate).toFixed(1)+'%', color=confusionColor(rate,i===j);
matrix+='<td class="'+(i===j?'cm-diagonal':'cm-off-diagonal')+'" style="text-align:center;background:'+color.background+';color:'+color.foreground+'">'+pct+(mean?'':'<br><small>n = '+r.counts[i][j]+'</small>')+'</td>';
}matrix+='</tr>';
}matrix+='</tbody></table>';document.getElementById('cm-matrix').innerHTML=matrix;
let recalls='<table aria-label="Class recall"><thead><tr><th>Waveform</th><th>Recall</th><th>Held-out support</th><th>Correct / support'+(mean?' (pooled counts)':'')+'</th></tr></thead><tbody>';
for(let i=0;i<5;i++){const support=r.counts[i].reduce((a,b)=>a+b,0), recall=r.rates[i][i];recalls+='<tr><th>'+CM_NAMES[i]+'</th><td>'+(recall==null?'—':(100*recall).toFixed(1)+'%')+'</td><td>'+support+'</td><td>'+r.counts[i][i]+' / '+support+'</td></tr>';}
recalls+='</tbody></table>';document.getElementById('cm-recall').innerHTML=recalls;
}
for(const id of ['cm-mode','cm-selector','cm-position','cm-subject'])document.getElementById(id).addEventListener('change',renderConfusion);renderConfusion();
</script>'''
    section = section.replace('SUBJECTS', ''.join(f'<option value="{s}">{s}</option>' for s in range(3, 13)))
    section = section.replace('PAYLOAD', json.dumps(records + aggregates, allow_nan=False)).replace('CM_NAMES=NAMES', 'CM_NAMES=' + json.dumps(NAMES))
    return section, summary
