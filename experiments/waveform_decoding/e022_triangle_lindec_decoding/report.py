"""Standalone WD-009 accuracy and class-recall reports."""

from html import escape
import json


SUBSETS = {
    'all_contexts': 'All comparison contexts',
    'matched_all_contexts': 'Nominal-charge-matched participants, all contexts',
    'direct': 'Held-out direct triangle/LinDec pairs',
    'matched_direct': 'Held-out nominal-charge-matched direct pairs',
    'non_direct': 'Other waveform-pair comparisons',
    'matched_non_direct': 'Nominal-charge-matched other-pair comparisons',
}


def render_markdown(summary):
    lines = [
        '# WD-009. Triangle versus linear-decrease EEG decoding', '',
        '**23 November 2026.** The target is the delivered waveform from the verified trigger. '
        'Naturalness votes are used only by the inherited trial-validity QC.',
        '',
        f"{summary['epochs']} epochs from {summary['participants']} participants; "
        f"{summary['direct_epochs']} direct-pair epochs, including {summary['matched_epochs']} nominal-charge-matched epochs.",
        '',
        '## Research question and origin', '',
        'Collu et al. reported 70% median behavioral match-to-sample discrimination for triangle versus '
        'linear decrease at matched charge and peak current. WD-009 tests whether their delivered waveform '
        'identities can also be distinguished from EEG. The EEG target comes from triggers. '
        '[Collu et al.](https://doi.org/10.1038/s41598-023-28594-0)',
        '',
        '## Which epochs are nominal-charge matched?', '',
        f"The matched direct-pair subset contains {summary['matched_epochs']} epochs. "
        f"The broader all-context matched subset contains {summary.get('matched_all_epochs', 247)} epochs "
        'from participants 3, 4, 5, 6 and 10. Only these participants have equal Triangle/LinDec nominal charge; '
        'the other five have different waveform-specific threshold amplitudes.',
        '',
        'Charge matching and comparison context are independent evaluation filters. The default report view '
        'uses matched nominal charge across all contexts: Triangle/LinDec epochs paired with each other or with '
        'Sine, Square or linear increase. Direct pairs contain only Triangle–LinDec comparisons. '
        'Other-pair-only evaluation excludes those direct comparisons and contains 375 epochs, including 181 matched-charge epochs.',
        '',
        'For the first pulse, the second stimulus occurs after the analyzed EEG window. Comparison context '
        'can still carry block expectations and recording-order effects. For the second pulse, the preceding '
        'waveform also changes with comparison context. Both presentation position and whole-block validation '
        'remain explicit. Filters summarize existing held-out predictions and do not refit models.',
        '',
        'Nominal cathodic charge equals peak amplitude times 0.4 ms times the waveform area factor. '
        'Triangle and LinDec both have area factor 0.5. Matching their shape factors does not match '
        'their waveform-specific threshold amplitudes. Task 4 uses a race-to-five pair-preference protocol. '
        'The ten-per-amplitude repetitions describe Task 3.',
        '',
        '## Held-out balanced accuracy', '',
        'Each participant and presentation position has its own model. Whole comparison groups are held out. '
        'Scores average participants equally; the binary balanced-accuracy reference is 50%.',
        '',
        '| Evaluation subset | Position | Bin | Model | Pre BA | Post BA | Post minus pre, pp [95% descriptive interval] | Sign-test p | Holm p |',
        '| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |',
    ]
    for row in summary['aggregate']:
        change = row['post_minus_pre']
        test = row.get('post_pre_test', {'p': 1., 'p_holm': 1.})
        lines.append(
            f"| {SUBSETS[row['subset']]} | {row['position']} | {row['width_ms']} ms | "
            f"{summary['models'][row['model']]} | {row['pre']['mean']:.2%} | {row['post']['mean']:.2%} | "
            f"{100*change['mean']:+.2f} [{100*change['ci95'][0]:+.2f}, {100*change['ci95'][1]:+.2f}] | "
            f"{test['p']:.4f} | {test['p_holm']:.4f} |"
        )
    lines += [
        '', '## Models, features and controls', '',
        '- All 62 scalp electrodes, with 50 or 100 ms voltage means over the same one-second window.',
        '- Post window is 50–1,050 ms. The matched pre window is −1,050 to −50 ms.',
        '- A shared pair baseline is measured −1,250 to −1,050 ms before the first pulse, before either feature period. '
        'Processing uses the instantaneous average reference, without temporal filtering or ICA.',
        '- StandardScaler and PCA are fitted on training epochs only. PCA retains 90% of training variance.',
        '- PCA + shrinkage LDA, PCA + linear SVM, and linear SVM without PCA are fixed comparisons. '
        'LDA uses equal priors; SVM uses C=1 and balanced class weights.',
        '- Training excludes the entire held-out waveform comparison within a participant and position. '
        'Direct triangle/LinDec blocks are tested using models trained on comparisons against other waveforms.',
        '- Nominal charge defines the matched evaluation subset and is excluded from predictors.',
        '- The original complete-pair QC is retained. Additional channels receive finite-value checks and descriptive '
        'peak-to-peak flags. All-channel artifact rejection and interpolation are outside this run.',
        '', '## Participant-level post/pre significance', '',
        'The exact two-sided sign test counts participants with higher post than pre BA, excluding ties. '
        'It tests equal probabilities of improvement and decline across independent participants, not the size '
        'of the change or individual above-chance decoding. Holm correction covers all 72 configurations/subsets/positions. '
        'With five non-tied participants the smallest two-sided p is 0.0625. '
        '[Exact binomial test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html) · '
        '[Holm correction](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html).',
        '', '## Interpretation', '',
        '- Earlier work decoded five classes or one waveform versus the remaining four. This run trains dedicated '
        'triangle/LinDec classifiers and evaluates unseen comparison contexts.',
        '- The direct and matched subset rows reuse held-out predictions and do not refit within their 5–7 trial blocks.',
        '- First-position results are the primary view. Second-position pre and post EEG can reflect the preceding pulse. '
        'The [WD-008 behavioral order audit](https://htmlpreview.github.io/?https://github.com/neurocontrol-lab/tens-eeg-waveform-decoding/blob/main/Docs/Reports/WD-008/index.html#behavioral-order-audit) '
        'establishes an association under its assumptions, without identifying physiological carryover.',
        '- Naturalness labels and behavioral order effects do not determine the waveform-identity targets.',
        '- Participant intervals condition on fixed held-out predictions. Corrected sign tests compare post with pre. '
        'No refitted label-permutation test of above-chance decoding is included.',
        '- Physical trigger/current delay and behavioral-response timing remain unverified.',
        '',
    ]
    return '\n'.join(lines)


def render_html(summary, figures=None):
    # Publish scores and methods, keeping raw file provenance in the numerical run.
    shared = {key: summary[key] for key in
              ('experiment', 'models', 'epochs', 'participants', 'direct_epochs',
               'matched_epochs', 'aggregate', 'per_subject')}
    shared['matched_all_epochs'] = summary.get('matched_all_epochs', 247)
    shared['figures'] = figures or {}
    charge_rows = ''.join(
        f'<tr><td>{row["subject"]}</td><td>{row["triangle_uC"]:.2f}</td>'
        f'<td>{row["lindec_uC"]:.2f}</td><td>{"Yes" if row["matched"] else "No"}</td>'
        f'<td>{row["epochs"]}</td></tr>'
        for row in summary.get('charge_table', [])
    )
    common_figures = ''.join(
        f'<figure><img src="{item["url"]}" alt="{escape(item["caption"])}">'
        f'<figcaption>{escape(item["caption"])}</figcaption></figure>'
        for item in shared['figures'].get('common', [])
    )
    return (PAGE.replace('__DATA__', json.dumps(shared, allow_nan=False).replace('<', '\\u003c'))
            .replace('__CHARGE_ROWS__', charge_rows).replace('__COMMON_FIGURES__', common_figures))


PAGE = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>WD-009 · Triangle versus linear decrease</title>
<style>
body{font:16px/1.55 system-ui,sans-serif;color:#172c37;background:#f7f8fa;margin:0}
*{box-sizing:border-box}main{max-width:1150px;margin:auto;padding:2rem 1.25rem 4rem}h1,h2{line-height:1.2}
h3{font-size:1.1rem;line-height:1.35;margin:1.3rem 0 .45rem}
section{background:white;padding:1.25rem;margin:1.5rem 0;border:1px solid #dce4e5;border-radius:.5rem}
p{margin:.65rem 0 1rem}.filters{display:flex;gap:1.25rem;flex-wrap:wrap}label{display:grid;gap:.3rem}
select{font:inherit;padding:.4rem}.scroll{overflow:auto}table{border-collapse:collapse;width:100%}
th,td{padding:.55rem;text-align:right;border-bottom:1px solid #ddd;vertical-align:top}
th:first-child,td:first-child{text-align:left}thead{background:#edf3f3}
.data-table{min-width:740px}.scroll{max-width:100%}
figure{margin:1.25rem 0}figure img{display:block;width:100%;height:auto}
figcaption{font-size:.9rem;color:#465762}
</style><main><h1>Triangle versus linear-decrease EEG decoding</h1><p id="coverage"></p>
<p>The classifier predicts the delivered waveform, Triangle (241) or LinDec (242).
Each participant and presentation position has its own model. Entire waveform-comparison groups are held out.</p>
<section><h2>Research question and origin</h2>
<p>Collu et al. reported 70% median behavioral match-to-sample discrimination of triangle versus linear decrease
under matched charge and peak current. WD-009 asks whether the delivered waveforms can also be distinguished
from EEG. The targets come from verified triggers.
<a href="https://doi.org/10.1038/s41598-023-28594-0">Collu et al.</a></p>
</section>
<section id="settings"><h2>Experimental settings and filters</h2>
<p><strong>Target:</strong> distinguish Triangle from LinDec using individual Task 4 EEG epochs.
Models are fitted separately for each participant and first/second presentation; entire comparison blocks are held out.</p>
<p><strong>Nominal charge:</strong> the default includes participants 3, 4, 5, 6 and 10, whose Triangle and LinDec
settings give equal nominal charge (247 epochs). The other five participants have different waveform-specific
amplitudes and are included when selecting “All charge settings” (501 epochs). Charge is never a model input.</p>
<p><strong>Comparison context:</strong> “All waveform pairs” includes every eligible Triangle or LinDec epoch,
whether paired with each other or with Sine, Square or linear increase. “Direct pairs” includes only
Triangle–LinDec comparisons (126 epochs; 66 with matched charge). “Other pairs” excludes those direct comparisons
(375 epochs; 181 with matched charge). Each filter summarizes saved held-out predictions without retraining.</p>
<p><strong>Presentation:</strong> first and second pulses are evaluated separately. The participant selector
changes the tables; cohort plots always show all participants eligible for the chosen filters.</p></section>
<section><h2>Accuracy and class recall</h2><div class="filters">
<label>Nominal charge<select id="charge"><option value="matched">Matched within participant × presentation</option>
<option value="all">All charge settings</option></select></label>
<label>Comparison context<select id="context"><option value="all">All waveform pairs</option>
<option value="direct">Direct triangle/LinDec pairs only</option>
<option value="other">Other waveform pairs only</option></select></label>
<label>Presentation<select id="position"><option value="1">First</option><option value="2">Second</option></select></label>
<label>Participant<select id="participant"><option value="all">Participant mean</option></select></label>
</div><p id="selection"></p><div class="scroll"><table class="data-table"><thead><tr><th>Bin / model</th><th>Pre BA</th><th>Post BA</th>
<th>Post accuracy</th><th>Triangle recall</th><th>LinDec recall</th></tr></thead><tbody id="scores"></tbody></table></div>
<h3>Post versus pre</h3><div class="scroll"><table class="data-table"><thead><tr><th>Bin / model</th>
<th>Post − pre, pp</th><th>95% interval, pp</th><th>Improved / worse / tied</th><th>p</th><th>Holm p</th>
</tr></thead><tbody id="contrasts"></tbody></table></div>
<p>BA is balanced accuracy, the mean of the two waveform recalls, with a 50% reference.
Participant means weight participants equally. Charge and context filters select existing held-out
predictions; they do not refit the models. Post − pre is a paired difference in percentage points (pp).
Intervals and p-values apply to participant means; an individual selection shows its difference only.</p></section>
<section id="plots"><h2>Cohort plots</h2>
<p>These plots follow the selected evaluation subset and presentation. Tables above can select an individual participant.</p>
<figure><img id="comparison_plot" alt="Pre and post balanced accuracy across all model configurations">
<figcaption>Pre/post participant means and descriptive 95% participant intervals. The dotted line is the 50% reference.
Labels give Holm-adjusted p-values for the post-versus-pre participant sign tests.</figcaption></figure>
<figure><img id="participant_plot" alt="Pre and post balanced accuracy by participant and model configuration">
<figcaption>Each panel shows one configuration: blue is pre-stimulation, orange is post-stimulation.
The dotted horizontal line marks 50% balanced accuracy. These reference lines are not significance thresholds.</figcaption></figure>
<h3>Significance test</h3><p>For each configuration, count participants whose post BA exceeds pre BA and whose
post BA is lower; ties are omitted. The exact two-sided sign test asks whether improvements and decreases
are equally likely across independent participants. It tests consistency of the direction of change, not its magnitude
or individual decoding above chance. Holm correction covers all 72 comparisons: six configurations × six filter subsets
× two positions. With only five non-tied participants, even five improvements give a two-sided p of 0.0625.</p>
<p><a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html">Exact binomial test</a> ·
<a href="https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html">Holm correction</a>.</p></section>
<section><h2>Epoch and charge coverage</h2>
<p>The table shows why only five participants qualify for charge matching. Counts combine both presentation
positions and all waveform-pair blocks. The filters above then select the desired epochs. Trial counts vary
because each Task 4 comparison stops when one waveform receives five votes.</p>
<p>Nominal cathodic charge is peak amplitude × 0.4 ms × 0.5 for both Triangle and LinDec.
Their shape factors are equal, but threshold amplitudes vary by participant and waveform.
These are documented nominal settings; current-monitor measurements remain unavailable.</p>
<div class="scroll"><table><thead><tr><th>Participant</th><th>Triangle charge (µC)</th>
<th>LinDec charge (µC)</th><th>Matched?</th><th>Epochs, all contexts</th></tr></thead>
<tbody>__CHARGE_ROWS__</tbody></table></div>__COMMON_FIGURES__</section>
<section id="methods"><h2>Features and models</h2>
<h3>Voltage features</h3>
<p>All 62 electrodes contribute voltage means in 50 or 100 ms bins. Post EEG spans 50–1,050 ms;
pre EEG spans −1,050 to −50 ms. This gives 1,240 or 620 original features.</p>
<h3>Reference and baseline</h3>
<p>A common pair baseline measured −1,250 to −1,050 ms before the first pulse is subtracted from all windows.
Processing uses the instantaneous 62-channel average reference, without temporal filtering or ICA.</p>
<h3>PCA</h3>
<p>Feature scaling and principal component analysis are fitted on the training fold.
PCA retains components accounting for 90% of training variance and transforms the held-out epochs using that fitted basis.</p>
<h3>Shrinkage LDA</h3>
<p>Linear discriminant analysis uses automatic covariance shrinkage and equal class priors,
with PCA-reduced features as its input.</p>
<h3>Linear SVM</h3>
<p>The support vector machine uses a linear kernel, fixed C=1 and balanced class weights.
It is evaluated both with PCA-reduced features and with the original scaled voltage features.</p>
<h3>Validation</h3>
<p>Within each participant and presentation, training excludes the entire held-out comparison.
For a direct triangle/LinDec block, the model learns from triangle and LinDec presentations paired with other
waveforms. All scaling and dimension reduction use training data only.</p></section>
<section><h2>Interpretation</h2>
<p>First-presentation performance is the primary view. Second-presentation pre and post EEG can
contain effects of the preceding pulse. The <a href="https://htmlpreview.github.io/?https://github.com/neurocontrol-lab/tens-eeg-waveform-decoding/blob/main/Docs/Reports/WD-008/index.html#behavioral-order-audit">WD-008 behavioral
order audit</a> identifies an association between choices and order, without establishing physiological carryover
or showing that second presentations are unusable.</p>
<p>The first-pulse EEG window ends before the second pulse, so a direct pair is optional for that analysis.
Comparison blocks can still differ in expectation and recording order. For the second pulse, pair context
also changes which waveform preceded the epoch. Whole-comparison-group validation addresses shared block context.</p>
<p>All-channel finite values and fixed original complete-pair QC are checked.
Additional channels have descriptive amplitude flags. The intervals resample participants conditional on
fixed held-out predictions. Sign tests compare post with pre across participants; no refitted label-permutation
test of above-chance decoding is included. Physical delivery and response timing remain unverified.</p></section></main>
<script type="application/json" id="data">__DATA__</script>
<script>
const data=JSON.parse(document.getElementById('data').textContent);
const el=id=>document.getElementById(id),pct=x=>(100*x).toFixed(2)+'%';
const mean=values=>values.reduce((a,b)=>a+b,0)/values.length;
const subsetKeys={
 all:{all:'all_contexts',matched:'matched_all_contexts'},
 direct:{all:'direct',matched:'matched_direct'},
 other:{all:'non_direct',matched:'matched_non_direct'}
};
const selectedSubset=()=>subsetKeys[el('context').value][el('charge').value];
el('coverage').textContent='WD-009 · '+data.epochs+' eligible epochs · '+data.participants+
 ' participants · '+data.direct_epochs+' direct-pair epochs · '+data.matched_all_epochs+
 ' matched-charge epochs across contexts · '+data.matched_epochs+' matched direct-pair epochs';
function menus(){
 const current=el('participant').value;
 const rows=data.per_subject.filter(r=>r.subset===selectedSubset()&&r.position===Number(el('position').value));
 const subjects=[...new Set(rows.map(r=>r.subject))];
 el('participant').replaceChildren(new Option('Participant mean','all'),...subjects.map(s=>new Option(String(s),String(s))));
 if(subjects.includes(Number(current)))el('participant').value=current;draw();
}
function draw(){
 const subset=selectedSubset(),position=Number(el('position').value),subject=el('participant').value;
 const all=data.per_subject.filter(r=>r.subset===subset&&r.position===position);
 const selected=subject==='all'?all:all.filter(r=>r.subject===Number(subject));
 el('scores').replaceChildren();el('contrasts').replaceChildren();
 const sample=selected.filter(r=>r.width_ms===50&&r.model==='pca_lda'&&r.period==='pre');
 el('selection').textContent=sample.reduce((n,r)=>n+r.epochs,0)+' epochs from '+sample.length+' participants';
 const figures=data.figures.contexts?.[subset+':'+position];
 el('plots').hidden=!figures;
 if(figures){el('comparison_plot').src=figures.overview;el('participant_plot').src=figures.participants;}
 for(const width of [50,100])for(const [model,label] of Object.entries(data.models)){
  const records=selected.filter(r=>r.width_ms===width&&r.model===model);
  const pre=records.filter(r=>r.period==='pre'),post=records.filter(r=>r.period==='post');
  const row=document.createElement('tr');
  const values=[width+' ms · '+label,mean(pre.map(r=>r.balanced_accuracy)),
   mean(post.map(r=>r.balanced_accuracy)),mean(post.map(r=>r.accuracy)),
   mean(post.map(r=>r.triangle_recall)),mean(post.map(r=>r.lindec_recall))];
  for(const [i,v]of values.entries()){const td=document.createElement('td');td.textContent=i===0?v:pct(v);row.append(td);}
  const delta=mean(post.map(r=>r.balanced_accuracy))-mean(pre.map(r=>r.balanced_accuracy));
  const extra=[(100*delta).toFixed(2), '—', '—', '—', '—'];
  if(subject==='all'){
   const comparison=data.aggregate.find(r=>r.subset===subset&&r.position===position&&r.width_ms===width&&r.model===model);
   const d=comparison.post_minus_pre,t=comparison.post_pre_test;
   extra[1]='['+d.ci95.map(v=>(100*v).toFixed(2)).join(', ')+']';
   if(t){extra[2]=t.improved+' / '+t.worsened+' / '+t.tied;extra[3]=t.p.toFixed(4);extra[4]=t.p_holm.toFixed(4);}
  }
  el('scores').append(row);
  const contrast=document.createElement('tr');
  for(const value of [values[0],...extra]){const td=document.createElement('td');td.textContent=value;contrast.append(td);}
  el('contrasts').append(contrast);
 }
}
el('charge').addEventListener('change',menus);el('context').addEventListener('change',menus);
el('position').addEventListener('change',menus);
el('participant').addEventListener('change',draw);menus();
</script></html>
"""
