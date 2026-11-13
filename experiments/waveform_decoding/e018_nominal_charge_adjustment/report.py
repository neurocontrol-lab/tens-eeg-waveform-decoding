"""Render the standalone WD-007 report from summary statistics."""

import json


def render(summary, waveforms):
    fields = ('subjects', 'complete_subjects', 'trials', 'complete_trials', 'charge_ranges',
              'calibration_missing', 'overlap', 'changes', 'scores')
    data = json.dumps({**{key: summary[key] for key in fields}, 'waveforms': waveforms}).replace('<', '\\u003c')
    return PAGE.replace('__SUMMARY__', data)


PAGE = """<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>WD-007 · Task 4 nominal-charge adjustment</title>
<style>
:root{color:#172c37;background:#f5f7f7;font:16px/1.55 system-ui,sans-serif}
body{margin:0}main{max-width:1100px;padding:2rem 1.25rem 4rem;margin:auto}
h1,h2{line-height:1.2;margin-bottom:.55rem}h1{font-size:clamp(2rem,5vw,3rem);max-width:18ch}
p{max-width:80ch}section{margin-top:2rem;padding:1.25rem;background:white;border:1px solid #dce4e5;border-radius:.65rem}
header p:first-child{font-weight:650;color:#17605b}.filters{display:flex;flex-wrap:wrap;gap:1rem;margin:1rem 0}
label{display:grid;gap:.2rem;font-weight:600}select{font:inherit;padding:.3rem .55rem;background:white;color:inherit;border:1px solid #7c9299;border-radius:.3rem}
.scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;white-space:nowrap}
th,td{text-align:right;padding:.45rem .65rem;border-bottom:1px solid #e0e7e8}th:first-child,td:first-child{text-align:left}thead th{background:#edf3f3}
small{color:#445b64}a{color:#17605b}summary{cursor:pointer;font-weight:600}
</style>
<main>
<header>
<p>WD-007 · Task 4 · E018</p>
<h1>Waveform decoding after nominal-charge adjustment</h1>
<p id="finding"></p>
<p>Nominal charge uses the true waveform shape. The waveform-specific variant also uses the true waveform to select the slope. These comparisons measure sensitivity to a label-informed adjustment.</p>
</header>
<section aria-labelledby="performance">
<h2 id="performance">Held-out performance</h2>
<div class="filters">
<label>Cohort <select id="cohort"><option value="all">All participants · pooled slope</option><option value="complete">Matched cohort · both slopes</option></select></label>
<label>Presentation <select id="position"><option value="0">Both presentations</option><option value="1">First</option><option value="2">Second</option></select></label>
</div>
<p id="coverage"></p>
<div class="scroll"><table id="metrics" aria-label="Participant-mean balanced accuracy and per-waveform recall"><thead></thead><tbody></tbody></table></div>
<p><small>Scores average participants equally. Balanced accuracy is mean class recall, with a five-class reference of 20%. Changes compare each adjustment with the original EEG in the same window. Intervals are descriptive paired participant bootstrap intervals.</small></p>
<p><small>The charge and correction controls have one fixed value or vector per waveform within a participant. Their zero within-class covariance makes shrinkage LDA degenerate. Nearest-centroid scores on the same folds show which templates can be distinguished.</small></p>
</section>
<section aria-labelledby="matrix-title">
<h2 id="matrix-title">Confusion matrix</h2>
<label>Model <select id="matrix-model"></select></label>
<p><small>Rows show the delivered waveform. Columns show the predicted waveform. Cells show mean within-participant row percentages. The diagonal is recall.</small></p>
<div class="scroll"><table id="matrix"><thead></thead><tbody></tbody></table></div>
</section>
<section aria-labelledby="overlap-title">
<h2 id="overlap-title">Nominal charges shared by waveform pairs</h2>
<p>Equal nominal charges produce identical pooled correction vectors within a participant. Waveform-specific slopes can produce different vectors even when nominal charges match.</p>
<label>Participant <select id="overlap-subject"><option value="all">All participants</option></select></label>
<div class="scroll"><table id="overlap"><thead></thead><tbody></tbody></table></div>
</section>
<section aria-labelledby="methods">
<h2 id="methods">Methods and coverage</h2>
<p>WD-001 voltage features used 14 electrodes and nine 50 ms bins. Post-stimulus features span 50 to 500 ms and matched pre-stimulus features span −500 to −50 ms. All comparisons reuse the original comparison-held-out folds and the same shrinkage LDA with training-fold standardization. First- and second-presentation scores use subsets of those held-out predictions.</p>
<p>Nominal cathodic charge equals peak amplitude × 0.4 ms × the ideal waveform area factor. The Task 4 CSV records one amplitude per pair. Its value was constant for every participant and second waveform, so the lookup uses that association for both presentations. Delivered current remains unverified.</p>
<p>Charge-only slopes were frozen from Task 3 and applied as −slope × nominal charge, with reference charge zero. The exact post-stimulus correction vectors were also applied to matched pre-stimulus EEG. The charge-only control uses the scalar. Each correction-only control uses the corresponding vector without EEG.</p>
<p id="calibration"></p>
</section>
</main>
<script type="application/json" id="data">__SUMMARY__</script>
<script>
const data = JSON.parse(document.querySelector('#data').textContent);
const names = {
  post: 'Original post-stimulus EEG', post_pooled: 'Post-stimulus EEG · pooled adjustment',
  post_specific: 'Post-stimulus EEG · waveform-specific adjustment', pre: 'Original pre-stimulus EEG',
  pre_pooled: 'Pre-stimulus EEG · pooled adjustment', pre_specific: 'Pre-stimulus EEG · waveform-specific adjustment',
  nominal_charge: 'Nominal charge alone · LDA', nominal_charge_centroid: 'Nominal charge alone · nearest centroid',
  pooled_correction: 'Pooled correction alone · LDA', pooled_correction_centroid: 'Pooled correction alone · nearest centroid',
  specific_correction: 'Waveform-specific correction alone · LDA',
  specific_correction_centroid: 'Waveform-specific correction alone · nearest centroid'
};
const find = id => document.getElementById(id);
const mean = values => values.reduce((a, b) => a + b, 0) / values.length;
const pct = value => `${(100 * value).toFixed(1)}%`;
const points = value => `${value >= 0 ? '+' : '−'}${Math.abs(100 * value).toFixed(1)}`;
const recall = (matrix, row, col) => matrix[row][col] / matrix[row].reduce((a, b) => a + b, 0);
let models = {};
function table(id, headings, rows) {
  const element = find(id);
  const makeRow = (items, header) => {
    const row = document.createElement('tr');
    items.forEach((value, index) => {
      const cell = document.createElement(header || index === 0 ? 'th' : 'td');
      if (header || index === 0) cell.scope = header ? 'col' : 'row';
      cell.textContent = value;
      row.append(cell);
    });
    return row;
  };
  element.tHead.replaceChildren(makeRow(headings, true));
  element.tBodies[0].replaceChildren(...rows.map(row => makeRow(row, false)));
}
function renderMatrix() {
  const matrices = models[find('matrix-model').value].map(row => row.confusion_matrix);
  table('matrix', ['Delivered / Predicted', ...data.waveforms], data.waveforms.map((wave, i) =>
    [wave, ...data.waveforms.map((_, j) => pct(mean(matrices.map(m => recall(m, i, j)))))]));
}
function update() {
  const cohort = find('cohort').value;
  const position = Number(find('position').value);
  const subjects = cohort === 'complete' ? data.complete_subjects : data.subjects;
  const rows = data.scores.filter(row => subjects.includes(row.subject) && row.position === position);
  models = {};
  for (const key of Object.keys(names)) {
    const scores = rows.filter(row => row.model === key);
    if (scores.length === subjects.length) models[key] = scores;
  }
  find('coverage').textContent = `${subjects.length} participants · ${models.post.reduce((sum, row) => sum + row.n_events, 0)} trials`;
  table('metrics', ['Model', 'Balanced accuracy', 'Change in points [95% interval]', ...data.waveforms],
    Object.entries(models).map(([key, values]) => {
      const change = data.changes.find(row => row.cohort === cohort && row.position === position && row.model === key);
      return [names[key], pct(mean(values.map(row => row.balanced_accuracy))),
        change ? `${points(change.mean)} [${change.ci95.map(points).join(', ')}]` : '—',
        ...data.waveforms.map((_, i) => pct(mean(values.map(row => recall(row.confusion_matrix, i, i)))) )];
    }));
  const select = find('matrix-model');
  const selected = select.value;
  select.replaceChildren(...Object.keys(models).map(key => new Option(names[key], key)));
  select.value = selected in models ? selected : 'post';
  renderMatrix();
}
const main = data.changes.find(row => row.cohort === 'all' && row.position === 0 && row.model === 'post_pooled');
const average = key => mean(data.scores.filter(row => row.position === 0 && row.model === key).map(row => row.balanced_accuracy));
find('finding').textContent = `The pooled adjustment changed post-stimulus balanced accuracy from ${pct(average('post'))} to ${pct(average('post_pooled'))} (${points(main.mean)} points). The same adjustment raised pre-stimulus accuracy from ${pct(average('pre'))} to ${pct(average('pre_pooled'))}.`;
function renderOverlap() {
  const selected = find('overlap-subject').value;
  if (selected === 'all') {
    const counts = new Map();
    for (const {waveform_a, waveform_b} of data.overlap) {
      const pair = `${waveform_a} and ${waveform_b}`;
      counts.set(pair, (counts.get(pair) || 0) + 1);
    }
    table('overlap', ['Waveforms', 'Participants with equal charge'], [...counts.entries()].sort((a, b) => b[1] - a[1]));
  } else {
    const pairs = data.overlap.filter(row => row.subject === Number(selected));
    table('overlap', ['Waveforms', 'Nominal charge (µC)'], pairs.length
      ? pairs.map(row => [`${row.waveform_a} and ${row.waveform_b}`, row.charge_uC.toFixed(3)])
      : [['No equal charges', '—']]);
  }
}
find('overlap-subject').append(...data.subjects.map(subject => new Option(`Participant ${subject}`, subject)));
find('overlap-subject').addEventListener('change', renderOverlap);
const missing = Object.entries(data.calibration_missing).map(([s, waves]) => `${s} (${waves.join(', ')})`).join(', ');
find('calibration').textContent = `${data.complete_subjects.length} participants had all five waveform-specific slopes. The remaining participants were ${missing || 'none'}. Across all trials, ${data.charge_ranges.outside || 0} nominal charges fell outside their waveform’s Task 3 fitted range and ${data.charge_ranges.missing || 0} had no waveform-specific range.`;
find('cohort').addEventListener('change', update);
find('position').addEventListener('change', update);
find('matrix-model').addEventListener('change', renderMatrix);
update();
renderOverlap();
</script>
</html>
"""
