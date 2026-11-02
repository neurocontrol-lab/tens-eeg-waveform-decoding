"""Audit saved E016 predictions and build a lightweight local results report."""
import argparse
import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .core import CODES, WAVES, binary_metrics


def within_fold_auc(folds):
    """Only compare positive/negative scores produced by the same fitted model."""
    usable = [f['test_metrics'] for f in folds if f['test_metrics']['roc_auc'] is not None]
    weights = [m['n_positive'] * m['n_negative'] for m in usable]
    return dict(within_comparison_auc=float(np.average([m['roc_auc'] for m in usable], weights=weights))
                if sum(weights) else None,
                evaluable_auc_folds=len(usable), within_fold_score_pairs=int(sum(weights)))


def audit(run):
    status = json.loads((run / 'status.json').read_text())
    config = json.loads((run / 'config.json').read_text())
    if not status['complete']:
        raise ValueError('Numerical run is incomplete')
    hashes = {**config['input_sha256'], **config['implementation_sha256']}
    changed = [p for p, h in hashes.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest() != h]
    if changed:
        raise ValueError(f'Frozen inputs or implementation changed: {changed}')
    models = json.loads((run / 'models.json').read_text())
    audits = json.loads((run / 'fold_audits.json').read_text())
    rows = []
    parents = set()
    for model in models:
        mode, subject, position, target = [model[k] for k in ['mode', 'subject', 'position', 'target']]
        folder = run / mode / f'sub-{subject}' / f'position-{position}'
        pred = pd.read_csv(folder / 'predictions.csv')
        if pred.event_index.duplicated().any() or not (pred.position == position).all():
            raise ValueError('Duplicate event or wrong presentation position')
        probability = pred[f'p_{target}'].to_numpy()
        predicted = pred[f'pred_{target}'].to_numpy()
        np.testing.assert_array_equal(predicted, probability >= .5)
        metrics = binary_metrics((pred.trigger.to_numpy() == target).astype(int), probability, predicted)
        for key, value in model['binary_metrics'].items():
            if isinstance(value, (int, float)):
                np.testing.assert_allclose(metrics[key], value, rtol=1e-12, atol=1e-12)
            elif metrics[key] != value:
                raise ValueError(f'Saved metric differs: {key}')
        folds = [f for f in audits if all(f[k] == model[k] for k in ['mode', 'subject', 'position', 'target'])]
        if len(folds) != 10 or len({f['fold'] for f in folds}) != 10:
            raise ValueError('Expected ten independent held-out comparisons per detector')
        for fold in folds:
            test = pred[pred.fold == fold['fold']]
            actual = binary_metrics((test.trigger.to_numpy() == target).astype(int),
                                    test[f'p_{target}'].to_numpy(), test[f'pred_{target}'].to_numpy())
            for key in ['roc_auc', 'balanced_accuracy']:
                expected = fold['test_metrics'][key]
                if expected is None:
                    assert actual[key] is None
                else:
                    np.testing.assert_allclose(actual[key], expected, atol=1e-12)
        parents.add((mode, subject, position))
        rows.append({k: model[k] for k in ['mode', 'subject', 'position', 'target', 'waveform', 'unreliable',
                                         'minimum_training_positives', 'untrainable_folds', 'single_class_test_folds']}
                    | metrics | within_fold_auc(folds)
                    | dict(source_argmax_ovr_balanced_accuracy=model['source_argmax_ovr_balanced_accuracy'],
                           flags='; '.join(model['flags'])))
    checked_parent_folds = 0
    for mode, subject, position in sorted(parents):
        folder = run / mode / f'sub-{subject}' / f'position-{position}'
        for file in folder.glob('fold-*.json'):
            fold = json.loads(file.read_text())
            assert fold['n_features'] == 100 and fold['source_scaling_reproduced']
            assert set(fold['training_event_indices']).isdisjoint(fold['test_event_indices'])
            pred = pd.read_csv(folder / 'predictions.csv')
            np.testing.assert_array_equal(pred[pred.fold == fold['fold']].event_index, fold['test_event_indices'])
            checked_parent_folds += 1
    df = pd.DataFrame(rows)
    assert len(df) == 155 and not df.duplicated(['mode', 'subject', 'position', 'target']).any()
    assert (df['mode'] == 'exploratory').sum() == 100 and (df['mode'] == 'strict').sum() == 55
    assert checked_parent_folds == 310 and len(audits) == 1550
    assert sum(f['kind'] != 'shrinkage_lda' for f in audits) == 1
    checks = dict(complete=True, frozen_hashes_verified=len(hashes), saved_detector_metrics_recomputed=len(df),
                  detector_folds_verified=len(audits), parent_folds_verified=checked_parent_folds,
                  probability_threshold=.5, new_fitting=False, permutations=0)
    return df, checks


def plot_heatmap(df, mode, output):
    subset = df[df['mode'] == mode]
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)
    for row, metric in enumerate(['balanced_accuracy', 'within_comparison_auc']):
        for col, position in enumerate([1, 2]):
            ax = axes[row, col]
            matrix = np.full((10, 5), np.nan)
            for record in subset[subset.position == position].to_dict('records'):
                matrix[record['subject'] - 3, record['target'] - 239] = record[metric]
            im = ax.imshow(np.ma.masked_invalid(matrix), vmin=0, vmax=1, cmap='RdBu', aspect='auto')
            ax.set(xticks=range(5), xticklabels=[WAVES[int(c)] for c in CODES],
                   yticks=range(10), yticklabels=range(3, 13), ylabel='Participant',
                   title=f'Position {position}: ' + ('balanced accuracy' if row == 0 else 'within-comparison AUC'))
            ax.set_facecolor('#dddddd')
            for r in range(10):
                for c in range(5):
                    if np.isfinite(matrix[r, c]):
                        mark = '*' if mode == 'exploratory' and r == 9 and col == 0 and c == 1 else ''
                        ax.text(c, r, f'{matrix[r,c]:.2f}{mark}', ha='center', va='center', fontsize=8,
                                color='white' if abs(matrix[r,c] - .5) > .28 else 'black')
    fig.colorbar(im, ax=axes, shrink=.65, label='Score (reference 0.50)')
    fig.suptitle(f'E016: {mode} detectors | fixed 100 features\n'
                 'Gray = unsupported source case; * = one missing-target training fold', fontsize=12)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def run(args):
    if args.output.exists():
        raise ValueError('Use a fresh summary output directory')
    df, checks = audit(args.run)
    args.output.mkdir(parents=True)
    df.drop(columns='confusion_matrix').to_csv(args.output / 'detectors.csv', index=False)
    metrics = ['balanced_accuracy', 'within_comparison_auc', 'roc_auc', 'sensitivity', 'specificity',
               'accuracy', 'always_rest_accuracy', 'source_argmax_ovr_balanced_accuracy']
    summary = df.groupby(['mode', 'position'])[metrics].mean().reset_index()
    summary.to_csv(args.output / 'position_summary.csv', index=False)
    df.groupby(['mode', 'position', 'waveform'])[metrics].mean().to_csv(args.output / 'waveform_summary.csv')
    for mode in ['exploratory', 'strict']:
        plot_heatmap(df, mode, args.output / f'{mode}.png')
    for name in ['models.json', 'fold_audits.json', 'config.json', 'status.json']:
        checks.setdefault('numerical_output_sha256', {})[name] = hashlib.sha256((args.run / name).read_bytes()).hexdigest()
    (args.output / 'verification.json').write_text(json.dumps(checks, indent=2) + '\n')
    records = df.drop(columns='confusion_matrix').to_dict('records')
    table = summary[['mode', 'position', 'balanced_accuracy', 'within_comparison_auc', 'roc_auc']].to_html(index=False, float_format=lambda x:f'{x:.4f}')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>E016 — Binary waveform detectors</title><style>
body{font:16px/1.55 system-ui,sans-serif;max-width:1250px;margin:32px auto;padding:0 22px;color:#1b3040;background:#f7fafc}
h1,h2{line-height:1.2}table{border-collapse:collapse;font-size:13px;background:white;width:100%}td,th{border:1px solid #ccd9e2;padding:7px;text-align:right}th{background:#eaf1f6}select{margin:0 16px 12px 6px;padding:6px}img{max-width:100%;height:auto}.scroll{overflow-x:auto}.note{background:#fff3d6;padding:14px;border-left:4px solid #c28a00}
</style><h1>E016: one-versus-rest waveform detection</h1>
<p>2 November 2026 · Task 4 · participants 3–12 · separate first/second presentations</p>
<p><b>100 exploratory binary detectors</b>, plus 55 strict-reference detectors. Each uses the same 100 selected features as its E015 parent fold and binary shrinkage LDA with equal target/rest priors. No new feature selection, spatial fitting or threshold tuning.</p>
<p>Features include voltage, slope, curvature, alpha/beta power and saved xDAWN components over 50–1,050 ms. Tests hold out complete waveform comparisons, keep pairs together, and preserve the 15.312-second train/test trigger separation. Five binary probabilities are independent outputs and need not sum to one.</p>
<h2>Observed summaries</h2><p>Equal-weight means over the available participant × target detectors in each position. The exploratory set includes all requested cases with the E015 reliability flags. Strict references include six participants at position 1 and five at position 2; they are a different subset.</p>TABLE
<p class="note"><b>Metric distinction:</b> balanced accuracy uses the fixed 0.5 threshold on held-out predictions. Within-comparison AUC compares positives and negatives only within the same fitted fold model, weighted by the number of positive–negative pairs. It excludes single-class test comparisons. Pooled AUC also compares scores across different fold models and is retained as a diagnostic, not an interchangeable estimate. These estimates do not establish statistical significance; no permutations have been run.</p>
<p>Mean exploratory balanced accuracy is below the 50% reference, while within-comparison AUC is close to 0.50. There is no clear overall gain from switching to the binary formulation. No score directions were reversed after inspecting results.</p>
<h2>Participant × waveform</h2><img src="exploratory.png" alt="Heatmaps for all 100 exploratory detectors"><img src="strict.png" alt="Heatmaps for 55 strict-reference detectors">
<h2>Inspect individual detectors</h2><label>Source<select id="mode"><option value="">All</option><option>exploratory</option><option>strict</option></select></label>
<label>Position<select id="position"><option value="">Both</option><option>1</option><option>2</option></select></label>
<label>Participant<select id="subject"><option value="">All</option>SUBJECTS</select></label>
<label>Target<select id="target"><option value="">All</option>TARGETS</select></label>
<p id="count"></p><div class="scroll"><table><thead><tr><th>Source</th><th>Participant</th><th>Position</th><th>Target</th><th>Positive / total</th><th>BA</th><th>Within-comparison AUC</th><th>Pooled AUC</th><th>Sensitivity</th><th>Specificity</th><th>Accuracy</th><th>Always-rest accuracy</th><th>AUC folds</th><th>Fallback folds</th><th>Flags</th></tr></thead><tbody id="rows"></tbody></table></div>
<h2>Support and provenance</h2><p>Targets have only 3–21 held-out positives per participant/position. Participant 12, first-position Square has one fold with no training positives: that fold uses a flagged constant-negative fallback and its errors remain in the score. Single-class test-fold AUC is undefined. A total of 1,549 LDA fits and one fallback cover 1,550 detector-folds.</p>
<p>All 155 detector metrics were recomputed from saved predictions. All 310 parent folds reproduced E015 scaling and used exactly 100 features. Frozen source/input hashes were checked. The full numerical run and fitted parameters remain local and ignored by Git.</p>
<p>Formal inference requires an appropriate full-pipeline null, including label-dependent feature selection and spatial fitting, plus multiple-comparison control. Current-monitor timing remains unverified. Reusing E015 features answers whether a binary model helps with this representation; it does not evaluate target-specific selection.</p>
<p><a href="detectors.csv">Detector CSV</a> · <a href="position_summary.csv">Position summary</a> · <a href="waveform_summary.csv">Waveform summary</a> · <a href="verification.json">Verification</a> · <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html">Scikit-learn guidance on pooled cross-validation metrics</a></p>
<script>const DATA=PAYLOAD;function fmt(x){return x==null?'—':Number(x).toFixed(3)}
function render(){let data=DATA.filter(r=>['mode','position','subject','target'].every(k=>!document.getElementById(k).value||String(r[k])===document.getElementById(k).value));document.getElementById('count').textContent=data.length+' detectors';document.getElementById('rows').innerHTML=data.map(r=>'<tr>'+[r.mode,r.subject,r.position,r.waveform,r.n_positive+' / '+r.n_events,fmt(r.balanced_accuracy),fmt(r.within_comparison_auc),fmt(r.roc_auc),fmt(r.sensitivity),fmt(r.specificity),fmt(r.accuracy),fmt(r.always_rest_accuracy),r.evaluable_auc_folds,r.untrainable_folds,(r.unreliable?'Unreliable source; ':'')+r.flags].map(v=>'<td>'+v+'</td>').join('')+'</tr>').join('')}
for(const k of ['mode','position','subject','target'])document.getElementById(k).addEventListener('change',render);render();</script></html>'''
    page = page.replace('TABLE', table).replace('SUBJECTS', ''.join(f'<option>{s}</option>' for s in range(3, 13)))
    page = page.replace('TARGETS', ''.join(f'<option value="{c}">{html.escape(WAVES[int(c)])}</option>' for c in CODES))
    page = page.replace('PAYLOAD', json.dumps(records, allow_nan=False))
    (args.output / 'index.html').write_text(page, encoding='utf-8')
    print(summary.to_string(index=False))
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('output/e016_binary_observed_20261102'))
    parser.add_argument('--output', type=Path, default=Path('output/e016_binary_summary_20261102'))
    run(parser.parse_args())
