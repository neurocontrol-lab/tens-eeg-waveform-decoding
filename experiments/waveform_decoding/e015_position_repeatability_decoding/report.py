"""Observed E015 results, with strict and unreliable estimates kept separate."""
import argparse
import collections
import html
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.waveform_decoding.e001_voltage_baseline.evaluate import scores
from experiments.response_characterization.e011_electrode_bin_repeatability.__main__ import sha,write_json
from .confusion import prepare_confusions


def load_checked(root):
    status=json.loads((root/'status.json').read_text());config=json.loads((root/'config.json').read_text())
    if not status['complete'] or status['fit_failed']:raise ValueError(f'Run incomplete: {root}')
    for p,h in {**config['input_sha256'],**config['implementation_sha256']}.items():
        if sha(p)!=h:raise ValueError(f'Frozen run input/source changed: {p}')
    models=json.loads((root/'models.json').read_text())
    for m in models:
        if m['status']!='observed_complete':continue
        p=root/f'sub-{m["subject"]}'/f'position-{m["position"]}'
        pred=pd.read_csv(p/'predictions.csv')
        if pred.event_index.duplicated().any() or (pred.fold<0).any():raise ValueError('Missing/duplicate held-out events')
        for name,r in m['models'].items():
            got=scores(pred.trigger.to_numpy(),pred[name+'_predicted'].to_numpy())
            for key in ['balanced_accuracy','macro_f1','accuracy']:
                if not np.isclose(r[key],got[key],atol=1e-12,rtol=0):raise ValueError('Stored score differs from held-out predictions')
            if r['confusion_matrix']!=got['confusion_matrix']:raise ValueError('Confusion matrix mismatch')
            if set(r['selected_counts'])!={100}:raise ValueError('Feature budget shortfall must be reported')
            prob=[f'{name}_prob_{c}' for c in range(239,244)]
            if all(c in pred for c in prob):
                values=pred[prob].to_numpy()
                if not np.isfinite(values).all() or not np.allclose(values.sum(axis=1),1.,atol=1e-10):raise ValueError('Invalid held-out probabilities')
    return config,status,models


def render(strict,exploratory,output,replace_report=False):
    if output.exists() and not replace_report:raise ValueError('Use a fresh summary directory')
    if output.exists() and not (output/'verification.json').is_file():raise ValueError('Refuse to overwrite an unknown directory')
    config_s,status_s,ms=load_checked(strict);config_e,status_e,me=load_checked(exploratory)
    if len(me)!=20 or any(m['status']!='observed_complete' or not m['unreliable'] for m in me):raise ValueError('Need all 20 flagged exploratory cases')
    output.mkdir(parents=True,exist_ok=True);(output/'figures').mkdir(exist_ok=True)
    rows=[]
    for mode,models in [('strict',ms),('exploratory',me)]:
        for m in models:
            if m['status']=='observed_complete':
                for name,r in m['models'].items():
                    rows.append(dict(mode=mode,subject=m['subject'],position=m['position'],selector=name,
                        balanced_accuracy=r['balanced_accuracy'],n_events=r['n_events'],unreliable=mode=='exploratory'))
    frame=pd.DataFrame(rows);frame.to_csv(output/'model_scores.csv',index=False)
    confusion_html,class_summary=prepare_confusions([('strict',strict,ms),('exploratory',exploratory,me)],output)
    means=frame.groupby(['mode','position','selector']).agg(mean_balanced_accuracy=('balanced_accuracy','mean'),
        participants=('subject','nunique')).reset_index();means.to_csv(output/'position_means.csv',index=False)
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharey=True,layout='constrained')
    for row,mode in enumerate(['strict','exploratory']):
        for col,pos in enumerate([1,2]):
            ax=axes[row,col]
            for selector,color,marker in [('repeatability','#0072B2','o'),('anova','#D55E00','s')]:
                d=frame[(frame['mode']==mode)&(frame.position==pos)&(frame.selector==selector)].sort_values('subject')
                ax.scatter(d.subject,d.balanced_accuracy*100,color=color,marker=marker,label=selector,s=42)
            ax.axhline(20,color='#555',ls='--',lw=1,label='20% reference')
            ax.set(title=f'{"Strict supported subset" if mode=="strict" else "Unreliable exploratory: all participants"} — P{pos}',
                   xlabel='Participant',ylabel='Held-out balanced accuracy (%)',xticks=range(3,13),ylim=(0,40))
            ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8,loc='upper right')
    fig.suptitle('E015: top-100 multiscale features + shrinkage LDA\nObserved scores only; no permutation inference',fontsize=14)
    fig.savefig(output/'figures/observed_accuracy.png',dpi=150);plt.close(fig)
    counts=[];calibration=collections.Counter();fold_count=collections.Counter()
    for mode,root in [('strict',strict),('exploratory',exploratory)]:
        for p in root.glob('sub-*/position-*/fold-*.json'):
            d=json.loads(p.read_text());fold_count[mode]+=1
            if 'calibration' in d:calibration[d['calibration']['calibration_per_class']]+=1
            for name,r in d['models'].items():
                c=collections.Counter(f['family'] for f in r['selected_features'])
                counts.extend(dict(mode=mode,subject=d['subject'],position=d['position'],fold=d['fold'],selector=name,family=k,count=v) for k,v in c.items())
            with np.load(p.with_suffix('.npz')) as z:
                for name in d['models']:
                    scale=z[name+'_scaler_scale']
                    if not np.isfinite(scale).all() or (scale<=0).any():raise ValueError('Invalid training-only scaler')
    pd.DataFrame(counts).to_csv(output/'selected_family_counts.csv',index=False)
    strict_map={(m['subject'],m['position']):m for m in ms}
    lines=['# E015 — Position-specific top-100 decoding: observed findings','',
        '**Date:** 2 November 2026. **Status:** Observed fitting complete; permutation inference pending.','',
        'All 20 participant/position cases were run with an explicitly unreliable exploratory pipeline. '
        'The original strict design also fitted 11 supported cases; nine were ineligible under that design. '
        'No model fit failed. Every fold selected 100 scalar features, separately per participant and position.','',
        'The primary selector ranks positive median split-profile correlations, squared after retaining the sign. '
        'A matched ANOVA selector uses the same candidate pool and shrinkage-LDA decoder. '
        'Features are mean voltage, signed/RMS slope, signed/RMS curvature, alpha/beta power and xDAWN amplitudes. '
        'The usual pool has 25,364 candidates. One exploratory fold lacking a training class has eight rather than ten xDAWN components, giving 25,214 candidates.','',
        '## Mean held-out balanced accuracy','',
        '| Pipeline | Position | Participants | Repeatability-selected | ANOVA-selected |',
        '| --- | --- | ---: | ---: | ---: |']
    for mode in ['strict','exploratory']:
        for pos in [1,2]:
            d=means[(means['mode']==mode)&(means.position==pos)].set_index('selector')
            lines.append(f'| {mode} | {pos} | {int(d.loc["repeatability","participants"])} | {100*d.loc["repeatability","mean_balanced_accuracy"]:.2f}% | {100*d.loc["anova","mean_balanced_accuracy"]:.2f}% |')
    lines+=['','Both pipeline means are below the nominal 20% five-class reference. '
        'These are descriptive point estimates, not permutation-test results. '
        'The two positions in the strict subset have different participant coverage; their means are not a paired carryover test. '
        'The exploratory and strict pipelines also use different scoring/calibration procedures and must not be pooled into a single accuracy.','',
        '## Individual repeatability-selected results','',
        '| Participant | Strict P1 | Strict P2 | Unreliable exploratory P1 | Unreliable exploratory P2 |',
        '| --- | ---: | ---: | ---: | ---: |']
    lookup={(r['mode'],r['subject'],r['position']):r['balanced_accuracy'] for r in rows if r['selector']=='repeatability'}
    for s in range(3,13):
        vals=[lookup.get((mode,s,pos)) for mode,pos in [('strict',1),('strict',2),('exploratory',1),('exploratory',2)]]
        lines.append('| '+str(s)+' | '+' | '.join('Ineligible' if v is None else f'{100*v:.2f}%' for v in vals)+' |')
    lines+=['','## Class recall and confusion matrices','',
        'The interactive report includes confusion matrices for each participant, position and selector, '
        'plus pooled and equal-participant summaries. Rows are true delivered waveforms; columns are predicted waveforms. '
        'Diagonal percentages are class recall. The table below averages participant recalls equally, consistent with the reported mean balanced accuracy.','',
        '| Pipeline | Position | Selector | Sine | Square | Triangle | LinDec | LinInc |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for mode in ['strict','exploratory']:
        for pos in [1,2]:
            for selector in ['repeatability','anova']:
                d=class_summary[(class_summary['mode']==mode)&(class_summary.position==pos)&(class_summary.selector==selector)].sort_values('trigger')
                lines.append('| '+f'{mode} | {pos} | {selector} | '+' | '.join(f'{100*v:.2f}%' for v in d.mean_participant_recall)+' |')
    lines+=['','Pooled count and row-normalized matrices retain event support; equal-participant percentages give each participant the same weight within each true class. '
        'Different trial counts can make these summaries differ. Class-level differences are descriptive, not significance-tested. '
        'Individual and summarized recalls are exported in `class_recall.csv` and `class_recall_summary.csv`, with all matrices in `confusion_matrices.json`.','',
        '## Meaning of the unreliable flag','',
        'All exploratory models retain the original comparison-held-out outer folds, complete-pair grouping and temporal embargo. '
        'xDAWN fits only outer-training epochs/labels, but there is no calibration set independent of the internal A/B ranking halves. '
        'The resulting split correlation is an in-sample selection heuristic, not an independently validated repeatability estimate. '
        'Observed comparison-balanced class profiles replace regression-adjusted waveform contrasts; they can retain context and drift effects.','',
        'Ranking uses classes having at least two training epochs, so both A/B halves can contain them. '
        'If a waveform is absent from training, the classifier cannot learn or predict it. '
        'Its predicted probability remains zero, and its held-out errors remain in the five-class score; no class or test epoch is silently removed.','',
        'Participant 12, position 1, has one outer fold without Square training examples. '
        'That fold uses four training/ranking classes and eight xDAWN components. '
        'Its output is explicitly flagged with `missing_training_class` and `fewer_than_five_ranking_classes`.','',
        '## Controls and reproducibility','',
        '- Strict xDAWN calibration is disjoint from both ranking halves; all feature families use those same ranking data. '
        f'Calibration used two epochs per class in {calibration[2]} strict folds and one per class in {calibration[1]} folds, based only on metadata feasibility.',
        '- Selection and scaling are fitted on training data. Held-out values enter transformation and prediction only. '
        'Tests verify that changing held-out signals does not alter selected features or fitted coefficients.',
        '- The effective filter guard remains seven seconds per side at an L1-tail tolerance of 1e-4. '
        'Baseline/derivative support gives a 15.312-second train/test trigger separation; complete training pairs inside this guard are purged.',
        '- First and second presentation models remain separate. Their performance difference alone does not identify physiological carryover.',
        '- No matched one-second pre-trigger control or permutation inference has run. The cached pre-trigger period is only 300 ms. '
        'The code includes a tested conditional pair-label shuffle helper; a full null runner still requires the complete scoring/spatial-fitting pipeline to be rerun.',
        '- Saved artifacts include all held-out predictions, confusion matrices, selected scalar feature IDs, calibration/ranking memberships, '
        'spatial filters, scaling parameters, classifier coefficients, support failures and source/input digests.','',
        '## Interpretation','',
        'The current observed results do not show a cohort-level accuracy above the 20% reference. '
        'A few individual point estimates exceed 20%, but no above-chance significance or improvement over ANOVA is established. '
        'Repeatable waveform-mean patterns do not necessarily translate into useful single-epoch features. '
        'These runs do not establish absence of waveform information, and unreliable cases remain explicitly marked.','',
        '## Local output directories','',
        '- `output/e015_strict_observed_20261102/`: strict supported fits and explicit ineligible cases.',
        '- `output/e015_exploratory_observed_20261102/`: all 20 flagged exploratory models.',
        '- `output/e015_observed_summary_20261102/`: aggregate tables, feature-family counts, figure and local HTML report.','',
        'See the [run guide](../experiments/waveform_decoding/e015_position_repeatability_decoding/README.md) '
        'and [design audit](E015_DESIGN_FEASIBILITY.md).','']
    markdown='\n'.join(lines);(output/'RESULTS.md').write_text(markdown,encoding='utf-8')
    means_html=means.copy();means_html['mean_balanced_accuracy']=means_html.mean_balanced_accuracy.map(lambda v:f'{100*v:.2f}%')
    individual=frame.pivot(index=['subject','position','mode','unreliable'],columns='selector',values='balanced_accuracy').reset_index()
    for column in ['repeatability','anova']:individual[column]=individual[column].map(lambda v:f'{100*v:.2f}%')
    body='<h2>Position means</h2>'+means_html.to_html(index=False,border=0)
    body+='<h2>Individual models</h2>'+individual.to_html(index=False,border=0)
    body+=confusion_html
    body+='<h2>Reliability flags</h2><p>All exploratory rows lack independent xDAWN calibration and use observed, not regression-adjusted, class profiles. Participant 12 P1 also has a fold with no Square training examples. Missing classes receive zero predicted probability; their held-out errors remain in the five-class score. Full diagnostics remain in the saved fold JSONs.</p>'
    body+='<details><summary>Complete methods and findings</summary><pre style="white-space:pre-wrap">'+html.escape(markdown)+'</pre></details>'
    selector_intro='''<section aria-labelledby="feature-selection-title" style="background:#f2f6fa;padding:18px;border-radius:8px;margin:18px 0">
<h2 id="feature-selection-title" style="margin-top:0">What do the two feature-selection methods mean?</h2>
<p>Both methods choose <b>100 scalar features</b> from the same candidate pool: voltage, slope, curvature, alpha/beta power and xDAWN amplitudes. A feature is one measurement, such as mean C3 voltage in a particular time bin. The selected features then enter the <b>same shrinkage-LDA classifier</b>.</p>
<ul><li><b>Repeatability-selected:</b> for each candidate feature, estimate its response to each waveform in two training-data halves, giving a pattern of values across the represented waveforms in each half. Correlate those two patterns across 100 repeated splits. Rank features by the squared positive part of the median correlation: <code>max(median r, 0)<sup>2</sup></code>. This favors a waveform-response pattern that recurs across splits; it does not guarantee accurate single-epoch decoding.</li>
<li><b>ANOVA-selected:</b> rank each feature by its one-way ANOVA F score: how much its mean differs between waveform classes relative to variability within those classes. This favors separation of class means, without explicitly testing split-half repeatability. The F score is a ranking criterion, not a declaration that each selected feature is statistically significant.</li></ul>
<p><b>Training only:</b> feature selection is repeated separately for each participant, presentation position and outer validation fold. Its top 100 features are shared across waveform classes; held-out comparisons do not determine the selection. In the strict pipeline, xDAWN calibration is separate from ranking and repeatability uses adjusted waveform effects. In the exploratory pipeline, xDAWN uses the same outer-training data as ranking and the patterns use observed context-balanced means, so repeatability is an in-sample selection heuristic.</p>
</section>'''
    page='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>E015 observed decoding</title><style>body{font:15px system-ui;max-width:1200px;margin:25px auto;padding:20px;color:#172437}img{width:100%;height:auto}pre{font:14px system-ui;line-height:1.6}table{border-collapse:collapse;width:100%;margin:15px 0}td,th{padding:8px;text-align:left;border-bottom:1px solid #ddd}summary{cursor:pointer;font-weight:600}</style><h1>E015 observed decoding</h1><p>Strict and unreliable exploratory cases are reported separately. No permutation inference.</p><img src="figures/observed_accuracy.png" alt="Held-out balanced accuracy by participant and position">'+body+'</html>'
    page=page.replace('<img src=',selector_intro+'<img src=',1)
    (output/'index.html').write_text(page,encoding='utf-8')
    write_json(output/'verification.json',dict(strict_models=11,exploratory_models=20,
        strict_folds=fold_count['strict'],exploratory_folds=fold_count['exploratory'],
        all_saved_scores_recomputed_from_held_out_predictions=True,all_selected_counts_100=True,
        all_scalers_finite_positive=True,all_probabilities_valid=True,source_and_input_hashes_unchanged=True,
        confusion_matrices_recomputed=62,class_recall_rows=310,confusion_reporter_sha256=sha(Path(__file__).with_name('confusion.py')),
        inference_complete=False,permutations=0,reporter_sha256=sha(Path(__file__))))
    print(means.to_string(index=False),flush=True)
    return markdown


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--strict',type=Path,default=Path('output/e015_strict_observed_20261102'))
    p.add_argument('--exploratory',type=Path,default=Path('output/e015_exploratory_observed_20261102'))
    p.add_argument('--output',type=Path,default=Path('output/e015_observed_summary_20261102'))
    p.add_argument('--replace-report',action='store_true')
    a=p.parse_args();render(a.strict,a.exploratory,a.output,a.replace_report)
