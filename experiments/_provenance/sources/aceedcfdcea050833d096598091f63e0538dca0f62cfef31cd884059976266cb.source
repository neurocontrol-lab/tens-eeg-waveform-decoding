"""Self-contained descriptive comparison and paired participant uncertainty."""
import html
import json
import numpy as np
import pandas as pd


def holm(values):
    values = np.asarray(values,float)
    order = np.argsort(values)
    adjusted = np.empty(len(values))
    adjusted[order] = np.minimum(1,np.maximum.accumulate(values[order]*(len(values)-np.arange(len(values)))))
    return adjusted


def bootstrap_interval(values,seed=20261023):
    values = np.asarray(values,float)
    if len(values)<2:
        return [None,None]
    rng = np.random.default_rng(seed)
    samples = rng.choice(values,size=(10000,len(values)),replace=True).mean(axis=1)
    return np.quantile(samples,[.025,.975]).tolist()


def write_report(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    reports = json.loads((out/'summary.json').read_text())['reports']
    config = json.loads((out/'run_config.json').read_text())
    within = [(key,report) for key,report in reports.items() if report['mode']=='within']
    rows = []
    for identity,report in reports.items():
        for name,result in report['models'].items():
            rows.append(dict(evaluation=identity,mode=report['mode'],representation=name,
                n_events=result['n_events'],balanced_accuracy=result['balanced_accuracy'],
                macro_f1=result['macro_f1'],conditional_p=result['conditional_p'],
                first_balanced_accuracy=result['position_scores']['1']['balanced_accuracy'],
                second_balanced_accuracy=result['position_scores']['2']['balanced_accuracy']))
    table = pd.DataFrame(rows)
    table['loso_xdawn_two_window_holm_p'] = np.nan
    if 'loso' in reports and config['permutations']:
        mask = (table.evaluation=='loso') & table.representation.str.startswith('xdawn_')
        table.loc[mask,'loso_xdawn_two_window_holm_p'] = holm(table.loc[mask,'conditional_p'])
    table.to_csv(out/'scores.csv',index=False)
    aggregate = []
    for window in ['post','pre']:
        sensor = np.array([r['models'][f'sensor_{window}']['balanced_accuracy'] for _,r in within])
        xdawn = np.array([r['models'][f'xdawn_{window}']['balanced_accuracy'] for _,r in within])
        for name,values in [('sensor',sensor),('xdawn',xdawn),('xdawn_minus_sensor',xdawn-sensor)]:
            lo,hi = bootstrap_interval(values)
            aggregate.append(dict(window=window,representation=name,n_subjects=len(values),
                mean_balanced_accuracy_or_difference=float(values.mean()),
                descriptive_bootstrap_lower=lo,descriptive_bootstrap_upper=hi))
    pd.DataFrame(aggregate).to_csv(out/'within_participant_summary.csv',index=False)
    audits = pd.concat([pd.read_csv(out/key/'projected_fold_audit.csv').assign(evaluation=key,mode=r['mode'])
                        for key,r in reports.items()],ignore_index=True)
    audits.to_csv(out/'projected_fold_audit.csv',index=False)
    lines = ['# Task 4 xDAWN single-epoch waveform decoding','',
        f'Participants: {config["subjects"]}. Completed evaluations: {list(reports)}.', '',
        '## Fixed comparison','',
        'Exact original baseline cohort, epochs, average reference, eligibility and comparison-held-out/LOSO splits. '
        'Fourteen predefined electrodes; no temporal filter, ICA, onset interpolation, resampling or new exclusions. '
        'Post epochs are [50,500) ms with [−250,−50) ms voltage baseline; matched pre epochs are [−500,−50) ms '
        'with [−750,−550) ms baseline. Entire pairs and comparison groups remain together.', '',
        'xDAWN is fitted separately for each window using full-rate **training epochs and labels only**. '
        'Two components per each of five waveform classes, fixed 10% spatial covariance shrinkage, full rank; '
        'no parameter search. Nine 50-ms temporal means give 90 projected features versus 126 sensor features. '
        'Both branches use training-only scaling and equal-prior shrinkage LDA. '
        'Temporal averaging commutes with an already-fitted linear spatial projection; this shortcut is tested, '
        'but xDAWN fitting itself always uses full-rate epochs.', '',
        'Reconstructed sensor features and out-of-fold predictions/probabilities must match the original baseline. '
        'Original sensor null scores may be reused only when model code, folds, dependencies, seed and permutation '
        'sequence match. Every xDAWN null refits the spatial filters, scaler and LDA; no observed-label filter is reused.', '',
        '## Across-participant within-participant results','',
        'Balanced accuracy pools out-of-fold predictions over all five classes within each participant; '
        'participants then receive equal weight. Bootstrap intervals resample participants, not epochs, '
        'and are descriptive. Paired differences use the same participant in both branches.', '',
        '| Window | Representation | Mean balanced accuracy / difference | Descriptive 95% participant-bootstrap interval |',
        '| --- | --- | --- | --- |']
    for r in aggregate:
        lo,hi = r['descriptive_bootstrap_lower'],r['descriptive_bootstrap_upper']
        interval = f'{100*lo:.1f} to {100*hi:.1f}%' if lo is not None else 'not estimated'
        lines.append(f'| {r["window"]} | {r["representation"]} | {100*r["mean_balanced_accuracy_or_difference"]:.1f}% | {interval} |')
    lines += ['', '## Leave-one-participant-out results','',
        '| Representation | Balanced accuracy | Conditional p | Two-window xDAWN Holm p | First position BA | Second position BA |',
        '| --- | --- | --- | --- | --- | --- |']
    if 'loso' in reports:
        for r in table[table.evaluation=='loso'].itertuples():
            adjusted = f'{r.loso_xdawn_two_window_holm_p:.3f}' if np.isfinite(r.loso_xdawn_two_window_holm_p) else '—'
            p = f'{r.conditional_p:.3f}' if pd.notna(r.conditional_p) else 'not run'
            lines.append(f'| {r.representation} | {100*r.balanced_accuracy:.1f}% | {p} | {adjusted} | '
                         f'{100*r.first_balanced_accuracy:.1f}% | {100*r.second_balanced_accuracy:.1f}% |')
    else:
        lines.append('| Not run | — | — | — | — | — |')
    lines += ['', f'{config["permutations"]} restricted diagnostic permutations circularly shift complete pairs '
        'within participant/comparison and may reverse orientation comparison-wide. They preserve paired support '
        'and class counts, but exchangeability can be limited by temporal nonstationarity. '
        'The 20% reference alone is not evidence of above-chance information. The two LOSO xDAWN windows receive '
        'Holm correction; participant-specific p-values in scores.csv are exploratory and uncorrected. '
        'Within-participant nulls are not combined into a cohort p-value.', '',
        '## Projected-feature audit','',
        'All observed training/test projected features are finite. Training-only distribution/scaling and '
        'actual LDA covariance reconstruction checks are rerun inside each fold. Component identities belong '
        'to each training fit, so their raw magnitudes are not pooled across fold-specific bases.', '',
        '| Evaluation | Window | Folds | Columns | Empirical deficient folds | Regularized deficient folds | Median regularized condition | Largest held-out abs z |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for (mode,window),group in audits.groupby(['mode','window']):
        lines.append(f'| {mode} | {window} | {len(group)} | {int(group.n_features.iloc[0])} | '
            f'{int((group.empirical_rank<group.n_features).sum())} | {int((group.regularized_rank<group.n_features).sum())} | '
            f'{group.regularized_condition.median():.1f} | {group.heldout_z_max_abs.max():.1f} |')
    lines += ['', 'Ten whole-pair training covariance resamples per fold are saved for descriptive stability checks '
        '(or the run-configured count). The xDAWN filters and outer training scaler remain fixed in these '
        'resamples: they assess downstream covariance conditional on the learned representation, not uncertainty '
        'of the whole xDAWN pipeline. No flags select channels, components or trials.', '',
        '## Limits','',
        '- This is one fixed exploratory spatial representation. A null result does not exclude other EEG information.',
        '- A post-trigger score can reflect electrical artifacts, intensity/charge differences or contextual history. '
        'Physical current timing remains unverified. Spatial filtering cannot establish a cortical origin.',
        '- Pre-stimulus decoding diagnoses context/carryover; a low score does not rule out post-only artifacts.',
        '- Position-specific scores are descriptive; nonsignificance does not establish equivalence or remove confounds.',
        '- Saved spatial patterns are fold-specific sensor mixtures, not source-localized neural generators.', '',
        '## Figures','', '![Participant performance](participant_balanced_accuracy.png)','',
        '![LOSO confusion matrices](loso_confusion_matrices.png)','']
    (out/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    fig,axes = plt.subplots(1,2,figsize=(12,5),layout='constrained')
    subjects = [int(key.split('-')[1]) for key,_ in within]
    for ax,window in zip(axes,['post','pre']):
        for name,marker in [('sensor','o'),('xdawn','s')]:
            values = [r['models'][f'{name}_{window}']['balanced_accuracy']*100 for _,r in within]
            ax.plot(subjects,values,marker=marker,label=name)
        ax.axhline(20,color='.4',ls='--',label='20% reference')
        ax.set(title=window,xlabel='Participant',ylabel='Balanced accuracy (%)',xticks=subjects,ylim=(0,100))
        ax.legend()
    fig.suptitle('Held-out waveform comparisons; fixed sensor versus fold-fitted xDAWN')
    fig.savefig(out/'participant_balanced_accuracy.png',dpi=150)
    plt.close(fig)
    fig,axes = plt.subplots(1,2,figsize=(11,5),layout='constrained')
    labels = ['Sine','Square','Triangle','LinDec','LinInc']
    for ax,name in zip(axes,['sensor_post','xdawn_post']):
        if 'loso' not in reports:
            ax.text(.5,.5,'LOSO not run',ha='center',va='center')
            continue
        counts = np.asarray(reports['loso']['models'][name]['confusion_matrix'])
        values = counts/counts.sum(axis=1,keepdims=True)
        im = ax.imshow(values,vmin=0,vmax=1,cmap='Blues')
        for i in range(5):
            for j in range(5):
                ax.text(j,i,f'{values[i,j]:.2f}',ha='center',va='center',color='white' if values[i,j]>.6 else 'black')
        ax.set(title=name,xlabel='Predicted',ylabel='Delivered',xticks=range(5),yticks=range(5),
               xticklabels=labels,yticklabels=labels)
        ax.tick_params(axis='x',rotation=30)
    fig.suptitle('LOSO post-trigger confusion matrices; row-normalized')
    fig.savefig(out/'loso_confusion_matrices.png',dpi=150)
    plt.close(fig)
    pictures = ''.join(f'<img src="{name}">' for name in ['participant_balanced_accuracy.png','loso_confusion_matrices.png'])
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Task 4 xDAWN</title>'
        '<style>body{max-width:1200px;margin:30px auto;font:16px/1.5 system-ui}pre{white-space:pre-wrap}img{max-width:100%}</style>'
        '<pre>'+html.escape('\n'.join(lines[:lines.index('## Figures')]))+'</pre>'+pictures,encoding='utf-8')
