"""Aggregate matched decoding and audit findings without selecting features."""
import html
import json
import numpy as np
import pandas as pd
from experiments.waveform_decoding.e004_xdawn_decoding.report import holm, bootstrap_interval


def write_report(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    reports = json.loads((out/'summary.json').read_text())['reports']
    config = json.loads((out/'run_config.json').read_text())
    within = [r for r in reports.values() if r['mode']=='within']
    if config['permutations']:
        resolution = 1/(config['permutations']+1)
        resolution_note = (f'Raw Monte Carlo p resolution = {resolution:.6g}; minimum possible '
            f'Holm p = {min(1,6*resolution):.6g} for the six-test LOSO family and '
            f'{min(1,6*len(within)*resolution):.6g} for the participant-specific family. '
            'A family whose minimum exceeds 0.05 cannot establish significance at that threshold; '
            'this limitation does not establish a negative finding. ')
    else:
        resolution_note = 'No permutations: descriptive smoke run only. '
    rows = []
    for identity, report in reports.items():
        for name, result in report['models'].items():
            rows.append(dict(evaluation=identity,mode=report['mode'],representation=name,
                n_events=result['n_events'],balanced_accuracy=result['balanced_accuracy'],
                macro_f1=result['macro_f1'],conditional_p=result['conditional_p'],
                first_balanced_accuracy=result['position_scores']['1']['balanced_accuracy'],
                second_balanced_accuracy=result['position_scores']['2']['balanced_accuracy']))
    table = pd.DataFrame(rows)
    table['family_holm_p'] = np.nan
    if config['permutations']:
        for mode in table['mode'].unique():
            mask = table['mode']==mode
            table.loc[mask,'family_holm_p'] = holm(table.loc[mask,'conditional_p'])
    table.to_csv(out/'scores.csv',index=False)
    aggregate = []
    for window in ['post','pre']:
        voltage = np.array([r['models'][f'voltage_{window}']['balanced_accuracy'] for r in within])
        for name in ['voltage','power','combined']:
            values = np.array([r['models'][f'{name}_{window}']['balanced_accuracy'] for r in within])
            for label, data in [(name,values)]+([(name+'_minus_voltage',values-voltage)] if name!='voltage' else []):
                lo, hi = bootstrap_interval(data)
                aggregate.append(dict(window=window,representation=label,n_subjects=len(data),
                    mean_balanced_accuracy_or_difference=float(data.mean()),
                    descriptive_bootstrap_lower=lo,descriptive_bootstrap_upper=hi))
    pd.DataFrame(aggregate).to_csv(out/'within_participant_summary.csv',index=False)
    audits = pd.concat([pd.read_csv(out/key/'fold_audit.csv').assign(evaluation=key,mode=r['mode'])
                        for key,r in reports.items()],ignore_index=True)
    audits.to_csv(out/'fold_audit.csv',index=False)
    lines = ['# Task 4 fixed single-trial spectral decoding','',
        f'Participants: {config["subjects"]}; {sum(r["n_events"] for r in within)} eligible events.','',
        '## Fixed design','',
        'The original baseline cohort, 14 electrodes, average reference, eligibility, comparison-held-out '
        'within-participant folds and complete-subject LOSO folds are preserved. No temporal filters, ICA, '
        'onset interpolation or new exclusions. Reconstructed voltage features, predictions and probabilities '
        'must reproduce the original baseline. Every voltage null is independently recomputed; compatible '
        'original null scores must match exactly.','',
        'Post epochs: [50,500) ms, voltage baseline [−250,−50) ms. Pre epochs: [−500,−50) ms, separate '
        'voltage baseline [−750,−550) ms. Power uses each entire 450-ms epoch: constant detrend, Hann taper, '
        'one-sided density periodogram, no zero-padding or between-epoch averaging. Alpha 8–13 Hz and beta '
        '13–30 Hz are integrated using a piecewise-linear PSD with interpolated exact band boundaries. '
        'Native frequency spacing is approximately 2.22 Hz; band boundaries are conventions, not sharply '
        'resolved physiological separations. PSD uses µV²/Hz; band integrals µV²; features 10 log10(power / 1 µV²). '
        f'Fixed numerical floor: {config["power_floor_uv2"]:g} µV²; hits are saved. '
        'Epoch detrending cancels the constant voltage-baseline subtraction for power; this is not a pre/post '
        'power ratio or a baseline-corrected spectral change.','',
        'Voltage = 126 fixed temporal means; power = 28 band/electrode values; combined = 154 columns. '
        'All use training-only StandardScaler and equal-prior auto-shrinkage LDA. No supervised feature '
        'transform, channel/window/band selection or parameter search. Every permutation refits scaler and '
        'LDA with the same restricted labels across all six representations/windows.','',
        '## Within-participant results','',
        'Each participant pools all out-of-fold predictions over five classes, then participants receive '
        'equal weight. Descriptive 95% intervals resample participants (10,000 draws, seed 20261023), '
        'not epochs or folds. Differences use paired participants. These intervals are not a formal '
        'test of model superiority.','',
        '| Window | Representation | Mean BA / difference | Descriptive 95% interval |',
        '| --- | --- | ---: | --- |']
    for r in aggregate:
        lo, hi = r['descriptive_bootstrap_lower'],r['descriptive_bootstrap_upper']
        interval = f'{100*lo:.2f} to {100*hi:.2f}%' if lo is not None else 'not estimated'
        lines.append(f'| {r["window"]} | {r["representation"]} | {100*r["mean_balanced_accuracy_or_difference"]:.2f}% | {interval} |')
    lines += ['', '## Leave-one-participant-out results','',
        '| Representation | BA | Conditional p | Six-test Holm p | First BA | Second BA |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in table[table['mode']=='loso'].itertuples():
        p = f'{r.conditional_p:.3f}' if pd.notna(r.conditional_p) else 'not run'
        adj = f'{r.family_holm_p:.3f}' if np.isfinite(r.family_holm_p) else 'not run'
        lines.append(f'| {r.representation} | {100*r.balanced_accuracy:.2f}% | {p} | {adj} | '
            f'{100*r.first_balanced_accuracy:.2f}% | {100*r.second_balanced_accuracy:.2f}% |')
    lines += ['', f'{config["permutations"]} restricted permutations shift complete pairs within each '
        'participant/comparison and may flip orientation comparison-wide. This conditional diagnostic null '
        'retains paired support and comparison-specific class availability; temporal nonstationarity can '
        'limit exchangeability. Chance reference = 20%; a numerical value above 20% alone is not evidence. '
        'LOSO receives Holm correction across all six tests. A separate Holm family covers all '
        f'{6*len(within)} participant-specific tests; no within-participant nulls are pooled into a cohort p value. '
        +resolution_note+
        'Position-specific scores are descriptive; their folds need not contain balanced class support.','',
        '## Repeated feature audit','',
        'Distribution, constant/near-constant, training-only scaling and actual LDA covariance reconstruction '
        'checks run for every observed training/test fold. Units are kept separate for near-constant flags '
        'in the combined representation. Whole pairs are resampled within participant/comparison to check '
        'covariance stability, conditional on each fixed outer-training scaler. Flags cause no exclusions, '
        'clipping or retuning.','',
        '| Evaluation | Representation | Folds | Columns | Empirical deficient | Regularized deficient | Median regularized condition | Max held-out abs z |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for (mode, key), group in audits.groupby(['mode','representation']):
        lines.append(f'| {mode} | {key} | {len(group)} | {int(group.n_features.iloc[0])} | '
            f'{int((group.empirical_rank<group.n_features).sum())} | '
            f'{int((group.regularized_rank<group.n_features).sum())} | '
            f'{group.regularized_condition.median():.1f} | {group.heldout_z_max_abs.max():.1f} |')
    hits = sum(sum(d['floor_hits'] for d in r['spectral_diagnostics'].values()) for r in within)
    verified = all(r['voltage_null_replay_verified'] for r in reports.values())
    replayed = sum(sum(r['voltage_null_replay_counts_per_window'].values()) for r in reports.values())
    lines += ['',f'Numerical floor hits across unique participant features: {hits}. '
        f'Original voltage null replay verified for all evaluations: {verified} '
        f'({replayed} scores in the shared original permutation prefix).','',
        '## Interpretation limits','',
        'These are total single-trial power features, containing evoked as well as ongoing activity; they '
        'are not isolated induced oscillations. No class-specific ERP is subtracted from held-out trials. '
        'Different stimulation shapes need not produce EEG oscillations at their stimulus frequencies. '
        'Power differences can also reflect muscle, movement, drift or stimulation contamination. '
        'Pre-trigger activity may encode anticipation or preceding stimulation, especially before the second '
        'presentation; it is a confound diagnostic, not a guaranteed null. Trigger-to-current timing remains '
        'unverified. Negative results apply to these bands/windows/features and this validation design, '
        'not all possible waveform information.','',
        '## Figures','', '![Participant scores](participant_balanced_accuracy.png)','',
        '![LOSO confusion matrices](loso_confusion_matrices.png)']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fig, axes = plt.subplots(1,2,figsize=(12,4.8),layout='constrained',sharey=True)
    colors = ['#4c78a8','#e88725','#459c75']
    for ax, window in zip(axes,['post','pre']):
        for j, (name,color) in enumerate(zip(['voltage','power','combined'],colors)):
            values = table[(table['mode']=='within') & (table.representation==f'{name}_{window}')]
            ax.plot(values.evaluation,100*values.balanced_accuracy,'o-',label=name,color=color,alpha=.85)
        ax.axhline(20,color='black',ls='--',lw=1,label='20% reference')
        ax.set(title=f'{window.capitalize()}-trigger',ylabel='Balanced accuracy (%)',ylim=(0,50))
        ax.tick_params(axis='x',rotation=50)
        ax.legend(fontsize=8)
    fig.savefig(out/'participant_balanced_accuracy.png',dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1,3,figsize=(13,4.3),layout='constrained')
    labels = ['Sine','Square','Triangle','LinDec','LinInc']
    for ax, name in zip(axes,['voltage','power','combined']):
        if 'loso' not in reports:
            ax.text(.5,.5,'LOSO not run',ha='center')
            continue
        matrix = np.array(reports['loso']['models'][name+'_post']['confusion_matrix'])
        fractions = matrix/matrix.sum(axis=1,keepdims=True)
        ax.imshow(fractions,vmin=0,vmax=1,cmap='Blues')
        for i in range(5):
            for j in range(5):
                ax.text(j,i,f'{fractions[i,j]:.2f}',ha='center',va='center',color='white' if fractions[i,j]>.6 else 'black',fontsize=9)
        ax.set(title=name,xlabel='Predicted',ylabel='Delivered',xticks=range(5),yticks=range(5),xticklabels=labels,yticklabels=labels)
        ax.tick_params(axis='x',rotation=35)
    fig.suptitle('LOSO post-trigger confusion matrices; row-normalized')
    fig.savefig(out/'loso_confusion_matrices.png',dpi=160)
    plt.close(fig)
    pictures = ''.join(f'<img src="{name}">' for name in ['participant_balanced_accuracy.png','loso_confusion_matrices.png'])
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Task 4 spectral comparison</title>'
        '<style>body{max-width:1300px;margin:30px auto;font:16px/1.5 system-ui}pre{white-space:pre-wrap}img{max-width:100%}</style>'
        '<pre>'+html.escape('\n'.join(lines[:lines.index('## Figures')]))+'</pre>'+pictures,encoding='utf-8')
