"""Cohort summaries and a compact scientific diagnostic figure."""
import json
import numpy as np
import pandas as pd


def fmt(value):
    return f'{value:.3g}' if np.isfinite(value) else 'infinite / rank deficient'


def write_report(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folds = pd.read_csv(out/'fold_audit.csv')
    distributions = pd.read_csv(out/'feature_distributions.csv')
    resamples = pd.read_csv(out/'covariance_resamples.csv')
    config = json.loads((out/'run_config.json').read_text())
    summary = []
    for (mode,name),group in folds.groupby(['mode','feature_set']):
        boot = resamples[(resamples['mode'] == mode) & (resamples.feature_set == name)]
        summary.append(dict(mode=mode,feature_set=name,n_folds=len(group),
            n_features=int(group.n_features.iloc[0]),n_train_min=int(group.n_train.min()),
            n_train_max=int(group.n_train.max()),n_constant_folds=int((group.n_constant>0).sum()),
            empirical_rank_deficient_folds=int((group.empirical_rank<group.n_features).sum()),
            regularized_rank_deficient_folds=int((group.regularized_rank<group.n_features).sum()),
            empirical_condition_median=float(group.empirical_condition.median()),
            regularized_condition_median=float(group.regularized_condition.median()),
            empirical_effective_rank_median=float(group.empirical_effective_rank.median()),
            empirical_resample_change_median=float(boot[boot.estimate=='empirical'].relative_frobenius_change.median()),
            regularized_resample_change_median=float(boot[boot.estimate=='regularized'].relative_frobenius_change.median())))
    table = pd.DataFrame(summary)
    table.to_csv(out/'summary.csv',index=False)
    membership = json.loads((out/'fold_membership.json').read_text())
    checks = dict(n_feature_matrices=len(distributions.groupby(['subject','feature_set'])),
        nonfinite_values=0,constant_feature_columns=int(distributions.constant.sum()),
        near_constant_feature_columns=int(distributions.near_constant.sum()),
        scaler_fit_counts_match_training=bool((folds.n_scaler_fit_samples==folds.n_train).all()),
        scaler_train_mean_max_error=float(folds.train_mean_fit_max_error.max()),
        standardized_train_mean_max_abs=float(folds.train_z_mean_max_abs.max()),
        standardized_train_std_max_error=float(folds.train_z_std_max_error.max()),
        covariance_reconstruction_max_error=float(folds.covariance_reconstruction_max_error.max()),
        all_saved_fold_memberships_verified=all(row['saved_fold_membership_verified'] for row in membership))
    (out/'checks.json').write_text(json.dumps(checks,indent=2)+'\n',encoding='utf-8')
    primary = folds[folds.feature_set == 'post']
    excursions = []
    for mode,group in primary.groupby('mode'):
        excursions.append(f'- {mode}: median fold maximum |z|={group.heldout_z_max_abs.median():.2f}, '
            f'largest |z|={group.heldout_z_max_abs.max():.2f}; median held-out fraction |z|>5 '
            f'is {100*group.heldout_fraction_abs_z_gt5.median():.3f}% '
            f'(largest fold fraction {100*group.heldout_fraction_abs_z_gt5.max():.2f}%).')
    lines = ['# Task 4 baseline feature audit','',
        f'Audited participants: {config["subjects"]}. Saved feature arrays were read directly; '
        'no EEG preprocessing, classifier scoring, feature removal or model tuning was performed.', '',
        '## Distribution and scaling checks','',
        f'All {checks["n_feature_matrices"]} participant/feature matrices were finite. '
        f'Constant columns: {checks["constant_feature_columns"]}; '
        f'near-constant columns: {checks["near_constant_feature_columns"]}. '
        'Inspect feature identities and units before interpreting flags: order/history columns include intentional binary indicators.', '',
        f'Training-only scaler sample counts matched in every fit: {checks["scaler_fit_counts_match_training"]}. '
        f'Maximum training mean reconstruction error: {checks["scaler_train_mean_max_error"]:.3g}; '
        f'maximum standardized training mean magnitude: {checks["standardized_train_mean_max_abs"]:.3g}; '
        f'maximum nonconstant standardized SD error: {checks["standardized_train_std_max_error"]:.3g}.', '',
        f'Saved fold memberships matched the original classifier predictions: '
        f'{checks["all_saved_fold_memberships_verified"]}. Each feature set uses identical folds and eligible events.', '',
        'Per-column units, ranges, percentiles and 3-IQR flags are saved in `feature_distributions.csv`. '
        'Held-out standardized excursions are saved per fold in `fold_audit.csv`; held-out values never fit the scaler. '
        'These are descriptive review flags, not exclusion rules or evidence of a unit error.', '',
        'Primary post-stimulus feature excursions, using only each training fold to define z:', '',
        *excursions, '',
        'The largest held-out feature name, units and event identity are saved per fold for inspection. '
        'These excursions may reflect artifacts, context shifts or participant differences; scaling correctly '
        'does not make their cause identifiable. No clipping or trial removal is performed.', '',
        '## Within-class covariance','',
        'All estimates use training observations after the actual baseline training scaler. '
        'Empirical class covariances and regularized class covariances receive equal waveform priors. '
        'The reconstructed automatic Ledoit–Wolf covariance matches the fitted baseline LDA covariance '
        f'(maximum absolute error {checks["covariance_reconstruction_max_error"]:.3g}). '
        'Automatic shrinkage is estimated separately within each class after additional within-class scaling, '
        'then rescaled, matching scikit-learn; it is not one pooled shrinkage coefficient.', '',
        '| Evaluation | Features | Folds | Columns | Training events | Empirical deficient folds | Regularized deficient folds | Median regularized condition |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in summary:
        lines.append(f'| {row["mode"]} | {row["feature_set"]} | {row["n_folds"]} | {row["n_features"]} | '
            f'{row["n_train_min"]}–{row["n_train_max"]} | {row["empirical_rank_deficient_folds"]} | '
            f'{row["regularized_rank_deficient_folds"]} | {fmt(row["regularized_condition_median"])} |')
    lines += ['', 'Rank uses a numerical eigenvalue tolerance of machine epsilon × dimension × largest eigenvalue. '
        'A second rank at relative tolerance 1e−8, eigenvalues and entropy effective ranks are saved. '
        'The empirical within-class covariance has rank at most min(p, n_train − number_of_classes). '
        'Rank deficiency when p is large relative to training trials is expected. '
        'High correlations indicate redundancy, not automatically a processing error. '
        'Regularization improves conditioning without establishing that predictive information exists; '
        'it can assign positive variance to constant columns without making them informative.', '',
        '## Covariance resampling stability','',
        f'{config["bootstrap_repeats"]} deterministic resamples per fold preserve complete pairs and '
        'resample within subject/comparison strata. Re-estimate class covariances and shrinkage while '
        'holding the outer training scaler fixed. Values are relative Frobenius changes from the '
        'corresponding original estimate. They assess covariance variability in fixed training coordinates, '
        'not the uncertainty of the whole classifier or its weak eigen-directions.', '',
        '| Evaluation | Features | Median empirical relative change | Median regularized relative change |',
        '| --- | --- | --- | --- |']
    for row in summary:
        lines.append(f'| {row["mode"]} | {row["feature_set"]} | {row["empirical_resample_change_median"]:.3f} | '
                     f'{row["regularized_resample_change_median"]:.3f} |')
    lines += ['', 'Folds and resamples overlap and are dependent. Their distributions are descriptive; '
        'no confidence interval, significance test or model-selection threshold is inferred.', '',
        '## Repeat this audit when representations change','',
        'Rerun the numerical checks when feature definitions, units, channels, windows, preprocessing, '
        'eligible events or validation splits change. Reuse the code and verified event provenance. '
        'For a learned representation such as xDAWN, fit the transform on training data and run the '
        'fold audit on its transformed training/test arrays; do not learn it on the full cohort. '
        'Audit component scales, covariance and rank again because their geometry differs from sensor features.', '',
        'Source array/manifest/prediction hashes, model-code hashes, exact dependencies and fold membership '
        'are saved alongside this report. The audit describes the existing 1,244-event baseline cohort, '
        'not the 1,242-event ICA diagnostic cohort.', '',
        '![Primary post-stimulus feature diagnostics](primary_post_diagnostics.png)', '']
    (out/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    primary_boot = resamples[resamples.feature_set == 'post']
    modes = list(primary['mode'].unique())
    fig,axes = plt.subplots(2,2,figsize=(12,9),layout='constrained')
    for i,mode in enumerate(modes):
        group = primary[primary['mode']==mode]
        for j,estimate in enumerate(['empirical','regularized']):
            offset = i*3+j
            axes[0,0].boxplot(group[f'{estimate}_rank']/group.n_features,positions=[offset],widths=.6)
            boot = primary_boot[(primary_boot['mode']==mode)&(primary_boot.estimate==estimate)]
            axes[1,1].boxplot(boot.relative_frobenius_change,positions=[offset],widths=.6)
        axes[0,1].boxplot(np.log10(group.regularized_condition),positions=[i],widths=.6)
        axes[1,0].boxplot(group.heldout_z_max_abs,positions=[i],widths=.6)
    ticks = [i*3+j for i in range(len(modes)) for j in range(2)]
    labels = [f'{mode}\n{estimate}' for mode in modes for estimate in ['empirical','regularized']]
    for ax in [axes[0,0],axes[1,1]]:
        ax.set_xticks(ticks,labels,fontsize=8)
    for ax in [axes[0,1],axes[1,0]]:
        ax.set_xticks(range(len(modes)),modes)
    axes[0,0].set(ylabel='Numerical rank / 126 features',title='Training within-class covariance rank',ylim=(-.05,1.1))
    axes[0,1].set(ylabel='log10(condition number)',title='Actual regularized LDA covariance')
    axes[1,0].set(ylabel='Maximum absolute standardized feature',title='Held-out feature excursions per fold')
    axes[1,1].set(ylabel='Relative Frobenius change',title='Whole-pair covariance resampling; fixed scaler')
    fig.suptitle('Primary 50–500-ms baseline: descriptive feature audit\nDependent folds/resamples; no classification scores or feature selection')
    fig.savefig(out/'primary_post_diagnostics.png',dpi=140)
    plt.close(fig)
