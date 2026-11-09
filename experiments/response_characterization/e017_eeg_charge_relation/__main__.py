"""Compare absolute-charge regressions on an amplitude tier held out of fitting."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .extract import WAVES, digest


MODELS = ('constant', 'shared', 'separate')
WINDOWS = ('post_uv', 'pre_uv')
MIN_TIER_TRIALS = 3
BOOTSTRAPS = 10000
SEED = 1709


def fit_slopes(calibration, outcome):
    """Separate absolute-charge intercepts, with shared or separate slopes."""
    cells = calibration.groupby('waveform', sort=False)
    q = calibration.charge_uC - cells.charge_uC.transform('mean')
    x = calibration[outcome] - cells[outcome].transform('mean')
    pooled = float(q @ x / (q @ q))
    rows = []
    for waveform, group in cells:
        q_centered, x_centered = q.loc[group.index], x.loc[group.index]
        separate = float(q_centered @ x_centered / (q_centered @ q_centered))
        for model, beta in zip(MODELS, (0., pooled, separate)):
            alpha = float(group[outcome].mean() - beta * group.charge_uC.mean())
            rows.append(dict(waveform=waveform, model=model, alpha_uv=alpha, beta_uv_per_uc=beta))
    return pd.DataFrame(rows)


def eligible_cells(data, responded_only):
    valid = data.eeg_qc.isna() & np.isfinite(data.post_uv) & np.isfinite(data.pre_uv)
    if responded_only:
        valid &= data.response_valid
    counts = (data.loc[valid].groupby(['subject', 'waveform', 'tier']).size()
              .unstack(fill_value=0).reindex(columns=[0, 1, 2], fill_value=0))
    counts = counts.reindex(pd.MultiIndex.from_frame(data[['subject', 'waveform']].drop_duplicates()), fill_value=0)
    counts.columns = ['threshold_n', 'middle_n', 'upper_n']
    counts['included'] = counts.min(axis=1) >= MIN_TIER_TRIALS
    keys = pd.MultiIndex.from_frame(data[['subject', 'waveform']])
    return data.loc[valid & keys.isin(counts.index[counts.included])].copy(), counts.reset_index()


def interval(values):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(SEED)
    means = rng.choice(values, size=(BOOTSTRAPS, len(values)), replace=True).mean(axis=1)
    return dict(mean=float(values.mean()), ci95=np.quantile(means, [.025, .975]).tolist(), participants=len(values))


def analyze(data, responded_only=False):
    usable, counts = eligible_cells(data, responded_only)
    if usable.subject.nunique() < 2:
        raise ValueError('Fewer than two participants have usable amplitude tiers')
    predictions, coefficients = [], []
    for subject, trials in usable.groupby('subject'):
        calibration = trials.loc[trials.tier.lt(2)]
        upper = trials.loc[trials.tier.eq(2)]
        for outcome in WINDOWS:
            fitted = fit_slopes(calibration, outcome).assign(subject=subject, window=outcome, fit='lower_two_tiers')
            descriptive = fit_slopes(trials, outcome).assign(subject=subject, window=outcome, fit='all_tiers')
            coefficients.extend([fitted, descriptive])
            scored = upper[['waveform', 'charge_uC', 'csv_row', outcome]].merge(fitted, on='waveform')
            scored = scored.rename(columns={outcome: 'observed_uv'})
            scored['predicted_uv'] = scored.alpha_uv + scored.beta_uv_per_uc * scored.charge_uC
            scored['error_uv'] = scored.predicted_uv - scored.observed_uv
            predictions.append(scored)
    predictions = pd.concat(predictions, ignore_index=True)
    coefficients = pd.concat(coefficients, ignore_index=True)
    predictions['squared_error'] = predictions.error_uv ** 2
    cells = (predictions.groupby(['subject', 'waveform', 'window', 'model'], as_index=False)
             .agg(observed_uv=('observed_uv', 'mean'), predicted_uv=('predicted_uv', 'mean'),
                  error_uv=('error_uv', 'mean'), trial_mse=('squared_error', 'mean'), n=('csv_row', 'size')))
    cells['mean_response_absolute_error'] = cells.error_uv.abs()
    participant = (cells.groupby(['subject', 'window', 'model'], as_index=False)
                   .agg(mae_uv=('mean_response_absolute_error', 'mean'), trial_mse=('trial_mse', 'mean')))
    participant['trial_rmse_uv'] = np.sqrt(participant.trial_mse)
    slopes = (coefficients.loc[coefficients.fit.eq('all_tiers') & coefficients.model.eq('shared')]
              .drop_duplicates(['subject', 'window']).pivot(index='subject', columns='window', values='beta_uv_per_uc'))
    summary = dict(participants=int(usable.subject.nunique()), cells=int(counts.included.sum()), trials=len(usable),
                   upper_trials=int(usable.tier.eq(2).sum()), unreported_sensation_trials=int((~usable.response_valid).sum()),
                   windows={}, shared_slope_post_minus_pre=interval(slopes.post_uv - slopes.pre_uv))
    gain = {}
    for outcome in WINDOWS:
        scores = participant.loc[participant.window.eq(outcome)].pivot(index='subject', columns='model', values='mae_uv')
        gain[outcome] = scores.constant - scores.shared
        summary['windows'][outcome] = dict(
            mae_uv={model: interval(scores[model]) for model in MODELS},
            shared_gain_vs_constant_uv=interval(gain[outcome]),
            separate_gain_vs_constant_uv=interval(scores.constant - scores.separate),
            shared_gain_vs_separate_uv=interval(scores.separate - scores.shared),
            all_tier_shared_slope_uv_per_uc=interval(slopes[outcome]),
        )
    summary['shared_prediction_gain_post_minus_pre_uv'] = interval(gain['post_uv'] - gain['pre_uv'])
    return summary, counts, coefficients, predictions, cells, participant


def draw_results(summary, participant, coefficients, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout='constrained')
    x = np.arange(len(MODELS))
    for window, shift, color, label in [('post_uv', -.14, '#276c96', 'Post-stimulus'),
                                        ('pre_uv', .14, '#c26a42', 'Pre-stimulus')]:
        stats = [summary['windows'][window]['mae_uv'][model] for model in MODELS]
        points = np.array([s['mean'] for s in stats])
        bounds = np.array([s['ci95'] for s in stats]).T
        axes[0].errorbar(x + shift, points, yerr=np.array([points - bounds[0], bounds[1] - points]),
                         fmt='o', capsize=4, color=color, label=label)
        subset = participant.loc[participant.window.eq(window)]
        for _, group in subset.groupby('subject'):
            axes[0].scatter(x + shift, group.set_index('model').loc[list(MODELS), 'mae_uv'], s=10, color=color, alpha=.22)
    axes[0].set(xticks=x, xticklabels=['Constant', 'Shared slope', 'Separate slopes'],
                ylabel='Mean absolute error (µV)', title='Prediction at the held-out upper tier')
    axes[0].legend(frameon=False)
    all_tiers = coefficients.loc[coefficients.fit.eq('all_tiers') & coefficients.model.eq('separate')]
    for window, shift, color in [('post_uv', -.14, '#276c96'), ('pre_uv', .14, '#c26a42')]:
        summaries = [interval(all_tiers.loc[all_tiers.window.eq(window) & all_tiers.waveform.eq(wave), 'beta_uv_per_uc'])
                     for wave in WAVES]
        means = np.array([s['mean'] for s in summaries])
        bounds = np.array([s['ci95'] for s in summaries]).T
        axes[1].errorbar(np.arange(len(WAVES)) + shift, means, yerr=np.array([means - bounds[0], bounds[1] - means]),
                         fmt='o', capsize=4, color=color)
    axes[1].axhline(0, color='#777777', lw=1)
    axes[1].set(xticks=np.arange(len(WAVES)), xticklabels=WAVES,
                ylabel='Slope (µV/µC)', title='Waveform slopes fitted across all tiers')
    fig.savefig(path, dpi=175)
    plt.close(fig)


def run(extraction, output):
    source = extraction / 'aligned_features.csv'
    data = pd.read_csv(source)
    output.mkdir(parents=True, exist_ok=False)
    summary, counts, coefficients, predictions, cells, participant = analyze(data)
    sensitivity, *_ = analyze(data, responded_only=True)
    summary['reported_sensation_sensitivity'] = sensitivity
    summary['settings'] = dict(experiment_id='E017', feature_cache=str(source), cache_sha256=digest(source),
                               extraction_audit_sha256=digest(extraction / 'extraction_audit.json'),
                               min_trials_per_tier=MIN_TIER_TRIALS, bootstrap_draws=BOOTSTRAPS, seed=SEED,
                               primary_error='absolute error of held-out tier mean, equal waveform and participant weights')
    for name, frame in [('cell_support', counts), ('coefficients', coefficients), ('upper_tier_predictions', predictions),
                        ('upper_tier_cell_scores', cells), ('participant_scores', participant)]:
        frame.to_csv(output / f'{name}.csv', index=False)
    draw_results(summary, participant, coefficients, output / 'charge_relation.png')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'reported_sensation_sensitivity'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extraction', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run(args.extraction, args.output)
