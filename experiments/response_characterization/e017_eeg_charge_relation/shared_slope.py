"""Evaluate Task 3 dose slopes by holding out one entire waveform at a time."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .__main__ import interval
from .extract import digest


DOSES = ('charge_uC', 'amplitude_mA')
MODELS = ('zero_change', 'nominal_charge', 'peak_current')


def block_statistics(trials, observed):
    """Remove the voltage level within each chronological chunk before fitting slopes."""
    numerators, denominators, contrasts, steps = [], [], [], []
    for _, chunk in trials.groupby('fold'):
        y = observed[chunk.index]
        dose = chunk[list(DOSES)].to_numpy()
        centered = dose - dose.mean(axis=0)
        numerators.append(np.tensordot(centered.T, y - y.mean(axis=0), axes=1))
        denominators.append((centered ** 2).sum(axis=0))
        low, high = chunk.tier.eq(0).to_numpy(), chunk.tier.eq(2).to_numpy()
        if low.any() and high.any():
            contrasts.append(y[high].mean(axis=0) - y[low].mean(axis=0))
            steps.append(dose[high].mean(axis=0) - dose[low].mean(axis=0))
    return sum(numerators), sum(denominators), contrasts, steps


def predict_contrasts(events, observed):
    cells, actual, predicted, slopes = [], [], [], []
    for subject, trials in events.groupby('subject'):
        stats = {waveform: block_statistics(group, observed) for waveform, group in trials.groupby('waveform')}
        for waveform, (_, _, contrasts, steps) in stats.items():
            if not contrasts:
                continue
            training = [(num, den) for other, (num, den, _, _) in stats.items() if other != waveform and den.all()]
            if not training:
                continue
            num, den = zip(*training)
            shape = (len(DOSES),) + (1,) * (observed.ndim - 1)
            beta = sum(num) / sum(den).reshape(shape)
            contrast = np.mean(contrasts, axis=0)
            dose_step = np.mean(steps, axis=0)
            cells.append(dict(subject=subject, held_out_waveform=waveform, training_waveforms=len(training),
                              contrast_chunks=len(contrasts), delta_charge_uC=dose_step[0],
                              delta_amplitude_mA=dose_step[1]))
            actual.append(contrast)
            predicted.append(np.concatenate((np.zeros_like(contrast)[None], beta * dose_step.reshape(shape))))
            slopes.append(beta)
    return pd.DataFrame(cells), np.asarray(actual), np.asarray(predicted), np.asarray(slopes)


def score(cells, actual, predicted, widths, periods):
    rows = []
    for subject, group in cells.groupby('subject'):
        errors = predicted[group.index] - actual[group.index, None]
        mae = np.abs(errors).mean(axis=(0, 4, 5))
        mse = (errors ** 2).mean(axis=(0, 4, 5))
        for w, width in enumerate(widths):
            for p, period in enumerate(periods):
                for m, model in enumerate(MODELS):
                    rows.append(dict(subject=subject, width_ms=width, window=period, model=model,
                                     mae_uv=mae[m, w, p], incremental_r2=1 - mse[m, w, p] / mse[0, w, p]))
    return pd.DataFrame(rows)


def analyze(forecast_fit, output):
    events = pd.read_csv(forecast_fit / 'fold_assignments.csv')
    target_path = forecast_fit / 'held_out_predictions.npz'
    with np.load(target_path) as data:
        np.testing.assert_array_equal(events.csv_row, data['csv_rows'])
        observed = data['observed']
        widths, periods = data['widths_ms'], data['windows']
        starts, channels = data['starts_ms'], data['channels']

    cells, actual, predicted, slopes = predict_contrasts(events, observed)
    if cells.empty or cells.subject.nunique() < 2:
        raise ValueError('Fewer than two participants have test contrasts and training waveforms')
    scores = score(cells, actual, predicted, widths, periods)
    summary = dict(participants=int(cells.subject.nunique()), scored_cells=len(cells), source_trials=len(events),
                   contrast_chunks=int(cells.contrast_chunks.sum()), electrodes=len(channels),
                   target_starts_ms=dict(zip(periods, starts.tolist())), target_archive_sha256=digest(target_path),
                   folds_sha256=digest(forecast_fit / 'fold_assignments.csv'), widths={})
    for width, subset in scores.groupby('width_ms'):
        result = {period: {model: {key: interval(group[key]) for key in ('mae_uv', 'incremental_r2')}
                           for model, group in by_period.groupby('model')}
                  for period, by_period in subset.groupby('window')}
        paired = subset.pivot(index='subject', columns=['window', 'model'], values='incremental_r2')
        result['post_minus_pre_incremental_r2'] = {
            model: interval(paired['post', model] - paired['pre', model]) for model in MODELS[1:]
        }
        result['post_charge_minus_current_incremental_r2'] = interval(
            paired['post', 'nominal_charge'] - paired['post', 'peak_current']
        )
        summary['widths'][str(width)] = result

    output.mkdir(parents=True, exist_ok=False)
    cells.to_csv(output / 'held_out_waveforms.csv', index=False)
    scores.to_csv(output / 'participant_scores.csv', index=False)
    np.savez_compressed(output / 'held_out_contrasts.npz', observed=actual, predictions=predicted, slopes=slopes,
                        subjects=cells.subject.to_numpy(int), waveforms=cells.held_out_waveform.to_numpy(str),
                        channels=channels, models=MODELS, slope_inputs=DOSES, windows=periods,
                        widths_ms=widths, starts_ms=starts)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--forecast-fit', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    analyze(args.forecast_fit, args.output)
