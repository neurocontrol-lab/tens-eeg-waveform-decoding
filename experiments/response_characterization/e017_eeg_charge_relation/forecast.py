"""Test whether nominal charge improves an out-of-sample Task 3 EEG forecast."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .__main__ import interval
from .extract import digest
from .full_montage import FOLDS, PERIODS, extract, fit_models


WIDTHS_MS = (100, 50)
PERIOD_FIRST_BIN_MS = (50, -500)
START_BINS = (3, 5, 7)
MODELS = ('history', 'history_charge', 'waveform_mean', 'waveform_charge')
PER_WAVEFORM_MODELS = ('separate_history', 'separate_history_charge', 'separate_waveform_charge')
COMPARATORS = {
    'history_charge': 'history',
    'waveform_charge': 'waveform_mean',
    'separate_history_charge': 'separate_history',
    'separate_waveform_charge': 'waveform_mean',
}


def forecast_arrays(bins, widths_ms=WIDTHS_MS, start_bins=START_BINS):
    """Construct non-overlapping histories and targets from 50 ms bin means."""
    if not start_bins or min(start_bins) < 3 or max(start_bins) + max(widths_ms) // 50 > bins.shape[2]:
        raise ValueError('History or target extends beyond the cached voltage bins')
    history = np.stack([bins[:, :, start - 3:start, :] for start in start_bins], axis=2)
    history = history.transpose(0, 1, 2, 4, 3)
    targets = np.stack([
        np.stack([bins[:, :, start:start + width // 50, :].mean(axis=2) for start in start_bins], axis=2)
        for width in widths_ms
    ], axis=1)
    return history, targets


def fit_forecasts(history, targets, q, train, test):
    """Fit target widths together, with one regression per period, time and electrode."""
    design = np.concatenate([np.ones_like(history[..., :1]), history], axis=-1)
    response = np.moveaxis(targets, 1, -1)
    predictions = np.empty((len(test), 2, *targets.shape[1:]))
    for index in np.ndindex(history.shape[1:-1]):
        local = design[(slice(None), *index)]
        with_charge = np.column_stack((local, q))
        y = response[(train, *index)]
        for model, features in enumerate((local, with_charge)):
            beta = np.linalg.lstsq(features[train], y, rcond=None)[0]
            predictions[(slice(None), model, slice(None), *index)] = features[test] @ beta
    return predictions


def trial_scores(observed, predicted, events, widths_ms, starts_ms, models, *, by_waveform=False):
    """Weight waveform–tier cells equally within each participant."""
    rows = []
    keys = ['subject', 'waveform'] if by_waveform else ['subject']
    for key, group in events.groupby(keys):
        indices = group.index.to_numpy()
        cell_sizes = group.groupby(['waveform', 'tier']).csv_row.transform('size').to_numpy()
        weights = 1. / cell_sizes
        y = observed[indices]
        errors = predicted[indices] - y[:, None]
        variance = np.average((y - np.average(y, axis=0, weights=weights)) ** 2, axis=0, weights=weights)
        mae = np.average(np.abs(errors), axis=0, weights=weights)
        mse = np.average(errors ** 2, axis=0, weights=weights)
        for w, width in enumerate(widths_ms):
            for p, period in enumerate(PERIODS):
                for start, time in [(None, 'all'), *enumerate(starts_ms[p])]:
                    selected = slice(None) if start is None else start
                    for m, model in enumerate(models):
                        error = mse[m, w, p, selected].mean()
                        base = COMPARATORS.get(model)
                        comparator = mse[models.index(base), w, p, selected].mean() if base else None
                        rows.append(dict(zip(keys, key), model=model, window=period, target_width_ms=width,
                                         target_start_ms=time, mae_uv=float(mae[m, w, p, selected].mean()),
                                         r2=float(1 - error / variance[w, p, selected].mean()),
                                         incremental_r2=None if comparator is None else float(1 - error / comparator)))
    return pd.DataFrame(rows)


def analyze(cache, fold_path, output, *, include_200ms=False, per_waveform=False):
    all_events = pd.read_csv(cache / 'montage_events.csv')
    folds = pd.read_csv(fold_path)
    columns = ['csv_row', 'subject', 'waveform', 'tier', 'recording', 'sample', 'charge_uC', 'montage_qc_pass']
    pd.testing.assert_frame_equal(all_events[columns], folds[columns])
    with np.load(cache / 'montage_features.npz') as data:
        if bool(data['subtract_trial_baseline']):
            raise ValueError('Forecast requires voltage before trial-baseline subtraction')
        np.testing.assert_array_equal(data['csv_rows'], all_events.csv_row)
        channels = data['channels']
        included = folds.fold.ge(0).to_numpy()
        bins = data['features'][included].astype(float)
    widths_ms = (*WIDTHS_MS, 200) if include_200ms else WIDTHS_MS
    # Keep identical target starts for each width and never truncate a target at the epoch end.
    start_bins = tuple(start for start in START_BINS if start + max(widths_ms) // 50 <= bins.shape[2])
    starts_ms = tuple(tuple(first + 50 * start for start in start_bins) for first in PERIOD_FIRST_BIN_MS)
    history, targets = forecast_arrays(bins, widths_ms, start_bins)

    events = folds.loc[included].reset_index(drop=True)
    if not events.csv_row.is_unique or not events.montage_qc_pass.all():
        raise ValueError('Fold assignments contain duplicate or ineligible trials')
    models = (*MODELS, *PER_WAVEFORM_MODELS) if per_waveform else MODELS
    predictions = np.full((len(events), len(models), *targets.shape[1:]), np.nan)
    slopes, subjects = [], []
    q = events.charge_uC.to_numpy()
    for subject, group in events.groupby('subject'):
        for fold in range(FOLDS):
            train = group.loc[group.fold.ne(fold)]
            test = group.loc[group.fold.eq(fold)]
            predictions[test.index, :2] = fit_forecasts(history, targets, q, train.index.to_numpy(),
                                                       test.index.to_numpy())
            calibration, _ = fit_models(train, targets, test)
            predictions[test.index, 2:4] = calibration[:, :2]
            if per_waveform:
                predictions[test.index, 6] = calibration[:, 2]
                for waveform, test_cell in test.groupby('waveform'):
                    train_cell = train.loc[train.waveform.eq(waveform)]
                    predictions[test_cell.index, 4:6] = fit_forecasts(
                        history, targets, q, train_cell.index.to_numpy(), test_cell.index.to_numpy(),
                    )
        _, beta = fit_models(group, targets, group)
        slopes.append(beta)
        subjects.append(subject)
    if not np.isfinite(predictions).all():
        raise ValueError('At least one eligible trial lacks finite out-of-fold predictions')

    scores = trial_scores(targets, predictions, events, widths_ms, starts_ms, models)
    overall = scores.loc[scores.target_start_ms.eq('all')]
    summary = dict(trials=len(events), participants=len(subjects), electrodes=len(channels),
                   folds=FOLDS, history_bins_ms=[[-150, -100], [-100, -50], [-50, 0]],
                   target_starts_ms=dict(zip(PERIODS, starts_ms)), subtract_trial_baseline=False,
                   feature_cache_sha256=digest(cache / 'montage_features.npz'), fold_assignments_sha256=digest(fold_path),
                   widths={})
    for width, by_width in overall.groupby('target_width_ms'):
        result = {}
        for period, by_period in by_width.groupby('window'):
            result[period] = {model: {key: interval(group[key].dropna()) for key in ('mae_uv', 'r2', 'incremental_r2')
                                      if group[key].notna().any()}
                              for model, group in by_period.groupby('model')}
        differences = {}
        for model in (name for name in COMPARATORS if name in models):
            paired = by_width.loc[by_width.model.eq(model)].pivot(index='subject', columns='window', values='incremental_r2')
            differences[model] = interval(paired.post - paired.pre)
        result['post_minus_pre_incremental_r2'] = differences
        summary['widths'][str(width)] = result

    output.mkdir(parents=True, exist_ok=False)
    scores.to_csv(output / 'participant_scores.csv', index=False)
    if per_waveform:
        by_wave = trial_scores(targets, predictions, events, widths_ms, starts_ms, models, by_waveform=True)
        by_wave.to_csv(output / 'participant_waveform_scores.csv', index=False)
    events.to_csv(output / 'fold_assignments.csv', index=False)
    np.savez_compressed(output / 'held_out_predictions.npz', observed=targets, predictions=predictions,
                        csv_rows=events.csv_row.to_numpy(), channels=channels, models=models, windows=PERIODS,
                        widths_ms=widths_ms, starts_ms=starts_ms)
    np.savez_compressed(output / 'descriptive_shared_charge_slopes.npz', slopes=np.asarray(slopes), subjects=subjects,
                        channels=channels, windows=PERIODS, widths_ms=widths_ms, starts_ms=starts_ms)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--extraction', type=Path, help='Original E017 trigger-alignment audit')
    source.add_argument('--cache', type=Path, help='Existing full-montage cache without trial-baseline subtraction')
    parser.add_argument('--fold-assignments', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--include-200ms', action='store_true', help='Compare 50, 100 and 200 ms at shared target starts')
    parser.add_argument('--per-waveform', action='store_true', help='Also fit models within each known waveform')
    args = parser.parse_args()
    cache = args.cache
    if args.extraction is not None:
        cache = args.output / 'extraction'
        cache.mkdir(parents=True, exist_ok=False)
        extract(args.extraction, cache, subtract_trial_baseline=False)
    analyze(cache, args.fold_assignments, args.output / 'fit', include_200ms=args.include_200ms,
            per_waveform=args.per_waveform)
