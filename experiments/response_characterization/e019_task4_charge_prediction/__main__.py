"""Evaluate EEG-only ridge regression for nominal Task 4 cathodic charge."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from experiments.response_characterization.e017_eeg_charge_relation.__main__ import interval
from experiments.response_characterization.e017_eeg_charge_relation.extract import AREA_FACTORS, PHASE_MS, digest
from experiments.waveform_decoding.e001_voltage_baseline.evaluate import make_splits
from experiments.waveform_decoding.e001_voltage_baseline.events import CODES, WAVEFORMS
from experiments.waveform_decoding.e001_voltage_baseline.features import CHANNELS, WINDOWS
from experiments.waveform_decoding.e018_nominal_charge_adjustment.__main__ import amplitude_lookup


ALPHAS = (1., 10., 100., 1000., 10000.)
PERIODS = ('pre', 'post')


def purge(meta, train, held_out):
    """Exclude whole training pairs whose feature support could touch held-out EEG."""
    time = meta.elapsed_s.to_numpy()
    close = (np.abs(time[train, None] - time[held_out]) <= 1.25).any(axis=1)
    pair = meta.pair_id.to_numpy()
    return train[~np.isin(pair[train], pair[train[close]])]


def waveform_splits(meta):
    """Train on comparisons without the test waveform and score its trials once."""
    for code in CODES:
        test = np.flatnonzero(meta.trigger.eq(code))
        excluded = meta.group.isin(meta.group.iloc[test]).to_numpy()
        train = purge(meta, np.flatnonzero(~excluded), np.flatnonzero(excluded))
        if not len(test) or set(meta.trigger.iloc[train]) != set(CODES) - {code}:
            raise ValueError(f'Insufficient training or test support for waveform {code}')
        yield train, test


def ridge(alpha):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def select_alpha(x, q, splits):
    mse = []
    for alpha in ALPHAS:
        errors = [ridge(alpha).fit(x[train], q[train]).predict(x[valid]) - q[valid] for train, valid in splits]
        mse.append(np.mean(np.concatenate(errors) ** 2))
    return ALPHAS[int(np.argmin(mse))]


def choose_alpha(meta, x, q, outer_train):
    inner = []
    for fit, valid in GroupKFold(n_splits=3).split(outer_train, groups=meta.group.iloc[outer_train]):
        held_out = outer_train[valid]
        train = purge(meta, outer_train[fit], held_out)
        if not len(train):
            raise ValueError('Inner fold has no training trials after temporal purging')
        inner.append((train, held_out))
    return select_alpha(x, q, inner)


def predict(meta, features, q, splits):
    table = meta[['subject', 'event_index', 'group', 'position', 'trigger', 'waveform']].copy()
    table['nominal_charge_uC'] = q
    table['fold'] = -1
    seen = np.zeros(len(table), dtype=int)
    for fold, (train, test) in enumerate(splits):
        seen[test] += 1
        table.loc[test, 'fold'] = fold
        table.loc[test, 'training_mean_uC'] = q[train].mean()
        table.loc[test, 'outside_training_range'] = (q[test] < q[train].min()) | (q[test] > q[train].max())
        table.loc[test, 'charge_seen_in_training'] = np.isin(q[test], q[train])
        for period in PERIODS:
            x = features[period]
            alpha = choose_alpha(meta, x, q, train)
            table.loc[test, f'{period}_alpha'] = alpha
            table.loc[test, f'{period}_predicted_uC'] = ridge(alpha).fit(x[train], q[train]).predict(x[test])
    if not np.all(seen == 1):
        raise ValueError('Each trial must receive one held-out prediction')
    return table


def metrics(group, period):
    y = group.nominal_charge_uC.to_numpy()
    err = group[f'{period}_predicted_uC'].to_numpy() - y
    base = group.training_mean_uC.to_numpy() - y
    mse = np.mean(err ** 2)
    baseline_mse = np.mean(base ** 2)
    return dict(mae_uC=float(np.abs(err).mean()), baseline_mae_uC=float(np.abs(base).mean()),
                r2=float(1 - mse / y.var()) if y.var() > 0 else None,
                r2_vs_training_mean=float(1 - mse / baseline_mse) if baseline_mse > 0 else None)


def run(baseline, output):
    config = json.loads((baseline / 'run_config.json').read_text())
    if config['channels'] != list(CHANNELS) or any(config['windows'][p] != list(WINDOWS[p]) for p in PERIODS):
        raise ValueError('Expected the original WD-001 voltage features')
    subjects = sorted(int(s) for s in config['subjects'].split(','))
    master = Path(config['master_csv'])
    amplitude = amplitude_lookup(master, subjects)
    hashes = {'master_csv': digest(master), 'run_config.json': digest(baseline / 'run_config.json')}
    tables = []
    with threadpool_limits(limits=1):
        for subject in subjects:
            folder = baseline / f'sub-{subject}'
            for name in ('events.csv', 'features.npz', 'predictions.csv'):
                hashes[f'sub-{subject}/{name}'] = digest(folder / name)
            meta = pd.read_csv(folder / 'events.csv').query('analysis_eligible').reset_index(drop=True)
            expected = pd.read_csv(folder / 'predictions.csv')
            keys = ['subject', 'event_index', 'position', 'trigger']
            pd.testing.assert_frame_equal(meta[keys], expected[keys])
            if not meta.waveform.eq(meta.trigger.map(WAVEFORMS)).all():
                raise ValueError(f'Waveform mismatch for participant {subject}')
            charges = {wave: round(amplitude.loc[subject, wave] * PHASE_MS * factor, 12)
                       for wave, factor in AREA_FACTORS.items()}
            q = meta.waveform.map(charges).to_numpy()
            with np.load(folder / 'features.npz') as data:
                features = {period: data[period] for period in PERIODS}
            shape = (len(meta), 9 * len(CHANNELS))
            if not all(x.shape == shape and np.isfinite(x).all() for x in features.values()):
                raise ValueError(f'Invalid feature matrix for participant {subject}')
            comparison = predict(meta, features, q, make_splits(meta))
            np.testing.assert_array_equal(comparison.fold, expected.fold)
            tables.append(comparison.assign(validation='comparison'))
            tables.append(predict(meta, features, q, waveform_splits(meta)).assign(validation='waveform'))

    predictions = pd.concat(tables, ignore_index=True)
    rows = [dict(validation=validation, subject=int(subject), period=period, trials=len(group), **metrics(group, period))
            for (validation, subject), group in predictions.groupby(['validation', 'subject']) for period in PERIODS]
    scores = pd.DataFrame(rows)
    summary = dict(participants=len(subjects), trials=int(predictions.validation.value_counts()['comparison']), alphas=ALPHAS,
                   inputs_sha256=hashes, validation={})
    for validation, group in predictions.groupby('validation'):
        result = {'trials': len(group), 'outside_training_range': int(group.outside_training_range.sum()),
                  'charge_seen_in_training': int(group.charge_seen_in_training.sum())}
        coverage = np.select([group.charge_seen_in_training.to_numpy(bool), group.outside_training_range.to_numpy(bool)],
                             ['seen', 'outside_training_range'], default='unseen_inside_training_range')
        result['charge_coverage'] = {}
        for label, subset in group.groupby(coverage):
            result['charge_coverage'][label] = dict(
                trials=len(subset), participants=int(subset.subject.nunique()),
                mae_uC={period: interval([metrics(g, period)['mae_uC'] for _, g in subset.groupby('subject')])
                        for period in PERIODS},
                baseline_mae_uC=interval([metrics(g, 'post')['baseline_mae_uC'] for _, g in subset.groupby('subject')]),
            )
        per_subject = scores.loc[scores.validation.eq(validation)]
        for period in PERIODS:
            selected = per_subject.loc[per_subject.period.eq(period)]
            result[period] = {key: interval(selected[key].dropna()) for key in
                              ('mae_uC', 'baseline_mae_uC', 'r2', 'r2_vs_training_mean')}
        paired = per_subject.pivot(index='subject', columns='period', values='mae_uC')
        result['post_minus_pre_mae_uC'] = interval(paired.post - paired.pre)
        summary['validation'][validation] = result
    output.mkdir(parents=True, exist_ok=False)
    predictions.to_csv(output / 'predictions.csv', index=False)
    scores.to_csv(output / 'participant_scores.csv', index=False)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=Path('output/task4_baseline'))
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run(args.baseline, args.output)
