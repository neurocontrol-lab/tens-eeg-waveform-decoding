"""Predict nominal cathodic charge from Task 3 EEG with grouped ridge regression."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.response_characterization.e017_eeg_charge_relation.__main__ import interval
from experiments.response_characterization.e017_eeg_charge_relation.extract import digest
from experiments.response_characterization.e017_eeg_charge_relation.full_montage import PERIODS
from experiments.response_characterization.e019_task4_charge_prediction.__main__ import ALPHAS, ridge, select_alpha
from experiments.waveform_decoding.e001_voltage_baseline.features import CHANNELS


VALIDATIONS = {'within_waveform': 'fold', 'held_out_waveform': 'waveform'}


def load_inputs(montage):
    fit = montage / 'fit'
    archive = montage / 'extraction/montage_features.npz'
    assignments = fit / 'fold_assignments.csv'
    hashes = {path.name: digest(path) for path in (archive, assignments)}
    source = json.loads((fit / 'summary.json').read_text(encoding='utf-8'))
    if hashes[archive.name] != source['montage_features_sha256']:
        raise ValueError('Montage cache differs from the fitted Task 3 run')
    events = pd.read_csv(assignments)
    eligible = events.fold.ge(0).to_numpy()
    with np.load(archive) as data:
        np.testing.assert_array_equal(events.csv_row, data['csv_rows'])
        if 'subtract_trial_baseline' in data and not bool(data['subtract_trial_baseline']):
            raise ValueError('Expected baseline-corrected montage features')
        picks = [data['channels'].tolist().index(channel) for channel in CHANNELS]
        x = data['features'][eligible][..., picks]
    events = events.loc[eligible].reset_index(drop=True)
    if not events.csv_row.is_unique or not events.montage_qc_pass.all() or not np.isfinite(x).all():
        raise ValueError('Duplicate trials or failed EEG quality checks in the analysis set')
    if x.shape != (len(events), len(PERIODS), 9, len(CHANNELS)):
        raise ValueError(f'Unexpected montage feature shape {x.shape}')
    if events.sort_values(['recording', 'sample']).groupby('recording').timestamp_s.diff().le(1.25).any():
        raise ValueError('Epoch support overlaps within a recording')
    return events, x.reshape(len(events), len(PERIODS), -1), hashes


def leave_group_out(indices, labels):
    for label in np.unique(labels[indices]):
        test = labels[indices] == label
        yield indices[~test], indices[test]


def predict(events, features, validation):
    table = events[['subject', 'waveform', 'csv_row', 'tier', 'fold', 'charge_uC']].copy()
    q = events.charge_uC.to_numpy()
    if not np.isfinite(q).all():
        raise ValueError('Nominal charges must be finite')
    groups = ['subject', 'waveform'] if validation == 'within_waveform' else ['subject']
    labels = events[VALIDATIONS[validation]].to_numpy()
    seen = np.zeros(len(table), dtype=int)
    for _, group in events.groupby(groups):
        for train, test in leave_group_out(group.index.to_numpy(), labels):
            if validation == 'within_waveform' and set(events.tier.iloc[train]) != {0, 1, 2}:
                raise ValueError('Within-waveform training fold lacks an amplitude tier')
            inner = list(leave_group_out(train, labels))
            if len(inner) < 2:
                raise ValueError('Insufficient inner groups for ridge selection')
            seen[test] += 1
            table.loc[test, 'training_mean_uC'] = q[train].mean()
            table.loc[test, 'charge_seen_in_training'] = np.isin(q[test].round(12), q[train].round(12))
            table.loc[test, 'outside_training_range'] = (q[test] < q[train].min() - 1e-12) | (q[test] > q[train].max() + 1e-12)
            for p, period in enumerate(PERIODS):
                x = features[:, p]
                alpha = select_alpha(x, q, inner)
                table.loc[test, f'{period}_alpha'] = alpha
                table.loc[test, f'{period}_predicted_uC'] = ridge(alpha).fit(x[train], q[train]).predict(x[test])
    if not np.all(seen == 1):
        raise ValueError('Each trial must receive exactly one held-out prediction')
    return table.assign(validation=validation)


def summarize(predictions):
    errors = predictions[['validation', 'subject', 'waveform']].copy()
    for name in (*PERIODS, 'baseline'):
        column = 'training_mean_uC' if name == 'baseline' else f'{name}_predicted_uC'
        residual = predictions[column] - predictions.charge_uC
        errors[f'{name}_mae_uC'] = residual.abs()
        errors[f'{name}_mse_uC2'] = residual ** 2
    cells = errors.groupby(['validation', 'subject', 'waveform']).mean()
    participants = cells.groupby(['validation', 'subject']).mean().reset_index()
    for period in PERIODS:
        participants[f'{period}_r2_vs_training_mean'] = 1 - participants[f'{period}_mse_uC2'] / participants.baseline_mse_uC2
    summary = {}
    for validation, group in participants.groupby('validation'):
        stats = {key: interval(group[key]) for key in group if key.endswith(('mae_uC', 'r2_vs_training_mean'))}
        stats['post_minus_pre_mae_uC'] = interval(group.post_mae_uC - group.pre_mae_uC)
        data = predictions.loc[predictions.validation.eq(validation)]
        stats['charge_seen_in_training'] = int(data.charge_seen_in_training.sum())
        stats['outside_training_range'] = int(data.outside_training_range.sum())
        summary[validation] = stats
    return participants, cells.reset_index(), summary


def run(montage, output):
    if output.exists():
        raise FileExistsError(output)
    events, features, hashes = load_inputs(montage)
    with threadpool_limits(limits=1):
        predictions = pd.concat([predict(events, features, validation) for validation in VALIDATIONS], ignore_index=True)
    participants, cells, summary = summarize(predictions)
    output.mkdir(parents=True, exist_ok=False)
    predictions.to_csv(output / 'predictions.csv', index=False)
    participants.to_csv(output / 'participant_scores.csv', index=False)
    cells.to_csv(output / 'waveform_scores.csv', index=False)
    result = dict(trials=len(events), participants=int(events.subject.nunique()),
                  waveform_cells=events.groupby(['subject', 'waveform']).ngroups, alphas=ALPHAS,
                  montage=str(montage), inputs_sha256=hashes, validation=summary)
    (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--montage', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run(args.montage, args.output)
