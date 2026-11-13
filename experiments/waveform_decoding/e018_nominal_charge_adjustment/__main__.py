"""Compare label-informed charge adjustments on the original Task 4 folds."""

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.response_characterization.e017_eeg_charge_relation.__main__ import interval
from experiments.response_characterization.e017_eeg_charge_relation.extract import AREA_FACTORS, PHASE_MS, digest
from experiments.response_characterization.e017_eeg_charge_relation.full_montage import charge_coefficients
from experiments.waveform_decoding.e001_voltage_baseline.evaluate import make_splits, predict_oof, scores
from experiments.waveform_decoding.e001_voltage_baseline.events import CODES, WAVEFORMS
from experiments.waveform_decoding.e001_voltage_baseline.features import CHANNELS, WINDOWS
from .report import render


def amplitude_lookup(master, subjects):
    data = pd.read_csv(master)
    data.columns = data.columns.str.strip().str.lower()
    t4 = data.loc[data.task_id.eq(4) & data.subject_id.isin(subjects)].copy()
    t4['waveform'] = t4['t4 waveform 2'].add(238).map(WAVEFORMS)
    if t4.waveform.isna().any() or not np.isfinite(t4.amplitude).all() or (t4.amplitude <= 0).any():
        raise ValueError('Invalid Task 4 amplitude or second waveform')
    groups = t4.groupby(['subject_id', 'waveform']).amplitude
    if len(groups) != len(subjects) * len(CODES) or not groups.nunique().eq(1).all():
        raise ValueError('Expected one recorded amplitude for each participant and second waveform')
    return groups.first()


def calibrate(cache, fit):
    events = pd.read_csv(fit / 'fold_assignments.csv')
    with np.load(cache / 'montage_features.npz') as data, np.load(fit / 'shared_slopes.npz') as saved:
        np.testing.assert_array_equal(events.csv_row, data['csv_rows'])
        np.testing.assert_array_equal(saved['channels'], data['channels'])
        np.testing.assert_array_equal(saved['windows'], ['post', 'pre'])
        if digest(cache / 'montage_features.npz') != json.loads((fit / 'summary.json').read_text())['montage_features_sha256']:
            raise ValueError('Task 3 feature cache differs from the calibration run')
        picks = [list(data['channels']).index(channel) for channel in CHANNELS]
        x = data['features'][:, 0, :, :]
        shared, separate, charge_ranges = {}, {}, {}
        for subject, group in events.loc[events.fold.ge(0)].groupby('subject'):
            _, pooled, individual = charge_coefficients(group, x)
            saved_index, = np.flatnonzero(saved['subjects'] == subject)
            np.testing.assert_allclose(pooled, saved['slopes'][saved_index, 0], atol=1e-10, rtol=1e-12)
            shared[subject] = pooled[:, picks].ravel()
            separate[subject] = {wave: beta[:, picks].ravel() for wave, beta in individual.items()}
            charge_ranges[subject] = group.groupby('waveform').charge_uC.agg(['min', 'max'])
    return shared, separate, charge_ranges


def centroid_oof(x, y, splits):
    """Score fixed class templates even when within-class covariance is zero."""
    predicted = np.empty_like(y)
    for train, test in splits:
        templates = []
        for code in CODES:
            rows = x[train[y[train] == code]]
            if not np.all(rows == rows[0]):
                raise ValueError('Expected a fixed control template per waveform')
            templates.append(rows[0])
        distances = ((x[test, None] - np.asarray(templates)) ** 2).sum(axis=-1)
        predicted[test] = np.asarray(CODES)[distances.argmin(axis=1)]
    return predicted


def run(baseline, cache, fit, output, web_report):
    config = json.loads((baseline / 'run_config.json').read_text())
    if config['channels'] != list(CHANNELS) or any(config['windows'][k] != list(WINDOWS[k]) for k in ('post', 'pre')):
        raise ValueError('Task 4 baseline features differ from Task 3 calibration geometry')
    subjects = sorted(int(s) for s in config['subjects'].split(','))
    master = Path(config['master_csv'])
    amplitudes = amplitude_lookup(master, subjects)
    shared, separate, ranges = calibrate(cache, fit)
    results, predictions, overlap = [], [], []
    hashes = {'master_csv': digest(master), 'task3_slopes': digest(fit / 'shared_slopes.npz'),
              'task3_features': digest(cache / 'montage_features.npz'),
              'task3_folds': digest(fit / 'fold_assignments.csv')}
    for subject in subjects:
        directory = baseline / f'sub-{subject}'
        for file in ('events.csv', 'features.npz', 'predictions.csv'):
            hashes[f'sub-{subject}/{file}'] = digest(directory / file)
        meta = pd.read_csv(directory / 'events.csv').query('analysis_eligible').reset_index(drop=True)
        expected = pd.read_csv(directory / 'predictions.csv')
        np.testing.assert_array_equal(meta[['event_index', 'trigger', 'position']], expected[['event_index', 'trigger', 'position']])
        if not meta.waveform.eq(meta.trigger.map(WAVEFORMS)).all():
            raise ValueError(f'Trigger and waveform mismatch for participant {subject}')
        q_by_wave = {wave: round(amplitudes.loc[subject, wave] * PHASE_MS * AREA_FACTORS[wave], 12)
                     for wave in WAVEFORMS.values()}
        for a, b in combinations(q_by_wave, 2):
            if q_by_wave[a] == q_by_wave[b]:
                overlap.append(dict(subject=subject, waveform_a=a, waveform_b=b, charge_uC=q_by_wave[a]))
        q = meta.waveform.map(q_by_wave).to_numpy()
        correction = -q[:, None] * shared[subject]
        with np.load(directory / 'features.npz') as data:
            post, pre = data['post'], data['pre']
        features = dict(post=post, pre=pre, post_pooled=post + correction, pre_pooled=pre + correction,
                        nominal_charge=q[:, None], pooled_correction=correction)
        if set(separate[subject]) == set(WAVEFORMS.values()):
            specific = -q[:, None] * np.stack([separate[subject][wave] for wave in meta.waveform])
            features.update(post_specific=post + specific, pre_specific=pre + specific, specific_correction=specific)
        y = meta.trigger.to_numpy(int)
        splits = make_splits(meta)
        for fold, (_, test) in enumerate(splits):
            np.testing.assert_array_equal(expected.fold.iloc[test], np.full(len(test), fold))
        table = meta[['subject', 'event_index', 'pair_id', 'group', 'position', 'trigger']].copy()
        table['fold'] = expected.fold
        table['nominal_charge_uC'] = q
        calibration_range = ranges[subject].reindex(meta.waveform)
        table['task3_charge_range'] = np.where(calibration_range['min'].isna(), 'missing', np.where(
            (q < calibration_range['min'].to_numpy() - 1e-10) | (q > calibration_range['max'].to_numpy() + 1e-10),
            'outside', 'inside'))
        fitted = {}
        with threadpool_limits(limits=1):
            for name, x in features.items():
                fitted[name], _ = predict_oof(x, y, splits)
                if name.endswith('correction') or name == 'nominal_charge':
                    fitted[f'{name}_centroid'] = centroid_oof(x, y, splits)
        for name, predicted in fitted.items():
            if name in ('post', 'pre'):
                np.testing.assert_array_equal(predicted, expected[f'{name}_predicted'])
            table[name] = predicted
            for pos in (0, 1, 2):
                use = np.ones(len(y), dtype=bool) if pos == 0 else meta.position.to_numpy() == pos
                result = scores(y[use], predicted[use])
                if result['n_classes'] != len(CODES):
                    raise ValueError(f'Participant {subject}, position {pos} lacks a waveform')
                results.append(dict(subject=subject, position=pos, model=name, **result))
        predictions.append(table)

    output.mkdir(parents=True, exist_ok=False)
    predictions = pd.concat(predictions, ignore_index=True)
    predictions.to_csv(output / 'predictions.csv', index=False)
    complete = sorted({r['subject'] for r in results if r['model'] == 'post_specific'})
    stats = pd.DataFrame(results)
    changes = []
    for label, subset in [('all', stats), ('complete', stats.loc[stats.subject.isin(complete)])]:
        for pos, group in subset.groupby('position'):
            wide = group.pivot(index='subject', columns='model', values='balanced_accuracy')
            for variant in ('pooled', 'specific'):
                if variant == 'specific' and label == 'all' and len(complete) != len(subjects):
                    continue
                for window in ('post', 'pre'):
                    adjusted = f'{window}_{variant}'
                    if adjusted in wide:
                        changes.append(dict(cohort=label, position=int(pos), model=adjusted,
                                            **interval(wide[adjusted] - wide[window])))
    summary = dict(subjects=subjects, complete_subjects=complete, trials=len(predictions),
                   complete_trials=int(predictions.subject.isin(complete).sum()),
                   charge_ranges=predictions.task3_charge_range.value_counts().to_dict(),
                   calibration_missing={s: sorted(set(WAVEFORMS.values()) - set(separate[s])) for s in subjects if s not in complete},
                   overlap=overlap, changes=changes, scores=results,
                   inputs_sha256=hashes)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    page = render(summary, list(WAVEFORMS.values()))
    (output / 'index.html').write_text(page, encoding='utf-8', newline='\n')
    if web_report is not None:
        web_report.parent.mkdir(parents=True, exist_ok=True)
        web_report.write_text(page, encoding='utf-8', newline='\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=Path('output/task4_baseline'))
    parser.add_argument('--task3-cache', type=Path, default=Path('output/e017_task3_charge_full_montage_20261109/extraction'))
    parser.add_argument('--task3-fit', type=Path, default=Path('output/e017_task3_charge_full_montage_20261109/fit'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--web-report', type=Path)
    args = parser.parse_args()
    run(args.baseline, args.task3_cache, args.task3_fit, args.output, args.web_report)
