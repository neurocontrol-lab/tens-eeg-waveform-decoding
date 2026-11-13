"""Predict voltage at all three Task 3 doses using held-out trial chunks."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyxdf

from experiments.waveform_decoding.e001_voltage_baseline.features import WINDOWS
from .__main__ import MODELS, MIN_TIER_TRIALS, interval
from .extract import digest


PERIODS = ('post', 'pre')
FOLDS = 5
EPOCH = (-.75, .50)
PTP_LIMIT_UV = 500.


def read_features(path, events, source, *, subtract_trial_baseline=True):
    if digest(path) != source['sha256']:
        raise ValueError(f'XDF changed since the trigger join: {path}')
    fs = source['sampling_hz']
    first, stop = (round(t * fs) for t in EPOCH)
    starts = events['sample'].to_numpy(int) + first
    ends = starts + stop - first
    stamps = np.full((len(events), stop - first), np.nan)
    offset = 0
    epochs = None
    channels = None
    picks = None
    recorded_triggers = np.full(len(events), np.nan)

    def collect(values, timestamps, stream, _stream_id):
        nonlocal offset, epochs, channels, picks, factors, trigger
        if picks is None:
            desc = stream['info']['desc'][0]['channels'][0]['channel']
            labels = [item['label'][0] for item in desc]
            picks = [i for i, item in enumerate(desc)
                     if item.get('type') == ['EEG'] and labels[i] not in ('CREF', 'M1', 'M2')]
            channels = [labels[i] for i in picks]
            scale = {'uvolt': 1., 'uv': 1., 'microvolt': 1., 'volt': 1e6, 'v': 1e6}
            units = [desc[i]['unit'][0].lower().replace('µ', 'u').replace('μ', 'u') for i in picks]
            factors = np.array([scale[unit] for unit in units])
            trigger = labels.index('TRIGGERS')
            epochs = np.full((*stamps.shape, len(picks)), np.nan, dtype=np.float32)
        end = offset + len(values)
        left = np.searchsorted(ends, offset, side='right')
        right = np.searchsorted(starts, end, side='left')
        for row in range(left, right):
            a, b = max(starts[row], offset), min(ends[row], end)
            source_slice = slice(a - offset, b - offset)
            epoch_slice = slice(a - starts[row], b - starts[row])
            uv = values[source_slice, :][:, picks] * factors
            epochs[row, epoch_slice] = uv - uv.mean(axis=1, keepdims=True)
            stamps[row, epoch_slice] = timestamps[source_slice]
            onset = starts[row] - first
            if a <= onset < b:
                recorded_triggers[row] = values[onset - offset, trigger]
        offset = end
        # The XDF reader should retain no continuous samples after this callback.
        return values[:0, :0].copy(), timestamps[:0].copy(), stream

    factors = None
    trigger = None
    pyxdf.load_xdf(str(path), select_streams=[{'name': 'SAGA', 'type': 'EEG'}], on_chunk=collect,
                   synchronize_clocks=False, dejitter_timestamps=False)
    if offset != source['samples'] or not np.array_equal(recorded_triggers, events.raw_trigger):
        raise ValueError(f'Cached event sample or trigger differs in {path.name}')
    np.testing.assert_allclose(stamps[:, -first], events.timestamp_s, rtol=0., atol=1e-6)

    def segment(lo, hi):
        return epochs[:, round(lo * fs) - first:round(hi * fs) - first]

    finite = np.isfinite(epochs).all(axis=(1, 2))
    regular = np.isfinite(stamps).all(axis=1) & (np.abs(np.diff(stamps) - 1 / fs) <= .1 / fs).all(axis=1)
    ptp = np.maximum(np.ptp(segment(-.75, -.05), axis=1).max(axis=1),
                     np.ptp(segment(.05, .50), axis=1).max(axis=1))
    valid = finite & regular & (ptp <= PTP_LIMIT_UV)
    matrices = []
    for period in PERIODS:
        lo, hi, step, blo, bhi = WINDOWS[period]
        bins = [segment(lo + k * step, lo + (k + 1) * step).mean(axis=1)
                for k in range(round((hi - lo) / step))]
        matrix = np.stack(bins, axis=1)
        if subtract_trial_baseline:
            matrix -= segment(blo, bhi).mean(axis=1)[:, None, :]
        matrices.append(matrix)
    return np.stack(matrices, axis=1), channels, valid, ptp


def extract(extraction, output, *, subtract_trial_baseline=True):
    source = json.loads((extraction / 'extraction_audit.json').read_text())
    events = pd.read_csv(extraction / 'aligned_features.csv')
    timeline = pd.read_csv(extraction / 'xdf_events.csv').sort_values(['recording', 'sample'])
    if timeline.groupby('recording').timestamp_s.diff().le(EPOCH[1] - EPOCH[0]).any():
        raise ValueError('Neighboring trigger epochs overlap. Review temporal grouping.')
    recordings = {Path(r['recording']).name: r for r in source['recordings']}
    event_frames, features = [], []
    names = None
    for name, group in events.groupby('recording', sort=True):
        group = group.sort_values('sample').copy()
        data, channels, valid, ptp = read_features(
            Path(recordings[name]['recording']), group, recordings[name],
            subtract_trial_baseline=subtract_trial_baseline,
        )
        if names is not None and channels != names:
            raise ValueError(f'Scalp channel order differs in {name}')
        names = channels
        group['montage_qc_pass'] = valid & group.eeg_qc.isna().to_numpy()
        group['montage_ptp_uv'] = ptp
        event_frames.append(group)
        features.append(data)
        print(f'{name}: {group.montage_qc_pass.sum()}/{len(group)} full-montage trials pass QC', flush=True)
    events = pd.concat(event_frames, ignore_index=True)
    events.to_csv(output / 'montage_events.csv', index=False)
    np.savez_compressed(output / 'montage_features.npz', features=np.concatenate(features), channels=names,
                        csv_rows=events.csv_row.to_numpy(), subtract_trial_baseline=subtract_trial_baseline)


def charge_coefficients(train, x):
    """Fit featurewise models with waveform intercepts and absolute charge inputs."""
    offsets, numerators, denominators = {}, {}, {}
    for waveform, group in train.groupby('waveform'):
        q = group.charge_uC.to_numpy()
        y = x[group.index]
        centered = q - q.mean()
        offsets[waveform] = (q.mean(), y.mean(axis=0))
        numerators[waveform] = np.tensordot(centered, y, axes=1)
        denominators[waveform] = centered @ centered
    shared = sum(numerators.values()) / sum(denominators.values())
    separate = {waveform: numerators[waveform] / denominators[waveform] for waveform in offsets}
    return offsets, shared, separate


def fit_models(train, x, test):
    offsets, shared, separate = charge_coefficients(train, x)
    predictions = np.empty((len(test), len(MODELS), *x.shape[1:]))
    for waveform, positions in test.groupby('waveform').indices.items():
        q_mean, y_mean = offsets[waveform]
        q = test.iloc[positions].charge_uC.to_numpy().reshape((-1,) + (1,) * (x.ndim - 1))
        for m, beta in enumerate((0., shared, separate[waveform])):
            alpha = y_mean - beta * q_mean
            predictions[positions, m] = alpha + q * beta
    return predictions, shared


def analyze(cache, output):
    events = pd.read_csv(cache / 'montage_events.csv')
    with np.load(cache / 'montage_features.npz') as z:
        np.testing.assert_array_equal(z['csv_rows'], events.csv_row)
        x, channels = z['features'], z['channels']
    events['fold'] = -1
    for _, group in events.loc[events.montage_qc_pass].groupby(['subject', 'waveform']):
        chunks = np.array_split(group.sort_values(['recording', 'sample']).index.to_numpy(), FOLDS)
        support = [events.loc[group.index.difference(chunk)].tier.value_counts().reindex([0, 1, 2], fill_value=0).min()
                   for chunk in chunks]
        if all(len(chunk) for chunk in chunks) and min(support) >= MIN_TIER_TRIALS:
            for fold, chunk in enumerate(chunks):
                events.loc[chunk, 'fold'] = fold
    included = events.loc[events.fold.ge(0)]
    if included.subject.nunique() < 2:
        raise ValueError('Insufficient participants with all tiers in every training fold')
    predictions = np.full((len(events), len(MODELS), *x.shape[1:]), np.nan, dtype=np.float32)
    subject_slopes = []
    for _, group in included.groupby('subject'):
        for fold in range(FOLDS):
            test = group.loc[group.fold.eq(fold)]
            train = group.loc[group.fold.ne(fold)]
            predictions[test.index], _ = fit_models(train, x, test)
        _, shared = fit_models(group, x, group)
        subject_slopes.append(shared)
    assert np.isfinite(predictions[included.index]).all()
    scores = []
    for subject, group in included.groupby('subject'):
        tiers = [predictions[g.index] - x[g.index, None] for _, g in group.groupby(['waveform', 'tier'])]
        mae = np.mean([np.abs(err).mean(axis=(0, 3, 4)) for err in tiers], axis=0)
        mse = np.mean([(err ** 2).mean(axis=(0, 3, 4)) for err in tiers], axis=0)
        for m, model in enumerate(MODELS):
            for p, period in enumerate(PERIODS):
                scores.append(dict(subject=subject, model=model, window=period, trial_mae_uv=float(mae[m, p]),
                                   mse_uv2=float(mse[m, p]),
                                   incremental_r2=float(1 - mse[m, p] / mse[0, p])))
    scores = pd.DataFrame(scores)
    summary = dict(participants=int(included.subject.nunique()), cells=len(included.groupby(['subject', 'waveform'])),
                   trials=len(included), montage_qc_trials=int(events.montage_qc_pass.sum()), electrodes=len(channels),
                   bins_per_window=x.shape[2], folds=FOLDS, min_train_trials_per_tier=MIN_TIER_TRIALS,
                   montage_features_sha256=digest(cache / 'montage_features.npz'), windows={})
    for period, group in scores.groupby('window'):
        summary['windows'][period] = {model: {key: interval(g[key]) for key in
                                              ['trial_mae_uv', 'incremental_r2']}
                                       for model, g in group.groupby('model')}
    shared = scores.loc[scores.model.eq('shared')].pivot(index='subject', columns='window', values='incremental_r2')
    summary['post_minus_pre_shared_incremental_r2'] = interval(shared.post - shared.pre)
    output.mkdir(parents=True, exist_ok=False)
    scores.to_csv(output / 'participant_scores.csv', index=False)
    events.to_csv(output / 'fold_assignments.csv', index=False)
    np.savez_compressed(output / 'held_out_predictions.npz', predictions=predictions[included.index],
                        observed=x[included.index], csv_rows=included.csv_row, models=MODELS, windows=PERIODS,
                        channels=channels)
    slopes = np.asarray(subject_slopes)
    np.savez_compressed(output / 'shared_slopes.npz', slopes=slopes, subjects=np.sort(included.subject.unique()),
                        windows=PERIODS, channels=channels)
    means = slopes.mean(axis=0)
    panels = np.stack([means[0], means[1], means[0] - means[1]])
    limit = np.abs(panels).max()
    fig, axes = plt.subplots(1, 3, figsize=(12, 12), sharey=True, layout='constrained')
    for ax, matrix, title in zip(axes, panels, ['Post-stimulus', 'Pre-stimulus', 'Post minus pre']):
        im = ax.imshow(matrix.T, aspect='auto', cmap='RdBu_r', vmin=-limit, vmax=limit)
        ax.set(title=title, xlabel='Matched 50 ms bin', xticks=np.arange(x.shape[2]),
               xticklabels=np.arange(1, x.shape[2] + 1), yticks=np.arange(len(channels)), yticklabels=channels)
        ax.tick_params(axis='y', labelsize=6)
    fig.colorbar(im, ax=axes, label='Participant mean shared slope (µV/µC)', shrink=.75)
    fig.savefig(output / 'shared_charge_slopes.png', dpi=150)
    plt.close(fig)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extraction', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    cache = args.output / 'extraction'
    cache.mkdir(parents=True, exist_ok=False)
    extract(args.extraction, cache)
    analyze(cache, args.output / 'fit')
