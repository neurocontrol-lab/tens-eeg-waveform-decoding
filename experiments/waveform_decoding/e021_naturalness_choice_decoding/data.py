"""Extract fixed voltage features with strict reuse of the verified trial alignment."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.response_characterization.e017_eeg_charge_relation.extract import (
    AREA_FACTORS, PHASE_MS, digest,
)
from experiments.waveform_decoding.e001_voltage_baseline.events import (
    WAVEFORMS, align_events, load_recording, read_behavior, trigger_runs,
)
from experiments.waveform_decoding.e001_voltage_baseline.features import interval_samples
from experiments.waveform_decoding.e018_nominal_charge_adjustment.__main__ import amplitude_lookup


CHANNELS = ('C4', 'CP2', 'CP4', 'CP6', 'P2', 'P4')
WINDOWS = {'pre': (-.350, -.100), 'post': (.100, .350)}
BIN_SECONDS = .050
BASELINE = (-.550, -.350)


def voltage_means(selected, sample, window, timestamps, sfreq):
    """Time-major bin means in microvolts, with full support and timestamp checks."""
    lo, hi = window
    start, stop = interval_samples(sample, lo, hi, sfreq)
    if start < 0 or stop > selected.shape[1]:
        raise ValueError('Requested feature window crosses a recording boundary')
    if np.any(np.abs(np.diff(timestamps[start:stop]) - 1 / sfreq) > .1 / sfreq):
        raise ValueError('Timestamp discontinuity in a requested feature window')
    values = selected[:, start:stop]
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite EEG in a requested feature window')
    bins = []
    for index in range(int(round((hi - lo) / BIN_SECONDS))):
        a, b = interval_samples(sample, lo + index * BIN_SECONDS,
                                lo + (index + 1) * BIN_SECONDS, sfreq)
        bins.append(selected[:, a:b].mean(axis=1) * 1e6)
    return np.concatenate(bins)


def prepare_subject(subject, args, baseline_config):
    """Make one row per original pair before applying common EEG eligibility."""
    old = args.baseline / f'sub-{subject}'
    source = args.data_dir / f'Subject{subject}_data/EEGDATA_SUB{subject}-T4.xdf'
    visual_audit = args.visual / f'sub-{subject}/audit.json'
    hashes = {str(path): digest(path) for path in
              (source, old / 'events.csv', visual_audit, args.master_csv)}
    if hashes[str(source)] != json.loads(visual_audit.read_text())['source_sha256']:
        raise ValueError(f'Participant {subject}: raw XDF differs from the verified input')
    folder = args.cache / f'sub-{subject}'
    signature = {
        'inputs_sha256': hashes, 'channels': list(CHANNELS),
        'windows': {name: list(window) for name, window in WINDOWS.items()},
        'baseline': list(BASELINE),
        'bin_seconds': BIN_SECONDS,
        'implementation_sha256': digest(Path(__file__)),
    }
    if (folder / 'complete.json').exists():
        complete = json.loads((folder / 'complete.json').read_text())
        if complete['signature'] != signature:
            raise ValueError(f'Participant {subject}: feature-cache provenance differs')
        for name, expected in complete['outputs_sha256'].items():
            if digest(folder / name) != expected:
                raise ValueError(f'Participant {subject}: cached {name} changed')
        with np.load(folder / 'features.npz') as data:
            features = {name: data[name].copy() for name in data.files}
        return pd.read_csv(folder / 'pairs.csv'), features, complete

    print(f'Participant {subject}: extract minimally processed raw EEG', flush=True)
    raw, names, triggers, timestamps, sfreq = load_recording(source)
    if len(names) != 62 or not set(CHANNELS).issubset(names):
        raise ValueError('Expected the full 62-channel scalp montage and predefined electrodes')
    aligned = align_events(
        trigger_runs(triggers, sfreq, baseline_config['min_trigger_samples']),
        read_behavior(args.master_csv, subject), subject, timestamps,
    )
    original = pd.read_csv(old / 'events.csv')
    fields = ['subject', 'event_index', 'sample', 'trigger', 'pair_id', 'position',
              'csv_row', 'response_value', 'response_valid', 'block_index', 'comparison']
    pd.testing.assert_frame_equal(aligned[fields], original[fields], check_dtype=False)
    picks = [names.index(channel) for channel in CHANNELS]
    selected = raw[picks] - raw.mean(axis=0, keepdims=True)
    del raw, triggers
    rows, pre, first_post, difference = [], [], [], []
    for _, pair in original.groupby('pair_id', sort=False):
        pair = pair.sort_values('position')
        if len(pair) != 2 or pair.position.tolist() != [1, 2]:
            raise ValueError('Expected two ordered events per pair')
        first, second = pair.iloc[0], pair.iloc[1]
        if first.response_value != second.response_value:
            raise ValueError('The two events disagree about the behavioral choice')
        why = []
        if not pair.analysis_eligible.all():
            why.append('original_pair_qc')
        if np.isfinite(first.previous_gap_s) and first.previous_gap_s <= -BASELINE[0]:
            why.append('preceding_stimulus_overlaps_pre_window')
        if first.next_gap_s <= .350 or (
            np.isfinite(second.next_gap_s) and second.next_gap_s <= .350
        ):
            why.append('next_stimulus_overlaps_post_window')
        values = [np.full(30, np.nan) for _ in range(3)]
        if not why:
            values = [
                voltage_means(selected, int(first['sample']), WINDOWS['pre'], timestamps, sfreq),
                voltage_means(selected, int(first['sample']), WINDOWS['post'], timestamps, sfreq),
                voltage_means(selected, int(second['sample']), WINDOWS['post'], timestamps, sfreq),
            ]
            baseline = voltage_means(
                selected, int(first['sample']), BASELINE, timestamps, sfreq,
            ).reshape(-1, len(CHANNELS)).mean(axis=0)
            reference = np.tile(baseline, 5)
            values = [value - reference for value in values]
        pre.append(values[0])
        first_post.append(values[1])
        difference.append(values[2] - values[1])
        rows.append(dict(
            subject=subject, pair_id=first.pair_id, pair_index=int(first.pair_index),
            block_index=int(first.block_index), comparison=first.comparison,
            first_trigger=int(first.trigger), second_trigger=int(second.trigger),
            csv_row=int(first.csv_row), response_value=float(first.response_value),
            target_second=int(first.response_value == 2),
            eligible=not why, exclusion_reason=';'.join(why),
            preceding_gap_s=float(first.previous_gap_s),
            within_pair_gap_s=float(first.next_gap_s),
            following_gap_s=float(second.next_gap_s),
        ))
    table = pd.DataFrame(rows)
    grouped = table.groupby('block_index', sort=False)
    table['within_block_index'] = grouped.cumcount()
    table['terminal_vote'] = table.pair_index.eq(grouped.pair_index.transform('max'))
    table['ordered_pair'] = table.first_trigger.astype(str) + '-' + table.second_trigger.astype(str)
    features = dict(pre=np.array(pre), first_post=np.array(first_post), paired_post=np.array(difference))
    folder.mkdir(parents=True, exist_ok=True)
    table.to_csv(folder / 'pairs.csv', index=False)
    np.savez_compressed(folder / 'features.npz', **features)
    complete = dict(
        signature=signature,
        outputs_sha256={name: digest(folder / name) for name in ('pairs.csv', 'features.npz')},
        pairs=len(table), eligible=int(table.eligible.sum()),
        sfreq=float(sfreq), referenced_channels=len(names),
    )
    (folder / 'complete.json').write_text(json.dumps(complete, indent=2) + '\n')
    print(f'  {table.eligible.sum()}/{len(table)} eligible pairs', flush=True)
    return table, features, complete


def assemble(args):
    """Check data provenance, then combine participant caches in recording order."""
    config = json.loads((args.baseline / 'run_config.json').read_text())
    if digest(args.master_csv) != config['master_csv_sha256']:
        raise ValueError('The master CSV differs from WD-001')
    subjects = list(range(3, 13))
    amplitude = amplitude_lookup(args.master_csv, subjects)
    tables, matrices, audits = [], [], {}
    for subject in subjects:
        table, features, audit = prepare_subject(subject, args, config)
        for position in ('first', 'second'):
            table[f'{position}_nominal_charge_uC'] = [
                round(float(amplitude.loc[subject, WAVEFORMS[code]]) * PHASE_MS *
                      AREA_FACTORS[WAVEFORMS[code]], 12)
                for code in table[f'{position}_trigger']
            ]
        table['matched_triangle_lindec'] = (
            table.comparison.eq('241-242') &
            np.isclose(table.first_nominal_charge_uC, table.second_nominal_charge_uC,
                       rtol=0, atol=1e-12)
        )
        tables.append(table)
        matrices.append(features)
        audits[str(subject)] = audit
    return (
        pd.concat(tables, ignore_index=True),
        {name: np.concatenate([matrix[name] for matrix in matrices]) for name in matrices[0]},
        audits,
    )
