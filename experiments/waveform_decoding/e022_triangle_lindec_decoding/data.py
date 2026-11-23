"""All-electrode raw-voltage means with verified event alignment and causal baselines."""

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


WINDOWS = {'pre': (-1.05, -.05), 'post': (.05, 1.05)}
BASELINE = (-1.25, -1.05)


def bin_means(data, sample, window, timestamps, sfreq):
    start, stop = interval_samples(sample, *window, sfreq)
    if start < 0 or stop > data.shape[1]:
        raise ValueError('Epoch crosses recording boundary')
    if np.any(np.abs(np.diff(timestamps[start:stop]) - 1 / sfreq) > .1 / sfreq):
        raise ValueError('Timestamp discontinuity in feature window')
    if not np.isfinite(data[:, start:stop]).all():
        raise ValueError('Nonfinite EEG in feature window')
    vectors = []
    for index in range(20):
        a, b = interval_samples(sample, window[0] + index * .05,
                                window[0] + (index + 1) * .05, sfreq)
        vectors.append(data[:, a:b].mean(axis=1) * 1e6)
    return np.stack(vectors)


def prepare_subject(subject, args, config):
    source = args.data_dir / f'Subject{subject}_data/EEGDATA_SUB{subject}-T4.xdf'
    old_path = args.baseline / f'sub-{subject}/events.csv'
    visual = args.visual / f'sub-{subject}/audit.json'
    inputs = {str(path): digest(path) for path in (source, old_path, visual, args.master_csv)}
    if inputs[str(source)] != json.loads(visual.read_text())['source_sha256']:
        raise ValueError('XDF differs from the verified raw input')
    signature = dict(inputs_sha256=inputs, windows=WINDOWS, pair_baseline=BASELINE,
                     implementation_sha256=digest(Path(__file__)))
    signature = json.loads(json.dumps(signature))
    folder = args.cache / f'sub-{subject}'
    if (folder / 'complete.json').exists():
        audit = json.loads((folder / 'complete.json').read_text())
        if audit['signature'] != signature:
            raise ValueError('Cache provenance differs')
        for name, expected in audit['outputs_sha256'].items():
            if digest(folder / name) != expected:
                raise ValueError('Cached file changed')
        with np.load(folder / 'features.npz') as arrays:
            values = {name: arrays[name].copy() for name in arrays.files}
        return pd.read_csv(folder / 'events.csv'), values, audit

    print(f'Participant {subject}: raw 62-electrode triangle/LinDec features', flush=True)
    data, names, triggers, timestamps, sfreq = load_recording(source)
    if len(names) != 62 or sfreq != 1000:
        raise ValueError('Expected verified 62-channel, 1000 Hz recording')
    aligned = align_events(trigger_runs(triggers, sfreq, config['min_trigger_samples']),
                           read_behavior(args.master_csv, subject), subject, timestamps)
    old = pd.read_csv(old_path)
    fields = ['event_index', 'sample', 'trigger', 'pair_id', 'position', 'csv_row']
    pd.testing.assert_frame_equal(aligned[fields], old[fields], check_dtype=False)
    data -= data.mean(axis=0, keepdims=True)
    selected = old.loc[old.analysis_eligible & old.trigger.isin([241, 242])].copy()
    first_events = old.loc[old.position.eq(1)].set_index('pair_id')
    features = {name: [] for name in WINDOWS}
    maximum_ptp = []
    for row in selected.itertuples():
        first = first_events.loc[row.pair_id]
        a, b = interval_samples(int(first['sample']), *BASELINE, sfreq)
        if a < 0 or b > len(timestamps) or not np.isfinite(data[:, a:b]).all():
            raise ValueError('Invalid pair baseline support')
        if np.any(np.abs(np.diff(timestamps[a:b]) - 1 / sfreq) > .1 / sfreq):
            raise ValueError('Timestamp gap in pair baseline')
        if np.isfinite(first.previous_gap_s) and first.previous_gap_s <= -BASELINE[0]:
            raise ValueError('Previous stimulus overlaps the pair baseline')
        if np.isfinite(row.previous_gap_s) and row.previous_gap_s <= 1.05:
            raise ValueError('Previous trigger falls inside the pre window')
        if np.isfinite(row.next_gap_s) and row.next_gap_s <= 1.05:
            raise ValueError('Next trigger falls inside the post window')
        baseline = data[:, a:b].mean(axis=1) * 1e6
        for name, window in WINDOWS.items():
            features[name].append(bin_means(data, row.sample, window, timestamps, sfreq) - baseline)
        start, stop = interval_samples(row.sample, .05, 1.05, sfreq)
        maximum_ptp.append(float(np.ptp(data[:, start:stop], axis=1).max() * 1e6))
    selected['all_channel_post_ptp_uv'] = maximum_ptp
    selected['target_lindec'] = selected.trigger.eq(242).astype(int)
    selected = selected.reset_index(drop=True)
    arrays = {name: np.array(values) for name, values in features.items()}
    folder.mkdir(parents=True, exist_ok=True)
    selected.to_csv(folder / 'events.csv', index=False)
    np.savez_compressed(folder / 'features.npz', **arrays)
    audit = dict(
        signature=signature, channels=names, events=len(selected),
        outputs_sha256={name: digest(folder / name) for name in ('events.csv', 'features.npz')},
        channel_policy='All 62 channels; original 14-channel complete-pair QC retained. Additional channels receive finite-value checks and descriptive peak-to-peak flags.',
    )
    (folder / 'complete.json').write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8')
    return selected, arrays, audit


def assemble(args):
    config = json.loads((args.baseline / 'run_config.json').read_text())
    if digest(args.master_csv) != config['master_csv_sha256']:
        raise ValueError('Master CSV differs from WD-001')
    amplitude = amplitude_lookup(args.master_csv, list(range(3, 13)))
    tables, matrices, audits = [], [], {}
    for subject in range(3, 13):
        table, arrays, audit = prepare_subject(subject, args, config)
        charges = {code: round(float(amplitude.loc[subject, WAVEFORMS[code]]) *
                              PHASE_MS * AREA_FACTORS[WAVEFORMS[code]], 12)
                   for code in (241, 242)}
        table['nominal_charge_uC'] = table.trigger.map(charges)
        table['matched_direct'] = table.comparison.eq('241-242') & np.isclose(
            charges[241], charges[242], rtol=0, atol=1e-12)
        tables.append(table)
        matrices.append(arrays)
        audits[str(subject)] = audit
    return pd.concat(tables, ignore_index=True), {
        name: np.concatenate([matrix[name] for matrix in matrices]) for name in WINDOWS
    }, audits
