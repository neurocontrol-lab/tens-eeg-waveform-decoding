"""Build a compact Task 3 feature cache with an auditable trigger join."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyxdf

from experiments.waveform_decoding.e001_voltage_baseline.features import interval_samples


WAVES = ('Sine', 'Square', 'Triangle', 'LinDec', 'LinInc')
ROI = ('C4', 'CP4')
POST = (0.150, 0.250, -0.250, -0.050)
PRE = (-0.250, -0.150, -0.650, -0.450)
CSV = Path('EEG_Codes_Export/02_SourceLocalization_Task4/Master_Data_ToSub12 (1).csv')
AREA_FACTORS = dict(zip(WAVES, (2 / np.pi, 1., .5, .5, .5)))
PHASE_MS = .4


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read_trials(path):
    trials = pd.read_csv(path)
    trials.columns = trials.columns.str.strip()
    trials['csv_row'] = np.arange(len(trials)) + 2
    trials = trials.loc[trials.Task_id.eq(3)].copy()
    trials['subject'] = trials.Subject_id.astype(int)
    trials['waveform'] = trials['Waveform/Spot'].map(dict(enumerate(WAVES, 1)))
    trials['amplitude_mA'] = pd.to_numeric(trials.Amplitude, errors='raise')
    if trials.waveform.isna().any() or not np.isfinite(trials.amplitude_mA).all():
        raise ValueError('Missing waveform or amplitude in Task 3')
    trials['charge_uC'] = trials.amplitude_mA * PHASE_MS * trials.waveform.map(AREA_FACTORS)
    trials['tier'] = -1
    trials['standard_cell'] = False
    for (_, _), cell in trials.groupby(['subject', 'waveform'], sort=False):
        levels = np.sort(cell.amplitude_mA.unique())
        if len(levels) == 3 and np.allclose(np.diff(levels), .25):
            trials.loc[cell.index, 'standard_cell'] = True
            trials.loc[cell.index, 'tier'] = np.searchsorted(levels, cell.amplitude_mA)
    sensation_columns = ['Pulse', 'Electric', 'Needle', 'Touch', 'Tingling', 'Vibration', 'Beat', 'Contraction']
    sensations = trials[sensation_columns].apply(pd.to_numeric, errors='coerce')
    trials['response_valid'] = sensations.notna().all(axis=1) & sensations.ne(0).any(axis=1)
    trials['expected_trigger'] = 224 + 3 * trials.waveform.map(dict(zip(WAVES, range(5)))) + trials.tier
    return trials


def load_roi(path):
    # Reduce each XDF chunk immediately. Only the two referenced ROI channels and
    # triggers need to remain in memory for these predefined voltage features.
    selection = None

    def reduce_chunk(values, stamps, stream, _stream_id):
        nonlocal selection
        if selection is None:
            channels = stream['info']['desc'][0]['channels'][0]['channel']
            labels = [ch['label'][0] for ch in channels]
            scalp = [i for i, ch in enumerate(channels)
                     if ch.get('type') == ['EEG'] and labels[i] not in ('CREF', 'M1', 'M2')]
            units = [channels[i].get('unit', [''])[0].lower().replace('µ', 'u').replace('μ', 'u') for i in scalp]
            scale = {'uvolt': 1., 'uv': 1., 'microvolt': 1., 'volt': 1e6, 'v': 1e6}
            factors = np.asarray([scale[unit] for unit in units])
            picks = [scalp.index(labels.index(ch)) for ch in ROI]
            selection = (scalp, factors, picks, labels.index('TRIGGERS'))
        scalp, factors, picks, trigger = selection
        uv = values[:, scalp] * factors
        roi = uv[:, picks] - uv.mean(axis=1, keepdims=True)
        return np.column_stack([values[:, trigger], roi]), stamps, stream

    streams, _ = pyxdf.load_xdf(str(path), select_streams=[{'name': 'SAGA', 'type': 'EEG'}],
                               on_chunk=reduce_chunk, synchronize_clocks=False, dejitter_timestamps=False)
    if len(streams) != 1:
        raise ValueError(f'Expected one SAGA EEG stream in {path.name}')
    stream = streams[0]
    sfreq = float(stream['info']['nominal_srate'][0])
    times = np.asarray(stream['time_stamps'])
    if len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError(f'Nonmonotonic timestamps in {path.name}')
    data = np.asarray(stream['time_series'])
    return data[:, 0], data[:, 1:], times, sfreq


def recorded_events(path, subject):
    values, roi, times, sfreq = load_roi(path)
    if not np.isfinite(values).all() or not np.equal(values, np.rint(values)).all():
        raise ValueError(f'Noninteger triggers in {path.name}')
    starts = np.r_[0, np.flatnonzero(np.diff(values) != 0) + 1]
    ends = np.r_[starts[1:], len(values)]
    raw = values[starts].astype(int)
    stable = (ends - starts >= 3) & ~np.isin(raw, [0, 65535])
    starts, ends, raw = starts[stable], ends[stable], raw[stable]
    codes = np.where(np.isin(raw, np.arange(224, 239)), raw, 65535 - raw)
    known = np.isin(codes, np.arange(224, 239))
    rows = []
    for index in np.flatnonzero(known):
        sample = int(starts[index])
        start, end = interval_samples(sample, -.65, .25, sfreq)
        reasons = []
        if start < 0 or end > len(times):
            reasons.append('recording_boundary')
        else:
            if np.any(np.abs(np.diff(times[start:end]) - 1 / sfreq) > .1 / sfreq):
                reasons.append('timestamp_gap')
            if (index and starts[index - 1] >= start) or (index + 1 < len(starts) and starts[index + 1] < end):
                reasons.append('neighbor_trigger')
            if not np.isfinite(roi[start:end]).all():
                reasons.append('nonfinite_eeg')
            a, b = interval_samples(sample, .05, .25, sfreq)
            c, d = interval_samples(sample, -.65, -.05, sfreq)
            if max(np.ptp(roi[a:b], axis=0).max(), np.ptp(roi[c:d], axis=0).max()) > 500.:
                reasons.append('ptp_over_500uV')
        features = {}
        for label, (lo, hi, blo, bhi) in [('post_uv', POST), ('pre_uv', PRE)]:
            a, b = interval_samples(sample, lo, hi, sfreq)
            c, d = interval_samples(sample, blo, bhi, sfreq)
            features[label] = float(roi[a:b].mean() - roi[c:d].mean()) if not reasons else np.nan
        code = int(codes[index])
        rows.append(dict(subject=subject, recording=path.name, sample=sample,
                         timestamp_s=float(times[sample]), trigger=code, raw_trigger=int(raw[index]),
                         waveform=WAVES[(code - 224) // 3], eeg_qc=';'.join(reasons), **features))
    counts = pd.Series(raw[~known]).value_counts().to_dict()
    audit = dict(recording=str(path), sha256=digest(path), sampling_hz=sfreq, samples=len(times),
                 recognized_events=len(rows), other_stable_raw_codes={str(k): int(v) for k, v in counts.items()})
    return pd.DataFrame(rows), audit


def align_cells(trials, events):
    aligned, audit = [], []
    for (subject, waveform), cell in trials.groupby(['subject', 'waveform'], sort=True):
        found = events.loc[events.subject.eq(subject) & events.waveform.eq(waveform)].reset_index(drop=True)
        expected = cell.expected_trigger.to_numpy()
        actual = found.trigger.to_numpy()
        standard = bool(cell.standard_cell.all())
        matches = bool(np.array_equal(expected, actual)) if standard else False
        row = dict(subject=int(subject), waveform=waveform, csv_trials=len(cell), xdf_trials=len(found),
                   standard_cell=standard, exact_trigger_sequence=matches)
        if not matches:
            row['reason'] = 'nonstandard_amplitudes' if not standard else 'trigger_sequence_mismatch'
            row['csv_expected_codes'] = expected.tolist() if standard else []
            row['xdf_codes'] = actual.tolist()
        else:
            joined = found.assign(csv_row=cell.csv_row.to_numpy(), trial_id=cell.Trial_id.to_numpy(),
                                  amplitude_mA=cell.amplitude_mA.to_numpy(), charge_uC=cell.charge_uC.to_numpy(),
                                  tier=cell.tier.to_numpy(), response_valid=cell.response_valid.to_numpy())
            aligned.append(joined)
        audit.append(row)
    if not aligned:
        raise ValueError('No complete Task 3 waveform sequences align')
    return pd.concat(aligned, ignore_index=True), audit


def run(output, subjects):
    output.mkdir(parents=True, exist_ok=False)
    trials = read_trials(CSV)
    if subjects:
        trials = trials.loc[trials.subject.isin(subjects)]
    events, audits = [], []
    for subject in sorted(trials.subject.unique()):
        paths = sorted((Path('EEGData') / f'Subject{subject}_data').glob('*-T3*.xdf'))
        if not paths:
            raise FileNotFoundError(f'No Task 3 XDF for participant {subject}')
        for path in paths:
            frame, audit = recorded_events(path, int(subject))
            events.append(frame)
            audits.append(audit)
            print(f'{path.name}: {len(frame)} recognized triggers', flush=True)
    events = pd.concat(events, ignore_index=True)
    events.to_csv(output / 'xdf_events.csv', index=False)
    aligned, cells = align_cells(trials, events)
    aligned.to_csv(output / 'aligned_features.csv', index=False)
    metadata = dict(csv=str(CSV), csv_sha256=digest(CSV), roi=ROI, post_window_baseline_s=POST,
                    pre_window_baseline_s=PRE, nominal_cathodic_phase_ms=PHASE_MS, area_factors=AREA_FACTORS,
                    recordings=audits, cells=cells, aligned_trials=len(aligned), pyxdf_version=pyxdf.__version__)
    (output / 'extraction_audit.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Aligned {len(aligned)} trials in {sum(r["exact_trigger_sequence"] for r in cells)} cells', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--subjects', nargs='+', type=int)
    args = parser.parse_args()
    run(args.output, args.subjects)
