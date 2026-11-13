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
REVIEWED_CSV_SHA256 = '19596071570fb8c152e0a4bd6468590ad3754abf0ce6c14ab63504fd4db0300e'
REVIEWED_CSV_BLOCKS = {(6, 'LinInc'): (2002, 2031), (12, 'Sine'): (4560, 4589)}
REVIEWED_XDF_EVENT = ('EEGDATA_SUB9-T3_W1-2-3.xdf', 3178732, 233)


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


def reviewed_cell(subject, waveform, cell, found):
    key = (subject, waveform)
    if key in REVIEWED_CSV_BLOCKS:
        first, last = REVIEWED_CSV_BLOCKS[key]
        keep = cell.csv_row.between(first, last)
        selected = cell.loc[keep].copy()
        if not np.array_equal(selected.csv_row, np.arange(first, last + 1)):
            raise ValueError(f'Unexpected row identities in reviewed CSV block {key}')
        levels, counts = np.unique(selected.amplitude_mA, return_counts=True)
        if len(levels) != 3 or not np.allclose(np.diff(levels), .25) or not (counts == 10).all():
            raise ValueError(f'Unexpected amplitude tiers in reviewed CSV block {key}')
        selected['tier'] = np.searchsorted(levels, selected.amplitude_mA)
        selected['standard_cell'] = True
        selected['expected_trigger'] = 224 + 3 * WAVES.index(waveform) + selected.tier
        return selected, found, {'excluded_csv_rows': cell.loc[~keep, 'csv_row'].tolist()}
    if key == (9, 'LinDec'):
        recording, sample, trigger = REVIEWED_XDF_EVENT
        extra = found.recording.eq(recording) & found['sample'].eq(sample) & found.trigger.eq(trigger)
        if extra.sum() != 1:
            raise ValueError('Expected exactly one reviewed participant 9 LinDec trigger')
        return cell, found.loc[~extra], {'excluded_xdf_event': dict(recording=recording, sample=sample, trigger=trigger)}
    return cell, found, None


def align_cells(trials, events, *, reviewed=False):
    aligned, audit = [], []
    for (subject, waveform), cell in trials.groupby(['subject', 'waveform'], sort=True):
        found = events.loc[events.subject.eq(subject) & events.waveform.eq(waveform)].reset_index(drop=True)
        row = dict(subject=int(subject), waveform=waveform, csv_trials=len(cell), xdf_trials=len(found))
        exception = None
        if reviewed:
            cell, found, exception = reviewed_cell(subject, waveform, cell, found)
        expected = cell.expected_trigger.to_numpy()
        actual = found.trigger.to_numpy()
        standard = bool(cell.standard_cell.all())
        matches = bool(np.array_equal(expected, actual)) if standard else False
        row.update(standard_cell=standard, exact_trigger_sequence=matches)
        if exception is not None:
            if not matches:
                raise ValueError(f'Reviewed Task 3 cell {subject}, {waveform} no longer aligns exactly')
            row.update(alignment_exception=exception, retained_csv_trials=len(cell), retained_xdf_trials=len(found))
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


def run(output, subjects, *, reviewed=False, reuse_extraction=None):
    if output.exists():
        raise FileExistsError(output)
    csv_sha256 = digest(CSV)
    if reviewed and csv_sha256 != REVIEWED_CSV_SHA256:
        raise ValueError('Master CSV differs from the version used to review the Task 3 exceptions')
    trials = read_trials(CSV)
    if subjects:
        trials = trials.loc[trials.subject.isin(subjects)]
    source_metadata = {}
    if reuse_extraction is None:
        frames, audits = [], []
        for subject in sorted(trials.subject.unique()):
            paths = sorted((Path('EEGData') / f'Subject{subject}_data').glob('*-T3*.xdf'))
            if not paths:
                raise FileNotFoundError(f'No Task 3 XDF for participant {subject}')
            for path in paths:
                frame, audit = recorded_events(path, int(subject))
                frames.append(frame)
                audits.append(audit)
                print(f'{path.name}: {len(frame)} recognized triggers', flush=True)
        events = pd.concat(frames, ignore_index=True)
    else:
        audit_path = reuse_extraction / 'extraction_audit.json'
        events_path = reuse_extraction / 'xdf_events.csv'
        source = json.loads(audit_path.read_text(encoding='utf-8'))
        if source['csv_sha256'] != csv_sha256:
            raise ValueError('Reused extraction has a different master CSV digest')
        if source['roi'] != list(ROI) or source['post_window_baseline_s'] != list(POST) or source['pre_window_baseline_s'] != list(PRE):
            raise ValueError('Reused extraction has different EEG feature settings')
        events = pd.read_csv(events_path, float_precision='round_trip')
        recorded_counts = {Path(a['recording']).name: a['recognized_events'] for a in source['recordings']}
        if events.recording.value_counts().to_dict() != recorded_counts:
            raise ValueError('Reused event table does not match the audited recording counts')
        requested = set(trials.subject)
        if not requested.issubset(set(events.subject)):
            raise ValueError('Reused extraction is missing requested Task 3 participants')
        events = events.loc[events.subject.isin(requested)]
        audits = [a for a in source['recordings'] if Path(a['recording']).name in set(events.recording)]
        source_metadata = dict(reuse_extraction=str(reuse_extraction), source_audit_sha256=digest(audit_path),
                               source_xdf_events_sha256=digest(events_path))
    aligned, cells = align_cells(trials, events, reviewed=reviewed)
    output.mkdir(parents=True, exist_ok=False)
    events.to_csv(output / 'xdf_events.csv', index=False)
    aligned.to_csv(output / 'aligned_features.csv', index=False)
    metadata = dict(csv=str(CSV), csv_sha256=csv_sha256, roi=ROI, post_window_baseline_s=POST,
                    pre_window_baseline_s=PRE, nominal_cathodic_phase_ms=PHASE_MS, area_factors=AREA_FACTORS,
                    recordings=audits, cells=cells, aligned_trials=len(aligned),
                    pyxdf_version=source['pyxdf_version'] if reuse_extraction is not None else pyxdf.__version__,
                    reviewed_alignment=reviewed, **source_metadata)
    (output / 'extraction_audit.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Aligned {len(aligned)} trials in {sum(r["exact_trigger_sequence"] for r in cells)} cells', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--subjects', nargs='+', type=int)
    parser.add_argument('--reviewed-alignment', action='store_true', help='Apply the three documented Task 3 alignment exceptions')
    parser.add_argument('--reuse-extraction', type=Path, help='Reuse the recorded-event and ROI cache from a previous extraction')
    args = parser.parse_args()
    run(args.output, args.subjects, reviewed=args.reviewed_alignment, reuse_extraction=args.reuse_extraction)
