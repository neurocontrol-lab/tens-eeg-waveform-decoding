"""Fixed features with no temporal filtering or fitted whole-recording transforms."""
import numpy as np

CHANNELS = ('FC1', 'FC2', 'C3', 'C1', 'Cz', 'C2', 'C4', 'CP5', 'CP1', 'CP2', 'CP6', 'P3', 'Pz', 'P4')
WINDOWS = {
    'post': (0.05, 0.50, 0.05, -0.25, -0.05),
    'pre': (-0.50, -0.05, 0.05, -0.75, -0.55),
    'late': (0.15, 0.50, 0.05, -0.25, -0.05),
    'onset': (0.0, 0.020, 0.005, -0.25, -0.05),
}


def interval_samples(sample, lo, hi, sfreq):
    return sample + int(round(lo*sfreq)), sample + int(round(hi*sfreq))


def build_features(data, names, events, timestamps, sfreq, ptp_limit_uv=500.):
    """Return features plus eligibility; retain all original events in the manifest.

    Pre/post use equal-length windows and separate preceding 200 ms baselines.
    Only within-time spatial referencing is applied; no future samples enter pre.
    """
    missing = set(CHANNELS) - set(names)
    if missing:
        raise ValueError(f'Missing predefined channels: {sorted(missing)}')
    picks = [names.index(ch) for ch in CHANNELS]
    # Fixed average reference, no data-fitted component rejection or channel choice.
    selected = data[picks] - data.mean(axis=0, keepdims=True)
    n = len(events)
    feature_sets = {name: [] for name in WINDOWS}
    eligible, reasons, ptps = [], [], []
    for row in events.itertuples():
        start, end = interval_samples(row.sample, -0.75, 0.50, sfreq)
        why = []
        if start < 0 or end >= data.shape[1]:
            why.append('recording_boundary')
        else:
            dt = np.diff(timestamps[start:end+1])
            if np.any(np.abs(dt - 1/sfreq) > .1/sfreq):
                why.append('timestamp_gap_or_irregularity')
            if (row.previous_gap_s == row.previous_gap_s and row.previous_gap_s <= .75) or (
                    row.next_gap_s == row.next_gap_s and row.next_gap_s <= .50):
                why.append('neighbor_stimulus_in_epoch')
            audit_segments = [selected[:, slice(*interval_samples(row.sample, a, b, sfreq))]
                              for a,b in [(-.75,-.05),(.05,.50)]]
            ptp = max(float(np.ptp(segment, axis=1).max()) for segment in audit_segments)
            if not np.isfinite(selected[:, start:end]).all():
                why.append('nonfinite_eeg')
            if ptp > ptp_limit_uv*1e-6:
                why.append('fixed_ptp_limit')
        ptps.append(ptp*1e6 if start >= 0 and end < data.shape[1] else np.nan)
        eligible.append(not why)
        reasons.append(';'.join(why))
        for name, (lo, hi, step, blo, bhi) in WINDOWS.items():
            n_bins = int(round((hi-lo)/step))
            if why:
                feature_sets[name].append(np.full(len(CHANNELS)*n_bins, np.nan))
                continue
            bstart, bend = interval_samples(row.sample, blo, bhi, sfreq)
            baseline = selected[:, bstart:bend].mean(axis=1)
            vectors = []
            for k in range(n_bins):
                a, b = interval_samples(row.sample, lo+k*step, lo+(k+1)*step, sfreq)
                vectors.append((selected[:, a:b].mean(axis=1)-baseline)*1e6)
            feature_sets[name].append(np.concatenate(vectors))
    result = events.copy()
    result['eeg_eligible'] = eligible
    result['eeg_exclusion_reason'] = reasons
    result['ptp_uv'] = ptps
    # Whole pairs retained/excluded together to preserve permutation and fold units.
    result['pair_eeg_eligible'] = result.groupby('pair_id').eeg_eligible.transform('all')
    result['analysis_eligible'] = result.pair_eeg_eligible & result.response_valid
    order = np.column_stack([
        result.elapsed_s.to_numpy(), result.event_index.to_numpy(),
        result.position.to_numpy(), result.previous_gap_s.fillna(0).to_numpy(),
        *[(result.previous_trigger == code).to_numpy(float) for code in [0,239,240,241,242,243]],
    ])
    feature_sets['order'] = order
    return {name: np.asarray(values) for name, values in feature_sets.items()}, result
