"""Fixed signal operations; no ICA, fitted artifact rejection or classification."""
import numpy as np
from scipy.signal import butter, sosfiltfilt

BASELINE = (-.25, -.05)
EPOCH = (-.30, 1.20)
FILTER_EDGE_SECONDS = 8.


def contiguous_segments(data, timestamps, sfreq):
    """Never filter through missing/nonfinite data or irregular timestamp steps."""
    valid = np.isfinite(data).all(axis=0) & np.isfinite(timestamps)
    breaks = np.r_[True, np.abs(np.diff(timestamps)-1/sfreq) > .1/sfreq]
    starts = np.flatnonzero(valid & (breaks | np.r_[True, ~valid[:-1]]))
    stops = np.flatnonzero(valid & (np.r_[~valid[1:], True] | np.r_[breaks[1:], True]))+1
    if len(starts) != len(stops):
        raise ValueError('Unmatched continuous-segment boundaries')
    return list(zip(starts.tolist(), stops.tolist()))


def filter_continuous(data, timestamps, sfreq):
    """Fixed Butterworth .5-30 Hz, prototype order 4, forward/backward SOS.

    Band-pass transformation gives eight poles per pass. This descriptive,
    noncausal filter can spread stimulation artifacts before and after onset.
    """
    if sfreq <= 60:
        raise ValueError('Sample rate must exceed twice the upper cutoff')
    sos = butter(4, [.5, 30.], btype='bandpass', fs=sfreq, output='sos')
    filtered = np.full(data.shape, np.nan, dtype=float)
    edge_ok = np.zeros(data.shape[1], dtype=bool)
    edge = round(FILTER_EDGE_SECONDS*sfreq)
    segments = contiguous_segments(data, timestamps, sfreq)
    for start, stop in segments:
        if stop-start <= 2*edge:
            continue
        filtered[:, start:stop] = sosfiltfilt(sos, data[:, start:stop], axis=-1)
        edge_ok[start+edge:stop-edge] = True
    return filtered, edge_ok, sos, segments


def extract_epochs(data, samples, sfreq):
    """Volts to baseline-corrected uV, retaining true 1-kHz samples."""
    first, last = [round(v*sfreq) for v in EPOCH]
    time = np.arange(first, last)/sfreq
    base = (time >= BASELINE[0]) & (time < BASELINE[1])
    if not base.any():
        raise ValueError('No baseline samples')
    epochs = np.stack([data[:, int(s)+first:int(s)+last] for s in samples])
    if epochs.shape[-1] != len(time) or not np.isfinite(epochs).all():
        raise ValueError('Invalid epoch support')
    epochs = (epochs-epochs[..., base].mean(axis=-1, keepdims=True))*1e6
    return epochs, time


def epoch_eligibility(events, timestamps, edge_ok, sfreq):
    """Freeze original QC and add only explicit longer-support/filter-edge checks."""
    reasons = []
    for row in events.itertuples():
        why = []
        if not row.analysis_eligible:
            why.append('original_baseline_exclusion')
        a = int(row.sample)+round(EPOCH[0]*sfreq)
        b = int(row.sample)+round(EPOCH[1]*sfreq)
        if a < 0 or b >= len(timestamps):
            why.append('extended_recording_boundary')
        else:
            if not edge_ok[a:b].all():
                why.append('filter_edge_gap_or_nonfinite')
            if np.any(np.abs(np.diff(timestamps[a:b])-1/sfreq) > .1/sfreq):
                why.append('extended_timestamp_irregularity')
        if np.isfinite(row.previous_gap_s) and row.previous_gap_s <= -EPOCH[0]:
            why.append('previous_stimulus_in_display_epoch')
        if np.isfinite(row.next_gap_s) and row.next_gap_s <= EPOCH[1]:
            why.append('next_stimulus_in_display_epoch')
        reasons.append(';'.join(why))
    result = events.copy()
    result['visual_exclusion_reason'] = reasons
    result['visual_event_eligible'] = result.visual_exclusion_reason.eq('')
    result['visual_pair_eligible'] = result.groupby('pair_id').visual_event_eligible.transform('all')
    return result


def cell_balanced_evokeds(epochs, meta, codes):
    """Equal observed comparison-cell weights within waveform and position.

    Context support differs by waveform: these are descriptive means, not
    regression-adjusted contrasts or a causal comparison.
    """
    means, counts, cells = [], [], []
    for position in (1, 2):
        wave_means, wave_counts, wave_cells = [], [], []
        for code in codes:
            mask = (meta.position.to_numpy() == position) & (meta.trigger.to_numpy() == code)
            contexts = meta.loc[mask, 'comparison'].unique()
            if not len(contexts):
                raise ValueError(f'Missing waveform/position cell: {code}/{position}')
            estimates = [epochs[mask & meta.comparison.eq(c).to_numpy()].mean(axis=0) for c in contexts]
            wave_means.append(np.mean(estimates, axis=0))
            wave_counts.append(int(mask.sum()))
            wave_cells.append(len(contexts))
        means.append(wave_means)
        counts.append(wave_counts)
        cells.append(wave_cells)
    return np.asarray(means), np.asarray(counts), np.asarray(cells)
