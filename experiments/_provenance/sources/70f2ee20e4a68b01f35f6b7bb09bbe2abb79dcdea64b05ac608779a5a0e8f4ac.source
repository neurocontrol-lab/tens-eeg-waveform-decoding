"""Event extraction and strict behavioral alignment, independent of EEG models."""
import numpy as np
import pandas as pd

WAVEFORMS = {239: 'Sine', 240: 'Square', 241: 'Triangle', 242: 'LinDec', 243: 'LinInc'}
CODES = tuple(WAVEFORMS)


def trigger_runs(values, sfreq, min_samples=3):
    """Keep original run onsets; audit glitches and active-low normalization."""
    if min_samples < 2:
        raise ValueError('Debouncing requires at least two stable samples')
    values = np.asarray(values)
    if not np.isfinite(values).all() or not np.allclose(values, np.rint(values)):
        raise ValueError('Trigger samples must be finite integers')
    values = np.rint(values).astype(np.int64)
    starts = np.r_[0, np.flatnonzero(np.diff(values)) + 1]
    ends = np.r_[starts[1:], len(values)]
    rows = []
    for start, end in zip(starts, ends):
        raw = int(values[start])
        if raw in (0, 65535):
            continue
        code = raw if raw in CODES else 65535 - raw
        expected = code in CODES
        rows.append(dict(sample=int(start), raw_trigger=raw,
                         trigger=int(code) if expected else raw,
                         inverted=bool(expected and code != raw),
                         duration_samples=int(end-start), duration_ms=1000*(end-start)/sfreq,
                         accepted=bool(expected and end-start >= min_samples),
                         reason='accepted' if expected and end-start >= min_samples else
                         ('short_run' if expected else 'unexpected_code')))
    return pd.DataFrame(rows, columns=['sample', 'raw_trigger', 'trigger', 'inverted',
                                      'duration_samples', 'duration_ms', 'accepted', 'reason'])


def read_behavior(path, subject):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    df['csv_row'] = np.arange(len(df)) + 2
    df = df[(df.subject_id == subject) & (df.task_id == 4)].copy()
    if df.empty:
        raise ValueError(f'No Task 4 behavioral rows for subject {subject}')
    # Preserve file order: a reset/repeated trial ID must not silently reorder data.
    for col in ['t4 waveform 1', 't4 waveform 2']:
        if not df[col].isin(range(1, 6)).all():
            raise ValueError(f'Invalid waveform number in {col}')
    if (df['t4 waveform 1'] == df['t4 waveform 2']).any():
        raise ValueError('A Task 4 pair must contain distinct waveforms')
    return df.reset_index(drop=True)


def align_events(runs, behavior, subject, timestamps):
    """Require exact full sequence agreement; never scan forward into later trials."""
    events = runs[runs.accepted].copy().reset_index(drop=True)
    expected = behavior[['t4 waveform 1', 't4 waveform 2']].to_numpy(int).ravel() + 238
    if len(events) != len(expected) or not np.array_equal(events.trigger, expected):
        n = min(len(events), len(expected))
        mismatch = np.flatnonzero(events.trigger.to_numpy()[:n] != expected[:n])
        first = int(mismatch[0]) if len(mismatch) else n
        raise ValueError(f'Strict CSV alignment failed: {len(events)} EEG events vs '
                         f'{len(expected)} expected; first disagreement at event {first}. '
                         'Audit saved trigger_runs.csv; no automatic resynchronization attempted.')
    events['subject'] = subject
    events['event_index'] = np.arange(len(events))
    events['waveform'] = events.trigger.map(WAVEFORMS)
    events['timestamp'] = timestamps[events['sample']]
    events['elapsed_s'] = events.timestamp - timestamps[0]
    events['previous_gap_s'] = events.timestamp.diff()
    events['next_gap_s'] = events.timestamp.shift(-1) - events.timestamp
    events['previous_trigger'] = events.trigger.shift(1).fillna(0).astype(int)
    events['pair_index'] = np.arange(len(events)) // 2
    events['position'] = np.arange(len(events)) % 2 + 1
    events['csv_row'] = np.repeat(behavior.csv_row.to_numpy(), 2)
    events['trial_id'] = np.repeat(behavior.trial_id.to_numpy(), 2)
    choice = pd.to_numeric(behavior['t4 naturalness value'], errors='coerce')
    events['response_value'] = np.repeat(choice.to_numpy(), 2)
    events['response_valid'] = events.response_value.isin([1, 2])
    events['legacy_detected_value'] = np.repeat(behavior['t4 waveform detected'].to_numpy(), 2)
    pair_types = [tuple(sorted(p)) for p in expected.reshape(-1, 2)]
    blocks, current = [], -1
    for i, pair in enumerate(pair_types):
        if i == 0 or pair != pair_types[i-1]:
            current += 1
        blocks.append(current)
    events['block_index'] = np.repeat(blocks, 2)
    # Group repeated/restarted occurrences of the same comparison together as well.
    events['comparison'] = np.repeat([f'{a}-{b}' for a, b in pair_types], 2)
    events['group'] = events.subject.astype(str) + ':' + events.comparison
    events['pair_id'] = events.subject.astype(str) + ':' + events.pair_index.astype(str)
    return events


def protocol_audit(events):
    rows = []
    for block, group in events.groupby('block_index', sort=False):
        pairs = group[group.position == 1]
        wins = {}
        early = False
        for i, row in enumerate(pairs.itertuples()):
            if row.response_valid:
                pair = group[group.pair_index == row.pair_index]
                chosen = int(pair.iloc[int(row.response_value)-1].trigger)
                wins[chosen] = wins.get(chosen, 0) + 1
                if wins[chosen] >= 5 and i < len(pairs)-1:
                    early = True
        gaps = group[group.position == 1].next_gap_s
        rows.append(dict(block=int(block), comparison=pairs.comparison.iloc[0],
                         n_pairs=len(pairs), missed_responses=int((~pairs.response_valid).sum()),
                         maximum_wins=max(wins.values(), default=0),
                         continued_after_five=early,
                         race_to_five_consistent=max(wins.values(), default=0) == 5 and not early,
                         pair_gap_min=float(gaps.min()), pair_gap_max=float(gaps.max())))
    return rows


def load_recording(path):
    import pyxdf
    # Preserve the stream's recorded timestamps rather than silently regularizing gaps.
    streams, _ = pyxdf.load_xdf(str(path), synchronize_clocks=False, dejitter_timestamps=False)
    candidates = []
    for stream in streams:
        channels = stream['info'].get('desc', [{}])[0].get('channels', [{}])[0].get('channel', [])
        if any(ch.get('label') == ['TRIGGERS'] for ch in channels):
            candidates.append((stream, channels))
    if len(candidates) != 1:
        raise ValueError(f'Expected one stream with TRIGGERS, found {len(candidates)}')
    stream, channels = candidates[0]
    labels = [ch['label'][0] for ch in channels]
    if len(set(labels)) != len(labels):
        raise ValueError('Duplicate channel labels')
    sfreq = float(stream['info']['nominal_srate'][0])
    times = np.asarray(stream['time_stamps'])
    if len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError('Nonmonotonic recording timestamps')
    eeg_indices, eeg_names, factors = [], [], []
    for i, channel in enumerate(channels):
        if channel.get('type') == ['EEG'] and labels[i] not in ['CREF', 'M1', 'M2']:
            unit = channel.get('unit', [''])[0].lower().replace('µ', 'u').replace('μ', 'u')
            if unit not in ['uvolt', 'uv', 'microvolt', 'volt', 'v']:
                raise ValueError(f'Unknown EEG unit {unit!r} for {labels[i]}')
            eeg_indices.append(i)
            eeg_names.append(labels[i])
            factors.append(1e-6 if unit in ['uvolt', 'uv', 'microvolt'] else 1.)
    data = stream['time_series'][:, eeg_indices].astype(np.float64).T
    data *= np.asarray(factors)[:, None]
    trigger = stream['time_series'][:, labels.index('TRIGGERS')]
    return data, eeg_names, trigger, times, sfreq
