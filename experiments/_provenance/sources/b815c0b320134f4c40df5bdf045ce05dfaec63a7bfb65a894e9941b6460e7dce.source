"""Fixed slope features, context-adjusted effects, and participant-level tests."""
import itertools
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

WINDOWS = {'pre': (-.20, -.05), 'early': (.05, .15),
           'middle': (.15, .30), 'late': (.30, .50)}
METRICS = ('voltage_uv', 'signed_slope_uv_s', 'absolute_slope_uv_s')
SLOPE_WINDOW_S = .025


def epoch_and_slopes(data, events, times, sfreq, ptp_limit_uv=500.):
    """Extract padded epochs; derivatives never see padded-edge extrapolation.

    QC uses all 62 scalp channels, baseline [-250,-50) ms and post [50,500) ms.
    Interpolated onset is excluded from amplitude QC. Exclude entire pairs.
    """
    offsets = np.arange(round(-.30*sfreq), round(.55*sfreq)+1)
    rel = offsets/sfreq
    baseline = (rel >= -.25) & (rel < -.05)
    pre = (rel >= -.25) & (rel < -.05)
    post = (rel >= .05) & (rel < .5)
    epochs = np.full((len(events), data.shape[0], len(rel)), np.nan)
    reasons, amplitudes = [], []
    for i, row in enumerate(events.itertuples()):
        indices = int(row.sample)+offsets
        why, amplitude = [], np.nan
        if indices[0] < 0 or indices[-1] >= data.shape[1]:
            why.append('boundary')
        else:
            dt = np.diff(times[indices])
            if np.any(np.abs(dt-1/sfreq) > .1/sfreq):
                why.append('timestamp_irregularity')
            if row.previous_gap_s <= .30 or row.next_gap_s <= .55:
                why.append('neighbor_in_padded_epoch')
            trial = data[:, indices]*1e6
            if not np.isfinite(trial).all():
                why.append('nonfinite')
            else:
                amplitude = max(np.ptp(trial[:, pre], axis=1).max(),
                                np.ptp(trial[:, post], axis=1).max())
                if amplitude > ptp_limit_uv:
                    why.append('fixed_all_channel_ptp')
                epochs[i] = trial-trial[:, baseline].mean(axis=1, keepdims=True)
        reasons.append(';'.join(why))
        amplitudes.append(amplitude)
    meta = events.copy()
    meta['diagnostic_eeg_eligible'] = np.array(reasons) == ''
    meta['diagnostic_exclusion_reason'] = reasons
    meta['diagnostic_ptp_uv'] = amplitudes
    meta['diagnostic_pair_eligible'] = meta.groupby('pair_id').diagnostic_eeg_eligible.transform('all')
    meta['diagnostic_eligible'] = meta.diagnostic_pair_eligible & meta.response_valid
    keep = meta.diagnostic_eligible.to_numpy()
    window = max(3, int(round(SLOPE_WINDOW_S*sfreq)) | 1)
    slopes = (savgol_filter(epochs[keep], window, 1, deriv=1, delta=1/sfreq, axis=-1)
              if keep.any() else epochs[keep].copy())
    selected = epochs[keep]
    values = []
    for lo, hi in WINDOWS.values():
        mask = (rel >= lo) & (rel < hi)
        values.append(np.stack([selected[:, :, mask].mean(axis=-1),
                                slopes[:, :, mask].mean(axis=-1),
                                np.abs(slopes[:, :, mask]).mean(axis=-1)], axis=1))
    # Trial x metric x fixed window x channel.
    features = np.stack(values, axis=2)
    return selected, slopes, rel, features, meta


def effect_design(meta):
    """Sum-coded waveform/position effects with comparison and time nuisance.

    Position -0.5/+0.5 makes position coefficients second-minus-first and
    waveform coefficients averages over positions. Each comparison gets its
    own intercept. Time is centered within comparison and scaled by overall SD.
    """
    codes = meta.trigger.to_numpy()-239
    wave = np.eye(5)[codes, :4]
    wave[codes == 4] = -1.
    position = meta.position.to_numpy(float)-1.5
    comparison = pd.get_dummies(meta.comparison, dtype=float).to_numpy()
    elapsed = meta.elapsed_s-meta.groupby('comparison').elapsed_s.transform('mean')
    elapsed = elapsed.to_numpy()/max(float(elapsed.std()), 1e-12)
    start = comparison.shape[1]
    design = np.column_stack([comparison, elapsed, position, wave, wave*position[:, None]])
    slices = dict(position=slice(start+1, start+2), waveform=slice(start+2, start+6),
                  interaction=slice(start+6, start+10))
    return design, slices


def fit_effects(features, meta):
    design, slices = effect_design(meta)
    rank = int(np.linalg.matrix_rank(design))
    if rank != design.shape[1] or len(meta) <= rank:
        raise ValueError(f'Effects not identifiable: design rank {rank}/{design.shape[1]}, n={len(meta)}')
    beta = np.linalg.lstsq(design, features.reshape(len(meta), -1), rcond=None)[0]
    effects = {key: beta[sl].reshape(-1, *features.shape[1:]) for key, sl in slices.items()}
    return effects, dict(rank=rank, columns=design.shape[1], n_events=len(meta),
                         condition_number=float(np.linalg.cond(design)))


def t_squared(coefficients):
    """Studentized group effects; sum across waveform contrasts when needed."""
    n = len(coefficients)
    mean = coefficients.mean(axis=0)
    variance = coefficients.var(axis=0, ddof=1)/n
    denominator = np.maximum(variance, np.finfo(float).eps*np.maximum(1., mean**2))
    return (mean**2/denominator).sum(axis=0)


def group_tests(subject_effects, channels):
    """Exact participant sign-flips; max-stat across metrics/windows/electrodes.

    Three effect families receive an additional Bonferroni factor of three.
    Requires independent participants and symmetric errors in their coefficients.
    These are exploratory model-conditional tests, not stimulus-randomization tests.
    """
    n = len(subject_effects)
    if n < 3 or n > 16:
        raise ValueError('Exact sign-flip diagnostic supports 3-16 participants')
    signs = np.array(list(itertools.product([-1., 1.], repeat=n)))
    rows = []
    for family in ('position', 'waveform', 'interaction'):
        coefficients = np.stack([effects[family] for effects in subject_effects])
        observed = t_squared(coefficients)
        exceed = np.zeros(observed.shape, int)
        exceed_max = np.zeros(observed.shape, int)
        for sign in signs:
            null = t_squared(coefficients*sign.reshape(n, 1, 1, 1, 1))
            # Tiny tolerance retains mathematically tied sign patterns.
            exceed += null >= observed-1e-10
            exceed_max += null.max() >= observed-1e-10
        for m, metric in enumerate(METRICS):
            for w, window in enumerate(WINDOWS):
                for c, channel in enumerate(channels):
                    rows.append(dict(effect=family, metric=metric, window=window,
                                     channel=channel, n_subjects=n,
                                     statistic=float(observed[m,w,c]),
                                     p_uncorrected=float(exceed[m,w,c]/len(signs)),
                                     p_max_family=float(exceed_max[m,w,c]/len(signs)),
                                     p_fwer=float(min(1.,3*exceed_max[m,w,c]/len(signs))),
                                     mean_position_difference=(float(coefficients[:,0,m,w,c].mean())
                                                               if family == 'position' else np.nan)))
    return pd.DataFrame(rows)


def condition_means(values, meta):
    """Waveform x position means, giving each observed comparison equal weight.

    Missing comparison/position cells remain a limitation; save support separately.
    """
    out = np.full((5, 2, *values.shape[1:]), np.nan)
    for code in range(239,244):
        for position in [1,2]:
            cells = []
            for _, indices in meta[(meta.trigger == code) & (meta.position == position)].groupby('comparison').groups.items():
                cells.append(values[np.asarray(list(indices), int)].mean(axis=0))
            if cells:
                out[code-239,position-1] = np.mean(cells, axis=0)
    return out


def split_half_repeatability(epochs, slopes, meta, rel, seed=20261021, repeats=100):
    """Descriptive within-participant reliability, not independent replication.

    Split complete pairs within each ordered comparison. Thus every eligible
    cell occurs in both halves, and paired presentations never cross halves.
    Singleton ordered cells are omitted only from this repeatability diagnostic.
    """
    first = meta[meta.position == 1]
    groups = [g.pair_id.tolist() for _,g in first.groupby(['comparison','trigger']) if len(g) >= 2]
    used = {pair for group in groups for pair in group}
    rng = np.random.default_rng(seed)
    mask = (rel >= .05) & (rel < .5)
    results = []
    def corr(a,b):
        a,b = a.ravel(),b.ravel()
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 3 or np.std(a[ok]) < 1e-12 or np.std(b[ok]) < 1e-12:
            return np.nan
        return float(np.corrcoef(a[ok],b[ok])[0,1])
    for repeat in range(repeats):
        left = set()
        for group in groups:
            shuffled = rng.permutation(group)
            left.update(shuffled[:len(group)//2])
        a = meta.pair_id.isin(left).to_numpy()
        b = meta.pair_id.isin(used-left).to_numpy()
        for metric, values in [('voltage',epochs),('slope',slopes)]:
            ma = condition_means(values[a][:,:,mask], meta[a].reset_index(drop=True))
            mb = condition_means(values[b][:,:,mask], meta[b].reset_index(drop=True))
            for p in range(2):
                overall_a = np.nanmean(ma[:,p], axis=0)
                overall_b = np.nanmean(mb[:,p], axis=0)
                results.append(dict(repeat=repeat, metric=metric, position=p+1,
                    common_response_r=corr(overall_a,overall_b),
                    waveform_deviation_r=corr(ma[:,p]-overall_a, mb[:,p]-overall_b),
                    n_pairs_in_repeatability=len(used)))
    return pd.DataFrame(results)
