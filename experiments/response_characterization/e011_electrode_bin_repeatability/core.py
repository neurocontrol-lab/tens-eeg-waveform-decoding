"""Fixed local derivatives, real-support binning and paired split estimators."""
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from experiments.response_characterization.e002_electrode_position_slope.analysis import effect_design

WIDTHS_MS = (1, 5, 25, 50, 100, 200)
REPRESENTATIONS = ('voltage_mean', 'slope_mean', 'slope_rms', 'curvature_mean', 'curvature_rms')
UNITS = dict(voltage_mean='uV', slope_mean='uV/s', slope_rms='uV/s',
             curvature_mean='uV/s^2', curvature_rms='uV/s^2')
WINDOW_SAMPLES = 25
POLYORDER = 2
START_MS = 50
END_MS = 1188  # Half-open; final centre 1187 uses samples through 1199 ms.
REPEATS = 100
SEED = 20261030


def local_representations(epochs_uv, time_s, sfreq):
    """Differentiate each full-rate trial before RMS or trial averaging."""
    x = np.asarray(epochs_uv)
    if sfreq != 1000. or x.ndim != 3 or x.shape[-1] != len(time_s) or not np.isfinite(x).all():
        raise ValueError('Require finite trial/channel/time epochs at the original 1 kHz')
    if not np.allclose(np.diff(time_s), 1/sfreq, atol=1e-12, rtol=0):
        raise ValueError('Irregular epoch grid')
    radius = WINDOW_SAMPLES//2
    valid = (time_s >= START_MS/1000-1e-12) & (time_s < END_MS/1000-1e-12)
    ix = np.flatnonzero(valid)
    if not len(ix) or ix[0] < radius or ix[-1]+radius >= x.shape[-1]:
        raise ValueError('Insufficient real derivative support')
    first = savgol_filter(x, WINDOW_SAMPLES, POLYORDER, deriv=1, delta=1/sfreq, axis=-1)
    second = savgol_filter(x, WINDOW_SAMPLES, POLYORDER, deriv=2, delta=1/sfreq, axis=-1)
    return {'voltage':x, 'slope':first, 'curvature':second}


def bin_features(values, time_s, width_ms, rms=False):
    """Full nonoverlapping bins only, all anchored at +50 ms."""
    if width_ms not in WIDTHS_MS:
        raise ValueError('Unregistered bin width')
    width = int(width_ms)  # original 1-kHz grid
    first = np.flatnonzero(np.isclose(time_s, START_MS/1000, rtol=0, atol=1e-12))
    if len(first) != 1:
        raise ValueError('Missing +50-ms anchor')
    bins = (END_MS-START_MS)//width
    start = int(first[0])
    chosen = np.asarray(values)[...,start:start+bins*width]
    if chosen.shape[-1] != bins*width or not np.isfinite(chosen).all():
        raise ValueError('Missing or invalid complete bin support')
    shaped = chosen.reshape(*chosen.shape[:-1], bins, width)
    features = np.sqrt(np.mean(shaped**2, axis=-1)) if rms else shaped.mean(axis=-1)
    left = START_MS+np.arange(bins)*width
    return features, left, left+width


def eligible_split_pairs(meta):
    """Both halves must retain the same observed ordered-comparison strata."""
    if not meta.groupby('pair_id').size().eq(2).all():
        raise ValueError('Need complete paired presentations')
    groups = [g.pair_id.tolist() for _,g in meta[meta.position == 1].groupby(['comparison','trigger']) if len(g)>=2]
    used = {p for group in groups for p in group}
    if not used:
        raise ValueError('No ordered comparison has at least two pairs')
    return groups, used


def waveform_operator(meta):
    """Five sum-zero adjusted waveform deviations for each presentation.

    Comparison intercepts and within-comparison elapsed time are nuisances.
    Position and waveform-by-position terms allow distinct first/second profiles.
    """
    design, slices = effect_design(meta.reset_index(drop=True))
    rank = int(np.linalg.matrix_rank(design))
    if rank != design.shape[1] or len(meta) <= rank:
        raise ValueError(f'Unidentifiable split design: {rank}/{design.shape[1]}, n={len(meta)}')
    inverse = np.linalg.pinv(design)
    contrast = np.vstack([np.eye(4), -np.ones((1,4))])
    waves = np.stack([contrast@(inverse[slices['waveform']]+p*inverse[slices['interaction']]) for p in (-.5,.5)])
    # Fitted common response at zero waveform contrasts and centred time,
    # equally averaging comparison intercepts. Descriptive reference only.
    n_comparisons = design.shape[1]-10
    common = np.stack([inverse[:n_comparisons].mean(axis=0)+p*inverse[slices['position']][0] for p in (-.5,.5)])
    return waves, common, dict(n_events=len(meta),rank=rank,columns=design.shape[1],condition_number=float(np.linalg.cond(design)))


def split_operators(meta, subject, repeats=REPEATS):
    groups, used = eligible_split_pairs(meta)
    if used != set(meta.pair_id):
        raise ValueError('Remove unsplittable pairs explicitly before constructing operators')
    rng = np.random.default_rng(SEED+subject)
    operators, rows, audits = [], [], []
    for repeat in range(repeats):
        left = set()
        for group in groups:
            left.update(rng.permutation(group)[:len(group)//2])
        halves = []
        for half, pairs in enumerate((left, used-left)):
            ix = np.flatnonzero(meta.pair_id.isin(pairs).to_numpy())
            wave, _, audit = waveform_operator(meta.iloc[ix].reset_index(drop=True))
            padded = np.zeros((2,5,len(meta)))
            padded[...,ix] = wave
            halves.append(padded)
            audits.append(dict(repeat=repeat,half=half,**audit))
            rows.extend(dict(repeat=repeat,half=half,pair_id=p) for p in sorted(pairs))
        operators.append(halves)
    return np.stack(operators),pd.DataFrame(rows),pd.DataFrame(audits)


def repeatability(features, operators, batch=5):
    """Per-position/channel/bin profile correlation and split cross-product.

    r compares five adjusted, sum-zero waveform values (only four contrast df).
    Cross-product is mean(A_k*B_k) over five waveforms, retaining squared units.
    Under independent mean-zero errors it avoids a squared-noise offset; shared
    drift/artifacts or model misspecification can still make it positive.
    """
    if not np.isfinite(features).all() or features.shape[0] != operators.shape[-1]:
        raise ValueError('Feature/split dimensions disagree or nonfinite features')
    n,channels,bins = features.shape
    flat = np.ascontiguousarray(features).reshape(n,-1)
    repeats = len(operators)
    correlations = np.empty((repeats,2,channels,bins), dtype=np.float64)
    products = np.empty_like(correlations)
    norms = np.empty_like(correlations)
    # Only numerical-zero contrasts are undefined; this is not a physiological cutoff.
    tolerance = 100*np.finfo(float).eps*np.maximum(np.sqrt(np.mean(features**2,axis=0)),np.finfo(float).tiny)
    for start in range(0,repeats,batch):
        op = operators[start:start+batch]
        profile = (op.reshape(-1,n)@flat).reshape(len(op),2,2,5,channels,bins)
        profile -= profile.mean(axis=3,keepdims=True)
        a,b = profile[:,0],profile[:,1]
        product = np.mean(a*b,axis=2)
        energy_a,energy_b = np.mean(a*a,axis=2),np.mean(b*b,axis=2)
        denominator = np.sqrt(energy_a*energy_b)
        valid = (energy_a > tolerance[None,None]**2) & (energy_b > tolerance[None,None]**2)
        correlations[start:start+len(op)] = np.divide(product,denominator,
            out=np.full(product.shape,np.nan),where=valid)
        products[start:start+len(op)] = product
        norms[start:start+len(op)] = .5*(energy_a+energy_b)
    with np.errstate(invalid='ignore'):
        median_r = np.nanmedian(correlations,axis=0)
    return dict(median_split_r=median_r,mean_split_product=products.mean(axis=0),
        split_product_sd=products.std(axis=0,ddof=1) if repeats>1 else np.full(products.shape[1:],np.nan),
        valid_r_splits=np.isfinite(correlations).sum(axis=0),
        mean_half_contrast_energy=norms.mean(axis=0))


def group_interval(values, indices):
    """Participant mean and pointwise bootstrap CI; split repetitions are not N."""
    values = np.asarray(values)
    finite = np.isfinite(values)
    count = finite.sum(axis=0)
    total = np.where(finite,values,0.)
    mean = np.divide(total.sum(axis=0),count,out=np.full(count.shape,np.nan),where=count>0)
    weights = np.array([np.bincount(row,minlength=len(values)) for row in indices],dtype=float)
    flat, valid = total.reshape(len(values),-1),finite.reshape(len(values),-1).astype(float)
    low,high = np.empty(flat.shape[1]),np.empty(flat.shape[1])
    for start in range(0,flat.shape[1],512):
        denom = weights@valid[:,start:start+512]
        estimates = np.divide(weights@flat[:,start:start+512],denom,
            out=np.full(denom.shape,np.nan),where=denom>0)
        with np.errstate(invalid='ignore'):
            low[start:start+512],high[start:start+512] = np.nanquantile(estimates,[.025,.975],axis=0)
    return mean,low.reshape(count.shape),high.reshape(count.shape),count
