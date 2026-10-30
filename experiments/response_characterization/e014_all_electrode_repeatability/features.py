"""Fixed power windows and leakage-safe calibration/A/B trial partitions."""
import numpy as np
import pandas as pd
from scipy.signal import periodogram
from experiments.waveform_decoding.e005_spectral_decoding.core import integrate_band
from experiments.response_characterization.e011_electrode_bin_repeatability.core import waveform_operator

POWER_WINDOW_MS=500
POWER_STEP_MS=50
POWER_STARTS_MS=np.arange(50,551,50)
TRIPLET_SEED=20261031


def power_features(epochs_uv,times):
    """500-ms Hann density estimates, 50-ms step; total power, not induced power.

    Centre locations are 300-800 ms. Overlapping windows are not independent.
    Native Fourier spacing is 2 Hz. Exact alpha/beta boundaries are integrated.
    """
    values={name:[] for name in ('alpha_power','beta_power')}
    floor_hits={name:0 for name in values}
    for start in POWER_STARTS_MS:
        indices=np.flatnonzero((times>=start/1000-1e-12)&(times<(start+POWER_WINDOW_MS)/1000-1e-12))
        if len(indices)!=500:raise ValueError('Require 500 real 1-kHz spectral samples')
        f,psd=periodogram(np.asarray(epochs_uv)[...,indices],fs=1000.,window='hann',
            detrend='constant',scaling='density',nfft=500,axis=-1)
        for name,lo,hi in [('alpha_power',8.,13.),('beta_power',13.,30.)]:
            linear=integrate_band(f,psd,lo,hi)
            floor_hits[name]+=int((linear<=1e-20).sum())
            values[name].append(10*np.log10(np.maximum(linear,1e-20)))
    features={name:np.stack(arrays,axis=-1) for name,arrays in values.items()}
    centres=POWER_STARTS_MS+POWER_WINDOW_MS/2
    # Heatmap cells locate window centres; true window edges are saved separately.
    return features,centres-POWER_STEP_MS/2,centres+POWER_STEP_MS/2,dict(
        window_starts_ms=POWER_STARTS_MS.tolist(),window_ends_ms=(POWER_STARTS_MS+500).tolist(),
        window_ms=500,step_ms=50,frequency_spacing_hz=2.,floor_hits=floor_hits)


def calibration_splits(meta,subject,repeats=100):
    """Calibration/A/B are disjoint complete pairs, preserving A/B cell support.

    Two-pair strata supply one pair to A and B and no calibration pair. Larger
    strata rotate three balanced allocations. All five classes must be present
    in calibration; A and B must support the original nuisance model.
    """
    groups=[g.pair_id.tolist() for _,g in meta[meta.position==1].groupby(['comparison','trigger'])]
    if any(len(g)<2 for g in groups):raise ValueError('Remove original singleton strata explicitly first')
    rng=np.random.default_rng(TRIPLET_SEED+subject)
    partitions=[];rows=[];audit=[]
    for repeat in range(repeats):
        roles=[set(),set(),set()]
        for group in groups:
            order=rng.permutation(group)
            if len(order)==2:
                roles[1].add(order[0]);roles[2].add(order[1])
            else:
                thirds=np.array_split(order,3)
                for role in range(3):roles[role].update(thirds[(role+repeat)%3])
        if any(roles[a]&roles[b] for a,b in [(0,1),(0,2),(1,2)]):raise AssertionError('Calibration leakage')
        if set.union(*roles)!=set(meta.pair_id):raise AssertionError('Missing paired observations')
        indices=[np.flatnonzero(meta.pair_id.isin(role).to_numpy()) for role in roles]
        if set(meta.iloc[indices[0]].trigger)!=set(range(239,244)):raise ValueError('Calibration lacks a waveform class')
        for role,ix in enumerate(indices):
            if role:
                _,_,design=waveform_operator(meta.iloc[ix].reset_index(drop=True))
                audit.append(dict(repeat=repeat,role=('calibration','A','B')[role],**design))
            rows.extend(dict(repeat=repeat,role=('calibration','A','B')[role],pair_id=p) for p in sorted(roles[role]))
        partitions.append(indices)
    return partitions,pd.DataFrame(rows),pd.DataFrame(audit)


def fit_spatial(calibration_uv,labels):
    """xDAWN calibrated only on calibration epochs/labels; no classifier."""
    from mne.decoding import XdawnTransformer
    if set(labels)!=set(range(239,244)):raise ValueError('Calibration needs all five classes')
    model=XdawnTransformer(n_components=2,reg=.1,rank='full')
    model.fit(calibration_uv*1e-6,np.asarray(labels))
    # Transform an identity basis to obtain the selected public transform,
    # independent of the internal storage layout of class-specific filters.
    filters=model.transform(np.eye(calibration_uv.shape[1])[None])[0]
    sign=np.sign(filters[np.arange(len(filters)),np.abs(filters).argmax(axis=1)])
    filters*=np.where(sign==0,1,sign)[:,None]
    projected=filters@calibration_uv
    rms=np.sqrt(np.mean(projected**2,axis=(0,2)))
    if not np.isfinite(rms).all() or (rms<=0).any():raise ValueError('Degenerate calibration components')
    normalized=filters/rms[:,None]
    sensor_rms=np.sqrt(np.mean(calibration_uv**2,axis=(0,2)))
    if (sensor_rms<=0).any():raise ValueError('Degenerate calibration electrode')
    return normalized,sensor_rms,model.classes_


def centre_feature_time(values):
    """Remove each feature's temporal mean before joint shape correlation.

    Applied equally to matched sensors and learned components. Joint agreement
    is then invariant to simultaneous component sign flips in A and B.
    """
    return values-values.mean(axis=-1,keepdims=True)
