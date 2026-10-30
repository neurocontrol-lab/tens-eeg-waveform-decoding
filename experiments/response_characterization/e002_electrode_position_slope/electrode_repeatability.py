"""Descriptive per-electrode repeatability of fixed waveform/window features.

Complements the continuous scalp-time check so one high-amplitude electrode
cannot dominate the measure. No feature selection or significance testing.
"""
import numpy as np
import pandas as pd
from .analysis import METRICS,condition_means


def electrode_repeatability(features,meta,channels,repeats=100,seed=20261021):
    if len(features) != len(meta) or features.shape[1:] != (3,4,len(channels)):
        raise ValueError('Feature/metadata/channel dimensions disagree')
    first = meta[meta.position == 1]
    groups = [g.pair_id.tolist() for _,g in first.groupby(['comparison','trigger']) if len(g) >= 2]
    used = {pair for group in groups for pair in group}
    rng = np.random.default_rng(seed)
    correlations = []
    for _ in range(repeats):
        left = set()
        for group in groups:
            chosen = rng.permutation(group)
            left.update(chosen[:len(group)//2])
        a = meta.pair_id.isin(left).to_numpy()
        b = meta.pair_id.isin(used-left).to_numpy()
        ma = condition_means(features[a],meta[a].reset_index(drop=True))[:,:,:,1:]
        mb = condition_means(features[b],meta[b].reset_index(drop=True))[:,:,:,1:]
        # Waveform deviations, then correlate across five waveforms x three post windows.
        ma -= np.nanmean(ma,axis=0,keepdims=True)
        mb -= np.nanmean(mb,axis=0,keepdims=True)
        ma = ma.transpose(1,2,4,0,3).reshape(2,3,len(channels),-1)
        mb = mb.transpose(1,2,4,0,3).reshape(2,3,len(channels),-1)
        finite = np.isfinite(ma) & np.isfinite(mb)
        count = finite.sum(axis=-1,keepdims=True)
        aa = np.where(finite,ma,0.)
        bb = np.where(finite,mb,0.)
        aa = np.where(finite,aa-aa.sum(axis=-1,keepdims=True)/np.maximum(count,1),0.)
        bb = np.where(finite,bb-bb.sum(axis=-1,keepdims=True)/np.maximum(count,1),0.)
        denominator = np.sqrt((aa**2).sum(axis=-1)*(bb**2).sum(axis=-1))
        r = np.divide((aa*bb).sum(axis=-1),denominator,
                      out=np.full(denominator.shape,np.nan),where=(denominator>1e-12)&(count[...,0]>=3))
        correlations.append(r)
    median = np.nanmedian(correlations,axis=0)
    rows = []
    for position in range(2):
        for metric,name in enumerate(METRICS):
            for c,channel in enumerate(channels):
                rows.append(dict(position=position+1,metric=name,channel=channel,
                                 median_waveform_feature_r=median[position,metric,c],
                                 n_pairs_in_repeatability=len(used),n_splits=repeats))
    return pd.DataFrame(rows)
