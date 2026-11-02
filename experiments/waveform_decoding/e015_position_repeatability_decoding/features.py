"""Label-free scalar features; supervised xDAWN is fitted separately per fold."""
import numpy as np
from experiments.response_characterization.e011_electrode_bin_repeatability.core import (
    local_representations, bin_features, REPRESENTATIONS, UNITS)
from experiments.response_characterization.e014_all_electrode_repeatability.features import power_features

WIDTHS=(25,50,100,200)


def contract_rows(family,names,left,right,width,unit):
    return [dict(feature=f'{family}:{ch}:{a:g}-{b:g}ms:w{width}',family=family,
                 channel=ch,start_ms=float(a),end_ms=float(b),width_ms=width,unit=unit)
            for ch in names for a,b in zip(left,right)]


def sensor_features(epochs_uv,times,names):
    """24,614 features per epoch; no labels, selection, scaling or ICA."""
    local=local_representations(epochs_uv,times,1000.)
    blocks=[];contract=[]
    for family in REPRESENTATIONS:
        for width in WIDTHS:
            x,left,right=bin_features(local[family.split('_')[0]],times,width,rms=family.endswith('_rms'))
            keep=right<=1050;x=x[...,keep];left=left[keep];right=right[keep]
            blocks.append(x.reshape(len(x),-1))
            contract+=contract_rows(family,names,left,right,width,UNITS[family])
    powers,_,_,audit=power_features(epochs_uv,times)
    for family,x in powers.items():
        blocks.append(x.reshape(len(x),-1))
        contract+=contract_rows(family,names,audit['window_starts_ms'],audit['window_ends_ms'],500,'dB re 1 uV^2')
    result=np.concatenate(blocks,axis=1)
    if result.shape[1]!=len(contract) or not np.isfinite(result).all():raise ValueError('Invalid feature contract')
    return result,contract


def xdawn_features(epochs_uv,times,filters,component_names):
    """750 scalars in one fixed calibration basis; no fitting here."""
    blocks=[];contract=[]
    for width in WIDTHS:
        bins,left,right=bin_features(epochs_uv,times,width);keep=right<=1050
        x=np.einsum('kc,nct->nkt',filters,bins[...,keep],optimize=True)
        blocks.append(x.reshape(len(x),-1))
        contract+=contract_rows('xdawn',component_names,left[keep],right[keep],width,'calibration RMS units')
    return np.concatenate(blocks,axis=1),contract
