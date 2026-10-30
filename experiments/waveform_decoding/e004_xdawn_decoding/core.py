"""Fixed xDAWN spatial representation; no full-cohort fitted transforms."""
import numpy as np
from mne.decoding import XdawnTransformer
from experiments.waveform_decoding.e001_voltage_baseline.evaluate import model
from experiments.waveform_decoding.e001_voltage_baseline.events import CODES

COMPONENTS_PER_CLASS = 2
SPATIAL_SHRINKAGE = .1


def bin_epochs(epochs, samples_per_bin):
    if epochs.ndim != 3 or epochs.shape[-1] % samples_per_bin:
        raise ValueError('Epoch length must contain complete fixed temporal bins')
    return epochs.reshape(*epochs.shape[:2],-1,samples_per_bin).mean(axis=-1)


def flatten_bins(bins):
    """Bin-major, component-minor µV representation, matching the sensor layout."""
    return bins.transpose(0,2,1).reshape(len(bins),-1)*1e6


def projected_contract(n_bins=9, start=.05, step=.05):
    return [dict(feature=f'{code}:component-{component+1}:{1000*(start+k*step):g}-{1000*(start+(k+1)*step):g}ms',
                 unit='weighted_uV',trigger=code,component=component+1,
                 start_s=start+k*step,end_s=start+(k+1)*step)
            for k in range(n_bins) for code in CODES for component in range(COMPONENTS_PER_CLASS)]


def fit_fold(epochs_train, epochs_test, labels_train, samples_per_bin):
    """Only training epochs/labels enter the spatial and classifier fits.

    Fit spatial filters on full-rate epochs. For transformation only, temporal
    averaging commutes with a fixed linear spatial projection; project bins
    rather than creating full projected test epochs. This equivalence is tested.
    """
    if set(labels_train) != set(CODES):
        raise ValueError('Training data must include all five waveforms')
    spatial = XdawnTransformer(n_components=COMPONENTS_PER_CLASS,
                              reg=SPATIAL_SHRINKAGE,rank='full')
    spatial.fit(epochs_train,labels_train)
    if not np.array_equal(spatial.classes_,CODES):
        raise ValueError('Unexpected xDAWN class order')
    train = flatten_bins(spatial.transform(bin_epochs(epochs_train,samples_per_bin)))
    test = flatten_bins(spatial.transform(bin_epochs(epochs_test,samples_per_bin)))
    if train.shape[1] != 9*len(CODES)*COMPONENTS_PER_CLASS:
        raise ValueError('Unexpected projected feature count')
    if not np.isfinite(train).all() or not np.isfinite(test).all():
        raise ValueError('Nonfinite projected features')
    classifier = model().fit(train,labels_train)
    return classifier.predict(test),classifier.predict_proba(test),spatial,classifier,train,test


def predict_xdawn_oof(epochs,labels,splits,samples_per_bin,on_fold=None):
    predicted = np.full(len(labels),-1,int)
    probabilities = np.zeros((len(labels),len(CODES)))
    seen = np.zeros(len(labels),int)
    for fold,(train,test) in enumerate(splits):
        result = fit_fold(epochs[train],epochs[test],labels[train],samples_per_bin)
        predicted[test],probabilities[test] = result[:2]
        seen[test] += 1
        if on_fold is not None:
            on_fold(fold,train,test,*result[2:])
    if not np.all(seen == 1):
        raise ValueError('Every event must have exactly one held-out prediction')
    return predicted,probabilities
