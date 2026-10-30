import itertools
import mne
import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits
from experiments.waveform_decoding.e001_voltage_baseline.evaluate import make_splits,permute_pairs,scores
from experiments.waveform_decoding.e001_voltage_baseline.features import CHANNELS,build_features
from experiments.waveform_decoding.e004_xdawn_decoding import core
from experiments.waveform_decoding.e004_xdawn_decoding.data import extract_epochs
from experiments.waveform_decoding.e004_xdawn_decoding.report import holm


@pytest.fixture(autouse=True)
def quiet_single_thread():
    with threadpool_limits(limits=1),mne.use_log_level('ERROR'):
        yield


def example():
    rows = []
    index = 0
    for a,b in itertools.combinations(range(239,244),2):
        for pair in range(4):
            for position,code in enumerate((a,b) if pair%2 else (b,a),1):
                rows.append(dict(subject=3,group=f'3:{a}-{b}',pair_id=f'3:{index}',
                    trigger=code,position=position,elapsed_s=index*4+position*1.4))
            index += 1
    meta = pd.DataFrame(rows)
    rng = np.random.default_rng(14)
    x = rng.normal(0,.04e-6,(len(meta),14,90))
    # Stable distributed spatial response across the entire epoch, plus noise.
    for i,code in enumerate(meta.trigger):
        x[i,code-239] += 1e-6
    return x,meta


def test_full_rate_spatial_projection_commutes_with_bin_means():
    x,meta = example()
    _,_,spatial,_,train,test = core.fit_fold(x[:64],x[64:],meta.trigger.to_numpy()[:64],10)
    expected = core.flatten_bins(core.bin_epochs(spatial.transform(x[64:]),10))
    np.testing.assert_allclose(test,expected,atol=1e-12,rtol=1e-12)
    assert train.shape == (64,90)


def test_heldout_values_cannot_change_fitted_spatial_or_classifier_parameters():
    x,meta = example()
    first = core.fit_fold(x[:64],x[64:],meta.trigger.to_numpy()[:64],10)
    second = core.fit_fold(x[:64],x[64:]+1e-3,meta.trigger.to_numpy()[:64],10)
    np.testing.assert_array_equal(first[2].filters_,second[2].filters_)
    np.testing.assert_array_equal(first[3].steps[0][1].mean_,second[3].steps[0][1].mean_)
    np.testing.assert_array_equal(first[3].steps[-1][1].coef_,second[3].steps[-1][1].coef_)


def test_spatial_fit_uses_only_training_rows_and_is_repeated_for_null_labels(monkeypatch):
    x,meta = example()
    y = meta.trigger.to_numpy()
    splits = make_splits(meta)
    fits = []
    original = core.XdawnTransformer
    class RecordedXdawn(original):
        def fit(self,X,y):
            fits.append((X.copy(),y.copy()))
            return super().fit(X,y)
    monkeypatch.setattr(core,'XdawnTransformer',RecordedXdawn)
    p,_ = core.predict_xdawn_oof(x,y,splits,10)
    assert scores(y,p)['balanced_accuracy'] > .9
    py = permute_pairs(y,meta,np.random.default_rng(4))
    core.predict_xdawn_oof(x,py,splits,10)
    assert len(fits) == 2*len(splits)
    for fold,(train,test) in enumerate(splits):
        np.testing.assert_array_equal(fits[fold][0],x[train])
        np.testing.assert_array_equal(fits[fold][1],y[train])
        np.testing.assert_array_equal(fits[fold+len(splits)][1],py[train])
        assert len(fits[fold][0]) < len(x)


def test_epoch_reconstruction_matches_sensor_features_and_protects_pre_window():
    rng = np.random.default_rng(3)
    data = rng.normal(0,1e-6,(len(CHANNELS),6000))
    meta = pd.DataFrame(dict(sample=[1000,3000],pair_id=['3:0']*2,response_valid=True,
        previous_gap_s=[np.nan,2.],next_gap_s=[2.,np.nan],elapsed_s=[1.,3.],
        event_index=[0,1],position=[1,2],previous_trigger=[0,239]))
    features,manifest = build_features(data,list(CHANNELS),meta,np.arange(6000)/1000,1000)
    for window in ['post','pre']:
        epochs,width = extract_epochs(data,list(CHANNELS),meta,1000,window)
        np.testing.assert_allclose(core.flatten_bins(core.bin_epochs(epochs,width)),features[window],atol=1e-12)
    before,_ = extract_epochs(data,list(CHANNELS),meta,1000,'pre')
    data[0,1050:1500] += 1e-3
    after,_ = extract_epochs(data,list(CHANNELS),meta,1000,'pre')
    np.testing.assert_array_equal(before,after)


def test_each_epoch_has_one_prediction_and_two_window_p_correction():
    x,meta = example()
    splits = make_splits(meta)
    with pytest.raises(ValueError,match='exactly one'):
        core.predict_xdawn_oof(x,meta.trigger.to_numpy(),splits[:-1],10)
    np.testing.assert_allclose(holm([.04,.01]),[.04,.02])
