import numpy as np
import pytest
from threadpoolctl import threadpool_limits
from experiments.waveform_decoding.e005_spectral_decoding.core import integrate_band, power_features, representations, contracts


def tones(amplitude_uv=2., frequency=10., n_events=3):
    t = np.arange(450)/1000
    signal = amplitude_uv*1e-6*np.sin(2*np.pi*frequency*t)
    return np.tile(signal,(n_events,14,1))


def test_exact_band_edges_and_partition_preserve_integrated_area():
    f = np.arange(0,50,2.5)
    density = np.tile(2*f+3,(2,14,1))
    alpha = integrate_band(f,density,8,13)
    beta = integrate_band(f,density,13,30)
    np.testing.assert_allclose(alpha,13**2+3*13-(8**2+3*8))
    np.testing.assert_allclose(alpha+beta,integrate_band(f,density,8,30))
    with pytest.raises(ValueError):
        integrate_band(f,density,8,50)


def test_physical_units_amplitude_scaling_and_dc_invariance():
    x = tones(frequency=20)
    logged, linear, info = power_features(x,1000)
    # A 2-uV peak sinusoid has 2 uV^2 mean square, captured almost entirely in beta.
    np.testing.assert_allclose(linear[:,14:],2,rtol=.01)
    # Put nonzero energy in both bands; exact bin-centred tones may hit the
    # documented numerical floor in the empty band and cannot scale there.
    x = x+tones(amplitude_uv=1.,frequency=10)
    logged,_,info = power_features(x,1000)
    twice,_,_ = power_features(2*x,1000)
    np.testing.assert_allclose(twice-logged,20*np.log10(2),atol=1e-9)
    offset,_,_ = power_features(x+100e-6,1000)
    np.testing.assert_allclose(offset,logged,atol=1e-8)
    assert info['frequency_spacing_hz']==pytest.approx(1000/450)
    assert info['floor_hits']==0


def test_power_features_are_trial_local_and_distinguish_bands():
    x = tones(frequency=10)
    first,_,_ = power_features(x,1000)
    assert (first[:,:14] > first[:,14:]+15).all()
    changed = x.copy()
    changed[1:] = tones(frequency=20,n_events=2)
    second,_,_ = power_features(changed,1000)
    np.testing.assert_array_equal(first[0],second[0])
    assert (second[1:,14:] > second[1:,:14]+15).all()


def test_floor_and_feature_contracts_do_not_change_eligibility_or_units():
    _, linear, info = power_features(np.zeros((2,14,450)),1000)
    assert np.isfinite(power_features(np.zeros((2,14,450)),1000)[0]).all()
    assert info['floor_hits']==56 and not linear.any()
    bundles,_,_ = representations(tones(),1000,50)
    for window in ['post','pre']:
        for name, width in [('voltage',126),('power',28),('combined',154)]:
            assert bundles[name].shape==(3,width)
            assert len(contracts(window)[name])==width
    assert {x['unit'] for x in contracts('post')['combined']}=={'uV','dB_re_1_uV2'}
    bad = tones()
    bad[0,0,0] = np.nan
    with pytest.raises(ValueError,match='Finite'):
        power_features(bad,1000)


def test_known_spectral_class_information_survives_grouped_decoding_and_null_refits(monkeypatch):
    from tests.test_e004_xdawn_decoding import example
    from experiments.waveform_decoding.e001_voltage_baseline import evaluate
    _, meta = example()
    y = meta.trigger.to_numpy()
    rng = np.random.default_rng(222)
    x = rng.normal(0,.2e-6,(len(y),14,450))
    t = np.arange(450)/1000
    for i,code in enumerate(y):
        x[i,code-239] += 3e-6*np.sin(2*np.pi*20*t+rng.uniform(0,2*np.pi))
    features,_,_ = power_features(x,1000)
    splits = evaluate.make_splits(meta)
    fitted = []
    original = evaluate.model
    def record_model():
        estimator = original()
        fitted.append(estimator)
        return estimator
    monkeypatch.setattr(evaluate,'model',record_model)
    with threadpool_limits(limits=1):
        predictions,_ = evaluate.predict_oof(features,y,splits)
        py = evaluate.permute_pairs(y,meta,np.random.default_rng(19))
        evaluate.predict_oof(features,py,splits)
    assert evaluate.scores(y,predictions)['balanced_accuracy']>.9
    assert len(fitted)==2*len(splits)
    for i,(train,_) in enumerate(splits):
        np.testing.assert_allclose(fitted[i].steps[0][1].mean_,features[train].mean(axis=0))
        np.testing.assert_allclose(fitted[i+len(splits)].steps[0][1].mean_,features[train].mean(axis=0))
