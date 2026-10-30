"""Physical derivative units, Gaussian support, pairing and inference invariants."""
import itertools
import numpy as np
import pandas as pd
import pytest
from experiments.response_characterization.e006_multiscale_responses.analysis import (
    specifications,transform,validate_support,condition_weights,split_weights,
    repeatability,window_features,group_tests)
from experiments.response_characterization.e002_electrode_position_slope.analysis import condition_means


def meta_table():
    rows = []
    for comp,(a,b) in enumerate(itertools.combinations(range(239,244),2)):
        for pair in range(4):
            order = (a,b) if pair % 2 else (b,a)
            for p,trigger in enumerate(order,1):
                rows.append(dict(trigger=trigger,position=p,comparison=f'{a}-{b}',pair_id=f'{comp}:{pair}'))
    return pd.DataFrame(rows)


def test_gaussian_derivative_retains_polarity_and_physical_units():
    fs = 1000.
    t = np.arange(-1,1.001,1/fs)
    ramp = np.stack([2*t,-3*t])[None]
    for spec in specifications():
        values = transform(ramp,fs,spec)[:,:,200:-200]
        if spec['operator'] in ('gaussian_derivative','local_slope'):
            np.testing.assert_allclose(values,np.broadcast_to([[[2.],[-3.]]],values.shape),atol=1e-10)
        if spec['operator'] == 'difference_of_gaussians':
            np.testing.assert_allclose(values,0.,atol=1e-10)


def test_gaussian_derivative_of_a_sinusoid_has_expected_scale_attenuation():
    fs = 1000.
    t = np.arange(-1,1.001,1/fs)
    omega = 2*np.pi*8
    spec = next(s for s in specifications() if s['name'] == 'deriv20')
    result = transform(np.sin(omega*t)[None,None],fs,spec)[0,0]
    expected = omega*np.exp(-(.020*omega)**2/2)*np.cos(omega*t)
    np.testing.assert_allclose(result[200:-200],expected[200:-200],atol=.08)


def test_support_rejects_short_epochs_and_direct_onset_overlap():
    validate_support(np.arange(-400,651)/1000,1000.,specifications())
    with pytest.raises(ValueError,match='padding'):
        validate_support(np.arange(-250,551)/1000,1000.,specifications())
    # A 30-ms sigma would reach through the post-window guard into the onset.
    with pytest.raises(ValueError,match='blank'):
        validate_support(np.arange(-500,801)/1000,1000.,[
            dict(operator='gaussian_derivative',sigma_ms=30.)])


def test_condition_weights_match_balanced_comparison_means_and_keep_pairs_together():
    meta = meta_table()
    rng = np.random.default_rng(2)
    values = rng.normal(size=(len(meta),2,7))
    w = condition_weights(meta)
    np.testing.assert_allclose((w.reshape(10,-1)@values.reshape(len(meta),-1)).reshape(5,2,2,7),
                               condition_means(values,meta),atol=1e-12)
    splits,audit = split_weights(meta,repeats=4)
    assert audit['n_pairs_used'] == 40
    for split in splits:
        active = split.sum(axis=(1,2)) > 0
        assert not (active[0] & active[1]).any()
        for _,group in meta.groupby('pair_id'):
            assert active[0,group.index].all() or active[1,group.index].all()


def test_repeatability_recovers_stable_patterns_per_position_and_electrode():
    meta = meta_table()
    times = np.arange(-400,651)/1000
    waveform = np.sin(2*np.pi*8*times)
    values = (meta.trigger.to_numpy()-238)[:,None,None]*waveform[None,None]
    values = np.concatenate([values,values*10000],axis=1)
    weights,_ = split_weights(meta,repeats=4)
    result,electrodes = repeatability(values,times,weights,['C3','C4'],'test',batch=2)
    np.testing.assert_allclose(result.waveform_deviation_r,1.)
    np.testing.assert_allclose(electrodes.waveform_deviation_r,1.)
    features = window_features(values,times)
    assert features.shape == (len(meta),2,4,2)
    assert (features[:,1] >= 0).all()


def test_group_correction_includes_every_representation():
    rng = np.random.default_rng(2)
    effects = []
    for _ in range(8):
        one = dict(position=np.zeros((1,4,4,2)),waveform=np.zeros((4,4,4,2)),
                   interaction=np.zeros((4,4,4,2)))
        one['position'][0,3,2,1] = 10+rng.normal(0,.05)
        effects.append(one)
    tests = group_tests(effects,['C3','C4'],['a_mean','a_rms','b_mean','b_rms'])
    hit = tests.query("effect == 'position' and metric == 'b_rms' and window == 'middle' and channel == 'C4'")
    assert hit.iloc[0].p_fwer < .05
    assert len(tests) == 3*4*4*2
    assert (tests.query("effect == 'waveform'").p_fwer == 1).all()
