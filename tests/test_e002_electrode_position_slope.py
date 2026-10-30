"""Scientific invariants: derivative units, whole-pair QC, adjustment and inference."""
import itertools
import numpy as np
import pandas as pd
import pytest

from experiments.response_characterization.e002_electrode_position_slope.analysis import (
    epoch_and_slopes, effect_design, fit_effects, group_tests, condition_means,
    split_half_repeatability)
from experiments.response_characterization.e002_electrode_position_slope.preprocess import interpolate_stimuli
from experiments.response_characterization.e002_electrode_position_slope.electrode_repeatability import electrode_repeatability


def trial_table():
    rows = []
    for comparison, (a,b) in enumerate(itertools.combinations(range(239,244),2)):
        for pair in range(8):
            ordered = (a,b) if pair % 2 else (b,a)
            for position, code in enumerate(ordered,1):
                rows.append(dict(subject=3,trigger=code,position=position,
                                 comparison=f'{a}-{b}',elapsed_s=comparison*100+pair*4+position,
                                 pair_id=f'{comparison}:{pair}'))
    return pd.DataFrame(rows)


def test_slopes_have_physical_units_and_pair_qc_is_joint():
    sfreq = 1000.
    times = np.arange(5000)/sfreq
    data = np.stack([2*times,-3*times])*1e-6
    meta = pd.DataFrame(dict(sample=[1000,2400],pair_id=['a','a'],response_valid=True,
                             previous_gap_s=[np.nan,1.4],next_gap_s=[1.4,np.nan]))
    epochs,slopes,rel,features,manifest = epoch_and_slopes(data,meta,times,sfreq)
    mask = (rel >= .05) & (rel < .5)
    np.testing.assert_allclose(slopes[:,:,mask],np.broadcast_to([[[2.],[-3.]]],slopes[:,:,mask].shape),atol=1e-10)
    np.testing.assert_allclose(features[:,1,:,0],2.,atol=1e-10)
    np.testing.assert_allclose(features[:,2,:,1],3.,atol=1e-10)
    assert manifest.diagnostic_eligible.all()
    # One bad event excludes both presentations, regardless of which electrode is bad.
    data[1,1100] = .01
    _,_,_,_,manifest = epoch_and_slopes(data,meta,times,sfreq)
    assert not manifest.diagnostic_eligible.any()
    assert manifest.diagnostic_eeg_eligible.tolist() == [False,True]


def test_context_adjustment_recovers_position_and_waveform_interaction():
    meta = trial_table()
    design,slices = effect_design(meta)
    beta = np.zeros((design.shape[1],3,4,2))
    beta[:10] = np.arange(10)[:,None,None,None]*200 # large comparison confound
    beta[10] = 70 # large within-comparison time drift
    beta[slices['position']] = 3.
    beta[slices['waveform']] = np.array([1,2,-1,-2])[:,None,None,None]
    beta[slices['interaction']] = np.array([4,-3,2,-1])[:,None,None,None]
    data = (design@beta.reshape(len(beta),-1)).reshape(len(meta),3,4,2)
    effects,audit = fit_effects(data,meta)
    assert audit['rank'] == design.shape[1]
    for family,sl in slices.items():
        np.testing.assert_allclose(effects[family],beta[sl],atol=1e-10)


def test_unidentifiable_position_effect_is_rejected():
    meta = trial_table()
    meta['position'] = 1
    with pytest.raises(ValueError,match='not identifiable'):
        fit_effects(np.zeros((len(meta),3,4,2)),meta)


def test_group_inference_corrects_the_complete_prespecified_family():
    rng = np.random.default_rng(3)
    effects = []
    for _ in range(8):
        values = dict(position=np.zeros((1,3,4,2)), waveform=np.zeros((4,3,4,2)),
                      interaction=np.zeros((4,3,4,2)))
        values['position'][0,0,1,0] = 10+rng.normal(0,.05)
        effects.append(values)
    results = group_tests(effects,['C3','C4'])
    target = results.query("effect == 'position' and metric == 'voltage_uv' and window == 'early' and channel == 'C3'").iloc[0]
    assert target.p_fwer < .05
    assert (results.query("effect == 'waveform'").p_fwer == 1).all()
    assert (results.p_fwer >= results.p_max_family).all()


def test_condition_averages_weight_comparisons_equally():
    meta = pd.DataFrame(dict(trigger=[239]*4,position=[1]*4,comparison=['a','a','a','b']))
    means = condition_means(np.array([[0.],[0.],[0.],[10.]]),meta)
    assert means[0,0,0] == 5. # not trial-weighted 2.5


def test_repeatability_recovers_stable_waveform_patterns():
    meta = trial_table()
    rel = np.arange(-.3,.551,.001)
    pattern = np.sin(2*np.pi*8*rel)
    data = (meta.trigger.to_numpy()-238)[:,None,None]*pattern[None,None,:]
    data = np.concatenate([data,2*data],axis=1)
    result = split_half_repeatability(data,data,meta,rel,repeats=3)
    np.testing.assert_allclose(result.waveform_deviation_r,1.)
    assert (result.n_pairs_in_repeatability == 80).all()


def test_electrode_repeatability_is_not_dominated_by_electrode_amplitude():
    meta = trial_table()
    values = (meta.trigger.to_numpy()-238)[:,None,None,None]*np.ones((1,3,4,2))
    values[:,:,:,0] *= 10000
    summary = electrode_repeatability(values,meta,['C3','C4'],repeats=3)
    np.testing.assert_allclose(summary.median_waveform_feature_r,1.)


def test_blanking_matches_supplied_endpoint_convention():
    data = np.arange(100,dtype=float)[None,:]
    data[:,18:30] = 10000
    interpolate_stimuli(data,[20],1000.)
    np.testing.assert_allclose(data[0,18:30],np.linspace(17,30,12))
    assert data[0,30] == 30
