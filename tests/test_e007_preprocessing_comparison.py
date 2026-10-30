"""Matched baselines, onset guards, interval lengths and paired inference."""
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from experiments.signal_quality.e007_preprocessing_comparison.analysis import (
    epoch_input,pre_values,interval_repeatability,signal_health,paired_contrasts,
    mask_for,INTERVALS,BASELINES,source_digest,choose_verified_fit)
from experiments.response_characterization.e006_multiscale_responses.analysis import specifications,transform,split_weights


def table():
    rows = []
    for comp,(a,b) in enumerate(itertools.combinations(range(239,244),2)):
        for trial in range(4):
            for p,code in enumerate((a,b) if trial % 2 else (b,a),1):
                rows.append(dict(trigger=code,position=p,comparison=f'{a}-{b}',pair_id=f'{comp}:{trial}'))
    return pd.DataFrame(rows)


def test_prior_source_paths_accept_relative_or_absolute_but_not_ambiguous_entries():
    target = 'experiments/response_characterization/e006_multiscale_responses/analysis.py'
    assert source_digest({target:'abc'},target) == 'abc'
    assert source_digest({'C:\\project\\experiments\\task4_multiscale\\analysis.py':'abc'},target) == 'abc'
    with pytest.raises(ValueError,match='unambiguous'):
        source_digest({target:'abc','C:/project/'+target:'def'},target)


def test_byte_identical_ica_candidates_prefer_the_recorded_cache():
    old = Path('cache/sub-3-old/cleaning-ica.fif')
    current = Path('cache/sub-3-current/cleaning-ica.fif')
    assert choose_verified_fit([old,current],'sub-3-current') == current
    assert choose_verified_fit([old,current],'relocated') == sorted([old,current])[0]
    with pytest.raises(ValueError,match='No original ICA'):
        choose_verified_fit([],'sub-3-current')


def test_minimal_reference_and_independent_pre_baseline():
    fs = 1000.
    timestamps = np.arange(5000)/fs
    data = np.array([2*timestamps,-3*timestamps])*1e-6
    events = pd.DataFrame(dict(sample=[1000,2400],previous_gap_s=[np.nan,1.4],next_gap_s=[1.4,np.nan]))
    x,t,delta = epoch_input(data,timestamps,events,fs,average_reference=True)
    np.testing.assert_allclose(x.mean(axis=1),0.,atol=1e-12)
    np.testing.assert_allclose(x[:,:,mask_for(t,BASELINES['post'])].mean(axis=-1),0.,atol=1e-12)
    pre = pre_values(x,delta,dict(operator='voltage'))
    np.testing.assert_allclose(pre[:,:,mask_for(t,BASELINES['pre'])].mean(axis=-1),0.,atol=1e-12)
    assert mask_for(t,INTERVALS['pre']).sum() == mask_for(t,INTERVALS['post']).sum() == 400
    np.testing.assert_allclose(np.asarray(BASELINES['pre'])+.6,BASELINES['post'])


def test_baseline_translation_commutes_with_each_representation():
    t = np.arange(-900,651)/1000
    x = np.sin(2*np.pi*8*t)[None,None,:]+np.array([0.,5.])[:,None,None]
    delta = np.array([2.,-3.])[:,None,None]
    for spec in specifications():
        explicit = transform(x-delta,1000.,spec)
        shortcut = pre_values(transform(x,1000.,spec),delta,spec)
        np.testing.assert_allclose(shortcut,explicit,atol=1e-9)


def test_local_features_cannot_reach_onset_from_either_analysis_interval():
    t = np.arange(-900,651)/1000
    x = np.zeros((1,1,len(t)))
    x[:,:,t == 0] = 1e6
    for spec in specifications():
        y = transform(x,1000.,spec)
        for interval in INTERVALS.values():
            np.testing.assert_allclose(y[:,:,mask_for(t,interval)],0.,atol=1e-10)


def test_repeatability_preserves_matched_pre_and_post_sample_counts():
    meta = table()
    t = np.arange(-900,651)/1000
    x = (meta.trigger.to_numpy()-238)[:,None,None]*np.sin(2*np.pi*5*t)[None,None]
    weights,_ = split_weights(meta,repeats=4)
    results = []
    for period in INTERVALS:
        r,_ = interval_repeatability(x,t,weights,['C3'],'test',period,1000.)
        results.append(r)
        np.testing.assert_allclose(r.waveform_deviation_r,1.)
    np.testing.assert_allclose(results[0].waveform_deviation_r,results[1].waveform_deviation_r)


def test_signal_checks_flag_contamination_without_removing_events():
    t = np.arange(-900,651)/1000
    x = np.stack([np.sin(2*np.pi*60*t),1000*np.sin(2*np.pi*60*t)])[:,None]
    h = signal_health(x,t,1000.,'post')
    assert len(h) == 2
    assert h.exceeds_500uv.tolist() == [False,True]
    assert (h.line60_fraction > .6).all()


def test_paired_tests_correct_all_contrasts_and_reject_missing_pairs():
    rows = []
    for subject in range(8):
        for rep in ('a','b'):
            for position in (1,2):
                for branch in ('minimal','cleaned'):
                    for period in ('pre','post'):
                        r = .1
                        if branch == 'minimal' and period == 'post':
                            r += .4+subject*.001
                        rows.append(dict(subject=subject,representation=rep,position=position,
                            branch=branch,period=period,waveform_deviation_r=r))
    frame = pd.DataFrame(rows)
    _,tests = paired_contrasts(frame)
    assert len(tests) == 12
    assert (tests.n_joint_tests == 12).all()
    assert (tests.query("contrast == 'post_minimal_minus_cleaned'").p_fwer < .05).all()
    assert (tests.query("contrast == 'pre_minimal_minus_cleaned'").p_fwer == 1).all()
    assert (tests.p_fwer >= tests.p_uncorrected).all()
    with pytest.raises(ValueError,match='Every participant'):
        paired_contrasts(frame.iloc[1:])
