"""Onset guards, five-condition OLS, and exact joint-vector max statistics."""
import itertools
import numpy as np
import pandas as pd
import pytest
from experiments.response_characterization.e008_time_resolved_effects.analysis import (
    specifications, interval_masks, regression_operator, fit_trace,
    sign_flip_family, family_adjustment)
from experiments.response_characterization.e006_multiscale_responses.analysis import transform
from experiments.response_characterization.e002_electrode_position_slope.analysis import effect_design, t_squared


def metadata():
    rows = []
    for c,(a,b) in enumerate(itertools.combinations(range(239,244),2)):
        for trial in range(6):
            for position,code in enumerate((a,b) if trial % 2 else (b,a),1):
                rows.append(dict(trigger=code,position=position,comparison=f'{a}:{b}',
                                 elapsed_s=c*100+trial*3+position))
    return pd.DataFrame(rows)


def test_all_five_scales_have_actual_padding_and_equal_guarded_periods():
    times = np.arange(-900,651)/1000
    masks = interval_masks(times,1000.)
    assert [m.sum() for m in masks.values()] == [300,300]
    assert [s['sigma_ms'] for s in specifications()] == [2.5,5.,10.,20.,40.]
    with pytest.raises(ValueError,match='padding'):
        interval_masks(np.arange(-600,601)/1000,1000.)


def test_new_scales_preserve_ramp_units_and_cannot_see_an_onset_impulse():
    times = np.arange(-900,651)/1000
    masks = interval_masks(times,1000.)
    impulse = np.zeros((1,1,len(times)))
    impulse[:,:,times == 0] = 1e6
    ramp = 7*times[None,None,:]+20
    for spec in specifications():
        y = transform(impulse,1000.,spec)
        slope = transform(ramp,1000.,spec)
        for mask in masks.values():
            np.testing.assert_allclose(y[:,:,mask],0.,atol=1e-12)
            np.testing.assert_allclose(slope[:,:,mask],7.,atol=1e-10)


def test_context_adjusted_ols_recovers_all_four_waveform_and_interaction_coefficients():
    meta = metadata()
    design,slices = effect_design(meta)
    rng = np.random.default_rng(41)
    beta = rng.normal(size=(design.shape[1],3,7))
    data = (design @ beta.reshape(len(beta),-1)).reshape(len(meta),3,7)
    operator,actual_slices,audit = regression_operator(meta)
    fitted = fit_trace(data,operator,actual_slices)
    assert audit['rank'] == design.shape[1]
    for family,sl in slices.items():
        np.testing.assert_allclose(fitted[family],beta[sl].transpose(0,2,1),atol=1e-12)
    assert fitted['waveform'].shape[0] == fitted['interaction'].shape[0] == 4
    with pytest.raises(ValueError,match='identifiable'):
        regression_operator(meta[meta.trigger != 243].reset_index(drop=True))


def test_chunked_omnibus_matches_brute_force_joint_sign_flips():
    rng = np.random.default_rng(7)
    coefs = rng.normal(size=(5,4,2,3))+.2
    observed = t_squared(coefs)
    null = np.stack([t_squared(coefs*np.asarray(sign)[:,None,None,None])
                     for sign in itertools.product((-1.,1.),repeat=5)])
    expected_raw = np.mean(null >= observed[None]-1e-10,axis=0)
    maxima = null.reshape(len(null),-1).max(axis=1)
    expected_max = np.mean(maxima[:,None,None] >= observed[None]-1e-10,axis=0)
    for chunk in (1,4,100):
        result = sign_flip_family(coefs,chunk_size=chunk)
        np.testing.assert_allclose(result['statistic'],observed)
        np.testing.assert_array_equal(result['p_uncorrected'],expected_raw)
        np.testing.assert_array_equal(result['p_max_family'],expected_max)
        np.testing.assert_allclose(result['null_maxima'],maxima,atol=1e-10)


def test_exact_null_and_planted_consistent_group_effect():
    zero = sign_flip_family(np.zeros((8,4,2,3)))
    np.testing.assert_array_equal(zero['p_uncorrected'],1.)
    np.testing.assert_array_equal(zero['p_max_family'],1.)
    rng = np.random.default_rng(8)
    x = rng.normal(0,.05,size=(8,4,2,3))
    x[:,0,1,2] += 2.
    result = sign_flip_family(x,chunk_size=2)
    assert family_adjustment(result)[1,2] < .05
    assert np.all(result['p_max_family'] >= result['p_uncorrected'])


def test_global_contrast_sign_reversal_preserves_two_sided_results():
    rng = np.random.default_rng(9)
    x = rng.normal(size=(6,4,5))
    a,b = sign_flip_family(x),sign_flip_family(-x)
    for key in ('statistic','p_uncorrected','p_max_family'):
        np.testing.assert_array_equal(a[key],b[key])
    with pytest.raises(ValueError,match='finite effects'):
        sign_flip_family(np.full((5,4,2),np.nan))


def test_complete_small_report_preserves_one_event_count_per_participant(tmp_path,monkeypatch):
    import json
    from experiments.response_characterization.e008_time_resolved_effects import report
    rng = np.random.default_rng(90)
    (tmp_path/'config.json').write_text(json.dumps(dict(subjects=[3,4,5],implementation={})))
    (tmp_path/'status.json').write_text(json.dumps(dict(completed=[3,4,5],errors={})))
    for subject in (3,4,5):
        destination = tmp_path/f'sub-{subject}'
        destination.mkdir()
        values = {key:rng.normal(size=(k,2,2,5,3,2))
                  for key,k in (('waveform',4),('position',1),('interaction',4))}
        np.savez_compressed(destination/'effects.npz',**values,channels=['C3','C4'],
            plot_channels=['C3','C4'],pre_times=[-.3,-.299,-.298],post_times=[.3,.301,.302],
            evoked_display=rng.normal(size=(2,2,5,5,2,2,3)))
        audit = dict(n_events=100,design=dict(n_events=100,rank=20,columns=20,condition_number=6.),
                     prior_feature_max_error=0.,n_padding_exclusions=0,ica_refitted=False)
        (destination/'audit.json').write_text(json.dumps(audit))
    # Numerical inference and all report tables run; figures are inspected separately.
    monkeypatch.setattr(report,'plot_p_map',lambda *args:None)
    monkeypatch.setattr(report,'plot_evoked',lambda *args:None)
    report.render(tmp_path)
    audit = pd.read_csv(tmp_path/'participant_audit.csv')
    assert audit.n_events.sum() == 300
    assert len(pd.read_csv(tmp_path/'test_summary.csv')) == 60
    assert (tmp_path/'group_tests.npz').exists()
    assert (tmp_path/'render_provenance.json').exists()
    assert '300 frozen matched events' in (tmp_path/'RESULTS.md').read_text(encoding='utf-8')
