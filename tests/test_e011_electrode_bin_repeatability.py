import numpy as np
import pandas as pd
import pytest
from experiments.response_characterization.e011_electrode_bin_repeatability.core import (
    WIDTHS_MS,local_representations,bin_features,eligible_split_pairs,waveform_operator,
    split_operators,repeatability,group_interval)


def metadata():
    import itertools
    rows = []
    ordinal = 0
    for context,(a,b) in enumerate(itertools.combinations(range(239,244),2)):
        for trial in range(6):
            for position,code in enumerate((a,b) if trial%2 else (b,a),1):
                rows.append(dict(pair_id=f'{context}:{trial}',comparison=f'{a}-{b}',position=position,
                    trigger=code,elapsed_s=ordinal+position*.7))
            ordinal += 5
    return pd.DataFrame(rows)


def test_derivative_units_and_quadratic_recovery_on_real_support():
    t = np.arange(-300,1200)/1000
    x = (3+2*t+4*t*t)[None,None]
    values = local_representations(x,t,1000.)
    mask = (t>=.05)&(t<1.188)
    np.testing.assert_allclose(values['slope'][0,0,mask],2+8*t[mask],atol=1e-10)
    np.testing.assert_allclose(values['curvature'][0,0,mask],8,atol=1e-8)
    shifted = local_representations(x+100.,t,1000.)
    np.testing.assert_allclose(shifted['slope'],values['slope'],atol=1e-9)
    np.testing.assert_allclose(shifted['curvature'],values['curvature'],atol=1e-7)
    with pytest.raises(ValueError,match='real derivative support'):
        local_representations(x[...,:-10],t[:-10],1000.)


def test_bin_anchors_complete_support_and_rms_before_averaging():
    t = np.arange(-300,1200)/1000
    x = np.tile([-2.,-1.,0.,1.,2.],300)[None,None]
    signed,left,right = bin_features(x,t,5)
    rms,_,_ = bin_features(x,t,5,rms=True)
    np.testing.assert_allclose(signed,0)
    np.testing.assert_allclose(rms,np.sqrt(2))
    ends = (1188,1185,1175,1150,1150,1050)
    for width,end in zip(WIDTHS_MS,ends):
        values,left,right = bin_features(x,t,width)
        assert left[0] == 50 and right[-1] == end
        assert (right-left == width).all()
        assert values.shape[-1] == (1188-50)//width
    with pytest.raises(ValueError): bin_features(x,t,7)


def test_pairs_never_cross_halves_and_splits_are_deterministic():
    meta = metadata()
    operators,rows,audit = split_operators(meta,3,repeats=4)
    assert np.isfinite(operators).all() and (audit['rank']==audit['columns']).all()
    for _,group in rows.groupby('repeat'):
        assert group.pair_id.nunique() == len(group) == meta.pair_id.nunique()
        a,b = [set(g.pair_id) for _,g in group.groupby('half')]
        assert not a & b and a | b == set(meta.pair_id)
    again,_,_ = split_operators(meta,3,repeats=4)
    np.testing.assert_array_equal(operators,again)
    np.testing.assert_allclose(operators.sum(axis=3),0,atol=1e-14)


def test_singleton_ordered_strata_are_explicit_and_whole_pair():
    meta = metadata()
    # Remove two pairs of one orientation, leaving one paired presentation.
    meta = meta[~meta.pair_id.isin(['0:2','0:4'])].reset_index(drop=True)
    _,used = eligible_split_pairs(meta)
    assert '0:0' not in used and '0:1' in used
    with pytest.raises(ValueError,match='unsplittable'):
        split_operators(meta,3,repeats=2)


def test_adjustment_recovers_waveform_profiles_despite_context_and_time():
    meta = metadata()
    wave = np.array([-2.,-1.,0.,1.,2.])
    interaction = np.array([.2,.1,0.,-.1,-.2])
    y = wave[meta.trigger-239]+(meta.position-1.5)*interaction[meta.trigger-239]
    y += 1000*pd.factorize(meta.comparison)[0]+.3*meta.elapsed_s
    op,_,audit = waveform_operator(meta)
    estimate = op@y.to_numpy()
    np.testing.assert_allclose(estimate,np.stack([wave-.5*interaction,wave+.5*interaction]),atol=1e-10)
    assert audit['rank'] == audit['columns']


def test_repeatability_scale_invariance_and_squared_physical_units():
    meta = metadata()
    rng = np.random.default_rng(42)
    features = (meta.trigger.to_numpy()-241)[:,None,None]+rng.normal(0,.1,(len(meta),2,7))
    operators,_,_ = split_operators(meta,3,repeats=8)
    original = repeatability(features,operators)
    scaled = repeatability(7*features,operators)
    np.testing.assert_allclose(scaled['median_split_r'],original['median_split_r'],atol=1e-12)
    np.testing.assert_allclose(scaled['mean_split_product'],49*original['mean_split_product'],atol=1e-10)
    assert original['median_split_r'].min() > .95
    assert (original['mean_split_product'] > 1).all()


def test_participant_bootstrap_not_split_count_and_missing_values():
    values = np.array([[1.,np.nan],[2.,4.],[3.,6.]])
    draws = np.array([[0,0,0],[2,2,2],[1,1,1],[0,1,2]])
    mean,lo,hi,n = group_interval(values,draws)
    np.testing.assert_allclose(mean,[2.,5.])
    np.testing.assert_array_equal(n,[3,2])
    assert (lo<=mean).all() and (hi>=mean).all()


def test_cross_product_can_be_negative_instead_of_clipped_to_zero():
    meta = metadata()
    operators,rows,_ = split_operators(meta,3,repeats=1)
    left = set(rows[rows.half==0].pair_id)
    y = (meta.trigger.to_numpy()-241)*np.where(meta.pair_id.isin(left),1.,-1.)
    result = repeatability(y[:,None,None],operators)
    assert (result['mean_split_product'] < -1.).all()
    np.testing.assert_allclose(result['median_split_r'],-1.,atol=1e-12)
