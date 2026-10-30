import base64
import numpy as np
from tests.test_e011_electrode_bin_repeatability import metadata
from experiments.response_characterization.e011_electrode_bin_repeatability.core import (
    split_operators,waveform_operator,repeatability)
from experiments.response_characterization.e011_electrode_bin_repeatability.waveform_views import (
    observed_weights,total_split_operators,pattern_r,individual_estimates,encode)


def test_waveform_products_decompose_original_pooled_estimator():
    meta = metadata()
    rng = np.random.default_rng(91)
    features = rng.normal(size=(len(meta),2,10))+(meta.trigger.to_numpy()-241)[:,None,None]
    operators,_,_ = split_operators(meta,3,repeats=6)
    full,_,_ = waveform_operator(meta)
    result = individual_estimates(features,operators,full,np.ones(10,bool),specific=True)
    previous = repeatability(features,operators)
    np.testing.assert_allclose(result['mean_split_product'].mean(axis=1),previous['mean_split_product'],atol=1e-12)
    assert result['mean_split_product'].shape == (2,5,2,10)
    assert result['temporal_profile_r'].shape == (2,5,2)
    assert result['joint_profile_r'].shape == (2,5)


def test_total_repeatability_can_be_shared_with_no_specific_waveform_effect():
    meta = metadata()
    t = np.linspace(0,2*np.pi,20)
    common = np.stack([np.sin(t),2*np.sin(t)])
    features = np.broadcast_to(common,(len(meta),2,20)).copy()
    specific,assignments,_ = split_operators(meta,3,repeats=4)
    total = total_split_operators(meta,assignments,4)
    full_specific,_,_ = waveform_operator(meta)
    full_total = observed_weights(meta)
    mask = np.ones(20,bool)
    total_result = individual_estimates(features,total,full_total,mask,specific=False)
    specific_result = individual_estimates(features,specific,full_specific,mask,specific=True)
    np.testing.assert_allclose(total_result['joint_profile_r'],1.,atol=1e-12)
    assert np.max(np.abs(specific_result['mean_split_product'])) < 1e-25
    np.testing.assert_allclose(total.sum(axis=-1),1.)


def test_one_waveform_is_not_averaged_away_and_contrasts_are_relative():
    meta = metadata()
    t = np.linspace(0,2*np.pi,20)
    features = (meta.trigger.to_numpy()==239)[:,None,None]*np.sin(t)[None,None]
    _,assignments,_ = split_operators(meta,3,repeats=4)
    total = total_split_operators(meta,assignments,4)
    result = individual_estimates(features,total,observed_weights(meta),np.ones(20,bool),specific=False)
    assert np.all(result['mean_split_product'][:,0].sum(axis=-1) > 1)
    np.testing.assert_array_equal(result['mean_split_product'][:,1:],0.)
    np.testing.assert_allclose(result['joint_profile_r'][:,0],1.)
    assert np.isnan(result['joint_profile_r'][:,1:]).all()


def test_temporal_correlation_uses_time_not_waveform_identity_or_split_index():
    a = np.array([[1.,2.,4.,8.],[8.,4.,2.,1.]])
    b = np.array([[2.,4.,8.,16.],[-8.,-4.,-2.,-1.]])
    np.testing.assert_allclose(pattern_r(a,b),[1.,-1.])
    assert np.isnan(pattern_r(np.ones((1,4)),np.ones((1,4)))).all()


def test_browser_payload_retains_signed_float32_values():
    values = np.array([-2.5,0.,3.25,np.nan])
    restored = np.frombuffer(base64.b64decode(encode(values)),dtype='<f4')
    np.testing.assert_allclose(restored,values,equal_nan=True)
