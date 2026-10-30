import numpy as np
from tests.test_e011_electrode_bin_repeatability import metadata
from experiments.response_characterization.e014_all_electrode_repeatability.features import (
    power_features,calibration_splits,fit_spatial,centre_feature_time)
from experiments.response_characterization.e014_all_electrode_repeatability.compute import accumulate_profiles
from experiments.response_characterization.e014_all_electrode_repeatability.data import channel_audit


def test_spectral_units_window_support_scaling_and_dc_invariance():
    t=np.arange(-300,1200)/1000
    x=(2*np.sin(2*np.pi*10*t)+np.sin(2*np.pi*20*t))[None,None]
    powers,left,right,audit=power_features(x,t)
    assert powers['alpha_power'].shape==(1,1,11)
    # Hann spreads the 10-Hz line into 8/12 Hz. Exact 8/13-Hz boundaries
    # truncate part of its main lobe; the beta integral includes its 13-14-Hz
    # interpolated tail. Band power is not the full sinusoid's variance.
    np.testing.assert_allclose(powers['alpha_power'],10*np.log10(43/24),atol=1e-9)
    np.testing.assert_allclose(powers['beta_power'],10*np.log10(13/24),atol=1e-9)
    bigger,_,_,_=power_features(2*x,t)
    shifted,_,_,_=power_features(x+100,t)
    for name in powers:
        np.testing.assert_allclose(bigger[name]-powers[name],20*np.log10(2),atol=1e-9)
        np.testing.assert_allclose(shifted[name],powers[name],atol=1e-9)
    assert left[0]==275 and right[-1]==825
    assert audit['window_starts_ms'][0]==50 and audit['window_ends_ms'][-1]==1050
    assert audit['frequency_spacing_hz']==2.


def test_calibration_and_both_evaluation_halves_are_disjoint_complete_pairs():
    meta=metadata()
    meta=meta[meta.pair_id!='0:4'].reset_index(drop=True)  # one ordered stratum has two pairs
    parts,rows,designs=calibration_splits(meta,3,repeats=5)
    for cal,a,b in parts:
        sets=[set(meta.iloc[ix].pair_id) for ix in [cal,a,b]]
        assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
        assert set.union(*sets)==set(meta.pair_id)
        for ix in [cal,a,b]:assert meta.iloc[ix].groupby('pair_id').size().eq(2).all()
        assert set(meta.iloc[cal].trigger)==set(range(239,244))
        assert not (sets[0]&{'0:0','0:2'})
        assert len(sets[1]&{'0:0','0:2'})==len(sets[2]&{'0:0','0:2'})==1
    assert (designs['rank']==designs['columns']).all()
    again,_,_=calibration_splits(meta,3,repeats=5)
    for p,q in zip(parts,again):
        for a,b in zip(p,q):np.testing.assert_array_equal(a,b)


def test_calibration_only_spatial_normalization_and_public_transform_shape():
    from threadpoolctl import threadpool_limits
    rng=np.random.default_rng(15)
    y=np.repeat(np.arange(239,244),4)
    x=rng.normal(size=(20,6,150))
    with threadpool_limits(limits=1):f,rms,classes=fit_spatial(x,y)
    assert f.shape==(10,6)
    np.testing.assert_allclose(np.sqrt(np.mean((f@x)**2,axis=(0,2))),1.,atol=1e-10)
    np.testing.assert_allclose(rms,np.sqrt(np.mean(x*x,axis=(0,2))))
    np.testing.assert_array_equal(classes,np.arange(239,244))


def test_matched_joint_temporal_metric_is_invariant_to_component_signs():
    rng=np.random.default_rng(7)
    a=rng.normal(size=(4,2,5,6,20));b=a+rng.normal(0,.2,a.shape)
    flip=np.array([1,-1,1,-1,-1,1])[None,None,None,:,None]
    original=accumulate_profiles(iter([(a,b)]))
    changed=accumulate_profiles(iter([(a*flip,b*flip)]))
    np.testing.assert_allclose(original['joint_profile_r'],changed['joint_profile_r'],atol=1e-12)
    np.testing.assert_allclose(original['temporal_profile_r'],changed['temporal_profile_r'],atol=1e-12)
    np.testing.assert_allclose(original['mean_split_product'],changed['mean_split_product'])
    np.testing.assert_allclose(centre_feature_time(a).mean(axis=-1),0,atol=1e-14)


def test_channel_flags_do_not_modify_or_exclude_arrays():
    raw=np.array([np.zeros(2000),np.arange(2000)*1e-6])
    t=np.arange(-300,1200)/1000
    epochs=np.zeros((2,2,1500));epochs[:,1]=np.linspace(-300,300,1500)
    before=epochs.copy()
    audit=channel_audit(raw,epochs,t,['flat','large'])
    assert 'near_flat' in audit.iloc[0]['flags']
    assert len(audit)==2 and 'flags' in audit
    np.testing.assert_array_equal(epochs,before)


def test_explorer_has_variable_channels_spectral_centres_and_quality_notes():
    from experiments.response_characterization.e014_all_electrode_repeatability.report import build_template
    text=build_template()
    script=text.split('<script>')[1].split('</script>')[0]
    assert '*14' not in script and 'c<14' not in script
    assert 'active.channels.length' in script
    assert 'info.centres_ms[j]' in script and 'info.step_ms' in script
    assert 'Quality note' in script
    assert 'calibration' in text and 'spectral' in text
