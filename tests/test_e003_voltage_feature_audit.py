import numpy as np
import pandas as pd
import pytest
from experiments.signal_quality.e003_voltage_feature_audit.core import (
    feature_contract,describe_features,near_constant_columns,pair_bootstrap_indices,audit_training_fold)


def fixture():
    rows = []
    for group,(a,b) in enumerate([(239,240),(240,241),(241,242),(242,243),(239,243)]):
        for pair in range(4):
            for position,code in enumerate((a,b),1):
                rows.append(dict(subject=3,group=str(group),pair_id=f'{group}:{pair}',position=position,trigger=code))
    meta = pd.DataFrame(rows)
    rng = np.random.default_rng(12)
    x = rng.normal(size=(len(meta),3))
    x = np.column_stack([x,x[:,0],np.zeros(len(x))])
    return x,meta


def test_audit_reconstructs_actual_covariance_and_exposes_redundancy():
    x,meta = fixture()
    row,shrink,boot,eig = audit_training_fold(x,x[:4],meta.trigger.to_numpy(),meta,repeats=2)
    assert row['empirical_rank'] < x.shape[1]
    assert row['n_constant'] == 1
    assert row['covariance_reconstruction_max_error'] < 1e-12
    assert len(shrink) == 5
    assert len(boot) == 4
    assert np.isfinite(eig[1]).all()


def test_heldout_extremes_do_not_change_training_scaler_or_covariance():
    x,meta = fixture()
    first = audit_training_fold(x,x[:4],meta.trigger.to_numpy(),meta,repeats=2)
    second = audit_training_fold(x,x[:4]+1e6,meta.trigger.to_numpy(),meta,repeats=2)
    for key in ['n_scaler_fit_samples','empirical_rank','regularized_condition','train_mean_fit_max_error']:
        assert first[0][key] == second[0][key]
    np.testing.assert_allclose(first[3],second[3])
    assert second[0]['heldout_z_max_abs'] > first[0]['heldout_z_max_abs']


def test_resampling_keeps_whole_pairs_and_comparison_support():
    _,meta = fixture()
    sampled = pair_bootstrap_indices(meta,np.random.default_rng(13))
    assert len(sampled) == len(meta)
    for a,b in sampled.reshape(-1,2):
        assert meta.pair_id.iloc[a] == meta.pair_id.iloc[b]
        assert set(meta.position.iloc[[a,b]]) == {1,2}
    assert set(meta.group.iloc[sampled]) == set(meta.group)
    assert set(meta.trigger.iloc[sampled]) == set(meta.trigger)


def test_units_and_bin_order_are_explicit_and_nonfinite_features_fail():
    contract = feature_contract('post',126)
    assert contract[0]['feature'] == 'FC1:50-100ms'
    assert contract[14]['feature'] == 'FC1:100-150ms'
    assert all(row['unit']=='uV' for row in contract)
    with pytest.raises(ValueError,match='contract'):
        feature_contract('post',125)
    with pytest.raises(ValueError,match='nonfinite'):
        describe_features(np.full((3,126),np.nan),contract)


def test_near_constant_check_does_not_confuse_physical_units():
    flags = near_constant_columns(np.array([1.,1e-12,1e-12,0.]),['uV','uV','V','V'])
    assert flags.tolist() == [False,True,False,True]
