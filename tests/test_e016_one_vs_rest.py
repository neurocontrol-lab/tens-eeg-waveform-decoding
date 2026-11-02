import numpy as np
import pytest
from threadpoolctl import threadpool_limits
from experiments.waveform_decoding.e016_one_vs_rest_decoding.core import (
    fit_binary_targets,binary_metrics,cached_spatial_features,selected_matrix)
from experiments.waveform_decoding.e015_position_repeatability_decoding.features import sensor_features,xdawn_features


def test_always_rest_accuracy_is_not_mistaken_for_binary_decoding():
    y=np.r_[np.ones(5,int),np.zeros(95,int)];m=binary_metrics(y,np.zeros(100))
    assert m['accuracy']==m['always_rest_accuracy']==.95
    assert m['balanced_accuracy']==m['roc_auc']==.5
    assert m['sensitivity']==0 and m['specificity']==1
    assert m['average_precision']==.05


def test_single_class_test_fold_has_undefined_auc_and_balanced_accuracy():
    m=binary_metrics(np.zeros(4,int),np.array([.1,.9,.2,.3]))
    assert m['roc_auc'] is None and m['balanced_accuracy'] is None
    assert m['n_positive']==0 and m['n_negative']==4
    with pytest.raises(ValueError):binary_metrics([0,1],[0,np.nan])


def test_within_fold_auc_avoids_cross_model_score_comparisons_and_undefined_folds():
    from experiments.waveform_decoding.e016_one_vs_rest_decoding.report import within_fold_auc
    first=binary_metrics([1,0],[.2,.1])
    second=binary_metrics([1,0],[.9,.8])
    negative_only=binary_metrics([0],[.99])
    result=within_fold_auc([{'test_metrics':m} for m in [first,second,negative_only]])
    assert result['within_comparison_auc']==1
    assert result['evaluable_auc_folds']==2 and result['within_fold_score_pairs']==2
    assert binary_metrics([1,0,1,0,0],[.2,.1,.9,.8,.99])['roc_auc'] < 1
    unequal=within_fold_auc([{'test_metrics':first},
                            {'test_metrics':binary_metrics([1,1,0,0],[0,0,1,1])}])
    assert unequal['within_comparison_auc']==.2


def test_missing_training_targets_use_flagged_constants_without_fake_lda():
    rng=np.random.default_rng(4);train=rng.normal(size=(12,4));test=rng.normal(size=(3,4))
    _,fits=fit_binary_targets(train,test,np.full(12,239))
    assert fits[239]['kind']=='constant_positive' and fits[239]['estimator'] is None
    np.testing.assert_array_equal(fits[239]['probability'],1)
    assert fits[240]['kind']=='constant_negative' and fits[240]['training_positive']==0
    assert 'target_absent_from_training' in fits[240]['flags']
    np.testing.assert_array_equal(fits[240]['probability'],0)


def test_held_out_signal_changes_do_not_change_binary_model_or_scaler():
    rng=np.random.default_rng(19);y=np.tile(np.arange(239,244),12)
    train=rng.normal(size=(60,12))+rng.normal(size=(5,12))[y-239];test=rng.normal(size=(4,12))
    with threadpool_limits(limits=1):
        sa,a=fit_binary_targets(train,test,y)
        sb,b=fit_binary_targets(train,test*1000,y)
    np.testing.assert_array_equal(sa.mean_,sb.mean_);np.testing.assert_array_equal(sa.scale_,sb.scale_)
    for c in range(239,244):
        np.testing.assert_array_equal(a[c]['estimator'].coef_,b[c]['estimator'].coef_)
        np.testing.assert_array_equal(a[c]['estimator'].priors_,[.5,.5])
        assert a[c]['training_positive']==12 and a[c]['training_negative']==48


def test_cached_voltage_projection_reproduces_original_epoch_spatial_features():
    rng=np.random.default_rng(21);times=np.arange(-300,1200)/1000
    epochs=rng.normal(size=(4,2,len(times)));classes=np.arange(239,244);filters=rng.normal(size=(10,2))
    sensor,contract=sensor_features(epochs,times,['C3','C4'])
    names=[f'{int(c)}:component-{i+1}' for c in classes for i in range(2)]
    original,original_contract=xdawn_features(epochs,times,filters,names)
    cached,cached_contract=cached_spatial_features(sensor,contract,filters,classes)
    np.testing.assert_allclose(cached,original,atol=1e-14);assert cached_contract==original_contract
    all_contract=contract+original_contract;chosen=np.array([0,793,794,800,1540])
    picked=[all_contract[i] for i in chosen]
    result=selected_matrix(sensor,contract,filters,classes,chosen,picked)
    np.testing.assert_allclose(result,np.column_stack([sensor,original])[:,chosen],atol=1e-14)
    wrong=[dict(r) for r in picked];wrong[0]['feature']='wrong-column'
    with pytest.raises(ValueError,match='feature IDs'):selected_matrix(sensor,contract,filters,classes,chosen,wrong)
