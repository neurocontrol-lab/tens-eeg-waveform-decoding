import numpy as np
import pytest
from tests.test_e011_electrode_bin_repeatability import metadata
from experiments.waveform_decoding.e015_position_repeatability_decoding.features import sensor_features,xdawn_features
from experiments.waveform_decoding.e015_position_repeatability_decoding.design import (
    position_operator,calibration_and_halves,filter_guard,outer_splits)
from experiments.waveform_decoding.e015_position_repeatability_decoding.scoring import repeatability_scores,select_columns


def test_position_adjustment_recovers_known_waveform_effects_without_redundant_terms():
    meta=metadata();meta=meta[meta.position==1].reset_index(drop=True)
    op,audit=position_operator(meta)
    truth=np.array([-2.,-1.,0.,1.,2.])
    context=meta.comparison.astype('category').cat.codes.to_numpy()*4.
    elapsed=(meta.elapsed_s-meta.groupby('comparison').elapsed_s.transform('mean')).to_numpy()
    signal=truth[meta.trigger.to_numpy()-239]+context+.17*elapsed
    np.testing.assert_allclose(op@signal,truth,atol=1e-11)
    assert audit['rank']==audit['columns']==15
    with pytest.raises(ValueError,match='one position'):position_operator(metadata())


def test_calibration_is_independent_of_both_adjusted_ranking_halves():
    meta=metadata();meta=meta[meta.position==2].reset_index(drop=True)
    cal,ops,audits,info=calibration_and_halves(meta,99,repeats=4)
    assert set(meta.iloc[cal].trigger)==set(range(239,244))
    assert np.all(ops[...,cal]==0)
    for a,b in ops:
        assert not (np.any(abs(a)>1e-12,axis=0)&np.any(abs(b)>1e-12,axis=0)).any()
    assert all(x['rank']==x['columns'] and x['n']>x['rank'] for x in audits)
    again=calibration_and_halves(meta,99,repeats=4)
    np.testing.assert_array_equal(cal,again[0]);np.testing.assert_allclose(ops,again[1])


def test_rare_class_cannot_be_shared_between_calibration_and_two_halves():
    meta=metadata();meta=meta[meta.position==1].reset_index(drop=True)
    rare=np.flatnonzero(meta.trigger.to_numpy()==239)
    meta=meta.drop(rare[2:]).reset_index(drop=True)
    with pytest.raises(ValueError,match='No independent'):calibration_and_halves(meta,1,repeats=2)


def test_label_free_feature_contract_and_fixed_projection_commute_with_binning():
    rng=np.random.default_rng(17);times=np.arange(-300,1200)/1000
    epochs=rng.normal(size=(3,2,len(times)))
    x,contract=sensor_features(epochs,times,['C3','C4'])
    assert x.shape==(3,794) and len({r['feature'] for r in contract})==794
    first=contract[0];assert first['channel']=='C3' and first['start_ms']==50 and first['end_ms']==75
    np.testing.assert_allclose(x[:,0],epochs[:,0,350:375].mean(axis=-1))
    assert sum(r['family']=='alpha_power' for r in contract)==22
    filters=np.array([[1.,2.],[-1.,.5]])
    a,ca=xdawn_features(epochs,times,filters,['a','b'])
    b,cb=xdawn_features(filters@epochs,times,np.eye(2),['a','b'])
    np.testing.assert_allclose(a,b,atol=1e-14);assert ca==cb and a.shape==(3,150)


def test_effective_filter_guard_and_pair_embargo_are_explicit():
    guard=filter_guard();assert guard['measured_l1_tail_fraction']<=1e-4
    assert guard['minimum_trigger_separation_s']==2*guard['one_sided_seconds']+1.312
    meta=metadata();meta['group']=meta.comparison
    for fold in outer_splits(meta,guard['minimum_trigger_separation_s']):
        a,b=meta.iloc[fold['train']],meta.iloc[fold['test']]
        assert not set(a.pair_id)&set(b.pair_id) and not set(a.group)&set(b.group)
        if len(a):assert np.min(abs(a.elapsed_s.to_numpy()[:,None]-b.elapsed_s.to_numpy()[None,:]))>guard['minimum_trigger_separation_s']


def test_negative_agreement_is_not_rewarded_and_calibration_does_not_enter_score():
    # Five class observations in each half, plus one independent calibration row.
    a=np.zeros((5,11));b=a.copy()
    for c in range(5):a[c,c]=1;b[c,5+c]=1
    operators=np.stack([a,b])[None]
    pattern=np.array([-2.,-1.,0.,1.,2.])
    x=np.column_stack([np.r_[pattern,pattern,0],np.r_[pattern,-pattern,0],np.zeros(11)])
    r,s,count=repeatability_scores(x,operators,np.arange(10))
    np.testing.assert_allclose(r[:2],[1,-1]);np.testing.assert_allclose(s[:2],[1,0])
    assert np.isnan(s[2]);np.testing.assert_array_equal(select_columns(s,100),[0])
    x[10]=1e20
    r2,s2,_=repeatability_scores(x,operators,np.arange(10))
    np.testing.assert_allclose(r,r2,equal_nan=True);np.testing.assert_allclose(s,s2,equal_nan=True)


def test_held_out_values_cannot_change_calibration_selection_scaling_or_model(monkeypatch):
    from experiments.waveform_decoding.e015_position_repeatability_decoding import scoring
    from threadpoolctl import threadpool_limits
    rng=np.random.default_rng(121);meta=metadata();meta=meta[meta.position==1].reset_index(drop=True)
    times=np.arange(-300,1200)/1000;labels=meta.trigger.to_numpy()-239
    epochs=rng.normal(size=(len(meta),2,len(times)))+labels[:,None,None]*.2
    sensor=rng.normal(size=(len(meta),8))+rng.normal(size=(5,8))[labels]
    test_sensor=rng.normal(size=(3,8));test_epochs=rng.normal(size=(3,2,len(times)))
    filters=rng.normal(size=(10,2));calls=[]
    def fit_only_calibration(x,y):
        calls.append((x.copy(),y.copy()));return filters.copy(),np.ones(2),np.arange(239,244)
    monkeypatch.setattr(scoring,'fit_spatial',fit_only_calibration)
    contract=[dict(feature=f'synthetic-{i}',family='voltage_mean') for i in range(8)]
    with threadpool_limits(limits=1):
        a=scoring.fit_fold(sensor,test_sensor,epochs,test_epochs,times,meta,contract,7,k=5,repeats=3)
        b=scoring.fit_fold(sensor,test_sensor*1000,epochs,test_epochs*1000,times,meta,contract,7,k=5,repeats=3)
    mask=(times>=.05)&(times<1.05)
    np.testing.assert_array_equal(calls[0][0],epochs[a['calibration_indices']][...,mask])
    assert not set(a['calibration_indices'])&set(a['ranking_indices'])
    for family in ['repeatability','anova']:
        aa,bb=a['models'][family],b['models'][family]
        np.testing.assert_array_equal(aa['selected'],bb['selected'])
        np.testing.assert_allclose(aa['estimator'][0].mean_,bb['estimator'][0].mean_)
        np.testing.assert_allclose(aa['estimator'][1].coef_,bb['estimator'][1].coef_)


def test_exploratory_missing_class_is_flagged_not_invented(monkeypatch):
    from experiments.waveform_decoding.e015_position_repeatability_decoding import exploratory
    from threadpoolctl import threadpool_limits
    rng=np.random.default_rng(91);meta=metadata();meta=meta[(meta.position==1)&(meta.trigger!=240)].reset_index(drop=True)
    times=np.arange(-300,1200)/1000;labels=meta.trigger.to_numpy()-239;classes=np.unique(meta.trigger)
    epochs=rng.normal(size=(len(meta),2,len(times)))+labels[:,None,None]*.3
    sensor=rng.normal(size=(len(meta),8))+rng.normal(size=(5,8))[labels]
    filters=rng.normal(size=(8,2));calls=[]
    def fit_train(x,y):calls.append(x.copy());return filters.copy(),np.unique(y)
    monkeypatch.setattr(exploratory,'spatial_fit',fit_train)
    contract=[dict(feature=f'f{i}',family='voltage_mean') for i in range(8)]
    test=rng.normal(size=(2,8));test_epochs=rng.normal(size=(2,2,len(times)))
    with threadpool_limits(limits=1):
        a=exploratory.fit_fold(sensor,test,epochs,test_epochs,times,meta,contract,8,k=5,repeats=3)
        b=exploratory.fit_fold(sensor,test*1000,epochs,test_epochs*1000,times,meta,contract,8,k=5,repeats=3)
    assert a['missing_training_classes']==[240] and 'missing_training_class' in a['flags']
    assert a['training_classes']==classes.tolist() and a['candidate_count']==8+600
    np.testing.assert_array_equal(calls[0],epochs[...,(times>=.05)&(times<1.05)])
    for name in ['repeatability','anova']:
        np.testing.assert_array_equal(a['models'][name]['selected'],b['models'][name]['selected'])
        np.testing.assert_allclose(a['models'][name]['estimator'][1].coef_,b['models'][name]['estimator'][1].coef_)
        assert np.all(a['models'][name]['probabilities'][:,1]==0)


def test_exploratory_halves_and_conditional_label_shuffle_preserve_their_contracts():
    from experiments.waveform_decoding.e015_position_repeatability_decoding.exploratory import observed_operators,permute_pair_labels
    m=metadata();m['group']=m.comparison
    p=m[m.position==2].reset_index(drop=True);ops,classes,rank=observed_operators(p,7,repeats=4)
    assert ops.shape==(4,2,5,len(p)) and len(rank)==len(p)
    np.testing.assert_allclose(ops.sum(axis=-1),1.)
    for a,b in ops:assert not (np.any(a>0,axis=0)&np.any(b>0,axis=0)).any()
    shuffled=permute_pair_labels(m,np.random.default_rng(12))
    for _,g in m.groupby('group'):
        for position in [1,2]:
            ix=g.index[g.position==position]
            np.testing.assert_array_equal(np.sort(shuffled[ix]),np.sort(m.loc[ix,'trigger']))
    for _,g in m.groupby('pair_id'):assert set(shuffled[g.index])==set(g.trigger)
