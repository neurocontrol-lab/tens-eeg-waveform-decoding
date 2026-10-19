import numpy as np
import pandas as pd
import pytest

from experiments.task4_baseline.events import align_events, trigger_runs
from experiments.task4_baseline.features import CHANNELS, build_features
from experiments.task4_baseline.evaluate import make_splits, permute_pairs, predict_oof, scores


def behavior(pairs, choices=None):
    n = len(pairs)
    return pd.DataFrame({'t4 waveform 1':[a-238 for a,b in pairs],
                         't4 waveform 2':[b-238 for a,b in pairs],
                         'trial_id':range(1,n+1), 'csv_row':range(2,n+2),
                         't4 naturalness value':choices or [1]*n,
                         't4 waveform detected':[0]*n})


def events_for(pairs):
    codes = np.asarray(pairs).ravel()
    values = np.zeros(2000+2000*len(codes))
    for i, code in enumerate(codes):
        values[1000+2000*i:1008+2000*i] = code
    runs = trigger_runs(values, 1000)
    return align_events(runs, behavior(pairs), 3, np.arange(len(values))/1000)


def test_debounce_keeps_onset_and_audits_active_low_glitches():
    values = np.r_[np.zeros(10), [242], np.zeros(2), [243]*8, [65535]*3, [65535-239]*4, [111]*5]
    runs = trigger_runs(values, 1000, 3)
    assert runs[runs.accepted]['sample'].tolist() == [13,24]
    assert runs[runs.accepted].trigger.tolist() == [243,239]
    assert runs[runs.accepted].inverted.tolist() == [False,True]
    assert set(runs[~runs.accepted].reason) == {'short_run','unexpected_code'}


def test_missing_event_fails_instead_of_shifting_behavior_labels():
    runs = trigger_runs(np.r_[np.zeros(10),[239]*8,np.zeros(20),[241]*8],1000)
    with pytest.raises(ValueError, match='Strict CSV alignment failed'):
        align_events(runs, behavior([(239,240)]),3,np.arange(46)/1000)


def test_legacy_zeros_do_not_mark_current_trials_failed():
    e = events_for([(239,240),(241,242)])
    assert e.response_valid.all()
    assert (e.legacy_detected_value == 0).all()
    values = np.r_[np.zeros(10),[239]*8,np.zeros(20),[240]*8]
    e = align_events(trigger_runs(values,1000),behavior([(239,240)],[0]),3,np.arange(len(values))/1000)
    assert not e.response_valid.any()


def test_post_stimulus_signal_cannot_enter_pre_features():
    e = events_for([(239,240)])
    n = 6000
    data = np.zeros((len(CHANNELS),n))
    # Post-onset signal only on one selected channel, no acausal filtering.
    for sample in e['sample']:
        data[0,sample+50:sample+500] = 10e-6
    features, meta = build_features(data,list(CHANNELS),e,np.arange(n)/1000,1000)
    assert np.allclose(features['pre'],0)
    assert np.max(np.abs(features['post'])) > 1
    assert meta.analysis_eligible.all()


def test_bad_eeg_drops_whole_pair_but_retains_manifest_rows():
    e = events_for([(239,240)])
    data = np.zeros((len(CHANNELS),6000))
    data[0,1100] = 1e-2
    _, meta = build_features(data,list(CHANNELS),e,np.arange(6000)/1000,1000)
    assert len(meta) == 2
    assert not meta.analysis_eligible.any()
    assert meta.eeg_eligible.tolist() == [False,True]


def full_meta():
    pairs = [(a,b) for a in range(239,244) for b in range(a+1,244) for _ in range(4)]
    return events_for(pairs)


def test_comparison_and_pair_groups_never_cross_folds():
    meta = full_meta()
    for train,test in make_splits(meta):
        assert not set(meta.group.iloc[train]) & set(meta.group.iloc[test])
        assert not set(meta.pair_id.iloc[train]) & set(meta.pair_id.iloc[test])
        assert set(meta.trigger.iloc[train]) == set(range(239,244))


def test_restricted_permutation_preserves_comparison_pair_membership():
    meta = full_meta()
    y = meta.trigger.to_numpy()
    permuted = permute_pairs(y,meta,np.random.default_rng(12))
    assert not np.array_equal(y,permuted)
    for start in range(0,len(y),2):
        assert set(permuted[start:start+2]) == set(y[start:start+2])


def test_synthetic_waveform_signal_is_recovered_on_heldout_comparisons():
    meta = full_meta()
    y = meta.trigger.to_numpy()
    rng = np.random.default_rng(1)
    x = np.eye(5)[y-239] + rng.normal(0,.03,(len(y),5))
    prediction,_ = predict_oof(x,y,make_splits(meta))
    assert scores(y,prediction)['balanced_accuracy'] > .9


def test_purging_removes_complete_training_pairs_with_shared_signal_support():
    meta = full_meta()
    # Force overlap between different comparison groups (an acquisition anomaly).
    meta.loc[8,'elapsed_s'] = meta.loc[0,'elapsed_s'] + .1
    for train,test in make_splits(meta):
        for idx in train:
            same = test[meta.subject.to_numpy()[test] == meta.subject.iloc[idx]]
            assert np.all(np.abs(meta.elapsed_s.to_numpy()[same]-meta.elapsed_s.iloc[idx]) > 1.25)
        for _,group in meta.iloc[train].groupby('pair_id'):
            assert len(group) == 2


def test_loso_never_uses_heldout_participant_for_training():
    first = full_meta()
    second = first.copy()
    second['subject'] = 4
    second['pair_id'] = '4:' + second.pair_index.astype(str)
    second['group'] = '4:' + second.comparison
    for train,test in make_splits(pd.concat([first,second],ignore_index=True),'loso'):
        assert not set(pd.concat([first,second],ignore_index=True).subject.iloc[train]) & set(pd.concat([first,second],ignore_index=True).subject.iloc[test])
