"""True-pair alignment, feature construction and held-out-choice isolation."""

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.waveform_decoding.e021_naturalness_choice_decoding.paired_evaluate import (
    MODELS,build_pairs,post_features,predict,summarize,
)


def sample():
    rows=[]
    for i in range(12):
        a,b=[(239,240),(239,241),(240,241)][i//4]
        for position in (1,2):
            rows.append(dict(subject=3,position=position,pair_id=f'3:{i}',pair_index=i,event_index=2*i+position-1,
                comparison=f'{a}-{b}',block_index=i//4,trigger=a if position==1 else b,
                opponent_trigger=b if position==1 else a,response_value=1+i%2,
                within_block_index=i%4,terminal_vote=i%4==3))
    rng=np.random.default_rng(42)
    events=pd.DataFrame(rows)
    return events,{k:rng.normal(size=(24,20,62)) for k in ('pre','post')}


def test_actual_pair_alignment_survives_shuffled_epoch_rows():
    events,features=sample()
    expected,values=build_pairs(events,features)
    order=np.random.default_rng(2).permutation(len(events))
    actual,changed=build_pairs(events.iloc[order],{k:v[order] for k,v in features.items()})
    pd.testing.assert_frame_equal(expected,actual)
    for name in values:np.testing.assert_array_equal(values[name],changed[name])
    assert expected.target_second.tolist()==[0,1]*6
    np.testing.assert_array_equal(values['first_post'],features['post'][::2])
    np.testing.assert_array_equal(values['second_post'],features['post'][1::2])


def test_both_post_responses_enter_in_order_and_difference_is_exact():
    events,features=sample();_,values=build_pairs(events,features)
    paired=post_features(values,50,'concatenated')
    assert paired.shape==(12,2480)
    np.testing.assert_array_equal(paired[:,:1240],values['first_post'].reshape(12,-1))
    np.testing.assert_array_equal(paired[:,1240:],values['second_post'].reshape(12,-1))
    np.testing.assert_array_equal(post_features(values,50,'difference'),paired[:,1240:]-paired[:,:1240])
    assert post_features(values,100,'concatenated').shape==(12,1240)


def test_pair_fold_predictions_do_not_use_held_out_choices():
    events,features=sample();table,features=build_pairs(events,features)
    changed=table.copy();selected=table.comparison.eq('239-240')
    changed.loc[selected,'target_second']=1-changed.loc[selected,'target_second']
    with threadpool_limits(limits=1):
        expected,folds,unsupported=predict(table,features)
        actual,_,_=predict(changed,features)
    assert not unsupported and len(expected)==4*(12+9)
    np.testing.assert_allclose(expected.loc[expected.comparison.eq('239-240'),list(MODELS)],
                               actual.loc[actual.comparison.eq('239-240'),list(MODELS)],rtol=0,atol=0)
    for row in folds:
        assert row['training_subjects']==[3]
        assert row['held_out_comparison'] not in row['training_comparisons']
        assert not set(row['training_pairs'])&set(row['test_pairs'])
    result=summarize(expected)
    assert len(result['aggregate'])==56
    assert all(r['balanced_accuracy_gain']['test']['family_size']==32 for r in result['contrasts'])
