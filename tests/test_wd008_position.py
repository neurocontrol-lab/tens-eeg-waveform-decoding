"""Meaningful leakage and support checks for the revised naturalness experiment."""

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.waveform_decoding.e021_naturalness_choice_decoding.position_evaluate import (
    MODELS, metadata_features, predict, summarize,
)


def sample():
    rows = []
    for i in range(12):
        a,b = [(239,240),(239,241),(240,241)][i//4]
        for position in (1,2):
            rows.append(dict(subject=3,position=position,pair_id=f'3:{i}',event_index=2*i+position-1,
                comparison=f'{a}-{b}',block_index=i//4,trigger=a if position==1 else b,
                opponent_trigger=b if position==1 else a,target_chosen=(i+position)%2,
                pair_index=i,within_block_index=i%4,terminal_vote=i%4==3))
    table = pd.DataFrame(rows)
    rng = np.random.default_rng(24)
    return table,{period:rng.normal(size=(len(table),20,62)) for period in ('pre','post')}


def test_metadata_has_no_choice_charge_or_stopping_information():
    table,_ = sample()
    expected = metadata_features(table)
    table['target_chosen']=1-table.target_chosen
    table['terminal_vote']=~table.terminal_vote
    table['nominal_charge_uC']=999
    np.testing.assert_array_equal(metadata_features(table),expected)
    assert expected.shape==(24,12)


def test_position_fits_keep_pairs_and_comparisons_out_and_ignore_held_out_choices():
    table,features = sample()
    changed=table.copy()
    selected=table.comparison.eq('239-240')
    changed.loc[selected,'target_chosen']=1-changed.loc[selected,'target_chosen']
    with threadpool_limits(limits=1):
        before,folds,unsupported=predict(table,features)
        after,_,_=predict(changed,features)
    assert not unsupported
    assert len(before)==2*(24+18)
    np.testing.assert_allclose(before.loc[before.comparison.eq('239-240'),list(MODELS)],
                               after.loc[after.comparison.eq('239-240'),list(MODELS)],rtol=0,atol=0)
    for fold in folds:
        assert fold['held_out_comparison'] not in fold['training_comparisons']
        assert not set(fold['training_pairs']) & set(fold['test_pairs'])
        assert fold['training_subjects']==[3]
        assert fold['training_positions']==[fold['position']]
    for analysis in ('full','nonterminal'):
        group=before.loc[before.analysis.eq(analysis)]
        assert group.groupby(['width_ms','pair_id']).target_chosen.sum().eq(1).all()
    result=summarize(before)
    assert len(result['aggregate'])==56
    assert all(row['balanced_accuracy_gain']['test']['family_size']==32 for row in result['contrasts'])


def test_unsupported_training_folds_are_recorded_without_using_test_choices():
    table,features=sample()
    table['target_chosen']=1
    table.loc[table.comparison.eq('239-240'),'target_chosen']=0
    with threadpool_limits(limits=1):
        predictions,_,unsupported=predict(table,features)
    assert unsupported
    assert {row['comparison'] for row in unsupported}=={'239-240'}
    assert not predictions.comparison.eq('239-240').any()
