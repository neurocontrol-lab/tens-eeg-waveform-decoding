"""Leakage and binning checks for the dedicated triangle/LinDec experiment."""

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.waveform_decoding.e021_naturalness_choice_decoding.behavior_audit import order_test
from experiments.waveform_decoding.e022_triangle_lindec_decoding.evaluate import (
    predict, resolution, summarize, paired_sign_test, holm_adjust,
)


def test_exact_participant_sign_test_omits_ties_and_has_small_sample_limit():
    test = paired_sign_test([.1, .2, .03, .12, .08, 0, -1e-14])
    assert test == dict(improved=5, worsened=0, tied=2, p=.0625)
    assert paired_sign_test([0, 0])['p'] == 1
    assert paired_sign_test([.1, -.1, .2, -.2])['p'] == 1


def test_holm_adjustment_keeps_original_order_and_is_monotone():
    np.testing.assert_allclose(holm_adjust([.04, .01, .03]), [.06, .03, .06])
    np.testing.assert_allclose(holm_adjust([.8, .8, 1]), [1, 1, 1])


def test_conditional_order_test_matches_four_trial_exact_distribution():
    table = pd.DataFrame(dict(
        subject=[3] * 4, block_index=[0] * 4, low=[241] * 4,
        first_trigger=[241, 241, 242, 242], chosen_trigger=[241, 241, 242, 242],
    ))
    result = order_test(table)
    np.testing.assert_allclose(result['two_sided_p'], 1 / 3)
    np.testing.assert_allclose(result['null_mass'], 1)


def test_100_ms_features_average_adjacent_50_ms_bins():
    values = np.arange(2 * 20 * 62).reshape(2, 20, 62).astype(float)
    binned = resolution(values, 100).reshape(2, 10, 62)
    np.testing.assert_array_equal(binned[:, 0], (values[:, 0] + values[:, 1]) / 2)
    np.testing.assert_array_equal(binned[:, -1], (values[:, -2] + values[:, -1]) / 2)
    assert resolution(values, 50).shape == (2, 1240)


def test_held_out_direct_waveform_labels_do_not_change_direct_predictions():
    rng = np.random.default_rng(22)
    y = np.array([0] * 4 + [1] * 4 + [0, 1, 0, 1])
    events = pd.DataFrame(dict(
        subject=[3] * 12, position=[1] * 12, event_index=np.arange(12),
        pair_id=[f'3:{i}' for i in range(12)],
        comparison=['239-241'] * 4 + ['239-242'] * 4 + ['241-242'] * 4,
        trigger=np.where(y == 0, 241, 242), target_lindec=y,
        nominal_charge_uC=[.5] * 12, matched_direct=[False] * 8 + [True] * 4,
    ))
    features = {name: rng.normal(size=(12, 20, 62)) for name in ('pre', 'post')}
    changed = events.copy()
    mask = events.comparison.eq('241-242')
    changed.loc[mask, 'target_lindec'] = 1 - changed.loc[mask, 'target_lindec']
    with threadpool_limits(limits=1):
        before, folds = predict(events, features)
        after, _ = predict(changed, features)
    columns = ['position', 'width_ms', 'period', 'model', 'event_index', 'prediction', 'score']
    pd.testing.assert_frame_equal(
        before.loc[before.comparison.eq('241-242'), columns].reset_index(drop=True),
        after.loc[after.comparison.eq('241-242'), columns].reset_index(drop=True),
    )
    for row in folds.itertuples():
        assert row.held_out_comparison not in row.training_comparisons


def test_charge_matched_all_contexts_keeps_epochs_outside_direct_pairs():
    rows = []
    for subject in (3, 4):
        for period in ('pre', 'post'):
            for index, target in enumerate((0, 1, 0, 1)):
                rows.append(dict(
                    subject=subject, position=1, width_ms=50, model='pca_lda', period=period,
                    target_lindec=target, prediction=target, score=2 * target - 1,
                    comparison='241-242' if index >= 2 else ('239-241' if target == 0 else '239-242'),
                    matched_direct=subject == 3 and index >= 2,
                    nominal_charge_uC=.6 if subject == 3 else (.5 if target == 0 else .7),
                ))
    result = summarize(pd.DataFrame(rows))
    cohorts = {row['subset']: row for row in result['aggregate']}
    assert cohorts['all_contexts']['epochs'] == 8
    assert cohorts['matched_all_contexts']['epochs'] == 4
    assert cohorts['matched_direct']['epochs'] == 2
    assert cohorts['matched_all_contexts']['participants'] == 1
    assert cohorts['non_direct']['epochs'] == 4
    assert cohorts['matched_non_direct']['epochs'] == 2
    assert all(row['post_pre_test']['family_size'] == 6 for row in result['aggregate'])
    assert all(row['post_pre_test']['p_holm'] == 1 for row in result['aggregate'])

    extra = pd.DataFrame(rows).loc[lambda table: table.subject.eq(3)].copy()
    extra['position'] = 2
    extra['nominal_charge_uC'] = np.where(extra.target_lindec.eq(0), .4, .5)
    combined = summarize(pd.concat([pd.DataFrame(rows), extra], ignore_index=True))
    matched = [row for row in combined['aggregate'] if row['subset'] == 'matched_all_contexts']
    assert {row['position'] for row in matched} == {1}
