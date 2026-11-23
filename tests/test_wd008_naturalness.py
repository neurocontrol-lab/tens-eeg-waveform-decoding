"""Checks for future-sample and held-out-choice leakage in WD-008."""

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.waveform_decoding.e021_naturalness_choice_decoding.data import voltage_means
from experiments.waveform_decoding.e021_naturalness_choice_decoding.evaluate import (
    MODEL_LABELS, model_features, predict,
)


def sample_data():
    rng = np.random.default_rng(8)
    table = pd.DataFrame(dict(
        subject=np.repeat([3, 4, 5], 12),
        ordered_pair=np.tile(['241-242', '242-241'], 18),
        pair_index=np.tile(np.arange(12), 3),
        within_block_index=np.tile(np.arange(6), 6),
        target_second=np.tile([0, 1, 1, 0], 9),
    ))
    return table, {name: rng.normal(size=(36, 30))
                   for name in ('pre', 'first_post', 'paired_post')}


def test_pre_features_do_not_use_future_samples():
    data = np.arange(6 * 1000).reshape(6, 1000).astype(float)
    timestamps = np.arange(1000) / 1000
    expected = voltage_means(data, 500, (-.350, -.100), timestamps, 1000)
    data[:, 500:] = 1e20
    actual = voltage_means(data, 500, (-.350, -.100), timestamps, 1000)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(expected[:6], np.mean(data[:, 150:200], axis=1) * 1e6)


def test_charge_and_target_are_excluded_from_predictors():
    table, features = sample_data()
    original = model_features(table, features)
    table['target_second'] = 1 - table.target_second
    table['first_nominal_charge_uC'] = 999
    table['second_nominal_charge_uC'] = -999
    for name, values in model_features(table, features).items():
        np.testing.assert_array_equal(values, original[name])


def test_held_out_choices_do_not_change_held_out_predictions():
    table, features = sample_data()
    changed = table.copy()
    changed.loc[changed.subject.eq(3), 'target_second'] = 1 - changed.loc[
        changed.subject.eq(3), 'target_second']
    with threadpool_limits(limits=1):
        expected, _ = predict(table, features)
        actual, _ = predict(changed, features)
    np.testing.assert_allclose(
        actual.loc[table.subject.eq(3), list(MODEL_LABELS)],
        expected.loc[table.subject.eq(3), list(MODEL_LABELS)],
        rtol=0, atol=0,
    )
