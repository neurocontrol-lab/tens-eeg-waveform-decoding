import numpy as np
from experiments.waveform_decoding.e015_position_repeatability_decoding.confusion import aggregate, row_normalize


def test_empty_true_class_has_undefined_recall_not_zero():
    counts = np.array([[2, 3], [0, 0]])
    rates = row_normalize(counts)
    np.testing.assert_allclose(rates[0], [.4, .6])
    assert np.isnan(rates[1]).all()


def test_participant_balanced_recall_differs_from_trial_weighted_recall():
    small = np.array([[8, 2], [1, 9]])
    large = np.array([[20, 80], [3, 7]])
    counts, pooled, mean = aggregate([small, large])
    assert counts.sum() == small.sum() + large.sum()
    np.testing.assert_allclose(pooled[0, 0], 28 / 110)
    np.testing.assert_allclose(mean[0, 0], .5)
    participant_ba = [np.diag(row_normalize(m)).mean() for m in [small, large]]
    np.testing.assert_allclose(np.diag(mean).mean(), np.mean(participant_ba))
    # A participant with no trials for a class contributes no false zero recall.
    absent = np.array([[0, 0], [5, 5]])
    _, _, missing_mean = aggregate([small, absent])
    np.testing.assert_allclose(missing_mean[0], [.8, .2])
