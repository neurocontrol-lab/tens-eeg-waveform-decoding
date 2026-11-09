import numpy as np
import pandas as pd
import pytest

from experiments.response_characterization.e017_eeg_charge_relation.__main__ import analyze
from experiments.response_characterization.e017_eeg_charge_relation.forecast import forecast_arrays
from experiments.response_characterization.e017_eeg_charge_relation.shared_slope import predict_contrasts


def test_upper_tier_outcomes_cannot_change_its_predictions():
    rows = []
    for subject in (3, 4):
        for waveform, alpha, step in [('Square', 5., .1), ('Triangle', -2., .05)]:
            for tier in range(3):
                charge = .4 + tier * step
                for _ in range(3):
                    rows.append(dict(subject=subject, waveform=waveform, tier=tier, charge_uC=charge,
                                     post_uv=alpha + 12 * charge, pre_uv=alpha,
                                     eeg_qc=np.nan, response_valid=True, csv_row=len(rows) + 2))
    data = pd.DataFrame(rows)
    _, _, _, original, _, _ = analyze(data)
    data.loc[data.tier.eq(2), 'post_uv'] += 10
    _, _, _, changed, _, _ = analyze(data)
    selected = original.window.eq('post_uv') & original.model.ne('constant')
    np.testing.assert_allclose(original.loc[selected, 'predicted_uv'], original.loc[selected, 'observed_uv'])
    np.testing.assert_array_equal(original.predicted_uv, changed.predicted_uv)


def test_200ms_targets_use_four_complete_bins():
    bins = np.arange(9, dtype=float).reshape(1, 1, 9, 1)
    history, targets = forecast_arrays(bins, widths_ms=(200,), start_bins=(3, 5))
    np.testing.assert_array_equal(history[0, 0, :, 0], [[0, 1, 2], [2, 3, 4]])
    np.testing.assert_array_equal(targets[0, 0, 0, :, 0], [4.5, 6.5])
    with pytest.raises(ValueError, match='extends beyond'):
        forecast_arrays(bins, widths_ms=(200,), start_bins=(7,))


def test_shared_charge_slope_excludes_held_out_waveform_and_absorbs_block_offsets():
    rows, outcomes = [], []
    for waveform, factor in [('Square', .4), ('Triangle', .2), ('Sine', .25)]:
        for fold, offset in enumerate([30., -40.]):
            for tier in [0, 2] if fold == 0 else [1, 2]:
                amplitude = 1. + .25 * tier
                charge = factor * amplitude
                rows.append(dict(subject=1, waveform=waveform, fold=fold, tier=tier,
                                 amplitude_mA=amplitude, charge_uC=charge))
                outcomes.append(offset + 7. * charge)
    events = pd.DataFrame(rows)
    observed = np.asarray(outcomes)[:, None, None, None, None]
    cells, actual, predictions, slopes = predict_contrasts(events, observed)
    np.testing.assert_allclose(predictions[:, 1], actual)
    np.testing.assert_allclose(slopes[:, 0], 7.)
    observed[events.waveform.eq('Square') & events.tier.eq(2)] += 500.
    _, changed, predictions_after, _ = predict_contrasts(events, observed)
    held_out = cells.held_out_waveform.eq('Square')
    np.testing.assert_array_equal(predictions[held_out], predictions_after[held_out])
    np.testing.assert_allclose(changed[held_out] - actual[held_out], 500.)
