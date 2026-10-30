import numpy as np
import pandas as pd
from experiments.signal_quality.e010_simple_visual_inspection.core import (
    contiguous_segments, filter_continuous, extract_epochs, epoch_eligibility, cell_balanced_evokeds)


def test_segment_filter_never_bridges_a_timestamp_gap():
    fs = 200.
    t = np.arange(16000)/fs
    t[8000:] += 2.
    x = np.tile(np.sin(2*np.pi*10*t), (2, 1))
    assert contiguous_segments(x, t, fs) == [(0,8000),(8000,16000)]
    first, edge, _, _ = filter_continuous(x, t, fs)
    changed = x.copy()
    changed[:, 8000:] += 1e5
    second, _, _, _ = filter_continuous(changed, t, fs)
    np.testing.assert_array_equal(first[:, :8000], second[:, :8000])
    assert not edge[:1600].any() and not edge[6400:9600].any()
    x[:, 4000] = np.nan
    assert contiguous_segments(x, t, fs) == [(0,4000),(4001,8000),(8000,16000)]


def test_physical_units_baseline_and_filter_frequency_behavior():
    fs = 200.
    t = np.arange(16000)/fs
    x = 7e-6 + 2e-6*np.sin(2*np.pi*10*t) + 4e-6*np.sin(2*np.pi*70*t)
    filtered, _, _, _ = filter_continuous(x[None], t, fs)
    epochs, time = extract_epochs(x[None], [6000], fs)
    base = (time >= -.25) & (time < -.05)
    np.testing.assert_allclose(epochs[..., base].mean(axis=-1), 0, atol=1e-12)
    offset, _ = extract_epochs(x[None]+100e-6, [6000], fs)
    np.testing.assert_allclose(offset, epochs, atol=1e-12)
    fit = np.column_stack([np.sin(2*np.pi*f*t[4000:12000]) for f in (10,70)])
    amplitudes = np.linalg.lstsq(fit, filtered[0,4000:12000], rcond=None)[0]*1e6
    assert 1.9 < amplitudes[0] < 2.1
    assert abs(amplitudes[1]) < .05


def test_longer_epoch_failure_excludes_complete_pair_in_both_views():
    t = np.arange(10000)/1000
    meta = pd.DataFrame(dict(sample=[3000,4400,7000,8400], pair_id=['a','a','b','b'],
        analysis_eligible=[True]*4, previous_gap_s=[np.nan,1.4,2.6,1.4], next_gap_s=[1.4,2.6,1.4,.9]))
    result = epoch_eligibility(meta, t, np.ones(len(t), bool), 1000)
    assert result.visual_pair_eligible.tolist() == [True,True,False,False]
    assert 'next_stimulus' in result.visual_exclusion_reason.iloc[3]


def test_cell_balance_is_not_trial_count_weighting_and_keeps_positions():
    rows, values = [], []
    for position in (1,2):
        for code in (239,240):
            for context, n, value in [('a',1,2.),('b',3,10.)]:
                for _ in range(n):
                    rows.append(dict(position=position, trigger=code, comparison=context))
                    values.append(np.full((2,3), value+position+code))
    means, counts, cells = cell_balanced_evokeds(np.array(values),pd.DataFrame(rows),(239,240))
    assert means.shape == (2,2,2,3)
    np.testing.assert_allclose(means[0,0], 6+1+239)
    np.testing.assert_allclose(means[1,1], 6+2+240)
    assert (counts == 4).all() and (cells == 2).all()
