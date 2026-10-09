"""
preprocess_artifact_rejection.py
=================================
Pure per-trial artifact-rejection logic: flags trials whose FRONTAL channel
amplitude, within a specific post-stimulus TIME WINDOW, is a robust outlier
relative to the other trials.
"""

import numpy as np


def _robust_z(values):
    """Robust z-scores via median + 1.4826*MAD (matches this project's
    existing preprocess_paper_method.detect_bad_channels() convention)."""
    values = np.asarray(values, dtype=float)
    med = np.median(values)
    mad = np.median(np.abs(values - med)) * 1.4826 + 1e-12
    return (values - med) / mad


def flag_frontal_artifact_trials(data, times_ms, window_ms, threshold_z=3.0):
    """Flag trials whose peak-to-peak amplitude, within ``window_ms`` only,
    is a robust outlier on ANY of the given (already channel-selected)
    frontal channels.

    Parameters
    ----------
    data : ndarray, shape (n_trials, n_channels, n_times)
        Epoch data restricted to the candidate artifact channels (pick
        those, e.g. Fp1/Fp2/Fpz/AF7/AF8, before calling this).
    times_ms : ndarray, shape (n_times,)
    window_ms : (float, float)
        (lo, hi) window (ms) to compute peak-to-peak amplitude within --
        amplitude outside this window never affects the decision.
    threshold_z : float
        Robust-z cutoff (MAD-based), matching this project's existing
        ICA find_bads_eog(threshold=3.0) convention.

    Returns
    -------
    bad : ndarray of bool, shape (n_trials,) -- True = reject.
    """
    if threshold_z <= 0:
        raise ValueError("threshold_z must be > 0")
    data = np.asarray(data, dtype=float)
    if data.ndim != 3:
        raise ValueError("data must be (n_trials, n_channels, n_times)")
    times_ms = np.asarray(times_ms, dtype=float)
    lo, hi = window_ms
    if hi <= lo:
        raise ValueError("window_ms must have hi > lo")
    mask = (times_ms >= lo) & (times_ms <= hi)
    if not mask.any():
        raise ValueError(f"window {window_ms} contains no samples")

    seg = data[:, :, mask]                       # (n_trials, n_ch, n_win)
    ptp = seg.max(axis=2) - seg.min(axis=2)       # (n_trials, n_ch)

    bad = np.zeros(data.shape[0], dtype=bool)
    for ch in range(ptp.shape[1]):
        bad |= _robust_z(ptp[:, ch]) > threshold_z
    return bad
