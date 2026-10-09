"""
task3_roi_erp.py
=================
Pure helper for the source-localized ROI ANOVA analysis
(Task3_ROI_SensorimotorANOVA.py): generates the sliding time windows used in
place of the original electrode analysis's fixed canonical (N140/P200/P300...)
windows, since dSPM/sLORETA source activation is magnitude-only (no polarity
to key a "canonical component" off of).

No MNE / I/O dependencies -- numpy only, so this stays unit-testable.
"""

import numpy as np


def sliding_windows(tmin_ms, tmax_ms, width_ms=50.0, step_ms=25.0):
    """Generate (lo, hi) sliding time windows covering [tmin_ms, tmax_ms].

    Windows start at ``tmin_ms`` and advance by ``step_ms``; the last window
    is clipped to end exactly at ``tmax_ms`` if a full-width window wouldn't
    fit (so the whole epoch is covered, matching "start to stop of epoch").

    Parameters
    ----------
    tmin_ms, tmax_ms : float
    width_ms : float
    step_ms : float

    Returns
    -------
    list of (float, float) tuples, low-to-high, non-decreasing starts.
    """
    if width_ms <= 0 or step_ms <= 0:
        raise ValueError("width_ms and step_ms must be > 0")
    if tmax_ms <= tmin_ms:
        raise ValueError("tmax_ms must be > tmin_ms")

    windows = []
    lo = tmin_ms
    while lo < tmax_ms:
        hi = min(lo + width_ms, tmax_ms)
        windows.append((lo, hi))
        if hi >= tmax_ms:
            break
        lo += step_ms
    return windows


def build_significance_ranking(stat_rows, artifact_zone_ms=None):
    """Summarize per-ROI significant-window counts from a sliding-window
    ANOVA sweep (one row per ROI per window), ranked by how many windows
    were significant.

    Parameters
    ----------
    stat_rows : sequence of dict
        Each must have "roi", "significant", "window_lo_ms", "window_hi_ms".
    artifact_zone_ms : (float, float) | None
        If given, each ROI's summary also flags whether any of its
        significant windows overlap this range (e.g. a known stimulus-
        artifact window), so a contaminated result is visible without
        cross-referencing the full window table.

    Returns
    -------
    list of dict: {roi, n_significant, n_windows, any_artifact_zone_overlap},
    sorted by n_significant descending (ties broken alphabetically by roi).
    """
    by_roi = {}
    for r in stat_rows:
        by_roi.setdefault(r["roi"], []).append(r)

    def _overlaps(lo, hi, zone):
        return lo < zone[1] and hi > zone[0]

    out = []
    for roi, rows in by_roi.items():
        sig_rows = [r for r in rows if r["significant"]]
        flagged = (artifact_zone_ms is not None and
                  any(_overlaps(r["window_lo_ms"], r["window_hi_ms"], artifact_zone_ms)
                      for r in sig_rows))
        out.append({"roi": roi, "n_significant": len(sig_rows), "n_windows": len(rows),
                   "any_artifact_zone_overlap": flagged})
    out.sort(key=lambda d: (-d["n_significant"], d["roi"]))
    return out


def mask_time_window(times_ms, y, window_ms):
    """Return a copy of ``y`` with samples inside ``window_ms`` set to NaN,
    for plotting a known-unreliable span (e.g. the ~100-140 ms eye-blink
    artifact -- see project memory / chat) as a visible gap rather than
    letting it dominate the trace and the y-axis autoscale.

    Parameters
    ----------
    times_ms : ndarray, shape (n_times,)
    y : ndarray, shape (..., n_times) -- masked along the last axis.
    window_ms : (float, float)
        (lo, hi) to mask, inclusive.

    Returns
    -------
    ndarray, same shape as ``y``, dtype float.
    """
    times_ms = np.asarray(times_ms, dtype=float)
    y = np.array(y, dtype=float, copy=True)  # copy: never mutate the caller's data
    lo, hi = window_ms
    if hi <= lo:
        raise ValueError("window_ms must have hi > lo")
    mask = (times_ms >= lo) & (times_ms <= hi)
    y[..., mask] = np.nan
    return y
