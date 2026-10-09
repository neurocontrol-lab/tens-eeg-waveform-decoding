"""
task3_erp_peaks.py
==================
Pure peak-detection utilities for Task 3 ERP analysis.

Detects positive and negative deflections in an evoked time series, labels them
against canonical somatosensory/cognitive ERP windows, and extracts the extreme
within a fixed window for per-subject statistics.

Honesty note: canonical component *names* (N140, P200, P300, ...) are only
asserted for broadband signals.  For band-limited signals (alpha, beta) the
detected extrema are oscillatory deflections, so they are tagged by the latency
window they fall in rather than named as the classical component.

No MNE / I/O dependencies -- numpy + scipy only, so this stays unit-testable.
"""

import numpy as np
from scipy.signal import find_peaks

# (name, low_ms, high_ms, polarity)  polarity: +1 positive, -1 negative
CANONICAL_WINDOWS = [
    ("N20",  18,  30, -1),
    ("P27",  25,  40, +1),
    ("P50",  40,  70, +1),
    ("N70",  60,  95, -1),
    ("P100", 90, 135, +1),
    ("N140", 120, 175, -1),
    ("P200", 175, 260, +1),
    ("P300", 260, 500, +1),
    ("LPC",  500, 800, +1),
]


def _robust_noise(x):
    """1.4826 * MAD -> robust sigma estimate; small floor avoids zero."""
    if x.size == 0:
        return 1e-12
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med)) + 1e-12


def detect_peaks(times_ms, signal, baseline_mask=None, prominence_k=3.0,
                 min_distance_ms=20.0, tmin_ms=0.0):
    """Detect positive and negative peaks in an evoked signal.

    Parameters
    ----------
    times_ms : (n,) array            time axis in milliseconds
    signal   : (n,) array            amplitude in microvolts
    baseline_mask : (n,) bool or None  samples used to estimate noise
                                        (default: pre-stimulus, times < 0)
    prominence_k : float             prominence threshold = k * robust-sigma
    min_distance_ms : float          minimum separation between peaks
    tmin_ms : float                  ignore peaks earlier than this (default 0)

    Returns
    -------
    list of dict, sorted by latency:
        {latency_ms, amplitude_uv, polarity (+1/-1), prominence}
    """
    times_ms = np.asarray(times_ms, dtype=float)
    signal = np.asarray(signal, dtype=float)

    if baseline_mask is None:
        baseline_mask = times_ms < 0
    noise = _robust_noise(signal[baseline_mask])
    prom = prominence_k * noise

    dt = np.median(np.diff(times_ms))
    distance = max(1, int(round(min_distance_ms / dt)))

    peaks = []
    for polarity, sig in ((+1, signal), (-1, -signal)):
        idx, props = find_peaks(sig, prominence=prom, distance=distance)
        for i, p in zip(idx, props["prominences"]):
            if times_ms[i] < tmin_ms:
                continue
            peaks.append({
                "latency_ms": float(times_ms[i]),
                "amplitude_uv": float(signal[i]),
                "polarity": int(polarity),
                "prominence": float(p),
            })
    peaks.sort(key=lambda d: d["latency_ms"])
    return peaks


def _canonical_name(latency_ms, polarity, windows):
    for name, lo, hi, pol in windows:
        if lo <= latency_ms <= hi and polarity == pol:
            return name
    return None


def label_peaks(peaks, band="broadband", windows=CANONICAL_WINDOWS):
    """Add 'label' and 'canonical' keys to each peak.

    broadband  -> canonical component name when latency+polarity match a window
                  (else 'pos@Xms' / 'neg@Xms')
    alpha/beta -> band-limited tag, e.g. 'pos@312ms [P300-window, alpha]'
    """
    out = []
    for pk in peaks:
        pk = dict(pk)
        name = _canonical_name(pk["latency_ms"], pk["polarity"], windows)
        sign = "pos" if pk["polarity"] > 0 else "neg"
        pk["window"] = name   # matched latency window (all bands), or None
        if band == "broadband":
            pk["label"] = name if name else f"{sign}@{pk['latency_ms']:.0f}ms"
            pk["canonical"] = name
        else:
            tag = f"[{name}-window, {band}]" if name else f"[{band}]"
            pk["label"] = f"{sign}@{pk['latency_ms']:.0f}ms {tag}"
            pk["canonical"] = None  # not a canonical component under band-pass
        out.append(pk)
    return out


def dedupe_peaks(labeled_peaks):
    """Keep at most one peak per latency window (the highest-prominence one), and
    drop peaks that fall in no canonical window.  Produces a clean, one-marker-
    per-component set for publication-ready annotation.  Requires peaks already
    passed through label_peaks (i.e. carrying a 'window' key).  Sorted by latency.
    """
    best = {}
    for pk in labeled_peaks:
        win = pk.get("window")
        if win is None:
            continue
        if win not in best or pk["prominence"] > best[win]["prominence"]:
            best[win] = pk
    return sorted(best.values(), key=lambda d: d["latency_ms"])


def extract_window_peak(times_ms, signal, t_lo_ms, t_hi_ms, polarity):
    """Return (latency_ms, amplitude_uv) of the extreme of given polarity within
    [t_lo_ms, t_hi_ms].  Used for per-subject peak LATENCY.  NaN if window empty.
    """
    times_ms = np.asarray(times_ms, dtype=float)
    signal = np.asarray(signal, dtype=float)
    m = (times_ms >= t_lo_ms) & (times_ms <= t_hi_ms)
    if not m.any():
        return float("nan"), float("nan")
    seg = signal[m]
    tt = times_ms[m]
    j = int(np.argmax(seg)) if polarity > 0 else int(np.argmin(seg))
    return float(tt[j]), float(seg[j])


def window_mean_amplitude(times_ms, signal, t_lo_ms, t_hi_ms):
    """Mean signal amplitude over [t_lo_ms, t_hi_ms] (signed).  This is the
    recommended, noise-robust amplitude measure (Luck 2014) -- unlike peak
    amplitude it is not biased upward by noise and equals what the averaged
    waveform shows.  NaN if the window is empty.
    """
    times_ms = np.asarray(times_ms, dtype=float)
    signal = np.asarray(signal, dtype=float)
    m = (times_ms >= t_lo_ms) & (times_ms <= t_hi_ms)
    if not m.any():
        return float("nan")
    return float(signal[m].mean())
