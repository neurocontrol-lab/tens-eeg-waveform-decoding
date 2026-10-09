"""
preprocess_paper_method.py
==========================
"PAPER METHOD" preprocessing — Alibou, Artoni, D'Anna & Micera, "Cortical
connectivity and spectral perturbations underlying TENS stimulation of hand
nerves: a case study" (IEEE) — applied to our Task 3 XDF data.

Adapted to OUR recording (per user): line-noise notch at **60 Hz** (paper: 50 Hz,
EU mains) and **1000 Hz** sampling (paper: 2000 Hz). 

Paper steps implemented here:
 1. load XDF, standard 10-20 montage, decode triggers (sub-9 split/inverted +
    sub-2 base-only handled via preprocess_task3_missing.load_concat)
 2. re-reference to behind-the-ear channels (M1, M2); then drop M1/M2/CREF
 3. high-pass 0.5 Hz, 94th-order Chebyshev type II (slow-drift removal; SOS)
 4. 60 Hz comb notch (line noise; fundamental + harmonics)        [paper: 50 Hz]
 5. automated bad-channel detection + interpolation (stand-in for visual rejection)
 6. re-reference to global average
 7. ICA (infomax-extended); auto-remove ocular (Fpz surrogate) + muscle components
 8. stimulation-artifact removal: blank +/-3 ms around each event, then a
    50.1 Hz comb notch (7.5 Hz width) and a 190 Hz low-pass
 9. epoch -250..+500 ms, baseline (-250, 0)
 10. frontal eye-blink/startle-artifact trial rejection (added after
     diagnosing a residual ~100-140 ms deflection -- see below); save

Assumptions where the paper is silent (documented): Cheby2 rs = 40 dB; blanking
+/-3 ms; comb = fundamental + harmonics below Nyquist; behind-the-ear = M1/M2;
automated ICA rejection stands in for manual component selection.

Step 10 -- frontal artifact trial rejection (added [see project memory]):
Grand-average evoked responses showed a sharp, tightly time-locked, frontal-
dominant (Fpz/F7/AF7-max, not centroparietal) deflection at ~100-140 ms post-
stimulus in EVERY waveform condition -- an eye-blink/startle reflex, not a
genuine cortical response (electrical hand-nerve stimulation, especially at
higher amplitudes, can trigger an involuntary blink around this latency).
Diagnosed via subject-level ICA source inspection: the responsible
component(s) (e.g. sub-3's ICA00, Fp2-dominant pattern) were ALREADY being
correctly excluded by step 7's existing threshold=3.0 auto-detection -- but
removing them only accounts for the majority of the artifact; ~1/3 of the
peak amplitude survives in the grand average even with ICA fully applied,
because a blink's morphology varies trial-to-trial and doesn't decompose
into one fixed spatial ICA component. So a second, explicit step drops
individual trials directly: any trial whose peak-to-peak amplitude on
Fp1/Fp2/Fpz/AF7/AF8, computed ONLY within 100-140 ms (so it cannot reject a
trial for anything outside that window/those channels), is a robust
amplitude outlier (MAD-based z > 3.0, matching the ICA threshold already
used in step 7) relative to that subject's own other trials. Pure logic in
preprocess_artifact_rejection.py (tested).

Output: EEG_Practice/preprocessed_data_paper_method/sub-{N}_ses-Task3_*.fif
Run: python preprocess_paper_method.py [--smoke]
"""
import os
import sys
import argparse
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "EEG_Practice"))
import mne
from eeg_waveform_analysis_with_save import (
    save_preprocessed_data, TRIGGER_MAPPING,
)
from preprocess_task3_missing import load_concat
from preprocess_artifact_rejection import flag_frontal_artifact_trials

warnings.filterwarnings("ignore")
mne.set_log_level("WARNING")

OUTPUT_DIR = os.path.join("EEG_Practice", "preprocessed_data_paper_method")
SESSION_ID = "Task3"

# --- Paper-method parameters (with our adaptations) ---
LINE_FREQ = 60.0                       # USA mains (paper: 50 Hz)
HP_CUT, HP_ORDER, HP_RS = 0.5, 94, 40  # Chebyshev II high-pass (rs assumed 40 dB)
STIM_COMB, STIM_BW, STIM_LP = 50.1, 7.5, 190.0   # stim-artifact removal
BLANK_MS = 3.0                         # +/- blanking window around each event (legacy)
BLANK_PRE_MS = 3.0                     # blanking before each event
BLANK_POST_MS = 3.0                    # blanking after each event
APPLY_STIM_COMB = True                 # apply STIM_COMB notch (off for transient stim)
TMIN, TMAX, BASELINE = -0.250, 0.500, (-0.250, 0)
BEHIND_EAR = ["M1", "M2"]
DROP_AFTER_REF = ["M1", "M2", "CREF"]

# --- Step 10: frontal eye-blink/startle-artifact trial rejection ---
FRONTAL_ARTIFACT_CHANNELS = ["Fp1", "Fp2", "Fpz", "AF7", "AF8"]
FRONTAL_ARTIFACT_WINDOW_MS = (100.0, 140.0)
FRONTAL_ARTIFACT_THRESHOLD_Z = 3.0

SUBJECT_FILES = {
    2:  ["EEG_Practice/Subject2_data/EEGDATA_SUB2-T3.xdf"],
    3:  ["EEG_Practice/Subject3_data/EEGDATA_SUB3-T3.xdf"],
    4:  ["EEG_Practice/Subject4_data/EEGDATA_SUB4-T3.xdf"],
    5:  ["EEG_Practice/Subject5_data/EEGDATA_SUB5-T3.xdf"],
    6:  ["EEG_Practice/Subject6_data/EEGDATA_SUB6-T3.xdf"],
    7:  ["EEG_Practice/Subject7_data/EEGDATA_SUB7-T3.xdf"],
    8:  ["EEG_Practice/Subject8_data/EEGDATA_SUB8-T3.xdf"],
    9:  ["EEG_Practice/Subject9_data/EEGDATA_SUB9-T3_W1-2-3.xdf",
         "EEG_Practice/Subject9_data/EEGDATA_SUB9-T3_W4-5.xdf"],
    10: ["EEG_Practice/Subject10_data/EEGDATA_SUB10-T3.xdf"],
    11: ["EEG_Practice/Subject11_data/EEGDATA_SUB11-T3.xdf"],
    12: ["EEG_Practice/Subject12_data/EEGDATA_SUB12-T3.xdf"],
}


def detect_bad_channels(raw):
    """Conservative automated stand-in for visual rejection: flag flat channels
    and log-variance outliers (robust |z| > 5) among EEG channels."""
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    data = raw.get_data(picks=picks)
    names = [raw.ch_names[p] for p in picks]
    sd = data.std(axis=1)
    bad = {nm for nm, s in zip(names, sd) if s < 1e-9}
    logv = np.log(sd + 1e-12)
    med = np.median(logv)
    mad = np.median(np.abs(logv - med)) * 1.4826 + 1e-12
    z = (logv - med) / mad
    bad |= {nm for nm, zz in zip(names, z) if abs(zz) > 5}
    return sorted(bad)


def run_ica(raw):
    ica = mne.preprocessing.ICA(n_components=15, method="infomax",
                                fit_params=dict(extended=True),
                                random_state=42, max_iter="auto")
    r = raw.copy().filter(1.0, None, fir_design="firwin", verbose=False)
    ica.fit(r)
    excl = set()
    eog_ref = "Fpz" if "Fpz" in r.ch_names else ("Fp1" if "Fp1" in r.ch_names else None)
    try:
        eog, _ = ica.find_bads_eog(r, ch_name=eog_ref, threshold=3.0)
        excl |= set(eog)
    except Exception:
        pass
    try:
        mus, _ = ica.find_bads_muscle(r)
        excl |= set(mus)
    except Exception:
        pass
    ica.exclude = sorted(excl)
    return ica


def blank_around_events(raw, events, pre_ms, post_ms=None):
    """Linear-interpolate (blank) from pre_ms before to post_ms after each event.
    If post_ms is None, blanks symmetrically +/- pre_ms (back-compatible)."""
    if post_ms is None:
        post_ms = pre_ms
    sf = raw.info["sfreq"]
    n_pre = int(round(pre_ms * sf / 1000.0))
    n_post = int(round(post_ms * sf / 1000.0))
    if n_pre < 1 and n_post < 1:
        return raw
    data = raw.get_data()
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    ntot = data.shape[1]
    for s in events[:, 0]:
        lo = max(1, int(s) - n_pre)
        hi = min(ntot - 1, int(s) + n_post)
        if hi - lo >= 2:
            for ch in picks:
                data[ch, lo:hi] = np.linspace(data[ch, lo - 1], data[ch, hi], hi - lo)
    raw._data[:] = data
    return raw


def paper_preprocess(raw, events):
    raw = raw.copy().load_data()
    # 2. behind-the-ear reference, then drop reference/aux channels
    avail = [c for c in BEHIND_EAR if c in raw.ch_names]
    if avail:
        raw.set_eeg_reference(avail, projection=False, verbose=False)
    drop = [c for c in DROP_AFTER_REF if c in raw.ch_names]
    if drop:
        raw.drop_channels(drop)
    # 3. high-pass Chebyshev II (0.5 Hz, order 94, SOS)
    raw.filter(l_freq=HP_CUT, h_freq=None, method="iir",
               iir_params=dict(order=HP_ORDER, ftype="cheby2", rs=HP_RS, output="sos"),
               phase="zero", verbose=False)
    # 4. line-noise comb notch (60 Hz + harmonics)
    raw.notch_filter(np.arange(LINE_FREQ, raw.info["sfreq"] / 2, LINE_FREQ), verbose=False)
    # 5. bad channels -> interpolate
    bads = detect_bad_channels(raw)
    raw.info["bads"] = bads
    if bads:
        try:
            raw.interpolate_bads(reset_bads=True, verbose=False)
        except Exception:
            raw.info["bads"] = []
    # 6. global average reference
    raw.set_eeg_reference("average", projection=False, verbose=False)
    # 7. ICA
    ica = run_ica(raw)
    ica.apply(raw)
    # 8. stimulation-artifact removal
    raw = blank_around_events(raw, events, BLANK_PRE_MS, BLANK_POST_MS)
    if APPLY_STIM_COMB:
        raw.notch_filter(np.arange(STIM_COMB, raw.info["sfreq"] / 2, STIM_COMB),
                         notch_widths=STIM_BW, verbose=False)
    raw.filter(l_freq=None, h_freq=STIM_LP, verbose=False)
    return raw, ica, bads


def reject_frontal_artifact_trials(epochs):
    """Step 10: drop trials that are still amplitude outliers on frontal
    channels within the ~100-140 ms eye-blink/startle-artifact window, after
    ICA (see FRONTAL_ARTIFACT_* constants + the module docstring for why).
    Mutates and returns ``epochs``; a no-op if none of the target channels
    are present."""
    picks = [c for c in FRONTAL_ARTIFACT_CHANNELS if c in epochs.ch_names]
    if not picks:
        return epochs, 0
    data = epochs.get_data(picks=picks, verbose=False)
    bad = flag_frontal_artifact_trials(
        data, epochs.times * 1000.0, FRONTAL_ARTIFACT_WINDOW_MS,
        FRONTAL_ARTIFACT_THRESHOLD_Z)
    n_bad = int(bad.sum())
    if n_bad:
        epochs.drop(bad, reason="FRONTAL_ARTIFACT", verbose=False)
    return epochs, n_bad


def process_subject(sid, files):
    print(f"\n{'='*70}\nSUBJECT {sid} (PAPER METHOD)\n{'='*70}")
    for f in files:
        if not os.path.exists(f):
            print(f"  MISSING {f}"); return False
    raw, events = load_concat(files)
    print(f"  raw {len(raw.ch_names)}ch {raw.n_times/raw.info['sfreq']:.0f}s, {len(events)} events")
    raw_pp, ica, bads = paper_preprocess(raw, events)
    present = set(int(c) for c in events[:, 2])
    event_dict = {k: v for k, v in TRIGGER_MAPPING.items() if v in present}
    epochs = mne.Epochs(raw_pp, events, event_id=event_dict, tmin=TMIN, tmax=TMAX,
                        baseline=BASELINE, preload=True, reject_by_annotation=True,
                        verbose=False)
    n_before = len(epochs)
    epochs, n_frontal_rejected = reject_frontal_artifact_trials(epochs)
    save_preprocessed_data(raw_pp, epochs, ica, OUTPUT_DIR, sid, SESSION_ID)
    ev = epochs.average()
    print(f"  SAVED sub-{sid}: {len(epochs)}/{n_before} epochs "
          f"({n_frontal_rejected} dropped for frontal artifact), "
          f"{len(epochs.event_id)} conds, {len(raw_pp.ch_names)}ch, "
          f"|evoked|max={np.abs(ev.data).max():.2f}uV, "
          f"bads={bads}, ICA_excl={ica.exclude}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    subs = {3: SUBJECT_FILES[3]} if args.smoke else SUBJECT_FILES
    ok = {}
    for sid, files in subs.items():
        try:
            ok[sid] = process_subject(sid, files)
        except Exception as e:
            import traceback
            print(f"  ERROR sub {sid}: {e}"); traceback.print_exc(); ok[sid] = False
    print(f"\n{'='*70}\nDONE (paper method): {ok}\n{'='*70}")


if __name__ == "__main__":
    main()
