"""
August27_Task4_ICA_ManualReview_Common.py
=============================================
Shared helpers for the two-stage manual-ICA-review workflow:

  Stage 1 (interactive, one subject at a time, run on a machine with a
  display): August27_Task4_ICA_ManualReview.py -- fits ICA, walks you
  through each component with ica.plot_properties() (topomap + trial-
  sorted activity image + ERP + power spectrum + variance -- the same
  panels EEGLAB's component-properties dialog shows), and saves your
  accept/reject decisions.

  Stage 2 (batch, no display needed): August27_Task4_Preprocess_ManualICA.py
  -- reads Stage 1's saved decisions + cached ICA fit for each reviewed
  subject and runs the rest of the pipeline (blanking, low-pass, epoch,
  save) using exactly your manual exclusions -- no automatic detection.

Separate from every existing pipeline (Task4_Preprocess_blank10ms.py,
the August_23-27 chain, August26v3/v4): does not import from or write
into any of those. New, self-contained artifacts only:

  August27_Task4_ICA_ManualReview_cache/
    August27_manual_sub-{N}_raw-preica.fif   (post HP+notch+ref, pre-ICA;
                                               this is exactly what the ICA
                                               was fit on AND what it gets
                                               applied to -- so Stage 2
                                               reproduces Stage 1 exactly)
    August27_manual_sub-{N}_ica.fif          (the fitted decomposition)
  August27_Task4_ICA_ManualExclusions.json   (your per-subject decisions)
"""
import os, sys, json, datetime
import numpy as np
import mne

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "EEG_Practice"))
sys.path.insert(0, os.path.dirname(__file__))
from preprocess_paper_method import HP_CUT, HP_ORDER, HP_RS, LINE_FREQ
from eeg_waveform_analysis_with_save import load_xdf_to_mne

EOG_REF_CHANNEL = "Fp1"
UV_TO_V_SCALE = 1e-6
REF_DROP = ["CREF", "M1", "M2"]
ZSCORE_THRESHOLD = 3.0
TARGET_LOW, TARGET_HIGH, TARGET_MID = 10.0, 20.0, 15.0
MULT_GRID = np.geomspace(0.02, 5.0, 60)
N_COMPONENTS = 15

CACHE_DIR = "August27_Task4_ICA_ManualReview_cache"
MANIFEST_PATH = "August27_Task4_ICA_ManualExclusions.json"

SUBJECT_FILES = {
    3:  "EEG_Practice/Subject3_data/EEGDATA_SUB3-T4.xdf",
    4:  "EEG_Practice/Subject4_data/EEGDATA_SUB4-T4.xdf",
    5:  "EEG_Practice/Subject5_data/EEGDATA_SUB5-T4.xdf",
    6:  "EEG_Practice/Subject6_data/EEGDATA_SUB6-T4.xdf",
    7:  "EEG_Practice/Subject7_data/EEGDATA_SUB7-T4.xdf",
    8:  "EEG_Practice/Subject8_data/EEGDATA_SUB8-T4.xdf",
    9:  "EEG_Practice/Subject9_data/EEGDATA_SUB9-T4.xdf",
    10: "EEG_Practice/Subject10_data/EEGDATA_SUB10-T4.xdf",
    11: "EEG_Practice/Subject11_data/EEGDATA_SUB11-T4.xdf",
    12: "EEG_Practice/Subject12_data/EEGDATA_SUB12-T4.xdf",
}


def raw_preica_path(subject_id):
    return os.path.join(CACHE_DIR, f"August27_manual_sub-{subject_id}_raw-preica.fif")


def ica_path(subject_id):
    return os.path.join(CACHE_DIR, f"August27_manual_sub-{subject_id}_ica.fif")


def prepare_raw_for_ica(subject_id):
    """Load + preprocess up to (but not including) ICA fitting:
    uV->V fix -> 0.5Hz Cheby2 HP -> 60Hz comb notch -> drop CREF/M1/M2 ->
    average reference. Matches August27_Task4_ICA_ReferenceFix_Diagnostic.py's
    'avg_ref' variant, which this session confirmed helps (or at worst
    doesn't hurt) component identification. Returns the preprocessed Raw
    (NOT yet ICA-cleaned) plus the original stim events (for downstream
    blanking/epoching in Stage 2)."""
    fpath = SUBJECT_FILES[subject_id]
    raw, events = load_xdf_to_mne(fpath)
    raw.apply_function(lambda x: x * UV_TO_V_SCALE, picks="eeg", channel_wise=True, verbose=False)
    raw.filter(l_freq=HP_CUT, h_freq=None, method="iir",
              iir_params=dict(order=HP_ORDER, ftype="cheby2", rs=HP_RS), verbose=False)
    raw.notch_filter(np.arange(LINE_FREQ, raw.info["sfreq"] / 2, LINE_FREQ), verbose=False)
    present = [c for c in REF_DROP if c in raw.ch_names]
    if present:
        raw.drop_channels(present)
    raw.set_eeg_reference("average", projection=False, verbose=False)
    return raw, events


def calibration_path(subject_id):
    return os.path.join(CACHE_DIR, f"August27_manual_sub-{subject_id}_calibration.json")


def calibrate_and_suggest(raw, subject_id=None, force_recalibrate=False):
    """Per-subject calibrated EOG threshold (August26v4's method, reused
    unchanged) on a 1Hz-HP copy of the reference-fixed raw, used only to
    PRE-POPULATE a suggested exclusion set for the reviewer -- final say
    is always the human's in Stage 1.

    The 60-point multiplier grid search re-scans the WHOLE recording once
    per grid point (60x find_eog_events over ~10min/65ch data) -- cheaper
    than the ICA fit, but still real work. Cached per subject (like the
    ICA fit) when subject_id is given, so repeat review runs don't redo
    it. Always still needs the 1Hz-HP copy `r` for later use (component
    scoring), so filtering itself is never skipped -- only the grid
    search is."""
    r = raw.copy().filter(1.0, None, fir_design="firwin", verbose=False)

    if subject_id is not None and not force_recalibrate:
        cpath = calibration_path(subject_id)
        if os.path.exists(cpath):
            with open(cpath) as f:
                cached = json.load(f)
            return r, cached["chosen_mult"], cached["chosen_rate"]

    dur_min = r.times[-1] / 60
    ref_std = r.get_data(picks=[EOG_REF_CHANNEL])[0].std()
    best_mult, best_rate, best_dist = None, None, np.inf
    in_band = []
    for mult in MULT_GRID:
        thresh = mult * ref_std
        n_ev = len(mne.preprocessing.find_eog_events(r, ch_name=EOG_REF_CHANNEL, thresh=thresh, verbose=False))
        rate = n_ev / dur_min
        if TARGET_LOW <= rate <= TARGET_HIGH:
            in_band.append((mult, rate))
        dist = abs(rate - TARGET_MID)
        if dist < best_dist:
            best_dist, best_mult, best_rate = dist, mult, rate
    chosen_mult, chosen_rate = min(in_band, key=lambda mr: abs(mr[1] - TARGET_MID)) if in_band else (best_mult, best_rate)

    if subject_id is not None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(calibration_path(subject_id), "w") as f:
            json.dump(dict(chosen_mult=chosen_mult, chosen_rate=chosen_rate), f)

    return r, chosen_mult, chosen_rate


def fit_or_load_ica(subject_id, force_refit=False):
    """Cache-aware ICA fit. Checks the cache FIRST, before doing any
    preprocessing -- on a cache hit this is just two fast .fif reads.
    Only on a genuine miss (or force_refit) does it call
    prepare_raw_for_ica() (the slow part: order-94 zero-phase Chebyshev-II
    high-pass + 8-harmonic notch over the full ~10min/65ch recording) and
    fit fresh (infomax-extended, 15 components, on a 1Hz-HP copy).

    Previously this took a preprocessed `raw` as a required argument,
    which forced EVERY caller to run prepare_raw_for_ica() up front even
    on a cache hit, where the result was immediately discarded -- that
    was the actual cause of "cached" reviews still taking 5+ minutes.
    Now it's lazy: preprocessing only happens when actually needed."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    rpath, ipath = raw_preica_path(subject_id), ica_path(subject_id)
    if not force_refit and os.path.exists(rpath) and os.path.exists(ipath):
        cached_raw = mne.io.read_raw_fif(rpath, preload=True, verbose=False)
        ica = mne.preprocessing.read_ica(ipath, verbose=False)
        return cached_raw, ica, True

    raw, _events = prepare_raw_for_ica(subject_id)
    r_hp = raw.copy().filter(1.0, None, fir_design="firwin", verbose=False)
    ica = mne.preprocessing.ICA(n_components=N_COMPONENTS, method="infomax",
                                fit_params=dict(extended=True),
                                random_state=42, max_iter="auto")
    ica.fit(r_hp)
    raw.save(rpath, overwrite=True)
    ica.save(ipath, overwrite=True)
    return raw, ica, False


def load_manifest():
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    return {}


def save_manifest_entry(subject_id, excluded, auto_suggested):
    manifest = load_manifest()
    manifest[str(subject_id)] = dict(
        excluded=sorted(int(c) for c in excluded),
        auto_suggested=sorted(int(c) for c in auto_suggested),
        reviewed_at=datetime.datetime.now().isoformat(timespec="seconds"),
    )
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return manifest
