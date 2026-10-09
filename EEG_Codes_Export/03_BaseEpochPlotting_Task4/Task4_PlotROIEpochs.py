"""
Task4_PlotROIEpochs.py
=======================
Electrode-ROI-averaged epoch plotting for Task4, updated for the current
AutoICA preprocessing pipeline's file naming (August27_Task4_Preprocess_AutoICA.py
/ August28_Task4_Preprocess_AutoICA33.py and friends) -- the variant naming
scheme is copied verbatim from August27_Task4_ROI_ChosenUnchosen_Auto.py's
VARIANTS dict so this script always points at real, existing preprocessed
output directories.

This does NOT touch raw .xdf files and does NOT run any preprocessing --
it only reads already-saved *_epo.fif files. Compare to
EEG_Practice/eeg_waveform_analysis_with_save.py's plot_waveform_evoked(),
which this replaces for the new pipeline: that function's cohort-level
loaders (load_and_combine_subjects / load_preprocessed_data) expect the
older "sub-{id}_ses-{session}_epo.fif" naming and can't find the AutoICA
variants' files, and its default trigger mapping does not carry the
LinInc/LinDec 242/243 swap fix -- this script reuses only the reusable
pieces of that module (ELECTRODE_GROUPS, FREQUENCY_BANDS, apply_frequency_band)
and takes the corrected WAVEFORM_TRIGGERS mapping from task4_source_loc.py
instead (see project memory: "LinInc/LinDec Trigger Swap Bug").

Usage is deliberately reduced to four things: which preprocessed variant,
which subjects, which electrodes (ROI), and which frequency band --
everything else (waveform trigger mapping, colors, CI style) is fixed.

Usage
-----
    python Task4_PlotROIEpochs.py --variant autoeog33
    python Task4_PlotROIEpochs.py --variant autoeog33 --subjects 3,4,5 --electrodes C3,C4,Cz --band alpha
    python Task4_PlotROIEpochs.py --variant autoeog33 --electrodes central --band beta
    python Task4_PlotROIEpochs.py --variant autoeog33muscle --electrodes parietal --waveforms Sine,LinInc

    --variant     required. One of: autoeog, autoeogmuscle, autoeog33,
                  autoeog33muscle, manualica33 (see VARIANTS below).
    --subjects    comma-separated subject IDs. Default: all of 3-12.
    --electrodes  a named ROI from ELECTRODE_GROUPS (central, frontal,
                  parietal, occipital, temporal, frontocentral,
                  centroparietal, midline, left_hemisphere,
                  right_hemisphere, all) OR a comma-separated list of raw
                  channel names (e.g. C3,C4,Cz). Default: all.
    --band        one of delta, theta, alpha, beta, gamma, broadband.
                  Default: broadband.
    --waveforms   comma-separated subset of Sine,Square,Triangle,LinInc,
                  LinDec. Default: all five.
    --ci          confidence-interval level for the between-subject band
                  (bootstrapped across each subject's evoked). Default 0.90.

Output: <outdir>/<variant>_<electrodes>_<band>_ROI_epoch.png
        (outdir defaults to "Task4_ROI_EpochPlot_Output")
"""

from __future__ import annotations

import argparse
import os
import sys

import mne
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings_mod = __import__("warnings")
warnings_mod.filterwarnings("ignore")
mne.set_log_level("ERROR")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "EEG_Practice"))

from eeg_waveform_analysis_with_save import ELECTRODE_GROUPS, FREQUENCY_BANDS, apply_frequency_band
from task4_source_loc import WAVEFORM_TRIGGERS

# ---------------------------------------------------------------------------
# Configuration -- kept in sync with August27_Task4_ROI_ChosenUnchosen_Auto.py
# ---------------------------------------------------------------------------

VARIANTS = {
    "autoeog": dict(
        data_dir="August27_Task4_Preprocessed_Data_AutoEOG",
        fname_template="August27_autoeog_sub-{sid}_epo.fif",
        label="EOG-only (15 comps)",
    ),
    "autoeogmuscle": dict(
        data_dir="August27_Task4_Preprocessed_Data_AutoEOGMuscle",
        fname_template="August27_autoeogmuscle_sub-{sid}_epo.fif",
        label="EOG+muscle (15 comps)",
    ),
    "autoeog33": dict(
        data_dir="August28_Task4_Preprocessed_Data_AutoEOG33",
        fname_template="August28_autoeog33_sub-{sid}_epo.fif",
        label="EOG-only, 33 comps",
    ),
    "autoeog33muscle": dict(
        data_dir="August28_Task4_Preprocessed_Data_AutoEOG33Muscle",
        fname_template="August28_autoeog33muscle_sub-{sid}_epo.fif",
        label="EOG+muscle, 33 comps",
    ),
    "manualica33": dict(
        data_dir="August28_Task4_Preprocessed_Data_ManualICA33",
        fname_template="August28_manualica33_sub-{sid}_epo.fif",
        label="Manual review, 33 comps",
    ),
}

ALL_SUBJECTS = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
WAVEFORM_ORDER = ["Sine", "Square", "Triangle", "LinInc", "LinDec"]
WAVEFORM_COLORS = {"Sine": "#1f77b4", "Square": "#ff7f0e", "Triangle": "#2ca02c",
                   "LinInc": "#d62728", "LinDec": "#9467bd"}
OUTDIR_DEFAULT = "Task4_ROI_EpochPlot_Output"


# ---------------------------------------------------------------------------
# Argument resolution
# ---------------------------------------------------------------------------

def resolve_electrodes(spec):
    """--electrodes value -> (picks_or_None, label). None picks == all channels."""
    if spec in ELECTRODE_GROUPS:
        return ELECTRODE_GROUPS[spec], spec
    names = [ch.strip() for ch in spec.split(",") if ch.strip()]
    return names, "-".join(names)


def resolve_waveforms(spec):
    if spec is None:
        return WAVEFORM_ORDER
    chosen = [w.strip() for w in spec.split(",") if w.strip()]
    unknown = [w for w in chosen if w not in WAVEFORM_TRIGGERS]
    if unknown:
        raise ValueError(f"Unknown waveform(s) {unknown}; choices are {WAVEFORM_ORDER}")
    return [w for w in WAVEFORM_ORDER if w in chosen]


def parse_subjects(spec):
    if spec is None:
        return ALL_SUBJECTS
    return [int(s.strip()) for s in spec.split(",") if s.strip()]


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def load_subject_epochs(data_dir, fname_template, subject, picks, band_name):
    """Load one subject's preprocessed epochs, band-filter, restrict channels.

    Returns None (with a printed warning) if the file is missing, rather
    than raising -- a single absent subject shouldn't abort a cohort plot.
    """
    fpath = os.path.join(data_dir, fname_template.format(sid=subject))
    if not os.path.exists(fpath):
        print(f"  [sub-{subject}] WARNING: file not found, skipping: {fpath}")
        return None

    epochs = mne.read_epochs(fpath, preload=True, verbose=False)
    epochs, _band_info = apply_frequency_band(epochs, band_name)

    if picks:
        available = [ch for ch in picks if ch in epochs.ch_names]
        missing = set(picks) - set(available)
        if missing:
            print(f"  [sub-{subject}] Note: channels not found, skipping: {sorted(missing)}")
        if not available:
            print(f"  [sub-{subject}] WARNING: none of the requested channels present, skipping subject")
            return None
        epochs = epochs.copy().pick_channels(available)

    return epochs


def collect_evokeds_by_waveform(data_dir, fname_template, subjects, picks, band_name, waveforms):
    """{waveform_name: [per-subject Evoked, ...]} for the between-subject CI band."""
    evokeds_by_waveform = {wf: [] for wf in waveforms}
    n_loaded = 0

    for subject in subjects:
        epochs = load_subject_epochs(data_dir, fname_template, subject, picks, band_name)
        if epochs is None:
            continue
        n_loaded += 1

        for wf in waveforms:
            trigger = WAVEFORM_TRIGGERS[wf]
            epochs_wf = epochs[epochs.events[:, 2] == trigger]
            if len(epochs_wf) == 0:
                print(f"  [sub-{subject}] Note: no '{wf}' (trigger {trigger}) epochs found")
                continue
            evoked = epochs_wf.average()
            evoked.comment = f"sub-{subject}"
            evokeds_by_waveform[wf].append(evoked)

    return evokeds_by_waveform, n_loaded


def plot_roi_epochs(evokeds_by_waveform, waveforms, ci, title, save_path):
    evoked_dict = {wf: evokeds_by_waveform[wf] for wf in waveforms if evokeds_by_waveform[wf]}
    dropped = [wf for wf in waveforms if not evokeds_by_waveform[wf]]
    if dropped:
        print(f"  Note: no data for {dropped}, omitting from plot")
    if not evoked_dict:
        raise RuntimeError("No evoked data available for any requested waveform/subject -- nothing to plot")

    fig = mne.viz.plot_compare_evokeds(
        evoked_dict,
        picks="all",
        combine="mean",
        ci=ci,
        colors={wf: WAVEFORM_COLORS[wf] for wf in evoked_dict},
        title=title,
        show=False,
        show_sensors=False,
        truncate_yaxis="auto",
    )
    ax = fig[0].axes[0]
    ax.axvline(x=0, color="k", linestyle="--", alpha=0.5, linewidth=1.2)
    ax.axhline(y=0, color="k", linestyle="-", alpha=0.3, linewidth=0.5)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (µV)")
    ax.grid(True, alpha=0.3)
    fig[0].tight_layout()
    fig[0].savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True, choices=list(VARIANTS.keys()))
    ap.add_argument("--subjects", default=None, help="comma-separated subject IDs, default all of 3-12")
    ap.add_argument("--electrodes", default="all",
                    help=f"ROI name ({', '.join(ELECTRODE_GROUPS.keys())}) or comma-separated channel names")
    ap.add_argument("--band", default="broadband", choices=list(FREQUENCY_BANDS.keys()))
    ap.add_argument("--waveforms", default=None,
                    help=f"comma-separated subset of {WAVEFORM_ORDER}, default all")
    ap.add_argument("--ci", type=float, default=0.90)
    ap.add_argument("--outdir", default=OUTDIR_DEFAULT)
    args = ap.parse_args()

    variant = VARIANTS[args.variant]
    subjects = parse_subjects(args.subjects)
    picks, electrodes_label = resolve_electrodes(args.electrodes)
    waveforms = resolve_waveforms(args.waveforms)

    print(f"Variant: {args.variant} ({variant['label']})  |  data dir: {variant['data_dir']}")
    print(f"Subjects: {subjects}")
    print(f"Electrodes: {args.electrodes} -> {picks if picks else 'ALL'}")
    print(f"Band: {args.band}  |  Waveforms: {waveforms}  |  CI: {args.ci}")

    evokeds_by_waveform, n_loaded = collect_evokeds_by_waveform(
        variant["data_dir"], variant["fname_template"], subjects, picks, args.band, waveforms)

    os.makedirs(args.outdir, exist_ok=True)
    ci_pct = int(round(args.ci * 100))
    band_label = args.band.capitalize()
    title = (f"Task4 [{variant['label']}] -- {electrodes_label} -- {band_label} "
            f"(n={n_loaded} subjects, {ci_pct}% CI)")
    fname = f"{args.variant}_{electrodes_label}_{args.band}_ROI_epoch.png"
    save_path = os.path.join(args.outdir, fname)

    plot_roi_epochs(evokeds_by_waveform, waveforms, args.ci, title, save_path)

    for wf in waveforms:
        print(f"  {wf}: {len(evokeds_by_waveform[wf])}/{n_loaded} subjects contributed")
    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()
