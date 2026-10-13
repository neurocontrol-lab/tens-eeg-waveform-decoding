"""
August28_Task4_Preprocess_AutoICA33.py
==========================================
Expanded-ICA variant of August27_Task4_Preprocess_AutoICA.py

Fits ICA fresh at 33 components (the existing 15-component cache in
August27_Task4_ICA_ManualReview_cache/ is NOT reused -- it's a different
decomposition entirely) but REUSES the existing per-subject calibrated
EOG threshold from that same cache (August27_manual_sub-{N}_calibration.json)
-- the threshold grid search operates purely on the Fp1 channel and is
independent of how many ICA components are fit, so recomputing it here
would be redundant.

New, separate cache: August28_Task4_ICA_Cache33/ (33-component ICA fits +
their preprocessed raw, so this script is re-runnable/idempotent like its
15-component counterpart). The SAME 33-comp fit is reused for both
--muscle and non---muscle runs (only the exclusion DECISION differs, not
the decomposition itself), mirroring how August27_Task4_Preprocess_AutoICA.py
handles the 15-comp case.

Output (EOG-only, default):
  August28_Task4_Preprocessed_Data_AutoEOG33/
  August28_Task4_ICA_Components_AutoEOG33/
Output (--muscle):
  August28_Task4_Preprocessed_Data_AutoEOG33Muscle/
  August28_Task4_ICA_Components_AutoEOG33Muscle/

Usage:
  python August28_Task4_Preprocess_AutoICA33.py
  python August28_Task4_Preprocess_AutoICA33.py --muscle
  python August28_Task4_Preprocess_AutoICA33.py --subjects 3,4
  python August28_Task4_Preprocess_AutoICA33.py --refit   # ignore the 33-comp cache, refit
"""
import os, sys, argparse, warnings
import numpy as np
import pandas as pd
import mne
warnings.filterwarnings("ignore")
mne.set_log_level("WARNING")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from August27_Task4_ICA_ManualReview_Common import (
    SUBJECT_FILES, EOG_REF_CHANNEL, ZSCORE_THRESHOLD,
    prepare_raw_for_ica, calibrate_and_suggest,
)
from preprocess_paper_method import blank_around_events, STIM_LP

N_COMPONENTS = 33
CACHE_DIR = "August28_Task4_ICA_Cache33"

BLANK_PRE_MS, BLANK_POST_MS = 2.0, 10.0
# Jason Whitson's clarification supplied 2026-10-13: higher ramp code = LinInc.
TRIGGER_MAPPING = {"Sine": 239, "Square": 240, "Triangle": 241, "LinInc": 243, "LinDec": 242}
TMIN, TMAX, BASELINE = -0.250, 0.500, (-0.250, 0)


def raw33_path(sid):
    return os.path.join(CACHE_DIR, f"August28_sub-{sid}_raw-preica.fif")


def ica33_path(sid):
    return os.path.join(CACHE_DIR, f"August28_sub-{sid}_ica33.fif")


def fit_or_load_ica33(sid, force_refit=False):
    os.makedirs(CACHE_DIR, exist_ok=True)
    rpath, ipath = raw33_path(sid), ica33_path(sid)
    if not force_refit and os.path.exists(rpath) and os.path.exists(ipath):
        raw = mne.io.read_raw_fif(rpath, preload=True, verbose=False)
        ica = mne.preprocessing.read_ica(ipath, verbose=False)
        return raw, ica, True

    raw, _events = prepare_raw_for_ica(sid)
    r_hp = raw.copy().filter(1.0, None, fir_design="firwin", verbose=False)
    ica = mne.preprocessing.ICA(n_components=N_COMPONENTS, method="infomax",
                                fit_params=dict(extended=True),
                                random_state=42, max_iter="auto")
    ica.fit(r_hp)
    raw.save(rpath, overwrite=True)
    ica.save(ipath, overwrite=True)
    return raw, ica, False


def detect_exclusions(ica, r_hp, chosen_mult, use_muscle):
    ref_std = r_hp.get_data(picks=[EOG_REF_CHANNEL])[0].std()
    eog_epochs = mne.preprocessing.create_eog_epochs(
        r_hp, ch_name=EOG_REF_CHANNEL, thresh=chosen_mult * ref_std, verbose=False)
    idx, _ = ica.find_bads_eog(eog_epochs, ch_name=EOG_REF_CHANNEL,
                               threshold=ZSCORE_THRESHOLD, verbose=False)
    eog_excl = sorted(int(c) for c in idx)
    excl = set(eog_excl)

    muscle_excl = []
    if use_muscle:
        try:
            midx, _ = ica.find_bads_muscle(r_hp, verbose=False)
            muscle_excl = sorted(int(c) for c in midx)
            excl |= set(muscle_excl)
        except Exception as e:
            print(f"    [warn] find_bads_muscle failed: {e}")

    return sorted(excl), eog_excl, muscle_excl


def process_subject(sid, force_refit, use_muscle, output_dir, plot_dir, tag):
    print(f"\n{'='*70}\nSUBJECT {sid} ({tag}, N_COMPONENTS={N_COMPONENTS})\n{'='*70}")

    raw, ica, from_cache = fit_or_load_ica33(sid, force_refit=force_refit)
    print(f"  ICA33 {'loaded from cache' if from_cache else 'fit fresh'} ({ica.n_components_} comps)")

    r_hp, chosen_mult, chosen_rate = calibrate_and_suggest(raw, subject_id=sid)
    print(f"  calibrated threshold (reused from 15-comp cache): {chosen_mult:.3f}xstd "
         f"-> ~{chosen_rate:.1f} blinks/min")

    excl, eog_excl, muscle_excl = detect_exclusions(ica, r_hp, chosen_mult, use_muscle)
    ica = ica.copy()
    ica.exclude = excl
    print(f"  EOG-flagged: {eog_excl}" + (f"  |  muscle-flagged: {muscle_excl}" if use_muscle else ""))
    print(f"  FINAL exclusion: {excl}  ({len(excl)}/{ica.n_components_})")

    raw_clean = raw.copy()
    ica.apply(raw_clean, verbose=False)

    from eeg_waveform_analysis_with_save import load_xdf_to_mne
    _, events = load_xdf_to_mne(SUBJECT_FILES[sid])

    raw_clean = blank_around_events(raw_clean, events, BLANK_PRE_MS, BLANK_POST_MS)
    raw_clean.filter(l_freq=None, h_freq=STIM_LP, verbose=False)

    present = set(int(c) for c in events[:, 2])
    event_dict = {k: v for k, v in TRIGGER_MAPPING.items() if v in present}
    epochs = mne.Epochs(raw_clean, events, event_id=event_dict, tmin=TMIN, tmax=TMAX,
                        baseline=BASELINE, preload=True, reject_by_annotation=True,
                        verbose=False)
    print(f"  Epochs: {len(epochs)} | conditions: {list(epochs.event_id.keys())}")

    os.makedirs(output_dir, exist_ok=True)
    raw_clean.save(os.path.join(output_dir, f"August28_{tag}_sub-{sid}_raw-preprocessed.fif"),
                   overwrite=True)
    epochs.save(os.path.join(output_dir, f"August28_{tag}_sub-{sid}_epo.fif"), overwrite=True)
    ica.save(os.path.join(output_dir, f"August28_{tag}_sub-{sid}_ica.fif"), overwrite=True)

    os.makedirs(plot_dir, exist_ok=True)
    figs = ica.plot_components(inst=r_hp, picks=range(ica.n_components_), show=False)
    if not isinstance(figs, list):
        figs = [figs]
    for i, fig in enumerate(figs):
        suffix = f"_page{i+1}" if len(figs) > 1 else ""
        fig.set_size_inches(14, 10)
        muscle_note = f" | muscle-flagged: {muscle_excl}" if use_muscle else ""
        fig.suptitle(
            f"Subject {sid} ({tag}, {N_COMPONENTS} comps): calibrated EOG mult={chosen_mult:.2f}xstd "
            f"(~{chosen_rate:.1f}/min): gray = EXCLUDED, bold black = kept  |  "
            f"EOG-flagged: {eog_excl}{muscle_note}  |  FINAL excluded: {excl}",
            fontsize=8, y=1.02)
        fname = os.path.join(plot_dir, f"August28_ICA_components_sub-{sid}{suffix}.png")
        fig.savefig(fname, dpi=140, bbox_inches="tight", pad_inches=0.3)
        plt.close(fig)
    print(f"  Saved sub-{sid} to {output_dir}/ and component plot to {plot_dir}/")

    return dict(subject=sid, n_epochs=len(epochs), n_components=ica.n_components_,
               calibrated_mult=chosen_mult, calibrated_rate=chosen_rate,
               eog_excluded=eog_excl, muscle_excluded=muscle_excl,
               final_excluded=excl, n_excluded=len(excl))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--muscle", action="store_true",
                    help="Also run find_bads_muscle and union it into the exclusion set")
    ap.add_argument("--subjects", type=str, default=None,
                    help="Comma-separated subject IDs (default: all 3-12)")
    ap.add_argument("--refit", action="store_true",
                    help="Ignore the 33-comp ICA cache and refit everyone")
    args = ap.parse_args()

    tag = "autoeog33muscle" if args.muscle else "autoeog33"
    output_dir = f"August28_Task4_Preprocessed_Data_{'AutoEOG33Muscle' if args.muscle else 'AutoEOG33'}"
    plot_dir = f"August28_Task4_ICA_Components_{'AutoEOG33Muscle' if args.muscle else 'AutoEOG33'}"

    subs = sorted(SUBJECT_FILES.keys())
    if args.subjects:
        want = set(int(s) for s in args.subjects.split(","))
        subs = [s for s in subs if s in want]

    print(f"Running AutoICA33 preprocessing ({'EOG+muscle' if args.muscle else 'EOG-only'}, "
         f"{N_COMPONENTS} components) for subjects: {subs}")

    rows, failed = [], []
    for sid in subs:
        try:
            rows.append(process_subject(sid, args.refit, args.muscle, output_dir, plot_dir, tag))
        except Exception as e:
            import traceback
            print(f"  ERROR sub {sid}: {e}")
            traceback.print_exc()
            failed.append(sid)

    df = pd.DataFrame(rows)
    os.makedirs(output_dir, exist_ok=True)
    summary_csv = os.path.join(output_dir, f"August28_{tag}_summary.csv")
    df.to_csv(summary_csv, index=False)
    print(f"\n{'='*70}\nDone. OK: {[r['subject'] for r in rows]}  |  Failed: {failed}")
    print(f"Output: {output_dir}/  |  Plots: {plot_dir}/  |  Summary: {summary_csv}")


if __name__ == "__main__":
    main()
