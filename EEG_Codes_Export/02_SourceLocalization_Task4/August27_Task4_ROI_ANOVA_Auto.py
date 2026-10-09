"""
August27_Task4_ROI_ANOVA_Auto.py
====================================
Source-localization + ROI-ANOVA driver for the Task4 waveform-type (5-
condition) comparison, run against this session's new automated ICA-fix
preprocessing (August27_Task4_Preprocess_AutoICA.py). Adapted from
August_25_Task4_ROI_ANOVA_ICAfix.py (the most recent/best version in the
prior investigation chain) -- keeps its rank-informed noise-covariance/
inverse fix (rank=dict(eeg=ica.n_components_-len(exclude)-1), verified in
that investigation to fix near-singular noise covariance from MNE's
compute_rank(rank=None) reporting a flat wrong rank of 61 regardless of
actual ICA aggressiveness) unchanged, since that fix is orthogonal to
which ICA-exclusion method produced the data.

Parameterized via --variant so the SAME script runs against either of
this session's two new preprocessing outputs (letting you directly
compare EOG-only vs EOG+muscle exclusion downstream, as requested):
  --variant autoeog        -> August27_Task4_Preprocessed_Data_AutoEOG/
  --variant autoeogmuscle  -> August27_Task4_Preprocessed_Data_AutoEOGMuscle/



Outputs nest under August27_Task4_SourceLocalization_{Variant}/waveform_type/:
  roi_anova_sensorimotor/{dSPM,eLORETA}/...
  roi_anova_all68/{dSPM,eLORETA}/...

Run : python August27_Task4_ROI_ANOVA_Auto.py --variant autoeog
      python August27_Task4_ROI_ANOVA_Auto.py --variant autoeogmuscle
      python August27_Task4_ROI_ANOVA_Auto.py --variant autoeog --rois all --method eLORETA
      python August27_Task4_ROI_ANOVA_Auto.py --variant autoeog --smoke
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mne

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from task3_source_loc import (
    build_forward_model, apply_inverse_stc,
    load_dk_atlas_labels, compute_region_activation_matrix,
)
from task4_source_loc import WAVEFORM_TRIGGERS, load_condition_epochs_task4
from task3_erp_stats import rm_anova, friedman_test, posthoc_pairwise, cluster_test_rm
from task3_erp_peaks import window_mean_amplitude
from task3_roi_erp import sliding_windows, build_significance_ranking

mne.set_log_level("ERROR")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VARIANTS = {
    "autoeog": dict(
        data_dir="August27_Task4_Preprocessed_Data_AutoEOG",
        fname_template="August27_autoeog_sub-{sid}_epo.fif",
        ica_fname_template="August27_autoeog_sub-{sid}_ica.fif",
        source_loc_root=os.path.join("August27_Task4_SourceLocalization_AutoEOG", "waveform_type"),
        label="EOG-only (15 comps)",
        file_prefix="August27",
    ),
    "autoeogmuscle": dict(
        data_dir="August27_Task4_Preprocessed_Data_AutoEOGMuscle",
        fname_template="August27_autoeogmuscle_sub-{sid}_epo.fif",
        ica_fname_template="August27_autoeogmuscle_sub-{sid}_ica.fif",
        source_loc_root=os.path.join("August27_Task4_SourceLocalization_AutoEOGMuscle", "waveform_type"),
        label="EOG+muscle (15 comps)",
        file_prefix="August27",
    ),
    "autoeog33": dict(
        data_dir="August28_Task4_Preprocessed_Data_AutoEOG33",
        fname_template="August28_autoeog33_sub-{sid}_epo.fif",
        ica_fname_template="August28_autoeog33_sub-{sid}_ica.fif",
        source_loc_root=os.path.join("August28_Task4_SourceLocalization_AutoEOG33", "waveform_type"),
        label="EOG-only, 33 comps",
        file_prefix="August28",
    ),
    "autoeog33muscle": dict(
        data_dir="August28_Task4_Preprocessed_Data_AutoEOG33Muscle",
        fname_template="August28_autoeog33muscle_sub-{sid}_epo.fif",
        ica_fname_template="August28_autoeog33muscle_sub-{sid}_ica.fif",
        source_loc_root=os.path.join("August28_Task4_SourceLocalization_AutoEOG33Muscle", "waveform_type"),
        label="EOG+muscle, 33 comps",
        file_prefix="August28m",
    ),
    "manualica33": dict(
        data_dir="August28_Task4_Preprocessed_Data_ManualICA33",
        fname_template="August28_manualica33_sub-{sid}_epo.fif",
        ica_fname_template="August28_manualica33_sub-{sid}_ica.fif",
        source_loc_root=os.path.join("August28_Task4_SourceLocalization_ManualICA33", "waveform_type"),
        label="Manual review, 33 comps",
        file_prefix="August28man",
    ),
}

SESSION = "Task4"
SUBJECTS = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
SCALE = 1.0  # data already correctly scaled at the preprocessing stage -- see docstring

WAVEFORMS = {name: [trig] for name, trig in WAVEFORM_TRIGGERS.items()}
WAVEFORM_ORDER = ["Sine", "Square", "Triangle", "LinInc", "LinDec"]
WAVEFORM_COLORS = {"Sine": "#1f77b4", "Square": "#ff7f0e", "Triangle": "#2ca02c",
                   "LinInc": "#d62728", "LinDec": "#9467bd"}

ROI_NAMES = ["precentral-lh", "precentral-rh", "postcentral-lh", "postcentral-rh",
            "paracentral-lh", "paracentral-rh"]
METHODS = ("dSPM", "sLORETA", "eLORETA")
METHOD = "dSPM"
SNR = 3.0
LAMBDA2 = 1.0 / SNR ** 2
ALPHA = 0.05
WINDOW_WIDTH_MS = 50.0
WINDOW_STEP_MS = 25.0
N_PERM = 2000

CI90_Z = 1.645
ARTIFACT_ZONE_MS = (100.0, 140.0)

_RANK_CACHE = {}
DATA_DIR = None
FNAME_TEMPLATE = None
ICA_FNAME_TEMPLATE = None
VARIANT_LABEL = None
FILE_PREFIX = None


def ensure_dirs(outdir):
    os.makedirs(os.path.join(outdir, "figures"), exist_ok=True)


def true_rank_for_subject(subj):
    if subj not in _RANK_CACHE:
        ica = mne.preprocessing.read_ica(
            os.path.join(DATA_DIR, ICA_FNAME_TEMPLATE.format(sid=subj)), verbose=False)
        _RANK_CACHE[subj] = max(1, ica.n_components_ - len(ica.exclude) - 1)
    return _RANK_CACHE[subj]


def compute_noise_cov_ranked(epochs, rank, tmax=0.0, method="shrunk"):
    return mne.compute_covariance(epochs, tmax=tmax, method=method, rank=rank, verbose=False)


def make_inverse_ranked(info, fwd, noise_cov, rank, loose=0.2, depth=0.8):
    from mne.minimum_norm import make_inverse_operator
    return make_inverse_operator(info, fwd, noise_cov, loose=loose, depth=depth,
                                 rank=rank, verbose=False)


def compute_per_subject_roi_matrices(subjects, roi_labels, fwd, src):
    roi_names = [lab.name for lab in roi_labels]
    data = {name: {wf: [] for wf in WAVEFORMS} for name in roi_names}
    times = None
    rank_by_subject = {}

    for subj in subjects:
        print(f"  subject {subj}...")
        true_rank = true_rank_for_subject(subj)
        rank_by_subject[subj] = true_rank
        for wf, trigger_codes in WAVEFORMS.items():
            epochs = load_condition_epochs_task4(
                [subj], DATA_DIR, SESSION, trigger_codes,
                fname_template=FNAME_TEMPLATE, scale=SCALE)
            evoked = epochs.average()
            if times is None:
                times = evoked.times.copy()

            noise_cov = compute_noise_cov_ranked(epochs, rank=dict(eeg=true_rank), tmax=0.0)
            inv = make_inverse_ranked(evoked.info, fwd, noise_cov, rank=dict(eeg=true_rank))
            stc = apply_inverse_stc(evoked, inv, method=METHOD, lambda2=LAMBDA2)

            names, matrix = compute_region_activation_matrix(stc, roi_labels, src)
            for i, name in enumerate(names):
                data[name][wf].append(matrix[i])

    for name in roi_names:
        for wf in WAVEFORMS:
            data[name][wf] = np.array(data[name][wf])
    return data, times, rank_by_subject


def _cluster_to_mask(cluster, n_times):
    mask = np.zeros(n_times, dtype=bool)
    mask[cluster] = True
    return mask


def _merge_spans(times_ms, mask):
    spans, in_run, start = [], False, None
    for i, m in enumerate(mask):
        if m and not in_run:
            in_run, start = True, times_ms[i]
        elif not m and in_run:
            in_run = False
            spans.append((start, times_ms[i - 1]))
    if in_run:
        spans.append((start, times_ms[-1]))
    return spans


def analyze_roi(roi_name, per_wf, times, n_perm, skip_cluster=False):
    times_ms = times * 1e3
    conds = WAVEFORM_ORDER
    n_subjects = per_wf[conds[0]].shape[0]

    stat_rows, posthoc_rows = [], []
    sig_windows_ms = []

    for lo, hi in sliding_windows(times_ms[0], times_ms[-1], WINDOW_WIDTH_MS, WINDOW_STEP_MS):
        mat = np.full((n_subjects, len(conds)), np.nan)
        for ci, wf in enumerate(conds):
            for si in range(n_subjects):
                mat[si, ci] = window_mean_amplitude(times_ms, per_wf[wf][si], lo, hi)
        if np.isnan(mat).any():
            continue

        an = rm_anova(mat)
        try:
            fr_stat, fr_p = friedman_test(mat)
        except Exception:
            fr_stat, fr_p = np.nan, np.nan
        sig = bool(np.isfinite(an["p"]) and an["p"] < ALPHA)
        means = mat.mean(axis=0)
        order = " > ".join(np.array(conds)[np.argsort(-means)])
        stat_rows.append({
            "roi": roi_name, "window_lo_ms": round(lo, 1), "window_hi_ms": round(hi, 1),
            "F": round(an["F"], 3) if np.isfinite(an["F"]) else np.nan,
            "p": round(an["p"], 5) if np.isfinite(an["p"]) else np.nan,
            "eta2_partial": round(an["eta2_partial"], 3) if np.isfinite(an["eta2_partial"]) else np.nan,
            "friedman_p": round(fr_p, 5) if np.isfinite(fr_p) else np.nan,
            "significant": sig, "ordering": order,
        })
        if sig:
            sig_windows_ms.append((lo, hi))
            for ph in posthoc_pairwise(mat, conds):
                posthoc_rows.append({
                    "roi": roi_name, "window_lo_ms": round(lo, 1), "window_hi_ms": round(hi, 1),
                    "pair": f"{ph['a']} vs {ph['b']}",
                    "dz": round(ph["dz"], 3), "p_holm": round(ph["p_holm"], 5),
                })

    cluster_rows = []
    if skip_cluster:
        return stat_rows, posthoc_rows, cluster_rows, sig_windows_ms
    for i in range(len(conds)):
        for j in range(i + 1, len(conds)):
            a, b = per_wf[conds[i]], per_wf[conds[j]]
            try:
                _, clusters, cl_pv = cluster_test_rm(a, b, n_permutations=n_perm)
            except Exception:
                clusters, cl_pv = [], []
            for cl, p in zip(clusters, cl_pv):
                if p < ALPHA:
                    mask = _cluster_to_mask(cl, len(times))
                    for (t0, t1) in _merge_spans(times_ms, mask):
                        cluster_rows.append({
                            "roi": roi_name, "pair": f"{conds[i]} vs {conds[j]}",
                            "tmin_ms": round(float(t0), 1), "tmax_ms": round(float(t1), 1),
                            "cluster_p": round(float(p), 4),
                        })

    return stat_rows, posthoc_rows, cluster_rows, sig_windows_ms


def plot_roi_comparison(roi_name, per_wf, times, sig_windows_ms, fname, band_mode="ci90"):
    band_z = CI90_Z if band_mode == "ci90" else 1.0
    band_label = "90% CI" if band_mode == "ci90" else "SEM"

    times_ms = times * 1e3
    fig, ax = plt.subplots(figsize=(10, 5))
    for lo, hi in sig_windows_ms:
        ax.axvspan(lo, hi, color="gray", alpha=0.15, lw=0)
    for wf in WAVEFORM_ORDER:
        mean = per_wf[wf].mean(axis=0)
        sem = per_wf[wf].std(axis=0, ddof=1) / np.sqrt(per_wf[wf].shape[0])
        band = band_z * sem
        ax.plot(times_ms, mean, color=WAVEFORM_COLORS[wf], label=wf, lw=1.5)
        ax.fill_between(times_ms, mean - band, mean + band, color=WAVEFORM_COLORS[wf], alpha=0.15)
    ax.axvline(0.0, color="black", lw=1, ls=":")
    ax.set(xlabel="time (ms)", ylabel=f"{METHOD} value (mean ± {band_label})")
    ax.set_title(f"[August27 AutoICA, {VARIANT_LABEL}, rank-informed] {roi_name} -- Task4 "
                f"waveform-type comparison (shaded = ANOVA p<{ALPHA} window)", fontsize=9)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)


def plot_significance_ranking(ranking, fname, top_n=None):
    from task3_source_loc import region_base_and_hemi, DK_LOBE_MAP, _LOBE_COLORS

    rows = ranking[:top_n] if top_n else ranking
    names = [r["roi"] for r in rows]
    counts = [r["n_significant"] for r in rows]
    colors = [_LOBE_COLORS[DK_LOBE_MAP[region_base_and_hemi(n)[0]]] for n in names]
    hatches = ["//" if r["any_artifact_zone_overlap"] else None for r in rows]

    fig, ax = plt.subplots(figsize=(10, max(4, len(rows) * 0.22)))
    y = np.arange(len(rows))[::-1]
    for yi, name, count, color, hatch in zip(y, names, counts, colors, hatches):
        ax.barh(yi, count, color=color, hatch=hatch, edgecolor="white" if hatch else color)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7)
    n_windows = ranking[0]["n_windows"] if ranking else 0
    ax.set_xlabel(f"significant sliding windows (out of {n_windows}, p<{ALPHA})")
    ax.set_title(f"[August27 AutoICA, {VARIANT_LABEL}] Task4 waveform-type ROI sweep -- "
                 "significant ANOVA windows, ranked (hatched = overlaps the ~100-140ms artifact zone)",
                 fontsize=8)
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)


def write_findings(fname, roi_names, subjects, rank_by_subject, all_stat_rows):
    with open(fname, "w") as f:
        f.write(f"# August27 AutoICA ({VARIANT_LABEL}) -- Task4 ROI ANOVA "
               "(waveform-type comparison) -- Findings\n\n")
        f.write(f"Method: {METHOD}, SNR={SNR}. Subjects: {subjects} (N={len(subjects)}).\n")
        f.write(f"True rank (eeg) passed to compute_covariance/make_inverse_operator per subject: "
               f"{rank_by_subject}\n")
        f.write(f"Preprocessing: August27_Task4_Preprocess_AutoICA.py "
               f"({'--muscle' if VARIANT_LABEL == 'EOG+muscle' else '(EOG-only)'}) -- see "
               "August27_Task4_EOG_ICA_Investigation_Report.md for the reference-fix + "
               "per-subject-calibrated-threshold methodology this preprocessing uses.\n")
        f.write(f"Sliding windows: {WINDOW_WIDTH_MS:.0f} ms wide, {WINDOW_STEP_MS:.0f} ms step, "
               "covering the whole epoch. ANOVA uncorrected across windows (exploratory).\n\n")
        for roi_name in roi_names:
            sig = [r for r in all_stat_rows if r["roi"] == roi_name and r["significant"]]
            f.write(f"## {roi_name}\n\n")
            if not sig:
                f.write("No sliding windows reached p<0.05.\n\n")
                continue
            for r in sig:
                f.write(f"- {r['window_lo_ms']:.0f}-{r['window_hi_ms']:.0f} ms: "
                       f"F={r['F']}, p={r['p']}, eta2p={r['eta2_partial']}, "
                       f"ordering: {r['ordering']}\n")
            f.write("\n")


def main():
    global METHOD, DATA_DIR, FNAME_TEMPLATE, ICA_FNAME_TEMPLATE, VARIANT_LABEL, FILE_PREFIX

    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=list(VARIANTS.keys()), required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--n-perm", type=int, default=N_PERM)
    parser.add_argument("--rois", choices=["sensorimotor", "all"], default="sensorimotor")
    parser.add_argument("--method", choices=METHODS, default="dSPM")
    parser.add_argument("--skip-cluster", action="store_true")
    parser.add_argument("--no-skip-cluster", action="store_true")
    args = parser.parse_args()

    v = VARIANTS[args.variant]
    DATA_DIR = v["data_dir"]
    FNAME_TEMPLATE = v["fname_template"]
    ICA_FNAME_TEMPLATE = v["ica_fname_template"]
    VARIANT_LABEL = v["label"]
    FILE_PREFIX = v["file_prefix"]
    source_loc_root = v["source_loc_root"]
    outdir_sensorimotor = os.path.join(source_loc_root, "roi_anova_sensorimotor")
    outdir_all68 = os.path.join(source_loc_root, "roi_anova_all68")

    METHOD = args.method
    subjects = SUBJECTS[:3] if args.smoke else SUBJECTS
    outdir_root = outdir_sensorimotor if args.rois == "sensorimotor" else outdir_all68
    outdir_root = os.path.join(outdir_root, METHOD)
    outdir = os.path.join(outdir_root, "_smoke") if args.smoke else outdir_root
    ensure_dirs(outdir)

    skip_cluster = args.skip_cluster or (args.rois == "all" and not args.no_skip_cluster)

    print(f"Variant: {args.variant} ({VARIANT_LABEL})  |  DATA_DIR: {DATA_DIR}")
    print(f"Subjects: {subjects}")
    print(f"ROI scope: {args.rois}")
    print(f"Method: {METHOD}")
    print("Rank-informed covariance/inverse: ON")

    probe_epochs = load_condition_epochs_task4(
        [subjects[0]], DATA_DIR, SESSION, WAVEFORMS[WAVEFORM_ORDER[0]],
        fname_template=FNAME_TEMPLATE, scale=SCALE)
    fwd, src, trans = build_forward_model(probe_epochs.info)
    fs_dir = mne.datasets.fetch_fsaverage(verbose=False)
    subjects_dir = os.path.dirname(fs_dir)

    all_labels = load_dk_atlas_labels("fsaverage", subjects_dir)
    if args.rois == "sensorimotor":
        roi_labels = [lab for lab in all_labels if lab.name in ROI_NAMES]
        roi_labels.sort(key=lambda lab: ROI_NAMES.index(lab.name))
    else:
        roi_labels = all_labels
    if args.smoke:
        roi_labels = roi_labels[:3] if args.rois == "all" else roi_labels[:1]
    roi_names = [lab.name for lab in roi_labels]
    print(f"Loaded {len(roi_labels)} target ROI labels: {roi_names}")

    print("\nComputing per-subject source-localized ROI time series...")
    data, times, rank_by_subject = compute_per_subject_roi_matrices(subjects, roi_labels, fwd, src)
    print(f"True rank per subject: {rank_by_subject}")

    all_stat_rows, all_posthoc_rows, all_cluster_rows = [], [], []
    for roi_name in roi_names:
        print(f"\n=== {roi_name} ===")
        stat_rows, posthoc_rows, cluster_rows, sig_windows_ms = analyze_roi(
            roi_name, data[roi_name], times, args.n_perm, skip_cluster=skip_cluster)
        n_sig = sum(r["significant"] for r in stat_rows)
        print(f"  {n_sig}/{len(stat_rows)} sliding windows significant (p<{ALPHA})")
        all_stat_rows += stat_rows
        all_posthoc_rows += posthoc_rows
        all_cluster_rows += cluster_rows

        fig_dir = os.path.join(outdir, "figures")
        plot_roi_comparison(roi_name, data[roi_name], times, sig_windows_ms,
                            os.path.join(fig_dir, f"{FILE_PREFIX}_{roi_name}_comparison.png"))
        plot_roi_comparison(roi_name, data[roi_name], times, sig_windows_ms,
                            os.path.join(fig_dir, f"{FILE_PREFIX}_{roi_name}_comparison_sem_shading.png"),
                            band_mode="sem")

    pd.DataFrame(all_stat_rows).to_csv(os.path.join(outdir, f"{FILE_PREFIX}_anova_by_window.csv"), index=False)
    pd.DataFrame(all_posthoc_rows).to_csv(os.path.join(outdir, f"{FILE_PREFIX}_posthoc_pairwise.csv"), index=False)
    pd.DataFrame(all_cluster_rows).to_csv(os.path.join(outdir, f"{FILE_PREFIX}_cluster_pairwise.csv"), index=False)
    write_findings(os.path.join(outdir, f"{FILE_PREFIX}_Findings.md"), roi_names, subjects,
                   rank_by_subject, all_stat_rows)

    ranking = build_significance_ranking(all_stat_rows, artifact_zone_ms=ARTIFACT_ZONE_MS)
    pd.DataFrame(ranking).to_csv(os.path.join(outdir, f"{FILE_PREFIX}_roi_significance_ranking.csv"), index=False)
    plot_significance_ranking(ranking, os.path.join(outdir, f"{FILE_PREFIX}_roi_significance_ranking.png"))
    print("\nTop ROIs by significant-window count:")
    for r in ranking[:10]:
        flag = " [overlaps ~100-140ms artifact zone]" if r["any_artifact_zone_overlap"] else ""
        print(f"  {r['roi']:20s} {r['n_significant']:2d}/{r['n_windows']}{flag}")

    print(f"\nDone. Outputs in {outdir}/")


if __name__ == "__main__":
    main()
