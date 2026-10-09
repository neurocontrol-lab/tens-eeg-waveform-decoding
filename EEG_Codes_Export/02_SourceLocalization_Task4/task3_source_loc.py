"""
task3_source_loc.py
====================
Helper functions for MNE dSPM/sLORETA source localization of Task 3
grand-average evoked responses, replicating
https://mne.tools/stable/auto_tutorials/inverse/30_mne_dspm_loreta.html
on a template (fsaverage) anatomy since no individual subject MRIs exist.

Pure/testable logic (no file or 3D-rendering I/O):
    select_waveform_mask, subsample_source_data, percentile_clim,
    filter_cortical_labels, _unit, _choose_best_view, draw_marker

I/O and MNE-object-heavy functions (exercised by the orchestrator /
smoke test rather than unit tests):
    load_condition_epochs, build_forward_model, compute_noise_cov,
    make_inverse, apply_inverse_stc, save_evoked_plot, save_whitened_plot,
    save_dipole_activation_plot, save_source_video,
    load_dk_atlas_labels, compute_region_activation_matrix,
    save_region_matrix_csv
"""

from __future__ import annotations

import os

import numpy as np


# ---------------------------------------------------------------------------
# Pure logic
# ---------------------------------------------------------------------------

def select_waveform_mask(codes, trigger_codes):
    """Boolean mask selecting the epochs whose trigger code is in
    ``trigger_codes``.

    Parameters
    ----------
    codes : array-like of int, shape (n_epochs,)
        Trigger code of each epoch (e.g. ``epochs.events[:, 2]``).
    trigger_codes : sequence of int
        Trigger codes belonging to one waveform (its Base/Mid/High triggers).

    Returns
    -------
    ndarray of bool, shape (n_epochs,)

    Raises
    ------
    ValueError
        If ``trigger_codes`` is empty or matches no epoch.
    """
    if not trigger_codes:
        raise ValueError("trigger_codes must not be empty")
    codes = np.asarray(codes)
    mask = np.isin(codes, list(trigger_codes))
    if not mask.any():
        raise ValueError(f"none of the trigger codes {trigger_codes} were found")
    return mask


def subsample_source_data(data, stride=100):
    """Subsample source-estimate data along the vertex axis for a readable
    activation-time-course plot (mirrors the tutorial's
    ``stc.data[::100, :]``).

    Parameters
    ----------
    data : ndarray, shape (n_vertices, n_times)
    stride : int
        Keep every ``stride``-th vertex.

    Returns
    -------
    ndarray, shape (ceil(n_vertices / stride), n_times)
    """
    if stride < 1:
        raise ValueError("stride must be >= 1")
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError("data must be 2-D (n_vertices, n_times)")
    return data[::stride, :]


def percentile_clim(data, percentiles=(96.0, 97.5, 99.9)):
    """Data-driven (lo, mid, hi) color limits for a source-estimate plot.

    Using percentiles of ``|data|`` (rather than the tutorial's hard-coded
    MEG-scale ``[8, 12, 15]``) keeps the color scale sensible regardless of
    the inverse method's numeric range (dSPM vs sLORETA differ by orders of
    magnitude).

    Parameters
    ----------
    data : ndarray
        Source-estimate data (any shape; flattened internally).
    percentiles : sequence of 3 floats, increasing
        Percentiles of ``|data|`` to use for (lo, mid, hi).

    Returns
    -------
    list of 3 floats, strictly increasing
    """
    if len(percentiles) != 3:
        raise ValueError("percentiles must have exactly 3 values")
    mag = np.abs(np.asarray(data)).ravel()
    lims = list(np.percentile(mag, percentiles))
    # guard against degenerate (flat/near-zero) data collapsing the scale
    for i in range(1, 3):
        if lims[i] <= lims[i - 1]:
            lims[i] = lims[i - 1] + 1e-12
    return lims


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

# The preprocessed .fif files store EEG amplitudes in microvolts numerically,
# but the FIFF channel metadata declares FIFF_UNIT_V (volts) -- the same
# mislabeling bug documented for preprocessed_data_new_baseline (see project
# memory). Left uncorrected, MNE's forward/noise-covariance/inverse math
# (which trusts the declared unit, not the numbers) treats a ~10 uV evoked
# response as a ~10 V signal, corrupting SNR regularization and the dSPM/
# sLORETA normalization. Multiplying by this factor makes the stored values
# true SI volts, matching the declared unit.
UV_MISLABELED_AS_V_SCALE = 1e-6


def load_condition_epochs(subjects, data_dir, session, trigger_codes,
                          fname_template="sub-{sid}_ses-{session}_epo.fif",
                          scale=UV_MISLABELED_AS_V_SCALE):
    """Load and pool one waveform condition's epochs across subjects.

    Parameters
    ----------
    subjects : sequence of int
    data_dir : str
    session : str
    trigger_codes : sequence of int
        Trigger codes belonging to the waveform (already corrected for the
        LinInc/LinDec label swap -- pass raw trigger integers, not names).
    fname_template : str
    scale : float
        Applied to every channel's data to correct the mislabeled-unit bug
        (see ``UV_MISLABELED_AS_V_SCALE``). Pass 1.0 to disable.

    Returns
    -------
    mne.Epochs
        Concatenated epochs (all matching trials, all subjects), preloaded,
        with corrected (true-volt) amplitudes.
    """
    import mne

    chunks = []
    for sid in subjects:
        fname = os.path.join(data_dir, fname_template.format(sid=sid, session=session))
        if not os.path.exists(fname):
            print(f"  [skip] missing {fname}")
            continue
        ep = mne.read_epochs(fname, preload=True, verbose=False)
        mask = select_waveform_mask(ep.events[:, 2], trigger_codes)
        ep = ep[mask]
        if scale != 1.0:
            ep.apply_function(lambda x: x * scale, channel_wise=True, verbose=False)
        chunks.append(ep)
    if not chunks:
        raise RuntimeError("no subject files were found / matched")
    return mne.concatenate_epochs(chunks, add_offset=False, verbose=False)


# ---------------------------------------------------------------------------
# Forward / inverse model (fsaverage template anatomy)
# ---------------------------------------------------------------------------

def build_forward_model(info, subject="fsaverage", subjects_dir=None):
    """Build the fsaverage-template forward solution for the given montage.

    Parameters
    ----------
    info : mne.Info
        Info of the (average-referenced) data; must already have a montage
        with digitized electrode positions set.
    subject : str
    subjects_dir : str | None
        If None, uses ``mne.datasets.fetch_fsaverage`` to locate/download it.

    Returns
    -------
    fwd : mne.Forward
    src : mne.SourceSpaces
    trans : str
        Path to the fsaverage template trans file.
    """
    import mne

    if subjects_dir is None:
        fs_dir = mne.datasets.fetch_fsaverage(verbose=False)
        subjects_dir = os.path.dirname(fs_dir)
    else:
        fs_dir = os.path.join(subjects_dir, subject)

    trans = os.path.join(fs_dir, "bem", f"{subject}-trans.fif")
    src = mne.read_source_spaces(os.path.join(fs_dir, "bem", f"{subject}-ico-5-src.fif"))
    bem = os.path.join(fs_dir, "bem", f"{subject}-5120-5120-5120-bem-sol.fif")

    fwd = mne.make_forward_solution(
        info, trans=trans, src=src, bem=bem, eeg=True, meg=False,
        mindist=5.0, verbose=False,
    )
    return fwd, src, trans


def compute_noise_cov(epochs, tmax=0.0, method="shrunk"):
    """Regularized baseline noise covariance from the pre-stimulus window."""
    import mne

    return mne.compute_covariance(epochs, tmax=tmax, method=method, verbose=False)


def make_inverse(info, fwd, noise_cov, loose=0.2, depth=0.8):
    """Build the (method-agnostic) inverse operator."""
    from mne.minimum_norm import make_inverse_operator

    return make_inverse_operator(info, fwd, noise_cov, loose=loose, depth=depth,
                                 verbose=False)


def apply_inverse_stc(evoked, inverse_operator, method="dSPM", lambda2=1.0 / 9.0,
                      pick_ori=None):
    """Apply the inverse operator to an evoked response.

    Parameters
    ----------
    method : {"dSPM", "sLORETA", "eLORETA", "MNE"}
    lambda2 : float
        Regularization, ``1 / SNR**2`` (tutorial default SNR=3).
    """
    from mne.minimum_norm import apply_inverse

    return apply_inverse(evoked, inverse_operator, lambda2=lambda2, method=method,
                         pick_ori=pick_ori, verbose=False)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def save_evoked_plot(evoked, fname, title=None):
    """Save the butterfly plot of all EEG channels (tutorial's
    ``evoked.plot()``)."""
    fig = evoked.plot(show=False, verbose=False)
    if title:
        fig.suptitle(title)
    fig.savefig(fname, dpi=150)
    import matplotlib.pyplot as plt
    plt.close(fig)


def save_whitened_plot(evoked, noise_cov, fname, title=None):
    """Save the whitened-data plot (tutorial's ``evoked.plot_white()``)."""
    fig = evoked.plot_white(noise_cov, show=False, verbose=False)
    if title:
        fig.suptitle(title)
    fig.savefig(fname, dpi=150)
    import matplotlib.pyplot as plt
    plt.close(fig)


def save_dipole_activation_plot(stc, method, fname, stride=100):
    """Save the source-activation time-course plot -- exactly the tutorial's
    own snippet, no added title:
        ax.plot(1e3 * stc.times, stc.data[::100, :].T)
        ax.set(xlabel="time (ms)", ylabel=f"{method} value")
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = subsample_source_data(stc.data, stride=stride)
    fig, ax = plt.subplots()
    ax.plot(1e3 * stc.times, sub.T)
    ax.set(xlabel="time (ms)", ylabel=f"{method} value")
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Desikan-Killiany (68-region) atlas activation matrix
# ---------------------------------------------------------------------------

def filter_cortical_labels(labels):
    """Keep only the genuine cortical regions from an ``aparc`` label list.

    FreeSurfer's Desikan-Killiany parcellation (``aparc``) ships 34 named
    cortical regions per hemisphere (68 total) plus a non-cortical
    "unknown" label per hemisphere (the medial-wall/corpus-callosum gap,
    which has no source-space vertices of interest) -- this drops those so
    ``len(result) == 68`` for a standard full ``aparc`` label list.

    Parameters
    ----------
    labels : sequence of objects with a ``.name`` attribute (mne.Label)

    Returns
    -------
    list, same type as input, excluding names starting with "unknown"
    """
    return [lab for lab in labels if not lab.name.startswith("unknown")]


def load_dk_atlas_labels(subject="fsaverage", subjects_dir=None):
    """Load the 68-region Desikan-Killiany cortical labels for ``subject``.

    Returns
    -------
    list of mne.Label, length 68, ordered lh-then-rh as returned by
    ``read_labels_from_annot`` (names like "bankssts-lh", "bankssts-rh", ...).
    """
    import mne

    labels = mne.read_labels_from_annot(subject, parc="aparc", hemi="both",
                                        subjects_dir=subjects_dir, verbose=False)
    return filter_cortical_labels(labels)


def compute_region_activation_matrix(stc, labels, src, mode="mean"):
    """Average a source estimate within each of the 68 DK regions.

    Thin wrapper around ``mne.extract_label_time_course`` -- the standard
    MNE tool for parcellating a vertex-wise source estimate into anatomical
    ROIs. ``mode="mean"`` is a plain average (appropriate here since our
    dSPM/sLORETA estimates, computed with a loose orientation constraint and
    ``pick_ori=None``, are already non-negative per-vertex magnitudes, so
    there's no sign-cancellation risk that would call for "pca_flip").

    Parameters
    ----------
    stc : mne.SourceEstimate
    labels : sequence of mne.Label
        Typically ``load_dk_atlas_labels()`` (68 regions).
    src : mne.SourceSpaces
        The source space the inverse operator/stc was built on.
    mode : str
        Passed to ``mne.extract_label_time_course``.

    Returns
    -------
    region_names : list of str, length len(labels)
    matrix : ndarray, shape (len(labels), n_times)
    """
    import mne

    matrix = mne.extract_label_time_course(stc, labels, src, mode=mode, verbose=False)
    region_names = [lab.name for lab in labels]
    return region_names, matrix


def save_region_matrix_csv(region_names, matrix, times, fname):
    """Save a (region x time) activation matrix as a labeled CSV: one row
    per region, one column per time point (header = time in ms)."""
    import csv

    if len(region_names) != matrix.shape[0]:
        raise ValueError(f"{len(region_names)} region names but matrix has "
                         f"{matrix.shape[0]} rows")
    if len(times) != matrix.shape[1]:
        raise ValueError(f"{len(times)} time points but matrix has "
                         f"{matrix.shape[1]} columns")

    with open(fname, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["region"] + [f"{t * 1e3:.1f}" for t in times])
        for name, row in zip(region_names, matrix):
            writer.writerow([name] + [f"{v:.6g}" for v in row])


# Desikan-Killiany 34-region-per-hemisphere -> lobe grouping (standard
# grouping, verified against fsaverage's actual aparc label set).
DK_LOBE_MAP = {
    "caudalmiddlefrontal": "Frontal", "lateralorbitofrontal": "Frontal",
    "medialorbitofrontal": "Frontal", "paracentral": "Frontal",
    "parsopercularis": "Frontal", "parsorbitalis": "Frontal",
    "parstriangularis": "Frontal", "precentral": "Frontal",
    "rostralmiddlefrontal": "Frontal", "superiorfrontal": "Frontal",
    "frontalpole": "Frontal",
    "inferiorparietal": "Parietal", "postcentral": "Parietal",
    "precuneus": "Parietal", "superiorparietal": "Parietal",
    "supramarginal": "Parietal",
    "bankssts": "Temporal", "entorhinal": "Temporal", "fusiform": "Temporal",
    "inferiortemporal": "Temporal", "middletemporal": "Temporal",
    "parahippocampal": "Temporal", "superiortemporal": "Temporal",
    "temporalpole": "Temporal", "transversetemporal": "Temporal",
    "cuneus": "Occipital", "lateraloccipital": "Occipital",
    "lingual": "Occipital", "pericalcarine": "Occipital",
    "caudalanteriorcingulate": "Cingulate", "isthmuscingulate": "Cingulate",
    "posteriorcingulate": "Cingulate", "rostralanteriorcingulate": "Cingulate",
    "insula": "Insula",
}
DK_LOBE_ORDER = ["Frontal", "Parietal", "Temporal", "Occipital", "Cingulate", "Insula"]


def region_base_and_hemi(name):
    """Split a DK region name like "superiorfrontal-lh" into
    ("superiorfrontal", "lh")."""
    if name.endswith("-lh"):
        return name[:-3], "lh"
    if name.endswith("-rh"):
        return name[:-3], "rh"
    raise ValueError(f"region name has no -lh/-rh hemisphere suffix: {name!r}")


def sort_regions_for_heatmap(region_names, lobe_map=DK_LOBE_MAP, lobe_order=DK_LOBE_ORDER):
    """Row order for a readable 68-region heatmap: grouped into lobes (in
    ``lobe_order``), and within each lobe, left/right homologous regions
    placed adjacently (so hemispheric asymmetry is easy to spot by eye).

    Parameters
    ----------
    region_names : sequence of str
        e.g. ``["bankssts-lh", "bankssts-rh", ...]`` (any order).
    lobe_map : dict[str, str]
        Base region name -> lobe.
    lobe_order : sequence of str
        Display order of lobes.

    Returns
    -------
    list of int
        Indices into ``region_names`` giving the sorted row order.
    """
    def key(i):
        base, hemi = region_base_and_hemi(region_names[i])
        if base not in lobe_map:
            raise ValueError(f"no lobe mapping for region {base!r}")
        return (lobe_order.index(lobe_map[base]), base, hemi)
    return sorted(range(len(region_names)), key=key)


def save_region_heatmap(region_names, matrix, times, fname, title=None, cmap="hot"):
    """Save a 68-region x time heatmap (lobe-grouped rows, see
    ``sort_regions_for_heatmap``), matching the pipeline's other dSPM/sLORETA
    plots' hot-colormap visual style.

    Parameters
    ----------
    region_names : sequence of str, length n_regions
    matrix : ndarray, shape (n_regions, n_times)
    times : ndarray, shape (n_times,) -- seconds
    fname : str
    title : str | None
    cmap : str
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = sort_regions_for_heatmap(region_names)
    ordered_names = [region_names[i] for i in order]
    ordered_matrix = matrix[order]
    lobes = [DK_LOBE_MAP[region_base_and_hemi(n)[0]] for n in ordered_names]

    vmax = percentile_clim(matrix)[-1]
    times_ms = times * 1e3
    n_regions = len(ordered_names)

    fig, ax = plt.subplots(figsize=(10, max(8, n_regions * 0.16)))
    # row i is centered at y=i (shading="nearest" treats the Y array as
    # sample centers, not cell edges)
    im = ax.pcolormesh(times_ms, np.arange(n_regions), ordered_matrix,
                       cmap=cmap, vmin=0, vmax=vmax, shading="nearest")
    ax.set_yticks(np.arange(n_regions))
    ax.set_yticklabels(ordered_names, fontsize=6)
    ax.invert_yaxis()
    ax.axvline(0.0, color="deepskyblue", lw=1, ls="--")
    ax.set_xlabel("time (ms)")

    # lobe-block separator lines (at the midpoint between two rows) + a
    # lobe-name label centered on each block
    boundaries = [0]
    for i in range(1, n_regions):
        if lobes[i] != lobes[i - 1]:
            boundaries.append(i)
    boundaries.append(n_regions)
    for b in boundaries[1:-1]:
        ax.axhline(b - 0.5, color="white", lw=1)
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        # placed well left of the (fairly long) per-region y-tick labels so
        # the two don't overlap
        ax.text(-0.28, (start + end - 1) / 2, lobes[start], transform=ax.get_yaxis_transform(),
               ha="right", va="center", fontsize=9, fontweight="bold")

    fig.colorbar(im, ax=ax, label="activation (a.u.)", fraction=0.03, pad=0.08)
    if title:
        ax.set_title(title)
    # explicit margins (not tight_layout): the lobe labels sit outside the
    # axes at a fixed fraction, so the left margin must be reserved by hand
    # or tight_layout would clip them.
    fig.subplots_adjust(left=0.32, right=0.90, top=0.96, bottom=0.05)
    fig.savefig(fname, dpi=150)
    plt.close(fig)


def filter_hemisphere_regions(region_names, hemi):
    """Indices of ``region_names`` belonging to one hemisphere.

    Parameters
    ----------
    region_names : sequence of str, e.g. ["bankssts-lh", "bankssts-rh", ...]
    hemi : {"lh", "rh"}

    Returns
    -------
    list of int
    """
    return [i for i, name in enumerate(region_names)
           if region_base_and_hemi(name)[1] == hemi]


_LOBE_COLORS = {
    "Frontal": "tab:blue", "Parietal": "tab:orange", "Temporal": "tab:green",
    "Occipital": "tab:red", "Cingulate": "tab:purple", "Insula": "tab:brown",
}


def save_region_timeseries_plot(region_names, matrix, times, fname, title=None,
                                lane_height=1.0):
    """Save a stacked per-region time-course plot: one non-overlapping
    horizontal lane per region (top to bottom, in ``region_names`` order),
    each trace scaled to its own lane -- the classic MNE ``raw.plot()``/
    ``epochs.plot()`` stacked-channel layout, as a static image. Each
    region's own peak amplitude is normalized to fit within its lane, so
    every trace is clearly visible regardless of its true magnitude
    (cross-reference the CSV/heatmap for actual relative amplitude).

    Parameters
    ----------
    region_names : sequence of str, length n_regions
    matrix : ndarray, shape (n_regions, n_times)
    times : ndarray, shape (n_times,) -- seconds
    fname : str
    title : str | None
    lane_height : float
        Vertical spacing between adjacent region lanes.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(region_names)
    times_ms = times * 1e3
    # top row = first region, matching MNE's own top-to-bottom channel order
    offsets = np.arange(n)[::-1] * lane_height

    fig, ax = plt.subplots(figsize=(12, max(6, n * 0.22)))
    for off, name, row in zip(offsets, region_names, matrix):
        base, _hemi = region_base_and_hemi(name)
        color = _LOBE_COLORS[DK_LOBE_MAP[base]]
        peak = np.max(np.abs(row))
        scale = (lane_height * 0.4) / peak if peak > 0 else 1.0
        ax.plot(times_ms, row * scale + off, color=color, lw=0.7)

    ax.axvline(0.0, color="black", lw=1, ls=":")
    ax.set_yticks(offsets)
    ax.set_yticklabels(region_names, fontsize=6)
    ax.set_ylim(-lane_height, n * lane_height)
    ax.set_xlabel("time (ms)")
    if title:
        ax.set_title(title)

    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)


def build_roi_raw(region_names, matrix, times):
    """Wrap a (region x time) matrix as an ``mne.io.RawArray``, one MNE
    "channel" per region, so MNE's own native ``raw.plot()`` browser can be
    used to explore it -- scrollable through both regions (like scrolling
    channels) and time, exactly like the classic MNE EEG-channel viewer,
    rather than a fixed static image.

    A single annotation marks the stimulus onset (t=0).

    Parameters
    ----------
    region_names : sequence of str, length n_regions
    matrix : ndarray, shape (n_regions, n_times)
    times : ndarray, shape (n_times,) -- seconds, evenly spaced

    Returns
    -------
    mne.io.RawArray
    """
    import mne

    sfreq = 1.0 / np.mean(np.diff(times))
    info = mne.create_info(ch_names=list(region_names), sfreq=sfreq, ch_types="misc")
    raw = mne.io.RawArray(np.asarray(matrix, dtype=float), info, verbose=False)
    raw.set_meas_date(None)
    raw.set_annotations(mne.Annotations(onset=[0.0 - times[0]], duration=[0.0],
                                        description=["stim onset"]))
    return raw


def broadcast_regions_to_vertices(stc_vertices, labels, matrix):
    """Paint a (n_regions, n_times) region matrix back onto every vertex
    belonging to each region -- the reverse of ``extract_label_time_course``
    -- for a "parcellated" (flat color per region) source array, instead of
    the continuous per-vertex gradient. Vertices not covered by any label
    (e.g. the non-cortical medial-wall gap) are left at 0.

    Parameters
    ----------
    stc_vertices : list of 2 arrays [lh_vertno, rh_vertno]
        Sorted vertex-number arrays, e.g. ``stc.vertices``.
    labels : sequence of objects with ``.hemi`` ("lh"/"rh") and
        ``.vertices`` (vertex numbers), row-matched to ``matrix``.
    matrix : ndarray, shape (len(labels), n_times)

    Returns
    -------
    ndarray, shape (len(stc_vertices[0]) + len(stc_vertices[1]), n_times)
    """
    if len(labels) != matrix.shape[0]:
        raise ValueError(f"{len(labels)} labels but matrix has {matrix.shape[0]} rows")
    n_lh = len(stc_vertices[0])
    n_total = n_lh + len(stc_vertices[1])
    out = np.zeros((n_total, matrix.shape[1]), dtype=matrix.dtype)
    for lab_idx, lab in enumerate(labels):
        hemi_verts = stc_vertices[0] if lab.hemi == "lh" else stc_vertices[1]
        offset = 0 if lab.hemi == "lh" else n_lh
        present = np.intersect1d(lab.vertices, hemi_verts)
        idx = np.searchsorted(hemi_verts, present)
        out[idx + offset, :] = matrix[lab_idx]
    return out


def make_parcellated_stc(stc, labels, matrix, src):
    """Build a source estimate with every vertex replaced by its region's
    averaged value (``broadcast_regions_to_vertices``) -- a 68-flat-patch
    view of the same data, suitable for the same brain-plotting/video code
    used for the full continuous map.

    Built on the FULL native cortical surface (``src[i]["np"]`` vertices per
    hemisphere, ~163842 on fsaverage), not ``stc.vertices``. ``stc.vertices``
    is only the sparse ico5 source-space *subset* used for the inverse
    computation (~10242 of ~163842 vertices per hemisphere -- ~6%, spread
    icosahedrally across the whole surface, not spatially clustered). A
    region's DK label spans thousands of *native-surface* vertices but only
    a handful of that sparse 6% sample, so painting just those left each
    region looking like scattered specks rather than a solid patch -- and
    MNE's neighbor-diffusion smoothing (meant for naturally sparse/
    continuous source data, not "fill one flat blob") diluted even those
    specks to near invisibility. Painting the full surface fixes both.

    Parameters
    ----------
    stc : mne.SourceEstimate
        Only its timing (``tmin``/``tstep``) and ``subject`` are used.
    labels : sequence of mne.Label
    matrix : ndarray, shape (len(labels), n_times)
    src : mne.SourceSpaces
        The (2-element, lh/rh) source space the forward model was built on
        -- gives the true full-surface vertex counts.
    """
    import mne

    vertices = [np.arange(src[0]["np"]), np.arange(src[1]["np"])]
    data = broadcast_regions_to_vertices(vertices, labels, matrix)
    return mne.SourceEstimate(data, vertices=vertices, tmin=stc.tmin,
                              tstep=stc.tstep, subject=stc.subject)


def vertex_region_name(vertex, hemi, labels):
    """Name of whichever label in ``labels`` contains ``vertex`` on
    ``hemi``, or None if it's in none of them (e.g. the medial-wall gap)."""
    for lab in labels:
        if lab.hemi == hemi and vertex in lab.vertices:
            return lab.name
    return None


_VIEW_CANDIDATES = ("lateral", "medial", "dorsal", "ventral", "rostral", "caudal")


def _unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("cannot normalize a zero vector")
    return v / n


def _view_camera_direction(brain, view_name):
    """Outward camera-direction unit vector (from focal point toward camera)
    for a named standard view, read from the live camera after
    ``brain.show_view()``. Used to test whether a vertex faces that camera."""
    brain.show_view(view_name)
    cam = brain._renderer.plotter.camera
    return _unit(np.array(cam.position) - np.array(cam.focal_point))


def _best_facing_view(brain, hemi, vertex, preferred, candidates=_VIEW_CANDIDATES):
    """Pick whichever standard view's camera most directly faces ``vertex``.

    Uses the vertex's actual surface normal (``brain.geo[hemi].nn``) dotted
    against each candidate view's camera direction -- pure geometry, read
    directly from the live scene, so it's immune to the render-size
    inconsistencies seen when instead trying to measure "is a 3D marker
    actor visible" from rendered pixels (that approach gave wildly
    different results for the identical vertex/view across otherwise
    identical runs -- likely a VTK/offscreen quirk -- so it isn't used).
    Leaves ``brain`` oriented to the winning view.
    """
    normal = _unit(brain.geo[hemi].nn[vertex])
    scores = {name: float(np.dot(normal, _view_camera_direction(brain, name)))
             for name in candidates}
    best = _choose_best_view(scores, preferred)
    brain.show_view(best)
    return best, scores[best]


def _project_vertex_to_pixel(brain, hemi, vertex, img_shape):
    """World -> screen-pixel coordinates (row, col) of a vertex under the
    Brain's current camera, for drawing a marker directly onto a
    screenshot. ``img_shape`` is the screenshot's ``(height, width, ...)``."""
    from vtkmodules.vtkRenderingCore import vtkCoordinate

    coord = brain.geo[hemi].coords[vertex]
    renderer = brain._renderer.plotter.renderer
    vc = vtkCoordinate()
    vc.SetCoordinateSystemToWorld()
    vc.SetValue(*(float(c) for c in coord))
    dx, dy = vc.GetComputedDisplayValue(renderer)
    row = img_shape[0] - 1 - dy  # vtk's display Y is bottom-up; arrays are top-down
    return row, dx


def draw_marker(img, row, col, radius=14, fill=(30, 100, 255), outline=(255, 255, 255)):
    """Draw a filled circle marker at ``(row, col)`` on a copy of ``img``.

    Used instead of MNE's 3D ``add_foci`` actor, whose apparent on-screen
    size proved unreliable for grazing-angle vertices (see
    ``_best_facing_view``) -- this guarantees a fixed, clearly visible
    marker regardless of 3D-rendering quirks.

    Parameters
    ----------
    img : ndarray, shape (H, W, 3), uint8
    row, col : number
        Pixel center (array coordinates: row from top, col from left).
    radius : int
    fill, outline : RGB tuples

    Returns
    -------
    ndarray, same shape/dtype as ``img``
    """
    from PIL import Image, ImageDraw

    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    draw.ellipse([col - radius, row - radius, col + radius, row + radius],
                fill=fill, outline=outline, width=2)
    return np.asarray(pil)


def _settle_after_close():
    """Give Qt/VTK a moment to finish tearing down a just-closed Brain
    figure. Repeatedly creating/closing pyvistaqt figures in one process can
    otherwise leave the teardown mid-flight, causing the *next* stc.plot()
    call to return a figure with plotter=None."""
    import gc
    gc.collect()
    try:
        from qtpy import QtWidgets
        app = QtWidgets.QApplication.instance()
        if app is not None:
            for _ in range(5):
                app.processEvents()
    except ImportError:
        pass


def _choose_best_view(scores, preferred):
    """Pick the view with the highest foci-marker visibility score.

    Ties resolve toward ``preferred`` when it's among the tied views,
    otherwise toward whichever tied view name sorts first (deterministic).

    Parameters
    ----------
    scores : dict[str, int]
        Marker-visibility score (``_foci_marker_score``) per view name.
    preferred : str
        The caller's requested default view.

    Returns
    -------
    str
    """
    if not scores:
        raise ValueError("scores must not be empty")
    best_score = max(scores.values())
    tied = sorted(name for name, s in scores.items() if s == best_score)
    return preferred if preferred in tied else tied[0]


def save_source_video(stc, subject, subjects_dir, fname, method, hemi="rh",
                      view="lateral", size=(800, 800), smoothing_steps=10,
                      n_frames=120, framerate=15, marker_radius=14,
                      auto_view=True, region_labels=None, title_suffix=""):
    """Render and save the 3D brain-activation video, matching the attached
    reference image: colored cortex, a marker at the peak-activation vertex,
    and a bottom activation time-course trace (RMS across vertices + the
    peak vertex's own time course) with a moving time cursor.

    ``Brain.save_movie(..., show_traces=True)`` does not actually capture the
    traces dock into exported frames (it only screenshots the 3D renderer),
    so the two panels are composited manually here: each frame pairs a
    ``brain.screenshot()`` with a matplotlib trace plot rendered for that
    same time point, stacked vertically. The peak-vertex marker is likewise
    drawn manually (``draw_marker``) at its exact projected screen position
    rather than via MNE's 3D ``add_foci`` actor -- see ``_best_facing_view``
    for why.

    Parameters
    ----------
    stc : mne.SourceEstimate
    subject, subjects_dir : str
    fname : str
        Output video path (``.mp4``).
    method : str
        Inverse method name, used for the on-brain title/colorbar label and
        the trace y-axis label.
    hemi : {"lh", "rh", "both"}
    view : str
        PyVista/Surfer view name, e.g. "lateral".
    n_frames : int
        Number of evenly spaced time points to render.
    framerate : int
        Output video frame rate (fps).
    marker_radius : int
        Pixel radius of the drawn peak-vertex marker.
    auto_view : bool
        If True, pick whichever standard view most directly faces the peak
        vertex instead of always using ``view`` (see ``_best_facing_view``).
    region_labels : sequence of mne.Label | None
        If given (e.g. ``load_dk_atlas_labels()``), the peak annotation
        names the anatomical region containing the peak vertex instead of
        printing its raw MNI coordinates -- most useful for a parcellated
        stc (``make_parcellated_stc``), where the peak already IS one whole
        named region.
    title_suffix : str
        Appended to the on-brain title text (e.g. ``" - right view"``), so
        two renders of the same waveform/method saved side by side stay
        visually distinguishable, not just by filename.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio
    import mne

    peak_hemi = hemi if hemi != "both" else "rh"
    lims = percentile_clim(stc.data)
    vertno_max, time_max = stc.get_peak(hemi=peak_hemi)
    idx_max, _ = stc.get_peak(hemi=peak_hemi, vert_as_index=True, time_as_index=True)

    # Matches the tutorial's surfer_kwargs idiom (time_label left on its
    # default "auto" so the on-brain clock keeps updating as set_time()
    # advances each frame, same as the reference figure's "time=0.090s"
    # line).
    brain = stc.plot(
        subject=subject, subjects_dir=subjects_dir, hemi=hemi, views=view,
        clim=dict(kind="value", lims=lims), initial_time=time_max,
        time_unit="s", size=size, smoothing_steps=smoothing_steps,
        time_viewer=False, backend="pyvistaqt", brain_kwargs=dict(show=False),
        verbose=False,
    )

    if auto_view:
        chosen_view, facing_score = _best_facing_view(brain, peak_hemi, vertno_max, view)
        print(f"    view: {chosen_view} (facing score {facing_score:.2f})")

    brain.add_text(0.03, 0.9,
                   f"{method} (plus location of maximal activation){title_suffix}",
                   "title", font_size=13)

    # Supplementary readout (not part of the tutorial): the peak vertex's
    # location, either its named anatomical region (if region_labels given)
    # or its raw MNI coordinates.
    trace_label = f"vert {vertno_max}"
    if region_labels is not None:
        region_name = vertex_region_name(int(vertno_max), peak_hemi, region_labels)
        if region_name:
            trace_label = region_name
        peak_text = f"peak region: {region_name}" if region_name else f"peak vertex {vertno_max}"
    else:
        mni = mne.vertex_to_mni([int(vertno_max)], hemis=1 if peak_hemi == "rh" else 0,
                                subject=subject, subjects_dir=subjects_dir)[0]
        peak_text = f"peak vertex {vertno_max}  MNI ({mni[0]:.1f}, {mni[1]:.1f}, {mni[2]:.1f})"
    brain.add_text(0.02, 0.02, peak_text, name="peak_label", color="deepskyblue", font_size=10)

    rms = np.sqrt(np.mean(stc.data ** 2, axis=0))
    peak_series = stc.data[idx_max, :]
    times_ms = stc.times * 1e3

    from PIL import Image

    # The camera doesn't move once the view is chosen, so the marker's
    # screen position is the same for every frame -- compute it once from
    # an initial screenshot (also fixes the exact brain-panel pixel size).
    probe_img = brain.screenshot()
    brain_shape = probe_img.shape[:2]
    marker_row, marker_col = _project_vertex_to_pixel(brain, peak_hemi, vertno_max, probe_img.shape)

    frame_indices = np.linspace(0, len(stc.times) - 1, n_frames).astype(int)
    frames = []
    trace_dpi = 100
    for i in frame_indices:
        t = stc.times[i]
        brain.set_time(t)
        brain_img = brain.screenshot()
        # mne/pyvista's off-screen render size can drift by a pixel or two
        # between screenshots (window-chrome rounding); every frame is
        # forced to this fixed size so the video writer never sees a size
        # change mid-stream.
        if brain_img.shape[:2] != brain_shape:
            brain_img = np.asarray(
                Image.fromarray(brain_img).resize((brain_shape[1], brain_shape[0])))
        brain_img = draw_marker(brain_img, marker_row, marker_col, radius=marker_radius)

        fig, ax = plt.subplots(figsize=(brain_shape[1] / trace_dpi, 2.4),
                               dpi=trace_dpi)
        ax.plot(times_ms, rms, ":", color="gray", label="RMS")
        ax.plot(times_ms, peak_series, "-", color="tab:blue", label=trace_label)
        ax.axvline(t * 1e3, color="black", lw=1)
        ax.set(xlabel="time (ms)", ylabel=f"{method} value")
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.canvas.draw()
        trace_img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        plt.close(fig)

        if trace_img.shape[1] != brain_shape[1]:
            trace_img = np.asarray(
                Image.fromarray(trace_img).resize((brain_shape[1], trace_img.shape[0])))
        frames.append(np.concatenate([brain_img, trace_img], axis=0))

    brain.close()
    _settle_after_close()

    # final guard: force every frame to the first frame's exact pixel shape.
    target_shape = frames[0].shape
    frames = [f if f.shape == target_shape
             else np.asarray(Image.fromarray(f).resize((target_shape[1], target_shape[0])))
             for f in frames]

    with imageio.get_writer(fname, fps=framerate, codec="libx264") as writer:
        for frame in frames:
            writer.append_data(frame)
