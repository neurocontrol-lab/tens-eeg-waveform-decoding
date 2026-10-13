"""
task4_source_loc.py
====================
Helper functions for source-localizing Task 4 (two-alternative forced
choice, 2AFC) EEG data, reusing task3_source_loc.py's generic
forward/inverse/ROI-extraction machinery (build_forward_model,
compute_noise_cov, make_inverse, apply_inverse_stc,
compute_region_activation_matrix, load_dk_atlas_labels, etc. are all
task-agnostic and imported directly by the Task4 orchestrator scripts, not
duplicated here).

This module adds only what's genuinely Task4-specific:
  - loading Task4_Preprocessed_Data_blank10ms/ epochs (unlike Task3's
    preprocessed data, these still include CREF/M1/M2 and have no average
    reference applied yet -- Task4_Preprocess_blank10ms.py's own pipeline
    never did that step)
  - the corrected waveform-trigger mapping (Task4_Preprocess_blank10ms.py /
    FeatureExtractionT4.py's WAVEFORM_NUM_TO_TRIGGER has the same 242/243
    LinInc/LinDec swap documented for Task3; Jason confirmed the correction
    in the email clarification supplied 2026-10-13)
  - chosen ("natural") vs unchosen ("unnatural") epoch labeling from
    Master_Data_ToSub12.csv, replicating the sequential trigger-pointer
    matching algorithm already established in FeatureExtractionT4.py
    (build_chosen_unchosen_labels + the epoch_ptr loop), so results are
    consistent with the existing Task4 ML pipeline

Pure/testable logic (no file or MNE I/O):
    parse_trial_pairs, match_chosen_unchosen_epochs

I/O and MNE-object-heavy functions:
    load_task4_epochs
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Jason Whitson's authoritative table, supplied in the email thread
# 2026-10-13: LinDec=242, LinInc=243 (higher ramp trigger = increase).
# Earlier exported labels were reversed; raw integer codes are unchanged.
# ---------------------------------------------------------------------------

WAVEFORM_TRIGGERS = {
    "Sine": 239, "Square": 240, "Triangle": 241, "LinInc": 243, "LinDec": 242,
}
TRIGGER_TO_WAVEFORM = {v: k for k, v in WAVEFORM_TRIGGERS.items()}
# Jason also confirms CSV waveform numbers:
# 1=Sine, 2=Square, 3=Triangle, 4=LinDec, 5=LinInc.
WAVEFORM_NUM_TO_TRIGGER = {
    1: WAVEFORM_TRIGGERS["Sine"], 2: WAVEFORM_TRIGGERS["Square"],
    3: WAVEFORM_TRIGGERS["Triangle"], 4: WAVEFORM_TRIGGERS["LinDec"],
    5: WAVEFORM_TRIGGERS["LinInc"],
}


def parse_trial_pairs(rows, wf1_col="t4 waveform 1", wf2_col="t4 waveform 2",
                      nat_col="t4 naturalness value"):
    """Parse Master_Data_ToSub12.csv rows (already filtered to one subject,
    Task 4, in original row order) into (wf1_trigger, wf2_trigger,
    chosen_trigger, unchosen_trigger) tuples.

    Replicates FeatureExtractionT4.py's build_chosen_unchosen_labels, with
    one deliberate difference: rows with a naturalness value that isn't
    exactly 1 or 2 (missing/undecided/malformed) are SKIPPED rather than
    silently treated as "waveform 2 chosen" (a real mislabeling risk in the
    original -- see project memory).

    Parameters
    ----------
    rows : sequence of dict-like (e.g. DataFrame.to_dict("records") or a
        list of dicts), each with (case-sensitive, already-lowercased)
        keys ``wf1_col``, ``wf2_col``, ``nat_col`` holding numeric values.
    wf1_col, wf2_col, nat_col : str
        Column names to read (already-lowercased, matching
        FeatureExtractionT4.py's own header normalization).

    Returns
    -------
    list of (wf1_trigger, wf2_trigger, chosen_trigger, unchosen_trigger)
    """
    pairs = []
    for row in rows:
        wf1_num = int(row[wf1_col])
        wf2_num = int(row[wf2_col])
        nat_val = int(row[nat_col])
        if wf1_num not in WAVEFORM_NUM_TO_TRIGGER or wf2_num not in WAVEFORM_NUM_TO_TRIGGER:
            continue
        if nat_val not in (1, 2):
            continue
        trig1 = WAVEFORM_NUM_TO_TRIGGER[wf1_num]
        trig2 = WAVEFORM_NUM_TO_TRIGGER[wf2_num]
        if nat_val == 1:
            pairs.append((trig1, trig2, trig1, trig2))
        else:
            pairs.append((trig1, trig2, trig2, trig1))
    return pairs


def match_chosen_unchosen_epochs(event_triggers, trial_pairs):
    """Match CSV trial pairs to epoch indices by sequential trigger
    scanning, replicating FeatureExtractionT4.py's epoch_ptr algorithm
    exactly (same behavior, including never rewinding the pointer, so a
    mismatched/dropped epoch permanently desyncs later trials -- this is a
    known fragility of the established approach, kept for consistency
    rather than silently fixed, since there's no Trial_id join key
    available to do better).

    Parameters
    ----------
    event_triggers : sequence of int, length n_epochs
        Trigger code of each epoch, in the order they appear in the epochs
        object (e.g. ``epochs.events[:, 2]``).
    trial_pairs : sequence of (wf1_trig, wf2_trig, chosen_trig, unchosen_trig)
        From ``parse_trial_pairs``, in original CSV row order.

    Returns
    -------
    list of (chosen_epoch_idx, unchosen_epoch_idx) -- trials that couldn't
    be matched (either trigger not found forward of the current pointer)
    are omitted.
    """
    event_triggers = list(event_triggers)
    n = len(event_triggers)
    ptr = 0
    matched = []

    for wf1_trig, wf2_trig, chosen_trig, unchosen_trig in trial_pairs:
        idx1 = None
        for p in range(ptr, n):
            if event_triggers[p] == wf1_trig:
                idx1 = p
                ptr = p + 1
                break
        idx2 = None
        for p in range(ptr, n):
            if event_triggers[p] == wf2_trig:
                idx2 = p
                ptr = p + 1
                break
        if idx1 is None or idx2 is None:
            continue
        chosen_idx = idx1 if chosen_trig == wf1_trig else idx2
        unchosen_idx = idx2 if chosen_trig == wf1_trig else idx1
        matched.append((chosen_idx, unchosen_idx))

    return matched


def load_condition_epochs_task4(subjects, data_dir, session, trigger_codes,
                                fname_template="sub-{sid}_ses-{session}_epo.fif",
                                scale=1e-6, drop_channels=("CREF", "M1", "M2")):
    """Load and pool one waveform condition's epochs across subjects, for the
    5-waveform-type comparison (pools every presentation of a waveform
    regardless of whether it was the chosen or unchosen member of its pair).

    Mirrors task3_source_loc.load_condition_epochs, but additionally drops
    the reference/aux channels and applies the average reference per
    subject before pooling -- Task4_Preprocess_blank10ms.py's own pipeline
    never did that step (unlike Task3's preprocessed data).

    Parameters
    ----------
    trigger_codes : sequence of int
        Trigger codes for one waveform, e.g. [WAVEFORM_TRIGGERS["Sine"]]
        (Task4 has one trigger per waveform, not three amplitude tiers like
        Task3).

    Returns
    -------
    mne.Epochs, concatenated across subjects, preloaded.
    """
    import os
    import mne
    from task3_source_loc import select_waveform_mask

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
        drop = [c for c in drop_channels if c in ep.ch_names]
        if drop:
            ep.drop_channels(drop)
        ep.set_eeg_reference("average", projection=True, verbose=False)
        chunks.append(ep)
    if not chunks:
        raise RuntimeError("no subject files were found / matched")
    # add_offset=True (default): Task4 epochs from different subjects can
    # land on identical absolute sample numbers (same experiment timing
    # template reused per subject), which collides under add_offset=False
    # ("Event time samples were not unique") -- offsetting keeps pooling
    # safe since downstream code (evoked averaging, source localization)
    # never reads the raw sample numbers.
    return mne.concatenate_epochs(chunks, add_offset=True, verbose=False)


def load_chosen_unchosen_epochs(subject, data_dir, trial_pairs, session="Task4",
                                fname_template="sub-{sid}_ses-{session}_epo.fif",
                                scale=1e-6, drop_channels=("CREF", "M1", "M2")):
    """Load one subject's Task4 epochs and split into chosen ("natural") vs
    unchosen ("unnatural") subsets using the CSV-driven trial-pair matching
    (parse_trial_pairs + match_chosen_unchosen_epochs).

    Parameters
    ----------
    trial_pairs : list of (wf1_trig, wf2_trig, chosen_trig, unchosen_trig)
        From ``parse_trial_pairs``, already filtered to this subject (and
        Task 4) and in original CSV row order.

    Returns
    -------
    (chosen_epochs, unchosen_epochs) : mne.Epochs, mne.Epochs
        Both preloaded, avg-ref'd, same length (one epoch per matched
        trial, paired at the same index).
    """
    epochs = load_task4_epochs(subject, data_dir, session=session,
                               fname_template=fname_template, scale=scale,
                               drop_channels=drop_channels)
    matched = match_chosen_unchosen_epochs(epochs.events[:, 2], trial_pairs)
    if not matched:
        raise RuntimeError(f"no trials matched for subject {subject}")
    chosen_idx = [m[0] for m in matched]
    unchosen_idx = [m[1] for m in matched]
    return epochs[chosen_idx], epochs[unchosen_idx]


def load_task4_epochs(subject, data_dir, session="Task4",
                      fname_template="sub-{sid}_ses-{session}_epo.fif",
                      scale=1e-6, drop_channels=("CREF", "M1", "M2")):
    """Load one subject's Task4 epochs, corrected for the same
    microvolts-mislabeled-as-volts unit bug as Task3 (see project memory),
    with the reference/aux channels dropped and average reference applied
    -- Task4_Preprocess_blank10ms.py's own pipeline never did that step
    (unlike Task3's preprocessed data, which already had it), so it has to
    happen here before any forward/inverse modeling.

    Parameters
    ----------
    subject : int
    data_dir : str
    session : str
    scale : float
        Applied to correct the mislabeled-unit bug; pass 1.0 to disable.
    drop_channels : sequence of str
        Reference/aux channels to drop before re-referencing.

    Returns
    -------
    mne.Epochs, preloaded, average-referenced (projection=True), all
    conditions (not filtered to any particular waveform/label).
    """
    import os
    import mne

    fname = os.path.join(data_dir, fname_template.format(sid=subject, session=session))
    epochs = mne.read_epochs(fname, preload=True, verbose=False)
    if scale != 1.0:
        epochs.apply_function(lambda x: x * scale, channel_wise=True, verbose=False)
    drop = [c for c in drop_channels if c in epochs.ch_names]
    if drop:
        epochs.drop_channels(drop)
    epochs.set_eeg_reference("average", projection=True, verbose=False)
    return epochs
