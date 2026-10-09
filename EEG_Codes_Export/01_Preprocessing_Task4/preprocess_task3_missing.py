"""
preprocess_task3_missing.py
===========================
Preprocess the two Task 3 subjects that were missing from
EEG_Practice/preprocessed_data_new_baseline (subjects 2 and 9), using the SAME
deterministic pipeline that produced the other nine
(eeg_waveform_analysis_with_save.preprocess_eeg: 0.5-40 Hz, 60 Hz notch,
average reference, automated infomax ICA with EOG+muscle rejection, seed 42),
so all 11 subjects are processed identically.

Subject 9's Task 3 recording is split across two XDF files
(W1-2-3 and W4-5); they are concatenated into one continuous raw BEFORE
preprocessing so event sample indices stay aligned with the saved raw (the
downstream analysis re-epochs from the raw using the epochs' events).

Run: python preprocess_task3_missing.py
"""

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "EEG_Practice"))
import mne
from eeg_waveform_analysis_with_save import (
    load_xdf_to_mne, preprocess_eeg, apply_frequency_band, create_epochs,
    save_preprocessed_data, TRIGGER_MAPPING, TMIN, TMAX, BASELINE,
)

warnings.filterwarnings("ignore")
mne.set_log_level("WARNING")

OUTPUT_DIR = os.path.join("EEG_Practice", "preprocessed_data_new_baseline")
SESSION_ID = "Task3"

# Task 3 XDF source files per missing subject (subject 9 = two files, in order)
SUBJECT_FILES = {
    2: [os.path.join("EEG_Practice", "Subject2_data", "EEGDATA_SUB2-T3.xdf")],
    9: [os.path.join("EEG_Practice", "Subject9_data", "EEGDATA_SUB9-T3_W1-2-3.xdf"),
        os.path.join("EEG_Practice", "Subject9_data", "EEGDATA_SUB9-T3_W4-5.xdf")],
}


def _normalize_events(events):
    """Map trigger codes to the canonical 224-238 scheme and drop everything else.

    Some Task 3 recordings (e.g. subject 9, file W4-5) used an inverted/active-low
    16-bit trigger line, so the code appears as 65535 - actual (e.g. 65297 -> 238).
    Idle (65535) and stray codes are discarded.
    """
    out = []
    for s, _, c in events:
        c = int(c)
        if not (224 <= c <= 238) and (224 <= 65535 - c <= 238):
            c = 65535 - c          # undo active-low inversion
        if 224 <= c <= 238:
            out.append([int(s), 0, c])
    return np.array(out, dtype=int)


def load_concat(files):
    """Load one or more XDF files into a single continuous raw with aligned,
    normalized events (mne.concatenate_raws offsets later files' event samples)."""
    raws, ev_list = [], []
    for f in files:
        r, ev = load_xdf_to_mne(f)
        raws.append(r)
        ev_list.append(_normalize_events(ev))
    if len(raws) == 1:
        return raws[0], ev_list[0]
    # align channels if the two recordings differ
    common = [c for c in raws[0].ch_names if all(c in r.ch_names for r in raws)]
    raws = [r.copy().pick(common) for r in raws]
    raw, events = mne.concatenate_raws(raws, events_list=ev_list)
    return raw, events


def process_subject(sid, files):
    print(f"\n{'='*70}\nSUBJECT {sid}  ({len(files)} file(s))\n{'='*70}")
    for f in files:
        if not os.path.exists(f):
            print(f"  MISSING FILE: {f} -- cannot process subject {sid}")
            return False
    raw, events = load_concat(files)
    print(f"  raw: {len(raw.ch_names)} ch, {raw.n_times/raw.info['sfreq']:.0f}s; "
          f"{len(events)} trigger events")

    raw_pre, ica = preprocess_eeg(raw, apply_ica=True, n_components=15, plot_ica=False)
    raw_band, _ = apply_frequency_band(raw_pre, "broadband")
    # Use only the conditions actually present (subject 2 codes waveform only,
    # so the full 15-key TRIGGER_MAPPING would error on the absent Mid/High keys).
    present = set(int(c) for c in events[:, 2])
    event_dict = {k: v for k, v in TRIGGER_MAPPING.items() if v in present}
    epochs = create_epochs(raw_band, events, event_dict)
    save_preprocessed_data(raw_pre, epochs, ica, OUTPUT_DIR, sid, SESSION_ID)

    # verification
    import collections
    trig = collections.Counter(epochs.events[:, 2])
    ev = epochs.average()
    print(f"  SAVED sub-{sid}_ses-{SESSION_ID}: {len(epochs)} epochs, "
          f"{len(epochs.event_id)} conditions")
    print(f"  trigger counts: {dict(sorted(trig.items()))}")
    print(f"  |evoked| max={np.abs(ev.data).max():.2f} (expect uV-scale ~1-30), "
          f"ICA excluded={sorted(ica.exclude) if ica else None}")
    return True


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ok = {}
    for sid, files in SUBJECT_FILES.items():
        epo = os.path.join(OUTPUT_DIR, f"sub-{sid}_ses-{SESSION_ID}_epo.fif")
        if os.path.exists(epo):
            print(f"\nsub-{sid}: already preprocessed -- skip")
            ok[sid] = True
            continue
        try:
            ok[sid] = process_subject(sid, files)
        except Exception as e:
            import traceback
            print(f"  ERROR processing subject {sid}: {e}")
            traceback.print_exc()
            ok[sid] = False
    print(f"\n{'='*70}\nDONE. Results: {ok}\n{'='*70}")


if __name__ == "__main__":
    main()
