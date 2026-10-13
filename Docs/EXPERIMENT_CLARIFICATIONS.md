# Experiment clarifications and first decoding experiment

Updated 2026-10-13 from Jason Whitson's reply supplied in the collaboration email thread. His table and explicit correction are the authoritative dataset label reference. The original PDFs are retained unchanged as historical documentation.

## Confirmed trigger mapping

| Waveform | Task 3: threshold | Task 3: +0.25 mA | Task 3: +0.5 mA | Task 4 |
| --- | --- | --- | --- | --- |
| Sine | 224 | 225 | 226 | 239 |
| Square | 227 | 228 | 229 | 240 |
| Triangle | 230 | 231 | 232 | 241 |
| Linear decrease (`LinDec`) | 233 | 234 | 235 | 242 |
| Linear increase (`LinInc`) | 236 | 237 | 238 | 243 |

Within each task, increasing ramp has the higher trigger codes. Earlier exported labels reversed the ramps. Task 4 CSV waveform numbers are 1 = Sine, 2 = Square, 3 = Triangle, 4 = LinDec, 5 = LinInc. Preserve numeric trigger codes alongside descriptive labels in every event manifest.

The preprocessing and legacy helper mappings have been corrected to agree with the already-correct downstream mapping. Previously generated epochs/caches should be audited for stale event names before reuse; this update does not rewrite existing data or caches.

## Confirmed Task 4 protocol

- Use subjects 3-12. Subject 2 experienced an older, incompatible Task 4 version and is excluded from Task 4 analysis. This does not imply exclusion from Task 3.
- A waveform comparison ends when either member has been selected five times: race to five. Do not impose ten selections or assume a fixed number of trials per comparison.
- Within a presented pair, stimulation is approximately 1.4 seconds apart. This is not a claim about every successive event in the recording; measure actual intervals from XDF.
- Jason advises at most 1.2 seconds post-stimulus for Task 4 epochs.
- Naturalness selections are subjective preferences, not correct/incorrect answers. The separate detected-waveform field's interpretation and missing-response handling still need confirmation.

## Event extraction and timing

Jason recommends rejecting errant transitions by accepting a trigger only when it remains constant for multiple samples, and checking the expected sequence against the CSV. Inspect trigger run lengths before specifying a minimum duration. Preserve the first sample of each accepted stable run as its onset; log rejected runs, unexpected codes, missing matches, and any polarity normalization. The current exported loader detects transitions without this stability check; debouncing remains implementation work, not a completed feature.

The thesis describes sending a trigger at waveform start. It also describes oscilloscope checks and a CBLCFMA connection from stimulator output to a DAQ analog input for monitoring. Thus the documented convention is onset-aligned triggers, but measured trigger-to-current delay/jitter and whether the monitor signal was saved are not established. Use the term trigger-aligned for current analyses.

The XDF events and Jason's mapping are sufficient to begin a trigger-aligned baseline; availability of acquisition code or current-monitor recordings is not a prerequisite. Saved current traces or delivery logs would strengthen the timing and delivery audit if available.

Reference: [MNE event handling](https://mne.tools/stable/auto_tutorials/raw/20_event_arrays.html). This is generic event handling guidance, not acquisition-specific synchronization evidence.

## Planned first experiment (not implemented yet)

Goal: given one post-stimulation EEG epoch, predict one of five delivered waveforms with balanced accuracy reliably above the 20% reference.

- Begin with Task 4, subjects 3-12, a simple sensor-space classifier, and fixed-window mean-amplitude features from 50-500 ms after the trigger.
- Retain a pre-stimulus segment for baseline/control analysis. The exported main preprocessing already saves -250 to +500 ms epochs; classification features would use only the selected post-trigger interval.
- Evaluate within participants first, then unseen participants. Keep both presentations of a pair in one validation fold; prefer comparison-block grouping where feasible. Fit scaling, feature selection, and tuning within training folds.
- Check pre-stimulus decoding, first/second presentation differences, preceding waveform and recording-order effects, and residual stimulation/frontal artifacts. A 50 ms feature start is an initial choice, not evidence that artifacts have ended.
- Predefine secondary windows ending at 800 ms and 1.2 seconds, subject to actual neighboring-event gaps. The current saved +500 ms epochs cannot support these longer windows without regenerating epochs from continuous data.
- A shorter feature window does not remove physiological carryover from the preceding stimulus. At a 1.4-second gap, a first epoch ending at 1.2 seconds overlaps the second epoch's -250 ms baseline by about 50 ms. Account for shared samples and filter temporal support rather than assuming a 200 ms gap proves independence.
- Report balanced accuracy, macro-F1, confusion matrices, subject-level scores, and structure-preserving permutation controls. Above-chance label decoding alone does not establish shape-specific cortical or naturalness decoding.

## Pending follow-up questions

1. Was the DAQ current-monitor signal saved, where are those recordings or delivery logs, and how were they aligned with SAGA EEG? Was trigger-to-current timing checked or hardware-synchronized?
2. For Tasks 3 and 4, were failed, interrupted, or repeated trials retained/marked or removed before sharing? Do not assume failed trials were repeated or excluded.
3. For Task 4, how were missed responses represented in the CSV and handled in the race-to-five count?
4. What does the separate detected-waveform field encode? Task 3 includes subjective sensation reports; neither those nor Task 4 naturalness preferences are correctness labels.
5. The omitted Task 2 and the thesis's inconsistent Task 3 epoch-count statement remain unresolved; neither blocks the first Task 4 baseline. Original classifier/feature-extraction code would help reproduce historical preference results.
