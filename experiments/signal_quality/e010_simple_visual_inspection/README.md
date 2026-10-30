# E010 - Simple-pipeline EEG visual inspection

**Dataset:** Task 4 | **Status:** Completed | [Central registry](../../README.md)

Descriptive EEG waveform/position overlays and chronological single-epoch images.
This is not another classifier, a per-electrode/bin reliability analysis or a
significance test. No supplied ICA outputs are used.

## Two matched views

- **Minimal:** original XDF EEG, fixed average over the same 62 scalp channels,
  retaining the original 14 classifier electrodes, baseline [-250,-50) ms.
  No added temporal filter. Acquisition filtering may already be present.
- **Bandpass:** the same referenced data plus fixed 0.5-30 Hz Butterworth SOS,
  prototype order 4 (eight poles for the band-pass per pass), forward/backward.
  Baseline is applied after filtering. No ICA, notch, onset interpolation,
  resampling, amplitude clipping or new amplitude-based exclusion.

The filter is applied separately to continuous finite timestamp-regular segments,
never independently to short epochs. Eight seconds at either segment boundary
are ineligible for the entire displayed support. This guard reduces edge concerns;
it does not guarantee freedom from filter transients. Forward/backward filtering
can spread stimulation contamination both before and after the trigger.

## Cohort and provenance

Subjects 3-12, original baseline response/QC eligibility and complete pairs.
Longer displayed [-300,1200)-ms support must be finite, regular, within recording
and filter-edge guards, and contain no neighboring accepted stimulus. Any added
failure excludes both presentations identically from both views. Save all rows
and reasons; do not silently replace the original 1,244-event cohort.

Require exact XDF/CSV event-sequence agreement, original baseline input hashes,
and a completed baseline-verified xDAWN epoch-cache provenance file identifying
the raw XDF hash. Cached EEG or learned transforms are not used. Minimal-view
50-500 ms bin means must reproduce saved baseline features to 1e-9 uV on every
common event. Source/settings/input hashes and all added exclusions are saved.

Within waveform/position, averages give each **observed comparison cell** equal
weight; group curves then give participants equal weight. Different waveform
cells can have different comparison support: these are descriptive overlays,
not context-adjusted causal effects. Trial/position/cell counts accompany them.

## Views and interpretation

- Fourteen-electrode waveform overlays, separate first/second presentations,
  both per participant and equal-participant grand averages.
- Waveform deviations from the five-waveform common mean, group level.
- Pointwise descriptive 95% participant-bootstrap bands (1,000 resamples),
  not corrected significance regions or single-epoch prediction uncertainty.
- C3/C4 chronological trial images by waveform and position; no amplitude or
  latency sorting. A common participant-specific 99th-percentile absolute-voltage
  color limit spans both branches. Actual clipping fractions are recorded;
  stored epochs are never clipped.
- Onset panels [-20,50] ms retain actual peri-trigger amplitudes. Main overlays
  and images omit [0,50) ms for readability and explicitly mark the omission.
  This is a plotting choice, not artifact removal or a physiological safe window.
- Filter response and per-channel amplitude summaries for review.

Pre-trigger activity can include preceding stimulation/anticipation; later
activity can include decisions and movement. Physical current timing remains
unverified. Visual differences and overlapping/nonoverlapping intervals do not
establish waveform decoding, significance, neural origin or absence of effects.
Full-cohort plots must not guide channel/window selection for a held-out classifier
unless selection is confined to its training folds.

## Run

Use the existing diagnostics environment; no new dependencies.

```powershell
$env:MPLCONFIGDIR = Join-Path (Get-Location) 'output/matplotlib-config'
& ./.venv/Scripts/python.exe -m pytest tests/test_e010_simple_visual_inspection.py -q --rootdir=.
& ./.venv/Scripts/python.exe -m experiments.signal_quality.e010_simple_visual_inspection --output output/task4_visual_inspection_20261030
# Re-render saved numerical summaries/epochs without loading XDF again:
& ./.venv/Scripts/python.exe -m experiments.signal_quality.e010_simple_visual_inspection.report output/task4_visual_inspection_20261030
```

Fresh output directories are required. Participant plots, epochs and manifests
remain Git-ignored under `output/`. `index.html` links all views. Failed subjects
are recorded, partial cohorts labeled, and failures return a nonzero exit status.

Original voltage, xDAWN and spectral **classifiers** used the minimal view with
baseline correction, without added temporal filtering or the supplied ICA.
The separate response diagnostics used the supplied preprocessing adaptation.
This inspection does not change previous classifier inputs/results.

Filter interpretation: [MNE background on filtering](https://mne.tools/stable/auto_tutorials/preprocessing/25_background_filtering.html).
