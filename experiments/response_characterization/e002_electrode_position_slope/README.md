# E002 - Electrode, presentation-position and EEG-slope responses

**Dataset:** Task 4 | **Status:** Completed | [Central registry](../../README.md)

This exploratory experiment asks whether electrode responses and their local slopes differ consistently across five delivered waveforms, first/second presentations, or their interaction. It also checks whether within-participant responses repeat across independent trial halves. It does **not** train a new classifier, select classifier features, establish physical current timing, or prove a cortical origin for condition differences.

EEG slopes are included in advance regardless of significance. A group-level effect need not yield single-epoch prediction; a feature without an individual significant effect may still contribute jointly to decoding.

## Run

From the repository root:

```powershell
& ./.venv/Scripts/python.exe -m pip install -r requirements-diagnostics.txt
& ./.venv/Scripts/python.exe -m pytest -q --rootdir=. tests

# One participant: preparation and descriptive figures, no group inference.
& ./.venv/Scripts/python.exe -m experiments.response_characterization.e002_electrode_position_slope --subjects 3 --output output/task4_responses_smoke

# Full cohort: subjects 3-12, exact participant sign-flip diagnostics.
& ./.venv/Scripts/python.exe -m experiments.response_characterization.e002_electrode_position_slope --output output/task4_response_diagnostics

# Regenerate plots/report from completed summaries, without preprocessing again.
& ./.venv/Scripts/python.exe -m experiments.response_characterization.e002_electrode_position_slope.render output/task4_response_diagnostics
```

Use a fresh output directory. Defaults use the same dataset/master CSV as `task4_baseline`. `--prepare-only` skips plotting and group statistics. The per-subject cache under `output/task4_response_cache/` is reusable only when recording/CSV hashes, numerical-method source hashes, dependencies, and settings match. Incomplete caches are preserved and rejected; use a fresh `--cache-dir` after investigating an interrupted run. Final `status.json` records completed subjects and all failures; failures produce a nonzero exit status. Partial cohorts are explicitly reported.

Installation adds MNE and Matplotlib to the existing baseline environment. `requirements-diagnostics-lock.txt` captures the extended environment; the original baseline lock remains unchanged. Set `MPLCONFIGDIR` to a writable project-local directory if desired; an unwritable global font cache does not prevent figure generation.

The supported MNE range is 1.13.x: it supplies the montage name used by the diagnostic scalp maps while retaining the legacy montage alias used by the supplied preprocessing. The lock records the tested 1.13.2 release.

## Processing and provenance

Adapted from the supplied `EEG_Codes_Export/01_Preprocessing_Task4/August28_Task4_Preprocess_AutoICA33.py` and its common/paper-method helpers:

1. Reuse the baseline's verified-unit loader, stable-run debounce and original run onsets; require exact full CSV sequence alignment. Preserve Jason's corrected ramp mapping and use `T4 Naturalness Value` for response validity. Subject 2 is excluded.
2. Retain 62 scalp electrodes, excluding CREF/M1/M2. Apply the supplied 0.5-Hz order-94 Chebyshev-II high-pass (40 dB stopband, explicit SOS), 60-Hz harmonic notches and average reference. Dropping reference channels before these channelwise filters is an implementation adaptation.
3. Fit 33-component extended Infomax ICA, seed 42, to a 1-Hz high-pass copy at the full sample rate. MNE `max_iter='auto'` resolves to 500 for Infomax in the recorded environment. Preserve reported iteration counts, component exclusions and calibration. In MNE 1.13.2, the weight-change stopping rule can also return `n_iter_=500` after an earlier stop; this value alone does not distinguish convergence from exhausting the iteration limit. Detailed optimizer traces were not retained.
4. Recompute the supplied 60-value Fp1 EOG threshold grid (0.02–5 × standard deviation); prefer 10–20 detected blinks/minute and the value closest to 15. Use the supplied component correlation threshold 3. No manual approval, muscle rejection or automatic bad-channel interpolation is added. Missing EOG calibration epochs cause an explicit failure.
5. Apply ICA exclusions; interpolate −2 to +10 ms using the supplied half-open endpoint convention; low-pass at 190 Hz. The supplied ordering is preserved: filtering before interpolation can spread electrical artifacts outside the interpolated interval.
6. Extract padded −300 to +550-ms epochs; subtract mean voltage over [−250,−50) ms. Require regular timestamps, finite values, valid boundaries, no neighboring accepted trigger inside support, and ≤500 µV peak-to-peak per electrode in the separate [−250,−50) and [50,500) ms spans. Reject complete pairs together. Unlike the initial baseline, this QC covers all 62 scalp electrodes and uses a shorter pre-trigger support.

ICA and calibration use each complete participant recording for **descriptive analysis**. These saved transforms are not automatically appropriate for a future inductive classifier evaluation. Pre-trigger effects can reflect preceding stimulation, expectation, drift, baseline choices, or acausal filter spread.

## Fixed metrics

Slopes use a local first-order least-squares fit over 25 ms (25 samples at 1 kHz), implemented with Savitzky–Golay differentiation in µV/s. The fit extends 12 samples either side of its center at 1 kHz; epochs are padded so analyzed windows avoid boundary extrapolation. Baseline subtraction changes offsets, not slopes. The slope is estimated per trial **before** taking its absolute value or averaging.

| Window | Half-open interval |
| --- | --- |
| Pre-stimulus diagnostic | −200 to −50 ms |
| Early response | 50 to 150 ms |
| Middle response | 150 to 300 ms |
| Late response | 300 to 500 ms |

For every electrode/window, compute mean voltage (µV), mean signed slope (µV/s), and mean absolute slope (µV/s). Signed averages can cancel rising/falling deflections. Absolute slopes measure rapid variation but do not identify its neural origin. A derivative is a deterministic representation of the EEG, not an independent measurement of stimulus current or sensation.

The pre-voltage diagnostic interval is contained in the voltage baseline, so its condition contrasts can be attenuated by baseline correction. It is not a matched negative-control classifier window. A null pre-voltage effect cannot rule out predictive context; slopes are invariant to the constant baseline subtraction but remain susceptible to filtering and previous-stimulus effects.

Voltage trace grids use 25-ms smoothing for readability only; voltage statistics use unsmoothed cleaned EEG. Both positions share an electrode's display scale. Eight fixed diagnostic electrodes are shown in trace grids (including frontal Fp1); all 62 enter summary heatmaps and tests. No electrode is selected from the observed effects.

## Position, context and inference

Fit each participant's trial-level features with:

`comparison intercepts + within-comparison elapsed time + position + waveform + waveform × position`

Waveform uses sum coding: four explicit deviations for Sine/Square/Triangle/LinDec; LinInc is their negative sum. Position is −0.5 for first and +0.5 for second. Thus the position coefficient is second minus first equally averaged over waveforms; waveform main effects average equally over positions. Require full column rank and positive residual degrees of freedom; save the condition number. Interactions test whether waveform differences change with presentation position. This adjusts linear terms rather than removing all history, adaptation, recording-order or comparison confounds.

Group tests treat participants as the independent units. Enumerate all `2^N` whole-vector sign flips. Position uses a squared one-sample t statistic; waveform and interaction use sums of four squared contrast t statistics. Within each effect family, the maximum statistic corrects over all 62 electrodes × four windows × three metrics (744 tests). Multiply these family-corrected p-values by three for waveform/position/interaction. Exact enumeration includes the observed sign pattern and all ties; no Monte Carlo +1 correction is needed. These model-conditional exploratory tests require independent participants and symmetric coefficient errors. They are not randomized-stimulation tests or proof of equivalence when nonsignificant.

Descriptive waveform/position averages first average within each observed comparison cell, then give those cells equal weight. They do not synthesize missing comparison cells. Save the support table so unequal counts and missing orientations remain visible.

## Repeatability

For each participant, make 100 deterministic random split-halves within ordered comparisons. Keep both presentations of every pair together. Omit ordered comparison cells with fewer than two pairs **only for repeatability**, not from the primary response model. Compare channel × time patterns over 50–500 ms with Pearson correlation:

- **Common response:** average the five waveform patterns within a position.
- **Waveform deviations:** subtract that common response from each waveform pattern before comparing halves.

Save each split and participant medians, without treating repeated splits as independent samples or confidence intervals. The diagnostic shares comparison contexts across halves and may retain repeatable electrical artifacts or drift. Low repeatability does not establish absence of a population effect.

A complementary per-electrode check uses the same complete-pair splitting scheme on fixed trial features. Within each electrode, metric and position, correlate five-waveform × three-post-window patterns after removing the across-waveform mean separately in each window. Save the median across 100 splits per participant and the median across participants. This avoids a high-amplitude electrode dominating the scalp-wide measure. It is descriptive, has no significance threshold, and does not select classifier electrodes. Rendering provenance includes this additional numerical summary module.

## Outputs

All participant-level/derived data remain under Git-ignored `output/`:

- `run_config.json`, `status.json`: settings, code/dependency provenance, completed subjects/failures.
- `sub-N/events.csv`, `cell_support.csv`: all aligned events with QC flags and retained support.
- `sub-N/preprocessing.json`, `provenance.json`: ICA/EOG audit, design rank, iteration counts and hashes.
- `sub-N/responses.npz`: evoked voltage/slope arrays, fixed trial features, sum-coded model coefficients and channel/time labels. No continuous raw EEG is copied into reports.
- `sub-N/repeatability.csv`, `repeatability_summary.csv`: repeated split checks and participant medians.
- `sub-N/electrode_repeatability.csv`, `electrode_repeatability_summary.csv`: electrode-level fixed-feature repeatability and group medians.
- `effect_tests.csv`, `strongest_exploratory_effects.csv`: all corrected tests and a descriptive ranking. Rankings must not select classifier features outside training folds.
- `RESULTS.md`, `index.html`, group/participant PNG figures: inspectable local report.
- Cache only: saved ICA fits/exclusions, trigger runs and completed-stage markers.

Report regeneration overwrites derived plots/reports only and records separate rendering source hashes in `render_provenance.json`. The original minimal baseline and research-note PDF are preserved.
