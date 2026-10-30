# E007 - Minimal versus supplied preprocessing comparison

**Dataset:** Task 4 | **Status:** Completed | [Central registry](../../README.md)

Matched sensitivity analysis of minimally processed recorded EEG versus the supplied filtering/ICA pipeline. The question is whether the pipeline materially changes derivative/DoG waveform-specific repeatability. This is not a classifier or a validation of a neural origin. The comparison changes the processing bundle; it does not isolate individual stages.

Completed cohort findings: [23 October 2026 preprocessing comparison](../../../Docs/TASK4_PREPROCESSING_COMPARISON.md). The canonical local report is `output/task4_preprocessing_comparison_final_20261023/index.html`.

## Run

Use the existing `requirements-diagnostics.txt` environment, completed multiscale results and saved ICA fits:

```powershell
& ./.venv/Scripts/python.exe -m pytest -q --rootdir=. tests/test_e007_preprocessing_comparison.py
$env:MPLCONFIGDIR = Join-Path (Get-Location) 'output/matplotlib-config'
& ./.venv/Scripts/python.exe -m experiments.signal_quality.e007_preprocessing_comparison --output output/task4_preprocessing_comparison_new

# Regenerate group summaries and plots from completed stages:
& ./.venv/Scripts/python.exe -m experiments.signal_quality.e007_preprocessing_comparison.report output/task4_preprocessing_comparison_new
```

`--subjects 3` runs one descriptive stage. `--prior` defaults to `output/task4_multiscale_20261023/`; `--ica-cache` defaults to `output/task4_response_cache/`. Use fresh output directories; existing analyses/caches are read-only. Errors are preserved in `status.json` and cause nonzero exit. No ICA fit or dependency installation is performed.

## Matched inputs

- Ten current-protocol participants, 62 scalp electrodes, verified XDF units and strict trigger/master-CSV alignment.
- Freeze the preceding multiscale experiment's eligibility flags and row order. Both branches use the same 1,242 events in 621 pairs. Never select a different raw cohort based on its amplitudes. Expanded real padding (âˆ’900 to +650 ms) must be finite, in bounds, timestamp-regular and free of neighboring accepted triggers for every selected event; otherwise fail explicitly.
- Verify original input/CSV hashes, numerical sources, dependency versions and saved ICA SHA256/channel order. Reconstruct the supplied continuous pipeline from original recordings with the saved exclusions; never refit ICA. The complete preceding fixed-feature arrays and post split correlations must reproduce before a stage is accepted.

## Branches

**Minimal:** recorded EEG in ÂµV with the same 62-channel average reference, interval-specific baseline subtraction and feature operators only. No added high-pass, notch, 190-Hz low-pass, ICA or onset interpolation. Acquisition-system filtering may already exist. This does not mean artifact-free or physically unfiltered EEG.

**Cleaned:** supplied 0.5-Hz order-94 Chebyshev-II high-pass, 60-Hz harmonic notches, average reference, original 33-component ICA with saved EOG exclusions, âˆ’2 to +10-ms interpolation and 190-Hz low-pass, in the same order as the previous experiment.

## Features and controls

Reuse all ten fixed representations: voltage, 25-ms local-linear slope, Gaussian voltage and first derivatives at Ïƒ=5/10/20 ms, and secondary DoG(5,10)/DoG(10,20). Gaussian support is Â±4Ïƒ; derivative units are calibrated ÂµV/s. No scale/electrode selection or temporal resampling.

Post trace [100,500) ms uses baseline [âˆ’250,âˆ’50) ms. The equal-length pre trace [âˆ’500,âˆ’100) ms uses its own earlier baseline [âˆ’850,âˆ’650) ms: both are exact 600-ms translations of the post interval/baseline. Thus voltage controls do not use the post baseline. Baseline constants cancel from derivative and DoG representations. The largest local kernel reaches no earlier than +20 ms in the post trace and no later than âˆ’20 ms in the pre trace; local features avoid the explicit trigger-aligned blank. This cannot undo the earlier continuous filter spread in the cleaned branch.

The same 100 whole-pair splits, seed 20261023, are used in both branches and periods, within ordered comparisons. Omit singleton cells only from repeatability (599 pairs used, 22 singleton pairs omitted); balance observed comparison cells. Calculate common-response and waveform-deviation correlations over scalp Ã— time, plus per-electrode waveform-deviation checks. Summarize medians over splits within participant first, then medians over participants. Shared recording-level transformations and comparison contexts make these descriptive repeatability checks, not independent replication.

For waveform-deviation r, predeclare three paired participant contrasts: minimalâˆ’cleaned post; minimalâˆ’cleaned pre; and (minimal postâˆ’minimal pre)âˆ’(cleaned postâˆ’cleaned pre). Exact participant sign-flip enumeration uses squared studentized statistics and a single maximum over all three contrasts Ã— ten representations Ã— two positions (60 tests). This assumes independent participants and symmetric difference errors. A positive postâˆ’pre contrast is not a proof of current-stimulus specificity. Pre-trigger signals can include expectation, previous-stimulus carryover and context.

## Signal checks and outputs

For each branch/period, report max-channel peak-to-peak amplitudes, >500-ÂµV flags and Welch spectral fractions (58â€“62 Hz / 1â€“100 Hz; 80â€“190 Hz / 1â€“190 Hz). Four-hundred-ms windows give 2.5-Hz bins. These are contamination indicators and do not change eligibility or measure neural SNR. The post-feature arrays (mean/RMS in the original three post windows) are saved for audit; no classifier is trained. The original short pre-feature window is used only internally for reconstruction validation, not as the matched negative control.

All participant summaries, pre/post repeats, health checks, saved display-only fixed-channel evoked traces, source/input/ICA hashes and reports remain under Git-ignored `output/`. Float64 is used for numerical features and correlations; float32 evoked traces are for display. The report contains every participant median and all corrected paired contrasts, plus waveform-specific/common-response comparisons and fixed C3 derivative plots. No continuous recordings or ICA fits are overwritten.

Results are conditional on the original cleaned-QC cohort. An increase in minimal-branch repeatability can reflect preserved neural signals or preserved artifacts/drift; pre controls and signal checks help interpret it but cannot resolve neural origin alone. Unknown physical trigger latency, preceding stimulation, charge/intensity and position effects remain. Responses before +100 ms, true scale invariance and classification accuracy are outside this experiment.
