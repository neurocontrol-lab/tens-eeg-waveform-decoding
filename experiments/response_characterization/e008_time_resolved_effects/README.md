# E008 - Time-resolved waveform, position and interaction effects

**Dataset:** Task 4 | **Status:** Completed | [Central registry](../../README.md)

Test whether brief Gaussian-derivative waveform effects were hidden by earlier window summaries. All five fixed scales (Ïƒ=2.5/5/10/20/40 ms) are tested together. This is a context-adjusted group response analysis, not classification or true scale invariance. The earlier repeatability results remain unchanged.

Completed findings: [23 October 2026 time-resolved analysis](../../../Docs/TASK4_TIME_RESOLVED_RESULTS.md). Canonical local report: `output/task4_time_resolved_20261023/index.html`.

## Run

Use the existing `requirements-diagnostics.txt` environment and original saved ICA fits. No fit or installation is performed.

```powershell
& ./.venv/Scripts/python.exe -m pytest -q --rootdir=. tests/test_e008_time_resolved_effects.py
$env:MPLCONFIGDIR = Join-Path (Get-Location) 'output/matplotlib-config'
& ./.venv/Scripts/python.exe -m experiments.response_characterization.e008_time_resolved_effects --output output/task4_time_resolved_new
# Rerun inference/reporting from completed coefficient files:
& ./.venv/Scripts/python.exe -m experiments.response_characterization.e008_time_resolved_effects.report output/task4_time_resolved_new
```

Use a fresh output directory. Defaults read `output/task4_multiscale_20261023`, `output/task4_preprocessing_comparison_final_20261023`, `output/task4_response_cache`, the original XDF recordings and master CSV. Earlier outputs and numerical sources are read-only. Complete original input, CSV, ICA and source hashes/dependency checks must pass. The previous fixed features at shared scales must reproduce. Errors cause nonzero exit; group inference requires every requested participant to complete. `--subjects` supports a descriptive subset of subjects 3â€“12, with at least three required for group tests.

## Frozen design

- Same 1,242-event/621-pair eligible cohort as the completed preprocessing comparison; no branch-specific rejection. Fail if real padding, timestamps, finiteness, alignment or design rank differs.
- Minimal branch: recorded EEG with verified units, 62-channel average reference and constant baseline subtraction. No added high-pass/notch/low-pass, ICA or interpolation. Recording-system filtering may already exist.
- Supplied branch: exact original filtering, original saved ICA/EOG exclusions, âˆ’2 to +10-ms interpolation and 190-Hz low-pass. No ICA refit.
- Apply only Gaussian first derivatives at the five scales, unit-ramp calibrated to ÂµV/s. DoG and Gaussian-smoothed voltage are not added to this inferential family.
- Post [180,480) ms and pre [âˆ’480,âˆ’180) ms each contain 300 full-rate samples (1 kHz). The largest kernel has Â±160-ms support, keeping post kernels â‰¥20 ms and pre kernels â‰¤âˆ’20 ms from the recorded trigger. Real âˆ’900 to +650-ms padding is retained. Derivatives cancel constant baselines. Continuous filtering may already spread artifacts. This interval differs from earlier [100,500)-ms repeats; it excludes earlier effects.

## Inference

Fit the earlier participant OLS model separately at every branch/period/scale/time/channel: comparison intercepts, within-comparison centered elapsed time, centered first/second position, four sum-coded waveform coefficients, and four waveform Ã— position coefficients. Waveform coefficients represent averages across positions; position represents second minus first averaged equally across waveforms. Linear adjustment does not remove arbitrary order/history/intensity effects.

For each effect family, use the sum of squared group t-statistics across its coefficients (four for waveform/interaction, one for position). This is an omnibus statistic, not a parametric F test. Enumerate every whole-participant sign pattern (1,024 at N=10), applying each participant's sign jointly to their full coefficient vector/map. No individual trial or class label is shuffled. Independent participants and joint symmetry of the coefficient errors under the null are required. This is exploratory model-conditional group inference, not randomized causal inference.

Within each family, max correction covers 2 branches Ã— 2 periods Ã— 5 scales Ã— 300 times Ã— 62 channels = 372,000 locations. Bonferroni across three effect families gives control across the 1,116,000 tested locations. Both uncorrected and corrected p maps are saved. Participants, not repeated trials/splits, are the units of group inference. Chunked calculations use the same exact null as brute-force tests; no cluster adjacency, threshold selection or scale tuning is used.

## Outputs and limits

All participant coefficients, descriptive evoked traces, events, provenance, audit tables, full p/statistic arrays (`group_tests.npz`), summaries, significant points and plots remain under ignored `output/`. Calculations use float64; display-only traces use float32. Fixed plots show C3/C4/CP1/Pz, with all 62 electrodes tested. Descriptive condition means are equally weighted over observed comparisons and are not the regression coefficients.

Non-significance does not establish equivalent waveforms or absent decodable information. Group tests can miss subject-specific effects or variable latencies. Pre activity can include anticipation and preceding stimulation. A significant post effect and non-significant pre effect do not establish a post-minus-pre difference. Physical delivery timing and neural versus artifact origin remain unresolved. Full-recording ICA is retained for descriptive analysis; no held-out classifier preprocessing is claimed.

## Individual participant plots

```powershell
& ./.venv/Scripts/python.exe -m experiments.response_characterization.e008_time_resolved_effects.participant_plots --output output/task4_participant_views_new
```

This separate viewer preserves the group report and tests. It renders C3 post-trigger averages for every participant from saved display traces, with a selector for full waveform averages or deviations from the five-waveform common response. Trial averages give equal weight to observed comparison contexts, separately by waveform and first/second position. Y-limits are shared across pipelines/positions within each scale row; they are not shared between participants or scales. Additional C3 pre and C4/CP1/Pz post plots are provided for the reference participant (default Subject 3, chosen as the first eligible participant, not by signal strength).

The default also reconstructs only the reference participant using the original saved ICA to plot every eligible C3 post-trigger voltage/Ïƒ=20-ms derivative trial. Black traces show context-balanced averages; thin traces show individual trials, with counts. Reconstructed derivative means must match the saved display traces. `--no-trials` avoids loading original recordings and generates the saved-average views only. Outputs and provenance remain Git-ignored. These are descriptive displays without new individual significance tests or classification. Within-participant averaging can still blur trial-level latency differences.
