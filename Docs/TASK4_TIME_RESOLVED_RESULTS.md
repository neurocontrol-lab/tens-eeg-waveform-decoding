# Task 4 time-resolved Gaussian-derivative effects

**Follow-up:** [Earlier scale-specific time coverage](TASK4_RESPONSE_FOLLOWUP.md) extends testing to 50 ms where kernel support permits, retaining global correction. It also found no corrected group effects. This original common late-window analysis remains preserved.

**23 October 2026.** Time-point-by-time-point testing did not identify a consistent group-level waveform, first/second-position or interaction effect after correction across the tested scales, electrodes, periods and preprocessing branches. This extends the earlier window-based diagnostics. It does not establish waveform equivalence, absence of participant-specific information or classification at chance.

## Cohort and signal representation

All ten participants (subjects 3–12) completed both branches using the same frozen **1,242 events in 621 presentation pairs** as the preceding preprocessing comparison. All 62 scalp electrodes were retained. No new amplitude exclusions or ICA fitting were performed.

The minimal branch uses verified recorded EEG units, average reference and constant baseline subtraction, followed by the local derivative operator. It has no added high-pass, notch, low-pass, ICA or onset interpolation. The supplied branch reconstructs the original filtering, saved ICA/EOG exclusions, −2 to +10-ms interpolation and 190-Hz low-pass. Acquisition-system filtering may already exist in either branch's recorded input.

Five first derivatives of Gaussian-smoothed EEG were evaluated: **σ=2.5, 5, 10, 20 and 40 ms**, calibrated in µV/s. This analysis tests derivatives only; it does not add DoG or smoothed-voltage tests. The previously computed split-half correlations remain a separate descriptive repeatability measure.

Both periods contain 300 full-rate samples at 1 kHz:

| Period | Tested interval relative to recorded trigger |
| --- | --- |
| Pre-trigger control | [−480,−180) ms |
| Post-trigger response | [180,480) ms |

These are matched 300-ms intervals, translated by 660 ms. At the largest scale, ±160-ms kernels stay at least 20 ms from trigger onset and are supported by real −900 to +650-ms epoch padding. This guards against directly incorporating the explicit onset blank. It cannot undo earlier continuous-filter spread. Constant baseline offsets cancel from the derivatives. The common interval differs from the earlier 100–500-ms checks; effects before 180 ms are outside this analysis.

## Statistical method

At every branch, period, scale, electrode and time point, the earlier participant regression was fitted with comparison-specific intercepts, within-comparison centered elapsed time, waveform, presentation position and their interaction. Four sum-coded coefficients represent the five waveform conditions. Waveform effects average over positions; the position coefficient is second minus first averaged equally across waveforms.

The omnibus waveform and interaction statistics sum the squared group t-statistics across their four coefficients; the position statistic uses one squared t-statistic. All **1,024 whole-participant sign patterns** were enumerated, flipping each participant's full contrast vector/map together. Individual trial labels were not shuffled. The test assumes independent participants and joint symmetry of participant coefficient errors under its null.

Maximum-statistic correction covers both branches × both periods × five scales × 300 time points × 62 electrodes: **372,000 tested locations per effect family**. Bonferroni correction across the three families covers the complete set of 1,116,000 locations. This is a regression-based, model-conditional sign-flip test, not a parametric F test or a randomized causal comparison.

## Results

| Effect | Locations with corrected p ≤ .05 | Smallest corrected p |
| --- | ---: | ---: |
| Waveform | 0 / 372,000 | 1.0000 |
| First/second position | 0 / 372,000 | 1.0000 |
| Waveform × position | 0 / 372,000 | 0.8848 |

Nominal uncorrected p≤.05 locations occurred in each family, but none remained significant after the declared correction. They should not be used to select electrodes or scales as validated waveform features. Fixed C3/C4/CP1/Pz plots show descriptive group mean waveform deviations, separately by position and preprocessing. Their condition averaging differs from the nuisance-adjusted regression coefficients, and visual separation alone does not establish an effect.

## Verification and limits

All participant designs were full rank (20/20 columns), with condition numbers 5.37–6.65. Shared-scale supplied-pipeline features reproduced the prior results exactly. There were zero new padding exclusions. The optimized sign-flip implementation matched brute-force enumeration in synthetic tests and independently checked actual-data locations. **All 29 relevant tests passed**, including seven new checks. A report assembly error was repaired; the completed report was regenerated from saved coefficients without repeating EEG preprocessing or changing the statistical analysis.

Ten participants and broad simultaneous correction limit sensitivity. Group tests can miss participant-specific patterns or variable latencies, and the modeled linear adjustments do not eliminate all history, intensity or recording-order confounds. Pre-trigger activity can contain anticipation and preceding-stimulus carryover. A significant post effect alongside a non-significant pre effect would not itself establish a post-minus-pre difference. Physical trigger-to-current timing and neural versus artifact origin remain unresolved. This experiment does not test held-out classification, true temporal scale invariance or absence of waveform information.

See the [experiment README](../experiments/response_characterization/e008_time_resolved_effects/README.md) for reproduction. Complete local results, full uncorrected/corrected p maps, coefficient arrays, audit records and figures are in Git-ignored `output/task4_time_resolved_20261023/`, with `RESULTS.md` and `index.html` as entry points. This note retains cohort aggregates only.
