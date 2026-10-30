# Task 4 electrode, presentation-position and EEG-slope diagnostics: 2026-10-21

These exploratory diagnostics did not identify a consistent waveform-specific voltage or slope signature across the ten participants. The common voltage response was descriptively more repeatable than the differences between waveforms, especially during second presentations. Slopes did not show a clear repeatability advantage in these checks. This does not establish that waveform information is absent, and no new classifier was evaluated.

## Cohort and fixed analysis

Subjects 3–12 contributed 1,246 stimulation events, all matching the full behavioral CSV sequence. The all-electrode QC retained 1,242 events in 621 complete pairs; two pairs exceeded the fixed 500 µV peak-to-peak rule. All matched naturalness responses were valid. This differs from the initial baseline's 1,244 eligible events because the diagnostic QC covers all 62 scalp electrodes and uses a different epoch support.

The analysis adapts the supplied August28 AutoEOG33 preprocessing: verified voltage units, 0.5-Hz high-pass, 60-Hz harmonic notches, average reference, full-rate 33-component extended Infomax ICA on a 1-Hz high-pass copy, automatic Fp1-based EOG rejection, −2 to +10-ms interpolation, and 190-Hz low-pass. The supplied ICA fitting settings were retained. Event loading instead uses the audited debounce and CSV alignment. ICA is fitted separately to each complete participant recording for this descriptive analysis.

For every electrode, compute mean voltage, mean signed slope and mean absolute per-trial slope. Slopes use a fixed 25-ms local linear fit. Analyze pre −200 to −50 ms, early 50–150 ms, middle 150–300 ms and late 300–500 ms. The voltage baseline is −250 to −50 ms. No channels or windows were selected from the results.

## Waveform and first/second presentation effects

Each participant's regression includes comparison-specific intercepts, within-comparison elapsed time, waveform, position, and waveform × position. Every design was full rank, with condition numbers 5.4–6.6. The position coefficient represents second minus first averaged equally over waveforms. These adjustments account for specified linear effects; they do not eliminate all expectation, adaptation, preceding-waveform or recording-order influences.

Exact participant sign-flip tests enumerate 1,024 sign patterns. Max-stat correction covers 62 electrodes × four windows × three metrics within each effect family, followed by correction for three families. The tests assume independent participants and symmetric coefficient errors.

| Effect | Tests surviving corrected p ≤ .05 | Smallest corrected p |
| --- | --- | --- |
| Waveform | 0 / 744 | 0.7910 |
| First/second position | 0 / 744 | 0.0820 |
| Waveform × position | 0 / 744 | 1.0000 |

There is no corrected evidence of a consistent effect in this analysis. In particular, the position result does not justify assuming that first and second presentations are equivalent or free of confounds. The strongest position association was the 150–300-ms signed slope at FT7, with a mean second-minus-first difference of +4.53 µV/s and corrected p=.0820; this is an exploratory ranking, not an electrode-selection rule.

## Repeatable responses versus waveform-specific structure

Within each ordered comparison, 100 deterministic split-halves keep complete pairs together. Singleton ordered cells are omitted only from this check. Correlations compare 50–500-ms scalp × time patterns; waveform-deviation correlations first subtract the response common to all five waveforms. Values below are medians of participant median correlations, not independent split observations or confidence intervals.

| Representation | Presentation | Common-response r | Waveform-deviation r |
| --- | --- | --- | --- |
| Voltage | First | 0.386 | 0.020 |
| Voltage | Second | 0.686 | 0.078 |
| Signed slope trace | First | 0.176 | 0.006 |
| Signed slope trace | Second | 0.207 | 0.023 |

Thus, a repeatable common response can coexist with weak separation between waveforms. The higher second-presentation voltage repeatability is descriptive; it is not an inferential test of the difference between positions.

A separate electrode-level check correlates five-waveform × three-post-window feature patterns after removing the common waveform mean in each window. Even the largest electrode-specific group median correlations were modest: signed slope 0.253 first / 0.219 second; absolute slope 0.151 / 0.220; voltage 0.206 / 0.238. The leading electrodes differed across metrics and positions. These descriptive results do not establish a robust electrode or slope feature to select for decoding.

## Interpretation and next use

Slopes remain eligible as predefined classifier features regardless of statistical significance. The comparison contributes interpretability: with this estimator and these windows, waveform-specific slope structure was weakly repeatable. Whether slopes help jointly with voltage features requires held-out classification; this report does not answer that question.

The supplied filtering occurs before onset interpolation and can spread electrical artifacts into pre- and post-trigger periods. Physical trigger-to-current latency remains unverified. The pre-voltage window lies inside the voltage baseline, so a null effect there is not a clean negative control. These results cannot establish a cortical origin, isolate shape from charge/intensity, or rule out contextual information. Ten participants and broad correction also limit sensitivity.

MNE 1.13.2 can report ICA `n_iter_=500` after its weight-change stopping rule as well as after exhausting the configured iteration limit. The saved count alone does not establish the stopping reason; detailed optimizer traces were not retained.

## Reproduction and local report

See the [experiment run guide](../experiments/response_characterization/e002_electrode_position_slope/README.md). Exact extended dependencies are recorded in `requirements-diagnostics-lock.txt`. The complete local report is `output/task4_response_diagnostics_final_20261021/index.html`, with `RESULTS.md`, group and participant figures, full corrected tests, trial flags, repeatability summaries, and input/source/dependency provenance alongside it. All ten stages were consolidated through verified caches and rerun through the public experiment CLI without refitting ICA.

All 18 project tests passed. Participant-level outputs and caches remain Git-ignored; this note contains cohort aggregates only.
