# Task 4 response-analysis refinements

**23–24 October 2026.** Participant-level uncertainty and earlier, scale-specific time coverage strengthen the interpretation of the preceding response analyses. The expanded tests found no corrected group waveform, presentation-position or interaction effect. The sensitivity pilot recovered stronger injected reference responses, but no 0.25–1 µV injections; participant latency variation reduced recovery. The original multiscale, preprocessing-comparison and late time-resolved results remain preserved. This follow-up is exploratory and does not establish absence of waveform information or evaluate another classifier.

## Participant uncertainty

The 100 split-halves from the multiscale experiment are dependent repeated partitions, not independent replications. The follow-up first takes each participant's median correlation over splits, then resamples the ten participants 10,000 times (seed 20261023). It reports medians across participants and median paired differences, keeping each participant matched across representations. The intervals are descriptive pointwise 95% percentile intervals, without multiplicity adjustment; they do not replace the earlier corrected tests.

Waveform-specific Gaussian-derivative repeatability relative to the original 25-ms local-linear slope:

| Gaussian sigma | Position | Paired median difference in r | Descriptive 95% interval |
| --- | --- | ---: | --- |
| 5 ms | First | +0.005 | −0.001 to +0.010 |
| 5 ms | Second | −0.004 | −0.012 to +0.001 |
| 10 ms | First | −0.018 | −0.046 to +0.018 |
| 10 ms | Second | +0.011 | −0.002 to +0.042 |
| 20 ms | First | −0.032 | −0.089 to +0.048 |
| 20 ms | Second | +0.033 | −0.015 to +0.103 |

All six intervals span zero. Gaussian-smoothed voltage and DoG waveform-repeatability differences also had intervals spanning zero. These summaries provide no clear repeatability advantage for the tested alternatives. They do not test their joint usefulness as classifier features.

For the response common across waveforms, the 20-ms derivative had positive paired changes relative to the local-linear slope: +0.137 r for first presentations (interval +0.040 to +0.187) and +0.261 for second presentations (+0.129 to +0.358). These pointwise descriptive intervals support the earlier distinction between more repeatable common structure and weak waveform-specific differences. They are not new multiplicity-corrected evidence or proof of neural origin.

The complete output includes intervals for all representation/common-response summaries, all paired multiscale differences and all 60 preprocessing contrasts. The new preprocessing intervals concern median paired differences; the earlier corrected tests studentized mean differences. Their estimands differ, and the intervals must not be used as substitute p values.

Three of the 60 preprocessing pointwise intervals excluded zero: second-presentation post-minus-pre branch differences for voltage and 5/10-ms smoothed voltage. These descriptive median contrasts do not overturn the earlier globally corrected mean-contrast tests, and they do not establish a pipeline advantage. No scales or processing settings were selected from them.

## Earlier time-resolved effects

All ten participants, 1,242 retained events in 621 complete pairs and all 62 scalp electrodes were unchanged. Both minimal and supplied preprocessing branches were reconstructed from verified EEG/CSV files and original saved ICA fits; no ICA was refitted and no new exclusions were made. The same participant regression adjusts comparison intercepts, within-comparison elapsed time, waveform, first/second position and their interaction.

The earlier common [180,480)-ms interval omitted earlier responses even at smaller scales. This extension uses fixed scale-specific support:

| Gaussian sigma | Post interval (ms) | Matched pre interval (ms) |
| --- | --- | --- |
| 2.5 ms | [50,480) | [−480,−50) |
| 5 ms | [50,480) | [−480,−50) |
| 10 ms | [51,480) | [−480,−51) |
| 20 ms | [91,480) | [−480,−91) |
| 40 ms | [171,480) | [−480,−171) |

At every scale, the local derivative kernel avoids the explicit −2 to +10-ms onset interpolation and uses actual recorded padding. The intervals have matched duration and mirrored continuous half-open limits; they are not one fixed translation. These guards cannot undo spread introduced by earlier continuous filtering or resolve unknown trigger-to-current delay.

Exact joint-vector participant sign flips enumerate all 1,024 patterns. Maximum-statistic correction jointly spans both branches, both periods, all five scales, every valid time and all electrodes: 492,776 locations per effect family. A further factor of three covers waveform, position and interaction. No correction was relaxed or time/channel selected from a peak.

| Effect | Corrected significant locations | Smallest corrected p |
| --- | ---: | ---: |
| Waveform | 0 / 492,776 | 1.0000 |
| Position | 0 / 492,776 | 0.6973 |
| Waveform × position | 0 / 492,776 | 1.0000 |

This exploratory extension was specified after the preceding negative results. It is not an independently preregistered replication. Non-significance cannot exclude participant-specific effects, variable latency or information useful jointly for prediction. A post-only significant effect would not itself establish a post-minus-pre difference.

## Preprocessing interpretation retained

The preceding comparison answers whether the **complete processing bundles** change waveform-specific repeatability on the cohort retained by supplied-pipeline QC. It does not establish equivalence, validate individual stages or identify a causal mechanism for processing effects. Minimal recordings are not guaranteed to be artifact-free. No stage ablations were added.

Use the later comparison's separate earlier pre-voltage baseline for its matched control; the preserved original multiscale pre-voltage window still overlaps its voltage baseline. Constant baselines cancel for derivatives and DoG. Full-recording ICA remains appropriate only to the stated descriptive use, not a claim of classifier-safe fitted preprocessing.

## Conditional sensitivity calibration

The separate calibration targets the expanded **time-resolved waveform test**, rather than multiscale RMS, repeatability, classification or signal attenuation by preprocessing.

A fixed 15-ms-width Gaussian voltage pulse is centred at 100 or 300 ms. Nominal maximum class peaks are 0, 0.25, 0.5, 1, 2, 4 and 8 µV; class multipliers are −1, −0.5, 0, 0.5 and 1. Sparse scalp weights sum to zero: C3=1, CP1=0.5, C4=−1, CP2=−0.5. This is a benchmark pattern, not a neural-source model or a claim about actual stimulation-waveform responses. Between-participant latency jitter is 0 or 10-ms SD, clipped at ±2 SD, with no trial-level jitter. The pulse is truncated at ±60 ms, so its earliest possible source sample is +20 ms.

Recorded participant coefficient maps provide fixed background magnitudes, with whole-participant sign randomization. Thirty-two distinct sign backgrounds are shared across amplitudes, latency scenarios and branches. An identical ideal signal is introduced into each **already-preprocessed** branch before Gaussian differentiation/OLS, using verified linear equivalence. Thus the exercise assesses detectability conditional on the processed recordings; it does not measure attenuation by ICA or the continuous filters, physiological population power or a minimum detectable real neural effect.

Each alternative retains the full location family and factor-three correction, recomputing its exact sign null. Only mathematically unchanged locations reuse their sign-orbit statistics; the optimized calculation matches full brute force in tests. Recovery requires a corrected effect within nonzero injected support. The complete zero-injection sign orbit checks the test's conditional false-alarm behavior, without validating real-data symmetry assumptions.

Across the complete zero-injection orbit, **46/1,024 sign backgrounds (4.49%)** produced an alarm in at least one of the three corrected effect families. This is the conditional finite-orbit calibration, not an empirical demonstration of the false-positive rate in new participants.

Recovery inside the injected support, out of 32 conditional replicates per setting:

| Pulse centre | Jitter SD | Nominal maximum class peak | Minimal recovered | Supplied recovered |
| --- | --- | --- | ---: | ---: |
| 100 ms | 0 ms | 2 µV | 14/32 | 32/32 |
| 100 ms | 10 ms | 2 µV | 2/32 | 5/32 |
| 300 ms | 0 ms | 2 µV | 24/32 | 25/32 |
| 300 ms | 10 ms | 2 µV | 9/32 | 12/32 |
| 100 ms | 0 ms | 4 µV | 32/32 | 32/32 |
| 100 ms | 10 ms | 4 µV | 23/32 | 28/32 |
| 300 ms | 0 ms | 4 µV | 32/32 | 32/32 |
| 300 ms | 10 ms | 4 µV | 32/32 | 32/32 |

At 0.25, 0.5 and 1 µV, no injected signals were recovered in either branch under any latency setting (0/32 in each case). At 8 µV, every setting recovered the signal in 32/32 cases. These are nominal EEG pulse peaks, not stimulation-current amplitudes; the maximum class-to-class peak difference in the benchmark is twice the stated amplitude.

This test can recover stronger, consistent instances of the reference pattern, and recovery declines with participant latency variation. Therefore the observed negative findings remain compatible with weaker or latency-variable effects. The pilot does not estimate a general neural detection threshold: pulse shape, sparse scalp pattern, fixed recorded effect magnitudes, full correction and only 32 conditional replicates constrain it. The supplied branch's higher recovery in some settings reflects the processed background under an identical **post-processing** injection; it is not evidence that the pipeline preserves real neural signals better or that a particular preprocessing step is superior.

## Verification and outputs

The shortcut of fitting voltage coefficients before the fixed linear derivative operator reproduces every prior late-window coefficient, with maximum absolute difference 1.08×10⁻¹¹ in the derived coefficient units. Its equivalence to trial-level filtering is also tested. All 62 project tests passed, including full optimized-versus-brute-force sensitivity simulations. The final sensitivity run completed without errors, reproduced the original recorded waveform null maxima and verified its frozen input/source hashes. Its case definitions and zero-injection calibration matched the interrupted, slower implementation exactly. Numerical input/source hashes and plotting provenance are retained. Plot re-rendering uses nearest-neighbor display so excluded kernel support is not blended into apparent p-map values, and fixed amplitude ticks prevent overlapping labels on sensitivity curves.

See the [run guide](../experiments/uncertainty_and_sensitivity/e009_response_uncertainty_sensitivity/README.md). Canonical local outputs are Git-ignored `output/task4_response_followup_20261023/` and `output/task4_response_sensitivity_final_20261023/`. They contain full coefficient/p maps, participant audits and interval tables, simulation settings/cases, exact null maxima, recovery summaries and HTML/Markdown reports. Participant-specific data remain excluded from Git.
