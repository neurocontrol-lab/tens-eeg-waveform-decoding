# Task 4: does minimal preprocessing preserve waveform-specific responses?

**Follow-up:** [Participant uncertainty and response-analysis refinements](TASK4_RESPONSE_FOLLOWUP.md) add descriptive paired intervals. This comparison remains conditional on the supplied-QC-selected cohort and concerns complete processing bundles; it does not establish equivalence or validate individual stages. The original corrected tests below are unchanged.

**23 October 2026.** Minimal preprocessing did not reveal strong, repeatable waveform-specific EEG slope patterns in the tested 100–500-ms interval. Small numerical differences between processing branches did not survive correction. These are response-repeatability results, not classification accuracies, and do not establish equivalence between pipelines or absence of waveform information.

## Matched comparison

All ten participants (subjects 3–12) completed both branches using the same 1,242 events in 621 pairs and the same 62 scalp electrodes. Eligibility was frozen from the preceding multiscale analysis. Expanded real-data padding passed for every selected event; no additional events were removed.

- **Minimal:** recorded EEG with verified units, average reference, interval-specific baseline subtraction and the feature operators. No added high-pass, notch, low-pass, ICA or onset interpolation. Acquisition-system filtering may already exist.
- **Supplied:** the preceding filtering pipeline, average reference, original saved 33-component ICA fit and EOG exclusions, −2 to +10-ms interpolation and 190-Hz low-pass. ICA was not refitted.

The post-trigger trace was [100,500) ms with baseline [−250,−50) ms. Its matched pre-trigger control was [−500,−100) ms with its own earlier baseline [−850,−650) ms. Both trace and baseline are exact 600-ms translations. Local Gaussian kernels do not cross recorded trigger onset; earlier continuous filtering can still spread contamination in the supplied branch.

Both branches used voltage, a 25-ms local-linear slope, Gaussian-smoothed voltage and Gaussian first derivatives at σ=5/10/20 ms, and two secondary difference-of-Gaussians representations. The same 100 whole-pair splits were used within ordered waveform comparisons. Repeatability used 599 pairs; 22 singleton comparison-cell pairs could not be split and were omitted from this calculation only.

For each split, the common response across five waveforms was subtracted to obtain waveform-specific deviations. Correlations compared these electrode-by-time patterns between halves, separately for first and second presentations. Each participant's median correlation across splits was computed before the group median.

## Results

Post-trigger waveform-specific derivative repeatability, expressed as group median correlation **r**:

| Gaussian derivative | Presentation | Minimal | Supplied |
| --- | --- | ---: | ---: |
| σ=5 ms | First | 0.013 | 0.011 |
| σ=5 ms | Second | 0.033 | 0.010 |
| σ=10 ms | First | −0.006 | −0.003 |
| σ=10 ms | Second | 0.052 | 0.013 |
| σ=20 ms | First | −0.033 | 0.006 |
| σ=20 ms | Second | 0.041 | 0.044 |

Minimal processing gave small numerical increases for second-presentation derivatives at 5 and 10 ms, without a consistent improvement across scales or positions. The DoG representations also remained weak. Common responses shared across waveforms were more repeatable than waveform-specific deviations in both branches.

Exact participant-level sign-flip tests compared branches post-trigger, pre-trigger, and on the post-minus-pre difference. A single maximum-statistic correction covered all 60 comparisons (three contrasts × ten representations × two positions). **None survived correction.** The smallest corrected p values were 1.0000, 0.6289 and 0.8359, respectively. These tests concern preprocessing differences in repeatability; they do not test classification against 20% chance.

The group median event-wise maximum-channel peak-to-peak amplitude was 47.3 µV with minimal processing versus 40.1 µV with supplied processing post-trigger. Relative 60-Hz-band power was also higher with minimal processing. Neither branch had post-trigger events exceeding the diagnostic 500-µV threshold; pre-trigger counts were three minimal and one supplied. No flags changed eligibility. Spectral fractions depend on their total-power denominators and are not direct measures of neural signal-to-noise.

## Interpretation and verification

This comparison provides no clear evidence that the supplied processing was hiding repeatable waveform-specific slopes at these scales. It compares the complete processing bundles and cannot identify the contribution of individual stages. Pre-trigger activity can include anticipation and preceding-stimulus responses, especially before second presentations, so it is not a guaranteed null. Trigger-to-current delay remains unverified. Earlier responses, classification and true scale invariance remain outside this experiment.

The reconstructed supplied branch reproduced prior fixed features exactly and prior post-trigger correlations within 1.11×10⁻¹⁶. All eight new tests passed; the recorded full-suite check passed 42 tests. Numerical outputs and participant details remain in Git-ignored `output/task4_preprocessing_comparison_final_20261023/`, including `RESULTS.md`, `index.html`, CSV summaries and figures. See the [experiment README](../experiments/signal_quality/e007_preprocessing_comparison/README.md) for settings and reproduction commands.
