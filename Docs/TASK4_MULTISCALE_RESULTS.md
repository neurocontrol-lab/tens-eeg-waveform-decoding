# Task 4 multiscale EEG response findings: 2026-10-23

**Follow-up:** [Participant-bootstrap uncertainty and earlier time-resolved tests](TASK4_RESPONSE_FOLLOWUP.md) supplement this preserved analysis. The new descriptive intervals do not replace the corrected tests below.

Gaussian smoothing made the waveform-averaged response more repeatable, particularly for second presentations. It did not reveal a clearly repeatable waveform-specific signature at the tested scales. Gaussian derivatives and the optional difference-of-Gaussians (DoG) contrasts showed weak waveform-deviation repeatability. No fixed response-feature effect survived the planned correction. These are exploratory response diagnostics, not new classification results or evidence that waveform information is absent.

## Questions and implementation

The first question was whether smoother derivative estimates expose reproducible rising/falling dynamics that were weak with the original 25-ms local-linear slope. The second was whether examining multiple temporal scales reveals waveform-specific structure. True invariance to stretching/compressing a response remains a separate, untested hypothesis.

The new experiment uses Gaussian widths σ = 5, 10 and 20 ms at the original 1-kHz sample rate. It analyzes smoothed voltage and first Gaussian derivatives; discrete derivative kernels are calibrated on a unit ramp to retain µV/s units. Signed DoG(5,10) and DoG(10,20) are secondary µV-valued band-pass contrasts, not first derivatives. Original voltage and the existing 25-ms slope are matched references. No scale pooling or data-driven peak/channel selection is applied.

All ten participants, subjects 3–12, were processed from the original recordings using their verified saved ICA fits. No ICA was refitted. Reconstructed earlier features agreed to a maximum absolute error of 1.36×10⁻¹¹ in the original feature units, with identical prior event/QC flags. The same 1,242 events in 621 pairs were retained; expanded epoch padding caused no additional exclusions.

## Shared time support and comparisons

Every representation uses [100,500) ms for trace repeatability. This is narrower than the earlier [50,500)-ms experiment, so the reference slope is recomputed on the new support and the same trial splits. At the largest Gaussian width, ±4σ spans ±80 ms; the first reported post-trigger value therefore uses samples from +20 ms onward, beyond the explicit onset interpolation. Real epoch padding spans −400 to +650 ms. This does not remove artifact spread from the earlier continuous filters. Responses before +100 ms are outside this analysis.

For each representation, calculate signed mean and per-trial RMS in pre [−250,−100), early [100,200), middle [200,350), and late [350,500) ms. Regress these features against waveform, position and their interaction, with comparison intercepts and within-comparison elapsed time. All participant designs were full rank. Exact participant sign flips use max-stat correction across all ten representations, two summaries, four windows and 62 electrodes, followed by correction for three effect families.

| Effect | Corrected p ≤ .05 | Smallest corrected p |
| --- | --- | --- |
| Waveform | 0 / 4,960 | 1.0000 |
| First/second position | 0 / 4,960 | 1.0000 |
| Waveform × position | 0 / 4,960 | 0.1582 |

These tests assume independent participants and symmetric coefficient errors. Their feature definitions, windows and correction family differ from the earlier experiment; changes in corrected p-values are not evidence that a prior effect has disappeared. Ten participants and the broad correction limit sensitivity.

## Repeatability

Use the same 100 complete-pair splits for every representation, splitting within ordered comparisons. Repeatability uses 599 pairs: 22 singleton ordered-comparison cells are omitted only from this check, while all 621 QC-eligible pairs enter the response-feature models. Give observed comparisons equal weight. First calculate each participant's median correlation over splits, then the median across participants. Waveform-deviation correlations subtract the response averaged equally over the five waveforms. Repeated splits are dependent and their spread is not a confidence interval.

| Representation | First: waveform-deviation r | Second: waveform-deviation r |
| --- | --- | --- |
| Original voltage | ≈0.000 | 0.053 |
| Gaussian-smoothed voltage, σ=20 ms | 0.007 | 0.067 |
| Reference slope, 25-ms local fit | 0.011 | 0.012 |
| Gaussian derivative, σ=5 ms | 0.011 | 0.010 |
| Gaussian derivative, σ=10 ms | −0.003 | 0.013 |
| Gaussian derivative, σ=20 ms | 0.006 | 0.044 |
| DoG(5,10) | 0.010 | −0.005 |
| DoG(10,20) | −0.018 | 0.025 |

The 20-ms Gaussian derivative had a median paired change from the reference of −0.032 for first presentations and +0.033 for second presentations. These are medians of within-participant differences, which need not equal differences of group medians. No inferential test of repeatability improvement was performed; these changes do not establish superiority.

The common response was more repeatable. For second presentations, its group median r rose from 0.708 for original voltage to 0.794 for 20-ms smoothed voltage; the reference slope had r=0.227 versus 0.507 for the 20-ms Gaussian derivative. This suggests that smoothing emphasizes shared response structure more strongly than the distinctions between waveforms. Frequency weighting and temporal autocorrelation also change with smoothing; a higher r alone does not establish more discriminative or neural information.

## Interpretation and artifacts

The tested multiscale derivatives did not provide a consistent repeatability advantage across both presentation positions. The DoG features also showed weak waveform-specific structure. This does not determine their joint usefulness in a single-epoch classifier. Any future choice of representation, scale or electrode based on prediction must remain inside training folds or be fixed before testing.

The report includes time × scale maps at four fixed electrodes (C3/C4/CP1/Pz), C3 waveform traces and all-62-electrode repeatability summaries. These maps are descriptive, not significance maps. Strong-looking local features must reproduce across trials before being interpreted as waveform signatures. Full-recording ICA, earlier zero-phase filtering and shared comparison contexts can contribute dependence between halves; this is a descriptive repeatability check, not independent replication. Timing uncertainty, electrical artifacts, charge/intensity differences, prior stimulation, expectation and nonlinear drift remain limitations. The pre-voltage window overlaps the voltage baseline and is not a matched classifier negative control.

## Reproduction

See the [run guide](../experiments/response_characterization/e006_multiscale_responses/README.md). The complete local report is `output/task4_multiscale_20261023/index.html`; accompanying files contain all corrected tests, repeatability correlations, paired comparisons, event flags, reconstruction checks, source/dependency/input hashes and figures. Participant-level outputs remain Git-ignored. The extended diagnostic environment was reused without dependency changes. All 24 project tests passed, including six new scientific invariants.
