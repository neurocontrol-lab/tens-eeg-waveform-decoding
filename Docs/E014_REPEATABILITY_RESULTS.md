# E014 - All-electrode, spectral and calibrated xDAWN repeatability

**Dataset:** Task 4, participants 3-12. **Analysis date:** 30 October 2026.

This extends the individual-waveform repeatability explorer to all **62 scalp
electrodes**, alpha/beta power and independently calibrated xDAWN components.
Participant, waveform and first/second presentation remain separate in the
primary maps. This is response characterization; no new classifier, best-feature
selection or latency alignment is performed. E012 decoding remains planned.

## Cohort, preprocessing and preserved reference

The analysis retains E011's **595 complete pairs / 1,190 events** and its exact
100 two-half paired allocations. All 62 channels use the same simple processing:
62-channel average reference, fixed 0.5-30 Hz Butterworth forward/backward filter,
and [-250,-50)-ms voltage baseline. No supplied ICA, resampling, additional
channel rejection or additional pair rejection is introduced.

The original 14-channel full-rate epochs reproduce exactly (maximum absolute
difference 0 uV). For all participants, widths and voltage/derivative
representations, the original per-electrode response, split product,
disagreement, sign agreement, example traces, energy and temporal correlations
also reproduce within the recorded numerical tolerances. A joint correlation
over 62 channels is a different spatial summary from one over 14 channels.

## Representations

| Representation | Spatial features | Temporal support |
| --- | --- | --- |
| Mean voltage | 62 electrodes | 1/5/25/50/100/200-ms complete bins |
| Signed mean and RMS first derivative | 62 electrodes | Same bins; fixed 25-ms local quadratic estimator |
| Signed mean and RMS second derivative | 62 electrodes | Same bins and estimator |
| Alpha power, 8-13 Hz | 62 electrodes | Eleven 500-ms Hann windows, advanced by 50 ms |
| Beta power, 13-30 Hz | 62 electrodes | Same spectral windows |
| Matched calibration/A/B sensor controls | 62 or original 14 electrodes | 25/50/100/200-ms bins over [50,1050) ms |
| Calibration-only xDAWN | Ten learned components | Same bins and held-out A/B trials as matched controls |

Voltage and derivative maps retain valid complete bins within [50,1188) ms;
their temporal correlation summaries use [50,1050) ms. Native 1-ms spacing is
not independent 1-ms physiological resolution. Signed and RMS derivatives answer
different questions and retain different units.

Spectral windows start at 50,100,...,550 ms and end at 550,...,1050 ms. Their
displayed centres are **300-800 ms**. Adjacent windows overlap substantially;
the 50-ms step is not independent spectral resolution. Periodograms have 2-Hz
native frequency spacing, constant detrending and exact-edge band integration.
Features are total single-trial power in dB re 1 uV^2, with an audited numerical
floor. They do not isolate induced power. The 30-Hz band-pass edge attenuates
upper beta activity, so these features differ from the earlier E005 inputs.

## Independent calibration for xDAWN

Each participant has 100 deterministic calibration/A/B allocations of complete
pairs. Calibration is disjoint from both evaluation halves. Ordered-comparison
strata with at least three pairs rotate balanced thirds; two-pair strata supply
A and B only. Every allocation retains the cohort, all five calibration classes
and identifiable nuisance designs in both evaluation halves.

xDAWN fits only calibration epochs and labels over [50,1050) ms, using two
components per waveform and 0.1 covariance regularization. Component signs and
RMS normalization are determined from calibration only. The matched 62- and
14-sensor controls use the same A/B trials and calibration-only RMS scaling.
There are **1,000 spatial fits** across the ten participants and no classifier
fits. A fitted projection is reused across the four bin widths.

Component names identify learned waveform/class-rank slots; they are not fixed
scalp electrodes or stable anatomical sources across fits. The joint shape
correlation removes each feature's temporal mean before flattening, equally for
all three matched representations. It is invariant to a joint sign flip in A
and B. Compare these matched representations with one another; their trial
counts, normalization and joint-r definition differ from the two-half views.
Grey xDAWN curves show the preselected split-0 held-out A/B average, rather than
an all-data projection.

## Descriptive matched comparison

For orientation, the following values average the participant/waveform/position
median joint-shape correlations across the four predefined bin widths. Each
participant contributes equally. These summaries do not replace the separate
participant/waveform maps and are not corrected significance tests.

| Matched representation | Total observed response | Adjusted waveform-specific deviation |
| --- | --- | --- |
| Original 14 sensors | 0.222 | 0.050 |
| All 62 sensors | 0.207 | 0.047 |
| Ten xDAWN components | 0.325 | 0.038 |

In this descriptive comparison, xDAWN has greater agreement for the **total**
response, without a corresponding increase in the adjusted waveform-specific
summary. Shared responses, comparison context and artifacts can all repeat.
These results do not establish a decoding improvement or identify a reliable
single-epoch feature. Individual waveform, electrode/component and presentation
patterns remain the intended objects of inspection.

## Channel-quality notes

All 620 participant/electrode entries have raw and processed quality summaries.
Three entries exceed the descriptive post-epoch 500-uV peak-to-peak flag:
participant 5 Fpz and Fp2, and participant 8 Fp2. No channel or pair is removed
because of these flags. Absence of a flag does not establish neural origin or
exclude lower-amplitude artifacts. Strong frontal activity and derivative/power
responses require the same interpretation caution as voltage responses.

## Outputs, sharing and limits

The canonical local explorer is
`output/e014_repeatability_20261030/index.html`. Its controls follow the E011
individual-waveform report, with variable electrode/component counts, spectral
window centres and quality notes. The original E011 report is preserved.

Numerical outputs include 8,800 participant/waveform summary rows, 880 separate
waveform group-summary rows, 600 original-14 reproduction checks, 100 browser
payloads and 880 representation/view/width result arrays. Calibration
allocations, nuisance-design checks, all spatial fits and input/source hashes
are retained locally. Stage provenance is separate from rendering provenance.

The canonical E011 and E014 local index.html pages are explicitly unignored,
with their companion numerical data still local. The full E014 numerical output
is 7.21 GB and remains ignored, together with the raw reconstruction cache.

A separate [image edition](Reports/E014/index.html) in Docs/Reports/E014/ preserves all participant,
waveform, first/second presentation, representation, bin and electrode/component
choices. Its HTML switches/crops pre-rendered map and curve sheets instead of
fetching numerical map arrays. A static gallery links directly to the sheets.
Images use fixed pixel resolution and a discrete display palette, followed by
lossless WebP compression. Exact arbitrary-sample numeric readout remains a
capability of the local numerical report. No estimates are recomputed, selected
or changed for publication. Source-array and image digests are recorded in the
publication manifest. Keep the images directory beside the HTML when sharing.

The maps supply products, response magnitude, disagreement and sign fractions
at individual feature/bin locations. Temporal correlations span a declared
time course; a single scalar location has no correlation on its own. Group
intervals are descriptive participant bootstraps. No binary repeatability
threshold, corrected feature discovery, physical-current timing validation or
population calibration is claimed. Selecting electrodes, times or learned
features for future decoding must occur inside its training data.

See the [E014 run guide](../experiments/response_characterization/e014_all_electrode_repeatability/README.md)
and [central experiment registry](../experiments/README.md).
