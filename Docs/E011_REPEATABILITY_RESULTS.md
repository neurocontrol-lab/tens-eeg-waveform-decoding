# E011 - Electrode-by-time-bin waveform repeatability

**30 October 2026.** Completed the fixed voltage, first-derivative and
second-derivative repeatability analysis on all ten current-protocol Task 4
participants. Native 1-ms maps and 5/25/50/100/200-ms bins were retained, with
separate first/second presentations. Overall location-averaged split-profile
correlations were small; the descriptive findings do not establish a best bin
width, informative electrode list or advantage of higher derivatives.

## Inputs and paired support

Reuse E010's verified simple **0.5-30 Hz Butterworth band-pass**, scalp-average
reference and [-250,-50)-ms baseline, without ICA, resampling, onset interpolation
or new amplitude rejection. All 14 predefined classifier electrodes remain.
E010 supplied 1,238 events in 619 complete pairs. Repeatability requires at least
two pairs in an ordered comparison stratum; **24 singleton-stratum pairs** were
flagged and omitted for splitting only, leaving **1,190 events in 595 pairs**.
Every representation/bin width uses this same retained cohort.

Use 100 deterministic within-stratum splits per participant, keeping each pair's
two presentations together. Every half fits comparison intercepts, centred
within-comparison elapsed time, position, four sum-coded waveform contrasts and
four waveform-by-position terms independently. All 2,000 half designs had full
rank; their largest condition number was 10.43. Some waveform/position cells have
only one event in a half, so support and estimation uncertainty remain important.
No design was dropped or feature selected after inspecting results.

## Derivatives, bins and estimands

Compute first and second derivatives per trial using a fixed **25-sample local
quadratic Savitzky-Golay fit** at 1 kHz. Physical units are uV/s and uV/s^2.
Summaries are mean voltage, signed mean slope, RMS slope, signed mean curvature
and RMS curvature. RMS is calculated within each trial/bin before regression.
The derivative-estimation width is fixed independently of reporting bin width.

Real local support restricts valid centres to [50,1188) ms, inside Jason's
1.2-second maximum. Bins are anchored at 50 ms; only complete bins are retained.
Endpoints for 1/5/25/50/100/200 ms are respectively 1188/1185/1175/1150/1150/1050 ms.
No reflected/extrapolated derivative-edge samples enter reported features.
The 1-ms grid is sample spacing, not independent 1-ms temporal resolution.

For each electrode/bin/position, compare five fitted, sum-zero waveform values
between halves. Save median split-profile correlation, mean amplitude-sensitive
split cross-product and valid split counts. Correlation has only four waveform
contrast dimensions and can be unstable. The cross-product retains squared
physical units and can be negative; it is not a normalized reliability coefficient.
Store common fitted responses and adjusted waveform estimates separately.

Group maps weight participants equally. Pointwise descriptive 95% bootstrap
intervals use 1,000 participant resamples; overlapping split repetitions are not
treated as independent participants. No multiplicity-corrected significance
test or automatic feature selection was performed.

## Fixed one-second summaries

For comparable coverage, first average per-location median split-profile r over
all 14 electrodes and complete bins in **[50,1050) ms**, then average participants.
This is an average of local correlations, not a correlation of one full-epoch
pattern and not classifier performance.

Selected predefined widths are shown below. Each cell gives first / second
presentation mean r; the full report includes all six widths, position-specific
uncertainty, per-bin maps and split-product estimates.

| Representation | 25 ms | 50 ms | 100 ms | 200 ms |
| --- | ---: | ---: | ---: | ---: |
| Mean voltage | +0.018 / -0.015 | +0.023 / -0.012 | +0.025 / -0.021 | +0.032 / -0.039 |
| Signed mean slope | -0.015 / -0.006 | -0.001 / +0.007 | +0.045 / +0.045 | +0.029 / -0.005 |
| RMS slope | -0.043 / -0.026 | -0.032 / -0.027 | -0.037 / -0.022 | -0.025 / -0.024 |
| Signed mean curvature | -0.009 / -0.034 | -0.008 / -0.023 | +0.026 / -0.030 | +0.022 / -0.003 |
| RMS curvature | -0.041 / -0.034 | -0.049 / -0.034 | -0.066 / -0.025 | -0.079 / -0.052 |

Across all 60 representation/width/position summaries, mean location r ranged
from -0.079 to +0.045. One r interval was entirely above zero: signed slope at
100 ms for second presentations, r=0.0446 with a descriptive interval approximately
[0.001,0.085]. Four split-product intervals were above zero. These are uncorrected
descriptive observations across multiple comparisons, not confirmed findings or
evidence that a selected representation improves decoding. Some intervals were
entirely negative; negative split estimates can arise from unstable/noisy fitted
contrasts and do not establish physiological inversion of waveform responses.

The inspected group maps show local positive and negative patches rather than
an obvious uniformly repeatable waveform signature. Coarser bins change or cancel
temporal structure; higher derivatives did not establish a general improvement.
These broad averages can obscure participant-specific or localized responses,
so small averages do not prove that waveform information is absent.

## Limits and next step

This is a within-recording, context-adjusted exploratory analysis. Halves share
comparison contexts. Shared drift, filter spread, artifacts, unequal support and
misspecified nuisance terms can still affect repeatability. An independent-noise
interpretation of split cross-products is not guaranteed. Squared-unit products
cannot be ranked across derivative orders as though they shared units. No matched
pre-trigger repeatability analysis is provided here; physical current timing is
still unverified.

E012 remains **planned**: one fixed 50-1050-ms epoch, predefined 25/50/100/200-ms
feature widths for shrinkage LDA and RBF SVM, plus compact EEGNet on sampled
voltage over the same epoch. Any selection/tuning/early stopping must stay inside
training data with appropriate grouping. Full-cohort E011 maps will not choose
the best channels, bins or representations for the same held-out evaluation.
No decoder was fitted in E011 and no classifier result changed.

## Reproduction and verification

See the [E011 run guide](../experiments/response_characterization/e011_electrode_bin_repeatability/README.md).
Canonical local output is Git-ignored `output/e011_repeatability_20261030/`,
containing HTML/Markdown reports, per-participant/group numerical maps, 15 figures,
pair allocations, cell counts, design audits, input/source hashes and pointwise
participant-bootstrap intervals. Source evoked means reproduced exactly
(maximum error 0.0 uV); frozen input/source hashes remained unchanged throughout.
Original E010 outputs, research PDFs, ICA and classifier results were preserved.

All **81 project tests passed**, including eight E011 checks of physical derivative
units/quadratic recovery, complete-bin support/RMS ordering, whole-pair splits,
singleton handling, nuisance adjustment, scale behavior, participant resampling
and negative cross-products. Figures were rendered and visually inspected.

## Extension: each waveform and participant kept separate

**30 October 2026.** A separate explorer now addresses repeatability for one
waveform without averaging over the five waveform identities or over participants
in its primary view. The original pooled report, source records and numerical
results remain preserved. Canonical local extension:
`output/e011_waveform_views_20261030/index.html`.

It retains the same 595 pairs, all representations/widths and exact split
allocations, and separates total observed responses from adjusted waveform
deviations. Every participant/waveform/position has product, signed response,
half-disagreement and sign-agreement maps. The mean of the five individual
specific-waveform products reproduces the original pooled product at every
electrode/bin. Agreement was checked for all 300 participant/representation/width
combinations; no new exclusions, classifier or feature selection were added.

For one waveform, **temporal pattern r** compares the halves across the fixed
one-second time course at one electrode. **Joint pattern r** compares the
14-electrode x one-second patterns. Each is summarized by the median over 100
splits. These correlations use different axes from the original correlation of
five waveform values at a single electrode/bin, so their numerical magnitudes
cannot be compared directly. A single waveform/electrode/bin contains a scalar
per half, where Pearson correlation is undefined.

For illustration, the predefined 50-ms voltage representation gives the following
equal-participant mean **joint pattern r**, retaining separate waveform rows:

| Waveform | Adjusted deviation: first / second | Total response: first / second |
| --- | ---: | ---: |
| Sine | -0.014 / +0.020 | +0.198 / +0.278 |
| Square | +0.066 / -0.084 | +0.124 / +0.298 |
| Triangle | +0.052 / +0.122 | +0.142 / +0.353 |
| Linear decrease | +0.139 / +0.002 | +0.311 / +0.448 |
| Linear increase | +0.051 / +0.133 | +0.243 / +0.359 |

There is participant heterogeneity: individual adjusted joint correlations in
this fixed representation span approximately -0.542 to +0.539 across waveform/
position combinations. Thus pooled summaries do not characterize every waveform
or participant. At group level, the linear-decrease first-presentation adjusted
estimate has a descriptive interval [0.040,0.243]; this uncorrected observation
across many waveform/representation/width comparisons is not a confirmed
waveform-specific finding or a basis for post-hoc feature selection.

Repeatable total responses may be shared across all waveforms or reflect nuisance
activity. Adjusted deviations are relative contrasts: a single waveform that
differs from an otherwise common set can make several centred contrasts repeat.
They are not evidence of five independently distinct waveform mechanisms.
No binary repeatable/not-repeatable classification or multiplicity-corrected
significance claim is made. Sparse support and correlated temporal samples still
limit interpretation; 200-ms temporal correlations contain only five time bins.

All **86 project tests passed**, including five extension checks of product
decomposition, common-response versus waveform-specific repeatability,
single-waveform isolation, correct correlation axes and signed browser-data
encoding. Browser-client logic was exercised against real local payloads for
62 control scenarios; a static waveform-trace preview was visually inspected.
