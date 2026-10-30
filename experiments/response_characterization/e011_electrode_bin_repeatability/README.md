# E011 - Electrode-by-time-bin waveform repeatability

**Dataset:** Task 4 | **Status:** Completed | [Central registry](../../README.md)

Completed findings: [E011 repeatability results](../../../Docs/E011_REPEATABILITY_RESULTS.md).

An additional [individual-waveform explorer](#individual-waveform-and-participant-extension)
keeps every waveform and participant separate; it preserves the original pooled analysis.

Question: do waveform-specific EEG differences reproduce between trial halves
at an individual electrode and time bin? This is descriptive response analysis,
not a classifier, feature selector or independent replication.

## Fixed inputs and processing

Reuse the completed E010 simple **0.5-30 Hz band-pass** epochs, scalp-average
reference and [-250,-50)-ms baseline. No supplied ICA, notch, onset interpolation,
resampling or new amplitude rejection. Verify the earlier source/settings,
cohort, cell-mean reconstruction and all reused file hashes; record immutable
input provenance for this run. Old source paths resolve via the exact relocation
manifest. Original E010 results are read-only inputs.

All 14 predefined classifier electrodes are retained. First/second presentations
remain distinct. Full-rate epochs span [-300,1200) ms at 1 kHz. Slopes and
curvature use a **25-sample local quadratic Savitzky-Golay fit**, with delta=1 ms,
on each trial before binning or RMS. Centre derivatives use 12 recorded samples
on either side, so valid centres are [50,1188) ms. Filtering is noncausal and can
spread artifacts: neither a 50-ms start nor real local padding establishes a
physiologically clean response or physical-current alignment.

Representations are mean voltage (uV), signed mean slope and RMS slope (uV/s),
signed mean curvature and RMS curvature (uV/s^2). Second derivatives are included
alongside first derivatives; no scale/order is selected from the observed maps.
RMS is computed within each trial/bin before trial averaging or regression.
Derivative magnitude can reflect noise/artifacts as well as response shape.

Bins are fixed at **1, 5, 25, 50, 100 and 200 ms**, all anchored at +50 ms. Only
complete bins are reported; usable endpoints are 1188/1185/1175/1150/1150/1050 ms,
respectively. The 1-ms representation contains individual samples, not independent
1-ms-resolution physiological measurements. The derivative estimator is fixed
at 25 ms regardless of the reporting bin width. No reflected or extrapolated
edge samples enter reported features. A common [50,1050)-ms interval supplies
width-comparison summaries, avoiding different coverage when comparing widths.

## Paired splitting and nuisance adjustment

Start from E010's frozen complete-pair cohort. Ordered comparison strata are
defined by unordered waveform comparison and the trigger in its first position.
A stratum needs at least two pairs to enter repeatability; singleton-stratum
pairs are explicitly flagged and excluded only for this split analysis.
Every width/representation uses the same retained pairs and 100 deterministic
splits (seed 20261030 + participant ID). Both presentations always stay together.
Split repetitions overlap and are not independent observations.

Within each half, fit the existing fixed regression:

`comparison intercepts + centred within-comparison time + position + waveform + waveform x position`

Require full column rank and positive residual degrees of freedom. Use its four
sum-coded waveform coefficients and interactions to recover five sum-zero
waveform deviations separately at each position. Halves estimate nuisance terms
independently; no full-data nuisance regression precedes splitting. Save every
pair allocation, stratum/cell counts, rank and condition number. Adjustment only
handles the stated linear/context terms, not every order/history confound.

## Repeatability summaries and uncertainty

For each electrode/bin/position:

- **Profile r:** correlation of the five adjusted waveform values between halves,
  summarized by the median across splits. Five values give only four contrast
  dimensions; correlations can be unstable. Numerical-zero profiles yield NaN,
  with valid split counts saved, rather than inventing a zero correlation.
- **Split product:** mean over five waveforms of A_k times B_k, then mean across
  splits. It retains squared feature units and can be negative. Independent,
  mean-zero partition errors avoid a squared-noise offset, but shared drift,
  artifacts and misspecified regression can still bias this measure.
- Also retain half-contrast energy, split variability (not a confidence interval),
  fitted common response and full-data adjusted contrasts for interpretation.

Group maps give participants equal weight. Pointwise descriptive 95% bootstrap
intervals resample the participants (1,000 draws), never the 100 overlapping
splits as independent replicates. These are not corrected significance regions.
No thresholded significant-feature list or best-resolution recommendation is
produced. Squared-unit products cannot be ranked across derivative orders as
though they shared units. Neither averaged-response repeatability nor a positive
descriptive interval establishes single-epoch decoding or neural origin.

Split halves share comparison contexts, so repeatability need not generalize to
E012's held-out comparison contexts. Global E011 maps must not select features
for the same held-out evaluation. E012 either fixes candidates beforehand or
performs all selection inside training folds and refits it under permutations.

## Run

Use the existing diagnostics environment; no new dependencies.

```powershell
$env:MPLCONFIGDIR = Join-Path (Get-Location) 'output/matplotlib-config'
& ./.venv/Scripts/python.exe -m experiments.response_characterization.e011_electrode_bin_repeatability --output output/e011_repeatability_20261030
& ./.venv/Scripts/python.exe -m experiments.response_characterization.e011_electrode_bin_repeatability.report output/e011_repeatability_20261030
```

Use a fresh output directory. Local outputs include input/source hashes, manifests,
split assignments, per-participant maps, group pointwise intervals, support/audit
tables, HTML/Markdown reports and figures; all participant-derived data stay
Git-ignored. Errors leave explicit incomplete status and return a nonzero exit.
No previous report/PDF, recording, ICA or classifier result is overwritten.

Method context: [cross-partition inner products](https://pmc.ncbi.nlm.nih.gov/articles/PMC5421820/)
and [Savitzky-Golay derivatives](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html).

## Individual-waveform and participant extension

```powershell
& ./.venv/Scripts/python.exe -m experiments.response_characterization.e011_electrode_bin_repeatability.waveform_views --output output/e011_waveform_views_20261030
```

This separate output reuses the original cohort, features, all six widths and
exact pair splits. It does not overwrite original E011 outputs or their hashes.
The interactive HTML explorer defaults to one participant and waveform, with
separate first/second maps. It offers two explicitly different response views:

- **Total observed response:** equal observed-comparison-cell means for that
  waveform/position. This retains shared evoked responses and nuisance activity.
- **Waveform-specific deviation:** the original independently half-fitted,
  context/time/position-adjusted waveform contrast relative to the common response.

At each electrode/bin, save that waveform's split product, signed full-data
response, RMS half disagreement and same-sign split fraction. Averaging the five
specific-waveform products exactly reproduces the original pooled product.
A single waveform at one electrode/bin supplies only a scalar per half, so it
does not have its own scalar Pearson correlation.

Instead, report a **temporal pattern correlation** across fixed [50,1050)-ms bins
within one electrode, and a **joint pattern correlation** across 14 electrodes x
the same time bins, taking the median over the 100 paired splits. Neither
correlates across the five waveform identities or across complementary splits.
Coarse 200-ms temporal correlations use only five bins. Temporal dependence and
repeated overlapping splits prevent interpreting these as independent samples.

The explorer selects participant, waveform, representation, width, response view,
electrode and map statistic. All five waveform rows remain visible for the
selected participant. Curves show the fixed first split, not a best-looking
example; their displayed r still summarizes all splits over the fixed one second.
Optional group tables preserve waveform identity and give descriptive participant
bootstrap intervals, without corrected significance or binary repeatability labels.

The per-waveform correlations have a different axis/estimand from the original
five-waveform profile correlation; their magnitudes cannot be compared directly.
A positive waveform deviation is relative to other waveforms, not an isolated
waveform mechanism: one waveform differing from a common set can make all
grand-mean-centred contrasts repeat. A repeatable total response likewise need not
be waveform-distinct, neural or useful for held-out classification. No features
or channels are selected from these maps.
