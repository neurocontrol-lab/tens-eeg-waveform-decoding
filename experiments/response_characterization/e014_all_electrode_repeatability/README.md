# E014 - All-electrode, spectral and calibrated xDAWN repeatability

**Dataset:** Task 4 | **Status:** Completed | [Central registry](../../README.md)

Completed on 30 October 2026. See the [findings](../../../Docs/E014_REPEATABILITY_RESULTS.md).

[Open the interactive image report online](https://htmlpreview.github.io/?https://github.com/neurocontrol-lab/tens-eeg-waveform-decoding/blob/main/Docs/Reports/E014/index.html).

This extends repeatability, not classification. E012 decoding remains planned.
Keep all 62 scalp electrodes, voltage/slope/curvature, alpha/beta total power and
independently calibrated xDAWN features. Participant, waveform and first/second
presentation remain separate. E011's code, results and 14-channel reference are
read-only inputs. No automatic best-electrode/window/feature selection is added.

## Fixed scalp reconstruction and quality flags

Reconstruct XDF units, triggers and CSV sequence with the original audited loader.
Verify each raw recording hash. Apply the same 62-channel average reference,
fixed 0.5-30 Hz Butterworth forward/backward filter and [-250,-50)-ms baseline as
E010, now retaining all 62 channels. No ICA, notch, resampling or onset interpolation.
Keep the frozen E011 595-pair cohort and exact original two-half splits. Fail if
expanded channel support would silently change that cohort. Verify the original
14 full-rate epochs and every per-channel repeatability field against E011.

Raw continuous standard deviation and nonfinite fraction, and filtered post-epoch
amplitude/near-flat metrics, are saved for every channel. Flags use raw SD <0.1 uV,
post SD <0.1 uV, post peak-to-peak >500 uV or nonfinite raw data. These descriptive
flags do not automatically drop channels/pairs, interpolate EEG or establish
neural origin. Referencing can make a flat raw electrode look non-flat, so raw
and referenced summaries remain separate. Large repeatable frontal/artifact
responses must not be treated as useful neural features simply because they repeat.

## Sensor and alpha/beta power representations

Preserve E011's mean voltage, signed/RMS slope and signed/RMS curvature,
25-sample local quadratic derivative estimator, native 1-ms grid and
5/25/50/100/200-ms complete bins. The same half-specific nuisance model supplies
adjusted deviations; total responses retain observed-cell means.

Alpha (8-13 Hz) and beta (13-30 Hz) use **500-ms Hann periodograms**, constant
detrending, 2-Hz native spacing, exact-edge PSD integration and 10 log10(power / 1
uV^2). A fixed 1e-20-uV^2 numerical floor is audited. Eleven overlapping windows
start at 50,100,...,550 ms and end at 550,...,1050 ms. They advance by 50 ms, but
this is not independent 50-ms spectral resolution; alpha/beta are not estimated
in 1/5-ms windows. Heatmaps locate their centres (300-800 ms), with actual window
edges stored. This is total single-trial power, not isolated induced oscillations.
The simple band-pass attenuates the upper beta edge; unlike E005, these inputs
are filtered. Constant voltage baseline subtraction cancels under PSD detrending.

## xDAWN: calibration is independent of both repeatability halves

Fitting spatial filters on A/B themselves would bias a supervised-repeatability
assessment. Use 100 deterministic **calibration/A/B** complete-pair allocations:

- Every ordered-comparison stratum with >=3 pairs rotates three balanced groups.
- A two-pair stratum supplies one pair to A and B and none to calibration, retaining
  support instead of silently dropping subjects or identifiable contrasts.
- Require disjoint complete pairs, all five calibration classes and full-rank
  nuisance designs independently in A and B. Calibration is never scored as A/B.
- Fit xDAWN only on calibration epochs/labels over [50,1050) ms: two components per
  class, covariance regularization 0.1, full numerical rank. No classifier is fit.
- Resolve component sign from the largest filter coefficient, and normalize each
  component by calibration RMS only. The matched 62- and 14-electrode controls
  use the same calibration RMS convention and exact A/B allocations.

Evaluate voltage means at 25/50/100/200 ms. A fixed fitted linear projection
commutes with half-specific regression and binning; transform held-out condition
profiles rather than creating all projected trial arrays. This shortcut does not
move xDAWN fitting outside calibration. Store every fit and allocation.

Components track calibration class/rank, not fixed scalp electrodes or fixed brain
sources across fits. Per-component repeatability is conditional on this learning
procedure. Joint shape correlation demeans each feature time course before
flattening, equally for matched sensors/components, and is invariant to joint
component sign flips. This estimand differs from the original E011 joint r.
Compare xDAWN against its matched three-way sensor controls, not a two-half sensor
result with more evaluation trials. Curves show the fixed split-0 calibration
fit's held-out A/B traces; grey is their average, not an all-data fitted projection.

## Outputs and interpretation

The explorer selects participant, waveform, representation, bin/window support,
feature/electrode and total versus adjusted responses. Native channel labels and
quality flags remain visible. xDAWN components are explicitly labelled as learned
features. Group intervals are optional, descriptive participant bootstraps without
multiplicity-corrected significance or binary repeatability labels. No peak
selection, temporal alignment/calibration of neural latency or model training is
performed. Shared context, artifacts, variable support and temporal dependence
remain limitations. Observed maps must not select features for the same held-out
classification evaluation later.

## Run

Use the existing diagnostics environment. No new packages or deep-learning
dependencies are required. Create fresh cache/output paths for new runs.

```powershell
$env:MPLCONFIGDIR = Join-Path (Get-Location) 'output/matplotlib-config'
& ./.venv/Scripts/python.exe -m experiments.response_characterization.e014_all_electrode_repeatability.data
& ./.venv/Scripts/python.exe -m experiments.response_characterization.e014_all_electrode_repeatability --stage sensors
& ./.venv/Scripts/python.exe -m experiments.response_characterization.e014_all_electrode_repeatability --stage xdawn
& ./.venv/Scripts/python.exe -m experiments.response_characterization.e014_all_electrode_repeatability.report output/e014_repeatability_20261030
```

Both numerical stages must finish before the combined report is finalized.
Their configs/status/hashes remain separate; raw reconstruction is cached once.
The 7.21-GB numerical output, including arrays, CSVs, spatial fits and browser
payloads, remains Git-ignored. Its local index.html is an explicit exception.
A separate [image publication](../../../Docs/Reports/E014/index.html) is generated under Docs/Reports/E014/. It preserves
all participant, waveform, representation, bin, presentation and electrode/
component choices using pre-rendered map and curve sheets. The HTML crops images;
it does not fetch numerical map arrays or trial EEG. A static gallery supplies
direct image links without JavaScript. Compression is lossless for the rendered
pixels; fixed display resolution and palette discretization are documented.

```powershell
& ./.venv/Scripts/python.exe -m experiments.response_characterization.e014_all_electrode_repeatability.image_report --output Docs/Reports/E014 --workers 4
```

Use a fresh separate publication directory. The image export reads frozen result
arrays, verifies they do not change during rendering, and records source/image
digests. It does not rerun any numerical analysis or deploy the report. Keep the
publication's images/ folder beside its index.html and gallery.html when hosting.
Previously generated reports and research PDFs are preserved.
