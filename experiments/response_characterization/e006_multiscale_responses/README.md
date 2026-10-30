# E006 - Multiscale EEG response characterization

**Dataset:** Task 4 | **Status:** Completed | [Central registry](../../README.md)

This exploratory experiment tests whether Gaussian-derivative response dynamics or other temporal scales expose repeatable waveform-specific structure. It adds optional difference-of-Gaussians (DoG) contrasts as a secondary representation. It does not fit a classifier or test true invariance to temporal stretching. All existing experiments and their outputs are read-only inputs.

Completed full-cohort findings are in [the results note](../../../Docs/TASK4_MULTISCALE_RESULTS.md).

## Run

Use the existing `requirements-diagnostics.txt` environment and completed diagnostic cohort/ICA caches:

```powershell
& ./.venv/Scripts/python.exe -m pytest -q --rootdir=. tests/test_e006_multiscale_responses.py
$env:MPLCONFIGDIR = Join-Path (Get-Location) 'output/matplotlib-config'
& ./.venv/Scripts/python.exe -m experiments.response_characterization.e006_multiscale_responses --output output/task4_multiscale_20261023

# Secondary DoG representations can be omitted before a run:
& ./.venv/Scripts/python.exe -m experiments.response_characterization.e006_multiscale_responses --no-dog --output output/task4_multiscale_without_dog

# Regenerate the report from saved summaries, without reloading EEG:
& ./.venv/Scripts/python.exe -m experiments.response_characterization.e006_multiscale_responses.report output/task4_multiscale_20261023
```

Use a fresh output directory. `--subjects 3` allows a single-participant descriptive run. Defaults read `output/task4_response_diagnostics_final_20261021/` and `output/task4_response_cache/`; both can be overridden. Missing or mismatched ICA fits fail explicitly; no fitting is performed. Participant outputs remain under Git-ignored `output/`. Failures are recorded in `status.json` and produce a nonzero exit status. Report completeness is stated explicitly.

## Reconstruction and fixed cohort

Raw XDF, master CSV, original numerical source and dependency hashes must match the prior diagnostic run. A completed cache must match that participant's exact prior provenance, channel order and ICA exclusions. Reapply the same continuous filtering, average reference, saved ICA, onset interpolation and low-pass. No EOG threshold search or ICA refitting is needed because the saved cleaning fit already contains the exclusions.

Before extracting new epochs, reproduce all previous fixed trial features within `rtol=2e-7, atol=2e-5` in their original ÂµV/ÂµV/s units, and require identical event/QC flags. Save the observed maximum error and ICA file hash. The original 62-electrode, whole-pair QC cohort is frozen. Additional real padding from âˆ’400 to +650 ms must be finite, in bounds, timestamp-regular and free of neighboring accepted triggers; any padding failure excludes a complete pair and is counted separately. The original voltage baseline, [âˆ’250,âˆ’50) ms, is retained. No continuous EEG or ICA fit is overwritten.

## Representations and guard intervals

- Original cleaned voltage and the existing 25-ms first-order local-linear slope serve as matched references.
- Gaussian-smoothed voltage at Ïƒ = 5, 10, 20 ms.
- First Gaussian derivatives at the same scales, calibrated with a discrete unit ramp so slopes retain ÂµV/s units despite kernel truncation.
- Optional signed DoG = G5âˆ’G10 and G10âˆ’G20 smoothed voltage, in ÂµV. DoG is temporal band-pass contrast, not first-derivative slope.

All filters operate only along time at the native sample rate, with Gaussian support truncated at Â±4Ïƒ. This is full-resolution Gaussian scale-space rather than a downsampled image pyramid. No scale-normalized pooling, peak selection or per-trial amplitude normalization is used. Scales retain physical time and voltage units.

Primary trace comparisons use [100,500) ms for **all** representations, including references. At Ïƒ=20 ms, the earliest contributing sample is +20 ms, beyond the explicit âˆ’2 to +10-ms blank. Fixed post feature windows are [100,200), [200,350), [350,500) ms; the pre diagnostic is [âˆ’250,âˆ’100) ms. Padding and direct blank overlap are validated for every scale. This guard does not remove artifact spread from the preceding zero-phase preprocessing. Responses earlier than +100 ms are outside the reported comparisons; null findings do not address those early responses.

For every trial/representation/electrode/window, calculate signed mean and RMS **before** condition averaging. RMS measures magnitude, including noise and artifacts. The earlier 50â€“500-ms diagnostic correlations must not be used as the comparator for this new 100â€“500-ms analysis.

## Repeatability and exploratory inference

Use 100 deterministic splits (seed 20261023), identical across representations. Split within ordered comparison cells, preserving both presentations of each pair. Singleton ordered cells are omitted only from repeatability. Give each observed comparison equal weight when averaging each waveform/position. Linear weighting exactly reproduces the existing `condition_means` method.

Correlate the two half-averages over electrode Ã— time for the common response, and waveform Ã— electrode Ã— time after removing the common waveform average. Also calculate waveform-deviation repeatability per electrode to avoid an individual high-amplitude channel determining the full-scalp result. Save all split correlations; summarize medians within participant first and then across participants. Repeated splits are not independent observations or confidence intervals. Shared full-recording ICA, temporal filtering and comparison contexts can contribute dependence between halves; this is a descriptive repeatability check, not independent replication.

Paired differences of participant median correlations compare each Gaussian derivative with the matched 25-ms reference. These differences are descriptive and do not claim significant improvement. Smoothing changes frequency weighting and temporal dependence; higher r does not alone demonstrate more waveform information.

For fixed mean/RMS features, reuse the previous participant regression: comparison intercepts, within-comparison centered elapsed time, sum-coded waveform, first/second position and their interaction. Require full-rank designs. Exact whole-participant coefficient sign flips use the maximum statistic across **all** operators/scales, mean/RMS summaries, four windows and 62 electrodes per effect family, followed by Bonferroni correction for three families. This assumes independent participants and symmetric coefficient errors; it is an exploratory model-conditional association test, not a physical stimulus randomization test. No channels, scales or windows are selected for classification from these full-cohort results.

## Outputs and limits

Each participant has an event manifest, reconstruction provenance, input/ICA hashes, model audit, complete marker, cell support, per-trial fixed features and float32 **display-only** evoked traces; feature fitting and repeatability use float64. Group outputs include corrected tests, participant/group repeatability medians, paired derivative differences, per-electrode summaries, and a Markdown/HTML report with fixed-channel time Ã— scale maps. Fixed illustrated electrodes are C3/C4/CP1/Pz; all 62 are analyzed.

The resulting plots characterize response dynamics and repeatability. They do not establish a neural origin, remove all position/history/charge confounds, validate physical trigger timing, prove absence of waveform information, or establish decoding accuracy. Full-recording ICA remains a descriptive transform, not a validated held-out classifier preprocessing procedure. The pre-voltage window overlaps the voltage baseline and is not a matched classifier negative control. Gaussian scales are chosen in advance; later tuning must remain inside training folds.

Conceptual motivation: [Itti, Koch and Niebur (1998)](https://doi.org/10.1109/34.730558), multiscale centerâ€“surround features. Implementation background: [SciPy Gaussian derivatives](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter1d.html) and [DoG band-pass filtering](https://scikit-image.org/docs/stable/auto_examples/filters/plot_dog.html). This temporal EEG adaptation is exploratory, not a validation of visual saliency as a neural decoding mechanism.
