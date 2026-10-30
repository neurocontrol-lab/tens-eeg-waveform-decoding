# E009 - Response uncertainty, earlier time coverage and sensitivity

**Dataset:** Task 4 | **Status:** Completed | [Central registry](../../README.md)

Preserve the completed multiscale, preprocessing-comparison and late time-resolved runs, while adding participant uncertainty, earlier kernel-guarded tests and a conditional recorded-background sensitivity pilot. This is an exploratory extension specified after the earlier negative results, not an independently preregistered replication or another classifier.

## Participant uncertainty

Read the saved multiscale split-half results. First take the median of the 100 dependent split correlations within each participant. Then resample the ten participants 10,000 times (seed 20261023). Summarize median correlations across participants and median paired differences against voltage (Gaussian voltage/DoG) or the original 25-ms slope (Gaussian derivatives), separately by presentation position and common/waveform-specific response.

The same bootstrap participant indices are used jointly for all summaries. This preserves pairing and does not turn 100 splits into 100 independent observations. Intervals are **descriptive pointwise 95% percentile intervals**, without multiplicity adjustment or new p values. They do not replace the earlier corrected tests or establish superiority/equivalence. Paired preprocessing contrasts also receive descriptive intervals for their median differences; earlier inferential tests used studentized mean differences, so the estimands are not identical.

## Earlier time-resolved analysis

Use the same 1,242 events/621 complete pairs, all 62 scalp channels, context-adjusted participant OLS model, verified original EEG and saved ICA fits as the preceding analyses. Both minimal and supplied branches are reconstructed; ICA is never refitted. No new rejection, channel selection or parameter tuning.

| Gaussian sigma | Post interval (ms) | Matched pre interval (ms) | Samples per period |
| --- | --- | --- | ---: |
| 2.5 ms | [50,480) | [âˆ’480,âˆ’50) | 430 |
| 5 ms | [50,480) | [âˆ’480,âˆ’50) | 430 |
| 10 ms | [51,480) | [âˆ’480,âˆ’51) | 429 |
| 20 ms | [91,480) | [âˆ’480,âˆ’91) | 389 |
| 40 ms | [171,480) | [âˆ’480,âˆ’171) | 309 |

The post start is `max(50, 11 + round(4*sigma))` ms on the original 1-kHz grid. The earliest local source sample is after +10 ms; pre kernels stay before âˆ’2 ms. Real padded epochs span âˆ’900 to +650 ms. These matched-duration intervals mirror the continuous half-open limits around onset; they are not one fixed time translation. Earlier continuous filtering can still spread artifacts.

Estimate the nine waveform/position/interaction voltage coefficient traces first, then apply the fixed linear Gaussian derivative. OLS and this temporal transform commute. Tests compare this shortcut with trial-level filtering; every coefficient in the prior late maps is independently reproduced on their exact sample support. Derivatives retain ÂµV/s units. This reduces repeated filtering without changing the fitted effects.

Joint whole-participant sign flips enumerate all 1,024 patterns at N=10. Within each of three effect families, maximum-statistic correction spans both branches, both periods, all five scales, all valid times and all electrodes: **492,776 locations**. Bonferroni across three families retains the original overall correction principle. The participant-error joint-symmetry and independence assumptions remain required. No data-selected peaks, cluster threshold or relaxed correction is used. This is a group response analysis, not held-out decoding or true scale invariance.

## Conditional sensitivity pilot

The separate `sensitivity` module calibrates the **expanded time-resolved waveform test**, using recorded participant waveform coefficient maps as fixed background magnitudes. It does not calibrate multiscale RMS/repeatability statistics or physiological population power.

- Randomize whole-participant coefficient signs, jointly over the full map. This retains each participant's recorded spatial/temporal structure, but imposes the null sign symmetry by construction. Existing genuine effects may contribute to these background magnitudes; they are not pure instrument noise. No coefficient centering or trial resampling.
- Use 32 distinct sign backgrounds (seed 20261023), shared across amplitude/latency/branch comparisons. Latency jitter draws are also shared between early and late scenarios. These finite conditional replicates are a pilot, not independent new participants or precise population-power estimates.
- Inject a fixed 15-ms-sigma Gaussian voltage pulse truncated at Â±60 ms, centred at 100 or 300 ms. Participant-level latency jitter is 0 or 10-ms SD, clipped at Â±2 SD. There is no trial-level jitter. Earliest possible injected sample: +20 ms.
- Nominal maximum class peak amplitudes: 0, 0.25, 0.5, 1, 2, 4 and 8 ÂµV. Five waveform multipliers: âˆ’1, âˆ’0.5, 0, 0.5, 1. These are benchmark differences, not claims about actual waveform-evoked EEG amplitudes or geometry.
- Fixed scalp weights: C3=1, CP1=0.5, C4=âˆ’1, CP2=âˆ’0.5, others zero. Their zero sum preserves average reference. This arbitrary sparse benchmark is not a cortical-source model.
- Introduce the same ideal signal into **already-preprocessed** branch inputs, before local differentiation/OLS, implemented by tested linear equivalence. This tests derivative/group-test detectability conditional on each processed background. It does **not** estimate signal attenuation by high-pass, notch, interpolation, low-pass or ICA.
- Recompute the exact null maxima under every injected alternative, retaining the full location family and factor-three correction. For efficiency, mathematically unchanged locations reuse their exact sign-orbit maxima with the appropriate sign-index permutation. Tests verify this against a full brute-force recalculation.
- Recovery requires a corrected waveform effect within nonzero injected support. An unrelated significant point is not recovery. At zero injection, enumerate all 1,024 sign backgrounds and tabulate family/union false-alarm fractions. This mechanically checks the finite sign-orbit test, not whether real EEG errors satisfy symmetry.

## Run

From the repository root, use the existing diagnostics environment; no dependency changes:

```powershell
& ./.venv/Scripts/python.exe -m pytest tests/test_e009_response_uncertainty_sensitivity.py tests/test_e009_response_sensitivity.py -q
$env:MPLCONFIGDIR = Join-Path (Get-Location) 'output/matplotlib-config'
& ./.venv/Scripts/python.exe -m experiments.uncertainty_and_sensitivity.e009_response_uncertainty_sensitivity --output output/task4_response_followup_new
& ./.venv/Scripts/python.exe -m experiments.uncertainty_and_sensitivity.e009_response_uncertainty_sensitivity.figures output/task4_response_followup_new
& ./.venv/Scripts/python.exe -m experiments.uncertainty_and_sensitivity.e009_response_uncertainty_sensitivity.sensitivity --source output/task4_response_followup_new --output output/task4_response_sensitivity_new
& ./.venv/Scripts/python.exe -m experiments.uncertainty_and_sensitivity.e009_response_uncertainty_sensitivity.figures --sensitivity output/task4_response_sensitivity_new
```

Always use fresh output directories. Defaults cover subjects 3â€“12. Custom subsets require at least three participants and cannot support more distinct sensitivity replicates than their 2^N sign patterns. Source input/configuration/EEG/CSV/ICA/coefficient hashes are recorded and checked; earlier numerical modules remain unchanged. The sensitivity module requires a complete observed follow-up.

Observed outputs: corrected full maps, location definitions, participant effects/audits, paired participant-bootstrap tables, `RESULTS.md`, `index.html`, three corrected p-map figures and an uncertainty figure. Sensitivity outputs: configuration, template/jitter cases, all replicate outcomes, recovery summaries, exact alternative null maxima, zero-injection orbit calibration, report and recovery curves. Everything participant-specific or generated remains in Git-ignored `output/`.

The `figures` command only re-renders saved arrays, uses nearest-neighbor display to avoid blending colors across excluded kernel support, and puts scales in numerical order. It records plotting provenance and checks that numerical input hashes are unchanged; it does not repeat inference.

The `--sensitivity` plotting option uses fixed, readable log-axis labels at the six injected positive amplitudes and suppresses crowded minor ticks. Neither plotting mode changes the statistics or recovery counts.

Canonical local runs: `output/task4_response_followup_20261023/` and `output/task4_response_sensitivity_final_20261023/`. See [the aggregate findings](../../../Docs/TASK4_RESPONSE_FOLLOWUP.md).

## Interpretation

The preprocessing comparison remains a comparison of complete bundles on the supplied-QC-selected cohort, not equivalence or step-level validation; stage ablations are deferred. Full-recording ICA remains descriptive, not classifier-safe fitted preprocessing. The later comparison's independent pre-voltage baseline is the useful matched voltage control; the original multiscale pre-voltage baseline overlap is still a limitation of that preserved run.

An apparent post-only association does not test a post-minus-pre difference. The local onset guards cannot correct earlier filter spread or unknown trigger-to-current latency. Higher repeatability does not establish discrimination or neural origin. A null group test cannot exclude participant-specific or latency-variable information. Conditional injection recovery does not turn the observed negative result into evidence of physiological absence.
