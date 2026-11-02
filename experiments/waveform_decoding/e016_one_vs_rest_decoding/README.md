# E016 - One-versus-rest waveform detectors on fixed E015 features

**Dataset:** Task 4 | **Status:** In progress | [Central registry](../../README.md)

One detector per participant x presentation position x target waveform gives
100 exploratory binary detectors. An additional 55 strict-reference detectors
can be fitted for E015's 11 supported participant/position cases. These are
model configurations, not independent datasets; cross-validation fits each
configuration repeatedly. No CNN or target-specific feature selection is added.

Each detector reuses the exact 100 repeatability-selected scalar columns and
saved spatial basis of its parent E015 fold. Cached voltage means are projected
through the saved xDAWN matrix, and scalar feature IDs and source scaling are
checked. The five targets share the same columns within a fold. Standardization
is fitted on training events only; each binary LDA uses automatic covariance
shrinkage and equal target/rest priors. A fixed 0.5 probability threshold is used.

The original held-out comparisons, complete-pair assignment, position separation
and 15.312-second temporal embargo are unchanged. No held-out labels enter the
fit API. Source exploratory flags remain attached. If a target is absent from
training, an explicit constant-negative fallback is recorded; no LDA is fitted
for that fold/target and its held-out errors remain in the final score.

Metrics include pooled out-of-fold binary balanced accuracy (reference 50%),
within-comparison ROC-AUC, and diagnostic pooled ROC-AUC (reference 0.5).
Within-comparison AUC compares scores from the same fitted fold model, weighted
by the number of positive-negative pairs. Pooled AUC also compares scores across
fold models and is not interchangeable with it. Single-class test folds have undefined fold AUC and
balanced accuracy, rather than invented chance values. Sensitivity, specificity,
precision, average precision and positive counts are retained. Ordinary accuracy
is accompanied by the always-rest baseline. Fixed features isolate the change
from five-class decoding to target detection; scores are not target-selected.

This stage provides observed estimates only. Formal inference must rerun all
label-dependent E015 feature selection and spatial fitting under permutations,
and address the multiple detector comparisons. Frozen true-label features must
not be used for an ostensibly full-pipeline permutation test.

## Observed run — 2 November 2026

All 100 exploratory detectors and 55 strict references completed. Exploratory
mean binary balanced accuracy was **45.86% / 45.36%** at first/second positions;
within-comparison AUC was **0.5035 / 0.5085**. There is no clear overall gain over
the existing E015 representation. Strict-reference balanced accuracy was
46.86% / 49.86% on its smaller eligible subset. No permutation inference is complete.

One participant-12 first-position Square fold had no positive training samples
and uses an explicit constant-negative fallback. All other 1,549 detector-folds
fitted LDA. All five targets within each parent fold use the same exact 100
columns; the independent binary probabilities need not sum to one.

[Full findings and metric interpretation](../../../Docs/E016_BINARY_RESULTS.md)

[Waveform and participant subgroup audit](../../../Docs/E015_E016_SUBGROUP_FINDINGS.md)
compares the saved E015 and E016 predictions without refitting. It retains class
support, sensitivity and specificity, and separates nominal-reference exceedances
from significance. Run `python -m experiments.waveform_decoding.e016_one_vs_rest_decoding.subgroup_audit`
to reproduce its local CSV tables in a fresh output directory.

```powershell
$env:OPENBLAS_NUM_THREADS='2'
$env:OMP_NUM_THREADS='2'
$env:MPLCONFIGDIR=Join-Path (Get-Location) 'output/matplotlib-config'
& ./.venv/Scripts/python.exe -m experiments.waveform_decoding.e016_one_vs_rest_decoding
& ./.venv/Scripts/python.exe -m experiments.waveform_decoding.e016_one_vs_rest_decoding.report
```

Both commands require fresh output directories. Use `--output` to choose a new
run, and `--run` on the report command to select its numerical source. Defaults:
`output/e016_binary_observed_20261102/` and
`output/e016_binary_summary_20261102/`. Open the latter's `index.html` locally;
the HTML, images and CSVs form a small report bundle. Numerical outputs and this
local bundle remain Git-ignored. Summary findings above are versioned in Docs.
