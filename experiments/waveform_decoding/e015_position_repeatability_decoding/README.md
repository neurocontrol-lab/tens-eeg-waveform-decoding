# E015 - Position-specific repeatability-selected waveform decoding

**Dataset:** Task 4 | **Status:** In progress | [Central registry](../../README.md)

Target: one separate five-class shrinkage-LDA decoder per participant and
presentation position, with 100 scalar features selected inside each training
fold. The label-free pool has 24,614 voltage, signed/RMS slope, signed/RMS
curvature and alpha/beta power features. A training-calibrated xDAWN basis adds
750 features. Voltage/derivative/component widths are 25/50/100/200 ms over
[50,1050) ms; power retains eleven overlapping 500-ms windows advanced by 50 ms.

The current implementation starts with a fail-closed metadata feasibility audit.
The original position/interaction terms are absent from the one-position
comparison/time nuisance model. Five adjusted waveform responses, rather than
one waveform's temporal profile, supply the candidate's split correlation.
Independent xDAWN calibration and A/B ranking halves must each retain all five
classes; both ranking halves must have identifiable nuisance designs. Calibration
prefers two epochs per class and falls back to one only on metadata feasibility.
The same A/B data score every feature family. Unsupported cases are reported,
not repaired using a minimum-norm coefficient vector or silent class removal.

Comparison-held-out outer folds use identical pair assignments for both
positions. A quantitative impulse-response audit sets an effective IIR guard at
an L1-tail tolerance of 1e-4, including baseline and derivative support. This
does not assert exactly compact support for the noncausal filter.

Feature ranking, xDAWN fitting, scaling and LDA must be training-only. Saved E014
full-cohort scores/filters cannot be used for selection. Any label-permutation
assessment must repeat all label-dependent calibration, scoring and selection.
The short cached pre-trigger period is not a matched one-second negative control.

E012's broader model comparison remains Planned. This experiment does not fit
CNNs, SVMs or manifold models. Numerical outputs and caches remain Git-ignored.

## Implemented stages

The label-free cache is complete for all ten participants (1,190 epochs). A
metadata audit checking 100 ranking splits found 11 of 20 participant/position
models supported in every comparison-held-out fold, and 178 of 200 supported
individual folds. Nine models have one or more failed folds. These are design
support findings, not decoding results. See the
[feasibility findings](../../../Docs/E015_DESIGN_FEASIBILITY.md).

The scoring and fold-fitting implementation uses a signed median split r,
requiring finite correlations in all ranking repetitions. Selection ranks its
positive square, with deterministic feature-ID tie breaking, and caps the set
at 100 positive-scoring features. Any shortfall is explicit; negative r is not
squared into apparent reliability. The matched ANOVA reference ranks the same
candidate pool using the same non-calibration training events. Both classifiers
fit training-only scalers and automatic-shrinkage LDA with equal class priors.
Calibration events may enter the final classifier fit; held-out events do not.

`evaluate.py` provides the strict observed-stage runner. It preflights all ten
folds before fitting a model and reports an unsupported model as ineligible,
without computing an accuracy from only its successful folds. It saves feature
IDs, calibration/ranking memberships, spatial filters, scaling/coefficients and
out-of-fold predictions. Position scores are calculated from pooled held-out
predictions, not averaged fold scores with missing test classes. This command
does not yet implement permutation inference and cannot establish above-chance
significance by itself. Real-data observed fitting is now complete: 11 strict supported models and all 20 unreliable exploratory models. See the [observed findings](../../../Docs/E015_OBSERVED_RESULTS.md). Permutation inference remains pending.

```powershell
& ./.venv/Scripts/python.exe -m experiments.waveform_decoding.e015_position_repeatability_decoding audit --audit-repeats 100 --output output/e015_feasibility_full_20261102
& ./.venv/Scripts/python.exe -m experiments.waveform_decoding.e015_position_repeatability_decoding prepare --output output/e015_feature_cache_20261102
# Strict supported models; unsupported models are explicitly reported:
& ./.venv/Scripts/python.exe -m experiments.waveform_decoding.e015_position_repeatability_decoding.evaluate --output output/e015_strict_observed_20261102
```

No old E011/E014 numerical algorithms, cached epochs, published images or
published reports are modified. The later permutation stage must recompute
label-dependent allocation, spatial fitting, scoring and selection. Its
exchangeability restrictions and handling of metadata-infeasible null designs
remain to be specified before inferential testing.

## Unreliable exploratory companion

At the user's request, all 20 participant/position cases were also run using a separately labelled exploratory pipeline. Outer comparison folds, complete-pair grouping and the temporal guard are unchanged. xDAWN fits all outer-training events, so its basis is not independent of internal ranking halves. Ranking uses observed comparison-balanced class profiles rather than regression-adjusted contrasts. Its r is an in-sample stability heuristic. All such outputs have `unreliable: true`.

Classes with fewer than two training epochs cannot appear in both ranking halves and are excluded from the correlation axis, explicitly recorded. A class absent from training cannot be learned: its probability stays zero and its errors remain in the five-class held-out score. Participant 12 P1 has one such fold, with four learned classes and eight xDAWN components. No minimum-norm adjusted contrast is substituted.

```powershell
& ./.venv/Scripts/python.exe -m experiments.waveform_decoding.e015_position_repeatability_decoding.exploratory --output output/e015_exploratory_observed_20261102
& ./.venv/Scripts/python.exe -m experiments.waveform_decoding.e015_position_repeatability_decoding.report
```

The observed report recomputes all stored scores from held-out predictions, checks selected feature counts, scaler validity, probability sums and source/input digests, and keeps strict and unreliable estimates separate. `--replace-report` regenerates only a recognized derived summary directory; numerical runs are not overwritten. Both observed runs selected exactly 100 features in every fold. No formal above-chance inference is claimed before a full label-permutation stage. The permutation helper preserves complete ordered-pair label vectors and their within-comparison position counts, but is not itself a completed null assessment.

## Confusion matrices and class recall

The local `output/e015_observed_summary_20261102/index.html` includes a matrix
viewer with pipeline, feature selector, presentation position and participant
controls. Rows are delivered waveforms and columns are predictions. The diagonal
is recall, not precision. Individual and pooled matrices show cell counts and
row percentages. The default equal-participant view averages row-normalized
matrices; its mean diagonal equals the reported mean participant balanced accuracy.
Pooled recall weights participants by their true-class trial counts and can differ.
Class support accompanies the recall table; unsupported strict models remain empty.
Blue marks correct predictions on the diagonal; orange marks off-diagonal errors.
Each has a labeled 0–100% legend, fixed across views. Stronger intensity means
a higher row-normalized percentage; colors do not encode statistical significance.

`class_recall.csv`, `class_recall_summary.csv` and `confusion_matrices.json` export
the same held-out results. `confusion.py` derives these without fitting classifiers
or changing predictions. Refresh this recognized derived report in place with:

```powershell
& ./.venv/Scripts/python.exe -m experiments.waveform_decoding.e015_position_repeatability_decoding.report --replace-report
```
