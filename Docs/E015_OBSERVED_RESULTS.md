# E015 — Position-specific top-100 decoding: observed findings

**Date:** 2 November 2026. **Status:** Observed fitting complete; permutation inference pending.

All 20 participant/position cases were run with an explicitly unreliable exploratory pipeline. The original strict design also fitted 11 supported cases; nine were ineligible under that design. No model fit failed. Every fold selected 100 scalar features, separately per participant and position.

The primary selector ranks positive median split-profile correlations, squared after retaining the sign. A matched ANOVA selector uses the same candidate pool and shrinkage-LDA decoder. Features are mean voltage, signed/RMS slope, signed/RMS curvature, alpha/beta power and xDAWN amplitudes. The usual pool has 25,364 candidates. One exploratory fold lacking a training class has eight rather than ten xDAWN components, giving 25,214 candidates.

## Mean held-out balanced accuracy

| Pipeline | Position | Participants | Repeatability-selected | ANOVA-selected |
| --- | --- | ---: | ---: | ---: |
| strict | 1 | 6 | 17.79% | 12.45% |
| strict | 2 | 5 | 16.44% | 13.85% |
| exploratory | 1 | 10 | 13.63% | 14.13% |
| exploratory | 2 | 10 | 11.67% | 12.45% |

Both pipeline means are below the nominal 20% five-class reference. These are descriptive point estimates, not permutation-test results. The two positions in the strict subset have different participant coverage; their means are not a paired carryover test. The exploratory and strict pipelines also use different scoring/calibration procedures and must not be pooled into a single accuracy.

## Individual repeatability-selected results

| Participant | Strict P1 | Strict P2 | Unreliable exploratory P1 | Unreliable exploratory P2 |
| --- | ---: | ---: | ---: | ---: |
| 3 | Ineligible | Ineligible | 7.60% | 8.04% |
| 4 | Ineligible | Ineligible | 7.82% | 5.62% |
| 5 | 14.05% | 20.17% | 14.58% | 12.50% |
| 6 | 14.72% | 25.32% | 25.47% | 15.69% |
| 7 | 18.03% | 10.78% | 10.48% | 15.32% |
| 8 | 19.30% | 8.14% | 23.76% | 13.60% |
| 9 | 23.17% | 17.78% | 16.51% | 5.02% |
| 10 | Ineligible | Ineligible | 9.75% | 8.33% |
| 11 | 17.49% | Ineligible | 10.70% | 16.37% |
| 12 | Ineligible | Ineligible | 9.67% | 16.19% |

## Class recall and confusion matrices

The interactive report includes confusion matrices for each participant, position and selector, plus pooled and equal-participant summaries. Rows are true delivered waveforms; columns are predicted waveforms. Diagonal percentages are class recall. The table below averages participant recalls equally, consistent with the reported mean balanced accuracy.

| Pipeline | Position | Selector | Sine | Square | Triangle | LinDec | LinInc |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| strict | 1 | repeatability | 15.34% | 18.41% | 15.39% | 16.30% | 23.55% |
| strict | 1 | anova | 9.13% | 23.32% | 6.59% | 16.77% | 6.45% |
| strict | 2 | repeatability | 19.15% | 16.71% | 13.97% | 18.69% | 13.68% |
| strict | 2 | anova | 19.30% | 10.26% | 17.73% | 17.24% | 4.71% |
| exploratory | 1 | repeatability | 11.67% | 19.86% | 11.62% | 13.58% | 11.44% |
| exploratory | 1 | anova | 12.87% | 19.62% | 11.83% | 19.37% | 6.95% |
| exploratory | 2 | repeatability | 6.35% | 11.60% | 12.83% | 20.26% | 7.29% |
| exploratory | 2 | anova | 8.56% | 7.34% | 15.81% | 20.89% | 9.63% |

Pooled count and row-normalized matrices retain event support; equal-participant percentages give each participant the same weight within each true class. Different trial counts can make these summaries differ. Class-level differences are descriptive, not significance-tested. Individual and summarized recalls are exported in `class_recall.csv` and `class_recall_summary.csv`, with all matrices in `confusion_matrices.json`.

## Meaning of the unreliable flag

All exploratory models retain the original comparison-held-out outer folds, complete-pair grouping and temporal embargo. xDAWN fits only outer-training epochs/labels, but there is no calibration set independent of the internal A/B ranking halves. The resulting split correlation is an in-sample selection heuristic, not an independently validated repeatability estimate. Observed comparison-balanced class profiles replace regression-adjusted waveform contrasts; they can retain context and drift effects.

Ranking uses classes having at least two training epochs, so both A/B halves can contain them. If a waveform is absent from training, the classifier cannot learn or predict it. Its predicted probability remains zero, and its held-out errors remain in the five-class score; no class or test epoch is silently removed.

Participant 12, position 1, has one outer fold without Square training examples. That fold uses four training/ranking classes and eight xDAWN components. Its output is explicitly flagged with `missing_training_class` and `fewer_than_five_ranking_classes`.

## Controls and reproducibility

- Strict xDAWN calibration is disjoint from both ranking halves; all feature families use those same ranking data. Calibration used two epochs per class in 107 strict folds and one per class in 3 folds, based only on metadata feasibility.
- Selection and scaling are fitted on training data. Held-out values enter transformation and prediction only. Tests verify that changing held-out signals does not alter selected features or fitted coefficients.
- The effective filter guard remains seven seconds per side at an L1-tail tolerance of 1e-4. Baseline/derivative support gives a 15.312-second train/test trigger separation; complete training pairs inside this guard are purged.
- First and second presentation models remain separate. Their performance difference alone does not identify physiological carryover.
- No matched one-second pre-trigger control or permutation inference has run. The cached pre-trigger period is only 300 ms. The code includes a tested conditional pair-label shuffle helper; a full null runner still requires the complete scoring/spatial-fitting pipeline to be rerun.
- Saved artifacts include all held-out predictions, confusion matrices, selected scalar feature IDs, calibration/ranking memberships, spatial filters, scaling parameters, classifier coefficients, support failures and source/input digests.

## Interpretation

The current observed results do not show a cohort-level accuracy above the 20% reference. A few individual point estimates exceed 20%, but no above-chance significance or improvement over ANOVA is established. Repeatable waveform-mean patterns do not necessarily translate into useful single-epoch features. These runs do not establish absence of waveform information, and unreliable cases remain explicitly marked.

## Local output directories

[Open the shared interactive report](https://htmlpreview.github.io/?https://github.com/neurocontrol-lab/tens-eeg-waveform-decoding/blob/main/Docs/Reports/E015/index.html)
for feature-selection explanations, participant results and confusion matrices.
Its self-contained publication is `Docs/Reports/E015/index.html`.

- `output/e015_strict_observed_20261102/`: strict supported fits and explicit ineligible cases.
- `output/e015_exploratory_observed_20261102/`: all 20 flagged exploratory models.
- `output/e015_observed_summary_20261102/`: aggregate tables, feature-family counts, figure and local HTML report.

See the [run guide](../experiments/waveform_decoding/e015_position_repeatability_decoding/README.md) and [design audit](E015_DESIGN_FEASIBILITY.md).
