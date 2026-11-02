# E016 — One-versus-rest waveform detectors

Observed analysis completed **2 November 2026**. Formal permutation inference is pending.

## Question and design

Does converting the E015 five-class model into separate waveform-versus-rest
detectors improve held-out discrimination using the same representation?

- **100 exploratory detectors:** participants 3–12 × two separate presentation
  positions × five target waveforms. All retain their E015 unreliable-source flags.
- **55 strict-reference detectors:** the 11 E015 participant/position cases that
  supported independent calibration and ranking. Position 1 includes participants
  5–9 and 11; position 2 includes participants 5–9. These are a different subset,
  not an independent replication or a matched-cohort comparison.
- Each fold reuses exactly the **100 E015 repeatability-selected scalar features**
  and saved spatial filters. The five target detectors share the same feature set.
  There is no new target-specific selection or spatial fitting.
- Features come from voltage, first and second derivatives, alpha/beta power and
  xDAWN components over **50–1,050 ms**. Input EEG uses the E014 simple processing
  branch: 0.5–30 Hz filtering, average reference and prestimulus baseline, without ICA.
- Binary LDA uses automatic covariance shrinkage, training-only standardization,
  equal target/rest priors, and a fixed **0.5 probability threshold**. The five
  independent probabilities need not sum to one.
- Validation preserves E015's ten held-out waveform comparisons, complete-pair
  grouping, separate positions and **15.312-second** minimum train/test trigger
  separation. No test labels enter fitting.

## Observed results

Equal-weight means across the available participant × target detectors within
each position. Binary balanced accuracy has a 50% reference; AUC has a 0.50 reference.

| Source | Position | Detectors | Balanced accuracy | Within-comparison AUC | Pooled OOF AUC |
| --- | --- | ---: | ---: | ---: | ---: |
| Exploratory | First | 50 | 45.86% | 0.5035 | 0.3437 |
| Exploratory | Second | 50 | 45.36% | 0.5085 | 0.3292 |
| Strict reference | First | 30 | 46.86% | 0.4710 | 0.3911 |
| Strict reference | Second | 25 | 49.86% | 0.5333 | 0.4362 |

For comparison, the original E015 five-class argmax predictions, converted into
target-versus-rest decisions on these same exploratory events, give mean binary
balanced accuracies of **45.88%** and **44.81%** at first and second positions.
These are binary reinterpretations of the original predictions, not E015's
five-class balanced accuracies.

There is no clear overall improvement from this binary formulation. Mean
exploratory sensitivity is **14.43% / 14.37%**, with specificity **77.28% / 76.35%**
for first/second presentations. Mean ordinary accuracy is **65.49% / 65.01%**;
the mean always-rest accuracy is **80%**, illustrating why ordinary accuracy
alone would be misleading.

Individual exploratory balanced accuracies range from **28.85% to 67.55%**.
For example, participant 6's first-position Sine detector has 67.55% balanced
accuracy, 12 positives among 59 epochs, and within-comparison AUC 0.6250.
These individual observations have not passed a null test or multiplicity correction.

## Why report two AUCs?

**Pooled OOF AUC** ranks all held-out positive and negative scores together,
including scores produced by different fold models. In this design the held-out
waveform composition changes between folds. Between-model score shifts can
therefore dominate this quantity; its low value should not be read as a clean
estimate of within-comparison discrimination or used to justify reversing scores.
Scikit-learn cautions that metrics computed on pooled cross-validation predictions
can differ from fold-wise evaluation, particularly for metrics that do not decompose
over samples. [Official cross-validation guidance](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html)

**Within-comparison AUC** ranks positive and negative events only when they were
scored by the same fitted fold model. Fold AUCs are weighted by
`n_positive × n_negative`, then averaged within each detector. Single-class test
folds contribute no AUC and are not assigned 0.5. Only **1–4 of ten comparisons**
are evaluable per exploratory target detector. This conditional diagnostic was
added after inspecting the pooled scores; it has a different estimand and does
not assess between-comparison score calibration. Both quantities are preserved.

Balanced accuracy above uses every held-out event at the original fixed threshold;
thresholds and score directions have not been optimized after looking at the results.

## Sparse support and reliability

Each participant/position target has only **3–21 positive epochs**. Participant 12's
first-position Square detector has one training fold with no Square examples.
That fold has an explicit constant-negative fallback, rather than a fabricated
LDA fit, and its held-out errors remain included. All other detector-folds fitted
binary LDA: **1,549 fits plus one fallback**, covering 1,550 detector-folds.

The exploratory source reuses training events for xDAWN fitting and its internal
repeatability ranking, and lacks the strict independent calibration partition.
Its flags remain visible. Strict references have the stronger partitioning design
but still small samples. Neither set establishes statistical significance.

## Verification and reproducibility

- All **155 detector summaries** were recomputed from saved held-out predictions.
- All **1,550 detector-fold AUC/BA pairs** were checked against saved predictions.
- All **310 parent folds** reproduced E015 scaling, selected exactly 100 columns,
  and kept training/test event IDs disjoint.
- **685 frozen input/implementation hashes** verified unchanged.
- **110 tests passed**, including held-out independence of scaling and fitting,
  saved spatial projection equivalence, missing-class handling, imbalance metrics,
  and exclusion of cross-model pairs from within-fold AUC.

Code and commands: [E016 run guide](../experiments/waveform_decoding/e016_one_vs_rest_decoding/README.md).
E015 context: [observed results](E015_OBSERVED_RESULTS.md).

Local Git-ignored outputs:

- `output/e016_binary_observed_20261102/`: fitted fold models, predictions,
  configuration, source hashes and reliability audits.
- `output/e016_binary_summary_20261102/index.html`: lightweight filterable report,
  two heatmaps, CSV summaries and verification record. Keep companion files beside
  the HTML when copying the local report.

The [shared standalone report](https://htmlpreview.github.io/?https://github.com/neurocontrol-lab/tens-eeg-waveform-decoding/blob/main/Docs/Reports/E016/index.html)
embeds the plots and small summary exports in `Docs/Reports/E016/index.html`;
the full numerical run remains local.

Formal inference requires an appropriate full-pipeline null that repeats
label-dependent feature selection and spatial fitting, with multiplicity control.
Permuting labels after freezing the true-label feature masks would not provide
that test. Trigger-to-current timing remains unverified. No CNN, manifold model,
new preprocessing comparison or target-specific feature selection was added here.
