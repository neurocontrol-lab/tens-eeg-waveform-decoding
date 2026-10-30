# Task 4 baseline feature audit: 2026-10-23

The audit found correct training-only scaling and substantial feature redundancy. Automatic shrinkage resolves the primary covariance rank deficiency, but covariance variability and held-out extremes remain. There is no basis here to attribute the previous chance-level decoding to a scaling implementation error, or to conclude that regularization has removed all estimation problems.

## Scope and verification

Read the exact saved baseline arrays for all ten participants: 1,244 events in 622 pairs, five representations (post, pre, late, onset and order/history), 50 participant/representation matrices. This is the original baseline cohort, not the 1,242-event ICA response diagnostic cohort. Reconstruct 50 within-participant and ten LOSO folds for each representation: 300 training-fold audits in total. Every fold's event membership matched the original saved classifier predictions.

All arrays were finite. No EEG feature was constant or near constant under the documented unit-aware review threshold. One participant-level order/history indicator was constant; this is an intentionally binary context feature. Order indicators were also constant in some training folds. These flags were recorded without removing columns or trials.

The actual baseline scaler was fitted only on each training fold. Its fitted means matched training means exactly; standardized training means and nonconstant SDs agreed with zero and one to numerical precision. Reconstructed class-specific Ledoit–Wolf covariances matched the actual fitted LDA covariance exactly. No classifier scores were recomputed and no features were selected.

## Primary post-stimulus representation

The primary baseline uses 126 features: nine 50-ms bins × 14 fixed electrodes, in µV.

| Diagnostic | Within participant | LOSO |
| --- | --- | --- |
| Training epochs per fold | 88–116 | 1,102–1,130 |
| Empirical covariance rank-deficient folds | 50 / 50 | 0 / 10 |
| Regularized covariance rank-deficient folds | 0 / 50 | 0 / 10 |
| Median regularized condition number | 90.9 | 140.2 |
| Median empirical entropy effective rank | 9.7 / 126 | 15.3 / 126 |
| Median fold maximum held-out absolute z | 5.09 | 9.36 |
| Largest held-out absolute z | 16.93 | 36.99 |

With five waveform classes, the empirical within-class covariance has rank at most min(p, n_train − 5). Rank deficiency is therefore expected for all primary within-participant folds. The effective ranks also show that the nominal 126 features contain substantial redundant covariance structure. Effective rank is not the number of informative neural dimensions.

Large held-out standardized values show that correctly fitting a scaler does not eliminate recording/context shifts, participant heterogeneity or artifacts. Median held-out fractions exceeding |z|=5 were 0.029% within participant and 0.391% LOSO; the largest fold fractions were 11.45% and 3.88%, respectively. The local report records feature names and event identities for inspection. These observations do not establish a unit error or justify automatic clipping or exclusion.

## Conditioning versus resampling variability

Ten descriptive resamples per fold draw complete pairs within subject/comparison strata. Keep the outer training scaler fixed and re-estimate empirical and automatic-shrinkage covariances. Median relative Frobenius changes for the primary features were:

| Evaluation | Empirical covariance | Regularized covariance |
| --- | --- | --- |
| Within participant | 0.303 | 0.373 |
| LOSO | 0.188 | 0.204 |

Shrinkage improved numerical conditioning but did not reduce this particular resampling-variability measure. These are changes around each estimator's own reference covariance, not errors relative to an unknown population covariance. They do not prove that shrinkage harms decoding. Overlapping folds/resamples are dependent; ten resamples are a compact diagnostic, not confidence intervals or a significance test.

## Implications and reusable checks

Further representations should keep feature count modest and preserve shrinkage. A small xDAWN spatial representation is a reasonable next comparison: train it using waveform labels inside each training fold, project held-out epochs without refitting, then apply the fixed temporal bins and training-only scaling/LDA. Refit xDAWN during every permutation test. It may combine weak distributed responses even though individual-electrode contrasts were nonsignificant; improvement is not established by this audit.

Rerun the numerical audit after adding features or changing units, channels, windows, preprocessing, eligibility or splits. Reuse verified event provenance and the audit functions. For learned representations, audit the transformed arrays separately inside each fold. Do not use full-cohort outlier plots to select channels, exclusions or parameters for the same held-out evaluation.

See the [audit run guide](../experiments/signal_quality/e003_voltage_feature_audit/README.md). The complete local report is `output/task4_feature_audit_final_20261023/RESULTS.md`, with distribution tables, fold diagnostics, class shrinkage coefficients, covariance resamples/eigenspectra, input hashes and fold membership. Participant-level outputs remain Git-ignored; this note contains cohort aggregates. Five new scientific-invariant tests passed; all 29 project tests at the time of this audit passed. The separate [xDAWN experiment](../experiments/waveform_decoding/e004_xdawn_decoding/README.md) implements the follow-on decoding comparison; this baseline audit itself contains no xDAWN fits.
