# Task 4 xDAWN waveform-decoding comparison: 2026-10-23

This fixed xDAWN representation gave a small increase in the point estimates but did not demonstrate reliable above-chance decoding or a reliable improvement over the sensor baseline. It improved within-participant covariance conditioning; the same did not hold for the pooled unseen-participant fits. The result concerns this representation, preprocessing and validation design, not the absence of waveform information in EEG.

## Fixed experiment and verification

All ten participants, subjects 3–12, contributed the same 1,244 events in 622 pairs as the original minimal baseline. Original event identities, eligibility and validation folds were reproduced. Full-rate epoch reconstruction reproduced the saved sensor features to a maximum absolute error of 5.8 × 10⁻¹² µV, and sensor predictions/probabilities matched the original run. The comparator therefore uses the same trials and splits rather than the separate 1,242-event ICA diagnostic cohort.

The pipeline retains the original 14 electrodes, fixed scalp-average reference, voltage baselines, 50–500-ms post-trigger interval and matched −500 to −50-ms pre-trigger control. No temporal filtering, ICA, stimulation interpolation, resampling or additional exclusions were introduced.

An independent replay from the reconstructed sensor-bin arrays exactly reproduced all 2,178 reused sensor-null balanced-accuracy scores, including LOSO. This additional check is recorded in the local `sensor_null_replay_validation.json`.

Fit xDAWN separately inside each training fold and independently for pre/post windows, using full-rate training epochs and training waveform labels only. Two components per class and fixed 10% spatial covariance shrinkage yield ten spatial mixtures. Nine 50-ms means produce 90 projected features, compared with 126 sensor features. Both branches use training-only standardization and equal-prior shrinkage LDA. There was no component-count, channel or parameter search.

Within-participant validation holds out entire waveform comparisons and preserves complete pairs and the original support purge. Leave-one-subject-out (LOSO) evaluation holds out each complete participant. Every xDAWN fit is repeated under every restricted label permutation, including the scaler and classifier. The full experiment performed 120 observed and 11,880 permutation spatial fits. Original sensor null scores were reused only after source/dependency/seed/fold and prediction equivalence checks passed in every evaluation.

## Accuracy and uncertainty

| Representation/window | Mean within-participant balanced accuracy | Pooled LOSO balanced accuracy |
| --- | --- | --- |
| Sensor, post | 19.1% | 18.5% |
| xDAWN, post | 20.7% | 20.0% |
| Sensor, pre | 16.8% | 19.3% |
| xDAWN, pre | 17.8% | 20.2% |

The descriptive participant-bootstrap 95% interval for the mean post-trigger xDAWN score was 17.4–23.9%, including the 20% reference. Its paired improvement over sensor features was **+1.65 percentage points**, with a descriptive interval of **−0.15 to +3.44 points**, including zero. Participants, not epochs or overlapping folds, are the bootstrap units.

For LOSO post-trigger xDAWN, the conditional restricted-permutation p-value was .41; for the pre-trigger control it was .59. Holm correction across the two LOSO xDAWN windows gave .82 for both. There were 99 permutations, so p-value resolution is .01. Participant-specific conditional p-values remain exploratory and uncorrected; their null distributions were not combined into a cohort test.

The pre control does not demonstrate reliable above-chance context decoding in LOSO. It also does not establish absence of preceding-stimulus effects, expectation or post-only artifacts. Position-specific scores and confusion matrices are in the full local report; they are descriptive, not equivalence tests.

## Audit of the learned features

All observed projected training/test arrays were finite. The scaling and within-class covariance audit was repeated separately in each learned fold basis. The actual fitted LDA covariances matched their class-specific automatic-shrinkage reconstructions exactly. All regularized projected covariances were full rank.

| Primary post-trigger diagnostic | Sensor baseline | xDAWN |
| --- | --- | --- |
| Features | 126 | 90 |
| Within-participant empirical deficient folds | 50 / 50 | 20 / 50 |
| Within-participant median regularized condition number | 90.9 | 34.7 |
| LOSO median regularized condition number | 140.2 | 237.4 |
| Largest within-participant held-out absolute z | 16.9 | 15.9 |
| Largest LOSO held-out absolute z | 37.0 | 26.5 |

Fewer columns therefore helped within-participant conditioning but did not uniformly improve the covariance geometry. The pre-trigger xDAWN branch also had a held-out excursion of approximately |z|=65. These are inspection flags; no audit-driven clipping, component rejection or trial deletion was applied.

Ten whole-pair covariance resamples per fold are saved. They hold the learned xDAWN filters and outer training scaler fixed, measuring downstream covariance variability conditional on the representation. They do not quantify uncertainty in the entire supervised spatial-fitting procedure. Raw component magnitudes and identities are not pooled across independently fitted bases.

## Interpretation and reproduction

The outcome supports a modest numerical benefit within participants but not a demonstrated decoding benefit. It does not show that xDAWN can never help, or that waveform information is absent. This was one fixed configuration with minimally processed EEG, a small cohort and strict generalization across comparison contexts/participants. Any further component tuning or feature selection must remain inside training folds.

The restricted null is conditional on paired comparison structure and can be limited by temporal nonstationarity. Near-chance pre performance does not certify artifact-free post responses. Charge/intensity differences, residual electrical contamination and unknown physical trigger-to-current timing remain unresolved; spatial filters cannot establish a cortical origin.

See the [run guide](../experiments/waveform_decoding/e004_xdawn_decoding/README.md). The complete local report is `output/task4_xdawn_20261023/index.html`, with `RESULTS.md`, participant scores, paired intervals, confusion matrices, null scores, exact fold memberships, observed spatial/classifier parameters and repeated feature audits. Input/source/dependency hashes and epoch reconstruction checks are saved. No participant-level derived files are added to Git.

Five new scientific-invariant tests passed; the project suite passed 34 tests during this experiment's validation. Tests cover full-rate projection/bin equivalence, isolation of fitted parameters from held-out values, training-row isolation and spatial refitting for null labels, reconstruction of the original sensor features and protection of the pre window, and complete out-of-fold predictions/correction arithmetic.
