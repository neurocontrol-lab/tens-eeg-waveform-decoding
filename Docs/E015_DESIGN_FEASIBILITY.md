# E015 - Position-specific decoding implementation and design feasibility

**Date:** 2 November 2026. **Status:** Pre-fitting design audit; observed fitting subsequently completed. See the [observed results](E015_OBSERVED_RESULTS.md).

The requested analysis has one five-class classifier per participant and
presentation position, using 100 training-selected scalar features and shrinkage
LDA. The complete intended pool is 25,364 features: mean voltage, signed/RMS slope,
signed/RMS curvature, alpha/beta power and independently calibrated xDAWN.
This note records a design audit, not decoding accuracy or evidence of absence
of waveform information.

## Work completed

- Implemented label-free extraction and built the cache for all 1,190 epochs from
  the preserved 595 pairs, with 24,614 sensor/derivative/power values per epoch.
- Implemented a fixed, fold-calibrated xDAWN basis adding 750 scalar features.
- Implemented one-position comparison/time-adjusted waveform contrasts, paired
  ranking splits, positive-correlation selection, training-only scaling and LDA,
  plus a matched ANOVA-selected LDA reference.
- Implemented an observed-stage runner that preflights complete participant/
  position models and never reports accuracy from only their successful folds.
- Tested that changing held-out values cannot change calibration inputs,
  selected columns, training scalers or classifier coefficients.

This audit preceded real-data fitting. The subsequent strict run fitted 11 supported models, and the unreliable exploratory companion ran all 20 participant/position cases. The boundary tests also use synthetic inputs and mock spatial fits to check that held-out signals cannot influence training.

## Validation and temporal support

The audit uses leave-one-comparison-out validation, with identical pair folds
for both positions. This retains more training data than five-fold comparison
validation for independent calibration and two ranking halves. Complete pairs
stay together. A label-independent impulse-response check of the existing
0.5-30-Hz forward/backward filter finds an effective seven-second one-sided
support at an L1-tail tolerance of 1e-4 (measured fraction 7.9005e-5).

Baseline and derivative support span [-250,1062) ms. Expanding both training and
test dependencies gives a minimum trigger separation of **15.312 seconds**;
nearby training pairs are purged. This is an effective-support tolerance, not a
claim of an exactly finite IIR impulse response. It does not remove within-pair
physiological carryover or establish artifact-free EEG.

## Full metadata audit

The audit checks 100 A/B ranking allocations per supported fold. xDAWN
calibration is independent of both halves and prefers two epochs per class,
falling back to one on metadata feasibility. All five classes must appear in
calibration and both ranking halves. Each half's comparison/time-adjusted
waveform model must be full rank with residual observations.

**11 of 20 participant/position models pass every fold. 178 of 200 individual
folds pass.** A failed bounded allocation search is not proof that no alternative
allocation exists, but missing classes and rank-deficient outer designs are
explicit hard failures under the specified model.

| Participant | Position 1 supported folds | Position 2 supported folds |
| --- | ---: | ---: |
| 3 | 6/10 | 7/10 |
| 4 | 9/10 | 8/10 |
| 5 | 10/10 | 10/10 |
| 6 | 10/10 | 10/10 |
| 7 | 10/10 | 10/10 |
| 8 | 10/10 | 10/10 |
| 9 | 10/10 | 10/10 |
| 10 | 7/10 | 8/10 |
| 11 | 10/10 | 9/10 |
| 12 | 7/10 | 7/10 |

Participant 12 has only **three first-position Square epochs in the entire
eligible cohort**. Holding even one out leaves at most two; three disjoint
training roles (calibration, A and B), each containing Square, cannot be formed.
Other failures involve scarce classes after purging, unidentifiable adjusted
contrasts, or unsuccessful constrained calibration/A/B allocations.

Consequently, the intended 20-model evaluation cannot be executed unchanged
under this strict design. The user requested that unsupported cases also be run and explicitly labelled unreliable. A strict run and a separate, consistently defined exploratory run were therefore produced. The latter fits xDAWN on outer-training data and uses observed comparison-balanced class profiles instead of unidentified regression-adjusted contrasts. Removing xDAWN calibration alone does not repair folds with missing
classes or unidentifiable comparison-adjusted contrasts; those checks remain.
No unsupported fold will be repaired by silently dropping a class or reporting
a minimum-norm coefficient as an identified waveform effect.

## Local artifacts

- `output/e015_feature_cache_20261102/`: label-free features, scalar contract,
  event metadata, source/input digests and completed-cache status.
- `output/e015_feasibility_full_20261102/`: filter audit, fold/model support CSVs,
  and metadata-audit status. No classifier accuracy is present.

All numerical files remain Git-ignored. See the
[run guide](../experiments/waveform_decoding/e015_position_repeatability_decoding/README.md)
and [central experiment registry](../experiments/README.md).
