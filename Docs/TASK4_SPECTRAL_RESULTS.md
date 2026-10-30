# Task 4: fixed single-trial alpha/beta power decoding

**23 October 2026.** Broad-band power did not provide a clear overall improvement over the voltage baseline in this fixed experiment. Power alone gave 15.80% mean within-participant post-trigger balanced accuracy and 17.70% leave-one-participant-out (LOSO) accuracy. Adding power to voltage gave 19.09% and 18.12%, respectively. None of the six LOSO representation/window tests demonstrated reliable above-chance decoding. These findings do not establish absence of waveform information in EEG.

## Matched design

All ten current-protocol participants (subjects 3–12) contributed the original baseline's **1,244 events in 622 complete pairs**. Fourteen predefined electrodes, scalp average reference, original eligibility, comparison-held-out within-participant folds, complete-subject LOSO folds and temporal-support purging were unchanged. No temporal filters, ICA, onset interpolation or new exclusions were introduced.

Post-trigger epochs were [50,500) ms with a [−250,−50) ms voltage baseline. Matched pre-trigger epochs were [−500,−50) ms with their own earlier [−750,−550) ms baseline. Epochs remain trigger-aligned; physical trigger-to-current delay is unverified.

Three representations used the same training-only StandardScaler and equal-prior auto-shrinkage LDA:

- **Voltage:** 126 features, comprising nine 50-ms temporal means at 14 electrodes.
- **Power:** 28 features, comprising alpha (8–13 Hz) and beta (13–30 Hz) log-power at each electrode.
- **Combined:** the 154-column concatenation of voltage and power.

Each power estimate used one full 450-ms epoch, constant detrending, a Hann taper and a one-sided density periodogram, without zero-padding or trial averaging. Piecewise-linear PSD integration included interpolated exact band boundaries. Native frequency spacing was 2.22 Hz; interpolation does not improve resolution or sharply separate these bands. Input volts were converted to µV; integrated powers were expressed in µV² and transformed using 10 log10(power / 1 µV²). A fixed 1e−20 µV² numerical floor was never reached. Constant detrending cancels the voltage-baseline offset for power; this is not a pre/post power ratio.

No bands, channels, windows or model settings were selected from the results. Feature extraction was label-independent and local to each epoch. Every restricted permutation refitted the training scaler and classifier.

## Results

Balanced accuracy pools out-of-fold predictions over all five classes within each participant, then weights participants equally. The 95% intervals below are **descriptive participant-bootstrap intervals**, using 10,000 resamples with seed 20261023; folds and epochs were not treated as independent participants.

| Post-trigger features | Within-participant mean BA | Descriptive 95% interval | LOSO BA | LOSO conditional p |
| --- | ---: | --- | ---: | ---: |
| Voltage | 19.05% | 15.78–22.32% | 18.52% | 0.860 |
| Power | 15.80% | 12.64–18.98% | 17.70% | 1.000 |
| Voltage + power | 19.09% | 15.40–23.04% | 18.12% | 0.980 |

The paired within-participant post-trigger difference was **−3.25 percentage points** for power minus voltage (descriptive interval −6.84 to +0.66 points), and **+0.04 points** for combined minus voltage (−2.00 to +2.06 points). These intervals do not provide clear evidence of an improvement or a formal test of model superiority.

| Pre-trigger control | Within-participant mean BA | Descriptive 95% interval | LOSO BA | LOSO conditional p |
| --- | ---: | --- | ---: | ---: |
| Voltage | 16.75% | 14.49–19.00% | 19.35% | 0.820 |
| Power | 19.14% | 17.03–20.97% | 18.48% | 0.960 |
| Voltage + power | 17.21% | 14.24–20.10% | 19.79% | 0.695 |

LOSO post-trigger power accuracy was 16.58% for first presentations and 18.64% for second presentations. Pre-trigger power accuracy was 20.41% and 16.49%, respectively. These position-specific scores are descriptive; a numerical score above 20% is not itself evidence of decoding or a position effect. No separate position-specific significance tests were performed.

There were isolated higher post-trigger participant scores with unadjusted conditional p values at or below 0.05; the lowest raw p was 0.005. These are exploratory observations, not confirmed participant-specific decoding or tests that power improves prediction. No participant, channel or feature was selected from them. Participant-specific tables remain in the ignored local outputs.

## Statistical limits

The 199 restricted permutations (seed 42) shifted complete pairs within each participant/comparison and optionally reversed presentation orientation comparison-wide. The same permutations were shared across representations/windows. This conditional diagnostic null preserves paired structure and comparison-specific waveform availability; temporal nonstationarity can limit exchangeability. Fold accuracies were not treated as independent observations, and participant null distributions were not pooled into a cohort p value.

Holm correction across the six LOSO tests gave **adjusted p=1.000 for every test**. With 199 permutations the raw p resolution is 0.005, allowing the six-test family to reach a minimum adjusted p of 0.030.

A separate correction covers the 60 exploratory participant-specific tests. Its minimum attainable adjusted p is 0.300 at this permutation count, so it cannot establish significance at 0.05. This resolution limit must not be presented as evidence that every participant lacks information. The within-participant means and intervals characterize performance; there is no formal cohort-level within-participant permutation test here. The 20% figure remains a reference, rather than a substitute for the conditional null under these structured validation splits.

## Feature audit and verification

All **360 observed folds** were audited again across representations/windows. Every feature array was finite, with no constant or near-constant columns. Training-only scaler means and the reconstructed actual LDA covariance agreed exactly with their reference fits. All regularized covariance matrices had full numerical rank. Distribution and covariance-bootstrap flags caused no clipping, exclusions or retuning.

Power's smaller representation improved some within-participant numerical properties, without improving decoding. Its empirical within-class covariance had full rank in all 50 post-trigger folds; the 126-column voltage and 154-column combined representations were deficient in all 50. Median post-trigger regularized covariance condition numbers were:

| Evaluation | Voltage | Power | Voltage + power |
| --- | ---: | ---: | ---: |
| Within participant | 90.9 | 22.4 | 80.7 |
| LOSO | 140.2 | 261.5 | 155.0 |

Thus conditioning did not improve uniformly across evaluations. Largest held-out absolute z scores for post-trigger power were 5.3 within participants and 4.5 in LOSO, versus 16.9 and 37.0 for voltage/combined features. These are inspection flags, not measures of neural signal quality. Ten whole-pair covariance resamples per fold were saved, conditional on each outer-training scaler.

Original voltage features, observed predictions/probabilities and fold membership were reproduced. All **2,178 shared original voltage-null scores** (99 permutations × two windows × 11 evaluations) were independently recomputed and matched exactly; another 100 permutations per evaluation/window were newly evaluated. All 13 recorded implementation hashes remained unchanged through the completed run. The project suite passed **47 tests**, including five new spectral tests covering units, band integration, amplitude scaling, epoch independence, numerical-floor handling and recovery of synthetic phase-varying spectral class information. Both generated figures were visually checked.

## Interpretation and reproduction

These features measure **total single-trial power**, which can include evoked transients, ongoing activity and non-neural contamination. They do not isolate induced oscillations; no class-specific ERP was subtracted from held-out trials. Stimulation-waveform shape does not imply a corresponding EEG spectral signature. Pre-trigger activity can contain anticipation and preceding-stimulus responses, especially before second presentations, and is not a guaranteed null. Negative results here concern these two bands, 14 electrodes, 450-ms windows, classifier and validation design.

See the [run guide](../experiments/waveform_decoding/e005_spectral_decoding/README.md) for exact settings and commands. The canonical local output is Git-ignored `output/task4_spectral_complete_20261023/`, containing `RESULTS.md`, `index.html`, figures, aggregate tables, feature audits, provenance and per-evaluation features, folds, predictions, null scores and fitted parameters. Raw and derived participant data remain excluded from Git.
