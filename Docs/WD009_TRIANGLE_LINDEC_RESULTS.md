# WD-009. Triangle versus linear-decrease EEG decoding

[Interactive report](Reports/WD-009/index.html) · [Run guide](../experiments/waveform_decoding/e022_triangle_lindec_decoding/README.md)

**23 November 2026.** The target is the delivered waveform from the verified trigger. Naturalness votes are used only by the inherited trial-validity QC.

501 epochs from 10 participants; 126 direct-pair epochs, including 66 nominal-charge-matched epochs.

## Research question and origin

Collu et al. reported 70% median behavioral match-to-sample discrimination for triangle versus linear decrease at matched charge and peak current. WD-009 tests whether their delivered waveform identities can also be distinguished from EEG. The EEG target comes from triggers. [Collu et al.](https://doi.org/10.1038/s41598-023-28594-0)

## Which epochs are nominal-charge matched?

The matched direct-pair subset contains 66 epochs. The broader all-context matched subset contains 247 epochs from participants 3, 4, 5, 6 and 10. Only these participants have equal Triangle/LinDec nominal charge; the other five have different waveform-specific threshold amplitudes.

Charge matching and comparison context are independent evaluation filters. The default report view uses matched nominal charge across all contexts: Triangle/LinDec epochs paired with each other or with Sine, Square or linear increase. Direct pairs contain only Triangle–LinDec comparisons. Other-pair-only evaluation excludes those direct comparisons and contains 375 epochs, including 181 matched-charge epochs.

For the first pulse, the second stimulus occurs after the analyzed EEG window. Comparison context can still carry block expectations and recording-order effects. For the second pulse, the preceding waveform also changes with comparison context. Both presentation position and whole-block validation remain explicit. Filters summarize existing held-out predictions and do not refit models.

Nominal cathodic charge equals peak amplitude times 0.4 ms times the waveform area factor. Triangle and LinDec both have area factor 0.5. Matching their shape factors does not match their waveform-specific threshold amplitudes. Task 4 uses a race-to-five pair-preference protocol. The ten-per-amplitude repetitions describe Task 3.

## Held-out balanced accuracy

Each participant and presentation position has its own model. Whole comparison groups are held out. Scores average participants equally; the binary balanced-accuracy reference is 50%.

| Evaluation subset | Position | Bin | Model | Pre BA | Post BA | Post minus pre, pp [95% descriptive interval] | Sign-test p | Holm p |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| All comparison contexts | 1 | 50 ms | PCA + shrinkage LDA | 43.51% | 49.12% | +5.61 [-2.38, +13.71] | 0.7539 | 1.0000 |
| All comparison contexts | 1 | 50 ms | PCA + linear SVM | 39.99% | 52.01% | +12.03 [+1.29, +23.52] | 0.7539 | 1.0000 |
| All comparison contexts | 1 | 50 ms | Linear SVM | 40.32% | 50.42% | +10.09 [-2.88, +23.79] | 0.5078 | 1.0000 |
| All comparison contexts | 1 | 100 ms | PCA + shrinkage LDA | 43.71% | 51.73% | +8.01 [+1.43, +15.19] | 0.5078 | 1.0000 |
| All comparison contexts | 1 | 100 ms | PCA + linear SVM | 41.52% | 48.57% | +7.05 [-1.48, +16.02] | 0.3438 | 1.0000 |
| All comparison contexts | 1 | 100 ms | Linear SVM | 38.44% | 49.98% | +11.54 [-0.95, +23.53] | 0.1797 | 1.0000 |
| All comparison contexts | 2 | 50 ms | PCA + shrinkage LDA | 52.78% | 50.38% | -2.41 [-14.21, +9.11] | 0.7539 | 1.0000 |
| All comparison contexts | 2 | 50 ms | PCA + linear SVM | 51.60% | 47.47% | -4.13 [-16.90, +8.25] | 0.5078 | 1.0000 |
| All comparison contexts | 2 | 50 ms | Linear SVM | 50.45% | 45.32% | -5.14 [-19.27, +7.00] | 0.7539 | 1.0000 |
| All comparison contexts | 2 | 100 ms | PCA + shrinkage LDA | 49.49% | 49.84% | +0.35 [-12.46, +12.85] | 0.7539 | 1.0000 |
| All comparison contexts | 2 | 100 ms | PCA + linear SVM | 50.14% | 47.15% | -2.99 [-14.18, +8.75] | 0.7539 | 1.0000 |
| All comparison contexts | 2 | 100 ms | Linear SVM | 50.42% | 42.23% | -8.20 [-22.26, +5.93] | 0.7539 | 1.0000 |
| Held-out direct triangle/LinDec pairs | 1 | 50 ms | PCA + shrinkage LDA | 55.58% | 44.75% | -10.83 [-24.58, +2.50] | 0.4531 | 1.0000 |
| Held-out direct triangle/LinDec pairs | 1 | 50 ms | PCA + linear SVM | 47.08% | 48.58% | +1.50 [-11.25, +13.83] | 0.7266 | 1.0000 |
| Held-out direct triangle/LinDec pairs | 1 | 50 ms | Linear SVM | 54.50% | 56.33% | +1.83 [-13.33, +15.42] | 1.0000 | 1.0000 |
| Held-out direct triangle/LinDec pairs | 1 | 100 ms | PCA + shrinkage LDA | 55.58% | 54.50% | -1.08 [-15.00, +11.17] | 1.0000 | 1.0000 |
| Held-out direct triangle/LinDec pairs | 1 | 100 ms | PCA + linear SVM | 51.25% | 52.83% | +1.58 [-11.67, +15.17] | 1.0000 | 1.0000 |
| Held-out direct triangle/LinDec pairs | 1 | 100 ms | Linear SVM | 55.00% | 58.00% | +3.00 [-14.08, +17.25] | 0.5078 | 1.0000 |
| Held-out direct triangle/LinDec pairs | 2 | 50 ms | PCA + shrinkage LDA | 64.58% | 55.33% | -9.25 [-21.67, +3.08] | 0.4531 | 1.0000 |
| Held-out direct triangle/LinDec pairs | 2 | 50 ms | PCA + linear SVM | 58.33% | 58.00% | -0.33 [-15.84, +14.58] | 1.0000 | 1.0000 |
| Held-out direct triangle/LinDec pairs | 2 | 50 ms | Linear SVM | 56.58% | 54.50% | -2.08 [-15.00, +8.75] | 1.0000 | 1.0000 |
| Held-out direct triangle/LinDec pairs | 2 | 100 ms | PCA + shrinkage LDA | 60.00% | 55.58% | -4.42 [-18.17, +8.00] | 1.0000 | 1.0000 |
| Held-out direct triangle/LinDec pairs | 2 | 100 ms | PCA + linear SVM | 59.17% | 54.25% | -4.92 [-18.33, +8.17] | 0.7266 | 1.0000 |
| Held-out direct triangle/LinDec pairs | 2 | 100 ms | Linear SVM | 53.92% | 50.75% | -3.17 [-14.00, +8.08] | 1.0000 | 1.0000 |
| Nominal-charge-matched participants, all contexts | 1 | 50 ms | PCA + shrinkage LDA | 30.70% | 42.18% | +11.47 [+1.05, +18.51] | 0.3750 | 1.0000 |
| Nominal-charge-matched participants, all contexts | 1 | 50 ms | PCA + linear SVM | 26.96% | 46.59% | +19.63 [+7.97, +29.29] | 0.3750 | 1.0000 |
| Nominal-charge-matched participants, all contexts | 1 | 50 ms | Linear SVM | 34.55% | 43.91% | +9.36 [+0.29, +19.33] | 0.6250 | 1.0000 |
| Nominal-charge-matched participants, all contexts | 1 | 100 ms | PCA + shrinkage LDA | 35.61% | 46.30% | +10.70 [+2.41, +17.81] | 0.3750 | 1.0000 |
| Nominal-charge-matched participants, all contexts | 1 | 100 ms | PCA + linear SVM | 33.11% | 45.42% | +12.31 [+2.11, +22.48] | 0.3750 | 1.0000 |
| Nominal-charge-matched participants, all contexts | 1 | 100 ms | Linear SVM | 31.38% | 43.47% | +12.09 [-2.37, +28.46] | 0.3750 | 1.0000 |
| Nominal-charge-matched participants, all contexts | 2 | 50 ms | PCA + shrinkage LDA | 52.89% | 47.58% | -5.32 [-27.07, +16.83] | 1.0000 | 1.0000 |
| Nominal-charge-matched participants, all contexts | 2 | 50 ms | PCA + linear SVM | 55.21% | 43.65% | -11.56 [-32.54, +11.63] | 1.0000 | 1.0000 |
| Nominal-charge-matched participants, all contexts | 2 | 50 ms | Linear SVM | 62.63% | 47.60% | -15.04 [-38.67, +5.88] | 0.3750 | 1.0000 |
| Nominal-charge-matched participants, all contexts | 2 | 100 ms | PCA + shrinkage LDA | 49.50% | 48.26% | -1.25 [-26.22, +21.58] | 1.0000 | 1.0000 |
| Nominal-charge-matched participants, all contexts | 2 | 100 ms | PCA + linear SVM | 50.62% | 44.95% | -5.67 [-25.82, +16.49] | 1.0000 | 1.0000 |
| Nominal-charge-matched participants, all contexts | 2 | 100 ms | Linear SVM | 61.23% | 42.90% | -18.33 [-38.34, +5.34] | 0.3750 | 1.0000 |
| Held-out nominal-charge-matched direct pairs | 1 | 50 ms | PCA + shrinkage LDA | 50.33% | 36.17% | -14.17 [-40.00, +11.67] | 1.0000 | 1.0000 |
| Held-out nominal-charge-matched direct pairs | 1 | 50 ms | PCA + linear SVM | 35.00% | 41.33% | +6.33 [-18.33, +27.33] | 0.3750 | 1.0000 |
| Held-out nominal-charge-matched direct pairs | 1 | 50 ms | Linear SVM | 44.83% | 41.00% | -3.83 [-28.50, +20.83] | 1.0000 | 1.0000 |
| Held-out nominal-charge-matched direct pairs | 1 | 100 ms | PCA + shrinkage LDA | 50.33% | 53.17% | +2.83 [-25.00, +24.00] | 0.6250 | 1.0000 |
| Held-out nominal-charge-matched direct pairs | 1 | 100 ms | PCA + linear SVM | 45.83% | 52.33% | +6.50 [-17.50, +28.67] | 1.0000 | 1.0000 |
| Held-out nominal-charge-matched direct pairs | 1 | 100 ms | Linear SVM | 45.83% | 41.00% | -4.83 [-33.00, +17.67] | 1.0000 | 1.0000 |
| Held-out nominal-charge-matched direct pairs | 2 | 50 ms | PCA + shrinkage LDA | 66.67% | 61.50% | -5.17 [-21.67, +11.33] | 1.0000 | 1.0000 |
| Held-out nominal-charge-matched direct pairs | 2 | 50 ms | PCA + linear SVM | 64.17% | 64.33% | +0.17 [-18.33, +18.17] | 1.0000 | 1.0000 |
| Held-out nominal-charge-matched direct pairs | 2 | 50 ms | Linear SVM | 70.67% | 63.17% | -7.50 [-30.00, +10.00] | 1.0000 | 1.0000 |
| Held-out nominal-charge-matched direct pairs | 2 | 100 ms | PCA + shrinkage LDA | 64.17% | 65.33% | +1.17 [-15.00, +17.00] | 1.0000 | 1.0000 |
| Held-out nominal-charge-matched direct pairs | 2 | 100 ms | PCA + linear SVM | 65.83% | 62.67% | -3.17 [-15.67, +8.33] | 1.0000 | 1.0000 |
| Held-out nominal-charge-matched direct pairs | 2 | 100 ms | Linear SVM | 68.67% | 61.50% | -7.17 [-23.00, +5.67] | 1.0000 | 1.0000 |
| Nominal-charge-matched other-pair comparisons | 1 | 50 ms | PCA + shrinkage LDA | 27.56% | 43.06% | +15.49 [+3.83, +24.32] | 0.3750 | 1.0000 |
| Nominal-charge-matched other-pair comparisons | 1 | 50 ms | PCA + linear SVM | 25.82% | 44.04% | +18.22 [+5.42, +30.61] | 0.3750 | 1.0000 |
| Nominal-charge-matched other-pair comparisons | 1 | 50 ms | Linear SVM | 35.91% | 40.52% | +4.61 [-9.17, +17.05] | 1.0000 | 1.0000 |
| Nominal-charge-matched other-pair comparisons | 1 | 100 ms | PCA + shrinkage LDA | 34.04% | 45.63% | +11.59 [-0.98, +24.51] | 1.0000 | 1.0000 |
| Nominal-charge-matched other-pair comparisons | 1 | 100 ms | PCA + linear SVM | 33.13% | 43.06% | +9.92 [-0.83, +24.51] | 0.3750 | 1.0000 |
| Nominal-charge-matched other-pair comparisons | 1 | 100 ms | Linear SVM | 29.38% | 39.35% | +9.97 [-6.14, +29.37] | 1.0000 | 1.0000 |
| Nominal-charge-matched other-pair comparisons | 2 | 50 ms | PCA + shrinkage LDA | 52.01% | 48.35% | -3.66 [-25.99, +18.67] | 1.0000 | 1.0000 |
| Nominal-charge-matched other-pair comparisons | 2 | 50 ms | PCA + linear SVM | 58.06% | 41.33% | -16.73 [-38.80, +5.67] | 1.0000 | 1.0000 |
| Nominal-charge-matched other-pair comparisons | 2 | 50 ms | Linear SVM | 63.37% | 45.95% | -17.42 [-45.85, +8.75] | 0.3750 | 1.0000 |
| Nominal-charge-matched other-pair comparisons | 2 | 100 ms | PCA + shrinkage LDA | 47.91% | 48.73% | +0.82 [-25.10, +29.61] | 1.0000 | 1.0000 |
| Nominal-charge-matched other-pair comparisons | 2 | 100 ms | PCA + linear SVM | 50.96% | 43.59% | -7.37 [-26.33, +15.26] | 1.0000 | 1.0000 |
| Nominal-charge-matched other-pair comparisons | 2 | 100 ms | Linear SVM | 62.26% | 41.13% | -21.13 [-46.81, +7.64] | 0.3750 | 1.0000 |
| Other waveform-pair comparisons | 1 | 50 ms | PCA + shrinkage LDA | 41.64% | 50.42% | +8.77 [-0.73, +18.55] | 0.7539 | 1.0000 |
| Other waveform-pair comparisons | 1 | 50 ms | PCA + linear SVM | 38.82% | 51.82% | +13.00 [+0.78, +26.26] | 0.7539 | 1.0000 |
| Other waveform-pair comparisons | 1 | 50 ms | Linear SVM | 38.80% | 47.39% | +8.58 [-7.55, +26.19] | 0.7539 | 1.0000 |
| Other waveform-pair comparisons | 1 | 100 ms | PCA + shrinkage LDA | 41.94% | 51.79% | +9.85 [+0.35, +20.43] | 1.0000 | 1.0000 |
| Other waveform-pair comparisons | 1 | 100 ms | PCA + linear SVM | 41.41% | 48.03% | +6.61 [-2.80, +16.99] | 0.3438 | 1.0000 |
| Other waveform-pair comparisons | 1 | 100 ms | Linear SVM | 35.01% | 45.87% | +10.86 [-3.05, +24.27] | 0.3438 | 1.0000 |
| Other waveform-pair comparisons | 2 | 50 ms | PCA + shrinkage LDA | 50.36% | 49.79% | -0.57 [-12.78, +11.49] | 1.0000 | 1.0000 |
| Other waveform-pair comparisons | 2 | 50 ms | PCA + linear SVM | 52.71% | 45.09% | -7.62 [-21.34, +4.69] | 1.0000 | 1.0000 |
| Other waveform-pair comparisons | 2 | 50 ms | Linear SVM | 49.30% | 43.31% | -6.00 [-23.61, +9.56] | 0.7539 | 1.0000 |
| Other waveform-pair comparisons | 2 | 100 ms | PCA + shrinkage LDA | 47.25% | 49.41% | +2.15 [-11.58, +17.02] | 0.3438 | 1.0000 |
| Other waveform-pair comparisons | 2 | 100 ms | PCA + linear SVM | 50.16% | 45.95% | -4.21 [-15.15, +7.84] | 0.7266 | 1.0000 |
| Other waveform-pair comparisons | 2 | 100 ms | Linear SVM | 49.99% | 41.03% | -8.97 [-26.28, +7.49] | 0.7539 | 1.0000 |

## Models, features and controls

- All 62 scalp electrodes, with 50 or 100 ms voltage means over the same one-second window.
- Post window is 50–1,050 ms. The matched pre window is −1,050 to −50 ms.
- A shared pair baseline is measured −1,250 to −1,050 ms before the first pulse, before either feature period. Processing uses the instantaneous average reference, without temporal filtering or ICA.
- StandardScaler and PCA are fitted on training epochs only. PCA retains 90% of training variance.
- PCA + shrinkage LDA, PCA + linear SVM, and linear SVM without PCA are fixed comparisons. LDA uses equal priors; SVM uses C=1 and balanced class weights.
- Training excludes the entire held-out waveform comparison within a participant and position. Direct triangle/LinDec blocks are tested using models trained on comparisons against other waveforms.
- Nominal charge defines the matched evaluation subset and is excluded from predictors.
- The original complete-pair QC is retained. Additional channels receive finite-value checks and descriptive peak-to-peak flags. All-channel artifact rejection and interpolation are outside this run.

## Participant-level post/pre significance

The exact two-sided sign test counts participants with higher post than pre BA, excluding ties. It tests equal probabilities of improvement and decline across independent participants, not the size of the change or individual above-chance decoding. Holm correction covers all 72 configurations/subsets/positions. With five non-tied participants the smallest two-sided p is 0.0625. [Exact binomial test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html) · [Holm correction](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html).

## Interpretation

- Earlier work decoded five classes or one waveform versus the remaining four. This run trains dedicated triangle/LinDec classifiers and evaluates unseen comparison contexts.
- The direct and matched subset rows reuse held-out predictions and do not refit within their 5–7 trial blocks.
- First-position results are the primary view. Second-position pre and post EEG can reflect the preceding pulse. The [WD-008 behavioral order audit](https://htmlpreview.github.io/?https://github.com/neurocontrol-lab/tens-eeg-waveform-decoding/blob/main/Docs/Reports/WD-008/index.html#behavioral-order-audit) establishes an association under its assumptions, without identifying physiological carryover.
- Naturalness labels and behavioral order effects do not determine the waveform-identity targets.
- Participant intervals condition on fixed held-out predictions. Corrected sign tests compare post with pre. No refitted label-permutation test of above-chance decoding is included.
- Physical trigger/current delay and behavioral-response timing remain unverified.
