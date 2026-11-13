# WD-007. Task 4 waveform decoding after nominal-charge adjustment

**13 November 2026.** The pooled adjustment raised balanced accuracy in both windows. The pre-stimulus increase was larger than the post-stimulus increase. Waveform-specific adjustment gave near-perfect decoding in both windows, and its correction vector alone identified all five waveform classes in the matched cohort.

[One-page research note](WD007_TASK4_CHARGE_ADJUSTMENT.pdf) · [Interactive report with class recall and confusion matrices](Reports/WD-007/index.html)

## Matched comparisons

The run reproduced all saved post- and pre-stimulus WD-001 predictions on the original comparison-held-out folds. It used 14 electrodes and nine 50 ms means from 50 to 500 ms after stimulation and −500 to −50 ms before stimulation. Both adjustments subtract the Task 3 charge-only slope times nominal cathodic charge. The same post-stimulus correction vector is applied to the matched pre-stimulus features. All EEG variants use the original shrinkage LDA and training-fold scaling.

| Cohort | EEG window | Original | Pooled adjustment | Waveform-specific adjustment |
| --- | --- | ---: | ---: | ---: |
| All 10 participants | Post | 19.05% | 26.35% | — |
| All 10 participants | Pre | 16.75% | 26.12% | — |
| Seven with complete calibration | Post | 20.73% | 28.09% | 99.87% |
| Seven with complete calibration | Pre | 16.73% | 26.62% | 99.77% |

The all-participant pooled gains were **+7.30 percentage points** post-stimulus (95% paired participant bootstrap interval +4.47 to +10.46) and **+9.37 points** pre-stimulus (+5.39 to +13.65). The all-participant cohort has 1,244 trials and the matched cohort has 854. Participants 6, 9 and 12 lack a waveform-specific Task 3 slope for linear increase, linear decrease and sine, respectively. The available waveform-specific Task 3 charge ranges covered all 1,170 corresponding Task 4 estimates. The remaining 74 trials had no waveform-specific range.

## Charge and correction controls

Each participant has one nominal charge scalar and one correction template per waveform. This gives zero within-class covariance and a degenerate shrinkage LDA fit. The report shows those LDA scores and a nearest-centroid diagnostic on the same folds. For equal templates the diagnostic resolves ties in waveform-code order. With balanced accuracy, each tied group contributes the recall of one recoverable class.

| Input without EEG | All 10 participants, nearest centroid | Matched seven, nearest centroid |
| --- | ---: | ---: |
| Nominal charge | 76.00% | 82.86% |
| Pooled correction vector | 76.00% | 82.86% |
| Waveform-specific correction vector | — | 100.00% |

There were 15 equal-charge participant–waveform pairs across six waveform combinations. Triangle and linear decrease shared a charge in five participants. The pooled correction preserves these overlaps. The waveform-specific slope can distinguish equal-charge waveforms through the slope template.

## Interpretation

The Task 4 CSV gives one amplitude per pair. The field is constant within every participant and second waveform cell. The analysis uses this association as an amplitude lookup for both presentations and computes nominal cathodic charge using the waveform area factor and 0.4 ms phase width. The lookup and slope selection both depend on the known waveform in the waveform-specific condition. Aligned current-monitor traces would allow a fresh pooled-slope test with charge estimated independently of the classification label.

The pre-stimulus gains and correction-only scores show that these label-informed adjustments can introduce waveform information. The higher post-stimulus scores therefore measure sensitivity to the adjustment. [Run guide](../experiments/waveform_decoding/e018_nominal_charge_adjustment/README.md)
