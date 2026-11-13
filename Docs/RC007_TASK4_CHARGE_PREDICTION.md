# RC-007. Predict nominal Task 4 charge from EEG

[Research report with separate Task 3 and Task 4 results](EEG_NOMINAL_CHARGE_PREDICTION.pdf)

**13 November 2026.** Ridge regression using EEG voltage gave slightly higher mean absolute error than the training-fold mean charge in both validation designs. Post-stimulus predictions were slightly closer than pre-stimulus predictions, with paired participant intervals spanning zero.

## Model and validation

The analysis used 1,244 Task 4 trials from 10 participants. Nominal cathodic charge is the waveform-specific amplitude from the WD-007 lookup times 0.4 ms and the waveform area factor. Each participant has a fixed nominal charge per waveform. The target uses the programmed stimulus and estimates delivered charge indirectly.

Each regression used 126 voltage features from the original WD-001 baseline, with nine 50 ms means at 14 electrodes. The post-stimulus window spans 50 to 500 ms and the matched pre-stimulus window spans −500 to −50 ms. Both windows use the original scalp average reference and separate pre-trigger baselines. Each participant and window has its own regression. The model uses voltage features as its only predictors.

- **Comparison holdout.** The same five outer folds as WD-001 hold out complete waveform comparisons and keep pairs together. The run verified every test fold against the saved baseline predictions.
- **Waveform holdout.** Each fold excludes all comparisons involving one waveform from training and scores that waveform's trials. This excludes the paired presentations from training. Every trial is scored once, and the same nominal charge can still occur in another waveform in training.
- **Model selection.** Inner comparison-grouped validation selects ridge alpha from 1, 10, 100, 1,000 and 10,000 by mean squared error. Scaling is fitted in each training split. Both validation designs purge potentially overlapping EEG support using the WD-001 1.25-second rule.

## Held-out results

Scores average participants equally. The last column compares squared error with the prediction that always returns the corresponding training-fold mean charge. The conventional held-out R² uses the observed within-participant test variance as its denominator.

| Validation | Window | EEG MAE (µC) | Training-mean MAE (µC) | Held-out R² | R² vs. training-fold mean |
| --- | --- | ---: | ---: | ---: | ---: |
| Comparison holdout | Pre | 0.04885 | 0.04834 | −0.194 | −0.075 |
| Comparison holdout | Post | 0.04858 | 0.04834 | −0.125 | −0.017 |
| Waveform holdout | Pre | 0.05975 | 0.05812 | −0.674 | −0.047 |
| Waveform holdout | Post | 0.05900 | 0.05812 | −0.627 | −0.018 |

Post minus pre MAE was −0.00027 µC in comparison holdout (95% paired participant bootstrap interval −0.00098 to +0.00040) and −0.00075 µC in waveform holdout (−0.00219 to +0.00039). Relative to the training-fold mean, the 95% interval for post-stimulus R² was −0.051 to +0.012 in comparison holdout and −0.031 to −0.005 in waveform holdout.

All comparison-held-out trials had a target charge already seen in training. The waveform holdout covered the following three cases. Each row weights participants with eligible trials equally.

| Held-out waveform charge | Trials | Participants | Pre EEG MAE (µC) | Post EEG MAE (µC) | Training-mean MAE (µC) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Seen in a training waveform | 543 | 8 | 0.03335 | 0.03413 | 0.03286 |
| Unseen, inside the training range | 315 | 8 | 0.04509 | 0.04228 | 0.04103 |
| Outside the training range | 386 | 10 | 0.10528 | 0.10422 | 0.10483 |

The largest subgroup errors occurred on charges outside the training range. The fixed charge per waveform limits what Task 4 can establish about dose sensitivity within a waveform. [Run guide](../experiments/response_characterization/e019_task4_charge_prediction/README.md)
