# RC-008. Predict nominal Task 3 charge from EEG

[Research report with separate Task 3 and Task 4 results](EEG_NOMINAL_CHARGE_PREDICTION.pdf)

**13 November 2026.** Within the same waveform, EEG voltage gave higher held-out error than the training-fold mean charge. The held-out-waveform test gave a small post-stimulus improvement with an interval spanning zero. The strict and reviewed alignments gave the same overall interpretation.

## Inputs and validation

Ridge regression used the same 14 electrodes and nine 50 ms voltage means as RC-007, with matched pre-stimulus (−500 to −50 ms) and post-stimulus (50 to 500 ms) windows. Each model used voltage features only. The target was the nominal cathodic charge derived from the recorded amplitude, 0.4 ms phase width and waveform area factor. The Task 3 tiers vary amplitude within each waveform.

- **Within waveform.** Fit separately for each participant and waveform. Hold out one of the five RC-006 chronological chunks at a time. All three tiers were present in every outer training fold.
- **Held-out waveform.** Fit within each participant on the other waveforms and predict charge for the entirely held-out waveform. This allows repeated charge values across waveform types.
- **Model selection.** Select ridge alpha from 1, 10, 100, 1,000 and 10,000 by inner held-out MSE. The inner validation holds out chronological chunks in the within-waveform analysis and entire waveforms in the transfer analysis. Scaling is fitted inside each training split. Each comparison uses the same trials and folds for pre- and post-stimulus EEG.

The primary run uses the reviewed alignment and includes 1,469 trials across all 50 participant–waveform groups. The original strict alignment includes 1,382 trials in 47 groups. Both include the same ten participants. [Alignment exceptions](../experiments/response_characterization/e017_eeg_charge_relation/README.md#reviewed-task-3-alignment-exceptions)

## Held-out results

The mean charge from each training fold is the reference. Errors weight waveforms equally within participant and then weight participants equally. The reported R² is the mean of participant-level relative squared-error reductions versus that reference. Intervals use paired participant bootstrap draws.

| Alignment | Validation | Training mean MAE (µC) | Pre EEG MAE (µC) | Post EEG MAE (µC) | Post R² vs. training mean (95% interval) |
| --- | --- | ---: | ---: | ---: | ---: |
| Reviewed | Within waveform | 0.04527 | 0.04718 | 0.04642 | −0.032 [−0.084, +0.018] |
| Reviewed | Held-out waveform | 0.08176 | 0.08226 | 0.08035 | +0.022 [−0.026, +0.087] |
| Strict | Within waveform | 0.04583 | 0.04793 | 0.04694 | −0.031 [−0.082, +0.018] |
| Strict | Held-out waveform | 0.08332 | 0.08385 | 0.08250 | +0.008 [−0.040, +0.075] |

In the reviewed run, post minus pre MAE was −0.00076 µC within waveform (95% interval −0.00185 to +0.00034) and −0.00191 µC for held-out waveforms (−0.00448 to +0.00007).

In the reviewed waveform holdout, 920 trials had a charge seen in training, 331 had a new charge within the training range, and 218 fell outside that range. For the 331 new in-range trials, post-stimulus MAE was 0.06685 µC versus 0.06667 µC for the training mean, weighting the available waveform groups equally within each participant. [Run guide](../experiments/response_characterization/e020_task3_charge_prediction/README.md)
