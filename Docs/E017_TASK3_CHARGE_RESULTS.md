# E017. Task 3 EEG response and nominal charge

[Research note on pooled and waveform-specific forecasts](E017_TASK3_POOLED_FORECAST.pdf)

**9 November 2026.** The charge-only regressions gave higher held-out error than the waveform means. With recent EEG voltage in the model, nominal charge reduced post-stimulus squared error by 1.40% for the primary 100 ms target, with a participant bootstrap interval of −0.33% to +3.49%.

## Feature and nominal charge

The primary feature was mean C4/CP4 voltage at 150–250 ms after the trigger, using a −250 to −50 ms baseline. The matched pre-stimulus feature used −250 to −150 ms and a −650 to −450 ms baseline. Both used the scalp average reference and unfiltered samples. The fixed QC screened both feature periods for recording boundaries, timestamp discontinuities, neighboring stable triggers, nonfinite values and ROI excursions above 500 µV peak to peak.

Nominal cathodic charge equals peak current in mA multiplied by the documented 0.4 ms phase duration and the ideal waveform area factor. The factor is 1 for square, 2/π for half-sine, and 0.5 for triangle and either ramp. The resulting charge is in µC. These estimates use the amplitude settings in the study CSV. The delivered current traces are pending. [Collu et al. waveform definitions](https://pmc.ncbi.nlm.nih.gov/articles/PMC9884304/) · [Project protocol](../PROJECT_REVIEW.md)

## Trial alignment and coverage

Task 3 has 1,671 CSV trials and 1,651 recognized XDF triggers across 11 participants and 12 recordings. The primary analysis required exact trigger-sequence agreement within each standard participant and waveform cell.

- Participant 2 has 30 events per waveform, but the recording uses the same trigger for all three amplitude tiers of each waveform. All five cells were excluded from the exact-tier join.
- Participant 9 has an extra LinDec event. The LinDec cell was excluded under the exact sequence rule.
- Participant 6 LinInc and participant 12 Sine contain four recorded CSV amplitudes. These cells were excluded from the standard-tier fit.
- The remaining 47 cells contain 1,410 aligned trials. Five trials failed the EEG QC threshold. The fit uses 1,405 trials from 10 participants, including 468 upper-tier trials. Each cell had at least three eligible trials at every tier.

The primary analysis includes trigger-confirmed trials irrespective of sensation ratings. Excluding all-zero or missing sensation reports was a sensitivity check. It retained 1,342 trials in 46 cells.

The [reviewed alignment exceptions](../experiments/response_characterization/e017_eeg_charge_relation/README.md#reviewed-task-3-alignment-exceptions) provide a separate corrected extraction for the three participant–waveform cells. The results on this page describe the original strict extraction.

## Held-out prediction

For each participant, the three models used threshold and threshold +0.25 mA for calibration and predicted threshold +0.50 mA. All models used separate waveform intercepts. The constant model used a zero slope, the shared model used one slope across waveforms for that participant, and the separate model estimated one slope for each waveform. Fits used absolute charge. Within-waveform demeaning in the least-squares calculation removes the intercept from the slope calculation. Reported intercepts and predictions use the original charge scale.

| Model | Post-stimulus mean absolute error (µV) | Matched pre-stimulus mean absolute error (µV) |
| --- | ---: | ---: |
| Constant | 1.36 | 1.73 |
| Shared slope within participant | 1.71 | 1.98 |
| Separate waveform slopes | 3.21 | 3.15 |

Error is the absolute difference between the predicted and observed upper-tier mean, averaged equally across waveforms within participant and then across participants. The 95% intervals below use 10,000 participant bootstrap draws.

- Post-stimulus error was **0.35 µV higher** with a shared slope than with a constant response (95% interval −0.07 to +0.82 µV).
- Post-stimulus error was **1.50 µV lower** with a shared slope than with separate waveform slopes (0.91 to 2.10 µV lower).
- The post-minus-pre improvement from adding the shared slope was **−0.10 µV** (−0.69 to +0.44 µV).
- In the reported-sensation sensitivity check, post-stimulus errors were 1.38, 1.69 and 3.17 µV for the constant, shared and separate models respectively.

## Descriptive slopes across all tiers

The mean participant shared slope was +3.20 µV/µC post-stimulus and +3.88 µV/µC pre-stimulus. Their paired difference was **−0.69 µV/µC** (95% interval −4.50 to +3.18). These slopes use all three tiers and are separate from the held-out prediction fits.

LinDec had the largest descriptive post-stimulus slope, +19.66 µV/µC across nine participants. Its post-minus-pre contrast was +25.18 µV/µC, with an exploratory, unadjusted interval of +4.09 to +50.41. Its separate-slope upper-tier prediction error was 4.61 µV, compared with 1.41 µV for a constant response. The descriptive contrast and the extrapolation result answer different questions.

This test covers one late voltage feature and an extrapolation from the lower two tiers to the upper tier. Within waveform, current and nominal charge vary together. Application to Task 4 would require evidence that the relationship transfers between tasks.

## Full montage with all three amplitude tiers in training

This extension predicted mean signed voltage separately for each of 62 scalp electrodes and nine consecutive 50 ms bins from 50 to 500 ms after the trigger. The matched pre-stimulus control used nine bins from −500 to −50 ms. Both windows used unfiltered, scalp-average-referenced EEG. The post-stimulus baseline was −250 to −50 ms and the control baseline was −750 to −550 ms. The all-electrode QC required continuous finite epochs and peak-to-peak voltage at most 500 µV at every scalp electrode in the scored periods and their baseline support. This left 1,382 trials from 47 participant–waveform cells across 10 participants.

For each participant and waveform, the eligible trials were divided into five chronological chunks. Each chunk was held out once. Every training fold contained at least three trials at each amplitude tier per included waveform. The regressions predicted voltage at each electrode and time bin independently. Each model used a waveform intercept within each participant. The shared model estimated one charge slope across waveforms for a given participant, electrode and bin. The separate model estimated a slope for each waveform. This evaluates new trials from the same participants and recorded amplitude tiers.

| Model | Post MAE (µV) | Pre MAE (µV) | Post change in MSE vs constant |
| --- | ---: | ---: | ---: |
| Constant | 4.58 | 4.52 | Reference |
| Shared slope within participant | 4.60 | 4.57 | +0.29% |
| Separate waveform slopes | 4.74 | 4.76 | +4.14% |

Mean absolute error (MAE) averages held-out trial errors across electrodes and bins, then gives each waveform–tier cell equal weight within each participant and each participant equal weight. The relative mean squared error (MSE) change is calculated within participants and then averaged. The post-stimulus MSE change had a 95% participant bootstrap interval of −0.15% to +0.77% for the shared slope and +3.22% to +5.03% for separate slopes. In the matched pre-stimulus control, the MSE increases were 0.99% and 5.59%, respectively.

| Amplitude tier | Post MAE, constant (µV) | Post MAE, shared slope (µV) | Post MAE, separate slopes (µV) |
| --- | ---: | ---: | ---: |
| Threshold | 4.72 | 4.72 | 4.98 |
| Threshold +0.25 mA | 4.47 | 4.47 | 4.47 |
| Threshold +0.50 mA | 4.56 | 4.63 | 4.76 |

Nominal charge added little to single-trial prediction across the montage. The shared-slope fit was close to the constant baseline, and fitting separate slopes increased held-out error. These are aggregate scores over a fixed window and montage.

## Forecasting from recent EEG and nominal charge

At each of the 62 scalp electrodes, three consecutive 50 ms voltage means predicted a 100 ms target mean at the same electrode. The post-stimulus targets began at 200, 300 and 400 ms, and the matched pre-stimulus targets began at −350, −250 and −150 ms. Histories end at the target start. The sensitivity analysis shortened the targets to 50 ms with the same starts and histories. Both analyses used scalp-average-referenced voltages before any trial-baseline subtraction.

The forecasts used participant-specific intercepts and no waveform label as a model input. A separate calibration comparison used waveform-specific intercepts with and without a shared charge slope, and omitted the voltage history. Both used the same five chronological held-out trial folds as the full-montage run, with all three amplitude tiers represented in each training fold. Scores weight waveform–tier cells equally within participants and participants equally across the cohort.

| Model for the 100 ms target | Post MAE (µV) | Post held-out R² | Pre MAE (µV) | Pre held-out R² |
| --- | ---: | ---: | ---: | ---: |
| Voltage history | 2.629 | 0.99915 | 2.174 | 0.99955 |
| Voltage history + nominal charge | 2.606 | 0.99920 | 2.195 | 0.99955 |
| Waveform mean | 58.923 | 0.83685 | 59.219 | 0.83611 |
| Waveform intercepts + shared charge slope | 59.454 | 0.83363 | 59.755 | 0.83288 |

Held-out R² uses the trial mean at each electrode and target window as the reference, then pools squared errors and reference variance within participant. Carrying the last 50 ms mean forward already yielded post-stimulus R² of 0.99895 and MAE of 2.766 µV. The high absolute R² mainly reflects the persistence of voltage levels over these adjacent windows. When scored against the mean change from the last history bin, the history model's R² was 0.0378.

| Target width | Post incremental R² from adding charge to history | Pre incremental R² | Paired post-minus-pre difference |
| --- | ---: | ---: | ---: |
| 100 ms | +1.40% (−0.33% to +3.49%) | −1.00% (−1.42% to −0.63%) | +2.40 points (+0.76 to +4.28) |
| 50 ms | +1.50% (−0.31% to +3.73%) | −1.16% (−1.37% to −0.93%) | +2.66 points (+0.80 to +4.93) |

Incremental R² is the relative reduction in mean squared error from adding charge, averaged across participants. Intervals use 10,000 participant bootstrap draws. The post-minus-pre contrast was positive, while the post-stimulus improvement alone was small and uncertain. In the separate 100 ms calibration comparison, adding the shared charge slope increased squared error by 1.60% both post-stimulus and pre-stimulus. The forecast comparison tests additional information beyond voltage history. The separate calibration comparison is the one that estimates a charge slope for possible Task 4 adjustment.

### Matched 200 ms target comparison

All three widths used the same trials, electrodes, histories and target starts of 200 and 300 ms post-trigger and −350 and −250 ms pre-trigger. Within each participant, the voltage-history regressions pooled eligible trials across waveforms and predicted the voltage mean of each held-out trial. The 200 ms post-stimulus targets were [200, 400) and [300, 500) ms. The 400 ms start from the earlier run was dropped for every width because a 200 ms target would extend beyond the cache.

| Target width | History MAE (µV), post | History + charge MAE (µV), post | Post incremental R² from adding charge | Pre incremental R² |
| --- | ---: | ---: | ---: | ---: |
| 50 ms | 2.198 | 2.195 | +0.21% (−1.03% to +1.62%) | −0.97% |
| 100 ms | 2.297 | 2.295 | +0.29% (−0.92% to +1.79%) | −0.78% |
| 200 ms | 2.906 | 2.872 | +0.62% (−1.04% to +2.81%) | −0.98% |

For the 200 ms targets, the paired post-minus-pre difference was +1.60 percentage points (−0.03 to +3.66). The charge calibration without voltage history increased post-stimulus squared error by 1.61%. A one-second Task 4 decoder would require Task 3 calibration slopes fitted to the exact electrode and window features used by the decoder, as well as independently measured charge for each Task 4 trial.

### Separate forecasts for each waveform

The matched comparison also fitted the intercept, history weights and charge slope separately for each known waveform within a participant. It held out the same trials as the pooled forecast. The pooled predictions were reproduced exactly.

| Target width | Post MAE, separate history (µV) | Post MAE, separate history + charge (µV) | Post error reduction from adding charge (95% interval) |
| --- | ---: | ---: | ---: |
| 50 ms | 2.454 | 2.515 | −3.50% (−6.14% to −0.93%) |
| 100 ms | 2.613 | 2.699 | −4.25% (−7.03% to −1.66%) |
| 200 ms | 3.293 | 3.389 | −3.60% (−5.15% to −2.00%) |

Pooling gave lower held-out MAE at all three widths. In the 200 ms calibration without EEG history, the waveform mean gave a post-stimulus MAE of 58.85 µV, compared with 59.38 µV for the shared charge slope and 60.20 µV for separate slopes. The PDF includes the per-waveform forecast results and the proposed Task 4 adjustment.

## Reproduction

The original trial-alignment audit, predictions, coefficients and figure are under `output/e017_task3_charge_20261109`. The full-montage run is under `output/e017_task3_charge_full_montage_20261109`. The forecast cache, predictions, participant scores and descriptive calibration slopes are under `output/e017_task3_charge_forecast_20261109`. The repeated pooled matched-width forecast is under `output/e017_task3_pooled_forecast_20261109`. The comparison with waveform-specific models is under `output/e017_task3_forecast_by_waveform_20261109`. [E017 run guide](../experiments/response_characterization/e017_eeg_charge_relation/README.md)
