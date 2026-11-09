# E017. Task 3 EEG response and nominal charge

E017 fits linear EEG response functions within participants and checks whether charge improves prediction with either shared or waveform-specific slopes. The [results](../../../Docs/E017_TASK3_CHARGE_RESULTS.md) cover the fixed feature, full-montage and voltage-history analyses.

Run from the repository root with fresh output directories.

```powershell
& ./.venv/Scripts/python.exe -m experiments.response_characterization.e017_eeg_charge_relation.extract --output output/e017_task3_charge_20261109/extraction
& ./.venv/Scripts/python.exe -m experiments.response_characterization.e017_eeg_charge_relation --extraction output/e017_task3_charge_20261109/extraction --output output/e017_task3_charge_20261109/fit
```

The extractor retains the recorded timestamps and uses the Task 3 trigger codebook. Each standard participant and waveform cell must match its complete CSV trigger sequence. The extraction audit records excluded cells and source digests. The model runner reads the compact feature cache and can be rerun without decoding the XDF files again.

The fitted outcome is mean C4/CP4 voltage in a fixed window. The models use absolute nominal cathodic charge in µC. Lower tiers determine the coefficients used for prediction at the upper tier. Separate full-range fits describe the slopes across all three tiers. The pre-stimulus control uses the same eligible trials and model comparisons.

`fit/summary.json` contains participant bootstrap intervals. `fit/coefficients.csv` distinguishes calibration and all-tier estimates, and `fit/upper_tier_predictions.csv` holds each prediction. The `fit/charge_relation.png` figure summarizes prediction error and descriptive slopes.

## Full-montage prediction at recorded tiers

Use the alignment audit from the first command to extract per-electrode 50 ms mean voltages. The runner predicts held-out trials in five chronological folds with all three amplitude tiers in each training fold. It scores nine post-stimulus bins and nine matched pre-stimulus bins.

```powershell
& ./.venv/Scripts/python.exe -m experiments.response_characterization.e017_eeg_charge_relation.full_montage --extraction output/e017_task3_charge_20261109/extraction --output output/e017_task3_charge_full_montage_20261109
```

`fit/summary.json` reports trial MAE and relative MSE changes. `fit/fold_assignments.csv` and `fit/held_out_predictions.npz` identify the scored trials and preserve each observed and predicted feature. `fit/shared_charge_slopes.png` shows descriptive slopes fitted across all eligible trials.

## Forecasting with voltage history

Run after the full-montage fit to reuse its fold assignments. The forecast re-extracts the same voltage bins before trial-baseline subtraction. It tests the addition of nominal charge to three preceding 50 ms means and evaluates a separate charge-only calibration with waveform intercepts.

```powershell
& ./.venv/Scripts/python.exe -m experiments.response_characterization.e017_eeg_charge_relation.forecast --extraction output/e017_task3_charge_20261109/extraction --fold-assignments output/e017_task3_charge_full_montage_20261109/fit/fold_assignments.csv --output output/e017_task3_charge_forecast_20261109
```

To reuse the extracted voltage bins with a fresh fit destination, pass `--cache output/e017_task3_charge_forecast_20261109/extraction` in place of `--extraction`. The fit saves aggregate and participant-level scores, held-out predictions, and descriptive shared charge slopes fitted to all eligible trials.

Use `--include-200ms` to compare all three target widths at starts where a full 200 ms target fits inside the cache.

```powershell
& ./.venv/Scripts/python.exe -m experiments.response_characterization.e017_eeg_charge_relation.forecast --cache output/e017_task3_charge_forecast_20261109/extraction --fold-assignments output/e017_task3_charge_full_montage_20261109/fit/fold_assignments.csv --output output/e017_task3_pooled_forecast_20261109 --include-200ms
```
