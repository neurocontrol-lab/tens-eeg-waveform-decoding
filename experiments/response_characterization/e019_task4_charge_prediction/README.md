# RC-007. Predict nominal Task 4 charge from EEG

The target is nominal cathodic charge in µC. It uses the WD-007 participant-by-waveform amplitude lookup and nominal waveform area factors. The regressors use the original WD-001 EEG voltage features. Pre- and post-stimulus windows are fitted independently.

```powershell
& ./.venv/Scripts/python.exe -m experiments.response_characterization.e019_task4_charge_prediction --output output/rc007_task4_charge_prediction_20261113_verified
```

Run from the repository root with the WD-001 outputs available and choose a fresh output path for each rerun. `predictions.csv` contains both sets of held-out predictions and their training-fold mean references. `participant_scores.csv` and `summary.json` contain errors, bootstrap intervals, charge coverage and input hashes. [Methods and results](../../../Docs/RC007_TASK4_CHARGE_PREDICTION.md)
