# RC-008. Predict nominal Task 3 charge from EEG

Fit EEG-only ridge regressions using the 14 WD-001 channels and nine 50 ms voltage means from the RC-006 full-montage cache. Run from the repository root with the RC-006 strict cache and reviewed trial join available. [Reviewed alignment procedure](../e017_eeg_charge_relation/README.md#reviewed-task-3-alignment-exceptions)

```powershell
& ./.venv/Scripts/python.exe -m experiments.response_characterization.e017_eeg_charge_relation.full_montage --extraction output/e017_task3_charge_reviewed_20261113/extraction --output output/rc008_task3_reviewed_montage_20261113
& ./.venv/Scripts/python.exe -m experiments.response_characterization.e020_task3_charge_prediction --montage output/rc008_task3_reviewed_montage_20261113 --output output/rc008_task3_charge_prediction_reviewed_20261113
& ./.venv/Scripts/python.exe -m experiments.response_characterization.e020_task3_charge_prediction --montage output/e017_task3_charge_full_montage_20261109 --output output/rc008_task3_charge_prediction_strict_20261113
```

Use fresh output directories for reruns. Each regression run saves held-out trial predictions, participant and waveform scores, and a summary with charge coverage, bootstrap intervals and input hashes. [Methods and results](../../../Docs/RC008_TASK3_CHARGE_PREDICTION.md)
