# WD-007. Task 4 nominal-charge adjustment

Compare original and adjusted EEG on the WD-001 within-participant folds. The frozen RC-006 charge-only slopes match the 14 channels and nine 50 ms baseline bins. Both variants use a zero reference charge. The waveform-specific comparison uses participants with all five waveform calibrations and evaluates the pooled model and baseline on that same subset.

Run from the repository root with the original WD-001 and RC-006 full-montage outputs available.

```powershell
& ./.venv/Scripts/python.exe -m experiments.waveform_decoding.e018_nominal_charge_adjustment --output output/wd007_nominal_charge_20261113_verified --web-report Docs/Reports/WD-007/index.html
```

Use a fresh output path when rerunning. The local `summary.json` records input hashes, paired accuracy changes and control scores, and `predictions.csv` stores the held-out predictions. The standalone HTML embeds participant-level summary scores, recall and charge-overlap summaries. Trial-level predictions and fitted slopes remain in ignored local outputs.

[Findings](../../../Docs/WD007_NOMINAL_CHARGE_RESULTS.md) · [Interactive report](../../../Docs/Reports/WD-007/index.html)
