# E003 - Voltage feature audit

**Dataset:** Task 4 | **Status:** Completed | [Central registry](../../README.md)

Reusable numerical diagnostics for feature distributions, training-only scaling, within-class covariance conditioning and covariance resampling variability. The default adapter audits the exact saved arrays used by the initial baseline, including its post, pre, late, onset and order/history representations. It does not train a new decoding experiment, report new classification scores, remove features or change existing outputs.

## Run

Use the existing diagnostics environment (`requirements-diagnostics.txt`; Matplotlib is needed for the figure). From the repository root:

```powershell
& ./.venv/Scripts/python.exe -m pytest -q --rootdir=. tests/test_e003_voltage_feature_audit.py
$env:MPLCONFIGDIR = Join-Path (Get-Location) 'output/matplotlib-config'
& ./.venv/Scripts/python.exe -m experiments.signal_quality.e003_voltage_feature_audit --output output/task4_feature_audit_20261023
```

Choose a fresh output directory. Defaults read `output/task4_baseline`, subjects 3â€“12, and use 10 descriptive covariance resamples per fold. Override `--source`, `--subjects`, `--bootstrap-repeats` and `--seed` as needed. Smaller participant subsets are smoke checks; their LOSO folds differ from the original full-cohort evaluation and are marked accordingly.

The existing baseline cohort has 1,244 events, distinct from the ICA response diagnostic's 1,242 events. This audit reads the original baseline feature arrays and eligible manifests directly. It verifies all feature-row counts and first/second pairs and reproduces the saved classifier fold memberships using subject/event identity. No XDF loading or ICA fitting is needed. Source hashes and exact dependencies are saved. The original event/feature code has evolved since the first run; auditing the saved arrays avoids silently replacing the features actually evaluated.

## Checks

- Explicit feature order and units: saved EEG columns are bin-major and channel-minor, in ÂµV; order/history columns have separate units and include intentional binary indicators.
- Finite values, per-column ranges, percentiles, SD, exact constants, near constants and observations outside three IQRs. Near-constant flags compare SD with 1eâˆ’8 times the median positive SD **within the same unit**. These flags prompt interpretation, not automatic removal; the threshold is not a universal scientific cutoff.
- Refit the original `StandardScaler â†’ equal-prior, lsqr LDA(shrinkage='auto')` on each original training fold. Record training-only fit counts and mean/SD checks, then inspect held-out standardized excursions. No held-out label enters fitting, and no classification score is computed.
- Compare empirical within-class covariance with the actual fitted LDA covariance, using equal class priors. Reconstruct scikit-learn's automatic estimator from its within-class standardization, Ledoitâ€“Wolf shrinkage and rescaling, and require agreement with `LDA.covariance_`. Save class-specific shrinkage coefficients rather than inventing one pooled coefficient.
- Record eigenvalue spectra, numerical rank, rank at relative tolerance 1eâˆ’8, entropy effective rank, condition number and feature correlations. Numerical rank uses machine epsilon Ã— feature dimension Ã— largest eigenvalue. High correlation is redundancy, not automatically an error. The empirical within-class rank cannot exceed min(p, n_train âˆ’ number_of_classes).
- Resample complete pairs with replacement within subject/comparison strata. Keep the outer training scaler fixed so covariance estimates remain in the same coordinates; re-estimate each class covariance and shrinkage. Record relative Frobenius change from the corresponding original covariance. This describes variability conditional on the scaler, not whole-pipeline uncertainty or guaranteed stability of weak eigen-directions. Ten resamples are a compact diagnostic, not an inferential bootstrap; overlapping folds/resamples are dependent.

## Reusing the audit

Rerun numerical checks whenever features, units, windows, channels, preprocessing, eligible trials or splits change. Preserve input hashes and reuse unchanged event/protocol verification. A new feature bundle needs a names/units contract; `describe_features` and `audit_training_fold` accept generic arrays independently of the baseline adapter.

For learned features, fit the transform inside each training fold and audit the resulting training/test arrays in that fold. A global precomputed array is inappropriate if its construction used held-out observations or labels. The global descriptive distributions here must not become a rule for selecting channels, clipping values or tuning parameters against the test data. Any learned exclusion/scaling/selection rule belongs in the training pipeline.

## xDAWN comparison

xDAWN is a reasonable additional spatial representation despite nonsignificant individual-electrode contrasts. It learns combinations emphasizing event-related responses; usefulness for five-waveform decoding remains untested. A candidate fixed pipeline is training-only, regularized xDAWN with two components per waveform class, then the same temporal bins, training-only scaling and shrinkage LDA. Retain identical eligible trials and validation splits and rerun this audit on the projected features. Refit xDAWN for every fold and every label permutation; tune component counts only inside training folds if tuning is later introduced. It cannot establish a neural origin or fix unknown physical trigger timing.

The separate [task4_xdawn experiment](../../waveform_decoding/e004_xdawn_decoding/README.md) implements this comparison and reruns this audit on its fold-specific projected features. This baseline audit itself contains no xDAWN fits. See the official [XdawnTransformer documentation](https://mne.tools/stable/generated/mne.decoding.XdawnTransformer.html) and [cross-validation example](https://mne.tools/stable/auto_examples/decoding/decoding_xdawn_eeg.html).

## Local outputs

All participant-level outputs remain Git-ignored:

- `RESULTS.md`, `primary_post_diagnostics.png`, `summary.csv`, `checks.json`.
- `feature_distributions.csv`: named, unit-aware column summaries.
- `fold_audit.csv`: fold-specific scaling, rank, correlation and conditioning checks.
- `class_shrinkage.csv`, `covariance_resamples.csv`, `covariance_eigenspectra.npz`.
- `run_config.json`, `input_hashes.json`, `fold_membership.json`, `status.json`: source/dependency provenance and explicit completion/failure status.
