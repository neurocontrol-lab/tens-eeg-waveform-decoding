# E004 - Waveform decoding with xDAWN spatial features

**Dataset:** Task 4 | **Status:** Completed | [Central registry](../../README.md)

This exploratory experiment tests a fixed supervised spatial representation against the original sensor-voltage baseline. It decodes the five delivered waveforms, not naturalness. The post-stimulus representation is [50,500) ms; an independently fitted [âˆ’500,âˆ’50) ms pre-stimulus representation is a matched context/carryover control.

## Run

Use the existing `requirements-diagnostics.txt` environment, with MNE 1.13.x and the original saved `output/task4_baseline` run. From the repository root:

```powershell
& ./.venv/Scripts/python.exe -m pytest -q --rootdir=. tests/test_e004_xdawn_decoding.py
$env:MPLCONFIGDIR = Join-Path (Get-Location) 'output/matplotlib-config'

# One-participant smoke check; two nulls cannot support inference.
& ./.venv/Scripts/python.exe -m experiments.waveform_decoding.e004_xdawn_decoding --subjects 3 --no-loso --permutations 2 --audit-resamples 2 --output output/task4_xdawn_smoke

# All ten participants: within participant, then unseen-participant evaluation.
& ./.venv/Scripts/python.exe -m experiments.waveform_decoding.e004_xdawn_decoding --output output/task4_xdawn_20261023

# Prepare provenance-matched epochs without fitting models.
& ./.venv/Scripts/python.exe -m experiments.waveform_decoding.e004_xdawn_decoding --prepare-only --output output/task4_xdawn_prepare
```

Choose a fresh output directory. Defaults use subjects 3â€“12, 99 restricted permutations, seed 42 and ten descriptive covariance resamples per observed fold. Override local paths with `--baseline`, `--data-dir`, `--master-csv`, `--cache-dir`. Partial cohorts require `--no-loso` because matched LOSO reproduces the original full-cohort folds. Failures are explicit in `status.json` and return a nonzero exit code; a partial cohort is not silently reported as complete.

## Frozen cohort and signal representation

Reconstruct the original minimal baseline directly from XDF with its audited trigger/parser and complete master-CSV alignment. Require identical event identities, samples, eligibility, all five saved feature arrays and temporal-bin reconstructions. The master CSV must match the original baseline hash. No new exclusions are made: the comparison uses the original 1,244 events in 622 pairs, distinct from the descriptive ICA cohort of 1,242 events.

Retain the same 14 predefined electrodes and fixed spatial average reference across the 62 scalp electrodes. There is no temporal filtering, ICA, onset interpolation or resampling. Post voltage baseline is [âˆ’250,âˆ’50) ms; pre voltage baseline is [âˆ’750,âˆ’550) ms. Thus a post-trigger sample cannot enter the pre feature through a temporal filter. Physical carryover and contextual expectations remain possible.

The cache stores only these 14-channel epoched voltages, in volts, at the native sample rate; it never copies continuous EEG. Recording/CSV/baseline artifact hashes, numerical extraction source hashes and dependencies determine its identity. Incomplete stages are preserved and rejected. Cached epoch generation is deterministic and uses no fitted model or labels beyond protocol alignment.

## Fixed training-only xDAWN branch

- Fit MNE `XdawnTransformer(n_components=2, reg=0.1, rank='full')` on full-rate **training epochs and training waveform labels**. Two components per five classes produce ten spatial mixtures. The fixed 10% covariance shrinkage is not selected from accuracy. No component/channel search or nested tuning is performed.
- Project onto that fold's learned filters, then form the same nine 50-ms means. There are 90 projected features versus 126 sensor features. For efficiency, transformation projects the temporal means directly: this equals temporal means of the full projection because the fitted spatial map is linear. An explicit numerical test verifies the equivalence. This does not move temporal averaging before xDAWN **fitting**, which always uses full-rate epochs.
- Fit the original training-only scaler and equal-prior `LDA(solver='lsqr', shrinkage='auto')`. The post and pre branches fit independently; each window's spatial filters see only that window.
- Projected values are weighted ÂµV after the linear transformation. Their signs, scales and component axes belong to the training fit; raw component magnitudes are not treated as common across folds.

The comparator refits the original sensor model and must reproduce its saved out-of-fold predictions and probabilities. Every validation fold matches the saved original event membership. Both presentations of each pair, repeated comparison groups and the original support purge are preserved. Within-participant five-fold validation holds out waveform comparisons; LOSO holds out each complete participant. Balanced accuracy pools predictions across folds using all five classes, rather than averaging folds with different available test classes.

## Null controls and uncertainty

Restricted permutations circularly shift complete ordered pairs within participant/comparison and may reverse orientation comparison-wide, preserving pair membership and class support. **Every permutation refits xDAWN, the scaler and LDA inside every training fold.** No observed-label spatial filters are reused for null scoring.

The original sensor null scores may be reused only when source model code, dependencies, exact folds, seed, null-count availability and observed predictions agree. The deterministic label sequence is shared by both representations/windows. Otherwise sensor nulls are recomputed. Reuse is recorded per evaluation. Label-independent epoch caching never caches a supervised projection.

These nulls are conditional diagnostics, not iid-label or physical acquisition randomization tests; temporal nonstationarity can limit exchangeability. Ninety-nine permutations give p-value resolution .01. Two LOSO xDAWN windows receive Holm correction. Participant-specific p-values are exploratory and uncorrected; their null draws are not combined into a cohort p-value. Participant bootstrap intervals summarize mean within-participant accuracy and paired xDAWN-minus-sensor differences descriptively. They do not provide an untouched confirmatory test after exploring representations on this dataset.

## Repeated projected-feature audit

Inside every observed training fold, rerun the reusable feature audit on the actual projected arrays: finite values, named component/bin units, training/test distributions, training-only scaler checks, within-class covariance reconstruction/rank/conditioning, class-specific shrinkage and whole-pair covariance resampling. Held-out feature extremes remain inspection flags, not new exclusion rules.

Covariance resampling holds that fold's xDAWN filters and outer scaler fixed. It measures downstream covariance variability conditional on the representation; it does not resample the entire supervised xDAWN procedure. Spatial models, scaler parameters and LDA parameters are saved per observed fold. No full-cohort fit is used to generate classifier inputs or interpretation maps.

## Outputs and limits

All derived participant data stay under Git-ignored `output/`:

- `run_config.json`, `epoch_provenance.json`, `status.json`, `summary.json`.
- `scores.csv`, `within_participant_summary.csv`, `projected_fold_audit.csv`, Markdown/HTML report and scientific figures.
- Per evaluation: events, exact fold membership, predictions/probabilities, restricted-null scores, spatial/classifier models, feature distributions and covariance resamples.

Pre-stimulus performance diagnoses possible contextual information; low pre performance does not rule out post-only artifacts. A post score cannot establish a cortical origin or isolate pulse shape from charge/intensity. Physical trigger-to-current timing remains unverified. xDAWN does not correct physiological carryover merely because the analyzed 450-ms epochs do not overlap. Saved spatial patterns are sensor mixtures, not localized neural generators. A null result concerns these fixed features and validation conditions, not every possible EEG representation.

Method reference: [MNE xDAWN transformer](https://mne.tools/stable/generated/mne.decoding.XdawnTransformer.html) and its [cross-validation example](https://mne.tools/stable/auto_examples/decoding/decoding_xdawn_eeg.html).
