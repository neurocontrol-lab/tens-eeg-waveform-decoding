# E005 - Waveform decoding with spectral and combined features

**Dataset:** Task 4 | **Status:** Completed | [Central registry](../../README.md)

This experiment asks whether broad alpha/beta power adds waveform information beyond the original temporal-mean voltage baseline. It follows the fixed xDAWN comparison; it does not search frequency bands, channels, windows or classifiers.

## Fixed protocol

- Subjects 3â€“12, exact original baseline cohort (1,244 events in 622 complete pairs), the same 14 predefined electrodes, scalp average reference and original minimal preprocessing. No added temporal filtering, ICA, onset interpolation or new exclusions. Acquisition-system filtering may already exist.
- Post [50,500) ms, voltage baseline [âˆ’250,âˆ’50) ms. Matched pre [âˆ’500,âˆ’50) ms, independent voltage baseline [âˆ’750,âˆ’550) ms. Pre features never use post-trigger samples. Epochs are trigger-aligned; physical current-delivery delay remains unverified.
- Alpha 8â€“13 Hz and beta 13â€“30 Hz, one full-epoch Hann periodogram per event/electrode. Constant detrend, one-sided density, no padding or trial averaging. A 450-ms epoch has approximately 2.22-Hz native frequency spacing; interpolated boundaries do not improve resolution.
- Input volts become ÂµV before PSD estimation. Integrate piecewise-linear density (ÂµVÂ²/Hz) with exact interpolated band boundaries to obtain ÂµVÂ². Features are `10*log10(power / 1 ÂµVÂ²)`, with a fixed numerical floor of `1e-20 ÂµVÂ²`; every floor hit is recorded. Shared alpha/beta boundaries have zero measure and do not double count area.
- Constant detrending removes the earlier voltage-baseline offset for power. Power is absolute epoch band power, **not** a pre/post power ratio or spectral baseline correction.
- Three fixed representations: 126 voltage features (nine 50-ms bins Ã— 14 electrodes), 28 power features (two bands Ã— 14 electrodes), and their 154-column concatenation. Contracts explicitly preserve ÂµV versus dB units. No class-specific ERP subtraction or learned feature transform.
- Original training-only StandardScaler and equal-prior shrinkage LDA (`solver='lsqr', shrinkage='auto'`). Original five comparison-held-out folds within each participant and complete-subject LOSO; whole pairs and comparisons stay together, with the original temporal-support purge.
- Original sensor features, observed predictions/probabilities and fold identities must reproduce the baseline. For compatible dependencies/code/seed, the independently recomputed voltage null distributions must also match exactly. Null scores are not reused.
- 199 restricted permutations (seed 42), using the same label permutation across representations/windows. Shift complete pairs within participant/comparison and optionally reverse orientation comparison-wide; refit scaler and classifier for every null. These are conditional diagnostic nulls, not iid randomizations; temporal nonstationarity may limit exchangeability. The shared first 99 voltage nulls must reproduce the original run; the other 100 are new evaluations.
- Six LOSO tests form one Holm family (three representations Ã— two windows). All 60 participant-specific tests form a separate Holm family; these are exploratory, and their nulls are not pooled into a cohort p value. First/second presentation scores are descriptive.
- At 199 permutations, raw p resolution is 0.005. The six-test LOSO family can reach a corrected p of 0.03; the 60-test participant-specific family cannot reach 0.05 (its minimum is 0.30). Participant-specific adjusted tests therefore cannot support an affirmative or definitive negative result at this resolution. Use the participant distributions and descriptive intervals to characterize those results.
- Within-participant group summaries weight participants equally. Descriptive 95% bootstrap intervals use 10,000 participant resamples (seed 20261023); paired differences preserve participant identity. They are not formal evidence of classifier superiority and do not resample folds as independent observations.

## Repeated feature audit

Every observed fold/representation/window repeats distribution, constant/near-constant, training-only scaling, held-out extremes, feature correlation, empirical/shrinkage covariance rank and conditioning checks. Near-constant comparisons remain within physical units for combined features. Covariance stability uses ten whole-pair resamples within participant/comparison, conditional on the outer training scaler. The audit reconstructs the actual classifier covariance and checks saved predictions against fitted parameters. No flags change eligibility, features, clipping or model settings.

The canonical run has 360 audited folds: 50 within-participant plus ten LOSO folds, each with six representations/windows. Detailed training/test distributions, class shrinkages and pair-bootstrap covariance results remain in ignored output files.

## Run

Run from the repository root with the existing diagnostics environment (`requirements-diagnostics-lock.txt`); no new dependencies. The shared epoch/cache helper imports the existing MNE/xDAWN module, but this experiment fits no xDAWN or ICA models.

```powershell
$env:MPLCONFIGDIR = Join-Path (Get-Location) 'output/matplotlib-config'
& ./.venv/Scripts/python.exe -m experiments.waveform_decoding.e005_spectral_decoding --output output/task4_spectral_new
& ./.venv/Scripts/python.exe -m pytest tests/test_e005_spectral_decoding.py -q
```

Use a fresh output directory. Defaults: all ten participants, 199 permutations, ten audit resamples, seed 42. A small diagnostic subset requires `--subjects 3 --no-loso`; zero permutations are accepted only as a descriptive smoke run, with no inferential p values. The default provenance-verified full-rate cache is `output/task4_xdawn_epoch_cache/`. If missing, epochs are reconstructed from XDF/CSV and compared against saved baseline events/features. All source data hashes, epoch cache hashes, dependencies, parameters and implementation hashes are saved; implementation hashes must remain unchanged during a run.

Outputs include `RESULTS.md`, `index.html`, two figures, aggregate scores/intervals/audits, and per-evaluation features, linear powers, contracts, numerical-floor counts, folds, predictions, null distributions, model parameters and detailed audits. Participant data and generated outputs remain Git-ignored. See [the aggregate findings](../../../Docs/TASK4_SPECTRAL_RESULTS.md) for the completed canonical run.

## Interpretation

These are **total single-trial power** features: evoked transients and ongoing activity both contribute. They do not isolate induced oscillations or establish neural origin. Power can also reflect muscle, movement, stimulation contamination or recording-order effects. A shaped stimulation pulse need not yield EEG oscillations matching its spectrum. Pre-trigger activity may contain anticipation and preceding-stimulus responses, particularly before second presentations; it is not guaranteed to be a null.

Negative results apply to this fixed representation/window/validation design, not every possible EEG feature. A numerical accuracy increase must be assessed against its uncertainty and conditional null before being interpreted. No channels, bands or epochs are selected using the full-cohort findings.

Primary method references: [SciPy periodogram](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.periodogram.html), [MNE decoding](https://mne.tools/stable/auto_tutorials/machine-learning/50_decoding.html), and [FieldTrip on evoked versus induced power](https://www.fieldtriptoolbox.org/faq/spectral/evoked_vs_induced/).
