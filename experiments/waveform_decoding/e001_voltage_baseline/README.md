# E001 - Sensor-space voltage baseline

**Dataset:** Task 4 | **Status:** Completed | [Central registry](../../README.md)

This pipeline reads original SAGA XDF recordings and the master behavioral CSV. It predicts the delivered trigger class, not naturalness preference. Jason's table supplies the labels: 239=Sine, 240=Square, 241=Triangle, 242=LinDec, 243=LinInc. Only subjects 3-12 are supported.

## Setup and execution

From the repository root, create a Python 3.12 virtual environment and install dependencies:

```powershell
python -m venv .venv
& ./.venv/Scripts/python.exe -m pip install -r requirements-baseline.txt
& ./.venv/Scripts/python.exe -m pytest -q --rootdir=. tests
```

Explicit `--rootdir=.` avoids pytest interpreting brackets in the Windows workspace name as parametrization syntax. `requirements-baseline-lock.txt` records the exact environment used for the first real-data run; the smaller requirements file specifies supported dependency ranges.

```powershell
# Inspect and prepare one participant without fitting classifiers.
& ./.venv/Scripts/python.exe -m experiments.waveform_decoding.e001_voltage_baseline --subjects 3 --audit-only --output output/task4_audit

# Small end-to-end run; 19 permutations are a smoke check, not strong inference.
& ./.venv/Scripts/python.exe -m experiments.waveform_decoding.e001_voltage_baseline --subjects 3 --permutations 19 --output output/task4_smoke

# Full cohort, including leave-one-subject-out evaluation.
& ./.venv/Scripts/python.exe -m experiments.waveform_decoding.e001_voltage_baseline --permutations 99 --loso --output output/task4_baseline
```

Defaults are `--data-dir EEGData` and `--master-csv "EEG_Codes_Export/02_SourceLocalization_Task4/Master_Data_ToSub12 (1).csv"`. Override either for another authorized local dataset location. Outputs are ignored by Git because manifests and predictions contain participant-level information. No data is uploaded.

## Event audit

- Select exactly one XDF stream containing TRIGGERS. Verify channel units and convert microvolts to volts. Retain recorded stream timestamps without clock correction/dejittering; triggers and EEG are in the same stream. This does not establish physical current-delivery timing.
- Enumerate non-idle trigger runs, retaining raw code, original start sample, duration, and active-low inversion where applicable. Accept expected codes stable for at least three samples by default. Subject 3's observed pulses were eight or nine samples long; unexpected one-sample transitions were rejected. Check each participant's saved duration distribution before accepting the default.
- Preserve the CSV's original row order and require the complete accepted event sequence to match its first/second waveform columns exactly. On any discrepancy, save trigger diagnostics and stop that participant. No forward scanning, invented labels, or automatic ambiguous recovery is performed.
- Annotate original CSV row/trial ID, within-pair position, preceding code, actual neighboring time gaps, and contiguous comparison block. Repeated occurrences of the same unordered waveform comparison are grouped together for validation.
- Read response status from `T4 Naturalness Value` (1 or 2). Retain `T4 Waveform Detected` for provenance only: its zeros do not mark these current-protocol trials as failed. A missing/invalid naturalness response is flagged, not interpreted as proof of failed delivery.
- Audit race-to-five counts and within-pair gaps. Protocol discrepancies are reported, not silently fixed or used to fabricate boundaries.

## Signal processing and features

This intentionally minimal baseline is separate from the legacy ICA/source-localization pipeline. It applies a fixed spatial average reference after removing CREF/M1/M2. It uses 14 predefined central/frontocentral/centroparietal/parietal channels, with no data-driven channel or component selection.

There is **no temporal filtering, resampling, ICA, or stimulation interpolation**. This avoids backward smearing of an onset artifact by zero-phase filtering, while leaving slow drift, line noise, ocular/muscle signals, and possible electrical artifacts as explicit limitations. Fixed-bin means provide the first simple feature representation; they do not prove artifact removal.

| Model | Feature interval | Bin width | Baseline interval |
| --- | --- | --- | --- |
| post (primary) | +50 to +500 ms | 50 ms | -250 to -50 ms |
| pre (control) | -500 to -50 ms | 50 ms | -750 to -550 ms |
| late (sensitivity) | +150 to +500 ms | 50 ms | -250 to -50 ms |
| onset (diagnostic) | 0 to +20 ms | 5 ms | -250 to -50 ms |
| order/history (control) | No EEG | Not applicable | Not applicable |

Intervals are half-open. Pre and post use equal-length feature windows and separate preceding 200 ms baselines, so post-trigger samples cannot enter pre features. This extends the required pre-trigger data to -750 ms. The pre interval and its baseline can still carry preceding-stimulus information, especially for second presentations. Order features comprise elapsed time, event ordinal, first/second position, preceding-event gap, and one-hot preceding trigger; they exclude current waveform and comparison identity.

Reject boundary/irregular-timestamp epochs and epochs containing another accepted stimulus in the required signal span. A fixed 500 microvolt peak-to-peak limit over selected referenced channels is applied separately to pre/post non-onset spans, without fitting thresholds on held-out data. If either presentation fails EEG quality or the behavioral response is invalid, exclude the entire pair from primary evaluation but preserve both rows and reasons in the manifest. Onset amplitudes do not trigger the non-onset rejection rule. Inspect exclusions and channel quality before interpreting scores.

## Validation and statistical interpretation

Every model uses training-fold StandardScaler followed by shrinkage LDA (`solver=lsqr`, automatic shrinkage, equal class priors). There is no hyperparameter search.

Within each participant, five GroupKFold splits hold out entire unordered waveform comparisons. Both presentations of every trial stay together; repeated comparison blocks stay together too. Training samples with overlapping [-750,+500] ms signal support relative to a held-out sample are purged as complete pairs. Training folds must contain all five classes. These splits test generalization to held-out comparison contexts, which is stricter than random trial splitting.

LOSO holds out each complete participant. Scaling and LDA are refitted on the remaining participants. A participant contributing to LOSO must have successfully audited features; within-participant and LOSO outcomes are distinct.

Headline balanced accuracy is macro recall over five classes using pooled out-of-fold predictions. Individual comparison folds may contain fewer than five test classes; their mean balanced accuracy would not have the same interpretation. Reports also include macro-F1, confusion matrices, per-position and per-subject metrics, predicted probabilities, and explicit fold membership.

Restricted permutations circularly shift complete ordered trial pairs within each subject/comparison and optionally reverse orientation for a whole comparison. This preserves paired structure, comparison-specific waveform availability, and class counts. Scores are recomputed with the original fixed folds. The diagnostic null is conditional on comparison context, not an iid trial-label null; temporal nonstationarity can limit exchangeability. Permutation p=(1+number of null scores >= observed)/(1+number of permutations). The primary post score is prespecified; other scores/p-values are exploratory and are not corrected for multiple comparisons.

The 20% balanced-accuracy reference is not sufficient evidence by itself. Post scores can reflect charge/intensity differences, drift, context, or residual artifacts. High pre or order scores need investigation; low pre scores do not exclude post-only contamination. High onset scores indicate recoverable peri-trigger information, not its origin. This implementation does not establish charge matching or measured physical onset timing.

## Local outputs

- `run_config.json`: parameters, channel/window definitions, dependency versions, source-code hashes, CSV hash.
- `sub-N/trigger_runs.csv`: all non-idle runs and rejection reasons.
- `sub-N/events.csv`: full aligned manifest, response/EEG quality flags, and analysis eligibility.
- `sub-N/audit.json`: pulse durations, gaps, race-to-five checks, and exclusion counts.
- `sub-N/features.npz`: eligible fixed features, never raw EEG.
- `sub-N/scores.json`, `predictions.csv`: metrics, restricted nulls, folds, and held-out predictions.
- `loso_scores.json`, `loso_predictions.csv`: optional unseen-participant evaluation.
- `summary.json`, `RESULTS.md`: combined results and explicit failures/skips.

Sequence mismatches return a nonzero exit status even if other participants succeed. Do not silently treat a partial cohort as complete. Choose a fresh output directory for each run so earlier artifacts cannot be mistaken for current outputs.

Further work includes current-monitor timing validation, missed-trial sensitivity analysis, bad-channel/ocular-artifact investigation, and larger permutation runs. No source-localization or preference classifier is trained here.
