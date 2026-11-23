# Experiment registry

Each category has its own three-digit sequence starting at `001`. The prefixes are **WD** for waveform decoding, **RC** for response characterization, **SQ** for signal quality, and **US** for uncertainty and sensitivity. The Reference column lists the `E` numbers used in package names and earlier reports. For example, **RC-006** is the EEG charge study referenced as **E017**. Task and cohort information belongs in the registry and run guide. Run all commands from the repository root.

**Updated: 2026-11-24.** 18 completed, 2 planned, 2 in progress. Completed means the specified analysis finished; it does not imply a positive scientific result.

Planned IDs are reserved; no empty executable packages or results are created for them. Use Planned, In progress, Completed, Blocked or Superseded. Use Scheduled only when an execution has an actual schedule. Status labels do not start jobs or recurring automations.

## Status of completed and planned experiments

### [WD — Waveform and naturalness-choice decoding](waveform_decoding/README.md)

Can EEG features predict the delivered waveform or the participant's choice between two stimuli on held-out data?

| ID | Experiment | Dataset | Status | Last completed run | Findings | Reference |
| --- | --- | --- | --- | --- | --- | --- |
| WD-001 | [Sensor-space voltage baseline](waveform_decoding/e001_voltage_baseline/README.md) | Task 4 | Completed | 2026-10-16 | [Results](../Docs/BASELINE_RESULTS.md) | E001 |
| WD-002 | [Waveform decoding with xDAWN spatial features](waveform_decoding/e004_xdawn_decoding/README.md) | Task 4 | Completed | 2026-10-23 | [Results](../Docs/TASK4_XDAWN_RESULTS.md) | E004 |
| WD-003 | [Waveform decoding with spectral and combined features](waveform_decoding/e005_spectral_decoding/README.md) | Task 4 | Completed | 2026-10-23 | [Results](../Docs/TASK4_SPECTRAL_RESULTS.md) | E005 |
| WD-004 | Longer-window waveform decoding with LDA, RBF SVM and EEGNet | Task 4 | Planned | - | - | E012 |
| WD-005 | [Position-specific multiscale repeatability-selected decoding](waveform_decoding/e015_position_repeatability_decoding/README.md) | Task 4 | In progress | 2026-11-02 | [Results](../Docs/E015_OBSERVED_RESULTS.md) | E015 |
| WD-006 | [One-versus-rest waveform detection on fixed E015 features](waveform_decoding/e016_one_vs_rest_decoding/README.md) | Task 4 | In progress | 2026-11-02 | [Results](../Docs/E016_BINARY_RESULTS.md) | E016 |
| WD-007 | [Task 4 waveform decoding with nominal-charge adjustment](waveform_decoding/e018_nominal_charge_adjustment/README.md) | Task 4 | Completed | 2026-11-13 | [Results](../Docs/WD007_NOMINAL_CHARGE_RESULTS.md) | E018 |
| WD-008 | [Task 4 naturalness-choice decoding from paired EEG](waveform_decoding/e021_naturalness_choice_decoding/README.md) | Task 4 | Completed | 2026-11-23 | [Results](../Docs/WD008_NATURALNESS_CHOICE_RESULTS.md) | E021 |
| WD-009 | [Triangle versus linear-decrease waveform decoding](waveform_decoding/e022_triangle_lindec_decoding/README.md) | Task 4 | Completed | 2026-11-23 | [Results](../Docs/WD009_TRIANGLE_LINDEC_RESULTS.md) | E022 |

### [RC — Stimulus-aligned response characterization](response_characterization/README.md)

How do voltage and slope patterns vary with waveform, electrode, time and presentation position?

| ID | Experiment | Dataset | Status | Last completed run | Findings | Reference |
| --- | --- | --- | --- | --- | --- | --- |
| RC-001 | [Electrode, presentation-position and EEG-slope responses](response_characterization/e002_electrode_position_slope/README.md) | Task 4 | Completed | 2026-10-21 | [Results](../Docs/TASK4_RESPONSE_DIAGNOSTICS.md) | E002 |
| RC-002 | [Multiscale EEG response characterization](response_characterization/e006_multiscale_responses/README.md) | Task 4 | Completed | 2026-10-23 | [Results](../Docs/TASK4_MULTISCALE_RESULTS.md) | E006 |
| RC-003 | [Time-resolved waveform, position and interaction effects](response_characterization/e008_time_resolved_effects/README.md) | Task 4 | Completed | 2026-10-23 | [Results](../Docs/TASK4_TIME_RESOLVED_RESULTS.md) | E008 |
| RC-004 | [Electrode-by-time-bin waveform repeatability](response_characterization/e011_electrode_bin_repeatability/README.md) | Task 4 | Completed | 2026-10-30 | [Results](../Docs/E011_REPEATABILITY_RESULTS.md) | E011 |
| RC-005 | [All-electrode, alpha/beta power and calibrated xDAWN repeatability](response_characterization/e014_all_electrode_repeatability/README.md) | Task 4 | Completed | 2026-10-30 | [Results](../Docs/E014_REPEATABILITY_RESULTS.md) | E014 |
| RC-006 | [EEG response and nominal cathodic charge](response_characterization/e017_eeg_charge_relation/README.md) | Task 3 | Completed | 2026-11-09 | [Results](../Docs/E017_TASK3_CHARGE_RESULTS.md) | E017 |
| RC-007 | [Predict nominal Task 4 charge from EEG](response_characterization/e019_task4_charge_prediction/README.md) | Task 4 | Completed | 2026-11-13 | [Results](../Docs/RC007_TASK4_CHARGE_PREDICTION.md) | E019 |
| RC-008 | [Predict nominal Task 3 charge from EEG](response_characterization/e020_task3_charge_prediction/README.md) | Task 3 | Completed | 2026-11-13 | [Results](../Docs/RC008_TASK3_CHARGE_PREDICTION.md) | E020 |

### [SQ — Signal quality, preprocessing and feature audits](signal_quality/README.md)

What do recordings and feature matrices look like, and how do processing choices affect responses?

| ID | Experiment | Dataset | Status | Last completed run | Findings | Reference |
| --- | --- | --- | --- | --- | --- | --- |
| SQ-001 | [Voltage feature audit](signal_quality/e003_voltage_feature_audit/README.md) | Task 4 | Completed | 2026-10-23 | [Results](../Docs/TASK4_FEATURE_AUDIT.md) | E003 |
| SQ-002 | [Minimal versus supplied preprocessing comparison](signal_quality/e007_preprocessing_comparison/README.md) | Task 4 | Completed | 2026-10-23 | [Results](../Docs/TASK4_PREPROCESSING_COMPARISON.md) | E007 |
| SQ-003 | [Simple-pipeline EEG visual inspection](signal_quality/e010_simple_visual_inspection/README.md) | Task 4 | Completed | 2026-10-30 | [Results](../Docs/TASK4_VISUAL_INSPECTION.md) | E010 |

### [US — Uncertainty and sensitivity](uncertainty_and_sensitivity/README.md)

How uncertain are the summaries, and which specified reference effects can the tests recover?

| ID | Experiment | Dataset | Status | Last completed run | Findings | Reference |
| --- | --- | --- | --- | --- | --- | --- |
| US-001 | [Response uncertainty, earlier time coverage and sensitivity](uncertainty_and_sensitivity/e009_response_uncertainty_sensitivity/README.md) | Task 4 | Completed | 2026-10-24 | [Results](../Docs/TASK4_RESPONSE_FOLLOWUP.md) | E009 |
| US-002 | Classifier sensitivity to known EEG patterns | Task 4 | Planned | - | - | E013 |

## Run locations and historical provenance

The machine-readable [registry](registry.json) records canonical local output directories and old package names. Existing `output/task4_*` results, caches and report URLs stay in place; ID-based package names do not rename historical runs. Raw data, participant outputs and local caches remain Git-ignored.

The [migration record](_provenance/README.md) preserves original source bytes and exact before/after digests. Historical inputs, fitted ICA and numerical results are unchanged. Old recorded source paths are resolved only through the explicit relocation mapping; further source edits still fail historical checks. New runs record current paths and actual current hashes. The verified raw-epoch cache can be reused across this migration without refitting ICA.

The original classifiers (E001/E004/E005) use minimal preprocessing without added temporal filters or supplied ICA. E002 and its response follow-ups use the separately documented descriptive processing branches. E010 compares minimal and simple band-pass views, with no new classification.

## Running an experiment

```powershell
& ./.venv/Scripts/python.exe -m experiments.waveform_decoding.e001_voltage_baseline --help
& ./.venv/Scripts/python.exe -m experiments.signal_quality.e010_simple_visual_inspection --help
```

Each package README supplies full run commands, cohort definitions and interpretation limits. Feature audits and controls embedded in a classifier remain part of its experiment. E009 also contributes earlier-window response characterization; its existing injection calibration assesses the response-statistics test, not classifier sensitivity or preprocessing attenuation.

## Adding or updating an experiment

- Register the next number in the experiment's category as `category_number` in `registry.json`. Assign the next `ENNN` reference for package naming. Include planned entries when allocating numbers. Run `python -m experiments.catalog` to refresh the status tables.
- Create `category/eNNN_descriptive_name/` when implementing the experiment. Use a new ID for a materially different experiment and reuse the existing ID for routine reruns.
- Keep dataset/task, cohort, preprocessing, features, validation and limitations explicit in the experiment README and run configuration.
- Update the category README and central status when work starts or finishes. Record the canonical run and findings; preserve earlier runs.
- Use fresh Git-ignored output directories. Keep tests under root `tests/` and document dependencies in the existing requirements files.

The baseline moved into `experiments/` on 20 October 2026. On 30 October the ten packages moved into the category/ID layout. These organizational changes do not constitute new numerical analysis results.
