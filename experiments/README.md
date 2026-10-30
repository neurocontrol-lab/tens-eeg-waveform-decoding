# Experiment registry

Experiments have stable project-wide IDs, descriptive package names and physical category folders. Task/cohort information belongs in metadata and each run guide. Run all commands from the repository root.

**Updated: 2026-10-30.** 12 completed, 2 planned. Completed means the specified analysis finished; it does not imply a positive scientific result.

## Status of completed and planned experiments

| ID | Experiment | Category | Dataset | Status | Last completed run | Findings |
| --- | --- | --- | --- | --- | --- | --- |
| E001 | [Sensor-space voltage baseline](waveform_decoding/e001_voltage_baseline/README.md) | waveform decoding | Task 4 | Completed | 2026-10-16 | [Results](../Docs/BASELINE_RESULTS.md) |
| E002 | [Electrode, presentation-position and EEG-slope responses](response_characterization/e002_electrode_position_slope/README.md) | response characterization | Task 4 | Completed | 2026-10-21 | [Results](../Docs/TASK4_RESPONSE_DIAGNOSTICS.md) |
| E003 | [Voltage feature audit](signal_quality/e003_voltage_feature_audit/README.md) | signal quality | Task 4 | Completed | 2026-10-23 | [Results](../Docs/TASK4_FEATURE_AUDIT.md) |
| E004 | [Waveform decoding with xDAWN spatial features](waveform_decoding/e004_xdawn_decoding/README.md) | waveform decoding | Task 4 | Completed | 2026-10-23 | [Results](../Docs/TASK4_XDAWN_RESULTS.md) |
| E005 | [Waveform decoding with spectral and combined features](waveform_decoding/e005_spectral_decoding/README.md) | waveform decoding | Task 4 | Completed | 2026-10-23 | [Results](../Docs/TASK4_SPECTRAL_RESULTS.md) |
| E006 | [Multiscale EEG response characterization](response_characterization/e006_multiscale_responses/README.md) | response characterization | Task 4 | Completed | 2026-10-23 | [Results](../Docs/TASK4_MULTISCALE_RESULTS.md) |
| E007 | [Minimal versus supplied preprocessing comparison](signal_quality/e007_preprocessing_comparison/README.md) | signal quality | Task 4 | Completed | 2026-10-23 | [Results](../Docs/TASK4_PREPROCESSING_COMPARISON.md) |
| E008 | [Time-resolved waveform, position and interaction effects](response_characterization/e008_time_resolved_effects/README.md) | response characterization | Task 4 | Completed | 2026-10-23 | [Results](../Docs/TASK4_TIME_RESOLVED_RESULTS.md) |
| E009 | [Response uncertainty, earlier time coverage and sensitivity](uncertainty_and_sensitivity/e009_response_uncertainty_sensitivity/README.md) | uncertainty and sensitivity | Task 4 | Completed | 2026-10-24 | [Results](../Docs/TASK4_RESPONSE_FOLLOWUP.md) |
| E010 | [Simple-pipeline EEG visual inspection](signal_quality/e010_simple_visual_inspection/README.md) | signal quality | Task 4 | Completed | 2026-10-30 | [Results](../Docs/TASK4_VISUAL_INSPECTION.md) |
| E011 | [Electrode-by-time-bin waveform repeatability](response_characterization/e011_electrode_bin_repeatability/README.md) | response characterization | Task 4 | Completed | 2026-10-30 | [Results](../Docs/E011_REPEATABILITY_RESULTS.md) |
| E012 | Longer-window waveform decoding with LDA, RBF SVM and EEGNet | waveform decoding | Task 4 | Planned | - | - |
| E013 | Classifier sensitivity to known EEG patterns | uncertainty and sensitivity | Task 4 | Planned | - | - |
| E014 | [All-electrode, alpha/beta power and calibrated xDAWN repeatability](response_characterization/e014_all_electrode_repeatability/README.md) | response characterization | Task 4 | Completed | 2026-10-30 | [Results](../Docs/E014_REPEATABILITY_RESULTS.md) |

Planned IDs are reserved; no empty executable packages or results are created for them. Use Planned, In progress, Completed, Blocked or Superseded. Use Scheduled only when an execution has an actual schedule. Status labels do not start jobs or recurring automations.

## Category directories

### [Signal quality, preprocessing and feature audits](signal_quality/README.md)

What do recordings and feature matrices look like, and how do processing choices affect responses?

- **E003** - Voltage feature audit (completed).
- **E007** - Minimal versus supplied preprocessing comparison (completed).
- **E010** - Simple-pipeline EEG visual inspection (completed).

### [Stimulus-aligned response characterization](response_characterization/README.md)

How do voltage and slope patterns vary with waveform, electrode, time and presentation position?

- **E002** - Electrode, presentation-position and EEG-slope responses (completed).
- **E006** - Multiscale EEG response characterization (completed).
- **E008** - Time-resolved waveform, position and interaction effects (completed).
- **E011** - Electrode-by-time-bin waveform repeatability (completed).
- **E014** - All-electrode, alpha/beta power and calibrated xDAWN repeatability (completed).

### [Single-epoch waveform decoding](waveform_decoding/README.md)

Can EEG features predict the delivered waveform on held-out data?

- **E001** - Sensor-space voltage baseline (completed).
- **E004** - Waveform decoding with xDAWN spatial features (completed).
- **E005** - Waveform decoding with spectral and combined features (completed).
- **E012** - Longer-window waveform decoding with LDA, RBF SVM and EEGNet (planned).

### [Uncertainty and sensitivity](uncertainty_and_sensitivity/README.md)

How uncertain are the summaries, and which specified reference effects can the tests recover?

- **E009** - Response uncertainty, earlier time coverage and sensitivity (completed).
- **E013** - Classifier sensitivity to known EEG patterns (planned).

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

- Register the next unused ID in `registry.json` and this status table; never renumber or reuse an ID.
- Create `category/eNNN_descriptive_name/` when implementing it. A materially different scientific experiment receives a new ID; routine reruns retain their ID.
- Keep dataset/task, cohort, preprocessing, features, validation and limitations explicit in the experiment README and run configuration.
- Update the category README and central status when work starts or finishes. Record the canonical run and findings; preserve earlier runs.
- Use fresh Git-ignored output directories. Keep tests under root `tests/` and document dependencies in the existing requirements files.

The baseline moved into `experiments/` on 20 October 2026. On 30 October the ten packages moved into the category/ID layout. These organizational changes do not constitute new numerical analysis results.
