# TENS EEG Waveform Decoding

Research project investigating whether EEG can identify the waveform delivered during transcutaneous electrical nerve stimulation (TENS) of the median nerve.

The immediate decoding target comprises five waveform classes: sine, square, centered triangle, increasing ramp, and decreasing ramp. The wider motivation is understanding waveform-dependent sensory responses and naturalness for sensory restoration.

## Project materials

- `EEG_Codes_Export/01_Preprocessing_Task4/`: supplied XDF-to-MNE preprocessing and supporting functions.
- `EEG_Codes_Export/02_SourceLocalization_Task4/`: supplied source-space waveform and preference comparisons.
- `EEG_Codes_Export/03_BaseEpochPlotting_Task4/`: supplied sensor-space ROI epoch plotting.
- `EEG_Codes_Export/*.pdf`: experimental procedures and Jason Whitson's undergraduate thesis.
- `Docs/`: motivation paper.
- `PROJECT_REVIEW.md`: document/code review, dataset overview, known inconsistencies, and proposed decoding analysis.
- `Docs/EXPERIMENT_CLARIFICATIONS.md`: Jason's confirmed trigger table, Task 4 protocol, timing limits, and pending audit questions.
- `experiments/`: central home for executable analysis pipelines and future experiment versions. See the [experiment index](experiments/README.md).
- `experiments/waveform_decoding/e001_voltage_baseline/`: auditable event extraction and fixed sensor-space waveform classifiers, with pre-stimulus, onset, late-window, and order/history controls. See its [run guide](experiments/waveform_decoding/e001_voltage_baseline/README.md).
- `Docs/BASELINE_RESULTS.md`: initial full-cohort results and interpretation limits (2026-10-16).
- `experiments/response_characterization/e002_electrode_position_slope/`: supplied-preprocessing adaptation for exploratory electrode, first/second presentation, repeatability and EEG-slope comparisons. See its [run guide](experiments/response_characterization/e002_electrode_position_slope/README.md).
- `Docs/TASK4_RESPONSE_DIAGNOSTICS.md`: full-cohort electrode, position and slope findings (2026-10-21).
- `experiments/signal_quality/e003_voltage_feature_audit/`: reusable distribution, training-only scaling and covariance checks; see the [audit findings](Docs/TASK4_FEATURE_AUDIT.md) (2026-10-23).
- `experiments/waveform_decoding/e004_xdawn_decoding/`: fixed, training-fold xDAWN spatial features with matched sensor/pre-stimulus controls and repeated projected-feature audits. See its [run guide](experiments/waveform_decoding/e004_xdawn_decoding/README.md).
- `Docs/TASK4_XDAWN_RESULTS.md`: full-cohort xDAWN comparison and interpretation (2026-10-23).
- `experiments/waveform_decoding/e005_spectral_decoding/`: fixed single-trial alpha/beta power versus voltage and combined features, with matched validation, pre-trigger controls and repeated feature audits. See its [run guide](experiments/waveform_decoding/e005_spectral_decoding/README.md).
- `Docs/TASK4_SPECTRAL_RESULTS.md`: full-cohort spectral decoding comparison and interpretation (2026-10-23).
- `experiments/uncertainty_and_sensitivity/e009_response_uncertainty_sensitivity/`: participant-level uncertainty, earlier scale-specific time coverage and conditional recorded-background sensitivity. See its [run guide](experiments/uncertainty_and_sensitivity/e009_response_uncertainty_sensitivity/README.md).
- `Docs/TASK4_RESPONSE_FOLLOWUP.md`: aggregate findings and interpretation of the response-analysis refinements (2026-10-23).
- `Docs/task4_followup_research_note.pdf`: shareable research note consolidating the response, preprocessing, feature/decoding and sensitivity analyses (2026-10-26). Its LaTeX source project is kept locally and Git-ignored.
- `experiments/signal_quality/e010_simple_visual_inspection/`: matched minimal and fixed 0.5-30 Hz band-pass EEG views by waveform, electrode and presentation position, without supplied ICA. See the [visual inspection note](Docs/TASK4_VISUAL_INSPECTION.md) (2026-10-30).

## Experimental task numbering

| Experiment | Thesis label | Dataset/code label |
| --- | --- | --- |
| Threshold calculation | Task 1 | Task 1 |
| Additional task omitted from thesis | Not described | Task 2 |
| Superthreshold stimulation | Task 2 | Task 3 |
| Pairwise naturalness preference | Task 3 | Task 4 |

Waveform identity and chosen-versus-unchosen preference are separate targets. The thesis's reported preference classifier accuracy does not represent five-class waveform decoding accuracy.

## Current status

The original PDFs are preserved for provenance. The legacy exported code has been reviewed and its ramp labels corrected using Jason's clarification supplied on 2026-10-13. The original source-localization pipeline has not been validated end to end in this workspace. The separate minimal Task 4 baseline in `experiments/waveform_decoding/e001_voltage_baseline/` runs directly from XDF and the behavioral CSV without MNE or ICA; it did not demonstrate reliable above-chance decoding. See the [baseline findings](Docs/BASELINE_RESULTS.md).

The response diagnostic experiment has also run on all ten current-protocol participants using an adaptation of the supplied full-rate ICA preprocessing. It retained 1,242 events in 621 pairs. No waveform, position or interaction effect survived the planned correction; waveform-specific slope patterns were weakly repeatable. These exploratory associations do not establish absence of decodable information or eliminate artifacts and position confounds. See the [diagnostic findings](Docs/TASK4_RESPONSE_DIAGNOSTICS.md) and experiment run guide for methods, limits and local output locations.

The confirmed Task 4 mapping is Sine=239, Square=240, Triangle=241, LinDec=242, LinInc=243; CSV numbers 4 and 5 mean LinDec and LinInc respectively. Task 3 ramp codes are LinDec=233-235 and LinInc=236-238. Task 4 uses subjects 3-12 (subject 2 had an incompatible earlier protocol) and a race-to-five preference rule.

Within-pair stimulation is approximately 1.4 seconds apart; Jason recommends a maximum 1.2-second post-trigger epoch. The implemented primary baseline uses 50-500 ms; longer windows beyond 500 ms remain future secondary analyses. See [experiment clarifications](Docs/EXPERIMENT_CLARIFICATIONS.md).

The new baseline accepts local dataset/CSV paths, debounces stable runs, and requires exact full-sequence CSV alignment. The legacy exported loader still detects transitions without debouncing. The baseline uses `T4 Naturalness Value` for responses and ignores zero-filled legacy `T4 Waveform Detected` for validity. Jason confirms saved current-monitor recordings exist on the collection laptop; their availability and physical trigger-to-current timing remain follow-up work. Raw data and per-participant results remain excluded from Git.

The baseline environment is recorded in `requirements-baseline-lock.txt`. The response diagnostics add MNE and Matplotlib through `requirements-diagnostics.txt`, with the extended environment recorded in `requirements-diagnostics-lock.txt`. Legacy optional visualization helpers may also need imageio; source localization requires MNE fsaverage assets. Treat the original scripts' working-directory assumptions as part of setup.

## Data handling

This repository contains code and project documentation. Participant EEG recordings, behavioral CSV records, hand-map images, archives, derived outputs, and the personal training certificate are excluded by `.gitignore`. The accompanying dataset is held separately under the project's approved access arrangements.

In the original local workspace, recordings reside in `EEGData/SubjectN_data/`. The exported scripts generally expect `EEG_Practice/SubjectN_data/`; configure paths before running rather than uploading the dataset to GitHub.

## Attribution and research context

The supplied experimental documentation and analysis export are by Jason Whitson, for research supervised by Prof. Mohsen Rakhshan at the University of Central Florida. Repository setup and the initial project review are by Satyam Awasthi.

Motivation: Collu, Earley, Barbaro, and Ortiz-Catalan (2023), *Non-rectangular neurostimulation waveforms elicit varied sensation quality and perceptive fields on the hand*, Scientific Reports 13, 1588. DOI: https://doi.org/10.1038/s41598-023-28594-0 (CC BY 4.0).

Thesis: Whitson (2026), *Beyond the Square Pulse: Waveform Shape, EEG Correlates, and the Pursuit of Natural Sensation in TENS*. https://stars.library.ucf.edu/hut2024/661/

No software license has been assigned to the supplied code; the rights to third-party materials remain with their respective authors.
