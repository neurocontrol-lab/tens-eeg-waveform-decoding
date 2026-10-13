# Collaboration and project review

Reviewed 9 October 2026. Scope: email attachment, motivation paper, procedures, thesis, exported Python scripts and READMEs, dataset inventory, and a sample XDF header. No preprocessing or model training was run, and no analysis dependencies were installed.

Updated 13 October 2026 with Jason Whitson's clarification supplied in the email thread. His trigger table resolves the ramp identity, Task 4 stopping rule, and subject 2 exclusion. Full confirmed protocol, planned baseline, and remaining audit questions: [experiment clarifications](Docs/EXPERIMENT_CLARIFICATIONS.md).

## Objective

Prof. Mohsen Rakhshan's latest email (25 September) sets the immediate goal: classify the delivered stimulation waveform from EEG. The wider goal is understanding which stimulation conditions produce more natural-feeling sensations for sensory restoration. The thread records completed CIRTification training and approval of the user's addition to the team on 24 September.

Distinguish five-class waveform identity, pairwise naturalness preference, and reported sensation type/intensity/location. These are different targets.

## Motivation

Collu et al. (Scientific Reports, 2023) studied 11 healthy participants using square, sine, centered triangle, increasing ramp, and decreasing ramp charge-balanced stimulation of the median nerve. It measured detection thresholds, percept locations/qualities, and match-to-sample discrimination. Triangle versus either ramp was discriminated at a median 70%; the two ramps were at 50%, despite charge and peak-current matching. Chronaxie differed for some shapes, while rheobase and total percept area differences were not significant. Shape is therefore a plausible information-bearing stimulation parameter. The paper did not test EEG decoding or establish that nonrectangular shapes invariably feel more natural.

Source: Docs/Non-rectangular neurostimulation waveforms elicit varied sensation quality and perceptive fields on the hand.pdf

## Experiment

The procedures note explicitly corrects the thesis numbering:

| Experiment | Thesis | Data/code |
| --- | --- | --- |
| Thresholds | Task 1 | Task 1 |
| Additional experiment omitted from thesis | Not described | Task 2 |
| Superthreshold stimulation | Task 2 | Task 3 |
| Pairwise preference | Task 3 | Task 4 |

The thesis describes 11 healthy participants (nine male, two female), left median nerve stimulation, a Measurement Computing DAQ and STMISOLA stimulator, and 64-channel SAGA EEG at 1,000 Hz. Cathodic duration is 400 microseconds, with a 100 microsecond interphase delay and charge-balancing rectangular anodic phase. Stimulation is described as single-pulse: the 1,250 Hz statement concerns generation of a 400 microsecond half-cycle, not a pulse train repetition rate.

Thresholds use a one-up/two-down staircase, 250 microamp steps, and five reversals. Task 3 uses threshold, threshold +250 microamps, and threshold +500 microamps, nominally ten repetitions per combination: 150 trials total per participant. Two subjects received extra amplitude levels for particular waveforms, requiring exclusions from relevant amplitude-specific analyses.

Task 4 presents all ten waveform pairs at threshold with randomized order and asks which felt more natural. This is a preference task, unlike Collu's match-to-sample discrimination. Jason confirms race to five: a comparison ends when either waveform has five preference selections, resolving the thesis's ten-selection statement. Subject 2 used an incompatible older Task 4 protocol and is excluded from Task 4, not automatically from Task 3. Within-pair stimulation is approximately 1.4 seconds apart; Jason recommends at most 1.2 seconds post-trigger. Another thesis section says 150 epochs per waveform, conflicting with 150 total Task 3 trials in the design; this count discrepancy remains unresolved.

Sources: EEG_Codes_Export/Procedures of stimulation project.pdf; EEG_Codes_Export/UG Thesis - Beyond the Square Pulse_ Waveform Shape EEG Correlates and the.pdf

## Existing results

The thesis reports differences between waveform-averaged EEG responses near N140, P200, and P300, especially in contralateral centroparietal/parietal electrodes. Sine and square generally evoke larger responses than the triangle/ramp group. Charge, intensity, and percept area also differ, so these comparisons do not isolate waveform shape.

The thesis classifier predicts chosen versus unchosen preference from paired sum/difference features. It describes 903 base LDA classifiers and an L1 logistic regression meta-learner, with ten subjects grouped into five cross-validation folds. Reported mean balanced accuracy is 77.3% across six seeds; averaging seed predictions yields 83.6%. These are not five-class waveform results and have not been reproduced here. Audit the described OOF tuning and subject-wide normalization before interpreting them as fully independent future-subject performance. The export lacks the training and feature-extraction scripts referenced by this analysis.

## Pipeline

1. Stage 1's main entry point is August28_Task4_Preprocess_AutoICA33.py. It loads Task 4 for subjects 3–12, converts microvolts to volts, high-passes at 0.5 Hz using order-94 Chebyshev-II, applies 60 Hz harmonic notches, drops CREF/M1/M2, average-references, and fits 33-component extended Infomax ICA on a 1 Hz high-pass copy. It detects EOG components using Fp1 with a calibrated threshold; muscle exclusion is optional and was not used for the final README-described results. After ICA it interpolates approximately -2 to +10 ms around events, low-passes at 190 Hz, and saves raw/ICA/epochs plus summaries and component figures. Epochs cover -250 to +500 ms with a -250 to 0 ms baseline.
2. Stage 2 has waveform ANOVA and chosen/unchosen source-analysis drivers, both supporting autoeog33. Shared helpers use standard electrode positions and an fsaverage template forward/BEM model with dSPM, sLORETA, or eLORETA estimates, then six sensorimotor or 68 Desikan-Killiany ROIs. Statistics use per-subject averages and 50 ms windows stepped by 25 ms. These scripts do not train a waveform classifier.
3. Stage 3's Task4_PlotROIEpochs.py reads saved epochs, selects subjects/channels/bands, and plots waveform averages with between-subject confidence bands. It can follow stage 1 directly.

Older Task 3 and general helper preprocessing variants differ from the main Task 4 branch. The two copies of eeg_waveform_analysis_with_save.py are identical, as are the two task4_source_loc.py copies.

## Dataset

EEGData contains eleven subject directories (2–12), 44 XDF files, six CSVs, and 2,517 hand-map PNGs: 2,567 files, approximately 16.53 GB / 15.40 GiB excluding root ZIP archives. Every subject has a Task 4 recording. Subject 2 uses Subject2_Task4.xdf rather than the later standard name. Subject 9 Task 3 and subject 6 Task 1 have split recordings; some early names are nonstandard.

Master_Data_ToSub12 (1).csv in stage 2 contains subject/task/trial IDs, amplitude, waveform, sensation ratings, and Task 4 pair/detection/naturalness fields. It includes IDs beyond available EEG subjects, including 1 and 200. Subjects 3–12 have 623 Task 4 behavioral rows with naturalness values 1 or 2; subject 2 has 69 and one value of 0. These are behavioral row counts, not verified matched EEG epoch counts.

A sample subject 3 Task 4 XDF header confirms a SAGA EEG stream at 1,000 Hz and microvolt EEG units. Its 71 total channels include EEG/reference channels, accelerometers, TRIGGERS, STATUS, and COUNTER. The main loader reads EEG channels and embedded triggers.

## Issues before decoding

- Ramp identity resolved by Jason's authoritative table: Task 4 LinDec=242, LinInc=243; Task 3 LinDec=233-235, LinInc=236-238; CSV waveform 4=LinDec, 5=LinInc. The main preprocessing and both legacy helper copies are now corrected to match downstream analysis. Preserve numeric codes and audit stale names in prior outputs/caches before reuse.
- Data paths differ: stage 1 expects EEG_Practice/SubjectN_data while this workspace uses EEGData/SubjectN_data. Later stages expect output folders in their working directory. Preference analysis expects Master_Data_ToSub12.csv, but the supplied file has ' (1)' in its name. README1's CLI typo is now corrected to --refit.
- Jason recommends stable-trigger debouncing and expected-sequence checks. The current loader detects transitions without a minimum stable duration. Inspect run lengths before implementing the duration criterion; log rejected/errant events and keep original onset samples.
- The thesis documents triggers sent at waveform start and a CBLCFMA current-monitor path into DAQ analog input. Saved monitor recordings, clock alignment, and measured trigger-to-current delay/jitter remain unconfirmed. Acquisition code is not a prerequisite for the first trigger-aligned baseline.
- Preference alignment uses sequential forward scanning of trigger pairs. Missing/rejected/extra events can match later occurrences and silently misalign labels. Preserve original event IDs and audit all matches and omissions.
- Main preprocessing omits explicit mastoid referencing and bad-channel detection/interpolation described in the thesis/older pipeline. Fp1 calibration targets 10–20 blink events/minute and needs visual validation. Source drivers assume rank from ICA component count/exclusions; verify this against reconstructed data before accepting it.
- Sliding-window significance is exploratory and uncorrected across windows/ROIs. Within-window Holm corrections do not address the full search. Cluster tests exist but are skipped by default for all-ROI analysis. Do not select classifier ROIs on the full dataset and then treat held-out scores as independent.
- Bundled Python lacks MNE, pyxdf, and scipy; no dependency manifest is supplied. Source localization also needs fsaverage assets. The analysis has not been executed.

## Next analysis

Build an event-level manifest first: subject, task, file, original sample/timestamp, numeric trigger, verified waveform, amplitude/threshold tier, behavioral trial/pair identifier, and quality flags. Validate trigger counts/polarity, split files, units, and behavior alignment.

Then establish five-class single-trial sensor-space baselines. Task 4 supplies threshold-level trials; Task 3 provides amplitude-stratified complementary analysis. Keep within-subject and held-out-subject evaluations separate, with pair/block grouping where needed and training-fold-only feature selection, scaling, and tuning. Report balanced accuracy, macro-F1, confusion matrices, and per-subject results, with 20% balanced five-class reference and structure-preserving permutations. Use pre-stimulus controls and post-stimulus window comparisons to detect block/order confounds and residual stimulation/frontal artifacts. Quantify amplitude/charge dependence. Waveform-label decoding alone does not establish shape-specific cortical or naturalness decoding.

The proposed first Task 4 experiment uses subjects 3-12, fixed-window mean-amplitude features from 50-500 ms, and a simple sensor-space classifier. Compare first/second presentations and preceding-waveform effects; retain pair/comparison grouping in folds. Predefine longer-window secondary analyses ending at 800 ms and at most 1.2 seconds after checking actual event gaps. Existing saved +500 ms epochs need regeneration for those analyses. Shorter windows do not eliminate physiological carryover; at a 1.4-second gap, a first epoch ending at 1.2 seconds shares about 50 ms with the second epoch's -250 ms baseline. The classifier and debounced manifest have not been implemented yet.

## Questions for Prof. Rakhshan / Jason

Resolved: waveform mapping and CSV numbering, subject 2's Task 4 incompatibility, and race-to-five stopping. Jason endorsed the proposed waveform-decoding goal. The current plan evaluates within participants first, then unseen participants.

1. Was the current-monitor input saved, where are those recordings or delivery logs, and how were they aligned with SAGA EEG? Was trigger-to-current timing checked?
2. For Tasks 3 and 4, were failed, interrupted, or repeated trials retained/marked or removed before sharing? Do not assume that only successful trials were retained.
3. How were Task 4 missed responses recorded and handled in race-to-five counts? Naturalness choices are preferences, not correctness labels; the separate detected-waveform field's meaning remains to be confirmed.
4. What is omitted Task 2, and how should the inconsistent Task 3 epoch-count statement be interpreted?
5. Are the historical feature-extraction, classifier, and diagnostic/report files available for reproducing preference results?
