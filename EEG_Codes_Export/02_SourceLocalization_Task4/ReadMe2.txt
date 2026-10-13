Source Localization via Preprocessed EEG data - Jason Whitson 09/16/26

The following details how to go from preprocessed EEG data to source localization using the codes provided. To do this, you must have generated preprocessed data from the last step (step 1) to use as inputs to these codes. The preprocessed data must be in the working directory of the codes mentioned below, as well as the Master data file. The master data file will be included in this zip. If you do not have it, it can be found in the projects one drive.

The two main codes used to perform source localization are "August27_Task4_ROI_ChosenUnchosen_Auto" and "August27_Task4_ROI_ANOVA_Auto". The names are pretty self explanatory; the Chosen Unchosen code will categorize the data as chosen or unchosen epochs, and the ROI ANOVA will categorize the data per waveform type. To run these codes, the following commands should be used in the terminal of the directory they are located within:

python August27_Task4_ROI_ChosenUnchosen_Auto.py --variant autoeog33
python August27_Task4_ROI_ANOVA_Auto.py --variant autoeog33 --rois all --method eLORETA

The implemented methods are dSPM, sLORETA, and eLORETA. One method runs per invocation; the default is dSPM. Use --variant autoeog33 for the final preprocessing supplied in stage one, although the CLI accepts other variants whose outputs may not be available. The supporting Python files provide crucial functions that the two working codes call. These codes are quite resource intensive, so allow time for the output.

The output ROIs will be shaded when an ANOVA analysis signifies significant differences (p < 0.05) between the classes in its specific window.

Project clarification update (2026-10-13):
- Jason confirms Sine=239, Square=240, Triangle=241, LinDec=242, LinInc=243. CSV waveform 4=LinDec and 5=LinInc; task4_source_loc.py already uses this mapping.
- Use subjects 3-12 for Task 4; subject 2 had an incompatible older protocol. A comparison ends when one waveform has five preference selections (race to five).
- Chosen/unchosen labels are relative naturalness preferences, not correct/incorrect responses or delivered-waveform identity. Missing-response and failed/repeated-trial handling are pending clarification.
- Audit sequential CSV-to-epoch pair matching for missing/extra events; do not infer that a matched pair is valid merely because its codes are expected.
- The supplied CSV is named Master_Data_ToSub12 (1).csv locally, while the driver expects Master_Data_ToSub12.csv. Resolve the local path/name before execution; do not commit behavioral CSVs.
- The implemented --method choices are dSPM, sLORETA, and eLORETA, with dSPM as the default; one method runs per invocation. autoeog33 is the supplied final preprocessing variant, although other variants are accepted by the CLI.
- Sliding-window significance is exploratory and uncorrected across windows/ROIs. These scripts do not implement the proposed single-epoch waveform classifier.
Timing, event-audit requirements, and the planned baseline: ../../Docs/EXPERIMENT_CLARIFICATIONS.md
