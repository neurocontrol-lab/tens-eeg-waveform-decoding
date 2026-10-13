Preprocessing XDF files into workable data - Jason Whitson 09/16/26

Code descriptions:

August28_Task4_Preprocess_AutoICA33.py

This is the main script for preprocessing . The first run of this code will take a considerable amount of time, as it will compute the ICA. Subsequent runs (unlikely to be required) will be quicker as the per participant ICA will be cached to a file. To run this script, you must have the participant xdf files within a folder titled EEG_Practice in the same directory as the code, as well as the two supporting codes mentioned below. These supporting codes provide legacy functions to this code (probably pretty inefficient and could be consolidated to a singular script, but it works). The xdf files should be used with the following file convention.

EEG_Practice/Subject{3..12}_data/EEGDATA_SUB{N}-T4.xdf

To clarify, relative to the your working directory, the xdf files should be within a folder titled EEG_Practice, then within a subfolder titled Subject{X}_data, with the .xdf titled EEGDATA_SUB{N}-T4. The code can be run using teminal commands of the following methods

  python August28_Task4_Preprocess_AutoICA33.py
  python August28_Task4_Preprocess_AutoICA33.py --muscle
  python August28_Task4_Preprocess_AutoICA33.py --subjects 3,4
  python August28_Task4_Preprocess_AutoICA33.py --refit

Muscle filtering is available for use, but was not used for the final results shown to Dr. Rakhshan due to their tendancy to remove a large number of ICAs, thus weakening the overall neural signal that was used for analysis. Once this script is ran for all subjects, you can move onto the second batch of codes to perform source localization.

The rest of the codes within this directory are complimentary codes that provide crucial functions to the main two codes.

Project clarification update (2026-10-13; Jason's reply supplied in the email thread):
- Authoritative Task 4 mapping: Sine=239, Square=240, Triangle=241, LinDec=242, LinInc=243. CSV waveform 4=LinDec, 5=LinInc.
- Task 3 mapping: Sine=224-226, Square=227-229, Triangle=230-232, LinDec=233-235, LinInc=236-238; each triplet is threshold, +0.25 mA, +0.5 mA.
- Main and legacy helper mappings now agree. Audit old saved epoch names/caches before reuse; this update does not rewrite existing outputs.
- Subject 2 used an incompatible older Task 4 protocol. Task 4 subjects 3-12 remain the intended cohort; this exclusion does not apply automatically to Task 3.
- Task 4 is race to five, with approximately 1.4 seconds between stimuli within a pair. Post-trigger epochs must be at most 1.2 seconds per Jason's advice. Current epochs remain -250 to +500 ms; planned classifier features are 50-500 ms.
- The loader still detects trigger transitions without debouncing. Inspect stable-run durations and reject/log errant transitions before building the manifest; preserve accepted runs' original onset samples.
- Local recordings are under EEGData/, whereas this script expects EEG_Practice/ relative to its working directory. Configure those paths before running; participant data remain outside Git.
- Physical trigger-to-current delay, saved monitoring traces, and failed/missed-trial handling remain unconfirmed.
Full reference: ../../Docs/EXPERIMENT_CLARIFICATIONS.md

If you have any questions, please reach out!
jasonwh2003@gmail.com
321-289-0538
