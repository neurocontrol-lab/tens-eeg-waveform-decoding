Base epoch plotting Task 4 user guide - Jason Whitson 09/17/26

One you have preprocessed the xdf files to obtain the .fif files, you can begin plotting the epochs. The data should be in its original file from part 1 in the working directory of these codes. The main script within this section is "Task4_PlotROIEpochs". This code requires the following inputs: subjects, electrodes, and the frequency band. The confidence interval may also be specified, as well as the waveforms you want to have included in analysis (defaults as all). To change the other features on the plot itself (Colors, labeling, etc), you must go directly into the code itself. Example usages of this code are as follows.

    python Task4_PlotROIEpochs.py --variant autoeog33
    python Task4_PlotROIEpochs.py --variant autoeog33 --subjects 3,4,5 --electrodes C3,C4,Cz --band alpha
    python Task4_PlotROIEpochs.py --variant autoeog33 --electrodes central --band beta
    python Task4_PlotROIEpochs.py --variant autoeog33muscle --electrodes parietal --waveforms Sine,LinInc

It may become easier for a larger analysis to go through various electrode groups and output those plots to a folder. This can be added to the script simply. The other python files here are helper files for the code to run.

Project clarification update (2026-10-13):
- Jason's authoritative mapping is Sine=239, Square=240, Triangle=241, LinDec=242, LinInc=243. CSV waveform 4=LinDec and 5=LinInc. The plotting helper already uses this mapping; its legacy helper constants have now been corrected too.
- Task 4 subjects are 3-12; subject 2 used an incompatible earlier Task 4 protocol. Preference comparisons use race to five.
- Stimuli within a pair are approximately 1.4 seconds apart. Jason recommends at most 1.2 seconds post-trigger; check actual intervals rather than assuming a fixed gap.
- Existing main preprocessing saves only through +500 ms. The first planned decoding window is 50-500 ms; longer windows require regenerated epochs from continuous data. Shorter windows do not prove the absence of carryover or artifacts.
- Plotting produces averaged responses, not single-epoch classifier performance. Keep paired presentations together in validation folds when implementing the classifier.
Full reference: ../../Docs/EXPERIMENT_CLARIFICATIONS.md
