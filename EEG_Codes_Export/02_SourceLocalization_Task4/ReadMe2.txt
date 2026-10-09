Source Localization via Preprocessed EEG data - Jason Whitson 09/16/26

The following details how to go from preprocessed EEG data to source localization using the codes provided. To do this, you must have generated preprocessed data from the last step (step 1) to use as inputs to these codes. The preprocessed data must be in the working directory of the codes mentioned below, as well as the Master data file. The master data file will be included in this zip. If you do not have it, it can be found in the projects one drive.

The two main codes used to perform source localization are "August27_Task4_ROI_ChosenUnchosen_Auto" and "August27_Task4_ROI_ANOVA_Auto". The names are pretty self explanatory; the Chosen Unchosen code will categorize the data as chosen or unchosen epochs, and the ROI ANOVA will categorize the data per waveform type. To run these codes, the following commands should be used in the terminal of the directory they are located within:

python August27_Task4_ROI_ChosenUnchosen_Auto.py --variant autoeog33
python August27_Task4_ROI_ANOVA_Auto.py --variant autoeog33 --rois all --method eLORETA

Two methods can be used, ELORETA or dSPM. If no method is specified, then both will be generated. The variant is locked to auroeog33, as that is the preprocessing method that I chose as final, and represents the only method of preprocessing provided in the stage one files. The other two python files provide crucial functions that the two working codes call. These codes are quite resource intensive, so ensure you run these when you have a decent amount of time on your hands to wait on the output.

The output ROIs will be shaded when an ANOVA analysis signifies significant differences (p < 0.05) between the classes in its specific window.