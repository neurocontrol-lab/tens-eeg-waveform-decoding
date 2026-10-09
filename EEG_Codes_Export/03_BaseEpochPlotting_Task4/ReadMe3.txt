Base epoch plotting Task 4 user guide - Jason Whitson 09/17/26

One you have preprocessed the xdf files to obtain the .fif files, you can begin plotting the epochs. The data should be in its original file from part 1 in the working directory of these codes. The main script within this section is "Task4_PlotROIEpochs". This code requires the following inputs: subjects, electrodes, and the frequency band. The confidence interval may also be specified, as well as the waveforms you want to have included in analysis (defaults as all). To change the other features on the plot itself (Colors, labeling, etc), you must go directly into the code itself. Example usages of this code are as follows.

    python Task4_PlotROIEpochs.py --variant autoeog33
    python Task4_PlotROIEpochs.py --variant autoeog33 --subjects 3,4,5 --electrodes C3,C4,Cz --band alpha
    python Task4_PlotROIEpochs.py --variant autoeog33 --electrodes central --band beta
    python Task4_PlotROIEpochs.py --variant autoeog33muscle --electrodes parietal --waveforms Sine,LinInc

It may become easier for a larger analysis to go through various electrode groups and output those plots to a folder. This can be added to the script simply. The other python files here are helper files for the code to run.