Preprocessing XDF files into workable data - Jason Whitson 09/16/26

Code descriptions:

August28_Task4_Preprocess_AutoICA33.py

This is the main script for preprocessing . The first run of this code will take a considerable amount of time, as it will compute the ICA. Subsequent runs (unlikely to be required) will be quicker as the per participant ICA will be cached to a file. To run this script, you must have the participant xdf files within a folder titled EEG_Practice in the same directory as the code, as well as the two supporting codes mentioned below. These supporting codes provide legacy functions to this code (probably pretty inefficient and could be consolidated to a singular script, but it works). The xdf files should be used with the following file convention.

EEG_Practice/Subject{3..12}_data/EEGDATA_SUB{N}-T4.xdf

To clarify, relative to the your working directory, the xdf files should be within a folder titled EEG_Practice, then within a subfolder titled Subject{X}_data, with the .xdf titled EEGDATA_SUB{N}-T4. The code can be run using teminal commands of the following methods

  python August28_Task4_Preprocess_AutoICA33.py
  python August28_Task4_Preprocess_AutoICA33.py --muscle
  python August28_Task4_Preprocess_AutoICA33.py --subjects 3,4
  python August28_Task4_Preprocess_AutoICA33.py --refi

Muscle filtering is available for use, but was not used for the final results shown to Dr. Rakhshan due to their tendancy to remove a large number of ICAs, thus weakening the overall neural signal that was used for analysis. Once this script is ran for all subjects, you can move onto the second batch of codes to perform source localization.

The rest of the codes within this directory are complimentary codes that provide crucial functions to the main two codes.

If you have any questions, please reach out!
jasonwh2003@gmail.com
321-289-0538