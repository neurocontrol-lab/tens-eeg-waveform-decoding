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

## Experimental task numbering

| Experiment | Thesis label | Dataset/code label |
| --- | --- | --- |
| Threshold calculation | Task 1 | Task 1 |
| Additional task omitted from thesis | Not described | Task 2 |
| Superthreshold stimulation | Task 2 | Task 3 |
| Pairwise naturalness preference | Task 3 | Task 4 |

Waveform identity and chosen-versus-unchosen preference are separate targets. The thesis's reported preference classifier accuracy does not represent five-class waveform decoding accuracy.

## Current status

The original export is preserved for provenance. It has been reviewed but has not been executed or validated in this workspace. A waveform-decoding training pipeline is not yet included.

Before execution, resolve the increasing/decreasing ramp trigger-name mismatch, configure local dataset/output paths, reconcile the supplied master CSV filename, and validate behavioral-to-EEG event alignment. Details are in `PROJECT_REVIEW.md` and the supplied stage READMEs.

The code imports NumPy, pandas, Matplotlib, MNE, pyxdf, and SciPy; some optional visualization helpers also use imageio. Dependency versions have not yet been pinned or tested. Source localization requires MNE fsaverage assets. Treat the original scripts' working-directory assumptions as part of setup.

## Data handling

This repository contains code and project documentation. Participant EEG recordings, behavioral CSV records, hand-map images, archives, derived outputs, and the personal training certificate are excluded by `.gitignore`. The accompanying dataset is held separately under the project's approved access arrangements.

In the original local workspace, recordings reside in `EEGData/SubjectN_data/`. The exported scripts generally expect `EEG_Practice/SubjectN_data/`; configure paths before running rather than uploading the dataset to GitHub.

## Attribution and research context

The supplied experimental documentation and analysis export are by Jason Whitson, for research supervised by Prof. Mohsen Rakhshan at the University of Central Florida. Repository setup and the initial project review are by Satyam Awasthi.

Motivation: Collu, Earley, Barbaro, and Ortiz-Catalan (2023), *Non-rectangular neurostimulation waveforms elicit varied sensation quality and perceptive fields on the hand*, Scientific Reports 13, 1588. DOI: https://doi.org/10.1038/s41598-023-28594-0 (CC BY 4.0).

Thesis: Whitson (2026), *Beyond the Square Pulse: Waveform Shape, EEG Correlates, and the Pursuit of Natural Sensation in TENS*. https://stars.library.ucf.edu/hut2024/661/

No software license has been assigned to the supplied code; the rights to third-party materials remain with their respective authors.
