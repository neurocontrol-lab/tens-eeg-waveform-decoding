# TENS EEG Waveform Decoding

Investigating how peripheral stimulation waveforms relate to EEG responses and perceived sensation, toward more natural sensory feedback for bidirectional interfaces. The decoding target is one of five delivered waveforms: sine, square, centered triangle, increasing ramp, or decreasing ramp. Waveform identity and subjective naturalness preference are separate research questions.

## Explore the results

**[Open the interactive EEG report](https://htmlpreview.github.io/?https://github.com/neurocontrol-lab/tens-eeg-waveform-decoding/blob/main/Docs/Reports/E014/index.html)**

Explore waveform repeatability by participant, first/second presentation, electrode or learned component, feature representation, and time bin. The current report covers all 62 scalp electrodes, voltage and derivative features, alpha/beta power, and independently calibrated xDAWN components. It characterizes responses; repeatability alone does not establish waveform decoding or neural origin.

[Static image gallery](https://htmlpreview.github.io/?https://github.com/neurocontrol-lab/tens-eeg-waveform-decoding/blob/main/Docs/Reports/E014/gallery.html) · [Findings and interpretation](Docs/E014_REPEATABILITY_RESULTS.md) · [Report index and local viewing](Docs/Reports/README.md)

Interactive reports are the main entry point for inspecting individual response patterns. The [experiment registry](experiments/README.md) links analysis-specific findings, methods, run guides, and status.

Latest decoding: [E016 binary waveform detectors](Docs/E016_BINARY_RESULTS.md)
compares 100 position-specific waveform-versus-rest detectors on the fixed E015
top-100 features, with 55 strict-reference detectors and an audit of AUC aggregation.

**Decoding reports:** [E015 multiclass and confusion matrices](https://htmlpreview.github.io/?https://github.com/neurocontrol-lab/tens-eeg-waveform-decoding/blob/main/Docs/Reports/E015/index.html)
· [E016 binary detectors](https://htmlpreview.github.io/?https://github.com/neurocontrol-lab/tens-eeg-waveform-decoding/blob/main/Docs/Reports/E016/index.html)
· [Waveform and participant comparison](Docs/E015_E016_SUBGROUP_FINDINGS.md).
Both decoding reports are self-contained pages smaller than 1 MB.

## Research areas

| Area | Main question |
| --- | --- |
| [Signal quality and preprocessing](experiments/signal_quality/README.md) | Are recordings and features suitable for analysis, and how do processing choices affect them? |
| [Response characterization](experiments/response_characterization/README.md) | How do stimulus-aligned responses vary across waveforms, electrodes, time, and presentation position? |
| [Waveform decoding](experiments/waveform_decoding/README.md) | Can single-epoch EEG predict the delivered waveform on held-out data? |
| [Uncertainty and sensitivity](experiments/uncertainty_and_sensitivity/README.md) | How uncertain are the results, and which reference effects can the analyses recover? |

## Methods and setup

- [Project review](PROJECT_REVIEW.md): dataset overview, experimental background, and supplied-code review.
- [Protocol clarifications](Docs/EXPERIMENT_CLARIFICATIONS.md): task numbering, corrected trigger labels, timing, and behavioral fields.
- [Experiment registry and run guides](experiments/README.md): reproducible commands, dependencies, validation, and limitations for each analysis.
- [Research note](Docs/task4_followup_research_note.pdf): consolidated response, preprocessing, decoding, and sensitivity findings through 26 October 2026.

Current Task 4 analyses use participants 3–12 and trigger-aligned epochs. Physical trigger-to-current timing remains unverified. The original study materials and legacy analysis export are retained in `EEG_Codes_Export/`; each experiment documents its own preprocessing and evaluation.

## Data and sharing

Raw EEG, behavioral records, fitted filters, caches, and full numerical outputs remain local and Git-ignored. Shared image reports contain pre-rendered responses and small summaries. Keep each report's `images/` directory beside its HTML when downloading or hosting it. Dataset access follows the collaboration's approved arrangements.

## Attribution

Experimental acquisition and supplied study materials: Jason Whitson, supervised by Prof. Mohsen Rakhshan at the University of Central Florida. Research analysis in this repository: Satyam Awasthi.

Motivation: R. Collu et al. (2023), [*Non-rectangular neurostimulation waveforms elicit varied sensation quality and perceptive fields on the hand*](https://doi.org/10.1038/s41598-023-28594-0). Experimental foundation: [Whitson's thesis (2026)](https://stars.library.ucf.edu/hut2024/661/).

No software license has been assigned to the supplied code; third-party materials remain subject to their authors' rights.
