# WD-008 · Task 4 naturalness-choice decoding from paired EEG

The model predicts the participant's choice of the first or second stimulus as more natural.
One behavioral pair supplies one target. Participants 3–12 form the cohort.

## Behavioral audit and current decision

Further naturalness decoding is deferred at the user's request. The completed EEG fits are preserved.
The independent behavior-only audit covers 623 choices and 100 participant/comparison blocks.
Its exact conditional order-association test gives p=0.0000454475, favoring second-position choices.
The direct triangle/LinDec subset gives p=0.1107; removing terminal votes gives p=0.04649.
The latter two tests are exploratory sensitivities with unadjusted p-values.

The test conditions on waveform choices and observed order counts in each block. It assumes exchangeable
presentation order within a block under the null, following the documented randomization protocol.
The acquisition randomization code is unavailable. The audit identifies an order association without
establishing incorrect log entries, a physiological carryover mechanism or unusable second-presentation EEG.
A metadata model could quantify predictability, while high accuracy could also reflect consistent preferences.

For each participant/comparison block, the exact test holds waveform votes and presentation-order counts fixed.
If A is the reference waveform, X counts trials where A was first and chosen. X has a hypergeometric
null distribution. Convolution sums these distributions over 100 blocks. The observed total is 133,
against a conditional expectation of 151.6726. Summing both null tails at least 18.6726 from the expectation
gives p=0.0000454475. The report includes the exact null-distribution plot and its assumptions.

The research question originates from Jason's naturalness-consistency result. It asks whether EEG
predicts the participant's choice and whether post-stimulation EEG adds information beyond pre EEG.

The behavioral explorer includes all five waveforms. Select a waveform and either one comparison
waveform or all other waveforms to inspect first/second presentation rates and participant vote counts.
These descriptive views use the 623 valid behavioral choices. Missing position counts appear as gaps.

Run the independent audit with a fresh output directory.

```powershell
& ./.venv/Scripts/python.exe -m experiments.waveform_decoding.e021_naturalness_choice_decoding.behavior_audit
```

## What the completed run covers

| Item | Completed WD-008 analysis |
| --- | --- |
| Dataset | Task 4, 622 eligible pairs from participants 3–12, covering all five waveforms and all ten waveform comparisons |
| Behavioral labels | Jason's populated `T4 Naturalness Value`: 1 means the first stimulus was chosen as more natural, 2 means the second. The classifier encodes these as 0 and 1. |
| Validation | Leave one participant out, train on the other nine. Each participant's displayed score is from that participant's held-out trials. |
| Presentation handling | The main post-stimulation representation subtracts first-presentation EEG from second-presentation EEG. Standalone pre/post models use the first presentation. |
| Triangle versus linear decrease subset | 33 nominal-charge-matched pairs from participants 3, 4, 5, 6 and 10. These scores select predictions from the full-cohort naturalness models. |
| Within-participant naturalness decoders | Deferred following the behavioral audit. Existing fits use cross-participant models. |

The naturalness test predicts the participant's actual preference using Jason's behavioral log.
Further naturalness decoding is deferred following the behavioral audit.

Collu et al. reported 70% median correct match-to-sample discrimination for triangle versus
linear decrease under matched charge and peak current. That percentage measures the ability
to distinguish the two stimuli. Their result does not specify a 70% naturalness preference for
triangle over linear decrease. In the current matched Task 4 subset, linear decrease received
25 of 33 naturalness votes. [Collu et al.](https://doi.org/10.1038/s41598-023-28594-0)

[Results](../../../Docs/WD008_NATURALNESS_CHOICE_RESULTS.md) ·
[Interactive report](../../../Docs/Reports/WD-008/index.html)

Run from the repository root using a fresh output directory.

```powershell
& ./.venv/Scripts/python.exe -m experiments.waveform_decoding.e021_naturalness_choice_decoding
```

The default result directory is `output/wd008_naturalness_baselined_20261123`. Raw feature extraction
can resume from `output/wd008_feature_cache_baselined_20261123` after checking all recorded cache hashes.

## Features and validation

Six anatomically fixed contralateral electrodes are C4, CP2, CP4, CP6, P2 and P4.
Each contributes five 50 ms voltage means. First-pulse pre EEG spans −350 to −100 ms.
Post EEG spans 100 to 350 ms after each pulse. The paired post representation is
second minus first. Processing applies the 62-channel instantaneous average reference
and microvolt conversion. The same first-pulse baseline from −550 to −350 ms is subtracted
from both windows and both post responses. It cancels in the paired difference.
No temporal filtering or ICA is applied.

The initial unbaselined run at `output/wd008_naturalness_20261123` is preserved as a processing
reference. Participant-specific DC offsets reached approximately 20,000 µV and dominated the
standalone features. Baseline subtraction was added after inspecting those features and scores.
This preprocessing adaptation is exploratory and is disclosed in the reports.

Eight fixed models use the same eligible pairs and leave-one-participant-out folds.

- Training choice proportion.
- Metadata alone, using 20 ordered-waveform-pair indicators and chronological and within-block pair indices.
- Metadata plus first-pulse pre EEG.
- Metadata plus paired post EEG.
- Metadata plus first-pulse pre EEG and paired post EEG.
- First-pulse pre EEG alone.
- First-pulse post EEG alone.
- Paired post EEG alone.

Logistic regression uses fixed C=1 and training-only standardization, with no class weighting,
feature selection or held-out participant normalization. The primary comparison is the change
in participant-mean log loss when paired post EEG is added to metadata and pre EEG.
The standalone comparison tests first-pulse post versus first-pulse pre EEG with identical
feature counts. Balanced accuracy, accuracy, ROC AUC, Brier score and confusion counts are saved.
Intervals resample participants 10,000 times conditional on the fixed held-out predictions.

## Checks and secondary analyses

The loader verifies raw-recording hashes against the earlier visual audit and reconstructs
the full trigger-to-CSV alignment. It preserves WD-001 complete-pair eligibility, including the
fixed 500 µV peak-to-peak QC, and checks timestamps and neighboring stimuli within each new
feature window. The additional electrodes receive finite-value checks. Cache hashes, training
constant features and solver convergence are recorded.

Removing each block's terminal vote refits all models on the reduced cohort.
The varying-choice and nominal-charge-matched triangle/linear-decrease subsets summarize
held-out predictions from the full cohort. Charge is used only for secondary subset membership.
Varying-choice membership uses the observed choices and is descriptive.

The master CSV has no response timestamps. The 350 ms endpoint cannot be verified to precede
the behavioral response, and overlap with the previous response cannot be timed. Second-pulse
features may contain first-pulse carryover. No permutation significance test is run.
The 68.29% behavioral race-to-five winner share and 50% binary balanced-accuracy reference
describe different statistics.

The run saves a complete pair manifest, feature cache, input and code hashes, fold audits,
held-out predictions, participant and cohort metrics, a Markdown report and a standalone HTML report.
