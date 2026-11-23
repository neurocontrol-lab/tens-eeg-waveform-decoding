# WD-008. Task 4 naturalness-choice decoding

[Interactive report](Reports/WD-008/index.html) · [Run guide](../experiments/waveform_decoding/e021_naturalness_choice_decoding/README.md)

## Research question and origin

Jason Whitson reported consistent naturalness preferences, with an average winner share of 83.5% against the race-to-five null expectation of 68.29%. WD-008 asks whether EEG predicts the recorded first-or-second choice and whether post EEG improves on pre EEG and task metadata. [Whitson thesis](https://stars.library.ucf.edu/hut2024/661/)

## Behavioral order audit and current decision

**Further naturalness decoding is deferred.** Existing EEG results below are retained as exploratory results.

The behavior-only audit includes 623 choices from 10 participants and 100 comparison blocks. The first stimulus was selected 284 times and the second 339 times.

| Behavioral data | Trials | Exact conditional two-sided p |
| --- | ---: | ---: |
| All Task 4 comparison blocks | 623 | 4.54475e-05 |
| Direct triangle versus linear decrease | 63 | 0.110693 |
| Direct pairs with terminal votes removed; sensitivity analysis | 53 | 0.0464948 |

### How the p-value was calculated

We checked whether presentation order was associated with participants’ choices.

We kept each participant’s waveform preferences fixed and calculated what could happen if the presentation orders were rearranged within each comparison block, preserving how often each waveform appeared first.

Across **623 trials**, the first stimulus would be chosen about **321 times (51.6%)** on average under the null model. It was actually chosen **284 times (45.6%)**—about **37 fewer choices**, a difference of **6 percentage points**.

A difference this large, in either direction, occurs in about **1 out of 22,000** allowed order arrangements. This gives a **p-value of approximately 0.000045**.

This provides evidence that choices were associated with presentation order. The test alone cannot identify what caused that association.

### Statistical method and references

Exact conditional randomization test. Hold waveform choices and observed order counts fixed within each participant/comparison block. Convolve the block hypergeometric distributions and use the absolute deviation from the conditional expectation.
The implementation defines a two-sided result by its absolute distance from the conditional mean. Exact-test software can use other two-sided conventions. [Jung (2014), stratified exact testing](https://doi.org/10.1002/bimj.201300048) · [SciPy, fixed-margin hypergeometric calculation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html)

Presentation order is exchangeable within each recorded block under the null of no order effect. This follows the documented randomized-order protocol; acquisition randomization code is unavailable.

The primary result favors second-position choices after conditioning on participant and waveform comparison. The two direct-pair results are exploratory sensitivity checks with unadjusted p-values. Their different outcomes show sensitivity to the stopping-rule selection.

This audit supports order-dependent preferences. It does not establish incorrect behavioral entries, physiological carryover, or unusable second-presentation EEG. The log records each choice in its presentation context. A metadata classifier could measure predictability, while high accuracy can also reflect consistent waveform preferences.

**2026-11-23.** 622 eligible pairs from 10 participants.

The target is the participant’s first-or-second naturalness choice on each pair. Each pair receives one prediction from models trained on the other participants.

## What the completed run covers

| Item | Completed WD-008 analysis |
| --- | --- |
| Dataset | Task 4, 622 eligible pairs from 10 participants; all five waveforms and all ten waveform comparisons |
| Labels | T4 Naturalness Value from Jason’s behavioral log: 1 = first stimulus chosen, 2 = second stimulus chosen. The classifier encodes these as 0 and 1. |
| Validation | Train on nine participants and test on the remaining participant. The participant selector displays held-out participant scores. |
| Presentations | Main paired post EEG is second minus first. Standalone pre/post models use the first presentation. |
| Triangle/linear-decrease subset | 33 matched-charge pairs from participants 3, 4, 5, 6 and 10; scores reuse the full-cohort naturalness-model predictions. |
| Requested within-participant naturalness decoders | Deferred following the behavioral order audit |

The naturalness follow-up is deferred following the behavioral order audit.

Collu et al. reported 70% median match-to-sample discrimination accuracy for triangle versus linear decrease under matched stimulation settings. This measures correct distinction of the stimuli. A directional naturalness preference requires a separate preference judgment. [Collu et al.](https://doi.org/10.1038/s41598-023-28594-0)

## Held-out results

Scores average participants equally. Log loss measures probability prediction error; lower is better.

| Model | Balanced accuracy | Accuracy | Log loss | ROC AUC |
| --- | ---: | ---: | ---: | ---: |
| Training choice proportion | 50.00% | 54.04% | 0.6936 | 0.500 |
| Waveform identities and trial order | 46.91% | 47.22% | 0.7464 | 0.460 |
| Metadata + pre EEG | 45.89% | 46.84% | 0.7846 | 0.450 |
| Metadata + paired post EEG | 47.89% | 47.46% | 0.7786 | 0.483 |
| Metadata + pre + paired post EEG | 47.52% | 48.05% | 0.8606 | 0.454 |
| First-pulse pre EEG alone | 48.48% | 49.78% | 0.7315 | 0.457 |
| First-pulse post EEG alone | 49.55% | 51.02% | 0.7499 | 0.469 |
| Paired post EEG alone | 53.98% | 53.59% | 0.7177 | 0.528 |

## Paired comparisons

Positive changes favor the model with post-stimulus EEG. Intervals resample participants 10,000 times using fixed held-out predictions and are descriptive.

| Analysis | Log-loss reduction [95% interval] | Balanced-accuracy change, percentage points [95% interval] |
| --- | ---: | ---: |
| All eligible pairs: Add paired post EEG to metadata and pre EEG | -0.0760 [-0.1604, -0.0113] | +1.63 [-1.30, +4.56] |
| All eligible pairs: First-pulse post EEG versus first-pulse pre EEG | -0.0185 [-0.0710, +0.0295] | +1.08 [-3.32, +5.84] |
| Terminal votes removed and models refitted: Add paired post EEG to metadata and pre EEG | -0.0639 [-0.1556, +0.0079] | +1.33 [-1.22, +4.01] |
| Terminal votes removed and models refitted: First-pulse post EEG versus first-pulse pre EEG | +0.0081 [-0.0301, +0.0442] | +3.96 [+0.10, +8.38] |

## Secondary subsets

Varying-choice and charge-matched subsets summarize the full-cohort held-out predictions; they are not separate model fits. Nonterminal models are refitted on the reduced training data.

| Subset | Pairs | Participants | Metadata BA | Metadata + pre BA | Metadata + pre + paired post BA | Pre alone BA | Post alone BA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Same ordered pair with varying choices | 310 | 10 | 50.46% | 44.19% | 43.62% | 47.50% | 46.02% |
| Nominal-charge-matched triangle versus linear decrease | 33 | 5 | 75.33% | 65.00% | 67.83% | 47.17% | 45.17% |

Linear decrease received 25 of 33 votes in the matched subset. This is an observed vote share; decoder scores predict first-versus-second choices.

## Methods and checks

- Six fixed contralateral electrodes: C4, CP2, CP4, CP6, P2 and P4.
- Five 50 ms means per electrode, giving 30 EEG features per window.
- First-pulse pre EEG spans −350 to −100 ms; first- and second-pulse post EEG span 100 to 350 ms. The paired representation is second minus first post EEG.
- Minimal processing uses a 62-channel instantaneous average reference and microvolt conversion. The same first-pulse baseline from −550 to −350 ms is subtracted from both windows and both post responses. It cancels in the paired difference. No temporal filtering or ICA is applied.
- Metadata consists of 20 fixed ordered-pair indicators and chronological and within-block pair indices.
- Logistic regression uses fixed C=1 and training-only standardization. No label-based feature selection, class weighting or held-out participant normalization is used.
- Identical leave-one-participant-out folds and eligible pairs are used for all eight models.
- Existing complete-pair QC and verified XDF-to-CSV alignment are reused. Raw-input hashes, feature-cache hashes, feature counts, constant-feature counts and convergence are saved.

## Interpretation and timing

- The standalone comparison uses the first pulse in both periods. The primary paired post EEG also contains the second response and can include carryover from the first.
- The CSV contains choices but no response timestamps. A 350 ms post-trigger endpoint cannot be verified as preceding the behavioral response. Overlap with the previous response is also unverified.
- Nominal charge determines secondary subset membership only and is excluded from model predictors.
- Subsets selected for varying observed choices are descriptive. Terminal-vote removal is a sensitivity check for the stopping rule; the reduced cohort retains the history of the original task.
- The 68.29% expected winner share describes the race-to-five behavioral statistic. Balanced accuracy has a 50% binary reference. A useful EEG improvement must also be compared with the metadata and pre-EEG models.
- Bootstrap intervals condition on the fitted cross-validation models. No permutation significance test was run, and these summaries do not establish an artifact-free neural mechanism.

## Processing adaptation

Large participant-specific DC offsets were identified after inspecting the initial run. A shared causal baseline was added as a documented preprocessing adaptation.
The initial run is retained locally. Its uncorrected DC offsets reached approximately 20,000 µV. The adaptation followed inspection of the initial features and scores and is exploratory.

| Unbaselined model | Balanced accuracy | Log loss |
| --- | ---: | ---: |
| First-pulse pre EEG alone | 49.86% | 0.9551 |
| First-pulse post EEG alone | 49.86% | 0.9535 |
| Metadata + pre EEG | 49.24% | 1.0513 |
| Metadata + pre + paired post EEG | 48.47% | 1.1912 |
