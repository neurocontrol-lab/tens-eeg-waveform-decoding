# WD-008. Participant-specific naturalness-choice EEG decoding

[Interactive report](Reports/WD-008/index.html) · [Run guide](../experiments/waveform_decoding/e021_naturalness_choice_decoding/README.md)

**Analysis: 24 November 2026.** Predict whether an individual presentation was chosen as more natural, using Jason’s T4 Naturalness Value. Separate models are fitted for each participant and presentation position. The previous leave-one-participant-out analysis is archived; it is not the current validation design.

622 eligible behavioral pairs, 1244 EEG epochs and 10 participants (3–12). Supported analysis/position/resolution/block folds: 800; unsupported folds: 0.

## Question and design

Does post-stimulation EEG predict the recorded choice beyond waveform identities, trial order and pre-stimulation EEG? The label is chosen = 1 and not chosen = 0 for the current presentation. Each pair contributes one epoch to each position-specific dataset, with complementary labels. Responses are never subtracted or pooled across positions.

- Hold out the entire waveform comparison, including any repeated blocks of that comparison, within a participant and position. No participant contributes training data to another participant’s model. Single-class test blocks are retained; BA is computed from all held-out predictions pooled within each participant/position.
- All 62 electrodes; fixed 50/100 ms voltage means over post 50–1,050 ms and pre −1,050 to −50 ms. These provide 1,240/620 original features per period.
- Instantaneous average reference and a common pair baseline −1,250 to −1,050 ms before the first pulse; no temporal filtering or ICA. The archived complete-pair QC is retained, with finite/timestamp checks on all channels.
- Training-only standardization, separate pre/post PCA retaining 90% variance, then training-only standardization of components. No supervised feature selection or tuning. Logistic regression uses L2 regularization, C=1 and no class weighting.
- Metadata: five current-waveform and five opponent-waveform indicators, chronological pair index and within-block index. No charge, choice history, terminal-vote flag, final block length or response-derived predictors.
- Seven models include a training-proportion baseline, metadata alone, pre/post EEG alone and their combinations. The sensitivity analysis removes the original terminal vote from each block (522 remaining pairs) and refits everything.

## Cohort held-out results

Scores weight participants equally. The binary BA reference is 50%. BA averages chosen/not-chosen recall. Accuracy and BA differ because choice labels are not equally frequent.

| Trials | Position | Bin | Model | BA | Accuracy | Log loss | AUC |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| full | 1 | 50 ms | Training choice proportion | 47.68% | 55.50% | 0.6910 | 0.291 |
| full | 1 | 50 ms | Waveform identities and trial order | 68.78% | 70.52% | 0.6128 | 0.769 |
| full | 1 | 50 ms | Pre EEG alone | 45.99% | 50.68% | 0.8228 | 0.417 |
| full | 1 | 50 ms | Post EEG alone | 50.90% | 55.70% | 0.7596 | 0.483 |
| full | 1 | 50 ms | Metadata + pre EEG | 66.16% | 68.02% | 0.7813 | 0.721 |
| full | 1 | 50 ms | Metadata + post EEG | 69.27% | 70.94% | 0.6629 | 0.763 |
| full | 1 | 50 ms | Metadata + pre + post EEG | 66.75% | 68.68% | 0.8001 | 0.718 |
| full | 1 | 100 ms | Training choice proportion | 47.68% | 55.50% | 0.6910 | 0.291 |
| full | 1 | 100 ms | Waveform identities and trial order | 68.78% | 70.52% | 0.6128 | 0.769 |
| full | 1 | 100 ms | Pre EEG alone | 47.65% | 52.41% | 0.7920 | 0.432 |
| full | 1 | 100 ms | Post EEG alone | 51.29% | 56.29% | 0.7532 | 0.486 |
| full | 1 | 100 ms | Metadata + pre EEG | 67.27% | 68.65% | 0.7749 | 0.721 |
| full | 1 | 100 ms | Metadata + post EEG | 69.29% | 70.96% | 0.6646 | 0.761 |
| full | 1 | 100 ms | Metadata + pre + post EEG | 67.60% | 69.19% | 0.7934 | 0.729 |
| full | 2 | 50 ms | Training choice proportion | 47.42% | 55.21% | 0.6910 | 0.291 |
| full | 2 | 50 ms | Waveform identities and trial order | 68.78% | 70.52% | 0.6128 | 0.769 |
| full | 2 | 50 ms | Pre EEG alone | 51.46% | 56.17% | 0.7741 | 0.475 |
| full | 2 | 50 ms | Post EEG alone | 50.97% | 54.99% | 0.7841 | 0.493 |
| full | 2 | 50 ms | Metadata + pre EEG | 69.48% | 70.80% | 0.6888 | 0.752 |
| full | 2 | 50 ms | Metadata + post EEG | 68.35% | 69.83% | 0.7229 | 0.745 |
| full | 2 | 50 ms | Metadata + pre + post EEG | 68.36% | 69.95% | 0.7633 | 0.728 |
| full | 2 | 100 ms | Training choice proportion | 47.42% | 55.21% | 0.6910 | 0.291 |
| full | 2 | 100 ms | Waveform identities and trial order | 68.78% | 70.52% | 0.6128 | 0.769 |
| full | 2 | 100 ms | Pre EEG alone | 51.46% | 56.68% | 0.7629 | 0.475 |
| full | 2 | 100 ms | Post EEG alone | 51.02% | 55.15% | 0.7767 | 0.494 |
| full | 2 | 100 ms | Metadata + pre EEG | 69.97% | 71.11% | 0.6823 | 0.758 |
| full | 2 | 100 ms | Metadata + post EEG | 69.53% | 71.08% | 0.7310 | 0.740 |
| full | 2 | 100 ms | Metadata + pre + post EEG | 69.00% | 70.61% | 0.7569 | 0.727 |
| nonterminal | 1 | 50 ms | Training choice proportion | 48.01% | 56.01% | 0.6942 | 0.275 |
| nonterminal | 1 | 50 ms | Waveform identities and trial order | 66.91% | 68.79% | 0.6447 | 0.732 |
| nonterminal | 1 | 50 ms | Pre EEG alone | 46.73% | 51.12% | 0.8681 | 0.411 |
| nonterminal | 1 | 50 ms | Post EEG alone | 47.48% | 51.89% | 0.7864 | 0.465 |
| nonterminal | 1 | 50 ms | Metadata + pre EEG | 64.60% | 66.36% | 0.8437 | 0.684 |
| nonterminal | 1 | 50 ms | Metadata + post EEG | 65.44% | 66.92% | 0.7293 | 0.730 |
| nonterminal | 1 | 50 ms | Metadata + pre + post EEG | 62.07% | 64.13% | 0.8894 | 0.679 |
| nonterminal | 1 | 100 ms | Training choice proportion | 48.01% | 56.01% | 0.6942 | 0.275 |
| nonterminal | 1 | 100 ms | Waveform identities and trial order | 66.91% | 68.79% | 0.6447 | 0.732 |
| nonterminal | 1 | 100 ms | Pre EEG alone | 48.48% | 53.03% | 0.8127 | 0.430 |
| nonterminal | 1 | 100 ms | Post EEG alone | 47.25% | 51.64% | 0.7696 | 0.458 |
| nonterminal | 1 | 100 ms | Metadata + pre EEG | 65.10% | 66.83% | 0.8083 | 0.699 |
| nonterminal | 1 | 100 ms | Metadata + post EEG | 65.86% | 67.34% | 0.7248 | 0.725 |
| nonterminal | 1 | 100 ms | Metadata + pre + post EEG | 65.01% | 66.76% | 0.8687 | 0.696 |
| nonterminal | 2 | 50 ms | Training choice proportion | 50.00% | 58.25% | 0.6942 | 0.275 |
| nonterminal | 2 | 50 ms | Waveform identities and trial order | 66.91% | 68.79% | 0.6447 | 0.732 |
| nonterminal | 2 | 50 ms | Pre EEG alone | 50.50% | 55.08% | 0.7877 | 0.476 |
| nonterminal | 2 | 50 ms | Post EEG alone | 52.27% | 56.46% | 0.7939 | 0.500 |
| nonterminal | 2 | 50 ms | Metadata + pre EEG | 68.62% | 69.72% | 0.6976 | 0.733 |
| nonterminal | 2 | 50 ms | Metadata + post EEG | 66.01% | 67.44% | 0.7559 | 0.714 |
| nonterminal | 2 | 50 ms | Metadata + pre + post EEG | 65.92% | 67.50% | 0.7727 | 0.709 |
| nonterminal | 2 | 100 ms | Training choice proportion | 50.00% | 58.25% | 0.6942 | 0.275 |
| nonterminal | 2 | 100 ms | Waveform identities and trial order | 66.91% | 68.79% | 0.6447 | 0.732 |
| nonterminal | 2 | 100 ms | Pre EEG alone | 49.19% | 53.73% | 0.7808 | 0.473 |
| nonterminal | 2 | 100 ms | Post EEG alone | 51.75% | 56.04% | 0.7983 | 0.482 |
| nonterminal | 2 | 100 ms | Metadata + pre EEG | 67.93% | 69.14% | 0.6927 | 0.738 |
| nonterminal | 2 | 100 ms | Metadata + post EEG | 64.65% | 66.37% | 0.7613 | 0.706 |
| nonterminal | 2 | 100 ms | Metadata + pre + post EEG | 65.90% | 67.61% | 0.7835 | 0.705 |

## Paired participant comparisons

Positive values favor post EEG: BA change is comparison minus baseline; log-loss reduction is baseline minus comparison. Intervals are descriptive 10,000-resample participant bootstrap intervals conditional on fitted predictions.

| Trials | Position | Bin | Comparison | BA change, pp [95% interval] | Log-loss reduction [95% interval] | BA Holm p | Loss Holm p |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| full | 1 | 50 ms | Post EEG versus pre EEG | +4.91 [+0.76, +9.83] | +0.0633 [+0.0181, +0.1110] | 1.0000 | 1.0000 |
| full | 1 | 50 ms | Metadata + pre + post EEG versus metadata + pre EEG | +0.59 [-0.53, +1.69] | -0.0188 [-0.0527, +0.0138] | 1.0000 | 1.0000 |
| full | 1 | 100 ms | Post EEG versus pre EEG | +3.64 [+1.70, +5.78] | +0.0389 [+0.0057, +0.0805] | 1.0000 | 1.0000 |
| full | 1 | 100 ms | Metadata + pre + post EEG versus metadata + pre EEG | +0.33 [-1.51, +2.16] | -0.0185 [-0.0602, +0.0196] | 1.0000 | 1.0000 |
| full | 2 | 50 ms | Post EEG versus pre EEG | -0.49 [-6.08, +3.97] | -0.0100 [-0.0462, +0.0235] | 1.0000 | 1.0000 |
| full | 2 | 50 ms | Metadata + pre + post EEG versus metadata + pre EEG | -1.12 [-3.98, +2.70] | -0.0745 [-0.1325, -0.0102] | 1.0000 | 1.0000 |
| full | 2 | 100 ms | Post EEG versus pre EEG | -0.44 [-5.60, +4.15] | -0.0138 [-0.0536, +0.0210] | 1.0000 | 1.0000 |
| full | 2 | 100 ms | Metadata + pre + post EEG versus metadata + pre EEG | -0.97 [-3.80, +2.42] | -0.0746 [-0.1250, -0.0219] | 1.0000 | 0.6875 |
| nonterminal | 1 | 50 ms | Post EEG versus pre EEG | +0.75 [-5.15, +6.18] | +0.0817 [+0.0142, +0.1532] | 1.0000 | 1.0000 |
| nonterminal | 1 | 50 ms | Metadata + pre + post EEG versus metadata + pre EEG | -2.54 [-4.78, -0.29] | -0.0456 [-0.1066, +0.0173] | 1.0000 | 1.0000 |
| nonterminal | 1 | 100 ms | Post EEG versus pre EEG | -1.23 [-5.63, +3.04] | +0.0431 [-0.0195, +0.1297] | 1.0000 | 1.0000 |
| nonterminal | 1 | 100 ms | Metadata + pre + post EEG versus metadata + pre EEG | -0.09 [-2.59, +2.14] | -0.0604 [-0.1300, +0.0147] | 1.0000 | 1.0000 |
| nonterminal | 2 | 50 ms | Post EEG versus pre EEG | +1.77 [-1.92, +5.45] | -0.0062 [-0.0431, +0.0300] | 1.0000 | 1.0000 |
| nonterminal | 2 | 50 ms | Metadata + pre + post EEG versus metadata + pre EEG | -2.70 [-6.10, +0.11] | -0.0751 [-0.1304, -0.0143] | 1.0000 | 1.0000 |
| nonterminal | 2 | 100 ms | Post EEG versus pre EEG | +2.56 [-1.18, +6.54] | -0.0175 [-0.0460, +0.0104] | 1.0000 | 1.0000 |
| nonterminal | 2 | 100 ms | Metadata + pre + post EEG versus metadata + pre EEG | -2.03 [-5.64, +1.18] | -0.0908 [-0.1635, -0.0238] | 1.0000 | 1.0000 |

## Interpretation and inference

- Full-cohort standalone post EEG reached 50.90%/51.29% BA for first presentations and 50.97%/51.02% for second presentations (50/100 ms bins). Metadata alone reached 68.78%. Waveform identity and trial order can predict subjective preferences; this does not establish incorrect logs or isolate an order effect.
- Adding post EEG to metadata + pre changed first-position BA by +0.59/+0.33 percentage points, and second-position BA by -1.12/-0.97 points. Mean log loss worsened in 4 of the four full-data incremental comparisons. Across all paired tests, 0 of 32 survived Holm correction at 0.05.
- Exact two-sided participant sign tests compare the direction of the paired changes; ties are omitted. Holm correction covers 32 tests: two contrasts × two metrics × two widths × two positions × two analyses. These are not individual above-chance tests or label-permutation tests. [Exact sign-test computation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html) · [Holm correction](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html).
- Second-position pre EEG occurs after the first stimulus and may contain its response; it is not an unstimulated baseline. The two positions share behavioral pairs and are not independent replications. Behavioral-response timestamps and current/trigger synchronization remain unverified.

## Per-participant class support

| Participant | Position | Epochs | Chosen | Not chosen |
| --- | ---: | ---: | ---: | ---: |
| 3 | 1 | 57 | 36 | 21 |
| 3 | 2 | 57 | 21 | 36 |
| 4 | 1 | 61 | 28 | 33 |
| 4 | 2 | 61 | 33 | 28 |
| 5 | 1 | 71 | 23 | 48 |
| 5 | 2 | 71 | 48 | 23 |
| 6 | 1 | 61 | 24 | 37 |
| 6 | 2 | 61 | 37 | 24 |
| 7 | 1 | 58 | 20 | 38 |
| 7 | 2 | 58 | 38 | 20 |
| 8 | 1 | 65 | 27 | 38 |
| 8 | 2 | 65 | 38 | 27 |
| 9 | 1 | 68 | 35 | 33 |
| 9 | 2 | 68 | 33 | 35 |
| 10 | 1 | 57 | 31 | 26 |
| 10 | 2 | 57 | 26 | 31 |
| 11 | 1 | 58 | 29 | 29 |
| 11 | 2 | 58 | 29 | 29 |
| 12 | 1 | 66 | 31 | 35 |
| 12 | 2 | 66 | 35 | 31 |

## Behavioral order audit and current decision

The within-participant test is reported above. The order association is retained as behavioral context.

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

## Archived population analysis

The earlier leave-one-participant-out fits are superseded for the current research question. They remain available in the interactive report’s collapsed archive and [the preserved report version](https://github.com/neurocontrol-lab/tens-eeg-waveform-decoding/blob/5436f02/Docs/WD008_NATURALNESS_CHOICE_RESULTS.md).
