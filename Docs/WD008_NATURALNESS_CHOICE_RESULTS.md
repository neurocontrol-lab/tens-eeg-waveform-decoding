# WD-008. Paired naturalness-choice EEG decoding

[Interactive report](Reports/WD-008/index.html) · [Run guide](../experiments/waveform_decoding/e021_naturalness_choice_decoding/README.md)

**24 November 2026.** One observation is one original two-stimulus trial. The model receives EEG from both presentations and predicts first chosen (0) versus second chosen (1), from Jason’s T4 Naturalness Value. Models are fitted separately within each participant.

622 eligible pairs, 1244 response epochs and 10 participants. Supported block/configuration folds: 800; unsupported: 0.

Paired post EEG alone reached 51.21%/49.40% balanced accuracy with ordered concatenation and 55.45%/55.17% with second-minus-first features (50/100 ms bins). Metadata alone reached 68.78%. Of 32 paired-change tests, 0 survived Holm correction at 0.05.

## Inputs and validation

- Actual pairs are matched by pair ID, retaining first/second order. No artificial chosen/unchosen pairs are constructed.
- All 62 electrodes in 50/100 ms bins. Both post windows span 50–1,050 ms relative to their respective pulses. The pre-pair control spans −1,050 to −50 ms before the first pulse.
- Ordered concatenation is the primary representation: first-response features followed by second-response features (2,480/1,240 original post features at 50/100 ms). The restored second-minus-first representation is a fixed comparison (1,240/620 original post features). Pre-pair controls have 1,240/620 original features.
- Instantaneous average reference and common pair baseline −1,250 to −1,050 ms before the first pulse; no temporal filter or ICA. Original complete-pair QC and all-channel finite/timestamp checks are retained.
- Within each participant, hold out every block of the same waveform comparison together. Both responses stay in one observation. No cross-participant fitting. Test-block labels can be single-class; pool held-out predictions across comparisons before scoring each participant.
- Training-only scaling, separate pre/post PCA retaining 90% variance, then training-only component scaling. Fixed L2 logistic regression C=1; no class weighting or tuning.
- Metadata: first/second waveform indicators, chronological pair index and within-block index. No charge, previous choices, terminal-vote flag or final block length enters the inputs.
- Optional last-trial exclusion removes the original final trial per block and refits models on 522 pairs. It checks dependence on those trials and reduced training data; it does not remove the stopping rule.

## Held-out results

Scores average participants equally. Balanced accuracy averages first-choice and second-choice recall; its reference is 50%. Ordinary accuracy is the fraction of correctly predicted pair choices.

| Trials | Representation | Bin | Model | BA | Accuracy | Log loss | AUC |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| full | concatenated | 50 ms | Training choice proportion | 47.42% | 55.21% | 0.6910 | 0.291 |
| full | concatenated | 50 ms | Waveform identities and trial order | 68.78% | 70.52% | 0.6128 | 0.769 |
| full | concatenated | 50 ms | Pre-pair EEG alone | 45.99% | 50.68% | 0.8228 | 0.417 |
| full | concatenated | 50 ms | Paired post EEG alone | 51.21% | 55.05% | 0.8201 | 0.494 |
| full | concatenated | 50 ms | Metadata + pre-pair EEG | 66.16% | 68.02% | 0.7813 | 0.721 |
| full | concatenated | 50 ms | Metadata + paired post EEG | 69.68% | 71.26% | 0.7187 | 0.742 |
| full | concatenated | 50 ms | Metadata + pre-pair + paired post EEG | 67.24% | 69.05% | 0.8232 | 0.702 |
| full | concatenated | 100 ms | Training choice proportion | 47.42% | 55.21% | 0.6910 | 0.291 |
| full | concatenated | 100 ms | Waveform identities and trial order | 68.78% | 70.52% | 0.6128 | 0.769 |
| full | concatenated | 100 ms | Pre-pair EEG alone | 47.65% | 52.41% | 0.7920 | 0.432 |
| full | concatenated | 100 ms | Paired post EEG alone | 49.40% | 53.23% | 0.8131 | 0.485 |
| full | concatenated | 100 ms | Metadata + pre-pair EEG | 67.27% | 68.65% | 0.7749 | 0.721 |
| full | concatenated | 100 ms | Metadata + paired post EEG | 69.89% | 71.39% | 0.7058 | 0.746 |
| full | concatenated | 100 ms | Metadata + pre-pair + paired post EEG | 67.14% | 68.51% | 0.8033 | 0.716 |
| full | difference | 50 ms | Training choice proportion | 47.42% | 55.21% | 0.6910 | 0.291 |
| full | difference | 50 ms | Waveform identities and trial order | 68.78% | 70.52% | 0.6128 | 0.769 |
| full | difference | 50 ms | Pre-pair EEG alone | 45.99% | 50.68% | 0.8228 | 0.417 |
| full | difference | 50 ms | Paired post EEG alone | 55.45% | 58.69% | 0.8652 | 0.582 |
| full | difference | 50 ms | Metadata + pre-pair EEG | 66.16% | 68.02% | 0.7813 | 0.721 |
| full | difference | 50 ms | Metadata + paired post EEG | 68.49% | 70.05% | 0.7506 | 0.753 |
| full | difference | 50 ms | Metadata + pre-pair + paired post EEG | 65.04% | 67.19% | 0.8445 | 0.714 |
| full | difference | 100 ms | Training choice proportion | 47.42% | 55.21% | 0.6910 | 0.291 |
| full | difference | 100 ms | Waveform identities and trial order | 68.78% | 70.52% | 0.6128 | 0.769 |
| full | difference | 100 ms | Pre-pair EEG alone | 47.65% | 52.41% | 0.7920 | 0.432 |
| full | difference | 100 ms | Paired post EEG alone | 55.17% | 58.51% | 0.8534 | 0.579 |
| full | difference | 100 ms | Metadata + pre-pair EEG | 67.27% | 68.65% | 0.7749 | 0.721 |
| full | difference | 100 ms | Metadata + paired post EEG | 68.05% | 69.63% | 0.7626 | 0.751 |
| full | difference | 100 ms | Metadata + pre-pair + paired post EEG | 66.85% | 68.52% | 0.8701 | 0.719 |
| nonterminal | concatenated | 50 ms | Training choice proportion | 50.00% | 58.25% | 0.6942 | 0.275 |
| nonterminal | concatenated | 50 ms | Waveform identities and trial order | 66.91% | 68.79% | 0.6447 | 0.732 |
| nonterminal | concatenated | 50 ms | Pre-pair EEG alone | 46.73% | 51.12% | 0.8681 | 0.411 |
| nonterminal | concatenated | 50 ms | Paired post EEG alone | 55.27% | 58.63% | 0.8316 | 0.526 |
| nonterminal | concatenated | 50 ms | Metadata + pre-pair EEG | 64.60% | 66.36% | 0.8437 | 0.684 |
| nonterminal | concatenated | 50 ms | Metadata + paired post EEG | 67.28% | 68.74% | 0.7380 | 0.724 |
| nonterminal | concatenated | 50 ms | Metadata + pre-pair + paired post EEG | 65.41% | 67.12% | 0.8565 | 0.688 |
| nonterminal | concatenated | 100 ms | Training choice proportion | 50.00% | 58.25% | 0.6942 | 0.275 |
| nonterminal | concatenated | 100 ms | Waveform identities and trial order | 66.91% | 68.79% | 0.6447 | 0.732 |
| nonterminal | concatenated | 100 ms | Pre-pair EEG alone | 48.48% | 53.03% | 0.8127 | 0.430 |
| nonterminal | concatenated | 100 ms | Paired post EEG alone | 53.66% | 57.01% | 0.8096 | 0.530 |
| nonterminal | concatenated | 100 ms | Metadata + pre-pair EEG | 65.10% | 66.83% | 0.8083 | 0.699 |
| nonterminal | concatenated | 100 ms | Metadata + paired post EEG | 66.27% | 67.63% | 0.7168 | 0.732 |
| nonterminal | concatenated | 100 ms | Metadata + pre-pair + paired post EEG | 65.03% | 66.63% | 0.8332 | 0.700 |
| nonterminal | difference | 50 ms | Training choice proportion | 50.00% | 58.25% | 0.6942 | 0.275 |
| nonterminal | difference | 50 ms | Waveform identities and trial order | 66.91% | 68.79% | 0.6447 | 0.732 |
| nonterminal | difference | 50 ms | Pre-pair EEG alone | 46.73% | 51.12% | 0.8681 | 0.411 |
| nonterminal | difference | 50 ms | Paired post EEG alone | 56.44% | 59.61% | 0.8565 | 0.574 |
| nonterminal | difference | 50 ms | Metadata + pre-pair EEG | 64.60% | 66.36% | 0.8437 | 0.684 |
| nonterminal | difference | 50 ms | Metadata + paired post EEG | 66.67% | 68.75% | 0.7426 | 0.732 |
| nonterminal | difference | 50 ms | Metadata + pre-pair + paired post EEG | 64.94% | 66.79% | 0.8496 | 0.696 |
| nonterminal | difference | 100 ms | Training choice proportion | 50.00% | 58.25% | 0.6942 | 0.275 |
| nonterminal | difference | 100 ms | Waveform identities and trial order | 66.91% | 68.79% | 0.6447 | 0.732 |
| nonterminal | difference | 100 ms | Pre-pair EEG alone | 48.48% | 53.03% | 0.8127 | 0.430 |
| nonterminal | difference | 100 ms | Paired post EEG alone | 54.78% | 58.17% | 0.8370 | 0.581 |
| nonterminal | difference | 100 ms | Metadata + pre-pair EEG | 65.10% | 66.83% | 0.8083 | 0.699 |
| nonterminal | difference | 100 ms | Metadata + paired post EEG | 66.75% | 68.96% | 0.7337 | 0.743 |
| nonterminal | difference | 100 ms | Metadata + pre-pair + paired post EEG | 66.22% | 68.34% | 0.8346 | 0.705 |

## What does paired post EEG add?

Positive BA changes and log-loss reductions favor paired post EEG. Intervals are descriptive participant bootstrap intervals conditional on fixed held-out predictions.

| Trials | Representation | Bin | Comparison | BA change, pp [95% interval] | Log-loss reduction [95% interval] | BA Holm p | Loss Holm p |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| full | concatenated | 50 ms | Paired post EEG versus pre-pair EEG | +5.23 [+1.32, +8.74] | +0.0028 [-0.0584, +0.0628] | 1.0000 | 1.0000 |
| full | concatenated | 50 ms | Metadata + pre-pair + paired post EEG versus metadata + pre-pair EEG | +1.08 [-3.06, +5.04] | -0.0419 [-0.1019, +0.0133] | 1.0000 | 1.0000 |
| full | concatenated | 100 ms | Paired post EEG versus pre-pair EEG | +1.75 [-1.37, +4.34] | -0.0210 [-0.0837, +0.0555] | 1.0000 | 1.0000 |
| full | concatenated | 100 ms | Metadata + pre-pair + paired post EEG versus metadata + pre-pair EEG | -0.13 [-3.39, +3.37] | -0.0284 [-0.0833, +0.0208] | 1.0000 | 1.0000 |
| full | difference | 50 ms | Paired post EEG versus pre-pair EEG | +9.46 [+2.82, +17.53] | -0.0423 [-0.1552, +0.0515] | 0.6875 | 1.0000 |
| full | difference | 50 ms | Metadata + pre-pair + paired post EEG versus metadata + pre-pair EEG | -1.12 [-5.04, +3.64] | -0.0631 [-0.1772, +0.0366] | 1.0000 | 1.0000 |
| full | difference | 100 ms | Paired post EEG versus pre-pair EEG | +7.52 [+2.48, +13.56] | -0.0614 [-0.1915, +0.0422] | 1.0000 | 1.0000 |
| full | difference | 100 ms | Metadata + pre-pair + paired post EEG versus metadata + pre-pair EEG | -0.43 [-3.57, +3.53] | -0.0952 [-0.2194, +0.0128] | 1.0000 | 1.0000 |
| nonterminal | concatenated | 50 ms | Paired post EEG versus pre-pair EEG | +8.53 [+2.91, +14.15] | +0.0365 [-0.0603, +0.1375] | 0.6875 | 1.0000 |
| nonterminal | concatenated | 50 ms | Metadata + pre-pair + paired post EEG versus metadata + pre-pair EEG | +0.80 [-1.85, +3.45] | -0.0128 [-0.1172, +0.0925] | 1.0000 | 1.0000 |
| nonterminal | concatenated | 100 ms | Paired post EEG versus pre-pair EEG | +5.19 [-2.20, +12.37] | +0.0031 [-0.1034, +0.1234] | 1.0000 | 1.0000 |
| nonterminal | concatenated | 100 ms | Metadata + pre-pair + paired post EEG versus metadata + pre-pair EEG | -0.07 [-2.72, +3.15] | -0.0250 [-0.1399, +0.0899] | 1.0000 | 1.0000 |
| nonterminal | difference | 50 ms | Paired post EEG versus pre-pair EEG | +9.71 [+5.35, +13.72] | +0.0116 [-0.0838, +0.0960] | 0.6875 | 1.0000 |
| nonterminal | difference | 50 ms | Metadata + pre-pair + paired post EEG versus metadata + pre-pair EEG | +0.34 [-3.94, +4.90] | -0.0059 [-0.1080, +0.0858] | 1.0000 | 1.0000 |
| nonterminal | difference | 100 ms | Paired post EEG versus pre-pair EEG | +6.30 [+0.45, +11.90] | -0.0243 [-0.1337, +0.0711] | 1.0000 | 1.0000 |
| nonterminal | difference | 100 ms | Metadata + pre-pair + paired post EEG versus metadata + pre-pair EEG | +1.12 [-2.92, +5.11] | -0.0264 [-0.1497, +0.0751] | 1.0000 | 1.0000 |

## Interpretation

The full-data metadata-only BA is 68.78%. Paired EEG does not show a consistent incremental benefit over metadata + pre-pair EEG across the tested representations and widths. Observed above-50% scores are not individual significance claims. Exact two-sided participant sign tests assess paired BA/log-loss changes; Holm correction covers 32 tests across two contrasts, two metrics, two representations, two widths and two trial analyses. [Sign-test computation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html) · [Holm correction](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html).

The second post response can contain carryover, position and decision-related activity. All EEG controls before the pair use only the first pre window; the second pre window is excluded because it follows the first stimulus. Concatenated post EEG contains two one-second windows, while the pre control contains one. Physical delivery synchronization and behavioral-response timestamps remain unavailable.

Superseded single-presentation and population analyses are retained locally for provenance and are omitted from the shared report. No acquisition or archived numerical files were deleted.

## Behavioral order audit and current decision

The behavioral association provides context for the within-participant paired EEG analysis.

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
