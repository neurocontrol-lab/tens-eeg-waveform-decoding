# WD-008 · Participant-specific naturalness-choice EEG decoding

The current question is whether an individual Task 4 presentation’s EEG predicts whether it was
chosen as more natural, beyond waveform identities, trial order and pre-stimulation EEG.
Labels come from Jason’s `T4 Naturalness Value`: 1 chooses the first presentation and 2 the second.
For each presentation-specific classifier, chosen is encoded 1 and not chosen 0.

[Current results](../../../Docs/WD008_NATURALNESS_CHOICE_RESULTS.md) ·
[Interactive report](../../../Docs/Reports/WD-008/index.html)

## Current design

The revised analysis uses 622 eligible behavioral pairs (1,244 epochs) from participants 3–12.
Each participant × presentation has separate fitted models. A pair contributes one epoch to each
position dataset with complementary labels; the responses are not subtracted or pooled.

Within a participant/position, the entire waveform comparison is held out, including any repeated
blocks of that comparison. All preprocessing is fitted on the other comparison groups. Single-class
test blocks are retained; scores pool held-out predictions across blocks before calculating each
participant’s balanced accuracy. The common 622-pair cohort and original complete-pair QC are preserved.
All 800 analysis/position/resolution/comparison folds were supported; none were discarded.

## Features and models

- All 62 electrodes, using fixed 50/100 ms voltage means over post 50–1,050 ms and pre −1,050 to −50 ms.
  This yields 1,240/620 original features per period.
- Instantaneous average reference and microvolt conversion; shared pair baseline −1,250 to −1,050 ms
  before the first pulse. No temporal filtering or ICA. Finite-value/timestamp checks cover all channels;
  this does not extend the inherited amplitude-rejection policy to all channels.
- Training-only feature scaling, separate pre/post PCA retaining 90% variance, and training-only component scaling.
- Fixed L2 logistic regression, C=1, no class weighting, supervised selection or hyperparameter search.
- Twelve metadata features: five current-waveform and five opponent-waveform indicators, chronological
  pair index and within-block index. No charge, choice history, terminal-vote flag, final block length
  or response-derived predictor enters the models.
- Seven models: training choice proportion, metadata, pre EEG, post EEG, metadata + pre,
  metadata + post, and metadata + pre + post.

The original terminal vote in each block is removed and all models are refitted for the sensitivity
analysis (522 behavioral pairs). The new last trial is not relabeled as terminal.

## Results and interpretation

For full-data first presentations, standalone post EEG reached 50.90/51.29% BA at 50/100 ms resolution;
second presentations reached 50.97/51.02%. Metadata alone reached 68.78% BA.
Adding post EEG to metadata + pre changed first-position BA by +0.59/+0.33 percentage points and
second-position BA by −1.12/−0.97 points; mean log loss worsened in all four full-data comparisons.
No paired comparison survived Holm correction, including the terminal-vote sensitivity.

Accuracy, balanced accuracy, both choice recalls, AUC, Brier score and log loss are saved per participant.
Cohort means weight participants equally. Descriptive 10,000-resample participant bootstrap intervals
condition on the fitted held-out predictions. Exact two-sided sign tests compare participant directions
of change, omitting ties, for post versus pre and post added to metadata + pre. Holm correction covers
32 tests (two contrasts × two metrics × two bin widths × two positions × two trial analyses).
These are not individual above-chance tests or refitted label-permutation tests.

Second-position pre EEG follows the first stimulus and can contain its response. The two positions
share behavioral pairs and are not independent replications. Behavioral-response timestamps and physical
current/trigger synchronization remain unavailable. The behavioral order audit provides context;
it does not make the recorded choices incorrect. Metadata performance can reflect consistent preferences.

## Run and saved outputs

```powershell
& ./.venv/Scripts/python.exe -m experiments.waveform_decoding.e021_naturalness_choice_decoding.run_position
```

The current numerical run is `output/wd008_position_naturalness_20261124`; the independently verified
cache is `output/wd008_position_feature_cache_20261124`. Outputs include `epoch_manifest.csv`,
`predictions.csv`, `fold_audit.json`, `unsupported.json`, `config.json`, `summary.json` and `status.json`.
Each fold audit records its training/test pair IDs, comparisons, participant, position and PCA dimensions.
Use a fresh output directory for a rerun; verified feature caches may be reused.

```powershell
& ./.venv/Scripts/python.exe -m experiments.publish_wd008_wd009_reports --output output/wd008_position_report_refresh
```

The publisher includes the revised position run by default. Pass `--position-run` for a different
completed run or `--legacy-only` to reproduce the historical presentation without this revision.
The current report publication record is `output/wd008_without_counts_table_20261124/publication_manifest.json`.

## Archived population analysis

The original command without `.run_position` fits the older leave-one-participant-out models.
It is retained to reproduce `output/wd008_naturalness_baselined_20261123`, not as the current analysis.
Its six-electrode, 100–350 ms features and second-minus-first representation differ from this revision;
the new results are not a comparison changing validation alone. Archived inputs and predictions are
unchanged, and their report is available in the interactive report’s collapsed archive and
[the preserved version](https://github.com/neurocontrol-lab/tens-eeg-waveform-decoding/blob/5436f02/Docs/WD008_NATURALNESS_CHOICE_RESULTS.md).
