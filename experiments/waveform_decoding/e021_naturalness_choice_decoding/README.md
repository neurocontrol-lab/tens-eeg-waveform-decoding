# WD-008 · Within-participant paired naturalness decoding

Each observation is one original Task 4 two-stimulus trial. The model receives both post-stimulation
EEG responses and predicts which stimulus the participant chose as more natural. Labels come from
Jason’s `T4 Naturalness Value`: first chosen = 0, second chosen = 1.

[Results](../../../Docs/WD008_NATURALNESS_CHOICE_RESULTS.md) ·
[Interactive report](../../../Docs/Reports/WD-008/index.html)

## Dataset and validation

The cohort contains 622 complete eligible behavioral pairs (1,244 response epochs) from participants
3–12, retaining all five waveforms and all ten comparisons. Actual first/second responses are matched
by pair ID. No artificial pairs are constructed from independently chosen and unchosen stimuli.

Fit models separately for each participant. Hold out all blocks of an entire waveform comparison,
keeping both responses in one observation. No model trains on another participant. Single-class test
blocks remain in the dataset; participant scores pool all held-out predictions across comparisons.
All 800 configuration/comparison folds were supported.

## Paired representations and features

- Primary: concatenate first-response and second-response features in presentation order.
- Fixed comparison: subtract first-response features from second-response features, restoring the
  original paired-difference representation.
- Both post windows span 50–1,050 ms relative to their respective pulse. The pre-pair control uses
  only −1,050 to −50 ms before the first pulse, before either stimulus.
- All 62 electrodes in fixed 50/100 ms bins. Concatenated post features number 2,480/1,240;
  differences and pre-pair controls number 1,240/620.
- Instantaneous average reference and shared baseline −1,250 to −1,050 ms before the first pulse;
  no temporal filtering or ICA. Original complete-pair QC and all-channel finite/timestamp checks remain.
- Fit scaling, separate pre/post PCA retaining 90% variance, and component scaling on training pairs only.
- Fixed L2 logistic regression C=1, without class weighting, tuning or supervised feature selection.
- Metadata: five first-waveform and five second-waveform indicators, chronological pair index and
  within-block index. No charge, prior choices, terminal-vote flag or final block length is used.

Seven models: training choice proportion, metadata alone, pre-pair EEG alone, paired post EEG alone,
metadata + pre, metadata + paired post, and metadata + pre + paired post.

The optional original-last-trial exclusion refits models on 522 pairs after dropping each block’s
original final trial. This checks robustness to excluding those trials and losing training data;
it does not correct for or remove the race-to-five stopping rule.

## Results and interpretation

Standalone paired post EEG reached 51.21/49.40% BA with concatenation and 55.45/55.17% with
second-minus-first features (50/100 ms bins). Metadata alone reached 68.78% BA.
Adding paired post EEG to metadata + pre changed BA by +1.08/−0.13 percentage points with
concatenation and −1.12/−0.43 points with differences. No paired comparison survived Holm correction.

Balanced accuracy averages first-choice and second-choice recall; its reference is 50%.
Ordinary accuracy is the fraction of correctly predicted pair choices. Cohort means average separately
fitted participant scores equally; there is no population model.

Descriptive 10,000-resample participant bootstrap intervals condition on the fitted predictions.
Exact two-sided participant sign tests compare BA and log-loss changes, omitting ties; Holm correction
covers 32 tests across two contrasts, two metrics, two representations, two bin widths and two trial
analyses. These are not individual above-chance tests or refitted label-permutation tests.

The second post response can contain carryover, presentation and decision effects. Concatenated post
inputs contain two one-second windows, while the pre-pair control contains one. Behavioral-response
timestamps and physical delivery synchronization remain unavailable. The behavioral order audit
provides context; recorded preference labels are not declared invalid.

## Run and saved outputs

```powershell
& ./.venv/Scripts/python.exe -m experiments.waveform_decoding.e021_naturalness_choice_decoding.run_paired
```

The numerical run is `output/wd008_paired_naturalness_20261124`. It verifies and reuses the all-electrode
cache `output/wd008_position_feature_cache_20261124` without changing its feature values or source audit.
Outputs include `pair_manifest.csv`, `predictions.csv`, `fold_audit.json`, `unsupported.json`, `config.json`,
`summary.json` and `status.json`. Fold audits record training/test pair IDs, comparisons, participant,
representation and PCA dimensions. Use a fresh output directory for a rerun.

```powershell
& ./.venv/Scripts/python.exe -m experiments.publish_wd008_wd009_reports --output output/wd008_paired_report_refresh
```

The report publisher defaults to the paired run (`--paired-run` overrides it). Publication provenance
is in `output/wd008_paired_publication_20261124/publication_manifest.json`.

## Local archives

The superseded single-presentation run (`output/wd008_position_naturalness_20261124`) and population run
(`output/wd008_naturalness_baselined_20261123`) are preserved unchanged locally. Their results and
model panels are omitted from the shared report. The old `.run_position` and original package command
remain available for reproduction; they are not the current analysis. Explicit `--position-run` or
`--legacy-only` publisher options reproduce those older views if needed.
