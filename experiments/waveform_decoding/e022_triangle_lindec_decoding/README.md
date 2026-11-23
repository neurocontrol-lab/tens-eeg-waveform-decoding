# WD-009 · Triangle versus linear-decrease EEG decoding

This dedicated binary waveform classifier extends the earlier five-class and one-versus-rest analyses.
Targets come from the verified triggers, 241 for Triangle and 242 for LinDec.
Naturalness votes contribute only to the inherited trial-validity QC.

The research question follows Collu et al.'s 70% median behavioral discrimination of triangle versus
linear decrease under matched charge and peak current. WD-009 asks whether EEG distinguishes the
delivered waveforms. [Collu et al.](https://doi.org/10.1038/s41598-023-28594-0)

[Results](../../../Docs/WD009_TRIANGLE_LINDEC_RESULTS.md) ·
[Interactive report](../../../Docs/Reports/WD-009/index.html)

## Dataset and validation

The audited cohort contains 501 epochs from Task 4 participants 3–12: 244 Triangle and 257 LinDec.
First presentation has 241 epochs; second has 260. Separate models are trained for each participant
and presentation position.

Each complete waveform-comparison group is held out in turn within a participant and position.
For direct triangle/LinDec testing, the entire direct block is held out and the model learns from
the other comparison contexts. The direct evaluation contains 126 epochs. The nominal-charge-matched
direct evaluation contains 66 epochs from participants 3, 4, 5, 6 and 10. These subsets reuse held-out
predictions from the full models.

Charge matching and comparison context are independent evaluation filters. The default interactive
view uses matched nominal charge across all comparison contexts, totaling 247 epochs from those five
participants. Other-pair-only evaluation contains 375 epochs, or 181 with nominal charge matching.
Matching is checked within each participant × presentation dataset. Filters select held-out predictions
and preserve the original model fits.

Direct pairs are optional for single-epoch decoding. The first-pulse EEG window ends before the second
stimulus, while comparison blocks can still differ in expectations and recording order. For the second
pulse, context also changes the preceding stimulus. Whole-comparison-group validation is retained.

Triangle and LinDec have the same ideal area factor (0.5) and cathodic duration (0.4 ms), but their
participant-specific threshold amplitudes can differ. Therefore nominal charge matches in five of the
ten participants. Task 4's race-to-five protocol produces variable counts; the ten repetitions per
amplitude belong to Task 3. Approximately ten epochs per waveform and position do not imply equal charge.

## Features and models

- All 62 electrodes, using fixed 50 or 100 ms voltage means.
- Post period 50–1,050 ms; pre period −1,050 to −50 ms.
- A shared pair baseline is measured −1,250 to −1,050 ms before the first pulse, before the first-pulse
  pre-control window. It removes DC offsets from both presentations.
- Instantaneous 62-channel average reference; no temporal filtering or ICA.
- Training-only standardization. PCA retains 90% of training variance.
- PCA + shrinkage LDA, PCA + linear SVM, and linear SVM without PCA.
- LDA uses equal class priors. Linear SVM uses fixed C=1 and balanced class weights.

The original complete-pair QC is retained, with finite checks and descriptive amplitude flags on all
additional channels. No channels are selected using waveform labels. Nominal charge, preference votes,
comparison identity and presentation order are excluded from model input features.

Balanced accuracy and ordinary accuracy are accompanied by per-class recall, AUC and confusion counts.
Scores average participants equally within each presentation. Participant intervals condition on the
fixed held-out predictions. Exact two-sided participant sign tests compare post with pre BA, omitting ties.
They test the direction of change across independent participants. Holm correction covers all 72 comparisons
(six configurations × six subsets × two positions). Five non-tied participants cannot yield a two-sided
p below 0.0625. These tests do not establish individual above-chance decoding; no label-permutation refits are run.
[Exact binomial test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html) ·
[Holm correction](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html).

## Interpretation

The first presentation is the primary view. Second-presentation pre and post EEG can contain the preceding
pulse's response. The behavioral order association does not establish physiological carryover or unusable
EEG. The current-monitor traces and response timestamps remain unavailable.

The charge-matched subset uses documented nominal settings. Physical current and artifact contributions
remain unverified. The 70% result in Collu et al. is behavioral match-to-sample discrimination.
This experiment measures EEG waveform-identity prediction.

## Run

```powershell
& ./.venv/Scripts/python.exe -m experiments.waveform_decoding.e022_triangle_lindec_decoding
```

Use a fresh output directory for each fit. The verified cache at
`output/wd009_feature_cache_20261123` can be reused. The default numerical run is
`output/wd009_triangle_lindec_20261123`.

Refresh both linked reports and their embedded plots from saved predictions with a fresh output directory.

```powershell
& ./.venv/Scripts/python.exe -m experiments.publish_wd008_wd009_reports --output output/wd008_wd009_report_refresh
```
