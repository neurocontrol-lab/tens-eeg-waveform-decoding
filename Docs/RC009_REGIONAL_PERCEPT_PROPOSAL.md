# RC-009. Regional hand-map responses to stimulation and EEG

**Status: Planned.** Proposal dated 24 November 2026. Reference: E023.
No image segmentation, hypothesis testing or model fitting has been performed for this experiment.

## Research question

How do stimulation settings relate to the extent and anatomical distribution of the reported sensation,
and does EEG explain trial-to-trial variation in that report beyond stimulation settings and trial order?

The primary dataset is Task 3, the super-threshold task with repeated presentations at different amplitude
levels within each waveform. This provides continuous perceptual outcomes alongside the stimulus and EEG.
The Task 4 paired-preference results remain a separate experiment.

The first stage is a trial-level extension of the percept-area and amplitude comparisons already described
in Whitson's thesis. The EEG stage asks about the reported percept, extending the earlier EEG-charge studies.
It is not a claim that an area-amplitude relationship is new.

## One representation for area and location

Use anatomical landmarks to partition the common hand template into fixed, non-overlapping regions.
Start with the palm/body and five digits on the front view, and the dorsum/body and five digits on the back
view: twelve candidate regions. Final boundaries are frozen after template inspection and before testing
relationships with stimulation or EEG. A wrist/forearm region is included only if the actual template
supports an identifiable, consistently drawn region; it is not added based on interesting effects.

For trial i and region r:

\[
a_{i,r}=\frac{\text{highlighted pixels inside region }r}
{\text{total pixels inside the front-and-back hand-template masks}}.
\]

The trial outcome is one regional-coverage vector:

\[
\mathbf a_i=[a_{i,\mathrm{front\ palm}},a_{i,\mathrm{front\ thumb}},\ldots,
a_{i,\mathrm{back\ body}},a_{i,\mathrm{back\ thumb}},\ldots].
\]

Every component uses the same whole-template denominator. If the regions partition the template,
their sum is the total marked-area fraction. Percentages of each individual region's area are different
quantities and must not be summed as total coverage.

This single vector captures extent and coarse anatomical location. More coverage in the same regions
indicates expansion; changing contributions across regions indicates redistribution. For example,
10% palm coverage and 5% palm plus 5% index-finger coverage have the same total area but different vectors.
Total coverage and regional proportions are interpretation summaries of the same outcome, not separate
experiments. Within-region movement can be missed at this resolution.

The measurement is marked pixel coverage of a schematic, not physical skin area in square centimetres.
Brush width, drawing style and the distinction between a stroke and a filled percept field require visual QA.

## Hypotheses and expectations

| ID | Relationship | Hypothesis / expectation | Null or competing explanation |
| --- | --- | --- | --- |
| H1: dose and extent, primary behavioral | Amplitude / nominal cathodic charge versus the regional vector | Within a participant and waveform, higher dose is expected to increase total marked coverage on average, through increases in one or more regions. Individual regions need not all increase, and growth may saturate. | No reproducible within-condition dose trend; apparent pooled effects arise from differences between participants, waveforms or recording blocks. |
| H2: anatomical recruitment, exploratory | Dose and waveform shape versus the regional pattern | Higher dose may recruit additional regions or expand existing regions. Different waveforms may have different patterns even when total marked coverage is similar. No universal direction of movement is assumed. | Maps mainly scale up within stable regions, or apparent changes reflect drawing variability. Waveform-block differences may reflect order rather than shape. |
| H3: EEG and percept map, primary EEG | Post-stimulus EEG versus the regional vector after stimulus/order controls | Post EEG may improve held-out prediction of regional coverage beyond known stimulation settings, trial order and pre-stimulus EEG. The expectation is uncertain given the weak incremental EEG results so far; a null result is plausible. | EEG adds no reproducible information. Apparent associations are explained by dose, order, artifacts or reporting activity. |
| H4: EEG and nominal dose, secondary | EEG response features versus amplitude / nominal charge | Examine whether any dose-related EEG pattern is stimulus aligned and more informative than its pre-stimulus control. This overlaps earlier charge studies and is a consistency check, not the main novelty. | Pre and post behave similarly, or post associations are compatible with residual stimulation artifacts, recording drift or weak effects. |

For H1, the directional expectation is greater aggregate coverage, not a positive slope in every region.
For H2, expansion and redistribution are alternatives that will be read from the same regional vector.
For H3, improvement is an empirical hypothesis, not an assumption that percept information must be decodable.
These expectations will not be used to choose regions, electrodes or time windows after viewing outcomes.

## Stimulation quantities and identifiability

Use recorded peak current amplitude and documented **nominal cathodic charge**:

\[
Q_{i,\mathrm{nominal}}=I_{i,\mathrm{peak}}\,PW\,k_w,
\qquad PW=0.4\ \mathrm{ms}.
\]

The ideal shape factor is 1 for square, 2/pi for half-sine, and 0.5 for triangle and either ramp.
With current in mA and width in ms, nominal charge is in microcoulombs. This is cathodic-phase charge,
not signed net charge across a nominally charge-balanced biphasic pulse, and is not a current-monitor measurement.

Within a waveform, pulse width and shape factor are fixed, so nominal charge is proportional to amplitude.
Analyze these as alternative descriptions of dose; do not interpret their coefficients as independently
identified physiological effects. The fixed 400-microsecond width does not support a pulse-width effect test.
Across-waveform comparisons require examination of dose overlap and waveform/order confounding before interpretation.

## Stage 0: verify the hand-map dataset

1. Inventory images by participant, task, trial, waveform, amplitude and template version. Check repeated
   filenames/trial IDs, duplicate images, interrupted trials and extra amplitude levels. Image and CSV counts
   must be reconciled rather than assuming that all files map one-to-one to EEG events.
2. Join each image to the master CSV and then the verified XDF event using validated identifiers and order.
   Resolve the task-numbering difference in the thesis and protocol addendum. Do not infer labels solely
   from an image directory name, filename waveform number or an unverified trial offset.
3. Record missing/invalid reports separately from a genuinely valid empty drawing. Preserve failed/no-response
   flags and acquisition exceptions; do not silently convert missing drawings into zero area.
4. Verify common front/back template geometry, crop out GUI text, and extract highlight pixels only inside
   the hand masks. Flag marks outside the template, unusual dimensions, missing surfaces and possible duplicates.
5. Inspect overlays spanning participants, waveforms and amplitude levels. Freeze anatomical regions and the
   segmentation rule before examining hypothesis effects. Check sensitivity to a small, documented range
   of highlight thresholds and brush/edge conventions.
6. Report matched counts and exclusions before statistical fitting. Start with participants 3-12 to align with
   recent EEG analyses; inspect Subject 2 separately for Task 3 compatibility rather than inheriting its Task 4 exclusion.

Progress to the EEG stage only when trial matching is defensible and regional measures have interpretable
within-condition variation/repeatability. If drawings cannot be linked per trial, aggregate EEG to the actual
reporting unit and describe the resulting condition-level analysis; do not copy one report onto many epochs
and treat them as independent labels.

## Stage 1: relate stimulation to the regional map

- Use a single multivariate regional-vector framework. Begin with within-participant, within-waveform
  summaries and dose trends, retaining the repeated trials rather than correlating only waveform grand means.
- Account for participant, waveform/condition, chronological trial index and recording block. Show dose
  means and individual-participant patterns; pooling trials must not turn repeated observations into independent subjects.
- Compare amplitude-based and nominal-charge-based dose descriptions. Within-waveform results do not
  separate amplitude and charge mechanisms. Assess shape-related differences only where settings overlap;
  blocked waveform acquisition limits a causal shape interpretation.
- Summarize the fitted regional pattern as total expansion versus redistribution. Relative shares are undefined
  when total marked area is zero and are not set to an arbitrary uniform location. Do not use the observed
  total area as an extra predictor for held-out EEG prediction unless that quantity is independently available.
- Report participant-level uncertainty and a prespecified global vector effect for the main dose question.
  Regional follow-ups are exploratory and require multiplicity control. Final inference details are fixed
  after the identifier/variation audit and before fitting the outcome models.

## Stage 2: does EEG contribute beyond stimulation and task structure?

Predict the same regional-area vector using a simple regularized multi-output regression, initially within
participants. This is a continuous map-prediction task, not waveform classification or charge subtraction.

| Comparison | Inputs | Purpose |
| --- | --- | --- |
| Reference | Training-fold condition/mean regional map | Quantify the gain over predicting a typical map without single-trial EEG. |
| Stimulus/order baseline | Waveform identity, one dose description, chronological/within-block indices | Determine how much of the report is predictable from known settings and task structure. |
| Pre EEG alone | A fixed pre-stimulus EEG window | Check expectation, prior-stimulus and slow recording/reporting influences. |
| Post EEG alone | A fixed post-stimulus EEG window | Describe unadjusted EEG-map prediction. |
| Baseline + pre | Stimulus/order baseline and pre EEG | Controlled reference for the incremental EEG question. |
| Baseline + pre + post | The same inputs plus post EEG | Primary incremental test of whether post EEG contributes to the reported regional pattern. |

Candidate starting features are all available scalp electrodes, 50/100 ms voltage-bin means, and training-only
scaling/PCA with a regularized linear multi-output regressor. A proposed one-second post window is 50-1,050 ms
with a matched pre window, subject to actual Task 3 event gaps, recording support and reporting timing.
Do not assume that the Task 4 timing limits or caches establish Task 3 eligibility. Fix or revise these choices
before outcome-model fitting, document the choice, and keep regularization selection inside training data.

Use chronological held-out chunks within waveform, preserving repeated-trial dependence and sufficient
dose support in training. Keep neighboring/overlapping EEG windows together where necessary. Use the same
eligible trials and splits for all controls; no whole-dataset feature selection or ROI optimization from labels.
Explicitly separate within-condition performance from cross-waveform transfer, which is exploratory here.

Primary performance is participant-balanced held-out error reduction for the joint regional vector relative
to the prespecified controlled reference. Report raw error, regional errors and the corresponding total-area
and distribution summaries so a large region does not hide failure elsewhere. Nonnegative bounded targets
do not guarantee bounded linear predictions; retain and report out-of-range predictions rather than silently
improving scores through clipping. A global cohort comparison comes first; regional follow-ups are secondary.

## Confounds and interpretation

Correlation between two measured responses is not automatically unconfounded. Stimulation dose can influence
both the drawn map and EEG; residual electrical artifacts can also scale with dose. Trial order, attention,
adaptation and motor/visual activity during reporting may affect both measurements. Pre-stimulus controls,
known settings, temporal validation and reporting-timing checks therefore remain necessary.

Task 3 does not use Task 4's two-pulse preference structure, but its actual stimulus intervals and history
must still be audited. A positive EEG-map association is evidence of prediction/association under the tested
controls; it is not proof of cortical encoding or causality. A negative EEG result would not erase a reproducible
behavioral dose-map relationship.

## Deliverables and decision points

1. Audited image/CSV/EEG manifest, fixed template-region specification and segmentation overlay examples.
2. A report of within-participant dose-map relationships, including global extent and regional distributions
   from the same vector and a comparison with the existing thesis observations.
3. If the audit supports it, EEG-map prediction with stimulus-only, pre-stimulus and incremental controls,
   regional errors, participant summaries and appropriately corrected inference.
4. A decision on whether the data support a percept-map association, an incremental EEG contribution,
   behavioral findings only, or remain inconclusive because of map reliability/alignment/sample limitations.

## Sources and related analyses

- [Collu et al., Scientific Reports (2023)](https://www.nature.com/articles/s41598-023-28594-0): perceptive-field
  area/location and sensation quality motivate the question. Their matched-setting waveform comparison did
  not show significant total-area differences; shape-dependent regional changes are not assumed guaranteed.
- [Whitson thesis](https://stars.library.ucf.edu/hut2024/661/), methods and amplitude/area results: the present
  proposal is an extension of the original behavioral analyses, with explicit trial matching and EEG controls.
- [Project protocol and task numbering](../PROJECT_REVIEW.md) and
  [experiment clarifications](EXPERIMENT_CLARIFICATIONS.md).
- [RC-006: EEG and nominal charge](E017_TASK3_CHARGE_RESULTS.md) and
  [RC-008: Task 3 charge prediction](RC008_TASK3_CHARGE_PREDICTION.md): earlier weak EEG-charge findings
  motivate cautious expectations and distinguish the new percept-map outcome from another charge decoder.
- [RC-010: response stability and temporal generalization](RC010_TEMPORAL_RESPONSE_STABILITY_PLAN.md)
  is a separate planned investigation of within-block response changes, with regional maps as an optional companion outcome.

The registry reserves RC-009 / E023. The implementation directory is reserved only; no executable package
or numerical result is created by this proposal.
