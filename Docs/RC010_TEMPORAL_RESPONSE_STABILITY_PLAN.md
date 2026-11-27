# RC-010. Task 3 response stability and temporal generalization

**Status: Planned.** Created 25 November 2026; updated 27 November 2026. Reference: E024.
No response-trend testing, decoder fitting or numerical results have been produced for this experiment.

## Planned sequence at a glance

| Step | Work | Main output |
| --- | --- | --- |
| 1. Chronology and support | Reuse reviewed alignment/QC; retain original trial order, dose, timing, preceding stimulation and interruptions. | Audited history and early/late class counts. |
| 2. Fixed EEG measurements | Prespecify electrodes, voltage summaries and matched pre/post windows before inspecting effects. | A documented, reproducible feature definition. |
| 3. Response change | Inspect dose-matched early/middle/late responses and trends, with pre-period and timing controls. | Evidence about whether responses change before introducing a decoder. |
| 4. Temporal decoding | Where sample support permits, fit early and late amplitude-level classifiers within participant/waveform. | Matched independent within-period and cross-period scores. |
| 5. Weight and pattern stability | Compare aligned linear-model weights in common sensor/time coordinates, with calibration-resampling uncertainty and response-pattern checks. | Evidence about decoder changes without assuming neural reorganization. |
| 6. Perceptual companion | Examine existing sensation ratings; add regional hand maps after their separate extraction audit. | Whether reported perception changes alongside EEG. |
| 7. Interpretation | Summarize participant-level uncertainty and distinguish response change, temporal generalization loss, recalibration benefit and drift. | A report with supported conclusions and unresolved explanations. |

The first practical milestone is steps 1-3. The decoding stage proceeds only where independent early/late
calibration and evaluation sets have adequate class support. This plan is shared for review, not as a results report.

## Research question

Do responses to repeated stimulation change chronologically within Task 3 waveform blocks, and does a
decoder calibrated early remain useful later? If performance changes, is the pattern more consistent
with stimulus-related response change, a changing feature-to-target relationship, or broader recording/state drift?

The primary analysis characterizes response changes directly. Early-to-late decoding is a secondary
generalization test. Declining accuracy alone does not prove neural adaptation, an artifact or a confound.

This is separate from [RC-009: regional hand-map responses](RC009_REGIONAL_PERCEPT_PROPOSAL.md).
Regional drawings can later provide a behavioral companion outcome, but RC-010 does not depend on
successful image segmentation and is not a rerun of the earlier charge-prediction models.

## Dataset and feasibility

- Start with Task 3 participants 3-12 and the reviewed event alignment used in
  [RC-008](RC008_TASK3_CHARGE_PREDICTION.md). Preserve strict/reviewed exceptions and their provenance.
- Waveforms are presented in blocks; amplitude tiers are intermixed within the inspected blocks.
  Standard participant-waveform groups nominally have 30 trials, ten at each of three amplitudes.
  Audit actual counts, extra amplitude levels, exclusions and recording interruptions before using that assumption.
- Same waveform does not mean identical stimulation when amplitude varies. Direct response trends must
  hold current amplitude/dose constant or adjust for it; preceding stimuli can have different amplitudes.
- Use actual event timestamps as well as repetition indices. An initial inspection of the reviewed manifest
  found a median chronological event separation of about 13.5 seconds, with substantial variation. This is
  an inventory observation, not a validated stimulation-interval result. Reconstruct previous-stimulus gaps
  from the complete event sequence before EEG exclusions so omitted events do not inflate the gaps.
- The recording is not a controlled rapid-repetition habituation protocol. Existing waveform blocks cannot
  fully separate waveform identity from their recording-time context.

## Hypotheses and expectations

| ID | Question | Hypothesis / expectation | Important alternative |
| --- | --- | --- | --- |
| H1, primary | Does the stimulus-aligned response change with repetition at matched waveform/amplitude? | Response magnitude, latency or shape may change within a block. Attenuation is plausible, but an initial transient, plateau or no clear trend is also plausible. | Dose composition, trial history, variable intervals, attention, reporting activity or recording drift explain the change. |
| H2, secondary | Does an early-calibrated decoder lose performance later? | If early held-out decoding is informative and the mapping changes, matched later held-out performance may decline. | Stable decoding, insufficient early signal, mismatched class support or small test samples. |
| H3, secondary | Does late calibration recover late performance? | Better late-to-late than early-to-late performance would suggest a changing predictive relationship or a need for recalibration. | Poor decoding in both periods, loss of measurable information, increased noise or sampling uncertainty. |
| H4, diagnostic control | Are chronological changes also present before stimulation? | Similar pre/post changes favor a broad time/state explanation. Predominantly stimulus-aligned changes with stable pre EEG are more compatible with a stimulation-related effect. | Pre/post comparisons alone cannot identify a central neural mechanism or exclude all stimulus artifacts. |
| H5, optional behavioral companion | Do reported regional sensation maps change alongside EEG at fixed stimulation settings? | Reliable map changes may accompany altered response magnitude or distribution. | EEG changes without a reported percept change, drawing variability, or unrelated behavioral/recording trends. |

No analysis assumes that every EEG component decreases monotonically. Signed voltage and response magnitude
must be distinguished: attenuation of a negative component can appear as a numerical increase toward zero.

## Stage 0: audit chronology and class support

1. Build a manifest retaining participant, recording/segment, waveform, amplitude tier, nominal charge,
   trial ID, event timestamp, block, position within the block and repetitions of the same waveform-amplitude condition.
2. Preserve the complete stimulation history, response-validity/QC flags and reasons for exclusions.
   Record elapsed time, preceding interval and preceding amplitude. Do not renumber the surviving trials
   as though failed/excluded stimulations had never occurred.
3. Check whether blocks were repeated/restarted and whether waveform order varies across participants.
   Separate physical recording discontinuities from changes within a continuous block.
4. Inspect early/middle/late amplitude support using metadata only. Freeze temporal boundaries,
   matching rules and feature choices before inspecting decoding performance.
5. Confirm that feature periods do not cross neighboring stimuli or timestamp gaps. Check reporting timing
   where possible; unknown response times remain an interpretation limitation.

## Stage 1: response change without a decoder

Analyze chronological response variation within participants, controlling waveform and amplitude condition.
Begin with a fixed low-dimensional sensor-space representation, such as the existing contralateral ROI
(C4, CP2, CP4, CP6, P2, P4), and the same fixed 50-ms voltage bins used in prior Task 3 analyses.
Candidate matched periods are post 50-500 ms and pre -500 to -50 ms. Confirm compatible preprocessing,
reference, baseline and event support before reusing any cache.

Prespecify a small set of signed window-amplitude and response-magnitude summaries, supported by the
full response trace. Fix windows before testing; do not select the largest apparent early/late difference.
Latency/shape comparisons and all-electrode maps are exploratory extensions with their own multiplicity control.
There is no requirement to fit a classifier before assessing these response changes.

Model or summarize repetition-related trends after accounting for amplitude and waveform, retaining
participant-specific effects. Examine repetition number and elapsed time explicitly; do not force separate
causal coefficients when they are too strongly correlated. Consider preceding interval and stimulus history
with a limited prespecified adjustment set rather than overfitting many covariates to short sequences.
Plot dose-matched early/middle/late response profiles and pre-period counterparts.

Where regional drawings become available from RC-009, align them by actual trial ID and examine whether
percept changes accompany EEG changes. Stable or changing drawings are both informative; neither alone
establishes the mechanism of the EEG trend.

## Stage 2: a small temporal-generalization decoder

### Primary decoder target: amplitude tier within a waveform

Fit a separate three-class low/mid/high amplitude decoder within each participant and waveform.
Intermixed tiers make this a more focused within-block test than decoding the identity of a waveform
that stays constant throughout that block. The balanced-accuracy reference is one third when all three
classes are supported. This remains amplitude prediction, not proof of neural rather than artifact encoding.

Start with one fixed simple classifier, shrinkage LDA, and a small predefined feature set derived from
the response representation above. Keep scaling and any dimension reduction training-only. Do not tune
models against late test data or select features from the full block's apparent trends.

Approximately ten trials per tier limit calibration and test support. Select the early/late boundary
and common per-class training/evaluation quotas using a prespecified metadata-only rule. Check that
each period can supply distinct training and evaluation examples of every tier. Flag unsupported cells
and report their counts rather than silently dropping classes or forcing a fit. Boundary adjustments, if
needed, must follow a fixed support rule rather than maximize accuracy.

### Independent calibration and evaluation

Reserve disjoint early calibration, early evaluation, late calibration and late evaluation sets.
The early evaluation set must never be used to fit the early decoder. Match class composition and
training-set sizes across early/late calibration; use equal class-balanced evaluation support where feasible.
Keep overlapping windows and relevant temporal groups together and document temporal separation.

| Calibration data | Independent early evaluation | Independent late evaluation |
| --- | --- | --- |
| Early calibration | Within-early reference | Early-to-late generalization |
| Late calibration | Late-to-early diagnostic | Within-late reference |

The same late evaluation trials are used to compare early-calibrated and late-calibrated models;
likewise the same early evaluation trials support the reverse comparison. No calibration observation
enters its corresponding evaluation set. A within-period score can still exploit persistent local state;
it is a reference for the temporal contrast, not a guarantee of stimulus-specific decoding.

A decline from early **training accuracy** to later **test accuracy** is not evidence of temporal degradation.
Only comparisons with independent evaluation data and matched calibration support address that question.

If early held-out performance is uninformative, a performance-decline claim is not supported. Report the
decoder as inconclusive for temporal loss and retain the direct response-change analysis.

### Exploratory waveform-identity decoder

If examined, train on early examples from every waveform block within a participant and test on matched
later examples. A single waveform block cannot provide multiple waveform labels. Because waveform and
block identity coincide, this test cannot fully separate genuine waveform responses from recording-time
features. Do not treat arbitrary relabeling of one block per waveform as a valid independent null test:
it can merely rename the same classes. Counterbalanced repeated blocks would be needed for stronger separation.

## Stage 3: linear-model weights and response-pattern stability

Compare early- and late-calibrated linear models using the same predefined electrode/time-bin features.
Align class labels and express class-contrast coefficients in a common original sensor/time coordinate
system, back-transforming through training-only scaling and any PCA. Do not directly compare coefficients
in independently fitted PCA bases or differently scaled feature units. Report class priors/intercepts separately.

Assess coefficient stability under calibration-set resampling and matched training counts; small samples
and correlated features can change fitted weights even when prediction remains similar. Inspect class-mean
evoked response patterns alongside decoder coefficients. If activation-pattern estimates are used, keep
their estimation and interpretation explicit rather than treating large decoding weights as neural sources.

This stage describes changes in the decoder and measured response patterns. It does not by itself prove
central neural reorganization, learning or adaptation. See [Haufe et al. (2014)](https://pubmed.ncbi.nlm.nih.gov/24239590/)
for the distinction between decoding weights and interpretable activation patterns. This is a prespecified
follow-up to supported decoding, not a way to search retrospectively for a favorable electrode/time region.

## Controls, inference and interpretation

- Repeat the temporal comparison using matched pre-stimulus EEG and track channel quality, baseline
  magnitude/variance, artifact flags and interruptions. A post-stimulus dose decoder can reflect residual
  stimulation artifacts; evaluate feature timing and signal quality explicitly.
- Compare results with equal training counts and matched dose composition. Report class recall and
  confusion matrices as well as balanced accuracy; note the small and correlated evaluation samples.
- Summarize within-participant effects and then give participants equal weight. Uncertainty must respect
  participant/block dependence rather than treating every trial or repeated split as independent evidence.
- Prespecify a small primary contrast family before fitting and correct for multiple testing across
  contrasts, EEG periods and any additional feature choices. Any permutation test must have a justified
  exchangeability scheme and refit training preprocessing; unrestricted shuffling across time is not an
  automatic valid null for this experiment.
- A within-period versus cross-period gap is evidence about temporal generalization. It is not sufficient
  by itself to declare an earlier decoder invalid, identify adaptation as the cause or prove a confound.

| Observed pattern | Supported interpretation |
| --- | --- |
| Early-to-late performance falls, while late-to-late performance is stronger on the same late evaluation set | Distribution/mapping change and possible recalibration benefit; mechanism remains unresolved. |
| Pre and post responses/decoders change similarly | Broad recording or state variation is a plausible contributor. |
| Dose-matched stimulus-aligned responses change while pre EEG is comparatively stable | Compatible with a stimulation-related repetition effect, including adaptation; not proof of a central neural cause. |
| EEG and reliable percept maps change together at fixed settings | A percept-linked temporal change worth investigating, with common influences still possible. |
| Early and late decoding remain near their reference | No informative decoder-based evidence for a loss of applicability; response trends may still be detectable. |

Call a factor a confound only in relation to a specific claim: a time-related state becomes a concern for
waveform decoding when it covaries with waveform labels and can account for apparent stimulus prediction.
Genuine adaptation is itself a possible biological finding, not automatically an artifact or nuisance.

## Deliverables and execution gate

1. Audited chronological event/history manifest and early/late class-support report.
2. Prespecified response-trend analysis with pre-period, dose and timing controls.
3. Where support permits, within- and cross-period decoding matrices with matched calibration/evaluation sets,
   class recall, participant uncertainty and complete eligibility counts.
4. Common-coordinate weight and response-pattern comparisons with stability estimates, where decoding is supported.
5. A report separating response change, generalization loss, recalibration benefit and unresolved mechanisms.

After the chronology/support audit, freeze the splits, primary summaries and inferential family before
outcome testing. If decoder support is inadequate, complete the descriptive response analysis and clearly
mark decoding as unsupported or inconclusive. This document specifies future work; no numerical analysis has started.

## Sources and related work

- [Mancini et al., Characterizing the Short-Term Habituation of Event-Related Evoked Potentials](https://pmc.ncbi.nlm.nih.gov/articles/PMC6162078/):
  supports the possibility of repetition-related somatosensory ERP changes; its rapid-repetition findings
  are not assumed to transfer unchanged to this dataset's variable intervals.
- [Li et al., Training on the test set?](https://arxiv.org/abs/1812.07697): block-level temporal correlations
  motivate caution when interpreting stimulus classification from blocked EEG recordings.
- [Haufe et al., On the interpretation of weight vectors of linear models in multivariate neuroimaging](https://pubmed.ncbi.nlm.nih.gov/24239590/):
  motivates the separation of decoder-weight comparisons from physiological interpretation.
- [RC-008: Task 3 charge prediction](RC008_TASK3_CHARGE_PREDICTION.md) and its reviewed alignment/chronological
  features provide provenance and feasibility context. RC-010 adds explicit within/cross-period calibration
  comparisons and direct repetition-response characterization.
- [RC-009: regional hand-map proposal](RC009_REGIONAL_PERCEPT_PROPOSAL.md) supplies an optional later perceptual
  companion analysis. Neither experiment assumes marked-area measurements are exact physical skin areas.

RC-010 / E024 is reserved in the registry. The implementation directory is reserved only; no empty
executable package or result is created by this plan.
