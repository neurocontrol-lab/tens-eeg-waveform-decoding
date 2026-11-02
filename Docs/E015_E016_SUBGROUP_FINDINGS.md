# E015/E016 — Waveform, participant and presentation subgroup audit

**2 November 2026.** Descriptive analysis of existing held-out predictions.
No new training, thresholds, feature selection or significance tests.

## Answer

Some individual waveform/participant/presentation scores exceed their nominal
chance references. **Participant 6, second presentation, Triangle** is the
clearest current lead across strict/exploratory processing and multiclass/binary
formulations. There is no established statistically above-chance condition, and
no waveform is consistently easiest across the cohort.

## Measures that answer different questions

- **Five-class balanced accuracy:** mean of the five class recalls; nominal
  reference 20%.
- **One class's multiclass recall:** fraction of its epochs assigned the correct
  label. Uniform random predictions give 20% expected recall, but a model that
  guesses one label frequently can have high recall for that label without
  discriminating it. This is not a stand-alone test of waveform information.
- **Target-versus-rest balanced accuracy:** average of sensitivity and specificity;
  nominal reference 50%. E015 predictions can be converted into this form by
  treating the target label as positive and all four other labels as negative.
  This permits a matched comparison with E016's separately fitted binary detector.

The tables below compare repeatability-selected features. Both model formulations
use the same 100 features per source fold and exactly the same held-out events.

## Strongest target-level leads

| Source | Participant | Position | Target | E015 target/rest BA | E016 binary BA | Target support |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| Strict | 6 | Second | Triangle | 69.90% | 68.31% | 13 of 59 epochs |
| Exploratory | 6 | Second | Triangle | 57.27% | 66.05% | 13 of 59 epochs |
| Exploratory | 6 | First | Sine | 60.28% | 67.55% | 12 of 59 epochs |
| Exploratory | 6 | First | Linear decrease | 60.20% | 64.36% | 12 of 59 epochs |

For strict participant-6 second-position Triangle:

| Model | True positives | False positives | Recall | Specificity | Precision |
| --- | ---: | ---: | ---: | ---: | ---: |
| E015 multiclass, converted to target/rest | 8 / 13 | 10 / 46 | 61.54% | 78.26% | 44.44% |
| E016 binary | 9 / 13 | 15 / 46 | 69.23% | 67.39% | 37.50% |

The binary model detects one extra Triangle epoch but also produces five extra
false positives. Its higher recall does not give higher balanced accuracy here.
The strict binary within-comparison AUC is 0.8333, but only two of ten held-out
comparisons contain both target and rest epochs. Its pooled AUC is 0.7358;
these AUCs compare different sets of score pairs.

The large Triangle recall depends on representation: strict ANOVA-selected E015
features give 3/13 correct Triangle epochs (23.08% recall), compared with 8/13
for strict repeatability selection. This is a lead for checking representation,
not an independent confirmation of one selector's superiority.

## Participants and overall five-class decoding

The highest repeatability-selected five-class point estimates are:

| Source | Participant | Position | Five-class BA | Epochs |
| --- | ---: | --- | ---: | ---: |
| Exploratory | 6 | First | 25.47% | 59 |
| Exploratory | 8 | First | 23.76% | 64 |
| Strict | 6 | Second | 25.32% | 59 |
| Strict | 9 | First | 23.17% | 65 |
| Strict | 5 | Second | 20.17% | 70 |

Thus 2/20 exploratory and 3/11 strict repeatability-selected participant/position
models exceed 20%. ANOVA has 4/20 exploratory cases above 20%, with a maximum
of 25.41% for participant 8 at the second position; none of its 11 strict cases
exceed 20%. These are counts of observed point estimates, not discoveries.

Participant 6 has the highest exploratory binary mean across all ten target ×
position detectors, but that mean is only **50.65%**. Its first-position mean is
54.38%, whereas its second-position mean is 46.92%. A strong individual target
therefore does not imply good decoding of all five waveforms.

## Consistency and cohort-level waveform effects

- Among 100 exploratory target conditions, E015 target/rest BA exceeds 50% in
  **21**, and E016 binary BA exceeds 50% in **19**.
- Among 55 strict target conditions, both formulations exceed 50% in **22** cases
  each. These are separate, overlapping selections; counts alone do not show the
  same detector is above the reference in both formulations.
- The only exploratory binary participant/target with both positions above 50%
  is **participant 8, linear decrease: 54.69% / 57.32%**. Strict binary counterparts
  include participant 6 linear decrease (53.90% / 59.77%) and several weaker
  Sine/Square cases. None constitutes statistical replication.
- No exploratory waveform/position mean binary BA exceeds 50%. Strict second-
  position Sine and Square means are slightly above it, **51.03% and 51.54%**,
  on five participants. The other strict waveform/position means are below 50%.
- All E015 repeatability-selected cohort mean target/rest BAs are below 50%.
  There is no cohort-wide evidence that one waveform is generally easy to detect.
- Binary fitting changes mean matched target/rest BA by **+0.26 percentage points**
  in the exploratory set and **−0.02 points** in the strict set. It helps some
  conditions, but gives no clear overall gain.

Eight exploratory and three strict multiclass target conditions have recall above
20% while target/rest BA is at or below 50%. This directly illustrates why recall
alone can overstate apparent class-specific discrimination.

## Inference and next priority

All subgroup leads were identified after inspecting many conditions. No E015/E016
permutation inference has run; model agreement reuses the same trials and is not
independent replication. Strict/exploratory subsets and fitted representations
also differ. Cross-validation predictions share training data, and epochs within
comparisons are dependent; an independent-trial binomial test would not adequately
represent this analysis design.

The next priority is to validate the participant-6 Triangle lead while accounting
for the full inspected search: specify a defensible null respecting the paired
acquisition/order structure, repeat label-dependent spatial fitting and feature
selection within training data, and control the multiple comparisons. A fixed-
feature label shuffle would omit the selection stage. Permutation inference
tests performance against an empirical null rather than declaring an individual
point estimate significant from its nominal reference alone.
[Scikit-learn permutation-test documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.permutation_test_score.html)
describes that general approach; its generic shuffle needs adaptation to this design.

## Reproducibility

The audit verified the frozen numerical inputs and source hashes, recomputed E015
confusions and all 155 E016 detector metrics, and checked all 155 matched target-
level BA values against the saved E015 reference. Numerical models and predictions
are unchanged. All feature selectors, including ANOVA, remain in the exported tables.

Run from the project root:

```powershell
& ./.venv/Scripts/python.exe -m experiments.waveform_decoding.e016_one_vs_rest_decoding.subgroup_audit
```

Use `--output` for a fresh directory on rerun. Local output:
`output/e015_e016_subgroup_audit_20261102/`, including per-target metrics,
matched models, waveform/participant summaries, both-position checks and verification.

See [E015 findings](E015_OBSERVED_RESULTS.md) and [E016 findings](E016_BINARY_RESULTS.md).
