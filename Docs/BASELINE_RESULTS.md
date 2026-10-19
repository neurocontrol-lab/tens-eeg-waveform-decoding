# Initial Task 4 waveform baseline: 2026-10-16

The first fixed-feature baseline did **not** demonstrate reliable above-chance waveform decoding. This is a result for the specific minimal pipeline below, not evidence that the recordings contain no decodable waveform information.

## Dataset and event audit

- Ten participants, subjects 3-12; subject 2 excluded per Jason's protocol clarification.
- All 1,246 debounced stimulation events matched the complete CSV sequence exactly. No participant was silently skipped or resynchronized.
- Accepted pulses were checked in saved per-participant run-duration audits. Unexpected/short trigger runs were retained as rejected diagnostics.
- All comparison blocks passed the race-to-five consistency audit. Observed within-pair intervals ranged from 1.408 to 1.448 seconds.
- One EEG pair failed the prespecified 500 microvolt peak-to-peak threshold; 1,244 events remained eligible. All current-protocol naturalness responses in the matched cohort were valid. Zero-filled legacy detected-waveform values were ignored.

## Model and evaluation

The primary features are 50 ms bin means from +50 to +500 ms on 14 fixed scalp channels, referenced to each time sample's scalp average and to a -250 to -50 ms baseline. Training-fold scaling and shrinkage LDA use no tuning. There is no temporal filter, ICA, interpolation, or source localization. Pre-stimulus features have equal duration and their own earlier baseline; post-trigger samples cannot leak backward through filtering.

Within-participant validation holds out complete waveform comparisons, preserving paired presentations and repeated comparison groups. This tests generalization across pair contexts and is stricter than randomly splitting nearby trials. Leave-one-subject-out (LOSO) evaluation holds out each complete participant. Potential shared-sample support is purged from within-participant training folds. Scores pool out-of-fold predictions over five classes, rather than averaging fold scores with different available test classes.

| Features | Mean within-participant balanced accuracy | Pooled LOSO balanced accuracy |
| --- | --- | --- |
| Post-stimulus, 50-500 ms (primary) | 19.1% | 18.5% |
| Pre-stimulus, -500 to -50 ms | 16.8% | 19.3% |
| Late, 150-500 ms | 21.9% | 19.8% |
| Onset, 0-20 ms | 16.1% | 20.7% |
| Order/history only, no EEG | 10.5% | 19.5% |

The descriptive participant-bootstrap 95% interval for the mean within-participant primary score is 15.7-22.4%. This interval includes the 20% reference. The primary LOSO score's diagnostic restricted-permutation p-value is 0.87 (99 permutations). Restricted permutations preserve comparison membership and paired structure; they are not an iid trial-label test. Other models/participant comparisons are exploratory and uncorrected for multiplicity. The same permutation index is not an independent cohort-level randomization test, so individual nulls are not pooled into a cohort p-value.

## Interpretation

There is no evidence here of easy separation with these predefined mean-amplitude features. Pre-stimulus performance is not systematically stronger than the primary signal across the cohort; the slightly higher pooled LOSO pre score is itself near chance. That does not establish absence of contamination. Onset and late-window results also do not establish neural versus electrical origin.

The low within-participant order/history score is specific to holding out entire comparisons: a relationship learned in one pair context may fail in an unseen context. It should not be presented as proof that order/history confounds are absent. Its previous-trigger input is label-related contextual information, and its restricted-permutation distribution is only a diagnostic comparator, not a clean physical randomization of acquisition order.

The pipeline deliberately retains limitations of minimally processed EEG: no adaptive bad-channel handling, blink/muscle removal, or line-noise filtering; unknown trigger-to-current latency; possible preceding-stimulus carryover; and non-charge-matched conditions. These results neither isolate waveform shape from intensity/charge nor assess naturalness decoding.

## Reproduction and next work

See [the baseline run guide](../experiments/task4_baseline/README.md). Exact package versions are in `requirements-baseline-lock.txt`. Local `output/task4_baseline/` contains full metrics, confusion matrices, probabilities, manifests, configuration/source hashes, and permutation distributions. `output/task4_audit_confirmation/` independently reran final extraction/QC; all feature arrays were identical to those used for the classifier run, recorded in `extraction_validation.json`. The package moved into `experiments/` on 20 October; the saved original analysis and this result summary remain the reference.

Next inspect signal quality, channel-level responses, first/second presentation results, and covariance/feature scaling before changing the model. Any revised processing/features should be labeled a new exploratory experiment with fresh outputs and the same leakage controls. Timing validation with the saved monitor signal remains useful. Participant-level outputs remain ignored by Git; this note contains cohort aggregates only.
