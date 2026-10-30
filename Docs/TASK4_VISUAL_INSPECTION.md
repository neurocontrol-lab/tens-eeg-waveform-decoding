# Task 4: simple-pipeline visual inspection

**30 October 2026.** Completed descriptive waveform-by-waveform EEG inspection
for subjects 3-12, comparing a baseline-matched minimally processed view with a
fixed 0.5-30 Hz band-pass view. No classifier was fitted or changed, no new
repeatability estimate was calculated, and no visual peak selected a classifier
electrode/window. Earlier analyses and the research-note PDF are unchanged.

## Which processing did the classifiers use?

The original voltage, xDAWN, spectral-power and combined classifiers used the
**minimal baseline**: verified-unit XDF EEG, fixed average reference over 62 scalp
channels, 14 predefined electrodes and temporal baseline subtraction. No extra
temporal filtering, ICA, notch, onset interpolation or resampling was applied.
Acquisition filtering may already be present. xDAWN spatial filters were learned
inside training folds; they are separate from supplied ICA preprocessing.
Power features used epoch-local detrending and a Hann periodogram, not a
band-pass-filtered EEG preprocessing branch.

The **separate response diagnostics** used the supplied filtering/ICA adaptation.
The subsequent preprocessing comparison explicitly compared minimal and supplied
branches for descriptive response measures. Those were not new classifiers.

## Visual pipeline and cohort

Both views use the same original classifier electrodes and complete pairs.
Minimal is average-reference plus [-250,-50)-ms baseline. Bandpass adds only a
fixed Butterworth SOS 0.5-30 Hz filter (prototype order four, eight band-pass
poles per pass), forward/backward on continuous finite, timestamp-regular segments,
then the same baseline. No ICA, notch, resampling, onset interpolation, fitted
artifact rejection or new amplitude threshold was used.

Epochs span [-300,1200) ms. Eight-second recording/segment-edge guards and checks
of finite, regular, neighbor-free extended epoch support apply identically to
both views. The original 1,244 eligible events in 622 pairs become **1,238 events
in 619 pairs**: three additional complete pairs are excluded because a member
falls within the fixed filter-edge guard. No additional pairs were excluded on
amplitude. All original and additional reasons remain in local manifests.

Reconstructed minimal-view 50-500-ms voltage features match the saved baseline
on common events, with maximum absolute error **2.46e-11 uV**. XDF hashes match
the completed baseline-verified epoch-cache provenance; original event sequences
match CSV labels exactly. The cache supplies provenance, not preprocessed EEG or
ICA transforms. All recorded input/source hashes remained unchanged during the
preparation run.

Within waveform/position, evoked overlays give observed comparison cells equal
weight. Group curves give each participant equal weight. Different waveforms can
have different comparison support; these overlays are not regression-adjusted
effects. Individual and group views both remain available.

## Inspection views and observations

The local HTML report contains:

- Fourteen-electrode overlays of all five waveform conditions, separately for
  first/second presentations and both processing views, for every participant.
- Equal-participant group overlays with pointwise descriptive 95% bootstrap
  bands (1,000 participant resamples), plus waveform-minus-common-mean traces.
- Chronological single-trial C3/C4 images by waveform and position, without
  amplitude or latency sorting. The color limit is shared across both branches
  within a participant; clipping is for display only and its fraction is saved.
- Separate [-20,50]-ms onset panels retaining actual peri-trigger amplitudes.
- Filter frequency response, per-channel amplitude audits, cell counts and all
  additional support exclusions.

The inspected group overlays contain broad deflections shared by several
waveforms, particularly in some second-presentation central/parietal traces.
Waveform deviations cross over time and do not present an obvious, uniform
five-class visual separation. Participant overlays and single-trial images show
substantial heterogeneity, including large slow excursions in some recordings.
These are qualitative observations, not significance tests, repeatability
estimates, confirmed position effects or evidence of neural/artifact origin.
No participant, electrode or trial was removed after viewing the plots.

Band-pass filtering smooths high-frequency fluctuations but leaves some large
slow excursions. Smoother appearance is not evidence of improved decoding or
validated artifact removal. Main overlays/images omit [0,50) ms for readability,
mark that omission, and retain onset amplitudes in separate panels. The omission
does not make later activity artifact-free: noncausal filtering can spread
stimulation transients both before and after onset. See
[MNE's filtering background](https://mne.tools/stable/auto_tutorials/preprocessing/25_background_filtering.html).

Pre-trigger activity can contain preceding-stimulus responses/anticipation;
later activity can include decisions or movement. Physical trigger-to-current
timing remains unverified. Bootstrap bands are pointwise descriptive uncertainty,
not corrected significance regions. Averaged-response plots do not establish
single-epoch discriminability. Any later channel/window selection must occur
within training folds rather than use these full-cohort maps to select features
for the same held-out evaluation.

## Reproduction and verification

Canonical local output: Git-ignored
`output/task4_visual_inspection_20261030/index.html`, with numerical epochs,
summaries, manifests, input hashes and figures. Participant-specific information
and plots remain excluded from Git. Run instructions are in
[the experiment README](../experiments/signal_quality/e010_simple_visual_inspection/README.md).

All **66 project tests passed**, including four new checks of filter isolation
across timestamp gaps, physical units/baseline and frequency behavior, whole-pair
extended-support eligibility, and equal-comparison-cell averaging. The full test
suite used a fresh project-local temporary directory because the existing system
pytest temporary root was inaccessible. Figures were rendered and inspected;
the HTML links resolve to generated images. No new classifier accuracy or
electrode-by-bin repeatability conclusion is reported here.
