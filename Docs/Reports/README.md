# Shared analysis reports

- [E015 multiclass decoding and class recall](E015/index.html): interactive confusion
  matrices, first/second presentations, participant results and explanations of
  ANOVA versus repeatability selection. [Open online](https://htmlpreview.github.io/?https://github.com/neurocontrol-lab/tens-eeg-waveform-decoding/blob/main/Docs/Reports/E015/index.html).
- [E016 waveform-versus-rest detectors](E016/index.html): all 100 exploratory
  detectors and 55 strict references, with balanced accuracy, sensitivity,
  specificity and AUC diagnostics. [Open online](https://htmlpreview.github.io/?https://github.com/neurocontrol-lab/tens-eeg-waveform-decoding/blob/main/Docs/Reports/E016/index.html).
- [Waveform and participant subgroup findings](../E015_E016_SUBGROUP_FINDINGS.md)
  compare E015 and E016 and identify exploratory leads.

- [E014 waveform-repeatability explorer](E014/index.html): pre-rendered maps and
  electrode/component traces, with participant, waveform, presentation, feature
  and bin choices preserved. [Open interactive report online](https://htmlpreview.github.io/?https://github.com/neurocontrol-lab/tens-eeg-waveform-decoding/blob/main/Docs/Reports/E014/index.html) · [Open static gallery online](https://htmlpreview.github.io/?https://github.com/neurocontrol-lab/tens-eeg-waveform-decoding/blob/main/Docs/Reports/E014/gallery.html) · [Local gallery file](E014/gallery.html).

E015 and E016 are **self-contained HTML files**, each smaller than 1 MB. They embed
their figures and summary-table downloads and need no companion files or JavaScript
library downloads. E015 embeds the MIT-licensed markdown-it renderer so the full
methods and findings display as formatted Markdown, including tables and links,
while remaining usable offline. They contain summary statistics, not raw EEG, trial-level prediction
tables or fitted models. Rebuild them with `python -m experiments.publish_decoding_reports`
after regenerating the local numerical summaries. The publication manifest records
the source and embedded-asset hashes.

E014 uses a separate `images/` folder; keep it beside its HTML when copying or
hosting that report. Full EEG, numerical map arrays, fitted spatial filters and
local caches remain Git-ignored. Exporting a report does not itself deploy it online.
