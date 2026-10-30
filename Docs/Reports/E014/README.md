# E014 image publication

[Open the interactive report online](https://htmlpreview.github.io/?https://github.com/neurocontrol-lab/tens-eeg-waveform-decoding/blob/main/Docs/Reports/E014/index.html) using HTMLPreview.

Open index.html for the image explorer, or gallery.html for script-free direct image links. Keep the images/ directory beside both pages when copying or hosting this folder. All participant, waveform, first/second presentation, representation, bin and electrode/component choices are preserved. The page crops pre-rendered image sheets; it does not fetch numerical map arrays or trial EEG. Tiny joint-correlation/support summaries remain embedded in HTML.

Images use a fixed display raster and discrete palette, then lossless WebP compression. Images cannot provide exact numeric readout at arbitrary samples. Native bin widths and full-precision estimates remain in the separate local numerical report. Statistics are not recomputed or selected for this edition. See the expandable methods in index.html and source/image digests in manifest.json.

Regenerate from the project root with:

```powershell
& ./.venv/Scripts/python.exe -m experiments.response_characterization.e014_all_electrode_repeatability.image_report --output Docs/Reports/E014
```

Choose a fresh output path for regeneration; existing publications are not overwritten. This folder is Git-trackable; the large output/e014_repeatability_20261030 numerical payloads remain ignored. No deployment is performed by exporting.

This edition contains 8,800 image sheets across all ten participants and all 44 feature/bin combinations, in both response views. Component map labels use c1/c2 as short forms of component 1/2; dropdowns and curve sheets retain full names. The complete image archive is approximately 819.8 MB. Browsing loads only the selected map and curve sheet, not the full archive.
