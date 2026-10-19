# Experiments

This directory contains executable analysis pipelines and experiment versions. Run them as Python modules from the repository root so local data and output defaults resolve consistently.

| Experiment | Purpose | Run guide |
| --- | --- | --- |
| `task4_baseline` | Initial fixed-feature waveform classifier and diagnostic controls | [README](task4_baseline/README.md) |

```powershell
& ./.venv/Scripts/python.exe -m experiments.task4_baseline --subjects 3 --audit-only --output output/task4_audit_new
```

## Future experiments

- Add a sibling package with a descriptive name, for example `task4_v2_signal_quality`, when methods or preprocessing materially change. The name is an example, not an implemented pipeline.
- Give each experiment a `README.md`, an executable `__main__.py`, explicit settings, and tests. Document the hypothesis, differences from the baseline, cohort/exclusions, features, and validation design.
- Preserve the original baseline as a comparison. Do not replace its saved outputs when running a new version; use a fresh, clearly named directory under `output/`.
- Keep recordings in `EEGData/` and derived participant data under Git-ignored `output/`. Keep research PDFs and reference material in `Docs/`.
- Root-level `requirements-baseline.txt` and its lock file describe the baseline environment. Document any new environment requirements when adding a different experiment.

The baseline moved here on 20 October 2026. Existing saved run configuration and hashes describe the original execution and remain unchanged; this directory move does not constitute a new analysis result.
