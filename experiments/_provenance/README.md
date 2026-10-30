# Experiment relocation provenance - 30 October 2026

The ten experiment packages moved from `experiments/task4_*` to category folders
with stable E001-E010 IDs. Current commands, imports and public Markdown links
use the new layout. The [central registry](../README.md) records completed and
planned experiments; [registry.json](../registry.json) retains old package names
and canonical local output paths.

## Historical records stay immutable

No raw EEG, master CSV, fitted ICA, numerical result, saved configuration, cache,
research PDF or local output directory was renamed or overwritten by this
migration. Existing `output/task4_*` report links stay valid. Their recorded paths
and hashes describe the original runs and are not rewritten to imply a rerun.

`migration_20261030.json` records every original Python module's old path, new
path, original SHA256, exact migrated SHA256 and frozen source snapshot. The
content-addressed `sources/<original-sha256>.source` files preserve original bytes;
`.gitattributes` disables text conversion for them. They are code-only provenance
artifacts, not executable experiment packages or participant records. They are
not imported during current analyses. A historical source can be inspected or
restored in an isolated checkout using the manifest; the snapshots alone are not
a complete dependency/environment archive.

## Strict compatibility, rather than bypassed checks

`experiments/provenance.py` resolves only documented package moves. A source check
accepts either the exact recorded digest or a manifest entry for which both the
frozen original bytes and exact migrated current bytes match. Any further edit
fails that historical check. New runs record real current source digests and
paths. Source changes made during a new numerical run still fail its original
end-of-run checks.

Changes are package/import/path rewrites and the source-verification/cache-lookup
glue required by relocation. Numerical estimator code, preprocessing settings,
event labels, thresholds, folds and statistical definitions are unchanged.

Saved ICA reconstruction checks still verify the original preprocessing,
dependencies, raw/CSV hashes, fitted ICA and reproduced features. The three
sources defining original ICA-cache identity retain their original bytes.
The raw epoch-cache lookup additionally accepts an older cache only when all
non-source provenance is identical and every recorded/current source passes the
strict migration check. Multiple equivalent candidates fail rather than being
chosen silently. Future source edits beyond this manifest produce a new cache
identity instead of silently reusing historical results.

The manifest is an explicit reviewed compatibility record, not an authorization
to accept future scientific changes. If methods change, create a new experiment
or run and record its actual provenance. Do not replace manifest digests merely
to make a changed analysis pass a historical validation.
