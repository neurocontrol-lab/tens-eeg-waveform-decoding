"""Strict source verification across the documented 2026-10-30 relocation.

Historical run records are immutable. An old digest is accepted only when the
archived original bytes and the exact migrated current bytes match the manifest.
Further edits are not implicitly accepted. New runs record actual current hashes.
"""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT/'experiments/_provenance/migration_20261030.json'


def _manifest():
    return json.loads(MANIFEST.read_text(encoding='utf-8'))


def canonical_source_name(path, manifest=None):
    """Normalize recorded Windows/POSIX roots and the declared package moves."""
    manifest = _manifest() if manifest is None else manifest
    value = str(path).replace('\\', '/')
    match = re.search(r'(?:^|/)(experiments/.*)$', value, flags=re.IGNORECASE)
    if match:
        value = match.group(1)
    for old, new in manifest['packages'].items():
        if value.casefold() == old.casefold() or value.casefold().startswith(old.casefold()+'/'):
            value = new+value[len(old):]
            break
    return value.casefold()


def resolve_source_path(path, manifest=None):
    manifest = _manifest() if manifest is None else manifest
    canonical = canonical_source_name(path, manifest)
    for record in manifest['files'].values():
        if canonical == record['current_path'].casefold():
            return ROOT/record['current_path']
    # Non-migrated paths (including temporary test sources) retain normal semantics.
    original = Path(path)
    return original if original.is_absolute() else ROOT/original


def source_matches(path, expected, *, manifest=None, root=None):
    """Accept an exact hash or one explicit, byte-verified migration pair only."""
    manifest = _manifest() if manifest is None else manifest
    root = ROOT if root is None else Path(root)
    canonical = canonical_source_name(path, manifest)
    records = [r for r in manifest['files'].values() if r['current_path'].casefold() == canonical]
    if len(records) > 1:
        raise ValueError(f'Ambiguous migrated source: {path}')
    current = root/records[0]['current_path'] if records else resolve_source_path(path, manifest)
    if not current.is_file():
        return False
    actual = hashlib.sha256(current.read_bytes()).hexdigest()
    if actual == expected:
        return True
    if not records:
        return False
    record = records[0]
    if record['original_sha256'] != expected or record.get('migrated_sha256') != actual:
        return False
    frozen = root/record['snapshot']
    return frozen.is_file() and hashlib.sha256(frozen.read_bytes()).hexdigest() == expected


def recorded_source_digest(implementation, filename):
    """Find exactly one recorded digest, across relative/absolute legacy paths."""
    target = canonical_source_name(filename)
    matches = [value for key, value in implementation.items() if canonical_source_name(key) == target]
    if len(matches) != 1:
        raise ValueError(f'Need one unambiguous prior source digest for {filename}')
    return matches[0]


def equivalent_cache_provenance(recorded, current):
    """Reuse a legacy epoch cache only for otherwise identical inputs/settings.

    New source edits beyond the explicit migration produce a new cache key.
    This does not fit or apply any ICA transform.
    """
    if {k:v for k,v in recorded.items() if k != 'implementation'} != {
            k:v for k,v in current.items() if k != 'implementation'}:
        return False
    old_sources = recorded.get('implementation', {})
    new_sources = current.get('implementation', {})
    if not old_sources or {canonical_source_name(k) for k in old_sources} != {
            canonical_source_name(k) for k in new_sources}:
        return False
    return all(source_matches(path, digest) for path,digest in old_sources.items()) and all(
        source_matches(path, digest) for path,digest in new_sources.items())
