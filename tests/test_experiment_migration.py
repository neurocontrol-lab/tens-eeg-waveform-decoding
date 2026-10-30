"""Scientific package identity and fail-closed relocation provenance checks."""
import ast
import hashlib
import importlib
import json
from pathlib import Path
import re
import pytest
from experiments.provenance import (ROOT, MANIFEST, canonical_source_name,
    recorded_source_digest, source_matches, equivalent_cache_provenance)


def test_all_registered_completed_packages_import_and_planned_ids_are_only_reserved():
    registry = json.loads((ROOT/'experiments/registry.json').read_text())['experiments']
    assert len({e['id'] for e in registry}) == len(registry)
    assert [e['id'] for e in registry] == [f'E{i:03d}' for i in range(1,len(registry)+1)]
    for e in registry:
        if e['package'] is not None:
            assert e['dataset'] == 'Task 4'
            importlib.import_module(e['package'].replace('/','.'))
            assert (ROOT/e['package']/'README.md').is_file()
            assert e['status'] in {'In progress','Completed','Blocked','Superseded'}
            if e['status'] == 'Completed':
                assert (ROOT/e['findings']).is_file()
        else:
            assert e['status'] == 'Planned' and e['package'] is None
            assert not (ROOT/e['reserved_package']).exists()


def test_every_migrated_source_and_frozen_original_matches_manifest():
    manifest = json.loads(MANIFEST.read_text())
    assert len(manifest['packages']) == 10
    assert manifest['verification_helper_sha256'] == hashlib.sha256((ROOT/'experiments/provenance.py').read_bytes()).hexdigest()
    for old, record in manifest['files'].items():
        assert source_matches(old, record['original_sha256']), old
        assert source_matches(record['current_path'], record['migrated_sha256'])
        assert hashlib.sha256((ROOT/record['snapshot']).read_bytes()).hexdigest() == record['original_sha256']
        assert not source_matches(old, '0'*64)


def test_later_edits_and_corrupt_originals_are_rejected(tmp_path):
    old = b'original source\n'
    new = b'migrated source\n'
    original_sha = hashlib.sha256(old).hexdigest()
    current_sha = hashlib.sha256(new).hexdigest()
    current = tmp_path/'experiments/category/e001_example/module.py'
    current.parent.mkdir(parents=True)
    current.write_bytes(new)
    frozen = tmp_path/'frozen.source'
    frozen.write_bytes(old)
    manifest = dict(packages={'experiments/task4_example':'experiments/category/e001_example'},
        files={'experiments/task4_example/module.py':dict(current_path='experiments/category/e001_example/module.py',
            original_sha256=original_sha, migrated_sha256=current_sha,snapshot='frozen.source')})
    assert source_matches('experiments/task4_example/module.py', original_sha,manifest=manifest,root=tmp_path)
    current.write_bytes(new+b'changed estimator\n')
    assert not source_matches('experiments/task4_example/module.py', original_sha,manifest=manifest,root=tmp_path)
    current.write_bytes(new)
    frozen.write_bytes(b'corrupt original\n')
    assert not source_matches('experiments/task4_example/module.py', original_sha,manifest=manifest,root=tmp_path)


def test_recorded_windows_paths_and_duplicates_have_strict_identity():
    old = 'experiments/task4_multiscale/analysis.py'
    current = 'experiments/response_characterization/e006_multiscale_responses/analysis.py'
    assert canonical_source_name('C:\\old-root\\'+old.replace('/','\\')) == current
    assert recorded_source_digest({'C:/old-root/'+old:'abc'},current) == 'abc'
    with pytest.raises(ValueError,match='unambiguous'):
        recorded_source_digest({old:'a',current:'b'}, current)


def test_legacy_cache_requires_identical_inputs_and_exact_migration_sources():
    manifest = json.loads(MANIFEST.read_text())
    old = 'experiments/task4_xdawn/data.py'
    record = manifest['files'][old]
    previous = dict(subject=3, input_sha256='input',dependencies={'numpy':'fixed'},
        implementation={old:record['original_sha256']})
    current = dict(previous,implementation={record['current_path']:record['migrated_sha256']})
    assert equivalent_cache_provenance(previous,current)
    assert not equivalent_cache_provenance(previous,dict(current,input_sha256='different'))
    assert not equivalent_cache_provenance(previous,dict(current,dependencies={'numpy':'different'}))
    assert not equivalent_cache_provenance(previous,dict(current,implementation={record['current_path']:'0'*64}))


def test_core_numerical_algorithms_changed_only_import_paths():
    manifest = json.loads(MANIFEST.read_text())
    modules = [
        'experiments/task4_baseline/events.py', 'experiments/task4_baseline/features.py',
        'experiments/task4_baseline/evaluate.py',
        'experiments/task4_response_diagnostics/preprocess.py',
        'experiments/task4_response_diagnostics/analysis.py',
        'experiments/task4_multiscale/analysis.py',
        'experiments/task4_time_resolved/analysis.py',
        'experiments/task4_xdawn/core.py', 'experiments/task4_spectral/core.py',
        'experiments/task4_response_followup/core.py',
        'experiments/task4_visual_inspection/core.py']
    for old in modules:
        record = manifest['files'][old]
        original = (ROOT/record['snapshot']).read_text(encoding='utf-8')
        for previous,new in manifest['packages'].items():
            original = original.replace(previous.replace('/','.'),new.replace('/','.'))
            original = original.replace(previous,new)
        actual = (ROOT/record['current_path']).read_text(encoding='utf-8')
        assert ast.dump(ast.parse(original)) == ast.dump(ast.parse(actual)), old


def test_public_documentation_links_and_central_status_resolve():
    pages = [ROOT/'README.md',ROOT/'PROJECT_REVIEW.md',* (ROOT/'Docs').glob('*.md'),
        *[p for p in (ROOT/'experiments').rglob('README.md') if '_provenance/sources' not in p.as_posix()]]
    missing = []
    for page in pages:
        for target in re.findall(r'(?<!!)\[[^\]]*\]\(([^)]+)\)',page.read_text(encoding='utf-8')):
            if '://' in target or target.startswith('#'):
                continue
            target = target.split('#')[0].strip('<>')
            if target and not (page.parent/target).exists():
                missing.append(f'{page.relative_to(ROOT)} -> {target}')
    assert not missing, missing
