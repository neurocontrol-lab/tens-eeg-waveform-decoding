"""Run fixed time-resolved effects with verified original ICA reconstructions."""
import argparse
import importlib.metadata
import json
from pathlib import Path
from experiments.provenance import source_matches
import traceback
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from experiments.signal_quality.e007_preprocessing_comparison.__main__ import load_inputs
from experiments.signal_quality.e007_preprocessing_comparison.analysis import source_digest
from experiments.response_characterization.e002_electrode_position_slope.__main__ import sha, write_json
from experiments.response_characterization.e006_multiscale_responses.analysis import transform, window_features, condition_weights
from .analysis import (SCALES_MS, INTERVALS, BRANCHES, FAMILIES, specifications,
                       interval_masks, regression_operator, fit_trace)

PLOT_CHANNELS = ('C3', 'C4', 'CP1', 'Pz')


def prepare(subject, args, config):
    destination = args.output/f'sub-{subject}'
    destination.mkdir()
    print(f'Subject {subject}: reconstruct verified minimal/supplied inputs', flush=True)
    branches, times, channels, meta, audit = load_inputs(subject, args, config)
    masks = interval_masks(times, audit['sfreq'])
    operator, slices, design = regression_operator(meta)
    effects = {key: [] for key in FAMILIES}
    display = []
    means_weight = condition_weights(meta).reshape(10, len(meta))
    plot_ix = [channels.index(c) for c in PLOT_CHANNELS]
    with np.load(args.prior/f'sub-{subject}/summaries.npz') as old:
        prior_features = old['features']
        if old['channels'].tolist() != channels:
            raise ValueError('Prior channels differ')
    max_error = 0.
    for branch in BRANCHES:
        print(f'  {branch}: all five fixed scales, every electrode/time point', flush=True)
        branch_effects = {key: [] for key in FAMILIES}
        branch_display = []
        for period, mask in masks.items():
            period_effects = {key: [] for key in FAMILIES}
            period_display = []
            for spec in specifications():
                # Continuous reconstruction and actual padding precede local filtering.
                values = transform(branches[branch][0], audit['sfreq'], spec)
                if branch == 'cleaned' and period == 'post' and spec['sigma_ms'] in (5.,10.,20.):
                    old_specs = config['prior_config']['representations']
                    index = next(i for i,s in enumerate(old_specs) if s['name'] == spec['name'])
                    actual = window_features(values, times)
                    expected = prior_features[:,2*index:2*index+2]
                    np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=2e-5)
                    max_error = max(max_error, float(np.max(np.abs(actual-expected))))
                selected = values[:, :, mask]
                fitted = fit_trace(selected, operator, slices)
                for key in FAMILIES:
                    period_effects[key].append(fitted[key])
                d = selected[:, plot_ix]
                period_display.append((means_weight @ d.reshape(len(meta), -1)).reshape(
                    5, 2, len(plot_ix), -1).astype(np.float32))
                del values, selected, d, fitted
            for key in FAMILIES:
                # contrast × scale × time × channel
                branch_effects[key].append(np.stack(period_effects[key], axis=1))
            branch_display.append(period_display)
        for key in FAMILIES:
            # contrast × period × scale × time × channel
            effects[key].append(np.stack(branch_effects[key], axis=1))
        display.append(branch_display)
    effects = {key: np.stack(value, axis=1) for key,value in effects.items()}
    np.savez_compressed(destination/'effects.npz', **effects,
        channels=np.asarray(channels), plot_channels=np.asarray(PLOT_CHANNELS),
        pre_times=times[masks['pre']], post_times=times[masks['post']],
        evoked_display=np.asarray(display))
    meta.to_csv(destination/'events.csv', index=False)
    audit.update(design=design, prior_feature_max_error=max_error,
        numerical_dtype='float64', display_dtype='float32',
        n_time_points_per_period=int(masks['post'].sum()),
        previous_preprocessing_audit_sha256=sha(args.comparison/f'sub-{subject}/audit.json'))
    old_audit = json.loads((args.comparison/f'sub-{subject}/audit.json').read_text())
    for key in ('source_sha256','csv_sha256','saved_ica_sha256','n_events'):
        if audit[key] != old_audit[key]:
            raise ValueError(f'Previous comparison differs: {key}')
    previous_meta = pd.read_csv(args.comparison/f'sub-{subject}/events.csv')
    for key in ('sample','trigger','pair_id','position'):
        if not np.array_equal(meta[key], previous_meta[key]):
            raise ValueError(f'Comparison cohort differs: {key}')
    write_json(destination/'audit.json', audit)
    write_json(destination/'complete.json', dict(subject=subject))
    print(f'  Completed {subject}: {len(meta)} events, feature error {max_error:.3g}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subjects', default=','.join(map(str,range(3,13))))
    parser.add_argument('--prior', type=Path, default=Path('output/task4_multiscale_20261023'))
    parser.add_argument('--comparison', type=Path, default=Path('output/task4_preprocessing_comparison_final_20261023'))
    parser.add_argument('--ica-cache', type=Path, default=Path('output/task4_response_cache'))
    parser.add_argument('--data-dir', type=Path, default=Path('EEGData'))
    parser.add_argument('--master-csv', type=Path, default=Path('EEG_Codes_Export/02_SourceLocalization_Task4/Master_Data_ToSub12 (1).csv'))
    parser.add_argument('--output', type=Path, default=Path('output/task4_time_resolved_20261023'))
    args = parser.parse_args()
    subjects = [int(s) for s in args.subjects.split(',')]
    if not subjects or len(set(subjects)) != len(subjects) or not set(subjects) <= set(range(3,13)):
        parser.error('Choose unique subjects 3–12')
    if args.output.exists():
        parser.error('Use a fresh output directory')
    prior_config = json.loads((args.prior/'config.json').read_text())
    comparison_config = json.loads((args.comparison/'config.json').read_text())
    for base in (args.prior,args.comparison):
        status = json.loads((base/'status.json').read_text())
        if status['errors'] or not set(subjects) <= set(status['completed']):
            parser.error(f'Incomplete prior analysis: {base}')
    # The verified loader itself must be identical to the completed comparison.
    for file in ('experiments/signal_quality/e007_preprocessing_comparison/__main__.py',
                 'experiments/signal_quality/e007_preprocessing_comparison/analysis.py'):
        if not source_matches(file, source_digest(comparison_config['implementation'],file)):
            parser.error(f'Previous reconstruction code changed: {file}')
    sources = list(Path(__file__).parent.glob('*.py'))+[Path(p) for p in (
        'experiments/signal_quality/e007_preprocessing_comparison/__main__.py',
        'experiments/signal_quality/e007_preprocessing_comparison/analysis.py',
        'experiments/response_characterization/e006_multiscale_responses/analysis.py',
        'experiments/response_characterization/e002_electrode_position_slope/analysis.py',
        'experiments/response_characterization/e002_electrode_position_slope/preprocess.py','experiments/waveform_decoding/e001_voltage_baseline/events.py')]
    args.output.mkdir(parents=True)
    config = dict(subjects=subjects, prior_config=prior_config, scales_ms=SCALES_MS,
        representations=specifications(), intervals=INTERVALS, branches=BRANCHES,
        families=FAMILIES, epoch_support=[-.9,.65], truncate=4.,
        cohort='Frozen 1242-event multiscale/comparison cohort; fail on any new padding exclusions',
        model='Comparison intercepts + centered elapsed time + position + four sum-coded waveform coefficients + four waveform-by-position coefficients',
        correction='Exact participant joint-vector sign flips; maximum across branch/period/scale/time/channel within effect family; Bonferroni across three families',
        interpretation='Exploratory model-conditional group mean effects; not trial label randomization or a classifier',
        dependencies={n:importlib.metadata.version(n) for n in prior_config['dependencies']},
        implementation={p.as_posix():sha(p) for p in sources},
        prior_config_sha256=sha(args.prior/'config.json'),
        comparison_config_sha256=sha(args.comparison/'config.json'),
        classifier_fitted=False, ica_refitted=False, true_scale_invariance_tested=False)
    write_json(args.output/'config.json', config)
    completed, errors = [], {}
    with threadpool_limits(limits=2):
        for subject in subjects:
            try:
                prepare(subject,args,config)
                completed.append(subject)
            except Exception as exc:
                errors[str(subject)] = f'{type(exc).__name__}: {exc}'
                print(traceback.format_exc(),flush=True)
            write_json(args.output/'status.json',dict(completed=completed, errors=errors))
    if errors:
        raise SystemExit(1)
    from .report import render
    with threadpool_limits(limits=2):
        render(args.output)
    print(f'Report: {(args.output/"index.html").resolve()}',flush=True)


if __name__ == '__main__':
    main()
