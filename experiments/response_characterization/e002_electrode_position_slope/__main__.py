"""Run electrode/position/slope diagnostics, without fitting classifiers."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import traceback

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.waveform_decoding.e001_voltage_baseline.events import (align_events, load_recording, read_behavior,
                                               trigger_runs, protocol_audit)
from .analysis import (WINDOWS, METRICS, SLOPE_WINDOW_S, epoch_and_slopes, fit_effects,
                       condition_means, split_half_repeatability, group_tests)
from .preprocess import SETTINGS, supplied_preprocess


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str)+'\n', encoding='utf-8')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_subject(subject, args, config):
    source = args.data_dir/f'Subject{subject}_data'/f'EEGDATA_SUB{subject}-T4.xdf'
    provenance = dict(source_file=str(source.resolve()), source_sha256=sha(source),
                      csv_sha256=config['csv_sha256'], preprocessing=SETTINGS,
                      slope_seconds=SLOPE_WINDOW_S, windows=WINDOWS,
                      ptp_limit_uv=args.ptp_limit_uv,
                      implementation={p.name:sha(p) for p in [
                          Path(__file__).with_name('preprocess.py'),
                          Path(__file__).with_name('analysis.py'),
                          Path('experiments/waveform_decoding/e001_voltage_baseline/events.py')]},
                      dependencies=config['dependencies'])
    key = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()[:20]
    cache = args.cache_dir/f'sub-{subject}-{key}'
    destination = args.output/f'sub-{subject}'
    destination.mkdir()
    if not (cache/'complete.json').exists():
        if cache.exists() and any(cache.iterdir()):
            raise ValueError(f'Incomplete cache {cache}; preserve it and use a fresh --cache-dir')
        cache.mkdir(parents=True, exist_ok=True)
        write_json(cache/'provenance.json', provenance)
        print(f'Subject {subject}: load and align original recording', flush=True)
        data, names, trigger, timestamps, sfreq = load_recording(source)
        runs = trigger_runs(trigger, sfreq)
        runs.to_csv(cache/'trigger_runs.csv', index=False)
        events = align_events(runs, read_behavior(args.master_csv, subject), subject, timestamps)
        cleaned, audit = supplied_preprocess(data, names, sfreq, events['sample'].to_numpy(), cache)
        del data, trigger
        epochs, slopes, rel, features, manifest = epoch_and_slopes(
            cleaned, events, timestamps, sfreq, args.ptp_limit_uv)
        del cleaned, timestamps
        meta = manifest[manifest.diagnostic_eligible].reset_index(drop=True)
        effects, design = fit_effects(features, meta)
        manifest.to_csv(cache/'events.csv', index=False)
        repeatability = split_half_repeatability(epochs, slopes, meta, rel)
        repeatability.to_csv(cache/'repeatability.csv', index=False)
        support = meta.groupby(['waveform','position','comparison']).size().rename('n').reset_index()
        support.to_csv(cache/'cell_support.csv', index=False)
        # Save only fixed-window features and evoked summaries, not full continuous EEG.
        means = condition_means(epochs, meta)
        slope_means = condition_means(slopes, meta)
        summary_features = condition_means(features, meta)
        np.savez_compressed(cache/'responses.npz', channels=np.asarray(names), times=rel,
                            voltage=means, slope=slope_means,
                            features=features, condition_features=summary_features, **effects)
        audit.update(n_events=len(events), n_eligible=len(meta), sfreq=sfreq,
                     n_channels=len(names), design=design,
                     response_status_source='T4 Naturalness Value; legacy detected zeros ignored',
                     exclusion_counts=manifest.diagnostic_exclusion_reason.value_counts().to_dict(),
                     protocol=protocol_audit(events))
        write_json(cache/'preprocessing.json', audit)
        write_json(cache/'complete.json', dict(subject=subject, cache_key=key))
        print(f'  {len(meta)}/{len(events)} events retained; effects rank {design["rank"]}', flush=True)
    else:
        print(f'Subject {subject}: reuse provenance-matched diagnostics cache', flush=True)
    for filename in ['events.csv','cell_support.csv','repeatability.csv','responses.npz',
                     'preprocessing.json','provenance.json']:
        shutil.copyfile(cache/filename, destination/filename)
    saved = np.load(destination/'responses.npz')
    return dict(subject=subject, channels=saved['channels'].tolist(), times=saved['times'],
                voltage=saved['voltage'], slope=saved['slope'],
                condition_features=saved['condition_features'],
                effects={name:saved[name] for name in ['position','waveform','interaction']},
                audit=json.loads((destination/'preprocessing.json').read_text()),
                repeatability=pd.read_csv(destination/'repeatability.csv'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subjects', default=','.join(map(str,range(3,13))))
    parser.add_argument('--data-dir', type=Path, default=Path('EEGData'))
    parser.add_argument('--master-csv', type=Path,
                        default=Path('EEG_Codes_Export/02_SourceLocalization_Task4/Master_Data_ToSub12 (1).csv'))
    parser.add_argument('--output', type=Path, default=Path('output/task4_response_diagnostics'))
    parser.add_argument('--cache-dir', type=Path, default=Path('output/task4_response_cache'))
    parser.add_argument('--ptp-limit-uv', type=float, default=500.)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    subjects = [int(value) for value in args.subjects.split(',')]
    if not subjects or len(set(subjects)) != len(subjects) or not set(subjects) <= set(range(3,13)):
        parser.error('Choose unique participants 3-12')
    if args.ptp_limit_uv <= 0 or not np.isfinite(args.ptp_limit_uv):
        parser.error('Amplitude limit must be finite and positive')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Choose a fresh output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    config = dict(subjects=subjects, preprocessing=SETTINGS, windows=WINDOWS, metrics=METRICS,
                  slope_seconds=SLOPE_WINDOW_S, ptp_limit_uv=args.ptp_limit_uv,
                  csv_sha256=sha(args.master_csv),
                  dependencies={name:importlib.metadata.version(name) for name in
                                ['numpy','pandas','scipy','mne','matplotlib','scikit-learn','pyxdf']},
                  implementation={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
                  inference='exploratory participant coefficient sign-flips; max-stat and 3-family correction',
                  classifier_fitted=False)
    write_json(args.output/'run_config.json', config)
    results, errors = [], {}
    with threadpool_limits(limits=2):
        for subject in subjects:
            try:
                results.append(prepare_subject(subject,args,config))
            except Exception as exc:
                errors[str(subject)] = f'{type(exc).__name__}: {exc}'
                print(traceback.format_exc(), flush=True)
                write_json(args.output/f'sub-{subject}'/'error.json', dict(error=errors[str(subject)]))
    write_json(args.output/'status.json',dict(completed=[r['subject'] for r in results],errors=errors))
    if results and not args.prepare_only:
        from .report import write_report
        channels = results[0]['channels']
        if any(result['channels'] != channels for result in results):
            raise ValueError('Inconsistent channel ordering; do not silently pool')
        tests = group_tests([r['effects'] for r in results],channels) if len(results) >= 3 else pd.DataFrame()
        tests.to_csv(args.output/'effect_tests.csv',index=False)
        write_report(args.output,results,tests,errors)
    if errors:
        raise SystemExit(1)
    print(f'Results: {args.output.resolve()}',flush=True)


if __name__ == '__main__':
    main()
