"""Run with python -m experiments.waveform_decoding.e001_voltage_baseline --subjects 3 --permutations 19."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from .events import align_events, load_recording, protocol_audit, read_behavior, trigger_runs
from .features import CHANNELS, WINDOWS, build_features
from .evaluate import evaluate


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str)+'\n', encoding='utf-8')


def write_report(out, reports, errors):
    lines = ['# Task 4 baseline results', '',
             'Trigger-aligned condition decoding; no claim of artifact-free cortical decoding.', '',
             'Balanced accuracy is computed over all held-out predictions with five fixed classes.',
             'Conditional permutations preserve comparison membership and paired structure; p-values are diagnostic.',
             'Windows were predefined, not selected using these scores. No temporal filtering or ICA was applied.', '',
             '| Evaluation | Features | Events | Balanced accuracy | Macro-F1 | Conditional p |',
             '| --- | --- | --- | --- | --- | --- |']
    for name, report in reports.items():
        for feature, result in report['models'].items():
            p = result['permutation_p']
            lines.append(f"| {name} | {feature} | {result['n_events']} | {result['balanced_accuracy']:.3f} | {result['macro_f1']:.3f} | {p if p is not None else 'not run'} |")
    within = [report for name,report in reports.items() if name.startswith('sub-')]
    if within:
        lines += ['', '## Across-participant within-subject summary', '',
                  'Each participant receives equal weight. Bootstrap intervals resample participants, not epochs.', '',
                  '| Features | Participants | Mean balanced accuracy | Descriptive 95% bootstrap interval |',
                  '| --- | --- | --- | --- |']
        for feature in within[0]['models']:
            values = np.array([report['models'][feature]['balanced_accuracy'] for report in within])
            if len(values) >= 2:
                rng = np.random.default_rng(20261016)
                boot = rng.choice(values, size=(10000,len(values)), replace=True).mean(axis=1)
                lo,hi = np.quantile(boot,[.025,.975])
                interval = f'{lo:.3f}-{hi:.3f}'
            else:
                interval = 'not estimated from one participant'
            lines.append(f'| {feature} | {len(values)} | {values.mean():.3f} | {interval} |')
    lines += ['', '## Interpretation limits', '',
              '- A high onset score flags readily recoverable peri-trigger information, not its neural or electrical origin.',
              '- Pre-stimulus and order-only scores diagnose preceding-stimulus, block, position, and drift information.',
              '- Low pre-stimulus scores do not rule out post-stimulus artifacts. The late window is a sensitivity analysis.',
              '- Charge-balanced pulses need not be charge-matched across waveform conditions.',
              '- Primary cohort uses valid naturalness responses and complete EEG-eligible pairs; flagged trials remain in manifests.',
              '- Raw slow drift, bad channels, ocular/muscle activity, and unknown physical onset delay can still affect results.',
              '- Small permutation counts do not support strong above-chance claims; participant-level uncertainty needs the full cohort.',
              '- Some held-out comparison folds contain fewer than five classes; reported headline scores pool predictions across folds.',
              '', '## Errors / skipped participants', '', json.dumps(errors, indent=2)]
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('EEGData'))
    parser.add_argument('--master-csv', type=Path, default=Path('EEG_Codes_Export/02_SourceLocalization_Task4/Master_Data_ToSub12 (1).csv'))
    parser.add_argument('--subjects', default=','.join(map(str, range(3, 13))))
    parser.add_argument('--output', type=Path, default=Path('output/task4_baseline'))
    parser.add_argument('--min-trigger-samples', type=int, default=3)
    parser.add_argument('--ptp-limit-uv', type=float, default=500.)
    parser.add_argument('--permutations', type=int, default=99)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--loso', action='store_true')
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    subjects = [int(s) for s in args.subjects.split(',')]
    if not subjects or len(set(subjects)) != len(subjects) or not set(subjects) <= set(range(3, 13)):
        parser.error('Choose unique Task 4 subjects from 3-12; subject 2 is incompatible')
    if args.permutations < 0 or args.ptp_limit_uv <= 0:
        parser.error('Invalid permutation count or amplitude limit')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output directory must be empty; use a fresh path to avoid stale results')
    args.output.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.update(channels=CHANNELS, windows=WINDOWS, python=platform.python_version(),
                  dependencies={name: importlib.metadata.version(name) for name in ['numpy','pandas','scipy','scikit-learn','pyxdf']},
                  master_csv_sha256=hashlib.sha256(args.master_csv.read_bytes()).hexdigest())
    # Preserve code provenance even when Git is unavailable or sandbox ownership differs.
    config['implementation_sha256'] = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    write_json(args.output/'run_config.json', config)
    reports, errors, all_features, all_meta = {}, {}, [], []
    for subject in subjects:
        out = args.output/f'sub-{subject}'
        out.mkdir(exist_ok=True)
        print(f'Subject {subject}: loading XDF', flush=True)
        try:
            path = args.data_dir/f'Subject{subject}_data'/f'EEGDATA_SUB{subject}-T4.xdf'
            data, names, trigger, times, sfreq = load_recording(path)
            runs = trigger_runs(trigger, sfreq, args.min_trigger_samples)
            runs.to_csv(out/'trigger_runs.csv', index=False)
            behavior = read_behavior(args.master_csv, subject)
            events = align_events(runs, behavior, subject, times)
            features, events = build_features(data, names, events, times, sfreq, args.ptp_limit_uv)
            events['source_file'] = str(path.resolve())
            events['sfreq'] = sfreq
            events.to_csv(out/'events.csv', index=False)
            write_json(out/'audit.json', dict(source_file=str(path.resolve()),
                file_bytes=path.stat().st_size, n_events=len(events),
                n_eeg_channels=len(names), sfreq=sfreq,
                accepted_run_lengths=runs[runs.accepted].duration_samples.value_counts().to_dict(),
                rejected_runs=int((~runs.accepted).sum()),
                timestamp_step_quantiles=np.quantile(np.diff(times),[0,.5,.99,1]).tolist(),
                n_eligible=int(events.analysis_eligible.sum()),
                eeg_exclusions=events.eeg_exclusion_reason.value_counts().to_dict(),
                response_status_source='T4 Naturalness Value; legacy detected field ignored',
                blocks=protocol_audit(events)))
            print(f'  {len(events)} aligned events, {events.analysis_eligible.sum()} eligible', flush=True)
            keep = events.analysis_eligible.to_numpy()
            meta = events.loc[keep].reset_index(drop=True)
            subset = {name:values[keep] for name,values in features.items()}
            np.savez_compressed(out/'features.npz', **subset)
            all_features.append(subset)
            all_meta.append(meta)
            del data, trigger, times
            if not args.audit_only:
                report, predictions = evaluate(subset, meta, 'within', args.permutations, args.seed)
                reports[f'sub-{subject}'] = report
                write_json(out/'scores.json', report)
                predictions.to_csv(out/'predictions.csv', index=False)
                print('  '+', '.join(f'{name}={score["balanced_accuracy"]:.3f}' for name,score in report['models'].items()), flush=True)
        except Exception as exc:
            errors[str(subject)] = f'{type(exc).__name__}: {exc}'
            write_json(out/'error.json', dict(error=errors[str(subject)]))
            print(f'  ERROR: {errors[str(subject)]}', flush=True)
    if args.loso and not args.audit_only:
        if len(all_meta) >= 2:
            features = {name:np.concatenate([f[name] for f in all_features]) for name in all_features[0]}
            report, predictions = evaluate(features, pd.concat(all_meta, ignore_index=True), 'loso', args.permutations, args.seed)
            reports['loso'] = report
            write_json(args.output/'loso_scores.json', report)
            predictions.to_csv(args.output/'loso_predictions.csv', index=False)
        else:
            errors['loso'] = 'Need at least two successfully audited participants'
    write_json(args.output/'summary.json', dict(reports=reports, errors=errors))
    write_report(args.output, reports, errors)
    print(f'Results: {args.output.resolve()}', flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
