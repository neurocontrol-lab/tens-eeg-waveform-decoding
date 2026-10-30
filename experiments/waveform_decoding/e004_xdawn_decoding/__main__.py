"""Run a fixed, fold-fitted xDAWN comparison with matched sensor controls."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import time
import numpy as np
import mne
from threadpoolctl import threadpool_limits
from .core import COMPONENTS_PER_CLASS,SPATIAL_SHRINKAGE
from .data import prepare_subject,sha,write_json
from .evaluate import evaluate_job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,default=Path('output/task4_baseline'))
    parser.add_argument('--data-dir',type=Path,default=Path('EEGData'))
    parser.add_argument('--master-csv',type=Path,default=Path('EEG_Codes_Export/02_SourceLocalization_Task4/Master_Data_ToSub12 (1).csv'))
    parser.add_argument('--cache-dir',type=Path,default=Path('output/task4_xdawn_epoch_cache'))
    parser.add_argument('--output',type=Path,default=Path('output/task4_xdawn'))
    parser.add_argument('--subjects',default=','.join(map(str,range(3,13))))
    parser.add_argument('--permutations',type=int,default=99)
    parser.add_argument('--audit-resamples',type=int,default=10)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--no-loso',action='store_true')
    parser.add_argument('--prepare-only',action='store_true')
    args = parser.parse_args()
    subjects = [int(value) for value in args.subjects.split(',')]
    if not subjects or len(subjects)!=len(set(subjects)) or not set(subjects)<=set(range(3,13)):
        parser.error('Choose unique participants 3-12')
    if not args.no_loso and subjects!=list(range(3,13)) and not args.prepare_only:
        parser.error('Matched LOSO requires all subjects 3-12 in order; use --no-loso for a subset')
    if args.permutations<0 or args.seed<0 or args.audit_resamples<2:
        parser.error('Invalid permutation count, seed or covariance resample count')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Choose a fresh output directory')
    args.output.mkdir(parents=True,exist_ok=True)
    args.dependencies = {name:importlib.metadata.version(name) for name in
        ['numpy','pandas','scipy','scikit-learn','pyxdf','mne','matplotlib']}
    args.evaluate_source = Path('experiments/waveform_decoding/e001_voltage_baseline/evaluate.py')
    sources = [*Path(__file__).parent.glob('*.py'),args.evaluate_source,
        Path('experiments/waveform_decoding/e001_voltage_baseline/events.py'),Path('experiments/waveform_decoding/e001_voltage_baseline/features.py'),
        Path('experiments/signal_quality/e003_voltage_feature_audit/core.py'),Path('experiments/signal_quality/e003_voltage_feature_audit/__main__.py')]
    config = dict(subjects=subjects,baseline=str(args.baseline.resolve()),permutations=args.permutations,
        seed=args.seed,audit_resamples=args.audit_resamples,no_loso=args.no_loso,
        components_per_class=COMPONENTS_PER_CLASS,spatial_covariance_shrinkage=SPATIAL_SHRINKAGE,
        temporal_bins_ms=50,post_ms=[50,500],pre_ms=[-500,-50],
        baseline_ms=dict(post=[-250,-50],pre=[-750,-550]),
        spatial_fit='Full-rate training epochs only; separate pre/post fits; refit for every permutation',
        preprocessing='Exact original minimal baseline; no temporal filtering, ICA or stimulation blanking',
        dependencies=args.dependencies,implementation_sha256={str(p):sha(p) for p in sources},
        baseline_run_config_sha256=sha(args.baseline/'run_config.json'))
    write_json(args.output/'run_config.json',config)
    reports,prepared = {},[]
    started = time.perf_counter()
    mne.set_log_level('ERROR')
    try:
        with threadpool_limits(limits=1):
            for subject in subjects:
                prepared.append(prepare_subject(subject,args,args.dependencies))
                write_json(args.output/'status.json',dict(prepared=[p['subject'] for p in prepared],
                    completed_jobs=list(reports),complete=False,errors={}))
            write_json(args.output/'epoch_provenance.json',[
                dict(cache=str(p['cache'].resolve()),**p['info'],
                     provenance=json.loads((p['cache']/'provenance.json').read_text())) for p in prepared])
            if not args.prepare_only:
                for item in prepared:
                    identity = f'sub-{item["subject"]}'
                    reports[identity] = evaluate_job(identity,'within',item['epochs'],item['meta'],
                                                     item['info']['samples_per_bin'],args)
                    write_json(args.output/'status.json',dict(prepared=subjects,completed_jobs=list(reports),complete=False,errors={}))
                if not args.no_loso:
                    import pandas as pd
                    if len({p['info']['sfreq'] for p in prepared})!=1:
                        raise ValueError('Different sample rates; do not silently pool')
                    epochs = {window:np.concatenate([p['epochs'][window] for p in prepared]) for window in ['post','pre']}
                    meta = pd.concat([p['meta'] for p in prepared],ignore_index=True)
                    reports['loso'] = evaluate_job('loso','loso',epochs,meta,prepared[0]['info']['samples_per_bin'],args)
        write_json(args.output/'summary.json',dict(reports=reports))
        write_json(args.output/'status.json',dict(prepared=subjects,completed_jobs=list(reports),
            complete=True,prepare_only=args.prepare_only,seconds=time.perf_counter()-started,errors={}))
        if reports:
            from .report import write_report
            write_report(args.output)
    except Exception as exc:
        write_json(args.output/'status.json',dict(prepared=[p['subject'] for p in prepared],
            completed_jobs=list(reports),complete=False,errors=dict(run=f'{type(exc).__name__}: {exc}')))
        raise
    print(f'Results: {args.output.resolve()}',flush=True)


if __name__=='__main__':
    main()
