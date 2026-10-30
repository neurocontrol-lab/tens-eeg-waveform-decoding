"""Run repeatability stages only: all-electrode/power, or calibrated xDAWN."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import time
import traceback
from threadpoolctl import threadpool_limits
from experiments.response_characterization.e011_electrode_bin_repeatability.__main__ import sha,write_json
from .compute import sensor_subject,xdawn_subject


def run(args):
    cache_status=json.loads((args.cache/'status.json').read_text())
    cache_config=json.loads((args.cache/'config.json').read_text())
    if not cache_status['complete'] or cache_status['errors']:raise ValueError('All-electrode cache incomplete')
    for path,digest in {**cache_config['input_sha256'],**cache_config['implementation_sha256']}.items():
        if sha(path)!=digest:raise ValueError(f'Cache input/source changed: {path}')
    original=json.loads((args.source/'config.json').read_text())
    for path,digest in {**original['input_sha256'],**original['implementation_sha256']}.items():
        if sha(path)!=digest:raise ValueError(f'Original E011 input/source changed: {path}')
    subjects=[int(s) for s in args.subjects.split(',')]
    if not subjects or len(set(subjects))!=len(subjects) or not set(subjects)<=set(cache_status['completed']):
        raise ValueError('Choose unique cache participants')
    args.output.mkdir(parents=True,exist_ok=True);(args.output/'data').mkdir(exist_ok=True)
    configfile=args.output/f'config_{args.stage}.json'
    if configfile.exists():raise ValueError('Stage already exists; use a fresh output root')
    inputs=[args.cache/'config.json',args.cache/'status.json',args.source/'config.json']
    for s in subjects:
        inputs += [args.cache/f'sub-{s}'/n for n in ['epochs_uv.npy','layout.npz','events.csv','split_assignments.csv','channel_quality.csv']]
        if args.stage=='sensors':inputs+=list((args.waveform_source/f'sub-{s}').glob('w*.npz'))
    sources=[Path(__file__),Path(__file__).with_name('compute.py'),Path(__file__).with_name('features.py'),
        Path('experiments/response_characterization/e011_electrode_bin_repeatability/core.py'),
        Path('experiments/response_characterization/e011_electrode_bin_repeatability/waveform_views.py'),
        Path('experiments/waveform_decoding/e005_spectral_decoding/core.py')]
    before={str(p):sha(p) for p in inputs+sources}
    config=dict(experiment_id='E014',stage=args.stage,subjects=subjects,cache=str(args.cache),
        source=str(args.source),waveform_source=str(args.waveform_source),
        input_sha256={str(p):before[str(p)] for p in inputs},implementation_sha256={str(p):before[str(p)] for p in sources},
        dependencies={n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','mne']},
        classifier_fitted=False,features_selected=False,ica_refitted=False,
        sensor_notes='62 channels, preserved 595 pairs and E011 two-half allocations, 5 voltage/derivative summaries, six widths',
        power_notes='500-ms Hann windows, 50-ms step; alpha 8-13 Hz and beta 13-30 Hz; total dB power; overlapping windows',
        xdawn_notes='100 calibration/A/B pair partitions; calibration-only xDAWN(2 components/class, reg .1), RMS scaling; matched 62/14-channel controls; no classifier')
    write_json(configfile,config)
    complete=[];errors={};started=time.perf_counter()
    with threadpool_limits(limits=2):
        for subject in subjects:
            try:
                (sensor_subject if args.stage=='sensors' else xdawn_subject)(subject,args)
                complete.append(subject)
            except Exception as exc:
                errors[str(subject)]=f'{type(exc).__name__}: {exc}';print(traceback.format_exc(),flush=True)
            write_json(args.output/f'status_{args.stage}.json',dict(completed=complete,errors=errors,complete=False))
    changed=[p for p,h in before.items() if sha(p)!=h]
    if changed:errors['provenance']=f'Input/source changed during stage: {changed}'
    write_json(args.output/f'status_{args.stage}.json',dict(completed=complete,errors=errors,
        complete=not errors and complete==subjects,seconds=time.perf_counter()-started))
    if errors:raise SystemExit(1)
    print(f'E014 {args.stage} stage complete: {args.output.resolve()}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,default=Path('output/e014_all_electrode_cache_20261030'))
    parser.add_argument('--source',type=Path,default=Path('output/e011_repeatability_20261030'))
    parser.add_argument('--waveform-source',type=Path,default=Path('output/e011_waveform_views_20261030'))
    parser.add_argument('--output',type=Path,default=Path('output/e014_repeatability_20261030'))
    parser.add_argument('--subjects',default=','.join(map(str,range(3,13))))
    parser.add_argument('--stage',choices=['sensors','xdawn'],required=True)
    run(parser.parse_args())
