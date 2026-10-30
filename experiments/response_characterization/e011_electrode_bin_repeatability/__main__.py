"""E011: fixed per-electrode/bin voltage, slope and curvature repeatability."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import traceback
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from experiments.provenance import source_matches
from experiments.waveform_decoding.e001_voltage_baseline.events import CODES
from experiments.waveform_decoding.e001_voltage_baseline.features import CHANNELS
from experiments.signal_quality.e010_simple_visual_inspection.core import cell_balanced_evokeds
from .core import (WIDTHS_MS,REPRESENTATIONS,UNITS,REPEATS,SEED,WINDOW_SAMPLES,POLYORDER,
    START_MS,END_MS,eligible_split_pairs,split_operators,waveform_operator,
    local_representations,bin_features,repeatability)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False,default=str)+'\n',encoding='utf-8')


def verify_source(source):
    config = json.loads((source/'run_config.json').read_text())
    status = json.loads((source/'status.json').read_text())
    if status['errors'] or not status['complete']:
        raise ValueError('Require completed E010 inputs')
    if config['channels'] != list(CHANNELS) or config['epoch'] != [-.3,1.2] or config['baseline'] != [-.25,-.05]:
        raise ValueError('Unexpected source channels, epoch or baseline')
    if config['filter'] != dict(type='Butterworth SOS forward/backward',prototype_order=4,
            bandpass_poles_per_pass=8,cutoff_hz=[.5,30.],segment_edge_guard_s=8.):
        raise ValueError('Unexpected E010 simple filter settings')
    for path,digest in config['initial_sha256'].items():
        if not source_matches(path,digest):
            raise ValueError(f'E010 source/input changed beyond the recorded relocation: {path}')
    return config,status


def prepare(subject,args):
    source = args.source/f'sub-{subject}'
    out = args.output/f'sub-{subject}'
    out.mkdir()
    print(f'Subject {subject}: verified simple-pipeline epochs, derivatives and paired splits',flush=True)
    meta_all = pd.read_csv(source/'eligible_events.csv')
    epochs_all = np.load(source/'bandpass_epochs_uv.npy',mmap_mode='r')
    audit_previous = json.loads((source/'audit.json').read_text())
    with np.load(source/'summaries.npz') as saved:
        times = saved['time_s']
        sfreq = float(saved['sfreq'])
        if saved['channels'].tolist() != list(CHANNELS) or saved['triggers'].tolist() != list(CODES):
            raise ValueError('Source channel/trigger ordering differs')
        means,counts,cells = cell_balanced_evokeds(epochs_all,meta_all,CODES)
        np.testing.assert_allclose(means,saved['means'][1],rtol=0,atol=1e-10)
        reconstruction_error = float(np.max(np.abs(means-saved['means'][1])))
    if epochs_all.shape != (len(meta_all),len(CHANNELS),len(times)) or not (meta_all.subject == subject).all():
        raise ValueError('Source epoch/metadata/participant identity differs')
    if len(meta_all) != audit_previous['n_visual_events']:
        raise ValueError('Source cohort count differs')
    _,used = eligible_split_pairs(meta_all)
    keep = meta_all.pair_id.isin(used).to_numpy()
    manifest = meta_all.copy()
    manifest['repeatability_eligible'] = keep
    manifest['repeatability_exclusion_reason'] = np.where(keep,'','ordered_comparison_stratum_has_one_pair')
    manifest.to_csv(out/'events.csv',index=False)
    meta = meta_all[keep].reset_index(drop=True)
    epochs = np.asarray(epochs_all[keep],dtype=float)
    meta.to_csv(out/'eligible_events.csv',index=False)
    operators,assignments,designs = split_operators(meta,subject)
    assignments.to_csv(out/'split_assignments.csv',index=False)
    designs.to_csv(out/'split_designs.csv',index=False)
    meta.groupby(['comparison','trigger','position']).size().rename('n_events').to_csv(out/'cell_support.csv')
    # Count actual label/position support in every half, without treating splits as N.
    support = []
    for (repeat,half),group in assignments.groupby(['repeat','half']):
        selected = meta[meta.pair_id.isin(group.pair_id)]
        counts_half = selected.groupby(['trigger','position']).size()
        if len(counts_half) != 10 or (counts_half <= 0).any():
            raise ValueError('Missing waveform/position in a split half')
        support += [dict(repeat=repeat,half=half,trigger=code,position=position,n_events=int(n))
                    for (code,position),n in counts_half.items()]
    pd.DataFrame(support).to_csv(out/'split_cell_support.csv',index=False)
    full_wave,full_common,full_design = waveform_operator(meta)
    values = local_representations(epochs,times,sfreq)
    common_summaries = []
    for representation in REPRESENTATIONS:
        name = representation.split('_')[0]
        rms = representation.endswith('_rms')
        for width in WIDTHS_MS:
            features,left,right = bin_features(values[name],times,width,rms=rms)
            result = repeatability(features,operators)
            profile = (full_wave.reshape(-1,len(meta))@features.reshape(len(meta),-1)).reshape(2,5,len(CHANNELS),-1)
            common = (full_common@features.reshape(len(meta),-1)).reshape(2,len(CHANNELS),-1)
            np.savez_compressed(out/f'w{width:03d}_{representation}.npz',**result,
                adjusted_profile=profile,common_response=common,left_ms=left,right_ms=right,
                channels=np.array(CHANNELS),unit=UNITS[representation])
            onesecond = (left >= 50) & (right <= 1050)
            for position in range(2):
                common_summaries.append(dict(subject=subject,representation=representation,width_ms=width,
                    position=position+1,n_pairs=len(used),n_bins=int(onesecond.sum()),
                    mean_location_r=float(np.nanmean(result['median_split_r'][position,:,onesecond])),
                    mean_location_product=float(result['mean_split_product'][position,:,onesecond].mean()),
                    unit=UNITS[representation]))
        print(f'  {representation}: six reporting widths complete',flush=True)
    pd.DataFrame(common_summaries).to_csv(out/'one_second_summary.csv',index=False)
    audit = dict(subject=subject,sfreq=sfreq,source_events=len(meta_all),source_pairs=meta_all.pair_id.nunique(),
        n_events=len(meta),n_pairs=len(used),singleton_pairs_omitted=meta_all.pair_id.nunique()-len(used),
        splits=REPEATS,min_half_cell_events=min(r['n_events'] for r in support),
        source_mean_max_error_uv=reconstruction_error,full_design=full_design,
        maximum_split_condition=float(designs.condition_number.max()),
        ica_refitted=False,classifier_fitted=False,amplitude_exclusions_added=False)
    write_json(out/'audit.json',audit)
    print(f'  {len(used)} repeatability pairs; {audit["singleton_pairs_omitted"]} singleton-stratum pairs omitted',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('output/task4_visual_inspection_20261030'))
    parser.add_argument('--output',type=Path,default=Path('output/e011_repeatability_20261030'))
    parser.add_argument('--subjects',default=','.join(map(str,range(3,13))))
    parser.add_argument('--prepare-only',action='store_true')
    args = parser.parse_args()
    subjects = [int(s) for s in args.subjects.split(',')]
    if not subjects or len(set(subjects)) != len(subjects) or not set(subjects)<=set(range(3,13)):
        parser.error('Choose unique subjects 3-12')
    if args.output.exists():
        parser.error('Use a fresh output directory; previous outputs are read-only')
    source_config,source_status = verify_source(args.source)
    if not set(subjects)<=set(source_status['completed_subjects']):
        parser.error('Requested participants absent from source')
    inputs = [args.source/'run_config.json',args.source/'status.json']
    for subject in subjects:
        inputs += [args.source/f'sub-{subject}'/n for n in
                   ['eligible_events.csv','bandpass_epochs_uv.npy','summaries.npz','audit.json']]
    sources = list(Path(__file__).parent.glob('*.py'))+[
        Path('experiments/response_characterization/e002_electrode_position_slope/analysis.py'),
        Path('experiments/signal_quality/e010_simple_visual_inspection/core.py'),Path('experiments/provenance.py')]
    before = {str(p):sha(p) for p in inputs+sources}
    args.output.mkdir(parents=True)
    config = dict(experiment_id='E011',subjects=subjects,source=str(args.source),
        representations=REPRESENTATIONS,units=UNITS,widths_ms=WIDTHS_MS,
        derivative_window_samples=WINDOW_SAMPLES,derivative_polynomial_order=POLYORDER,
        post_support_ms=[START_MS,END_MS],common_summary_ms=[50,1050],repeats=REPEATS,seed=SEED,
        model='Comparison intercepts + within-comparison centred time + position + waveform + waveform x position',
        bootstrap='1000 participant draws, seed 20261030; pointwise descriptive intervals',
        inference='No corrected significance test, feature selection or classifier',
        source_filter=source_config['filter'],baseline_ms=[-250,-50],channels=CHANNELS,
        input_sha256={str(p):before[str(p)] for p in inputs},
        implementation_sha256={str(p):before[str(p)] for p in sources},
        dependencies={k:importlib.metadata.version(k) for k in ['numpy','pandas','scipy','matplotlib']})
    write_json(args.output/'config.json',config)
    status = dict(completed=[],errors={},complete=False)
    started = time.perf_counter()
    with threadpool_limits(limits=2):
        for subject in subjects:
            try:
                prepare(subject,args)
                status['completed'].append(subject)
            except Exception as exc:
                status['errors'][str(subject)] = f'{type(exc).__name__}: {exc}'
                print(traceback.format_exc(),flush=True)
            write_json(args.output/'status.json',status)
        changed = [p for p,h in before.items() if sha(p)!=h]
        if changed:
            status['errors']['provenance'] = f'Inputs/source changed during run: {changed}'
        status['complete'] = not status['errors'] and status['completed'] == subjects
        status['seconds'] = time.perf_counter()-started
        write_json(args.output/'status.json',status)
        if not args.prepare_only and status['completed']:
            from .report import render
            render(args.output)
    if status['errors']:
        raise SystemExit(1)
    destination = args.output if args.prepare_only else args.output/'index.html'
    print(f'Results: {destination.resolve()}',flush=True)


if __name__ == '__main__':
    main()
