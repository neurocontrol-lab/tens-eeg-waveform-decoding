"""Reconstruct full-rate epochs, requiring the exact saved baseline cohort."""
import hashlib
import json
from pathlib import Path
from experiments.provenance import equivalent_cache_provenance
import numpy as np
import pandas as pd
from experiments.waveform_decoding.e001_voltage_baseline.events import load_recording,trigger_runs,align_events,read_behavior
from experiments.waveform_decoding.e001_voltage_baseline.features import CHANNELS,WINDOWS,build_features,interval_samples
from .core import bin_epochs,flatten_bins


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False,default=str)+'\n',encoding='utf-8')


def extract_epochs(data,names,meta,sfreq,window):
    lo,hi,step,blo,bhi = WINDOWS[window]
    picks = [names.index(name) for name in CHANNELS]
    selected = data[picks]-data.mean(axis=0,keepdims=True)
    length = round((hi-lo)*sfreq)
    epochs = np.empty((len(meta),len(picks),length),float)
    for i,sample in enumerate(meta['sample']):
        a,b = interval_samples(sample,lo,hi,sfreq)
        ba,bb = interval_samples(sample,blo,bhi,sfreq)
        epochs[i] = selected[:,a:b]-selected[:,ba:bb].mean(axis=1,keepdims=True)
    if not np.isfinite(epochs).all():
        raise ValueError('Nonfinite reconstructed epochs')
    width = int(round(step*sfreq))
    if not np.isclose(width/sfreq,step) or length != 9*width:
        raise ValueError('Sample rate incompatible with fixed bin definition')
    return epochs,width


def prepare_subject(subject,args,dependencies):
    baseline = args.baseline/f'sub-{subject}'
    original_config = json.loads((args.baseline/'run_config.json').read_text())
    source = args.data_dir/f'Subject{subject}_data/EEGDATA_SUB{subject}-T4.xdf'
    provenance = dict(subject=subject,source_file=str(source.resolve()),source_sha256=sha(source),
        master_csv_sha256=sha(args.master_csv),baseline_config_sha256=sha(args.baseline/'run_config.json'),
        baseline_events_sha256=sha(baseline/'events.csv'),baseline_features_sha256=sha(baseline/'features.npz'),
        dependencies=dependencies,implementation={str(p):sha(p) for p in [Path(__file__),
            Path('experiments/waveform_decoding/e001_voltage_baseline/events.py'),Path('experiments/waveform_decoding/e001_voltage_baseline/features.py')]})
    if provenance['master_csv_sha256'] != original_config['master_csv_sha256']:
        raise ValueError('Master CSV differs from the original baseline')
    key = hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest()[:20]
    cache = args.cache_dir/f'sub-{subject}-{key}'
    if not (cache/'complete.json').exists():
        legacy = []
        for candidate in args.cache_dir.glob(f'sub-{subject}-*'):
            if (candidate/'complete.json').exists() and (candidate/'provenance.json').exists():
                recorded = json.loads((candidate/'provenance.json').read_text())
                if equivalent_cache_provenance(recorded, provenance):
                    legacy.append(candidate)
        if len(legacy) > 1:
            raise ValueError('Multiple equivalent epoch caches; select a dedicated --cache-dir')
        if legacy:
            cache = legacy[0]
    if not (cache/'complete.json').exists():
        if cache.exists() and any(cache.iterdir()):
            raise ValueError(f'Incomplete cache {cache}; preserve it and use a fresh --cache-dir')
        cache.mkdir(parents=True,exist_ok=True)
        write_json(cache/'provenance.json',provenance)
        print(f'Subject {subject}: reconstruct baseline-matched full-rate epochs',flush=True)
        data,names,trigger,timestamps,sfreq = load_recording(source)
        runs = trigger_runs(trigger,sfreq,original_config['min_trigger_samples'])
        events = align_events(runs,read_behavior(args.master_csv,subject),subject,timestamps)
        features,manifest = build_features(data,names,events,timestamps,sfreq,original_config['ptp_limit_uv'])
        saved_events = pd.read_csv(baseline/'events.csv')
        columns = ['sample','trigger','subject','event_index','pair_id','group','position','analysis_eligible']
        pd.testing.assert_frame_equal(manifest[columns],saved_events[columns],check_dtype=False)
        keep = manifest.analysis_eligible.to_numpy()
        meta = manifest[keep].reset_index(drop=True)
        with np.load(baseline/'features.npz') as saved:
            for name in saved.files:
                np.testing.assert_allclose(features[name][keep],saved[name],rtol=0,atol=1e-10)
        errors = {}
        for window in ['post','pre']:
            epochs,width = extract_epochs(data,names,meta,sfreq,window)
            sensor = flatten_bins(bin_epochs(epochs,width))
            np.testing.assert_allclose(sensor,features[window][keep],rtol=0,atol=1e-10)
            errors[window] = float(np.abs(sensor-features[window][keep]).max())
            np.save(cache/f'{window}.npy',epochs)
        meta.to_csv(cache/'events.csv',index=False)
        write_json(cache/'complete.json',dict(subject=subject,sfreq=sfreq,samples_per_bin=width,
            n_events=len(meta),n_channels=len(CHANNELS),feature_reconstruction_max_errors=errors))
    else:
        print(f'Subject {subject}: reuse provenance-matched epochs',flush=True)
    info = json.loads((cache/'complete.json').read_text())
    return dict(subject=subject,cache=cache,meta=pd.read_csv(cache/'events.csv'),info=info,
                epochs={name:np.load(cache/f'{name}.npy',mmap_mode='r') for name in ['post','pre']})
