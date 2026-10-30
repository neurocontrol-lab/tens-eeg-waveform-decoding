"""Preserved E011 extension: individual-waveform and participant repeatability."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import time
import warnings
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from experiments.waveform_decoding.e001_voltage_baseline.events import CODES,WAVEFORMS
from experiments.waveform_decoding.e001_voltage_baseline.features import CHANNELS
from .core import (WIDTHS_MS,REPRESENTATIONS,UNITS,SEED,local_representations,
    bin_features,split_operators,waveform_operator,group_interval)
from .__main__ import sha,write_json

VIEWS = ('specific','total')


def observed_weights(meta,keep=None):
    """Total observed response: equal observed comparison cells, within position.

    This deliberately retains the common evoked response and nuisance activity.
    It is not the context-adjusted waveform-specific regression coefficient.
    """
    if keep is None:
        keep = np.ones(len(meta),bool)
    weights = np.zeros((2,5,len(meta)))
    for p in (1,2):
        for w,code in enumerate(CODES):
            selected = meta[keep & (meta.position == p).to_numpy() & (meta.trigger == code).to_numpy()]
            cells = list(selected.groupby('comparison').groups.values())
            if not cells:
                raise ValueError('Missing waveform/position support')
            for cell in cells:
                ix = np.asarray(list(cell),int)
                weights[p-1,w,ix] = 1/(len(cells)*len(ix))
    return weights


def total_split_operators(meta,assignments,repeats):
    operators = []
    for repeat in range(repeats):
        halves = []
        for half in (0,1):
            pairs = assignments[(assignments['repeat']==repeat)&(assignments.half==half)].pair_id
            halves.append(observed_weights(meta,meta.pair_id.isin(pairs).to_numpy()))
        operators.append(halves)
    return np.stack(operators)


def pattern_r(a,b,axis=-1):
    """Correlation across a declared time/channel pattern, never across splits."""
    ac,bc = a-a.mean(axis=axis,keepdims=True),b-b.mean(axis=axis,keepdims=True)
    aa,bb = np.sum(ac*ac,axis=axis),np.sum(bc*bc,axis=axis)
    # Numerical-constant patterns have undefined correlation, not invented zero r.
    scale_a = np.sum(a*a,axis=axis)
    scale_b = np.sum(b*b,axis=axis)
    eps = (100*np.finfo(float).eps)**2
    valid = (aa>eps*np.maximum(scale_a,np.finfo(float).tiny)) & (bb>eps*np.maximum(scale_b,np.finfo(float).tiny))
    return np.divide(np.sum(ac*bc,axis=axis),np.sqrt(aa*bb),
        out=np.full(aa.shape,np.nan),where=valid)


def median_finite(values):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='All-NaN slice encountered',category=RuntimeWarning)
        return np.nanmedian(values,axis=0)


def individual_estimates(features,operators,full_operator,mask,specific,batch=5):
    """Separate every waveform: products/disagreement maps and time-pattern r."""
    n,channels,bins = features.shape
    repeats = len(operators)
    shape = (2,5,channels,bins)
    products = np.zeros(shape)
    difference_ss = np.zeros(shape)
    sign_agreement = np.zeros(shape)
    half_energy = np.zeros(shape)
    time_correlations = []
    full_correlations = []
    examples = {}
    flat = np.ascontiguousarray(features).reshape(n,-1)
    for start in range(0,repeats,batch):
        op = operators[start:start+batch]
        profile = (op.reshape(-1,n)@flat).reshape(len(op),2,2,5,channels,bins)
        if specific:
            profile -= profile.mean(axis=3,keepdims=True)
        a,b = profile[:,0],profile[:,1]
        if start == 0:
            examples = dict(example_half_a=a[0].copy(),example_half_b=b[0].copy())
        products += (a*b).sum(axis=0)
        difference_ss += ((a-b)**2).sum(axis=0)
        sign_agreement += (a*b>0).sum(axis=0)
        half_energy += (.5*(a*a+b*b)).sum(axis=0)
        a_time,b_time = a[...,mask],b[...,mask]
        time_correlations.append(pattern_r(a_time,b_time))
        full_correlations.append(pattern_r(a_time.reshape(len(op),2,5,-1),b_time.reshape(len(op),2,5,-1)))
    temporal = np.concatenate(time_correlations)
    joint = np.concatenate(full_correlations)
    response = (full_operator.reshape(-1,n)@flat).reshape(shape)
    if specific:
        response -= response.mean(axis=1,keepdims=True)
    return dict(**examples,mean_split_product=products/repeats,
        rms_half_disagreement=np.sqrt(difference_ss/repeats),
        split_sign_agreement=sign_agreement/repeats,
        mean_half_energy=half_energy/repeats,
        full_response=response,
        temporal_profile_r=median_finite(temporal),
        joint_profile_r=median_finite(joint),
        valid_temporal_splits=np.isfinite(temporal).sum(axis=0),
        valid_joint_splits=np.isfinite(joint).sum(axis=0))


def encode(values):
    """Compact local JS payload, preserving signed units at display precision."""
    return base64.b64encode(np.asarray(values,dtype='<f4').tobytes()).decode('ascii')


def finite_list(values):
    a = np.asarray(values,dtype=float)
    return np.where(np.isfinite(a),a,None).tolist()


def prepare(subject,args,config):
    source = args.source/f'sub-{subject}'
    previous_config = json.loads((args.source/'config.json').read_text())
    original_source = Path(previous_config['source'])/f'sub-{subject}'
    meta = pd.read_csv(source/'eligible_events.csv')
    all_meta = pd.read_csv(original_source/'eligible_events.csv')
    ids = pd.Index(all_meta.event_index).get_indexer(meta.event_index)
    if (ids<0).any():
        raise ValueError('E011 events missing from E010 cache')
    pd.testing.assert_frame_equal(meta.reset_index(drop=True),all_meta.iloc[ids].reset_index(drop=True),check_dtype=False)
    epochs = np.load(original_source/'bandpass_epochs_uv.npy',mmap_mode='r')[ids]
    with np.load(original_source/'summaries.npz') as saved:
        times = saved['time_s']
        sfreq = float(saved['sfreq'])
    operators,assignments,_ = split_operators(meta,subject)
    saved_splits = pd.read_csv(source/'split_assignments.csv')
    pd.testing.assert_frame_equal(assignments,saved_splits,check_dtype=False)
    total = total_split_operators(meta,assignments,len(operators))
    specific_full,_,_ = waveform_operator(meta)
    total_full = observed_weights(meta)
    values = local_representations(epochs,times,sfreq)
    destination = args.output/f'sub-{subject}'
    destination.mkdir()
    rows,channel_rows,support = [],[],[]
    for p in (1,2):
        for code in CODES:
            counts = []
            for _,allocation in assignments.groupby(['repeat','half']):
                counts.append(int((meta.pair_id.isin(allocation.pair_id)&(meta.position==p)&(meta.trigger==code)).sum()))
            support.append(dict(position=p,trigger=code,n_events=int(((meta.position==p)&(meta.trigger==code)).sum()),
                minimum_half_events=min(counts),maximum_half_events=max(counts)))
    pd.DataFrame(support).to_csv(destination/'support.csv',index=False)
    for representation in REPRESENTATIONS:
        payload = dict(subject=subject,representation=representation,unit=UNITS[representation],
            channels=list(CHANNELS),waves=[WAVEFORMS[c] for c in CODES],support=support,widths={})
        for width in WIDTHS_MS:
            features,left,right = bin_features(values[representation.split('_')[0]],times,width,
                rms=representation.endswith('_rms'))
            mask = (left>=50)&(right<=1050)
            results = {}
            for view,op,full in [('specific',operators,specific_full),('total',total,total_full)]:
                estimates = individual_estimates(features,op,full,mask,specific=(view=='specific'))
                if view == 'specific':
                    with np.load(source/f'w{width:03d}_{representation}.npz') as saved:
                        # The previous pooled product is exactly the mean of these
                        # five per-waveform contributions. Preserve/reproduce it.
                        pooled = estimates['mean_split_product'].mean(axis=1)
                        np.testing.assert_allclose(pooled,saved['mean_split_product'],rtol=1e-10,atol=1e-8)
                        np.testing.assert_allclose(estimates['full_response'],saved['adjusted_profile'],rtol=1e-10,atol=1e-8)
                np.savez_compressed(destination/f'w{width:03d}_{representation}_{view}.npz',**estimates,
                    left_ms=left,right_ms=right,triggers=np.array(CODES),channels=np.array(CHANNELS))
                results[view] = dict(
                    **{k:encode(estimates[k]) for k in ['mean_split_product','rms_half_disagreement','split_sign_agreement','full_response','example_half_a','example_half_b']},
                    temporal_profile_r=finite_list(estimates['temporal_profile_r']),
                    joint_profile_r=finite_list(estimates['joint_profile_r']))
                for p in range(2):
                    for w,code in enumerate(CODES):
                        rows.append(dict(subject=subject,representation=representation,width_ms=width,view=view,
                            position=p+1,trigger=code,waveform=WAVEFORMS[code],joint_pattern_r=estimates['joint_profile_r'][p,w],
                            mean_channel_temporal_r=float(np.nanmean(estimates['temporal_profile_r'][p,w])),
                            mean_location_product=float(estimates['mean_split_product'][p,w][:,mask].mean()),
                            unit=UNITS[representation]))
                        for c,ch in enumerate(CHANNELS):
                            channel_rows.append(dict(subject=subject,representation=representation,width_ms=width,view=view,
                                position=p+1,waveform=WAVEFORMS[code],channel=ch,
                                temporal_pattern_r=estimates['temporal_profile_r'][p,w,c],
                                valid_splits=int(estimates['valid_temporal_splits'][p,w,c])))
            payload['widths'][str(width)] = dict(bins=len(left),left_ms=int(left[0]),right_ms=int(right[-1]),views=results)
        key = f'{subject}:{representation}'
        script = 'window.e011WaveData=window.e011WaveData||{};window.e011WaveData['+json.dumps(key)+']='+json.dumps(payload,allow_nan=False,separators=(',',':'))+';\n'
        (args.output/'data'/f'sub-{subject}_{representation}.js').write_text(script,encoding='utf-8')
        print(f'  Subject {subject}: {representation}, five waveforms kept separate',flush=True)
    pd.DataFrame(rows).to_csv(destination/'waveform_summary.csv',index=False)
    pd.DataFrame(channel_rows).to_csv(destination/'channel_temporal_correlations.csv',index=False)
    return rows


def group_summary(rows,out,subjects):
    frame = pd.DataFrame(rows)
    frame.to_csv(out/'participant_waveform_summary.csv',index=False)
    rng = np.random.default_rng(SEED)
    indices = rng.integers(len(subjects),size=(1000,len(subjects)))
    results = []
    for keys,group in frame.groupby(['representation','width_ms','view','position','trigger','waveform']):
        ordered = group.set_index('subject').reindex(subjects)
        if ordered.joint_pattern_r.isna().all():
            continue
        mean,low,high,n = group_interval(ordered.joint_pattern_r.to_numpy(),indices)
        results.append(dict(zip(['representation','width_ms','view','position','trigger','waveform'],keys))|
            dict(mean_joint_pattern_r=float(mean),low=float(low),high=float(high),n_subjects=int(n)))
    pd.DataFrame(results).to_csv(out/'group_waveform_summary.csv',index=False)


def run(args):
    status = json.loads((args.source/'status.json').read_text())
    if not status['complete'] or status['errors']:
        raise ValueError('Require the completed original E011 run')
    source_config = json.loads((args.source/'config.json').read_text())
    # Do not rewrite or weaken the original E011 source/input provenance.
    for path,digest in {**source_config['input_sha256'],**source_config['implementation_sha256']}.items():
        if sha(path)!=digest:
            raise ValueError(f'Original E011 input/source changed: {path}')
    inputs = [args.source/'config.json',args.source/'status.json']
    for s in status['completed']:
        inputs += list((args.source/f'sub-{s}').glob('w*.npz'))+[
            args.source/f'sub-{s}'/'eligible_events.csv',args.source/f'sub-{s}'/'split_assignments.csv']
    inputs += [Path(p) for p in source_config['input_sha256']]
    sources = [Path(__file__),Path(__file__).with_name('waveform_report.py'),Path(__file__).with_name('core.py'),
        Path(__file__).with_name('__main__.py')]
    before = {str(p):sha(p) for p in inputs+sources}
    args.output.mkdir(parents=True)
    (args.output/'data').mkdir()
    config = dict(experiment_id='E011',extension='Waveform/participant views; total response and adjusted waveform deviation',
        source=str(args.source),subjects=status['completed'],representations=REPRESENTATIONS,widths_ms=WIDTHS_MS,
        views=VIEWS,temporal_correlation_interval_ms=[50,1050],
        notes='Correlations run across a declared temporal or channel-time pattern, never across five waveform identities or complementary split repetitions. Local maps use products, signed responses and disagreement.',
        source_config_sha256=sha(args.source/'config.json'),input_sha256={str(p):sha(p) for p in inputs},
        implementation_sha256={str(p):sha(p) for p in sources})
    write_json(args.output/'config.json',config)
    rows,completed = [],[]
    started = time.perf_counter()
    write_json(args.output/'status.json',dict(completed=[],complete=False,errors={}))
    try:
        for subject in status['completed']:
            rows += prepare(subject,args,config)
            completed.append(subject)
            write_json(args.output/'status.json',dict(completed=completed,complete=False,errors={}))
        group_summary(rows,args.output,completed)
        for path,digest in before.items():
            if sha(path)!=digest:
                raise ValueError(f'Input/source changed during waveform extension: {path}')
        from .waveform_report import render
        render(args.output)
        write_json(args.output/'status.json',dict(completed=completed,complete=True,errors={},seconds=time.perf_counter()-started))
    except Exception as exc:
        write_json(args.output/'status.json',dict(completed=completed,complete=False,errors=dict(run=f'{type(exc).__name__}: {exc}')))
        raise
    print(f'Waveform explorer: {(args.output/"index.html").resolve()}',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('output/e011_repeatability_20261030'))
    parser.add_argument('--output',type=Path,default=Path('output/e011_waveform_views_20261030'))
    args = parser.parse_args()
    if args.output.exists():parser.error('Use a fresh output directory')
    with threadpool_limits(limits=2):
        run(args)
