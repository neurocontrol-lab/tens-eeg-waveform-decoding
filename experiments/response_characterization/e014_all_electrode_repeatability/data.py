"""Reconstruct all scalp channels, preserving E011 trials and simple filtering."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from experiments.waveform_decoding.e001_voltage_baseline.events import load_recording,trigger_runs,align_events,read_behavior
from experiments.waveform_decoding.e001_voltage_baseline.features import CHANNELS
from experiments.signal_quality.e010_simple_visual_inspection.core import filter_continuous,extract_epochs,epoch_eligibility
from experiments.response_characterization.e011_electrode_bin_repeatability.__main__ import sha,write_json,verify_source


def channel_audit(raw_v,epochs_uv,times,names):
    """Fixed descriptive flags, without channel selection/interpolation/exclusions."""
    rows=[]
    post=(times>=.05)&(times<1.05)
    for c,name in enumerate(names):
        raw=raw_v[c]
        finite=np.isfinite(raw)
        std=float(np.std(raw[finite])*1e6)
        values=epochs_uv[:,c,post]
        ptp=np.ptp(values,axis=-1)
        sd=np.std(values,axis=-1)
        flags=[]
        if not finite.all():flags.append('nonfinite_raw_samples')
        if std<.1:flags.append('near_flat_raw_std_below_0.1uV')
        if np.any(ptp>500.):flags.append('post_epoch_ptp_above_500uV')
        if np.any(sd<.1):flags.append('post_epoch_std_below_0.1uV')
        rows.append(dict(channel=name,raw_continuous_std_uv=std,
            raw_nonfinite_fraction=float((~finite).mean()),post_median_ptp_uv=float(np.median(ptp)),
            post_max_ptp_uv=float(ptp.max()),post_fraction_above_500uv=float(np.mean(ptp>500)),
            post_fraction_below_0_1uv_std=float(np.mean(sd<.1)),flags=';'.join(flags)))
    return pd.DataFrame(rows)


def prepare(subject,args):
    print(f'Subject {subject}: reconstruct all 62 scalp electrodes',flush=True)
    source=args.data_dir/f'Subject{subject}_data/EEGDATA_SUB{subject}-T4.xdf'
    old_visual=args.visual/f'sub-{subject}'
    old_audit=json.loads((old_visual/'audit.json').read_text())
    digest=sha(source)
    if digest!=old_audit['source_sha256']:raise ValueError('Raw recording differs from E010')
    data,names,triggers,timestamps,sfreq=load_recording(source)
    if len(names)!=62 or any(n in names for n in ['CREF','M1','M2']):raise ValueError('Expected 62 scalp channels')
    meta=pd.read_csv(args.source/f'sub-{subject}/eligible_events.csv')
    aligned=align_events(trigger_runs(triggers,sfreq),read_behavior(args.master_csv,subject),subject,timestamps)
    selected=aligned.iloc[meta.event_index.to_numpy()].reset_index(drop=True)
    fields=['sample','trigger','pair_id','position','comparison','elapsed_s']
    pd.testing.assert_frame_equal(selected[fields],meta[fields],check_dtype=False,atol=1e-8,rtol=0)
    raw_std=np.std(data,axis=1)*1e6
    raw_bad=np.mean(~np.isfinite(data),axis=1)
    # Same 62-channel reference as the old 14-channel analysis, not a changed reference.
    data-=data.mean(axis=0,keepdims=True)
    filtered,edge_ok,sos,segments=filter_continuous(data,timestamps,sfreq)
    manifest=epoch_eligibility(meta,timestamps,edge_ok,sfreq)
    if not manifest.visual_pair_eligible.all():raise ValueError('All-channel support would change the frozen E011 cohort')
    epochs,times=extract_epochs(filtered,meta['sample'].to_numpy(),sfreq)
    # Audit referenced pre-filter data for consistency, then restore true raw std/nonfinite summaries.
    audit=channel_audit(data,epochs,times,names)
    audit['raw_continuous_std_uv']=raw_std
    audit['raw_nonfinite_fraction']=raw_bad
    for c in range(len(names)):
        flags=[x for x in audit.loc[c,'flags'].split(';') if x and not x.startswith(('near_flat_raw','nonfinite_raw'))]
        if raw_std[c]<.1:flags.append('near_flat_raw_std_below_0.1uV')
        if raw_bad[c]>0:flags.append('nonfinite_raw_samples')
        audit.loc[c,'flags']=';'.join(flags)
    del data,filtered,triggers,timestamps
    old_meta=pd.read_csv(old_visual/'eligible_events.csv')
    ix=pd.Index(old_meta.event_index).get_indexer(meta.event_index)
    picks=[names.index(ch) for ch in CHANNELS]
    old=np.load(old_visual/'bandpass_epochs_uv.npy',mmap_mode='r')[ix]
    np.testing.assert_allclose(epochs[:,picks],old,rtol=0,atol=1e-9)
    error=float(np.max(np.abs(epochs[:,picks]-old)))
    destination=args.output/f'sub-{subject}';destination.mkdir()
    np.save(destination/'epochs_uv.npy',epochs)
    np.savez(destination/'layout.npz',time_s=times,channels=np.array(names),sfreq=sfreq,sos=sos)
    meta.to_csv(destination/'events.csv',index=False)
    audit.to_csv(destination/'channel_quality.csv',index=False)
    pd.read_csv(args.source/f'sub-{subject}/split_assignments.csv').to_csv(destination/'split_assignments.csv',index=False)
    write_json(destination/'audit.json',dict(subject=subject,n_events=len(meta),n_pairs=meta.pair_id.nunique(),
        n_channels=len(names),raw_source_sha256=digest,old14_epoch_max_error_uv=error,
        segments=segments,flagged_channels=int(audit['flags'].ne('').sum()),channels_excluded=0,
        pairs_excluded=0,ica_refitted=False,classifier_fitted=False))
    print(f'  {len(meta)} events; original 14-channel maximum error {error:.3g} uV; flags only, no new exclusions',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('output/e011_repeatability_20261030'))
    parser.add_argument('--visual',type=Path,default=Path('output/task4_visual_inspection_20261030'))
    parser.add_argument('--data-dir',type=Path,default=Path('EEGData'))
    parser.add_argument('--master-csv',type=Path,default=Path('EEG_Codes_Export/02_SourceLocalization_Task4/Master_Data_ToSub12 (1).csv'))
    parser.add_argument('--output',type=Path,default=Path('output/e014_all_electrode_cache_20261030'))
    args=parser.parse_args()
    if args.output.exists():parser.error('Use a fresh cache directory')
    config0,status0=verify_source(args.visual)
    if sha(args.master_csv)!=config0['master_csv_sha256']:raise ValueError('CSV differs')
    status=json.loads((args.source/'status.json').read_text())
    if not status['complete'] or status['errors']:raise ValueError('E011 incomplete')
    inputs=[args.source/'config.json',args.source/'status.json',args.visual/'run_config.json',args.master_csv]
    for s in status['completed']:
        inputs += [args.source/f'sub-{s}/eligible_events.csv',args.source/f'sub-{s}/split_assignments.csv',
            args.visual/f'sub-{s}/audit.json',args.visual/f'sub-{s}/bandpass_epochs_uv.npy',args.visual/f'sub-{s}/eligible_events.csv']
    sources=[Path(__file__),Path('experiments/signal_quality/e010_simple_visual_inspection/core.py'),Path('experiments/waveform_decoding/e001_voltage_baseline/events.py')]
    before={str(p):sha(p) for p in inputs+sources}
    args.output.mkdir(parents=True)
    write_json(args.output/'config.json',dict(experiment_id='E014',subjects=status['completed'],
        source=str(args.source),visual_source=str(args.visual),input_sha256={str(p):before[str(p)] for p in inputs},
        implementation_sha256={str(p):before[str(p)] for p in sources},
        preprocessing=config0['filter'],channel_policy='All scalp electrodes; descriptive flags only; frozen E011 cohort',
        dependencies={n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','pyxdf']}))
    completed=[]
    try:
        with threadpool_limits(limits=2):
            for s in status['completed']:
                prepare(s,args);completed.append(s)
                write_json(args.output/'status.json',dict(completed=completed,errors={},complete=False))
        for p,digest in before.items():
            if sha(p)!=digest:raise ValueError(f'Input/source changed during reconstruction: {p}')
        write_json(args.output/'status.json',dict(completed=completed,errors={},complete=True))
    except Exception as exc:
        write_json(args.output/'status.json',dict(completed=completed,errors={'run':f'{type(exc).__name__}: {exc}'},complete=False));raise


if __name__=='__main__':main()
