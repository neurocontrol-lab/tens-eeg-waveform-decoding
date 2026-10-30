"""All-channel sensor/power maps and calibration-isolated xDAWN summaries."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.waveform_decoding.e001_voltage_baseline.events import CODES,WAVEFORMS
from experiments.waveform_decoding.e001_voltage_baseline.features import CHANNELS as ORIGINAL_CHANNELS
from experiments.response_characterization.e011_electrode_bin_repeatability.core import (
    WIDTHS_MS,REPRESENTATIONS,UNITS,local_representations,bin_features,split_operators,waveform_operator)
from experiments.response_characterization.e011_electrode_bin_repeatability.waveform_views import (
    observed_weights,total_split_operators,individual_estimates,pattern_r,median_finite,encode,finite_list)
from experiments.response_characterization.e011_electrode_bin_repeatability.__main__ import write_json
from .features import power_features,calibration_splits,fit_spatial,centre_feature_time

FIELDS=['mean_split_product','rms_half_disagreement','split_sign_agreement','full_response','example_half_a','example_half_b']


def support_rows(meta,assignments):
    result=[]
    for p in (1,2):
        for code in CODES:
            counts=[]
            for _,group in assignments.groupby(['repeat','half']):
                counts.append(int((meta.pair_id.isin(group.pair_id)&(meta.position==p)&(meta.trigger==code)).sum()))
            result.append(dict(position=p,trigger=code,n_events=int(((meta.position==p)&(meta.trigger==code)).sum()),
                minimum_half_events=min(counts),maximum_half_events=max(counts)))
    return result


def save_payload(out,subject,representation,payload):
    key=f'{subject}:{representation}'
    script='window.e011WaveData=window.e011WaveData||{};window.e011WaveData['+json.dumps(key)+']='+json.dumps(payload,allow_nan=False,separators=(',',':'))+';\n'
    (out/'data'/f'sub-{subject}_{representation}.js').write_text(script,encoding='utf-8')


def append_rows(rows,subject,representation,width,view,estimates,unit,mask):
    for p in range(2):
        for w,code in enumerate(CODES):
            rows.append(dict(subject=subject,representation=representation,width_ms=width,view=view,
                position=p+1,trigger=code,waveform=WAVEFORMS[code],
                joint_pattern_r=estimates['joint_profile_r'][p,w],
                mean_channel_temporal_r=float(np.nanmean(estimates['temporal_profile_r'][p,w])),
                mean_location_product=float(estimates['mean_split_product'][p,w][:,mask].mean()),unit=unit))


def packed(estimates):
    return dict(**{k:encode(estimates[k]) for k in FIELDS},
        temporal_profile_r=finite_list(estimates['temporal_profile_r']),joint_profile_r=finite_list(estimates['joint_profile_r']))


def sensor_subject(subject,args):
    src=args.cache/f'sub-{subject}';dest=args.output/f'sub-{subject}';dest.mkdir(exist_ok=True)
    meta=pd.read_csv(src/'events.csv');epochs=np.load(src/'epochs_uv.npy',mmap_mode='r')
    with np.load(src/'layout.npz') as layout:times=layout['time_s'];names=layout['channels'].tolist()
    quality=pd.read_csv(src/'channel_quality.csv').fillna('')
    quality.to_csv(dest/'channel_quality.csv',index=False)
    operators,assignments,_=split_operators(meta,subject)
    pd.testing.assert_frame_equal(assignments,pd.read_csv(src/'split_assignments.csv'),check_dtype=False)
    total=total_split_operators(meta,assignments,len(operators))
    full,_,_=waveform_operator(meta);full_total=observed_weights(meta)
    support=support_rows(meta,assignments)
    values=local_representations(epochs,times,1000.)
    powers,p_left,p_right,p_audit=power_features(epochs,times)
    write_json(dest/'power_audit.json',p_audit)
    rows=[];reproduction=[]
    for representation in list(REPRESENTATIONS)+['alpha_power','beta_power']:
        is_power=representation in powers
        unit='dB re 1 uV^2' if is_power else UNITS[representation]
        method=('500-ms Hann total-power estimates, 50-ms step, 2-Hz Fourier spacing. Markers are window centres; all true windows lie in [50,1050) ms. Not induced power.' if is_power else
            'Preserved E011 two-way pair splits, now all 62 scalp electrodes. Fixed 25-ms local derivatives; no new exclusions or learned spatial features.')
        payload=dict(subject=subject,representation=representation,unit=unit,channels=names,
            quality=quality[['channel','flags']].to_dict('records'),waves=[WAVEFORMS[c] for c in CODES],
            support=support,method_note=method,estimate_label='Full-data estimate',widths={})
        for width in ([500] if is_power else WIDTHS_MS):
            if is_power:
                features,left,right=powers[representation],p_left,p_right
                mask=np.ones(len(left),bool);step=50
            else:
                features,left,right=bin_features(values[representation.split('_')[0]],times,width,rms=representation.endswith('_rms'))
                mask=(left>=50)&(right<=1050);step=width
            views={}
            for view,op,fop in [('specific',operators,full),('total',total,full_total)]:
                estimates=individual_estimates(features,op,fop,mask,specific=(view=='specific'))
                if not is_power:
                    pick=[names.index(n) for n in ORIGINAL_CHANNELS]
                    oldfile=args.waveform_source/f'sub-{subject}/w{width:03d}_{representation}_{view}.npz'
                    with np.load(oldfile) as old:
                        for field in FIELDS+['mean_half_energy','temporal_profile_r','valid_temporal_splits']:
                            np.testing.assert_allclose(estimates[field][:,:,pick],old[field],rtol=2e-10,atol=1e-8,equal_nan=True)
                    reproduction.append(dict(representation=representation,width_ms=width,view=view,original14_verified=True))
                np.savez_compressed(dest/f'w{width:03d}_{representation}_{view}.npz',**estimates,left_ms=left,right_ms=right,channels=np.array(names))
                views[view]=packed(estimates)
                append_rows(rows,subject,representation,width,view,estimates,unit,mask)
            payload['widths'][str(width)]=dict(bins=len(left),left_ms=float(left[0]),right_ms=float(right[-1]),
                centres_ms=((left+right)/2).tolist(),step_ms=step,views=views)
        save_payload(args.output,subject,representation,payload)
        print(f'Subject {subject}: all-electrode {representation} complete',flush=True)
    pd.DataFrame(rows).to_csv(dest/'sensor_waveform_summary.csv',index=False)
    pd.DataFrame(reproduction).to_csv(dest/'original14_reproduction.csv',index=False)
    return rows


def accumulate_profiles(batches):
    """Summarize matched calibration-conditional feature profiles.

    Joint shape r demeans each feature over time before flattening, so a joint
    sign flip of an xDAWN component in both halves cannot change agreement.
    """
    products=disagreement=signs=energy=None;temporal=[];joint=[];count=0;examples={}
    for a,b in batches:
        if products is None:
            products=np.zeros(a.shape[1:]);disagreement=np.zeros_like(products)
            signs=np.zeros_like(products);energy=np.zeros_like(products)
            examples=dict(example_half_a=a[0].copy(),example_half_b=b[0].copy(),full_response=.5*(a[0]+b[0]))
        products+=(a*b).sum(axis=0);disagreement+=((a-b)**2).sum(axis=0)
        signs+=(a*b>0).sum(axis=0);energy+=(.5*(a*a+b*b)).sum(axis=0);count+=len(a)
        temporal.append(pattern_r(a,b))
        ac,bc=centre_feature_time(a),centre_feature_time(b)
        joint.append(pattern_r(ac.reshape(len(a),2,5,-1),bc.reshape(len(b),2,5,-1)))
    tr,jr=np.concatenate(temporal),np.concatenate(joint)
    return dict(**examples,mean_split_product=products/count,rms_half_disagreement=np.sqrt(disagreement/count),
        split_sign_agreement=signs/count,mean_half_energy=energy/count,
        temporal_profile_r=median_finite(tr),joint_profile_r=median_finite(jr),
        valid_temporal_splits=np.isfinite(tr).sum(axis=0),valid_joint_splits=np.isfinite(jr).sum(axis=0))


def xdawn_subject(subject,args):
    import mne
    mne.set_log_level('ERROR')
    src=args.cache/f'sub-{subject}';dest=args.output/f'sub-{subject}';dest.mkdir(exist_ok=True)
    meta=pd.read_csv(src/'events.csv');epochs=np.load(src/'epochs_uv.npy',mmap_mode='r')
    with np.load(src/'layout.npz') as layout:times=layout['time_s'];names=layout['channels'].tolist()
    mask=(times>=.05)&(times<1.05)
    partitions,allocations,designs=calibration_splits(meta,subject)
    allocations.to_csv(dest/'calibration_allocations.csv',index=False);designs.to_csv(dest/'calibration_evaluation_designs.csv',index=False)
    cal_filters=[];cal_sensor_rms=[];wave_ops=[];total_ops=[]
    # Hold out calibration from both repeatability halves. Fit once per split;
    # reuse its frozen linear projection across all four reporting widths.
    for repeat,(cal,a,b) in enumerate(partitions):
        f,rms,classes=fit_spatial(np.asarray(epochs[cal][...,mask]),meta.iloc[cal].trigger.to_numpy())
        cal_filters.append(f);cal_sensor_rms.append(rms)
        half_w=[];half_t=[]
        for ix in (a,b):
            small=meta.iloc[ix].reset_index(drop=True)
            op,_,_=waveform_operator(small)
            w=np.zeros((2,5,len(meta)));w[...,ix]=op;half_w.append(w)
            t=np.zeros_like(w);t[...,ix]=observed_weights(small);half_t.append(t)
        wave_ops.append(half_w);total_ops.append(half_t)
        if repeat==0 or (repeat+1)%20==0:print(f'Subject {subject}: xDAWN calibration {repeat+1}/100 (no classifier)',flush=True)
    filters=np.stack(cal_filters);sensor_rms=np.stack(cal_sensor_rms)
    wave_ops,total_ops=np.stack(wave_ops),np.stack(total_ops)
    np.savez_compressed(dest/'calibration_spatial_fits.npz',filters=filters,sensor_rms=sensor_rms,classes=classes)
    eval_assignments=allocations[allocations.role!='calibration'].rename(columns={'role':'half'})
    support=support_rows(meta,eval_assignments)
    picks=[names.index(n) for n in ORIGINAL_CHANNELS]
    rows=[]
    for representation,channels in [('calibrated_sensors62',names),('calibrated_sensors14',list(ORIGINAL_CHANNELS)),
            ('xdawn_components',[f'{WAVEFORMS[c]} component {i+1}' for c in CODES for i in range(2)])]:
        method=('Independent calibration/A/B pair partitions; same held-out A/B trials as both calibrated-sensor controls. '
            'Calibration-only RMS normalization. Joint r demeans each feature time course before flattening. '
            'xDAWN component names track calibration class/rank, not fixed electrodes or fixed brain sources. '
            'The plotted grey estimate is the split-0 held-out A/B average, not a full-data fit.')
        payload=dict(subject=subject,representation=representation,unit='calibration RMS units',channels=channels,
            quality=[dict(channel=n,flags='Learned component, not an electrode' if representation=='xdawn_components' else '') for n in channels],
            waves=[WAVEFORMS[c] for c in CODES],support=support,method_note=method,
            estimate_label='Split-0 held-out A/B average',widths={})
        for width in [25,50,100,200]:
            features,left,right=bin_features(epochs,times,width)
            keep=right<=1050;features=features[...,keep];left=left[keep];right=right[keep]
            flat=features.reshape(len(meta),-1);views={}
            for view,operators in [('specific',wave_ops),('total',total_ops)]:
                def batches():
                    for start in range(0,len(operators),5):
                        op=operators[start:start+5]
                        sensor=(op.reshape(-1,len(meta))@flat).reshape(len(op),2,2,5,len(names),-1)
                        if view=='specific':sensor-=sensor.mean(axis=3,keepdims=True)
                        if representation=='xdawn_components':
                            projected=np.einsum('rkc,rhpwct->rhpwkt',filters[start:start+len(op)],sensor,optimize=True)
                        else:
                            projected=sensor/sensor_rms[start:start+len(op),None,None,None,:,None]
                            if representation=='calibrated_sensors14':projected=projected[...,picks,:]
                        yield projected[:,0],projected[:,1]
                estimates=accumulate_profiles(batches())
                np.savez_compressed(dest/f'w{width:03d}_{representation}_{view}.npz',**estimates,
                    left_ms=left,right_ms=right,channels=np.array(channels))
                views[view]=packed(estimates)
                append_rows(rows,subject,representation,width,view,estimates,'calibration RMS units',np.ones(len(left),bool))
            payload['widths'][str(width)]=dict(bins=len(left),left_ms=float(left[0]),right_ms=float(right[-1]),
                centres_ms=((left+right)/2).tolist(),step_ms=width,views=views)
        save_payload(args.output,subject,representation,payload)
        print(f'Subject {subject}: {representation} matched repeatability complete',flush=True)
    pd.DataFrame(rows).to_csv(dest/'calibrated_waveform_summary.csv',index=False)
    return rows
