"""Conditional recovery calibration on sign-randomized recorded effect maps.

The injected voltage template is applied after the frozen preprocessing bundle,
before Gaussian differentiation/OLS (implemented by tested linear equivalence).
This calibrates the derivative/group test, not attenuation by ICA or filtering.
"""
import argparse
import json
from pathlib import Path
from experiments.provenance import source_matches
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from experiments.response_characterization.e002_electrode_position_slope.__main__ import sha,write_json
from experiments.response_characterization.e006_multiscale_responses.analysis import transform
from .core import all_signs,sign_statistics,pulse,WAVE_CODES,SPATIAL,SCALES,BRANCHES

AMPLITUDES = (0.,.25,.5,1.,2.,4.,8.)
CENTRES = (.100,.300)
JITTER_SD = (0.,.010)


def added_statistics(noise_means,noise_ss,cross,signal_ss,signal_means,amplitude,n):
    """Exact sign statistics of q*recorded_coefficients + amplitude*signal."""
    means = noise_means+amplitude*signal_means
    ss = noise_ss+amplitude*cross+amplitude**2*signal_ss
    # Reuse the squared-mean buffer instead of allocating it three times.
    np.square(means,out=means)
    variance = -n*means
    variance += ss[None]
    variance /= n*(n-1)
    floor = np.maximum(1.,means)
    floor *= np.finfo(float).eps
    np.maximum(variance,floor,out=variance)
    means /= variance
    return np.sum(means,axis=1)


def corrected_max_p(null_max,observed,families=3):
    return min(1.,families*float(np.mean(null_max>=observed-1e-10)))


def run(args):
    source_config = json.loads((args.source/'config.json').read_text())
    status = json.loads((args.source/'status.json').read_text())
    if not status.get('complete') or status['errors'] or status['completed']!=source_config['subjects']:
        raise ValueError('Require the complete observed follow-up first')
    for path,digest in source_config['implementation'].items():
        if not source_matches(path, digest):
            raise ValueError(f'Observed numerical source changed: {path}')
    subjects = status['completed']; n=len(subjects)
    signs = all_signs(n); nperm=len(signs)
    if args.replicates>nperm:
        raise ValueError('Replicates exceed the available distinct participant sign backgrounds')
    layout = json.loads((args.source/'layout.json').read_text())
    loc = dict(np.load(args.source/'locations.npz'))
    channels=layout['channels']
    spatial = np.array([SPATIAL.get(ch,0.) for ch in channels])
    affected=(loc['period']==1)&(spatial[loc['channel']]!=0)
    indices=np.flatnonzero(affected)
    coefs=[]
    inputs=[args.source/name for name in ['config.json','status.json','layout.json','locations.npz','group_tests.npz']]
    for subject in subjects:
        path=args.source/f'sub-{subject}/effects.npz';inputs.extend([path,path.with_name('audit.json')])
        with np.load(path) as data:
            coefs.append(data['waveform'])
    background=np.stack(coefs);del coefs
    rng=np.random.default_rng(args.seed)
    q_indices=rng.choice(nperm,size=args.replicates,replace=False)
    jitter_standard=np.clip(rng.normal(size=(args.replicates,n)),-2.,2.)
    times=np.arange(-900,651)/1000
    cases=[];derivatives=[]
    for centre in CENTRES:
        for sd in JITTER_SD:
            for repeat,q in enumerate(q_indices):
                latencies=centre+sd*jitter_standard[repeat]
                signals=pulse(times,latencies)
                d=np.stack([transform(signals[:,None,:],1000.,
                    dict(operator='gaussian_derivative',sigma_ms=s))[:,0] for s in SCALES],axis=1)
                derivatives.append(d[:,:,950:1380])  # fixed full post grid [50,480) ms
                cases.append(dict(centre_ms=centre*1000,jitter_sd_ms=sd*1000,repeat=repeat,
                    noise_sign_index=int(q),latencies_s=latencies.tolist()))
    derivatives=np.asarray(derivatives)
    config=dict(source=str(args.source.resolve()),subjects=subjects,amplitudes_uv=AMPLITUDES,
        centres_ms=[100,300],pulse_width_ms=15,pulse_truncate=4,jitter_sd_ms=[0,10],
        jitter_bound_sd=2,replicates=args.replicates,seed=args.seed,spatial=SPATIAL,
        class_amplitudes=WAVE_CODES.tolist(),conditional_noise='Independent whole-participant signs of recorded waveform coefficient vectors; fixed magnitudes, no coefficient centering',
        injection_stage='Same ideal voltage template in each already-preprocessed branch, before local Gaussian differentiation/OLS; no ICA/filter attenuation calibration',
        null='All exact participant sign patterns, full observed location family and factor-three correction; recomputed under each injected alternative',
        limitations='Conditional recovery pilot, not population power, cortical-source model, trial-latency model, preprocessing validation or proof of noise symmetry',
        chunk_size=args.chunk_size,dependencies=source_config['dependencies'],input_sha256={str(p):sha(p) for p in inputs},
        implementation_sha256={str(p):sha(p) for p in [Path(__file__),Path(__file__).with_name('core.py'),
            Path('experiments/response_characterization/e006_multiscale_responses/analysis.py')]})
    write_json(args.output/'config.json',config)
    write_json(args.output/'cases.json',cases)
    # The unmodified locations have the same exact sign orbit for every
    # alternative; only permutation row order changes with the chosen noise sign.
    unaffected_max=np.zeros(nperm);baseline_max=np.zeros(nperm)
    baseline_branch=np.zeros((2,nperm))
    for start in range(0,background.shape[-1],args.chunk_size):
        end=min(start+args.chunk_size,background.shape[-1])
        statistics=sign_statistics(background[:,:,start:end],signs)
        baseline_max=np.maximum(baseline_max,statistics.max(axis=1))
        off=~affected[start:end]
        if off.any():
            unaffected_max=np.maximum(unaffected_max,statistics[:,off].max(axis=1))
        for b in range(2):
            mask=loc['branch'][start:end]==b
            if mask.any():
                baseline_branch[b]=np.maximum(baseline_branch[b],statistics[:,mask].max(axis=1))
        if start==0 or end==background.shape[-1] or end//args.chunk_size%100==0:
            print(f'Sensitivity: cached unchanged null {end:,}/{background.shape[-1]:,} locations',flush=True)
    with np.load(args.source/'group_tests.npz') as old:
        np.testing.assert_allclose(baseline_max,old['waveform_null_maxima'],rtol=1e-10,atol=1e-10)
        # Complete-orbit zero-injection calibration of all three existing
        # families, using the SAME participant signs across families.
        zero={}
        for family in ('waveform','position','interaction'):
            maxima=old[f'{family}_null_maxima']
            order=np.sort(maxima)
            zero[family]=np.minimum(1.,3*(nperm-np.searchsorted(order,maxima-1e-10,side='left'))/nperm)
    zero_frame=pd.DataFrame(zero)
    zero_frame['any_family_alarm']=(zero_frame<=.05).any(axis=1)
    zero_frame.to_csv(args.output/'zero_injection_sign_orbit.csv',index=False)
    reindex=np.array([np.arange(nperm)^case['noise_sign_index']^(nperm-1) for case in cases])
    null_max=np.repeat(unaffected_max[reindex][:,None,:],len(AMPLITUDES)-1,axis=1)
    support_max=np.full((len(cases),len(AMPLITUDES)-1,2),-np.inf)
    affected_background=background[:,:,indices]
    del background
    for start in range(0,len(indices),args.chunk_size):
        end=min(start+args.chunk_size,len(indices))
        chosen=indices[start:end]
        bkg=affected_background[:,:,start:end]
        k=bkg.shape[1];m=bkg.shape[-1]
        means=(signs@bkg.reshape(n,-1)).reshape(nperm,k,m)/n
        ss=np.sum(bkg*bkg,axis=0)
        base_statistics=sign_statistics(bkg,signs)
        weights=spatial[loc['channel'][chosen]]
        scale=loc['scale'][chosen]
        ti=np.rint(loc['time_s'][chosen]*1000).astype(int)-50
        branch=loc['branch'][chosen]
        for c,case in enumerate(cases):
            h=derivatives[c][:,scale,ti]
            signal=h[:,None,:]*WAVE_CODES[:4][None,:,None]*weights[None,None,:]
            active=np.any(signal!=0,axis=(0,1))
            # Most early/late templates are exactly zero in most chunks. Their
            # location statistics are unchanged at every amplitude and can be
            # reused exactly, not estimated or omitted from the global maximum.
            if (~active).any():
                unchanged=base_statistics[reindex[c]][:,~active].max(axis=1)
                null_max[c]=np.maximum(null_max[c],unchanged[None])
            if not active.any():
                continue
            local_signal=signal[:,:,active]
            local_background=bkg[:,:,active]
            signal_means=(signs@local_signal.reshape(n,-1)).reshape(nperm,k,-1)/n
            cross=2*np.sum(local_background*signs[case['noise_sign_index'],:,None,None]*local_signal,axis=0)
            signal_ss=np.sum(local_signal*local_signal,axis=0)
            noise_means=means[reindex[c]][:,:,active]
            for a,amplitude in enumerate(AMPLITUDES[1:]):
                stats=added_statistics(noise_means,ss[:,active],cross,signal_ss,signal_means,amplitude,n)
                null_max[c,a]=np.maximum(null_max[c,a],stats.max(axis=1))
                observed=stats[-1]  # all-positive sign pattern = the observed alternative
                for b in range(2):
                    mask=branch[active]==b
                    if mask.any():
                        support_max[c,a,b]=max(support_max[c,a,b],float(observed[mask].max()))
        if start==0 or end==len(indices) or end//args.chunk_size%4==0:
            print(f'Sensitivity: alternatives at {end:,}/{len(indices):,} injected-support locations',flush=True)
    results=[]
    for c,case in enumerate(cases):
        q=case['noise_sign_index']
        for amplitude in AMPLITUDES:
            a=list(AMPLITUDES).index(amplitude)-1
            distribution=baseline_max[reindex[c]] if amplitude==0 else null_max[c,a]
            observed=baseline_max[q] if amplitude==0 else distribution[-1]
            p_anywhere=corrected_max_p(distribution,observed)
            for b,branch in enumerate(BRANCHES):
                support=baseline_branch[b,q] if amplitude==0 else support_max[c,a,b]
                p=corrected_max_p(distribution,support)
                results.append(dict(**{k:v for k,v in case.items() if k!='latencies_s'},amplitude_uv=amplitude,
                    branch=branch,support_p_fwer=p,support_detected=bool(p<=.05),
                    anywhere_p_fwer=p_anywhere,anywhere_detected=bool(p_anywhere<=.05),
                    outcome='Null alarm (no injected signal)' if amplitude==0 else 'Recovery inside nonzero injected support'))
    frame=pd.DataFrame(results)
    frame.to_csv(args.output/'recovery_replicates.csv',index=False)
    summary=frame.groupby(['centre_ms','jitter_sd_ms','amplitude_uv','branch']).agg(
        replicates=('support_detected','size'),detections=('support_detected','sum'),
        recovery_fraction=('support_detected','mean'),anywhere_fraction=('anywhere_detected','mean')).reset_index()
    summary.to_csv(args.output/'recovery_summary.csv',index=False)
    np.savez_compressed(args.output/'exact_null_maxima.npz',baseline=baseline_max,
        unaffected=unaffected_max,alternative=null_max,support_max=support_max)
    calibration=dict(n_sign_patterns=nperm,source_null_reproduced=True,
        family_alarm_rates={f:float((zero_frame[f]<=.05).mean()) for f in ('waveform','position','interaction')},
        any_family_alarm_rate=float(zero_frame.any_family_alarm.mean()),
        interpretation='Exact finite sign-orbit calibration conditional on recorded maps; does not validate real-data sign symmetry')
    write_json(args.output/'calibration.json',calibration)
    for path,digest in {**config['input_sha256'],**config['implementation_sha256']}.items():
        if sha(Path(path))!=digest:
            raise ValueError(f'Input or implementation changed during sensitivity run: {path}')
    report(args.output,summary,calibration,config)


def report(out,summary,calibration,config):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import html
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained',sharex=True,sharey=True)
    for r,centre in enumerate([100.,300.]):
        for c,sd in enumerate([0.,10.]):
            ax=axes[r,c]
            for branch,color in zip(BRANCHES,['#4c78a8','#e88725']):
                d=summary[(summary.centre_ms==centre)&(summary.jitter_sd_ms==sd)&
                    (summary.branch==branch)&(summary.amplitude_uv>0)]
                ax.plot(d.amplitude_uv,d.recovery_fraction,'o-',color=color,label=branch)
            ax.set(title=f'Pulse centre {centre:g} ms · participant jitter SD {sd:g} ms',
                xlabel='Nominal maximum class peak (µV)',ylabel='Conditional recovery fraction',ylim=(-.03,1.03),xscale='log')
            ax.legend()
    fig.suptitle(f'Recorded-effect sign-background sensitivity · {config["replicates"]} replicates per setting\n'
        'Frozen preprocessing; same ideal injection in both branches; full corrected waveform family')
    fig.savefig(out/'recovery_curves.png',dpi=150);plt.close(fig)
    lines=['# Conditional recorded-noise sensitivity pilot','',
        'This is a calibration of the expanded time-resolved waveform test, not a new classification result. '
        'It does NOT calibrate multiscale RMS or repeatability statistics, or signal attenuation by preprocessing.','',
        '## Frozen simulation','',
        f'{len(config["subjects"])} recorded participant coefficient vectors, {config["replicates"]} distinct '
        f'random participant-sign backgrounds per setting (seed {config["seed"]}), and amplitudes 0/0.25/0.5/1/2/4/8 µV. '
        'The same sign background is shared across amplitudes, branches and latency scenarios. '
        'A 15-ms-width Gaussian pulse (truncated ±60 ms) is centred at 100 or 300 ms. '
        'Between-participant jitter is 0 or 10-ms SD, clipped at ±2 SD; there is no trial-level jitter. '
        'The earliest possible pulse starts at +20 ms. Five class amplitudes are −1/−0.5/0/0.5/1 times '
        'the stated amplitude. Fixed sparse scalp weights: C3=1, CP1=0.5, C4=−1, CP2=−0.5, others zero. '
        'They sum to zero, preserving average reference; this is a benchmark pattern, not a neural-source model.','',
        'An identical ideal voltage signal is introduced into each already-preprocessed branch before '
        'Gaussian differentiation/OLS, implemented by tested linear equivalence. The original sign-randomized '
        'recorded coefficient vectors provide the background. Their full spatial/temporal covariance and '
        'participant magnitudes are retained; they are not pure instrument noise. There is no coefficient '
        'centering, trial resampling, new ICA fitting or assumption that real waveform effects are absent.','',
        'Every alternative recomputes ALL exact participant sign statistics. Unchanged-location null maxima '
        'are reused only through a proved sign-index permutation; affected locations are recomputed. '
        'The global null still spans both branches, both periods, all scales, valid times and electrodes, '
        'with the same factor-three correction. Recovery requires a corrected effect inside nonzero injected '
        'support, not an unrelated significant location.','',
        '## Zero-injection calibration','',
        f'All {calibration["n_sign_patterns"]} sign backgrounds were enumerated without injection. '
        f'The union alarm fraction across three corrected families was {calibration["any_family_alarm_rate"]:.4f}. '
        'This checks the exact finite sign-orbit implementation; it does not establish that real participant '
        'coefficient errors satisfy the required symmetry. The original recorded waveform null maxima '
        'were reproduced.','',
        '## Recovery','',
        'Recovery frequencies are conditional pilot summaries over the chosen finite recorded backgrounds '
        'and jitter draws. They are not a population-power estimate, a minimum detectable physiological '
        'effect, a superiority test between pipelines or a detection guarantee. The finite number of '
        'replicates limits precision. Zero-amplitude rows in CSVs describe false alarms, not recovery.','',
        '| Centre (ms) | Jitter SD (ms) | Nominal peak (µV) | Minimal recovered | Supplied recovered |',
        '| --- | --- | --- | --- | --- |']
    pivot=summary.pivot(index=['centre_ms','jitter_sd_ms','amplitude_uv'],columns='branch',values='detections')
    for (centre,sd,amp),row in pivot.iterrows():
        if amp==0:continue
        lines.append(f'| {centre:g} | {sd:g} | {amp:g} | {int(row["minimal"])}/{config["replicates"]} | {int(row["cleaned"])}/{config["replicates"]} |')
    lines+=['','![Recovery curves](recovery_curves.png)','',
        'Existing group effects, broad correction, ten participants, template choice and fixed participant '
        'magnitudes all constrain interpretation. The late and early pulse scenarios are prespecified; '
        'no location, scale, template or amplitude was tuned after reviewing recovery.']
    text='\n'.join(lines)+'\n';(out/'RESULTS.md').write_text(text,encoding='utf-8')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Sensitivity pilot</title>'
        '<style>body{max-width:1150px;margin:30px auto;font:16px/1.5 system-ui}pre{white-space:pre-wrap}img{max-width:100%}</style>'
        '<pre>'+html.escape(text)+'</pre><img src="recovery_curves.png">',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('output/task4_response_followup_20261023'))
    parser.add_argument('--output',type=Path,default=Path('output/task4_response_sensitivity_20261023'))
    parser.add_argument('--replicates',type=int,default=32)
    parser.add_argument('--seed',type=int,default=20261023)
    parser.add_argument('--chunk-size',type=int,default=256)
    args=parser.parse_args()
    if args.output.exists():parser.error('Use a fresh output directory')
    if not 2<=args.replicates<=1024 or args.chunk_size<1 or args.seed<0:
        parser.error('Invalid replicate count/chunk size/seed')
    args.output.mkdir(parents=True)
    started=time.perf_counter()
    try:
        with threadpool_limits(limits=2):run(args)
        write_json(args.output/'status.json',dict(complete=True,seconds=time.perf_counter()-started,errors={}))
    except Exception as exc:
        write_json(args.output/'status.json',dict(complete=False,errors=dict(run=f'{type(exc).__name__}: {exc}')))
        raise


if __name__=='__main__':main()
