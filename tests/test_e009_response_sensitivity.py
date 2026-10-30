import numpy as np
from experiments.uncertainty_and_sensitivity.e009_response_uncertainty_sensitivity.core import all_signs,sign_statistics
from experiments.uncertainty_and_sensitivity.e009_response_uncertainty_sensitivity.sensitivity import added_statistics,corrected_max_p


def test_cached_unaffected_null_and_injected_alternatives_equal_full_brute_force():
    rng=np.random.default_rng(192)
    background=rng.normal(size=(6,4,23));signal=np.zeros_like(background)
    affected=np.array([1,4,7,9,18])
    signal[:,:,affected]=rng.normal(size=(6,4,len(affected)))
    signs=all_signs(6);n=len(background)
    off=np.ones(23,bool);off[affected]=False
    base_off=sign_statistics(background[:,:,off],signs).max(axis=1)
    b=background[:,:,affected];s=signal[:,:,affected]
    noise_means=(signs@b.reshape(n,-1)).reshape(len(signs),4,-1)/n
    signal_means=(signs@s.reshape(n,-1)).reshape(len(signs),4,-1)/n
    for q in (0,9,63):
        index=np.arange(len(signs))^q^(len(signs)-1)
        cross=2*np.sum(b*signs[q,:,None,None]*s,axis=0)
        for amplitude in (0.,.25,2.):
            stats=added_statistics(noise_means[index],np.sum(b*b,axis=0),cross,
                np.sum(s*s,axis=0),signal_means,amplitude,n)
            cached=np.maximum(base_off[index],stats.max(axis=1))
            actual=sign_statistics(background*signs[q,:,None,None]+amplitude*signal,signs)
            np.testing.assert_allclose(cached,actual.max(axis=1),atol=1e-10)
            np.testing.assert_allclose(stats[-1],actual[-1,affected],atol=1e-10)
            expected=min(1.,3*float(np.mean(actual.max(axis=1)>=actual[-1].max()-1e-10)))
            assert corrected_max_p(cached,actual[-1].max())==expected


def test_complete_sign_orbit_controls_three_family_null_alarm_and_detects_strong_template():
    rng=np.random.default_rng(77);signs=all_signs(8)
    alarms=[]
    for k in (4,1,4):
        background=rng.normal(size=(8,k,11))
        maxima=sign_statistics(background,signs).max(axis=1)
        alarms.append(np.array([corrected_max_p(maxima,t)<=.05 for t in maxima]))
    assert np.any(alarms,axis=0).mean()<=.05
    background=rng.normal(0,.1,size=(8,4,11));background[:,0,5]+=5
    stats=sign_statistics(background,signs)
    assert corrected_max_p(stats.max(axis=1),stats[-1,5])<.05


def test_complete_conditional_simulation_matches_full_null_with_zero_signal_chunks(tmp_path,monkeypatch):
    import json
    import argparse
    import pandas as pd
    from experiments.uncertainty_and_sensitivity.e009_response_uncertainty_sensitivity import sensitivity as module
    from experiments.uncertainty_and_sensitivity.e009_response_uncertainty_sensitivity.core import pulse,SPATIAL,WAVE_CODES
    from experiments.response_characterization.e006_multiscale_responses.analysis import transform
    source=tmp_path/'source';source.mkdir()
    destination=tmp_path/'result';destination.mkdir()
    subjects=[3,4,5];channels=list(SPATIAL)
    (source/'config.json').write_text(json.dumps(dict(subjects=subjects,implementation={},dependencies={})))
    (source/'status.json').write_text(json.dumps(dict(complete=True,errors={},completed=subjects)))
    (source/'layout.json').write_text(json.dumps(dict(channels=channels)))
    rows=[(b,p,0,t,c) for b in (0,1) for p in (0,1)
          for t in ((-.4,-.1) if p==0 else (.05,.1,.15,.4)) for c in range(4)]
    loc={key:np.array([r[i] for r in rows]) for i,key in enumerate(['branch','period','scale','time_s','channel'])}
    np.savez(source/'locations.npz',**loc)
    rng=np.random.default_rng(442);signs=all_signs(3)
    background=rng.normal(size=(3,4,len(rows)))
    group={}
    for family,k in [('waveform',4),('position',1),('interaction',4)]:
        coefficients=background if family=='waveform' else rng.normal(size=(3,k,len(rows)))
        group[family+'_null_maxima']=sign_statistics(coefficients,signs).max(axis=1)
    np.savez(source/'group_tests.npz',**group)
    for i,subject in enumerate(subjects):
        d=source/f'sub-{subject}';d.mkdir()
        np.savez(d/'effects.npz',waveform=background[i])
        (d/'audit.json').write_text('{}')
    monkeypatch.setattr(module,'AMPLITUDES',(0.,.5))
    monkeypatch.setattr(module,'CENTRES',(.1,))
    monkeypatch.setattr(module,'JITTER_SD',(0.,))
    monkeypatch.setattr(module,'report',lambda *args:None)
    args=argparse.Namespace(source=source,output=destination,seed=12,replicates=2,chunk_size=4)
    module.run(args)
    cases=json.loads((destination/'cases.json').read_text())
    times=np.arange(-900,651)/1000
    h=transform(pulse(times,[.1])[:,None,:],1000.,dict(operator='gaussian_derivative',sigma_ms=2.5))[0,0]
    spatial=np.array(list(SPATIAL.values()))
    signal=np.zeros_like(background)
    for j,row in enumerate(rows):
        b,p,s,t,c=row
        if p==1:
            signal[:,:,j]=WAVE_CODES[:4][None,:]*spatial[c]*h[round(t*1000)+900]
    with np.load(destination/'exact_null_maxima.npz') as data:
        for c,case in enumerate(cases):
            q=case['noise_sign_index']
            brute=sign_statistics(background*signs[q,:,None,None]+.5*signal,signs)
            np.testing.assert_allclose(data['alternative'][c,0],brute.max(axis=1),atol=1e-10)
    assert len(pd.read_csv(destination/'recovery_replicates.csv'))==8
