import itertools
import numpy as np
import pandas as pd
from experiments.uncertainty_and_sensitivity.e009_response_uncertainty_sensitivity.core import (supports,coefficient_traces,
    build_layout,all_signs,sign_statistics,pulse,WAVE_CODES,SPATIAL)
from experiments.uncertainty_and_sensitivity.e009_response_uncertainty_sensitivity.uncertainty import median_intervals
from experiments.response_characterization.e006_multiscale_responses.analysis import transform
from experiments.response_characterization.e008_time_resolved_effects.analysis import fit_trace,regression_operator,sign_flip_family


def metadata():
    rows=[]
    for c,(a,b) in enumerate(itertools.combinations(range(239,244),2)):
        for trial in range(4):
            for pos,code in enumerate((a,b) if trial%2 else (b,a),1):
                rows.append(dict(trigger=code,position=pos,comparison=f'{a}:{b}',elapsed_s=c*100+trial*3+pos))
    return pd.DataFrame(rows)


def test_each_scale_has_earlier_matched_guarded_support():
    t=np.arange(-900,651)/1000
    masks=supports(t,1000)
    assert [np.flatnonzero(masks['post',s])[0]-900 for s in (2.5,5,10,20,40)]==[50,50,51,91,171]
    for s in (2.5,5,10,20,40):
        assert masks['pre',s].sum()==masks['post',s].sum()
        impulse=np.zeros((1,1,len(t))); impulse[:,:,t==0]=1e6
        y=transform(impulse,1000,dict(operator='gaussian_derivative',sigma_ms=s))
        for period in ('pre','post'):
            np.testing.assert_allclose(y[:,:,masks[period,s]],0.,atol=1e-12)
    _,loc=build_layout(t,1000,['C3','C4'])
    assert len(loc['branch'])==2*2*sum(masks['post',s].sum() for s in (2.5,5,10,20,40))*2


def test_ols_and_gaussian_differentiation_commute():
    meta=metadata(); rng=np.random.default_rng(92)
    x=rng.normal(size=(len(meta),3,1551))
    beta,indices,_=coefficient_traces(x,meta)
    operator,slices,_=regression_operator(meta)
    spec=dict(operator='gaussian_derivative',sigma_ms=10.)
    direct=fit_trace(transform(x,1000,spec),operator,slices)
    efficient=transform(beta,1000,spec)
    for family in indices:
        np.testing.assert_allclose(efficient[indices[family]].transpose(0,2,1),direct[family],atol=1e-10)


def test_injected_voltage_template_matches_coefficient_level_linear_injection():
    meta=metadata(); t=np.arange(-900,651)/1000
    weights=np.array(list(SPATIAL.values()))
    assert weights.sum()==0
    signal=pulse(t,[.1])[0]
    x=WAVE_CODES[meta.trigger.to_numpy()-239,None,None]*weights[None,:,None]*signal[None,None,:]
    beta,indices,_=coefficient_traces(x,meta)
    np.testing.assert_allclose(beta[indices['waveform']],WAVE_CODES[:4,None,None]*weights[None,:,None]*signal[None,None,:],atol=1e-14)
    np.testing.assert_allclose(beta[indices['position']],0.,atol=1e-14)
    np.testing.assert_allclose(beta[indices['interaction']],0.,atol=1e-14)
    assert np.all(pulse(t,[.08,.12])[:,t<.02]==0)


def test_exact_statistics_and_sign_reindexing_match_preserved_inference():
    rng=np.random.default_rng(4); x=rng.normal(size=(5,4,17))
    signs=all_signs(5)
    stat=sign_statistics(x,signs)
    np.testing.assert_allclose(stat.max(axis=1),sign_flip_family(x)['null_maxima'],atol=1e-10)
    for q in (0,3,31):
        reindex=np.arange(len(signs))^q^(len(signs)-1)
        np.testing.assert_allclose(sign_statistics(x*signs[q,:,None,None],signs),stat[reindex],atol=1e-10)


def test_participant_bootstrap_preserves_paired_identity_and_ignores_split_count():
    x=np.array([[.1,.3],[.2,.4],[.3,.5],[.4,.6]])
    centre,interval=median_intervals(x,repeats=1000)
    np.testing.assert_allclose(centre,[.25,.45])
    np.testing.assert_allclose(interval[:,1]-interval[:,0],.2)
    diff=x[:,1]-x[:,0]
    c,ci=median_intervals(diff[:,None],repeats=1000)
    np.testing.assert_allclose(c,.2);np.testing.assert_allclose(ci,.2)
