"""Explicitly unreliable fallback: train-only supervised basis and observed means.

Outer held-out comparisons/pairs and the temporal embargo are unchanged. The
supervised xDAWN basis sees both internal ranking halves, so their correlation
is an in-sample selection heuristic, not independently validated repeatability.
Observed comparison-balanced class means replace unidentified adjusted effects.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import time
import warnings
import numpy as np
import pandas as pd
from mne.decoding import XdawnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import f_classif
from threadpoolctl import threadpool_limits
from .design import filter_guard,outer_splits,CODES
from .features import xdawn_features
from .scoring import select_columns
from experiments.waveform_decoding.e001_voltage_baseline.evaluate import scores
from experiments.response_characterization.e011_electrode_bin_repeatability.__main__ import sha,write_json


def spatial_fit(epochs_uv,labels):
    classes=np.unique(labels)
    if len(classes)<2:raise ValueError('Fewer than two training classes')
    spatial=XdawnTransformer(n_components=2,reg=.1,rank='full')
    spatial.fit(epochs_uv*1e-6,labels)
    filters=spatial.transform(np.eye(epochs_uv.shape[1])[None])[0]
    signs=np.sign(filters[np.arange(len(filters)),abs(filters).argmax(axis=1)])
    filters*=np.where(signs==0,1,signs)[:,None]
    rms=np.sqrt(np.mean((filters@epochs_uv)**2,axis=(0,2)))
    if not np.isfinite(rms).all() or (rms<=0).any():raise ValueError('Degenerate training spatial filters')
    return filters/rms[:,None],classes


def observed_operators(meta,seed,repeats=100):
    """Correlate only classes represented independently in both training halves."""
    y=meta.trigger.to_numpy(int);classes=np.array([c for c in CODES if np.sum(y==c)>=2])
    if len(classes)<3:raise ValueError('Fewer than three classes support A/B ranking')
    rng=np.random.default_rng(seed);contexts=meta.comparison.to_numpy();operators=[]
    indices=[np.flatnonzero(y==c) for c in classes]
    for _ in range(repeats):
        a=[];b=[]
        for ix in indices:
            order=rng.permutation(ix);cut=len(ix)//2
            if len(ix)%2 and rng.integers(2):cut+=1
            a.extend(order[:cut]);b.extend(order[cut:])
        halves=[]
        for selected in [np.asarray(a,int),np.asarray(b,int)]:
            weights=np.zeros((len(classes),len(meta)))
            for ci,c in enumerate(classes):
                within=selected[y[selected]==c];cells=np.unique(contexts[within])
                for context in cells:
                    cell=within[contexts[within]==context];weights[ci,cell]=1/(len(cells)*len(cell))
            halves.append(weights)
        operators.append(halves)
    return np.asarray(operators),classes,np.flatnonzero(np.isin(y,classes))


def ranking_scores(x,operators,ranking_indices,batch=5):
    scale=np.sqrt(np.mean(x[ranking_indices]**2,axis=0));tol=(100*np.finfo(float).eps*np.maximum(scale,np.finfo(float).tiny))**2
    correlations=[];n_classes=operators.shape[2]
    for start in range(0,len(operators),batch):
        op=operators[start:start+batch]
        profile=(op.reshape(-1,len(x))@x).reshape(len(op),2,n_classes,x.shape[1])
        a,b=profile[:,0],profile[:,1];a=a-a.mean(axis=1,keepdims=True);b=b-b.mean(axis=1,keepdims=True)
        aa=np.mean(a*a,axis=1);bb=np.mean(b*b,axis=1)
        correlations.append(np.divide(np.mean(a*b,axis=1),np.sqrt(aa*bb),out=np.full(aa.shape,np.nan),where=(aa>tol)&(bb>tol)))
    r=np.concatenate(correlations)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning);median=np.nanmedian(r,axis=0)
    count=np.isfinite(r).sum(axis=0)
    return median,np.where(count==len(operators),np.maximum(median,0)**2,np.nan),count


def fit_fold(sensor_train,sensor_test,epochs_train,epochs_test,times,meta_train,contract,seed,k=100,repeats=100):
    if meta_train.position.nunique()!=1:raise ValueError('Do not pool presentation positions')
    y=meta_train.trigger.to_numpy(int);mask=(times>=.05)&(times<1.05)
    filters,classes=spatial_fit(epochs_train[...,mask],y)
    names=[f'{code}:component-{i+1}' for code in classes for i in range(2)]
    z_train,zc=xdawn_features(epochs_train,times,filters,names)
    z_test,_=xdawn_features(epochs_test,times,filters,names)
    x=np.column_stack([sensor_train,z_train]);test=np.column_stack([sensor_test,z_test]);contract=contract+zc
    if x.shape[1]!=len(contract):raise ValueError('Feature axis mismatch')
    operators,rank_classes,rank_indices=observed_operators(meta_train,seed,repeats)
    median,r_score,valid=ranking_scores(x,operators,rank_indices)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning);f_score,_=f_classif(x[rank_indices],y[rank_indices])
    output={}
    for name,score in [('repeatability',r_score),('anova',f_score)]:
        chosen=select_columns(score,k)
        estimator=make_pipeline(StandardScaler(),LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto',
            priors=np.full(len(classes),1/len(classes)))).fit(x[:,chosen],y)
        probability=np.zeros((len(test),5))
        probability[:,estimator.classes_-239]=estimator.predict_proba(test[:,chosen])
        output[name]=dict(predicted=estimator.predict(test[:,chosen]),probabilities=probability,estimator=estimator,
            selected=chosen,selected_contract=[contract[i] for i in chosen],selected_scores=score[chosen],n_selected=len(chosen))
    flags=['unreliable_exploratory','no_independent_xdawn_calibration','observed_context_balanced_means_not_adjusted_contrasts']
    if len(classes)<5:flags.append('missing_training_class')
    if len(rank_classes)<5:flags.append('fewer_than_five_ranking_classes')
    return dict(models=output,spatial_filters=filters,median_r=median,valid_r_splits=valid,
        training_classes=classes.tolist(),ranking_classes=rank_classes.tolist(),flags=flags,
        missing_training_classes=sorted(set(CODES.tolist())-set(classes.tolist())),candidate_count=x.shape[1])


def permute_pair_labels(meta,rng):
    """Conditional diagnostic null; preserves ordered-pair counts per comparison."""
    y=meta.trigger.to_numpy(int).copy()
    for _,group in meta.groupby('group',sort=False):
        pairs=[g.sort_values('position').index.to_numpy() for _,g in group.groupby('pair_id',sort=False)]
        if any(len(ix)!=2 for ix in pairs):raise ValueError('Incomplete pair in permutation')
        ix=np.stack(pairs);values=y[ix].copy();y[ix]=values[rng.permutation(len(ix))]
    return y


def run(args):
    import mne
    mne.set_log_level('ERROR')
    if args.output.exists():raise ValueError('Use a fresh exploratory output')
    cache=json.loads((args.features/'config.json').read_text());status=json.loads((args.features/'status.json').read_text())
    if not status['complete'] or status['errors']:raise ValueError('Feature cache incomplete')
    for path,digest in {**cache['input_sha256'],**cache['implementation_sha256']}.items():
        if sha(path)!=digest:raise ValueError(f'Cache source changed: {path}')
    subjects=args.subjects or cache['subjects'];contract=pd.read_csv(args.features/'feature_contract.csv').to_dict('records')
    if not set(subjects)<=set(cache['subjects']):raise ValueError('Unknown participants')
    inputs=[args.features/'config.json',args.features/'status.json',args.features/'feature_contract.csv']
    for s in subjects:
        inputs += [args.features/f'sub-{s}'/n for n in ['features.npy','events.csv']]
        inputs += [args.epochs/f'sub-{s}'/n for n in ['epochs_uv.npy','layout.npz']]
    sources=[Path(__file__),Path(__file__).with_name('design.py'),Path(__file__).with_name('features.py'),
        Path(__file__).with_name('scoring.py'),Path('experiments/waveform_decoding/e001_voltage_baseline/evaluate.py')]
    before={str(p):sha(p) for p in inputs+sources};args.output.mkdir(parents=True);guard=filter_guard();summaries=[];fold_audits=[]
    write_json(args.output/'config.json',dict(experiment_id='E015',stage='exploratory_observed',subjects=subjects,
        unreliable=True,features=args.k,ranking_repeats=args.repeats,guard=guard,
        differences_from_strict=['xDAWN fitted on all outer training events','Observed comparison-balanced class profiles, without regression adjustment',
            'A/B ranking uses classes with at least two training events','Missing training classes remain impossible to predict and are scored in five-class evaluation'],
        test_labels_used_for_fitting=False,permutations=0,
        input_sha256={str(p):before[str(p)] for p in inputs},implementation_sha256={str(p):before[str(p)] for p in sources},
        dependencies={n:importlib.metadata.version(n) for n in ['numpy','scipy','pandas','scikit-learn','mne']}))
    started=time.perf_counter()
    for subject in subjects:
        meta=pd.read_csv(args.features/f'sub-{subject}/events.csv').reset_index(drop=True)
        x=np.load(args.features/f'sub-{subject}/features.npy',mmap_mode='r');epochs=np.load(args.epochs/f'sub-{subject}/epochs_uv.npy',mmap_mode='r')
        with np.load(args.epochs/f'sub-{subject}/layout.npz') as f:times=f['time_s']
        folds=outer_splits(meta,guard['minimum_trigger_separation_s'])
        for pos in [1,2]:
            target=np.flatnonzero(meta.position.to_numpy()==pos);local={int(v):i for i,v in enumerate(target)}
            predictions=meta.iloc[target][['subject','event_index','pair_id','group','position','trigger']].copy().reset_index(drop=True)
            predictions['fold']=-1
            for name in ['repeatability','anova']:predictions[name+'_predicted']=-1
            counts={'repeatability':[],'anova':[]};failures=[];missing=0
            dest=args.output/f'sub-{subject}'/f'position-{pos}';dest.mkdir(parents=True)
            for fold in folds:
                tr=fold['train'][meta.iloc[fold['train']].position.to_numpy()==pos];te=fold['test'][meta.iloc[fold['test']].position.to_numpy()==pos]
                small=meta.iloc[tr].reset_index(drop=True);seed=20261102+1000*subject+100*pos+fold['fold']
                row=dict(subject=subject,position=pos,fold=fold['fold'],n_train=len(tr),n_test=len(te),unreliable=True)
                try:
                    result=fit_fold(np.asarray(x[tr]),np.asarray(x[te]),np.asarray(epochs[tr]),np.asarray(epochs[te]),times,small,contract,seed,args.k,args.repeats)
                    row.update(flags=result['flags'],training_classes=result['training_classes'],ranking_classes=result['ranking_classes'],
                        missing_training_classes=result['missing_training_classes'],candidate_count=result['candidate_count'])
                    missing+=int(bool(result['missing_training_classes']));ix=np.array([local[int(v)] for v in te]);predictions.loc[ix,'fold']=fold['fold']
                    detail={**row,'models':{}};arrays=dict(spatial_filters=result['spatial_filters'],median_r=result['median_r'])
                    for name,v in result['models'].items():
                        predictions.loc[ix,name+'_predicted']=v['predicted'];counts[name].append(v['n_selected'])
                        for c in range(5):predictions.loc[ix,f'{name}_prob_{239+c}']=v['probabilities'][:,c]
                        detail['models'][name]=dict(n_selected=v['n_selected'],requested_features=args.k,
                            selected_features=v['selected_contract'],selected_scores=v['selected_scores'].tolist())
                        estimator=v['estimator'];arrays[name+'_selected']=v['selected'];arrays[name+'_scaler_mean']=estimator[0].mean_
                        arrays[name+'_scaler_scale']=estimator[0].scale_;arrays[name+'_coef']=estimator[1].coef_;arrays[name+'_intercept']=estimator[1].intercept_
                    write_json(dest/f'fold-{fold["fold"]}.json',detail);np.savez_compressed(dest/f'fold-{fold["fold"]}.npz',**arrays)
                    print(f'Exploratory subject {subject} P{pos}, fold {fold["fold"]}: complete',flush=True)
                except Exception as e:
                    row['error']=f'{type(e).__name__}: {e}';failures.append(row['error']);print(row['error'],flush=True)
                fold_audits.append(row)
            report=dict(subject=subject,position=pos,unreliable=True,status='fit_failed' if failures else 'observed_complete',
                n_events=len(target),failed_folds=failures,folds_with_missing_training_class=missing,models={})
            if not failures:
                if (predictions['fold']<0).any():raise ValueError('Incomplete held-out predictions')
                for name in counts:
                    if (predictions[name+'_predicted']<0).any():raise ValueError('Incomplete model predictions')
                    report['models'][name]=scores(predictions.trigger.to_numpy(),predictions[name+'_predicted'].to_numpy())
                    report['models'][name]['selected_counts']=counts[name]
                predictions.to_csv(dest/'predictions.csv',index=False)
            write_json(dest/'scores.json',report);summaries.append(report);write_json(args.output/'models.json',summaries)
    write_json(args.output/'fold_audits.json',fold_audits)
    changed=[p for p,h in before.items() if sha(p)!=h];failed=sum(r['status']=='fit_failed' for r in summaries)
    write_json(args.output/'status.json',dict(complete=not changed and not failed,unreliable=True,
        processed_models=len(summaries),observed_complete=len(summaries)-failed,fit_failed=failed,
        changed_sources_or_inputs=changed,inference_complete=False,permutations=0,seconds=time.perf_counter()-started))
    if changed:raise ValueError('Exploratory input/source changed during run')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features',type=Path,default=Path('output/e015_feature_cache_20261102'))
    p.add_argument('--epochs',type=Path,default=Path('output/e014_all_electrode_cache_20261030'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--subjects',type=int,nargs='+')
    p.add_argument('--k',type=int,default=100);p.add_argument('--repeats',type=int,default=100)
    a=p.parse_args()
    with threadpool_limits(limits=2):run(a)
