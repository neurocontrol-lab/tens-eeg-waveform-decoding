"""Fit observed binary detectors on the exact feature sets saved by E015."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .core import CODES,WAVES,selected_matrix,fit_binary_targets,binary_metrics
from experiments.waveform_decoding.e015_position_repeatability_decoding.design import outer_splits
from experiments.response_characterization.e011_electrode_bin_repeatability.__main__ import sha,write_json


def load_source(root):
    config=json.loads((root/'config.json').read_text());status=json.loads((root/'status.json').read_text())
    if not status['complete'] or status['fit_failed']:raise ValueError(f'Incomplete E015 source: {root}')
    for path,digest in {**config['input_sha256'],**config['implementation_sha256']}.items():
        if sha(path)!=digest:raise ValueError(f'Frozen E015 source/input changed: {path}')
    return config,[r for r in json.loads((root/'models.json').read_text()) if r['status']=='observed_complete']


def run(args):
    if args.output.exists():raise ValueError('Use a fresh output directory')
    source_roots=dict(exploratory=args.exploratory,strict=args.strict)
    sources={name:load_source(root) for name,root in source_roots.items()}
    contract=pd.read_csv(args.features/'feature_contract.csv').to_dict('records')
    inputs=[args.features/'config.json',args.features/'status.json',args.features/'feature_contract.csv']
    subjects=sorted({r['subject'] for _,models in sources.values() for r in models})
    for s in subjects:inputs += [args.features/f'sub-{s}'/n for n in ['features.npy','events.csv']]
    for name,root in source_roots.items():
        inputs += [root/n for n in ['config.json','status.json','models.json']]
        for m in sources[name][1]:
            folder=root/f'sub-{m["subject"]}'/f'position-{m["position"]}'
            inputs += [folder/'predictions.csv']+list(folder.glob('fold-*.json'))+list(folder.glob('fold-*.npz'))
    implementations=list(Path(__file__).parent.glob('*.py'))+[
        Path('experiments/waveform_decoding/e015_position_repeatability_decoding/design.py'),
        Path('experiments/waveform_decoding/e015_position_repeatability_decoding/features.py')]
    before={str(p):sha(p) for p in inputs+implementations}
    args.output.mkdir(parents=True);started=time.perf_counter();models_out=[];fold_audits=[];seen_cases=0
    write_json(args.output/'config.json',dict(experiment_id='E016',stage='observed_binary_decoding',targets=CODES.tolist(),
        subjects=subjects,selector='E015 repeatability-selected features, frozen within each original training fold',
        n_features=100,new_feature_selection=False,new_spatial_fitting=False,binary_priors=[.5,.5],threshold=.5,
        test_labels_used_for_fitting=False,permutations=0,inference_complete=False,
        evaluation='Original held-out comparisons/pairs/temporal guard; metrics pooled over out-of-fold predictions',
        source_runs={k:str(v) for k,v in source_roots.items()},
        input_sha256={str(p):before[str(p)] for p in inputs},implementation_sha256={str(p):before[str(p)] for p in implementations},
        dependencies={n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','scikit-learn']}))
    for mode,root in source_roots.items():
        config,source_models=sources[mode]
        for source_model in source_models:
            s=source_model['subject'];position=source_model['position'];source_folder=root/f'sub-{s}'/f'position-{position}'
            meta=pd.read_csv(args.features/f'sub-{s}/events.csv').reset_index(drop=True)
            sensor=np.load(args.features/f'sub-{s}/features.npy',mmap_mode='r')
            target=np.flatnonzero(meta.position.to_numpy()==position);local={int(v):j for j,v in enumerate(target)}
            sensor_pos=np.asarray(sensor[target]);source_prediction=pd.read_csv(source_folder/'predictions.csv')
            if not np.array_equal(source_prediction.event_index.to_numpy(),meta.iloc[target].event_index.to_numpy()):raise ValueError('Source held-out row identity differs')
            if not np.array_equal(source_prediction.trigger.to_numpy(),meta.iloc[target].trigger.to_numpy()):raise ValueError('Source labels differ')
            prediction=meta.iloc[target][['subject','event_index','pair_id','group','position','trigger']].copy().reset_index(drop=True)
            prediction['fold']=-1
            for code in CODES:prediction[f'p_{code}']=np.nan;prediction[f'pred_{code}']=-1
            dest=args.output/mode/f'sub-{s}'/f'position-{position}';dest.mkdir(parents=True)
            per_target={int(c):[] for c in CODES}
            for fold in outer_splits(meta,config['guard']['minimum_trigger_separation_s']):
                f=fold['fold'];tr=fold['train'][meta.iloc[fold['train']].position.to_numpy()==position]
                te=fold['test'][meta.iloc[fold['test']].position.to_numpy()==position]
                train=np.array([local[int(i)] for i in tr]);test=np.array([local[int(i)] for i in te])
                expected=source_prediction[source_prediction.fold==f].event_index.to_numpy()
                if not np.array_equal(expected,meta.iloc[te].event_index.to_numpy()):raise ValueError('Outer folds differ from E015')
                detail=json.loads((source_folder/f'fold-{f}.json').read_text());chosen_contract=detail['models']['repeatability']['selected_features']
                with np.load(source_folder/f'fold-{f}.npz') as z:
                    chosen=z['repeatability_selected'];filters=z['spatial_filters']
                    old_mean=z['repeatability_scaler_mean'];old_scale=z['repeatability_scaler_scale']
                if len(chosen)!=100:raise ValueError('The parent fold did not select exactly 100 features')
                classes=np.unique(meta.iloc[tr].trigger)
                matrix=selected_matrix(sensor_pos,contract,filters,classes,chosen,chosen_contract)
                scaler,fits=fit_binary_targets(matrix[train],matrix[test],meta.iloc[tr].trigger.to_numpy())
                np.testing.assert_allclose(scaler.mean_,old_mean,rtol=1e-9,atol=1e-8)
                np.testing.assert_allclose(scaler.scale_,old_scale,rtol=1e-9,atol=1e-8)
                prediction.loc[test,'fold']=f
                fold_detail=dict(mode=mode,subject=s,position=position,fold=f,n_features=100,
                    source_fold=str(source_folder/f'fold-{f}.json'),source_scaling_reproduced=True,
                    training_event_indices=meta.iloc[tr].event_index.tolist(),test_event_indices=meta.iloc[te].event_index.tolist(),
                    source_flags=detail.get('flags',[]),detectors={})
                arrays=dict(scaler_mean=scaler.mean_,scaler_scale=scaler.scale_,source_selected=chosen,
                    coef=np.full((5,100),np.nan),intercept=np.full(5,np.nan),fitted=np.zeros(5,bool))
                for ci,code in enumerate(CODES):
                    c=int(code);v=fits[c];prediction.loc[test,f'p_{c}']=v['probability'];prediction.loc[test,f'pred_{c}']=v['predicted']
                    flags=list(v['flags']);missing_rest=sorted(set(CODES.tolist())-set(classes.tolist())-{c})
                    if missing_rest:flags.append('rest_waveform_absent_from_training')
                    row=dict(mode=mode,subject=s,position=position,fold=f,target=c,waveform=WAVES[c],
                        source_unreliable=mode=='exploratory',kind=v['kind'],training_positive=v['training_positive'],
                        training_negative=v['training_negative'],missing_rest_waveforms=missing_rest,flags=flags,warnings=v['warnings'],
                        test_metrics=binary_metrics((meta.iloc[te].trigger.to_numpy()==c).astype(int),v['probability'],v['predicted']))
                    fold_detail['detectors'][str(c)]=row;per_target[c].append(row);fold_audits.append(row)
                    if v['estimator'] is not None:
                        arrays['coef'][ci]=v['estimator'].coef_[0];arrays['intercept'][ci]=v['estimator'].intercept_[0];arrays['fitted'][ci]=True
                write_json(dest/f'fold-{f}.json',fold_detail);np.savez_compressed(dest/f'fold-{f}.npz',**arrays)
            if (prediction.fold<0).any():raise ValueError('Missing held-out predictions')
            prediction.to_csv(dest/'predictions.csv',index=False)
            case_scores=[]
            for code in CODES:
                c=int(code);truth=(prediction.trigger.to_numpy()==c).astype(int)
                probability=prediction[f'p_{c}'].to_numpy();pred=prediction[f'pred_{c}'].to_numpy()
                metrics=binary_metrics(truth,probability,pred)
                old_binary=(source_prediction.repeatability_predicted.to_numpy()==c).astype(int)
                old=binary_metrics(truth,old_binary.astype(float),old_binary)
                extra={}
                if f'repeatability_prob_{c}' in source_prediction:
                    extra['source_multiclass_probability_auc']=binary_metrics(truth,source_prediction[f'repeatability_prob_{c}'].to_numpy())['roc_auc']
                rows=per_target[c]
                record=dict(mode=mode,subject=s,position=position,target=c,waveform=WAVES[c],
                    source_unreliable=mode=='exploratory',unreliable=mode=='exploratory' or any(r['kind']!='shrinkage_lda' for r in rows),
                    binary_metrics=metrics,source_argmax_ovr_balanced_accuracy=old['balanced_accuracy'],
                    minimum_training_positives=min(r['training_positive'] for r in rows),
                    untrainable_folds=sum(r['kind']!='shrinkage_lda' for r in rows),
                    single_class_test_folds=sum(r['test_metrics']['roc_auc'] is None for r in rows),
                    flags=sorted({flag for r in rows for flag in r['flags']}),**extra)
                models_out.append(record);case_scores.append(record)
            write_json(dest/'scores.json',case_scores);seen_cases+=1
            write_json(args.output/'models.json',models_out)
            print(f'Binary detectors complete: {mode} subject {s} P{position}; five targets ({seen_cases} parent cases)',flush=True)
    write_json(args.output/'fold_audits.json',fold_audits)
    changed=[p for p,h in before.items() if sha(p)!=h]
    write_json(args.output/'status.json',dict(complete=not changed,primary_exploratory_detectors=sum(r['mode']=='exploratory' for r in models_out),
        strict_reference_detectors=sum(r['mode']=='strict' for r in models_out),detectors=len(models_out),
        held_out_detector_folds=len(fold_audits),lda_fits=sum(r['kind']=='shrinkage_lda' for r in fold_audits),
        constant_fallback_folds=sum(r['kind']!='shrinkage_lda' for r in fold_audits),new_feature_selection=False,
        new_spatial_fitting=False,inference_complete=False,permutations=0,changed_sources_or_inputs=changed,
        seconds=time.perf_counter()-started))
    if changed:raise ValueError('Frozen binary experiment inputs/source changed')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features',type=Path,default=Path('output/e015_feature_cache_20261102'))
    p.add_argument('--exploratory',type=Path,default=Path('output/e015_exploratory_observed_20261102'))
    p.add_argument('--strict',type=Path,default=Path('output/e015_strict_observed_20261102'))
    p.add_argument('--output',type=Path,default=Path('output/e016_binary_observed_20261102'))
    a=p.parse_args()
    with threadpool_limits(limits=2):run(a)
