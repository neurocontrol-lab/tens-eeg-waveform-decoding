"""Observed strict-design evaluation; ineligible models receive no accuracy."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import time
import traceback
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .design import filter_guard,outer_splits,position_operator,calibration_and_halves
from .scoring import fit_fold
from experiments.waveform_decoding.e001_voltage_baseline.evaluate import scores
from experiments.response_characterization.e011_electrode_bin_repeatability.__main__ import sha,write_json


def run(args):
    if args.output.exists():raise ValueError('Use a fresh evaluation directory')
    cache_status=json.loads((args.features/'status.json').read_text())
    config=json.loads((args.features/'config.json').read_text())
    if not cache_status['complete'] or cache_status['errors']:raise ValueError('Feature cache incomplete')
    for path,digest in {**config['input_sha256'],**config['implementation_sha256']}.items():
        if sha(path)!=digest:raise ValueError(f'Feature input/source changed: {path}')
    subjects=args.subjects or config['subjects']
    if not set(subjects)<=set(config['subjects']):raise ValueError('Unknown participants')
    contract=pd.read_csv(args.features/'feature_contract.csv').to_dict('records')
    inputs=[args.features/'config.json',args.features/'status.json',args.features/'feature_contract.csv']
    for s in subjects:
        inputs += [args.features/f'sub-{s}'/n for n in ['features.npy','events.csv']]
        inputs += [args.epochs/f'sub-{s}'/n for n in ['epochs_uv.npy','layout.npz']]
    sources=list(Path(__file__).parent.glob('*.py'))+[
        Path('experiments/response_characterization/e014_all_electrode_repeatability/features.py'),
        Path('experiments/waveform_decoding/e001_voltage_baseline/evaluate.py')]
    before={str(p):sha(p) for p in inputs+sources}
    args.output.mkdir(parents=True);guard=filter_guard();summary=[];fold_rows=[];started=time.perf_counter()
    write_json(args.output/'config.json',dict(experiment_id='E015',stage='observed_decoding',
        policy='Strict independent calibration and identifiable adjusted halves; skip entire unsupported models',
        subjects=subjects,positions=[1,2],features=args.k,ranking_repeats=args.repeats,
        outer_validation='Leave one entire comparison out, with pair-wise temporal embargo',guard=guard,
        primary='Repeatability-selected shrinkage LDA',reference='ANOVA-selected matched shrinkage LDA',
        permutation_testing='Not run by this observed-stage command; no inferential p-values',
        input_sha256={str(p):before[str(p)] for p in inputs},implementation_sha256={str(p):before[str(p)] for p in sources},
        dependencies={n:importlib.metadata.version(n) for n in ['numpy','scipy','pandas','scikit-learn','mne']}))
    for subject in subjects:
        meta=pd.read_csv(args.features/f'sub-{subject}/events.csv').reset_index(drop=True)
        x=np.load(args.features/f'sub-{subject}/features.npy',mmap_mode='r')
        epochs=np.load(args.epochs/f'sub-{subject}/epochs_uv.npy',mmap_mode='r')
        with np.load(args.epochs/f'sub-{subject}/layout.npz') as z:times=z['time_s']
        folds=outer_splits(meta,guard['minimum_trigger_separation_s'])
        for pos in (1,2):
            target=meta.index[meta.position==pos].to_numpy();local={int(i):j for j,i in enumerate(target)}
            subsets=[];failures=[]
            for f in folds:
                tr=f['train'][meta.iloc[f['train']].position.to_numpy()==pos]
                te=f['test'][meta.iloc[f['test']].position.to_numpy()==pos]
                small=meta.iloc[tr].reset_index(drop=True);seed=20261102+1000*subject+100*pos+f['fold']
                row=dict(subject=subject,position=pos,fold=f['fold'],n_train=len(tr),n_test=len(te),
                    test_group=f['test_groups'][0],eligible=False,reason='')
                try:
                    position_operator(small)
                    calibration_and_halves(small,seed,repeats=args.repeats)
                    row['eligible']=True;subsets.append((f['fold'],tr,te,small,seed))
                except ValueError as e:row['reason']=str(e);failures.append(row)
                fold_rows.append(row)
            if failures:
                summary.append(dict(subject=subject,position=pos,status='ineligible',failed_folds=len(failures),
                    reason='Strict metadata support failed; no partial-fold accuracy reported'))
                print(f'Subject {subject} P{pos}: ineligible ({len(failures)} folds); no classifier fit',flush=True)
                continue
            dest=args.output/f'sub-{subject}'/f'position-{pos}';dest.mkdir(parents=True)
            prediction=meta.iloc[target][['subject','event_index','pair_id','group','position','trigger']].copy().reset_index(drop=True)
            prediction['fold']=-1
            for name in ['repeatability','anova']:prediction[name+'_predicted']=-1
            selected_counts={'repeatability':[],'anova':[]}
            try:
                for fold,tr,te,small,seed in subsets:
                    result=fit_fold(np.asarray(x[tr]),np.asarray(x[te]),np.asarray(epochs[tr]),np.asarray(epochs[te]),
                                    times,small,contract,seed,k=args.k,repeats=args.repeats)
                    ix=np.array([local[int(i)] for i in te]);prediction.loc[ix,'fold']=fold
                    details=dict(subject=subject,position=pos,fold=fold,calibration=result['allocation'],
                        calibration_event_indices=small.iloc[result['calibration_indices']].event_index.tolist(),
                        ranking_event_indices=small.iloc[result['ranking_indices']].event_index.tolist(),models={})
                    arrays=dict(spatial_filters=result['spatial_filters'],median_r=result['median_r'],
                                valid_r_splits=result['valid_r_splits'])
                    for name,value in result['models'].items():
                        prediction.loc[ix,name+'_predicted']=value['predicted'];selected_counts[name].append(value['n_selected'])
                        estimator=value['estimator']
                        details['models'][name]=dict(n_selected=value['n_selected'],requested_features=args.k,
                            selected_features=value['selected_contract'],selected_scores=value['selected_scores'].tolist())
                        arrays[name+'_selected']=value['selected'];arrays[name+'_scaler_mean']=estimator[0].mean_
                        arrays[name+'_scaler_scale']=estimator[0].scale_;arrays[name+'_coef']=estimator[1].coef_
                        arrays[name+'_intercept']=estimator[1].intercept_
                    np.savez_compressed(dest/f'fold-{fold}.npz',**arrays)
                    write_json(dest/f'fold-{fold}.json',details)
                    print(f'Subject {subject} P{pos}: held-out fold {fold} complete',flush=True)
                if (prediction['fold']<0).any():raise ValueError('Missing held-out prediction')
                report=dict(subject=subject,position=pos,status='observed_complete',n_events=len(target),models={})
                for name in selected_counts:
                    if (prediction[name+'_predicted']<0).any():raise ValueError('Missing model prediction')
                    report['models'][name]=scores(prediction.trigger.to_numpy(),prediction[name+'_predicted'].to_numpy())
                    report['models'][name]['selected_counts']=selected_counts[name]
                prediction.to_csv(dest/'predictions.csv',index=False);write_json(dest/'scores.json',report)
                summary.append(report)
            except Exception as e:
                summary.append(dict(subject=subject,position=pos,status='fit_failed',reason=f'{type(e).__name__}: {e}'))
                print(traceback.format_exc(),flush=True)
            write_json(args.output/'models.json',summary)
    pd.DataFrame(fold_rows).to_csv(args.output/'fold_support.csv',index=False)
    write_json(args.output/'models.json',summary)
    changed=[path for path,digest in before.items() if sha(path)!=digest]
    failed=sum(r['status']=='fit_failed' for r in summary)
    write_json(args.output/'status.json',dict(complete=not changed and not failed,processed_models=len(summary),
        observed_complete=sum(r['status']=='observed_complete' for r in summary),
        ineligible=sum(r['status']=='ineligible' for r in summary),fit_failed=failed,
        changed_sources_or_inputs=changed,inference_complete=False,permutations=0,
        note='Observed point estimates only; unsupported models are explicitly reported',seconds=time.perf_counter()-started))
    if changed:raise ValueError('Inputs/source changed during run')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features',type=Path,default=Path('output/e015_feature_cache_20261102'))
    p.add_argument('--epochs',type=Path,default=Path('output/e014_all_electrode_cache_20261030'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--subjects',type=int,nargs='+')
    p.add_argument('--k',type=int,default=100);p.add_argument('--repeats',type=int,default=100)
    a=p.parse_args()
    with threadpool_limits(limits=2):run(a)
