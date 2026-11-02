"""Prepare label-free features or audit strict position-specific decoding support."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .design import filter_guard,outer_splits,position_operator,calibration_and_halves,CODES
from .features import sensor_features
from experiments.response_characterization.e011_electrode_bin_repeatability.__main__ import sha,write_json


def verify_cache(root):
    config=json.loads((root/'config.json').read_text());status=json.loads((root/'status.json').read_text())
    if not status['complete'] or status['errors']:raise ValueError('Require completed E014 epoch cache')
    for path,digest in {**config['input_sha256'],**config['implementation_sha256']}.items():
        if sha(path)!=digest:raise ValueError(f'Frozen source changed: {path}')
    return status['completed']


def audit(args):
    subjects=verify_cache(args.cache)
    if args.output.exists():raise ValueError('Use a fresh audit output')
    args.output.mkdir(parents=True);guard=filter_guard();rows=[];started=time.perf_counter()
    write_json(args.output/'config.json',dict(experiment_id='E015',stage='metadata_feasibility',
        subjects=subjects,guard=guard,scoring_splits_checked=args.audit_repeats,classifier_fitted=False,
        selection_fitted=False,calibration_fitted=False))
    for subject in subjects:
        meta=pd.read_csv(args.cache/f'sub-{subject}/events.csv').reset_index(drop=True)
        for fold in outer_splits(meta,guard['minimum_trigger_separation_s']):
            for pos in (1,2):
                take=fold['train'][meta.iloc[fold['train']].position.to_numpy()==pos]
                small=meta.iloc[take].reset_index(drop=True)
                counts=small.trigger.value_counts().reindex(CODES,fill_value=0)
                row=dict(subject=subject,position=pos,fold=fold['fold'],test_group=fold['test_groups'][0],
                    n_train=len(small),minimum_class_events=int(counts.min()),purged_pairs=len(fold['purged_pairs']),
                    eligible=False,reason='',calibration_per_class=None)
                try:
                    _,d=position_operator(small);row.update(design_rank=d['rank'],design_columns=d['columns'])
                    cal,operators,checks,detail=calibration_and_halves(small,20261102+1000*subject+100*pos+fold['fold'],
                        repeats=args.audit_repeats,calibration_attempts=5,split_attempts=50)
                    row.update(eligible=True,calibration_per_class=detail['calibration_per_class'],ranking_events=detail['ranking_events'])
                except ValueError as e:row['reason']=str(e)
                rows.append(row)
        print(f'Metadata support audited: subject {subject}',flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(args.output/'fold_support.csv',index=False)
    models=frame.groupby(['subject','position']).agg(eligible=('eligible','all'),eligible_folds=('eligible','sum'),
        total_folds=('fold','count'),minimum_training_epochs=('n_train','min')).reset_index()
    models.to_csv(args.output/'model_support.csv',index=False)
    summary=dict(complete=True,metadata_audit_only=True,classifier_fitted=False,calibration_fitted=False,
        eligible_models=int(models.eligible.sum()),models=len(models),eligible_folds=int(frame.eligible.sum()),
        folds=len(frame),scoring_splits_checked=args.audit_repeats,guard=guard,seconds=time.perf_counter()-started)
    write_json(args.output/'status.json',summary);print(json.dumps(summary,indent=2),flush=True)


def prepare(args):
    subjects=verify_cache(args.cache)
    if args.output.exists():raise ValueError('Use a fresh feature cache')
    args.output.mkdir(parents=True);inputs={};started=time.perf_counter();contract=None
    for subject in subjects:
        src=args.cache/f'sub-{subject}';dest=args.output/f'sub-{subject}';dest.mkdir()
        for name in ['epochs_uv.npy','layout.npz','events.csv']:inputs[str(src/name)]=sha(src/name)
        epochs=np.load(src/'epochs_uv.npy',mmap_mode='r')
        with np.load(src/'layout.npz') as layout:times=layout['time_s'];names=layout['channels'].tolist()
        matrix,rows=sensor_features(epochs,times,names)
        if matrix.shape[1]!=24614:raise ValueError('Wrong label-free feature count')
        if contract is None:contract=rows
        elif contract!=rows:raise ValueError('Participant feature contracts differ')
        np.save(dest/'features.npy',matrix)
        pd.read_csv(src/'events.csv').to_csv(dest/'events.csv',index=False)
        print(f'Label-free feature cache: subject {subject}, {matrix.shape}',flush=True)
    for path,digest in inputs.items():
        if sha(path)!=digest:raise ValueError('Epoch cache changed during feature extraction')
    pd.DataFrame(contract).to_csv(args.output/'feature_contract.csv',index=False)
    sources=[Path(__file__),Path(__file__).with_name('features.py'),
        Path('experiments/response_characterization/e011_electrode_bin_repeatability/core.py'),
        Path('experiments/response_characterization/e014_all_electrode_repeatability/features.py'),
        Path('experiments/waveform_decoding/e005_spectral_decoding/core.py')]
    write_json(args.output/'config.json',dict(experiment_id='E015',stage='label_free_feature_cache',subjects=subjects,
        input_sha256=inputs,implementation_sha256={str(p):sha(p) for p in sources},
        label_free_features=24614,fold_fitted_xdawn_features=750,full_candidate_count=25364))
    write_json(args.output/'status.json',dict(complete=True,errors={},completed=subjects,
        classifier_fitted=False,xdawn_fitted=False,seconds=time.perf_counter()-started))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['audit','prepare'])
    p.add_argument('--cache',type=Path,default=Path('output/e014_all_electrode_cache_20261030'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--audit-repeats',type=int,default=3)
    a=p.parse_args()
    with threadpool_limits(limits=2):(audit if a.stage=='audit' else prepare)(a)
