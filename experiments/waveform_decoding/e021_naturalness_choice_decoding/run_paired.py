"""Fit true-pair, within-participant naturalness decoders from verified EEG caches."""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.response_characterization.e017_eeg_charge_relation.extract import digest
from . import paired_evaluate, position_data, position_evaluate


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,default=Path('output/wd008_position_feature_cache_20261124'))
    parser.add_argument('--archive',type=Path,default=Path('output/wd008_position_naturalness_20261124'))
    parser.add_argument('--output',type=Path,default=Path('output/wd008_paired_naturalness_20261124'))
    args=parser.parse_args()
    if args.output.exists():parser.error('Use a fresh paired-fit output directory')
    args.output.mkdir(parents=True)
    write(args.output/'status.json',dict(complete=False,stage='loading_verified_pair_features'))
    inputs={};tables=[];matrices=[];channels=None
    for subject in range(3,13):
        folder=args.cache/f'sub-{subject}';audit=json.loads((folder/'complete.json').read_text())
        if audit['signature']['implementation_sha256']!=digest(Path(position_data.__file__)):
            raise ValueError('Feature extraction code differs from cached provenance')
        names=audit['channels']
        if channels is None:channels=names
        if names!=channels or len(names)!=62:raise ValueError('Inconsistent 62-channel montage')
        for name,expected in audit['outputs_sha256'].items():
            path=folder/name
            if digest(path)!=expected:raise ValueError('Cached feature input changed')
            inputs[str(path)]=expected
        inputs[str(folder/'complete.json')]=digest(folder/'complete.json')
        tables.append(pd.read_csv(folder/'events.csv'))
        with np.load(folder/'features.npz') as data:matrices.append({k:data[k].copy() for k in data.files})
    events=pd.concat(tables,ignore_index=True)
    features={k:np.concatenate([m[k] for m in matrices]) for k in ('pre','post')}
    pairs,features=paired_evaluate.build_pairs(events,features)
    for name in ('summary.json','predictions.csv','fold_audit.json','config.json'):
        inputs[str(args.archive/name)]=digest(args.archive/name)
    config=dict(experiment='WD-008',date='2026-11-24',design='Actual paired presentations, separate within-participant models',
        target='0: first stimulus chosen; 1: second stimulus chosen, from T4 Naturalness Value',
        validation='Within each participant, hold out all blocks of an entire waveform comparison; both responses stay together',
        representations=paired_evaluate.REPRESENTATIONS,primary_representation='concatenated',
        windows_seconds=dict(pre_pair=[-1.05,-.05],first_post=[.05,1.05],second_post=[.05,1.05]),
        baseline_seconds_relative_to_first=[-1.25,-1.05],channels=channels,bin_widths_ms=[50,100],
        preprocessing='Instantaneous average reference and common pre-pair baseline; no temporal filter or ICA; original complete-pair QC and all-channel finite/timestamp checks',
        features='Scale features, fit separate pre/post PCA retaining 90% variance, standardize components; training fold only',
        classifier='L2 logistic regression C=1, no class weighting or tuning',
        metadata='First/second waveform indicators, chronological pair index and within-block index; no charge or choice/stopping inputs',
        controls='Training-proportion and metadata baselines; pre-pair EEG; incremental paired post EEG; optional original-last-trial exclusion with refits',
        inference='Descriptive participant bootstrap intervals; sign tests of BA/log-loss changes, Holm across both representations, resolutions and trial analyses',
        timing='Second response may include carryover; first pre window is before either pulse. No behavioral-response timestamps or verified physical delivery timing.',
        sources_sha256={str(p):digest(p) for p in (Path(__file__),Path(paired_evaluate.__file__),Path(position_data.__file__),Path(position_evaluate.__file__))},
        input_sha256=inputs)
    write(args.output/'config.json',config);pairs.to_csv(args.output/'pair_manifest.csv',index=False)
    write(args.output/'status.json',dict(complete=False,stage='fitting',pairs=len(pairs)))
    with threadpool_limits(limits=1):
        predictions,folds,unsupported=paired_evaluate.predict(pairs,features)
        summary=paired_evaluate.summarize(predictions)
    if not unsupported:
        expected=4*(len(pairs)+int((~pairs.terminal_vote).sum()))
        if len(predictions)!=expected:raise ValueError('Missing paired prediction')
    predictions.to_csv(args.output/'predictions.csv',index=False)
    write(args.output/'fold_audit.json',folds);write(args.output/'unsupported.json',unsupported)
    summary.update(experiment='WD-008',date=config['date'],design=config['design'],config=config,
                   pairs=len(pairs),epochs=2*len(pairs),participants=int(pairs.subject.nunique()),
                   nonterminal_pairs=int((~pairs.terminal_vote).sum()),models=paired_evaluate.MODELS,
                   representations=paired_evaluate.REPRESENTATIONS,folds=len(folds),unsupported=unsupported)
    write(args.output/'summary.json',summary)
    for path,expected in {**inputs,**config['sources_sha256']}.items():
        if digest(Path(path))!=expected:raise ValueError('Input or source changed during paired fit')
    write(args.output/'status.json',dict(complete=True,stage='complete',pairs=len(pairs),folds=len(folds),unsupported=len(unsupported)))
    print('Completed paired naturalness:',args.output.resolve(),flush=True)


if __name__=='__main__':main()
