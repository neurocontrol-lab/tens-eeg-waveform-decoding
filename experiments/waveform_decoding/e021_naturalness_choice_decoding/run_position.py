"""Revised WD-008: participant/position models, full-montage PCA and block holdout."""

import argparse
import importlib.metadata
import json
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.response_characterization.e017_eeg_charge_relation.extract import digest
from . import position_data, position_evaluate


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=Path('output/task4_baseline'))
    parser.add_argument('--visual', type=Path, default=Path('output/task4_visual_inspection_20261030'))
    parser.add_argument('--data-dir', type=Path, default=Path('EEGData'))
    parser.add_argument('--master-csv', type=Path,
                        default=Path('EEG_Codes_Export/02_SourceLocalization_Task4/Master_Data_ToSub12 (1).csv'))
    parser.add_argument('--previous', type=Path, default=Path('output/wd008_naturalness_baselined_20261123'))
    parser.add_argument('--cache', type=Path, default=Path('output/wd008_position_feature_cache_20261124'))
    parser.add_argument('--output', type=Path, default=Path('output/wd008_position_naturalness_20261124'))
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output directory; the verified position cache can be reused.')
    args.output.mkdir(parents=True)
    write(args.output/'status.json', dict(complete=False, stage='extracting'))
    config = dict(
        experiment='WD-008', reference='E021', revision='within-participant and presentation', date='2026-11-24',
        paths={key:str(value) for key,value in vars(args).items()},
        target='1 if the current presentation was chosen as more natural, 0 if the other was chosen; T4 Naturalness Value',
        cohort='Same 622 eligible behavioral pairs as the archived full WD-008 cohort; both positions retained',
        validation='Separate subject x position models; all occurrences of a comparison are held out together; no across-participant fitting',
        windows_seconds=position_data.WINDOWS,
        pair_baseline_relative_to_first_trigger=position_data.BASELINE,
        preprocessing='62-channel instantaneous average reference, common pre-pair baseline, voltage bin means in microvolts; no temporal filter or ICA',
        eeg_features='All 62 electrodes, fixed 50/100 ms means; train-only StandardScaler, separate pre/post PCA (90% variance) and train-only component standardization',
        metadata='Five current-waveform and five opponent-waveform indicators plus chronological pair and within-block indices; no choice history, terminal flag, final block length or charge inputs',
        classifier='Fixed L2 logistic regression C=1, no class weighting; training choice proportion baseline',
        presentations='No response differences or combined first/second EEG; the target for the second position is the complement of the first',
        sensitivity='Remove each original terminal vote and refit using the remaining data; no relabeling of the new last trial',
        inference='Descriptive participant bootstrap intervals conditional on fitted predictions; exact participant sign tests of post/pre and incremental changes, Holm across both metrics, widths, positions and analyses; no individual above-chance permutation tests',
        timing='Second-position pre EEG may contain the first response. Response timestamps and physical current/trigger synchronization remain unavailable.',
        model_labels=position_evaluate.MODELS,
        dependencies={name: importlib.metadata.version(name) for name in ('numpy','pandas','scipy','scikit-learn','pyxdf')},
        source_sha256={str(path):digest(path) for path in (Path(__file__),Path(position_data.__file__),Path(position_evaluate.__file__))},
        archived_input_sha256={str(args.previous/name):digest(args.previous/name) for name in ('summary.json','pair_manifest.csv','predictions.csv')},
    )
    write(args.output/'config.json', config)
    events, features, audit = position_data.assemble(args)
    if events.duplicated(['subject','pair_id','position']).any():
        raise ValueError('Repeated presentation in the manifest')
    if not events.groupby('pair_id').target_chosen.sum().eq(1).all():
        raise ValueError('A behavioral pair must have exactly one chosen presentation')
    events.to_csv(args.output/'epoch_manifest.csv', index=False)
    write(args.output/'status.json', dict(complete=False, stage='fitting', epochs=len(events)))
    print(f'Fit participant/position models on {len(events)} epochs from {events.pair_id.nunique()} pairs', flush=True)
    with threadpool_limits(limits=1):
        predictions, folds, unsupported = position_evaluate.predict(events, features)
        summary = position_evaluate.summarize(predictions)
    for analysis in ('full','nonterminal'):
        expected = events if analysis=='full' else events.loc[~events.terminal_vote]
        observed = predictions.loc[predictions.analysis.eq(analysis)]
        if not unsupported and len(observed)!=2*len(expected):
            raise ValueError('Missing held-out prediction for an eligible epoch/resolution')
    predictions.to_csv(args.output/'predictions.csv', index=False)
    write(args.output/'fold_audit.json', folds)
    write(args.output/'unsupported.json', unsupported)
    summary.update(experiment='WD-008', date=config['date'], revision=config['revision'],
                   pairs=int(events.pair_id.nunique()), epochs=len(events), participants=int(events.subject.nunique()),
                   nonterminal_pairs=int(events.loc[~events.terminal_vote,'pair_id'].nunique()),
                   models=position_evaluate.MODELS, input_audit=audit, config=config,
                   unsupported=unsupported, folds=len(folds),
                   class_counts=events.groupby(['subject','position']).target_chosen.agg(['count','sum']).reset_index().to_dict('records'))
    write(args.output/'summary.json', summary)
    for path, expected in {**config['source_sha256'],**config['archived_input_sha256']}.items():
        if digest(Path(path)) != expected:
            raise ValueError('Analysis implementation or archived input changed during this run')
    write(args.output/'status.json',dict(complete=True,stage='complete',epochs=len(events),pairs=summary['pairs'],unsupported=len(unsupported)))
    print('Completed:', args.output.resolve(), flush=True)


if __name__ == '__main__':
    main()
