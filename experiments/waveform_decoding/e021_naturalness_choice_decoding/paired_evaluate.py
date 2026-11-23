"""True-pair naturalness models fitted within each participant."""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .evaluate import interval
from .position_evaluate import metadata_features, metrics as epoch_metrics
from experiments.waveform_decoding.e022_triangle_lindec_decoding.evaluate import (
    resolution, paired_sign_test, holm_adjust,
)

MODELS = dict(prior='Training choice proportion',metadata='Waveform identities and trial order',
              pre='Pre-pair EEG alone',post='Paired post EEG alone',
              metadata_pre='Metadata + pre-pair EEG',metadata_post='Metadata + paired post EEG',
              metadata_pre_post='Metadata + pre-pair + paired post EEG')
CONTRASTS = dict(post_vs_pre=('Paired post EEG versus pre-pair EEG','pre','post'),
                post_added=('Metadata + pre-pair + paired post EEG versus metadata + pre-pair EEG',
                            'metadata_pre','metadata_pre_post'))
REPRESENTATIONS = dict(concatenated='First and second responses concatenated',
                       difference='Second response minus first response')


def build_pairs(events, features):
    """Match actual paired presentations by ID, never by choice or waveform similarity."""
    events=events.reset_index(drop=True).assign(feature_row=np.arange(len(events)))
    if events.duplicated(['pair_id','position']).any():
        raise ValueError('Repeated presentation in a pair')
    first=events.loc[events.position.eq(1)].sort_values(['subject','pair_index']).copy()
    second=events.loc[events.position.eq(2)].set_index('pair_id').loc[first.pair_id].reset_index()
    if len(first)*2!=len(events) or set(first.pair_id)!=set(second.pair_id):
        raise ValueError('Every observation needs its actual two presentations')
    first=first.reset_index(drop=True)
    for field in ('subject','comparison','response_value','terminal_vote'):
        if not np.array_equal(first[field],second[field]):
            raise ValueError('Paired presentation metadata disagree: '+field)
    if not np.array_equal(first.trigger,second.opponent_trigger) or not np.array_equal(second.trigger,first.opponent_trigger):
        raise ValueError('Pair waveform identities disagree')
    table=first[['subject','pair_id','pair_index','block_index','comparison','within_block_index','terminal_vote']].copy()
    table['first_trigger']=first.trigger.to_numpy()
    table['second_trigger']=second.trigger.to_numpy()
    table['first_event_index']=first.event_index.to_numpy()
    table['second_event_index']=second.event_index.to_numpy()
    table['target_second']=first.response_value.eq(2).astype(int)
    if not first.response_value.isin([1,2]).all():
        raise ValueError('Missing naturalness choice')
    a,b=first.feature_row.to_numpy(),second.feature_row.to_numpy()
    return table,dict(pre=features['pre'][a],first_post=features['post'][a],second_post=features['post'][b])


def post_features(features, width, representation):
    first,second=resolution(features['first_post'],width),resolution(features['second_post'],width)
    if representation=='concatenated':return np.column_stack([first,second])
    if representation=='difference':return second-first
    raise ValueError('Unknown pair representation')


def predict(table, features):
    records,audits,unsupported=[],[],[]
    metadata=metadata_features(table.rename(columns={'first_trigger':'trigger','second_trigger':'opponent_trigger'}))
    for analysis,mask in [('full',np.ones(len(table),bool)),('nonterminal',~table.terminal_vote.to_numpy(bool))]:
        for representation in REPRESENTATIONS:
            for subject,cell in table.loc[mask].groupby('subject'):
                indices=cell.index.to_numpy();y=cell.target_second.to_numpy();groups=cell.comparison.to_numpy()
                for width in (50,100):
                    signals=dict(pre=resolution(features['pre'][indices],width),
                                 post=post_features({k:v[indices] for k,v in features.items()},width,representation))
                    meta=metadata[indices]
                    for comparison in sorted(set(groups)):
                        train,test=np.flatnonzero(groups!=comparison),np.flatnonzero(groups==comparison)
                        if set(cell.pair_id.iloc[train])&set(cell.pair_id.iloc[test]):raise ValueError('Pair leakage')
                        if len(set(y[train]))!=2 or min(np.bincount(y[train],minlength=2))<2:
                            unsupported.append(dict(analysis=analysis,representation=representation,subject=int(subject),
                                                    width_ms=width,comparison=comparison,reason='insufficient_training_choice_classes'))
                            continue
                        scaler=StandardScaler().fit(meta[train])
                        before,after={'metadata':scaler.transform(meta[train])},{'metadata':scaler.transform(meta[test])}
                        dims={}
                        for period,x in signals.items():
                            if not np.isfinite(x).all():raise ValueError('Nonfinite pair feature')
                            transform=make_pipeline(StandardScaler(),PCA(n_components=.9,svd_solver='full'),StandardScaler())
                            before[period]=transform.fit_transform(x[train]);after[period]=transform.transform(x[test])
                            dims[period]=before[period].shape[1]
                        result=cell.iloc[test].copy().assign(analysis=analysis,representation=representation,width_ms=width,
                                                          held_out_comparison=comparison,prior=float(y[train].mean()))
                        for model in MODELS:
                            if model=='prior':continue
                            parts=model.split('_')
                            classifier=LogisticRegression(C=1.,solver='lbfgs',max_iter=3000,random_state=24)
                            classifier.fit(np.column_stack([before[p] for p in parts]),y[train])
                            if int(classifier.n_iter_[0])>=classifier.max_iter:raise ValueError('Logistic regression did not converge')
                            result[model]=classifier.predict_proba(np.column_stack([after[p] for p in parts]))[:,1]
                        records.append(result)
                        audits.append(dict(analysis=analysis,representation=representation,subject=int(subject),width_ms=width,
                            held_out_comparison=comparison,training_comparisons=sorted(set(groups[train])),
                            training_pairs=cell.pair_id.iloc[train].tolist(),test_pairs=cell.pair_id.iloc[test].tolist(),
                            training_subjects=[int(subject)],training_pairs_count=len(train),test_pairs_count=len(test),
                            training_second_choices=int(y[train].sum()),test_second_choices=int(y[test].sum()),
                            pre_original_features=signals['pre'].shape[1],post_original_features=signals['post'].shape[1],
                            pre_components=int(dims['pre']),post_components=int(dims['post'])))
                print(f'{analysis}, {representation}: participant {subject}: {len(cell)} actual pairs',flush=True)
    if not records:raise ValueError('No supported paired model folds')
    result=pd.concat(records,ignore_index=True)
    if result.duplicated(['analysis','representation','subject','width_ms','pair_id']).any():raise ValueError('Repeated prediction')
    if not np.isfinite(result[list(MODELS)]).all().all():raise ValueError('Missing prediction')
    return result,audits,unsupported


def metrics(group,model):
    result=epoch_metrics(group.rename(columns={'target_second':'target_chosen'}),model)
    result['pairs']=result.pop('epochs');result['second_choices']=result.pop('chosen');result['first_choices']=result.pop('not_chosen')
    result['first_recall']=result.pop('not_chosen_recall');result['second_recall']=result.pop('chosen_recall')
    return result


def summarize(predictions):
    per_subject,aggregate,contrasts=[],[],[]
    for (analysis,representation,subject,width),group in predictions.groupby(['analysis','representation','subject','width_ms']):
        for model in MODELS:
            per_subject.append(dict(analysis=analysis,representation=representation,subject=int(subject),width_ms=int(width),
                                    model=model,**metrics(group,model)))
    cells=sorted(set((r['analysis'],r['representation'],r['width_ms']) for r in per_subject))
    for analysis,representation,width in cells:
        cell=[r for r in per_subject if (r['analysis'],r['representation'],r['width_ms'])==(analysis,representation,width)]
        for model in MODELS:
            rows=[r for r in cell if r['model']==model]
            aggregate.append(dict(analysis=analysis,representation=representation,width_ms=width,model=model,
                participants=len(rows),pairs=sum(r['pairs'] for r in rows),
                **{name:interval([r[name] for r in rows if r[name] is not None])
                   if any(r[name] is not None for r in rows) else None
                   for name in ('accuracy','balanced_accuracy','first_recall','second_recall','log_loss','brier','auc')}))
        for name,(label,baseline,comparison) in CONTRASTS.items():
            participants=[]
            for subject in sorted(set(r['subject'] for r in cell)):
                a=next(r for r in cell if r['subject']==subject and r['model']==baseline)
                b=next(r for r in cell if r['subject']==subject and r['model']==comparison)
                if a['balanced_accuracy'] is None or b['balanced_accuracy'] is None:continue
                participants.append(dict(subject=subject,balanced_accuracy_gain=b['balanced_accuracy']-a['balanced_accuracy'],
                                         log_loss_reduction=a['log_loss']-b['log_loss']))
            row=dict(analysis=analysis,representation=representation,width_ms=width,contrast=name,label=label,
                     baseline=baseline,comparison=comparison,participants=participants)
            for metric in ('balanced_accuracy_gain','log_loss_reduction'):
                values=[r[metric] for r in participants]
                row[metric]=dict(**interval(values),test=paired_sign_test(values)) if values else None
            contrasts.append(row)
    tests=[r[m]['test'] for r in contrasts for m in ('balanced_accuracy_gain','log_loss_reduction') if r[m] is not None]
    for test,corrected in zip(tests,holm_adjust([t['p'] for t in tests])):test.update(p_holm=float(corrected),family_size=len(tests))
    return dict(per_subject=per_subject,aggregate=aggregate,contrasts=contrasts)
