"""Within-person and position-specific naturalness decoding on unseen comparisons."""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .evaluate import interval
from experiments.waveform_decoding.e022_triangle_lindec_decoding.evaluate import (
    resolution, paired_sign_test, holm_adjust,
)


MODELS = {
    'prior': 'Training choice proportion',
    'metadata': 'Waveform identities and trial order',
    'pre': 'Pre EEG alone',
    'post': 'Post EEG alone',
    'metadata_pre': 'Metadata + pre EEG',
    'metadata_post': 'Metadata + post EEG',
    'metadata_pre_post': 'Metadata + pre + post EEG',
}
CONTRASTS = {
    'post_vs_pre': ('Post EEG versus pre EEG', 'pre', 'post'),
    'post_added': ('Add post EEG to metadata + pre EEG', 'metadata_pre', 'metadata_pre_post'),
}


def metadata_features(events):
    """Known waveform identities and chronological indices; no choice/stopping inputs."""
    return np.column_stack([
        *[events.trigger.eq(code).to_numpy() for code in range(239, 244)],
        *[events.opponent_trigger.eq(code).to_numpy() for code in range(239, 244)],
        events.pair_index.to_numpy(), events.within_block_index.to_numpy(),
    ]).astype(float)


def predict(events, features):
    records, audits, unsupported = [], [], []
    for analysis, mask in [('full', np.ones(len(events), dtype=bool)),
                           ('nonterminal', ~events.terminal_vote.to_numpy(bool))]:
        eligible = events.loc[mask]
        for (subject, position), cell in eligible.groupby(['subject', 'position']):
            indices = cell.index.to_numpy()
            y = cell.target_chosen.to_numpy()
            groups = cell.comparison.to_numpy()
            metadata = metadata_features(cell)
            for width in (50, 100):
                signals = {period: resolution(values[indices], width) for period, values in features.items()}
                for comparison in sorted(set(groups)):
                    train, test = np.flatnonzero(groups != comparison), np.flatnonzero(groups == comparison)
                    if set(cell.pair_id.iloc[train]) & set(cell.pair_id.iloc[test]):
                        raise ValueError('A behavioral pair crosses the training/test split')
                    if len(set(y[train])) != 2 or min(np.bincount(y[train], minlength=2)) < 2:
                        unsupported.append(dict(analysis=analysis, subject=int(subject), position=int(position),
                                                width_ms=width, comparison=comparison,
                                                reason='fewer_than_two_training_examples_per_choice'))
                        continue
                    scaler = StandardScaler().fit(metadata[train])
                    train_blocks = {'metadata': scaler.transform(metadata[train])}
                    test_blocks = {'metadata': scaler.transform(metadata[test])}
                    dimensions = {}
                    for period, x in signals.items():
                        if not np.isfinite(x).all():
                            raise ValueError('Nonfinite EEG input')
                        transform = make_pipeline(StandardScaler(), PCA(n_components=.90, svd_solver='full'),
                                                  StandardScaler())
                        train_blocks[period] = transform.fit_transform(x[train])
                        test_blocks[period] = transform.transform(x[test])
                        dimensions[period] = int(train_blocks[period].shape[1])
                    table = cell.iloc[test][[
                        'subject', 'position', 'pair_id', 'event_index', 'comparison', 'block_index',
                        'trigger', 'opponent_trigger', 'target_chosen', 'terminal_vote',
                    ]].copy().assign(analysis=analysis, width_ms=width, held_out_comparison=comparison)
                    table['prior'] = float(y[train].mean())
                    for model in MODELS:
                        if model == 'prior':
                            continue
                        blocks = model.split('_')
                        x_train = np.column_stack([train_blocks[name] for name in blocks])
                        x_test = np.column_stack([test_blocks[name] for name in blocks])
                        classifier = LogisticRegression(C=1., solver='lbfgs', max_iter=3000, random_state=24)
                        classifier.fit(x_train, y[train])
                        if int(classifier.n_iter_[0]) >= classifier.max_iter:
                            raise ValueError('Logistic regression did not converge')
                        table[model] = classifier.predict_proba(x_test)[:, 1]
                    records.append(table)
                    audits.append(dict(
                        analysis=analysis, subject=int(subject), position=int(position), width_ms=width,
                        held_out_comparison=comparison, training_comparisons=sorted(set(groups[train])),
                        training_pairs=cell.pair_id.iloc[train].tolist(), test_pairs=cell.pair_id.iloc[test].tolist(),
                        training_epochs=len(train), test_epochs=len(test),
                        training_chosen=int(y[train].sum()), test_chosen=int(y[test].sum()),
                        original_eeg_features=signals['pre'].shape[1], pre_components=dimensions['pre'],
                        post_components=dimensions['post'], metadata_features=metadata.shape[1],
                        training_subjects=[int(subject)], training_positions=[int(position)],
                    ))
            print(f'{analysis}: participant {subject}, position {position}: {len(cell)} epochs', flush=True)
    predictions = pd.concat(records, ignore_index=True)
    key = ['analysis', 'subject', 'position', 'width_ms', 'event_index']
    if predictions.duplicated(key).any():
        raise ValueError('Repeated held-out prediction')
    if not np.isfinite(predictions[list(MODELS)]).all().all():
        raise ValueError('Missing model prediction')
    return predictions, audits, unsupported


def metrics(group, model):
    y = group.target_chosen.to_numpy()
    p = group[model].to_numpy()
    predicted = (p >= .5).astype(int)
    matrix = confusion_matrix(y, predicted, labels=[0, 1])
    both = set(y) == {0, 1}
    recalls = [float(matrix[i, i] / matrix[i].sum()) if matrix[i].sum() else None for i in (0, 1)]
    return dict(epochs=len(y), chosen=int(y.sum()), not_chosen=int(len(y) - y.sum()),
                accuracy=float(np.mean(y == predicted)),
                balanced_accuracy=float(np.mean(recalls)) if both else None,
                chosen_recall=recalls[1], not_chosen_recall=recalls[0],
                log_loss=float(log_loss(y, p, labels=[0, 1])),
                brier=float(np.mean((p-y)**2)), auc=float(roc_auc_score(y, p)) if both else None,
                confusion=matrix.tolist())


def summarize(predictions):
    per_subject, aggregate, contrasts = [], [], []
    for (analysis, subject, position, width), group in predictions.groupby(
        ['analysis', 'subject', 'position', 'width_ms']
    ):
        for model in MODELS:
            per_subject.append(dict(analysis=analysis, subject=int(subject), position=int(position),
                                    width_ms=int(width), model=model, **metrics(group, model)))
    for analysis in sorted(predictions.analysis.unique()):
        for position in (1, 2):
            for width in (50, 100):
                cell = [r for r in per_subject if (r['analysis'],r['position'],r['width_ms']) ==
                        (analysis,position,width)]
                for model in MODELS:
                    rows = [r for r in cell if r['model'] == model]
                    aggregate.append(dict(analysis=analysis, position=position, width_ms=width, model=model,
                        participants=len(rows), epochs=sum(r['epochs'] for r in rows),
                        **{metric: interval([r[metric] for r in rows if r[metric] is not None])
                           if any(r[metric] is not None for r in rows) else None
                           for metric in ('accuracy','balanced_accuracy','log_loss','brier','auc',
                                          'chosen_recall','not_chosen_recall')}))
                for name, (label, before, after) in CONTRASTS.items():
                    participants = []
                    for subject in sorted(set(r['subject'] for r in cell)):
                        a = next(r for r in cell if r['subject']==subject and r['model']==before)
                        b = next(r for r in cell if r['subject']==subject and r['model']==after)
                        if a['balanced_accuracy'] is None or b['balanced_accuracy'] is None:
                            continue
                        participants.append(dict(subject=subject,
                            balanced_accuracy_gain=b['balanced_accuracy']-a['balanced_accuracy'],
                            log_loss_reduction=a['log_loss']-b['log_loss']))
                    row = dict(analysis=analysis, position=position, width_ms=width, contrast=name,
                               label=label, baseline=before, comparison=after, participants=participants)
                    for metric in ('balanced_accuracy_gain','log_loss_reduction'):
                        values = [r[metric] for r in participants]
                        row[metric] = dict(**interval(values), test=paired_sign_test(values)) if values else None
                    contrasts.append(row)
    tests = [r[metric]['test'] for r in contrasts for metric in ('balanced_accuracy_gain','log_loss_reduction')
             if r[metric] is not None]
    for test, corrected in zip(tests, holm_adjust([t['p'] for t in tests])):
        test.update(p_holm=float(corrected), family_size=len(tests))
    return dict(per_subject=per_subject, aggregate=aggregate, contrasts=contrasts)
