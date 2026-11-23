"""Participant-held-out probability predictions and paired descriptive uncertainty."""

from itertools import permutations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


MODEL_LABELS = {
    'prior': 'Training choice proportion',
    'metadata': 'Waveform identities and trial order',
    'metadata_pre': 'Metadata + pre EEG',
    'metadata_post': 'Metadata + paired post EEG',
    'metadata_pre_post': 'Metadata + pre + paired post EEG',
    'pre_only': 'First-pulse pre EEG alone',
    'post_only': 'First-pulse post EEG alone',
    'paired_post_only': 'Paired post EEG alone',
}


def model_features(table, features):
    """Fixed ordered-pair indicators and ordinal trial indices; EEG adds 30 columns."""
    comparisons = [f'{a}-{b}' for a, b in permutations(range(239, 244), 2)]
    indicators = np.column_stack([table.ordered_pair.eq(pair) for pair in comparisons])
    metadata = np.column_stack([indicators, table.pair_index, table.within_block_index])
    pre, post = features['pre'], features['paired_post']
    return dict(
        metadata=metadata, metadata_pre=np.column_stack([metadata, pre]),
        metadata_post=np.column_stack([metadata, post]),
        metadata_pre_post=np.column_stack([metadata, pre, post]),
        pre_only=pre, post_only=features['first_post'], paired_post_only=post,
    )


def predict(table, features):
    """One probability per pair and model, with training-only standardization."""
    x_by_model = model_features(table, features)
    result = table.copy()
    subjects = table.subject.to_numpy()
    y = table.target_second.to_numpy()
    if len(np.unique(subjects)) < 3:
        raise ValueError('At least three participants are required')
    audits = []
    for subject in np.unique(subjects):
        train, test = np.flatnonzero(subjects != subject), np.flatnonzero(subjects == subject)
        if set(y[train]) != {0, 1}:
            raise ValueError('A training fold lacks one behavioral choice class')
        result.loc[result.index[test], 'prior'] = y[train].mean()
        for name, x in x_by_model.items():
            if not np.isfinite(x).all():
                raise ValueError(f'Nonfinite model input for {name}')
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=1., solver='lbfgs', max_iter=2000, random_state=21),
            )
            model.fit(x[train], y[train])
            fitted = model[-1]
            if int(fitted.n_iter_[0]) >= fitted.max_iter:
                raise ValueError('Logistic regression did not converge')
            result.loc[result.index[test], name] = model.predict_proba(x[test])[:, 1]
            audits.append(dict(
                held_out_subject=int(subject), model=name, training_pairs=len(train),
                test_pairs=len(test), features=x.shape[1],
                training_constant_features=int(np.sum(np.ptp(x[train], axis=0) == 0)),
                training_max_abs=float(np.abs(x[train]).max()),
                iterations=int(fitted.n_iter_[0]),
            ))
    if not np.isfinite(result[list(MODEL_LABELS)].to_numpy()).all():
        raise ValueError('Missing held-out probabilities')
    return result, pd.DataFrame(audits)


def metrics(table, model):
    y = table.target_second.to_numpy()
    probability = table[model].to_numpy()
    prediction = (probability >= .5).astype(int)
    matrix = confusion_matrix(y, prediction, labels=[0, 1])
    return dict(
        pairs=len(table), accuracy=float(np.mean(y == prediction)),
        balanced_accuracy=float(balanced_accuracy_score(y, prediction)) if len(set(y)) == 2 else None,
        log_loss=float(log_loss(y, np.column_stack([1 - probability, probability]), labels=[0, 1])),
        brier=float(np.mean((probability - y) ** 2)),
        auc=float(roc_auc_score(y, probability)) if len(set(y)) == 2 else None,
        confusion=matrix.tolist(),
    )


def interval(values):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(20261123)
    boot = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
    return dict(mean=float(values.mean()), ci95=np.quantile(boot, [.025, .975]).tolist())


def summarize(predictions):
    """Equal participant summaries and intervals conditional on fitted LOSO predictions."""
    per_subject, aggregate, contrasts = [], {}, {}
    for name in MODEL_LABELS:
        records = [dict(subject=int(subject), model=name, **metrics(group, name))
                   for subject, group in predictions.groupby('subject')]
        per_subject.extend(records)
        aggregate[name] = dict(
            pooled=metrics(predictions, name),
            participant_mean={
                metric: interval([row[metric] for row in records if row[metric] is not None])
                for metric in ('accuracy', 'balanced_accuracy', 'log_loss', 'brier', 'auc')
            },
        )
    for name, baseline, addition in (
        ('post_added_to_metadata_pre', 'metadata_pre', 'metadata_pre_post'),
        ('post_vs_pre_alone', 'pre_only', 'post_only'),
        ('pre_added_to_metadata', 'metadata', 'metadata_pre'),
        ('paired_post_added_to_metadata', 'metadata', 'metadata_post'),
    ):
        records = []
        for subject, group in predictions.groupby('subject'):
            before, after = metrics(group, baseline), metrics(group, addition)
            records.append(dict(
                subject=int(subject),
                log_loss_reduction=before['log_loss'] - after['log_loss'],
                balanced_accuracy_gain=after['balanced_accuracy'] - before['balanced_accuracy']
                if before['balanced_accuracy'] is not None and after['balanced_accuracy'] is not None
                else None,
            ))
        contrasts[name] = dict(
            baseline=baseline, comparison=addition, participants=records,
            log_loss_reduction=interval([row['log_loss_reduction'] for row in records]),
            balanced_accuracy_gain=interval([row['balanced_accuracy_gain'] for row in records
                                            if row['balanced_accuracy_gain'] is not None]),
        )
    return dict(
        pairs=len(predictions), participants=int(predictions.subject.nunique()),
        models=aggregate, contrasts=contrasts, per_subject=per_subject,
    )
