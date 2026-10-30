"""Group-aware fixed-model evaluation with auditable out-of-fold predictions."""
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .events import CODES


def model():
    return make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(
        solver='lsqr', shrinkage='auto', priors=np.full(5, .2)))


def scores(y, predicted):
    return dict(n_events=len(y), n_classes=len(np.unique(y)),
                balanced_accuracy=float(recall_score(y, predicted, labels=CODES,
                                                     average='macro', zero_division=0)),
                macro_f1=float(f1_score(y, predicted, labels=CODES, average='macro', zero_division=0)),
                accuracy=float(accuracy_score(y, predicted)),
                confusion_matrix=confusion_matrix(y, predicted, labels=CODES).tolist())


def make_splits(meta, mode='within', n_splits=5):
    """Hold out entire waveform comparisons or subjects and purge shared samples."""
    if mode == 'within':
        if meta.subject.nunique() != 1:
            raise ValueError('Within-participant evaluation requires one subject')
        groups = meta.group.to_numpy()
        if len(set(groups)) < n_splits:
            raise ValueError('Too few comparison groups for requested folds')
        splitter = GroupKFold(n_splits=n_splits)
    elif mode == 'loso':
        groups = meta.subject.to_numpy()
        if len(set(groups)) < 2:
            raise ValueError('LOSO needs at least two participants')
        splitter = LeaveOneGroupOut()
    else:
        raise ValueError(mode)
    splits, seen = [], np.zeros(len(meta), int)
    for train, test in splitter.split(meta, groups=groups):
        # Every feature set draws only from [-.75,+.50] s around its trigger.
        remove_pairs = set()
        for idx in train:
            same = test[meta.subject.to_numpy()[test] == meta.subject.iloc[idx]]
            if np.any(np.abs(meta.elapsed_s.to_numpy()[same]-meta.elapsed_s.iloc[idx]) <= 1.25):
                remove_pairs.add(meta.pair_id.iloc[idx])
        train = np.array([i for i in train if meta.pair_id.iloc[i] not in remove_pairs], int)
        if set(meta.trigger.iloc[train]) != set(CODES):
            raise ValueError('Training fold lacks one or more waveform classes')
        if set(meta.pair_id.iloc[train]) & set(meta.pair_id.iloc[test]):
            raise AssertionError('Paired presentations leaked across folds')
        if set(meta.group.iloc[train]) & set(meta.group.iloc[test]):
            raise AssertionError('Comparison groups leaked across folds')
        seen[test] += 1
        splits.append((train, test))
    if not np.all(seen == 1):
        raise AssertionError('Every event must receive exactly one held-out prediction')
    return splits


def permute_pairs(y, meta, rng):
    """Conditional null: circularly shift whole trial pairs within comparisons.

    A comparison-wide orientation flip also breaks first/second associations.
    Pair membership and comparison-specific waveform availability are preserved.
    This is a restricted diagnostic null, not an unrestricted iid-label test.
    """
    out = np.asarray(y).copy()
    for _, group in meta.groupby('group', sort=False):
        indices = group.index.to_numpy()
        if len(indices) % 2 or not np.array_equal(group.position, np.tile([1, 2], len(indices)//2)):
            raise ValueError('Permutation requires complete chronological pairs')
        paired = y[indices].reshape(-1, 2)
        shifted = np.roll(paired, int(rng.integers(len(paired))), axis=0)
        if rng.integers(2):
            shifted = shifted[:, ::-1]
        out[indices] = shifted.ravel()
    return out


def predict_oof(x, y, splits):
    predicted = np.full(len(y), -1, int)
    probabilities = np.zeros((len(y), 5))
    for train, test in splits:
        estimator = model()
        estimator.fit(x[train], y[train])
        predicted[test] = estimator.predict(x[test])
        probabilities[test] = estimator.predict_proba(x[test])
    if np.any(predicted < 0):
        raise AssertionError('Missing held-out predictions')
    return predicted, probabilities


def evaluate(features, meta, mode, permutations=99, seed=42):
    meta = meta.reset_index(drop=True)
    y = meta.trigger.to_numpy(int)
    if set(y) != set(CODES):
        raise ValueError('Evaluation requires all five classes')
    splits = make_splits(meta, mode)
    rng = np.random.default_rng(seed)
    null_labels = [permute_pairs(y, meta, rng) for _ in range(permutations)]
    report, predictions = {}, meta[['subject', 'event_index', 'pair_id', 'group', 'position', 'trigger']].copy()
    predictions['fold'] = -1
    fold_rows = []
    for fold, (train, test) in enumerate(splits):
        predictions.loc[test, 'fold'] = fold
        fold_rows.append(dict(fold=fold, n_train=len(train), n_test=len(test),
                              train_groups=sorted(set(meta.group.iloc[train])),
                              test_groups=sorted(set(meta.group.iloc[test])),
                              test_classes=sorted(int(c) for c in set(y[test]))))
    with threadpool_limits(limits=1):
        for name, x in features.items():
            predicted, probabilities = predict_oof(x, y, splits)
            observed = scores(y, predicted)
            null = [scores(py, predict_oof(x, py, splits)[0])['balanced_accuracy'] for py in null_labels]
            observed.update(null_balanced_accuracies=null,
                            permutation_p=(1+sum(v >= observed['balanced_accuracy'] for v in null))/(1+len(null)) if null else None,
                            position_scores={str(p): scores(y[meta.position == p], predicted[meta.position == p]) for p in [1, 2]},
                            subject_scores={str(s): scores(y[meta.subject == s], predicted[meta.subject == s]) for s in meta.subject.unique()})
            report[name] = observed
            predictions[name+'_predicted'] = predicted
            for ci, code in enumerate(CODES):
                predictions[f'{name}_prob_{code}'] = probabilities[:, ci]
    return dict(mode=mode, seed=seed, permutations=permutations, class_order=list(CODES),
                chance_reference=.2, folds=fold_rows, models=report), predictions
