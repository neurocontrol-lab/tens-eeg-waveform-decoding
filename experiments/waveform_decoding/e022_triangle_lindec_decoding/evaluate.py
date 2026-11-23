"""Within-participant, position-separated, comparison-held-out binary decoding."""

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from experiments.waveform_decoding.e021_naturalness_choice_decoding.evaluate import interval


MODELS = {
    'pca_lda': 'PCA + shrinkage LDA',
    'pca_svm': 'PCA + linear SVM',
    'svm': 'Linear SVM',
}


def paired_sign_test(differences):
    """Exact two-sided participant sign test; omit numerical ties, not epochs."""
    differences = np.asarray(differences, dtype=float)
    if not np.isfinite(differences).all():
        raise ValueError('Participant differences must be finite')
    nonzero = differences[~np.isclose(differences, 0, atol=1e-12, rtol=0)]
    positive = int(np.sum(nonzero > 0))
    return dict(
        improved=positive, worsened=int(len(nonzero) - positive),
        tied=int(len(differences) - len(nonzero)),
        p=float(binomtest(positive, len(nonzero), .5).pvalue) if len(nonzero) else 1.,
    )


def holm_adjust(pvalues):
    """Holm familywise correction, valid with dependent comparisons."""
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(
        values[order] * np.arange(len(values), 0, -1)))
    return adjusted


def resolution(values, width):
    """Average adjacent 50 ms bins, preserving the identical one-second duration."""
    if values.shape[1:] != (20, 62) or width not in (50, 100):
        raise ValueError('Expected 20 time bins, 62 channels and width 50 or 100 ms')
    bins = values if width == 50 else values.reshape(len(values), 10, 2, 62).mean(axis=2)
    return bins.reshape(len(values), -1)


def predict(events, features):
    records, audits = [], []
    for (subject, position), cell in events.groupby(['subject', 'position']):
        indices = cell.index.to_numpy()
        y = cell.target_lindec.to_numpy()
        groups = cell.comparison.to_numpy()
        for width in (50, 100):
            for period in ('pre', 'post'):
                x = resolution(features[period][indices], width)
                for fold, comparison in enumerate(sorted(set(groups))):
                    test, train = np.flatnonzero(groups == comparison), np.flatnonzero(groups != comparison)
                    if min(np.bincount(y[train], minlength=2)) < 2:
                        raise ValueError('A training fold has fewer than two epochs per waveform')
                    if set(cell.pair_id.iloc[train]) & set(cell.pair_id.iloc[test]):
                        raise ValueError('A behavioral pair crosses a training/test split')
                    scaler = StandardScaler().fit(x[train])
                    train_scaled, test_scaled = scaler.transform(x[train]), scaler.transform(x[test])
                    pca = PCA(n_components=.90, svd_solver='full').fit(train_scaled)
                    train_pca, test_pca = pca.transform(train_scaled), pca.transform(test_scaled)
                    for name in MODELS:
                        fit_x, test_x = (train_scaled, test_scaled) if name == 'svm' else (train_pca, test_pca)
                        classifier = (
                            LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto', priors=[.5, .5])
                            if name == 'pca_lda' else
                            LinearSVC(C=1., class_weight='balanced', dual='auto',
                                      max_iter=20000, random_state=22)
                        )
                        classifier.fit(fit_x, y[train])
                        if name != 'pca_lda' and classifier.n_iter_ >= classifier.max_iter:
                            raise ValueError('Linear SVM did not converge')
                        decision = classifier.decision_function(test_x)
                        predicted = classifier.predict(test_x)
                        table = cell.iloc[test][[
                            'subject', 'position', 'event_index', 'pair_id', 'comparison',
                            'trigger', 'target_lindec', 'nominal_charge_uC', 'matched_direct',
                        ]].copy()
                        table = table.assign(
                            width_ms=width, period=period, model=name, fold=fold,
                            prediction=predicted, score=decision,
                        )
                        records.append(table)
                        audits.append(dict(
                            subject=int(subject), position=int(position), width_ms=width,
                            period=period, model=name, held_out_comparison=comparison,
                            training_comparisons=sorted(set(groups[train])),
                            training_epochs=len(train), test_epochs=len(test),
                            training_triangle=int(np.sum(y[train] == 0)),
                            training_lindec=int(np.sum(y[train] == 1)),
                            original_features=x.shape[1], fitted_features=fit_x.shape[1],
                            pca_explained_variance=float(pca.explained_variance_ratio_.sum())
                            if name != 'svm' else None,
                            training_constant_features=int(np.sum(scaler.var_ == 0)),
                        ))
        print(f'  participant {subject}, presentation {position}: {len(cell)} held-out epochs', flush=True)
    predictions = pd.concat(records, ignore_index=True)
    keys = ['subject', 'position', 'width_ms', 'period', 'model', 'event_index']
    if predictions.duplicated(keys).any() or len(predictions) != len(events) * 12:
        raise ValueError('Expected one held-out prediction per epoch and configuration')
    return predictions, pd.DataFrame(audits)


def score_group(group):
    y, predicted = group.target_lindec.to_numpy(), group.prediction.to_numpy()
    if set(y) != {0, 1}:
        raise ValueError('A participant/subset/position lacks one waveform')
    matrix = confusion_matrix(y, predicted, labels=[0, 1])
    return dict(
        epochs=len(group), triangle=int(np.sum(y == 0)), lindec=int(np.sum(y == 1)),
        accuracy=float(np.mean(y == predicted)),
        balanced_accuracy=float(balanced_accuracy_score(y, predicted)),
        auc=float(roc_auc_score(y, group.score)),
        triangle_recall=float(matrix[0, 0] / matrix[0].sum()),
        lindec_recall=float(matrix[1, 1] / matrix[1].sum()),
        confusion=matrix.tolist(),
    )


def summarize(predictions):
    per_subject, results = [], []
    matched_all = predictions.groupby(['subject', 'position']).nominal_charge_uC.transform('nunique').eq(1)
    direct = predictions.comparison.eq('241-242')
    for subset, selected in (
        ('all_contexts', predictions),
        ('matched_all_contexts', predictions.loc[matched_all]),
        ('direct', predictions.loc[direct]),
        ('matched_direct', predictions.loc[matched_all & direct]),
        ('non_direct', predictions.loc[~direct]),
        ('matched_non_direct', predictions.loc[matched_all & ~direct]),
    ):
        for (subject, position, width, model, period), group in selected.groupby(
            ['subject', 'position', 'width_ms', 'model', 'period']
        ):
            per_subject.append(dict(
                subset=subset, subject=int(subject), position=int(position),
                width_ms=int(width), model=model, period=period, **score_group(group),
            ))
    table = pd.DataFrame(per_subject)
    for (subset, position, width, model), group in table.groupby(
        ['subset', 'position', 'width_ms', 'model']
    ):
        before = group.loc[group.period.eq('pre')].set_index('subject').sort_index()
        after = group.loc[group.period.eq('post')].set_index('subject').sort_index()
        if not before.index.equals(after.index):
            raise ValueError('Pre and post participant support differs')
        results.append(dict(
            subset=subset, position=int(position), width_ms=int(width), model=model,
            participants=len(before), epochs=int(before.epochs.sum()),
            pre=interval(before.balanced_accuracy),
            post=interval(after.balanced_accuracy),
            post_minus_pre=interval(after.balanced_accuracy.to_numpy() - before.balanced_accuracy.to_numpy()),
            post_pre_test=paired_sign_test(after.balanced_accuracy.to_numpy() - before.balanced_accuracy.to_numpy()),
        ))
    adjusted = holm_adjust([row['post_pre_test']['p'] for row in results])
    for row, corrected in zip(results, adjusted):
        row['post_pre_test'].update(p_holm=float(corrected), family_size=len(results))
    return dict(aggregate=results, per_subject=per_subject)
