"""Matched comparisons, full null-model refits and repeated feature audits."""
import json
import numpy as np
import pandas as pd
from experiments.waveform_decoding.e001_voltage_baseline.evaluate import make_splits, predict_oof, permute_pairs, scores
from experiments.signal_quality.e003_voltage_feature_audit.__main__ import verify_saved_folds
from experiments.signal_quality.e003_voltage_feature_audit.core import audit_training_fold, describe_features
from experiments.waveform_decoding.e004_xdawn_decoding.data import sha, write_json
from .core import representations, contracts


def evaluate_job(identity, mode, epochs, meta, sfreq, width, args):
    out = args.output/identity
    out.mkdir()
    meta = meta.reset_index(drop=True)
    splits = make_splits(meta, mode)
    reference = verify_saved_folds(args.baseline, meta, splits, mode)
    original = pd.read_csv(reference)
    y = meta.trigger.to_numpy(int)
    predicted = meta[['subject', 'event_index', 'pair_id', 'group', 'position', 'trigger']].copy()
    predicted['fold'] = -1
    membership = []
    for fold, (train, test) in enumerate(splits):
        predicted.loc[test, 'fold'] = fold
        membership.append(dict(fold=fold,
            train_events=meta.iloc[train][['subject', 'event_index']].values.tolist(),
            test_events=meta.iloc[test][['subject', 'event_index']].values.tolist()))
    write_json(out/'fold_membership.json', membership)
    meta.to_csv(out/'events.csv', index=False)
    features, contract, diagnostics, arrays = {}, {}, {}, {}
    for window, x in epochs.items():
        bundles, linear, diagnostics[window] = representations(x, sfreq, width)
        arrays[f'{window}_linear_power_uv2'] = linear
        for name, matrix in bundles.items():
            key = f'{name}_{window}'
            features[key] = matrix
            arrays[key] = matrix
            contract[key] = contracts(window)[name]
    np.savez_compressed(out/'features.npz', **arrays)
    write_json(out/'feature_contract.json', contract)
    write_json(out/'spectral_diagnostics.json', diagnostics)
    report = dict(identity=identity, mode=mode, seed=args.seed, permutations=args.permutations,
        n_events=len(meta), original_fold_membership_verified=True,
        original_predictions_sha256=sha(reference), spectral_diagnostics=diagnostics, models={})
    audits, distributions, shrinkages, resamples = [], [], [], []
    models = out/'fold_models'
    models.mkdir()
    for key, x in features.items():
        p, prob = predict_oof(x, y, splits)
        if key.startswith('voltage_'):
            window = key.split('_')[1]
            np.testing.assert_array_equal(p, original[window+'_predicted'])
            np.testing.assert_allclose(prob, original[[f'{window}_prob_{code}' for code in range(239,244)]],
                                       rtol=1e-8, atol=1e-10)
        result = scores(y, p)
        result.update(position_scores={str(pos): scores(y[meta.position==pos], p[meta.position==pos]) for pos in [1,2]},
            subject_scores={str(s): scores(y[meta.subject==s], p[meta.subject==s]) for s in meta.subject.unique()})
        report['models'][key] = result
        predicted[key+'_predicted'] = p
        for j, code in enumerate(range(239,244)):
            predicted[f'{key}_prob_{code}'] = prob[:,j]
        for fold, (train, test) in enumerate(splits):
            row, shrink, boot, _ = audit_training_fold(x[train], x[test], y[train],
                meta.iloc[train].reset_index(drop=True), repeats=args.audit_resamples,
                seed=args.seed+fold, units=[entry['unit'] for entry in contract[key]])
            column = contract[key][row['heldout_max_column']]['feature']
            event = meta.iloc[test[row['heldout_max_row']]]
            row.update(representation=key, fold=fold, heldout_max_feature=column,
                heldout_max_subject=int(event.subject), heldout_max_event_index=int(event.event_index))
            audits.append(row)
            for split, values in [('train', x[train]), ('test', x[test])]:
                distributions.append(describe_features(values, contract[key]).assign(representation=key, fold=fold, split=split))
            shrinkages.append(shrink.assign(representation=key, fold=fold))
            resamples.append(boot.assign(representation=key, fold=fold))
            # Store actual parameters and confirm audited model predicts the same held-out rows.
            from experiments.waveform_decoding.e001_voltage_baseline.evaluate import model
            estimator = model().fit(x[train], y[train])
            np.testing.assert_array_equal(estimator.predict(x[test]), p[test])
            scaler, lda = estimator.steps[0][1], estimator.steps[-1][1]
            np.savez_compressed(models/f'{key}_fold-{fold}.npz', scaler_mean=scaler.mean_,
                scaler_scale=scaler.scale_, lda_covariance=lda.covariance_,
                lda_coef=lda.coef_, lda_intercept=lda.intercept_, classes=lda.classes_)
        print(f'{identity} {key}: BA={result["balanced_accuracy"]:.4f}; audit complete', flush=True)
    null = {key: [] for key in features}
    rng = np.random.default_rng(args.seed)
    for permutation in range(args.permutations):
        py = permute_pairs(y, meta, rng)
        for key, x in features.items():
            # No learned feature transform; every null refits scaler and LDA.
            null[key].append(scores(py, predict_oof(x, py, splits)[0])['balanced_accuracy'])
        if permutation==0 or (permutation+1)%10==0 or permutation+1==args.permutations:
            print(f'{identity}: {permutation+1}/{args.permutations} restricted permutations', flush=True)
    for key, result in report['models'].items():
        values = null[key]
        result.update(null_balanced_accuracies=values,
            conditional_p=(1+sum(value>=result['balanced_accuracy'] for value in values))/(1+len(values)) if values else None)
    # This independent replay also checks both original voltage null distributions.
    source = args.baseline/('loso_scores.json' if mode=='loso' else f'{identity}/scores.json')
    old = json.loads(source.read_text())
    baseline_config = json.loads((args.baseline/'run_config.json').read_text())
    comparable = (args.seed==old['seed'] and
        sha(args.evaluate_source)==baseline_config['implementation_sha256']['evaluate.py'] and
        all(args.dependencies.get(k)==v for k,v in baseline_config['dependencies'].items()))
    replay_counts = {}
    if comparable:
        for window in epochs:
            count = min(args.permutations,len(old['models'][window]['null_balanced_accuracies']))
            np.testing.assert_allclose(null[f'voltage_{window}'][:count],
                old['models'][window]['null_balanced_accuracies'][:count], rtol=0, atol=0)
            replay_counts[window] = count
    report['voltage_null_replay_verified'] = bool(comparable and replay_counts and min(replay_counts.values())>0)
    report['voltage_null_replay_counts_per_window'] = replay_counts
    predicted.to_csv(out/'predictions.csv', index=False)
    pd.DataFrame(null).to_csv(out/'null_balanced_accuracy.csv', index=False)
    pd.DataFrame(audits).to_csv(out/'fold_audit.csv', index=False)
    pd.concat(distributions, ignore_index=True).to_csv(out/'feature_distributions.csv', index=False)
    pd.concat(shrinkages, ignore_index=True).to_csv(out/'class_shrinkage.csv', index=False)
    pd.concat(resamples, ignore_index=True).to_csv(out/'covariance_resamples.csv', index=False)
    write_json(out/'scores.json', report)
    return report
