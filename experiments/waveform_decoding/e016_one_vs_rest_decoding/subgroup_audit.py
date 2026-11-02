"""Descriptive E015/E016 subgroup audit; no new training or significance tests."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from .core import CODES, WAVES, binary_metrics
from .report import audit
from experiments.waveform_decoding.e015_position_repeatability_decoding.report import load_checked


def run(args):
    if args.output.exists():
        raise ValueError('Use a fresh output directory')
    binary, verification = audit(args.binary)
    multiclass, model_rows = [], []
    for mode, root in [('strict', args.strict), ('exploratory', args.exploratory)]:
        _, _, models = load_checked(root)
        for model in models:
            if model['status'] != 'observed_complete':
                continue
            subject, position = model['subject'], model['position']
            pred = pd.read_csv(root / f'sub-{subject}' / f'position-{position}' / 'predictions.csv')
            for selector, saved in model['models'].items():
                model_rows.append(dict(mode=mode, subject=subject, position=position, selector=selector,
                                       balanced_accuracy=saved['balanced_accuracy'], n_events=saved['n_events']))
                for code in CODES:
                    truth = (pred.trigger.to_numpy() == code).astype(int)
                    guess = (pred[selector + '_predicted'].to_numpy() == code).astype(int)
                    metrics = binary_metrics(truth, guess.astype(float), guess)
                    # A hard-label AUC equals binary BA; keep its name out of this table
                    # so it cannot be mistaken for E015's probability-score ROC-AUC.
                    metrics.pop('roc_auc')
                    multiclass.append(dict(mode=mode, subject=subject, position=position, selector=selector,
                                           target=int(code), waveform=WAVES[int(code)],
                                           predicted_fraction=float(guess.mean()), **metrics))
    multi = pd.DataFrame(multiclass)
    overall = pd.DataFrame(model_rows)
    keys = ['mode', 'subject', 'position', 'target']
    matched = multi[multi.selector == 'repeatability'].merge(binary, on=keys, suffixes=('_multi', '_binary'), validate='one_to_one')
    np.testing.assert_allclose(matched.balanced_accuracy_multi, matched.source_argmax_ovr_balanced_accuracy, atol=1e-12)
    matched['binary_minus_multiclass_ba'] = matched.balanced_accuracy_binary - matched.balanced_accuracy_multi
    args.output.mkdir(parents=True)
    multi.drop(columns='confusion_matrix').to_csv(args.output / 'multiclass_per_target.csv', index=False)
    binary.drop(columns='confusion_matrix').to_csv(args.output / 'binary_per_target.csv', index=False)
    matched.drop(columns=['confusion_matrix_multi', 'confusion_matrix_binary']).to_csv(args.output / 'matched_models.csv', index=False)
    overall.to_csv(args.output / 'multiclass_overall.csv', index=False)
    multi.groupby(['mode', 'selector', 'position', 'waveform'])[['balanced_accuracy', 'sensitivity', 'specificity', 'precision']].mean().to_csv(args.output / 'multiclass_waveform_means.csv')
    binary.groupby(['mode', 'position', 'waveform'])[['balanced_accuracy', 'within_comparison_auc', 'sensitivity', 'specificity']].mean().to_csv(args.output / 'binary_waveform_means.csv')
    overall.groupby(['mode', 'selector', 'subject']).agg(mean_balanced_accuracy=('balanced_accuracy', 'mean'), positions=('position', 'count')).to_csv(args.output / 'multiclass_participant_means.csv')
    binary.groupby(['mode', 'subject']).agg(mean_balanced_accuracy=('balanced_accuracy', 'mean'), detectors=('position', 'count')).to_csv(args.output / 'binary_participant_means.csv')
    consistency = []
    for mode in ['strict', 'exploratory']:
        wide = binary[binary['mode'] == mode].pivot(index=['subject', 'waveform'], columns='position', values='balanced_accuracy').dropna()
        for (subject, waveform), row in wide[(wide[1] > .5) & (wide[2] > .5)].iterrows():
            consistency.append(dict(mode=mode, subject=subject, waveform=waveform, position1=row[1], position2=row[2]))
    pd.DataFrame(consistency).to_csv(args.output / 'binary_above_reference_both_positions.csv', index=False)
    summary = dict(inference_complete=False, permutations=0, significance_tests=0, posthoc_subgroup_search=True,
                   comparisons_are_not_independent_replications=True, source_verification=verification, counts={})
    for mode in ['strict', 'exploratory']:
        m = multi[(multi['mode'] == mode) & (multi.selector == 'repeatability')]
        b = binary[binary['mode'] == mode]
        match = matched[matched['mode'] == mode]
        summary['counts'][mode] = dict(target_conditions=len(b), multiclass_target_ba_above_half=int((m.balanced_accuracy > .5).sum()),
            binary_ba_above_half=int((b.balanced_accuracy > .5).sum()),
            multiclass_recall_above_20_but_target_ba_not_above_half=int(((m.sensitivity > .2) & (m.balanced_accuracy <= .5)).sum()),
            binary_ba_higher_than_multiclass=int((match.binary_minus_multiclass_ba > 1e-12).sum()),
            mean_binary_ba_change=float(match.binary_minus_multiclass_ba.mean()))
    (args.output / 'verification_and_counts.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary['counts'], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strict', type=Path, default=Path('output/e015_strict_observed_20261102'))
    parser.add_argument('--exploratory', type=Path, default=Path('output/e015_exploratory_observed_20261102'))
    parser.add_argument('--binary', type=Path, default=Path('output/e016_binary_observed_20261102'))
    parser.add_argument('--output', type=Path, default=Path('output/e015_e016_subgroup_audit_20261102'))
    run(parser.parse_args())
