"""Run WD-008 naturalness-choice prediction with matched first-pulse controls."""

import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.response_characterization.e017_eeg_charge_relation.extract import digest
from .data import BASELINE, BIN_SECONDS, CHANNELS, WINDOWS, assemble
from .evaluate import MODEL_LABELS, predict, summarize
from .report import render_html, render_markdown


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=Path('output/task4_baseline'))
    parser.add_argument('--visual', type=Path, default=Path('output/task4_visual_inspection_20261030'))
    parser.add_argument('--data-dir', type=Path, default=Path('EEGData'))
    parser.add_argument('--master-csv', type=Path,
                        default=Path('EEG_Codes_Export/02_SourceLocalization_Task4/Master_Data_ToSub12 (1).csv'))
    parser.add_argument('--cache', type=Path, default=Path('output/wd008_feature_cache_baselined_20261123'))
    parser.add_argument('--output', type=Path, default=Path('output/wd008_naturalness_baselined_20261123'))
    parser.add_argument('--unbaselined-reference', type=Path,
                        default=Path('output/wd008_naturalness_20261123/summary.json'))
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output directory. The verified feature cache can be reused.')
    args.output.mkdir(parents=True)
    write_json(args.output / 'status.json', dict(complete=False, stage='extracting'))
    config = dict(
        experiment='WD-008', reference='E021', date='2026-11-23',
        paths={name: str(path) for name, path in vars(args).items()},
        channels=CHANNELS, windows_seconds=WINDOWS, bin_seconds=BIN_SECONDS,
        baseline_seconds_relative_to_first_trigger=BASELINE,
        preprocessing='62-channel instantaneous average reference; fixed bin means in microvolts; shared 200 ms baseline before the first pre-stimulus window; no temporal filtering or ICA',
        eligibility='WD-001 complete-pair QC plus feature-window support and neighboring-stimulus checks',
        classifier='Logistic regression, fixed C=1, training-only StandardScaler, no class weighting or feature selection',
        validation='One held-out participant per fold, one prediction per original behavioral pair',
        trial_metadata='20 fixed ordered-waveform-pair indicators, chronological pair index and within-block pair index',
        response_timing='Master CSV contains choices but no response timestamps. Feature windows are trigger-aligned; pre-response timing cannot be confirmed.',
        uncertainty='10,000 participant bootstrap resamples of fixed held-out scores. These descriptive intervals condition on the fitted LOSO models.',
        inference='Observed held-out performance and paired descriptive intervals; no permutation significance test was run.',
        dependencies={name: importlib.metadata.version(name)
                      for name in ('numpy', 'pandas', 'scipy', 'scikit-learn', 'pyxdf')},
        sources_sha256={str(path): digest(path) for path in Path(__file__).parent.glob('*.py')},
    )
    write_json(args.output / 'config.json', config)
    table, features, audit = assemble(args)
    table.to_csv(args.output / 'pair_manifest.csv', index=False)
    usable = table.eligible.to_numpy(bool)
    meta = table.loc[usable].reset_index(drop=True)
    matrices = {name: values[usable] for name, values in features.items()}
    write_json(args.output / 'status.json', dict(complete=False, stage='fitting', pairs=len(meta)))
    print(f'Fit eight fixed models on {len(meta)} pairs from {meta.subject.nunique()} participants', flush=True)
    with threadpool_limits(limits=1):
        predictions, fold_audit = predict(meta, matrices)
        nonterminal = ~meta.terminal_vote.to_numpy(bool)
        sensitivity, sensitivity_audit = predict(
            meta.loc[nonterminal].reset_index(drop=True),
            {name: values[nonterminal] for name, values in matrices.items()},
        )
    predictions.to_csv(args.output / 'predictions.csv', index=False)
    sensitivity.to_csv(args.output / 'nonterminal_predictions.csv', index=False)
    fold_audit.to_csv(args.output / 'fold_audit.csv', index=False)
    sensitivity_audit.to_csv(args.output / 'nonterminal_fold_audit.csv', index=False)
    full = summarize(predictions)
    changed = predictions.groupby(['subject', 'ordered_pair']).target_second.transform('nunique').gt(1)
    matched = predictions.matched_triangle_lindec
    summary = dict(
        experiment='WD-008', date=config['date'], model_labels=MODEL_LABELS,
        config=config, input_audit=audit, recorded_pairs=len(table),
        exclusions=table.loc[~table.eligible, 'exclusion_reason'].value_counts().to_dict(),
        full=full, nonterminal=summarize(sensitivity),
        varying_choice=summarize(predictions.loc[changed].copy()),
        matched_triangle_lindec=summarize(predictions.loc[matched].copy()),
        subset_policy='Varying-choice and charge-matched subsets summarize the full-cohort held-out predictions; they are not separate model fits. Nonterminal models are refitted on the reduced training data.',
        subsets=dict(
            varying_choice_pairs=int(changed.sum()), matched_triangle_lindec_pairs=int(matched.sum()),
            matched_triangle_lindec_subjects=sorted(predictions.loc[matched, 'subject'].unique().tolist()),
            matched_lindec_votes=int((
                ((predictions.first_trigger.eq(242) & predictions.target_second.eq(0)) |
                 (predictions.second_trigger.eq(242) & predictions.target_second.eq(1))) & matched
            ).sum()),
        ),
    )
    if args.unbaselined_reference.exists():
        reference = json.loads(args.unbaselined_reference.read_text())
        if not pd.read_csv(args.unbaselined_reference.parent / 'predictions.csv').pair_id.equals(
            predictions.pair_id
        ):
            raise ValueError('The unbaselined reference uses a different eligible pair sequence')
        summary['unbaselined_reference'] = dict(
            path=str(args.unbaselined_reference),
            sha256=digest(args.unbaselined_reference),
            full=reference['full'],
            reason='Large participant-specific DC offsets were identified after inspecting the initial run. A shared causal baseline was added as a documented preprocessing adaptation.',
        )
    full_block_shares = []
    for _, block in table.groupby(['subject', 'block_index']):
        chosen = np.where(block.target_second.eq(1), block.second_trigger, block.first_trigger)
        full_block_shares.append(float(pd.Series(chosen).value_counts().max() / len(block)))
    summary['behavioral_summary'] = dict(
        blocks=len(full_block_shares), mean_winner_vote_share=float(np.mean(full_block_shares)),
        second_choice_fraction=float(meta.target_second.mean()),
    )
    write_json(args.output / 'summary.json', summary)
    (args.output / 'RESULTS.md').write_text(render_markdown(summary), encoding='utf-8')
    (args.output / 'index.html').write_text(render_html(summary), encoding='utf-8')
    for path, expected in config['sources_sha256'].items():
        if digest(Path(path)) != expected:
            raise ValueError('Analysis code changed during the run')
    write_json(args.output / 'status.json', dict(complete=True, stage='complete', pairs=len(meta)))
    print('Primary log-loss reduction:', full['contrasts']['post_added_to_metadata_pre']['log_loss_reduction'], flush=True)
    print('Standalone post-minus-pre:', full['contrasts']['post_vs_pre_alone']['balanced_accuracy_gain'], flush=True)
    print('Results:', args.output.resolve(), flush=True)


if __name__ == '__main__':
    main()
