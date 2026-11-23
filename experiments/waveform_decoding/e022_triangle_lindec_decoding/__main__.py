"""Run WD-009 triangle versus linear-decrease waveform identity decoding."""

import argparse
import importlib.metadata
import json
from pathlib import Path

from threadpoolctl import threadpool_limits

from experiments.response_characterization.e017_eeg_charge_relation.extract import digest
from .data import BASELINE, WINDOWS, assemble
from .evaluate import MODELS, predict, summarize
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
    parser.add_argument('--cache', type=Path, default=Path('output/wd009_feature_cache_20261123'))
    parser.add_argument('--output', type=Path, default=Path('output/wd009_triangle_lindec_20261123'))
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output directory; verified feature extraction can be reused')
    args.output.mkdir(parents=True)
    write_json(args.output / 'status.json', dict(complete=False, stage='extracting'))
    config = dict(
        experiment='WD-009', reference='E022', date='2026-11-23',
        paths={name: str(path) for name, path in vars(args).items()},
        target='Delivered waveform: Triangle (trigger 241, target 0), LinDec (trigger 242, target 1)',
        windows_seconds=WINDOWS, bin_widths_ms=[50, 100], channels=62,
        pair_baseline_seconds_relative_to_first_trigger=BASELINE,
        baseline_policy='A common 200 ms baseline before the first pulse pre-control window is subtracted from all epochs in that pair.',
        preprocessing='Instantaneous 62-channel average reference, microvolt bin means, no temporal filtering or ICA',
        validation='Independent participant x presentation models; hold out each entire waveform-comparison group in turn; no shared pair IDs across folds',
        classifier_settings='PCA retains 90% of training variance; shrinkage LDA has equal priors; linear SVM uses fixed C=1 and balanced class weights. Scaling and PCA fit training data only. Linear SVM without PCA is a reference.',
        subsets='All triangle/LinDec epochs across comparison contexts, direct-pair epochs, and nominal-charge-matched direct-pair epochs. Subsets reuse held-out predictions. Every direct block is absent from its training fold.',
        limitations='First and second positions are evaluated separately. Second-position pre EEG can contain the first stimulus response. Raw units are verified, while current-monitor and response timestamps are unavailable. No permutation inference or corrected model selection is performed.',
        uncertainty='10,000 descriptive participant bootstrap resamples of fixed held-out predictions',
        sources_sha256={str(path): digest(path) for path in Path(__file__).parent.glob('*.py')},
        dependencies={name: importlib.metadata.version(name)
                      for name in ('numpy', 'pandas', 'scipy', 'scikit-learn', 'pyxdf')},
    )
    write_json(args.output / 'config.json', config)
    events, features, audit = assemble(args)
    events.to_csv(args.output / 'events.csv', index=False)
    if len(events) != 501 or events.loc[events.matched_direct].shape[0] != 66:
        raise ValueError('The verified analysis cohort differs from the audited counts')
    write_json(args.output / 'status.json', dict(complete=False, stage='fitting', epochs=len(events)))
    with threadpool_limits(limits=1):
        predictions, folds = predict(events, features)
        results = summarize(predictions)
    predictions.to_csv(args.output / 'predictions.csv', index=False)
    folds.to_csv(args.output / 'fold_audit.csv', index=False)
    summary = dict(
        experiment='WD-009', config=config, models=MODELS, input_audit=audit,
        epochs=len(events), participants=int(events.subject.nunique()),
        direct_epochs=int(events.comparison.eq('241-242').sum()),
        matched_epochs=int(events.matched_direct.sum()), **results,
    )
    write_json(args.output / 'summary.json', summary)
    (args.output / 'RESULTS.md').write_text(render_markdown(summary), encoding='utf-8')
    (args.output / 'index.html').write_text(render_html(summary), encoding='utf-8')
    for source, expected in config['sources_sha256'].items():
        if digest(Path(source)) != expected:
            raise ValueError('Source code changed during the run')
    write_json(args.output / 'status.json', dict(complete=True, epochs=len(events)))
    for row in summary['aggregate']:
        if row['position'] == 1:
            print(row['subset'], row['width_ms'], row['model'],
                  'pre', round(row['pre']['mean'], 4), 'post', round(row['post']['mean'], 4), flush=True)
    print('Results:', args.output.resolve(), flush=True)


if __name__ == '__main__':
    main()
