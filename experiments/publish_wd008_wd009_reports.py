"""Refresh linked WD-008/WD-009 reports and figures from saved predictions."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.response_characterization.e017_eeg_charge_relation.extract import digest
from experiments.waveform_decoding.e021_naturalness_choice_decoding import report as naturalness_report
from experiments.waveform_decoding.e021_naturalness_choice_decoding import position_report
from experiments.waveform_decoding.e022_triangle_lindec_decoding import report as waveform_report
from experiments.waveform_decoding.e022_triangle_lindec_decoding.evaluate import summarize
from experiments.waveform_decoding.e022_triangle_lindec_decoding.figures import wd008_figures, wd009_figures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wd008', type=Path, default=Path('output/wd008_naturalness_baselined_20261123'))
    parser.add_argument('--audit', type=Path, default=Path('output/wd008_behavior_audit_20261123'))
    parser.add_argument('--wd009', type=Path, default=Path('output/wd009_triangle_lindec_20261123'))
    parser.add_argument('--position-run', type=Path, default=Path('output/wd008_position_naturalness_20261124'),
                        help='Revised participant-specific naturalness run; use --legacy-only to omit it')
    parser.add_argument('--legacy-only', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('output/wd008_wd009_plots_20261123'))
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh report output directory')
    inputs = [
        args.wd008 / 'summary.json', args.audit / 'summary.json', args.audit / 'behavior.csv',
        args.wd009 / 'summary.json', args.wd009 / 'predictions.csv', args.wd009 / 'events.csv',
    ]
    if not args.legacy_only:
        if not json.loads((args.position_run/'status.json').read_text())['complete']:
            raise ValueError('The revised participant-specific run is incomplete')
        inputs += [args.position_run/name for name in ('summary.json','predictions.csv','fold_audit.json','config.json')]
    before = {str(path): digest(path) for path in inputs}
    naturalness = json.loads((args.wd008 / 'summary.json').read_text())
    audit = json.loads((args.audit / 'summary.json').read_text())
    behavior = pd.read_csv(args.audit / 'behavior.csv')
    naturalness['behavior_audit'] = audit
    waveform = json.loads((args.wd009 / 'summary.json').read_text())
    predictions = pd.read_csv(args.wd009 / 'predictions.csv')
    events = pd.read_csv(args.wd009 / 'events.csv')
    waveform.update(summarize(predictions))
    charges = []
    for subject, group in events.groupby('subject'):
        triangle = float(group.loc[group.trigger.eq(241), 'nominal_charge_uC'].iloc[0])
        lindec = float(group.loc[group.trigger.eq(242), 'nominal_charge_uC'].iloc[0])
        charges.append(dict(
            subject=int(subject), triangle_uC=triangle, lindec_uC=lindec,
            matched=bool(np.isclose(triangle, lindec, rtol=0, atol=1e-12)), epochs=len(group),
        ))
    waveform['charge_table'] = charges
    waveform['matched_all_epochs'] = sum(row['epochs'] for row in charges if row['matched'])
    waveform['evaluation_extension'] = dict(
        model_refitted=False,
        description='Added independent nominal-charge and comparison-context evaluation filters, including other-pair-only scores, from existing held-out predictions. Models were not refitted.',
    )
    args.output.mkdir(parents=True)
    image_folder = args.output / 'figures'
    print('Render behavioral audit and historical naturalness figures', flush=True)
    naturalness_figures = wd008_figures(naturalness, audit, behavior, image_folder)
    if not args.legacy_only:
        revised = json.loads((args.position_run/'summary.json').read_text())
        revision_figures = position_report.scientific_figures(revised, image_folder)
        naturalness['position_revision'] = position_report.public_summary(revised, revision_figures)
    print('Render waveform decoding, participant and coverage figures', flush=True)
    waveform_figures = wd009_figures(waveform, events, image_folder)
    for experiment, summary, figures, renderer, filename in (
        ('WD-008', naturalness, naturalness_figures, naturalness_report, 'WD008_NATURALNESS_CHOICE_RESULTS.md'),
        ('WD-009', waveform, waveform_figures, waveform_report, 'WD009_TRIANGLE_LINDEC_RESULTS.md'),
    ):
        output = args.output / experiment
        output.mkdir()
        html = renderer.render_html(summary, figures)
        markdown = renderer.render_markdown(summary)
        (output / 'index.html').write_text(html, encoding='utf-8')
        (output / 'RESULTS.md').write_text(markdown, encoding='utf-8')
        (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        publication = Path('Docs/Reports') / experiment
        publication.mkdir(parents=True, exist_ok=True)
        (publication / 'index.html').write_text(html, encoding='utf-8')
        package = 'e021_naturalness_choice_decoding' if experiment == 'WD-008' else 'e022_triangle_lindec_decoding'
        offset = markdown.index('\n\n')
        links = (f'\n\n[Interactive report](Reports/{experiment}/index.html) · '
                 f'[Run guide](../experiments/waveform_decoding/{package}/README.md)')
        Path('Docs', filename).write_text(markdown[:offset] + links + markdown[offset:], encoding='utf-8')
    for source, expected in before.items():
        if digest(Path(source)) != expected:
            raise ValueError('A saved numerical input changed during report generation')
    manifest = dict(
        inputs_sha256=before, model_refitted=False, matched_all_epochs=waveform['matched_all_epochs'],
        generator_sha256={str(path): digest(path) for path in [
            Path(__file__), Path(naturalness_report.__file__), Path(waveform_report.__file__),
            Path('experiments/waveform_decoding/e022_triangle_lindec_decoding/figures.py'),
            Path('experiments/waveform_decoding/e022_triangle_lindec_decoding/evaluate.py'),
            Path(position_report.__file__),
        ]},
        publication_sha256={f'Docs/Reports/{experiment}/index.html':
                            digest(Path(f'Docs/Reports/{experiment}/index.html'))
                            for experiment in ('WD-008', 'WD-009')},
    )
    (args.output / 'publication_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print('Matched all-context epochs:', waveform['matched_all_epochs'], flush=True)
    print('Published linked reports in Docs/Reports/WD-008 and Docs/Reports/WD-009', flush=True)


if __name__ == '__main__':
    main()
