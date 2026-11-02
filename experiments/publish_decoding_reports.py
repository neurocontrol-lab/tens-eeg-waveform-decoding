"""Publish compact E015/E016 HTML snapshots with embedded summary assets."""
import argparse
import base64
import hashlib
import json
import re
from pathlib import Path

REPO = 'https://github.com/neurocontrol-lab/tens-eeg-waveform-decoding/blob/main/'
PREVIEW = 'https://htmlpreview.github.io/?' + REPO
MIMES = {'.png': 'image/png', '.csv': 'text/csv', '.json': 'application/json'}
SOURCES = {'E015': 'output/e015_observed_summary_20261102',
           'E016': 'output/e016_binary_summary_20261102'}


def standalone(source, report_id):
    """Embed only explicitly linked figures and summary exports, never raw epochs."""
    source = Path(source).resolve()
    page = (source / 'index.html').read_text(encoding='utf-8')
    manifest = {}

    def embed(match):
        attr, relative = match.groups()
        if relative.startswith(('https://', 'http://', '#', 'data:')):
            return match.group(0)
        file = (source / relative).resolve()
        if not file.is_relative_to(source) or file.suffix not in MIMES:
            raise ValueError(f'Unexpected report asset: {relative}')
        data = file.read_bytes()
        manifest[relative] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        value = 'data:' + MIMES[file.suffix] + ';base64,' + base64.b64encode(data).decode('ascii')
        download = f' download="{file.name}"' if attr == 'href' else ''
        return f'{attr}="{value}"{download}'

    page = re.sub(r'(src|href)="([^"]+)"', embed, page)
    other = 'E016' if report_id == 'E015' else 'E015'
    banner = (f'<nav aria-label="Related reports"><a href="{PREVIEW}Docs/Reports/{other}/index.html">'
              f'{other} report</a> · <a href="{REPO}Docs/E015_E016_SUBGROUP_FINDINGS.md">'
              'Waveform and participant comparison</a></nav>'
              '<p><small>Snapshot: 2 November 2026. This standalone report embeds rendered plots '
              'and summary statistics. Raw EEG, trial-level prediction files and fitted models '
              'are excluded. Observed estimates only; permutation inference pending.</small></p>')
    page = page.replace('<h1>', banner + '<h1>', 1)
    if len(page.encode('utf-8')) >= 1_000_000:
        raise ValueError('Publication exceeds the one-megabyte page budget')
    return page, manifest


def run(args):
    publication = {}
    for report_id, folder in SOURCES.items():
        source = Path(folder)
        verification = json.loads((source / 'verification.json').read_text())
        if report_id == 'E015' and not verification['all_saved_scores_recomputed_from_held_out_predictions']:
            raise ValueError('E015 report has not passed verification')
        if report_id == 'E016' and not verification['complete']:
            raise ValueError('E016 report has not passed verification')
        page, assets = standalone(source, report_id)
        target = args.output / report_id
        target.mkdir(parents=True, exist_ok=True)
        (target / 'index.html').write_text(page, encoding='utf-8', newline='\n')
        publication[report_id] = dict(source=str(source), source_html_sha256=hashlib.sha256((source / 'index.html').read_bytes()).hexdigest(),
            publication_sha256=hashlib.sha256((target / 'index.html').read_bytes()).hexdigest(),
            publication_bytes=(target / 'index.html').stat().st_size, embedded_assets=assets,
            preview=PREVIEW + f'Docs/Reports/{report_id}/index.html', raw_eeg_included=False,
            trial_predictions_included=False, fitted_models_included=False)
    (args.output / 'decoding_publication_manifest.json').write_text(json.dumps(publication, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: dict(bytes=value['publication_bytes'], preview=value['preview']) for key, value in publication.items()}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('Docs/Reports'))
    run(parser.parse_args())
