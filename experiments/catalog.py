"""Refresh human-readable experiment statuses from registry.json."""
from collections import Counter
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def display_id(entry, category_prefixes):
    return f'{category_prefixes[entry["category"]]}-{entry["category_number"]:03d}'


def refresh(root=ROOT):
    root = Path(root)
    registry = json.loads((root/'experiments/registry.json').read_text(encoding='utf-8'))
    entries = registry['experiments']
    prefixes = registry['category_prefixes']
    registry_page = root/'experiments/README.md'
    text = registry_page.read_text(encoding='utf-8')
    counts = Counter(e['status'] for e in entries)
    summary = f'**Updated: {registry["updated"]}.** '+', '.join(f'{n} {s.lower()}' for s,n in counts.items())+'. Completed means the specified analysis finished; it does not imply a positive scientific result.'
    text = re.sub(r'\*\*Updated:.*?(?=\n\n)',summary,text,count=1,flags=re.S)
    sections = []
    for category, prefix in prefixes.items():
        category_page = root/'experiments'/category/'README.md'
        category_text = category_page.read_text(encoding='utf-8')
        heading, description, _ = category_text.split('\n\n', 2)
        heading = heading.removeprefix('# ')
        central_rows = [
            '| ID | Experiment | Dataset | Status | Last completed run | Findings | Reference |',
            '| --- | --- | --- | --- | --- | --- | --- |',
        ]
        category_rows = [
            '| ID | Experiment | Dataset | Status |',
            '| --- | --- | --- | --- |',
        ]
        for entry in sorted(
            (e for e in entries if e['category'] == category),
            key=lambda e: e['category_number'],
        ):
            experiment_id = display_id(entry, prefixes)
            central_title = category_title = entry['title']
            if entry['package']:
                package_path = entry['package'].removeprefix('experiments/')
                central_title = f'[{central_title}]({package_path}/README.md)'
                category_title = f'[{category_title}]({Path(package_path).name}/README.md)'
            elif entry.get('proposal'):
                central_title = f'[{central_title}](../{entry["proposal"]})'
                category_title = f'[{category_title}](../../{entry["proposal"]})'
            dataset, status = entry['dataset'], entry['status']
            last_run = entry['last_completed_run'] or '-'
            findings = f'[Results](../{entry["findings"]})' if entry['findings'] else '-'
            central_rows.append(
                f'| {experiment_id} | {central_title} | {dataset} | {status} | '
                f'{last_run} | {findings} | {entry["id"]} |'
            )
            category_rows.append(f'| {experiment_id} | {category_title} | {dataset} | {status} |')

        sections.append('\n\n'.join([
            f'### [{prefix} — {heading}]({category}/README.md)',
            description,
            '\n'.join(central_rows),
        ]))
        category_start = category_text.index('| ID |')
        category_page.write_text(
            category_text[:category_start]+'\n'.join(category_rows)+'\n',
            encoding='utf-8', newline='\n',
        )

    status_heading = '## Status of completed and planned experiments\n\n'
    start = text.index(status_heading)+len(status_heading)
    stop = text.index('\n## ',start)
    registry_page.write_text(
        text[:start]+'\n\n'.join(sections)+'\n'+text[stop:],
        encoding='utf-8', newline='\n',
    )


if __name__ == '__main__':
    refresh()
