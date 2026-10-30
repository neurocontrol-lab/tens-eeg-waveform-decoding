"""Refresh human-readable experiment statuses from registry.json."""
from collections import Counter
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def refresh(root=ROOT):
    root = Path(root)
    registry = json.loads((root/'experiments/registry.json').read_text())
    entries = registry['experiments']
    page = root/'experiments/README.md'
    text = page.read_text()
    counts = Counter(e['status'] for e in entries)
    summary = f'**Updated: {registry["updated"]}.** '+', '.join(f'{n} {s.lower()}' for s,n in counts.items())+'. Completed means the specified analysis finished; it does not imply a positive scientific result.'
    text = re.sub(r'\*\*Updated:.*?(?=\n\n)',summary,text,count=1,flags=re.S)
    header = '| ID | Experiment | Category | Dataset | Status | Last completed run | Findings |'
    rows = [header,'| --- | --- | --- | --- | --- | --- | --- |']
    for e in entries:
        title = f'[{e["title"]}]({e["package"].removeprefix("experiments/")}/README.md)' if e['package'] else e['title']
        findings = f'[Results](../{e["findings"]})' if e['findings'] else '-'
        rows.append(f'| {e["id"]} | {title} | {e["category"].replace("_"," ")} | {e["dataset"]} | {e["status"]} | {e["last_completed_run"] or "-"} | {findings} |')
    start = text.index(header)
    stop = text.index('\n\n',start)
    text = text[:start]+'\n'.join(rows)+text[stop:]
    for e in entries:
        text = re.sub(rf'- \*\*{e["id"]}\*\* - .*',f'- **{e["id"]}** - {e["title"]} ({e["status"].lower()}).',text)
    page.write_text(text,encoding='utf-8',newline='\n')
    for category in {e['category'] for e in entries}:
        page = root/'experiments'/category/'README.md'
        text = page.read_text()
        start = text.index('| ID |')
        rows = ['| ID | Experiment | Dataset | Status |','| --- | --- | --- | --- |']
        for e in entries:
            if e['category'] != category:
                continue
            title = f'[{e["title"]}]({Path(e["package"]).name}/README.md)' if e['package'] else e['title']
            rows.append(f'| {e["id"]} | {title} | {e["dataset"]} | {e["status"]} |')
        page.write_text(text[:start]+'\n'.join(rows)+'\n',encoding='utf-8',newline='\n')


if __name__ == '__main__':
    refresh()
