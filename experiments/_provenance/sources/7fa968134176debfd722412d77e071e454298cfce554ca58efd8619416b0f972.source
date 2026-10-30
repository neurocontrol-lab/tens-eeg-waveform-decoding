"""Regenerate diagnostic reports from completed, local participant summaries."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

from .analysis import group_tests
from .report import write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args = parser.parse_args()
    status = json.loads((args.directory/'status.json').read_text())
    results = []
    for subject in status['completed']:
        folder = args.directory/f'sub-{subject}'
        with np.load(folder/'responses.npz') as saved:
            results.append(dict(subject=subject,channels=saved['channels'].tolist(),
                times=saved['times'],voltage=saved['voltage'],slope=saved['slope'],
                condition_features=saved['condition_features'],
                effects={name:saved[name] for name in ['position','waveform','interaction']},
                audit=json.loads((folder/'preprocessing.json').read_text()),
                repeatability=pd.read_csv(folder/'repeatability.csv')))
    if not results:
        raise ValueError('No completed participant summaries')
    channels,times = results[0]['channels'],results[0]['times']
    if any(r['channels'] != channels or not np.array_equal(r['times'],times) for r in results):
        raise ValueError('Different channel or time axes; do not silently pool')
    tests = group_tests([r['effects'] for r in results],channels) if len(results) >= 3 else pd.DataFrame()
    tests.to_csv(args.directory/'effect_tests.csv',index=False)
    write_report(args.directory,results,tests,status['errors'])
    # Rendering can change without rerunning numerical preprocessing/feature extraction.
    provenance = {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in [Path(__file__),Path(__file__).with_name('report.py'),
                            Path(__file__).with_name('analysis.py'),
                            Path(__file__).with_name('electrode_repeatability.py')]}
    (args.directory/'render_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(f'Report regenerated: {(args.directory/"index.html").resolve()}')


if __name__ == '__main__':
    main()
