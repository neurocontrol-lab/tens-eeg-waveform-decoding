"""Behavior-only order association and epoch counts for the WD-008 report."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

from experiments.response_characterization.e017_eeg_charge_relation.extract import digest
from experiments.waveform_decoding.e001_voltage_baseline.events import read_behavior


def order_test(table):
    """Condition on votes and order counts in each participant/comparison block."""
    distribution = np.array([1.])
    observed, expected = 0, 0.
    for _, group in table.groupby(['subject', 'block_index']):
        low_first = group.first_trigger.eq(group.low.iloc[0]).to_numpy()
        low_chosen = group.chosen_trigger.eq(group.low.iloc[0]).to_numpy()
        n, selected, first = len(group), int(low_chosen.sum()), int(low_first.sum())
        observed += int(np.sum(low_first & low_chosen))
        expected += first * selected / n
        distribution = np.convolve(
            distribution, hypergeom.pmf(np.arange(first + 1), n, selected, first),
        )
    support = np.arange(len(distribution))
    probability = distribution[np.abs(support - expected) >= abs(observed - expected) - 1e-10].sum()
    return dict(
        trials=len(table), participants=int(table.subject.nunique()),
        blocks=int(table.groupby(['subject', 'block_index']).ngroups),
        observed_low_first_and_chosen=observed, null_expected=float(expected),
        two_sided_p=float(min(1., probability)),
        direction='Second-position preference' if observed < expected else 'First-position preference',
        null_mass=float(distribution.sum()),
    )


def audit(master, baseline):
    tables = []
    for subject in range(3, 13):
        behavior = read_behavior(master, subject)
        table = pd.DataFrame(dict(
            subject=subject, csv_row=behavior.csv_row,
            first_trigger=behavior['t4 waveform 1'].astype(int) + 238,
            second_trigger=behavior['t4 waveform 2'].astype(int) + 238,
            response=pd.to_numeric(behavior['t4 naturalness value'], errors='coerce'),
        ))
        table['low'] = table[['first_trigger', 'second_trigger']].min(axis=1)
        table['high'] = table[['first_trigger', 'second_trigger']].max(axis=1)
        table['comparison'] = table.low.astype(str) + '-' + table.high.astype(str)
        table['block_index'] = table.comparison.ne(table.comparison.shift()).cumsum() - 1
        table['terminal'] = table.index.isin(table.groupby('block_index').tail(1).index)
        table = table.loc[table.response.isin([1, 2])].copy()
        table['chosen_trigger'] = np.where(table.response.eq(1), table.first_trigger, table.second_trigger)
        tables.append(table)
    behavior = pd.concat(tables, ignore_index=True)
    direct = behavior.loc[behavior.comparison.eq('241-242')].copy()
    orders = []
    for first, group in direct.groupby(direct.first_trigger.eq(242)):
        orders.append(dict(
            lindec_position=1 if first else 2,
            lindec_votes=int(group.chosen_trigger.eq(242).sum()), trials=len(group),
        ))
    events = pd.concat([
        pd.read_csv(baseline / f'sub-{subject}/events.csv') for subject in range(3, 13)
    ], ignore_index=True)
    selected = events.loc[events.analysis_eligible & events.trigger.isin([241, 242])]
    epoch_counts = [
        dict(subject=int(subject), trigger=int(trigger), position=int(position), epochs=len(group))
        for (subject, trigger, position), group in selected.groupby(['subject', 'trigger', 'position'])
    ]
    result = dict(
        date='2026-11-23', primary=order_test(behavior), triangle_lindec=order_test(direct),
        triangle_lindec_nonterminal=order_test(direct.loc[~direct.terminal]),
        direct_order_counts=orders, first_chosen=int(behavior.response.eq(1).sum()),
        second_chosen=int(behavior.response.eq(2).sum()),
        triangle_lindec_votes=int(direct.chosen_trigger.eq(242).sum()),
        triangle_votes=int(direct.chosen_trigger.eq(241).sum()),
        epoch_counts=epoch_counts, all_context_epochs=len(selected),
        direct_epochs=int(selected.comparison.eq('241-242').sum()),
        inputs_sha256={str(path): digest(path) for path in [
            master, *[baseline / f'sub-{subject}/events.csv' for subject in range(3, 13)]
        ]},
        implementation_sha256=digest(Path(__file__)),
        method='Exact conditional randomization test. Hold waveform choices and observed order counts fixed within each participant/comparison block. Convolve the block hypergeometric distributions and use the absolute deviation from the conditional expectation.',
        assumptions='Presentation order is exchangeable within each recorded block under the null of no order effect. This follows the documented randomized-order protocol; acquisition randomization code is unavailable.',
        interpretation='Order association supports context-dependent choices. It does not establish recording errors, physiological carryover, or unusable second-presentation EEG. Direct-pair and terminal-vote analyses are exploratory sensitivity checks.',
        decision='Further naturalness decoding is deferred. Existing fitted WD-008 results are retained as exploratory results.',
    )
    return result, behavior


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--master-csv', type=Path,
                        default=Path('EEG_Codes_Export/02_SourceLocalization_Task4/Master_Data_ToSub12 (1).csv'))
    parser.add_argument('--baseline', type=Path, default=Path('output/task4_baseline'))
    parser.add_argument('--output', type=Path, default=Path('output/wd008_behavior_audit_20261123'))
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh audit output directory')
    result, behavior = audit(args.master_csv, args.baseline)
    args.output.mkdir(parents=True)
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    behavior.to_csv(args.output / 'behavior.csv', index=False)
    pd.DataFrame(result['epoch_counts']).to_csv(args.output / 'epoch_counts.csv', index=False)
    print(json.dumps({key: result[key] for key in ('primary', 'triangle_lindec', 'triangle_lindec_nonterminal')}, indent=2))


if __name__ == '__main__':
    main()
