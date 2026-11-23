"""All-electrode naturalness epochs; preserve the archived pair cohort and alignment."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.response_characterization.e017_eeg_charge_relation.extract import digest
from experiments.waveform_decoding.e001_voltage_baseline.events import (
    align_events, load_recording, read_behavior, trigger_runs,
)
from experiments.waveform_decoding.e001_voltage_baseline.features import interval_samples
from experiments.waveform_decoding.e022_triangle_lindec_decoding.data import (
    BASELINE, WINDOWS, bin_means,
)


def assemble(args):
    """One epoch per presentation of each previously eligible behavioral pair."""
    prior_path = args.previous / 'pair_manifest.csv'
    prior = pd.read_csv(prior_path)
    config_path = args.baseline / 'run_config.json'
    config = json.loads(config_path.read_text())
    if digest(args.master_csv) != config['master_csv_sha256']:
        raise ValueError('Master CSV differs from the verified alignment')
    tables, matrices, audits = [], [], {}
    for subject in range(3, 13):
        folder = args.cache / f'sub-{subject}'
        source = args.data_dir / f'Subject{subject}_data/EEGDATA_SUB{subject}-T4.xdf'
        old_path = args.baseline / f'sub-{subject}/events.csv'
        visual = args.visual / f'sub-{subject}/audit.json'
        inputs = {str(p): digest(p) for p in (source, old_path, visual, args.master_csv, prior_path, config_path)}
        if inputs[str(source)] != json.loads(visual.read_text())['source_sha256']:
            raise ValueError('Raw EEG differs from the verified recording')
        signature = dict(inputs_sha256=inputs, windows=WINDOWS, baseline=BASELINE,
                         implementation_sha256=digest(Path(__file__)),
                         bin_implementation_sha256=digest(Path(bin_means.__code__.co_filename)))
        signature = json.loads(json.dumps(signature))
        if (folder / 'complete.json').exists():
            audit = json.loads((folder / 'complete.json').read_text())
            if audit['signature'] != signature:
                raise ValueError('Position-feature cache provenance differs')
            for name, expected in audit['outputs_sha256'].items():
                if digest(folder / name) != expected:
                    raise ValueError('Position-feature cache changed')
            table = pd.read_csv(folder / 'events.csv')
            with np.load(folder / 'features.npz') as values:
                features = {name: values[name].copy() for name in values.files}
        else:
            print(f'Participant {subject}: extract both positions, all waveforms and 62 electrodes', flush=True)
            data, names, triggers, timestamps, sfreq = load_recording(source)
            if len(names) != 62 or sfreq != 1000:
                raise ValueError('Expected 62 scalp electrodes at 1000 Hz')
            old = pd.read_csv(old_path)
            aligned = align_events(trigger_runs(triggers, sfreq, config['min_trigger_samples']),
                                   read_behavior(args.master_csv, subject), subject, timestamps)
            fields = ['event_index', 'sample', 'trigger', 'pair_id', 'position', 'csv_row',
                      'response_value', 'comparison']
            pd.testing.assert_frame_equal(old[fields], aligned[fields], check_dtype=False)
            cohort = prior.loc[prior.subject.eq(subject) & prior.eligible].copy()
            table = old.loc[old.pair_id.isin(cohort.pair_id)].copy()
            if len(table) != 2 * len(cohort) or not table.analysis_eligible.all():
                raise ValueError('Position cohort differs from the archived complete-pair QC')
            table = table.merge(cohort[['pair_id', 'within_block_index', 'terminal_vote']],
                                on='pair_id', validate='many_to_one', sort=False)
            firsts = table.loc[table.position.eq(1)].set_index('pair_id')
            seconds = table.loc[table.position.eq(2)].set_index('pair_id')
            table['opponent_trigger'] = [
                int((seconds if row.position == 1 else firsts).loc[row.pair_id, 'trigger'])
                for row in table.itertuples()
            ]
            table['target_chosen'] = table.response_value.eq(table.position).astype(int)
            data -= data.mean(axis=0, keepdims=True)
            arrays = {name: [] for name in WINDOWS}
            for row in table.itertuples():
                first = firsts.loc[row.pair_id]
                a, b = interval_samples(int(first['sample']), *BASELINE, sfreq)
                if a < 0 or b > len(timestamps) or not np.isfinite(data[:, a:b]).all():
                    raise ValueError('Unsupported common pair baseline')
                if np.any(np.abs(np.diff(timestamps[a:b]) - 1 / sfreq) > .1 / sfreq):
                    raise ValueError('Timestamp gap in common baseline')
                if np.isfinite(first.previous_gap_s) and first.previous_gap_s <= -BASELINE[0]:
                    raise ValueError('Prior stimulus falls inside the common pair baseline')
                if np.isfinite(row.previous_gap_s) and row.previous_gap_s <= 1.05:
                    raise ValueError('Prior trigger falls inside the pre feature period')
                if np.isfinite(row.next_gap_s) and row.next_gap_s <= 1.05:
                    raise ValueError('Next trigger falls inside the post feature period')
                baseline = data[:, a:b].mean(axis=1) * 1e6
                for period, window in WINDOWS.items():
                    arrays[period].append(bin_means(data, row.sample, window, timestamps, sfreq) - baseline)
            features = {name: np.stack(values) for name, values in arrays.items()}
            table = table.reset_index(drop=True)
            folder.mkdir(parents=True, exist_ok=True)
            table.to_csv(folder / 'events.csv', index=False)
            np.savez_compressed(folder / 'features.npz', **features)
            audit = dict(signature=signature, channels=names, epochs=len(table), pairs=len(cohort),
                         outputs_sha256={name: digest(folder / name) for name in ('events.csv', 'features.npz')})
            (folder / 'complete.json').write_text(json.dumps(audit, indent=2)+'\n', encoding='utf-8')
            del data, triggers
        tables.append(table)
        matrices.append(features)
        audits[str(subject)] = audit
        print(f'  {len(table)} epochs / {table.pair_id.nunique()} behavioral pairs', flush=True)
    return pd.concat(tables, ignore_index=True), {
        period: np.concatenate([values[period] for values in matrices]) for period in WINDOWS
    }, audits
