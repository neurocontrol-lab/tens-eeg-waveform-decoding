"""Task 4 visual inspection: baseline-matched EEG and a simple band-pass view."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.waveform_decoding.e001_voltage_baseline.events import load_recording, trigger_runs, align_events, read_behavior, CODES
from experiments.waveform_decoding.e001_voltage_baseline.features import CHANNELS
from experiments.waveform_decoding.e004_xdawn_decoding.data import sha, write_json
from .core import filter_continuous, extract_epochs, epoch_eligibility, cell_balanced_evokeds, EPOCH, BASELINE, FILTER_EDGE_SECONDS


def prepare(subject, args, config, original):
    source = args.data_dir/f'Subject{subject}_data/EEGDATA_SUB{subject}-T4.xdf'
    baseline = args.baseline/f'sub-{subject}'
    # Require a previously verified raw-recording hash, without reusing any ICA.
    caches = list(args.verified_cache.glob(f'sub-{subject}-*/provenance.json'))
    matches = []
    for cache in caches:
        provenance = json.loads(cache.read_text())
        if ((cache.parent/'complete.json').exists()
                and provenance['baseline_events_sha256'] == sha(baseline/'events.csv')
                and provenance['baseline_features_sha256'] == sha(baseline/'features.npz')
                and provenance['baseline_config_sha256'] == sha(args.baseline/'run_config.json')
                and provenance['master_csv_sha256'] == config['master_csv_sha256']):
            matches.append(provenance)
    if not matches or len({p['source_sha256'] for p in matches}) != 1:
        raise ValueError('No unambiguous baseline-verified recording provenance')
    source_hash = sha(source)
    if source_hash != matches[0]['source_sha256']:
        raise ValueError('Recording differs from prior verified baseline input')
    out = args.output/f'sub-{subject}'
    out.mkdir()
    print(f'Subject {subject}: verify raw XDF and reconstruct simple views', flush=True)
    data, names, triggers, timestamps, sfreq = load_recording(source)
    events = align_events(trigger_runs(triggers, sfreq, original['min_trigger_samples']),
        read_behavior(args.master_csv, subject), subject, timestamps)
    saved = pd.read_csv(baseline/'events.csv')
    columns = ['sample', 'trigger', 'subject', 'event_index', 'pair_id', 'group', 'position']
    pd.testing.assert_frame_equal(events[columns], saved[columns], check_dtype=False)
    picks = [names.index(ch) for ch in CHANNELS]
    referenced = data[picks]-data.mean(axis=0, keepdims=True)
    del data, triggers
    filtered, edge_ok, sos, segments = filter_continuous(referenced, timestamps, sfreq)
    manifest = epoch_eligibility(saved, timestamps, edge_ok, sfreq)
    manifest.to_csv(out/'events.csv', index=False)
    meta = manifest.loc[manifest.visual_pair_eligible].reset_index(drop=True)
    if meta.empty or not meta.groupby('pair_id').size().eq(2).all():
        raise ValueError('No complete eligible display pairs')
    meta.to_csv(out/'eligible_events.csv', index=False)
    summaries, count_rows = [], []
    epoch_arrays = {}
    for branch, continuous in [('minimal', referenced), ('bandpass', filtered)]:
        epochs, time = extract_epochs(continuous, meta['sample'].to_numpy(), sfreq)
        np.save(out/f'{branch}_epochs_uv.npy', epochs)
        epoch_arrays[branch] = epochs
        means, counts, cells = cell_balanced_evokeds(epochs, meta, CODES)
        summaries.append(means)
        if branch == 'minimal':
            # Exact reconstruction of the original voltage features on common events.
            vectors = np.concatenate([epochs[..., (time >= k*.05+.05-1e-9) & (time < k*.05+.10-1e-9)].mean(axis=-1)
                                      for k in range(9)], axis=1)
            eligible_ids = saved.loc[saved.analysis_eligible, 'event_index'].to_numpy()
            indices = pd.Index(eligible_ids).get_indexer(meta.event_index)
            if (indices < 0).any():
                raise ValueError('Display cohort is not a subset of classifier cohort')
            with np.load(baseline/'features.npz') as stored:
                np.testing.assert_allclose(vectors, stored['post'][indices], rtol=0, atol=1e-9)
                feature_error = float(np.max(np.abs(vectors-stored['post'][indices])))
        for p in range(2):
            for w, code in enumerate(CODES):
                count_rows.append(dict(subject=subject, branch=branch, position=p+1, trigger=code,
                    n_events=int(counts[p, w]), n_comparison_cells=int(cells[p, w])))
    np.savez_compressed(out/'summaries.npz', means=np.stack(summaries), time_s=time,
        sos=sos, sfreq=sfreq, channels=np.array(CHANNELS), triggers=np.array(CODES))
    pd.DataFrame(count_rows).to_csv(out/'support.csv', index=False)
    inspect_time = (time >= -.25) & ~((time >= 0) & (time < .05))
    central = [CHANNELS.index(c) for c in ['C3', 'C4']]
    limit = max(1., float(np.quantile(np.abs(np.concatenate([
        x[:, central][:, :, inspect_time].ravel() for x in epoch_arrays.values()])), .99)))
    clipping = {branch:float(np.mean(np.abs(x[:, central][:, :, inspect_time]) > limit))
                for branch, x in epoch_arrays.items()}
    qc = []
    for branch, x in epoch_arrays.items():
        for c, name in enumerate(CHANNELS):
            ptp = np.ptp(x[:, c, (time >= .05)], axis=-1)
            qc.append(dict(subject=subject, branch=branch, channel=name,
                median_post_ptp_uv=float(np.median(ptp)), max_post_ptp_uv=float(ptp.max())))
    pd.DataFrame(qc).to_csv(out/'amplitude_audit.csv', index=False)
    audit = dict(subject=subject, n_aligned_events=len(saved), n_baseline_events=int(saved.analysis_eligible.sum()),
        n_visual_events=len(meta), n_visual_pairs=meta.pair_id.nunique(),
        exclusion_counts=manifest.visual_exclusion_reason.value_counts().to_dict(),
        n_segments=len(segments), segments=segments, trial_image_limit_uv=limit,
        trial_image_clipped_fraction=clipping, baseline_feature_max_abs_error_uv=feature_error,
        next_gap_min_s=float(meta.next_gap_s.min()), sfreq=sfreq,
        source_sha256=source_hash, baseline_events_sha256=sha(baseline/'events.csv'),
        baseline_features_sha256=sha(baseline/'features.npz'))
    write_json(out/'audit.json', audit)
    print(f'  {len(meta)} events / {meta.pair_id.nunique()} pairs; baseline feature error {feature_error:.3g} uV', flush=True)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('EEGData'))
    parser.add_argument('--master-csv', type=Path, default=Path('EEG_Codes_Export/02_SourceLocalization_Task4/Master_Data_ToSub12 (1).csv'))
    parser.add_argument('--baseline', type=Path, default=Path('output/task4_baseline'))
    parser.add_argument('--verified-cache', type=Path, default=Path('output/task4_xdawn_epoch_cache'))
    parser.add_argument('--subjects', default=','.join(map(str, range(3, 13))))
    parser.add_argument('--output', type=Path, default=Path('output/task4_visual_inspection_20261030'))
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    subjects = [int(s) for s in args.subjects.split(',')]
    if not subjects or len(set(subjects)) != len(subjects) or not set(subjects) <= set(range(3, 13)):
        parser.error('Choose unique subjects 3-12')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Use a fresh output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    original = json.loads((args.baseline/'run_config.json').read_text())
    files = [*Path(__file__).parent.glob('*.py'), Path('experiments/waveform_decoding/e001_voltage_baseline/events.py'),
        Path('experiments/waveform_decoding/e001_voltage_baseline/features.py'), args.master_csv, args.baseline/'run_config.json']
    initial_hashes = {str(p):sha(p) for p in files}
    config = dict(subjects=subjects, channels=CHANNELS, epoch=EPOCH, baseline=BASELINE,
        filter=dict(type='Butterworth SOS forward/backward', prototype_order=4,
                    bandpass_poles_per_pass=8, cutoff_hz=[.5,30.], segment_edge_guard_s=FILTER_EDGE_SECONDS),
        master_csv_sha256=sha(args.master_csv), initial_sha256=initial_hashes,
        dependencies={n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','matplotlib','pyxdf']},
        notes='Descriptive plots only; no ICA, notch, interpolation, resampling or classifier. Frozen baseline QC plus common extended support.')
    if config['master_csv_sha256'] != original['master_csv_sha256']:
        raise ValueError('Master CSV differs from baseline')
    write_json(args.output/'run_config.json', config)
    status = dict(completed_subjects=[], errors={}, complete=False)
    for subject in subjects:
        try:
            prepare(subject, args, config, original)
            status['completed_subjects'].append(subject)
        except Exception as exc:
            status['errors'][str(subject)] = f'{type(exc).__name__}: {exc}'
            print(status['errors'][str(subject)], flush=True)
        write_json(args.output/'status.json', status)
    changed = [p for p, digest in initial_hashes.items() if sha(Path(p)) != digest]
    if changed:
        status['errors']['provenance'] = f'Inputs/source changed during run: {changed}'
    status['complete'] = not status['errors'] and status['completed_subjects'] == subjects
    write_json(args.output/'status.json', status)
    if not args.prepare_only and status['completed_subjects']:
        from .report import render
        render(args.output)
    print(f'Visual report: {args.output.resolve()}', flush=True)
    if status['errors']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
