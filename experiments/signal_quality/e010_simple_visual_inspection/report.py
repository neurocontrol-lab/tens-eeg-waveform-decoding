"""Re-render saved aggregates and chronological single-trial images."""
import html
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import sosfreqz
from experiments.waveform_decoding.e001_voltage_baseline.events import CODES, WAVEFORMS
from experiments.waveform_decoding.e001_voltage_baseline.features import CHANNELS

COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#8B6508']
BRANCHES = ('minimal', 'bandpass')


def save(fig, path):
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def traces(means, time, path, title, deviations=False, band=None):
    fig, axes = plt.subplots(7, 4, figsize=(18, 22), sharex=True)
    mask = (time >= 0) & (time < .05)
    shown = means.copy()
    if deviations:
        shown -= shown.mean(axis=1, keepdims=True)
    shown[..., mask] = np.nan
    for c, channel in enumerate(CHANNELS):
        row, start = c//2, 2*(c % 2)
        limit = max(1., 1.08*float(np.nanmax(np.abs(shown[:, :, c]))))
        if band is not None:
            limit = max(limit, 1.08*float(np.nanmax(np.abs(band[:, :, :, c, ~mask]))))
        for p in range(2):
            ax = axes[row, start+p]
            for w, (code, color) in enumerate(zip(CODES, COLORS)):
                ax.plot(time*1000, shown[p, w, c], color=color, lw=1., label=WAVEFORMS[code])
                if band is not None:
                    lo, hi = band[:, p, w, c].copy()
                    lo[mask] = hi[mask] = np.nan
                    ax.fill_between(time*1000, lo, hi, color=color, alpha=.09, lw=0)
            ax.axvspan(0, 50, color='.85')
            ax.axvspan(-250, -50, color='#dbeafe', alpha=.3)
            ax.axhline(0, color='.6', lw=.5)
            ax.set(title=f'{channel}: presentation {p+1}', ylim=(-limit, limit), xlim=(-250, 1200))
            ax.set_ylabel('uV')
            ax.grid(alpha=.12)
            if row == 6:
                ax.set_xlabel('Time from recorded trigger (ms)')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=5, bbox_to_anchor=(.5, .965))
    subtitle = 'Waveform deviations from the equally weighted five-waveform mean.' if deviations else 'Equal observed-comparison-cell means within waveform/position.'
    if band is not None:
        subtitle += ' Shading: pointwise descriptive 95% participant-bootstrap intervals.'
    fig.suptitle(title+'\n'+subtitle+'\n0-50 ms omitted here; see onset panels. This omission does not guarantee artifact-free later activity.', fontsize=12, y=.997)
    fig.tight_layout(rect=(0, 0, 1, .95))
    save(fig, path)


def onset_figure(means, time, path, title):
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True)
    mask = (time >= -.02) & (time <= .05)
    for b, branch in enumerate(BRANCHES):
        for ci, channel in enumerate(('C3', 'C4')):
            c = CHANNELS.index(channel)
            for p in range(2):
                ax = axes[b, ci*2+p]
                for w, code in enumerate(CODES):
                    ax.plot(time[mask]*1000, means[b, p, w, c, mask], color=COLORS[w], label=WAVEFORMS[code])
                ax.axvline(0, color='k', lw=.6)
                ax.set(title=f'{branch}: {channel}, presentation {p+1}', ylabel='uV', xlabel='Trigger-relative ms')
                ax.grid(alpha=.15)
    fig.suptitle(title+'\nPeri-trigger voltage is displayed, not interpreted as a cortical response.', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .92))
    save(fig, path)


def trial_images(epochs, meta, time, path, title, limit):
    fig, axes = plt.subplots(5, 4, figsize=(17, 15), sharex=True)
    mask = (time >= -.25) & (time < 1.2)
    for w, code in enumerate(CODES):
        for ci, channel in enumerate(('C3', 'C4')):
            for p in range(2):
                ax = axes[w, ci*2+p]
                keep = (meta.trigger == code) & (meta.position == p+1)
                selected = epochs[keep.to_numpy(), CHANNELS.index(channel)][:, mask].copy()
                selected[:, (time[mask] >= 0) & (time[mask] < .05)] = np.nan
                im = ax.imshow(selected, aspect='auto', origin='lower', interpolation='nearest',
                    extent=[time[mask][0]*1000, (time[mask][-1]+time[1]-time[0])*1000, 0, len(selected)],
                    cmap='RdBu_r', vmin=-limit, vmax=limit)
                ax.axvspan(0, 50, color='.8')
                ax.set_title(f'{WAVEFORMS[code]} / {channel} / P{p+1}, n={len(selected)}', fontsize=9)
                ax.set_ylabel('Chronological trial index')
                if w == 4:
                    ax.set_xlabel('Trigger-relative ms')
    fig.suptitle(title+f'\nFixed C3/C4; no amplitude/latency sorting. Shared color clipping at +/-{limit:.1f} uV; 0-50 ms omitted.', fontsize=12)
    fig.tight_layout(rect=(0, 0, .93, .95))
    fig.colorbar(im, ax=axes.ravel().tolist(), location='right', fraction=.025, pad=.02, label='uV (clipped display only)')
    save(fig, path)


def render(out):
    out = Path(out)
    config = json.loads((out/'run_config.json').read_text())
    status = json.loads((out/'status.json').read_text())
    subjects = status['completed_subjects']
    if not subjects:
        raise ValueError('No completed participants to render')
    figure_dir = out/'figures'
    figure_dir.mkdir(exist_ok=True)
    all_means = []
    for subject in subjects:
        sub = out/f'sub-{subject}'
        with np.load(sub/'summaries.npz') as saved:
            means, time = saved['means'], saved['time_s']
        all_means.append(means)
        for b, branch in enumerate(BRANCHES):
            traces(means[b], time, figure_dir/f'sub-{subject}_{branch}_evoked.png', f'Subject {subject}: {branch}')
        onset_figure(means, time, figure_dir/f'sub-{subject}_onset.png', f'Subject {subject}: onset inspection')
        meta = pd.read_csv(sub/'eligible_events.csv')
        audit = json.loads((sub/'audit.json').read_text())
        for branch in BRANCHES:
            epochs = np.load(sub/f'{branch}_epochs_uv.npy', mmap_mode='r')
            trial_images(epochs, meta, time, figure_dir/f'sub-{subject}_{branch}_trials.png',
                f'Subject {subject}: {branch} single epochs', audit['trial_image_limit_uv'])
        print(f'Rendered subject {subject}', flush=True)
    all_means = np.stack(all_means)
    group = all_means.mean(axis=0)
    rng = np.random.default_rng(20261030)
    # Participant bootstrap, not independent epochs or overlapping split-halves.
    weights = np.array([np.bincount(rng.integers(len(subjects), size=len(subjects)), minlength=len(subjects))/len(subjects) for _ in range(1000)])
    for b, branch in enumerate(BRANCHES):
        band = np.empty((2, *group[b].shape)) if len(subjects) > 1 else None
        if band is not None:
            for c in range(len(CHANNELS)):
                boot = np.einsum('rs,spwt->rpwt', weights, all_means[:, b, :, :, c], optimize=True)
                band[:, :, :, c] = np.quantile(boot, [.025, .975], axis=0)
            del boot
        traces(group[b], time, figure_dir/f'group_{branch}_evoked.png', f'Equal-participant mean: {branch}, n={len(subjects)}', band=band)
        traces(group[b], time, figure_dir/f'group_{branch}_deviations.png', f'Equal-participant waveform deviations: {branch}', deviations=True)
    onset_figure(group, time, figure_dir/'group_onset.png', 'Equal-participant onset inspection')
    np.savez_compressed(out/'group_summaries.npz', means_uv=group, participant_means_uv=all_means, time_s=time)
    with np.load(out/f'sub-{subjects[0]}'/'summaries.npz') as saved:
        sos, sfreq = saved['sos'], float(saved['sfreq'])
    f, h = sosfreqz(sos, worN=16000, fs=sfreq)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(f, 20*np.log10(np.maximum(np.abs(h)**2, 1e-10)))
    ax.set(xlim=(0, 80), ylim=(-100, 3), xlabel='Hz', ylabel='Forward/backward gain (dB)', title='Fixed 0.5-30 Hz Butterworth band-pass: squared magnitude response')
    ax.grid(alpha=.25)
    save(fig, figure_dir/'filter_response.png')
    limitations = ('Descriptive inspection only; no new classifier, repeatability estimate or significance test. '
        'Both views use the same complete pairs and the original 14 classifier electrodes. '
        'Minimal means scalp-average reference plus baseline, without added temporal filtering. '
        'Bandpass adds only the fixed 0.5-30 Hz forward/backward Butterworth filter. '
        'No ICA, onset interpolation, notch, resampling or new amplitude threshold. '
        'Noncausal filtering can spread stimulation transients; 0-50 ms omission is only a display choice. '
        'Pre-trigger activity may include preceding-stimulus responses. Later activity may include decisions/movement. '
        'Physical current timing is unknown. Waveform means have unequal and different observed comparison support; '
        'cell balancing does not remove context/order confounds. Grand averages can conceal participant differences. '
        'Pointwise bootstrap bands are descriptive, not corrected tests. Trial-image clipping never changes stored data.')
    body = ['<!doctype html><meta charset="utf-8"><title>Task 4 simple-pipeline visual inspection</title>',
        '<style>body{max-width:1500px;margin:2em auto;font:16px system-ui;padding:1em}img{max-width:100%}p{line-height:1.6}summary{cursor:pointer;font-weight:bold}details{margin:1.5em 0}a{margin-right:1em}</style>',
        '<h1>Task 4: waveform-by-waveform EEG inspection</h1>', '<p>'+html.escape(limitations)+'</p>',
        '<p>2026-10-30. Baseline [-250,-50) ms; displayed support [-300,1200) ms. '
        'Full-rate data; eight-second guards around recording/segment edges. '
        'Missing/gapped support excludes complete pairs identically in both views.</p>',
        '<h2>Group views</h2>']
    for branch in BRANCHES:
        for kind in ('evoked', 'deviations'):
            file = f'group_{branch}_{kind}.png'
            body.append(f'<details><summary>{branch}: {kind}</summary><a href="figures/{file}">Open full-resolution image</a><img src="figures/{file}"></details>')
    body += ['<details><summary>Group onset and filter response</summary><img src="figures/group_onset.png"><img src="figures/filter_response.png"></details>', '<h2>Participant views</h2>']
    for subject in subjects:
        audit = json.loads((out/f'sub-{subject}'/'audit.json').read_text())
        body.append(f'<details><summary>Subject {subject}: {audit["n_visual_events"]} events / {audit["n_visual_pairs"]} pairs</summary><p>'+html.escape(json.dumps(audit['exclusion_counts']))+'</p>')
        for suffix in ('minimal_evoked', 'bandpass_evoked', 'onset', 'minimal_trials', 'bandpass_trials'):
            body.append(f'<details><summary>{suffix}</summary><a href="figures/sub-{subject}_{suffix}.png">Full-resolution image</a><img loading="lazy" src="figures/sub-{subject}_{suffix}.png"></details>')
        body.append('</details>')
    if status['errors']:
        body.append('<h2>Incomplete run</h2><pre>'+html.escape(json.dumps(status['errors'], indent=2))+'</pre>')
    (out/'index.html').write_text('\n'.join(body), encoding='utf-8')
    (out/'RESULTS.md').write_text('# Task 4 simple-pipeline visual inspection\n\n'+limitations+'\n\n'
        f'Completed subjects: {subjects}. Errors: {status["errors"]}.\n\n'
        'See index.html for waveform/position overlays, participant plots, chronological single-epoch images, onset panels and filter response.\n', encoding='utf-8')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    render(parser.parse_args().output)
