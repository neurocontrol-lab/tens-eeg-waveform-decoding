"""Display individual participants without changing saved group inference."""
import argparse
import html
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits
from experiments.response_characterization.e002_electrode_position_slope.__main__ import sha, write_json
from experiments.signal_quality.e007_preprocessing_comparison.__main__ import load_inputs
from experiments.response_characterization.e006_multiscale_responses.analysis import transform, condition_weights
from .analysis import SCALES_MS, BRANCHES, INTERVALS
from .report import WAVEFORMS


def participant_figure(values, times, subject, channel, period, mode, filename):
    """values: branch × scale × waveform × position × time."""
    fig,axes = plt.subplots(5,4,figsize=(15,12),layout='constrained',sharex=True)
    colors = plt.get_cmap('tab10').colors
    for s,sigma in enumerate(SCALES_MS):
        bound = max(1e-9,float(np.abs(values[:,s]).max()))*1.08
        for b,branch in enumerate(BRANCHES):
            for p in range(2):
                ax = axes[s,b*2+p]
                for w,name in enumerate(WAVEFORMS):
                    ax.plot(times*1000,values[b,s,w,p],color=colors[w],label=name,linewidth=1.2)
                ax.axhline(0,color='.6',linewidth=.6)
                ax.set_ylim(-bound,bound)
                if s == 0:
                    ax.set_title(f'{branch} · {"first" if p == 0 else "second"}')
                if b == 0 and p == 0:
                    ax.set_ylabel(f'σ={sigma:g} ms\nµV/s')
                if s == 4:
                    ax.set_xlabel('Time from recorded trigger (ms)')
    handles,labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=5)
    label = 'Full waveform averages (common response retained)' if mode == 'means' else 'Waveform deviations (five-waveform mean subtracted)'
    fig.suptitle(f'Subject {subject} · {channel} · {period}\n{label}; descriptive, no significance implied')
    fig.savefig(filename,dpi=130)
    plt.close(fig)


def trial_figure(data, times, meta, channel_index, subject, representation, filename):
    """All eligible individual trials, with context-balanced means in black."""
    fig,axes = plt.subplots(5,4,figsize=(15,12),layout='constrained',sharex=True)
    colors = plt.get_cmap('tab10').colors
    weights = condition_weights(meta).reshape(10,len(meta))
    means = {b:(weights @ x[:,channel_index]).reshape(5,2,-1) for b,x in data.items()}
    for w,name in enumerate(WAVEFORMS):
        wave_mask = meta.trigger.to_numpy() == 239+w
        bound = max(float(np.abs(x[wave_mask,channel_index]).max()) for x in data.values())*1.05
        for b,branch in enumerate(BRANCHES):
            for p in range(2):
                ax = axes[w,b*2+p]
                mask = wave_mask & (meta.position.to_numpy() == p+1)
                for trace in data[branch][mask,channel_index]:
                    ax.plot(times*1000,trace,color=colors[w],alpha=.25,linewidth=.65)
                ax.plot(times*1000,means[branch][w,p],color='black',linewidth=2)
                ax.axhline(0,color='.6',linewidth=.6)
                ax.set_ylim(-bound,bound)
                ax.text(.02,.94,f'n={int(mask.sum())}',transform=ax.transAxes,va='top',fontsize=9)
                if w == 0:
                    ax.set_title(f'{branch} · {"first" if p == 0 else "second"}')
                if b == 0 and p == 0:
                    ax.set_ylabel(f'{name}\n'+('µV' if representation == 'voltage' else 'µV/s'))
                if w == 4:
                    ax.set_xlabel('Time from recorded trigger (ms)')
    fig.suptitle(f'Subject {subject} · C3 · {representation} · post-trigger\n'
                 'Colored: every eligible trial; black: equal-comparison mean; y-limits shared per waveform row')
    fig.savefig(filename,dpi=130)
    plt.close(fig)


def render(args):
    source = args.source
    config = json.loads((source/'config.json').read_text())
    status = json.loads((source/'status.json').read_text())
    if status['errors'] or status['completed'] != config['subjects']:
        raise ValueError('Require complete saved time-resolved analysis')
    if args.output.exists():
        raise ValueError('Use a fresh participant-view output directory')
    args.output.mkdir(parents=True)
    reference = args.reference_subject
    if reference not in status['completed']:
        raise ValueError('Reference subject must be among completed participants')
    audits, inputs = [], {}
    for subject in status['completed']:
        print(f'Participant plots: subject {subject}',flush=True)
        destination = args.output/f'sub-{subject}'
        destination.mkdir()
        file = source/f'sub-{subject}/effects.npz'
        inputs[str(subject)] = sha(file)
        with np.load(file) as saved:
            values = saved['evoked_display'].astype(float)
            channels = saved['plot_channels'].tolist()
            times = {p:saved[p+'_times'] for p in INTERVALS}
        meta = pd.read_csv(source/f'sub-{subject}/events.csv')
        counts = meta.groupby(['trigger','waveform','position']).size().rename('n_events').reset_index()
        counts.to_csv(destination/'trial_counts.csv',index=False)
        deviations = values-values.mean(axis=3,keepdims=True)
        # Show C3 post for every subject. Additional fixed reference views are not signal-selected.
        views = [('C3','post')]
        if subject == reference:
            views += [('C3','pre'),('C4','post'),('CP1','post'),('Pz','post')]
        for channel,period in views:
            c,p = channels.index(channel),list(INTERVALS).index(period)
            for mode,x in (('means',values),('deviations',deviations)):
                participant_figure(x[:,p,:,:,:,c,:],times[period],subject,channel,period,mode,
                    destination/f'{channel}_{period}_{mode}.png')
        # Check that deviations have zero mean over the five waveform curves.
        if not np.allclose(deviations.mean(axis=3),0.,atol=1e-9):
            raise ValueError('Waveform common-response subtraction failed')
        audits.append(dict(subject=subject,n_events=len(meta),
                           n_pairs=int(meta.pair_id.nunique()),source_sha256=inputs[str(subject)]))
    if not args.no_trials:
        print(f'Reference subject {reference}: reconstruct original trials with saved ICA',flush=True)
        loader_args = SimpleNamespace(prior=args.prior,ica_cache=args.ica_cache,
            data_dir=args.data_dir,master_csv=args.master_csv)
        branches,times,channels,meta,audit = load_inputs(reference,loader_args,config)
        recorded = json.loads((source/f'sub-{reference}/audit.json').read_text())
        for key in ('source_sha256','csv_sha256','saved_ica_sha256','n_events'):
            if audit[key] != recorded[key]:
                raise ValueError(f'Reference input differs from previous analysis: {key}')
        previous = pd.read_csv(source/f'sub-{reference}/events.csv')
        for key in ('sample','trigger','pair_id','position'):
            np.testing.assert_array_equal(meta[key],previous[key])
        mask = (times >= INTERVALS['post'][0]) & (times < INTERVALS['post'][1])
        c = channels.index('C3')
        weight = condition_weights(meta).reshape(10,len(meta))
        trial_data = {}
        for representation in ('voltage','Gaussian derivative σ=20 ms'):
            selected = {}
            for b,branch in enumerate(BRANCHES):
                # Only C3 is needed here; transformation remains strictly temporal.
                epochs = branches[branch][0][:,c:c+1,:]
                x = epochs if representation == 'voltage' else transform(epochs,audit['sfreq'],
                    dict(operator='gaussian_derivative',sigma_ms=20.))
                selected[branch] = x[:,:,mask]
                trial_data[branch+('_voltage' if representation == 'voltage' else '_deriv20')] = selected[branch][:,0]
                if representation != 'voltage':
                    actual = (weight @ selected[branch][:,0]).reshape(5,2,-1)
                    with np.load(source/f'sub-{reference}/effects.npz') as saved:
                        c_saved = saved['plot_channels'].tolist().index('C3')
                        expected = saved['evoked_display'][b,1,3,:,:,c_saved]
                    np.testing.assert_allclose(actual,expected,rtol=2e-6,atol=2e-5)
            name = 'voltage' if representation == 'voltage' else 'deriv20'
            trial_figure(selected,times[mask],meta,0,reference,representation,
                args.output/f'sub-{reference}/C3_post_{name}_trials.png')
        np.savez_compressed(args.output/f'sub-{reference}/C3_trial_traces.npz',
                            times=times[mask],**trial_data)
        meta.to_csv(args.output/f'sub-{reference}/trial_events.csv',index=False)
        write_json(args.output/f'sub-{reference}/trial_reconstruction.json',audit)
    pd.DataFrame(audits).to_csv(args.output/'participant_audit.csv',index=False)
    options = ''.join(f'<option value="{s}"'+(' selected' if s == reference else '')+
                      f'>Subject {s}</option>' for s in status['completed'])
    page = ['<!doctype html><html><head><meta charset="utf-8"><title>Task 4 participant responses</title>',
        '<style>body{font:16px system-ui;max-width:1450px;margin:25px auto;padding:0 20px} '
        'img{max-width:100%;height:auto}select{font:inherit;margin:6px}p{max-width:1000px}</style></head><body>',
        '<h1>Task 4: individual participant responses</h1>',
        '<p>These plots retain each participant’s timing and waveform averages. They are descriptive, '
        'and do not establish individual statistical significance or single-trial decoding. '
        'Each participant’s curves average trials within each waveform/position, equally weighting '
        'the observed comparison contexts. Both processing branches use the same eligible events.</p>',
        '<label>Participant <select id="subject" onchange="update()">'+options+'</select></label>',
        '<label>Display <select id="mode" onchange="update()"><option value="means">Full waveform averages</option>'
        '<option value="deviations">Waveform deviations from common response</option></select></label>',
        '<p>Fixed C3, post-trigger 180–480 ms. Rows show σ=2.5/5/10/20/40 ms. '
        'Columns separate minimal/supplied preprocessing and first/second positions. '
        'Y-limits are shared within each scale row, not between participants or scales. '
        'Full averages retain the response shared across waveforms; deviations subtract '
        'the mean of the five waveform averages separately for each branch/scale/position.</p>',
        f'<img id="main" src="sub-{reference}/C3_post_means.png" alt="Participant waveform averages">',
        '<p><a id="image-link" href="'+f'sub-{reference}/C3_post_means.png'+ '">Open full-resolution plot</a> · '
        '<a id="count-link" href="'+f'sub-{reference}/trial_counts.csv'+ '">Trial counts</a></p>',
        '<script>function update(){const s=document.getElementById("subject").value;'
        'const m=document.getElementById("mode").value;const p="sub-"+s+"/C3_post_"+m+".png";'
        'document.getElementById("main").src=p;document.getElementById("image-link").href=p;'
        'document.getElementById("count-link").href="sub-"+s+"/trial_counts.csv";}</script>',
        f'<h2>Reference example: Subject {reference}</h2>',
        '<p>The reference is fixed to the first eligible participant by default, not selected for '
        'visually large effects. Individual trial overlays retain the common response. '
        'Every eligible trial is shown, with its count in each panel. Black curves equally '
        'weight comparison contexts, so they need not equal the ordinary mean of all thin traces. '
        'Within-participant averaging can still blur trial-to-trial latency variation.</p>']
    if not args.no_trials:
        for name in ('voltage','deriv20'):
            page.append(f'<h3>C3 individual trials: {name}</h3><img src="sub-{reference}/C3_post_{name}_trials.png">')
    page.append('<h3>Additional fixed views</h3><ul>')
    for channel,period in (('C3','pre'),('C4','post'),('CP1','post'),('Pz','post')):
        for mode in ('means','deviations'):
            label = html.escape(f'{channel} {period} {mode}')
            page.append(f'<li><a href="sub-{reference}/{channel}_{period}_{mode}.png">{label}</a></li>')
    page += ['</ul><p>No new inferential test or classifier was run. The earlier group results are unchanged.</p>',
             '</body></html>']
    (args.output/'index.html').write_text('\n'.join(page),encoding='utf-8')
    write_json(args.output/'provenance.json',dict(source=str(source.resolve()),
        source_config_sha256=sha(source/'config.json'),participant_inputs=inputs,
        display_source='Previously saved float32 per-participant condition means; converted to float64 for display arithmetic',
        reference_subject=reference,reference_selection='First eligible subject by default, not signal-selected',
        trial_overlays=not args.no_trials,new_inference=False,ica_refitted=False,
        renderer_sha256=sha(Path(__file__)),
        definitions='Full means retain common response; deviations subtract five-waveform mean; equal observed comparison weights'))
    print(f'Participant browser: {(args.output/"index.html").resolve()}',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('output/task4_time_resolved_20261023'))
    parser.add_argument('--output',type=Path,default=Path('output/task4_participant_views_20261023'))
    parser.add_argument('--reference-subject',type=int,default=3)
    parser.add_argument('--no-trials',action='store_true',help='Use saved condition averages only; do not reload EEG')
    parser.add_argument('--prior',type=Path,default=Path('output/task4_multiscale_20261023'))
    parser.add_argument('--ica-cache',type=Path,default=Path('output/task4_response_cache'))
    parser.add_argument('--data-dir',type=Path,default=Path('EEGData'))
    parser.add_argument('--master-csv',type=Path,default=Path('EEG_Codes_Export/02_SourceLocalization_Task4/Master_Data_ToSub12 (1).csv'))
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        render(args)


if __name__ == '__main__':
    main()
