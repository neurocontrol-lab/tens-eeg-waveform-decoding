"""Local scientific plots and a self-contained HTML/Markdown diagnostic report."""
import html
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from experiments.waveform_decoding.e001_voltage_baseline.events import WAVEFORMS
from .analysis import WINDOWS, METRICS
from .electrode_repeatability import electrode_repeatability

PLOT_CHANNELS = ('Fp1','FC1','C3','Cz','C4','CP1','CP2','Pz')
COLORS = ('#2171b5','#d94801','#238b45','#88419d','#a6761d')


def common_response_plot(results,channels,times,out):
    chosen = ('Fp1','C3','CP1','Pz')
    mask = (times >= -.2) & (times <= .5)
    smoothing = max(3,int(round(.025/np.median(np.diff(times)))) | 1)
    fig,axes = plt.subplots(4,2,figsize=(12,10),sharex=True)
    for column,metric in enumerate(['voltage','slope']):
        data = np.stack([r[metric].mean(axis=0) for r in results])
        if metric == 'voltage':
            data = savgol_filter(data,smoothing,1,axis=-1)
        mean = data.mean(axis=0)
        sem = data.std(axis=0,ddof=1)/np.sqrt(len(data)) if len(data)>1 else np.zeros_like(mean)
        for row,channel in enumerate(chosen):
            c = channels.index(channel)
            ax = axes[row,column]
            for position,color in enumerate(['#2166ac','#b35806']):
                y,error = mean[position,c,mask],sem[position,c,mask]
                ax.plot(times[mask]*1000,y,color=color,label=f'Position {position+1}')
                ax.fill_between(times[mask]*1000,y-error,y+error,color=color,alpha=.18)
            ax.axvline(0,color='.5',lw=.8)
            ax.axhline(0,color='.7',lw=.5)
            ax.axvspan(-2,10,color='.85',alpha=.5)
            ax.set_ylabel(f'{channel}\n'+('µV' if metric == 'voltage' else 'µV/s'))
            if row == 0:
                ax.set_title(metric)
                ax.legend(loc='best',fontsize=8)
            if row == 3:
                ax.set_xlabel('Time from recorded trigger (ms)')
    fig.suptitle('Common response averaged equally over waveforms and participants\n25-ms voltage display smoothing / local slopes; shading = ±1 participant SEM, descriptive')
    fig.tight_layout(rect=(0,0,1,.94))
    path = out/'group_common_response_positions.png'
    fig.savefig(path,dpi=140)
    plt.close(fig)
    return path


def position_topographies(results,tests,channels,out):
    import mne
    info = mne.create_info(channels,1000.,'eeg')
    info.set_montage('colin27_1020')
    coefficients = np.mean([r['effects']['position'][0] for r in results],axis=0)
    fig,axes = plt.subplots(3,3,figsize=(12,11),layout='constrained')
    for m,metric in enumerate(METRICS):
        bound = max(np.abs(coefficients[m,1:]).max(),1e-6)
        for w,window in enumerate(['early','middle','late'],1):
            subset = tests[(tests.effect == 'position') & (tests.metric == metric) & (tests.window == window)]
            hits = subset.set_index('channel').loc[channels].p_fwer.to_numpy() <= .05
            im,_ = mne.viz.plot_topomap(coefficients[m,w],info,axes=axes[m,w-1],show=False,
                                      vlim=(-bound,bound),cmap='RdBu_r',mask=hits,
                                      mask_params=dict(marker='o',markerfacecolor='none',
                                                       markeredgecolor='black',markeredgewidth=1.5,
                                                       linewidth=0,markersize=7))
            axes[m,w-1].set_title(f'{metric}\n{window}')
        fig.colorbar(im,ax=axes[m,:].tolist(),shrink=.6,
                     label='µV' if m == 0 else 'µV/s')
    fig.suptitle('Context-adjusted second minus first response\nApproximate standard electrode positions; black rings: corrected p ≤ .05')
    path = out/'group_position_topographies.png'
    fig.savefig(path,dpi=140)
    plt.close(fig)
    return path


def response_grid(voltage, slope, times, channels, prefix, title):
    """Fixed channels and axes; separate position panels, identical limits."""
    paths = []
    mask = (times >= -.2) & (times <= .5)
    smoothing = max(3, int(round(.025/np.median(np.diff(times)))) | 1)
    display_voltage = savgol_filter(voltage,smoothing,1,axis=-1)
    for metric,values,unit in [('voltage',display_voltage,'µV'),('slope',slope,'µV/s')]:
        fig,axes = plt.subplots(len(PLOT_CHANNELS),2,figsize=(12,16),sharex=True)
        for i,channel in enumerate(PLOT_CHANNELS):
            c = channels.index(channel)
            amplitude = np.nanmax(np.abs(values[:,:,c,mask]))
            amplitude = max(amplitude*1.1,1e-3)
            for position in range(2):
                ax = axes[i,position]
                for w,name in enumerate(WAVEFORMS.values()):
                    ax.plot(times[mask]*1000,values[w,position,c,mask],color=COLORS[w],
                            lw=1,label=name)
                ax.axvspan(-2,10,color='.85',alpha=.8)
                ax.axvline(0,color='.4',lw=.7)
                ax.axhline(0,color='.7',lw=.5)
                ax.set_ylim(-amplitude,amplitude)
                ax.set_ylabel(f'{channel}\n{unit}')
                if i == 0:
                    ax.set_title(f'Presentation {position+1}')
                if i == len(PLOT_CHANNELS)-1:
                    ax.set_xlabel('Time from recorded trigger (ms)')
        handles,labels = axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.965),ncol=5)
        detail = '25-ms display smoothing' if metric == 'voltage' else '25-ms local linear slopes'
        fig.suptitle(f'{title}: {metric}\n{detail}; equal comparison-cell weight; shaded interpolated onset',y=.995)
        fig.tight_layout(rect=(0,0,1,.945))
        path = prefix.with_name(prefix.name+f'_{metric}.png')
        fig.savefig(path,dpi=130)
        plt.close(fig)
        paths.append(path)
    return paths


def all_channel_heatmaps(voltage, slope, times, channels, out):
    paths = []
    mask = (times >= .05) & (times < .5)
    for name,values in [('voltage',voltage),('slope',slope)]:
        # Both panels are descriptive; RMS between waveforms is positively biased by noise.
        position_difference = (values[:,1]-values[:,0]).mean(axis=0)[:,mask]
        balanced = values.mean(axis=1)
        waveform_rms = np.sqrt(np.mean((balanced-balanced.mean(axis=0))**2,axis=0))[:,mask]
        fig,axes = plt.subplots(1,2,figsize=(14,16),sharey=True)
        bound = max(np.max(np.abs(position_difference)),1e-6)
        a = axes[0].imshow(position_difference,aspect='auto',origin='lower',cmap='RdBu_r',
                           vmin=-bound,vmax=bound,extent=(50,500,-.5,len(channels)-.5))
        b = axes[1].imshow(waveform_rms,aspect='auto',origin='lower',cmap='magma',
                           extent=(50,500,-.5,len(channels)-.5))
        axes[0].set_yticks(range(len(channels)),channels,fontsize=8)
        axes[0].set_title('Second minus first (equal waveform weight)')
        axes[1].set_title('RMS waveform differences (noise-biased; descriptive)')
        for ax in axes:
            ax.set_xlabel('Time from recorded trigger (ms)')
        unit = 'µV' if name == 'voltage' else 'µV/s'
        fig.colorbar(a,ax=axes[0],label=unit,shrink=.5)
        fig.colorbar(b,ax=axes[1],label=unit,shrink=.5)
        fig.suptitle(f'All scalp channels: group {name}; equal participant weight')
        fig.tight_layout()
        path = out/f'group_all_channels_{name}.png'
        fig.savefig(path,dpi=130)
        plt.close(fig)
        paths.append(path)
    return paths


def inference_plot(tests,channels,out):
    fig,axes = plt.subplots(3,3,figsize=(13,17),sharey=True)
    for i,effect in enumerate(['waveform','position','interaction']):
        for j,metric in enumerate(METRICS):
            subset = tests[(tests.effect == effect) & (tests.metric == metric)]
            matrix = subset.pivot(index='channel',columns='window',values='p_fwer').loc[channels,list(WINDOWS)]
            image = axes[i,j].imshow(-np.log10(matrix.to_numpy()),origin='lower',aspect='auto',
                                     vmin=0,vmax=3,cmap='viridis')
            axes[i,j].set_title(f'{effect}\n{metric}')
            axes[i,j].set_xticks(range(4),list(WINDOWS),rotation=30)
            axes[i,j].set_yticks(range(len(channels)),channels,fontsize=6)
    fig.suptitle('Exploratory context-adjusted effects; participant sign-flips\nMax-stat across electrodes/windows/metrics, plus correction for three effect families')
    fig.subplots_adjust(top=.92,bottom=.05,left=.08,right=.85,hspace=.25,wspace=.15)
    fig.colorbar(image,cax=fig.add_axes([.90,.27,.02,.46]),
                 label='−log10(corrected p); 1.30 corresponds to p=.05')
    path = out/'group_corrected_tests.png'
    fig.savefig(path,dpi=140)
    plt.close(fig)
    return path


def electrode_reliability_plot(summary,channels,out):
    fig,axes = plt.subplots(1,3,figsize=(10,16),sharey=True)
    for i,metric in enumerate(METRICS):
        subset = summary[summary.metric == metric]
        table = subset.pivot(index='channel',columns='position',values='median_waveform_feature_r').loc[channels,[1,2]]
        im = axes[i].imshow(table,origin='lower',aspect='auto',vmin=-1,vmax=1,cmap='RdBu_r')
        axes[i].set_title(['Voltage','Signed slope','Absolute slope'][i],fontsize=10)
        axes[i].set_xticks([0,1],['First','Second'])
        axes[i].set_yticks(range(len(channels)),channels,fontsize=7)
        for c in range(len(channels)):
            for p in range(2):
                value = table.iloc[c,p]
                axes[i].text(p,c,f'{value:.2f}' if np.isfinite(value) else 'NA',
                             ha='center',va='center',fontsize=6,
                             color='white' if abs(value)>.65 else 'black')
    fig.suptitle('Per-electrode repeatability of waveform-specific feature patterns\nFixed three post windows × five waveforms; common waveform mean removed')
    fig.subplots_adjust(top=.91,bottom=.04,left=.08,right=.85,wspace=.3)
    fig.colorbar(im,cax=fig.add_axes([.90,.27,.025,.46]),
                 label='Median participant split-half r (descriptive)')
    path = out/'group_electrode_repeatability.png'
    fig.savefig(path,dpi=140)
    plt.close(fig)
    return path


def write_report(out,results,tests,errors):
    channels,times = results[0]['channels'],results[0]['times']
    if any(not np.array_equal(r['times'],times) for r in results):
        raise ValueError('Cannot pool different time axes')
    voltage = np.mean([r['voltage'] for r in results],axis=0)
    slope = np.mean([r['slope'] for r in results],axis=0)
    figures = [common_response_plot(results,channels,times,out)]
    figures += response_grid(voltage,slope,times,channels,out/'group_responses',
                             f'Group mean ({len(results)} participants)')
    figures += all_channel_heatmaps(voltage,slope,times,channels,out)
    if not tests.empty:
        figures.append(position_topographies(results,tests,channels,out))
        figures.append(inference_plot(tests,channels,out))
    per_subject_figures = []
    for result in results:
        folder = out/f'sub-{result["subject"]}'
        per_subject_figures += response_grid(result['voltage'],result['slope'],times,channels,
                                             folder/'responses',f'Subject {result["subject"]}')
    repeatability = pd.concat([r['repeatability'].assign(subject=r['subject']) for r in results],ignore_index=True)
    repeatability.to_csv(out/'repeatability.csv',index=False)
    reliability = repeatability.groupby(['subject','metric','position'])[
        ['common_response_r','waveform_deviation_r']].median().reset_index()
    reliability.to_csv(out/'repeatability_summary.csv',index=False)
    electrode_rows = []
    for result in results:
        folder = out/f'sub-{result["subject"]}'
        with np.load(folder/'responses.npz') as saved:
            features = saved['features']
        meta = pd.read_csv(folder/'events.csv')
        meta = meta[meta.diagnostic_eligible].reset_index(drop=True)
        per_electrode = electrode_repeatability(features,meta,channels)
        per_electrode['subject'] = result['subject']
        per_electrode.to_csv(folder/'electrode_repeatability.csv',index=False)
        electrode_rows.append(per_electrode)
    electrode_rows = pd.concat(electrode_rows,ignore_index=True)
    electrode_rows.to_csv(out/'electrode_repeatability.csv',index=False)
    electrode_summary = electrode_rows.groupby(['metric','position','channel'],as_index=False).agg(
        median_waveform_feature_r=('median_waveform_feature_r','median'),
        n_subjects=('subject','nunique'),
        n_subjects_with_finite_r=('median_waveform_feature_r','count'))
    electrode_summary.to_csv(out/'electrode_repeatability_summary.csv',index=False)
    figures.append(electrode_reliability_plot(electrode_summary,channels,out))
    lines = ['# Task 4 electrode, position and EEG-slope diagnostics','',
        f'Completed subjects: {", ".join(str(r["subject"]) for r in results)}. Failures: {errors or "none"}.',
        '', '## Scope and processing','',
        'Exploratory response analysis; no new classifier or feature selection was performed. '
        'EEG slopes remain eligible as future fixed features regardless of these p-values. '
        'This analysis assesses associations and repeatability, not artifact-free cortical origin.', '',
        'Adapted supplied August28 AutoEOG33: verified µV→V conversion, average reference, '
        '0.5-Hz order-94 Chebyshev-II high-pass, 60-Hz harmonic notches, '
        '33-component extended Infomax ICA fitted to a 1-Hz high-pass copy, '
        'Fp1-calibrated automatic EOG component rejection, −2 to +10-ms interpolation, '
        '190-Hz low-pass. Strict debounced triggers and full CSV alignment replace the legacy event loader. '
        'No muscle component rejection or bad-channel interpolation. ICA/calibration use each full '
        'participant recording for description, not held-out classifier evaluation.', '',
        'Epoch support −300 to +550 ms; baseline −250 to −50 ms. '
        'QC excludes nonfinite samples, boundaries, timestamp irregularities, neighboring triggers, '
        'or >500 µV peak-to-peak on any scalp electrode in pre/post non-onset spans. '
        'Both presentations are excluded together. This all-channel rule differs from the baseline’s 14-channel QC.', '',
        'Slopes are local first-order least-squares fits over 25 ms (25 samples at 1 kHz), '
        'reported in µV/s. No slope-specific baseline subtraction. '
        'Fixed windows: pre −200 to −50 ms; early 50–150 ms; middle 150–300 ms; late 300–500 ms. '
        'Metrics: mean voltage, mean signed slope, mean absolute slope. Padding protects analyzed slope '
        'windows from epoch-edge extrapolation.', '',
        '## Cohort and processing checks','',
        '| Subject | Retained / aligned events | ICA components removed | Reported ICA iterations | Design condition number |',
        '| --- | --- | --- | --- | --- |']
    status_path = out/'status.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    if status.get('interim'):
        lines[0] = '# INTERIM pilot: Task 4 electrode, position and EEG-slope diagnostics'
        lines[2:2] = [f'INTERIM ONLY: {len(results)} completed participants; '
                      f'pending subjects {status.get("pending",[])}. The planned cohort is incomplete.', '']
    for r in results:
        a = r['audit']
        lines.append(f'| {r["subject"]} | {a["n_eligible"]} / {a["n_events"]} | {a["ica_excluded"]} | {a["ica_iterations"]} | {a["design"]["condition_number"]:.1f} |')
    reported_limit = [r['subject'] for r in results if r['audit']['ica_iterations'] >= 500]
    if reported_limit:
        lines += ['', f'MNE reports ICA n_iter_=500 for subjects {reported_limit}. '
                  'In the recorded MNE 1.13.2 Infomax implementation, the weight-change stopping rule '
                  'can also return this value after an earlier stop. The count alone therefore does not '
                  'establish whether these fits converged or exhausted the iteration limit; '
                  'detailed optimizer traces were not retained.']
    lines += ['', '## Waveform and position comparisons','',
        'Participant-level regression includes comparison-specific intercepts, centered within-comparison '
        'elapsed time, sum-coded waveform, first/second position, and waveform × position interaction. '
        'The position coefficient is second minus first averaged equally over five waveforms; '
        'waveform coefficients average equally over both positions. Full-rank design is required. '
        'These adjustments do not eliminate preceding-waveform, adaptation, expectation, or nonlinear drift confounds.', '',
        'Group inference flips the entire coefficient vector independently for each participant, '
        'enumerating all 2^N sign patterns. Position uses squared one-sample t; waveform/interaction use '
        'the sum of four squared contrast t statistics. Max-stat correction covers all 62 electrodes, '
        'four windows and three metrics per family, followed by a factor of three for the three families. '
        'This assumes independent participants and symmetric coefficient errors. It tests consistent '
        'model-adjusted associations, not randomized delivery or a causal waveform mechanism.', '',
        '| Effect | Tests with corrected p ≤ .05 | Smallest corrected p |',
        '| --- | --- | --- |']
    if tests.empty:
        lines.append('| Not run | Need at least three participants | — |')
    else:
        for effect in ['waveform','position','interaction']:
            subset = tests[tests.effect == effect]
            lines.append(f'| {effect} | {int((subset.p_fwer <= .05).sum())} / {len(subset)} | {subset.p_fwer.min():.4f} |')
        strongest = tests.sort_values(['p_fwer','p_uncorrected','statistic'],ascending=[True,True,False]).head(15)
        strongest.to_csv(out/'strongest_exploratory_effects.csv',index=False)
        lines += ['', 'The strongest effects are saved in `strongest_exploratory_effects.csv`; '
                  'their ranking is descriptive and must not be used for full-dataset classifier feature selection.']
        minimum = min(1.,6/(2**len(results)))
        lines += ['', f'With {len(results)} participants and the three-family correction, '
                  f'the smallest attainable corrected p is at least {minimum:.5f}. '
                  'Small pilot cohorts may therefore be unable to reach p ≤ .05 regardless of effect size.']
    lines += ['', '## Repeatability','',
        '100 deterministic random split-halves within each ordered comparison; complete pairs stay together. '
        'Ordered cells with fewer than two pairs are omitted only from this check. '
        'Each observed comparison contributes equally to waveform/position averages. '
        'Pearson r compares scalp × time patterns over 50–500 ms. Common-response r averages waveforms; '
        'waveform-deviation r removes the common response before comparing condition-specific patterns. '
        'These repeated splits are dependent descriptive checks; their spread is not a confidence interval. '
        'Within-comparison splits share context and can retain stable artifacts or drift.', '',
        '| Metric | Position | Median participant common-response r | Median participant waveform-deviation r |',
        '| --- | --- | --- | --- |']
    for (metric,position),group in reliability.groupby(['metric','position']):
        lines.append(f'| {metric} | {position} | {group.common_response_r.median():.3f} | {group.waveform_deviation_r.median():.3f} |')
    lines += ['', 'A complementary electrode-level check repeats the same whole-pair splits on the '
        'fixed trial features. For each electrode, metric and position, correlate the five-waveform × '
        'three-post-window pattern after removing the across-waveform mean in each window. '
        'Save participant median correlations and their group medians; there is no significance test '
        'or feature selection. This prevents a high-amplitude electrode from dominating the '
        'scalp-wide repeatability measure. See `electrode_repeatability_summary.csv` and the figure.']
    lines += ['', '## Interpretation limits','',
        '- Zero-phase filtering precedes stimulation interpolation in the supplied pipeline; '
        'pre-trigger differences and post-trigger slopes may include filter-spread electrical artifacts.',
        '- Baseline correction changes voltage offsets but not the fitted slopes. '
        'Signed window-mean slopes summarize net response change and may cancel opposing deflections.',
        '- The pre-voltage diagnostic window lies inside the voltage baseline. Its contrasts can be '
        'attenuated by that baseline choice; a null pre-voltage effect does not rule out contextual '
        'information. This is not the matched pre-stimulus classifier control from the initial baseline.',
        '- Condition means balance observed comparison cells, not missing cells. Inspect saved cell support. '
        'Unequal presentation counts and history remain relevant.',
        '- No corrected effect does not establish equivalence or absence of information. '
        f'{len(results)} participants and broad correction limit sensitivity.',
        '- Consistent waveform differences need not imply a pure shape effect: charge, '
        'intensity, perceived strength, and electrical contamination may contribute.',
        '- Repeatability or group differences do not establish single-epoch decoding. '
        'Future preprocessing and supervised feature selection must respect held-out evaluation.', '',
        '## Figures','', 'Voltage trace grids use 25-ms local-linear smoothing for display only; '
        'window-mean voltage statistics use unsmoothed cleaned EEG. Slopes use the fixed 25-ms estimator. '
        'The same voltage/slope scale is used for first and second presentations of each electrode.','']
    for path in figures:
        lines += [f'![{path.stem}]({path.relative_to(out).as_posix()})','']
    lines += ['## Participant response figures','']
    for path in per_subject_figures:
        lines.append(f'- [{path.parent.name}: {path.stem}]({path.relative_to(out).as_posix()})')
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    # No Markdown dependency; embed the report verbatim alongside viewable figures.
    pictures = '\n'.join(f'<figure><figcaption>{html.escape(p.stem)}</figcaption><img loading="lazy" src="{p.relative_to(out).as_posix()}"></figure>' for p in figures+per_subject_figures)
    content = ('<!doctype html><html><head><meta charset="utf-8"><title>Task 4 response diagnostics</title>'
               '<style>body{max-width:1200px;margin:30px auto;padding:0 20px;font:16px/1.5 system-ui}'
               'pre{white-space:pre-wrap}img{max-width:100%;height:auto}figure{margin:40px 0}</style>'
               '</head><body><pre>'+html.escape('\n'.join(lines[:lines.index('## Figures')]))+
               '</pre>'+pictures+'</body></html>')
    (out/'index.html').write_text(content,encoding='utf-8')
