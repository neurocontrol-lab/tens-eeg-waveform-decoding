"""Expanded corrected maps and participant-bootstrap interpretation."""
import html
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.response_characterization.e008_time_resolved_effects.analysis import sign_flip_family,family_adjustment
from .core import FAMILIES,BRANCHES,PERIODS,SCALES


def render(out):
    config = json.loads((out/'config.json').read_text())
    layout = json.loads((out/'layout.json').read_text())
    locations = dict(np.load(out/'locations.npz'))
    saved,rows = {},[]
    for family in FAMILIES:
        coefs = []
        for subject in config['subjects']:
            with np.load(out/f'sub-{subject}/effects.npz') as data:
                coefs.append(data[family])
        x = np.stack(coefs)
        print(f'Expanded inference: {family}, {x.shape[-1]:,} locations',flush=True)
        result = sign_flip_family(x)
        corrected = family_adjustment(result)
        saved.update({f'{family}_{key}':value for key,value in result.items()})
        saved[f'{family}_p_fwer'] = corrected
        saved[f'{family}_mean_coefficients'] = x.mean(axis=0)
        for b,branch in enumerate(BRANCHES):
            for p,period in enumerate(PERIODS):
                for s,sigma in enumerate(SCALES):
                    mask = (locations['branch']==b)&(locations['period']==p)&(locations['scale']==s)
                    pv = corrected[mask]
                    rows.append(dict(effect=family,branch=branch,period=period,sigma_ms=sigma,
                        n_tests=len(pv),n_significant=int((pv<=.05).sum()),min_p=float(pv.min())))
        plot_maps(corrected,locations,family,out)
        del coefs,x
    np.savez_compressed(out/'group_tests.npz',**saved)
    frame = pd.DataFrame(rows)
    frame.to_csv(out/'test_summary.csv',index=False)
    agg = frame.groupby('effect').agg(n_tests=('n_tests','sum'),n_significant=('n_significant','sum'),min_p=('min_p','min'))
    audits = [dict(subject=s,**json.loads((out/f'sub-{s}/audit.json').read_text())) for s in config['subjects']]
    n_events = sum(a['n_events'] for a in audits)
    uncertainty = pd.read_csv(out/'multiscale_difference_intervals.csv')
    plot_intervals(uncertainty,out)
    lines = ['# Task 4 response follow-up','',
        f'{len(audits)} participants, {n_events} frozen events; {layout["n_locations"]:,} locations per effect family.','',
        '## Earlier time-resolved effects','',
        'This exploratory extension was specified after the previous negative results. Earlier runs remain unchanged. '
        'Post intervals end at 480 ms and start at 50/50/51/91/171 ms for Gaussian sigma 2.5/5/10/20/40 ms. '
        'Each scale has matched pre [-480,-start) ms. All kernels use real padding and avoid the explicit onset blank. '
        'These are matched half-open intervals around onset, not one fixed time translation. '
        'Continuous supplied filters can still spread stimulation artifacts.','',
        'The same context-adjusted participant OLS model and exact joint-vector sign flips are used. '
        'The maximum spans BOTH preprocessing branches, BOTH periods, all five scales, all valid times and 62 electrodes. '
        'An additional factor of three covers waveform, position and interaction. No correction was dropped or '
        'window/channel selected from a peak. These are group response tests, not classification.','',
        '| Effect | Significant locations | Total locations | Minimum corrected p |',
        '| --- | ---: | ---: | ---: |']
    for family,row in agg.iterrows():
        lines.append(f'| {family} | {int(row.n_significant)} | {int(row.n_tests):,} | {row.min_p:.4f} |')
    lines += ['',f'Prior late-map reconstruction maximum absolute error: {max(a["late_coefficient_max_error"] for a in audits):.3g}. '
        'OLS and fixed Gaussian differentiation commute; applying differentiation after voltage-coefficient '
        'estimation was tested against trial-level filtering and checked against every saved late coefficient. '
        'No ICA was refitted and no new cohort exclusions were made.','',
        '## Participant uncertainty','',
        'First take each participant median over the 100 dependent split-halves. Then bootstrap the ten '
        'participants 10,000 times (seed 20261023). Report medians across participants and paired median '
        'differences, preserving participant identity. All intervals are descriptive, pointwise 95% percentile '
        'intervals, NOT multiplicity-adjusted tests. Earlier corrected tests are unchanged.','',
        '| Representation minus reference | Position | Waveform-repeatability median difference r | Descriptive 95% interval |',
        '| --- | --- | ---: | --- |']
    for r in uncertainty[uncertainty.metric=='waveform_deviation_r'].itertuples():
        lines.append(f'| {r.representation} minus {r.reference} | {r.position} | {r.median:.3f} | {r.lower:.3f} to {r.upper:.3f} |')
    lines += ['', 'All representation/common-response summaries are in multiscale_intervals.csv. '
        'Paired contrasts are in multiscale_difference_intervals.csv; preprocessing bundle contrasts are '
        'in preprocessing_difference_intervals.csv. Intervals must not be substituted for the earlier '
        'family-corrected p values.','',
        '## Preprocessing interpretation','',
        'The earlier comparison estimates the difference between complete processing bundles on a '
        'cohort selected by the supplied-pipeline QC. It does not establish equivalence, validate each '
        'step, isolate a causal processing mechanism or show that minimal data are artifact-free. '
        'The independent pre-voltage baseline from that comparison is retained; the older multiscale '
        'pre-voltage window still overlaps its baseline and is not a clean negative control. '
        'Stage ablations are deferred. Full-recording ICA remains descriptive and is not claimed as '
        'classifier-safe fitted preprocessing.','',
        '## Limits','',
        'Non-significance is not absence of waveform information. Group averages can miss participant-specific '
        'or latency-variable effects. Higher smoothed repeatability is not necessarily better discrimination '
        'or a neural signal. The tested scales do not establish true temporal scale invariance. '
        'A post-only significant result would not itself establish a post-minus-pre difference. '
        'Recorded-current timing remains unresolved. Conditional sensitivity calibration is a separate output.','',
        '## Figures','', '![Paired uncertainty](repeatability_intervals.png)']
    lines += [f'\n![{family} maps]({family}_expanded_maps.png)' for family in FAMILIES]
    text = '\n'.join(lines)+'\n'
    (out/'RESULTS.md').write_text(text,encoding='utf-8')
    pictures = ''.join(f'<img src="{name}">' for name in ['repeatability_intervals.png']+[f'{f}_expanded_maps.png' for f in FAMILIES])
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Response follow-up</title>'
        '<style>body{max-width:1250px;margin:30px auto;font:16px/1.5 system-ui}pre{white-space:pre-wrap}img{max-width:100%}</style>'
        '<pre>'+html.escape(text.split('## Figures')[0])+'</pre>'+pictures,encoding='utf-8')
    print(agg.to_string(),flush=True)


def plot_maps(pvalues,locations,family,out):
    fig,axes = plt.subplots(2,2,figsize=(12,6.5),layout='constrained')
    maximum = max(1.5,float(-np.log10(max(pvalues.min(),1e-10))))
    for b,branch in enumerate(BRANCHES):
        for p,period in enumerate(PERIODS):
            grid = np.arange(-480,-50) if period=='pre' else np.arange(50,480)
            image = np.full((5,len(grid)),np.nan)
            for s in range(5):
                mask = (locations['branch']==b)&(locations['period']==p)&(locations['scale']==s)
                times = np.rint(locations['time_s'][mask]*1000).astype(int)
                for time in np.unique(times):
                    image[s,time-grid[0]] = -np.log10(max(float(pvalues[mask][times==time].min()),1e-10))
            im = axes[b,p].imshow(image,origin='lower',aspect='auto',extent=[grid[0],grid[-1],-.5,4.5],
                vmin=0,vmax=maximum,cmap='viridis')
            axes[b,p].set(title=f'{branch} · {period}',xlabel='Time from trigger (ms)',ylabel='Gaussian sigma (ms)',
                yticks=range(5),yticklabels=[str(s) for s in SCALES],facecolor='.85')
    bar = fig.colorbar(im,ax=axes.ravel().tolist(),label='−log10(corrected p); min over electrodes')
    bar.ax.axhline(-np.log10(.05),color='red')
    fig.suptitle(f'{family}: full corrected family; grey = excluded by kernel support')
    fig.savefig(out/f'{family}_expanded_maps.png',dpi=150)
    plt.close(fig)


def plot_intervals(frame,out):
    fig,axes = plt.subplots(1,2,figsize=(11,5),layout='constrained',sharex=True)
    for ax,pos in zip(axes,[1,2]):
        sub = frame[(frame.position==pos)&(frame.metric=='waveform_deviation_r')].reset_index(drop=True)
        ax.hlines(np.arange(len(sub)),sub.lower,sub.upper,color='#4c78a8',lw=2)
        ax.scatter(sub['median'],np.arange(len(sub)),color='#4c78a8')
        ax.axvline(0,color='black',ls='--',lw=1)
        ax.set(yticks=np.arange(len(sub)),yticklabels=sub.representation+' − '+sub.reference,
            xlabel='Paired median difference in repeatability r',title=f'Position {pos}')
    fig.suptitle('Descriptive 95% participant-bootstrap intervals; no multiplicity adjustment')
    fig.savefig(out/'repeatability_intervals.png',dpi=150)
    plt.close(fig)
