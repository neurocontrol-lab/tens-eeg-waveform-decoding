"""Exact omnibus tests, corrected maps and fixed-channel derivative displays."""
import html
import json
from pathlib import Path
from experiments.provenance import resolve_source_path, source_matches
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.response_characterization.e002_electrode_position_slope.__main__ import sha, write_json
from .analysis import (SCALES_MS, BRANCHES, INTERVALS, FAMILIES,
                       sign_flip_family, family_adjustment)

WAVEFORMS = ('Sine','Square','Triangle','Linear decrease','Linear increase')


def render(out):
    out = Path(out)
    config = json.loads((out/'config.json').read_text())
    status = json.loads((out/'status.json').read_text())
    if status['errors'] or status['completed'] != config['subjects']:
        raise ValueError('Every requested participant must complete before group inference')
    subjects = status['completed']
    if len(subjects) < 3:
        raise ValueError('At least three participants are required for group inference')
    sources = {p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}
    # Extraction/inference sources are frozen; rendering may be repaired in place.
    for filename,digest in config['implementation'].items():
        if resolve_source_path(filename).resolve() == Path(__file__).resolve():
            continue
        if not source_matches(filename, digest):
            raise ValueError(f'Numerical source changed since extraction: {filename}')
    with np.load(out/f'sub-{subjects[0]}/effects.npz') as first:
        channels = first['channels'].tolist()
        plot_channels = first['plot_channels'].tolist()
        pre_times, post_times = first['pre_times'], first['post_times']
    expected_shape = (2,2,len(SCALES_MS),len(post_times),len(channels))
    saved_maps = dict(channels=np.asarray(channels), pre_times=pre_times, post_times=post_times)
    summary, significant, audits = [], [], []
    total_tests = np.prod(expected_shape)
    for family in FAMILIES:
        print(f'Group: {family}, {len(subjects)} participants, {total_tests:,} locations',flush=True)
        values = []
        for subject in subjects:
            with np.load(out/f'sub-{subject}/effects.npz') as data:
                np.testing.assert_array_equal(data['channels'], channels)
                np.testing.assert_array_equal(data['pre_times'], pre_times)
                np.testing.assert_array_equal(data['post_times'], post_times)
                values.append(data[family])
        coefs = np.stack(values)
        if coefs.shape[2:] != expected_shape:
            raise ValueError('Coefficient axes differ from declared branch/period/scale/time/channel axes')
        result = sign_flip_family(coefs)
        corrected = family_adjustment(result)
        saved_maps.update({f'{family}_{name}':value for name,value in result.items()})
        saved_maps[family+'_p_fwer'] = corrected
        saved_maps[family+'_mean_coefficients'] = coefs.mean(axis=0)
        if family == 'position':
            saved_maps['position_median_coefficients'] = np.median(coefs,axis=0)
        for b, branch in enumerate(BRANCHES):
            for p, period in enumerate(INTERVALS):
                for s, sigma in enumerate(SCALES_MS):
                    pv = corrected[b,p,s]
                    stat = result['statistic'][b,p,s]
                    ix = np.unravel_index(np.argmax(stat),stat.shape)
                    times = pre_times if period == 'pre' else post_times
                    summary.append(dict(effect=family,branch=branch,period=period,sigma_ms=sigma,
                        n_tests=pv.size,n_uncorrected_p_le_05=int((result['p_uncorrected'][b,p,s] <= .05).sum()),
                        n_corrected_p_le_05=int((pv <= .05).sum()),min_corrected_p=float(pv.min()),
                        max_statistic=float(stat[ix]),peak_statistic_time_ms=float(times[ix[0]]*1000),
                        peak_statistic_channel=channels[ix[1]],
                        n_subjects=len(subjects),n_sign_patterns=2**len(subjects)))
                    for t,c in np.argwhere(pv <= .05):
                        significant.append(dict(effect=family,branch=branch,period=period,sigma_ms=sigma,
                            time_ms=float(times[t]*1000),channel=channels[c],statistic=float(stat[t,c]),
                            p_uncorrected=float(result['p_uncorrected'][b,p,s,t,c]),p_fwer=float(pv[t,c])))
        plot_p_map(corrected, family, out, pre_times, post_times)
        del values, coefs
    np.savez_compressed(out/'group_tests.npz', **saved_maps)
    for subject in subjects:
        audit = json.loads((out/f'sub-{subject}/audit.json').read_text())
        audits.append({**audit['design'], 'subject':subject, 'n_events':audit['n_events'],
            'prior_feature_max_error':audit['prior_feature_max_error'],
            'n_padding_exclusions':audit['n_padding_exclusions'],'ica_refitted':audit['ica_refitted']})
    frame = pd.DataFrame(summary)
    frame.to_csv(out/'test_summary.csv',index=False)
    pd.DataFrame(significant,columns=('effect','branch','period','sigma_ms','time_ms','channel',
                                     'statistic','p_uncorrected','p_fwer')).to_csv(out/'significant_points.csv',index=False)
    audit_frame = pd.DataFrame(audits)
    audit_frame.to_csv(out/'participant_audit.csv',index=False)
    evoked = []
    for subject in subjects:
        with np.load(out/f'sub-{subject}/effects.npz') as data:
            evoked.append(data['evoked_display'].astype(float))
    evoked = np.asarray(evoked)
    # Remove each participant's five-waveform common response, separately by position.
    deviations = evoked-evoked.mean(axis=4,keepdims=True)
    np.savez_compressed(out/'group_display.npz',mean_evoked=evoked.mean(axis=0),
        mean_waveform_deviations=deviations.mean(axis=0),plot_channels=np.asarray(plot_channels),
        pre_times=pre_times,post_times=post_times)
    for period_index,period in enumerate(INTERVALS):
        for channel_index,channel in enumerate(plot_channels):
            plot_evoked(deviations.mean(axis=0),period_index,channel_index,channel,period,out,
                        pre_times if period == 'pre' else post_times)
    by_family = frame.groupby('effect').agg(n_tests=('n_tests','sum'),
        n_significant=('n_corrected_p_le_05','sum'),min_p=('min_corrected_p','min'))
    lines = ['# Task 4 time-resolved Gaussian-derivative effects','',
        f'Completed {len(subjects)} participants; {int(audit_frame.n_events.sum())} frozen matched events. '
        'No new ICA fit, trial-label shuffling or classifier.','',
        '## Fixed analysis','',
        'Five Gaussian first derivatives: σ=2.5/5/10/20/40 ms, calibrated in µV/s. '
        'Both minimal and supplied preprocessing; 62 scalp channels. '
        'Post [180,480) ms and pre [−480,−180) ms each contain 300 samples at 1 kHz. '
        'Actual epoch padding is −900 to +650 ms. At the largest scale, local kernel support '
        'extends no earlier than +20 ms post-trigger and no later than −20 ms pre-trigger. '
        'This excludes the explicit −2 to +10-ms blank but cannot undo prior continuous-filter spread. '
        'Constant baseline subtraction does not affect these derivative operators.','',
        'For every participant/branch/period/scale/electrode/time point, fit the preceding OLS design: '
        'comparison intercepts, within-comparison centered elapsed time, sum-coded waveform, '
        'position and waveform × position. Waveform effects average over positions; position is '
        'second minus first averaged equally over waveforms. Four waveform/interaction coefficients '
        'are jointly tested using the sum of squared group t-statistics. This is a regression-based '
        'omnibus statistic, not a parametric F-distribution test.','',
        f'Enumerate all {2**len(subjects):,} participant sign patterns, flipping each participant’s '
        'entire contrast vector/map together. Max-statistic correction spans both branches, '
        'both periods, five scales, 300 time points and 62 electrodes within each effect family '
        '(372,000 locations at defaults). A further Bonferroni factor of three covers waveform, '
        'position and interaction. All raw and corrected p maps are saved in group_tests.npz; '
        'test_summary.csv gives every branch/period/scale summary.','',
        '## Corrected effects','',
        '| Effect | Points with corrected p ≤ .05 | Smallest corrected p |',
        '| --- | --- | --- |']
    for family,row in by_family.iterrows():
        lines.append(f'| {family} | {int(row.n_significant):,} / {int(row.n_tests):,} | {row.min_p:.4f} |')
    lines += ['', '## Verification and interpretation','',
        f'Prior supplied fixed-feature reproduction maximum error: {audit_frame.prior_feature_max_error.max():.3g}. '
        f'New padding exclusions: {int(audit_frame.n_padding_exclusions.sum())}. '
        f'Design condition numbers: {audit_frame.condition_number.min():.2f}–{audit_frame.condition_number.max():.2f}.','',
        '- These are model-conditional tests of consistent group effects, assuming independent participants '
        'and joint symmetry of each participant’s coefficient-error vector under the null. No arbitrary '
        'within-participant label exchangeability is assumed. They do not establish neural origin or causal waveform effects.',
        '- Group means can miss participant-specific effects or latency differences. Broad correction and '
        'ten participants limit sensitivity. Failure to reject is not equivalence or evidence of absent information.',
        '- Pre-trigger responses can contain expectation and previous-stimulus carryover. A post effect '
        'without a significant pre effect is not itself a significant post-minus-pre difference.',
        '- Physical trigger-to-current delay remains unverified. Processing was reconstructed using full-recording '
        'ICA for descriptive analysis. These tests do not measure held-out decoding, true scale invariance or responses before 180 ms.',
        '- All three reported p-map families include both branches and periods in their correction. '
        'Uncorrected p-values are not an alternate minimally processed data branch.',
        '', '## Figures','',
        'P maps show the smallest corrected p across electrodes at each time/scale. '
        'Their channel reduction is descriptive; correction already included all channels. '
        'A nonzero color is not a significance declaration; the color bar marks −log10(.05).','']
    for family in FAMILIES:
        lines.append(f'![{family} corrected maps]({family}_p_maps.png)\n')
    lines += ['Fixed-channel waveform-deviation plots are descriptive group means; '
              'their equal-observed-comparison weighting differs from nuisance-adjusted regression coefficients. '
              'Axes are shared between branches and positions within each scale/period/channel.','']
    for period in INTERVALS:
        for channel in plot_channels:
            lines.append(f'![{channel} {period} derivatives]({channel}_{period}_derivatives.png)\n')
    text = '\n'.join(lines)+'\n'
    (out/'RESULTS.md').write_text(text,encoding='utf-8')
    sections = ['<!doctype html><html><head><meta charset="utf-8"><title>Task 4 time-resolved effects</title>',
        '<style>body{font:16px system-ui;max-width:1250px;margin:30px auto;padding:0 20px} '
        'pre{white-space:pre-wrap;line-height:1.5}img{max-width:100%}</style></head><body>',
        '<h1>Task 4 time-resolved effects</h1><pre>'+html.escape(text.split('## Figures')[0])+'</pre>',
        '<p>Corrected maps summarize the minimum corrected p across channels. '
        'Waveform traces show descriptive group mean deviations, not significance.</p>']
    for name in [f'{f}_p_maps.png' for f in FAMILIES]+[
            f'{c}_{p}_derivatives.png' for p in INTERVALS for c in plot_channels]:
        sections.append(f'<h2>{html.escape(name)}</h2><img src="{name}" alt="{name}">')
    sections.append('</body></html>')
    (out/'index.html').write_text('\n'.join(sections),encoding='utf-8')
    write_json(out/'render_provenance.json',dict(sources=sources,
        inputs={str(s):sha(out/f'sub-{s}/effects.npz') for s in subjects},
        n_sign_patterns=2**len(subjects),n_locations_per_family=int(total_tests),
        n_families=3,strict_pointwise_fwer=True,cluster_inference=False))
    print(by_family.to_string(),flush=True)


def plot_p_map(pvalues, family, out, pre_times, post_times):
    fig, axes = plt.subplots(2,2,figsize=(12,6.8),layout='constrained')
    maximum = max(1.5,float(-np.log10(np.maximum(pvalues.min(),1e-6))))
    for b,branch in enumerate(BRANCHES):
        for p,period in enumerate(INTERVALS):
            times = (pre_times if period == 'pre' else post_times)*1000
            ax = axes[b,p]
            image = ax.imshow(-np.log10(np.maximum(pvalues[b,p].min(axis=-1),1e-6)),
                origin='lower',aspect='auto',extent=[times[0],times[-1],-.5,4.5],
                vmin=0,vmax=maximum,cmap='viridis')
            ax.set_yticks(range(5),[str(s) for s in SCALES_MS])
            ax.set(xlabel='Time from trigger (ms)',ylabel='Gaussian σ (ms)',title=f'{branch} · {period}')
    bar = fig.colorbar(image,ax=axes.ravel().tolist(),label='−log10(corrected p); min over electrodes')
    bar.ax.axhline(-np.log10(.05),color='red',linewidth=2)
    fig.suptitle(f'{family}: corrected time/scale maps (threshold p=.05 marked red on bar)')
    fig.savefig(out/f'{family}_p_maps.png',dpi=150)
    plt.close(fig)


def plot_evoked(deviations, period_index, channel_index, channel, period, out, times):
    fig, axes = plt.subplots(5,4,figsize=(15,12),layout='constrained',sharex=True)
    colors = plt.get_cmap('tab10').colors
    for s,sigma in enumerate(SCALES_MS):
        row_values = deviations[:,period_index,s,:,:,channel_index,:]
        bound = max(1e-9,float(np.abs(row_values).max()))*1.08
        for b,branch in enumerate(BRANCHES):
            for position in range(2):
                ax = axes[s,2*b+position]
                for w,name in enumerate(WAVEFORMS):
                    ax.plot(times*1000,deviations[b,period_index,s,w,position,channel_index],
                            color=colors[w],label=name,linewidth=1.1)
                ax.axhline(0,color='.65',linewidth=.6)
                ax.set_ylim(-bound,bound)
                if s == 0:
                    ax.set_title(f'{branch} · {"first" if position == 0 else "second"}')
                if b == 0 and position == 0:
                    ax.set_ylabel(f'σ={sigma:g} ms\nµV/s')
                if s == 4:
                    ax.set_xlabel('Time from trigger (ms)')
    handles,labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=5)
    fig.suptitle(f'{channel} · {period}: waveform-specific derivative deviations\n'
                 'Descriptive group means; shared y-limits per scale; no significance implied')
    fig.savefig(out/f'{channel}_{period}_derivatives.png',dpi=140)
    plt.close(fig)


if __name__ == '__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):
        render(Path(sys.argv[1]))
