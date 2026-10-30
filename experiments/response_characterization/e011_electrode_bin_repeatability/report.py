"""Descriptive repeatability maps, participant uncertainty and support reports."""
import argparse
import html
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .core import WIDTHS_MS,REPRESENTATIONS,UNITS,group_interval,SEED
from .__main__ import sha,write_json


def heatmaps(results,representation,metric,out,channels):
    fig,axes = plt.subplots(2,6,figsize=(22,8),sharey=True)
    if metric == 'r':
        key,limit,label = 'mean_r',1.,'Mean participant median split-profile r'
    else:
        key = 'mean_product'
        limit = max(float(np.quantile(np.concatenate([np.abs(v[key]).ravel() for v in results]),.99)),1e-12)
        label = 'Mean participant split-product ('+UNITS[representation]+')^2'
    for col,(width,result) in enumerate(zip(WIDTHS_MS,results)):
        for p in range(2):
            ax = axes[p,col]
            image = ax.imshow(result[key][p],aspect='auto',origin='upper',interpolation='nearest',
                extent=[result['left_ms'][0],result['right_ms'][-1],len(channels)-.5,-.5],
                cmap='RdBu_r',vmin=-limit,vmax=limit)
            ax.axvline(1050,color='k',ls=':',lw=.7)
            if result['right_ms'][-1] < 1188:
                ax.axvspan(result['right_ms'][-1],1188,color='.92')
            ax.set(title=f'{width} ms / presentation {p+1}',xlim=(50,1188),xlabel='Trigger-relative ms')
            ax.set_yticks(range(len(channels)),channels)
    fig.suptitle(f'{representation}: electrode-by-bin repeatability\n'
        'Fixed local derivative window: 25 ms; 1-ms grid is not independent temporal resolution. Grey tails contain no complete bin.',fontsize=12)
    fig.tight_layout(rect=(0,0,.94,.92))
    fig.colorbar(image,ax=axes.ravel().tolist(),location='right',fraction=.02,pad=.02,label=label)
    path = out/'figures'/f'{representation}_{metric}.png'
    fig.savefig(path,dpi=125,bbox_inches='tight');plt.close(fig)


def interval_plot(result,representation,out,channels):
    chosen = ('C3','C4','CP1','CP2')
    fig,axes = plt.subplots(4,2,figsize=(13,11),sharex=True,sharey=True)
    time = .5*(result['left_ms']+result['right_ms'])
    for row,ch in enumerate(chosen):
        c = channels.index(ch)
        for p in range(2):
            ax = axes[row,p]
            ax.plot(time,result['mean_r'][p,c],color='#0072B2',lw=.9)
            ax.fill_between(time,result['r_low'][p,c],result['r_high'][p,c],color='#0072B2',alpha=.18)
            ax.axhline(0,color='.4',lw=.7)
            ax.axvline(1050,color='.4',ls=':',lw=.7)
            ax.set(title=f'{ch}, presentation {p+1}',ylim=(-1,1),ylabel='Split-profile r',xlim=(50,1188))
            ax.grid(alpha=.15)
            if row == 3: ax.set_xlabel('Trigger-relative ms')
    fig.suptitle(representation+': native-grid repeatability\nPointwise descriptive 95% participant-bootstrap bands; no multiplicity-corrected significance claim.',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.94))
    fig.savefig(out/'figures'/f'{representation}_intervals.png',dpi=125,bbox_inches='tight');plt.close(fig)


def render(out):
    out = Path(out)
    config = json.loads((out/'config.json').read_text())
    status = json.loads((out/'status.json').read_text())
    if not status['complete'] or status['errors'] or status['completed'] != config['subjects']:
        raise ValueError('Require all requested participants to complete before group summaries')
    for source,digest in {**config['input_sha256'],**config['implementation_sha256']}.items():
        if Path(source).resolve() != Path(__file__).resolve() and sha(source)!=digest:
            raise ValueError(f'Frozen numerical input/source changed: {source}')
    subjects = status['completed']
    if len(subjects) < 2:
        raise ValueError('At least two participants are required for group bootstrap intervals; use --prepare-only for a smoke run')
    channels = config['channels']
    (out/'figures').mkdir(exist_ok=True)
    (out/'group').mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    indices = rng.integers(len(subjects),size=(1000,len(subjects)))
    np.save(out/'participant_bootstrap_indices.npy',indices)
    table_rows = []
    for representation in REPRESENTATIONS:
        results = []
        for width in WIDTHS_MS:
            source_maps = []
            for subject in subjects:
                with np.load(out/f'sub-{subject}'/f'w{width:03d}_{representation}.npz') as data:
                    source_maps.append({k:data[k].copy() for k in data.files})
            left,right = source_maps[0]['left_ms'],source_maps[0]['right_ms']
            assert all(np.array_equal(m['left_ms'],left) and np.array_equal(m['right_ms'],right) for m in source_maps)
            r = np.stack([m['median_split_r'] for m in source_maps])
            product = np.stack([m['mean_split_product'] for m in source_maps])
            mean_r,r_low,r_high,r_n = group_interval(r,indices)
            mean_product,product_low,product_high,product_n = group_interval(product,indices)
            result = dict(mean_r=mean_r,r_low=r_low,r_high=r_high,r_n=r_n,
                mean_product=mean_product,product_low=product_low,product_high=product_high,product_n=product_n,
                left_ms=left,right_ms=right,subject_r=r,subject_product=product)
            np.savez_compressed(out/'group'/f'w{width:03d}_{representation}.npz',**result)
            results.append(result)
            # Fixed 1-s support for comparing reporting widths. Equal electrodes/bins,
            # then equal participants; do not choose bins from observed peaks.
            mask = (left >= 50) & (right <= 1050)
            for p in range(2):
                r_summary = np.nanmean(r[:,p][:,:,mask],axis=(1,2))
                product_summary = product[:,p][:,:,mask].mean(axis=(1,2))
                r_est = group_interval(r_summary,indices)
                product_est = group_interval(product_summary,indices)
                table_rows.append(dict(representation=representation,width_ms=width,position=p+1,
                    n_subjects=len(subjects),n_bins=int(mask.sum()),mean_r=float(r_est[0]),
                    r_low=float(r_est[1]),r_high=float(r_est[2]),
                    mean_product=float(product_est[0]),product_low=float(product_est[1]),product_high=float(product_est[2]),
                    unit=UNITS[representation]))
        heatmaps(results,representation,'r',out,channels)
        heatmaps(results,representation,'product',out,channels)
        interval_plot(results[0],representation,out,channels)
        print(f'Group maps/intervals: {representation}',flush=True)
    summary = pd.DataFrame(table_rows)
    summary.to_csv(out/'one_second_group_summary.csv',index=False)
    audits = [json.loads((out/f'sub-{s}'/'audit.json').read_text()) for s in subjects]
    pd.DataFrame(audits).drop(columns='full_design').to_csv(out/'participant_audit.csv',index=False)
    source_pairs = sum(a['source_pairs'] for a in audits)
    pairs = sum(a['n_pairs'] for a in audits)
    totals = dict(n_subjects=len(subjects),source_pairs=source_pairs,repeatability_pairs=pairs,
        repeatability_events=2*pairs,singleton_stratum_pairs_omitted=source_pairs-pairs,
        min_half_cell_events=min(a['min_half_cell_events'] for a in audits),
        max_split_condition=max(a['maximum_split_condition'] for a in audits),
        maximum_source_reconstruction_error=max(a['source_mean_max_error_uv'] for a in audits))
    write_json(out/'aggregate_audit.json',totals)
    lines = ['# E011 - Electrode-by-time-bin waveform repeatability','',
        f'{len(subjects)} participants; {pairs} complete pairs / {2*pairs} events. '
        f'{source_pairs-pairs} ordered-singleton-stratum pairs omitted from the E010 cohort for repeatability only.','',
        'Simple 0.5-30 Hz band-pass and baseline; no ICA or new amplitude exclusions. '
        'Each trial supplies voltage, signed/RMS slope and signed/RMS curvature; '
        'local derivatives use a fixed 25-sample quadratic fit. '
        'Both members of each pair stay together in 100 stratified splits. '
        'Comparison, elapsed time, position and waveform-by-position regression are fitted independently in each half.','',
        '## Fixed one-second summary','',
        'Average per-location split-profile r over all 14 electrodes and complete bins in [50,1050) ms, '
        'then equal participant weighting. Intervals resample participants, not overlapping splits. '
        'These averages are not full-epoch pattern correlations or classifier accuracy.','',
        '| Representation | Bin ms | Position | Mean location r | Descriptive 95% interval |',
        '| --- | ---: | ---: | ---: | --- |']
    for row in summary.itertuples():
        lines.append(f'| {row.representation} | {row.width_ms} | {row.position} | {row.mean_r:+.3f} | [{row.r_low:+.3f}, {row.r_high:+.3f}] |')
    limitations = ('This exploratory analysis does not report corrected significant channels/bins, select a best resolution, '
        'fit a classifier or establish neural origin. Correlation uses only five adjusted waveform values and can be unstable. '
        'Cross-products retain squared physical units, can be negative and cannot be ranked across derivative orders. '
        'Shared temporal drift, filtering/artifacts and misspecified adjustment can bias split estimates. '
        'Halves share comparison contexts; repeatability need not transfer to held-out comparisons. '
        'A positive pointwise bootstrap interval is not a multiplicity-corrected finding. '
        '1-ms sampling is not independent 1-ms temporal resolution after filtering/local fitting. '
        'Coarse signed means may cancel opposite deflections; RMS can reflect noise as well as response dynamics. '
        'Derivative padding is real; filtering may still spread stimulation artifacts. '
        'No matched pre-trigger repeatability test is supplied here. Physical current timing remains unverified. '
        'Global E011 maps must not select features for the same held-out E012 evaluation.')
    lines += ['', '## Interpretation limits','',limitations,'',
        'Maps span complete post bins beginning at 50 ms; endpoints differ by width '
        '(1188/1185/1175/1150/1150/1050 ms). The fixed one-second summary equalizes coverage. '
        'Common-response and full adjusted-waveform estimates are saved separately for interpretation.','',
        'See index.html for maps and pointwise intervals. Numeric group NPZ files retain participant maps; '
        'split allocations and cell/design audits make the analysis reproducible.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    body = ['<!doctype html><meta charset="utf-8"><title>E011 repeatability</title>',
        '<style>body{max-width:1600px;margin:2em auto;padding:1em;font:16px system-ui}img{max-width:100%}p{line-height:1.6}details{margin:1.5em 0}summary{cursor:pointer;font-weight:bold}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:.4em}</style>',
        '<h1>E011: electrode-by-time-bin waveform repeatability</h1>',
        f'<p>{len(subjects)} participants; {pairs} pairs; {2*pairs} events. '
        f'{source_pairs-pairs} singleton-stratum pairs omitted from E010 for splitting only.</p>',
        '<p>'+html.escape(limitations)+'</p>',
        '<h2>Fixed one-second summary</h2><p>Equal-location, then equal-participant averages over [50,1050) ms. '
        'No resolution or feature is selected. Intervals are descriptive, pointwise participant-bootstrap intervals.</p>',
        summary[['representation','width_ms','position','mean_r','r_low','r_high']].to_html(index=False,float_format=lambda x:f'{x:+.3f}'),
        '<h2>Electrode-by-bin maps</h2>']
    for representation in REPRESENTATIONS:
        body.append(f'<h3>{representation}</h3>')
        for suffix,label in [('r','Split-profile correlation'),('product','Amplitude-sensitive split product'),('intervals','Native-grid curves and pointwise intervals: fixed C3/C4/CP1/CP2')]:
            file = f'{representation}_{suffix}.png'
            body.append(f'<details><summary>{label}</summary><a href="figures/{file}">Full-resolution figure</a><img loading="lazy" src="figures/{file}"></details>')
    body += ['<h2>Audit files</h2><p><a href="aggregate_audit.json">Aggregate audit</a> | '
        '<a href="one_second_group_summary.csv">Group summary and product intervals</a> | '
        '<a href="participant_audit.csv">Participant audit</a> | <a href="RESULTS.md">Full methods/limitations</a></p>']
    (out/'index.html').write_text('\n'.join(body),encoding='utf-8')
    write_json(out/'render_provenance.json',dict(report_sha256=sha(Path(__file__)),
        input_config_sha256=sha(out/'config.json'),participants=subjects,bootstrap_draws=1000,
        bootstrap_seed=SEED,intervals='Pointwise descriptive; not corrected significance',
        classifier_fitted=False,features_selected=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    with threadpool_limits(limits=2):
        render(parser.parse_args().output)
