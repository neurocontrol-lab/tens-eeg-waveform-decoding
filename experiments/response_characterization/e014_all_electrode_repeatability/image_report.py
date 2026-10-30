"""Publish frozen E014 results as images, without browser numerical payloads."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from .report import LABELS

METRICS = ['mean_split_product', 'full_response', 'rms_half_disagreement', 'split_sign_agreement']
METRIC_LABELS = ['Mean split product A x B', 'Signed response estimate',
                 'RMS half-to-half disagreement', 'Fraction of splits with matching signs']
FIELDS = METRICS + ['example_half_a', 'example_half_b', 'temporal_profile_r', 'joint_profile_r']
WAVES = ['Sine', 'Square', 'Triangle', 'LinDec', 'LinInc']
WIDTH, ROW, CURVE_HEIGHT = 1000, 14, 145
BG, WHITE, TEXT, AXIS, BLUE, ORANGE, GREY = range(242,249)
FONT_PATH = fm.findfont('DejaVu Sans')
FONT = ImageFont.truetype(FONT_PATH,11)
SMALL = ImageFont.truetype(FONT_PATH,10)
TITLE = ImageFont.truetype(FONT_PATH,13)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def palette():
    colors=[]
    for i in range(121):
        t=(i-60)/60
        colors.append([round(250-70*t),round(250-205*t),round(250-205*t)] if t>=0 else
                      [round(250+205*t),round(250+135*t),round(250+65*t)])
    for i in range(121):
        t=i/120;colors.append([round(247-190*t),round(250-140*t),round(252-70*t)])
    colors += [[233,237,241],[255,255,255],[36,55,75],[188,198,209],
               [0,114,178],[213,94,0],[138,150,165]]
    colors += [[255,255,255]]*(256-len(colors))
    return np.array(colors,dtype=np.uint8)


PALETTE=palette()


def image(size):
    result=Image.new('P',size,WHITE);result.putpalette(PALETTE.ravel().tolist());return result


def number(x):
    if not np.isfinite(x):return 'undefined'
    return f'{x:.2e}' if abs(x)<1e-3 or abs(x)>=1e4 else f'{x:.3f}'


def raster(values,left,right,limit,positive,width=370):
    """Rasterize existing bins at fixed display pixels; never rebin statistics."""
    times=50+(np.arange(width)+.5)*1138/width
    indices=np.searchsorted(left,times,side='right')-1
    valid=(indices>=0)&(indices<len(left))
    safe=np.clip(indices,0,len(left)-1);valid &= times<right[safe]
    selected=values[:,safe]
    finite=np.isfinite(selected)&valid[None,:]
    scaled=np.nan_to_num(selected/limit,nan=0,posinf=0,neginf=0)
    codes=(121+np.rint(np.clip(scaled,0,1)*120) if positive else
           np.rint((np.clip(scaled,-1,1)+1)*60)).astype(np.uint8)
    codes[~finite]=BG
    return np.repeat(codes,ROW,axis=0)


def ticks(draw,x,y,w):
    for t in [50,250,450,650,850,1050,1188]:
        draw.text((x+(t-50)/1138*w,y),str(t),font=SMALL,fill=TEXT,anchor='mt')


def map_atlas(arrays,left,right,channels,limits,wave,unit,estimate):
    height=len(channels)*ROW+90
    result=image((WIDTH,4*height));draw=ImageDraw.Draw(result)
    for m,metric in enumerate(METRICS):
        top=m*height;positive=m in (2,3);limit=limits[m]
        suffix='dependent-split fraction' if m==3 else ('('+unit+')^2' if m==0 else unit)
        label=estimate if m==1 else METRIC_LABELS[m]
        draw.text((12,top+5),f'{label} | scale {"0 to " if positive else "+/- "}{number(limit)} {suffix}',font=TITLE,fill=TEXT)
        for p in range(2):
            x=p*500+112;y=top+40;w=370;h=len(channels)*ROW
            draw.text((p*500+250,top+23),f'Presentation {p+1}',font=FONT,fill=TEXT,anchor='mt')
            panel=Image.fromarray(raster(arrays[metric][p,wave],left,right,limit,positive))
            panel.putpalette(PALETTE.ravel().tolist());result.paste(panel,(x,y))
            for c,channel in enumerate(channels):
                display_channel=channel.replace(' component ',' c')
                draw.text((x-6,y+c*ROW+ROW/2),display_channel,font=FONT,fill=TEXT,anchor='rm')
            draw.rectangle((x,y,x+w,y+h),outline=AXIS)
            cutoff=x+1000/1138*w
            for dy in range(0,h,8):draw.line((cutoff,y+dy,cutoff,y+min(dy+4,h)),fill=AXIS)
            ticks(draw,x,y+h+7,w)
            draw.text((x+w/2,y+h+25),'Time from recorded trigger (ms)',font=FONT,fill=TEXT,anchor='mt')
    return result,height


def curve_atlas(arrays,left,right,channels,wave):
    result=image((WIDTH,len(channels)*CURVE_HEIGHT));draw=ImageDraw.Draw(result)
    t=(left+right)/2
    for c,channel in enumerate(channels):
        for p in range(2):
            x=p*500+65;y=c*CURVE_HEIGHT+29;w=417;h=80
            r=arrays['temporal_profile_r'][p,wave,c]
            draw.text((p*500+12,c*CURVE_HEIGHT+5),f'{channel} | P{p+1} | median temporal r = {number(r)}',font=FONT,fill=TEXT)
            full=arrays['full_response'][p,wave,c];a=arrays['example_half_a'][p,wave,c];b=arrays['example_half_b'][p,wave,c]
            limit=max(float(np.nanmax(np.abs(np.stack([full,a,b]))))*1.1,1e-9)
            draw.rectangle((x+1000/1138*w,y,x+w,y+h),fill=BG)
            draw.line((x,y+h/2,x+w,y+h/2),fill=AXIS)
            for values,color in [(full,GREY),(a,BLUE),(b,ORANGE)]:
                points=np.column_stack([x+(t-50)/1138*w,y+h/2-values/limit*h/2])
                good=np.isfinite(points).all(axis=1)
                # Disconnected finite runs avoid drawing across undefined samples.
                for run in np.split(np.flatnonzero(good),np.flatnonzero(np.diff(np.flatnonzero(good))>1)+1):
                    if len(run)>1:draw.line([tuple(v) for v in points[run]],fill=color,width=1)
            for value in [-limit,0,limit]:draw.text((x-5,y+h/2-value/limit*h/2),number(value),font=SMALL,fill=TEXT,anchor='rm')
            ticks(draw,x,y+h+6,w)
    return result


def save_image(im,path):
    # Lossless WebP compresses the plotted pixels, not the scientific arrays.
    im.convert('RGB').save(path,format='WEBP',lossless=True,method=3)


PAGE=r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>E014 — Image report</title><style>
body{margin:0;background:#f5f7fa;color:#172437;font:15px system-ui}main{max-width:1200px;margin:auto;padding:24px}h1{font-size:28px}h2{font-size:21px}p{line-height:1.55}.panel{background:white;border:1px solid #dbe1e9;border-radius:9px;padding:18px;margin:16px 0}.controls{display:flex;gap:12px;flex-wrap:wrap}label{display:flex;flex-direction:column;gap:5px;font-size:13px;font-weight:600}select{font:15px system-ui;padding:8px;max-width:390px}.note{font-size:13px;color:#526379}.clip{position:relative;overflow:hidden;background:white;max-width:100%}.atlas{position:absolute;left:0;top:0;max-width:none;width:100%;height:auto}.image-shell{width:100%;max-width:1000px}table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:8px;border-bottom:1px solid #e5e9ee}.selected{background:#e8f2fc}summary{cursor:pointer;font-weight:600}.error{color:#b71c1c}.button{display:inline-block;padding:6px 10px;border:1px solid #adb8c7;border-radius:5px;color:#0b5a96;text-decoration:none}noscript{display:block;padding:12px;background:#fff3d4}
</style></head><body><main>
<h1>E014 — Individual-waveform repeatability</h1>
<p>Image edition. Choose a participant and waveform; both stay separate, as do first and second presentations. All 62 electrodes and all ten feature representations are available. The page displays pre-rendered images and small summary tables. It does not download EEG epochs or numerical map arrays.</p>
<noscript>Use the <a href="gallery.html">static image gallery</a> to browse without JavaScript.</noscript>
<div class="controls panel"><label>Participant<select id="participant"></select></label><label>Waveform<select id="waveform"></select></label><label>Representation<select id="representation"></select></label><label>Bin / spectral window<select id="width"></select></label><label>Response view<select id="view"><option value="specific">Waveform-specific deviation</option><option value="total">Total observed response</option></select></label></div>
<p id="status" class="note" aria-live="polite"></p><p id="method" class="panel note"></p>
<section class="panel"><h2>All five waveforms — selected participant only</h2><p class="note">Median held-out joint pattern r, across 100 paired splits. Total-response agreement can reflect shared responses or artifacts; it is not evidence of waveform distinctiveness. No significance threshold is applied.</p><div id="summary"></div></section>
<section class="panel"><h2 id="mapTitle">Electrode / feature × time maps</h2><label>Map statistic<select id="metric"></select></label><p class="note">The two columns show first and second presentation. Color limits match the local report: shared across all five waveforms and both positions, clipped visually at the 99th-percentile magnitude (sign agreement uses 0–1). Grey has no reported bin/window centre. Dashed line: 1050 ms.</p><div class="image-shell"><div id="mapClip" class="clip"><img id="mapImage" class="atlas" alt="Selected waveform, both presentations, all electrodes or learned components"></div></div><p><a id="mapLink" class="button" target="_blank" rel="noopener">Open full-resolution map sheet</a></p></section>
<section class="panel"><h2>Inspect the two halves at one electrode / component</h2><label>Electrode / component<select id="channel"></select></label><p id="quality" class="note"></p><p class="note">Blue/orange: preselected split 0, A/B. Grey: full-data estimate for sensor/power views; split-0 held-out A/B average for matched calibration views. Each panel has its own vertical range, in the units shown above. The printed r summarizes all 100 splits over [50,1050) ms; the shaded tail is outside that summary.</p><div class="image-shell"><div id="curveClip" class="clip"><img id="curveImage" class="atlas" alt="Selected electrode, both presentation positions, fixed split-zero traces"></div></div><p><a id="curveLink" class="button" target="_blank" rel="noopener">Open all-electrode curve sheet</a></p></section>
<details class="panel"><summary>Methods, interpretation and image resolution</summary>
<p>Ten participants, 595 complete pairs, 1,190 events; simple 0.5–30 Hz filtering, average reference and [-250,-50)-ms baseline. No ICA refit, new exclusions, classifier, peak selection or latency alignment. The original 14-channel results reproduce the preserved E011 reference.</p>
<p>Waveform-specific deviations use half-specific adjustment for comparison context, elapsed time and presentation position. Total responses retain observed comparison-cell means. A repeatable total response need not distinguish waveform identity or have neural origin.</p>
<p>Voltage and first/second derivatives retain 1/5/25/50/100/200-ms bins; derivatives use the fixed 25-ms local estimator. 1-ms spacing does not imply independent physiological resolution. Images have fixed pixel resolution; this affects display only, not the estimates or reported correlations. Palette colors are discretized for rendering. WebP compression is lossless for the rendered pixels. The full-precision numerical report is retained locally.</p>
<p>Alpha/beta use eleven overlapping 500-ms Hann total-power windows with a 50-ms step and 2-Hz Fourier spacing. Their plotted centres are 300–800 ms; each marker summarizes a 500-ms window, not 50-ms independent activity. These are filtered total-power features, not induced power.</p>
<p>xDAWN fits only independent calibration pairs (two components per waveform, regularization 0.1). Both A/B halves are held out. Matched 62- and 14-sensor controls share the same held-out trials and calibration-only RMS normalization. Joint shape r removes each feature's temporal mean. Compare these three matched views to one another. Learned component names denote class/rank slots, not fixed electrodes or anatomical sources.</p>
<p>Channel-quality flags are descriptive and remove no channels or pairs. Absence of a flag does not establish neural origin. Separate participant/waveform patterns remain primary; this edition adds no group averaging, corrected feature discovery or decoding result.</p>
<p><a href="gallery.html">Static gallery</a> · <a href="manifest.json">Image/provenance manifest</a> · <a href="README.md">Sharing and regeneration</a></p></details>
</main><script>
const catalog=__CATALOG__;
const el=id=>document.getElementById(id);
const metricLabels=__METRIC_LABELS__;
for(const s of catalog.subjects)el('participant').add(new Option('Subject '+s,s));
catalog.waves.forEach((w,i)=>el('waveform').add(new Option(w,i)));
for(const [r,v] of Object.entries(catalog.representations))el('representation').add(new Option(v.label,r));
metricLabels.forEach((m,i)=>el('metric').add(new Option(m,i)));
let current=null;
function menus(){const rep=catalog.representations[el('representation').value],old=el('width').value;el('width').innerHTML='';for(const w of rep.widths)el('width').add(new Option(w===500?'500-ms window / 50-ms step':w+' ms',w));el('width').value=rep.widths.includes(Number(old))?old:String(rep.widths.includes(50)?50:rep.widths[0]);const ch=el('channel').value;el('channel').innerHTML='';for(const name of rep.channels)el('channel').add(new Option(name,name));el('channel').value=rep.channels.includes(ch)?ch:(rep.channels.includes('C3')?'C3':rep.channels[0]);draw();}
function cropped(id,src,top,height){const im=el(id+'Image'),clip=el(id+'Clip');im.src=src;clip.style.aspectRatio='1000 / '+height;im.style.transform='translateY('+(-top/(id==='map'?4*height:current.channels.length*catalog.curve_height)*100)+'%)';el(id+'Link').href=src;}
function draw(){const s=el('participant').value,r=el('representation').value,w=el('width').value,v=el('view').value,wave=Number(el('waveform').value);const rep=catalog.representations[r];current=catalog.entries[[s,r,w,v].join(':')];if(!current){el('status').textContent='This combination is not in this image export.';return;}const base='images/'+s+'_'+r+'_'+w+'_'+v+'_wave'+wave;el('mapTitle').textContent=catalog.waves[wave]+' — Subject '+s+' — '+rep.label;el('method').textContent=rep.method+' Units: '+rep.unit+'.';const c=rep.channels.indexOf(el('channel').value);cropped('map',base+'_maps.webp',Number(el('metric').value)*current.map_height,current.map_height);cropped('curve',base+'_curves.webp',c*catalog.curve_height,catalog.curve_height);const flags=catalog.quality[s]?.[el('channel').value];el('quality').textContent=r==='xdawn_components'?'Learned component, not a fixed scalp electrode.':(flags?'Quality note: '+flags:'No recorded amplitude flag for this channel; artifacts remain possible.');let table='<table><tr><th>Waveform</th><th>P1 joint r</th><th>P2 joint r</th><th>Available trials P1 / P2</th><th>Minimum A/B trials P1 / P2</th></tr>';for(let i=0;i<5;i++){const a=current.support.find(x=>x.position===1&&x.trigger===239+i),b=current.support.find(x=>x.position===2&&x.trigger===239+i);const f=x=>x===null?'undefined':x.toFixed(3);table+='<tr'+(i===wave?' class="selected"':'')+'><td>'+catalog.waves[i]+'</td><td>'+f(current.joint[0][i])+'</td><td>'+f(current.joint[1][i])+'</td><td>'+a.n_events+' / '+b.n_events+'</td><td>'+a.minimum_half_events+' / '+b.minimum_half_events+'</td></tr>';}el('summary').innerHTML=table+'</table>';el('status').textContent='Pre-rendered images; no EEG or numerical map-array download. Full-resolution sheets open in a new tab.';}
el('representation').addEventListener('change',menus);for(const id of ['participant','waveform','width','view','metric','channel'])el(id).addEventListener('change',draw);
for(const id of ['mapImage','curveImage'])el(id).addEventListener('error',()=>{el('status').textContent='Image could not load. Keep index.html and the companion images folder together, or use the static gallery.';});
menus();
</script></body></html>'''


def export(source,destination,subjects=None,representations=None,widths=None):
    source=Path(source);destination=Path(destination)
    if destination.exists():raise ValueError('Use a fresh separate publication directory')
    status=json.loads((source/'status.json').read_text())
    if not status['complete'] or status['errors']:raise ValueError('Require complete E014 results')
    config=json.loads((source/'config.json').read_text())
    subjects=subjects or config['subjects'];representations=representations or config['representations']
    if not set(subjects)<=set(config['subjects']) or not set(representations)<=set(config['representations']):raise ValueError('Unknown participant or representation')
    (destination/'images').mkdir(parents=True)
    quality=pd.read_csv(source/'channel_quality.csv').fillna('')
    catalog=dict(subjects=subjects,waves=WAVES,representations={},entries={},curve_height=CURVE_HEIGHT,
        quality={str(s):dict(zip(q.channel,q['flags'])) for s,q in quality.groupby('subject') if s in subjects})
    manifest=dict(experiment_id='E014',edition='Pre-rendered images',no_trial_data=True,
        source_config_sha256=sha(source/'config.json'),renderer_sha256=sha(Path(__file__)),
        lossless_pixel_compression=True,rendering='Fixed pixel raster and discrete color palette; no statistical recomputation',
        source_arrays={},files={})
    gallery=['<!doctype html><html lang="en"><meta charset="utf-8"><title>E014 static image gallery</title><style>body{font:15px system-ui;max-width:1100px;margin:30px auto;padding:15px}td,th{padding:7px;border-bottom:1px solid #ddd}a{color:#075a9a}</style><h1>E014 static image gallery</h1><p><a href="index.html">Open image explorer</a>. No scripts are required on this page. Each map sheet contains all four metrics, first/second presentation and all electrodes/components; each curve sheet contains all electrodes/components. Maps and curves preserve separate waveforms.</p>']
    started=time.perf_counter()
    # Small support metadata are read once per representation/participant, without
    # copying the large browser payload into the publication.
    for s in subjects:
        gallery.append(f'<h2>Subject {s}</h2>')
        for rep in representations:
            raw=(source/f'data/sub-{s}_{rep}.js').read_text(encoding='utf-8')
            payload=json.loads(raw[raw.index(']=')+2:].rstrip(';\n'))
            available=sorted(int(w) for w in payload['widths'] if widths is None or int(w) in widths)
            if not available:raise ValueError(f'No selected widths for {rep}')
            channels=payload['channels']
            catalog['representations'][rep]=dict(label=LABELS[rep],channels=channels,widths=available,
                unit=payload['unit'],method=payload['method_note'])
            support=payload['support'];estimate=payload['estimate_label'];del raw,payload
            gallery.append(f'<h3>{html.escape(LABELS[rep])}</h3><table><tr><th>Bin/window</th><th>View</th>'+''.join('<th>'+w+'</th>' for w in WAVES)+'</tr>')
            for width in available:
                for view in ['specific','total']:
                    file=source/f'sub-{s}/w{width:03d}_{rep}_{view}.npz'
                    digest=sha(file)
                    with np.load(file) as saved:
                        arrays={k:saved[k] for k in FIELDS};left=saved['left_ms'];right=saved['right_ms']
                        if saved['channels'].tolist()!=channels:raise ValueError('Channel axis mismatch')
                    limits=[1. if m==3 else max(float(np.quantile(np.abs(arrays[k][np.isfinite(arrays[k])]),.99)),1e-12) for m,k in enumerate(METRICS)]
                    key=f'{s}:{rep}:{width}:{view}'
                    joint=arrays['joint_profile_r'];joint=np.where(np.isfinite(joint),joint,None).tolist()
                    catalog['entries'][key]=dict(joint=joint,support=support,channels=channels,
                        map_height=len(channels)*ROW+90)
                    cells=[]
                    for wave in range(5):
                        prefix=f'{s}_{rep}_{width}_{view}_wave{wave}'
                        maps,height=map_atlas(arrays,left,right,channels,limits,wave,catalog['representations'][rep]['unit'],estimate)
                        curves=curve_atlas(arrays,left,right,channels,wave)
                        for name,im in [('maps',maps),('curves',curves)]:
                            path=destination/'images'/f'{prefix}_{name}.webp';save_image(im,path)
                            manifest['files'][path.relative_to(destination).as_posix()]=dict(bytes=path.stat().st_size,sha256=sha(path),width=im.width,height=im.height)
                        cells.append(f'<td><a href="images/{prefix}_maps.webp">Maps</a> / <a href="images/{prefix}_curves.webp">curves</a></td>')
                    if sha(file)!=digest:raise ValueError('Source result changed during image rendering')
                    manifest['source_arrays'][file.relative_to(source).as_posix()]=digest
                    gallery.append(f'<tr><td>{width} ms</td><td>{view}</td>'+''.join(cells)+'</tr>')
            gallery.append('</table>')
            print(f'Image export: subject {s}, {rep}, {len(available)} widths',flush=True)
    manifest.update(subjects=subjects,representations=representations,images=len(manifest['files']),
        image_bytes=sum(f['bytes'] for f in manifest['files'].values()),seconds=time.perf_counter()-started,
        numerical_stages_recomputed=False)
    finish(destination,catalog,gallery,manifest)
    return dict(manifest=manifest,catalog=catalog,gallery=gallery)


def finish(destination,catalog,gallery,manifest):
    page=PAGE.replace('__CATALOG__',json.dumps(catalog,allow_nan=False,separators=(',',':'))).replace('__METRIC_LABELS__',json.dumps(METRIC_LABELS))
    (destination/'index.html').write_text(page,encoding='utf-8')
    (destination/'gallery.html').write_text('\n'.join(gallery)+'</html>',encoding='utf-8')
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    (destination/'README.md').write_text('# E014 image publication\n\nOpen index.html for the image explorer, or gallery.html for script-free direct image links. Keep the images/ directory beside both pages when copying or hosting this folder. All participant, waveform, first/second presentation, representation, bin and electrode/component choices are preserved. The page crops pre-rendered image sheets; it does not fetch numerical map arrays or trial EEG. Tiny joint-correlation/support summaries remain embedded in HTML.\n\nImages use a fixed display raster and discrete palette, then lossless WebP compression. Images cannot provide exact numeric readout at arbitrary samples. Native bin widths and full-precision estimates remain in the separate local numerical report. Statistics are not recomputed or selected for this edition. See the expandable methods in index.html and source/image digests in manifest.json.\n\nRegenerate from the project root with:\n\n```powershell\n& ./.venv/Scripts/python.exe -m experiments.response_characterization.e014_all_electrode_repeatability.image_report --output Docs/Reports/E014\n```\n\nChoose a fresh output path for regeneration; existing publications are not overwritten. This folder is Git-trackable; the large output/e014_repeatability_20261030 numerical payloads remain ignored. No deployment is performed by exporting.\n',encoding='utf-8')
    print(json.dumps({k:manifest[k] for k in ['images','image_bytes','seconds']},indent=2),flush=True)


def export_all(source,destination,subjects=None,representations=None,widths=None,workers=4):
    """Render independent participants in worker processes, then join images."""
    source=Path(source);destination=Path(destination)
    config=json.loads((source/'config.json').read_text())
    subjects=subjects or config['subjects']
    if workers==1 or len(subjects)==1:return export(source,destination,subjects,representations,widths)
    if destination.exists():raise ValueError('Use a fresh separate publication directory')
    status=json.loads((source/'status.json').read_text())
    if not status['complete'] or status['errors']:raise ValueError('Require complete E014 results')
    if workers<1 or not set(subjects)<=set(config['subjects']) or len(set(subjects))!=len(subjects):raise ValueError('Invalid workers or participants')
    scratch=Path('output')/('e014_image_parts_'+uuid.uuid4().hex)
    (destination/'images').mkdir(parents=True)
    started=time.perf_counter();results={}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending={pool.submit(export,source,scratch/f'sub-{s}',[s],representations,widths):s for s in subjects}
        for future in as_completed(pending):
            s=pending[future];result=future.result();results[s]=result
            for rel in result['manifest']['files']:
                old=(scratch/f'sub-{s}'/rel).resolve();new=(destination/rel).resolve()
                if not old.is_relative_to(scratch.resolve()) or not new.is_relative_to(destination.resolve()):raise ValueError('Image path escaped export directory')
                if new.exists():raise ValueError('Duplicate publication image')
                old.rename(new)
            print(f'Publication participant {s} complete ({len(results)}/{len(subjects)})',flush=True)
    first=results[subjects[0]]
    catalog={**first['catalog'],'subjects':subjects,'entries':{},'quality':{}}
    manifest={**first['manifest']};gallery=first['gallery'][:1]
    all_entries={};all_quality={};all_files={};all_sources={}
    for s in subjects:
        r=results[s]
        if r['manifest']['renderer_sha256']!=manifest['renderer_sha256']:raise ValueError('Renderer changed during export')
        all_entries.update(r['catalog']['entries'])
        all_quality.update(r['catalog']['quality'])
        all_files.update(r['manifest']['files']);all_sources.update(r['manifest']['source_arrays'])
        gallery+=r['gallery'][1:]
    catalog['entries']=all_entries;catalog['quality']=all_quality
    manifest.update(subjects=subjects,files=all_files,source_arrays=all_sources,images=len(all_files),
        image_bytes=sum(f['bytes'] for f in all_files.values()),seconds=time.perf_counter()-started)
    finish(destination,catalog,gallery,manifest)
    return dict(manifest=manifest,catalog=catalog,gallery=gallery)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('output/e014_repeatability_20261030'))
    parser.add_argument('--output',type=Path,default=Path('Docs/Reports/E014'))
    parser.add_argument('--subjects',type=int,nargs='+')
    parser.add_argument('--representations',nargs='+')
    parser.add_argument('--widths',type=int,nargs='+')
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args();export_all(args.source,args.output,args.subjects,args.representations,args.widths,args.workers)
