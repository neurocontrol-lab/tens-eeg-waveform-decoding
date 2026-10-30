"""Variable-channel explorer for sensor, spectral and calibrated representations."""
import argparse
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
from experiments.response_characterization.e011_electrode_bin_repeatability.waveform_report import TEMPLATE
from experiments.response_characterization.e011_electrode_bin_repeatability.waveform_views import group_summary
from experiments.response_characterization.e011_electrode_bin_repeatability.__main__ import sha,write_json

LABELS=dict(voltage_mean='62 electrodes: mean voltage',slope_mean='62 electrodes: signed mean slope',
    slope_rms='62 electrodes: RMS slope',curvature_mean='62 electrodes: signed mean curvature',
    curvature_rms='62 electrodes: RMS curvature',alpha_power='62 electrodes: alpha power (500-ms windows)',
    beta_power='62 electrodes: beta power (500-ms windows)',
    calibrated_sensors62='Matched calibration/A/B: 62 sensors',
    calibrated_sensors14='Matched calibration/A/B: original 14 sensors',
    xdawn_components='Matched calibration/A/B: 10 xDAWN components')


def build_template():
    text=TEMPLATE.replace('E011','E014').replace('Preserved original E014 report','Preserved E011 report')
    text=text.replace('Is this waveform\'s response repeatable?','All-electrode and learned-feature repeatability')
    text=text.replace('500-ms windows','500-ms windows')
    text=text.replace('<label>Bin width','<label>Bin / spectral window')
    text=text.replace('<option value="full_response">Signed full-data response</option>',
        '<option value="full_response">Signed response estimate</option>')
    text=text.replace('Individual-waveform electrode × time maps','Individual-waveform feature × time maps')
    text=text.replace('<label>Electrode<select','<label>Electrode / component<select')
    text=text.replace('<div id="loading" class="note">Loading participant data…</div>',
        '<div id="loading" class="note">Loading participant data…</div><p id="method" class="panel note"></p>')
    text=re.sub(r'<p><b>Temporal pattern r</b>.*?</p>',
        '<p><b>Temporal pattern r</b> compares one feature\'s held-out A/B time courses within the declared one-second support; joint pattern r combines features and time. Sensor and power maps retain scalp electrodes. Power uses eleven overlapping 500-ms windows whose centres run from 300 to 800 ms. xDAWN uses ten learned components, not ten fixed electrodes.</p>'
        '<p><b>Calibration/A/B comparisons:</b> xDAWN fits only independent calibration trials. Its 62- and 14-sensor controls use exactly the same A/B trials and calibration-only RMS normalization. Joint shape r demeans each feature\'s time course before flattening. Compare these three matched representations to each other; their normalization, trial counts and joint-r definition differ from the two-way sensor/power views.</p>',text,flags=re.S)
    text=re.sub(r'<p>Paired splits, trial eligibility,.*?</p>',
        '<p>Voltage/derivative/power features use the original E011 paired splits and 595 pairs. xDAWN uses 100 calibration/A/B allocations, with disjoint complete pairs. Two-pair strata supply A/B only. Component class/rank names change spatial meaning across calibration fits and are not fixed brain sources. The selected method box explains each representation. All uncertainty is exploratory; no classifier or significant-feature selection is performed.</p>',text,flags=re.S)
    text=text.replace('Joint pattern r: median split correlation across 14 electrodes × [50,1050) ms.',
        'Joint pattern r: median held-out pattern agreement for the selected feature representation.')
    text=text.replace('Trial support is shown alongside.','Available trial support and minimum evaluation-half counts are shown alongside.')
    text=text.replace('Grey tails contain no complete bin.','Grey areas have no reported bin or spectral-window centre.')
    text=text.replace('Grey shows the full-data estimate.','Grey shows the estimate described in the selected representation.')
    text=text.replace('595 pairs / 1,190 events, identical to E014.','595 pairs / 1,190 events, identical to E011.')
    text=text.replace('RMS derivatives may reflect noise.','RMS derivatives and band power may reflect noise. Quality flags are descriptive and do not remove electrodes or trials. Learned xDAWN components are not scalp electrodes.')
    script=text.split('<script>')[1].split('</script>')[0]
    script=re.sub(r'const labels=\{.*?\};','const labels=config.labels;',script,count=1)
    old="for(const w of config.widths)el('width').add(new Option(w===1?'1 ms (native grid)':w+' ms',w));el('width').value='50';"
    assert old in script;script=script.replace(old,'')
    old="for(const c of config.channels)el('channel').add(new Option(c,c));el('channel').value='C3';"
    assert old in script;script=script.replace(old,'')
    script=script.replace('config.channels','active.channels')
    # The parent template's spatial count is its one hard-coded scientific axis.
    script=re.sub(r'\b14\b','active.channels.length',script)
    script=script.replace('W=810,H=350','W=810,H=canvas.height-90')
    old="const x=L+W*(50+b*Number(el('width').value)-50)/1138;const wid=W*Number(el('width').value)/1138;"
    assert old in script;script=script.replace(old,'const x=L+W*(info.left_ms+b*info.step_ms-50)/1138;const wid=W*info.step_ms/1138;')
    old="const x=L+((j+.5)*Number(el('width').value))/1138*W"
    assert old in script;script=script.replace(old,'const x=L+(info.centres_ms[j]-50)/1138*W')
    script=script.replace("ctx.fillText('Full data',L+165,16)","ctx.fillText(active.estimate_label,L+165,16)")
    old="cells:[ch,fmt(view.temporal_profile_r[0][w][ci]),fmt(view.temporal_profile_r[1][w][ci])]"
    assert old in script;script=script.replace(old,"cells:[ch,fmt(view.temporal_profile_r[0][w][ci]),fmt(view.temporal_profile_r[1][w][ci]),(active.quality?.[ci]?.flags||'')]")
    script=script.replace("['Electrode','P1 temporal pattern r','P2 temporal pattern r']",
        "['Electrode / component','P1 temporal pattern r','P2 temporal pattern r','Quality note']")
    script=script.replace("'Trials: P1 / P2'","'Available trials: P1 / P2'")
    old='active=window.e011WaveData[key];'
    insert="""active=window.e011WaveData[key];
const oldWidth=el('width').value;el('width').innerHTML='';
for(const width of Object.keys(active.widths))el('width').add(new Option(width==='500'?'500-ms power window / 50-ms step':(width==='1'?'1 ms (native grid)':width+' ms'),width));
el('width').value=active.widths[oldWidth]?oldWidth:(active.widths['50']?'50':Object.keys(active.widths)[0]);
const oldChannel=el('channel').value;el('channel').innerHTML='';
for(let c=0;c<active.channels.length;c++)el('channel').add(new Option(active.channels[c]+(active.quality?.[c]?.flags?' [note]':''),active.channels[c]));
el('channel').value=active.channels.includes(oldChannel)?oldChannel:(active.channels.includes('C3')?'C3':active.channels[0]);
for(const id of ['map0','map1'])el(id).height=Math.max(440,active.channels.length*20+95);
el('method').textContent=active.method_note;
"""
    assert old in script;script=script.replace(old,insert)
    text=text.split('<script>')[0]+'<script>'+script+'</script></body></html>'
    return text


def render(out):
    out=Path(out);configs={};statuses={}
    for stage in ['sensors','xdawn']:
        configs[stage]=json.loads((out/f'config_{stage}.json').read_text())
        statuses[stage]=json.loads((out/f'status_{stage}.json').read_text())
        if statuses[stage]['errors'] or not statuses[stage]['complete']:raise ValueError(f'Incomplete {stage} stage')
        for path,h in {**configs[stage]['input_sha256'],**configs[stage]['implementation_sha256']}.items():
            if sha(path)!=h:raise ValueError(f'Frozen stage input/source changed: {path}')
    subjects=configs['sensors']['subjects']
    if configs['xdawn']['subjects']!=subjects:raise ValueError('Matched stages have different participants')
    rows=[];quality=[];cache=Path(configs['sensors']['cache'])
    for s in subjects:
        for name in ['sensor_waveform_summary.csv','calibrated_waveform_summary.csv']:
            rows+=pd.read_csv(out/f'sub-{s}'/name).to_dict('records')
        table=pd.read_csv(cache/f'sub-{s}/channel_quality.csv').fillna('');table['subject']=s;quality.append(table)
    group_summary(rows,out,subjects)
    pd.concat(quality).to_csv(out/'channel_quality.csv',index=False)
    settings=dict(subjects=subjects,representations=list(LABELS),labels=LABELS,
        waves=['Sine','Square','Triangle','LinDec','LinInc'])
    groups=pd.read_csv(out/'group_waveform_summary.csv').to_dict('records')
    template=build_template()
    html=template.replace('__CONFIG__',json.dumps(settings)).replace('__GROUPS__',json.dumps(groups,allow_nan=False))
    html=html.replace('<h2>Limits and downloadable outputs</h2>',
        '<h2>Limits and downloadable outputs</h2><p><a href="channel_quality.csv">All 62-electrode quality flags</a> · '
        '<a href="config_sensors.json">Sensor/power stage provenance</a> · <a href="config_xdawn.json">Calibration/xDAWN stage provenance</a></p>')
    (out/'index.html').write_text(html,encoding='utf-8')
    write_json(out/'config.json',dict(experiment_id='E014',subjects=subjects,n_scalp_electrodes=62,
        stages={k:sha(out/f'config_{k}.json') for k in configs},representations=list(LABELS),
        no_classifier_fitted=True,no_features_selected=True,original_E011_preserved=True))
    write_json(out/'status.json',dict(completed=subjects,complete=True,errors={},
        sensor_seconds=statuses['sensors'].get('seconds'),xdawn_seconds=statuses['xdawn'].get('seconds')))
    write_json(out/'render_provenance.json',dict(renderer_sha256=sha(Path(__file__)),
        template_sha256=sha(Path('experiments/response_characterization/e011_electrode_bin_repeatability/waveform_report.py')),
        original_group_helper_sha256=sha(Path('experiments/response_characterization/e011_electrode_bin_repeatability/waveform_views.py')),
        numerical_stages_recomputed=False))
    (out/'RESULTS.md').write_text('# E014 - All-electrode, power and calibrated xDAWN repeatability\n\n'
        'All 62 scalp electrodes, 595 pairs and 1,190 events. The original 14 epochs and per-electrode '
        'fields reproduce the frozen E011 reference. Channel-quality flags cause no exclusions.\n\n'
        'Voltage/derivative and alpha/beta power use the original two-half splits. Power uses overlapping '
        '500-ms Hann windows with a 50-ms step; the displayed coordinates are their centres, not millisecond '
        'spectral resolution. These are total-power features after the simple band-pass.\n\n'
        'xDAWN uses 100 independent calibration/A/B partitions, with two components per class and 0.1 '
        'covariance regularization. Matched 62/14-sensor controls share the same held-out A/B data and '
        'calibration-only RMS normalization. Joint temporal shape correlations remove each feature time mean '
        'and differ from the original two-half metric. Component labels are learned class/rank slots, '
        'not fixed scalp locations. No classifier, peak alignment, feature selection or corrected significance '
        'claim is added. See the interactive explorer, channel flags, stage configs and per-participant arrays.\n',encoding='utf-8')
    print(f'Combined E014 explorer: {(out/"index.html").resolve()}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path)
    render(parser.parse_args().output)
