"""Portable report sections for the revised participant-specific naturalness test."""

import base64
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .position_evaluate import MODELS


def result_sentences(summary):
    def score(position, width, model):
        return next(r['balanced_accuracy']['mean'] for r in summary['aggregate']
                    if (r['analysis'],r['position'],r['width_ms'],r['model'])==('full',position,width,model))
    def contrasts(position):
        return [next(r for r in summary['contrasts'] if (r['analysis'],r['position'],r['width_ms'],r['contrast'])
                     ==('full',position,width,'post_added')) for width in (50,100)]
    first,second=contrasts(1),contrasts(2)
    tests=[r[metric]['test'] for r in summary['contrasts']
           for metric in ('balanced_accuracy_gain','log_loss_reduction')]
    count=sum(t['p_holm']<.05 for t in tests)
    main=(f"Full-cohort standalone post EEG reached {score(1,50,'post'):.2%}/{score(1,100,'post'):.2%} BA "
          f"for first presentations and {score(2,50,'post'):.2%}/{score(2,100,'post'):.2%} for second presentations "
          f"(50/100 ms bins). Metadata alone reached {score(1,50,'metadata'):.2%}.")
    gains=(f"Adding post EEG to metadata + pre changed first-position BA by "
           f"{100*first[0]['balanced_accuracy_gain']['mean']:+.2f}/{100*first[1]['balanced_accuracy_gain']['mean']:+.2f} "
           f"percentage points, and second-position BA by "
           f"{100*second[0]['balanced_accuracy_gain']['mean']:+.2f}/{100*second[1]['balanced_accuracy_gain']['mean']:+.2f} points.")
    negative=sum(r['log_loss_reduction']['mean']<0 for r in first+second)
    conclusion=(f"Mean log loss worsened in {negative} of the four full-data incremental comparisons. "
                f"Across all paired tests, {count} of {len(tests)} survived Holm correction at 0.05.")
    return main,gains,conclusion


def scientific_figures(summary, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    result = {}
    for analysis in ('full','nonterminal'):
        for position in (1,2):
            for width in (50,100):
                key = f'{analysis}:{position}:{width}'
                rows = [r for r in summary['aggregate'] if (r['analysis'],r['position'],r['width_ms'])==
                        (analysis,position,width)]
                fig,axes = plt.subplots(1,2,figsize=(14,5.5),gridspec_kw={'width_ratios':[1.15,1]})
                y = np.arange(len(MODELS))
                scores = [next(r for r in rows if r['model']==model)['balanced_accuracy'] for model in MODELS]
                means=np.array([r['mean'] for r in scores])*100
                bounds=np.array([r['ci95'] for r in scores])*100
                axes[0].errorbar(means,y,xerr=[means-bounds[:,0],bounds[:,1]-means],fmt='o',capsize=3,
                                 color='#1f6e9c')
                axes[0].set_yticks(y,list(MODELS.values()))
                axes[0].invert_yaxis()
                axes[0].set_xlim(0,100)
                axes[0].axvline(50,color='#707b83',linestyle=':')
                axes[0].set_xlabel('Participant-mean balanced accuracy (%)')
                axes[0].set_title('Held-out comparison blocks',loc='left')
                selected=[r for r in summary['per_subject'] if (r['analysis'],r['position'],r['width_ms'])==
                          (analysis,position,width)]
                subjects=sorted(set(r['subject'] for r in selected))
                x=np.arange(len(subjects))
                for model,color,offset,label in [('pre','#2879a9',-.08,'Pre EEG alone'),
                                                 ('post','#e98024',.08,'Post EEG alone')]:
                    values=[100*next(r['balanced_accuracy'] for r in selected if r['subject']==subject
                                     and r['model']==model) for subject in subjects]
                    axes[1].scatter(x+offset,values,color=color,label=label)
                axes[1].axhline(50,color='#707b83',linestyle=':')
                axes[1].set_xticks(x,[str(s) for s in subjects])
                axes[1].set_ylim(0,100)
                axes[1].set_xlabel('Participant')
                axes[1].set_ylabel('Balanced accuracy (%)')
                axes[1].set_title('Standalone EEG by participant',loc='left')
                axes[1].legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,-.15),ncol=2,fontsize=9)
                title='All eligible trials' if analysis=='full' else 'Original terminal votes removed; refitted'
                fig.suptitle(f'{title} · presentation {position} · {width} ms bins',x=.02,ha='left')
                fig.tight_layout(rect=(0,0,1,.94))
                buffer=BytesIO();fig.savefig(buffer,format='png',dpi=140,bbox_inches='tight');plt.close(fig)
                binary=buffer.getvalue()
                (output/f'wd008_position_{analysis}_p{position}_{width}ms.png').write_bytes(binary)
                result[key]='data:image/png;base64,'+base64.b64encode(binary).decode('ascii')
    return result


def public_summary(summary, figures):
    keys=('revision','date','pairs','epochs','participants','nonterminal_pairs','models','aggregate',
          'per_subject','contrasts','unsupported','class_counts','folds')
    result = dict({key:summary[key] for key in keys},figures=figures)
    result['contrasts'] = [dict(row, label='Metadata + pre + post EEG versus metadata + pre EEG')
                           if row['contrast']=='post_added' else row.copy() for row in summary['contrasts']]
    return result


def render_markdown(summary):
    lines=[
        '# WD-008. Participant-specific naturalness-choice EEG decoding','',
        '**Analysis: 24 November 2026.** Predict whether an individual presentation was chosen as more natural, '
        'using Jason’s T4 Naturalness Value. Separate models are fitted for each participant and presentation position. '
        'The previous leave-one-participant-out analysis is archived; it is not the current validation design.','',
        f"{summary['pairs']} eligible behavioral pairs, {summary['epochs']} EEG epochs and {summary['participants']} participants (3–12). "
        f"Supported analysis/position/resolution/block folds: {summary['folds']}; unsupported folds: {len(summary['unsupported'])}.",
        '', '## Question and design','',
        'Does post-stimulation EEG predict the recorded choice beyond waveform identities, trial order and pre-stimulation EEG? '
        'The label is chosen = 1 and not chosen = 0 for the current presentation. Each pair contributes one epoch to each '
        'position-specific dataset, with complementary labels. Responses are never subtracted or pooled across positions.','',
        '- Hold out the entire waveform comparison, including any repeated blocks of that comparison, within a participant and position. '
        'No participant contributes training data to another participant’s model. Single-class test blocks are retained; '
        'BA is computed from all held-out predictions pooled within each participant/position.',
        '- All 62 electrodes; fixed 50/100 ms voltage means over post 50–1,050 ms and pre −1,050 to −50 ms. '
        'These provide 1,240/620 original features per period.',
        '- Instantaneous average reference and a common pair baseline −1,250 to −1,050 ms before the first pulse; '
        'no temporal filtering or ICA. The archived complete-pair QC is retained, with finite/timestamp checks on all channels.',
        '- Training-only standardization, separate pre/post PCA retaining 90% variance, then training-only standardization of components. '
        'No supervised feature selection or tuning. Logistic regression uses L2 regularization, C=1 and no class weighting.',
        '- Metadata: five current-waveform and five opponent-waveform indicators, chronological pair index and within-block index. '
        'No charge, choice history, terminal-vote flag, final block length or response-derived predictors.',
        f"- Seven models include a training-proportion baseline, metadata alone, pre/post EEG alone and their combinations. "
        f"The sensitivity analysis removes the original terminal vote from each block ({summary['nonterminal_pairs']} remaining pairs) and refits everything.",
        '', '## Cohort held-out results','',
        'Scores weight participants equally. The binary BA reference is 50%. BA averages chosen/not-chosen recall. '
        'Accuracy and BA differ because choice labels are not equally frequent.','',
        '| Trials | Position | Bin | Model | BA | Accuracy | Log loss | AUC |',
        '| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |',
    ]
    for row in summary['aggregate']:
        lines.append(f"| {row['analysis']} | {row['position']} | {row['width_ms']} ms | {MODELS[row['model']]} | "
                     f"{row['balanced_accuracy']['mean']:.2%} | {row['accuracy']['mean']:.2%} | "
                     f"{row['log_loss']['mean']:.4f} | {row['auc']['mean']:.3f} |")
    lines += ['', '## Paired participant comparisons','',
              'Positive values favor post EEG: BA change is comparison minus baseline; log-loss reduction is baseline minus comparison. '
              'Intervals are descriptive 10,000-resample participant bootstrap intervals conditional on fitted predictions.','',
              '| Trials | Position | Bin | Comparison | BA change, pp [95% interval] | Log-loss reduction [95% interval] | BA Holm p | Loss Holm p |',
              '| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |']
    for row in summary['contrasts']:
        ba,ll=row['balanced_accuracy_gain'],row['log_loss_reduction']
        lines.append(f"| {row['analysis']} | {row['position']} | {row['width_ms']} ms | {row['label']} | "
            f"{100*ba['mean']:+.2f} [{100*ba['ci95'][0]:+.2f}, {100*ba['ci95'][1]:+.2f}] | "
            f"{ll['mean']:+.4f} [{ll['ci95'][0]:+.4f}, {ll['ci95'][1]:+.4f}] | "
            f"{ba['test']['p_holm']:.4f} | {ll['test']['p_holm']:.4f} |")
    lines += ['', '## Interpretation and inference','',
        '- '+result_sentences(summary)[0]+' Waveform identity and trial order can predict subjective preferences; '
        'this does not establish incorrect logs or isolate an order effect.',
        '- '+' '.join(result_sentences(summary)[1:]),
        '- Exact two-sided participant sign tests compare the direction of the paired changes; ties are omitted. '
        'Holm correction covers 32 tests: two contrasts × two metrics × two widths × two positions × two analyses. '
        'These are not individual above-chance tests or label-permutation tests. '
        '[Exact sign-test computation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html) · '
        '[Holm correction](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html).',
        '- Second-position pre EEG occurs after the first stimulus and may contain its response; it is not an unstimulated baseline. '
        'The two positions share behavioral pairs and are not independent replications. Behavioral-response timestamps and '
        'current/trigger synchronization remain unverified.',
        '', '## Per-participant class support','',
        '| Participant | Position | Epochs | Chosen | Not chosen |',
        '| --- | ---: | ---: | ---: | ---: |']
    for row in summary['class_counts']:
        lines.append(f"| {row['subject']} | {row['position']} | {row['count']} | {row['sum']} | {row['count']-row['sum']} |")
    return '\n'.join(lines)+'\n'


HTML = '''<section id="position-revision"><h2>Within-participant naturalness decoding</h2>
<p><strong>Current analysis · 24 November 2026.</strong> Each participant and presentation position has its own models.
The target is whether that presentation was <strong>chosen as more natural</strong> (1) or not chosen (0),
from Jason’s T4 Naturalness Value. First and second responses are analyzed separately, without taking their difference.</p>
<p id="nr_coverage"></p>
<p>__RESULTS__
Metadata performance reflects predictable choices from waveform identities and trial order, not proof of invalid behavioral labels.</p>
<div class="filters">
<label>Trials<select id="nr_analysis"><option value="full">All eligible trials</option>
<option value="nonterminal">Original terminal votes removed · refitted</option></select></label>
<label>Presentation<select id="nr_position"><option value="1">First</option><option value="2">Second</option></select></label>
<label>Bin width<select id="nr_width"><option value="50">50 ms</option><option value="100">100 ms</option></select></label>
<label>Participant<select id="nr_subject"><option value="all">Participant mean</option></select></label></div>
<p id="nr_filter_note"></p>
<details><summary>What do these filters mean?</summary>
<p><strong>Trials — All eligible trials:</strong> use all 622 EEG-eligible behavioral pairs.
Each presentation-specific dataset contains one epoch per pair.</p>
<p><strong>Trials — Original terminal votes removed:</strong> exclude the original last recorded trial of each
comparison block, which normally ends the race-to-five when one waveform receives its fifth vote.
This removes 100 trials, leaving 522 pairs. All models, scaling and PCA are refitted on the remaining data
with the same comparison-block holdout. The newly last trial is not removed again.
This checks sensitivity to the stopping rule; it does not undo how earlier trials were collected.</p>
<p><strong>Presentation:</strong> choose separate first- or second-pulse models. The label always means whether
that pulse was chosen as more natural. Second-pulse pre EEG follows the first pulse and may contain its response.</p>
<p><strong>Bin width:</strong> 50 or 100 ms determines how finely the same one-second EEG window is averaged.
It changes feature resolution (1,240 or 620 original features across 62 electrodes), not the analyzed duration.</p>
<p><strong>Participant:</strong> “Participant mean” averages the scores of the separately fitted participant models,
giving each person equal weight. Selecting a participant displays that person’s scores and counts.
The plots continue to show the full eligible cohort for the selected trial, presentation and bin-width settings.</p></details>
<p id="nr_selection"></p>
<h3>Held-out accuracy and choice recall</h3><div class="scroll"><table class="data-table"><thead><tr>
<th>Model</th><th>BA</th><th>Accuracy</th><th>Chosen recall</th><th>Not-chosen recall</th><th>Log loss ↓</th><th>AUC</th>
</tr></thead><tbody id="nr_scores"></tbody></table></div>
<p>BA averages chosen and not-chosen recall; its reference is 50%. Cohort scores weight participants equally.
Accuracy uses the observed choice proportions. All models use identical held-out epochs.</p>
<h3>What does post EEG add?</h3><div class="scroll"><table class="data-table"><thead><tr>
<th>Comparison</th><th>BA change, pp [95% interval]</th><th>BA p / Holm p</th>
<th>Log-loss reduction [95% interval]</th><th>Loss p / Holm p</th></tr></thead><tbody id="nr_contrasts"></tbody></table></div>
<p>Positive differences favor post EEG. Intervals resample participants using fixed held-out predictions;
individual selections show their paired differences without cohort intervals or p-values.</p>
<figure><img id="nr_plot" alt="Participant-specific naturalness decoding results">
<figcaption>Cohort plots follow trial, presentation and bin-width filters. The participant selector affects the tables only.
Dotted lines mark 50% BA, not significance thresholds. Error bars are descriptive participant bootstrap intervals.</figcaption></figure>
<h3>Features and model</h3>
<p><strong>EEG:</strong> all 62 electrodes, using 50/100 ms voltage means across post 50–1,050 ms and pre −1,050 to −50 ms
(1,240/620 original features). Processing uses the instantaneous average reference and a common pair baseline
−1,250 to −1,050 ms before the first pulse; no temporal filter or ICA. Original complete-pair QC is retained,
with finite/timestamp checks on all channels.</p>
<p><strong>PCA:</strong> fit scaling and separate pre/post PCA on the training fold only; retain 90% variance,
then standardize component scores using that training fold. <strong>Classifier:</strong> L2 logistic regression,
fixed C=1, without class weighting or tuning. The prior model uses the training choice proportion.</p>
<p><strong>Metadata:</strong> current-waveform and opponent-waveform indicators plus chronological pair and within-block indices.
No charge, choice history, terminal-vote flag, final block length or response-derived predictor is used.</p>
<h3>Validation and sensitivity analysis</h3>
<p>Within each participant and position, hold out the entire waveform comparison, grouping any repeated blocks together.
No training participant or presentation position is pooled with another. Single-class test blocks are retained;
scores pool held-out predictions across blocks before calculating each participant’s BA.
The sensitivity analysis removes the original terminal vote from every block and refits all preprocessing and models.</p>
<h3>Statistical interpretation and timing</h3>
<p>Exact two-sided participant sign tests compare improvement versus decline in BA and log loss, omitting ties.
Holm correction covers all 32 contrasts across metrics, widths, positions and analyses.
These tests evaluate paired changes, not individual decoding above chance.
<a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html">Exact sign-test computation</a> ·
<a href="https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html">Holm correction</a>.</p>
<p>The <strong>second-position pre period follows the first stimulus</strong> and can contain its response.
The two positions share behavioral pairs and are not independent replications. Response timestamps and physical
current/trigger synchronization remain unavailable. The earlier population models are archived below.</p></section>'''


JS = '''
if(data.position_revision){
 const revision=data.position_revision;
 const nr=id=>document.getElementById('nr_'+id);
 const fixed=(x,d=2)=>x===null||x===undefined?'—':x.toFixed(d);
 const percentage=x=>x===null||x===undefined?'—':(100*x).toFixed(2)+'%';
 const addRow=(body,values)=>{const tr=document.createElement('tr');for(const value of values){
   const td=document.createElement('td');td.textContent=value;tr.append(td);}body.append(tr);};
 const dimensions=()=>[nr('analysis').value,Number(nr('position').value),Number(nr('width').value)];
 const matches=(r,[a,p,w])=>r.analysis===a&&r.position===p&&r.width_ms===w;
 function revisionDraw(){
   const dim=dimensions(),subject=nr('subject').value;
   nr('filter_note').textContent=dim[0]==='full'?
     'All eligible trials: 622 pairs; no terminal-vote exclusion.':
     'Terminal-vote sensitivity: the original final trial from each comparison block is excluded; 522 pairs remain and all models are refitted.';
   const selected=revision.per_subject.filter(r=>matches(r,dim)&&(subject==='all'||r.subject===Number(subject)));
   const sample=selected.filter(r=>r.model==='prior');
   nr('selection').textContent=sample.reduce((n,r)=>n+r.epochs,0)+' held-out epochs from '+sample.length+
     ' participant'+(sample.length===1?'':'s')+' · chosen '+sample.reduce((n,r)=>n+r.chosen,0)+
     ' / not chosen '+sample.reduce((n,r)=>n+r.not_chosen,0);
   nr('scores').replaceChildren();nr('contrasts').replaceChildren();
   for(const [model,label] of Object.entries(revision.models)){
     const record=subject==='all'?revision.aggregate.find(r=>matches(r,dim)&&r.model===model):selected.find(r=>r.model===model);
     const value=name=>subject==='all'?record[name]?.mean:record[name];
     addRow(nr('scores'),[label,...['balanced_accuracy','accuracy','chosen_recall','not_chosen_recall'].map(k=>percentage(value(k))),
       fixed(value('log_loss'),4),fixed(value('auc'),3)]);
   }
   for(const contrast of revision.contrasts.filter(r=>matches(r,dim))){
     const ba=contrast.balanced_accuracy_gain,ll=contrast.log_loss_reduction;
     const individual=contrast.participants.find(r=>r.subject===Number(subject));
     const effect=(values,factor,d)=>subject==='all'?fixed(factor*values.mean,d)+' ['+
       values.ci95.map(x=>fixed(factor*x,d)).join(', ')+']':fixed(factor*individual?.[factor===100?'balanced_accuracy_gain':'log_loss_reduction'],d);
     const p=values=>subject==='all'?fixed(values.test.p,4)+' / '+fixed(values.test.p_holm,4):'—';
     addRow(nr('contrasts'),[contrast.label,effect(ba,100,2),p(ba),effect(ll,1,4),p(ll)]);
   }
   nr('plot').src=revision.figures[dim.join(':')];
 }
 nr('coverage').textContent=revision.pairs+' behavioral pairs · '+revision.epochs+' EEG epochs · '+revision.participants+
   ' participants · '+revision.folds+' supported block/position/resolution fits · '+revision.unsupported.length+' unsupported folds';
 const subjects=[...new Set(revision.per_subject.map(r=>r.subject))];
 nr('subject').replaceChildren(new Option('Participant mean','all'),...subjects.map(s=>new Option('Participant '+s,String(s))));
 for(const control of ['analysis','position','width','subject'])nr(control).addEventListener('change',revisionDraw);
 revisionDraw();
}
'''
