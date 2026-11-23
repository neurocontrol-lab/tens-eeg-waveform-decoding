"""Reports for actual-pair, within-participant naturalness prediction."""

import base64
from io import BytesIO
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def result_text(summary):
    def value(representation,width,model):
        return next(r['balanced_accuracy']['mean'] for r in summary['aggregate'] if
                    (r['analysis'],r['representation'],r['width_ms'],r['model'])==('full',representation,width,model))
    tests=[r[m]['test'] for r in summary['contrasts'] for m in ('balanced_accuracy_gain','log_loss_reduction')]
    return (f"Paired post EEG alone reached {value('concatenated',50,'post'):.2%}/{value('concatenated',100,'post'):.2%} "
            f"balanced accuracy with ordered concatenation and {value('difference',50,'post'):.2%}/{value('difference',100,'post'):.2%} "
            f"with second-minus-first features (50/100 ms bins). Metadata alone reached {value('concatenated',50,'metadata'):.2%}. "
            f"Of {len(tests)} paired-change tests, {sum(t['p_holm']<.05 for t in tests)} survived Holm correction at 0.05.")


def scientific_figures(summary,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);result={}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    for analysis in ('full','nonterminal'):
        for representation in summary['representations']:
            for width in (50,100):
                rows=[r for r in summary['aggregate'] if (r['analysis'],r['representation'],r['width_ms'])==
                      (analysis,representation,width)]
                fig,axes=plt.subplots(1,2,figsize=(14,5.5),gridspec_kw={'width_ratios':[1.2,1]})
                values=[next(r['balanced_accuracy'] for r in rows if r['model']==m) for m in summary['models']]
                means=np.array([v['mean'] for v in values])*100;bounds=np.array([v['ci95'] for v in values])*100
                axes[0].errorbar(means,np.arange(len(values)),xerr=[means-bounds[:,0],bounds[:,1]-means],fmt='o',capsize=3,color='#2879a9')
                axes[0].set_yticks(np.arange(len(values)),list(summary['models'].values()));axes[0].invert_yaxis()
                axes[0].set_xlim(0,100);axes[0].axvline(50,color='#727b82',linestyle=':')
                axes[0].set_xlabel('Participant-mean balanced accuracy (%)')
                axes[0].set_title('One held-out prediction per actual pair',loc='left')
                cell=[r for r in summary['per_subject'] if (r['analysis'],r['representation'],r['width_ms'])==
                      (analysis,representation,width)]
                subjects=sorted(set(r['subject'] for r in cell));x=np.arange(len(subjects))
                for model,color,offset,label in [('metadata_pre','#2879a9',-.08,'Metadata + pre'),
                                                 ('metadata_pre_post','#e98024',.08,'Add paired post EEG')]:
                    scores=[100*next(r['balanced_accuracy'] for r in cell if r['subject']==s and r['model']==model) for s in subjects]
                    axes[1].scatter(x+offset,scores,label=label,color=color)
                axes[1].axhline(50,color='#727b82',linestyle=':');axes[1].set_ylim(0,100)
                axes[1].set_xticks(x,[str(s) for s in subjects]);axes[1].set_xlabel('Participant')
                axes[1].set_ylabel('Balanced accuracy (%)');axes[1].set_title('Incremental paired EEG comparison',loc='left')
                axes[1].legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,-.15),ncol=2,fontsize=8)
                trials='All eligible pairs' if analysis=='full' else 'Original last trial excluded; refitted'
                fig.suptitle(f'{trials} · {summary["representations"][representation]} · {width} ms bins',x=.02,ha='left',fontsize=11)
                fig.tight_layout(rect=(0,0,1,.94));buffer=BytesIO();fig.savefig(buffer,format='png',dpi=140,bbox_inches='tight');plt.close(fig)
                binary=buffer.getvalue();(output/f'wd008_paired_{analysis}_{representation}_{width}ms.png').write_bytes(binary)
                result[f'{analysis}:{representation}:{width}']='data:image/png;base64,'+base64.b64encode(binary).decode('ascii')
    return result


def public_summary(summary,figures):
    return dict({k:summary[k] for k in ('date','design','pairs','epochs','participants','nonterminal_pairs','models',
                                     'representations','folds','unsupported','aggregate','per_subject','contrasts')},figures=figures)


def render_markdown(summary):
    lines=['# WD-008. Paired naturalness-choice EEG decoding','',
        '**24 November 2026.** One observation is one original two-stimulus trial. The model receives EEG from both '
        'presentations and predicts first chosen (0) versus second chosen (1), from Jason’s T4 Naturalness Value. '
        'Models are fitted separately within each participant.','',
        f"{summary['pairs']} eligible pairs, {summary['epochs']} response epochs and {summary['participants']} participants. "
        f"Supported block/configuration folds: {summary['folds']}; unsupported: {len(summary['unsupported'])}.",'',
        result_text(summary),'','## Inputs and validation','',
        '- Actual pairs are matched by pair ID, retaining first/second order. No artificial chosen/unchosen pairs are constructed.',
        '- All 62 electrodes in 50/100 ms bins. Both post windows span 50–1,050 ms relative to their respective pulses. '
        'The pre-pair control spans −1,050 to −50 ms before the first pulse.',
        '- Ordered concatenation is the primary representation: first-response features followed by second-response features '
        '(2,480/1,240 original post features at 50/100 ms). The restored second-minus-first representation is a fixed comparison '
        '(1,240/620 original post features). Pre-pair controls have 1,240/620 original features.',
        '- Instantaneous average reference and common pair baseline −1,250 to −1,050 ms before the first pulse; '
        'no temporal filter or ICA. Original complete-pair QC and all-channel finite/timestamp checks are retained.',
        '- Within each participant, hold out every block of the same waveform comparison together. Both responses stay in '
        'one observation. No cross-participant fitting. Test-block labels can be single-class; pool held-out predictions across '
        'comparisons before scoring each participant.',
        '- Training-only scaling, separate pre/post PCA retaining 90% variance, then training-only component scaling. '
        'Fixed L2 logistic regression C=1; no class weighting or tuning.',
        '- Metadata: first/second waveform indicators, chronological pair index and within-block index. '
        'No charge, previous choices, terminal-vote flag or final block length enters the inputs.',
        '- Optional last-trial exclusion removes the original final trial per block and refits models on 522 pairs. '
        'It checks dependence on those trials and reduced training data; it does not remove the stopping rule.',
        '', '## Held-out results','',
        'Scores average participants equally. Balanced accuracy averages first-choice and second-choice recall; '
        'its reference is 50%. Ordinary accuracy is the fraction of correctly predicted pair choices.','',
        '| Trials | Representation | Bin | Model | BA | Accuracy | Log loss | AUC |',
        '| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |']
    for r in summary['aggregate']:
        lines.append(f"| {r['analysis']} | {r['representation']} | {r['width_ms']} ms | {summary['models'][r['model']]} | "
                     f"{r['balanced_accuracy']['mean']:.2%} | {r['accuracy']['mean']:.2%} | {r['log_loss']['mean']:.4f} | {r['auc']['mean']:.3f} |")
    lines+=['','## What does paired post EEG add?','',
            'Positive BA changes and log-loss reductions favor paired post EEG. Intervals are descriptive participant '
            'bootstrap intervals conditional on fixed held-out predictions.','',
            '| Trials | Representation | Bin | Comparison | BA change, pp [95% interval] | Log-loss reduction [95% interval] | BA Holm p | Loss Holm p |',
            '| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |']
    for r in summary['contrasts']:
        ba,ll=r['balanced_accuracy_gain'],r['log_loss_reduction']
        lines.append(f"| {r['analysis']} | {r['representation']} | {r['width_ms']} ms | {r['label']} | "
                     f"{100*ba['mean']:+.2f} [{100*ba['ci95'][0]:+.2f}, {100*ba['ci95'][1]:+.2f}] | "
                     f"{ll['mean']:+.4f} [{ll['ci95'][0]:+.4f}, {ll['ci95'][1]:+.4f}] | {ba['test']['p_holm']:.4f} | {ll['test']['p_holm']:.4f} |")
    lines+=['','## Interpretation','',
        'The full-data metadata-only BA is 68.78%. Paired EEG does not show a consistent incremental benefit over '
        'metadata + pre-pair EEG across the tested representations and widths. Observed above-50% scores are not '
        'individual significance claims. Exact two-sided participant sign tests assess paired BA/log-loss changes; '
        'Holm correction covers 32 tests across two contrasts, two metrics, two representations, two widths and two trial analyses. '
        '[Sign-test computation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html) · '
        '[Holm correction](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html).','',
        'The second post response can contain carryover, position and decision-related activity. All EEG controls before '
        'the pair use only the first pre window; the second pre window is excluded because it follows the first stimulus. '
        'Concatenated post EEG contains two one-second windows, while the pre control contains one. '
        'Physical delivery synchronization and behavioral-response timestamps remain unavailable.','',
        'Superseded single-presentation and population analyses are retained locally for provenance and are omitted '
        'from the shared report. No acquisition or archived numerical files were deleted.','']
    return '\n'.join(lines)


HTML='''<section id="paired-analysis"><h2>Within-participant paired naturalness decoding</h2>
<p><strong>One observation = one actual stimulus pair.</strong> The model receives both post-stimulation EEG responses
and predicts <strong>first stimulus chosen</strong> versus <strong>second stimulus chosen</strong>, using Jason’s recorded choice.
Each participant has separate fitted models. First and second responses remain together throughout training and testing.</p>
<p id="pr_coverage"></p><p>__RESULTS__</p>
<div class="filters"><label>Trials<select id="pr_analysis"><option value="full">All eligible pairs</option>
<option value="nonterminal">Original last trial excluded · refitted</option></select></label>
<label>Paired EEG representation<select id="pr_representation"><option value="concatenated">First + second concatenated</option>
<option value="difference">Second minus first</option></select></label>
<label>Bin width<select id="pr_width"><option value="50">50 ms</option><option value="100">100 ms</option></select></label>
<label>Participant<select id="pr_subject"><option value="all">Participant mean</option></select></label></div>
<details><summary>What do these filters mean?</summary>
<p><strong>Trials:</strong> all eligible pairs uses 622 original pairs. Last-trial exclusion removes each block’s original
final trial, leaving 522 pairs, and refits all models. This is a limited robustness check, not a correction for the stopping rule.</p>
<p><strong>Representation:</strong> concatenation preserves first-response features followed by second-response features.
Second-minus-first restores the original difference representation. Both use the same actual pairs and recorded choices.</p>
<p><strong>Bin width:</strong> 50/100 ms changes the averaging resolution within each one-second response, not its duration.</p>
<p><strong>Participant:</strong> select one participant’s held-out scores or an equal-weight mean of the individually fitted models.
There is no population model. Cohort plots follow the other three filters and always show the full eligible cohort.</p></details>
<p id="pr_selection"></p>
<h3>Held-out accuracy and first/second-choice recall</h3><div class="scroll"><table class="data-table"><thead><tr>
<th>Model</th><th>BA</th><th>Accuracy</th><th>First-choice recall</th><th>Second-choice recall</th><th>Log loss ↓</th><th>AUC</th>
</tr></thead><tbody id="pr_scores"></tbody></table></div>
<p>One prediction per held-out pair. Accuracy is the fraction of correct first/second choices;
BA averages the two choice recalls and has a 50% reference. Cohort scores weight participants equally.</p>
<h3>What does paired post EEG add?</h3><div class="scroll"><table class="data-table"><thead><tr>
<th>Comparison</th><th>BA change, pp [95% interval]</th><th>BA p / Holm p</th>
<th>Log-loss reduction [95% interval]</th><th>Loss p / Holm p</th></tr></thead><tbody id="pr_contrasts"></tbody></table></div>
<p>Positive differences favor paired post EEG. Participant selections show individual differences without cohort intervals or p-values.</p>
<figure><img id="pr_plot" alt="Paired naturalness decoding results"><figcaption>Descriptive participant bootstrap intervals;
dotted 50% BA references are not significance thresholds. Blue/orange points compare metadata + pre-pair EEG
against the same model with paired post EEG added.</figcaption></figure>
<h3>Features and models</h3>
<p><strong>EEG:</strong> all 62 electrodes. Both post windows cover 50–1,050 ms relative to their own pulse.
The pre-pair control covers −1,050 to −50 ms before the first pulse. Ordered post concatenation has 2,480/1,240 original
features at 50/100 ms; differences and pre-pair controls have 1,240/620. Concatenation is the primary representation.</p>
<p><strong>Processing:</strong> instantaneous average reference; shared pair baseline −1,250 to −1,050 ms before the first pulse;
no temporal filter or ICA. Original complete-pair QC plus all-channel finite/timestamp checks are retained.</p>
<p><strong>Training:</strong> fit feature scaling, separate pre/post PCA retaining 90% variance, and component scaling
on training pairs only. Logistic regression uses fixed L2 regularization, C=1 and no class weighting or tuning.</p>
<p><strong>Metadata:</strong> first/second waveform indicators plus chronological pair and within-block indices.
No charge, earlier choices, terminal-vote flag or final block length is supplied.</p>
<h3>Validation and interpretation</h3>
<p>Within each participant, hold out all blocks of an entire waveform comparison. Test first/second responses always
belong to the same original pair. Single-class test blocks are retained; participant scores pool all held-out pair predictions.</p>
<p>Exact two-sided participant sign tests compare the direction of paired BA and log-loss changes, omitting ties.
Holm correction covers 32 tests across contrasts, metrics, representations, bin widths and trial analyses.
These are not individual above-chance tests.
<a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html">Sign-test computation</a> ·
<a href="https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html">Holm correction</a>.</p>
<p>The second response can contain carryover and decision-related activity. Only EEG before the first pulse is used
for pre-pair controls. Concatenated post inputs contain two one-second responses; the pre control contains one window.
Physical delivery timing and behavioral-response timestamps remain unverified.</p></section>'''

JS='''
if(data.paired_revision){
 const paired=data.paired_revision,pr=id=>document.getElementById('pr_'+id);
 const fixed=(x,d=2)=>x===null||x===undefined?'—':x.toFixed(d),pc=x=>x===null||x===undefined?'—':(100*x).toFixed(2)+'%';
 const add=(body,values)=>{const row=document.createElement('tr');for(const value of values){const td=document.createElement('td');td.textContent=value;row.append(td);}body.append(row);};
 const dimensions=()=>[pr('analysis').value,pr('representation').value,Number(pr('width').value)];
 const match=(r,d)=>r.analysis===d[0]&&r.representation===d[1]&&r.width_ms===d[2];
 function pairedDraw(){
  const dim=dimensions(),subject=pr('subject').value,individual=subject!=='all';
  const rows=paired.per_subject.filter(r=>match(r,dim)&&(!individual||r.subject===Number(subject)));
  const count=rows.filter(r=>r.model==='prior');
  pr('selection').textContent=count.reduce((n,r)=>n+r.pairs,0)+' held-out actual pairs · '+count.length+' participants';
  pr('scores').replaceChildren();pr('contrasts').replaceChildren();
  for(const [model,label] of Object.entries(paired.models)){
   const r=individual?rows.find(r=>r.model===model):paired.aggregate.find(r=>match(r,dim)&&r.model===model);
   const value=k=>individual?r[k]:r[k]?.mean;
   add(pr('scores'),[label,...['balanced_accuracy','accuracy','first_recall','second_recall'].map(k=>pc(value(k))),fixed(value('log_loss'),4),fixed(value('auc'),3)]);
  }
  for(const r of paired.contrasts.filter(r=>match(r,dim))){
   const person=r.participants.find(p=>p.subject===Number(subject));
   const effect=(name,factor,d)=>individual?fixed(factor*person[name],d):fixed(factor*r[name].mean,d)+' ['+r[name].ci95.map(x=>fixed(factor*x,d)).join(', ')+']';
   const p=name=>individual?'—':fixed(r[name].test.p,4)+' / '+fixed(r[name].test.p_holm,4);
   add(pr('contrasts'),[r.label,effect('balanced_accuracy_gain',100,2),p('balanced_accuracy_gain'),effect('log_loss_reduction',1,4),p('log_loss_reduction')]);
  }
  pr('plot').src=paired.figures[dim.join(':')];
 }
 pr('coverage').textContent=paired.pairs+' actual pairs · '+paired.epochs+' response epochs · '+paired.participants+' participants · '+paired.folds+' supported configuration/block fits · '+paired.unsupported.length+' unsupported folds';
 pr('subject').replaceChildren(new Option('Participant mean','all'),...[...new Set(paired.per_subject.map(r=>r.subject))].map(s=>new Option('Participant '+s,String(s))));
 for(const id of ['analysis','representation','width','subject'])pr(id).addEventListener('change',pairedDraw);
 pairedDraw();
}
'''
