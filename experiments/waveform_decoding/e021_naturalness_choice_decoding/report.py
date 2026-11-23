"""Build portable reports from saved held-out predictions and summaries."""

import argparse
import hashlib
from html import escape
import json
from pathlib import Path

from .evaluate import MODEL_LABELS


def behavior_markdown(audit):
    if audit is None:
        return []
    return [
        '## Behavioral order audit and current decision', '',
        '**Further naturalness decoding is deferred.** Existing EEG results below are retained as exploratory results.',
        '',
        f"The behavior-only audit includes {audit['primary']['trials']} choices from "
        f"{audit['primary']['participants']} participants and {audit['primary']['blocks']} comparison blocks. "
        f"The first stimulus was selected {audit['first_chosen']} times and the second {audit['second_chosen']} times.",
        '',
        '| Behavioral data | Trials | Exact conditional two-sided p |',
        '| --- | ---: | ---: |',
        *[
            f"| {title} | {audit[key]['trials']} | {audit[key]['two_sided_p']:.6g} |"
            for key, title in (
                ('primary', 'All Task 4 comparison blocks'),
                ('triangle_lindec', 'Direct triangle versus linear decrease'),
                ('triangle_lindec_nonterminal', 'Direct pairs with terminal votes removed; sensitivity analysis'),
            )
        ],
        '',
        '### How the p-value was calculated', '',
        'We checked whether presentation order was associated with participants’ choices.',
        '',
        'We kept each participant’s waveform preferences fixed and calculated what could happen if the '
        'presentation orders were rearranged within each comparison block, preserving how often each waveform appeared first.',
        '',
        'Across **623 trials**, the first stimulus would be chosen about **321 times (51.6%)** '
        'on average under the null model. It was actually chosen **284 times (45.6%)**—about '
        '**37 fewer choices**, a difference of **6 percentage points**.',
        '',
        'A difference this large, in either direction, occurs in about **1 out of 22,000** allowed order '
        'arrangements. This gives a **p-value of approximately 0.000045**.',
        '',
        'This provides evidence that choices were associated with presentation order. '
        'The test alone cannot identify what caused that association.',
        '',
        '### Statistical method and references', '',
        audit['method'],
        'The implementation defines a two-sided result by its absolute distance from the conditional mean. '
        'Exact-test software can use other two-sided conventions. '
        '[Jung (2014), stratified exact testing](https://doi.org/10.1002/bimj.201300048) · '
        '[SciPy, fixed-margin hypergeometric calculation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html)',
        '',
        audit['assumptions'], '',
        'The primary result favors second-position choices after conditioning on participant and waveform '
        'comparison. The two direct-pair results are exploratory sensitivity checks with unadjusted p-values. '
        'Their different outcomes show sensitivity to the stopping-rule selection.', '',
        'This audit supports order-dependent preferences. It does not establish incorrect behavioral entries, '
        'physiological carryover, or unusable second-presentation EEG. The log records each choice in its '
        'presentation context. A metadata classifier could measure predictability, while high accuracy can '
        'also reflect consistent waveform preferences.', '',
    ]


def behavior_html(audit):
    if audit is None:
        return ''
    rows = ''.join(
        f'<tr><td>{title}</td><td>{audit[key]["trials"]}</td>'
        f'<td>{audit[key]["two_sided_p"]:.6g}</td></tr>'
        for key, title in (
            ('primary', 'All Task 4 comparison blocks'),
            ('triangle_lindec', 'Direct triangle / linear decrease'),
            ('triangle_lindec_nonterminal', 'Direct pairs without terminal votes · sensitivity'),
        )
    )
    return (
        '<section id="behavioral-order-audit"><h2>Behavioral order audit</h2>'
        '<p><strong>Further naturalness decoding is deferred.</strong> Existing fitted EEG results are retained below.</p>'
        f'<p>The audit uses {audit["primary"]["trials"]} choices in {audit["primary"]["blocks"]} blocks from '
        f'{audit["primary"]["participants"]} participants. First was chosen {audit["first_chosen"]} times; '
        f'second was chosen {audit["second_chosen"]} times.</p>'
        '<div class="scroll"><table><thead><tr><th>Data</th><th>Trials</th>'
        '<th>Exact conditional two-sided p</th></tr></thead><tbody>'+rows+'</tbody></table></div>'
        '<p>The conditional test holds the observed waveform choices and order counts fixed within each '
        'participant/comparison block. It tests whether choices associate with first versus second presentation, '
        'assuming presentation order was exchangeable within blocks under the null. Acquisition randomization '
        'code remains unavailable. The direct-pair checks are exploratory and their p-values are unadjusted.</p>'
        '<h3>How the p-value was calculated</h3>'
        '<p>We checked whether presentation order was associated with participants’ choices.</p>'
        '<p>We kept each participant’s waveform preferences fixed and calculated what could happen if the '
        'presentation orders were rearranged within each comparison block, preserving how often each waveform appeared first.</p>'
        '<p>Across <strong>623 trials</strong>, the first stimulus would be chosen about '
        '<strong>321 times (51.6%)</strong> on average under the null model. It was actually chosen '
        '<strong>284 times (45.6%)</strong>—about <strong>37 fewer choices</strong>, a difference of '
        '<strong>6 percentage points</strong>.</p>'
        '<p>A difference this large, in either direction, occurs in about <strong>1 out of 22,000</strong> '
        'allowed order arrangements. This gives a <strong>p-value of approximately 0.000045</strong>.</p>'
        '<p>This provides evidence that choices were associated with presentation order. '
        'The test alone cannot identify what caused that association.</p>'
        '<details><summary>Statistical method and references</summary>'
        '<p>For each block, A is the lower trigger code used as a reference. Count N trials, m with A first, '
        'K choosing A, and X with A both first and chosen. Under the no-order-effect null and the assumed '
        'exchangeable randomization, X follows a hypergeometric distribution with parameters N, K and m.</p>'
        '<p>Sum X over all 100 blocks to get T. The observed T is 133 and its conditional null expectation '
        'is 151.6726. The two-sided p-value is the probability that a null T is at least 18.6726 away '
        'from that expectation. Exact convolution of the block distributions gives p = 0.0000454475. '
        'Waveform vote totals are held fixed, accounting for consistent waveform preferences within each block.</p>'
        '<p>The two-sided rule uses absolute distance from the conditional mean. '
        'Exact-test software can use other two-sided conventions.</p>'
        '<p><a href="https://doi.org/10.1002/bimj.201300048">Jung (2014), stratified exact conditional testing</a> · '
        '<a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html">'
        'SciPy, fixed-margin hypergeometric calculation</a></p></details>'
        '<p>The primary result favors second-position choices. An order association does not establish errors '
        'in the behavioral records, physiological carryover, or unusable second-presentation EEG. The logs '
        'record what was preferred in that trial context. A high metadata-classifier accuracy can also reflect '
        'consistent waveform preferences and would not by itself establish a recording problem.</p>'
        '</section>'
    )


def render_markdown(summary):
    if 'paired_revision' in summary:
        from .paired_report import render_markdown as paired_markdown
        audit = [line.replace('**Further naturalness decoding is deferred.** Existing EEG results below are retained as exploratory results.',
                              'The behavioral association provides context for the within-participant paired EEG analysis.')
                 for line in behavior_markdown(summary.get('behavior_audit'))]
        return paired_markdown(summary['paired_revision'])+'\n'+'\n'.join(audit)
    if 'position_revision' in summary:
        from .position_report import render_markdown as render_revision
        audit = behavior_markdown(summary.get('behavior_audit'))
        audit = [line.replace('**Further naturalness decoding is deferred.** Existing EEG results below are retained as exploratory results.',
                             'The within-participant test is reported above. The order association is retained as behavioral context.')
                 for line in audit]
        return (render_revision(summary['position_revision'])+'\n'+'\n'.join(audit)+
                '\n## Archived population analysis\n\nThe earlier leave-one-participant-out fits are superseded for the current research question. '
                'They remain available in the interactive report’s collapsed archive and '
                '[the preserved report version](https://github.com/neurocontrol-lab/tens-eeg-waveform-decoding/blob/5436f02/Docs/WD008_NATURALNESS_CHOICE_RESULTS.md).\n')
    full = summary['full']
    lines = [
        '# WD-008. Task 4 naturalness-choice decoding',
        '',
        '## Research question and origin', '',
        'Jason Whitson reported consistent naturalness preferences, with an average winner share of 83.5% '
        'against the race-to-five null expectation of 68.29%. WD-008 asks whether EEG predicts the recorded '
        'first-or-second choice and whether post EEG improves on pre EEG and task metadata. '
        '[Whitson thesis](https://stars.library.ucf.edu/hut2024/661/)',
        '',
        *behavior_markdown(summary.get('behavior_audit')),
        f"**{summary['date']}.** {full['pairs']} eligible pairs from {full['participants']} participants.",
        '',
        'The target is the participant’s first-or-second naturalness choice on each pair. '
        'Each pair receives one prediction from models trained on the other participants.',
        '',
        '## What the completed run covers',
        '',
        '| Item | Completed WD-008 analysis |',
        '| --- | --- |',
        f"| Dataset | Task 4, {full['pairs']} eligible pairs from {full['participants']} participants; "
        'all five waveforms and all ten waveform comparisons |',
        '| Labels | T4 Naturalness Value from Jason’s behavioral log: 1 = first stimulus chosen, '
        '2 = second stimulus chosen. The classifier encodes these as 0 and 1. |',
        '| Validation | Train on nine participants and test on the remaining participant. '
        'The participant selector displays held-out participant scores. |',
        '| Presentations | Main paired post EEG is second minus first. '
        'Standalone pre/post models use the first presentation. |',
        f"| Triangle/linear-decrease subset | {summary['subsets']['matched_triangle_lindec_pairs']} "
        'matched-charge pairs from participants 3, 4, 5, 6 and 10; '
        'scores reuse the full-cohort naturalness-model predictions. |',
        '| Requested within-participant naturalness decoders | Deferred following the behavioral order audit |',
        '',
        'The naturalness follow-up is deferred following the behavioral order audit.',
        '',
        'Collu et al. reported 70% median match-to-sample discrimination accuracy for triangle versus '
        'linear decrease under matched stimulation settings. This measures correct distinction of the stimuli. '
        'A directional naturalness preference requires a separate preference judgment. '
        '[Collu et al.](https://doi.org/10.1038/s41598-023-28594-0)',
        '',
        '## Held-out results',
        '',
        'Scores average participants equally. Log loss measures probability prediction error; lower is better.',
        '',
        '| Model | Balanced accuracy | Accuracy | Log loss | ROC AUC |',
        '| --- | ---: | ---: | ---: | ---: |',
    ]
    for name, label in MODEL_LABELS.items():
        values = full['models'][name]['participant_mean']
        lines.append(
            f"| {label} | {values['balanced_accuracy']['mean']:.2%} | "
            f"{values['accuracy']['mean']:.2%} | {values['log_loss']['mean']:.4f} | "
            f"{values['auc']['mean']:.3f} |"
        )
    lines += [
        '', '## Paired comparisons', '',
        'Positive changes favor the model with post-stimulus EEG. Intervals resample participants '
        '10,000 times using fixed held-out predictions and are descriptive.',
        '',
        '| Analysis | Log-loss reduction [95% interval] | Balanced-accuracy change, percentage points [95% interval] |',
        '| --- | ---: | ---: |',
    ]
    for cohort, title in (('full', 'All eligible pairs'), ('nonterminal', 'Terminal votes removed and models refitted')):
        for key, description in (
            ('post_added_to_metadata_pre', 'Add paired post EEG to metadata and pre EEG'),
            ('post_vs_pre_alone', 'First-pulse post EEG versus first-pulse pre EEG'),
        ):
            comparison = summary[cohort]['contrasts'][key]
            loss, accuracy = comparison['log_loss_reduction'], comparison['balanced_accuracy_gain']
            lines.append(
                f"| {title}: {description} | {loss['mean']:+.4f} "
                f"[{loss['ci95'][0]:+.4f}, {loss['ci95'][1]:+.4f}] | "
                f"{100*accuracy['mean']:+.2f} "
                f"[{100*accuracy['ci95'][0]:+.2f}, {100*accuracy['ci95'][1]:+.2f}] |"
            )
    lines += ['', '## Secondary subsets', '', summary['subset_policy'], '',
              '| Subset | Pairs | Participants | Metadata BA | Metadata + pre BA | Metadata + pre + paired post BA | Pre alone BA | Post alone BA |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for key, title in (('varying_choice', 'Same ordered pair with varying choices'),
                       ('matched_triangle_lindec', 'Nominal-charge-matched triangle versus linear decrease')):
        result = summary[key]
        scores = [result['models'][name]['participant_mean']['balanced_accuracy']['mean']
                  for name in ('metadata', 'metadata_pre', 'metadata_pre_post', 'pre_only', 'post_only')]
        lines.append(f"| {title} | {result['pairs']} | {result['participants']} | " +
                     ' | '.join(f'{value:.2%}' for value in scores) + ' |')
    subset = summary['subsets']
    lines += [
        '',
        f"Linear decrease received {subset['matched_lindec_votes']} of "
        f"{subset['matched_triangle_lindec_pairs']} votes in the matched subset. "
        'This is an observed vote share; decoder scores predict first-versus-second choices.',
        '', '## Methods and checks', '',
        '- Six fixed contralateral electrodes: C4, CP2, CP4, CP6, P2 and P4.',
        '- Five 50 ms means per electrode, giving 30 EEG features per window.',
        '- First-pulse pre EEG spans −350 to −100 ms; first- and second-pulse post EEG span 100 to 350 ms. '
        'The paired representation is second minus first post EEG.',
        '- Minimal processing uses a 62-channel instantaneous average reference and microvolt conversion. '
        'The same first-pulse baseline from −550 to −350 ms is subtracted from both windows and both post responses. '
        'It cancels in the paired difference. No temporal filtering or ICA is applied.',
        '- Metadata consists of 20 fixed ordered-pair indicators and chronological and within-block pair indices.',
        '- Logistic regression uses fixed C=1 and training-only standardization. No label-based feature selection, '
        'class weighting or held-out participant normalization is used.',
        '- Identical leave-one-participant-out folds and eligible pairs are used for all eight models.',
        '- Existing complete-pair QC and verified XDF-to-CSV alignment are reused. Raw-input hashes, feature-cache '
        'hashes, feature counts, constant-feature counts and convergence are saved.',
        '', '## Interpretation and timing', '',
        '- The standalone comparison uses the first pulse in both periods. The primary paired post EEG also '
        'contains the second response and can include carryover from the first.',
        '- The CSV contains choices but no response timestamps. A 350 ms post-trigger endpoint cannot be verified '
        'as preceding the behavioral response. Overlap with the previous response is also unverified.',
        '- Nominal charge determines secondary subset membership only and is excluded from model predictors.',
        '- Subsets selected for varying observed choices are descriptive. Terminal-vote removal is a sensitivity '
        'check for the stopping rule; the reduced cohort retains the history of the original task.',
        '- The 68.29% expected winner share describes the race-to-five behavioral statistic. '
        'Balanced accuracy has a 50% binary reference. A useful EEG improvement must also be compared with '
        'the metadata and pre-EEG models.',
        '- Bootstrap intervals condition on the fitted cross-validation models. No permutation significance test '
        'was run, and these summaries do not establish an artifact-free neural mechanism.',
        '',
    ]
    if 'unbaselined_reference' in summary:
        lines += [
            '## Processing adaptation', '',
            summary['unbaselined_reference']['reason'],
            'The initial run is retained locally. Its uncorrected DC offsets reached approximately 20,000 µV. '
            'The adaptation followed inspection of the initial features and scores and is exploratory.', '',
            '| Unbaselined model | Balanced accuracy | Log loss |',
            '| --- | ---: | ---: |',
        ]
        for name in ('pre_only', 'post_only', 'metadata_pre', 'metadata_pre_post'):
            values = summary['unbaselined_reference']['full']['models'][name]['participant_mean']
            lines.append(f"| {MODEL_LABELS[name]} | {values['balanced_accuracy']['mean']:.2%} | "
                         f"{values['log_loss']['mean']:.4f} |")
        lines.append('')
    return '\n'.join(lines)


def render_html(summary, figures=None):
    figures = figures or {}
    paired = summary.get('paired_revision')
    shared = (dict(paired_revision=paired, behavior_audit=summary.get('behavior_audit'),
                   preference_explorer=figures.get('preference', {})) if paired else
              dict(summary, preference_explorer=figures.get('preference', {})))
    data = json.dumps(shared, allow_nan=False).replace('<', '\\u003c')
    revision = summary.get('position_revision')
    audit = behavior_html(summary.get('behavior_audit'))
    if paired:
        from .paired_report import HTML as revision_html, JS as revision_js, result_text
        revision_html = revision_html.replace('__RESULTS__', escape(result_text(paired)))
        audit = audit.replace('<p><strong>Further naturalness decoding is deferred.</strong> Existing fitted EEG results are retained below.</p>',
                              '<p>The within-participant paired EEG tests below use these recorded choices with metadata and pre-pair EEG controls.</p>')
    elif revision:
        from .position_report import HTML as revision_html, JS as revision_js
        from .position_report import result_sentences
        revision_html = revision_html.replace('__RESULTS__', escape(' '.join(result_sentences(revision))))
        audit = audit.replace('<p><strong>Further naturalness decoding is deferred.</strong> Existing fitted EEG results are retained below.</p>',
                              '<p>The participant-specific EEG tests below retain the recorded choices and compare EEG against metadata and pre-stimulation controls.</p>')
    else:
        revision_html = revision_js = ''
    def image_html(item):
        return (f'<figure><img src="{item["url"]}" alt="{escape(item["caption"])}">'
                f'<figcaption>{escape(item["caption"])}</figcaption></figure>')
    page = PAGE
    if paired:
        page = page.split('__LEGACY_START__')[0]+page.split('__LEGACY_END__')[1]
    return (page.replace('__SUMMARY__', data)
            .replace('__REVISION__', revision_html).replace('__REVISION_JS__', revision_js)
            .replace('__LEGACY_START__', '<details id="legacy-analysis"><summary>Archived population analysis · superseded leave-one-participant-out models</summary><p>The following sections describe the earlier population analysis. The current participant-specific tests appear above.</p>' if revision else '')
            .replace('__LEGACY_END__', '</details>' if revision else '')
            .replace('__AUDIT__', audit)
            .replace('__BEHAVIOR_FIGURES__', ''.join(image_html(item) for item in figures.get('behavior', [])))
            .replace('__EEG_FIGURE__', ''.join(image_html(item) for item in figures.get('eeg', []))))


PAGE = """<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>WD-008 · Task 4 naturalness-choice decoding</title>
<style>
body{font:16px/1.55 system-ui,sans-serif;color:#172c37;background:#f7f8fa;margin:0}
*{box-sizing:border-box}main{max-width:1150px;margin:auto;padding:2rem 1.25rem 4rem}h1,h2{line-height:1.2}
h3{font-size:1.1rem;line-height:1.35;margin:1.3rem 0 .45rem}
section{background:white;padding:1.25rem;margin:1.5rem 0;border:1px solid #dce4e5;border-radius:.5rem}
p{margin:.65rem 0 1rem}.filters{display:flex;gap:1.25rem;flex-wrap:wrap}
label{display:grid;gap:.3rem}select{font:inherit;padding:.4rem}.scroll{overflow:auto}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:.55rem;text-align:right;border-bottom:1px solid #ddd;vertical-align:top}
th:first-child,td:first-child{text-align:left}thead{background:#edf3f3}small{color:#465762}
.scope-table{table-layout:fixed}.scope-table th,.scope-table td{text-align:left;white-space:normal;overflow-wrap:break-word}
.scope-table th:first-child,.scope-table td:first-child{width:28%;font-weight:600}
.data-table{min-width:650px}.scroll{max-width:100%}
details{margin:1rem 0}summary{cursor:pointer;font-weight:600}
figure{margin:1.25rem 0}figure img{display:block;width:100%;height:auto}
figcaption{font-size:.9rem;color:#465762}
</style>
<main><h1>Task 4 naturalness-choice decoding</h1>
<section><h2>Research question and origin</h2>
<p>Jason Whitson reported consistent naturalness preferences: 83.5% average winner vote share versus
68.29% expected under random choices with the race-to-five stopping rule. WD-008 tests whether EEG predicts
which stimulus a participant chose as more natural, and whether post EEG adds information beyond pre EEG
and task metadata.
<a href="https://stars.library.ucf.edu/hut2024/661/">Whitson thesis</a></p>
</section>
__AUDIT__
__BEHAVIOR_FIGURES__
<section id="preference_explorer" hidden><h2>Naturalness choices by waveform and presentation</h2>
<div class="filters">
<label>Waveform<select id="pref_waveform">
<option value="239">Sine</option><option value="240">Square</option><option value="241">Triangle</option>
<option value="242" selected>Linear decrease</option><option value="243">Linear increase</option>
</select></label>
<label>Compared with<select id="pref_opponent"><option value="0">All other waveforms</option></select></label>
</div>
<p id="preference_summary"></p>
<p>These descriptive rates use the behavioral choices for the selected comparison.
“All other waveforms” combines four different comparisons. The order audit above uses all Task 4 comparison blocks.</p>
<figure><img id="preference_plot" alt="Waveform naturalness choices by presentation position">
<figcaption id="preference_caption"></figcaption></figure>
<details><summary>Counts by participant</summary><div class="scroll"><table class="data-table">
<thead><tr><th>Participant</th><th>Chosen when first / presentations</th>
<th>Chosen when second / presentations</th></tr></thead><tbody id="preference_counts"></tbody>
</table></div></details></section>
__REVISION__
__LEGACY_START__
<p id="coverage"></p><p>Predict which of two stimuli a participant chose as more natural.
Each pair is held out with its participant. Metadata means waveform identities and trial order.</p>
<section><h2>What the completed run covers</h2>
<p>The main analysis uses all five Task 4 waveforms and all ten waveform comparisons from participants 3–12.</p>
<div class="scroll"><table class="scope-table"><thead><tr><th>Item</th><th>Completed analysis</th></tr></thead><tbody>
<tr><td>Target and label source</td><td>Jason’s T4 Naturalness Value: first or second stimulus chosen as more natural</td></tr>
<tr><td>Validation</td><td>Train on nine participants, test on the remaining participant</td></tr>
<tr><td>Presentations</td><td>Main post EEG is second minus first; standalone pre/post models use the first presentation</td></tr>
<tr><td>Triangle/linear-decrease view</td><td>Subset of full-cohort naturalness predictions, using nominal-charge-matched pairs</td></tr>
<tr><td>Within-participant naturalness decoders</td><td>Deferred following the behavioral audit</td></tr>
</tbody></table></div>
<p>The participant selector below displays each participant’s held-out score from the cross-participant models.
The triangle/linear-decrease view contains 33 matched-charge pairs from participants 3, 4, 5, 6 and 10.</p>
<p>The naturalness follow-up is deferred following the behavioral order audit.</p>
<p>Collu et al. reported 70% median correct discrimination of triangle versus linear decrease in a
match-to-sample task under matched stimulation settings. This measures distinguishing the stimuli.
A directional naturalness preference requires a separate preference judgment.
<a href="https://doi.org/10.1038/s41598-023-28594-0">Collu et al.</a></p>
</section>
<section><h2>Existing exploratory EEG performance</h2><div class="filters">
<label>Analysis<select id="analysis">
<option value="full">All eligible pairs</option><option value="nonterminal">Terminal votes removed · refitted</option>
<option value="varying_choice">Same ordered pair · varying choices</option>
<option value="matched_triangle_lindec">Triangle/linear decrease · naturalness subset</option></select></label>
<label>Participant<select id="participant"><option value="all">Participant mean</option></select></label></div>
<p id="selection"></p><div class="scroll"><table class="data-table"><thead><tr><th>Model</th><th>Balanced accuracy</th>
<th>Accuracy</th><th>Log loss ↓</th><th>ROC AUC</th></tr></thead><tbody id="metrics"></tbody></table></div>
<p><small>Log loss evaluates predicted probabilities. Lower is better. Balanced accuracy averages the
recall of choosing first and choosing second. Participant means give each participant equal weight.</small></p></section>
__EEG_FIGURE__
<section><h2>What does post-stimulation EEG add?</h2><div id="contrasts"></div>
<p><small>Intervals resample participants using fixed held-out scores. They are descriptive and condition
on the fitted models. Subset summaries use full-cohort predictions; terminal-vote removal refits the models.</small></p></section>
<section><h2>Choice recall</h2><div class="scroll"><table class="data-table"><thead><tr><th>Model</th><th>Choose first recall</th>
<th>Choose second recall</th><th>Pairs</th></tr></thead><tbody id="recall"></tbody></table></div>
<p><small>For participant means, recalls here pool the selected cohort's held-out pairs.</small></p></section>
<section id="methods"><h2>Methods and timing</h2>
<h3>Voltage features</h3>
<p>Six fixed electrodes (C4, CP2, CP4, CP6, P2, P4) and five 50 ms voltage means per electrode yield 30 EEG features.
First-pulse windows span −350 to −100 ms before the trigger and 100 to 350 ms afterward.
Paired post EEG subtracts the first response from the second over 100 to 350 ms.</p>
<h3>Reference and baseline</h3>
<p>Processing uses the instantaneous 62-channel average reference and microvolt conversion.
The same first-pulse baseline from −550 to −350 ms is subtracted from all windows.
It cancels in the paired difference. There is no temporal filtering or ICA.</p>
<h3>Logistic regression</h3>
<p>The classifier uses L2 regularization with fixed C=1 and training-only feature scaling.
It predicts the probability that the participant chose the second stimulus.</p>
<h3>Validation</h3>
<p>All models use identical leave-one-participant-out folds. Each pair receives one prediction
from a model trained on the other participants.</p>
<h3>Behavioral context and timing</h3>
<p id="behavior"></p>
<p>The CSV supplies naturalness choices without response timestamps. The post-trigger endpoint
and the pre-trigger window's separation from the preceding response cannot be checked against response timing.
The primary paired features include second-pulse carryover. The standalone pre/post comparison uses the first pulse.</p>
<p>Nominal charge defines the matched secondary subset and does not enter model features.
The varying-choice subset is selected descriptively using observed choices.
No permutation significance test was run.</p>
<p id="adaptation"></p></section>__LEGACY_END__</main>
<script type="application/json" id="data">__SUMMARY__</script>
<script>
const data=JSON.parse(document.getElementById('data').textContent);
const el=id=>document.getElementById(id);
const pct=x=>x===null?'—':(100*x).toFixed(2)+'%';
const number=x=>x===null?'—':x.toFixed(4);
const signed=x=>(x>=0?'+':'')+x.toFixed(4);
__REVISION_JS__
function preferenceMenus(){
 const explorer=data.preference_explorer;
 const target=el('pref_waveform').value,previous=el('pref_opponent').value;
 const others=Object.entries(explorer.waveforms).filter(([code])=>code!==target);
 el('pref_opponent').replaceChildren(new Option('All other waveforms','0'),
  ...others.map(([code,name])=>new Option(name,code)));
 if(previous==='0'||others.some(([code])=>code===previous))el('pref_opponent').value=previous;
 preferenceDraw();
}
function preferenceDraw(){
 const target=el('pref_waveform').value,opponent=el('pref_opponent').value;
 const explorer=data.preference_explorer,view=explorer.views[target+':'+opponent];
 const name=explorer.waveforms[target],comparison=opponent==='0'?'all other waveforms':explorer.waveforms[opponent];
 const first=view.totals[0],second=view.totals[1];
 el('preference_summary').textContent=name+' compared with '+comparison+' · '+view.trials+' pair choices. '+
  'Chosen when first: '+first.votes+'/'+first.trials+' ('+pct(first.trials?first.votes/first.trials:null)+
  '). Chosen when second: '+second.votes+'/'+second.trials+' ('+pct(second.trials?second.votes/second.trials:null)+').';
 el('preference_plot').src=view.image;
 el('preference_plot').alt=name+' choices compared with '+comparison+', split by presentation position';
 el('preference_caption').textContent=view.caption;
 el('preference_counts').replaceChildren();
 for(const record of view.participants){
  const row=document.createElement('tr');
  const values=[record.subject,...[1,2].map(position=>{
   const votes=record['position_'+position+'_votes'],trials=record['position_'+position+'_trials'];
   return trials?votes+'/'+trials+' ('+pct(votes/trials)+')':'—';
  })];
  for(const value of values){const td=document.createElement('td');td.textContent=value;row.append(td);}
  el('preference_counts').append(row);
 }
}
if(data.preference_explorer?.views){
 el('preference_explorer').hidden=false;
 el('pref_waveform').addEventListener('change',preferenceMenus);
 el('pref_opponent').addEventListener('change',preferenceDraw);
 preferenceMenus();
}
if(data.full){
if(data.unbaselined_reference)el('adaptation').textContent=data.unbaselined_reference.reason+
 ' The initial unbaselined run is preserved locally. The feature and score inspection that prompted this change makes the adaptation exploratory.';
el('coverage').textContent='WD-008 · '+data.date+' · '+data.full.pairs+' eligible pairs from '+data.full.participants+' participants';
el('behavior').textContent='Linear decrease was chosen on '+data.subsets.matched_lindec_votes+' of '+
data.subsets.matched_triangle_lindec_pairs+' nominal-charge-matched triangle/linear-decrease pairs. '+
'The 68.29% race-to-five winner share is a behavioral reference. Binary balanced accuracy has a 50% reference.';
function updateParticipants(){
 const current=el('participant').value;
 const subjects=[...new Set(data[el('analysis').value].per_subject.map(row=>row.subject))];
 el('participant').replaceChildren(new Option('Participant mean','all'),
   ...subjects.map(subject=>new Option('Participant '+subject,String(subject))));
 if(subjects.includes(Number(current)))el('participant').value=current;
 update();
}
function update(){
 const result=data[el('analysis').value], subject=el('participant').value;
 el('selection').textContent=result.pairs+' pairs · '+result.participants+' participants';
 el('metrics').replaceChildren();el('recall').replaceChildren();
 for(const [name,label] of Object.entries(data.model_labels)){
   const record=subject==='all'?result.models[name].participant_mean:
     result.per_subject.find(row=>row.model===name&&row.subject===Number(subject));
   const value=metric=>subject==='all'?record[metric].mean:record[metric];
   const row=document.createElement('tr');
   for(const content of [label,pct(value('balanced_accuracy')),pct(value('accuracy')),number(value('log_loss')),number(value('auc'))]){
     const td=document.createElement('td');td.textContent=content;row.append(td);
   }el('metrics').append(row);
   const pooled=subject==='all'?result.models[name].pooled:record,m=pooled.confusion;
   const recalls=m.map((r,i)=>r[i]/r.reduce((a,b)=>a+b,0));
   const recallRow=document.createElement('tr');
   for(const content of [label,...recalls.map(pct),pooled.pairs]){
     const td=document.createElement('td');td.textContent=content;recallRow.append(td);
   }el('recall').append(recallRow);
 }
 el('contrasts').replaceChildren();
 for(const [key,title] of [
   ['post_added_to_metadata_pre','Add paired post EEG to metadata and pre EEG'],
   ['post_vs_pre_alone','First-pulse post EEG versus pre EEG alone']]){
   const contrast=result.contrasts[key],p=document.createElement('p');
   if(subject==='all'){
     const loss=contrast.log_loss_reduction,ba=contrast.balanced_accuracy_gain;
     p.textContent=title+'. Log-loss reduction '+signed(loss.mean)+' ['+loss.ci95.map(signed).join(', ')+
       ']. Balanced-accuracy change '+signed(100*ba.mean)+' percentage points ['+ba.ci95.map(x=>signed(100*x)).join(', ')+'].';
   }else{
     const row=contrast.participants.find(row=>row.subject===Number(subject));
     p.textContent=title+'. Log-loss reduction '+signed(row.log_loss_reduction)+
       '. Balanced-accuracy change '+signed(100*row.balanced_accuracy_gain)+' percentage points.';
   }el('contrasts').append(p);
 }
}
el('analysis').addEventListener('change',updateParticipants);
el('participant').addEventListener('change',update);
updateParticipants();
}
</script></html>
"""


def main():
    parser = argparse.ArgumentParser(description='Refresh WD-008 presentation from saved predictions and an independent behavior audit')
    parser.add_argument('--run', type=Path, default=Path('output/wd008_naturalness_baselined_20261123'))
    parser.add_argument('--audit', type=Path, default=Path('output/wd008_behavior_audit_20261123/summary.json'))
    parser.add_argument('--output', type=Path, default=Path('output/wd008_audited_report_20261123'))
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh report directory')
    summary = json.loads((args.run / 'summary.json').read_text())
    summary['behavior_audit'] = json.loads(args.audit.read_text())
    args.output.mkdir(parents=True)
    (args.output / 'index.html').write_text(render_html(summary), encoding='utf-8')
    (args.output / 'RESULTS.md').write_text(render_markdown(summary), encoding='utf-8')
    provenance = {
        'numerical_summary': str(args.run / 'summary.json'),
        'numerical_summary_sha256': hashlib.sha256((args.run / 'summary.json').read_bytes()).hexdigest(),
        'audit': str(args.audit), 'audit_sha256': hashlib.sha256(args.audit.read_bytes()).hexdigest(),
        'renderer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'model_refitted': False,
    }
    (args.output / 'report_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    print('Updated report:', args.output.resolve())


if __name__ == '__main__':
    main()
