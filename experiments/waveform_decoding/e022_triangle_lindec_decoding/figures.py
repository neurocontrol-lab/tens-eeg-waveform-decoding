"""Publication figures derived from saved predictions and behavioral counts."""

import base64
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from scipy.stats import hypergeom

from .report import SUBSETS


BLUE = '#326d9c'
ORANGE = '#d47a34'
COLORS = LinearSegmentedColormap.from_list('balanced_accuracy', [BLUE, '#f7f7f7', ORANGE])
DISPLAY_WAVEFORMS = {
    239: 'Sine', 240: 'Square', 241: 'Triangle',
    242: 'Linear decrease', 243: 'Linear increase',
}


def save(fig, path):
    fig.savefig(path, dpi=135, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return 'data:image/png;base64,' + base64.b64encode(path.read_bytes()).decode('ascii')


def preference_view(behavior, target, opponent, output):
    selected = behavior.loc[
        behavior.first_trigger.eq(target) | behavior.second_trigger.eq(target)
    ]
    if opponent:
        selected = selected.loc[
            selected.first_trigger.eq(opponent) | selected.second_trigger.eq(opponent)
        ]
    name = DISPLAY_WAVEFORMS[target]
    comparator = DISPLAY_WAVEFORMS[opponent] if opponent else 'all other waveforms'
    totals, participants = [], []
    for position, field in ((1, 'first_trigger'), (2, 'second_trigger')):
        rows = selected.loc[selected[field].eq(target)]
        totals.append(dict(position=position, votes=int(rows.chosen_trigger.eq(target).sum()),
                           trials=len(rows)))
    for subject, group in selected.groupby('subject'):
        row = dict(subject=int(subject))
        for position, field in ((1, 'first_trigger'), (2, 'second_trigger')):
            rows = group.loc[group[field].eq(target)]
            row[f'position_{position}_votes'] = int(rows.chosen_trigger.eq(target).sum())
            row[f'position_{position}_trials'] = len(rows)
        participants.append(row)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.1), gridspec_kw={'width_ratios': [1, 2]})
    for result, color in zip(totals, (BLUE, ORANGE)):
        rate = 100 * result['votes'] / result['trials']
        axes[0].bar(result['position'], rate, color=color, width=.6)
        axes[0].text(result['position'], rate + 2, f'{result["votes"]}/{result["trials"]}', ha='center')
    axes[0].set_xticks([1, 2], [name + '\nfirst', name + '\nsecond'])
    axes[0].set_ylim(0, 112)
    axes[0].set_yticks([0, 25, 50, 75, 100])
    axes[0].set_ylabel('Chosen as more natural (%)')
    axes[0].set_title('Compared with ' + comparator, loc='left', fontsize=11)
    subjects = [row['subject'] for row in participants]
    rates = []
    for position in (1, 2):
        rates.append(np.array([
            100 * row[f'position_{position}_votes'] / row[f'position_{position}_trials']
            if row[f'position_{position}_trials'] else np.nan
            for row in participants
        ]))
    x = np.arange(len(subjects))
    both = np.isfinite(rates[0]) & np.isfinite(rates[1])
    for index in x[both]:
        axes[1].plot([index - .07, index + .07], [rates[0][index], rates[1][index]],
                     color='#acb6bd', linewidth=1.5)
    axes[1].scatter(x - .07, rates[0], color=BLUE, label='First', zorder=3)
    axes[1].scatter(x + .07, rates[1], color=ORANGE, label='Second', zorder=3)
    axes[1].set_xticks(x, [str(subject) for subject in subjects])
    axes[1].set_ylim(-3, 103)
    axes[1].set_xlabel('Participant')
    axes[1].set_ylabel('Chosen as more natural (%)')
    axes[1].set_title('Participant choices', loc='left')
    axes[1].legend(frameon=False, loc='lower left')
    fig.suptitle(name + ': naturalness choices by presentation position', x=.02, ha='left')
    fig.tight_layout(rect=(0, 0, 1, .94))
    image = save(fig, output / f'wd008_preference_{target}_vs_{opponent}.png')
    return dict(
        target=target, opponent=opponent, trials=len(selected), totals=totals,
        participants=participants, image=image,
        caption='Rates describe the selected waveform comparison. Counts are shown on the bars and in the participant table. A missing presentation position has no plotted point.',
    )


def wd008_figures(summary, audit, behavior, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    distribution = np.array([1.])
    for _, group in behavior.groupby(['subject', 'block_index']):
        n = len(group)
        k = int(group.chosen_trigger.eq(group.low.iloc[0]).sum())
        m = int(group.first_trigger.eq(group.low.iloc[0]).sum())
        distribution = np.convolve(distribution, hypergeom.pmf(np.arange(m + 1), n, k, m))
    support = np.arange(len(distribution))
    observed = audit['primary']['observed_low_first_and_chosen']
    expectation = audit['primary']['null_expected']
    extreme = np.abs(support - expectation) >= abs(observed - expectation) - 1e-10
    np.testing.assert_allclose(distribution[extreme].sum(), audit['primary']['two_sided_p'], atol=1e-12)
    # Express the same statistic as the total first-position choices.
    offset = audit['first_chosen'] - 2 * observed
    support = offset + 2 * support
    observed = audit['first_chosen']
    expectation = offset + 2 * expectation
    fig, ax = plt.subplots(figsize=(10, 4.7))
    ax.bar(support, distribution, color=np.where(extreme, ORANGE, BLUE), width=1.8)
    radius = abs(observed - expectation)
    ax.axvspan(expectation - radius - 10, expectation - radius, color=ORANGE, alpha=.12)
    ax.axvspan(expectation + radius, expectation + radius + 10, color=ORANGE, alpha=.12)
    ax.axvline(observed, color=ORANGE, linestyle='--', linewidth=2.6)
    ax.axvline(expectation, color='#4b5563', linestyle='--', linewidth=1.8)
    ax.set_ylim(0, float(distribution.max()) * 1.25)
    ax.annotate(f'Observed: {observed}', xy=(observed, .97), xycoords=('data', 'axes fraction'),
                xytext=(7, 0), textcoords='offset points', ha='left', va='top',
                color=ORANGE, fontweight='bold')
    ax.annotate(f'Expected: {expectation:.0f}', xy=(expectation, .97), xycoords=('data', 'axes fraction'),
                xytext=(7, 0), textcoords='offset points', ha='left', va='top',
                color='#4b5563', fontweight='bold')
    ax.set_xlim(expectation - radius - 8, expectation + radius + 8)
    ax.set_xlabel('Total number of first-position choices')
    ax.set_ylabel('Conditional null probability')
    ax.set_title('Order test: expected first-position choices under the null', loc='left')
    ax.text(.98, .78, f'Two-sided p = {audit["primary"]["two_sided_p"]:.8f}',
            transform=ax.transAxes, ha='right', va='top')
    null_image = save(fig, output / 'wd008_order_null.png')

    preference = dict(waveforms=DISPLAY_WAVEFORMS, views={})
    for target in DISPLAY_WAVEFORMS:
        for opponent in [0, *[code for code in DISPLAY_WAVEFORMS if code != target]]:
            preference['views'][f'{target}:{opponent}'] = preference_view(
                behavior, target, opponent, output,
            )

    fig, ax = plt.subplots(figsize=(11, 6))
    models = list(summary['model_labels'])
    for index, name in enumerate(models):
        result = summary['full']['models'][name]['participant_mean']['balanced_accuracy']
        mean = 100 * result['mean']
        low, high = np.array(result['ci95']) * 100
        color = ORANGE if 'post' in name else BLUE
        ax.errorbar(mean, index, xerr=[[mean - low], [high - mean]],
                    fmt='o', capsize=3, color=color)
    ax.set_yticks(np.arange(len(models)), [summary['model_labels'][name] for name in models])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.axvline(50, color='#727b82', linestyle='--', label='50% reference')
    ax.set_xlabel('Participant-mean balanced accuracy (%)')
    ax.set_title('Existing exploratory naturalness decoding', loc='left')
    ax.legend(frameon=False, loc='lower right')
    fig.tight_layout()
    eeg_image = save(fig, output / 'wd008_eeg_models.png')
    return dict(
        behavior=[
            dict(url=null_image, caption='The shaded tails define the two-sided exact conditional order test. Vote totals and presentation-order counts are fixed within blocks.'),
        ],
        preference=preference,
        eeg=[dict(url=eeg_image, caption='Historical WD-008 EEG fits with descriptive participant intervals. Further naturalness decoding is deferred.')],
    )


def wd009_figures(summary, events, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    contexts = {}
    configurations = [(width, model) for width in (50, 100) for model in summary['models']]
    labels = [f'{width} ms · {summary["models"][model]}' for width, model in configurations]
    for subset in SUBSETS:
        for position in (1, 2):
            aggregate = [next(
                row for row in summary['aggregate']
                if (row['subset'], row['position'], row['width_ms'], row['model']) ==
                (subset, position, width, model)
            ) for width, model in configurations]
            fig, ax = plt.subplots(figsize=(12, 5.4))
            y = np.arange(len(aggregate))
            for period, offset, color in (('pre', -.13, BLUE), ('post', .13, ORANGE)):
                means = np.array([row[period]['mean'] for row in aggregate]) * 100
                bounds = np.array([row[period]['ci95'] for row in aggregate]) * 100
                ax.errorbar(means, y + offset, xerr=[means - bounds[:, 0], bounds[:, 1] - means],
                            fmt='o', capsize=3, color=color, label=period.title() + ' EEG')
            ax.set_yticks(y, labels)
            ax.invert_yaxis()
            ax.set_xlim(0, 100)
            ax.axvline(50, linestyle=':', color='#727b82')
            for index, row in enumerate(aggregate):
                test = row.get('post_pre_test')
                if test:
                    ax.text(99, index, f"Holm p={test['p_holm']:.3f}",
                            ha='right', va='center', fontsize=8,
                            bbox=dict(facecolor='white', edgecolor='none', alpha=.85))
            ax.set_xlabel('Participant-mean balanced accuracy (%)')
            ax.set_title(f'{SUBSETS[subset]} · presentation {position}', loc='left')
            ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(.5, -.14), ncol=2)
            fig.tight_layout()
            overview = save(fig, output / f'wd009_{subset}_p{position}_overview.png')

            selected = [row for row in summary['per_subject']
                        if row['subset'] == subset and row['position'] == position]
            subjects = sorted(set(row['subject'] for row in selected))
            fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=True)
            x = np.arange(len(subjects))
            for ax, (width, model), label in zip(axes.flat, configurations, labels):
                for period, color, offset in (('pre', BLUE, -.08), ('post', ORANGE, .08)):
                    scores = [100 * next(row['balanced_accuracy'] for row in selected
                              if (row['subject'], row['width_ms'], row['model'], row['period']) ==
                              (subject, width, model, period)) for subject in subjects]
                    ax.scatter(x + offset, scores, color=color, label=period.title() + ' EEG', s=26)
                ax.axhline(50, linestyle=':', color='#727b82', linewidth=1.5)
                ax.set_xticks(x, [str(subject) for subject in subjects])
                ax.set_ylim(0, 100)
                ax.set_xlabel('Participant')
                ax.set_title(label, loc='left', fontsize=10)
            for ax in axes[:, 0]:
                ax.set_ylabel('Balanced accuracy (%)')
            axes[0, 0].legend(frameon=False, fontsize=8)
            fig.suptitle(f'{SUBSETS[subset]} · presentation {position}', x=.02, ha='left')
            fig.tight_layout(rect=(0, 0, 1, .93))
            participants = save(fig, output / f'wd009_{subset}_p{position}_participants.png')
            contexts[f'{subset}:{position}'] = dict(overview=overview, participants=participants)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    subjects = sorted(events.subject.unique())
    y = np.arange(len(subjects))
    for ax, position in zip(axes, (1, 2)):
        for trigger, offset, color, label in ((241, -.18, BLUE, 'Triangle'),
                                               (242, .18, ORANGE, 'LinDec')):
            counts = [int((events.subject.eq(subject) & events.position.eq(position) &
                           events.trigger.eq(trigger)).sum()) for subject in subjects]
            ax.barh(y + offset, counts, height=.34, color=color, label=label)
            for index, count in enumerate(counts):
                ax.text(count + .15, index + offset, str(count), va='center', fontsize=9)
        ax.set_yticks(y, [str(subject) for subject in subjects])
        ax.set_xlim(0, 20)
        ax.set_xlabel('Eligible epochs, all comparison contexts')
        ax.set_title(f'Presentation {position}', loc='left')
        ax.legend(frameon=False, loc='lower right')
    axes[0].set_ylabel('Participant')
    axes[0].invert_yaxis()
    fig.tight_layout()
    counts_image = save(fig, output / 'wd009_epoch_counts.png')

    fig, ax = plt.subplots(figsize=(10, 5.8))
    for index, row in enumerate(summary['charge_table']):
        ax.plot([row['triangle_uC'], row['lindec_uC']], [index - .1, index + .1],
                color='#9aa8b1', linewidth=1.5)
        ax.scatter(row['triangle_uC'], index - .1, color=BLUE, label='Triangle' if index == 0 else None)
        ax.scatter(row['lindec_uC'], index + .1, color=ORANGE, label='LinDec' if index == 0 else None)
    ax.set_yticks(np.arange(len(subjects)), [str(subject) for subject in subjects])
    ax.invert_yaxis()
    ax.set_xlabel('Nominal cathodic charge (µC)')
    ax.set_ylabel('Participant')
    ax.set_title('Charge matches only where both waveform-specific thresholds match', loc='left')
    ax.legend(frameon=False)
    fig.tight_layout()
    charge_image = save(fig, output / 'wd009_nominal_charge.png')
    return dict(
        contexts=contexts,
        common=[
            dict(url=counts_image, caption='Available epochs per participant, waveform and presentation across all Task 4 comparison contexts.'),
            dict(url=charge_image, caption='Documented nominal charge. Participants 3, 4, 5, 6 and 10 have equal Triangle/LinDec values; measured delivered charge is unavailable.'),
        ],
    )
