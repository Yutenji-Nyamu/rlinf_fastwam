"""Plot completed CP180 groups against realized takeover rates, with CSV export."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


def plot(summary_path, output):
    rows = json.loads(Path(summary_path).read_text())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    exported = []
    for rule, color, marker in [('high', '#D55E00', 'o'), ('low', '#0072B2', 's'), ('random', '#009E73', '^')]:
        selected = [row for row in rows if row['condition'].startswith(f'cp180_{rule}')]
        selected.sort(key=lambda row: row['strong_fraction'])
        if not selected:
            continue
        x = [100 * row['strong_fraction'] for row in selected]
        y = [100 * row['success_rate'] for row in selected]
        lower = [100 * (row['success_rate'] - row['wilson95'][0]) for row in selected]
        upper = [100 * (row['wilson95'][1] - row['success_rate']) for row in selected]
        ax.errorbar(x, y, yerr=[lower, upper], marker=marker,
                    linestyle='-' if len(selected) > 1 else 'none', color=color,
                    capsize=3, label=f'{rule.capitalize()} U' if rule != 'random' else 'Random')
        exported.extend(selected)
    for name, label, marker in [('weak_n10_c50', 'Full weak', 'x'), ('grpo_u_tau1_cp180', 'Full CP180', '*')]:
        row = next((row for row in rows if row['condition'] == name), None)
        if row:
            ax.scatter(100 * row['strong_fraction'], 100 * row['success_rate'],
                       marker=marker, s=90, color='#555555', label=label)
            exported.append(row)
    ax.set(xlim=(-3, 103), ylim=(0, 100), xlabel='Realized strong-policy decision fraction (%)',
           ylabel='Task success rate (%)', title='Pillbottle: frozen original / GRPO tau1 CP180')
    ax.grid(alpha=.2)
    ax.legend()
    fig.text(.5, .01, '128 paired episodes per point; error bars: Wilson 95% intervals. '
             'Lines connect completed settings, not fitted/extrapolated values.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .03, 1, 1))
    fig.savefig(output / 'cp180-takeover-rates.svg')
    fig.savefig(output / 'cp180-takeover-rates.png', dpi=140)
    plt.close(fig)
    with (output / 'cp180-takeover-rates.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['condition', 'successes', 'episodes', 'realized_decision_fraction', 'success_rate', 'wilson95_low', 'wilson95_high'])
        for row in exported:
            writer.writerow([row['condition'], row['successes'], row['episodes'], row['strong_fraction'], row['success_rate'], *row['wilson95']])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('summary')
    parser.add_argument('output')
    args = parser.parse_args()
    plot(args.summary, args.output)
