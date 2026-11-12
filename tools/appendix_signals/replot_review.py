"""Redraw numeric review panels from the released summaries; no GPU or weights.

Usage: python replot_review.py --data extracted_review --output figures
Raw K8 banks and simulation images remain in the companion simulation release.
The K statistics are within-observation, executed-position ranks, not ranks of
chunk means across observations. See REPORT.md for scope and definitions.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    a = json.loads((args.data / 'analysis.json').read_text())
    d = json.loads((args.data / 'display-data.json').read_text())
    plt.rcParams.update({'svg.fonttype': 'none', 'axes.spines.top': False, 'axes.spines.right': False})

    def save(fig, name):
        fig.tight_layout()
        for fmt in ['png', 'svg', 'pdf']:
            fig.savefig(args.output / (name + '.' + fmt), dpi=160, bbox_inches='tight')
        plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9))
    for ax, fid in zip(axes, ['4', '5', '7']):
        row = d['training'][fid]
        for i, series in reversed(list(enumerate(row['series']))):
            points = np.asarray(series['eval'])
            ax.plot(points[:, 0], points[:, 1], 'o-', ms=3, label=f'Segment {i}', color=['#197e98', '#859ba3', '#c3ced1'][i])
        ax.set(title=row['title'], xlabel='Logged training round', ylabel='Evaluation success (%)', ylim=(-5, 105))
        ax.legend(); ax.grid(alpha=.15)
    save(fig, 'training_three')

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    rows = a['K']; x = [r['K'] for r in rows]; y = np.asarray([r['rho_mean'] for r in rows]); ci = np.asarray([r['rho_ci95'] for r in rows])
    axes[0].errorbar(x, y, yerr=[y-ci[:, 0], ci[:, 1]-y], fmt='o-', capsize=4)
    axes[0].set(ylabel='Within-chunk Spearman', ylim=(0, 1))
    axes[1].plot(x, [100*r['top20_mean'] for r in rows], 'o-')
    axes[1].set(ylabel='Top-20% overlap (%)', ylim=(0, 100))
    for ax in axes: ax.set(xlabel='K samples', xticks=x); ax.grid(alpha=.15)
    save(fig, 'k_position_stability')

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), sharey=True)
    names = ['main_4', 'main_5', 'main_7']; labels = ['RT2 DSRL', 'Dojo DSRL', 'RT2 RLT']
    for ax, kind in zip(axes, ['normal_mean', 'amortized_mean']):
        left = np.zeros(3)
        for key, color in zip(['rollout', 'update', 'evaluate', 'other'], ['#2d829b', '#eaa33c', '#8f76bd', '#b7c3ca']):
            values = np.asarray([max(0, a['timing'][n][kind].get(key, 0))/60 for n in names])
            ax.barh(labels, values, left=left, label=key, color=color); left += values
        ax.set(title=kind, xlabel='Minutes / round')
    axes[1].legend(fontsize=8); save(fig, 'round_cost')


if __name__ == '__main__':
    main()
