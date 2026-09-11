"""ExIt at MCTS budget 200, K=10: random vs GNN-proposed progressive widening.

Both arms are the same experiment — warm start from the K=10 PPO checkpoint,
21 maps, 50 iterations of 96 self-play games, eval every 10 iterations against
MCTS-uniform(200) with 12 games per map. They differ only in where progressive
widening draws its candidate actions (see docs/EXPERIMENT_LOG.md §10).

Usage:
    PYTHONIOENCODING=utf-8 PYTHONUTF8=1 PYTHONPATH=. python experiments/plot_exit_widening.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RANDOM = Path('experiments/exit_b200_k10/results.json')          # pw_sampler=masked_random
GNN = Path('experiments/exit_b200_k10_pwgnn/results.json')       # pw_sampler=gnn
AB = Path('experiments/pw_sampler_ab/results.json')              # untrained checkpoint, both samplers
OUT = Path('experiments/exit_b200_k10_pwgnn/random_vs_gnn_widening.png')

_SURFACE, _INK, _INK2, _MUTED, _GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9'


def chrome(ax):
    ax.set_facecolor(_SURFACE)
    ax.yaxis.grid(True, color=_GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(_GRID)
    ax.tick_params(colors=_MUTED, labelsize=9)


def curve(d, maps):
    it = [e['iteration'] for e in d['eval_history']]
    y = [100 * np.mean([e['per_map'][m]['score'] for m in maps]) for e in d['eval_history']]
    return it, y


def main():
    rnd = json.loads(RANDOM.read_text(encoding='utf-8'))
    gnn = json.loads(GNN.read_text(encoding='utf-8'))
    ab = json.loads(AB.read_text(encoding='utf-8'))
    maps = gnn['map_names']
    n_games = gnn['eval_history'][-1]['per_map'][maps[0]]['total']
    se = 100 * (0.25 / n_games) ** 0.5 / len(maps) ** 0.5

    cmap = plt.get_cmap('tab10')
    c_rnd, c_gnn = cmap(3), cmap(0)
    fig, axes = plt.subplots(3, 1, figsize=(11, 12.5), facecolor=_SURFACE)

    ax = axes[0]
    for d, lab, c in ((rnd, 'random widening (masked_random)', c_rnd),
                      (gnn, 'GNN-proposed widening', c_gnn)):
        it, y = curve(d, maps)
        ax.plot(it, y, color=c, lw=2.4, marker='o', markersize=7,
                label=f'{lab}: {y[0]:.1f} -> {y[-1]:.1f} (best {max(y):.1f})')
        ax.fill_between(it, np.array(y) - se, np.array(y) + se, color=c, alpha=0.15)
    ab8 = list(ab['gnn']['per_map'])
    ax.axhline(100 * ab['gnn']['mean_score'], color=c_gnn, lw=1.2, ls=':', alpha=0.9)
    ax.text(10.5, 100 * ab['gnn']['mean_score'] + 1.2,
            f'untrained checkpoint, GNN widening ({100 * ab["gnn"]["mean_score"]:.1f} on {len(ab8)} maps)',
            color=c_gnn, fontsize=8)
    ax.axhline(50, color=_MUTED, lw=1, ls='--', alpha=0.7)
    ax.text(10.5, 51.2, 'parity with MCTS-uniform(200)', color=_MUTED, fontsize=8)
    ax.set_ylim(30, 100)
    ax.set_ylabel('mean score over all 21 maps (%)', color=_INK2, fontsize=10)
    ax.set_xlabel('iteration', color=_INK2, fontsize=10)
    ax.set_title('A. Where progressive widening draws candidates decides the outcome'
                 '  (band = +/- 1 SE)', color=_INK, fontsize=12, fontweight='bold', pad=6)
    ax.legend(frameon=True, facecolor=_SURFACE, edgecolor=_GRID, labelcolor=_INK2,
              fontsize=9, loc='center right')
    chrome(ax)

    ax = axes[1]
    for d, lab, c in ((rnd, 'random', c_rnd), (gnn, 'GNN', c_gnn)):
        it = [e['iteration'] for e in d['eval_history']]
        ax.plot(it, [100 * e['mean_win_rate'] for e in d['eval_history']], color=c, lw=2,
                marker='o', markersize=6, label=f'{lab}: outright wins')
        ax.plot(it, [100 * e['mean_draw_rate'] for e in d['eval_history']], color=c, lw=1.4,
                ls=':', marker='^', markersize=5, alpha=0.85, label=f'{lab}: draws')
    ax.set_ylim(0, 80)
    ax.set_ylabel('% of eval games', color=_INK2, fontsize=10)
    ax.set_xlabel('iteration', color=_INK2, fontsize=10)
    ax.set_title('B. Wins and draws — the score metric hides that random widening never wins',
                 color=_INK, fontsize=12, fontweight='bold', pad=6)
    ax.legend(frameon=True, facecolor=_SURFACE, edgecolor=_GRID, labelcolor=_INK2,
              fontsize=8, ncol=2)
    chrome(ax)

    ax = axes[2]
    from parallel_risk.env.map_config import MapRegistry
    order = sorted(maps, key=lambda m: MapRegistry.get(m).n_territories)
    best = max(gnn['eval_history'], key=lambda e: np.mean([e['per_map'][m]['score'] for m in maps]))
    x = np.arange(len(order))
    ax.bar(x - 0.2, [100 * rnd['eval_history'][-1]['per_map'][m]['score'] for m in order],
           0.4, color=c_rnd, label='random widening, iter 50')
    ax.bar(x + 0.2, [100 * best['per_map'][m]['score'] for m in order],
           0.4, color=c_gnn, label=f'GNN widening, iter {best["iteration"]}')
    ax.axhline(50, color=_MUTED, lw=1, ls='--', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{m}\n({MapRegistry.get(m).n_territories})' for m in order],
                       rotation=30, ha='right', fontsize=7.5)
    ax.set_ylabel('score (%)', color=_INK2, fontsize=10)
    ax.set_title('C. Per map, best eval of each arm — 0.50 with no wins means every game drew',
                 color=_INK, fontsize=12, fontweight='bold', pad=6)
    ax.legend(frameon=True, facecolor=_SURFACE, edgecolor=_GRID, labelcolor=_INK2, fontsize=9)
    chrome(ax)

    fig.suptitle('Expert Iteration, MCTS budget 200, K=10, warm start, 21 maps',
                 color=_INK, fontsize=13, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT, dpi=150, facecolor=_SURFACE)
    print(f'saved {OUT}')


if __name__ == '__main__':
    main()
