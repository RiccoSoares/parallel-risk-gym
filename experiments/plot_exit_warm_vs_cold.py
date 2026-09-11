"""Combined figure: ExIt at MCTS budget 200, K=10 — warm start vs cold start.

Panel A: mean eval score over the 14 informative maps (the 7 maps that draw
every game sit at exactly 0.50 and are excluded; they carry no signal).
Panel B: per-map score at the last eval, sorted by map size.
Panel C: self-play outcome mix, the quantity that collapsed the cold-start
signal at budget 40 / K=5.

Usage:
    PYTHONIOENCODING=utf-8 PYTHONUTF8=1 PYTHONPATH=. python experiments/plot_exit_warm_vs_cold.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

WARM = Path('experiments/exit_b200_k10/results.json')
COLD = Path('experiments/exit_b200_k10_cold/results.json')
OUT = Path('experiments/exit_b200_k10/warm_vs_cold.png')

_SURFACE = '#fcfcfb'
_INK = '#0b0b0b'
_INK2 = '#52514e'
_MUTED = '#898781'
_GRID = '#e1e0d9'


def chrome(ax):
    ax.set_facecolor(_SURFACE)
    ax.yaxis.grid(True, color=_GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(_GRID)
    ax.tick_params(colors=_MUTED, labelsize=9)


def informative(d):
    """Maps that are not 100% draws at every eval point."""
    eh, maps = d['eval_history'], d['map_names']
    return [m for m in maps
            if not all(e['per_map'][m]['draws'] == e['per_map'][m]['total'] for e in eh)]


def curve(d, keep):
    it = [e['iteration'] for e in d['eval_history']]
    y = [np.mean([e['per_map'][m]['score'] for m in keep]) * 100 for e in d['eval_history']]
    return it, y


def main():
    warm = json.loads(WARM.read_text(encoding='utf-8'))
    cold = json.loads(COLD.read_text(encoding='utf-8'))
    keep = sorted(set(informative(warm)) & set(informative(cold)))
    n_games = warm['eval_history'][-1]['per_map'][keep[0]]['total']
    # Standard error of a mean over `keep` maps of a proportion from n_games each.
    se = 100 * math_se(n_games, len(keep))

    cmap = plt.get_cmap('tab10')
    c_warm, c_cold = cmap(0), cmap(3)
    fig, axes = plt.subplots(3, 1, figsize=(11, 12), facecolor=_SURFACE)

    ax = axes[0]
    for d, lab, c in ((warm, 'warm start (PPO K=10 ckpt)', c_warm), (cold, 'cold start (random init)', c_cold)):
        it, y = curve(d, keep)
        ax.plot(it, y, color=c, lw=2.4, marker='o', markersize=7, label=f'{lab}: {y[0]:.1f} -> {y[-1]:.1f}')
        ax.fill_between(it, np.array(y) - se, np.array(y) + se, color=c, alpha=0.15)
    ax.axhline(50, color=_MUTED, lw=1, ls='--', alpha=0.7)
    ax.text(11, 51, 'parity with MCTS-uniform(200)', color=_MUTED, fontsize=8, va='bottom')
    ax.set_ylabel(f'mean score over {len(keep)} informative maps (%)', color=_INK2, fontsize=10)
    ax.set_xlabel('iteration', color=_INK2, fontsize=10)
    ax.set_title('A. ExIt at MCTS budget 200, K=10 — shaded band is +/- 1 SE',
                 color=_INK, fontsize=12, fontweight='bold', pad=6)
    ax.legend(frameon=True, facecolor=_SURFACE, edgecolor=_GRID, labelcolor=_INK2, fontsize=9)
    chrome(ax)

    ax = axes[1]
    from parallel_risk.env.map_config import MapRegistry
    order = sorted(keep, key=lambda m: MapRegistry.get(m).n_territories)
    x = np.arange(len(order))
    w_last = [warm['eval_history'][-1]['per_map'][m]['score'] * 100 for m in order]
    c_last = [cold['eval_history'][-1]['per_map'][m]['score'] * 100 for m in order]
    ax.bar(x - 0.2, w_last, 0.4, color=c_warm, label='warm', edgecolor='none')
    ax.bar(x + 0.2, c_last, 0.4, color=c_cold, label='cold', edgecolor='none')
    ax.axhline(50, color=_MUTED, lw=1, ls='--', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{m}\n({MapRegistry.get(m).n_territories})' for m in order],
                       rotation=25, ha='right', fontsize=8)
    ax.set_ylabel('score at iteration 50 (%)', color=_INK2, fontsize=10)
    ax.set_title('B. Per-map score at the last eval (7 all-draw maps excluded)',
                 color=_INK, fontsize=12, fontweight='bold', pad=6)
    ax.legend(frameon=True, facecolor=_SURFACE, edgecolor=_GRID, labelcolor=_INK2, fontsize=9)
    chrome(ax)

    ax = axes[2]
    for d, lab, c in ((warm, 'warm', c_warm), (cold, 'cold', c_cold)):
        m = d['metrics']
        ax.plot(m['iteration'], [x * 100 for x in m['selfplay_draw_frac']], color=c, lw=1.6,
                label=f'{lab}: draws')
        ax.plot(m['iteration'], [x * 100 for x in m['selfplay_win_frac']], color=c, lw=1.2,
                ls=':', alpha=0.8, label=f'{lab}: agent_0 wins')
    ax.set_ylim(0, 100)
    ax.set_ylabel('self-play games (%)', color=_INK2, fontsize=10)
    ax.set_xlabel('iteration', color=_INK2, fontsize=10)
    ax.set_title('C. Self-play outcome mix — at budget 40 / K=5 cold start sat near 100% draws',
                 color=_INK, fontsize=12, fontweight='bold', pad=6)
    ax.legend(frameon=True, facecolor=_SURFACE, edgecolor=_GRID, labelcolor=_INK2,
              fontsize=8, ncol=2)
    chrome(ax)

    fig.suptitle('Expert Iteration at MCTS budget 200, K=10, 21 maps, 50 iterations',
                 color=_INK, fontsize=13, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT, dpi=150, facecolor=_SURFACE)
    print(f'saved {OUT}')
    print(f'informative maps ({len(keep)}): {", ".join(keep)}')


def math_se(n_games: int, n_maps: int) -> float:
    """SE of the mean over n_maps of a proportion estimated from n_games each (p=0.5 worst case)."""
    return (0.25 / n_games) ** 0.5 / (n_maps ** 0.5)


if __name__ == '__main__':
    main()
