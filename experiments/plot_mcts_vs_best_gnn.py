"""Generate plots for MCTS vs best-GNN comparison (fair comparison run).

Reads: experiments/mcts_vs_best_gnn/comparison_results.json
Writes: experiments/mcts_vs_best_gnn/{agent_comparison.png, summary_dashboard.png}
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
CMP_DIR = ROOT / 'mcts_vs_best_gnn'


def load_json(path):
    with open(path) as f:
        return json.load(f)


def plot_comparison(data, out_dir):
    budget = 100
    n_eps = data['mcts_vs_random']['total_episodes']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f'MCTS vs Best GNN — Fair Comparison (n={n_eps} episodes)\n'
        f'GNN: 96.7% win rate vs random  |  MCTS budget={budget}: 100% win rate vs random',
        fontsize=13, fontweight='bold', y=1.02
    )

    colors = {'mcts': '#2563eb', 'gnn': '#059669', 'draw': '#94a3b8'}

    # Panel 1: Win rates vs random
    ax = axes[0]
    agents = [f'MCTS\n(budget={budget})', 'Best GNN\n(PPO, iter 40)']
    vrs = [
        data['mcts_vs_random']['win_rate'] * 100,
        data['gnn_vs_random']['win_rate'] * 100,
    ]
    bars = ax.bar(agents, vrs, color=[colors['mcts'], colors['gnn']],
                  alpha=0.85, edgecolor='white', width=0.5)
    ax.set_ylim(0, 110)
    ax.set_ylabel('Win Rate (%)', fontsize=10)
    ax.set_title('vs Masked-Random Baseline', fontsize=11, fontweight='bold')
    for bar, v in zip(bars, vrs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f'{v:.0f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel 2: Head-to-head (pie)
    ax = axes[1]
    mcts_w = data['mcts_vs_gnn']['win_rate'] * 100
    gnn_w  = data['mcts_vs_gnn']['loss_rate'] * 100
    draw_w = data['mcts_vs_gnn']['draw_rate'] * 100
    sizes  = [v for v in [mcts_w, gnn_w, draw_w] if v > 0]
    labels = [l for v, l in [(mcts_w, f'MCTS wins\n{mcts_w:.0f}%'),
                              (gnn_w, f'GNN wins\n{gnn_w:.0f}%'),
                              (draw_w, f'Draw\n{draw_w:.0f}%')] if v > 0]
    pie_colors = [colors['mcts'], colors['gnn'], colors['draw']][:len(sizes)]
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=pie_colors,
        autopct='', startangle=90, explode=[0.05] + [0] * (len(sizes) - 1),
        textprops={'fontsize': 10}
    )
    for at in autotexts:
        at.set_fontweight('bold')
    ax.set_title(f'Head-to-Head: MCTS(b={budget}) vs Best GNN', fontsize=11, fontweight='bold')

    # Panel 3: Episode length comparison (competitiveness proxy)
    ax = axes[2]
    matchups = ['MCTS vs\nRandom', 'GNN vs\nRandom', 'MCTS vs\nGNN']
    lengths = [
        data['mcts_vs_random']['avg_episode_length'],
        data['gnn_vs_random']['avg_episode_length'],
        data['mcts_vs_gnn']['avg_episode_length'],
    ]
    errs = [
        data['mcts_vs_random']['std_episode_length'],
        data['gnn_vs_random']['std_episode_length'],
        data['mcts_vs_gnn']['std_episode_length'],
    ]
    bar_colors = [colors['mcts'], colors['gnn'], '#7c3aed']
    bars = ax.bar(matchups, lengths, color=bar_colors, alpha=0.85,
                  edgecolor='white', width=0.5,
                  yerr=errs, capsize=4, error_kw={'elinewidth': 1.5})
    ax.set_ylabel('Average Episode Length (turns)', fontsize=10)
    ax.set_title('Game Length (longer = more competitive)', fontsize=11, fontweight='bold')
    for bar, l in zip(bars, lengths):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(errs) * 0.05 + 1,
                f'{l:.1f}', ha='center', va='bottom', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    note = (
        f'Prior comparison (old GNN, budget=200): MCTS won 90%.  '
        f'Best GNN (budget={budget}): MCTS wins 63% — gap narrowed significantly.'
    )
    fig.text(0.5, -0.04, note, ha='center', fontsize=9, color='#475569', style='italic')

    plt.tight_layout()
    out_path = out_dir / 'fair_comparison_dashboard.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    data = load_json(CMP_DIR / 'comparison_results.json')
    plot_comparison(data, CMP_DIR)
    print('Done.')
