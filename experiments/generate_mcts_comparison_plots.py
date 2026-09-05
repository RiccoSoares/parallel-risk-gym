"""Generate visualization plots for MCTS validation and MCTS vs GNN comparison.

Reads existing JSON results from:
  - experiments/mcts_validation_results/mcts_validation_results.json
  - experiments/mcts_vs_gnn_results_corrected/comparison_results.json

Writes plots to experiments/mcts_vs_gnn_results_corrected/.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).parent


def load_json(path):
    with open(path) as f:
        return json.load(f)


def plot_budget_sweep(mcts_data, out_dir):
    """Win rate and episode length vs MCTS simulation budget."""
    evals = mcts_data['evaluations']
    budgets = sorted(int(b) for b in evals)
    win_rates = [evals[str(b)]['win_rate'] * 100 for b in budgets]
    avg_lengths = [evals[str(b)]['avg_episode_length'] for b in budgets]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('MCTS Performance vs Simulation Budget\n(vs. Masked-Random Baseline)',
                 fontsize=14, fontweight='bold')

    # Win rate
    bars = ax1.bar([str(b) for b in budgets], win_rates,
                   color='#2563eb', alpha=0.85, edgecolor='white', linewidth=0.5)
    ax1.set_ylim(80, 101)
    ax1.set_xlabel('Simulation Budget (rollouts per move)', fontsize=11)
    ax1.set_ylabel('Win Rate (%)', fontsize=11)
    ax1.set_title('Win Rate vs Budget', fontsize=12)
    ax1.axhline(100, color='#64748b', linestyle='--', linewidth=0.8, alpha=0.6)
    for bar, wr in zip(bars, win_rates):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                 f'{wr:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Episode length (shorter = more decisive)
    ax2.plot([str(b) for b in budgets], avg_lengths,
             marker='o', color='#059669', linewidth=2.5, markersize=8)
    ax2.set_xlabel('Simulation Budget (rollouts per move)', fontsize=11)
    ax2.set_ylabel('Average Episode Length (turns)', fontsize=11)
    ax2.set_title('Episode Length vs Budget\n(shorter = more decisive play)', fontsize=12)
    for x, y in zip([str(b) for b in budgets], avg_lengths):
        ax2.annotate(f'{y:.1f}', (x, y), textcoords='offset points', xytext=(0, 8),
                     ha='center', fontsize=10)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    out_path = out_dir / 'mcts_budget_sweep.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')


def plot_agent_comparison(cmp_data, out_dir):
    """Bar chart comparing win rates: MCTS vs random, GNN vs random, MCTS vs GNN."""
    matchups = {
        'MCTS\nvs Random': cmp_data['mcts_vs_random']['win_rate'],
        'GNN\nvs Random': cmp_data['gnn_vs_random']['win_rate'],
        'MCTS\nvs GNN': cmp_data['mcts_vs_gnn']['win_rate'],
    }
    lengths = {
        'MCTS\nvs Random': cmp_data['mcts_vs_random']['avg_episode_length'],
        'GNN\nvs Random': cmp_data['gnn_vs_random']['avg_episode_length'],
        'MCTS\nvs GNN': cmp_data['mcts_vs_gnn']['avg_episode_length'],
    }
    n_eps = cmp_data['mcts_vs_random']['total_episodes']

    colors = ['#2563eb', '#059669', '#dc2626']
    labels = list(matchups.keys())
    win_rates = [v * 100 for v in matchups.values()]
    ep_lengths = list(lengths.values())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f'Agent Comparison (n={n_eps} episodes per matchup)',
                 fontsize=14, fontweight='bold')

    # Win rate comparison
    bars = ax1.bar(labels, win_rates, color=colors, alpha=0.85,
                   edgecolor='white', linewidth=0.5, width=0.5)
    ax1.set_ylim(0, 110)
    ax1.set_ylabel('Win Rate (%)', fontsize=11)
    ax1.set_title('Win Rates', fontsize=12)
    ax1.axhline(50, color='#94a3b8', linestyle='--', linewidth=1, alpha=0.8, label='50% (chance)')
    for bar, wr in zip(bars, win_rates):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                 f'{wr:.0f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Episode length (proxy for game decisiveness)
    bars2 = ax2.bar(labels, ep_lengths, color=colors, alpha=0.85,
                    edgecolor='white', linewidth=0.5, width=0.5)
    ax2.set_ylabel('Average Episode Length (turns)', fontsize=11)
    ax2.set_title('Episode Length\n(shorter = more decisive play)', fontsize=12)
    for bar, el in zip(bars2, ep_lengths):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f'{el:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    note = ('Note: GNN checkpoint predates best-known weights.\n'
            'Fair MCTS vs. best GNN comparison is pending (Phase 3 baseline).')
    fig.text(0.5, -0.03, note, ha='center', fontsize=9, color='#64748b', style='italic')

    plt.tight_layout()
    out_path = out_dir / 'agent_comparison.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')


def plot_combined_summary(mcts_data, cmp_data, out_dir):
    """Single summary figure: budget sweep + agent comparison side by side."""
    evals = mcts_data['evaluations']
    budgets = sorted(int(b) for b in evals)
    budget_wr = [evals[str(b)]['win_rate'] * 100 for b in budgets]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('MCTS & GNN Agent Evaluation Summary', fontsize=15, fontweight='bold', y=1.02)

    # 1) Budget sweep
    ax = axes[0]
    ax.plot([str(b) for b in budgets], budget_wr, marker='o',
            color='#2563eb', linewidth=2.5, markersize=9)
    ax.set_ylim(90, 101)
    ax.set_xlabel('MCTS Budget (rollouts/move)', fontsize=10)
    ax.set_ylabel('Win Rate vs Masked-Random (%)', fontsize=10)
    ax.set_title('MCTS Budget Sweep', fontsize=11, fontweight='bold')
    for x, y in zip([str(b) for b in budgets], budget_wr):
        ax.annotate(f'{y:.1f}%', (x, y), textcoords='offset points',
                    xytext=(0, 7), ha='center', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 2) Agent vs random
    ax = axes[1]
    agents = ['MCTS\n(budget=200)', 'GNN\n(PPO)']
    vrs = [cmp_data['mcts_vs_random']['win_rate'] * 100,
           cmp_data['gnn_vs_random']['win_rate'] * 100]
    bars = ax.bar(agents, vrs, color=['#2563eb', '#059669'], alpha=0.85,
                  edgecolor='white', width=0.45)
    ax.set_ylim(0, 110)
    ax.set_ylabel('Win Rate vs Random (%)', fontsize=10)
    ax.set_title('vs Masked-Random Baseline', fontsize=11, fontweight='bold')
    for bar, v in zip(bars, vrs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f'{v:.0f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 3) MCTS vs GNN head-to-head
    ax = axes[2]
    h2h = [cmp_data['mcts_vs_gnn']['win_rate'] * 100,
           (1 - cmp_data['mcts_vs_gnn']['win_rate']) * 100]
    explode = (0.05, 0)
    wedge_colors = ['#2563eb', '#059669']
    wedges, texts, autotexts = ax.pie(
        h2h, labels=['MCTS wins', 'GNN wins'],
        colors=wedge_colors, autopct='%1.0f%%',
        startangle=90, explode=explode,
        textprops={'fontsize': 10},
    )
    for at in autotexts:
        at.set_fontweight('bold')
        at.set_fontsize(11)
    ax.set_title('MCTS vs GNN Head-to-Head\n(budget=200)', fontsize=11, fontweight='bold')

    note = '* GNN evaluated against earlier (non-best) checkpoint'
    fig.text(0.5, -0.06, note, ha='center', fontsize=8, color='#64748b', style='italic')

    plt.tight_layout()
    out_path = out_dir / 'summary_dashboard.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')


def main():
    mcts_dir = ROOT / 'mcts_validation_results'
    cmp_dir = ROOT / 'mcts_vs_gnn_results_corrected'

    mcts_data = load_json(mcts_dir / 'mcts_validation_results.json')
    cmp_data = load_json(cmp_dir / 'comparison_results.json')

    plot_budget_sweep(mcts_data, mcts_dir)
    plot_agent_comparison(cmp_data, cmp_dir)
    plot_combined_summary(mcts_data, cmp_data, cmp_dir)

    print('\nAll plots generated.')


if __name__ == '__main__':
    main()
