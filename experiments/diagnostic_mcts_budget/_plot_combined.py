"""Combined Q3(K=5) + Q3(K=10) + Q4 comparison plot."""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from parallel_risk.env.map_config import MapRegistry


def load(path, score_key):
    with open(path) as f:
        d = json.load(f)
    per_map = d['per_map_summary']
    return {m: s[score_key] for m, s in per_map.items()}, per_map


def main():
    root = Path('experiments')
    q3_k5, q3_k5_raw = load(root / 'diagnostic_mcts_budget/full_k5/results.json',
                            'score_high')
    q3_k10, q3_k10_raw = load(root / 'diagnostic_mcts_budget/full_k10/results.json',
                              'score_high')
    q4, q4_raw = load(root / 'diagnostic_mcts_k_strength/full_b40/results.json',
                      'score_high_K')

    maps = sorted(q3_k5.keys(), key=lambda m: MapRegistry.get(m).n_territories)
    labels = [f"{m}\n(n={MapRegistry.get(m).n_territories})" for m in maps]

    x = np.arange(len(maps))
    width = 0.27

    fig, ax = plt.subplots(figsize=(14, 5.5))

    q3k5 = [q3_k5[m] for m in maps]
    q3k10 = [q3_k10[m] for m in maps]
    q4v = [q4[m] for m in maps]

    b1 = ax.bar(x - width, q3k5, width, color='#3b7ea1', alpha=0.90,
                edgecolor='#1c3d55',
                label='Q3(K=5):  MCTS-200 vs MCTS-40  (both K=5)')
    b2 = ax.bar(x, q3k10, width, color='#7fb069', alpha=0.90,
                edgecolor='#3d5c33',
                label='Q3(K=10): MCTS-200 vs MCTS-40  (both K=10)')
    b3 = ax.bar(x + width, q4v, width, color='#6a4c93', alpha=0.90,
                edgecolor='#3d2c5c',
                label='Q4:       MCTS-K10 vs MCTS-K5   (both budget=40)')

    ax.axhline(0.5, color='#7a7a7a', linestyle='--', linewidth=1.0,
               label='50% (no advantage)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=9)
    ax.set_ylabel('Head-to-head score (wins + 0.5·draws) / n')
    ax.set_ylim(0, 1.02)
    ax.set_title(
        'MCTS opponent-strength calibration — comparing three questions on the same 9-map roster\n'
        f'Aggregates:  Q3(K=5) = {np.mean(q3k5):.2f}    '
        f'Q3(K=10) = {np.mean(q3k10):.2f}    '
        f'Q4 = {np.mean(q4v):.2f}',
        fontsize=11,
    )
    ax.legend(loc='lower right', fontsize=9, framealpha=0.95)
    ax.grid(True, axis='y', alpha=0.25)

    fig.tight_layout()
    out_path = Path('experiments/diagnostic_mcts_budget/q3_q4_combined.png')
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
