"""Regenerate a 2-panel dashboard from the Q3 results.json (score + draw-rate)."""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from parallel_risk.env.map_config import MapRegistry


def main():
    results_path = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        Path('experiments/diagnostic_mcts_budget/full_k5/results.json')
    with open(results_path) as f:
        data = json.load(f)

    per_map = data['per_map_summary']
    cfg = data['config']
    budget_high = cfg['budget_high']
    budget_low = cfg['budget_low']
    k = cfg['action_budget']

    maps = list(per_map.keys())
    maps.sort(key=lambda m: MapRegistry.get(m).n_territories)

    scores = [per_map[m]['score_high'] for m in maps]
    ci_lo = [per_map[m]['ci_low'] for m in maps]
    ci_hi = [per_map[m]['ci_high'] for m in maps]
    errs_lo = [s - lo for s, lo in zip(scores, ci_lo)]
    errs_hi = [hi - s for s, hi in zip(scores, ci_hi)]
    draws = [per_map[m]['draws'] / per_map[m]['n'] for m in maps]
    win_rate = [per_map[m]['wins_high'] / per_map[m]['n'] for m in maps]
    loss_rate = [per_map[m]['losses_high'] / per_map[m]['n'] for m in maps]

    labels = [f"{m}\n(n={MapRegistry.get(m).n_territories})" for m in maps]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.3), gridspec_kw={'wspace': 0.28})

    # Panel A — score with 95% CI
    ax = axes[0]
    x = np.arange(len(maps))
    ax.bar(x, scores, yerr=[errs_lo, errs_hi], color='#3b7ea1', alpha=0.85,
           capsize=4, edgecolor='#1c3d55')
    ax.axhline(0.5, color='#7a7a7a', linestyle='--', linewidth=1.0,
               label='50% (no advantage)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=9)
    ax.set_ylabel(f'Score of MCTS-{budget_high} vs MCTS-{budget_low}\n(wins+0.5·draws)/n')
    ax.set_ylim(0, 1)
    ax.set_title(f'Panel A — Head-to-head score (K={k})', fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, axis='y', alpha=0.25)

    # Panel B — outcome breakdown (stacked)
    ax = axes[1]
    ax.bar(x, win_rate, color='#3b7ea1', alpha=0.85, edgecolor='#1c3d55',
           label=f'Win (MCTS-{budget_high})')
    ax.bar(x, draws, bottom=win_rate, color='#c4c1b8',
           edgecolor='#8a8880', label='Draw')
    ax.bar(x, loss_rate, bottom=[w + d for w, d in zip(win_rate, draws)],
           color='#c8695b', alpha=0.85, edgecolor='#8b4437',
           label=f'Loss (MCTS-{budget_low} wins)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=9)
    ax.set_ylabel('Outcome fraction')
    ax.set_ylim(0, 1)
    ax.set_title('Panel B — Outcome breakdown', fontsize=11)
    ax.legend(loc='lower right', fontsize=9)

    fig.suptitle(
        f'Q3: MCTS opponent-strength calibration — MCTS-uniform({budget_high}) vs MCTS-uniform({budget_low}) at K={k}\n'
        f'Aggregate score: {data["aggregate_score_high"]:.2f}   |   {cfg["num_games"]} games/map, colors alternated 50/50',
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = results_path.parent / f'q3_dashboard_b{budget_high}_vs_b{budget_low}_k{k}.png'
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
