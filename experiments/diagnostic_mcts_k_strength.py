"""
Q4: Does higher K make MCTS a stronger opponent, at fixed search budget?

Head-to-head:  MCTS-uniform(budget=B, K=K_high)  vs  MCTS-uniform(budget=B, K=K_low)
Both agents share the same search budget; only the number of action slots per
turn differs. Colors + K-assignment are alternated 50/50 across games.

Interpretation:
  - Score of K_high >> 0.5 : higher K makes MCTS a stronger opponent → K-sweep
    win-rate drop-offs at higher K may be partly opponent-strength driven.
  - Score ~ 0.5              : K doesn't change MCTS strength → any K-sweep
    drop-off is a training-difficulty story (PPO), not opponent inflation.
  - Score of K_high < 0.5   : higher K actually HURTS uniform MCTS at this
    search budget (bigger branching factor per turn, same 40 sims).

Usage:
  PYTHONPATH=. python experiments/diagnostic_mcts_k_strength.py \
      --maps simple_6,medium_8,large_10,triangle_6,ring_6,star_8,double_hub_8,hex_grid_10,dense_12 \
      --num-games 20 --k-high 10 --k-low 5 --budget 40 --workers 8
"""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


DEFAULT_MAPS = [
    'simple_6', 'medium_8', 'large_10',
    'triangle_6', 'ring_6', 'star_8', 'double_hub_8', 'hex_grid_10', 'dense_12',
]


def play_one_game(map_name, budget, k_high, k_low, max_turns, k_high_plays_as, seed):
    """Play one MCTS(K_high) vs MCTS(K_low) game at fixed search budget."""
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).parent.parent))

    from parallel_risk.env.parallel_risk_env import ParallelRiskEnv
    from parallel_risk.agents.mcts_agent import MCTSAgent

    # env must accommodate the larger K
    env = ParallelRiskEnv(
        map_name=map_name,
        max_turns=max_turns,
        max_actions_per_turn=max(10, k_high, k_low),
        reward_shaping_config=None,
    )

    agent_high_K = MCTSAgent.from_env(env, simulation_budget=budget, action_budget=k_high)
    agent_low_K = MCTSAgent.from_env(env, simulation_budget=budget, action_budget=k_low)

    if k_high_plays_as == 'agent_0':
        agents = {'agent_0': agent_high_K, 'agent_1': agent_low_K}
    else:
        agents = {'agent_0': agent_low_K, 'agent_1': agent_high_K}

    obs, _ = env.reset(seed=seed)
    done = False
    turns = 0
    while not done:
        actions = {}
        for aid in ['agent_0', 'agent_1']:
            if aid in obs:
                actions[aid] = agents[aid].get_action(env.game_state, aid)
        obs, rewards, terms, truncs, _ = env.step(actions)
        done = terms.get('__all__', False) or truncs.get('__all__', False)
        turns += 1

    other = 'agent_1' if k_high_plays_as == 'agent_0' else 'agent_0'
    r_high = rewards.get(k_high_plays_as, 0.0)
    r_low = rewards.get(other, 0.0)
    if r_high > r_low:
        result = +1
    elif r_low > r_high:
        result = -1
    else:
        result = 0
    return {'map_name': map_name, 'result': result, 'turns': turns,
            'k_high_plays_as': k_high_plays_as, 'seed': seed}


def _worker(kwargs):
    return play_one_game(**kwargs)


def run_map(map_name, num_games, budget, k_high, k_low, max_turns, workers, base_seed):
    jobs = []
    for i in range(num_games):
        k_high_plays_as = 'agent_0' if i % 2 == 0 else 'agent_1'
        jobs.append({
            'map_name': map_name,
            'budget': budget,
            'k_high': k_high, 'k_low': k_low,
            'max_turns': max_turns,
            'k_high_plays_as': k_high_plays_as,
            'seed': base_seed + i,
        })
    if workers <= 1:
        return [_worker(j) for j in jobs]
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_worker, j) for j in jobs]
        for fut in as_completed(futures):
            results.append(fut.result())
    return results


def summarize(results):
    n = len(results)
    wins = sum(1 for r in results if r['result'] == +1)
    losses = sum(1 for r in results if r['result'] == -1)
    draws = sum(1 for r in results if r['result'] == 0)
    score = (wins + 0.5 * draws) / n if n > 0 else 0.0
    avg_turns = float(np.mean([r['turns'] for r in results])) if results else 0.0
    if n > 0:
        z = 1.96
        denom = 1 + z * z / n
        centre = (score + z * z / (2 * n)) / denom
        halfw = z * np.sqrt(score * (1 - score) / n + z * z / (4 * n * n)) / denom
        ci_low, ci_high = max(0.0, centre - halfw), min(1.0, centre + halfw)
    else:
        ci_low = ci_high = 0.0
    return {
        'n': n, 'wins_high_K': wins, 'losses_high_K': losses, 'draws': draws,
        'win_rate_high_K': wins / n if n else 0.0,
        'score_high_K': score,
        'ci_low': ci_low, 'ci_high': ci_high,
        'avg_turns': avg_turns,
    }


def make_plot(per_map_summary, budget, k_high, k_low, out_path):
    """2-panel dashboard: score (with 95% CI) + outcome breakdown, sorted by map size."""
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).parent.parent))
    from parallel_risk.env.map_config import MapRegistry

    maps = list(per_map_summary.keys())
    maps.sort(key=lambda m: MapRegistry.get(m).n_territories)

    scores = [per_map_summary[m]['score_high_K'] for m in maps]
    ci_lo = [per_map_summary[m]['ci_low'] for m in maps]
    ci_hi = [per_map_summary[m]['ci_high'] for m in maps]
    errs_lo = [s - lo for s, lo in zip(scores, ci_lo)]
    errs_hi = [hi - s for s, hi in zip(scores, ci_hi)]
    draws = [per_map_summary[m]['draws'] / per_map_summary[m]['n'] for m in maps]
    win_frac = [per_map_summary[m]['wins_high_K'] / per_map_summary[m]['n'] for m in maps]
    loss_frac = [per_map_summary[m]['losses_high_K'] / per_map_summary[m]['n'] for m in maps]

    labels = [f"{m}\n(n={MapRegistry.get(m).n_territories})" for m in maps]
    x = np.arange(len(maps))

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.3), gridspec_kw={'wspace': 0.28})

    # Panel A — score with CI
    ax = axes[0]
    ax.bar(x, scores, yerr=[errs_lo, errs_hi], color='#6a4c93', alpha=0.85,
           capsize=4, edgecolor='#3d2c5c')
    ax.axhline(0.5, color='#7a7a7a', linestyle='--', linewidth=1.0,
               label='50% (K makes no difference)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=9)
    ax.set_ylabel(f'Score of K={k_high} side vs K={k_low} side\n(wins+0.5·draws)/n')
    ax.set_ylim(0, 1)
    ax.set_title(f'Panel A — Head-to-head score  (both MCTS at budget={budget})',
                 fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, axis='y', alpha=0.25)

    # Panel B — outcome breakdown
    ax = axes[1]
    ax.bar(x, win_frac, color='#6a4c93', alpha=0.85, edgecolor='#3d2c5c',
           label=f'Win (K={k_high} side)')
    ax.bar(x, draws, bottom=win_frac, color='#c4c1b8',
           edgecolor='#8a8880', label='Draw')
    ax.bar(x, loss_frac, bottom=[w + d for w, d in zip(win_frac, draws)],
           color='#c8695b', alpha=0.85, edgecolor='#8b4437',
           label=f'Loss (K={k_low} side wins)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=9)
    ax.set_ylabel('Outcome fraction')
    ax.set_ylim(0, 1)
    ax.set_title('Panel B — Outcome breakdown', fontsize=11)
    ax.legend(loc='lower right', fontsize=9)

    aggregate = float(np.mean(scores))
    fig.suptitle(
        f'Q4: Does K change MCTS opponent strength?  '
        f'MCTS(K={k_high}) vs MCTS(K={k_low}) at fixed budget={budget}\n'
        f'Aggregate score for K={k_high}: {aggregate:.2f}  |  colors alternated 50/50',
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Q4: MCTS K-strength calibration')
    parser.add_argument('--maps', type=str, default=','.join(DEFAULT_MAPS))
    parser.add_argument('--num-games', type=int, default=20)
    parser.add_argument('--budget', type=int, default=40, help='Shared MCTS search budget')
    parser.add_argument('--k-high', type=int, default=10)
    parser.add_argument('--k-low', type=int, default=5)
    parser.add_argument('--max-turns', type=int, default=40)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--base-seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str,
                        default='experiments/diagnostic_mcts_k_strength')
    args = parser.parse_args()

    maps = [m.strip() for m in args.maps.split(',') if m.strip()]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'[Q4] MCTS(K={args.k_high}) vs MCTS(K={args.k_low}) | '
          f'budget={args.budget} | {args.num_games} games/map | {len(maps)} maps')
    print(f'     Workers={args.workers}  max_turns={args.max_turns}')

    all_results = {}
    per_map_summary = {}
    t0 = time.time()

    for i, map_name in enumerate(maps, 1):
        t_m = time.time()
        print(f'\n[{i}/{len(maps)}] {map_name} ...')
        results = run_map(
            map_name=map_name,
            num_games=args.num_games,
            budget=args.budget,
            k_high=args.k_high, k_low=args.k_low,
            max_turns=args.max_turns,
            workers=args.workers,
            base_seed=args.base_seed + i * 1000,
        )
        s = summarize(results)
        per_map_summary[map_name] = s
        all_results[map_name] = results
        dt = time.time() - t_m
        print(f'     score(K={args.k_high})={s["score_high_K"]:.2f}  '
              f'W={s["wins_high_K"]} L={s["losses_high_K"]} D={s["draws"]}  '
              f'avg_turns={s["avg_turns"]:.1f}  '
              f'CI95=[{s["ci_low"]:.2f}, {s["ci_high"]:.2f}]  '
              f'({dt:.0f}s)')

        with open(out_dir / 'results_partial.json', 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'config': vars(args),
                'per_map_summary': per_map_summary,
                'raw_results': all_results,
            }, f, indent=2, default=str)

    total_dt = time.time() - t0
    aggregate = float(np.mean([s['score_high_K'] for s in per_map_summary.values()]))
    print(f'\n=== Aggregate ===')
    print(f'  Mean score(K={args.k_high}) across {len(maps)} maps: {aggregate:.3f}')
    print(f'  Total wall-clock: {total_dt / 60:.1f} min')

    with open(out_dir / 'results.json', 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'config': vars(args),
            'aggregate_score_high_K': aggregate,
            'wall_clock_min': total_dt / 60,
            'per_map_summary': per_map_summary,
            'raw_results': all_results,
        }, f, indent=2, default=str)

    plot_path = out_dir / f'q4_k{args.k_high}_vs_k{args.k_low}_b{args.budget}.png'
    make_plot(per_map_summary, args.budget, args.k_high, args.k_low, plot_path)
    print(f'\nSaved: {out_dir}/results.json')
    print(f'Saved: {plot_path}')


if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    main()
