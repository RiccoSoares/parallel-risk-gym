"""
Q3: MCTS opponent-strength calibration — MCTS-uniform(budget=200) vs MCTS-uniform(budget=40).

For each map, plays num_games head-to-head, alternating colors, and reports
win-rate of high-budget MCTS over low-budget MCTS. This tells us how far
our default MCTS-uniform-40 opponent is from a stronger reference, and
therefore how much of our earlier PPO-vs-MCTS-40 results might be inflated.

No trained model required. Pure MCTS vs pure MCTS with masked-random rollouts.

Usage:
    PYTHONPATH=. python experiments/diagnostic_mcts_budget.py \
        --maps simple_6,medium_8,large_10,triangle_6,ring_6,star_8,double_hub_8,hex_grid_10,dense_12 \
        --num-games 20 --k 5 --workers 8
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


def play_one_game(map_name, budget_high, budget_low, action_budget, max_turns,
                  high_plays_as, seed):
    """Play one MCTS(high) vs MCTS(low) game. Returns +1 if high won, -1 if low won, 0 draw."""
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).parent.parent))

    from parallel_risk.env.parallel_risk_env import ParallelRiskEnv
    from parallel_risk.agents.mcts_agent import MCTSAgent

    env = ParallelRiskEnv(
        map_name=map_name,
        max_turns=max_turns,
        max_actions_per_turn=max(10, action_budget),
        reward_shaping_config=None,
    )

    agent_high = MCTSAgent.from_env(env, simulation_budget=budget_high,
                                    action_budget=action_budget)
    agent_low = MCTSAgent.from_env(env, simulation_budget=budget_low,
                                   action_budget=action_budget)

    # Alternate colors: high_plays_as tells which slot the high-budget agent takes.
    if high_plays_as == 'agent_0':
        agents = {'agent_0': agent_high, 'agent_1': agent_low}
    else:
        agents = {'agent_0': agent_low, 'agent_1': agent_high}

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

    r_high = rewards.get(high_plays_as, 0.0)
    r_low = rewards.get('agent_1' if high_plays_as == 'agent_0' else 'agent_0', 0.0)
    if r_high > r_low:
        result = +1
    elif r_low > r_high:
        result = -1
    else:
        result = 0
    return {'map_name': map_name, 'result': result, 'turns': turns,
            'high_plays_as': high_plays_as, 'seed': seed}


def _worker(args):
    return play_one_game(**args)


def run_map(map_name, num_games, budget_high, budget_low, action_budget,
            max_turns, workers, base_seed):
    """Run num_games total for a single map, alternating colors 50/50."""
    jobs = []
    for i in range(num_games):
        high_plays_as = 'agent_0' if i % 2 == 0 else 'agent_1'
        jobs.append({
            'map_name': map_name,
            'budget_high': budget_high,
            'budget_low': budget_low,
            'action_budget': action_budget,
            'max_turns': max_turns,
            'high_plays_as': high_plays_as,
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
    """Aggregate: wins for high-budget = result==+1; losses = -1; draws = 0."""
    n = len(results)
    wins = sum(1 for r in results if r['result'] == +1)
    losses = sum(1 for r in results if r['result'] == -1)
    draws = sum(1 for r in results if r['result'] == 0)
    avg_turns = float(np.mean([r['turns'] for r in results])) if results else 0.0
    # Wilson score interval (95%) for win-rate incl. draws=0.5
    score = (wins + 0.5 * draws) / n if n > 0 else 0.0
    if n > 0:
        z = 1.96
        denom = 1 + z * z / n
        centre = (score + z * z / (2 * n)) / denom
        halfw = z * np.sqrt(score * (1 - score) / n + z * z / (4 * n * n)) / denom
        ci_low, ci_high = max(0.0, centre - halfw), min(1.0, centre + halfw)
    else:
        ci_low = ci_high = 0.0
    return {
        'n': n, 'wins_high': wins, 'losses_high': losses, 'draws': draws,
        'win_rate_high': wins / n if n else 0.0,
        'score_high': score,       # wins + 0.5*draws (win-rate with draws split)
        'ci_low': ci_low, 'ci_high': ci_high,
        'avg_turns': avg_turns,
    }


def make_plot(per_map_summary, budget_high, budget_low, action_budget, out_path):
    """Bar chart: high-budget score by map with 95% CI, sorted by n_territories."""
    from parallel_risk.env.map_config import MapRegistry
    maps = list(per_map_summary.keys())
    # sort by n_territories
    maps.sort(key=lambda m: MapRegistry.get(m).n_territories)

    scores = [per_map_summary[m]['score_high'] for m in maps]
    ci_lo = [per_map_summary[m]['ci_low'] for m in maps]
    ci_hi = [per_map_summary[m]['ci_high'] for m in maps]
    errs_lo = [s - lo for s, lo in zip(scores, ci_lo)]
    errs_hi = [hi - s for s, hi in zip(scores, ci_hi)]

    labels = [f"{m}\n(n={MapRegistry.get(m).n_territories})" for m in maps]

    fig, ax = plt.subplots(figsize=(max(9, len(maps) * 0.9), 5.2))
    x = np.arange(len(maps))
    ax.bar(x, scores, yerr=[errs_lo, errs_hi], color='#3b7ea1', alpha=0.85,
           capsize=4, edgecolor='#1c3d55')
    ax.axhline(0.5, color='#7a7a7a', linestyle='--', linewidth=1.0,
               label='50% (no advantage)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=9)
    ax.set_ylabel(f'Score of MCTS-{budget_high} vs MCTS-{budget_low}\n(win-rate, draws=0.5)')
    ax.set_ylim(0, 1)
    ax.set_title(f'MCTS opponent-strength calibration (K={action_budget})\n'
                 f'MCTS-uniform({budget_high} sims) vs MCTS-uniform({budget_low} sims)',
                 fontsize=11)
    ax.legend(loc='upper right')
    ax.grid(True, axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Q3: MCTS budget calibration')
    parser.add_argument('--maps', type=str, default=','.join(DEFAULT_MAPS))
    parser.add_argument('--num-games', type=int, default=20,
                        help='Total games per map (split 50/50 between colors)')
    parser.add_argument('--budget-high', type=int, default=200)
    parser.add_argument('--budget-low', type=int, default=40)
    parser.add_argument('--k', '--action-budget', dest='action_budget',
                        type=int, default=5)
    parser.add_argument('--max-turns', type=int, default=40)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--base-seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str,
                        default='experiments/diagnostic_mcts_budget')
    args = parser.parse_args()

    maps = [m.strip() for m in args.maps.split(',') if m.strip()]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'[Q3] MCTS-uniform({args.budget_high}) vs MCTS-uniform({args.budget_low}) | '
          f'K={args.action_budget} | {args.num_games} games/map | {len(maps)} maps')
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
            budget_high=args.budget_high,
            budget_low=args.budget_low,
            action_budget=args.action_budget,
            max_turns=args.max_turns,
            workers=args.workers,
            base_seed=args.base_seed + i * 1000,
        )
        s = summarize(results)
        per_map_summary[map_name] = s
        all_results[map_name] = results
        dt = time.time() - t_m
        print(f'     score={s["score_high"]:.2f}  '
              f'W={s["wins_high"]} L={s["losses_high"]} D={s["draws"]}  '
              f'avg_turns={s["avg_turns"]:.1f}  '
              f'CI95=[{s["ci_low"]:.2f}, {s["ci_high"]:.2f}]  '
              f'({dt:.0f}s)')

        # Save incrementally in case of interrupt
        with open(out_dir / 'results_partial.json', 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'config': vars(args),
                'per_map_summary': per_map_summary,
                'raw_results': all_results,
            }, f, indent=2, default=str)

    total_dt = time.time() - t0
    aggregate_score = float(np.mean([s['score_high'] for s in per_map_summary.values()]))
    print(f'\n=== Aggregate ===')
    print(f'  Mean score across {len(maps)} maps: {aggregate_score:.3f}')
    print(f'  Total wall-clock: {total_dt / 60:.1f} min')

    # Save final
    with open(out_dir / 'results.json', 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'config': vars(args),
            'aggregate_score_high': aggregate_score,
            'wall_clock_min': total_dt / 60,
            'per_map_summary': per_map_summary,
            'raw_results': all_results,
        }, f, indent=2, default=str)

    plot_path = out_dir / f'q3_budget_{args.budget_high}_vs_{args.budget_low}_k{args.action_budget}.png'
    make_plot(per_map_summary, args.budget_high, args.budget_low,
              args.action_budget, plot_path)
    print(f'\nSaved: {out_dir}/results.json')
    print(f'Saved: {plot_path}')


if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    main()
