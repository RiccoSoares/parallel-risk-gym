"""Validate MCTS against masked-random baseline across simulation budgets.

Runs MCTS (agent_0) vs MaskedRandomAgentRLlib (agent_1) for each budget
in a configurable sweep, tracking win rate and action timing.

Output JSON is compatible with parallel_risk.evaluation.visualize.plot_all()
(budget values are used as iteration keys).

Example:
    # Quick smoke test
    PYTHONPATH=. python experiments/validate_mcts.py --budgets 10 --num-episodes 5 --verbose

    # Full sweep
    PYTHONPATH=. python experiments/validate_mcts.py --budgets 50,100,200 --num-episodes 200
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from parallel_risk.env.parallel_risk_env import ParallelRiskEnv
from parallel_risk.agents.mcts_agent import MCTSAgent
from parallel_risk.agents.masked_random_agent import MaskedRandomAgentRLlib


def evaluate_mcts_vs_random(
    budget: int,
    num_episodes: int,
    map_name: str = "simple_6",
    max_turns: int = 100,
    action_budget: int = 5,
    seed: int = 42,
    uct_c: float = 1.41,
    verbose: bool = False,
) -> dict:
    """Run MCTS vs masked-random for a fixed simulation budget.

    Returns a result dict compatible with visualize.plot_all().
    """
    env = ParallelRiskEnv(
        map_name=map_name,
        max_turns=max_turns,
        reward_shaping_config=None,  # sparse rewards for fair evaluation
    )
    mcts_agent = MCTSAgent.from_env(
        env,
        simulation_budget=budget,
        uct_c=uct_c,
        action_budget=action_budget,
    )
    random_agent = MaskedRandomAgentRLlib.from_env(env, action_budget=action_budget)

    wins = losses = draws = 0
    episode_lengths = []
    action_times_ms = []

    for episode in range(num_episodes):
        obs, _ = env.reset(seed=seed + episode)
        done = False
        turn = 0

        while not done:
            actions = {}

            # MCTS needs the full game state
            t0 = time.perf_counter()
            actions['agent_0'] = mcts_agent.get_action(env.game_state, 'agent_0')
            action_times_ms.append((time.perf_counter() - t0) * 1000)

            if 'agent_1' in obs:
                actions['agent_1'] = random_agent.get_action_raw(obs['agent_1'])

            obs, rewards, terms, truncs, _ = env.step(actions)
            done = terms.get('__all__', False) or truncs.get('__all__', False)
            turn += 1

        episode_lengths.append(turn)
        r0 = rewards.get('agent_0', 0.0)
        r1 = rewards.get('agent_1', 0.0)
        if r0 > r1:
            wins += 1
        elif r1 > r0:
            losses += 1
        else:
            draws += 1

        if verbose and (episode + 1) % max(1, num_episodes // 10) == 0:
            print(
                f"  Budget={budget} | Episode {episode+1}/{num_episodes} | "
                f"W={wins} L={losses} D={draws} | "
                f"Avg action time: {np.mean(action_times_ms):.0f}ms"
            )

    total = wins + losses + draws
    avg_len = float(np.mean(episode_lengths))
    std_len = float(np.std(episode_lengths))
    avg_time = float(np.mean(action_times_ms)) if action_times_ms else 0.0

    return {
        'win_rate': wins / total,
        'loss_rate': losses / total,
        'draw_rate': draws / total,
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'total_episodes': total,
        'avg_episode_length': avg_len,
        'std_episode_length': std_len,
        'episode_lengths': episode_lengths,
        'avg_action_time_ms': avg_time,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Validate MCTS against masked-random baseline"
    )
    parser.add_argument(
        '--budgets',
        type=str,
        default='50,100,200',
        help='Comma-separated simulation budgets to sweep (default: 50,100,200)',
    )
    parser.add_argument('--num-episodes', type=int, default=200)
    parser.add_argument('--map-name', type=str, default='simple_6')
    parser.add_argument('--max-turns', type=int, default=100)
    parser.add_argument('--action-budget', type=int, default=5)
    parser.add_argument('--results-dir', type=str, default='experiments/mcts_validation_results')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--uct-c', type=float, default=1.41)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    budgets = [int(b.strip()) for b in args.budgets.split(',')]
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Validating MCTS vs masked-random")
    print(f"  Budgets:  {budgets}")
    print(f"  Episodes: {args.num_episodes} per budget")
    print(f"  Map:      {args.map_name}")
    print()

    evaluations = {}
    for budget in budgets:
        print(f"[Budget={budget}] Running {args.num_episodes} episodes...")
        result = evaluate_mcts_vs_random(
            budget=budget,
            num_episodes=args.num_episodes,
            map_name=args.map_name,
            max_turns=args.max_turns,
            action_budget=args.action_budget,
            seed=args.seed,
            uct_c=args.uct_c,
            verbose=args.verbose,
        )
        evaluations[str(budget)] = result
        print(
            f"  => Win rate: {result['win_rate']:.1%} | "
            f"Avg ep length: {result['avg_episode_length']:.1f} | "
            f"Avg action time: {result['avg_action_time_ms']:.0f}ms"
        )

    # Save JSON (compatible with visualize.plot_all())
    output = {'evaluations': evaluations}
    json_path = results_dir / 'mcts_validation_results.json'
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {json_path}")

    # Print summary table
    print("\n=== Summary ===")
    print(f"{'Budget':>8} | {'Win Rate':>9} | {'Avg Ep Len':>10} | {'Avg Action (ms)':>15}")
    print("-" * 52)
    for budget in budgets:
        r = evaluations[str(budget)]
        print(
            f"{budget:>8} | {r['win_rate']:>8.1%} | "
            f"{r['avg_episode_length']:>10.1f} | "
            f"{r['avg_action_time_ms']:>15.0f}"
        )

    # Try to generate plots
    try:
        from parallel_risk.evaluation.visualize import plot_all
        plot_all(str(json_path), str(results_dir))
        print(f"\nPlots saved to {results_dir}/")
    except Exception as e:
        print(f"\n(Could not generate plots: {e})")


if __name__ == '__main__':
    main()
