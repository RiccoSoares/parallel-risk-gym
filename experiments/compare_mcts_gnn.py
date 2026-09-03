"""Compare MCTS vs masked-random and MCTS vs trained GNN.

Three head-to-head matchups:
1. MCTS (agent_0) vs MaskedRandom (agent_1)
2. GNN (agent_0) vs MaskedRandom (agent_1)
3. MCTS (agent_0) vs GNN (agent_1)

Example:
    PYTHONPATH=. python experiments/compare_mcts_gnn.py \\
        --checkpoint checkpoints/gnn_training_parallel/checkpoint_000050.pt \\
        --budget 200 --num-episodes 200 --verbose
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from parallel_risk.env.parallel_risk_env import ParallelRiskEnv
from parallel_risk.env.map_config import MapRegistry
from parallel_risk.agents.mcts_agent import MCTSAgent
from parallel_risk.agents.masked_random_agent import MaskedRandomAgentRLlib


def run_matchup(
    agent_0,
    agent_1,
    env: ParallelRiskEnv,
    num_episodes: int,
    seed: int,
    agent_0_needs_game_state: bool = False,
    agent_1_needs_game_state: bool = False,
    label: str = '',
    verbose: bool = False,
) -> dict:
    """Generic matchup runner. Handles different agent interfaces:
    - MCTSAgent:             get_action(env.game_state, agent_id)
    - GNNAgent:              get_action(obs_dict)
    - MaskedRandomAgentRLlib: get_action_raw(obs_dict)
    """
    wins = losses = draws = 0
    episode_lengths = []
    action_times_ms = []

    for episode in range(num_episodes):
        obs, _ = env.reset(seed=seed + episode)
        done = False
        turn = 0

        while not done:
            actions = {}

            # agent_0
            t0 = time.perf_counter()
            if agent_0_needs_game_state:
                actions['agent_0'] = agent_0.get_action(env.game_state, 'agent_0')
            else:
                actions['agent_0'] = agent_0.get_action(obs['agent_0'])
            action_times_ms.append((time.perf_counter() - t0) * 1000)

            # agent_1
            if 'agent_1' in obs:
                if agent_1_needs_game_state:
                    actions['agent_1'] = agent_1.get_action(env.game_state, 'agent_1')
                elif hasattr(agent_1, 'get_action_raw'):
                    actions['agent_1'] = agent_1.get_action_raw(obs['agent_1'])
                else:
                    actions['agent_1'] = agent_1.get_action(obs['agent_1'])

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

        if verbose and (episode + 1) % max(1, num_episodes // 5) == 0:
            print(
                f"  {label} | Episode {episode+1}/{num_episodes} | "
                f"W={wins} L={losses} D={draws}"
            )

    total = wins + losses + draws
    return {
        'win_rate': wins / total,
        'loss_rate': losses / total,
        'draw_rate': draws / total,
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'total_episodes': total,
        'avg_episode_length': float(np.mean(episode_lengths)),
        'std_episode_length': float(np.std(episode_lengths)),
        'episode_lengths': episode_lengths,
        'avg_action_time_ms': float(np.mean(action_times_ms)) if action_times_ms else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare MCTS vs Random and MCTS vs GNN"
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/gnn_training_parallel/checkpoint_000050.pt',
        help='Path to GNN .pt checkpoint',
    )
    parser.add_argument('--budget', type=int, default=200,
                        help='MCTS simulation budget per move')
    parser.add_argument('--num-episodes', type=int, default=200)
    parser.add_argument('--map-name', type=str, default='simple_6')
    parser.add_argument('--max-turns', type=int, default=100)
    parser.add_argument('--action-budget', type=int, default=5)
    parser.add_argument('--results-dir', type=str,
                        default='experiments/mcts_vs_gnn_results')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--uct-c', type=float, default=1.41)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--skip-gnn', action='store_true',
                        help='Skip GNN matchups (if no checkpoint available)')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    map_config = MapRegistry.get(args.map_name)
    env = ParallelRiskEnv(
        map_name=args.map_name,
        max_turns=args.max_turns,
        reward_shaping_config=None,
    )

    print(f"Setting up agents...")
    mcts_agent = MCTSAgent.from_env(
        env,
        simulation_budget=args.budget,
        uct_c=args.uct_c,
        action_budget=args.action_budget,
    )
    random_agent = MaskedRandomAgentRLlib.from_env(env, action_budget=args.action_budget)

    gnn_agent = None
    if not args.skip_gnn:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            print(f"Warning: checkpoint not found at {checkpoint_path}. "
                  f"Use --skip-gnn or provide a valid --checkpoint path.")
            print("Running MCTS vs Random only.\n")
            args.skip_gnn = True
        else:
            try:
                from parallel_risk.agents.gnn_agent import GNNAgent
                print(f"Loading GNN from {checkpoint_path}...")
                gnn_agent = GNNAgent.from_checkpoint(
                    str(checkpoint_path),
                    map_config=map_config,
                    device=args.device,
                )
            except ImportError as e:
                print(f"Warning: Could not load GNNAgent ({e}). Running MCTS vs Random only.")
                args.skip_gnn = True

    all_results = {}

    # --- Matchup 1: MCTS vs Random ---
    print(f"\n[1/3] MCTS (budget={args.budget}) vs MaskedRandom | {args.num_episodes} episodes")
    result = run_matchup(
        agent_0=mcts_agent,
        agent_1=random_agent,
        env=env,
        num_episodes=args.num_episodes,
        seed=args.seed,
        agent_0_needs_game_state=True,
        agent_1_needs_game_state=False,
        label='MCTS vs Random',
        verbose=args.verbose,
    )
    all_results['mcts_vs_random'] = result
    print(f"  => Win rate: {result['win_rate']:.1%} | "
          f"Avg ep length: {result['avg_episode_length']:.1f} | "
          f"Avg action time: {result['avg_action_time_ms']:.0f}ms")

    if not args.skip_gnn and gnn_agent is not None:
        # --- Matchup 2: GNN vs Random ---
        print(f"\n[2/3] GNN vs MaskedRandom | {args.num_episodes} episodes")
        result = run_matchup(
            agent_0=gnn_agent,
            agent_1=random_agent,
            env=env,
            num_episodes=args.num_episodes,
            seed=args.seed,
            agent_0_needs_game_state=False,
            agent_1_needs_game_state=False,
            label='GNN vs Random',
            verbose=args.verbose,
        )
        all_results['gnn_vs_random'] = result
        print(f"  => Win rate: {result['win_rate']:.1%} | "
              f"Avg ep length: {result['avg_episode_length']:.1f} | "
              f"Avg action time: {result['avg_action_time_ms']:.0f}ms")

        # --- Matchup 3: MCTS vs GNN ---
        print(f"\n[3/3] MCTS (budget={args.budget}) vs GNN | {args.num_episodes} episodes")
        result = run_matchup(
            agent_0=mcts_agent,
            agent_1=gnn_agent,
            env=env,
            num_episodes=args.num_episodes,
            seed=args.seed,
            agent_0_needs_game_state=True,
            agent_1_needs_game_state=False,
            label='MCTS vs GNN',
            verbose=args.verbose,
        )
        all_results['mcts_vs_gnn'] = result
        print(f"  => Win rate: {result['win_rate']:.1%} | "
              f"Avg ep length: {result['avg_episode_length']:.1f} | "
              f"Avg action time: {result['avg_action_time_ms']:.0f}ms")

    # Save results
    json_path = results_dir / 'comparison_results.json'
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {json_path}")

    # Summary table
    print("\n=== Summary ===")
    print(f"{'Matchup':<22} | {'Win Rate':>9} | {'Avg Ep Len':>10} | {'Avg Action (ms)':>15}")
    print("-" * 64)
    labels = {
        'mcts_vs_random': f'MCTS(b={args.budget}) vs Rand',
        'gnn_vs_random':  'GNN vs Random        ',
        'mcts_vs_gnn':    f'MCTS(b={args.budget}) vs GNN ',
    }
    for key, label in labels.items():
        if key in all_results:
            r = all_results[key]
            print(
                f"{label:<22} | {r['win_rate']:>8.1%} | "
                f"{r['avg_episode_length']:>10.1f} | "
                f"{r['avg_action_time_ms']:>15.0f}"
            )


if __name__ == '__main__':
    main()
