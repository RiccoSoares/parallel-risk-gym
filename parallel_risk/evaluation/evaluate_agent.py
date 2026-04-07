"""
Evaluation script for trained Parallel Risk agents.

Run head-to-head matches between policies and compute win rates.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from ray.rllib.algorithms.ppo import PPO

from parallel_risk.agents.random_agent import RandomAgent
from parallel_risk.training.rllib_wrapper import make_rllib_env


def evaluate_policy(
    policy_checkpoint_path: str,
    opponent: str = "random",
    num_episodes: int = 100,
    map_name: str = "simple_6",
    max_turns: int = 100,
    action_budget: int = 5,
    seed: int = 42,
    verbose: bool = False,
) -> Dict:
    """
    Evaluate a trained policy against an opponent.

    Args:
        policy_checkpoint_path: Path to RLlib checkpoint directory
        opponent: "random" or path to another checkpoint
        num_episodes: Number of evaluation episodes
        map_name: Map to evaluate on
        max_turns: Max turns per episode
        action_budget: Actions per turn
        seed: Random seed
        verbose: Print progress

    Returns:
        Dict with evaluation results:
            - win_rate: float
            - loss_rate: float
            - draw_rate: float
            - avg_episode_length: float
            - wins: int
            - losses: int
            - draws: int
            - episode_lengths: list[int]
    """
    # Load trained policy
    if verbose:
        print(f"Loading policy from {policy_checkpoint_path}...")

    algorithm = PPO.from_checkpoint(policy_checkpoint_path)

    # Create environment
    env_config = {
        "map_name": map_name,
        "max_turns": max_turns,
        "action_budget": action_budget,
        "reward_shaping_type": "sparse",  # Evaluation with sparse rewards
    }
    env = make_rllib_env(env_config)

    # Get map info for random agent
    n_territories = env.env.env.map_config.n_territories

    # Create opponent
    if opponent == "random":
        opponent_agent = RandomAgent(
            n_territories=n_territories,
            action_budget=action_budget,
            mode="rllib"
        )
        opponent_policy = None
        if verbose:
            print("Opponent: Random agent")
    else:
        if verbose:
            print(f"Loading opponent from {opponent}...")
        opponent_policy = PPO.from_checkpoint(opponent)
        opponent_agent = None
        if verbose:
            print("Opponent: Trained policy")

    # Run evaluation episodes
    wins = 0
    losses = 0
    draws = 0
    episode_lengths = []

    if verbose:
        print(f"\nRunning {num_episodes} evaluation episodes...")

    for episode_idx in range(num_episodes):
        obs, info = env.reset(seed=seed + episode_idx)
        done = {"__all__": False}
        episode_length = 0

        while not done["__all__"]:
            actions = {}

            # Agent 0: trained policy
            if "agent_0" in obs:
                actions["agent_0"] = algorithm.compute_single_action(
                    obs["agent_0"],
                    policy_id="main_policy",
                    explore=False,  # Deterministic evaluation
                )

            # Agent 1: opponent
            if "agent_1" in obs:
                if opponent_policy is not None:
                    # Another trained policy
                    actions["agent_1"] = opponent_policy.compute_single_action(
                        obs["agent_1"],
                        policy_id="main_policy",
                        explore=False,
                    )
                else:
                    # Random agent
                    actions["agent_1"] = opponent_agent.get_action()

            obs, rewards, terminateds, truncateds, infos = env.step(actions)

            # Check for done
            done = {k: terminateds.get(k, False) or truncateds.get(k, False)
                   for k in ["agent_0", "agent_1", "__all__"]}
            if not done.get("__all__"):
                done["__all__"] = all([done.get("agent_0", False),
                                       done.get("agent_1", False)])

            episode_length += 1

        # Determine winner
        reward_0 = rewards.get("agent_0", 0)
        reward_1 = rewards.get("agent_1", 0)

        if reward_0 > reward_1:
            wins += 1
        elif reward_0 < reward_1:
            losses += 1
        else:
            draws += 1

        episode_lengths.append(episode_length)

        if verbose and (episode_idx + 1) % 10 == 0:
            current_win_rate = wins / (episode_idx + 1)
            print(f"  Episode {episode_idx + 1}/{num_episodes} - "
                  f"Win rate: {current_win_rate:.2%}")

    # Compute statistics
    total = wins + losses + draws
    results = {
        "win_rate": wins / total,
        "loss_rate": losses / total,
        "draw_rate": draws / total,
        "avg_episode_length": np.mean(episode_lengths),
        "std_episode_length": np.std(episode_lengths),
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "total_episodes": total,
        "episode_lengths": episode_lengths,
        "checkpoint": policy_checkpoint_path,
        "opponent": opponent,
        "map_name": map_name,
        "seed": seed,
    }

    if verbose:
        print(f"\n{'='*60}")
        print("EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"Total Episodes:  {total}")
        print(f"Wins:            {wins} ({results['win_rate']:.2%})")
        print(f"Losses:          {losses} ({results['loss_rate']:.2%})")
        print(f"Draws:           {draws} ({results['draw_rate']:.2%})")
        print(f"Avg Ep Length:   {results['avg_episode_length']:.1f} ± "
              f"{results['std_episode_length']:.1f}")
        print(f"{'='*60}\n")

    return results


def main():
    """CLI entry point for evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Parallel Risk agent"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to RLlib checkpoint directory"
    )
    parser.add_argument(
        "--opponent",
        type=str,
        default="random",
        help="Opponent type: 'random' or path to checkpoint"
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--map-name",
        type=str,
        default="simple_6",
        help="Map to evaluate on"
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=100,
        help="Max turns per episode"
    )
    parser.add_argument(
        "--action-budget",
        type=int,
        default=5,
        help="Actions per turn"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results (optional)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress"
    )

    args = parser.parse_args()

    # Run evaluation
    results = evaluate_policy(
        policy_checkpoint_path=args.checkpoint,
        opponent=args.opponent,
        num_episodes=args.num_episodes,
        map_name=args.map_name,
        max_turns=args.max_turns,
        action_budget=args.action_budget,
        seed=args.seed,
        verbose=args.verbose,
    )

    # Save results if output path specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy types to Python types for JSON serialization
        results_serializable = {
            k: (v.tolist() if isinstance(v, np.ndarray) else
                float(v) if isinstance(v, (np.floating, np.integer)) else v)
            for k, v in results.items()
        }

        with open(output_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)

        print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
