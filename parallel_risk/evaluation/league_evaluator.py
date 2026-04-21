"""
League evaluation for Parallel Risk.

Evaluates a policy against multiple opponents (random baseline + historical snapshots).
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from ray.rllib.algorithms.ppo import PPO

from parallel_risk.agents.random_agent import RandomAgent
from parallel_risk.agents.checkpoint_agent import CheckpointAgent
from parallel_risk.training.rllib.wrapper import make_rllib_env


def discover_snapshots(snapshot_dir: str) -> List[Dict]:
    """
    Discover all policy snapshots in a directory.

    Args:
        snapshot_dir: Directory containing snapshot subdirectories

    Returns:
        List of snapshot dicts with keys: iteration, path, metadata
        Sorted by iteration number (ascending)
    """
    snapshot_dir = Path(snapshot_dir)
    if not snapshot_dir.exists():
        return []

    snapshots = []

    # Look for iter_XXXXXX subdirectories
    for item in snapshot_dir.iterdir():
        if item.is_dir() and item.name.startswith("iter_"):
            try:
                # Extract iteration number
                iteration = int(item.name.split("_")[1])

                # Load metadata if available
                metadata_path = item / "metadata.json"
                metadata = {}
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)

                snapshots.append({
                    "iteration": iteration,
                    "path": str(item.resolve()),  # Use absolute path
                    "metadata": metadata
                })
            except (IndexError, ValueError):
                continue

    # Sort by iteration
    snapshots.sort(key=lambda x: x["iteration"])

    return snapshots


class LeagueEvaluator:
    """
    Evaluates a policy against a league of opponents.

    Memory-efficient: loads one opponent at a time, evaluates, then unloads.
    """

    def __init__(self, env_config: Optional[Dict] = None):
        """
        Initialize league evaluator.

        Args:
            env_config: Environment configuration dict
        """
        self.env_config = env_config or {
            "map_name": "simple_6",
            "max_turns": 100,
            "action_budget": 5,
            "reward_shaping_type": "sparse",
        }

    def evaluate_league(
        self,
        main_policy_path: str,
        opponent_specs: List[Dict],
        num_episodes: int = 100,
        seed: int = 42,
        verbose: bool = True
    ) -> Dict[str, Dict]:
        """
        Evaluate main policy against multiple opponents.

        Args:
            main_policy_path: Path to main policy checkpoint
            opponent_specs: List of opponent specifications
                Each spec is a dict with keys:
                - "type": "random" or "checkpoint"
                - "name": Display name (e.g., "random_baseline", "snapshot_iter_50")
                - "path": Checkpoint path (for type="checkpoint")
            num_episodes: Episodes per opponent matchup
            seed: Base random seed
            verbose: Print progress

        Returns:
            Dict mapping opponent name to evaluation results:
            {
                "random_baseline": {"win_rate": 0.65, ...},
                "snapshot_iter_50": {"win_rate": 0.72, ...},
                ...
            }
        """
        if verbose:
            print(f"\n{'='*70}")
            print("LEAGUE EVALUATION")
            print(f"{'='*70}")
            print(f"Main policy: {main_policy_path}")
            print(f"Opponents: {len(opponent_specs)}")
            print(f"Episodes per matchup: {num_episodes}")
            print(f"{'='*70}\n")

        # Load main policy
        if verbose:
            print(f"Loading main policy...")
        main_policy_path = str(Path(main_policy_path).resolve())
        main_algorithm = PPO.from_checkpoint(main_policy_path)

        # Create environment
        env = make_rllib_env(self.env_config)
        n_territories = env.env.map_config.n_territories

        # Evaluate against each opponent
        all_results = {}

        for idx, opponent_spec in enumerate(opponent_specs, 1):
            opponent_name = opponent_spec["name"]
            opponent_type = opponent_spec["type"]

            if verbose:
                print(f"\n[{idx}/{len(opponent_specs)}] Evaluating vs {opponent_name}...")

            # Create opponent agent
            opponent_agent = self._create_opponent(
                opponent_spec, n_territories, self.env_config["action_budget"]
            )

            # Run matchup
            results = self._run_matchup(
                main_algorithm,
                opponent_agent,
                env,
                num_episodes,
                seed,
                verbose
            )

            # Store results
            all_results[opponent_name] = results

            if verbose:
                print(f"  Win rate: {results['win_rate']:.2%} "
                      f"({results['wins']}W-{results['losses']}L-{results['draws']}D)")

            # Cleanup opponent to free memory
            if hasattr(opponent_agent, 'unload'):
                opponent_agent.unload()

        # Cleanup
        main_algorithm.stop()
        env.close()

        if verbose:
            print(f"\n{'='*70}")
            print("LEAGUE EVALUATION COMPLETE")
            print(f"{'='*70}\n")

        return all_results

    def _create_opponent(self, opponent_spec: Dict, n_territories: int, action_budget: int):
        """Create opponent agent from specification."""
        opponent_type = opponent_spec["type"]

        if opponent_type == "random":
            return RandomAgent(
                n_territories=n_territories,
                action_budget=action_budget,
                mode="rllib"
            )
        elif opponent_type == "checkpoint":
            checkpoint_path = opponent_spec["path"]
            return CheckpointAgent(
                checkpoint_path=checkpoint_path,
                action_budget=action_budget
            )
        else:
            raise ValueError(f"Unknown opponent type: {opponent_type}")

    def _run_matchup(
        self,
        main_algorithm,
        opponent_agent,
        env,
        num_episodes: int,
        seed: int,
        verbose: bool
    ) -> Dict:
        """Run episodes between main policy and opponent."""
        wins = 0
        losses = 0
        draws = 0
        episode_lengths = []

        for episode_idx in range(num_episodes):
            obs, info = env.reset(seed=seed + episode_idx)
            done = {"__all__": False}
            episode_length = 0

            while not done["__all__"]:
                actions = {}

                # Agent 0: main policy
                if "agent_0" in obs:
                    actions["agent_0"] = main_algorithm.compute_single_action(
                        obs["agent_0"],
                        policy_id="main_policy",
                        explore=False,
                    )

                # Agent 1: opponent
                if "agent_1" in obs:
                    actions["agent_1"] = opponent_agent.get_action(obs["agent_1"])

                obs, rewards, terminateds, truncateds, infos = env.step(actions)

                # Check done
                done = {k: terminateds.get(k, False) or truncateds.get(k, False)
                       for k in ["agent_0", "agent_1", "__all__"]}
                if not done.get("__all__"):
                    done["__all__"] = all([done.get("agent_0", False),
                                           done.get("agent_1", False)])

                episode_length += 1

            # Count result
            reward_0 = rewards.get("agent_0", 0)
            reward_1 = rewards.get("agent_1", 0)

            if reward_0 > reward_1:
                wins += 1
            elif reward_0 < reward_1:
                losses += 1
            else:
                draws += 1

            episode_lengths.append(episode_length)

        # Compute statistics
        total = wins + losses + draws
        return {
            "win_rate": wins / total,
            "loss_rate": losses / total,
            "draw_rate": draws / total,
            "avg_episode_length": float(np.mean(episode_lengths)),
            "std_episode_length": float(np.std(episode_lengths)),
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "total_episodes": total,
            "episode_lengths": episode_lengths,
        }
