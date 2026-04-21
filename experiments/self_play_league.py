"""
Self-play league experiment for Parallel Risk.

This script orchestrates a complete league experiment:
1. Training with PPO (self-play) and periodic policy snapshots
2. League evaluation: main policy vs random baseline + historical snapshots
3. Rich visualization showing learning progress across opponents

Usage:
    python experiments/self_play_league.py --num-iterations 500
    python experiments/self_play_league.py --num-iterations 10 --snapshot-interval 5 --eval-interval 5
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import ray
import yaml

from parallel_risk.training.rllib.train import (
    load_config,
    create_env_config,
    setup_algorithm
)
from parallel_risk.evaluation.league_evaluator import (
    LeagueEvaluator,
    discover_snapshots
)
from parallel_risk.evaluation.league_visualize import plot_league_results


def save_policy_snapshot(algo, iteration: int, snapshot_dir: Path, metadata: dict):
    """
    Save a policy snapshot with metadata.

    Args:
        algo: RLlib algorithm instance
        iteration: Training iteration number
        snapshot_dir: Directory to save snapshots
        metadata: Dict with training metadata
    """
    iter_dir = snapshot_dir / f"iter_{iteration:06d}"
    algo.save(str(iter_dir))

    # Save metadata
    metadata_path = iter_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return iter_dir


def run_training_with_snapshots(
    config_path: str,
    num_iterations: int,
    checkpoint_dir: Path,
    snapshot_dir: Path,
    snapshot_interval: int = 50,
    num_workers: int = 4,
    verbose: bool = True
):
    """
    Run training with periodic snapshot saving.

    Args:
        config_path: Path to training configuration YAML
        num_iterations: Number of training iterations
        checkpoint_dir: Directory for training checkpoints
        snapshot_dir: Directory for league snapshots
        snapshot_interval: Save snapshot every N iterations
        num_workers: Number of rollout workers
        verbose: Print progress

    Returns:
        Configured algorithm after training
    """
    print("\n" + "="*70)
    print("STEP 1: Training with Policy Snapshots")
    print("="*70)

    # Load config
    config = load_config(config_path)

    # Apply overrides
    config["training"]["num_workers"] = num_workers

    if verbose:
        print(f"Training for {num_iterations} iterations")
        print(f"Checkpoint dir: {checkpoint_dir}")
        print(f"Snapshot dir: {snapshot_dir}")
        print(f"Snapshot interval: {snapshot_interval} iterations")
        print(f"Workers: {num_workers}\n")

    # Create directories
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    # Setup algorithm
    algo = setup_algorithm(config, str(checkpoint_dir))

    checkpoint_freq = config["checkpoint"]["frequency"]

    # Training loop
    for iteration in range(1, num_iterations + 1):
        result = algo.train()

        # Print progress
        if verbose:
            print(f"\n=== Iteration {iteration}/{num_iterations} ===")

            # Extract metrics
            env_runners = result.get('env_runners', {})
            episode_reward = env_runners.get('episode_reward_mean',
                                            result.get('episode_reward_mean', 'N/A'))
            episode_length = env_runners.get('episode_len_mean',
                                            result.get('episode_len_mean', 'N/A'))

            if isinstance(episode_reward, (int, float)):
                print(f"  Episode reward mean: {episode_reward:.3f}")
            else:
                print(f"  Episode reward mean: {episode_reward}")

            if isinstance(episode_length, (int, float)):
                print(f"  Episode length mean: {episode_length:.1f}")
            else:
                print(f"  Episode length mean: {episode_length}")

        # Save training checkpoint
        if iteration % checkpoint_freq == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_{iteration:06d}"
            algo.save(str(checkpoint_path))
            if verbose:
                print(f"  💾 Checkpoint saved: {checkpoint_path}")

        # Save snapshot for league
        if iteration % snapshot_interval == 0:
            env_runners = result.get('env_runners', {})
            episode_reward = env_runners.get('episode_reward_mean',
                                            result.get('episode_reward_mean', None))

            metadata = {
                "iteration": iteration,
                "episode_reward_mean": float(episode_reward) if isinstance(episode_reward, (int, float)) else None,
                "timestamp": datetime.now().isoformat(),
            }

            snapshot_path = save_policy_snapshot(algo, iteration, snapshot_dir, metadata)
            if verbose:
                print(f"  📸 Snapshot saved: {snapshot_path}")

    if verbose:
        print(f"\n✓ Training completed ({num_iterations} iterations)")

    return algo


def run_league_evaluation(
    checkpoint_dir: Path,
    snapshot_dir: Path,
    eval_interval: int,
    num_episodes: int,
    env_config: dict,
    results_dir: Path,
    verbose: bool = True
):
    """
    Evaluate all checkpoints against league of opponents.

    Args:
        checkpoint_dir: Directory with training checkpoints
        snapshot_dir: Directory with league snapshots
        eval_interval: Evaluate every N iterations
        num_episodes: Episodes per matchup
        env_config: Environment configuration
        results_dir: Directory for results
        verbose: Print progress

    Returns:
        Dict with all evaluation results
    """
    print("\n" + "="*70)
    print("STEP 2: League Evaluation")
    print("="*70)

    # Discover checkpoints to evaluate
    checkpoints = []
    for item in sorted(checkpoint_dir.iterdir()):
        if item.is_dir() and item.name.startswith("checkpoint_"):
            try:
                iteration = int(item.name.split("_")[1])
                if iteration % eval_interval == 0:
                    checkpoints.append((iteration, item))
            except (IndexError, ValueError):
                continue

    if not checkpoints:
        print("✗ No checkpoints found for evaluation!")
        return {}

    if verbose:
        print(f"Found {len(checkpoints)} checkpoints to evaluate")
        print(f"Episodes per matchup: {num_episodes}\n")

    # Discover snapshots for opponents
    snapshots = discover_snapshots(snapshot_dir)

    if verbose:
        print(f"Found {len(snapshots)} historical snapshots as opponents")

    # Create league evaluator
    evaluator = LeagueEvaluator(env_config=env_config)

    # Evaluate each checkpoint
    all_results = {}

    for idx, (iteration, checkpoint_path) in enumerate(checkpoints, 1):
        if verbose:
            print(f"\n{'='*70}")
            print(f"Evaluating Checkpoint {idx}/{len(checkpoints)} (Iteration {iteration})")
            print(f"{'='*70}")

        # Build opponent specs
        opponent_specs = [{"type": "random", "name": "random_baseline"}]

        # Add snapshots as opponents (only those at or before current iteration)
        for snapshot in snapshots:
            if snapshot["iteration"] <= iteration:
                opponent_specs.append({
                    "type": "checkpoint",
                    "path": snapshot["path"],
                    "name": f"snapshot_iter_{snapshot['iteration']}"
                })

        # Run league evaluation
        results = evaluator.evaluate_league(
            main_policy_path=str(checkpoint_path),
            opponent_specs=opponent_specs,
            num_episodes=num_episodes,
            verbose=verbose
        )

        all_results[str(iteration)] = results

    if verbose:
        print(f"\n✓ League evaluation completed ({len(checkpoints)} checkpoints)")

    return all_results


def generate_summary(all_results: dict, results_dir: Path, metadata: dict, verbose: bool = True):
    """
    Generate summary and visualizations.

    Args:
        all_results: Evaluation results dict
        results_dir: Directory for results
        metadata: Experiment metadata
        verbose: Print progress
    """
    print("\n" + "="*70)
    print("STEP 3: Generating Summary and Plots")
    print("="*70)

    # Save combined results JSON
    combined_path = results_dir / "league_results.json"

    output_data = {
        "evaluations": all_results,
        "metadata": metadata
    }

    with open(combined_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"✓ Saved results to {combined_path}")

    # Generate plots
    try:
        plot_league_results(str(combined_path), output_dir=str(results_dir))
        print("✓ Generated league plots")
    except ImportError as e:
        print(f"⚠ Could not generate plots (matplotlib/seaborn not available): {e}")
    except Exception as e:
        print(f"⚠ Plot generation failed: {e}")

    # Generate text summary
    summary_path = results_dir / "league_summary.txt"

    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("PARALLEL RISK SELF-PLAY LEAGUE EXPERIMENT\n")
        f.write("="*70 + "\n\n")

        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("Experiment Configuration:\n")
        for key, value in metadata.items():
            f.write(f"  {key}: {value}\n")

        f.write("\n" + "="*70 + "\n")
        f.write("EVALUATION RESULTS\n")
        f.write("="*70 + "\n\n")

        # Analyze results by iteration
        iterations = sorted(all_results.keys(), key=int)

        for iteration in iterations:
            iter_results = all_results[iteration]

            f.write(f"\nIteration {iteration}:\n")
            f.write("-" * 50 + "\n")

            for opponent_name, results in iter_results.items():
                f.write(f"  vs {opponent_name:30s}: "
                       f"{results['win_rate']:6.2%} win rate "
                       f"({results['wins']}W-{results['losses']}L-{results['draws']}D)\n")

            # Compute average across all opponents
            avg_win_rate = sum(r['win_rate'] for r in iter_results.values()) / len(iter_results)
            f.write(f"  {'Average':32s}: {avg_win_rate:6.2%}\n")

        f.write("\n" + "="*70 + "\n")
        f.write("LEARNING VALIDATION\n")
        f.write("="*70 + "\n\n")

        # Check final iteration performance
        final_iteration = max(map(int, iterations))
        final_results = all_results[str(final_iteration)]

        random_win_rate = final_results.get("random_baseline", {}).get("win_rate", 0)
        f.write(f"Final win rate vs random: {random_win_rate:.2%}\n")

        # Check if agent beats past self
        snapshot_opponents = [name for name in final_results.keys()
                             if name.startswith("snapshot_")]
        if snapshot_opponents:
            snapshot_win_rates = [final_results[name]['win_rate']
                                 for name in snapshot_opponents]
            avg_vs_snapshots = sum(snapshot_win_rates) / len(snapshot_win_rates)
            f.write(f"Avg win rate vs historical snapshots: {avg_vs_snapshots:.2%}\n\n")

            if avg_vs_snapshots > 0.6:
                f.write("✓ SUCCESS: Agent demonstrates improvement over past versions\n")
            elif avg_vs_snapshots > 0.5:
                f.write("⚠ PARTIAL SUCCESS: Some improvement over past versions\n")
            else:
                f.write("✗ FAILURE: Not improving over past versions\n")

        f.write("\n" + "="*70 + "\n")

    print(f"✓ Saved text summary to {summary_path}")

    # Print summary to console
    if verbose:
        print("\n" + "="*70)
        print("FINAL RESULTS")
        print("="*70)

        final_iteration = max(map(int, iterations))
        final_results = all_results[str(final_iteration)]

        print(f"\nFinal iteration: {final_iteration}")
        print(f"Win rate vs random: {random_win_rate:.2%}")

        if snapshot_opponents:
            print(f"Avg vs historical snapshots: {avg_vs_snapshots:.2%}")

        print(f"\nAll results saved to: {results_dir}")
        print("="*70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run self-play league experiment for Parallel Risk"
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=500,
        help="Number of training iterations"
    )
    parser.add_argument(
        "--snapshot-interval",
        type=int,
        default=50,
        help="Save snapshot every N iterations"
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=50,
        help="Evaluate every N iterations"
    )
    parser.add_argument(
        "--num-eval-episodes",
        type=int,
        default=100,
        help="Episodes per opponent matchup"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of training workers"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="parallel_risk/training/rllib/configs/ppo_sparse.yaml",
        help="Training config file"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="experiments/league_results",
        help="Directory for results"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/league_training",
        help="Directory for training checkpoints"
    )
    parser.add_argument(
        "--snapshot-dir",
        type=str,
        default="league_snapshots",
        help="Directory for league snapshots"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training, only evaluate existing checkpoints"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output"
    )

    args = parser.parse_args()

    # Convert to paths
    results_dir = Path(args.results_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    snapshot_dir = Path(args.snapshot_dir)
    config_path = Path(args.config)

    # Create directories
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("PARALLEL RISK SELF-PLAY LEAGUE EXPERIMENT")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Training iterations:  {args.num_iterations}")
    print(f"  Snapshot interval:    {args.snapshot_interval}")
    print(f"  Eval interval:        {args.eval_interval}")
    print(f"  Eval episodes:        {args.num_eval_episodes}")
    print(f"  Training workers:     {args.num_workers}")
    print(f"  Config file:          {config_path}")
    print(f"  Checkpoint dir:       {checkpoint_dir}")
    print(f"  Snapshot dir:         {snapshot_dir}")
    print(f"  Results dir:          {results_dir}")

    # Load config for env settings
    config = load_config(str(config_path))
    env_config = create_env_config(config)

    # Save metadata
    metadata = {
        "map_name": env_config["map_name"],
        "max_turns": env_config["max_turns"],
        "action_budget": env_config["action_budget"],
        "reward_shaping": config["env"]["reward_shaping"],
        "num_iterations": args.num_iterations,
        "snapshot_interval": args.snapshot_interval,
        "eval_interval": args.eval_interval,
        "num_eval_episodes": args.num_eval_episodes,
        "num_workers": args.num_workers,
        "timestamp": datetime.now().isoformat(),
    }

    # Step 1: Training
    if not args.skip_training:
        algo = run_training_with_snapshots(
            config_path=str(config_path),
            num_iterations=args.num_iterations,
            checkpoint_dir=checkpoint_dir,
            snapshot_dir=snapshot_dir,
            snapshot_interval=args.snapshot_interval,
            num_workers=args.num_workers,
            verbose=args.verbose
        )

        # Cleanup
        algo.stop()
    else:
        print("\n⚠ Skipping training (--skip-training flag)")

    # Step 2: League Evaluation
    all_results = run_league_evaluation(
        checkpoint_dir=checkpoint_dir,
        snapshot_dir=snapshot_dir,
        eval_interval=args.eval_interval,
        num_episodes=args.num_eval_episodes,
        env_config=env_config,
        results_dir=results_dir,
        verbose=args.verbose
    )

    if not all_results:
        print("\n✗ No evaluation results obtained")
        sys.exit(1)

    # Step 3: Generate summary
    generate_summary(all_results, results_dir, metadata, verbose=args.verbose)

    # Cleanup
    ray.shutdown()

    print("\n✓ Self-play league experiment complete!")


if __name__ == "__main__":
    main()
