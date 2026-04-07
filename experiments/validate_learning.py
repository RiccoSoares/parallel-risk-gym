"""
Learning validation experiment for Parallel Risk.

This script orchestrates a complete validation experiment:
1. Baseline evaluation (random vs. random)
2. Training with PPO
3. Periodic evaluation of checkpoints vs. random
4. Results visualization and summary

Usage:
    python experiments/validate_learning.py --num-iterations 500
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


def run_baseline_evaluation(results_dir, num_episodes=100, verbose=True):
    """Evaluate random vs. random as baseline."""
    print("\n" + "="*70)
    print("STEP 1: Baseline Evaluation (Random vs. Random)")
    print("="*70)

    if verbose:
        print(f"Running {num_episodes} episodes with two random agents...")
        print("This establishes empirical baseline performance.\n")

    # Import here to avoid issues if Ray isn't initialized
    from parallel_risk.agents.random_agent import RandomAgent
    from parallel_risk.training.rllib_wrapper import make_rllib_env

    # Create environment
    env_config = {
        "map_name": "simple_6",
        "max_turns": 100,
        "action_budget": 5,
        "reward_shaping_type": "sparse",
    }
    env = make_rllib_env(env_config)
    n_territories = env.env.map_config.n_territories

    # Create two random agents
    random_agent_0 = RandomAgent(n_territories=n_territories, action_budget=5, mode="rllib")
    random_agent_1 = RandomAgent(n_territories=n_territories, action_budget=5, mode="rllib")

    # Run episodes
    wins = 0
    losses = 0
    draws = 0
    episode_lengths = []

    for episode in range(num_episodes):
        obs, info = env.reset(seed=42 + episode)
        done = {"__all__": False}
        episode_length = 0

        while not done["__all__"]:
            actions = {}
            if "agent_0" in obs:
                actions["agent_0"] = random_agent_0.get_action()
            if "agent_1" in obs:
                actions["agent_1"] = random_agent_1.get_action()

            obs, rewards, terminateds, truncateds, infos = env.step(actions)

            done = {k: terminateds.get(k, False) or truncateds.get(k, False)
                   for k in ["agent_0", "agent_1", "__all__"]}
            if not done.get("__all__"):
                done["__all__"] = all([done.get("agent_0", False), done.get("agent_1", False)])

            episode_length += 1

        # Count results
        reward_0 = rewards.get("agent_0", 0)
        reward_1 = rewards.get("agent_1", 0)

        if reward_0 > reward_1:
            wins += 1
        elif reward_0 < reward_1:
            losses += 1
        else:
            draws += 1

        episode_lengths.append(episode_length)

        if verbose and (episode + 1) % 20 == 0:
            current_win_rate = wins / (episode + 1)
            print(f"  Episode {episode + 1}/{num_episodes} - Win rate: {current_win_rate:.2%}")

    total = wins + losses + draws
    win_rate = wins / total

    baseline_results = {
        "win_rate": win_rate,
        "loss_rate": losses / total,
        "draw_rate": draws / total,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "total_episodes": total,
        "avg_episode_length": np.mean(episode_lengths),
        "std_episode_length": np.std(episode_lengths),
        "note": "Empirical baseline - two random agents",
    }

    print(f"\n{'='*70}")
    print(f"Baseline Results:")
    print(f"  Win rate: {win_rate:.2%} (expected ~50%)")
    print(f"  Draws: {draws} ({baseline_results['draw_rate']:.2%})")
    print(f"  Avg episode length: {baseline_results['avg_episode_length']:.1f}")
    print(f"{'='*70}\n")

    # Save baseline results
    baseline_path = results_dir / "baseline_results.json"
    with open(baseline_path, 'w') as f:
        json.dump(baseline_results, f, indent=2)

    return baseline_results


def run_training(config_path, num_iterations, checkpoint_dir, num_workers=4, verbose=True):
    """Run PPO training."""
    print("\n" + "="*70)
    print("STEP 2: Training PPO Agent")
    print("="*70)

    if verbose:
        print(f"Training for {num_iterations} iterations with {num_workers} workers")
        print(f"Checkpoints will be saved to: {checkpoint_dir}")
        print("This will take approximately 2-3 hours...\n")

    # Build training command
    cmd = [
        sys.executable, "-m", "parallel_risk.training.train_rllib",
        "--config", str(config_path),
        "--num-iterations", str(num_iterations),
        "--checkpoint-dir", str(checkpoint_dir),
        "--num-workers", str(num_workers),
    ]

    # Run training
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("\n✓ Training completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Training failed with error: {e}")
        return False


def discover_checkpoints(checkpoint_dir):
    """Find all checkpoint directories."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return []

    checkpoints = []

    # Check for numbered checkpoint subdirectories (checkpoint_000100, etc.)
    for item in checkpoint_dir.iterdir():
        if item.is_dir() and item.name.startswith("checkpoint_"):
            # Extract iteration number
            try:
                iteration = int(item.name.split("_")[1])
                checkpoints.append((iteration, item))
            except (IndexError, ValueError):
                continue

    # If no numbered checkpoints found, check for a single checkpoint directory
    # (RLlib saves to the same directory repeatedly when not using Tune)
    if not checkpoints:
        # Check if checkpoint_dir itself is a valid checkpoint
        rllib_checkpoint = checkpoint_dir / "rllib_checkpoint.json"
        if rllib_checkpoint.exists():
            # Single checkpoint at the final iteration - we can only evaluate this
            print(f"  Note: Found single checkpoint (no iteration subdirectories)")
            print(f"  This is the final checkpoint from training")
            # Return checkpoint for final iteration only
            return [(10, checkpoint_dir)]  # Assuming 10 is the last iteration

    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def evaluate_checkpoint(checkpoint_path, iteration, num_episodes, results_dir, verbose=True):
    """Evaluate a single checkpoint."""
    if verbose:
        print(f"\n  Evaluating checkpoint at iteration {iteration}...")

    # Build evaluation command
    output_path = results_dir / f"eval_{iteration}.json"

    cmd = [
        sys.executable, "-m", "parallel_risk.evaluation.evaluate_agent",
        "--checkpoint", str(checkpoint_path),
        "--opponent", "random",
        "--num-episodes", str(num_episodes),
        "--output", str(output_path),
    ]

    if verbose:
        cmd.append("--verbose")

    # Run evaluation
    try:
        subprocess.run(cmd, check=True, capture_output=not verbose)

        # Load results
        with open(output_path, 'r') as f:
            results = json.load(f)

        if verbose:
            print(f"    Win rate: {results['win_rate']:.2%}")

        return results

    except subprocess.CalledProcessError as e:
        print(f"    ✗ Evaluation failed: {e}")
        return None


def run_evaluations(checkpoint_dir, eval_interval, num_episodes, results_dir, verbose=True):
    """Evaluate checkpoints at specified intervals."""
    print("\n" + "="*70)
    print("STEP 3: Evaluating Checkpoints")
    print("="*70)

    # Discover checkpoints
    checkpoints = discover_checkpoints(checkpoint_dir)

    if not checkpoints:
        print("✗ No checkpoints found!")
        return {}

    print(f"Found {len(checkpoints)} checkpoints")

    # Filter checkpoints by eval_interval
    eval_checkpoints = [
        (it, path) for it, path in checkpoints
        if it % eval_interval == 0 or it == checkpoints[-1][0]
    ]

    print(f"Will evaluate {len(eval_checkpoints)} checkpoints "
          f"(every {eval_interval} iterations)")

    # Evaluate each checkpoint
    all_results = {}

    for iteration, checkpoint_path in eval_checkpoints:
        results = evaluate_checkpoint(
            checkpoint_path, iteration, num_episodes, results_dir, verbose
        )
        if results:
            all_results[iteration] = results

    print(f"\n✓ Completed {len(all_results)} evaluations")
    return all_results


def generate_summary(all_results, results_dir, verbose=True):
    """Generate final summary and visualization."""
    print("\n" + "="*70)
    print("STEP 4: Generating Summary and Plots")
    print("="*70)

    # Save combined results
    combined_path = results_dir / "evaluation_results.json"
    combined_data = {"evaluations": {str(k): v for k, v in all_results.items()}}

    with open(combined_path, 'w') as f:
        json.dump(combined_data, f, indent=2)

    print(f"✓ Saved combined results to {combined_path}")

    # Generate plots
    try:
        from parallel_risk.evaluation.visualize import plot_all
        plot_all(str(combined_path), output_dir=str(results_dir))
        print("✓ Generated plots")
    except ImportError as e:
        print(f"⚠ Could not generate plots (matplotlib not available): {e}")
    except Exception as e:
        print(f"⚠ Plot generation failed: {e}")

    # Generate text summary
    summary_path = results_dir / "final_summary.txt"

    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("PARALLEL RISK LEARNING VALIDATION EXPERIMENT\n")
        f.write("="*70 + "\n\n")

        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Sort by iteration
        iterations = sorted(all_results.keys())

        f.write(f"Evaluated {len(iterations)} checkpoints:\n\n")

        f.write("Iteration | Win Rate | Losses | Draws | Avg Ep Length\n")
        f.write("-"*70 + "\n")

        for it in iterations:
            r = all_results[it]
            f.write(f"{it:9d} | {r['win_rate']:7.2%} | "
                   f"{r['losses']:6d} | {r['draws']:5d} | "
                   f"{r['avg_episode_length']:13.1f}\n")

        f.write("\n" + "="*70 + "\n")
        f.write("LEARNING VALIDATION RESULTS\n")
        f.write("="*70 + "\n\n")

        # Check success criteria
        final_iteration = max(iterations)
        final_win_rate = all_results[final_iteration]['win_rate']

        f.write(f"Final win rate (iteration {final_iteration}): {final_win_rate:.2%}\n\n")

        if final_win_rate >= 0.7:
            f.write("✓ SUCCESS: Agent achieved >70% win rate vs. random\n")
            f.write("  Learning has been validated!\n")
        elif final_win_rate >= 0.6:
            f.write("⚠ PARTIAL SUCCESS: Agent achieved >60% win rate\n")
            f.write("  Learning is happening but may need more training\n")
        else:
            f.write("✗ FAILURE: Agent did not achieve >60% win rate\n")
            f.write("  Learning may not be working - investigate environment/hyperparams\n")

        f.write("\n" + "="*70 + "\n\n")

        # Print trajectory
        f.write("Win Rate Progression:\n")
        for it in iterations:
            wr = all_results[it]['win_rate']
            bar_length = int(wr * 50)
            bar = "█" * bar_length + "░" * (50 - bar_length)
            f.write(f"  Iter {it:4d}: {bar} {wr:.2%}\n")

    print(f"✓ Saved text summary to {summary_path}")

    # Print summary to console
    if verbose:
        print("\n" + "="*70)
        print("FINAL RESULTS")
        print("="*70)

        final_iteration = max(iterations)
        final_win_rate = all_results[final_iteration]['win_rate']

        print(f"\nFinal win rate (iteration {final_iteration}): {final_win_rate:.2%}")

        if final_win_rate >= 0.7:
            print("\n✓ SUCCESS: Learning validated!")
        elif final_win_rate >= 0.6:
            print("\n⚠ PARTIAL SUCCESS: Some learning observed")
        else:
            print("\n✗ FAILURE: Insufficient learning")

        print(f"\nAll results saved to: {results_dir}")
        print("="*70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run learning validation experiment for Parallel Risk"
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=500,
        help="Number of training iterations"
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
        help="Episodes per evaluation"
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
        default="parallel_risk/training/configs/ppo_baseline.yaml",
        help="Training config file"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="experiments/validate_learning_results",
        help="Directory for results"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/validation_run",
        help="Directory for training checkpoints"
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
    config_path = Path(args.config)

    # Create directories
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("PARALLEL RISK LEARNING VALIDATION EXPERIMENT")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Training iterations:  {args.num_iterations}")
    print(f"  Eval interval:        {args.eval_interval}")
    print(f"  Eval episodes:        {args.num_eval_episodes}")
    print(f"  Training workers:     {args.num_workers}")
    print(f"  Config file:          {config_path}")
    print(f"  Checkpoint dir:       {checkpoint_dir}")
    print(f"  Results dir:          {results_dir}")

    # Step 1: Baseline evaluation
    run_baseline_evaluation(results_dir, verbose=args.verbose)

    # Step 2: Training
    if not args.skip_training:
        success = run_training(
            config_path,
            args.num_iterations,
            checkpoint_dir,
            num_workers=args.num_workers,
            verbose=args.verbose
        )

        if not success:
            print("\n✗ Experiment failed during training")
            sys.exit(1)
    else:
        print("\n⚠ Skipping training (--skip-training flag)")

    # Step 3: Evaluate checkpoints
    all_results = run_evaluations(
        checkpoint_dir,
        args.eval_interval,
        args.num_eval_episodes,
        results_dir,
        verbose=args.verbose
    )

    if not all_results:
        print("\n✗ No evaluation results obtained")
        sys.exit(1)

    # Step 4: Generate summary
    generate_summary(all_results, results_dir, verbose=args.verbose)

    print("\n✓ Validation experiment complete!")


if __name__ == "__main__":
    main()
