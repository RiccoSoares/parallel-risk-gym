"""
Validation experiment for GNN-based Parallel Risk agent (Phase 2).

This script validates that the GNN+PPO training pipeline learns effectively:
1. Train GNN policy with PPO for 200 iterations
2. Evaluate checkpoints every 25 iterations vs random agent
3. Generate plots using existing visualization code (directly comparable to Phase 1)
4. Verify >70% win rate is achieved

Usage:
    python experiments/validate_gnn_learning.py
    python experiments/validate_gnn_learning.py --num-iterations 500 --eval-interval 50
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


def run_training(config_path, num_iterations, checkpoint_dir, verbose=True, parallel=False, num_workers=4):
    """Run GNN+PPO training."""
    print("\n" + "="*70)
    print("STEP 1: Training GNN+PPO Agent")
    print("="*70)

    if verbose:
        print(f"Training for {num_iterations} iterations")
        print(f"Checkpoints will be saved to: {checkpoint_dir}")
        if parallel:
            print(f"Using parallel training with {num_workers} workers")
        print("This will take approximately 30-60 minutes...\n")

    # Import training components
    if parallel:
        from parallel_risk.training.torchrl.train_parallel import PPOTrainerParallel, load_config
    else:
        from parallel_risk.training.torchrl.train import PPOTrainer, load_config

    # Load config
    config = load_config(config_path)

    # Override checkpoint directory
    config['checkpoint_dir'] = str(checkpoint_dir)
    if 'logging' not in config:
        config['logging'] = {}
    config['logging']['checkpoint_dir'] = str(checkpoint_dir)

    # Create trainer
    if parallel:
        trainer = PPOTrainerParallel(config, num_workers=num_workers)
    else:
        trainer = PPOTrainer(config)

    # Train
    try:
        if parallel:
            trainer.train(num_iterations, checkpoint_interval=10)
            trainer.vec_env.close()
        else:
            trainer.train(num_iterations)
        print("\n✓ Training completed successfully")
        return True
    except Exception as e:
        print(f"\n✗ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def discover_checkpoints(checkpoint_dir):
    """Find all checkpoint files."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return []

    checkpoints = []

    # Look for checkpoint_NNNNNN.pt files
    for item in checkpoint_dir.glob("checkpoint_*.pt"):
        # Extract iteration number
        try:
            iteration = int(item.stem.split("_")[1])
            checkpoints.append((iteration, item))
        except (IndexError, ValueError):
            continue

    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def evaluate_checkpoint(checkpoint_path, iteration, num_episodes, verbose=True):
    """Evaluate a GNN checkpoint against random agent."""
    if verbose:
        print(f"\n  Evaluating checkpoint at iteration {iteration}...")

    from parallel_risk import ParallelRiskEnv
    from parallel_risk.training.torchrl.graph_wrapper import GraphObservationWrapper
    from parallel_risk.models.gnn_gcn import GCNPolicy
    from parallel_risk.models.action_decoder import ActionDecoder
    from parallel_risk.agents.random_agent import RandomAgent
    from torch_geometric.data import Batch

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']

    # Create environment
    env = ParallelRiskEnv(
        map_name=config['env']['map_name'],
        max_turns=config['env'].get('max_turns', 100),
        seed=None,
        reward_shaping_config=None  # Sparse rewards for evaluation
    )
    wrapped_env = GraphObservationWrapper(env, device='cpu')

    # Create GNN policy
    obs_space = wrapped_env.observation_space
    policy = GCNPolicy(
        node_features_dim=obs_space['node_features_dim'],
        global_features_dim=obs_space['global_features_dim'],
        hidden_dim=config['model'].get('hidden_dim', 128),
        num_layers=config['model'].get('num_layers', 3),
        action_budget=config['env'].get('action_budget', 5),
        max_troops=20,
        dropout=config['model'].get('dropout', 0.1)
    )
    policy.load_state_dict(checkpoint['policy_state_dict'])
    policy.eval()

    # Create action decoder
    action_decoder = ActionDecoder(
        action_budget=config['env'].get('action_budget', 5),
        max_troops=20
    )

    # Create random opponent
    n_territories = env.map_config.n_territories
    random_agent = RandomAgent(
        n_territories=n_territories,
        action_budget=config['env'].get('action_budget', 5),
        mode='pettingzoo'
    )

    # Run evaluation episodes
    wins = 0
    losses = 0
    draws = 0
    episode_lengths = []

    for episode in range(num_episodes):
        obs, _ = wrapped_env.reset()
        done = False
        episode_length = 0

        while not done:
            # GNN agent (agent_0) action
            graph_0 = obs.get('agent_0')
            if graph_0 is not None:
                with torch.no_grad():
                    batched_graph = Batch.from_data_list([graph_0])
                    action_logits, _, _ = policy(batched_graph)
                    actions_tensor, _ = action_decoder.decode_actions(
                        action_logits, batched_graph.batch, deterministic=True
                    )
                    action_array = actions_tensor[0].cpu().numpy()
                    action_0 = {
                        'num_actions': config['env'].get('action_budget', 5),
                        'actions': np.vstack([action_array, np.zeros((10 - config['env'].get('action_budget', 5), 3))])
                    }
            else:
                action_0 = None

            # Random agent (agent_1) action
            if 'agent_1' in obs:
                action_1 = random_agent.get_action()
            else:
                action_1 = None

            # Step environment
            actions = {}
            if action_0 is not None:
                actions['agent_0'] = action_0
            if action_1 is not None:
                actions['agent_1'] = action_1

            obs, rewards, terminateds, truncateds, infos = wrapped_env.step(actions)

            done = terminateds.get('__all__', False) or truncateds.get('__all__', False)
            episode_length += 1

        # Count results
        reward_0 = rewards.get('agent_0', 0)
        reward_1 = rewards.get('agent_1', 0)

        if reward_0 > reward_1:
            wins += 1
        elif reward_0 < reward_1:
            losses += 1
        else:
            draws += 1

        episode_lengths.append(episode_length)

        if verbose and (episode + 1) % 20 == 0:
            current_win_rate = wins / (episode + 1)
            print(f"    Episode {episode + 1}/{num_episodes} - Win rate: {current_win_rate:.2%}")

    total = wins + losses + draws
    win_rate = wins / total if total > 0 else 0.0

    results = {
        "win_rate": win_rate,
        "loss_rate": losses / total if total > 0 else 0.0,
        "draw_rate": draws / total if total > 0 else 0.0,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "total_episodes": total,
        "avg_episode_length": float(np.mean(episode_lengths)),
        "std_episode_length": float(np.std(episode_lengths)),
        "episode_lengths": episode_lengths,
    }

    if verbose:
        print(f"    Win rate: {win_rate:.2%} | Avg length: {results['avg_episode_length']:.1f}")

    return results


def run_evaluations(checkpoint_dir, eval_interval, num_episodes, results_dir, verbose=True):
    """Evaluate checkpoints at specified intervals."""
    print("\n" + "="*70)
    print("STEP 2: Evaluating Checkpoints")
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
            checkpoint_path, iteration, num_episodes, verbose
        )
        if results:
            all_results[iteration] = results

    print(f"\n✓ Completed {len(all_results)} evaluations")
    return all_results


def generate_summary(all_results, results_dir, verbose=True):
    """Generate final summary and visualization using existing plotting code."""
    print("\n" + "="*70)
    print("STEP 3: Generating Summary and Plots")
    print("="*70)

    # Save combined results in format compatible with existing visualize.py
    combined_path = results_dir / "evaluation_results.json"
    combined_data = {"evaluations": {str(k): v for k, v in all_results.items()}}

    with open(combined_path, 'w') as f:
        json.dump(combined_data, f, indent=2)

    print(f"✓ Saved combined results to {combined_path}")

    # Generate plots using existing visualization code
    try:
        from parallel_risk.evaluation.visualize import plot_all
        plot_all(str(combined_path), output_dir=str(results_dir))
        print("✓ Generated plots (win_rate_curve.png, episode_length_curve.png, reward_distribution.png)")
    except ImportError as e:
        print(f"⚠ Could not generate plots (matplotlib not available): {e}")
    except Exception as e:
        print(f"⚠ Plot generation failed: {e}")
        import traceback
        traceback.print_exc()

    # Generate text summary
    summary_path = results_dir / "final_summary.txt"

    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("PHASE 2 GNN LEARNING VALIDATION EXPERIMENT\n")
        f.write("="*70 + "\n\n")

        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Architecture: GNN (GCN) + PPO\n")
        f.write(f"Map: simple_6\n")
        f.write(f"Reward shaping: Sparse (terminal only)\n\n")

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
            f.write("✓ SUCCESS: GNN agent achieved >70% win rate vs. random\n")
            f.write("  Learning has been validated!\n")
        elif final_win_rate >= 0.6:
            f.write("⚠ PARTIAL SUCCESS: GNN agent achieved >60% win rate\n")
            f.write("  Learning is happening but may need more training\n")
        else:
            f.write("✗ FAILURE: GNN agent did not achieve >60% win rate\n")
            f.write("  Learning may not be working - investigate hyperparameters\n")

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
            print("\n✓ SUCCESS: GNN learning validated!")
        elif final_win_rate >= 0.6:
            print("\n⚠ PARTIAL SUCCESS: Some learning observed")
        else:
            print("\n✗ FAILURE: Insufficient learning")

        print(f"\nAll results saved to: {results_dir}")
        print("="*70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate GNN+PPO learning for Parallel Risk (Phase 2)"
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=200,
        help="Number of training iterations (default: 200)"
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=25,
        help="Evaluate every N iterations (default: 25)"
    )
    parser.add_argument(
        "--num-eval-episodes",
        type=int,
        default=100,
        help="Episodes per evaluation (default: 100)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="parallel_risk/training/torchrl/configs/gnn_gcn.yaml",
        help="Training config file"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="experiments/gnn_validation_results",
        help="Directory for results"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/gnn_validation",
        help="Directory for training checkpoints"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training, only evaluate existing checkpoints"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Use parallel training with vectorized environments"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4, only used with --parallel)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Alias for --results-dir (for backwards compatibility)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print detailed output"
    )

    args = parser.parse_args()

    # Handle output-dir alias
    if args.output_dir:
        args.results_dir = args.output_dir

    # Convert to paths
    results_dir = Path(args.results_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    config_path = Path(args.config)

    # Create directories
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("PHASE 2: GNN LEARNING VALIDATION EXPERIMENT")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Training iterations:  {args.num_iterations}")
    print(f"  Eval interval:        {args.eval_interval}")
    print(f"  Eval episodes:        {args.num_eval_episodes}")
    print(f"  Config file:          {config_path}")
    print(f"  Checkpoint dir:       {checkpoint_dir}")
    print(f"  Results dir:          {results_dir}")
    print(f"  Architecture:         GNN (GCN) + PPO")
    print(f"  Map:                  simple_6")
    if args.parallel:
        print(f"  Training mode:        Parallel ({args.num_workers} workers)")
    else:
        print(f"  Training mode:        Single-threaded")

    # Step 1: Training
    if not args.skip_training:
        success = run_training(
            config_path,
            args.num_iterations,
            checkpoint_dir,
            verbose=args.verbose,
            parallel=args.parallel,
            num_workers=args.num_workers
        )

        if not success:
            print("\n✗ Experiment failed during training")
            sys.exit(1)
    else:
        print("\n⚠ Skipping training (--skip-training flag)")

    # Step 2: Evaluate checkpoints
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

    # Step 3: Generate summary
    generate_summary(all_results, results_dir, verbose=args.verbose)

    print("\n✓ GNN validation experiment complete!")


if __name__ == "__main__":
    main()
