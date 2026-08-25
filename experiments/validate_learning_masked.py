"""
Learning validation experiment for Parallel Risk (Phase 1 with action masking).

This script validates Phase 1 (RLlib/MLP) training using a masked random opponent
that applies the same autoregressive action masking as the Phase 2 GNN agent.
This provides a fair comparison baseline.

Steps:
1. Baseline evaluation (masked random vs. masked random)
2. Training with PPO (saving checkpoints at specified interval)
3. Periodic evaluation of checkpoints vs. masked random
4. Results visualization and summary

Usage:
    # Quick test (comparable to Phase 2 command)
    PYTHONPATH=. python experiments/validate_learning_masked.py \\
        --num-iterations 50 \\
        --eval-interval 1 \\
        --num-eval-episodes 50 \\
        --output-dir experiments/phase1_masked_results \\
        --checkpoint-dir checkpoints/phase1_masked \\
        --num-workers 8

    # Full run
    PYTHONPATH=. python experiments/validate_learning_masked.py \\
        --num-iterations 500 \\
        --eval-interval 10 \\
        --num-workers 4
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


def run_baseline_evaluation(results_dir, num_episodes=100, verbose=True):
    """Evaluate masked random vs. masked random as baseline."""
    print("\n" + "="*70)
    print("STEP 1: Baseline Evaluation (Masked Random vs. Masked Random)")
    print("="*70)

    if verbose:
        print(f"Running {num_episodes} episodes with two masked random agents...")
        print("This establishes empirical baseline performance.\n")

    from parallel_risk import ParallelRiskEnv
    from parallel_risk.agents.masked_random_agent import MaskedRandomAgentRLlib

    # Create environment
    env = ParallelRiskEnv(
        map_name="simple_6",
        max_turns=100,
        seed=None,
    )

    # Create two masked random agents
    masked_agent_0 = MaskedRandomAgentRLlib.from_env(env, action_budget=5)
    masked_agent_1 = MaskedRandomAgentRLlib.from_env(env, action_budget=5)

    # Run episodes
    wins = 0
    losses = 0
    draws = 0
    episode_lengths = []

    for episode in range(num_episodes):
        obs, _ = env.reset(seed=42 + episode)
        done = False
        episode_length = 0

        while not done:
            actions = {}
            if "agent_0" in obs:
                actions["agent_0"] = masked_agent_0.get_action_raw(obs["agent_0"])
            if "agent_1" in obs:
                actions["agent_1"] = masked_agent_1.get_action_raw(obs["agent_1"])

            obs, rewards, terminateds, truncateds, _ = env.step(actions)

            done = terminateds.get("__all__", False) or truncateds.get("__all__", False)
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
        "avg_episode_length": float(np.mean(episode_lengths)),
        "std_episode_length": float(np.std(episode_lengths)),
        "note": "Empirical baseline - two masked random agents with autoregressive masking",
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


def run_training(config_path, num_iterations, checkpoint_dir, checkpoint_interval=1,
                 num_workers=4, verbose=True):
    """Run PPO training with checkpoints at specified interval."""
    print("\n" + "="*70)
    print("STEP 2: Training PPO Agent")
    print("="*70)

    if verbose:
        print(f"Training for {num_iterations} iterations with {num_workers} workers")
        print(f"Checkpoints will be saved every {checkpoint_interval} iteration(s)")
        print(f"Checkpoint directory: {checkpoint_dir}\n")

    import os
    import ray
    import yaml
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.models import ModelCatalog

    from parallel_risk.training.rllib.action_mask_wrapper import ActionMaskRLlibEnv
    from parallel_risk.training.rllib.masked_model import SimpleMaskedModel
    from parallel_risk.env.reward_shaping import create_sparse_config

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)

    # Register custom model
    ModelCatalog.register_custom_model("masked_model", SimpleMaskedModel)

    # Create environment config
    env_config = {
        "map_name": config["env"]["map_name"],
        "max_turns": config["env"]["max_turns"],
        "action_budget": config["env"]["action_budget"],
        "reward_shaping_config": None,  # Sparse rewards
        "max_troops": 20,
    }

    # Get dimensions for model config
    n_territories = 6  # simple_6 map
    n_regions = 3  # simple_6 has 3 regions
    action_budget = env_config["action_budget"]
    max_troops = env_config["max_troops"]

    # Calculate observation size explicitly
    obs_size = (
        n_territories +  # ownership
        n_territories +  # troops
        n_territories * n_territories +  # adjacency
        1 +  # income
        1 +  # turn
        n_regions  # region control
    )

    # Policy mapping function
    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        return "main_policy"

    # Build PPO config
    ppo_cfg = config["ppo"]
    model_cfg = config["model"]
    training_cfg = config["training"]

    algo_config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(
            env=ActionMaskRLlibEnv,
            env_config=env_config,
        )
        .framework("torch")
        .resources(num_gpus=0)
        .env_runners(
            num_env_runners=num_workers,
            num_envs_per_env_runner=training_cfg.get("num_envs_per_worker", 1),
        )
        .training(
            train_batch_size=training_cfg["train_batch_size"],
            minibatch_size=training_cfg["sgd_minibatch_size"],
            num_sgd_iter=training_cfg["num_sgd_iter"],
            gamma=ppo_cfg["gamma"],
            lambda_=ppo_cfg["lambda"],
            clip_param=ppo_cfg["clip_param"],
            vf_clip_param=ppo_cfg["vf_clip_param"],
            entropy_coeff=ppo_cfg["entropy_coeff"],
            lr=ppo_cfg["lr"],
            model={
                "custom_model": "masked_model",
                "custom_model_config": {
                    "n_territories": n_territories,
                    "action_budget": action_budget,
                    "max_troops": max_troops,
                    "obs_size": obs_size,
                },
                "fcnet_hiddens": model_cfg["fcnet_hiddens"],
                "fcnet_activation": model_cfg["fcnet_activation"],
                "vf_share_layers": model_cfg["vf_share_layers"],
            },
        )
        .multi_agent(
            policies={"main_policy"},
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["main_policy"],
        )
    )

    # Build algorithm
    algo = algo_config.build()

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training loop
    try:
        for iteration in range(1, num_iterations + 1):
            result = algo.train()

            # Extract metrics
            env_runners = result.get('env_runners', {})
            episode_reward = env_runners.get('episode_reward_mean', result.get('episode_reward_mean', 'N/A'))
            episode_length = env_runners.get('episode_len_mean', result.get('episode_len_mean', 'N/A'))

            if verbose:
                reward_str = f"{episode_reward:.3f}" if isinstance(episode_reward, (int, float)) else str(episode_reward)
                length_str = f"{episode_length:.1f}" if isinstance(episode_length, (int, float)) else str(episode_length)
                print(f"  Iteration {iteration}/{num_iterations} - Reward: {reward_str}, Length: {length_str}")

            # Save checkpoint at interval
            if iteration % checkpoint_interval == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{iteration:06d}")
                algo.save(checkpoint_path)
                if verbose:
                    print(f"    💾 Saved checkpoint: {checkpoint_path}")

        print("\n✓ Training completed successfully")
        return True

    except Exception as e:
        print(f"\n✗ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        algo.stop()


def discover_checkpoints(checkpoint_dir):
    """Find all checkpoint directories."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return []

    checkpoints = []

    # Check for numbered checkpoint subdirectories (checkpoint_000010, etc.)
    for item in checkpoint_dir.iterdir():
        if item.is_dir() and item.name.startswith("checkpoint_"):
            # Extract iteration number
            try:
                iteration = int(item.name.split("_")[1])
                checkpoints.append((iteration, item))
            except (IndexError, ValueError):
                continue

    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def evaluate_checkpoint(checkpoint_path, iteration, num_episodes, verbose=True):
    """Evaluate a single checkpoint against masked random agent."""
    if verbose:
        print(f"\n  Evaluating checkpoint at iteration {iteration}...")

    import ray
    from ray.rllib.algorithms.ppo import PPO
    from ray.rllib.models import ModelCatalog

    from parallel_risk import ParallelRiskEnv
    from parallel_risk.agents.masked_random_agent import MaskedRandomAgentRLlib
    from parallel_risk.training.rllib.masked_model import SimpleMaskedModel

    # Initialize Ray if needed
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)

    # Register custom model (needed to load checkpoint)
    ModelCatalog.register_custom_model("masked_model", SimpleMaskedModel)

    # Load checkpoint - need absolute path for RLlib
    checkpoint_path_abs = Path(checkpoint_path).resolve()
    try:
        algo = PPO.from_checkpoint(str(checkpoint_path_abs))
    except Exception as e:
        print(f"    ✗ Failed to load checkpoint: {e}")
        return None

    # Create environment for evaluation
    env = ParallelRiskEnv(
        map_name="simple_6",
        max_turns=100,
        seed=None,
    )

    # Create masked random opponent
    masked_opponent = MaskedRandomAgentRLlib.from_env(env, action_budget=5)

    # Run evaluation episodes
    wins = 0
    losses = 0
    draws = 0
    episode_lengths = []

    for episode in range(num_episodes):
        obs, _ = env.reset(seed=42 + episode)
        done = False
        episode_length = 0

        while not done:
            actions = {}

            # Trained agent (agent_0) - use autoregressive sampling for valid actions
            if "agent_0" in obs:
                actions["agent_0"] = _sample_autoregressive_action(
                    algo, obs["agent_0"], policy_id="main_policy"
                )

            # Masked random opponent (agent_1)
            if "agent_1" in obs:
                actions["agent_1"] = masked_opponent.get_action_raw(obs["agent_1"])

            obs, rewards, terminateds, truncateds, _ = env.step(actions)

            done = terminateds.get("__all__", False) or truncateds.get("__all__", False)
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
            print(f"    Episode {episode + 1}/{num_episodes} - Win rate: {current_win_rate:.2%}")

    # Cleanup
    algo.stop()

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


def _flatten_observation(obs_dict):
    """Flatten observation dict to vector for RLlib."""
    return np.concatenate([
        obs_dict['territory_ownership'].astype(np.float32),
        obs_dict['territory_troops'].astype(np.float32),
        obs_dict['adjacency_matrix'].flatten().astype(np.float32),
        obs_dict['available_income'].astype(np.float32),
        obs_dict['turn_number'].astype(np.float32),
        obs_dict['region_control'].astype(np.float32),
    ])


def _create_masked_observation_for_eval(obs_dict, n_territories=6, max_troops=20):
    """Create observation dict with action masks for evaluation.

    This replicates the observation format from ActionMaskRLlibEnv so that
    checkpoints trained with action masking can be evaluated properly.
    """
    # Flatten observation
    flat_obs = _flatten_observation(obs_dict)

    # Compute conservative action masks (same as ActionMaskRLlibEnv)
    ownership = obs_dict['territory_ownership']
    troops = obs_dict['territory_troops']
    adjacency = obs_dict['adjacency_matrix']
    income = obs_dict['available_income'][0]

    # Source mask: owned territories
    source_mask = (ownership == 1).astype(np.float32)

    # Dest mask (conservative): owned territories + all neighbors of owned
    dest_mask = source_mask.copy()
    for i in range(n_territories):
        if source_mask[i]:
            dest_mask = np.maximum(dest_mask, adjacency[i].astype(np.float32))

    # Troops mask (conservative)
    owned_troops = troops[ownership == 1]
    if len(owned_troops) > 0:
        max_transferable = int(owned_troops.max()) - 1
    else:
        max_transferable = 0
    max_deployable = int(income)
    max_troops_available = max(max_transferable, max_deployable)

    troops_mask = np.zeros(max_troops, dtype=np.float32)
    if max_troops_available > 0:
        troops_mask[1:min(max_troops_available + 1, max_troops)] = 1.0

    return {
        "observations": flat_obs,
        "action_mask": {
            "source_mask": source_mask,
            "dest_mask": dest_mask,
            "troops_mask": troops_mask,
        },
        "raw_data": {
            "ownership": ownership.astype(np.float32),
            "troops": troops.astype(np.float32),
            "adjacency": adjacency.astype(np.float32),
            "income": np.array([income], dtype=np.float32),
        },
    }


def _tuple_to_env_action(action_tuple, action_budget=5):
    """Convert RLlib tuple action to environment format."""
    actions_array = np.zeros((10, 3), dtype=np.int32)
    for i, action in enumerate(action_tuple[:action_budget]):
        actions_array[i] = list(action)
    return {
        'num_actions': action_budget,
        'actions': actions_array
    }


def _sample_autoregressive_action(algo, obs_dict, policy_id="main_policy",
                                   n_territories=6, action_budget=5, max_troops=20):
    """
    Sample actions from the policy using autoregressive masking.

    This ensures valid actions by:
    1. Getting raw logits from the model
    2. Sampling source with ownership mask
    3. Sampling dest conditioned on chosen source
    4. Sampling troops conditioned on source and dest

    Args:
        algo: RLlib algorithm with loaded checkpoint
        obs_dict: Raw observation dict from environment
        policy_id: Policy to use
        n_territories: Number of territories
        action_budget: Number of actions per turn
        max_troops: Maximum troops

    Returns:
        Action dict in environment format
    """
    import torch

    # Get policy and model
    policy = algo.get_policy(policy_id)
    model = policy.model

    # Create base observation
    flat_obs = _flatten_observation(obs_dict)

    # Get observation data for masking
    ownership = obs_dict['territory_ownership']
    troops = obs_dict['territory_troops']
    adjacency = obs_dict['adjacency_matrix']
    income = obs_dict['available_income'][0]

    # Compute source mask (owned territories)
    source_mask = (ownership == 1).astype(np.float32)

    # Convert to tensor
    obs_tensor = torch.tensor(flat_obs, dtype=torch.float32).unsqueeze(0)

    # Get features and raw logits from model
    # SimpleMaskedModel uses a single policy_head that outputs all logits concatenated
    with torch.no_grad():
        features = model.feature_extractor(obs_tensor)
        raw_logits = model.policy_head(features).squeeze(0).numpy()

    # Parse logits for one action slot
    # Layout: [source(n_territories), dest(n_territories), troops(max_troops)] * action_budget
    single_action_size = n_territories + n_territories + max_troops

    actions = []

    for action_idx in range(action_budget):
        # Extract logits for this action slot
        offset = action_idx * single_action_size
        source_logits = raw_logits[offset:offset + n_territories]
        dest_logits = raw_logits[offset + n_territories:offset + 2*n_territories]
        troops_logits = raw_logits[offset + 2*n_territories:offset + single_action_size]

        # Step 1: Sample source with ownership mask
        masked_source = source_logits + (1 - source_mask) * (-1e10)
        source_probs = _softmax(masked_source)
        source = np.random.choice(n_territories, p=source_probs)

        # Step 2: Sample dest conditioned on source
        # Valid: self (deploy) or adjacent (transfer/attack)
        dest_mask = np.zeros(n_territories, dtype=np.float32)
        dest_mask[source] = 1.0  # Can always target self
        dest_mask = np.maximum(dest_mask, adjacency[source].astype(np.float32))  # Add adjacent

        masked_dest = dest_logits + (1 - dest_mask) * (-1e10)
        dest_probs = _softmax(masked_dest)
        dest = np.random.choice(n_territories, p=dest_probs)

        # Step 3: Sample troops conditioned on source and dest
        is_self = (source == dest)

        if is_self:
            # Deploy: limited by income
            max_troop = min(int(income), max_troops - 1)
        else:
            # Transfer/attack: limited by troops at source - 1
            max_troop = min(int(troops[source]) - 1, max_troops - 1)

        troops_mask_arr = np.zeros(max_troops, dtype=np.float32)
        if max_troop > 0:
            troops_mask_arr[1:max_troop + 1] = 1.0
        else:
            troops_mask_arr[1] = 1.0  # At least allow 1 troop

        masked_troops = troops_logits + (1 - troops_mask_arr) * (-1e10)
        troop_probs = _softmax(masked_troops)
        troop = np.random.choice(max_troops, p=troop_probs)

        actions.append([source, dest, troop])

    # Convert to environment format
    actions_array = np.zeros((10, 3), dtype=np.int32)
    for i, action in enumerate(actions):
        actions_array[i] = action

    return {
        'num_actions': action_budget,
        'actions': actions_array
    }


def _softmax(x):
    """Compute softmax with numerical stability."""
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / (exp_x.sum() + 1e-10)


def run_evaluations(checkpoint_dir, eval_interval, num_episodes, results_dir, verbose=True):
    """Evaluate checkpoints at specified intervals."""
    print("\n" + "="*70)
    print("STEP 3: Evaluating Checkpoints vs. Masked Random")
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

            # Save individual result
            output_path = results_dir / f"eval_{iteration}.json"
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)

    print(f"\n✓ Completed {len(all_results)} evaluations")
    return all_results


def generate_summary(all_results, results_dir, verbose=True):
    """Generate final summary and visualization."""
    print("\n" + "="*70)
    print("STEP 4: Generating Summary and Plots")
    print("="*70)

    # Save combined results (same format as Phase 2 for compatibility)
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
        f.write("PHASE 1 LEARNING VALIDATION (MASKED RANDOM OPPONENT)\n")
        f.write("="*70 + "\n\n")

        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Architecture: MLP + PPO (RLlib)\n")
        f.write(f"Opponent: Masked Random (autoregressive masking)\n")
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
            f.write("✓ SUCCESS: Agent achieved >70% win rate vs. masked random\n")
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
        description="Run learning validation experiment for Phase 1 with masked random opponent"
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=50,
        help="Number of training iterations (default: 50)"
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=1,
        help="Evaluate every N iterations (default: 1)"
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=None,
        help="Save checkpoint every N iterations (default: same as --eval-interval)"
    )
    parser.add_argument(
        "--num-eval-episodes",
        type=int,
        default=50,
        help="Episodes per evaluation (default: 50)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of training workers (default: 4)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="parallel_risk/training/rllib/configs/ppo_baseline.yaml",
        help="Training config file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/phase1_masked_results",
        help="Directory for results"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/phase1_masked",
        help="Directory for training checkpoints"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training, only evaluate existing checkpoints"
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline evaluation"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print detailed output"
    )

    args = parser.parse_args()

    # Default checkpoint_interval to eval_interval
    if args.checkpoint_interval is None:
        args.checkpoint_interval = args.eval_interval

    # Convert to paths
    results_dir = Path(args.output_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    config_path = Path(args.config)

    # Create directories
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("PHASE 1 LEARNING VALIDATION (MASKED RANDOM OPPONENT)")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Training iterations:  {args.num_iterations}")
    print(f"  Checkpoint interval:  {args.checkpoint_interval}")
    print(f"  Eval interval:        {args.eval_interval}")
    print(f"  Eval episodes:        {args.num_eval_episodes}")
    print(f"  Training workers:     {args.num_workers}")
    print(f"  Config file:          {config_path}")
    print(f"  Checkpoint dir:       {checkpoint_dir}")
    print(f"  Output dir:           {results_dir}")
    print(f"  Architecture:         MLP + PPO (RLlib)")
    print(f"  Opponent:             Masked Random (autoregressive)")

    # Step 1: Baseline evaluation
    if not args.skip_baseline:
        run_baseline_evaluation(results_dir, num_episodes=args.num_eval_episodes, verbose=args.verbose)
    else:
        print("\n⚠ Skipping baseline evaluation (--skip-baseline flag)")

    # Step 2: Training
    if not args.skip_training:
        success = run_training(
            config_path,
            args.num_iterations,
            checkpoint_dir,
            checkpoint_interval=args.checkpoint_interval,
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

    print("\n✓ Phase 1 masked validation experiment complete!")


if __name__ == "__main__":
    main()
