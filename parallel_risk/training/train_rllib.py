"""
Training script for Parallel Risk with RLlib.

This script handles:
- Loading configuration from YAML
- Setting up the RLlib environment and algorithm
- Configuring self-play
- Running training loop with checkpointing
- Logging metrics

Usage:
    python -m parallel_risk.training.train_rllib --config configs/ppo_baseline.yaml
    python -m parallel_risk.training.train_rllib --config configs/ppo_baseline.yaml --num-workers 8
"""

import argparse
import os
import yaml
from pathlib import Path
from typing import Dict, Any

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.tune.logger import pretty_print

from parallel_risk.training.rllib_wrapper import make_rllib_env, RLlibParallelRiskEnv
from parallel_risk.env.reward_shaping import (
    create_dense_config,
    create_sparse_config,
    create_territorial_config,
    create_aggressive_config,
)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_reward_shaping_config(shaping_type: str):
    """Get reward shaping config from string name."""
    configs = {
        "sparse": None,  # No shaping
        "dense": create_dense_config(),
        "territorial": create_territorial_config(),
        "aggressive": create_aggressive_config(),
    }
    return configs.get(shaping_type, None)


def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    """Map agents to policies for self-play.

    For now, both agents use the same policy (pure self-play).
    Future: implement policy pool with opponent sampling.
    """
    return "main_policy"


def create_env_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create environment configuration dict from loaded config."""
    env_cfg = config["env"]

    return {
        "map_name": env_cfg["map_name"],
        "max_turns": env_cfg["max_turns"],
        "action_budget": env_cfg["action_budget"],
        "reward_shaping_config": get_reward_shaping_config(env_cfg["reward_shaping"]),
    }


def setup_algorithm(config: Dict[str, Any], checkpoint_dir: str = None):
    """Set up RLlib PPO algorithm with configuration.

    Args:
        config: Loaded configuration dict
        checkpoint_dir: Directory to save checkpoints

    Returns:
        Configured PPO algorithm
    """
    env_config = create_env_config(config)
    training_cfg = config["training"]
    ppo_cfg = config["ppo"]
    model_cfg = config["model"]

    # Create PPO config
    algo_config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(
            env=RLlibParallelRiskEnv,
            env_config=env_config,
        )
        .framework("torch")
        .resources(
            num_gpus=training_cfg["num_gpus"],
        )
        .env_runners(
            num_env_runners=training_cfg["num_workers"],
            num_envs_per_env_runner=training_cfg["num_envs_per_worker"],
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
            lr_schedule=ppo_cfg["lr_schedule"],
        )
        .multi_agent(
            policies={"main_policy"},
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["main_policy"],
        )
        .debugging(
            log_level=config["logging"]["log_level"],
        )
    )

    # Add model configuration
    algo_config.model.update({
        "fcnet_hiddens": model_cfg["fcnet_hiddens"],
        "fcnet_activation": model_cfg["fcnet_activation"],
        "vf_share_layers": model_cfg["vf_share_layers"],
    })

    # Build algorithm
    algo = algo_config.build()

    return algo


def train(
    config_path: str,
    checkpoint_dir: str = None,
    num_iterations: int = None,
    **overrides
):
    """Run training loop.

    Args:
        config_path: Path to YAML configuration file
        checkpoint_dir: Directory to save checkpoints
        num_iterations: Override number of training iterations
        **overrides: Override config values (e.g., num_workers=8)
    """
    # Load config
    config = load_config(config_path)

    # Apply command-line overrides
    if num_iterations is not None:
        config["stop"]["training_iteration"] = num_iterations
    for key, value in overrides.items():
        # Simple override: assumes keys are in training section
        if key in config.get("training", {}):
            config["training"][key] = value

    # Set up checkpoint directory
    if checkpoint_dir is None:
        checkpoint_dir = "checkpoints/ppo_parallel_risk"
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("=" * 70)
    print("PARALLEL RISK - RLLIB TRAINING")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Environment: {config['env']['map_name']}")
    print(f"  Action budget: {config['env']['action_budget']}")
    print(f"  Reward shaping: {config['env']['reward_shaping']}")
    print(f"  Workers: {config['training']['num_workers']}")
    print(f"  GPUs: {config['training']['num_gpus']}")
    print(f"  Checkpoint dir: {checkpoint_dir}")
    print("=" * 70)

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    # Create algorithm
    algo = setup_algorithm(config, checkpoint_dir)

    # Training loop
    best_reward = float('-inf')
    checkpoint_freq = config["checkpoint"]["frequency"]
    max_iterations = config["stop"]["training_iteration"]

    try:
        for iteration in range(1, max_iterations + 1):
            # Train
            result = algo.train()

            # Print progress
            print(f"\n=== Iteration {iteration} ===")

            # Extract metrics from nested structure
            env_runners = result.get('env_runners', {})
            episode_reward = env_runners.get('episode_reward_mean', result.get('episode_reward_mean', 'N/A'))
            episode_length = env_runners.get('episode_len_mean', result.get('episode_len_mean', 'N/A'))

            if isinstance(episode_reward, (int, float)):
                print(f"  Episode reward mean: {episode_reward:.3f}")
            else:
                print(f"  Episode reward mean: {episode_reward}")

            if isinstance(episode_length, (int, float)):
                print(f"  Episode length mean: {episode_length:.1f}")
            else:
                print(f"  Episode length mean: {episode_length}")

            print(f"  Timesteps total: {result.get('timesteps_total', 'N/A')}")
            print(f"  Time elapsed: {result.get('time_total_s', 0):.1f}s")

            # Log additional metrics if available
            if 'policy_reward_mean' in result:
                for policy_id, reward in result['policy_reward_mean'].items():
                    print(f"  Policy {policy_id} reward: {reward:.3f}")

            # Checkpoint
            if iteration % checkpoint_freq == 0:
                # Save to iteration-specific subdirectory
                iteration_checkpoint_dir = os.path.join(checkpoint_dir, f"checkpoint_{iteration:06d}")
                checkpoint_path = algo.save(iteration_checkpoint_dir)
                print(f"  💾 Checkpoint saved: {checkpoint_path}")

                # Save best checkpoint separately
                env_runners = result.get('env_runners', {})
                current_reward = env_runners.get('episode_reward_mean', result.get('episode_reward_mean', float('-inf')))
                if isinstance(current_reward, (int, float)) and current_reward > best_reward:
                    best_reward = current_reward
                    best_path = os.path.join(checkpoint_dir, "best_checkpoint")
                    algo.save(best_path)
                    print(f"  ⭐ New best checkpoint: {best_reward:.3f}")

            # Check stop conditions
            if result.get('timesteps_total', 0) >= config["stop"]["timesteps_total"]:
                print(f"\n✅ Reached timestep limit: {config['stop']['timesteps_total']}")
                break

            env_runners = result.get('env_runners', {})
            current_reward = env_runners.get('episode_reward_mean', result.get('episode_reward_mean', float('-inf')))
            if isinstance(current_reward, (int, float)) and current_reward >= config["stop"]["episode_reward_mean"]:
                print(f"\n✅ Reached reward threshold: {config['stop']['episode_reward_mean']}")
                break

    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")

    finally:
        # Save final checkpoint with iteration number
        final_checkpoint_dir = os.path.join(checkpoint_dir, f"checkpoint_{iteration:06d}")
        final_path = algo.save(final_checkpoint_dir)
        print(f"\n💾 Final checkpoint saved: {final_path}")

        # Cleanup
        algo.stop()
        ray.shutdown()

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train Parallel Risk agent with RLlib PPO"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="parallel_risk/training/configs/ppo_baseline.yaml",
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=None,
        help="Override number of training iterations"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override number of rollout workers"
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Override number of GPUs"
    )

    args = parser.parse_args()

    # Build overrides dict
    overrides = {}
    if args.num_workers is not None:
        overrides["num_workers"] = args.num_workers
    if args.num_gpus is not None:
        overrides["num_gpus"] = args.num_gpus

    # Run training
    train(
        config_path=args.config,
        checkpoint_dir=args.checkpoint_dir,
        num_iterations=args.num_iterations,
        **overrides
    )


if __name__ == "__main__":
    main()
