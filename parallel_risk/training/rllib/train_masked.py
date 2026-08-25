"""
Training script for Parallel Risk with action masking.

Uses a custom model that applies action masks during training to ensure
only valid actions are sampled. This should significantly improve learning
compared to the baseline where invalid actions are silently ignored.

Usage:
    PYTHONPATH=. python -m parallel_risk.training.rllib.train_masked \
        --num-iterations 50 --num-workers 8
"""

import argparse
import os
import yaml
from pathlib import Path
from typing import Dict, Any

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog

from parallel_risk.training.rllib.action_mask_wrapper import ActionMaskRLlibEnv
from parallel_risk.training.rllib.masked_model import SimpleMaskedModel


def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    """Map agents to policies for self-play."""
    return "main_policy"


def create_env_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create environment configuration."""
    env_cfg = config["env"]
    return {
        "map_name": env_cfg["map_name"],
        "max_turns": env_cfg["max_turns"],
        "action_budget": env_cfg["action_budget"],
        "reward_shaping_config": None,  # Sparse rewards
        "max_troops": 20,
    }


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train(
    config_path: str,
    checkpoint_dir: str,
    num_iterations: int,
    checkpoint_interval: int = 1,
    num_workers: int = 4,
    verbose: bool = True,
):
    """Run training with action-masked model."""

    print("=" * 70)
    print("PARALLEL RISK - ACTION-MASKED TRAINING")
    print("=" * 70)

    # Load config
    config = load_config(config_path)
    env_config = create_env_config(config)
    ppo_cfg = config["ppo"]
    model_cfg = config["model"]
    training_cfg = config["training"]

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)

    # Register custom model
    ModelCatalog.register_custom_model("masked_model", SimpleMaskedModel)

    # Get environment dimensions
    n_territories = 6  # simple_6 map
    n_regions = 3  # simple_6 has 3 regions
    action_budget = env_config["action_budget"]
    max_troops = env_config["max_troops"]

    # Calculate observation size explicitly
    # (same formula as wrapper._create_observation_space)
    obs_size = (
        n_territories +  # ownership
        n_territories +  # troops
        n_territories * n_territories +  # adjacency
        1 +  # income
        1 +  # turn
        n_regions  # region control
    )

    print(f"\nConfiguration:")
    print(f"  Map: {env_config['map_name']}")
    print(f"  Action budget: {action_budget}")
    print(f"  Max troops: {max_troops}")
    print(f"  Iterations: {num_iterations}")
    print(f"  Checkpoint interval: {checkpoint_interval}")
    print(f"  Workers: {num_workers}")
    print(f"  Checkpoint dir: {checkpoint_dir}")
    print("=" * 70)

    # Build PPO config
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
            num_epochs=training_cfg["num_sgd_iter"],
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
    print("\nStarting training...\n")

    try:
        for iteration in range(1, num_iterations + 1):
            result = algo.train()

            # Extract metrics
            env_runners = result.get('env_runners', {})
            episode_reward = env_runners.get(
                'episode_reward_mean',
                result.get('episode_reward_mean', 'N/A')
            )
            episode_length = env_runners.get(
                'episode_len_mean',
                result.get('episode_len_mean', 'N/A')
            )

            if verbose:
                reward_str = f"{episode_reward:.3f}" if isinstance(episode_reward, (int, float)) else str(episode_reward)
                length_str = f"{episode_length:.1f}" if isinstance(episode_length, (int, float)) else str(episode_length)
                print(f"  Iteration {iteration}/{num_iterations} - Reward: {reward_str}, Length: {length_str}")

            # Save checkpoint
            if iteration % checkpoint_interval == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{iteration:06d}")
                algo.save(checkpoint_path)
                if verbose:
                    print(f"    💾 Saved checkpoint: {checkpoint_path}")

        print("\n✓ Training completed successfully")
        return True

    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        algo.stop()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train Parallel Risk with action masking"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="parallel_risk/training/rllib/configs/ppo_baseline.yaml",
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/masked_training",
        help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=50,
        help="Number of training iterations"
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1,
        help="Save checkpoint every N iterations"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of rollout workers"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print detailed output"
    )

    args = parser.parse_args()

    train(
        config_path=args.config,
        checkpoint_dir=args.checkpoint_dir,
        num_iterations=args.num_iterations,
        checkpoint_interval=args.checkpoint_interval,
        num_workers=args.num_workers,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
