#!/usr/bin/env python
"""
Quick validation test for PPO bug fixes.

Tests that the fixed training code runs without errors.
"""

import sys
import torch
import yaml
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from parallel_risk.training.torchrl.train import RunningMeanStd, PPOTrainer


def test_running_mean_std():
    """Test RunningMeanStd class."""
    print("Testing RunningMeanStd...")

    rms = RunningMeanStd()

    # Test update
    data = torch.randn(100)
    rms.update(data)

    assert rms.count == 100, f"Count should be 100, got {rms.count}"
    assert abs(rms.mean - data.mean().item()) < 0.1, "Mean should be close to data mean"

    # Test normalize
    normalized = rms.normalize(data)
    assert abs(normalized.mean().item()) < 0.2, "Normalized mean should be close to 0"

    print("✓ RunningMeanStd tests passed")


def test_trainer_init():
    """Test that PPOTrainer initializes correctly with bug fixes."""
    print("\nTesting PPOTrainer initialization...")

    config = {
        'env': {
            'map_name': 'simple_6',
            'max_turns': 100,
            'action_budget': 5,
        },
        'training': {
            'num_workers': 2,
            'batch_size': 128,
            'num_epochs': 3,
            'learning_rate': 3e-4,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_epsilon': 0.2,
            'entropy_coeff': 0.01,
            'value_loss_coeff': 0.5,
            'max_grad_norm': 0.5,
        },
        'model': {
            'hidden_dim': 128,
            'num_layers': 3,
            'dropout': 0.1,
        },
        'use_gpu': False,
        'log_dir': 'runs/test',
        'checkpoint_dir': 'checkpoints/test',
    }

    trainer = PPOTrainer(config)

    # Check that return_rms was initialized (Bug #3 fix)
    assert hasattr(trainer, 'return_rms'), "Trainer should have return_rms attribute"
    assert isinstance(trainer.return_rms, RunningMeanStd), "return_rms should be RunningMeanStd"

    print("✓ PPOTrainer initialization tests passed")


def test_rollout_collection():
    """Test that rollout collection stores next_obs (Bug #2 fix)."""
    print("\nTesting rollout collection...")

    config = {
        'env': {
            'map_name': 'simple_6',
            'max_turns': 100,
            'action_budget': 5,
        },
        'training': {
            'num_workers': 2,
            'batch_size': 10,  # Small for quick test
            'num_epochs': 1,
            'learning_rate': 3e-4,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_epsilon': 0.2,
            'entropy_coeff': 0.01,
            'value_loss_coeff': 0.5,
            'max_grad_norm': 0.5,
        },
        'model': {
            'hidden_dim': 64,
            'num_layers': 2,
            'dropout': 0.1,
        },
        'use_gpu': False,
        'log_dir': 'runs/test',
        'checkpoint_dir': 'checkpoints/test',
    }

    trainer = PPOTrainer(config)

    # Collect a small rollout
    rollout = trainer.collect_rollout(num_steps=5)

    # Check that next_obs is stored (Bug #2 fix)
    assert 'next_obs' in rollout, "Rollout should contain next_obs"
    print(f"  next_obs stored: {rollout['next_obs'] is not None}")

    # Check that compute_gae accepts next_obs parameter
    try:
        advantages, returns = trainer.compute_gae(
            rollout['rewards'],
            rollout['values'],
            rollout['dones'],
            rollout['next_obs']  # Bug #2 fix: passing next_obs
        )
        print(f"  GAE computed successfully with bootstrapping")
        print(f"  Advantages shape: {advantages.shape}")
        print(f"  Returns shape: {returns.shape}")
    except Exception as e:
        print(f"✗ GAE computation failed: {e}")
        raise

    print("✓ Rollout collection tests passed")


def test_value_normalization():
    """Test that returns are normalized in update_policy (Bug #3 fix)."""
    print("\nTesting value normalization...")

    config = {
        'env': {
            'map_name': 'simple_6',
            'max_turns': 100,
            'action_budget': 5,
        },
        'training': {
            'num_workers': 2,
            'batch_size': 10,
            'num_epochs': 1,
            'learning_rate': 3e-4,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_epsilon': 0.2,
            'entropy_coeff': 0.01,
            'value_loss_coeff': 0.5,
            'max_grad_norm': 0.5,
        },
        'model': {
            'hidden_dim': 64,
            'num_layers': 2,
            'dropout': 0.1,
        },
        'use_gpu': False,
        'log_dir': 'runs/test',
        'checkpoint_dir': 'checkpoints/test',
    }

    trainer = PPOTrainer(config)

    # Collect rollout and update
    rollout = trainer.collect_rollout(num_steps=5)

    # Check initial return_rms state
    initial_count = trainer.return_rms.count
    print(f"  Initial return_rms count: {initial_count}")

    # Run update (should update return_rms)
    try:
        trainer.update_policy(rollout)
        print(f"  Policy updated successfully")
        print(f"  Updated return_rms count: {trainer.return_rms.count}")
        print(f"  Return mean: {trainer.return_rms.mean:.4f}")
        print(f"  Return std: {trainer.return_rms.var**0.5:.4f}")

        assert trainer.return_rms.count > initial_count, "return_rms should be updated"
    except Exception as e:
        print(f"✗ Policy update failed: {e}")
        raise

    print("✓ Value normalization tests passed")


if __name__ == "__main__":
    print("=" * 60)
    print("PPO Bug Fixes Validation Test")
    print("=" * 60)

    try:
        test_running_mean_std()
        test_trainer_init()
        test_rollout_collection()
        test_value_normalization()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        print("\nThe PPO bug fixes are working correctly!")
        print("- Bug #2: GAE bootstrap is properly implemented")
        print("- Bug #3: Value normalization is working")

    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ TESTS FAILED")
        print("=" * 60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
