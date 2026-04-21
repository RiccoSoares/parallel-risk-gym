"""
Test training script for GNN policies.

Quick sanity check that training runs without errors.
"""

import sys
import os
import tempfile
import yaml
import torch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from parallel_risk.training.torchrl.train import PPOTrainer


def test_trainer_initialization():
    """Test that trainer can be initialized."""
    print("\n=== Test: Trainer Initialization ===")

    config = {
        'env': {
            'map_name': 'simple_6',
            'max_turns': 100,
            'action_budget': 3,
            'seed': 42
        },
        'model': {
            'type': 'gcn',
            'hidden_dim': 32,
            'num_layers': 2,
            'dropout': 0.1
        },
        'training': {
            'num_workers': 1,
            'batch_size': 128,
            'num_epochs': 2,
            'learning_rate': 3e-4,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_epsilon': 0.2,
            'entropy_coeff': 0.01,
            'value_loss_coeff': 0.5,
            'max_grad_norm': 0.5,
            'use_gpu': False
        },
        'log_dir': tempfile.mkdtemp(),
        'checkpoint_dir': tempfile.mkdtemp()
    }

    trainer = PPOTrainer(config)

    print(f"✓ Trainer initialized")
    print(f"  Device: {trainer.device}")
    print(f"  Node features: {trainer.node_features_dim}")
    print(f"  Global features: {trainer.global_features_dim}")
    print(f"  Action budget: {trainer.action_budget}")


def test_rollout_collection():
    """Test collecting rollout."""
    print("\n=== Test: Rollout Collection ===")

    config = {
        'env': {
            'map_name': 'simple_6',
            'max_turns': 100,
            'action_budget': 3,
            'seed': 42
        },
        'model': {
            'type': 'gcn',
            'hidden_dim': 32,
            'num_layers': 2,
            'dropout': 0.1
        },
        'training': {
            'num_workers': 1,
            'batch_size': 64,
            'num_epochs': 2,
            'learning_rate': 3e-4,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_epsilon': 0.2,
            'entropy_coeff': 0.01,
            'value_loss_coeff': 0.5,
            'max_grad_norm': 0.5,
            'use_gpu': False
        },
        'log_dir': tempfile.mkdtemp(),
        'checkpoint_dir': tempfile.mkdtemp()
    }

    trainer = PPOTrainer(config)

    # Collect small rollout
    rollout = trainer.collect_rollout(num_steps=20)

    print(f"✓ Rollout collected")
    print(f"  Steps: {len(rollout['observations'])}")
    print(f"  Rewards shape: {rollout['rewards'][0].shape}")
    print(f"  Values shape: {rollout['values'][0].shape}")
    print(f"  Log probs shape: {rollout['log_probs'][0].shape}")

    assert len(rollout['observations']) == 20, "Should have 20 steps"
    assert len(rollout['rewards']) == 20, "Should have 20 rewards"


def test_gae_computation():
    """Test GAE computation."""
    print("\n=== Test: GAE Computation ===")

    config = {
        'env': {
            'map_name': 'simple_6',
            'max_turns': 100,
            'action_budget': 3,
            'seed': 42
        },
        'model': {
            'type': 'gcn',
            'hidden_dim': 32,
            'num_layers': 2,
            'dropout': 0.1
        },
        'training': {
            'num_workers': 1,
            'batch_size': 64,
            'num_epochs': 2,
            'learning_rate': 3e-4,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_epsilon': 0.2,
            'entropy_coeff': 0.01,
            'value_loss_coeff': 0.5,
            'max_grad_norm': 0.5,
            'use_gpu': False
        },
        'log_dir': tempfile.mkdtemp(),
        'checkpoint_dir': tempfile.mkdtemp()
    }

    trainer = PPOTrainer(config)

    # Fake rollout data
    T = 10
    B = 2
    rewards = [torch.randn(B) for _ in range(T)]
    values = [torch.randn(B) for _ in range(T)]
    dones = [torch.zeros(B) for _ in range(T)]

    advantages, returns = trainer.compute_gae(rewards, values, dones)

    print(f"✓ GAE computed")
    print(f"  Advantages shape: {advantages.shape}")
    print(f"  Returns shape: {returns.shape}")

    assert advantages.shape == (T, B), f"Expected ({T}, {B}), got {advantages.shape}"
    assert returns.shape == (T, B), f"Expected ({T}, {B}), got {returns.shape}"


def test_policy_update():
    """Test policy update."""
    print("\n=== Test: Policy Update ===")

    config = {
        'env': {
            'map_name': 'simple_6',
            'max_turns': 100,
            'action_budget': 2,
            'seed': 42
        },
        'model': {
            'type': 'gcn',
            'hidden_dim': 32,
            'num_layers': 2,
            'dropout': 0.1
        },
        'training': {
            'num_workers': 1,
            'batch_size': 64,
            'num_epochs': 2,
            'learning_rate': 3e-4,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_epsilon': 0.2,
            'entropy_coeff': 0.01,
            'value_loss_coeff': 0.5,
            'max_grad_norm': 0.5,
            'use_gpu': False
        },
        'log_dir': tempfile.mkdtemp(),
        'checkpoint_dir': tempfile.mkdtemp()
    }

    trainer = PPOTrainer(config)

    # Collect rollout
    rollout = trainer.collect_rollout(num_steps=10)

    print(f"✓ Rollout collected ({len(rollout['observations'])} steps)")

    # Update policy
    initial_params = [p.clone() for p in trainer.policy.parameters()]
    trainer.update_policy(rollout)

    print(f"✓ Policy updated")

    # Check that parameters changed
    params_changed = False
    for initial, current in zip(initial_params, trainer.policy.parameters()):
        if not torch.allclose(initial, current):
            params_changed = True
            break

    assert params_changed, "Parameters should have changed after update"
    print(f"  Parameters changed: ✓")


def test_short_training_run():
    """Test a short training run."""
    print("\n=== Test: Short Training Run ===")

    config = {
        'env': {
            'map_name': 'simple_6',
            'max_turns': 50,
            'action_budget': 2,
            'seed': 42
        },
        'model': {
            'type': 'gcn',
            'hidden_dim': 32,
            'num_layers': 2,
            'dropout': 0.1
        },
        'training': {
            'num_workers': 1,
            'batch_size': 64,
            'num_epochs': 2,
            'learning_rate': 3e-4,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_epsilon': 0.2,
            'entropy_coeff': 0.01,
            'value_loss_coeff': 0.5,
            'max_grad_norm': 0.5,
            'use_gpu': False
        },
        'log_dir': tempfile.mkdtemp(),
        'checkpoint_dir': tempfile.mkdtemp()
    }

    trainer = PPOTrainer(config)

    # Train for 3 iterations
    trainer.train(num_iterations=3)

    print(f"✓ Training completed")
    print(f"  Episodes collected: {len(trainer.episode_rewards)}")
    print(f"  Global steps: {trainer.global_step}")


def run_all_tests():
    """Run all tests."""
    print("="*70)
    print("TRAINING SCRIPT TEST SUITE")
    print("="*70)

    passed = 0
    failed = 0

    tests = [
        test_trainer_initialization,
        test_rollout_collection,
        test_gae_computation,
        test_policy_update,
        test_short_training_run,
    ]

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"\n✗ Test failed: {e}")
            failed += 1
        except Exception as e:
            print(f"\n✗ Test error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*70)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
