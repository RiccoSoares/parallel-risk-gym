"""Smoke tests for the ExIt (Expert Iteration) training track.

Verifies:
1. `play_episode_exit` returns well-formed per-action training examples
   with backfilled game outcomes.
2. `ExitTrainer.update` produces finite losses and flows gradients through
   the policy network on a real (small) example batch.

All tests use a randomly-initialized GNN and a tiny MCTS budget so they
run in under a few seconds without any external checkpoint dependency.
"""

import sys

import numpy as np
import torch

from parallel_risk import ParallelRiskEnv
from parallel_risk.agents.mcts_gnn_agent import MCTSGNNAgent
from parallel_risk.models.action_decoder import ActionDecoder
from parallel_risk.models.gnn_gcn import GCNPolicy
from parallel_risk.training.mcts_gnn.exit_self_play import play_episode_exit
from parallel_risk.training.mcts_gnn.exit_trainer import ExitTrainer


def _build_random_policy(map_config, action_budget=5, hidden_dim=32, num_layers=2):
    n_regions = len(map_config.regions)
    return GCNPolicy(
        node_features_dim=3 + n_regions,
        global_features_dim=2 + n_regions,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        action_budget=action_budget,
        max_troops=20,
        dropout=0.0,
    )


def test_exit_episode_produces_examples():
    """`play_episode_exit` returns per-action examples with backfilled z."""
    print("\n=== Test: play_episode_exit produces well-formed examples ===")
    torch.manual_seed(0)
    np.random.seed(0)

    env = ParallelRiskEnv(map_name="simple_6", max_turns=15, seed=42,
                          reward_shaping_config=None)
    action_budget = 5
    policy = _build_random_policy(env.map_config, action_budget=action_budget)
    decoder = ActionDecoder(action_budget=action_budget, max_troops=20)

    agent = MCTSGNNAgent(
        policy=policy,
        decoder=decoder,
        map_config=env.map_config,
        simulation_budget=6,
        c_puct=1.4,
        action_budget=action_budget,
        max_turns=env.max_turns,
        device='cpu',
    )
    print("  Built MCTSGNNAgent (budget=6, random-init GNN)")

    self_play_cfg = {
        'dirichlet_alpha': 0.3,
        'noise_frac': 0.25,
        'dirichlet_min_actions': 6,
        'temperature_turns': 5,
        'max_temperature': 1.0,
    }
    examples = play_episode_exit(agent, env, self_play_cfg, map_name='simple_6')
    print(f"  Episode produced {len(examples)} examples")

    assert len(examples) > 0, "Episode produced zero examples"

    for ex in examples:
        for k in ('graph', 'agent_id', 'action_key', 'map_name', 'z'):
            assert k in ex, f"example missing key '{k}': {ex.keys()}"
        assert ex['agent_id'] in ('agent_0', 'agent_1')
        assert ex['map_name'] == 'simple_6'
        assert isinstance(ex['action_key'], tuple), \
            f"action_key must be a tuple, got {type(ex['action_key'])}"
        assert len(ex['action_key']) == action_budget, \
            f"action_key len {len(ex['action_key'])} != action_budget {action_budget}"
        for row in ex['action_key']:
            assert len(row) == 3, f"action_key row must be (src, dst, troops), got {row}"
        # z is a shaped terminal reward in [-1, +1].
        assert -1.0 - 1e-6 <= ex['z'] <= 1.0 + 1e-6, \
            f"z should be in [-1, +1], got {ex['z']}"

    ids = set(ex['agent_id'] for ex in examples)
    assert 'agent_0' in ids and 'agent_1' in ids

    env.close()
    print("[OK] play_episode_exit examples are well-formed")


def test_exit_trainer_update_step():
    """`ExitTrainer.update` runs, loss is finite, gradients flow."""
    print("\n=== Test: ExitTrainer.update runs and flows gradients ===")
    torch.manual_seed(0)
    np.random.seed(0)

    trainer = ExitTrainer({
        'env': {'map_names': ['simple_6'], 'max_turns': 15, 'action_budget': 5},
        'model': {'type': 'gcn', 'hidden_dim': 32, 'num_layers': 2, 'dropout': 0.0},
        'trainer': {
            'learning_rate': 1e-3,
            'value_loss_coeff': 1.0,
            'entropy_coeff': 0.005,
            'max_grad_norm': 0.5,
            'use_gpu': False,
        },
        'log_dir': '/tmp/exit_test_runs',
        'checkpoint_dir': '/tmp/exit_test_ckpts',
    })
    print("  Built ExitTrainer (hidden_dim=32, 2 layers)")

    env = ParallelRiskEnv(map_name="simple_6", max_turns=10, seed=1,
                          reward_shaping_config=None)
    agent = MCTSGNNAgent(
        policy=trainer.policy,
        decoder=trainer.decoder,
        map_config=env.map_config,
        simulation_budget=6,
        c_puct=1.4,
        action_budget=5,
        max_turns=env.max_turns,
        device='cpu',
    )
    examples = play_episode_exit(
        agent, env,
        {'dirichlet_alpha': 0.3, 'noise_frac': 0.25, 'dirichlet_min_actions': 6,
         'temperature_turns': 3, 'max_temperature': 1.0},
        map_name='simple_6',
    )
    print(f"  Generated {len(examples)} examples for the update step")
    assert len(examples) > 0, "need at least one example"

    before = next(trainer.policy.parameters()).detach().clone()
    metrics = trainer.update(examples)
    print(f"  Update metrics: {metrics}")

    for k in ('total_loss', 'policy_loss', 'value_loss'):
        assert np.isfinite(metrics[k]), f"non-finite loss: {k}={metrics[k]}"

    after = next(trainer.policy.parameters()).detach().clone()
    delta = (after - before).abs().max().item()
    print(f"  max abs weight change: {delta:.6e}")
    assert delta > 0.0, "no parameter update after trainer.update"

    trainer.close()
    env.close()
    print("[OK] ExitTrainer.update runs and flows gradients")


def run_all_tests():
    print("=" * 70)
    print("MCTS+GNN ExIt (Expert Iteration) tests")
    print("=" * 70)
    passed = failed = 0
    tests = [
        test_exit_episode_produces_examples,
        test_exit_trainer_update_step,
    ]
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"\n[FAIL] Test failed: {e}")
            failed += 1
        except Exception as e:
            print(f"\n[FAIL] Test error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
