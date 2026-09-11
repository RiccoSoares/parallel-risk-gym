"""
Test training script for GNN policies.

Quick sanity check that training runs without errors, plus checks of the
lockstep vectorized rollout (shapes, bootstrap semantics, per-column GAE,
worker transport).
"""

import sys
import os
import tempfile
import yaml
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from parallel_risk.training.torchrl.train import PPOTrainer
from parallel_risk.training.torchrl.vec_rollout import LockstepRollout, AGENTS
from parallel_risk.models.gnn_gcn import GCNPolicy
from parallel_risk.models.action_decoder import ActionDecoder
from torch_geometric.data import Batch


def make_config(map_names=('simple_6',), action_budget=3, max_turns=100, num_workers=1,
                batch_size=64, num_envs=None, use_reward_shaping=True, seed=42):
    """Config dict for a small trainer; num_envs=None leaves the trainer default."""
    training = {
        'num_workers': num_workers,
        'batch_size': batch_size,
        'num_epochs': 2,
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_epsilon': 0.2,
        'entropy_coeff': 0.01,
        'value_loss_coeff': 0.5,
        'max_grad_norm': 0.5,
        'use_gpu': False,
    }
    if num_envs is not None:
        training['num_envs'] = num_envs
    return {
        'env': {
            'map_names': list(map_names),
            'max_turns': max_turns,
            'action_budget': action_budget,
            'seed': seed,
            'use_reward_shaping': use_reward_shaping,
        },
        'model': {'type': 'gcn', 'hidden_dim': 32, 'num_layers': 2, 'dropout': 0.1},
        'training': training,
        'log_dir': tempfile.mkdtemp(),
        'checkpoint_dir': tempfile.mkdtemp(),
    }


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

    # 4 envs in lockstep: 20 env steps -> T=5 lockstep steps of B=8 samples
    trainer = PPOTrainer(make_config(num_envs=4))

    # Collect small rollout
    rollout = trainer.collect_rollout(num_steps=20)

    print(f"✓ Rollout collected")
    print(f"  Steps: {len(rollout['rewards'])}")
    print(f"  Rewards shape: {rollout['rewards'][0].shape}")
    print(f"  Values shape: {rollout['values'][0].shape}")
    print(f"  Log probs shape: {rollout['log_probs'][0].shape}")

    assert len(rollout['rewards']) == 5, "Should have 5 lockstep steps"
    assert rollout['rewards'][0].shape == (8,), "Should have 2 * num_envs samples per step"


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
    next_values = [torch.randn(B) for _ in range(T)]

    advantages, returns = trainer.compute_gae(rewards, values, dones, next_values)

    print(f"✓ GAE computed")
    print(f"  Advantages shape: {advantages.shape}")
    print(f"  Returns shape: {returns.shape}")

    assert advantages.shape == (T, B), f"Expected ({T}, {B}), got {advantages.shape}"
    assert returns.shape == (T, B), f"Expected ({T}, {B}), got {returns.shape}"


def test_policy_update():
    """Test policy update."""
    print("\n=== Test: Policy Update ===")

    trainer = PPOTrainer(make_config(action_budget=2, num_envs=2))

    # Collect rollout (10 env steps = 5 lockstep steps of 2 envs)
    rollout = trainer.collect_rollout(num_steps=10)

    print(f"✓ Rollout collected ({len(rollout['rewards'])} steps)")

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


def test_lockstep_rollout_shapes():
    """Vectorized path: [T, B] grid, per-step tensor shapes, graph_lists, episode bookkeeping."""
    print("\n=== Test: Lockstep Rollout Shapes ===")
    maps = ('simple_6', 'medium_8')
    trainer = PPOTrainer(make_config(map_names=maps, action_budget=3, max_turns=6, num_envs=4))

    rollout = trainer.collect_rollout(num_steps=40)  # 40 env steps / 4 envs -> T=10, B=8
    T, B, K = 10, 8, 3
    assert len(rollout['rewards']) == T
    for key in ('values', 'log_probs', 'dones', 'next_values', 'actions', 'graph_lists', 'terminateds'):
        assert len(rollout[key]) == T, f"{key}: {len(rollout[key])} != {T}"
    assert rollout['rewards'][0].shape == (B,)
    assert rollout['values'][0].shape == (B,)
    assert rollout['next_values'][0].shape == (B,)
    assert rollout['dones'][0].dtype == torch.bool and rollout['dones'][0].shape == (B,)
    assert rollout['log_probs'][0].shape == (B, K)
    assert rollout['actions'][0].shape == (B, K, 3)
    for step_graphs in rollout['graph_lists']:
        assert len(step_graphs) == B
        for g in step_graphs:
            assert g.x.shape == (g.num_nodes, trainer.node_features_dim)
            assert g.global_features.shape == (1, trainer.global_features_dim)
            assert g.edge_index.max().item() < g.num_nodes
    # column 2i and 2i+1 are the two agents of the same env: they see the same map size
    sizes = [[g.num_nodes for g in step_graphs] for step_graphs in rollout['graph_lists']]
    for row in sizes:
        assert all(row[2 * i] == row[2 * i + 1] for i in range(B // 2))
    # dones are per env (both columns of an env end together)
    dones = torch.stack(rollout['dones'])
    assert torch.equal(dones[:, 0::2], dones[:, 1::2])
    # max_turns=6 with T=10: every env finishes at least one episode (turn limit), so
    # episode bookkeeping must have entries and they must match the trainer state.
    assert len(rollout['map_names']) >= B // 2, rollout['map_names']
    assert set(rollout['map_names']) <= set(maps)
    assert len(trainer.episode_rewards) == len(rollout['map_names'])
    assert len(trainer.episode_lengths) == len(rollout['map_names'])
    assert sum(len(v) for v in trainer.episode_rewards_per_map.values()) == len(rollout['map_names'])
    assert max(trainer.episode_lengths) <= 6
    # the mega-batch the update builds has T*B graphs in (t, column) order
    all_graphs = [g for gl in rollout['graph_lists'] for g in gl]
    mega = Batch.from_data_list(all_graphs)
    assert mega.num_graphs == T * B
    print(f"✓ T={T} B={B} samples={T * B} episodes={len(rollout['map_names'])}")


def _record_next_obs(engine):
    """Wrap every slot env's step so the test can see the observation each step returned."""
    log = {}
    for i, slot in enumerate(engine.slots):
        log[i] = []
        env = slot.env
        orig_step = env.step

        def step(actions, _orig=orig_step, _log=log[i]):
            out = _orig(actions)
            _log.append(out[0])
            return out
        env.step = step
    return log


def test_lockstep_bootstrap_semantics():
    """next_values: reuse of V(s') for non-terminal, V(final obs) on truncation, 0 on termination."""
    print("\n=== Test: Lockstep Bootstrap Semantics ===")
    torch.manual_seed(0)
    np.random.seed(0)
    K = 2
    policy = GCNPolicy(node_features_dim=3 + 3, global_features_dim=2 + 3, hidden_dim=32,
                       num_layers=2, action_budget=K, max_troops=20, dropout=0.1)
    policy.eval()  # deterministic forward: reuse must be exact
    engine = LockstepRollout(policy, ActionDecoder(action_budget=K, max_troops=20),
                             map_names=['simple_6'], num_envs=3, action_budget=K, max_regions=3,
                             max_turns=3, use_reward_shaping=False, rng=np.random.RandomState(0))
    engine.begin()
    obs_log = _record_next_obs(engine)   # single map: each slot keeps one env object

    engine.step()                        # t=0
    # Force a termination in slot 0 on t=1: agent_0 owns everything, so after the
    # step the victory check fires (terminated, reward +1 for agent_0).
    engine.slots[0].env.env.game_state['territory_ownership'][:] = 0
    engine.step()                        # t=1: slot 0 terminated
    engine.step()                        # t=2: slots 1 and 2 hit max_turns=3 -> truncated
    engine.step()                        # t=3
    engine.step()                        # t=4: slot 0's second episode truncated
    arrays = engine.finish()
    T, B = arrays['T'], arrays['B']
    assert (T, B) == (5, 6)

    dones, terms = arrays['dones'], arrays['terminateds']
    values, next_values = arrays['values'], arrays['next_values']
    assert dones[1, 0:2].all() and terms[1, 0:2].all()
    assert torch.all(next_values[1, 0:2] == 0.0), "terminated must bootstrap with 0"
    assert dones[2, 2:6].all() and not terms[2, 2:6].any()
    assert dones[4, 0:2].all() and not terms[4, 0:2].any()
    assert int(dones.sum()) == 2 + 4 + 2
    # truncated: V(final observation) — the engine forwards exactly the truncated envs'
    # final graphs (slot order), so the same batch reproduces its values.
    final_graphs = [obs_log[s][2][a] for s in (1, 2) for a in AGENTS]
    with torch.no_grad():
        _, v_final, _ = policy(Batch.from_data_list(final_graphs))
    assert torch.allclose(next_values[2, 2:6], v_final.squeeze(-1), atol=1e-6), \
        (next_values[2, 2:6], v_final.squeeze(-1))
    with torch.no_grad():
        _, v_final0, _ = policy(Batch.from_data_list([obs_log[0][4][a] for a in AGENTS]))
    assert torch.allclose(next_values[4, 0:2], v_final0.squeeze(-1), atol=1e-6)
    # non-terminal: exactly the next step's value of the same column
    live = ~dones[:-1]
    assert torch.equal(next_values[:-1][live], values[1:][live])
    # last step, non-terminal columns: V of the observation the env is now in
    current = [slot.obs[a] for slot in engine.slots for a in AGENTS]
    with torch.no_grad():
        _, v_last, _ = policy(Batch.from_data_list(current))
    live_last = ~dones[-1]
    assert torch.allclose(next_values[-1][live_last], v_last.squeeze(-1)[live_last], atol=1e-6)
    # episode bookkeeping: order of completion (step, then slot)
    assert arrays['episode_lengths'] == [2, 3, 3, 3], arrays['episode_lengths']
    assert arrays['episode_rewards'] == [1.0, 0.0, 0.0, 0.0], arrays['episode_rewards']
    assert arrays['episode_map_names'] == ['simple_6'] * 4
    # rewards of the terminated step: +1 / -1 (no shaping)
    assert arrays['rewards'][1, 0] == 1.0 and arrays['rewards'][1, 1] == -1.0
    assert arrays['actions'].shape == (T, B, K, 3) and arrays['log_probs'].shape == (T, B, K)
    assert arrays['num_nodes'].tolist() == [6] * (T * B)
    assert arrays['x'].shape == (T * B * 6, 6) and arrays['global_features'].shape == (T * B, 5)
    print("✓ terminated -> 0, truncated -> V(final obs), non-terminal -> V(s') reused exactly")
    return arrays


def test_gae_per_column_boundaries():
    """compute_gae on a lockstep grid: columns independent, no propagation across dones."""
    print("\n=== Test: GAE Per-Column Boundaries ===")
    arrays = test_lockstep_bootstrap_semantics()
    trainer = PPOTrainer(make_config(action_budget=2, num_envs=2))
    gamma, lam = trainer.gamma, trainer.gae_lambda

    def gae_of(rewards):
        return trainer.compute_gae(list(rewards.unbind(0)), list(arrays['values'].unbind(0)),
                                   list(arrays['dones'].unbind(0)), list(arrays['next_values'].unbind(0)))

    rewards = arrays['rewards'].clone()
    adv, ret = gae_of(rewards)
    T, B = rewards.shape
    assert adv.shape == (T, B) and ret.shape == (T, B)
    assert torch.allclose(ret, adv + arrays['values'])
    delta = rewards + gamma * arrays['next_values'] - arrays['values']
    # at an episode boundary the advantage is the TD error alone
    for t in range(T):
        for c in range(B):
            if arrays['dones'][t, c]:
                assert torch.isclose(adv[t, c], delta[t, c]), (t, c, adv[t, c], delta[t, c])
            elif t < T - 1:
                assert torch.isclose(adv[t, c], delta[t, c] + gamma * lam * adv[t + 1, c])
    # a reward after a boundary must not leak backwards across it: slot 0 ends at t=1
    bumped = rewards.clone()
    bumped[2, 0] += 10.0
    adv2, _ = gae_of(bumped)
    assert torch.equal(adv2[:2, 0], adv[:2, 0]), "advantage leaked across a done"
    assert not torch.equal(adv2[2, 0], adv[2, 0])
    # columns are independent
    assert torch.equal(adv2[:, 1:], adv[:, 1:])
    trainer.writer.close()
    print("✓ GAE respects per-column episode boundaries")


def test_parallel_workers_compact_transfer():
    """Pool workers return compact arrays; merged columns give the same contract as in-process."""
    print("\n=== Test: Parallel Workers Compact Transfer ===")
    trainer = PPOTrainer(make_config(map_names=('simple_6', 'medium_8'), action_budget=2,
                                     max_turns=6, num_workers=2, num_envs=2, batch_size=32))
    try:
        for round_idx in range(2):  # second call exercises the per-worker cache
            rollout = trainer.collect_rollout(num_steps=16)  # 8 env steps per worker / 2 envs -> T=4
            T, B = 4, 8
            assert len(rollout['rewards']) == T
            assert rollout['rewards'][0].shape == (B,)
            assert rollout['actions'][0].shape == (B, 2, 3)
            assert rollout['log_probs'][0].shape == (B, 2)
            assert all(len(gl) == B for gl in rollout['graph_lists'])
            sizes = [[g.num_nodes for g in gl] for gl in rollout['graph_lists']]
            assert all(row[2 * i] == row[2 * i + 1] for row in sizes for i in range(B // 2))
            # node features of every sample must be consistent with its graph size
            for gl in rollout['graph_lists']:
                for g in gl:
                    assert g.x.shape[0] == g.num_nodes and g.x.shape[1] == trainer.node_features_dim
            dones = torch.stack(rollout['dones']); nv = torch.stack(rollout['next_values'])
            vals = torch.stack(rollout['values']); terms = torch.stack(rollout['terminateds'])
            live = ~dones[:-1]
            assert torch.equal(nv[:-1][live], vals[1:][live])
            assert torch.all(nv[terms] == 0.0)
            assert len(trainer.episode_rewards) == sum(len(v) for v in trainer.episode_rewards_per_map.values())
            trainer.update_policy(rollout)
        print(f"✓ 2 workers x 2 envs merged into T={T} B={B}; two rounds, update ran")
    finally:
        if trainer._worker_pool is not None:
            trainer._worker_pool.terminate()
            trainer._worker_pool.join()
        trainer.writer.close()


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
        test_lockstep_rollout_shapes,
        test_lockstep_bootstrap_semantics,
        test_gae_per_column_boundaries,
        test_parallel_workers_compact_transfer,
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
