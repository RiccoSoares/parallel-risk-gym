"""
Test GCN policy and action decoder.

Verifies that:
1. GCN policy can process graph observations
2. Action decoder can sample actions
3. Forward pass works with batched graphs
4. Log probabilities and entropy computation work
"""

import sys
import torch
import numpy as np

from parallel_risk import ParallelRiskEnv
from parallel_risk.training.torchrl.graph_wrapper import env_to_graph, GraphObservationWrapper
from parallel_risk.models.gnn_gcn import GCNPolicy
from parallel_risk.models.action_decoder import ActionDecoder


def test_gcn_policy_forward():
    """Test GCN policy forward pass."""
    print("\n=== Test: GCN Policy Forward Pass ===")

    # Create environment and get graph observation
    env = ParallelRiskEnv(map_name="simple_6", seed=42)
    obs, _ = env.reset()

    graph = env_to_graph(obs['agent_0'], env.map_config)

    # Create GCN policy
    node_features_dim = graph.x.shape[1]
    global_features_dim = graph.global_features.shape[0]

    policy = GCNPolicy(
        node_features_dim=node_features_dim,
        global_features_dim=global_features_dim,
        hidden_dim=64,
        num_layers=2,
        action_budget=5,
        max_troops=20,
    )

    print(f"✓ GCN policy created")
    print(f"  Node features dim: {node_features_dim}")
    print(f"  Global features dim: {global_features_dim}")
    print(f"  Hidden dim: 64")
    print(f"  Layers: 2")

    # Forward pass
    with torch.no_grad():
        action_logits, value, embeddings = policy(graph, return_embeddings=True)

    print(f"✓ Forward pass successful")
    print(f"  Action budget: {len(action_logits)}")
    print(f"  Value shape: {value.shape}")
    print(f"  Node embeddings shape: {embeddings.shape}")

    # Check action logits structure
    assert len(action_logits) == 5, f"Expected 5 action heads, got {len(action_logits)}"

    for i, logits_dict in enumerate(action_logits):
        assert 'source' in logits_dict, f"Action {i} missing 'source'"
        assert 'dest' in logits_dict, f"Action {i} missing 'dest'"
        assert 'troops' in logits_dict, f"Action {i} missing 'troops'"

        print(f"  Action {i}: source={logits_dict['source'].shape}, "
              f"dest={logits_dict['dest'].shape}, troops={logits_dict['troops'].shape}")

    print(f"✓ All action logits have correct structure")

    env.close()


def test_action_decoder():
    """Test action decoder."""
    print("\n=== Test: Action Decoder ===")

    # Create environment and get graph observation
    env = ParallelRiskEnv(map_name="simple_6", seed=42)
    obs, _ = env.reset()

    graph = env_to_graph(obs['agent_0'], env.map_config)

    # Create GCN policy
    node_features_dim = graph.x.shape[1]
    global_features_dim = graph.global_features.shape[0]

    policy = GCNPolicy(
        node_features_dim=node_features_dim,
        global_features_dim=global_features_dim,
        hidden_dim=64,
        num_layers=2,
        action_budget=5,
        max_troops=20,
    )

    # Get action logits
    with torch.no_grad():
        action_logits, value, _ = policy(graph)

    # Create action decoder
    decoder = ActionDecoder(action_budget=5, max_troops=20)

    print(f"✓ Action decoder created")

    # Decode actions (sampling)
    batch = torch.zeros(graph.num_nodes, dtype=torch.long)  # Single graph
    actions, log_probs = decoder.decode_actions(
        action_logits, batch, deterministic=False, return_log_probs=True
    )

    print(f"✓ Actions decoded")
    print(f"  Actions shape: {actions.shape}")
    print(f"  Log probs shape: {log_probs.shape}")
    print(f"  Sample actions:\n{actions[0]}")

    assert actions.shape == (1, 5, 3), f"Expected (1, 5, 3), got {actions.shape}"
    assert log_probs.shape == (1, 5), f"Expected (1, 5), got {log_probs.shape}"

    # Test deterministic decoding
    actions_det, _ = decoder.decode_actions(
        action_logits, batch, deterministic=True, return_log_probs=False
    )

    print(f"✓ Deterministic decoding works")
    print(f"  Deterministic actions:\n{actions_det[0]}")

    # Test log prob computation
    recomputed_log_probs = decoder.compute_log_probs(action_logits, actions, batch)

    print(f"✓ Log probability computation works")
    print(f"  Original log probs: {log_probs[0, :3]}")
    print(f"  Recomputed log probs: {recomputed_log_probs[0, :3]}")

    # They should match closely
    assert torch.allclose(log_probs, recomputed_log_probs, atol=1e-5), "Log probs don't match"

    # Test entropy computation
    entropies = decoder.compute_entropy(action_logits, batch)

    print(f"✓ Entropy computation works")
    print(f"  Entropies shape: {entropies.shape}")
    print(f"  Sample entropies: {entropies[0, :3]}")

    env.close()


def test_batched_graphs():
    """Test with batched graphs."""
    print("\n=== Test: Batched Graphs ===")

    # Create two graphs
    env = ParallelRiskEnv(map_name="simple_6", seed=42)
    obs1, _ = env.reset(seed=42)
    obs2, _ = env.reset(seed=43)

    graph1 = env_to_graph(obs1['agent_0'], env.map_config)
    graph2 = env_to_graph(obs2['agent_0'], env.map_config)

    # Batch graphs
    from torch_geometric.data import Batch as GeometricBatch
    batched_graph = GeometricBatch.from_data_list([graph1, graph2])

    print(f"✓ Batched graph created")
    print(f"  Total nodes: {batched_graph.num_nodes}")
    print(f"  Batch size: 2")

    # Create policy
    node_features_dim = graph1.x.shape[1]
    global_features_dim = graph1.global_features.shape[0]

    policy = GCNPolicy(
        node_features_dim=node_features_dim,
        global_features_dim=global_features_dim,
        hidden_dim=64,
        num_layers=2,
        action_budget=3,
        max_troops=20,
    )

    # Forward pass on batched graphs
    with torch.no_grad():
        action_logits, value, _ = policy(batched_graph)

    print(f"✓ Forward pass on batch successful")
    print(f"  Value shape: {value.shape}")

    # Decode actions for batch
    decoder = ActionDecoder(action_budget=3, max_troops=20)
    actions, log_probs = decoder.decode_actions(
        action_logits, batched_graph.batch, deterministic=False, return_log_probs=True
    )

    print(f"✓ Batch action decoding successful")
    print(f"  Actions shape: {actions.shape}")
    print(f"  Expected: (2, 3, 3)")

    assert actions.shape == (2, 3, 3), f"Expected (2, 3, 3), got {actions.shape}"

    print(f"  Graph 0 actions:\n{actions[0]}")
    print(f"  Graph 1 actions:\n{actions[1]}")

    env.close()


def test_gradient_flow():
    """Test that gradients flow through the network."""
    print("\n=== Test: Gradient Flow ===")

    # Create environment and graph
    env = ParallelRiskEnv(map_name="simple_6", seed=42)
    obs, _ = env.reset()
    graph = env_to_graph(obs['agent_0'], env.map_config)

    # Create policy
    node_features_dim = graph.x.shape[1]
    global_features_dim = graph.global_features.shape[0]

    policy = GCNPolicy(
        node_features_dim=node_features_dim,
        global_features_dim=global_features_dim,
        hidden_dim=32,
        num_layers=2,
        action_budget=2,
        max_troops=20,
    )

    # Forward pass with gradients enabled
    action_logits, value, _ = policy(graph)

    # Create decoder
    decoder = ActionDecoder(action_budget=2, max_troops=20)
    batch = torch.zeros(graph.num_nodes, dtype=torch.long)

    # Sample actions and get log probs
    actions, log_probs = decoder.decode_actions(
        action_logits, batch, deterministic=False, return_log_probs=True
    )

    # Compute a simple loss (policy gradient style)
    # Fake reward
    reward = torch.tensor([[1.0, 1.0]])

    policy_loss = -(log_probs * reward).mean()
    value_loss = (value - reward.mean()) ** 2

    total_loss = policy_loss + value_loss

    print(f"✓ Loss computed")
    print(f"  Policy loss: {policy_loss.item():.4f}")
    print(f"  Value loss: {value_loss.item():.4f}")
    print(f"  Total loss: {total_loss.item():.4f}")

    # Backward pass
    total_loss.backward()

    # Check that gradients exist
    has_grads = False
    for name, param in policy.named_parameters():
        if param.grad is not None:
            has_grads = True
            break

    assert has_grads, "No gradients found!"

    print(f"✓ Gradients flow through network")

    env.close()


def run_all_tests():
    """Run all tests."""
    print("="*70)
    print("GCN POLICY & ACTION DECODER TEST SUITE")
    print("="*70)

    passed = 0
    failed = 0

    tests = [
        test_gcn_policy_forward,
        test_action_decoder,
        test_batched_graphs,
        test_gradient_flow,
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
