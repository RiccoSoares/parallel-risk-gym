"""
Test graph wrapper for TorchRL integration.

Verifies that:
1. env_to_graph() converts observations correctly
2. GraphObservationWrapper works with ParallelRiskEnv
3. Graph structure is valid for different map sizes
"""

import sys
import numpy as np
import torch

from parallel_risk import ParallelRiskEnv
from parallel_risk.training.torchrl.graph_wrapper import env_to_graph, GraphObservationWrapper


def test_env_to_graph():
    """Test env_to_graph() function."""
    print("\n=== Test: env_to_graph() ===")

    # Create environment
    env = ParallelRiskEnv(map_name="simple_6")
    obs, _ = env.reset(seed=42)

    # Convert observation to graph
    agent_obs = obs['agent_0']
    graph = env_to_graph(agent_obs, env.map_config)

    print(f"✓ Graph created")
    print(f"  Nodes: {graph.num_nodes}")
    print(f"  Node features shape: {graph.x.shape}")
    print(f"  Edge index shape: {graph.edge_index.shape}")
    print(f"  Global features shape: {graph.global_features.shape}")

    # Verify node features
    n_territories = env.map_config.n_territories
    n_regions = len(env.map_config.regions)
    expected_feature_dim = 3 + n_regions  # troops, ownership, in_degree, + region one-hot

    assert graph.num_nodes == n_territories, f"Expected {n_territories} nodes, got {graph.num_nodes}"
    assert graph.x.shape[0] == n_territories, f"Expected {n_territories} node features rows"
    assert graph.x.shape[1] == expected_feature_dim, f"Expected {expected_feature_dim} features per node"

    # Verify edge index
    assert graph.edge_index.shape[0] == 2, "Edge index should have 2 rows (source, target)"
    assert graph.edge_index.shape[1] > 0, "Should have at least one edge"

    # Verify all edges are valid
    assert torch.all(graph.edge_index >= 0), "Edge indices should be non-negative"
    assert torch.all(graph.edge_index < n_territories), f"Edge indices should be < {n_territories}"

    # Verify global features
    expected_global_dim = 2 + n_regions  # income, turn, + region control
    assert graph.global_features.shape[0] == expected_global_dim, f"Expected {expected_global_dim} global features"

    print(f"✓ All graph structure checks passed")


def test_graph_wrapper():
    """Test GraphObservationWrapper."""
    print("\n=== Test: GraphObservationWrapper ===")

    # Create environment and wrapper
    env = ParallelRiskEnv(map_name="simple_6", seed=42)
    wrapped_env = GraphObservationWrapper(env)

    print(f"✓ Wrapper created")

    # Test reset
    graph_obs, infos = wrapped_env.reset(seed=42)

    print(f"✓ Reset successful")
    print(f"  Agents: {list(graph_obs.keys())}")
    print(f"  Agent 0 graph nodes: {graph_obs['agent_0'].num_nodes}")
    print(f"  Agent 0 graph edges: {graph_obs['agent_0'].edge_index.shape[1]}")

    assert len(graph_obs) == 2, "Should have 2 agents"
    assert 'agent_0' in graph_obs, "Should have agent_0"
    assert 'agent_1' in graph_obs, "Should have agent_1"

    # Test step with random actions
    actions = {
        'agent_0': {
            'num_actions': 3,
            'actions': np.array([
                [0, 1, 2],
                [1, 0, 1],
                [0, 3, 1],
            ] + [[0, 0, 0]] * 7)  # Padding
        },
        'agent_1': {
            'num_actions': 2,
            'actions': np.array([
                [2, 1, 1],
                [5, 4, 1],
            ] + [[0, 0, 0]] * 8)  # Padding
        }
    }

    graph_obs, rewards, terminateds, truncateds, infos = wrapped_env.step(actions)

    print(f"✓ Step successful")
    print(f"  Rewards: {rewards}")
    print(f"  Terminated: {terminateds}")

    assert len(graph_obs) == 2, "Should have 2 agent observations"
    assert len(rewards) == 2, "Should have 2 rewards"

    wrapped_env.close()
    print(f"✓ All wrapper checks passed")


def test_different_map_sizes():
    """Test graph wrapper with different map sizes."""
    print("\n=== Test: Different Map Sizes ===")

    map_names = ["simple_6", "basic_6"]

    for map_name in map_names:
        try:
            env = ParallelRiskEnv(map_name=map_name)
            wrapped_env = GraphObservationWrapper(env)

            graph_obs, _ = wrapped_env.reset(seed=42)
            agent_graph = graph_obs['agent_0']

            print(f"✓ {map_name}: {agent_graph.num_nodes} nodes, {agent_graph.edge_index.shape[1]} edges")

            wrapped_env.close()

        except ValueError:
            print(f"  {map_name}: not available (skipped)")


def test_observation_space():
    """Test observation space description."""
    print("\n=== Test: Observation Space ===")

    env = ParallelRiskEnv(map_name="simple_6")
    wrapped_env = GraphObservationWrapper(env)

    obs_space = wrapped_env.observation_space

    print(f"✓ Observation space:")
    print(f"  Type: {obs_space['type']}")
    print(f"  Node features dim: {obs_space['node_features_dim']}")
    print(f"  Global features dim: {obs_space['global_features_dim']}")
    print(f"  Territories: {obs_space['n_territories']}")
    print(f"  Regions: {obs_space['n_regions']}")

    assert obs_space['type'] == 'graph', "Observation space should be graph type"

    wrapped_env.close()


def test_edge_bidirectionality():
    """Test that edges are bidirectional (undirected graph)."""
    print("\n=== Test: Edge Bidirectionality ===")

    env = ParallelRiskEnv(map_name="simple_6")
    obs, _ = env.reset(seed=42)

    graph = env_to_graph(obs['agent_0'], env.map_config)
    edge_index = graph.edge_index.numpy()

    # Check that for each edge (i, j), there exists edge (j, i)
    edges = set()
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        edges.add((src, dst))

    all_bidirectional = True
    for src, dst in edges:
        if (dst, src) not in edges:
            print(f"  ✗ Edge ({src}, {dst}) has no reverse edge ({dst}, {src})")
            all_bidirectional = False

    if all_bidirectional:
        print(f"✓ All edges are bidirectional ({len(edges)} edges)")
    else:
        print(f"✗ Some edges are not bidirectional")

    env.close()


def run_all_tests():
    """Run all tests."""
    print("="*70)
    print("GRAPH WRAPPER TEST SUITE")
    print("="*70)

    passed = 0
    failed = 0

    tests = [
        test_env_to_graph,
        test_graph_wrapper,
        test_different_map_sizes,
        test_observation_space,
        test_edge_bidirectionality,
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
            failed += 1

    print("\n" + "="*70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*70)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
