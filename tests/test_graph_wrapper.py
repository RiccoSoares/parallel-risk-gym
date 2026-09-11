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

    # Verify global features (env_to_graph shapes as [1, dim] for PyG batching)
    expected_global_dim = 2 + n_regions  # income, turn, + region control
    assert graph.global_features.shape[-1] == expected_global_dim, (
        f"Expected {expected_global_dim} global features, got {graph.global_features.shape[-1]}"
    )

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


def _reference_env_to_graph(obs, map_config, max_regions=None):
    """Straightforward build from the observation alone (the pre-cache implementation).

    Kept verbatim so the cached env_to_graph can be checked against it.
    """
    from torch_geometric.data import Data

    n_territories = map_config.n_territories
    n_regions = len(map_config.regions)
    if max_regions is None:
        max_regions = n_regions

    ownership = obs['territory_ownership']
    troops = obs['territory_troops']
    adjacency_matrix = obs['adjacency_matrix']
    available_income = obs['available_income']
    turn_number = obs['turn_number']
    region_control = obs['region_control']

    node_features = [
        np.log1p(troops) / np.log1p(100.0),
        ownership,
        adjacency_matrix.sum(axis=1) / n_territories,
    ]
    territory_to_region = np.zeros((n_territories, max_regions), dtype=np.float32)
    for region_idx, (region_name, territories) in enumerate(map_config.regions.items()):
        for territory_id in territories:
            territory_to_region[territory_id, region_idx] = 1.0
    for region_idx in range(max_regions):
        node_features.append(territory_to_region[:, region_idx])
    node_features = np.stack(node_features, axis=1).astype(np.float32)

    edge_sources, edge_targets = [], []
    for i in range(n_territories):
        for j in range(n_territories):
            if adjacency_matrix[i, j] == 1:
                edge_sources.append(i)
                edge_targets.append(j)
    edge_index = np.array([edge_sources, edge_targets], dtype=np.int64)

    region_control_padded = np.zeros(max_regions, dtype=np.float32)
    region_control_padded[:n_regions] = region_control.astype(np.float32)
    global_features = np.concatenate([
        available_income.astype(np.float32) / 20.0,
        turn_number.astype(np.float32) / 100.0,
        region_control_padded,
    ])

    data = Data(
        x=torch.tensor(node_features, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        num_nodes=n_territories,
    )
    data.global_features = torch.tensor(global_features, dtype=torch.float32).unsqueeze(0)
    data.n_territories = n_territories
    data.n_regions = n_regions
    return data


def _assert_same_graph(got, ref, label):
    for name in ('x', 'edge_index', 'global_features'):
        a, b = getattr(got, name), getattr(ref, name)
        assert a.dtype == b.dtype, f"{label}: {name} dtype {a.dtype} != {b.dtype}"
        assert a.shape == b.shape, f"{label}: {name} shape {tuple(a.shape)} != {tuple(b.shape)}"
        assert torch.equal(a, b), f"{label}: {name} values differ"
    assert got.global_features.dim() == 2 and got.global_features.shape[0] == 1, (
        f"{label}: global_features must be [1, dim], got {tuple(got.global_features.shape)}"
    )
    assert got.num_nodes == ref.num_nodes, f"{label}: num_nodes {got.num_nodes} != {ref.num_nodes}"
    assert got.n_territories == ref.n_territories, f"{label}: n_territories differs"
    assert got.n_regions == ref.n_regions, f"{label}: n_regions differs"


def test_cached_build_matches_reference_all_maps():
    """The cached env_to_graph must equal the straightforward build on every map."""
    print("\n=== Test: cached env_to_graph == reference build (all maps) ===")
    from parallel_risk.env.map_config import MapRegistry
    from parallel_risk.agents.masked_random_agent import MaskedRandomAgentRLlib

    n_checked = 0
    for map_name in MapRegistry.list_maps():
        env = ParallelRiskEnv(map_name=map_name, max_actions_per_turn=10)
        sampler = MaskedRandomAgentRLlib.from_env(env, action_budget=10)
        n_regions = len(env.map_config.regions)
        np.random.seed(0)
        obs, _ = env.reset(seed=1)
        for step in range(6):
            for agent in env.agents:
                for max_regions in (None, n_regions + 3):
                    got = env_to_graph(obs[agent], env.map_config, max_regions=max_regions)
                    ref = _reference_env_to_graph(obs[agent], env.map_config, max_regions=max_regions)
                    _assert_same_graph(got, ref, f"{map_name} step={step} {agent} max_regions={max_regions}")
                    expected_dim = 2 + (max_regions if max_regions is not None else n_regions)
                    assert got.global_features.shape == (1, expected_dim), (
                        f"{map_name}: global_features shape {tuple(got.global_features.shape)}"
                    )
                    n_checked += 1
            actions = {agent: sampler.get_action_raw(obs[agent]) for agent in env.agents}
            obs, _, terms, truncs, _ = env.step(actions)
            if terms['__all__'] or truncs['__all__']:
                obs, _ = env.reset(seed=step + 2)
        env.close()
        print(f"[OK] {map_name}")
    print(f"[OK] {n_checked} graphs identical to the reference build (values, dtypes, edge order, attrs)")


def test_static_cache_follows_map_config_lifetime():
    """Cache entries are keyed by MapConfig identity and die with the object."""
    print("\n=== Test: static cache lifetime ===")
    import gc
    from parallel_risk.env.map_config import MapRegistry
    from parallel_risk.training.torchrl import graph_wrapper

    map_config = MapRegistry.get("simple_6")
    env = ParallelRiskEnv(map_name="simple_6")
    obs, _ = env.reset(seed=0)
    key = (id(map_config), len(map_config.regions))
    env_to_graph(obs['agent_0'], map_config)
    assert key in graph_wrapper._STATIC_CACHE, "entry not cached after first build"
    del map_config
    gc.collect()
    assert key not in graph_wrapper._STATIC_CACHE, "entry survived its MapConfig"
    env.close()
    print("[OK] entry created on first use and evicted when the MapConfig is collected")


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
        test_cached_build_matches_reference_all_maps,
        test_static_cache_follows_map_config_lifetime,
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
