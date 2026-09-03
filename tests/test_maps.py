"""Tests for all registered maps in map_config.py."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from parallel_risk.env.map_config import MapRegistry, _check_connected, _check_bidirectional, _check_region_ids
from parallel_risk import ParallelRiskEnv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_noop_actions(env):
    """Return a do-nothing action dict for all active agents."""
    n = env.map_config.n_territories
    return {
        agent: {
            'num_actions': 0,
            'actions': np.zeros((env.max_actions_per_turn, 3), dtype=np.int32),
        }
        for agent in env.agents
    }


# ---------------------------------------------------------------------------
# Test 1: All maps load without error
# ---------------------------------------------------------------------------

def test_all_maps_load():
    print("\nTest 1: All maps load without error")
    names = MapRegistry.list_maps()
    assert len(names) > 0, "No maps registered"
    for name in names:
        config = MapRegistry.get(name)
        assert config is not None, f"Map '{name}' returned None"
        assert config.n_territories > 0
        assert config.adjacency_matrix is not None
        assert config.initial_ownership is not None
        print(f"  {name}: n_territories={config.n_territories}  OK")
    print("  PASSED")


# ---------------------------------------------------------------------------
# Test 2: basic_6 == simple_6
# ---------------------------------------------------------------------------

def test_basic_6_alias():
    print("\nTest 2: basic_6 is an alias for simple_6")
    s6 = MapRegistry.get("simple_6")
    b6 = MapRegistry.get("basic_6")
    assert s6.n_territories == b6.n_territories, (
        f"n_territories mismatch: simple_6={s6.n_territories} basic_6={b6.n_territories}"
    )
    assert np.array_equal(s6.adjacency_matrix, b6.adjacency_matrix), (
        "adjacency_matrix mismatch between simple_6 and basic_6"
    )
    assert np.array_equal(s6.initial_ownership, b6.initial_ownership), (
        "initial_ownership mismatch between simple_6 and basic_6"
    )
    assert s6.regions == b6.regions, "regions mismatch between simple_6 and basic_6"
    print("  basic_6.n_territories matches simple_6:", s6.n_territories)
    print("  adjacency_matrix: identical")
    print("  initial_ownership: identical")
    print("  PASSED")


# ---------------------------------------------------------------------------
# Test 3: Every map's graph is connected
# ---------------------------------------------------------------------------

def test_all_maps_connected():
    print("\nTest 3: All maps have connected graphs")
    for name in MapRegistry.list_maps():
        config = MapRegistry.get(name)
        connected = _check_connected(config.adjacency_list, config.n_territories)
        assert connected, f"Map '{name}' graph is NOT connected"
        print(f"  {name}: connected  OK")
    print("  PASSED")


# ---------------------------------------------------------------------------
# Test 4: All adjacencies are bidirectional
# ---------------------------------------------------------------------------

def test_all_maps_bidirectional():
    print("\nTest 4: All map adjacencies are bidirectional")
    for name in MapRegistry.list_maps():
        config = MapRegistry.get(name)
        bad = _check_bidirectional(config.adjacency_list)
        assert len(bad) == 0, (
            f"Map '{name}' has non-bidirectional edges: {bad}"
        )
        print(f"  {name}: bidirectional  OK")
    print("  PASSED")


# ---------------------------------------------------------------------------
# Test 5: All region territory IDs are valid
# ---------------------------------------------------------------------------

def test_all_maps_region_ids():
    print("\nTest 5: All region territory IDs are within range")
    for name in MapRegistry.list_maps():
        config = MapRegistry.get(name)
        bad = _check_region_ids(config.regions, config.n_territories)
        assert len(bad) == 0, (
            f"Map '{name}' has out-of-range region IDs: {bad}"
        )
        print(f"  {name}: region IDs valid  OK")
    print("  PASSED")


# ---------------------------------------------------------------------------
# Test 6: initial_ownership assigns balanced territories
# ---------------------------------------------------------------------------

def test_all_maps_balanced_ownership():
    print("\nTest 6: initial_ownership is balanced (n//2 each, or (n+1)//2 vs n//2 for odd n)")
    for name in MapRegistry.list_maps():
        config = MapRegistry.get(name)
        n = config.n_territories
        count_0 = int(np.sum(config.initial_ownership == 0))
        count_1 = int(np.sum(config.initial_ownership == 1))
        assert count_0 + count_1 == n, (
            f"Map '{name}': total ownership {count_0 + count_1} != {n}"
        )
        # Allow at most 1 territory difference for odd-sized maps
        assert abs(count_0 - count_1) <= 1, (
            f"Map '{name}': imbalanced ownership agent_0={count_0} agent_1={count_1}"
        )
        print(f"  {name}: agent_0={count_0}  agent_1={count_1}  OK")
    print("  PASSED")


# ---------------------------------------------------------------------------
# Test 7: Smoke test — instantiate env with each map and run 3 steps
# ---------------------------------------------------------------------------

def test_all_maps_env_smoke():
    print("\nTest 7: ParallelRiskEnv smoke test (3 steps) for all maps")
    for name in MapRegistry.list_maps():
        env = ParallelRiskEnv(map_name=name)
        obs, infos = env.reset()
        assert set(obs.keys()) == {"agent_0", "agent_1"}, (
            f"Map '{name}': unexpected agents in obs"
        )
        for step in range(3):
            actions = make_noop_actions(env)
            obs, rewards, terms, truncs, infos = env.step(actions)
            # If the game ended unexpectedly on step 0-1 something is wrong
            if step < 2:
                assert not all(terms.values()), (
                    f"Map '{name}': game ended prematurely on step {step + 1}"
                )
        print(f"  {name}: 3 steps completed  OK")
    print("  PASSED")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all_tests():
    tests = [
        test_all_maps_load,
        test_basic_6_alias,
        test_all_maps_connected,
        test_all_maps_bidirectional,
        test_all_maps_region_ids,
        test_all_maps_balanced_ownership,
        test_all_maps_env_smoke,
    ]
    passed = 0
    failed = 0
    failures = []
    print("=" * 60)
    print("MAP TESTS")
    print("=" * 60)
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as exc:
            failed += 1
            failures.append((test.__name__, str(exc)))
            print(f"  FAILED: {exc}")
        except Exception as exc:
            failed += 1
            failures.append((test.__name__, str(exc)))
            print(f"  ERROR: {exc}")

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    if failures:
        print("\nFailures:")
        for name, msg in failures:
            print(f"  {name}: {msg}")
    else:
        print("All tests passed!")
    print("=" * 60)
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
