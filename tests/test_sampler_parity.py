"""
RNG-stream parity test: the fast MaskedRandomAgentRLlib must match the legacy
mask-based loop (parallel_risk/agents/_masked_random_legacy.py) exactly.

For every registered map, K in (5, 10), 300 observations reached by random
play, both agents, and seeds (step, 1000 + step):
  - get_action(): same tuples, same scalar types per element
  - get_action_raw(): same 'actions' array (values, dtype, shape), same num_actions
  - np.random.get_state() identical after each call (same number of draws
    consumed, so MCTS rollouts stay bit-identical)

Plus the fallback branches (no owned territory, no valid troop count, the
max_troops cap) on synthetic observations.

If this test fails, DO NOT ship the fast sampler: MCTS-uniform decisions and
every golden trajectory would silently change.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from parallel_risk import ParallelRiskEnv
from parallel_risk.env.map_config import MapRegistry
from parallel_risk.agents.masked_random_agent import MaskedRandomAgentRLlib
from parallel_risk.agents._masked_random_legacy import _MaskedRandomAgentRLlibLegacy

N_OBSERVATIONS = 300
BUDGETS = (5, 10)


def rng_state():
    name, key, pos, has_gauss, cached_gaussian = np.random.get_state()
    return name, key.tobytes(), pos, has_gauss, cached_gaussian


def scalar_types(actions):
    return [type(value) for action in actions for value in action]


def check_pair(legacy, fast, obs, seed, label):
    """Both entry points, same seed: actions, scalar types and RNG state must match."""
    np.random.seed(seed)
    a_legacy = legacy.get_action(obs)
    s_legacy = rng_state()
    np.random.seed(seed)
    a_fast = fast.get_action(obs)
    s_fast = rng_state()
    assert a_legacy == a_fast, f"{label} seed={seed}: get_action differs\n{a_legacy}\nvs\n{a_fast}"
    assert scalar_types(a_legacy) == scalar_types(a_fast), (
        f"{label} seed={seed}: scalar types differ\n{scalar_types(a_legacy)}\nvs\n{scalar_types(a_fast)}"
    )
    assert s_legacy == s_fast, f"{label} seed={seed}: np.random state differs after get_action"

    np.random.seed(seed)
    r_legacy = legacy.get_action_raw(obs)
    s_legacy = rng_state()
    np.random.seed(seed)
    r_fast = fast.get_action_raw(obs)
    s_fast = rng_state()
    assert r_legacy['num_actions'] == r_fast['num_actions'], f"{label}: num_actions differs"
    assert r_legacy['actions'].dtype == r_fast['actions'].dtype, (
        f"{label}: dtype {r_legacy['actions'].dtype} vs {r_fast['actions'].dtype}"
    )
    assert r_legacy['actions'].shape == r_fast['actions'].shape, f"{label}: shape differs"
    assert np.array_equal(r_legacy['actions'], r_fast['actions']), (
        f"{label} seed={seed}: get_action_raw differs\n{r_legacy['actions']}\nvs\n{r_fast['actions']}"
    )
    assert s_legacy == s_fast, f"{label} seed={seed}: np.random state differs after get_action_raw"


def check_map(map_name, action_budget):
    env = ParallelRiskEnv(map_name=map_name, max_actions_per_turn=max(10, action_budget))
    legacy = _MaskedRandomAgentRLlibLegacy.from_env(env, action_budget=action_budget)
    fast = MaskedRandomAgentRLlib.from_env(env, action_budget=action_budget)
    obs, _ = env.reset(seed=3)
    n_pairs = 0
    for step in range(N_OBSERVATIONS):
        for agent in env.agents:
            for seed in (step, 1000 + step):
                check_pair(legacy, fast, obs[agent], seed, f"{map_name} K={action_budget} {agent} step={step}")
                n_pairs += 1
        actions = {agent: legacy.get_action_raw(obs[agent]) for agent in env.agents}
        obs, _, terms, truncs, _ = env.step(actions)
        if terms['__all__'] or truncs['__all__']:
            obs, _ = env.reset(seed=step + 7)
    return n_pairs


def test_all_maps_random_play():
    total = 0
    for map_name in MapRegistry.list_maps():
        for action_budget in BUDGETS:
            n_pairs = check_map(map_name, action_budget)
            total += n_pairs
            print(f"[OK] {map_name:16s} K={action_budget:2d}: {n_pairs} (obs, seed) pairs identical "
                  f"(actions, scalar types, RNG state)")
    print(f"[OK] total {total} (obs, seed) pairs across {len(MapRegistry.list_maps())} maps")


def test_fallback_branches():
    """Synthetic observations that hit every fallback of the legacy loop."""
    env = ParallelRiskEnv(map_name='simple_6')
    obs, _ = env.reset(seed=0)
    base = obs['agent_0']
    cases = {
        'no_owned_territory': {
            **base, 'territory_ownership': np.full_like(base['territory_ownership'], -1)},
        'no_valid_troops': {
            **base, 'territory_troops': np.ones_like(base['territory_troops']),
            'available_income': np.zeros_like(base['available_income'])},
        'max_troops_cap': {
            **base, 'territory_troops': np.full_like(base['territory_troops'], 500),
            'available_income': np.array([50], dtype=base['available_income'].dtype)},
        'no_adjacency_key': {k: v for k, v in base.items() if k != 'adjacency_matrix'},
    }
    n_pairs = 0
    for name, case_obs in cases.items():
        for max_troops in (1, 3, 20):
            for action_budget in BUDGETS:
                legacy = _MaskedRandomAgentRLlibLegacy.from_env(env, action_budget=action_budget, max_troops=max_troops)
                fast = MaskedRandomAgentRLlib.from_env(env, action_budget=action_budget, max_troops=max_troops)
                for seed in range(20):
                    check_pair(legacy, fast, case_obs, seed, f"{name} max_troops={max_troops} K={action_budget}")
                    n_pairs += 1
    print(f"[OK] fallback branches: {n_pairs} (obs, seed) pairs identical")


def test_zero_budget():
    env = ParallelRiskEnv(map_name='simple_6')
    obs, _ = env.reset(seed=0)
    legacy = _MaskedRandomAgentRLlibLegacy.from_env(env, action_budget=0)
    fast = MaskedRandomAgentRLlib.from_env(env, action_budget=0)
    check_pair(legacy, fast, obs['agent_0'], 0, "K=0")
    print("[OK] K=0 returns an empty tuple / all-zero array without touching np.random")


if __name__ == "__main__":
    print("Testing MaskedRandomAgentRLlib fast vs legacy parity...")
    try:
        test_zero_budget()
        test_fallback_branches()
        test_all_maps_random_play()
    except AssertionError as exc:
        print(f"[FAIL] {exc}")
        sys.exit(1)
    print("\nAll sampler parity tests passed.")
