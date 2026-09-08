"""Smoke tests for MCTSGNNAgent and DuctMCTS PUCT branch.

Verifies that:
1. MCTSGNNAgent runs end-to-end with a randomly-initialized GNN.
2. The action it returns is well-formed and steps the environment cleanly.
3. Vanilla MCTSAgent (policy_fn=None) still works — no regression in the
   UCT path from the PUCT changes to DuctMCTS.
"""

import sys

import numpy as np
import torch

from parallel_risk import ParallelRiskEnv
from parallel_risk.agents.mcts_agent import MCTSAgent
from parallel_risk.agents.mcts_gnn_agent import MCTSGNNAgent
from parallel_risk.models.action_decoder import ActionDecoder
from parallel_risk.models.gnn_gcn import GCNPolicy


def _build_random_policy(env, action_budget=5):
    """Instantiate a fresh GCNPolicy sized for the given env.

    Uses the same node/global feature-dim formula as `GNNAgent.from_checkpoint`
    so shapes line up with what `env_to_graph` produces post-batching.
    """
    n_regions = len(env.map_config.regions)
    node_features_dim = 3 + n_regions
    global_features_dim = 2 + n_regions
    return GCNPolicy(
        node_features_dim=node_features_dim,
        global_features_dim=global_features_dim,
        hidden_dim=32,
        num_layers=2,
        action_budget=action_budget,
        max_troops=20,
        dropout=0.0,
    )


def test_mcts_gnn_agent_executes():
    """MCTSGNNAgent with a random-init GNN completes get_action + one env step."""
    print("\n=== Test: MCTSGNNAgent executes end-to-end ===")

    torch.manual_seed(0)
    np.random.seed(0)

    env = ParallelRiskEnv(map_name="simple_6", seed=42)
    env.reset(seed=42)

    action_budget = 5
    policy = _build_random_policy(env, action_budget=action_budget)
    decoder = ActionDecoder(action_budget=action_budget, max_troops=20)

    agent = MCTSGNNAgent(
        policy=policy,
        decoder=decoder,
        map_config=env.map_config,
        simulation_budget=8,          # small budget: forces PUCT + PW to fire, still fast
        c_puct=1.4,
        action_budget=action_budget,
        max_turns=env.max_turns,
        device='cpu',
    )
    print("  Built MCTSGNNAgent (budget=8, random-init GNN)")

    action = agent.get_action(env.game_state, 'agent_0')
    print(f"  Got action from MCTS: num_actions={action['num_actions']}")

    assert isinstance(action, dict), f"expected dict, got {type(action)}"
    assert action['num_actions'] == action_budget, (
        f"expected num_actions={action_budget}, got {action['num_actions']}"
    )
    assert isinstance(action['actions'], np.ndarray), "actions must be an ndarray"
    assert action['actions'].shape == (10, 3), f"shape {action['actions'].shape} != (10, 3)"
    assert action['actions'].dtype == np.int32, f"dtype {action['actions'].dtype} != int32"

    # Feed to the env — validates that indices/troop counts fit the action space.
    opponent = {'num_actions': 0, 'actions': np.zeros((10, 3), dtype=np.int32)}
    obs, rewards, terms, truncs, infos = env.step({
        'agent_0': action,
        'agent_1': opponent,
    })
    print(f"  env.step succeeded  |  rewards={rewards}")

    env.close()
    print("[OK] MCTSGNNAgent runs end-to-end")


def test_ductmcts_without_policy_fn_uses_uct():
    """MCTSAgent (no policy_fn) still runs — guards against PUCT-branch regressions."""
    print("\n=== Test: MCTSAgent (vanilla UCT branch) still runs ===")

    env = ParallelRiskEnv(map_name="simple_6", seed=42)
    env.reset(seed=42)

    agent = MCTSAgent.from_env(env, simulation_budget=8)
    print("  Built vanilla MCTSAgent (policy_fn=None -> UCT path)")

    action = agent.get_action(env.game_state, 'agent_0')
    assert isinstance(action, dict), f"expected dict, got {type(action)}"
    assert 'num_actions' in action and 'actions' in action
    print(f"  Got action from vanilla MCTS: num_actions={action['num_actions']}")

    opponent = {'num_actions': 0, 'actions': np.zeros((10, 3), dtype=np.int32)}
    obs, rewards, terms, truncs, infos = env.step({
        'agent_0': action,
        'agent_1': opponent,
    })
    print(f"  env.step succeeded  |  rewards={rewards}")

    env.close()
    print("[OK] Vanilla UCT branch unaffected by PUCT additions")


def run_all_tests():
    print("=" * 70)
    print("MCTSGNNAgent tests")
    print("=" * 70)
    passed = failed = 0
    tests = [
        test_mcts_gnn_agent_executes,
        test_ductmcts_without_policy_fn_uses_uct,
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
