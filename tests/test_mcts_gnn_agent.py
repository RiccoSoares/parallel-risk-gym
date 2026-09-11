"""Smoke tests for MCTSGNNAgent and DuctMCTS PUCT branch.

Verifies that:
1. MCTSGNNAgent runs end-to-end with a randomly-initialized GNN.
2. The action it returns is well-formed and steps the environment cleanly.
3. Vanilla MCTSAgent (policy_fn=None) still works — no regression in the
   UCT path from the PUCT changes to DuctMCTS.
4. The per-search GNNEvaluator cache returns bit-identical priors and values
   to an uncached reference, and a budget-50 search builds the same tree.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch

from parallel_risk import ParallelRiskEnv
from parallel_risk.agents.mcts_agent import MCTSAgent, RiskSimulator
from parallel_risk.agents.mcts_gnn_agent import MCTSGNNAgent
from parallel_risk.models.action_decoder import ActionDecoder
from parallel_risk.models.gnn_gcn import GCNPolicy
from parallel_risk.training.torchrl.graph_wrapper import env_to_graph

# Fixed checkpoint for the cache-equivalence test; falls back to a seeded
# random-init policy when it is not on disk (it is not tracked by git).
K10_CKPT = Path(__file__).resolve().parents[1] / 'experiments/k_sweep_ppo_200/K10/checkpoints/final.pt'
K10_MAX_REGIONS = 5


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


def _uncached_reference(agent):
    """The value_fn / policy_fn closures MCTSGNNAgent used before GNNEvaluator.

    Rebuilds the graph and runs one forward per call, exactly as the old code
    did (Batch.from_data_list + compute_log_probs(observations=[graph])).
    """
    policy, decoder, map_config = agent.policy, agent.decoder, agent.map_config
    device, max_regions, action_budget = agent.device, agent.max_regions, agent.action_budget
    simulator = agent.mcts.sim

    def value_fn(game_state, agent_id):
        obs = simulator.state_to_obs(game_state, agent_id)
        graph = env_to_graph(obs, map_config, device, max_regions=max_regions)
        batched = Batch.from_data_list([graph])
        with torch.no_grad():
            _, values, _ = policy(batched)
        return float(values.squeeze().item())

    def policy_fn(game_state, agent_id, action_dict):
        obs = simulator.state_to_obs(game_state, agent_id)
        graph = env_to_graph(obs, map_config, device, max_regions=max_regions)
        batched = Batch.from_data_list([graph])
        actions_tensor = torch.as_tensor(
            action_dict['actions'][:action_budget], dtype=torch.long, device=device
        ).unsqueeze(0)
        with torch.no_grad():
            action_logits, _, _ = policy(batched)
            log_probs = decoder.compute_log_probs(
                action_logits, actions_tensor, batched.batch, observations=[graph])
        return float(log_probs.sum().item())

    return value_fn, policy_fn


def _build_equivalence_agent(env, budget, **kwargs):
    if K10_CKPT.exists():
        agent = MCTSGNNAgent.from_checkpoint(
            str(K10_CKPT), env.map_config, simulation_budget=budget, device='cpu',
            max_regions=K10_MAX_REGIONS, max_turns=env.max_turns, **kwargs)
        return agent, 'checkpoint'
    torch.manual_seed(0)
    policy = _build_random_policy(env, action_budget=5)
    decoder = ActionDecoder(action_budget=5, max_troops=20)
    agent = MCTSGNNAgent(policy=policy, decoder=decoder, map_config=env.map_config,
                         simulation_budget=budget, action_budget=5,
                         max_turns=env.max_turns, device='cpu', **kwargs)
    return agent, 'random_init'


def _sample_triples(agent, env, n_states, seed):
    """(game_state, agent_id, action_dict) triples along a masked-random game."""
    sim = agent.mcts.sim
    np.random.seed(seed)
    env.reset(seed=seed)
    state = RiskSimulator.clone_state(env.game_state)
    triples = []
    while len(triples) < 2 * n_states:
        actions = {}
        for aid in sim.AGENTS:
            actions[aid] = agent.mcts.samplers[aid].get_action_raw(sim.state_to_obs(state, aid))
            triples.append((state, aid, actions[aid]))
        state, _, done = sim.step(state, actions)
        if done:
            env.reset(seed=seed + len(triples))
            state = RiskSimulator.clone_state(env.game_state)
    return triples


def _root_stats(root):
    return {a: {k: (s['n'], s['q'], s['p']) for k, s in root.stats[a].items()}
            for a in ('agent_0', 'agent_1')}


def test_gnn_evaluator_matches_uncached_reference():
    """Cached priors/values are bit-identical to the uncached closures over 100+ triples."""
    print("\n=== Test: GNNEvaluator == uncached reference (priors, values) ===")
    torch.set_num_threads(2)
    env = ParallelRiskEnv(map_name="medium_8", max_turns=50)
    agent, source = _build_equivalence_agent(env, budget=50)
    ref_value_fn, ref_policy_fn = _uncached_reference(agent)
    evaluator = agent.evaluator
    evaluator.new_search()

    triples = _sample_triples(agent, env, n_states=50, seed=11)
    assert len(triples) >= 100
    max_dp = max_dv = 0.0
    for state, aid, action_dict in triples:
        p_ref = ref_policy_fn(state, aid, action_dict)
        v_ref = ref_value_fn(state, aid)
        p_new = evaluator.policy_fn(state, aid, action_dict)
        v_new = evaluator.value_fn(state, aid)
        max_dp = max(max_dp, abs(p_ref - p_new))
        max_dv = max(max_dv, abs(v_ref - v_new))
        assert p_new == p_ref, f"prior differs: cached {p_new!r} vs reference {p_ref!r}"
        assert v_new == v_ref, f"value differs: cached {v_new!r} vs reference {v_ref!r}"
    print(f"  {len(triples)} triples ({source}): max|dlogprob|={max_dp:.1e} max|dvalue|={max_dv:.1e}")
    # One forward per (state, agent): 2 calls per triple, forwards == misses.
    assert evaluator.n_forwards == len(triples), (evaluator.n_forwards, len(triples))
    assert evaluator.n_hits == len(triples)
    # Re-asking the same triples is served entirely from the cache.
    before = evaluator.n_forwards
    for state, aid, action_dict in triples:
        evaluator.policy_fn(state, aid, action_dict)
        evaluator.value_fn(state, aid)
    assert evaluator.n_forwards == before
    evaluator.new_search()
    assert evaluator.n_forwards == 0 and evaluator.n_hits == 0

    # Paired forward (batch of 2 per state) is not bit-identical: check it stays close.
    paired, _ = _build_equivalence_agent(env, budget=50, pair_agents=True)
    max_dp = max_dv = 0.0
    for state, aid, action_dict in triples:
        max_dp = max(max_dp, abs(ref_policy_fn(state, aid, action_dict) - paired.evaluator.policy_fn(state, aid, action_dict)))
        max_dv = max(max_dv, abs(ref_value_fn(state, aid) - paired.evaluator.value_fn(state, aid)))
    print(f"  pair_agents=True: max|dlogprob|={max_dp:.1e} max|dvalue|={max_dv:.1e} "
          f"forwards={paired.evaluator.n_forwards} for {len(triples)} (state, agent) pairs")
    assert paired.evaluator.n_forwards == len(triples) // 2
    assert max_dp < 1e-4 and max_dv < 1e-5
    env.close()
    print("[OK] GNNEvaluator matches the uncached reference")


def test_gnn_evaluator_search_is_identical():
    """A budget-50 search yields the same root visit counts and best action with and without the cache."""
    print("\n=== Test: cached vs uncached search (budget 50) ===")
    torch.set_num_threads(2)
    budget = 50
    for map_name, seed in (("simple_6", 21), ("medium_8", 22)):
        env = ParallelRiskEnv(map_name=map_name, max_turns=50)
        agent, source = _build_equivalence_agent(env, budget=budget)
        env.reset(seed=seed)
        if map_name == "medium_8":
            # a mid-game state: play a few masked-random turns first
            sim = agent.mcts.sim
            np.random.seed(seed)
            for _ in range(4):
                acts = {a: agent.mcts.samplers[a].get_action_raw(sim.state_to_obs(env.game_state, a))
                        for a in sim.AGENTS}
                env.step(acts)
        state = RiskSimulator.clone_state(env.game_state)

        cached_value_fn, cached_policy_fn = agent.mcts.value_fn, agent.mcts.policy_fn
        ref_value_fn, ref_policy_fn = _uncached_reference(agent)

        agent.mcts.value_fn, agent.mcts.policy_fn = ref_value_fn, ref_policy_fn
        np.random.seed(seed)
        root_ref = agent.mcts.make_root(state)
        agent.mcts.run(root_ref, budget)
        best_ref = agent.mcts.best_action(root_ref, 'agent_0')

        agent.mcts.value_fn, agent.mcts.policy_fn = cached_value_fn, cached_policy_fn
        np.random.seed(seed)
        root_new = agent.mcts.make_root(state)
        agent.mcts.run(root_new, budget)
        best_new = agent.mcts.best_action(root_new, 'agent_0')

        assert root_ref.visit_count == root_new.visit_count == budget
        for a in ('agent_0', 'agent_1'):
            assert root_ref.available_actions[a] == root_new.available_actions[a], f"{map_name}: {a} action sets differ"
        assert _root_stats(root_ref) == _root_stats(root_new), f"{map_name}: root (n, q, p) differ"
        assert np.array_equal(best_ref['actions'], best_new['actions'])
        ev = agent.evaluator
        print(f"  {map_name} ({source}): identical root stats over "
              f"{sum(len(root_new.stats[a]) for a in ('agent_0', 'agent_1'))} actions; "
              f"forwards {ev.n_forwards} vs {ev.n_hits + ev.n_misses} network queries")
        assert ev.n_forwards < ev.n_hits + ev.n_misses
        env.close()
    print("[OK] cached search identical to uncached search")


def run_all_tests():
    print("=" * 70)
    print("MCTSGNNAgent tests")
    print("=" * 70)
    passed = failed = 0
    tests = [
        test_mcts_gnn_agent_executes,
        test_ductmcts_without_policy_fn_uses_uct,
        test_gnn_evaluator_matches_uncached_reference,
        test_gnn_evaluator_search_is_identical,
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
