"""Tests for the lockstep driver and the cross-game batched evaluator.

(a) Driver exactness: G=4 ExIt games in lockstep with single-graph forwards
    reproduce the same 4 games played one at a time with the same seeds,
    example for example, and G=2 gives the same games again, so the result
    depends neither on the batch size nor on which games share a round.
(b) Batched evaluator: `GNNEvaluator.evaluate_many` over one mixed-map batch
    gives priors and values within 1e-5 of the single-graph evaluator, its
    collate equals `Batch.from_data_list`, and a budget-30 search served by
    it has the same visit counts as the synchronous search.
(c) Eval games: `play_eval_games_lockstep` matches the sequential eval loop
    exactly in single-graph mode, and with batching runs, alternates colours
    and counts outcomes.
(d) CUDA (skipped without a GPU): evaluate_many forwarding on cuda against
    the CPU single-graph path, printed and bounded.

Uses the K10 PPO checkpoint when it can be found (see _find_checkpoint),
otherwise a seeded random-init policy.
"""

import copy
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch

from parallel_risk import ParallelRiskEnv
from parallel_risk.agents.masked_random_agent import MaskedRandomAgentRLlib
from parallel_risk.agents.mcts_agent import MCTSAgent, RiskSimulator
from parallel_risk.agents.mcts_gnn_agent import GNNEvaluator, MCTSGNNAgent
from parallel_risk.models.action_decoder import ActionDecoder
from parallel_risk.models.gnn_gcn import GCNPolicy
from parallel_risk.training.mcts_gnn.exit_self_play import (
    play_episode_exit, play_episode_exit_gen)
from parallel_risk.training.mcts_gnn.lockstep import (
    build_agent, eval_specs, play_eval_games_lockstep, self_play_game_specs,
    self_play_games_lockstep, summarize_eval_outcomes)

REPO = Path(__file__).resolve().parents[1]
CKPT_REL = 'experiments/k_sweep_ppo_200/K10/checkpoints/final.pt'
CKPT_MAX_REGIONS = 5

SP_CFG = {'dirichlet_alpha': 0.3, 'noise_frac': 0.25, 'dirichlet_min_actions': 4,
          'temperature_turns': 3, 'max_temperature': 1.0}


def _mcts_cfg(action_budget, budget):
    return {'simulation_budget': budget, 'c_puct': 1.4, 'uct_c': 1.41, 'pw_alpha': 0.5,
            'max_rollout_turns': 10, 'action_budget': action_budget,
            'pw_sampler': 'masked_random', 'use_value_fn': True}


def _find_checkpoint():
    """The K10 checkpoint is gitignored: try $PRG_K10_CKPT, this checkout, then the main checkout of a worktree."""
    candidates = [os.environ.get('PRG_K10_CKPT'), REPO / CKPT_REL]
    if REPO.parent.name == 'worktrees' and REPO.parent.parent.name == '.claude':
        candidates.append(REPO.parents[2] / CKPT_REL)
    for c in candidates:
        if c and Path(c).exists():
            return Path(c)
    return None


def _load_policy():
    """(policy on the CPU, action_budget, max_regions, source)."""
    ckpt = _find_checkpoint()
    if ckpt is not None:
        state = torch.load(ckpt, map_location='cpu', weights_only=False)
        policy = GCNPolicy(node_features_dim=3 + CKPT_MAX_REGIONS,
                           global_features_dim=2 + CKPT_MAX_REGIONS,
                           hidden_dim=128, num_layers=3, action_budget=10,
                           max_troops=20, dropout=0.1)
        policy.load_state_dict(state['policy_state_dict'])
        policy.eval()
        return policy, 10, CKPT_MAX_REGIONS, 'checkpoint'
    torch.manual_seed(0)
    policy = GCNPolicy(node_features_dim=3 + 3, global_features_dim=2 + 3, hidden_dim=32,
                       num_layers=2, action_budget=5, max_troops=20, dropout=0.0)
    policy.eval()
    return policy, 5, 3, 'random_init'


# ---------------------------------------------------------------------------
# (a) driver exactness
# ---------------------------------------------------------------------------

def _example_signature(examples):
    return [(ex['map_name'], ex['agent_id'], ex['action_key'], ex['z'],
             ex['graph'].x.numpy().tobytes()) for ex in examples]


def _sequential_exit(policy, action_budget, max_regions, env_configs, mcts_cfg,
                     num_games, base_seed):
    """The same games `self_play_games_lockstep` plays, one at a time on the synchronous path."""
    decoder = ActionDecoder(action_budget=action_budget, max_troops=20)
    max_turns = {e['map_name']: e['max_turns'] for e in env_configs}
    examples = []
    for map_name, seed in self_play_game_specs(list(max_turns), num_games, base_seed, 1):
        env = ParallelRiskEnv(map_name=map_name, max_turns=max_turns[map_name],
                              reward_shaping_config=None)
        agent = build_agent(policy, decoder, env.map_config, mcts_cfg,
                            max_turns[map_name], max_regions)
        examples.extend(play_episode_exit(agent, env, SP_CFG, map_name, seed=seed,
                                          rng=np.random.default_rng(seed)))
    return examples


def test_lockstep_driver_is_exact():
    """G=4 and G=2 lockstep batches (single-graph forwards) == the 4 games played sequentially."""
    print("\n=== Test: lockstep driver == sequential games (single-graph forwards) ===")
    torch.set_num_threads(2)
    policy, action_budget, max_regions, source = _load_policy()
    env_configs = [{'map_name': 'simple_6', 'max_turns': 6},
                   {'map_name': 'medium_8', 'max_turns': 6}]
    mcts_cfg = _mcts_cfg(action_budget, 8)

    ref = _sequential_exit(policy, action_budget, max_regions, env_configs, mcts_cfg, 4, 100)
    ref_sig = _example_signature(ref)
    assert len(ref) > 0
    maps = sorted({ex['map_name'] for ex in ref})
    print(f"  sequential: {len(ref)} examples over 4 games on {maps} ({source})")

    for size in (4, 2):
        stats = {}
        examples = self_play_games_lockstep(
            policy, env_configs, mcts_cfg, SP_CFG, 4, 100, max_regions,
            play_episode_exit_gen, games_per_batch=size, device='cpu',
            map_seed_offset=1, single_graph_forwards=True, stats=stats)
        sig = _example_signature(examples)
        n_same = sum(a == b for a, b in zip(sig, ref_sig))
        print(f"  G={size}: {len(examples)} examples, {n_same} identical to sequential, "
              f"rounds={stats['rounds']}")
        assert sig == ref_sig, f"G={size}: lockstep trajectories differ from sequential"
    print("[OK] lockstep driver reproduces the sequential games exactly")


# ---------------------------------------------------------------------------
# (b) batched evaluator
# ---------------------------------------------------------------------------

def _random_triples(action_budget, map_name, n_states, seed):
    """(env, sim, [(state, agent_id, action_dict)]) along masked-random play."""
    env = ParallelRiskEnv(map_name=map_name, max_turns=50)
    sim = RiskSimulator(env.map_config, max_turns=50)
    sampler = MaskedRandomAgentRLlib(env.map_config.n_territories, env.map_config.adjacency_matrix,
                                     action_budget=action_budget, max_troops=20)
    env.reset(seed=seed)
    np.random.seed(seed)
    state = RiskSimulator.clone_state(env.game_state)
    triples = []
    while len(triples) < 2 * n_states:
        actions = {}
        for agent_id in sim.AGENTS:
            actions[agent_id] = sampler.get_action_raw(sim.state_to_obs(state, agent_id))
            triples.append((state, agent_id, actions[agent_id]))
        state, _, done = sim.step(state, actions)
        if done:
            env.reset(seed=seed + len(triples))
            state = RiskSimulator.clone_state(env.game_state)
    return env, sim, triples


def _drive_batched(mcts, gen, policy, device='cpu'):
    """Run a search generator serving each request through evaluate_many (batched)."""
    try:
        request = next(gen)
        while True:
            GNNEvaluator.evaluate_many([(mcts.evaluator, s, a) for s, a in request], policy, device)
            request = gen.send(None)
    except StopIteration as stop:
        return stop.value


def _root_visits(root):
    return {a: [(k, root.stats[a][k]['n']) for k in root.available_actions[a]]
            for a in ('agent_0', 'agent_1')}


def _root_max_diff(root_a, root_b):
    dq = dp = 0.0
    for a in ('agent_0', 'agent_1'):
        for k in root_a.available_actions[a]:
            dq = max(dq, abs(root_a.stats[a][k]['q'] - root_b.stats[a][k]['q']))
            dp = max(dp, abs(root_a.stats[a][k]['p'] - root_b.stats[a][k]['p']))
    return dq, dp


def test_evaluate_many_matches_single_path():
    """One mixed-map batched forward vs the single-graph evaluator, then a budget-30 search."""
    print("\n=== Test: evaluate_many (mixed maps, one forward) vs single-graph evaluator ===")
    torch.set_num_threads(2)
    policy, action_budget, max_regions, source = _load_policy()
    decoder = ActionDecoder(action_budget=action_budget, max_troops=20)

    batched_evs, single_evs, all_triples = {}, {}, []
    for i, map_name in enumerate(('simple_6', 'medium_8')):
        env, sim, triples = _random_triples(action_budget, map_name, 12, 31 + i)
        for evs in (batched_evs, single_evs):
            evs[map_name] = GNNEvaluator(policy, decoder, env.map_config, sim, action_budget,
                                         'cpu', max_regions)
        all_triples.extend((map_name, s, a, act) for s, a, act in triples)

    requests = [(batched_evs[m], s, a) for m, s, a, _ in all_triples]
    unique = len({(m, GNNEvaluator._key(s, a)) for m, s, a, _ in all_triples})
    n_forwarded = GNNEvaluator.evaluate_many(requests, policy, 'cpu')
    assert n_forwarded == unique, (n_forwarded, unique)
    assert GNNEvaluator.evaluate_many(requests, policy, 'cpu') == 0, "second call must hit the cache"
    print(f"  {len(requests)} requests on 2 maps -> {n_forwarded} graphs in one forward ({source})")

    graphs = [batched_evs[m]._graph(s, a) for m, s, a, _ in all_triples[:7]]
    ref, col = Batch.from_data_list(graphs), GNNEvaluator._collate(graphs, 'cpu')
    for name in ('x', 'edge_index', 'global_features', 'batch'):
        assert torch.equal(getattr(ref, name), getattr(col, name)), f"collate differs on {name}"

    max_dp = max_dv = 0.0
    for m, s, a, act in all_triples:
        max_dp = max(max_dp, abs(batched_evs[m].policy_fn(s, a, act) - single_evs[m].policy_fn(s, a, act)))
        max_dv = max(max_dv, abs(batched_evs[m].value_fn(s, a) - single_evs[m].value_fn(s, a)))
    print(f"  batched vs single: max|dlogprob|={max_dp:.2e} max|dvalue|={max_dv:.2e}")
    assert max_dp < 1e-5 and max_dv < 1e-5, (max_dp, max_dv)
    for m in batched_evs:
        assert batched_evs[m].n_forwards == 1

    # Budget-30 searches: same visit counts whether requests are served batched or one by one.
    for map_name, seed in (('simple_6', 41), ('medium_8', 42)):
        env = ParallelRiskEnv(map_name=map_name, max_turns=50)
        cfg = _mcts_cfg(action_budget, 30)
        sync_agent = build_agent(policy, decoder, env.map_config, cfg, 50, max_regions)
        batch_agent = build_agent(policy, decoder, env.map_config, cfg, 50, max_regions)
        sim, sampler = batch_agent.mcts.sim, batch_agent.mcts.samplers['agent_0']
        env.reset(seed=seed)
        np.random.seed(seed)
        for _ in range(3):
            env.step({a: sampler.get_action_raw(sim.state_to_obs(env.game_state, a))
                      for a in sim.AGENTS})
        state = RiskSimulator.clone_state(env.game_state)

        np.random.seed(seed)
        root_sync = sync_agent.mcts.make_root(state)
        sync_agent.mcts.run(root_sync, 30)
        np.random.seed(seed)
        root_batch = _drive_batched(batch_agent.mcts, batch_agent.mcts.make_root_gen(state), policy)
        _drive_batched(batch_agent.mcts, batch_agent.mcts.search_gen(root_batch, 30), policy)

        assert _root_visits(root_sync) == _root_visits(root_batch), f"{map_name}: visit counts differ"
        dq, dp = _root_max_diff(root_sync, root_batch)
        ev = batch_agent.evaluator
        print(f"  {map_name} budget 30: identical visit counts over "
              f"{sum(len(root_batch.stats[a]) for a in ('agent_0', 'agent_1'))} actions, "
              f"max|dq|={dq:.1e} max|dp|={dp:.1e}, {ev.n_forwards} batched forwards / "
              f"{ev.n_graphs} graphs vs {sync_agent.evaluator.n_forwards} single forwards")
        assert ev.n_forwards < sync_agent.evaluator.n_forwards
    print("[OK] batched evaluator matches the single-graph path")


# ---------------------------------------------------------------------------
# (c) eval games
# ---------------------------------------------------------------------------

def _sequential_eval(policy, action_budget, max_regions, specs, max_turns, budget):
    """The eval loop of the experiment scripts, one game at a time."""
    decoder = ActionDecoder(action_budget=action_budget, max_troops=20)
    outcomes = []
    for map_name, seed, gnn_id in specs:
        env = ParallelRiskEnv(map_name=map_name, max_turns=max_turns, reward_shaping_config=None)
        gnn = MCTSGNNAgent(policy=policy, decoder=decoder, map_config=env.map_config,
                           simulation_budget=budget, c_puct=1.4, action_budget=action_budget,
                           max_turns=max_turns, device='cpu', max_regions=max_regions)
        uniform = MCTSAgent.from_env(env, simulation_budget=budget, action_budget=action_budget)
        obs, _ = env.reset(seed=seed)
        done, turns = False, 0
        while not done:
            actions = {}
            for aid in ('agent_0', 'agent_1'):
                if aid not in obs:
                    continue
                agent = gnn if aid == gnn_id else uniform
                actions[aid] = agent.get_action(env.game_state, aid)
            obs, rewards, terms, truncs, _ = env.step(actions)
            done = terms.get('__all__', False) or truncs.get('__all__', False)
            turns += 1
        opp = 'agent_1' if gnn_id == 'agent_0' else 'agent_0'
        r_g, r_o = float(rewards.get(gnn_id, 0.0)), float(rewards.get(opp, 0.0))
        result = 'win' if r_g > r_o else ('loss' if r_g < r_o else 'draw')
        outcomes.append((map_name, seed, gnn_id, result, turns))
    return outcomes


def test_eval_games_lockstep():
    """Eval games: exact in single-graph mode, and batched games run with alternating colours."""
    print("\n=== Test: eval games (MCTS+GNN vs MCTS-uniform) in lockstep ===")
    torch.set_num_threads(2)
    policy, action_budget, max_regions, source = _load_policy()
    specs = eval_specs(['simple_6', 'medium_8'], 2, base_seed=7)
    max_turns, budget = 5, 6

    ref = _sequential_eval(policy, action_budget, max_regions, specs, max_turns, budget)
    exact = play_eval_games_lockstep(policy, specs, max_turns=max_turns, action_budget=action_budget,
                                     mcts_budget=budget, max_regions=max_regions, games_per_batch=4,
                                     single_graph_forwards=True)
    got = [(o['map_name'], o['seed'], o['gnn_agent_id'], o['result'], o['length']) for o in exact]
    print(f"  sequential: {[(r[0], r[2], r[3], r[4]) for r in ref]}")
    assert got == ref, f"lockstep eval (single-graph) differs from sequential:\n{got}\n{ref}"

    stats = {}
    batched = play_eval_games_lockstep(policy, specs, max_turns=max_turns, action_budget=action_budget,
                                       mcts_budget=budget, max_regions=max_regions, games_per_batch=4,
                                       stats=stats)
    summary = summarize_eval_outcomes(batched)
    assert sum(v['total'] for v in summary.values()) == len(specs)
    assert [o['gnn_agent_id'] for o in batched] == ['agent_0', 'agent_1', 'agent_0', 'agent_1']
    for o in batched:
        assert o['result'] in ('win', 'loss', 'draw') and o['length'] >= 1
    assert stats['graphs_per_forward'] > 1.0, stats
    print(f"  batched: {[(o['map_name'], o['gnn_agent_id'], o['result'], o['length']) for o in batched]}")
    print(f"  summary: {summary}")
    print(f"  {stats['forwards']} batched forwards, {stats['graphs_per_forward']:.1f} graphs each")
    print("[OK] eval games run in lockstep and count outcomes")


# ---------------------------------------------------------------------------
# (d) CUDA forwards
# ---------------------------------------------------------------------------

def test_evaluate_many_cuda():
    """evaluate_many forwarding on CUDA (entries on the CPU) vs the CPU single-graph path."""
    print("\n=== Test: evaluate_many on CUDA vs CPU single-graph evaluator ===")
    if not torch.cuda.is_available():
        print("[OK] skipped (no CUDA)")
        return
    torch.set_num_threads(2)
    policy, action_budget, max_regions, source = _load_policy()
    policy_cuda = copy.deepcopy(policy).cuda().eval()
    decoder = ActionDecoder(action_budget=action_budget, max_troops=20)
    cuda_evs, single_evs, all_triples = {}, {}, []
    for i, map_name in enumerate(('simple_6', 'medium_8')):
        env, sim, triples = _random_triples(action_budget, map_name, 12, 51 + i)
        for evs in (cuda_evs, single_evs):
            evs[map_name] = GNNEvaluator(policy, decoder, env.map_config, sim, action_budget,
                                         'cpu', max_regions)
        all_triples.extend((map_name, s, a, act) for s, a, act in triples)
    n = GNNEvaluator.evaluate_many([(cuda_evs[m], s, a) for m, s, a, _ in all_triples],
                                   policy_cuda, 'cuda')
    max_dp = max_dv = 0.0
    for m, s, a, act in all_triples:
        entry = cuda_evs[m].evaluate(s, a)
        assert entry.batched.x.device.type == 'cpu' and entry.value is not None
        max_dp = max(max_dp, abs(cuda_evs[m].policy_fn(s, a, act) - single_evs[m].policy_fn(s, a, act)))
        max_dv = max(max_dv, abs(cuda_evs[m].value_fn(s, a) - single_evs[m].value_fn(s, a)))
    print(f"  {n} graphs forwarded on cuda (tf32 matmul={torch.backends.cuda.matmul.allow_tf32}, "
          f"precision={torch.get_float32_matmul_precision()}): "
          f"max|dlogprob|={max_dp:.2e} max|dvalue|={max_dv:.2e} vs CPU single-graph ({source})")
    assert max_dp < 1e-3 and max_dv < 1e-3, (max_dp, max_dv)
    print("[OK] CUDA batched forward agrees with the CPU path")


def run_all_tests():
    print("=" * 70)
    print("Lockstep driver tests")
    print("=" * 70)
    passed = failed = 0
    tests = [
        test_lockstep_driver_is_exact,
        test_evaluate_many_matches_single_path,
        test_eval_games_lockstep,
        test_evaluate_many_cuda,
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
