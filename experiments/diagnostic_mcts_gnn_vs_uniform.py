"""Diagnostic: is MCTS+GNN structurally disadvantaged vs MCTS(uniform)?

Runs three head-to-head comparisons on `simple_6` at budget=40:
  (a) MCTS(uniform)  vs MCTS(uniform)              — sanity, ~50% expected
  (b) MCTS+GNN(RANDOM-INIT weights) vs MCTS(uniform) — architecture baseline
  (c) MCTS+GNN(TRAINED weights)     vs MCTS(uniform) — what training delivered

If (b) is already ~0%, the loss is a design/architecture issue at low budget,
not a training-data issue. If (b) is ~50% and (c) is ~0%, training is
actively hurting priors.
"""

from __future__ import annotations

import torch  # noqa: F401 — force pre-load before RLlib chain

import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def _uniform_vs_uniform_worker(args):
    map_name, max_turns, action_budget, budget, num_games, seed = args
    import numpy as np, torch
    torch.set_num_threads(1)
    np.random.seed(seed); torch.manual_seed(seed)
    from parallel_risk import ParallelRiskEnv
    from parallel_risk.agents.mcts_agent import MCTSAgent
    env = ParallelRiskEnv(map_name=map_name, max_turns=max_turns, reward_shaping_config=None)
    a = MCTSAgent.from_env(env, simulation_budget=budget, action_budget=action_budget)
    b = MCTSAgent.from_env(env, simulation_budget=budget, action_budget=action_budget)
    w = l = d = 0
    for g in range(num_games):
        a_plays_0 = (g % 2 == 0)
        obs, _ = env.reset(seed=seed + g)
        done = False
        while not done:
            actions = {}
            for aid in ('agent_0', 'agent_1'):
                if aid not in obs: continue
                pick_a = (aid == 'agent_0') == a_plays_0
                actions[aid] = (a if pick_a else b).get_action(env.game_state, aid)
            obs, rewards, terms, truncs, _ = env.step(actions)
            done = terms.get('__all__', False) or truncs.get('__all__', False)
        r_a = rewards.get('agent_0' if a_plays_0 else 'agent_1', 0.0)
        r_b = rewards.get('agent_1' if a_plays_0 else 'agent_0', 0.0)
        if r_a > r_b: w += 1
        elif r_a < r_b: l += 1
        else: d += 1
    return {'wins': w, 'losses': l, 'draws': d, 'total': w + l + d}


def _gnn_vs_uniform_worker(args):
    (checkpoint_path, model_kwargs_or_none, map_name, max_turns, action_budget,
     budget, num_games, seed, max_regions) = args
    import numpy as np, torch
    torch.set_num_threads(1)
    np.random.seed(seed); torch.manual_seed(seed)
    from parallel_risk import ParallelRiskEnv
    from parallel_risk.agents.mcts_agent import MCTSAgent
    from parallel_risk.agents.mcts_gnn_agent import MCTSGNNAgent
    from parallel_risk.models.action_decoder import ActionDecoder
    from parallel_risk.models.gnn_gcn import GCNPolicy

    env = ParallelRiskEnv(map_name=map_name, max_turns=max_turns, reward_shaping_config=None)

    if checkpoint_path is None:
        # Random-init: build a fresh GCNPolicy with the given kwargs (or defaults)
        mk = model_kwargs_or_none or {}
        n_regions = len(env.map_config.regions)
        policy = GCNPolicy(
            node_features_dim=3 + (max_regions or n_regions),
            global_features_dim=2 + (max_regions or n_regions),
            hidden_dim=mk.get('hidden_dim', 128),
            num_layers=mk.get('num_layers', 3),
            action_budget=action_budget,
            max_troops=20,
            dropout=0.0,
        )
        policy.eval()
        decoder = ActionDecoder(action_budget=action_budget, max_troops=20)
        gnn_agent = MCTSGNNAgent(
            policy=policy, decoder=decoder, map_config=env.map_config,
            simulation_budget=budget, c_puct=1.4, action_budget=action_budget,
            max_turns=max_turns, device='cpu', max_regions=max_regions,
        )
    else:
        gnn_agent = MCTSGNNAgent.from_checkpoint(
            checkpoint_path, env.map_config,
            simulation_budget=budget, device='cpu', max_regions=max_regions,
        )

    uniform = MCTSAgent.from_env(env, simulation_budget=budget, action_budget=action_budget)

    w = l = d = 0
    for g in range(num_games):
        gnn_plays_0 = (g % 2 == 0)
        obs, _ = env.reset(seed=seed + g)
        done = False
        while not done:
            actions = {}
            for aid in ('agent_0', 'agent_1'):
                if aid not in obs: continue
                pick_gnn = (aid == 'agent_0') == gnn_plays_0
                if pick_gnn:
                    actions[aid] = gnn_agent.get_action(env.game_state, aid)
                else:
                    actions[aid] = uniform.get_action(env.game_state, aid)
            obs, rewards, terms, truncs, _ = env.step(actions)
            done = terms.get('__all__', False) or truncs.get('__all__', False)
        r_g = rewards.get('agent_0' if gnn_plays_0 else 'agent_1', 0.0)
        r_u = rewards.get('agent_1' if gnn_plays_0 else 'agent_0', 0.0)
        if r_g > r_u: w += 1
        elif r_g < r_u: l += 1
        else: d += 1
    return {'wins': w, 'losses': l, 'draws': d, 'total': w + l + d}


def run_parallel(worker_fn, args_per_worker, num_workers):
    ctx = mp.get_context('spawn')
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as ex:
        results = list(ex.map(worker_fn, args_per_worker))
    w = sum(r['wins'] for r in results)
    l = sum(r['losses'] for r in results)
    d = sum(r['draws'] for r in results)
    return {'wins': w, 'losses': l, 'draws': d, 'total': w + l + d,
            'win_rate': w / max(w + l + d, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--map-name', default='simple_6')
    ap.add_argument('--budget', type=int, default=40)
    ap.add_argument('--max-turns', type=int, default=40)
    ap.add_argument('--action-budget', type=int, default=5)
    ap.add_argument('--num-games', type=int, default=16)
    ap.add_argument('--num-workers', type=int, default=8)
    ap.add_argument('--trained-checkpoint',
                    default='experiments/mcts_gnn_training_run_v3/checkpoints/mcts_gnn_iter_000050.pt')
    args = ap.parse_args()

    per_worker = max(1, args.num_games // args.num_workers)
    print(f"Diagnostic on {args.map_name}  budget={args.budget}  max_turns={args.max_turns}")
    print(f"  {args.num_games} games / comparison, {args.num_workers} workers ({per_worker} each)")

    # ---- (a) uniform vs uniform (sanity) ------------------------------
    print("\n(a) MCTS(uniform) vs MCTS(uniform) — sanity (should be ~50%)")
    args_list = [
        (args.map_name, args.max_turns, args.action_budget, args.budget,
         per_worker, 1000 + w * 137)
        for w in range(args.num_workers)
    ]
    r = run_parallel(_uniform_vs_uniform_worker, args_list, args.num_workers)
    print(f"    win_rate={r['win_rate']:.2%}  ({r['wins']}W/{r['losses']}L/{r['draws']}D)")

    # ---- (b) random-init MCTS+GNN vs uniform --------------------------
    print("\n(b) MCTS+GNN(RANDOM-INIT) vs MCTS(uniform) — architecture baseline")
    args_list = [
        (None, {'hidden_dim': 128, 'num_layers': 3}, args.map_name, args.max_turns,
         args.action_budget, args.budget, per_worker, 2000 + w * 137, None)
        for w in range(args.num_workers)
    ]
    r = run_parallel(_gnn_vs_uniform_worker, args_list, args.num_workers)
    print(f"    win_rate={r['win_rate']:.2%}  ({r['wins']}W/{r['losses']}L/{r['draws']}D)")

    # ---- (c) trained MCTS+GNN vs uniform ------------------------------
    if Path(args.trained_checkpoint).exists():
        print(f"\n(c) MCTS+GNN(TRAINED @ {args.trained_checkpoint}) vs MCTS(uniform)")
        args_list = [
            (args.trained_checkpoint, None, args.map_name, args.max_turns,
             args.action_budget, args.budget, per_worker, 3000 + w * 137, None)
            for w in range(args.num_workers)
        ]
        r = run_parallel(_gnn_vs_uniform_worker, args_list, args.num_workers)
        print(f"    win_rate={r['win_rate']:.2%}  ({r['wins']}W/{r['losses']}L/{r['draws']}D)")
    else:
        print(f"\n(c) SKIPPED — no checkpoint at {args.trained_checkpoint}")


if __name__ == "__main__":
    main()
