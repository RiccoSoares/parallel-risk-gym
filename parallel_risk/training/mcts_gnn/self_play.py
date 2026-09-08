"""Self-play episode generator for MCTS+GNN training.

Plays games with an `MCTSGNNAgent` acting for both sides and records
`(state, visit_distribution, outcome)` tuples used as distillation targets
by `MCTSGNNTrainer`.

Parallel worker mirrors the spawn-based pattern in
`parallel_risk/training/torchrl/train.py::_rollout_worker`.
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Any, Dict, List

import numpy as np


# ---------------------------------------------------------------------------
# Temperature schedule
# ---------------------------------------------------------------------------

def _temperature(turn: int, temperature_turns: int,
                 max_temperature: float = 1.0) -> float:
    """T=max_temperature for the first `temperature_turns` turns, else 0 (argmax)."""
    return max_temperature if turn < temperature_turns else 0.0


# ---------------------------------------------------------------------------
# Single-episode driver (uses an MCTSGNNAgent directly)
# ---------------------------------------------------------------------------

def play_episode(mcts_gnn_agent, env, self_play_config: Dict[str, Any],
                 map_name: str) -> List[Dict[str, Any]]:
    """Play one self-play game and return training examples.

    Args:
        mcts_gnn_agent: Constructed `MCTSGNNAgent` (both agents share it).
        env: A raw `ParallelRiskEnv` — reward shaping should be disabled so
             the terminal reward is exactly ±1 or 0 (the value target).
        self_play_config: Dict with keys `dirichlet_alpha`, `noise_frac`,
             `temperature_turns`, `max_temperature`.
        map_name: Recorded on each example for per-map bookkeeping.

    Returns:
        List of example dicts, one per (agent, turn):
            {
                'graph': PyG Data,     # agent-relative graph obs BEFORE the turn
                'agent_id': str,
                'visit_dist': dict[tuple, float],
                'map_name': str,
                'z': float,            # +1 winner, -1 loser, 0 draw
            }
    """
    from parallel_risk.training.torchrl.graph_wrapper import env_to_graph

    mcts = mcts_gnn_agent.mcts
    map_config = mcts_gnn_agent.map_config
    device = mcts_gnn_agent.device
    max_regions = mcts_gnn_agent.max_regions

    dirichlet_alpha = float(self_play_config.get('dirichlet_alpha', 0.3))
    noise_frac = float(self_play_config.get('noise_frac', 0.25))
    dirichlet_min_actions = int(self_play_config.get('dirichlet_min_actions', 8))
    temperature_turns = int(self_play_config.get('temperature_turns', 10))
    max_temperature = float(self_play_config.get('max_temperature', 1.0))

    examples: List[Dict[str, Any]] = []

    obs, _ = env.reset()
    rewards = {a: 0.0 for a in env.possible_agents}
    done = False
    turn = 0

    while not done:
        temp = _temperature(turn, temperature_turns, max_temperature)

        root = mcts.make_root(env.game_state)
        if noise_frac > 0.0:
            mcts.apply_root_dirichlet(root, alpha=dirichlet_alpha,
                                      noise_frac=noise_frac,
                                      min_actions=dirichlet_min_actions)
        mcts.run(root, mcts_gnn_agent.simulation_budget)

        joint_action = {}
        for agent_id in ('agent_0', 'agent_1'):
            if agent_id not in obs:
                continue
            graph = env_to_graph(obs[agent_id], map_config, device,
                                 max_regions=max_regions)
            visit_dist = mcts.policy_target(root, agent_id)
            examples.append({
                'graph': graph,
                'agent_id': agent_id,
                'visit_dist': visit_dist,
                'map_name': map_name,
                'z': 0.0,  # backfilled after episode ends
            })
            joint_action[agent_id] = mcts.sampled_action(root, agent_id, temp)

        obs, rewards, terminateds, truncateds, _ = env.step(joint_action)
        done = terminateds.get('__all__', False) or truncateds.get('__all__', False)
        turn += 1

    # Backfill z from the raw terminal reward. With reward_shaping_config=None,
    # this is exactly +1 winner / -1 loser / 0 draw. Terminal-only by design —
    # matches the AlphaZero research formulation.
    z_by_agent = {a: float(rewards.get(a, 0.0)) for a in ('agent_0', 'agent_1')}
    for ex in examples:
        ex['z'] = z_by_agent.get(ex['agent_id'], 0.0)

    return examples


# ---------------------------------------------------------------------------
# Parallel worker — module-level so multiprocessing can pickle it
# ---------------------------------------------------------------------------

def _self_play_worker(args):
    """Run `num_games` self-play episodes in a spawned subprocess.

    Rebuilds `MCTSGNNAgent` on CPU from a state_dict, plays games (uniform
    over the configured map list), and returns a flat list of examples.
    """
    # Support both the pre-multi-map and post-multi-map arg tuples.
    if len(args) == 8:
        (policy_state_dict, model_kwargs, env_configs,
         mcts_config, self_play_config, num_games, base_seed, max_regions) = args
    else:
        (policy_state_dict, model_kwargs, env_configs,
         mcts_config, self_play_config, num_games, base_seed) = args
        max_regions = None

    import torch
    from parallel_risk import ParallelRiskEnv
    from parallel_risk.agents.mcts_gnn_agent import MCTSGNNAgent
    from parallel_risk.models.action_decoder import ActionDecoder
    from parallel_risk.models.gnn_gcn import GCNPolicy

    torch.set_num_threads(1)  # avoid oversubscription with sibling workers
    # Seed both RNGs so runs are reproducible given the same base_seed.
    # numpy for Dirichlet + temperature sampling, torch for the action
    # decoder's stochastic sampling inside the GNN sampler.
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)

    # Rebuild policy on CPU
    policy = GCNPolicy(**model_kwargs)
    policy.load_state_dict(policy_state_dict)
    policy.eval()

    action_budget = int(mcts_config.get('action_budget', 5))
    decoder = ActionDecoder(action_budget=action_budget, max_troops=20)

    # Build one MCTSGNNAgent per map (each is tied to a map_config).
    agents_by_map = {}
    envs_by_map = {}
    for ecfg in env_configs:
        map_name = ecfg['map_name']
        env = ParallelRiskEnv(
            map_name=map_name,
            max_turns=int(ecfg.get('max_turns', 50)),
            seed=None,
            reward_shaping_config=None,  # terminal reward is the value target
        )
        envs_by_map[map_name] = env
        agents_by_map[map_name] = MCTSGNNAgent(
            policy=policy,
            decoder=decoder,
            map_config=env.map_config,
            simulation_budget=int(mcts_config.get('simulation_budget', 50)),
            c_puct=float(mcts_config.get('c_puct', 1.4)),
            action_budget=action_budget,
            max_troops=20,
            max_turns=int(ecfg.get('max_turns', 50)),
            uct_c=float(mcts_config.get('uct_c', 1.41)),
            pw_alpha=float(mcts_config.get('pw_alpha', 0.5)),
            max_rollout_turns=int(mcts_config.get('max_rollout_turns', 20)),
            device='cpu',
            max_regions=max_regions,
            pw_sampler=mcts_config.get('pw_sampler', 'masked_random'),
            use_value_fn=bool(mcts_config.get('use_value_fn', True)),
        )

    rng = np.random.RandomState(base_seed)
    all_examples: List[Dict[str, Any]] = []
    map_names = list(agents_by_map.keys())

    for _ in range(num_games):
        map_name = map_names[rng.randint(len(map_names))]
        agent = agents_by_map[map_name]
        env = envs_by_map[map_name]
        all_examples.extend(
            play_episode(agent, env, self_play_config, map_name)
        )

    return all_examples


# ---------------------------------------------------------------------------
# Main-process entry
# ---------------------------------------------------------------------------

def collect_games(policy, model_kwargs, env_configs, mcts_config,
                  self_play_config, num_games: int, num_workers: int,
                  base_seed: int = 0,
                  max_regions: int = None) -> List[Dict[str, Any]]:
    """Collect `num_games` self-play episodes across `num_workers` processes.

    The trainer's policy is snapshotted to CPU once, then handed to each
    worker (workers hold independent CPU copies for the duration of the
    call). If `num_workers <= 1`, runs in-process (useful for tests + debug).

    `max_regions` should match the trainer's padding when the map set
    contains maps with different region counts. Defaults to None (each
    worker pads to its own map's region count — single-map only).
    """
    state_dict_cpu = {k: v.detach().cpu() for k, v in policy.state_dict().items()}

    per_worker = max(1, num_games // max(1, num_workers))
    worker_args = [
        (state_dict_cpu, model_kwargs, env_configs, mcts_config,
         self_play_config, per_worker, base_seed + i * 1000, max_regions)
        for i in range(max(1, num_workers))
    ]

    if num_workers <= 1:
        return _self_play_worker(worker_args[0])

    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=num_workers) as pool:
        results = pool.map(_self_play_worker, worker_args)

    flat: List[Dict[str, Any]] = []
    for r in results:
        flat.extend(r)
    return flat
