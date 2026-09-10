"""Benchmark suite for the performance work on branch perf/parallel-gpu.

Times the workloads that dominate real experiments, writes a JSON that
records the git commit, and can diff two JSONs into a before/after table.
Every metric is a time (lower is better) so the comparison is uniform.

Scenarios (all seeded so before/after runs see the same games):
    env_step        ParallelRiskEnv.step with masked-random self-play, K=10
    sim_step        RiskSimulator turn (the MCTS rollout primitive)
    mcts_uniform    MCTSAgent decision, budget 200, K=10, opening + mid-game
    mcts_gnn_cpu    MCTSGNNAgent decision (K10 PPO checkpoint), CPU
    mcts_gnn_gpu    same on CUDA
    ppo             PPOTrainer rollout + update, several worker/device layouts
    exit_selfplay   collect_games_exit over the 21-map roster, 12 workers
    eval_event      evaluate_all_maps on 3 maps (MCTS+GNN vs MCTS-uniform)

Usage:
    PYTHONIOENCODING=utf-8 PYTHONUTF8=1 PYTHONPATH=. python experiments/benchmark_suite.py \
        --label baseline                      # quick sizes (~6 min)
    ... --full                                # self-play/eval at budget 200
    ... --scenarios env_step,mcts_uniform     # subset
    ... --compare experiments/benchmarks/baseline.json experiments/benchmarks/after.json
"""

from __future__ import annotations

import torch  # noqa: F401  (import torch before parallel_risk.training on Windows)

import argparse
import json
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO = Path(__file__).resolve().parent.parent
CKPT_REL = Path('experiments') / 'k_sweep_ppo_200' / 'K10' / 'checkpoints' / 'final.pt'
CKPT_MAX_REGIONS = 5   # 21-map roster -> node_features_dim = 3 + 5


def resolve_k10_checkpoint(explicit: str = None) -> Path:
    """Find the K10 PPO checkpoint (gitignored, so absent from fresh worktrees).

    Order: explicit argument, $PRG_K10_CKPT, this checkout, then the main
    checkout when this file lives in an agent worktree
    (<main>/.claude/worktrees/<name>/...).
    """
    import os
    candidates = [explicit, os.environ.get('PRG_K10_CKPT'), REPO / CKPT_REL]
    if REPO.parent.name == 'worktrees' and REPO.parent.parent.name == '.claude':
        candidates.append(REPO.parents[2] / CKPT_REL)
    for c in candidates:
        if c and Path(c).exists():
            return Path(c)
    raise FileNotFoundError(
        f'K10 checkpoint not found; tried {[str(c) for c in candidates if c]}. '
        'Pass --ckpt or set PRG_K10_CKPT.')


CKPT_K10 = None  # set in main() via resolve_k10_checkpoint()
K = 10
MAX_TURNS = 40
MCTS_BUDGET = 200

ENV_MAPS = ['simple_6', 'grid_20', 'hub_ring_30']
MCTS_MAPS = ['simple_6', 'dense_12', 'grid_20', 'hub_ring_30']
PPO_MULTIMAP = ['simple_6', 'medium_8', 'large_10']
PPO_KSWEEP = ['simple_6', 'dense_12', 'hex_grid_18', 'grid_20', 'corridor_28', 'hub_ring_30']

ALL_SCENARIOS = ['env_step', 'sim_step', 'mcts_uniform', 'mcts_gnn_cpu', 'mcts_gnn_gpu',
                 'ppo', 'exit_selfplay', 'eval_event']


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def git_info() -> Dict[str, str]:
    def run(*args):
        try:
            return subprocess.check_output(['git', *args], cwd=REPO, text=True).strip()
        except Exception:
            return 'unknown'
    return {'commit': run('rev-parse', '--short', 'HEAD'),
            'branch': run('rev-parse', '--abbrev-ref', 'HEAD'),
            'dirty': run('status', '--porcelain') != ''}


def all_maps() -> List[str]:
    from parallel_risk.env.map_config import MapRegistry
    return sorted(m for m in MapRegistry.list_maps() if m != 'basic_6')


def make_env(map_name: str):
    from parallel_risk.env.parallel_risk_env import ParallelRiskEnv
    return ParallelRiskEnv(map_name=map_name, max_turns=MAX_TURNS,
                           max_actions_per_turn=max(10, K), reward_shaping_config=None)


def make_sampler(env):
    from parallel_risk.agents.masked_random_agent import MaskedRandomAgentRLlib
    return MaskedRandomAgentRLlib(
        n_territories=env.map_config.n_territories,
        adjacency_matrix=env.map_config.adjacency_matrix,
        action_budget=K, max_troops=20, max_actions_per_turn=max(10, K))


def advance_random(env, sampler, n_turns: int, seed: int):
    """Reset env with seed and play n_turns of masked-random self-play."""
    np.random.seed(seed)
    obs, _ = env.reset(seed=seed)
    for _ in range(n_turns):
        actions = {aid: sampler.get_action_raw(obs[aid]) for aid in obs}
        obs, _, terms, truncs, _ = env.step(actions)
        if terms.get('__all__') or truncs.get('__all__'):
            obs, _ = env.reset(seed=seed + 1)
    return obs


def load_k10_policy(device: str):
    from parallel_risk.models.gnn_gcn import GCNPolicy
    ckpt = torch.load(CKPT_K10, map_location='cpu', weights_only=False)
    kwargs = dict(node_features_dim=3 + CKPT_MAX_REGIONS,
                  global_features_dim=2 + CKPT_MAX_REGIONS,
                  hidden_dim=128, num_layers=3, action_budget=K,
                  max_troops=20, dropout=0.1)
    policy = GCNPolicy(**kwargs)
    policy.load_state_dict(ckpt['policy_state_dict'])
    policy.to(torch.device(device)).eval()
    return policy, kwargs


def _sync(device: str):
    if device.startswith('cuda'):
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# scenarios
# ---------------------------------------------------------------------------

def bench_env_step(n_steps: int) -> Dict[str, float]:
    out = {}
    for m in ENV_MAPS:
        env = make_env(m)
        sampler = make_sampler(env)
        np.random.seed(0)
        obs, _ = env.reset(seed=0)
        t0 = time.perf_counter()
        for i in range(n_steps):
            actions = {aid: sampler.get_action_raw(obs[aid]) for aid in obs}
            obs, _, terms, truncs, _ = env.step(actions)
            if terms.get('__all__') or truncs.get('__all__'):
                obs, _ = env.reset(seed=i + 1)
        out[f'{m}/ms_per_step'] = (time.perf_counter() - t0) / n_steps * 1000
    return out


def bench_sim_step(n_turns: int) -> Dict[str, float]:
    from parallel_risk.agents.mcts_agent import RiskSimulator
    out = {}
    for m in ENV_MAPS:
        env = make_env(m)
        sampler = make_sampler(env)
        sim = RiskSimulator(env.map_config, max_turns=MAX_TURNS)
        np.random.seed(0)
        env.reset(seed=0)
        fresh = RiskSimulator.clone_state(env.game_state)
        state = RiskSimulator.clone_state(fresh)
        t0 = time.perf_counter()
        for _ in range(n_turns):
            actions = {}
            for agent in sim.AGENTS:
                actions[agent] = sampler.get_action_raw(sim.state_to_obs(state, agent))
            state, _, done = sim.step(state, actions)
            if done:
                state = RiskSimulator.clone_state(fresh)
        out[f'{m}/ms_per_turn'] = (time.perf_counter() - t0) / n_turns * 1000
    return out


def _decision_points(env, sampler):
    """(label, game_state) pairs: opening position and after 10 random turns."""
    from parallel_risk.agents.mcts_agent import RiskSimulator
    advance_random(env, sampler, 0, seed=0)
    opening = RiskSimulator.clone_state(env.game_state)
    advance_random(env, sampler, 10, seed=0)
    mid = RiskSimulator.clone_state(env.game_state)
    return [('open', opening), ('mid', mid)]


def _time_decisions(agent, env, sampler, device: str = 'cpu') -> Dict[str, float]:
    out = {}
    for label, state in _decision_points(env, sampler):
        np.random.seed(123)
        torch.manual_seed(123)
        _sync(device)
        t0 = time.perf_counter()
        agent.get_action(state, 'agent_0')
        _sync(device)
        out[label] = time.perf_counter() - t0
    return out


def bench_mcts_uniform(budget: int) -> Dict[str, float]:
    from parallel_risk.agents.mcts_agent import MCTSAgent
    out = {}
    for m in MCTS_MAPS:
        env = make_env(m)
        sampler = make_sampler(env)
        agent = MCTSAgent.from_env(env, simulation_budget=budget, action_budget=K)
        for label, s in _time_decisions(agent, env, sampler).items():
            out[f'{m}/{label}/s_per_decision'] = s
    return out


def bench_mcts_gnn(device: str, budget: int) -> Dict[str, float]:
    from parallel_risk.agents.mcts_gnn_agent import MCTSGNNAgent
    out = {}
    for m in MCTS_MAPS:
        env = make_env(m)
        sampler = make_sampler(env)
        agent = MCTSGNNAgent.from_checkpoint(
            str(CKPT_K10), env.map_config, simulation_budget=budget, device=device,
            max_regions=CKPT_MAX_REGIONS, max_turns=MAX_TURNS, use_value_fn=True)
        # warm-up (CUDA context, cudnn autotune) on a tiny search
        env.reset(seed=0)
        agent.mcts.run(agent.mcts.make_root(env.game_state), 5)
        for label, s in _time_decisions(agent, env, sampler, device).items():
            out[f'{m}/{label}/s_per_decision'] = s
    return out


def _ppo_config(map_names, num_workers, use_gpu, batch_size, num_epochs, action_budget, run_dir):
    return {
        'env': {'map_names': list(map_names), 'max_turns': 50,
                'action_budget': action_budget, 'seed': None, 'use_reward_shaping': True},
        'model': {'type': 'gcn', 'hidden_dim': 128, 'num_layers': 3, 'dropout': 0.1},
        'training': {'num_workers': num_workers, 'batch_size': batch_size,
                     'num_epochs': num_epochs, 'learning_rate': 1e-4, 'gamma': 0.99,
                     'gae_lambda': 0.95, 'clip_epsilon': 0.2, 'entropy_coeff': 0.01,
                     'value_loss_coeff': 0.5, 'max_grad_norm': 0.5, 'use_gpu': use_gpu},
        'log_dir': str(run_dir / 'runs'),
        'checkpoint_dir': str(run_dir / 'ckpt'),
    }


def bench_ppo(num_timed: int, batch_size: int) -> Dict[str, float]:
    from parallel_risk.training.torchrl.train import PPOTrainer
    run_dir = Path(tempfile.mkdtemp(prefix='prg_bench_'))
    scenarios = [
        ('multimap_cpu_w1', PPO_MULTIMAP, False, 1, 5),
        ('multimap_gpu_w4', PPO_MULTIMAP, True, 4, 5),
        ('ksweep_cpu_w8', PPO_KSWEEP, False, 8, K),
        ('ksweep_gpu_w8', PPO_KSWEEP, True, 8, K),
    ]
    out = {}
    for name, maps, use_gpu, workers, ab in scenarios:
        if use_gpu and not torch.cuda.is_available():
            continue
        torch.manual_seed(0)
        np.random.seed(0)
        trainer = PPOTrainer(_ppo_config(maps, workers, use_gpu, batch_size, 3, ab, run_dir))
        steps = trainer.batch_size // 2
        rollout = trainer.collect_rollout(steps)     # warm-up: spawn pool, CUDA ctx
        trainer.update_policy(rollout)
        trainer.timers.enabled = True
        trainer.timers.reset()
        for _ in range(num_timed):
            rollout = trainer.collect_rollout(steps)
            trainer.update_policy(rollout)
        trainer.timers.enabled = False
        s = trainer.timers.summary()
        out[f'{name}/rollout_s'] = s['rollout.total']['avg_ms'] / 1000
        out[f'{name}/update_s'] = s['update.total']['avg_ms'] / 1000
        out[f'{name}/iter_s'] = out[f'{name}/rollout_s'] + out[f'{name}/update_s']
        if trainer._worker_pool is not None:
            trainer._worker_pool.terminate()
            trainer._worker_pool.join()
        trainer.writer.close()
    return out


def bench_exit_selfplay(num_games: int, num_workers: int, budget: int) -> Dict[str, float]:
    from parallel_risk.training.mcts_gnn.exit_self_play import collect_games_exit
    policy, kwargs = load_k10_policy('cpu')
    env_configs = [{'map_name': m, 'max_turns': MAX_TURNS} for m in all_maps()]
    mcts_config = {'simulation_budget': budget, 'c_puct': 1.4, 'uct_c': 1.41,
                   'pw_alpha': 0.5, 'max_rollout_turns': MAX_TURNS, 'action_budget': K,
                   'pw_sampler': 'masked_random', 'use_value_fn': True}
    self_play_config = {'dirichlet_alpha': 0.3, 'noise_frac': 0.25,
                        'dirichlet_min_actions': 8, 'temperature_turns': 10,
                        'max_temperature': 1.0}
    # games_per_worker/device mirror parallel_risk/training/mcts_gnn/configs/
    # mcts_gnn_exit.yaml, so this measures what a real ExIt run does.
    t0 = time.perf_counter()
    examples = collect_games_exit(policy=policy, model_kwargs=kwargs, env_configs=env_configs,
                                  mcts_config=mcts_config, self_play_config=self_play_config,
                                  num_games=num_games, num_workers=num_workers,
                                  base_seed=0, max_regions=CKPT_MAX_REGIONS,
                                  games_per_worker=16, device='cpu')
    wall = time.perf_counter() - t0
    return {f'games{num_games}_w{num_workers}_b{budget}/wall_s': wall,
            f'games{num_games}_w{num_workers}_b{budget}/s_per_game': wall / num_games,
            f'games{num_games}_w{num_workers}_b{budget}/examples': float(len(examples))}


def bench_eval_event(games_per_map: int, budget: int) -> Dict[str, float]:
    sys.path.insert(0, str(REPO / 'experiments'))
    from mcts_gnn_exit_training_run import evaluate_all_maps
    policy, kwargs = load_k10_policy('cpu')
    maps = ['simple_6', 'grid_20', 'hub_ring_30']
    t0 = time.perf_counter()
    evaluate_all_maps(policy=policy, model_kwargs=kwargs, map_names=maps,
                      max_turns=MAX_TURNS, action_budget=K, mcts_budget=budget,
                      num_games_per_map=games_per_map, num_workers=len(maps),
                      max_regions=CKPT_MAX_REGIONS, base_seed=7)
    wall = time.perf_counter() - t0
    return {f'3maps_g{games_per_map}_b{budget}/wall_s': wall}


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def run(scenarios: List[str], full: bool) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    budget = MCTS_BUDGET if full else 40
    runners = {
        'env_step': lambda: bench_env_step(n_steps=300),
        'sim_step': lambda: bench_sim_step(n_turns=300),
        'mcts_uniform': lambda: bench_mcts_uniform(budget=MCTS_BUDGET),
        'mcts_gnn_cpu': lambda: bench_mcts_gnn('cpu', budget=MCTS_BUDGET),
        'mcts_gnn_gpu': lambda: bench_mcts_gnn('cuda', budget=MCTS_BUDGET),
        'ppo': lambda: bench_ppo(num_timed=3, batch_size=1024),
        'exit_selfplay': lambda: bench_exit_selfplay(num_games=24 if full else 12,
                                                     num_workers=12, budget=budget),
        'eval_event': lambda: bench_eval_event(games_per_map=2, budget=budget),
    }
    for name in scenarios:
        print(f'\n[{name}] ...', flush=True)
        t0 = time.perf_counter()
        if name == 'mcts_gnn_gpu' and not torch.cuda.is_available():
            print('  skipped (no CUDA)')
            continue
        try:
            r = runners[name]()
        except Exception as exc:  # keep going: one broken scenario must not lose the rest
            import traceback
            traceback.print_exc()
            results[name] = {'error': str(exc)}
            continue
        results[name] = r
        for k, v in r.items():
            print(f'  {k:50s} {v:10.3f}')
        print(f'  ({time.perf_counter() - t0:.0f}s)')
    return results


def compare(path_a: str, path_b: str) -> None:
    a = json.load(open(path_a))
    b = json.load(open(path_b))
    print(f"before: {a['label']} @ {a['git']['commit']}   after: {b['label']} @ {b['git']['commit']}")
    print(f"{'metric':60s} {'before':>10s} {'after':>10s} {'speedup':>8s}")
    for scen, metrics in a['results'].items():
        if scen not in b['results']:
            continue
        for k, va in metrics.items():
            vb = b['results'][scen].get(k)
            if vb is None or k.endswith('/examples'):
                continue
            ratio = va / vb if vb else float('inf')
            print(f"{scen + '/' + k:60s} {va:10.3f} {vb:10.3f} {ratio:7.2f}x")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--label', default=None)
    p.add_argument('--scenarios', default=','.join(ALL_SCENARIOS))
    p.add_argument('--full', action='store_true', help='budget 200 + larger sizes for self-play/eval')
    p.add_argument('--output-dir', default='experiments/benchmarks')
    p.add_argument('--compare', nargs=2, metavar=('BEFORE', 'AFTER'))
    p.add_argument('--ckpt', default=None, help='K10 PPO checkpoint (default: auto-resolve)')
    p.add_argument('--threads', type=int, default=4,
                   help='torch threads in this process (fixed so runs are comparable)')
    args = p.parse_args()

    if args.compare:
        compare(*args.compare)
        return

    global CKPT_K10
    CKPT_K10 = resolve_k10_checkpoint(args.ckpt)
    torch.set_num_threads(max(1, args.threads))
    info = git_info()
    label = args.label or info['commit']
    scenarios = [s.strip() for s in args.scenarios.split(',') if s.strip()]
    print(f'benchmark label={label} commit={info["commit"]} branch={info["branch"]} '
          f'dirty={info["dirty"]} full={args.full} threads={args.threads} '
          f'ckpt={CKPT_K10} scenarios={scenarios}')
    t0 = time.perf_counter()
    results = run(scenarios, args.full)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{label}.json'
    with open(out_path, 'w') as f:
        json.dump({'label': label, 'timestamp': datetime.now().isoformat(), 'git': info,
                   'full': args.full, 'threads': args.threads, 'ckpt': str(CKPT_K10),
                   'platform': platform.platform(),
                   'torch': torch.__version__, 'cuda': torch.cuda.is_available(),
                   'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                   'results': results}, f, indent=2)
    print(f'\nSaved {out_path}  (total {(time.perf_counter() - t0) / 60:.1f} min)')


if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    main()
