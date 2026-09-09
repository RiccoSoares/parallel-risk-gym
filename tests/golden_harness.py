"""Golden-trajectory harness: proves a refactor preserved game/search/training semantics.

Two modes:

  record  : run a fixed, seeded battery and write a JSON of everything
            that must be preserved by a semantics-preserving refactor.
  compare : load a golden JSON and a new JSON and check them section by
            section — integer/structural data must match EXACTLY, floats
            must match within --rtol/--atol.

Battery (all seeds fixed inside this file):
  (a) 3 MaskedRandom self-play games per map x 3 maps
      -> full per-turn (ownership, troops) trajectory, invalid-action counts,
         terminal rewards, episode length.                       [EXACT]
  (b) 2 MCTS-uniform decisions (budget 30, K=5, simple_6): initial state and
      a mid-game state -> per-agent available action keys, visit counts n,
      chosen action                                              [EXACT]
      and Q values                                               [FLOAT]
  (c) 2 MCTS+GNN decisions (checkpoint K10 final.pt, CPU, budget 30):
      -> root visit distribution, chosen action                  [EXACT]
      -> priors p, Q, root value                                 [FLOAT]
  (d) one PPO rollout (sequential, simple_6, K=3, hidden 32, 64 steps):
      -> sample count, action histogram, dones                   [EXACT]
      -> mean reward, mean value, sum log_prob                   [FLOAT]
  (e) one AZ self-play episode via play_episode with Dirichlet+temperature,
      budget 8, with np.random.default_rng monkeypatched to a seeded
      Generator (the library leaves it unseeded — see notes) -> action keys
      per turn, visit dists, z                                   [EXACT/FLOAT]

Usage (from repo root):
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 PYTHONPATH=. python tests/golden_harness.py record --out tests/golden/golden.json
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 PYTHONPATH=. python tests/golden_harness.py record --out /tmp/new.json
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 PYTHONPATH=. python tests/golden_harness.py compare --golden tests/golden/golden.json --new /tmp/new.json [--rtol 1e-5 --atol 1e-6]

The committed tests/golden/golden.json was recorded on the unoptimized code (commit f2f3dd7).
Every optimization on branch perf/parallel-gpu must pass `compare` against it, or explain
exactly which section changed and why (e.g. a deliberate RNG-stream change).

Options:
  --ckpt PATH     GNN checkpoint (default: experiments/k_sweep_ppo_200/K10/checkpoints/final.pt).
                  If missing, section (c)/(e) use a random-init GCNPolicy under torch seed 0.
  --skip SECTION  comma list of sections to skip (a,b,c,d,e)
  --threads N     torch.set_num_threads (default 2). Record and compare with the SAME N
                  if you want float sections to be bit-identical.

Exit code: 0 if all compared sections pass, 1 otherwise.
"""
import argparse
import hashlib
import json
import math
import random
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = REPO / "experiments/k_sweep_ppo_200/K10/checkpoints/final.pt"
CKPT_MAX_REGIONS = 5   # K10 checkpoint input_proj is 3+5 wide; ckpt config lacks 'max_regions'

MAPS = ['simple_6', 'medium_8', 'large_10']
GAMES_PER_MAP = 3
GAME_SEED_BASE = 1000
MCTS_UNIFORM = dict(budget=30, K=5, map_name='simple_6', seeds=[2001, 2002])
MCTS_GNN = dict(budget=30, map_name='simple_6', seeds=[3001, 3002])
PPO = dict(seed=4001, num_steps=64, action_budget=3, hidden_dim=32, num_layers=2, map_name='simple_6')
SELFPLAY = dict(seed=5001, budget=8, map_name='simple_6', max_turns=12)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def key_list(keys):
    return [[list(map(int, row)) for row in k] for k in keys]


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def mid_game_state(map_name, seed, turns, K=5):
    """Play `turns` MaskedRandom turns to obtain a non-trivial game state."""
    from parallel_risk import ParallelRiskEnv
    from parallel_risk.agents.masked_random_agent import MaskedRandomAgentRLlib
    env = ParallelRiskEnv(map_name=map_name, max_turns=100, seed=seed)
    obs, _ = env.reset(seed=seed)
    ag = MaskedRandomAgentRLlib.from_env(env, action_budget=K)
    for _ in range(turns):
        obs, *_ = env.step({a: ag.get_action_raw(obs[a]) for a in env.agents})
        if not env.agents:
            break
    return env


def build_gnn_agent(map_config, budget, ckpt, action_budget_fallback=5, **kw):
    from parallel_risk.agents.mcts_gnn_agent import MCTSGNNAgent
    if ckpt is not None and Path(ckpt).exists():
        return MCTSGNNAgent.from_checkpoint(
            str(ckpt), map_config, simulation_budget=budget, device='cpu',
            max_regions=CKPT_MAX_REGIONS, max_turns=100, **kw), 'checkpoint'
    # Fallback: random init under fixed torch seed
    from parallel_risk.models.gnn_gcn import GCNPolicy
    from parallel_risk.models.action_decoder import ActionDecoder
    torch.manual_seed(0)
    n_regions = len(map_config.regions)
    policy = GCNPolicy(node_features_dim=3 + n_regions, global_features_dim=2 + n_regions,
                       hidden_dim=64, num_layers=2, action_budget=action_budget_fallback,
                       max_troops=20, dropout=0.0)
    decoder = ActionDecoder(action_budget=action_budget_fallback, max_troops=20)
    return MCTSGNNAgent(policy=policy, decoder=decoder, map_config=map_config,
                        simulation_budget=budget, action_budget=action_budget_fallback,
                        max_turns=100, device='cpu', **kw), 'random_init'


def root_summary(mcts, root, with_prior=False):
    out = {}
    for a in ('agent_0', 'agent_1'):
        keys = root.available_actions[a]
        st = root.stats[a]
        d = {
            'available': key_list(keys),
            'n': [int(st[k]['n']) for k in keys],
            'q': [float(st[k]['q']) for k in keys],
            'best': mcts.best_action(root, a)['actions'].tolist(),
            'policy_target': [[key_list([k])[0], float(p)] for k, p in mcts.policy_target(root, a).items()],
        }
        if with_prior:
            d['p'] = [float(st[k].get('p', 0.0)) for k in keys]
        out[a] = d
    out['root_visits'] = int(root.visit_count)
    out['n_children'] = int(len(root.children))
    return out


# ---------------------------------------------------------------------------
# record sections
# ---------------------------------------------------------------------------

def rec_a():
    from parallel_risk import ParallelRiskEnv
    from parallel_risk.agents.masked_random_agent import MaskedRandomAgentRLlib
    out = {}
    for mi, map_name in enumerate(MAPS):
        games = []
        for gi in range(GAMES_PER_MAP):
            seed = GAME_SEED_BASE + mi * 100 + gi
            env = ParallelRiskEnv(map_name=map_name, max_turns=100, seed=seed)
            obs, _ = env.reset(seed=seed)
            agents = {a: MaskedRandomAgentRLlib.from_env(env, action_budget=5) for a in env.possible_agents}
            traj = [{'own': env.game_state['territory_ownership'].tolist(),
                     'troops': env.game_state['territory_troops'].tolist()}]
            invalid = {a: 0 for a in env.possible_agents}
            done = False; rewards = None; nturn = 0
            while not done:
                acts = {a: agents[a].get_action_raw(obs[a]) for a in env.agents}
                obs, rewards, terms, truncs, infos = env.step(acts)
                for a in infos:
                    invalid[a] += infos[a]['invalid_actions']
                traj.append({'own': env.game_state['territory_ownership'].tolist(),
                             'troops': env.game_state['territory_troops'].tolist()})
                nturn += 1
                done = terms['__all__'] or truncs['__all__']
            games.append({'seed': seed, 'length': nturn, 'rewards': rewards,
                          'invalid_actions': invalid, 'trajectory': traj,
                          'trajectory_sha': sha(traj)})
        out[map_name] = games
    return out


def rec_b():
    from parallel_risk import ParallelRiskEnv
    from parallel_risk.agents.mcts_agent import MCTSAgent
    cfg = MCTS_UNIFORM
    decisions = []
    for i, seed in enumerate(cfg['seeds']):
        if i == 0:
            env = ParallelRiskEnv(map_name=cfg['map_name'], max_turns=100, seed=seed)
            env.reset(seed=seed)
        else:
            env = mid_game_state(cfg['map_name'], seed, turns=6, K=cfg['K'])
        seed_all(seed)
        agent = MCTSAgent(env.map_config, simulation_budget=cfg['budget'],
                          action_budget=cfg['K'], max_turns=100)
        t0 = time.perf_counter()
        root = agent.mcts.make_root(env.game_state)
        agent.mcts.run(root, cfg['budget'])
        dt = time.perf_counter() - t0
        s = root_summary(agent.mcts, root)
        s['state'] = {'own': env.game_state['territory_ownership'].tolist(),
                      'troops': env.game_state['territory_troops'].tolist(),
                      'turn': int(env.game_state['turn_number'])}
        s['seed'] = seed; s['_wall_s'] = dt
        decisions.append(s)
    return decisions


def rec_c(ckpt):
    from parallel_risk import ParallelRiskEnv
    cfg = MCTS_GNN
    decisions = []
    for i, seed in enumerate(cfg['seeds']):
        if i == 0:
            env = ParallelRiskEnv(map_name=cfg['map_name'], max_turns=100, seed=seed)
            env.reset(seed=seed)
        else:
            env = mid_game_state(cfg['map_name'], seed, turns=6, K=5)
        seed_all(seed)
        agent, src = build_gnn_agent(env.map_config, cfg['budget'], ckpt)
        t0 = time.perf_counter()
        root = agent.mcts.make_root(env.game_state)
        agent.mcts.run(root, cfg['budget'])
        dt = time.perf_counter() - t0
        s = root_summary(agent.mcts, root, with_prior=True)
        s['root_value'] = float(agent.mcts.value_fn(root.game_state, 'agent_0'))
        s['state'] = {'own': env.game_state['territory_ownership'].tolist(),
                      'troops': env.game_state['territory_troops'].tolist(),
                      'turn': int(env.game_state['turn_number'])}
        s['seed'] = seed; s['policy_source'] = src; s['_wall_s'] = dt
        decisions.append(s)
    return decisions


def rec_d():
    from parallel_risk.training.torchrl.train import PPOTrainer
    cfg = PPO
    seed_all(cfg['seed'])
    trainer = PPOTrainer({
        'env': {'map_name': cfg['map_name'], 'max_turns': 50,
                'action_budget': cfg['action_budget'], 'seed': cfg['seed']},
        'model': {'type': 'gcn', 'hidden_dim': cfg['hidden_dim'],
                  'num_layers': cfg['num_layers'], 'dropout': 0.1},
        'training': {'num_workers': 1, 'batch_size': 128, 'num_epochs': 1,
                     'learning_rate': 3e-4, 'gamma': 0.99, 'gae_lambda': 0.95,
                     'clip_epsilon': 0.2, 'entropy_coeff': 0.01, 'value_loss_coeff': 0.5,
                     'max_grad_norm': 0.5, 'use_gpu': False},
        'log_dir': tempfile.mkdtemp(), 'checkpoint_dir': tempfile.mkdtemp(),
    })
    t0 = time.perf_counter()
    ro = trainer.collect_rollout(num_steps=cfg['num_steps'])
    dt = time.perf_counter() - t0
    acts = torch.stack(ro['actions'])            # [T, B, K, 3]
    rews = torch.stack(ro['rewards'])            # [T, B]
    lps = torch.stack(ro['log_probs'])           # [T, B, K]
    vals = torch.stack(ro['values'])             # [T, B]
    dones = torch.stack(ro['dones'])             # [T, B]
    n_terr = trainer.env.map_config.n_territories
    hist = {
        'source': torch.bincount(acts[..., 0].flatten(), minlength=n_terr).tolist(),
        'dest': torch.bincount(acts[..., 1].flatten(), minlength=n_terr).tolist(),
        'troops': torch.bincount(acts[..., 2].flatten(), minlength=20).tolist(),
    }
    # Also run one PPO update and record the loss-relevant stats (float)
    adv, ret = trainer.compute_gae(ro['rewards'], ro['values'], ro['dones'], ro['next_values'])
    trainer.update_policy(ro)
    param_sum = float(sum(p.detach().double().sum().item() for p in trainer.policy.parameters()))
    trainer.writer.close()
    return {
        'sample_count': int(acts.shape[0] * acts.shape[1]),
        'T': int(acts.shape[0]), 'B': int(acts.shape[1]),
        'action_histogram': hist,
        'actions_sha': sha(acts.tolist()),
        'dones_count': int(dones.sum().item()),
        'episodes_completed': len(trainer.episode_rewards),
        'mean_reward': float(rews.mean().item()),
        'sum_reward': float(rews.sum().item()),
        'mean_value': float(vals.mean().item()),
        'sum_log_prob': float(lps.sum().item()),
        'adv_mean': float(adv.mean().item()), 'adv_std': float(adv.std().item()),
        'ret_mean': float(ret.mean().item()),
        'param_sum_after_1_update': param_sum,
        '_wall_s': dt,
    }


def rec_e(ckpt):
    from parallel_risk import ParallelRiskEnv
    from parallel_risk.training.mcts_gnn import self_play as sp
    cfg = SELFPLAY
    env = ParallelRiskEnv(map_name=cfg['map_name'], max_turns=cfg['max_turns'],
                          seed=cfg['seed'], reward_shaping_config=None)
    seed_all(cfg['seed'])
    agent, src = build_gnn_agent(env.map_config, cfg['budget'], ckpt)
    sp_cfg = {'dirichlet_alpha': 0.3, 'noise_frac': 0.25, 'dirichlet_min_actions': 6,
              'temperature_turns': 5, 'max_temperature': 1.0}
    # Library calls np.random.default_rng() (unseeded) inside apply_root_dirichlet.
    # Fixture: make it return one seeded Generator for the whole episode.
    gen = np.random.default_rng(cfg['seed'])
    orig = np.random.default_rng
    np.random.default_rng = lambda *a, **k: gen
    t0 = time.perf_counter()
    try:
        ex = sp.play_episode(agent, env, sp_cfg, cfg['map_name'])
    finally:
        np.random.default_rng = orig
    dt = time.perf_counter() - t0
    return {
        'policy_source': src,
        'num_examples': len(ex),
        'examples': [{
            'agent_id': e['agent_id'],
            'visit_dist': [[key_list([k])[0], float(p)] for k, p in e['visit_dist'].items()],
            'z': float(e['z']),
        } for e in ex],
        'final_state': {'own': env.game_state['territory_ownership'].tolist(),
                        'troops': env.game_state['territory_troops'].tolist(),
                        'turn': int(env.game_state['turn_number'])},
        '_wall_s': dt,
    }


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------

class Cmp:
    def __init__(self, rtol, atol):
        self.rtol, self.atol, self.fails, self.nfloat, self.nexact = rtol, atol, [], 0, 0

    def walk(self, a, b, path, exact_paths):
        # a value is compared with tolerance iff it's a float and NOT under an
        # exact-only path; ints/str/bool/None/structure are always exact.
        if isinstance(a, dict) and isinstance(b, dict):
            if set(a) != set(b):
                self.fails.append(f"{path}: key set differs {sorted(set(a)^set(b))}"); return
            for k in a:
                if k.startswith('_'):
                    continue  # timing etc.
                self.walk(a[k], b[k], f"{path}.{k}", exact_paths)
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                self.fails.append(f"{path}: length {len(a)} != {len(b)}"); return
            for i, (x, y) in enumerate(zip(a, b)):
                self.walk(x, y, f"{path}[{i}]", exact_paths)
        elif isinstance(a, float) and isinstance(b, (float, int)) and not isinstance(b, bool) \
                and not any(path.startswith(p) for p in exact_paths):
            self.nfloat += 1
            if not (math.isclose(a, b, rel_tol=self.rtol, abs_tol=self.atol)):
                self.fails.append(f"{path}: float {a!r} vs {b!r} (|d|={abs(a-b):.3e})")
        else:
            self.nexact += 1
            if a != b:
                self.fails.append(f"{path}: {a!r} != {b!r}")


def compare(golden, new, rtol, atol):
    ok_all = True
    # Section -> list of sub-paths that must be EXACT even if they hold floats
    exact_only = {
        'a': ['a'],                            # everything in (a) is integer
        'b': ['b'],                            # n, available, best exact; q is float but MCTS-uniform Q is a mean of
                                               # +-1/0 -> keep exact by default (bit-identical is the expectation)
        'c': [f'c[{i}].{k}' for i in range(4) for k in ('agent_0.available', 'agent_0.n', 'agent_0.best',
                                                          'agent_1.available', 'agent_1.n', 'agent_1.best',
                                                          'root_visits', 'n_children', 'state', 'seed')],
        'd': ['d.sample_count', 'd.T', 'd.B', 'd.action_histogram', 'd.actions_sha', 'd.dones_count',
              'd.episodes_completed'],
        'e': ['e.num_examples', 'e.final_state'] + [f'e.examples[{i}].agent_id' for i in range(400)],
    }
    for sec in ('a', 'b', 'c', 'd', 'e'):
        if sec not in golden or sec not in new:
            print(f"[skip] section {sec} missing in one file"); continue
        c = Cmp(rtol, atol)
        c.walk(golden[sec], new[sec], sec, exact_only[sec])
        status = "PASS" if not c.fails else "FAIL"
        ok_all &= not c.fails
        print(f"[{status}] section {sec}: {c.nexact} exact + {c.nfloat} float comparisons, {len(c.fails)} mismatches")
        for f in c.fails[:8]:
            print(f"        {f}")
        if len(c.fails) > 8:
            print(f"        ... {len(c.fails)-8} more")
    return ok_all


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['record', 'compare'])
    ap.add_argument('--out', default=str(Path(__file__).parent / 'golden' / 'golden.json'))
    ap.add_argument('--golden'); ap.add_argument('--new')
    ap.add_argument('--ckpt', default=str(DEFAULT_CKPT))
    ap.add_argument('--skip', default='')
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--rtol', type=float, default=1e-5)
    ap.add_argument('--atol', type=float, default=1e-6)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    if args.mode == 'record':
        skip = set(s.strip() for s in args.skip.split(',') if s.strip())
        out = {'_meta': {'torch': torch.__version__, 'numpy': np.__version__,
                         'threads': args.threads, 'ckpt': args.ckpt,
                         'ckpt_exists': Path(args.ckpt).exists()}}
        total0 = time.perf_counter()
        for sec, fn in (('a', rec_a), ('b', rec_b), ('c', lambda: rec_c(args.ckpt)),
                        ('d', rec_d), ('e', lambda: rec_e(args.ckpt))):
            if sec in skip:
                continue
            t0 = time.perf_counter()
            out[sec] = fn()
            dt = time.perf_counter() - t0
            out['_meta'][f'wall_{sec}_s'] = dt
            print(f"recorded section {sec} in {dt:.1f}s")
        out['_meta']['wall_total_s'] = time.perf_counter() - total0
        Path(args.out).write_text(json.dumps(out, indent=1), encoding='utf-8')
        print(f"wrote {args.out}  ({Path(args.out).stat().st_size/1024:.0f} KB, total {out['_meta']['wall_total_s']:.1f}s)")
    else:
        g = json.loads(Path(args.golden).read_text(encoding='utf-8'))
        n = json.loads(Path(args.new).read_text(encoding='utf-8'))
        ok = compare(g, n, args.rtol, args.atol)
        print("ALL SECTIONS PASS" if ok else "SOME SECTIONS FAIL")
        sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
