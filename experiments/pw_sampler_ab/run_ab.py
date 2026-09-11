"""Does the MCTS wrapper hurt because progressive widening proposes random actions?

No training. Takes the K=10 PPO checkpoint and plays it as MCTS+GNN(budget 200)
against MCTS-uniform(budget 200), twice: once with widening candidates drawn at
random (today's default) and once drawn from the policy (AlphaZero-style).
Same maps, same seeds, same colours, so the two arms play the same games.

If the mechanism is right, 'gnn' should gain most on the large maps, where the
random pool never contains the policy's preferred action, and little on
simple_6, where random sampling covers the space.
"""
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

REPO = Path(r'C:/Users/ricco/Documents/GitHub/parallel-risk-gym')
sys.path.insert(0, str(REPO))
OUT = Path(__file__).resolve().parent / 'ab_pw_sampler.json'
CKPT = REPO / 'experiments/k_sweep_ppo_200/K10/checkpoints/final.pt'
K, MR, MT, BUDGET, WORKERS = 10, 5, 40, 200, 12
MAPS = ['simple_6', 'medium_8', 'large_10', 'dense_12',
        'hub_spoke_16', 'hex_grid_18', 'grid_20', 'hub_ring_30']
GAMES_PER_MAP = int(sys.argv[1]) if len(sys.argv) > 1 else 12


def main():
    from parallel_risk.env.map_config import MapRegistry
    from parallel_risk.models.gnn_gcn import GCNPolicy
    from parallel_risk.training.mcts_gnn.lockstep import (
        eval_lockstep_worker, eval_specs, summarize_eval_outcomes)

    torch.set_num_threads(4)
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    kwargs = dict(node_features_dim=3 + MR, global_features_dim=2 + MR, hidden_dim=128,
                  num_layers=3, action_budget=K, max_troops=20, dropout=0.1)
    policy = GCNPolicy(**kwargs)
    policy.load_state_dict(ck['policy_state_dict'])
    policy.eval()
    state_dict = {k: v.detach().cpu() for k, v in policy.state_dict().items()}

    specs = eval_specs(MAPS, GAMES_PER_MAP, base_seed=7)
    order = sorted(range(len(specs)), key=lambda i: -MapRegistry.get(specs[i][0]).n_territories)
    print(f'{len(MAPS)} maps x {GAMES_PER_MAP} games = {len(specs)} games per arm, '
          f'budget {BUDGET}, K={K}, {WORKERS} workers', flush=True)

    results = {}
    for sampler in ('masked_random', 'gnn'):
        chunks = [[] for _ in range(WORKERS)]
        for pos, i in enumerate(order):
            chunks[pos % WORKERS].append(specs[i])
        args = [(state_dict, kwargs, ch, MT, K, BUDGET, MR, 16, 'cpu', sampler)
                for ch in chunks if ch]
        t0 = time.perf_counter()
        outcomes = []
        ctx = mp.get_context('spawn')
        with ProcessPoolExecutor(max_workers=WORKERS, mp_context=ctx) as ex:
            for res in ex.map(eval_lockstep_worker, args):
                outcomes.extend(res)
        dt = time.perf_counter() - t0
        per_map = summarize_eval_outcomes(outcomes)
        mean = sum(per_map[m]['score'] for m in MAPS) / len(MAPS)
        results[sampler] = {'per_map': per_map, 'mean_score': mean, 'wall_s': dt}
        print(f'\npw_sampler={sampler:14s} mean score {mean:.3f}   ({dt / 60:.1f} min)', flush=True)
        for m in MAPS:
            p = per_map[m]
            print(f'   {m:14s} n={MapRegistry.get(m).n_territories:2d}  score {p["score"]:.3f}  '
                  f'({p["wins"]}W/{p["losses"]}L/{p["draws"]}D)', flush=True)

    print('\n=== delta (gnn - masked_random), by map size ===')
    for m in sorted(MAPS, key=lambda x: MapRegistry.get(x).n_territories):
        a = results['masked_random']['per_map'][m]['score']
        b = results['gnn']['per_map'][m]['score']
        print(f'   {m:14s} n={MapRegistry.get(m).n_territories:2d}  '
              f'{a:.3f} -> {b:.3f}  {b - a:+.3f}')
    print(f"   {'MEAN':14s}       {results['masked_random']['mean_score']:.3f} -> "
          f"{results['gnn']['mean_score']:.3f}  "
          f"{results['gnn']['mean_score'] - results['masked_random']['mean_score']:+.3f}")
    OUT.write_text(json.dumps(results, indent=1), encoding='utf-8')
    print(f'\nsaved {OUT}')


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
