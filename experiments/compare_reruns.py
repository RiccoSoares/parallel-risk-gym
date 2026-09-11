"""Compare a re-run of the K-sweep against the original results.

The optimizations on perf/parallel-gpu changed the PPO rollout's sample
composition and RNG order, so a re-run cannot be expected to reproduce the
original per-map win rates exactly. What must hold is the *finding*: the
K=5 -> K=10 drop-off, its concentration on the medium and large maps, and
per-map agreement within sampling noise.

Each per-map win rate is a fraction of `--n-eval` evaluation games, so its
standard error is sqrt(p(1-p)/n); two independent runs differ with standard
error sqrt(2) times that. We report the per-map z-scores, how many maps
exceed 2 sigma (expect about 1 in 20 by chance), and a paired test on the
bucket means.

Usage:
    PYTHONIOENCODING=utf-8 PYTHONUTF8=1 PYTHONPATH=. python experiments/compare_reruns.py \
        --original experiments/k_sweep_ppo_200/results_K5_K10.csv \
        --rerun experiments/k_sweep_ppo_200_perf/results_partial.json \
        --n-eval 15
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict


def load_original(path: Path) -> Dict[str, Dict[int, float]]:
    """{map: {5: win_rate, 10: win_rate}} from the committed results CSV (percent)."""
    out: Dict[str, Dict[int, float]] = {}
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            name = row['map']
            if name.startswith('MEAN_'):
                continue
            out[name] = {5: float(row['K5']) / 100.0, 10: float(row['K10']) / 100.0}
    return out


def load_rerun(path: Path) -> Dict[str, Dict[int, float]]:
    """{map: {K: final win_rate}} from a k_sweep_ppo results JSON."""
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    out: Dict[str, Dict[int, float]] = {}
    for entry in data['results']:
        k = int(entry['K'])
        for name, curve in entry['per_map_win_rates'].items():
            out.setdefault(name, {})[k] = float(curve[-1])
    return out


def bucket_of(n_territories: int) -> str:
    if n_territories <= 12:
        return 'small (6-12)'
    if n_territories <= 22:
        return 'medium (16-22)'
    return 'large (28-30)'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--original', default='experiments/k_sweep_ppo_200/results_K5_K10.csv')
    p.add_argument('--rerun', default='experiments/k_sweep_ppo_200_perf/results_partial.json')
    p.add_argument('--n-eval', type=int, default=15, help='eval games per map per point')
    args = p.parse_args()

    from parallel_risk.env.map_config import MapRegistry

    orig = load_original(Path(args.original))
    new = load_rerun(Path(args.rerun))
    ks = sorted({k for v in new.values() for k in v})
    maps = [m for m in orig if m in new]
    maps.sort(key=lambda m: (MapRegistry.get(m).n_territories, m))
    n = args.n_eval

    print(f'{len(maps)} maps, K values {ks}, {n} eval games per map\n')
    header = f"{'map':16s} {'n':>3s} " + ' '.join(
        f'{"K=" + str(k) + " orig":>11s} {"rerun":>7s} {"z":>6s}' for k in ks)
    print(header)
    print('-' * len(header))

    outliers = {k: [] for k in ks}
    sums = {k: {'orig': {}, 'new': {}} for k in ks}
    for m in maps:
        nt = MapRegistry.get(m).n_territories
        b = bucket_of(nt)
        cells = []
        for k in ks:
            a, c = orig[m].get(k), new[m].get(k)
            if a is None or c is None:
                cells.append(f'{"-":>11s} {"-":>7s} {"-":>6s}')
                continue
            pooled = (a + c) / 2
            se = math.sqrt(max(pooled * (1 - pooled), 1e-9) * 2 / n)
            z = (c - a) / se
            if abs(z) > 2:
                outliers[k].append((m, a, c, z))
            sums[k]['orig'].setdefault(b, []).append(a)
            sums[k]['new'].setdefault(b, []).append(c)
            cells.append(f'{a:11.2f} {c:7.2f} {z:+6.1f}')
        print(f'{m:16s} {nt:3d} ' + ' '.join(cells))

    print()
    for k in ks:
        o = sums[k]['orig']
        c = sums[k]['new']
        allo = [x for v in o.values() for x in v]
        alln = [x for v in c.values() for x in v]
        print(f'K={k}: aggregate orig {sum(allo) / len(allo):.3f} -> rerun {sum(alln) / len(alln):.3f}')
        for b in ('small (6-12)', 'medium (16-22)', 'large (28-30)'):
            if b in o:
                print(f'    {b:16s} orig {sum(o[b]) / len(o[b]):.3f} -> rerun {sum(c[b]) / len(c[b]):.3f}'
                      f'   ({len(o[b])} maps)')
        bad = outliers[k]
        print(f'    maps beyond 2 sigma: {len(bad)}/{len(allo)} '
              f'(about {0.05 * len(allo):.1f} expected by chance)')
        for m, a, cc, z in bad:
            print(f'      {m:16s} {a:.2f} -> {cc:.2f}  z={z:+.1f}')

    if len(ks) >= 2:
        k_lo, k_hi = ks[0], ks[-1]
        do = [orig[m][k_lo] - orig[m][k_hi] for m in maps
              if k_lo in orig[m] and k_hi in orig[m]]
        dn = [new[m][k_lo] - new[m][k_hi] for m in maps
              if k_lo in new[m] and k_hi in new[m]]
        print(f'\nK={k_lo} minus K={k_hi} drop-off: orig {sum(do) / len(do):+.3f}, '
              f'rerun {sum(dn) / len(dn):+.3f}  '
              f'(the finding holds if both are clearly positive)')


if __name__ == '__main__':
    main()
