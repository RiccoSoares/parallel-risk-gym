"""Action-budget (K) sweep for PPO + GNN across the map roster.

For each K in the sweep, train a fresh PPO+GNN agent on the given map
roster (multi-map self-play) and evaluate periodically against MCTS-50
on every map. Aggregate results into a crossover dashboard:

- **Per-K per-map final win-rate** (heatmap)
- **Mean win-rate over training iterations, one line per K** (learning curves)
- **Compute vs performance trade-off** (peak win-rate vs K)

Motivation: at K=5 on 6-12 territory maps, we already know PPO scores
94-100% vs MCTS-50 (Phase 2.4 result). The interesting question is how
that performance changes as K grows and as maps get bigger — where does
each technique start to differentiate? This sweep gives the PPO baseline
across (K, map_size) that later MCTS+GNN sweeps will be compared against.

Usage:
    PYTHONPATH=. python experiments/k_sweep_ppo.py \
        --k-values 5,10,15,20 \
        --num-iterations 100 \
        --num-workers 8
"""

from __future__ import annotations

# Load torch first (Windows RLlib/torch import ordering guard).
import torch  # noqa: F401

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Default map roster (9 unique maps, basic_6 alias excluded)
# ---------------------------------------------------------------------------

DEFAULT_MAPS = [
    'simple_6', 'medium_8', 'large_10',
    'triangle_6', 'ring_6', 'star_8', 'double_hub_8',
    'hex_grid_10', 'dense_12',
]


# ---------------------------------------------------------------------------
# Plot chrome (matches other experiment dashboards on this repo)
# ---------------------------------------------------------------------------

_SURFACE = '#fcfcfb'
_INK_PRIMARY = '#0b0b0b'
_INK_SECONDARY = '#52514e'
_INK_MUTED = '#898781'
_GRIDLINE = '#e1e0d9'


def _chrome(ax):
    ax.set_facecolor(_SURFACE)
    ax.yaxis.grid(True, color=_GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    for spine in ('left', 'bottom'):
        ax.spines[spine].set_color(_GRIDLINE)
    ax.tick_params(colors=_INK_MUTED, labelsize=10)


# ---------------------------------------------------------------------------
# One-K training run (wraps multi_map_training.train_with_eval)
# ---------------------------------------------------------------------------

def train_one_k(K, map_names, num_iterations, num_workers, num_epochs,
                batch_size, mcts_budget, num_eval_episodes, eval_interval,
                output_dir, use_gpu):
    """Train PPO at action_budget=K on the map roster; return per-map + agg curves."""
    from experiments.multi_map_training import train_with_eval

    print(f"\n{'='*70}")
    print(f"K = {K}  (action_budget)")
    print(f"{'='*70}")

    k_dir = Path(output_dir) / f'K{K:02d}'
    k_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    trainer, per_map_win_rates, eval_iters = train_with_eval(
        map_names_to_train=map_names,
        num_iterations=num_iterations,
        eval_interval=eval_interval,
        num_episodes=num_eval_episodes,
        output_dir=str(k_dir),
        checkpoint_dir=k_dir / 'checkpoints',
        label=f'K={K}',
        verbose=True,
        batch_size=batch_size,
        num_epochs=num_epochs,
        save_weights_path=k_dir / 'checkpoints' / 'final.pt',
        mcts_budget=mcts_budget,
        num_workers=num_workers,
        use_gpu=use_gpu,
        parallel_eval=True,
        save_intermediate=False,   # per-K checkpoint every eval is too much
        action_budget=K,
    )
    wall = time.perf_counter() - t0

    trainer.writer.close()

    return {
        'K': K,
        'wall_clock_s': wall,
        'per_map_win_rates': per_map_win_rates,
        'eval_iterations': eval_iters,
        'final_win_rates': {m: (v[-1] if v else float('nan'))
                             for m, v in per_map_win_rates.items()},
    }


# ---------------------------------------------------------------------------
# Aggregation + plots
# ---------------------------------------------------------------------------

def _size_bucket(n_territories: int) -> str:
    if n_territories <= 12:
        return 'small (6-12)'
    if n_territories <= 22:
        return 'medium (16-22)'
    return 'large (28-30)'


def _map_sizes(map_names):
    from parallel_risk.env.map_config import MapRegistry
    return {m: MapRegistry.get(m).n_territories for m in map_names}


def make_dashboard(all_results: List[Dict[str, Any]], map_names: List[str],
                   output_path: Path):
    """4-panel main dashboard.

    A. Mean win-rate over iterations, one line per K, with shaded ±1σ
       across maps.
    B. Final per-map win-rate heatmap (K × maps sorted by n_territories),
       with bucket boundary lines.
    C. Per-bucket mean final win-rate (grouped bars, one group per bucket,
       one bar per K). Directly shows the K-crossover story as a function
       of map size — the point of the whole sweep.
    D. Peak mean win-rate vs K + wall-clock cost on secondary axis.
    """
    cmap = plt.get_cmap('viridis')
    K_values = sorted(r['K'] for r in all_results)
    k_color = {K: cmap(i / max(len(K_values) - 1, 1))
               for i, K in enumerate(K_values)}
    sizes = _map_sizes(map_names)
    # Sort maps by size (then by name for stable ties)
    sorted_maps = sorted(map_names, key=lambda m: (sizes[m], m))
    buckets = ['small (6-12)', 'medium (16-22)', 'large (28-30)']
    map_bucket = {m: _size_bucket(sizes[m]) for m in map_names}

    fig, axes = plt.subplots(4, 1, figsize=(13, 18), facecolor=_SURFACE)

    # ---- Panel A: learning curves with ±1σ shading ----
    ax = axes[0]
    for r in all_results:
        K = r['K']
        eval_iters = r['eval_iterations']
        if not eval_iters:
            continue
        matrix = np.array([
            r['per_map_win_rates'][m]
            for m in map_names if m in r['per_map_win_rates']
            and r['per_map_win_rates'][m]
        ])
        if matrix.size == 0:
            continue
        mean_curve = matrix.mean(axis=0) * 100
        std_curve = matrix.std(axis=0) * 100
        ax.plot(eval_iters, mean_curve, color=k_color[K], lw=2.5,
                marker='o', markersize=7, markerfacecolor=_SURFACE,
                markeredgewidth=2, markeredgecolor=k_color[K],
                label=f'K={K}  (peak={mean_curve.max():.1f}%)')
        ax.fill_between(eval_iters, mean_curve - std_curve,
                        mean_curve + std_curve, color=k_color[K], alpha=0.12)
    ax.axhline(50, color=_INK_MUTED, lw=1, linestyle='--', alpha=0.6, zorder=1)
    ax.set_ylim(0, 100)
    ax.set_xlabel('training iteration', color=_INK_SECONDARY, fontsize=11)
    ax.set_ylabel('mean win-rate vs MCTS-50 (%)', color=_INK_SECONDARY, fontsize=11)
    ax.set_title('A. Learning curves per action_budget K (shaded = ±1σ across maps)',
                 color=_INK_PRIMARY, fontsize=13, fontweight='bold', pad=6)
    ax.legend(frameon=True, facecolor=_SURFACE, edgecolor=_GRIDLINE,
              labelcolor=_INK_SECONDARY, fontsize=10, loc='lower right')
    _chrome(ax)

    # ---- Panel B: heatmap (K × sorted maps) with bucket separators ----
    ax = axes[1]
    heat = np.zeros((len(K_values), len(sorted_maps)))
    for i, K in enumerate(K_values):
        r = next(x for x in all_results if x['K'] == K)
        for j, m in enumerate(sorted_maps):
            heat[i, j] = r['final_win_rates'].get(m, np.nan) * 100
    im = ax.imshow(heat, aspect='auto', cmap='RdYlGn', vmin=0, vmax=100,
                   origin='lower')
    ax.set_xticks(range(len(sorted_maps)))
    ax.set_xticklabels([f'{m}\n({sizes[m]})' for m in sorted_maps],
                       rotation=45, ha='right', fontsize=8, color=_INK_SECONDARY)
    ax.set_yticks(range(len(K_values)))
    ax.set_yticklabels([f'K={K}' for K in K_values], color=_INK_SECONDARY, fontsize=10)
    # Bucket separator vertical lines (between last small and first medium, etc.)
    prev_bucket = map_bucket[sorted_maps[0]]
    for j, m in enumerate(sorted_maps[1:], start=1):
        if map_bucket[m] != prev_bucket:
            ax.axvline(j - 0.5, color='black', lw=1.5, alpha=0.7)
            prev_bucket = map_bucket[m]
    ax.set_title('B. Final per-map win-rate (%) — maps sorted by n_territories, black lines mark size buckets',
                 color=_INK_PRIMARY, fontsize=13, fontweight='bold', pad=6)
    cbar = fig.colorbar(im, ax=ax, shrink=0.7)
    cbar.ax.tick_params(labelsize=9, colors=_INK_MUTED)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            v = heat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f'{v:.0f}', ha='center', va='center',
                        color='black' if 30 < v < 70 else 'white',
                        fontsize=7, fontweight='bold')

    # ---- Panel C: per-bucket grouped bars ----
    ax = axes[2]
    bucket_maps = {b: [m for m in map_names if map_bucket[m] == b] for b in buckets}
    n_buckets = len(buckets)
    bar_w = 0.8 / max(len(K_values), 1)
    x_base = np.arange(n_buckets)
    for i, K in enumerate(K_values):
        r = next(x for x in all_results if x['K'] == K)
        means = []
        stds = []
        for b in buckets:
            vals = [r['final_win_rates'].get(m, np.nan) * 100
                    for m in bucket_maps[b] if m in r['final_win_rates']]
            vals = [v for v in vals if not np.isnan(v)]
            means.append(np.mean(vals) if vals else 0.0)
            stds.append(np.std(vals) if vals else 0.0)
        offset = (i - (len(K_values) - 1) / 2) * bar_w
        ax.bar(x_base + offset, means, width=bar_w * 0.9,
               yerr=stds, capsize=3, color=k_color[K], edgecolor=_SURFACE,
               linewidth=1.2, label=f'K={K}',
               error_kw={'ecolor': _INK_MUTED, 'lw': 1})
        # In-bar text
        for xb, m_val in zip(x_base + offset, means):
            if m_val > 5:
                ax.text(xb, m_val + 2, f'{m_val:.0f}', ha='center', va='bottom',
                        color=_INK_SECONDARY, fontsize=8, fontweight='bold')
    ax.axhline(50, color=_INK_MUTED, lw=1, linestyle='--', alpha=0.6, zorder=1)
    ax.set_xticks(x_base)
    ax.set_xticklabels([f'{b}\nn={len(bucket_maps[b])}' for b in buckets],
                       color=_INK_SECONDARY, fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_ylabel('final mean win-rate (%)', color=_INK_SECONDARY, fontsize=11)
    ax.set_title('C. Final win-rate by map-size bucket — the crossover story',
                 color=_INK_PRIMARY, fontsize=13, fontweight='bold', pad=6)
    ax.legend(frameon=True, facecolor=_SURFACE, edgecolor=_GRIDLINE,
              labelcolor=_INK_SECONDARY, fontsize=10, loc='upper right')
    _chrome(ax)

    # ---- Panel D: peak mean win-rate vs K + wall-clock cost ----
    ax = axes[3]
    peaks = []
    walls = []
    for K in K_values:
        r = next(x for x in all_results if x['K'] == K)
        matrix = np.array([
            r['per_map_win_rates'][m] for m in map_names
            if m in r['per_map_win_rates'] and r['per_map_win_rates'][m]
        ])
        if matrix.size == 0:
            peaks.append(np.nan); walls.append(np.nan); continue
        mean_curve = matrix.mean(axis=0)
        peaks.append(np.max(mean_curve) * 100)
        walls.append(r['wall_clock_s'] / 60.0)
    ax.plot(K_values, peaks, color=cmap(0.7), lw=2.5, marker='o', markersize=11,
            markerfacecolor=_SURFACE, markeredgewidth=2.5,
            markeredgecolor=cmap(0.7), label='peak mean win-rate', zorder=3)
    for K, p in zip(K_values, peaks):
        if not np.isnan(p):
            ax.text(K, p + 3, f'{p:.1f}%', ha='center', va='bottom',
                    color=cmap(0.7), fontsize=10, fontweight='bold')
    ax.axhline(50, color=_INK_MUTED, lw=1, linestyle='--', alpha=0.6, zorder=1)
    ax.set_ylim(0, 105)
    ax.set_xlabel('action_budget (K)', color=_INK_SECONDARY, fontsize=11)
    ax.set_ylabel('peak mean win-rate (%)', color=_INK_SECONDARY, fontsize=11)
    ax.set_title('D. Peak PPO performance vs K + wall-clock cost',
                 color=_INK_PRIMARY, fontsize=13, fontweight='bold', pad=6)
    ax.legend(frameon=True, facecolor=_SURFACE, edgecolor=_GRIDLINE,
              labelcolor=_INK_SECONDARY, fontsize=10, loc='center left')
    _chrome(ax)
    ax2 = ax.twinx()
    ax2.bar([K + 0.4 for K in K_values], walls, width=0.7,
            color=_INK_MUTED, alpha=0.25, zorder=1)
    ax2.set_ylabel('wall-clock per K (min)', color=_INK_MUTED, fontsize=11)
    ax2.tick_params(colors=_INK_MUTED, labelsize=10)
    ax2.spines['top'].set_visible(False)
    for K, w in zip(K_values, walls):
        if not np.isnan(w):
            ax2.text(K + 0.4, w + 5, f'{w:.0f}m',
                     ha='center', va='bottom', color=_INK_MUTED, fontsize=8)

    fig.suptitle('PPO+GNN action-budget sweep — Phase 3.5 (21 maps, sorted by size)',
                 color=_INK_PRIMARY, fontsize=15, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=200, facecolor=_SURFACE)
    plt.close(fig)


def make_per_map_trajectories(all_results: List[Dict[str, Any]],
                              map_names: List[str], output_path: Path):
    """Grid of per-map learning curves, one subplot per map."""
    K_values = sorted(r['K'] for r in all_results)
    cmap = plt.get_cmap('viridis')
    k_color = {K: cmap(i / max(len(K_values) - 1, 1))
               for i, K in enumerate(K_values)}
    sizes = _map_sizes(map_names)
    sorted_maps = sorted(map_names, key=lambda m: (sizes[m], m))

    n = len(sorted_maps)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                             facecolor=_SURFACE)
    axes_flat = axes.flatten()

    for idx, m in enumerate(sorted_maps):
        ax = axes_flat[idx]
        for r in all_results:
            K = r['K']
            series = r['per_map_win_rates'].get(m, [])
            iters = r['eval_iterations']
            if not series or not iters:
                continue
            ax.plot(iters, [v * 100 for v in series], color=k_color[K],
                    lw=1.7, marker='o', markersize=4, label=f'K={K}')
        ax.axhline(50, color=_INK_MUTED, lw=1, linestyle='--', alpha=0.5)
        ax.set_ylim(0, 100)
        ax.set_title(f'{m}  ({sizes[m]} territories)',
                     color=_INK_PRIMARY, fontsize=10, fontweight='bold')
        _chrome(ax)
        if idx == 0:
            ax.legend(frameon=True, facecolor=_SURFACE, edgecolor=_GRIDLINE,
                      labelcolor=_INK_SECONDARY, fontsize=8, loc='lower right')

    # Hide unused subplots
    for idx in range(n, len(axes_flat)):
        axes_flat[idx].axis('off')

    fig.suptitle('Per-map learning trajectories — win-rate vs MCTS-50 by iteration',
                 color=_INK_PRIMARY, fontsize=14, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=150, facecolor=_SURFACE)
    plt.close(fig)


def write_results_csv(all_results: List[Dict[str, Any]],
                      map_names: List[str], output_path: Path):
    """Wide-format CSV: rows = maps, cols = K values (final win-rate %)."""
    K_values = sorted(r['K'] for r in all_results)
    sizes = _map_sizes(map_names)
    sorted_maps = sorted(map_names, key=lambda m: (sizes[m], m))
    lines = ['map,n_territories,bucket,' + ','.join(f'K{K}' for K in K_values)]
    for m in sorted_maps:
        row = [m, str(sizes[m]), _size_bucket(sizes[m])]
        for K in K_values:
            r = next(x for x in all_results if x['K'] == K)
            v = r['final_win_rates'].get(m, float('nan'))
            row.append(f'{v*100:.1f}' if not (isinstance(v, float) and np.isnan(v)) else '')
        lines.append(','.join(row))
    # Aggregate rows
    for bucket in ['small (6-12)', 'medium (16-22)', 'large (28-30)']:
        bucket_maps = [m for m in map_names if _size_bucket(sizes[m]) == bucket]
        if not bucket_maps:
            continue
        row = [f'MEAN_{bucket}', '', bucket]
        for K in K_values:
            r = next(x for x in all_results if x['K'] == K)
            vals = [r['final_win_rates'].get(m, float('nan')) for m in bucket_maps]
            vals = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
            row.append(f'{100 * np.mean(vals):.1f}' if vals else '')
        lines.append(','.join(row))
    output_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PPO action-budget sweep")
    parser.add_argument('--output-dir', default='experiments/k_sweep_ppo')
    parser.add_argument('--map-names', default=None,
                        help='Comma-separated; default = 9 unique maps.')
    parser.add_argument('--k-values', default='5,10,15,20',
                        help='Comma-separated action_budget values to sweep.')
    parser.add_argument('--num-iterations', type=int, default=100)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--num-epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--mcts-budget', type=int, default=50,
                        help='MCTS opponent budget during eval (fixed across K).')
    parser.add_argument('--num-eval-episodes', type=int, default=25)
    parser.add_argument('--eval-interval', type=int, default=25)
    parser.add_argument('--use-gpu', dest='use_gpu', action='store_true', default=True)
    parser.add_argument('--no-gpu', dest='use_gpu', action='store_false')
    args = parser.parse_args()

    map_names = ([m.strip() for m in args.map_names.split(',')]
                 if args.map_names else list(DEFAULT_MAPS))
    K_values = [int(k.strip()) for k in args.k_values.split(',')]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PPO action-budget sweep")
    print(f"  K values:  {K_values}")
    print(f"  maps ({len(map_names)}): {map_names}")
    print(f"  iterations: {args.num_iterations}  |  eval every: {args.eval_interval}")
    print(f"  {args.num_eval_episodes} episodes/map/eval  vs  MCTS-{args.mcts_budget}")
    print(f"  {args.num_workers} workers  |  {'GPU' if args.use_gpu else 'CPU'}")
    print(f"  output: {output_dir}")
    print("=" * 70)

    all_results = []
    total_start = time.perf_counter()
    for K in K_values:
        r = train_one_k(
            K=K, map_names=map_names,
            num_iterations=args.num_iterations,
            num_workers=args.num_workers,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            mcts_budget=args.mcts_budget,
            num_eval_episodes=args.num_eval_episodes,
            eval_interval=args.eval_interval,
            output_dir=output_dir,
            use_gpu=args.use_gpu,
        )
        all_results.append(r)
        # Save incrementally so partial results survive a crash
        with open(output_dir / 'results_partial.json', 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'args': vars(args),
                'map_names': map_names,
                'results': all_results,
            }, f, indent=2, default=str)

    total_wall = time.perf_counter() - total_start

    # Final aggregated dump
    with open(output_dir / 'results.json', 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'args': vars(args),
            'total_wall_clock_s': total_wall,
            'map_names': map_names,
            'results': all_results,
        }, f, indent=2, default=str)

    dashboard = output_dir / 'dashboard.png'
    make_dashboard(all_results, map_names, dashboard)
    print(f"\nSaved dashboard             -> {dashboard}")

    traj_path = output_dir / 'per_map_trajectories.png'
    make_per_map_trajectories(all_results, map_names, traj_path)
    print(f"Saved per-map trajectories  -> {traj_path}")

    csv_path = output_dir / 'results_table.csv'
    write_results_csv(all_results, map_names, csv_path)
    print(f"Saved CSV table             -> {csv_path}")

    print(f"Saved results JSON          -> {output_dir / 'results.json'}")
    print(f"Total wall-clock: {total_wall/60:.1f} min")

    print("\nFinal per-K peak mean win-rate:")
    for r in all_results:
        matrix = np.array([
            r['per_map_win_rates'][m] for m in map_names
            if m in r['per_map_win_rates'] and r['per_map_win_rates'][m]
        ])
        peak = float(np.max(matrix.mean(axis=0))) if matrix.size else float('nan')
        print(f"  K={r['K']:2d}: peak mean = {peak:.2%}  "
              f"({r['wall_clock_s']/60:.1f} min)")


if __name__ == "__main__":
    main()
