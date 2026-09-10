"""ExIt (Expert Iteration) training run over all 9 unique maps.

Runs the ExIt loop (MCTS+GNN self-play + advantage-weighted actor-critic)
long enough to see whether the model is learning, evaluates it
periodically against MCTS(uniform) at the same search budget *on every
map*, and writes a 3-panel dashboard summarising the run.

Default map set: 9 unique maps registered in map_config.py (basic_6 alias
of simple_6 is excluded).

Usage (defaults are tuned for a ~2 hour run on a modern desktop):
    PYTHONPATH=. python experiments/mcts_gnn_exit_training_run.py
"""

from __future__ import annotations

# Load torch first: importing parallel_risk.training triggers this package's
# __init__ which pulls RLlib, whose try_import_torch path can fail on Windows
# if torch isn't already in sys.modules.
import torch  # noqa: F401

import argparse
import json
import math
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from parallel_risk.env.map_config import MapRegistry
from parallel_risk.training.mcts_gnn.exit_self_play import collect_games_exit
from parallel_risk.training.mcts_gnn.exit_trainer import ExitTrainer


# Default 9 unique maps (excluding basic_6 alias of simple_6).
DEFAULT_MAPS = [
    'simple_6', 'medium_8', 'large_10',
    'triangle_6', 'ring_6', 'star_8', 'double_hub_8',
    'hex_grid_10', 'dense_12',
]


# ---------------------------------------------------------------------------
# Evaluation (lockstep batched, see parallel_risk/training/mcts_gnn/lockstep.py)
# ---------------------------------------------------------------------------

def evaluate_all_maps(policy, model_kwargs, map_names, max_turns, action_budget,
                      mcts_budget, num_games_per_map, num_workers, max_regions,
                      base_seed, games_per_batch: int = 16,
                      device: str = 'cpu') -> Dict[str, Dict[str, float]]:
    """Evaluate MCTS+GNN(current) vs MCTS(uniform) on every map, in parallel.

    Each worker plays its share of the games in lockstep
    (`parallel_risk/training/mcts_gnn/lockstep.py`): one batched GNN forward
    per round serves all the games it holds, while the network-free
    MCTS-uniform side runs synchronously in the same round. Games are dealt
    to workers largest map first so the slow 30-territory games start early.
    Game g of map i (in `map_names` order) keeps the seed
    `base_seed + i * 1000 + g` and the colour of the per-game scheduling, so
    the same games are played.

    `games_per_batch=1` falls back to one game at a time per worker (exact
    single-graph forwards). `device` stays 'cpu': see the measurements in
    `parallel_risk/training/mcts_gnn/configs/mcts_gnn_exit.yaml`.
    """
    from parallel_risk.env.map_config import MapRegistry
    from parallel_risk.training.mcts_gnn.lockstep import (
        eval_lockstep_worker, eval_specs, summarize_eval_outcomes)

    state_dict_cpu = {k: v.detach().cpu() for k, v in policy.state_dict().items()}
    specs = eval_specs(map_names, num_games_per_map, base_seed)
    order = sorted(range(len(specs)),
                   key=lambda i: -MapRegistry.get(specs[i][0]).n_territories)
    chunks = [[] for _ in range(max(1, num_workers))]
    for pos, i in enumerate(order):
        chunks[pos % len(chunks)].append(specs[i])
    task_args = [(state_dict_cpu, model_kwargs, chunk, max_turns, action_budget,
                  mcts_budget, max_regions, games_per_batch, device)
                 for chunk in chunks if chunk]

    outcomes = []
    if len(task_args) <= 1:
        for args in task_args:
            outcomes.extend(eval_lockstep_worker(args))
    else:
        ctx = mp.get_context('spawn')
        with ProcessPoolExecutor(max_workers=len(task_args), mp_context=ctx) as ex:
            for res in ex.map(eval_lockstep_worker, task_args):
                outcomes.extend(res)

    counts = summarize_eval_outcomes(outcomes)
    per_map: Dict[str, Dict[str, float]] = {}
    for m in map_names:
        c = counts.get(m, {'wins': 0, 'losses': 0, 'draws': 0})
        total = c['wins'] + c['losses'] + c['draws']
        denom = max(total, 1)
        per_map[m] = {
            'wins': c['wins'], 'losses': c['losses'], 'draws': c['draws'],
            'total': total,
            'win_rate': c['wins'] / denom,
            'draw_rate': c['draws'] / denom,
            # Head-to-head score: draws count half. Used for the dashboard and
            # aggregate metric because several maps are draw-heavy.
            'score': (c['wins'] + 0.5 * c['draws']) / denom,
        }
    return per_map


# ---------------------------------------------------------------------------
# Plotting
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


def _map_colors(map_names):
    """Sorted-name colour lookup (matches experiments/multi_map_training.py)."""
    names_sorted = sorted(map_names)
    n = len(names_sorted)
    cmap = plt.get_cmap('tab10' if n <= 10 else 'tab20')
    return {name: cmap(i % cmap.N) for i, name in enumerate(names_sorted)}


def plot_dashboard(metrics: Dict[str, list],
                   eval_history: List[Dict[str, Any]],
                   output_path: Path, map_names: List[str]) -> None:
    cmap = plt.get_cmap('tab10')
    C_POLICY = cmap(0)
    C_VALUE = cmap(1)
    C_TOTAL = cmap(2)
    C_MEAN_EVAL = cmap(3)
    C_WIN = cmap(4)
    C_DRAW = cmap(5)
    C_LEN = cmap(6)
    map_color = _map_colors(map_names)

    iters = metrics['iteration']

    fig, axes = plt.subplots(3, 1, figsize=(11, 13), facecolor=_SURFACE)

    # ---- Panel A: losses ----------------------------------------------
    ax = axes[0]
    ax.plot(iters, metrics['policy_loss'], color=C_POLICY, lw=2, label='policy')
    ax.plot(iters, metrics['value_loss'], color=C_VALUE, lw=2, label='value')
    ax.plot(iters, metrics['total_loss'], color=C_TOTAL, lw=2, linestyle='--',
            label='total')
    ax.set_ylabel('loss', color=_INK_SECONDARY, fontsize=11)
    ax.set_title('ExIt training — losses',
                 color=_INK_PRIMARY, fontsize=13, fontweight='bold', pad=6)
    ax.legend(frameon=True, facecolor=_SURFACE, edgecolor=_GRIDLINE,
              labelcolor=_INK_SECONDARY, fontsize=10, loc='upper right')
    _chrome(ax)

    # ---- Panel B: per-map eval score + mean ---------------------------
    # Score = (wins + 0.5*draws) / n. Draw-heavy maps sit at 50 rather
    # than 0, so the mean draw-rate is drawn alongside to disambiguate
    # "parity" from "nobody wins".
    ax = axes[1]
    if eval_history:
        eval_iters = [e['iteration'] for e in eval_history]
        # One line per map
        for m in map_names:
            per_map = [e['per_map'].get(m, {}).get('score', float('nan')) * 100
                       for e in eval_history]
            ax.plot(eval_iters, per_map, color=map_color[m], lw=1.2,
                    marker='o', markersize=4, alpha=0.75, label=m)
        # Aggregate mean score
        mean_series = [e.get('mean_score', float('nan')) * 100
                       for e in eval_history]
        ax.plot(eval_iters, mean_series, color=C_MEAN_EVAL, lw=2.5,
                marker='s', markersize=8, markerfacecolor=_SURFACE,
                markeredgewidth=2, markeredgecolor=C_MEAN_EVAL,
                label='MEAN score', zorder=5)
        # Aggregate mean draw-rate
        draw_series = [e.get('mean_draw_rate', float('nan')) * 100
                       for e in eval_history]
        ax.plot(eval_iters, draw_series, color=C_DRAW, lw=2, linestyle=':',
                marker='^', markersize=6, label='MEAN draw %', zorder=4)
    ax.axhline(50, color=_INK_MUTED, lw=1, linestyle='--', alpha=0.6, zorder=1)
    ax.text(min(iters) if iters else 0, 51.5, '50 %',
            color=_INK_MUTED, fontsize=9, va='bottom')
    ax.set_ylim(0, 100)
    ax.set_ylabel('score (wins + 0.5·draws) / n  [%]',
                  color=_INK_SECONDARY, fontsize=11)
    ax.set_title('Eval per map: MCTS+GNN vs MCTS(uniform) at same budget',
                 color=_INK_PRIMARY, fontsize=13, fontweight='bold', pad=6)
    ax.legend(frameon=True, facecolor=_SURFACE, edgecolor=_GRIDLINE,
              labelcolor=_INK_SECONDARY, fontsize=8, loc='center left',
              bbox_to_anchor=(1.005, 0.5))
    _chrome(ax)

    # ---- Panel C: self-play sanity ------------------------------------
    ax = axes[2]
    ax.plot(iters, [w * 100 for w in metrics['selfplay_win_frac']],
            color=C_WIN, lw=1.5, label='agent_0 win %', alpha=0.85)
    ax.plot(iters, [d * 100 for d in metrics['selfplay_draw_frac']],
            color=C_DRAW, lw=1.5, label='draw %', alpha=0.85)
    ax.set_ylabel('%', color=_INK_SECONDARY, fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_xlabel('iteration', color=_INK_SECONDARY, fontsize=11)
    ax.legend(frameon=True, facecolor=_SURFACE, edgecolor=_GRIDLINE,
              labelcolor=_INK_SECONDARY, fontsize=10, loc='upper left')
    _chrome(ax)
    ax2 = ax.twinx()
    ax2.plot(iters, metrics['avg_game_length'], color=C_LEN, lw=1.5,
             linestyle=':', label='avg game length')
    ax2.set_ylabel('avg turns / game', color=C_LEN, fontsize=11)
    ax2.tick_params(colors=_INK_MUTED, labelsize=10)
    ax2.spines['top'].set_visible(False)
    ax2.legend(frameon=True, facecolor=_SURFACE, edgecolor=_GRIDLINE,
               labelcolor=_INK_SECONDARY, fontsize=10, loc='upper right')
    ax.set_title('Self-play sanity: outcome mix (aggregated) + game length',
                 color=_INK_PRIMARY, fontsize=13, fontweight='bold', pad=6)

    fig.suptitle(f'ExIt training run — {len(map_names)} maps',
                 color=_INK_PRIMARY, fontsize=14, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=150, facecolor=_SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Self-play sanity aggregation
# ---------------------------------------------------------------------------

def _selfplay_game_stats(examples: List[Dict[str, Any]]) -> Dict[str, float]:
    """Reduce a batch of examples to aggregated outcome + length stats.

    Groups by consecutive-same-z runs on agent_0 examples — approximate
    (misgroups two adjacent games with the same outcome) but adequate as
    a sanity trend line.
    """
    agent_0_zs = [ex['z'] for ex in examples if ex['agent_id'] == 'agent_0']
    outcomes: List[float] = []
    lengths: List[int] = []
    if agent_0_zs:
        cur = agent_0_zs[0]
        cur_len = 1
        for z in agent_0_zs[1:]:
            if z == cur:
                cur_len += 1
            else:
                outcomes.append(cur)
                lengths.append(cur_len)
                cur = z
                cur_len = 1
        outcomes.append(cur)
        lengths.append(cur_len)

    outcomes_arr = np.array(outcomes) if outcomes else np.zeros(0)
    if outcomes_arr.size == 0:
        return {'win_frac': 0.0, 'draw_frac': 0.0, 'avg_game_length': 0.0}
    return {
        'win_frac': float(np.mean(outcomes_arr > 0)),
        'draw_frac': float(np.mean(outcomes_arr == 0)),
        'avg_game_length': float(np.mean(lengths)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ExIt training run over multiple maps with dashboard plots"
    )
    parser.add_argument('--output-dir',
                        default='experiments/mcts_gnn_exit_training_run')
    parser.add_argument('--map-names', type=str, default=None,
                        help='Comma-separated list, or "all" for every registered '
                             'map (basic_6 alias excluded); defaults to the 9 '
                             'unique small maps.')
    parser.add_argument('--action-budget', type=int, default=5,
                        help='K: actions per turn for both self-play and eval.')
    parser.add_argument('--num-iterations', type=int, default=50)
    parser.add_argument('--num-games-per-iter', type=int, default=72)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--games-per-worker', type=int, default=16,
                        help='Games a worker advances in lockstep, sharing one batched '
                             'GNN forward per round. 1 = the old one-game-at-a-time worker.')
    parser.add_argument('--selfplay-device', default='cpu',
                        help="Where the batched self-play forwards run. Keep 'cpu': our "
                             'graphs are tiny and 12 CUDA contexts serialize on one device '
                             '(measured 4x slower). See configs/mcts_gnn_exit.yaml.')
    parser.add_argument('--num-epochs', type=int, default=4)
    parser.add_argument('--mcts-budget', type=int, default=40)
    parser.add_argument('--max-turns', type=int, default=40)
    parser.add_argument('--eval-interval', type=int, default=10)
    parser.add_argument('--num-eval-games', type=int, default=12,
                        help='Games per MAP per eval event.')
    parser.add_argument('--eval-workers', type=int, default=8)
    parser.add_argument('--use-gpu', dest='use_gpu', action='store_true', default=True)
    parser.add_argument('--no-gpu', dest='use_gpu', action='store_false')
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--num-layers', type=int, default=3)
    parser.add_argument('--checkpoint-interval', type=int, default=10)
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to a PPO/AZ/ExIt checkpoint to warm-start from.')
    parser.add_argument('--reset-optimizer', dest='reset_optimizer',
                        action='store_true', default=True)
    parser.add_argument('--keep-optimizer', dest='reset_optimizer',
                        action='store_false')
    parser.add_argument('--use-value-fn', dest='use_value_fn',
                        action='store_true', default=True,
                        help='SELF-PLAY only: use GNN value head at MCTS leaves. '
                             'Default True (matches inference-time behavior).')
    parser.add_argument('--no-value-fn', dest='use_value_fn', action='store_false',
                        help='SELF-PLAY only: fall back to random rollouts at MCTS '
                             'leaves. Use at cold-start when the value head is '
                             'still noise (AlphaGo-style bootstrap). Value head '
                             'still trained via MSE. Eval always uses value head.')
    args = parser.parse_args()

    if args.map_names is None:
        map_names = list(DEFAULT_MAPS)
    elif args.map_names.strip().lower() == 'all':
        map_names = sorted(m for m in MapRegistry.list_maps() if m != 'basic_6')
    else:
        map_names = [m.strip() for m in args.map_names.split(',')]
    K = args.action_budget

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / 'checkpoints'
    ckpt_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print(f"ExIt training run  |  maps={map_names}")
    print("=" * 70)
    print(f"  iterations={args.num_iterations}  games/iter={args.num_games_per_iter}  "
          f"workers={args.num_workers}  epochs/iter={args.num_epochs}")
    print(f"  mcts_budget={args.mcts_budget}  action_budget K={K}  "
          f"max_turns={args.max_turns}  eval every {args.eval_interval} iters "
          f"({args.num_eval_games} games/map, {args.eval_workers} workers)")
    print(f"  model: hidden={args.hidden_dim} layers={args.num_layers}  "
          f"trainer device={'GPU' if args.use_gpu else 'CPU'}")
    print(f"  self-play use_value_fn={args.use_value_fn}  "
          f"(eval always uses value head)")
    print(f"  output: {output_dir}")

    trainer_cfg = {
        'env': {'map_names': map_names, 'max_turns': args.max_turns,
                'action_budget': K},
        'model': {'type': 'gcn', 'hidden_dim': args.hidden_dim,
                  'num_layers': args.num_layers, 'dropout': 0.1},
        'trainer': {
            'learning_rate': 1e-4,
            'value_loss_coeff': 1.0,
            'entropy_coeff': 0.005,
            'max_grad_norm': 0.5,
            'use_gpu': args.use_gpu,
        },
        'log_dir': str(output_dir / 'runs'),
        'checkpoint_dir': str(ckpt_dir),
    }
    trainer = ExitTrainer(trainer_cfg)
    print(f"  trainer initialized on {trainer.device}  "
          f"(max_regions={trainer.max_regions}, "
          f"node_dim={trainer.node_features_dim})")

    if args.resume:
        loaded_iter = trainer.load_checkpoint(
            args.resume, load_optimizer=(not args.reset_optimizer),
        )
        print(f"  RESUMED from {args.resume} (iter {loaded_iter})  "
              f"optimizer_reset={args.reset_optimizer}")

    env_configs = [{'map_name': m, 'max_turns': args.max_turns} for m in map_names]
    mcts_config = {
        'simulation_budget': args.mcts_budget,
        'c_puct': 1.4, 'uct_c': 1.41, 'pw_alpha': 0.5,
        'max_rollout_turns': args.max_turns,
        'action_budget': K,
        'pw_sampler': 'masked_random',
        'use_value_fn': args.use_value_fn,
    }
    self_play_config = {
        'dirichlet_alpha': 0.3, 'noise_frac': 0.25, 'dirichlet_min_actions': 8,
        'temperature_turns': 10, 'max_temperature': 1.0,
    }

    metrics = {
        'iteration': [], 'policy_loss': [], 'value_loss': [], 'total_loss': [],
        'selfplay_win_frac': [], 'selfplay_draw_frac': [], 'avg_game_length': [],
        'collect_s': [], 'train_s': [],
    }
    eval_history: List[Dict[str, Any]] = []

    run_start = time.perf_counter()
    for it in range(args.num_iterations):
        t0 = time.perf_counter()
        examples = collect_games_exit(
            policy=trainer.policy,
            model_kwargs=trainer.model_kwargs(),
            env_configs=env_configs,
            mcts_config=mcts_config,
            self_play_config=self_play_config,
            num_games=args.num_games_per_iter,
            num_workers=args.num_workers,
            base_seed=it * 10000,
            max_regions=trainer.max_regions,
            games_per_worker=args.games_per_worker,
            device=args.selfplay_device,
        )
        collect_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        last: Dict[str, float] = {}
        for _ in range(args.num_epochs):
            last = trainer.update(examples)
        train_s = time.perf_counter() - t1

        sp = _selfplay_game_stats(examples)
        metrics['iteration'].append(it + 1)
        metrics['policy_loss'].append(float(last['policy_loss']))
        metrics['value_loss'].append(float(last['value_loss']))
        metrics['total_loss'].append(float(last['total_loss']))
        metrics['selfplay_win_frac'].append(sp['win_frac'])
        metrics['selfplay_draw_frac'].append(sp['draw_frac'])
        metrics['avg_game_length'].append(sp['avg_game_length'])
        metrics['collect_s'].append(collect_s)
        metrics['train_s'].append(train_s)

        for k in ('total_loss', 'policy_loss', 'value_loss'):
            assert math.isfinite(metrics[k][-1]), \
                f"iter {it}: non-finite {k}={metrics[k][-1]}"

        print(
            f"  iter {it+1:3d}/{args.num_iterations}  "
            f"examples={len(examples):5d}  "
            f"loss={last['total_loss']:6.3f} "
            f"(p={last['policy_loss']:5.3f} v={last['value_loss']:5.3f})  "
            f"win={sp['win_frac']:.2f} draw={sp['draw_frac']:.2f} "
            f"len={sp['avg_game_length']:.1f}  "
            f"collect={collect_s:4.1f}s train={train_s:4.1f}s"
        )

        if (it + 1) % args.eval_interval == 0:
            t2 = time.perf_counter()
            per_map = evaluate_all_maps(
                policy=trainer.policy,
                model_kwargs=trainer.model_kwargs(),
                map_names=map_names,
                max_turns=args.max_turns,
                action_budget=K,
                mcts_budget=args.mcts_budget,
                num_games_per_map=args.num_eval_games,
                num_workers=args.eval_workers,
                max_regions=trainer.max_regions,
                base_seed=100000 + (it + 1) * 137,
            )
            eval_s = time.perf_counter() - t2
            mean_wr = float(np.mean([per_map[m]['win_rate'] for m in map_names]))
            mean_score = float(np.mean([per_map[m]['score'] for m in map_names]))
            mean_draw = float(np.mean([per_map[m]['draw_rate'] for m in map_names]))
            eval_history.append({
                'iteration': it + 1,
                'per_map': per_map,
                'mean_win_rate': mean_wr,
                'mean_score': mean_score,
                'mean_draw_rate': mean_draw,
                'eval_time_s': eval_s,
            })
            per_map_summary = '  '.join(
                f"{m}={per_map[m]['score']:.0%}(d{per_map[m]['draws']})"
                for m in map_names)
            print(f"    [eval @ iter {it+1}]  mean_score={mean_score:.2%}  "
                  f"mean_win={mean_wr:.2%}  mean_draw={mean_draw:.2%}  "
                  f"({per_map_summary})  eval_time={eval_s:.1f}s")

        if (it + 1) % args.checkpoint_interval == 0:
            path = ckpt_dir / f"exit_iter_{it+1:06d}.pt"
            trainer.save_checkpoint(path, it + 1)

    run_s = time.perf_counter() - run_start
    trainer.save_checkpoint(ckpt_dir / f"exit_iter_{args.num_iterations:06d}.pt",
                            args.num_iterations)
    trainer.close()

    results = {
        'timestamp': datetime.now().isoformat(),
        'wall_clock_s': run_s,
        'args': vars(args),
        'map_names': map_names,
        'metrics': metrics,
        'eval_history': eval_history,
    }
    results_path = output_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics -> {results_path}")

    dashboard_path = output_dir / 'dashboard.png'
    plot_dashboard(metrics, eval_history, dashboard_path, map_names)
    print(f"Saved dashboard -> {dashboard_path}")

    print(f"\nTotal wall-clock: {run_s/60:.1f} min")
    if eval_history:
        first_mean = eval_history[0]['mean_score']
        last_mean = eval_history[-1]['mean_score']
        print(f"Mean eval score: first={first_mean:.2%}  last={last_mean:.2%}  "
              f"delta={(last_mean - first_mean):+.2%}")


if __name__ == "__main__":
    main()
