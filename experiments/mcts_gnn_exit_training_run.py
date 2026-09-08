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
# Per-map eval worker (module-level for ProcessPoolExecutor pickling)
# ---------------------------------------------------------------------------

def _eval_worker(args):
    """Play num_games games of MCTSGNNAgent vs MCTSAgent(uniform) on one map.

    Alternates colors — half the games GNN plays agent_0, half agent_1.
    Runs on CPU in a spawned subprocess.
    """
    (state_dict_cpu, model_kwargs, map_name, max_turns, action_budget,
     mcts_budget, num_games, base_seed, max_regions) = args

    import numpy as np
    import torch
    from parallel_risk import ParallelRiskEnv
    from parallel_risk.agents.mcts_agent import MCTSAgent
    from parallel_risk.agents.mcts_gnn_agent import MCTSGNNAgent
    from parallel_risk.models.action_decoder import ActionDecoder
    from parallel_risk.models.gnn_gcn import GCNPolicy

    torch.set_num_threads(1)
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)

    policy = GCNPolicy(**model_kwargs)
    policy.load_state_dict(state_dict_cpu)
    policy.eval()

    env = ParallelRiskEnv(map_name=map_name, max_turns=max_turns,
                          reward_shaping_config=None)
    decoder = ActionDecoder(action_budget=action_budget, max_troops=20)
    gnn_agent = MCTSGNNAgent(
        policy=policy, decoder=decoder, map_config=env.map_config,
        simulation_budget=mcts_budget, c_puct=1.4, action_budget=action_budget,
        max_turns=max_turns, device='cpu', max_regions=max_regions,
    )
    uniform_agent = MCTSAgent.from_env(env, simulation_budget=mcts_budget,
                                       action_budget=action_budget)

    wins = losses = draws = 0
    for g in range(num_games):
        gnn_plays_0 = (g % 2 == 0)
        obs, _ = env.reset(seed=base_seed + g)
        done = False
        while not done:
            actions = {}
            for aid in ('agent_0', 'agent_1'):
                if aid not in obs:
                    continue
                pick_gnn = (aid == 'agent_0') == gnn_plays_0
                if pick_gnn:
                    actions[aid] = gnn_agent.get_action(env.game_state, aid)
                else:
                    actions[aid] = uniform_agent.get_action(env.game_state, aid)
            obs, rewards, terms, truncs, _ = env.step(actions)
            done = terms.get('__all__', False) or truncs.get('__all__', False)

        gnn_aid = 'agent_0' if gnn_plays_0 else 'agent_1'
        opp_aid = 'agent_1' if gnn_plays_0 else 'agent_0'
        r_g = float(rewards.get(gnn_aid, 0.0))
        r_o = float(rewards.get(opp_aid, 0.0))
        if r_g > r_o:
            wins += 1
        elif r_g < r_o:
            losses += 1
        else:
            draws += 1

    return map_name, {
        'wins': wins, 'losses': losses, 'draws': draws,
        'total': wins + losses + draws,
        'win_rate': wins / max(wins + losses + draws, 1),
    }


def evaluate_all_maps(policy, model_kwargs, map_names, max_turns, action_budget,
                      mcts_budget, num_games_per_map, num_workers, max_regions,
                      base_seed) -> Dict[str, Dict[str, float]]:
    """Evaluate MCTS+GNN(current) vs MCTS(uniform) on every map, in parallel."""
    state_dict_cpu = {k: v.detach().cpu() for k, v in policy.state_dict().items()}
    args_per_map = [
        (state_dict_cpu, model_kwargs, m, max_turns, action_budget,
         mcts_budget, num_games_per_map, base_seed + i * 1000, max_regions)
        for i, m in enumerate(map_names)
    ]
    ctx = mp.get_context('spawn')
    per_map: Dict[str, Dict[str, float]] = {}
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as ex:
        for map_name, res in ex.map(_eval_worker, args_per_map):
            per_map[map_name] = res
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

    # ---- Panel B: per-map eval win-rate + mean ------------------------
    ax = axes[1]
    if eval_history:
        eval_iters = [e['iteration'] for e in eval_history]
        # One line per map
        for m in map_names:
            per_map = [e['per_map'].get(m, {}).get('win_rate', float('nan')) * 100
                       for e in eval_history]
            ax.plot(eval_iters, per_map, color=map_color[m], lw=1.2,
                    marker='o', markersize=4, alpha=0.75, label=m)
        # Aggregate mean
        mean_series = [e.get('mean_win_rate', float('nan')) * 100
                       for e in eval_history]
        ax.plot(eval_iters, mean_series, color=C_MEAN_EVAL, lw=2.5,
                marker='s', markersize=8, markerfacecolor=_SURFACE,
                markeredgewidth=2, markeredgecolor=C_MEAN_EVAL,
                label='MEAN', zorder=5)
    ax.axhline(50, color=_INK_MUTED, lw=1, linestyle='--', alpha=0.6, zorder=1)
    ax.text(min(iters) if iters else 0, 51.5, '50 %',
            color=_INK_MUTED, fontsize=9, va='bottom')
    ax.set_ylim(0, 100)
    ax.set_ylabel('% vs MCTS(uniform)', color=_INK_SECONDARY, fontsize=11)
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

    fig.suptitle('ExIt training run — 9-map cold-start',
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
                        help='Comma-separated list; defaults to the 9 unique maps.')
    parser.add_argument('--num-iterations', type=int, default=50)
    parser.add_argument('--num-games-per-iter', type=int, default=72)
    parser.add_argument('--num-workers', type=int, default=8)
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

    map_names = ([m.strip() for m in args.map_names.split(',')]
                 if args.map_names else list(DEFAULT_MAPS))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / 'checkpoints'
    ckpt_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print(f"ExIt training run  |  maps={map_names}")
    print("=" * 70)
    print(f"  iterations={args.num_iterations}  games/iter={args.num_games_per_iter}  "
          f"workers={args.num_workers}  epochs/iter={args.num_epochs}")
    print(f"  mcts_budget={args.mcts_budget}  max_turns={args.max_turns}  "
          f"eval every {args.eval_interval} iters "
          f"({args.num_eval_games} games/map, {args.eval_workers} workers)")
    print(f"  model: hidden={args.hidden_dim} layers={args.num_layers}  "
          f"trainer device={'GPU' if args.use_gpu else 'CPU'}")
    print(f"  self-play use_value_fn={args.use_value_fn}  "
          f"(eval always uses value head)")
    print(f"  output: {output_dir}")

    trainer_cfg = {
        'env': {'map_names': map_names, 'max_turns': args.max_turns,
                'action_budget': 5},
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
        'action_budget': 5,
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
                action_budget=5,
                mcts_budget=args.mcts_budget,
                num_games_per_map=args.num_eval_games,
                num_workers=min(args.eval_workers, len(map_names)),
                max_regions=trainer.max_regions,
                base_seed=100000 + (it + 1) * 137,
            )
            eval_s = time.perf_counter() - t2
            mean_wr = float(np.mean([per_map[m]['win_rate'] for m in map_names]))
            eval_history.append({
                'iteration': it + 1,
                'per_map': per_map,
                'mean_win_rate': mean_wr,
                'eval_time_s': eval_s,
            })
            per_map_summary = '  '.join(f"{m}={per_map[m]['win_rate']:.0%}"
                                        for m in map_names)
            print(f"    [eval @ iter {it+1}]  mean={mean_wr:.2%}  "
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
        first_mean = eval_history[0]['mean_win_rate']
        last_mean = eval_history[-1]['mean_win_rate']
        print(f"Mean eval win-rate: first={first_mean:.2%}  last={last_mean:.2%}  "
              f"delta={(last_mean - first_mean):+.2%}")


if __name__ == "__main__":
    main()
