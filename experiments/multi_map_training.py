"""
Multi-map training experiment for Parallel Risk GNN.

Trains a single GNN on all 3 maps simultaneously, evaluates per-map performance
over training, and runs a transfer-learning comparison (2-map vs 3-map model).

Usage:
    python experiments/multi_map_training.py --quick --output-dir /tmp/multimap_test
    python experiments/multi_map_training.py --num-iterations 200
"""

import argparse
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — safe for scripts without a display
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Validated categorical palette (dataviz skill, slots 1-3, adjacent-pair safe)
# slot 1 blue   #2a78d6   simple_6
# slot 2 orange #eb6834   medium_8
# slot 3 aqua   #1baf7a   large_10
# ---------------------------------------------------------------------------
_CAT = {
    'simple_6': '#2a78d6',
    'medium_8': '#eb6834',
    'large_10': '#1baf7a',
}
_SURFACE = '#fcfcfb'
_INK_PRIMARY = '#0b0b0b'
_INK_SECONDARY = '#52514e'
_INK_MUTED = '#898781'
_GRIDLINE = '#e1e0d9'


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def build_config(map_names, output_dir, checkpoint_dir, action_budget=5,
                 batch_size=4096, num_epochs=10, num_workers=1, use_gpu=True):
    """Build a PPOTrainer-compatible config dict for the given map list."""
    return {
        'env': {
            'map_names': list(map_names),
            'max_turns': 50,
            'action_budget': action_budget,
            'seed': None,
            'use_reward_shaping': True,
        },
        'model': {
            'type': 'gcn',
            'hidden_dim': 128,
            'num_layers': 3,
            'dropout': 0.1,
        },
        'training': {
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'num_workers': num_workers,
            'learning_rate': 1e-4,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_epsilon': 0.2,
            'entropy_coeff': 0.01,
            'value_loss_coeff': 0.5,
            'max_grad_norm': 0.5,
            'use_gpu': use_gpu,
        },
        'log_dir': str(Path(output_dir) / 'runs'),
        'checkpoint_dir': str(checkpoint_dir),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_policy_vs_mcts(policy, action_decoder, map_name, action_budget,
                            num_episodes, mcts_budget=50, seed_offset=0):
    """
    Evaluate a GNN policy (agent_0) vs an MCTS opponent (agent_1) on one map.

    Uses the raw env so MCTS can access env.game_state directly, while the GNN
    receives agent-relative obs dicts converted to PyG graphs via env_to_graph —
    exactly the same observation type the policy was trained on.

    Policy is set to eval() mode on entry; caller should switch it back to
    train() if training will continue.

    Returns:
        dict with win_rate, wins, losses, draws, total_episodes, avg_episode_length
    """
    from parallel_risk import ParallelRiskEnv
    from parallel_risk.agents.mcts_agent import MCTSAgent
    from parallel_risk.training.torchrl.graph_wrapper import env_to_graph
    from torch_geometric.data import Batch

    device = torch.device('cpu')
    env = ParallelRiskEnv(map_name=map_name, max_turns=50, seed=None,
                          reward_shaping_config=None)
    # Create MCTS once — it is map-specific (holds map_config internally)
    mcts_agent = MCTSAgent.from_env(env, simulation_budget=mcts_budget,
                                    action_budget=action_budget)

    policy.eval()

    wins = losses = draws = 0
    episode_lengths = []

    for episode in range(num_episodes):
        obs, _ = env.reset(seed=42 + seed_offset + episode)
        done = False
        episode_length = 0

        while not done:
            actions = {}

            if 'agent_0' in obs:
                graph = env_to_graph(obs['agent_0'], env.map_config, device)
                with torch.no_grad():
                    batched = Batch.from_data_list([graph])
                    action_logits, _, _ = policy(batched)
                    actions_tensor, _ = action_decoder.decode_actions(
                        action_logits, batched.batch,
                        deterministic=False, return_log_probs=False,
                        observations=[graph],
                    )
                action_array = actions_tensor[0].cpu().numpy()
                actions['agent_0'] = {
                    'num_actions': action_budget,
                    'actions': np.vstack([
                        action_array,
                        np.zeros((10 - action_budget, 3)),
                    ]),
                }

            if 'agent_1' in obs:
                actions['agent_1'] = mcts_agent.get_action(env.game_state, 'agent_1')

            obs, rewards, terminateds, truncateds, _ = env.step(actions)
            done = (terminateds.get('__all__', False)
                    or truncateds.get('__all__', False))
            episode_length += 1

        r0 = rewards.get('agent_0', 0)
        r1 = rewards.get('agent_1', 0)
        if r0 > r1:
            wins += 1
        elif r0 < r1:
            losses += 1
        else:
            draws += 1
        episode_lengths.append(episode_length)

    total = wins + losses + draws
    return {
        'win_rate': wins / total if total > 0 else 0.0,
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'total_episodes': total,
        'avg_episode_length': float(np.mean(episode_lengths)),
    }


# ---------------------------------------------------------------------------
# Parallel-eval worker (module-level so it's picklable by ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _eval_worker(args):
    """Rebuild the policy from a CPU state_dict and run evaluate_policy_vs_mcts.

    Runs in a spawned subprocess — everything imported here is duplicated per
    worker. Kept minimal for spawn cost. The GPU is not touched.
    """
    (state_dict_cpu, model_kwargs, map_name, action_budget,
     num_episodes, mcts_budget, seed_offset) = args

    import torch
    from parallel_risk.models.gnn_gcn import GCNPolicy
    from parallel_risk.models.action_decoder import ActionDecoder

    torch.set_num_threads(1)  # avoid oversubscription with sibling workers

    policy = GCNPolicy(**model_kwargs)
    policy.load_state_dict(state_dict_cpu)
    policy.eval()

    action_decoder = ActionDecoder(action_budget=action_budget, max_troops=20)

    result = evaluate_policy_vs_mcts(
        policy, action_decoder, map_name, action_budget,
        num_episodes, mcts_budget=mcts_budget, seed_offset=seed_offset,
    )
    return map_name, result


def _snapshot_policy_for_eval(trainer):
    """Return (state_dict_cpu, model_kwargs) suitable to hand to _eval_worker."""
    state_dict_cpu = {k: v.detach().cpu()
                      for k, v in trainer.policy.state_dict().items()}
    model_kwargs = dict(
        node_features_dim=trainer.node_features_dim,
        global_features_dim=trainer.global_features_dim,
        hidden_dim=trainer.policy.hidden_dim,
        num_layers=trainer.policy.num_layers,
        action_budget=trainer.action_budget,
        max_troops=20,
        dropout=trainer.policy.dropout,
    )
    return state_dict_cpu, model_kwargs


# ---------------------------------------------------------------------------
# Training loop with inline evaluation
# ---------------------------------------------------------------------------

def _save_training_checkpoint(trainer, path, iteration, extras=None):
    """
    Persist policy + optimizer + config to ``path``. Matches the checkpoint
    schema used by ``PPOTrainer.train`` in ``parallel_risk/training/torchrl/train.py``
    so downstream tools can load either interchangeably.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'iteration': iteration,
        'policy_state_dict': trainer.policy.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'config': trainer.config,
    }
    if extras:
        payload.update(extras)
    torch.save(payload, path)


def train_with_eval(map_names_to_train, num_iterations, eval_interval,
                    num_episodes, output_dir, checkpoint_dir,
                    label='', verbose=True, batch_size=4096, num_epochs=10,
                    save_weights_path=None, mcts_budget=50,
                    num_workers=1, use_gpu=True, parallel_eval=True,
                    save_intermediate=True):
    """
    Train a multi-map (or single-map) GNN with per-map evaluation at regular
    intervals.  Manages the PPO loop directly so evaluation happens in-memory
    without a checkpoint round-trip.

    Args:
        save_weights_path: If provided, saves final policy weights (state_dict)
                           to this path after training completes.
        save_intermediate: If True (default), also drop a
                           ``checkpoint_{iter:06d}.pt`` into ``checkpoint_dir``
                           at every eval interval. Needed to rerun evals
                           against past training states without retraining.

    Returns:
        trainer         — PPOTrainer (policy accessible via trainer.policy)
        per_map_win_rates — {map_name: [win_rate_at_eval_0, ...]}
        eval_iterations — [iteration_number, ...] matching the win_rate lists
    """
    from parallel_risk.training.torchrl.train import PPOTrainer

    config = build_config(
        map_names=map_names_to_train,
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        batch_size=batch_size,
        num_epochs=num_epochs,
        num_workers=num_workers,
        use_gpu=use_gpu,
    )

    label_str = label or ', '.join(map_names_to_train)
    print(f"\n{'='*60}")
    print(f"Training: {label_str}")
    print(f"  Maps:            {map_names_to_train}")
    print(f"  Iterations:      {num_iterations}  |  Eval every: {eval_interval}")
    print(f"  Episodes/map:    {num_episodes}")
    print('='*60)

    trainer = PPOTrainer(config)
    action_budget = trainer.action_budget

    per_map_win_rates = {name: [] for name in map_names_to_train}
    eval_iterations = []

    log_every = max(1, num_iterations // 10)

    # Persistent eval executor. One worker per map so all evals run in
    # parallel. Created lazily on first eval to avoid spawn cost when there
    # are zero eval intervals.
    eval_executor = None
    if parallel_eval and len(map_names_to_train) > 1:
        eval_executor = ProcessPoolExecutor(
            max_workers=len(map_names_to_train),
            mp_context=mp.get_context('spawn'),
        )

    try:
        for iteration in range(num_iterations):
            # Collect and update — mirrors what PPOTrainer.train() does internally
            rollout = trainer.collect_rollout(trainer.batch_size // 2)
            trainer.update_policy(rollout)

            if (iteration + 1) % eval_interval == 0:
                eval_iterations.append(iteration + 1)
                if verbose:
                    print(f"\n  -- eval @ iter {iteration+1} --")

                if save_intermediate:
                    ckpt_path = Path(checkpoint_dir) / f"checkpoint_{iteration+1:06d}.pt"
                    _save_training_checkpoint(
                        trainer, ckpt_path, iteration + 1,
                        extras={'map_names': list(map_names_to_train)},
                    )
                    if verbose:
                        print(f"    saved checkpoint: {ckpt_path}")

                if eval_executor is not None:
                    # Parallel: submit one job per map, wait for all.
                    state_dict_cpu, model_kwargs = _snapshot_policy_for_eval(trainer)
                    futures = {}
                    for idx, map_name in enumerate(map_names_to_train):
                        args = (state_dict_cpu, model_kwargs, map_name,
                                action_budget, num_episodes, mcts_budget,
                                idx * 1000)
                        futures[map_name] = eval_executor.submit(_eval_worker, args)

                    for map_name in map_names_to_train:
                        _, result = futures[map_name].result()
                        per_map_win_rates[map_name].append(result['win_rate'])
                        if verbose:
                            print(f"    {map_name}: {result['win_rate']:.2%}  "
                                  f"({result['wins']}W/{result['losses']}L/"
                                  f"{result['draws']}D)")
                else:
                    # Sequential fallback (single-map runs or --serial-eval).
                    for idx, map_name in enumerate(map_names_to_train):
                        result = evaluate_policy_vs_mcts(
                            trainer.policy, trainer.action_decoder,
                            map_name, action_budget, num_episodes,
                            mcts_budget=mcts_budget,
                            seed_offset=idx * 1000,
                        )
                        per_map_win_rates[map_name].append(result['win_rate'])
                        if verbose:
                            print(f"    {map_name}: {result['win_rate']:.2%}  "
                                  f"({result['wins']}W/{result['losses']}L/"
                                  f"{result['draws']}D)")

                # Return policy to training mode for next iteration
                trainer.policy.train()

            elif verbose and (iteration + 1) % log_every == 0:
                if trainer.episode_rewards:
                    avg = np.mean(trainer.episode_rewards[-20:])
                    n_ep = len(trainer.episode_rewards)
                    print(f"  iter {iteration+1:4d}/{num_iterations}  "
                          f"avg_reward={avg:+.3f}  episodes={n_ep}")
    finally:
        if eval_executor is not None:
            eval_executor.shutdown(wait=True)

    trainer.writer.close()

    if save_weights_path is not None:
        save_weights_path = Path(save_weights_path)
        save_weights_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'policy_state_dict': trainer.policy.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'iteration': num_iterations,
            'config': trainer.config,
            'map_names': map_names_to_train,
            'per_map_win_rates': per_map_win_rates,
            'eval_iterations': eval_iterations,
        }, save_weights_path)
        print(f"  Saved weights: {save_weights_path}")

    return trainer, per_map_win_rates, eval_iterations


# ---------------------------------------------------------------------------
# Transfer test
# ---------------------------------------------------------------------------

def run_transfer_test(full_trainer, num_iterations, eval_interval,
                      num_episodes, output_dir, checkpoint_dir, verbose=True,
                      batch_size=4096, num_epochs=10, mcts_budget=50,
                      num_workers=1, use_gpu=True, parallel_eval=True):
    """
    Train a 2-map model (simple_6 + medium_8 only) and compare zero-shot
    performance on large_10 against the already-trained 3-map model.

    Returns:
        dict:
            '2map_on_large10' — eval result dict
            '3map_on_large10' — eval result dict
    """
    TWO_MAPS = ['simple_6', 'medium_8']

    print(f"\n{'='*60}")
    print("Transfer test: 2-map model vs 3-map model on large_10")
    print('='*60)

    two_map_ckpt_dir = Path(checkpoint_dir) / 'transfer_2map'
    two_map_trainer, _, _ = train_with_eval(
        map_names_to_train=TWO_MAPS,
        num_iterations=num_iterations,
        eval_interval=eval_interval,
        num_episodes=num_episodes,
        output_dir=output_dir,
        checkpoint_dir=two_map_ckpt_dir,
        label='2-map model (simple_6 + medium_8)',
        verbose=verbose,
        batch_size=batch_size,
        num_epochs=num_epochs,
        save_weights_path=two_map_ckpt_dir / 'final.pt',
        mcts_budget=mcts_budget,
        num_workers=num_workers,
        use_gpu=use_gpu,
        parallel_eval=parallel_eval,
    )

    action_budget = two_map_trainer.action_budget

    # evaluate_policy_vs_mcts runs on CPU; if either trainer's policy sits on
    # GPU we need to move it before calling the sequential eval helper.
    def _eval_cpu(trainer, seed_offset):
        original_device = trainer.device
        trainer.policy.to('cpu')
        try:
            return evaluate_policy_vs_mcts(
                trainer.policy, trainer.action_decoder,
                'large_10', action_budget, num_episodes * 2,
                mcts_budget=mcts_budget, seed_offset=seed_offset,
            )
        finally:
            trainer.policy.to(original_device)

    print("\n  Evaluating 2-map model on large_10 (zero-shot) ...")
    result_2map = _eval_cpu(two_map_trainer, seed_offset=5000)

    print("  Evaluating 3-map model on large_10 ...")
    result_3map = _eval_cpu(full_trainer, seed_offset=6000)
    # Restore training mode (policy will not be trained further, but good practice)
    full_trainer.policy.train()

    print(f"\n  2-map (zero-shot) on large_10: {result_2map['win_rate']:.2%}  "
          f"({result_2map['wins']}W/{result_2map['losses']}L/{result_2map['draws']}D)")
    print(f"  3-map (trained on it) on large_10: {result_3map['win_rate']:.2%}  "
          f"({result_3map['wins']}W/{result_3map['losses']}L/{result_3map['draws']}D)")

    return {'2map_on_large10': result_2map, '3map_on_large10': result_3map}


# ---------------------------------------------------------------------------
# Plots  (form → color → mark specs → labels, per dataviz skill)
# ---------------------------------------------------------------------------

def _apply_chart_chrome(ax):
    """Apply shared axes/grid style."""
    ax.yaxis.grid(True, color=_GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(_GRIDLINE)
    ax.spines['bottom'].set_color(_GRIDLINE)
    ax.tick_params(colors=_INK_MUTED, labelsize=10)


def plot_learning_curves(per_map_win_rates, eval_iterations, output_path):
    """
    Line chart — 3 series (one per map), win rate over training iterations.
    Form: change-over-time → line chart.
    Colors: categorical slots 1–3 (blue/orange/aqua), same entity colours as
    final_performance chart to ensure cross-chart consistency.
    """
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=_SURFACE)
    ax.set_facecolor(_SURFACE)

    for map_name, win_rates in per_map_win_rates.items():
        if not win_rates:
            continue
        color = _CAT.get(map_name, '#888888')
        xs = eval_iterations[:len(win_rates)]
        ys = [wr * 100 for wr in win_rates]
        ax.plot(
            xs, ys,
            linewidth=2,
            color=color,
            marker='o',
            markersize=7,
            markerfacecolor=_SURFACE,
            markeredgewidth=2,
            markeredgecolor=color,
            label=map_name,
            zorder=3,
        )

    # 50 % reference
    ax.axhline(50, color=_INK_MUTED, linewidth=1, linestyle='--', alpha=0.55, zorder=1)
    ax.text(
        ax.get_xlim()[0] if ax.get_xlim()[0] != 0 else (eval_iterations[0] if eval_iterations else 0),
        51.5, '50 %', color=_INK_MUTED, fontsize=9, va='bottom',
    )

    ax.set_xlabel("Training iteration", color=_INK_SECONDARY, fontsize=11)
    ax.set_ylabel("Win rate vs masked-random (%)", color=_INK_SECONDARY, fontsize=11)
    ax.set_title("Multi-map GNN — learning curves per map",
                 color=_INK_PRIMARY, fontsize=13, fontweight='bold', pad=10)
    ax.set_ylim(0, 108)

    _apply_chart_chrome(ax)

    # Legend — always present for ≥ 2 series
    ax.legend(
        frameon=True, facecolor=_SURFACE, edgecolor=_GRIDLINE,
        labelcolor=_INK_SECONDARY, fontsize=10, loc='lower right',
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=_SURFACE)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_final_performance(final_win_rates, output_path):
    """
    Bar chart — final win rate per map.
    Form: magnitude per nominal category → bar chart.
    One series per bar; each map keeps its categorical slot colour so readers
    can link this chart to the learning-curves chart by colour.
    Direct value labels satisfy the relief rule (sub-3:1 fill for aqua on light).
    """
    maps = list(final_win_rates.keys())
    values = [final_win_rates[m] * 100 for m in maps]
    colors = [_CAT.get(m, '#888888') for m in maps]

    fig, ax = plt.subplots(figsize=(5.5, 4), facecolor=_SURFACE)
    ax.set_facecolor(_SURFACE)

    bars = ax.bar(maps, values, color=colors, width=0.5, zorder=3)

    # 2 px surface-colour edge creates a visual gap between adjacent fills
    for bar in bars:
        bar.set_edgecolor(_SURFACE)
        bar.set_linewidth(1.5)

    # Direct value labels (required: relief for low-contrast fills)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{val:.1f}%",
            ha='center', va='bottom',
            color=_INK_PRIMARY, fontsize=10, fontweight='bold',
        )

    ax.axhline(50, color=_INK_MUTED, linewidth=1, linestyle='--', alpha=0.55, zorder=2)

    ax.set_ylabel("Win rate vs masked-random (%)", color=_INK_SECONDARY, fontsize=11)
    ax.set_title("Final 3-map GNN performance per map",
                 color=_INK_PRIMARY, fontsize=13, fontweight='bold', pad=10)
    ax.set_ylim(0, 118)
    _apply_chart_chrome(ax)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=_SURFACE)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_transfer_comparison(transfer_results, output_path):
    """
    Bar chart — 2-map model vs 3-map model on large_10 (transfer test).
    Form: magnitude for 2 nominal categories (model identity) → bar chart.
    Colors: slot 1 blue for 3-map (trained-on), slot 2 orange for 2-map (zero-shot).
    Direct labels serve as the relief channel.
    """
    labels = ['2-map model\n(zero-shot on large_10)', '3-map model\n(trained on large_10)']
    values = [
        transfer_results['2map_on_large10']['win_rate'] * 100,
        transfer_results['3map_on_large10']['win_rate'] * 100,
    ]
    # slot 2 (orange) for the zero-shot model, slot 1 (blue) for the full model
    colors = ['#eb6834', '#2a78d6']

    fig, ax = plt.subplots(figsize=(5.5, 4), facecolor=_SURFACE)
    ax.set_facecolor(_SURFACE)

    bars = ax.bar(labels, values, color=colors, width=0.4, zorder=3)
    for bar in bars:
        bar.set_edgecolor(_SURFACE)
        bar.set_linewidth(1.5)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{val:.1f}%",
            ha='center', va='bottom',
            color=_INK_PRIMARY, fontsize=10, fontweight='bold',
        )

    ax.axhline(50, color=_INK_MUTED, linewidth=1, linestyle='--', alpha=0.55, zorder=2)

    ax.set_ylabel("Win rate on large_10 (%)", color=_INK_SECONDARY, fontsize=11)
    ax.set_title("Transfer test: large_10 zero-shot vs in-training",
                 color=_INK_PRIMARY, fontsize=13, fontweight='bold', pad=10)
    ax.set_ylim(0, 118)
    _apply_chart_chrome(ax)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=_SURFACE)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Profile mode — measure baseline bottlenecks across CPU/GPU × single/multi-map
# ---------------------------------------------------------------------------

def _run_profile_scenario(name, map_names, use_gpu, num_workers,
                          output_dir, batch_size, num_epochs, num_warmup, num_timed):
    """Run one profiling scenario: warm up, then time num_timed iterations."""
    from parallel_risk.training.torchrl.train import PPOTrainer

    print(f"\n  scenario '{name}': maps={map_names} gpu={use_gpu} workers={num_workers}")
    if use_gpu and not torch.cuda.is_available():
        print(f"    [skip: CUDA not available]")
        return None

    cfg = build_config(
        map_names=map_names,
        output_dir=output_dir,
        checkpoint_dir=Path(output_dir) / f'ckpt_{name}',
        batch_size=batch_size,
        num_epochs=num_epochs,
        num_workers=num_workers,
        use_gpu=use_gpu,
    )
    trainer = PPOTrainer(cfg)

    steps_per_iter = trainer.batch_size // 2  # 2 agents per env-step

    # Warmup (untimed) — primes CUDA context, cudnn benchmark, worker pool spawn
    for _ in range(num_warmup):
        rollout = trainer.collect_rollout(steps_per_iter)
        trainer.update_policy(rollout)

    # Timed
    trainer.timers.enabled = True
    trainer.timers.reset()
    for _ in range(num_timed):
        rollout = trainer.collect_rollout(steps_per_iter)
        trainer.update_policy(rollout)
    trainer.timers.enabled = False

    summary = trainer.timers.summary()

    # Cleanup — persistent worker pool + writer
    if trainer._worker_pool is not None:
        trainer._worker_pool.terminate()
        trainer._worker_pool.join()
        trainer._worker_pool = None
    trainer.writer.close()

    return {
        'name': name,
        'map_names': list(map_names),
        'use_gpu': use_gpu,
        'num_workers': num_workers,
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'num_timed_iters': num_timed,
        'timings': summary,
    }


def _print_bench_table(results):
    """Print a Markdown table of per-section avg-ms across all scenarios."""
    scenarios = [r for r in results if r is not None]
    if not scenarios:
        print("(no scenarios to display)")
        return

    # Collect all section names across scenarios
    section_names = sorted(set(
        s for r in scenarios for s in r['timings'].keys()
    ))
    scen_names = [r['name'] for r in scenarios]

    col_width = max(24, max((len(s) for s in section_names), default=24))
    val_width = max(14, max((len(n) for n in scen_names), default=14))

    def fmt_cell(v, w):
        return f"{v:>{w}}"

    # Header
    header = "| " + fmt_cell("section", col_width) + " |"
    for n in scen_names:
        header += " " + fmt_cell(n, val_width) + " |"
    sep = "|" + "-" * (col_width + 2) + "|"
    for _ in scen_names:
        sep += "-" * (val_width + 2) + "|"

    print()
    print(header)
    print(sep)

    for section in section_names:
        row = "| " + fmt_cell(section, col_width) + " |"
        for r in scenarios:
            info = r['timings'].get(section)
            if info is None:
                row += " " + fmt_cell("-", val_width) + " |"
            else:
                row += " " + fmt_cell(f"{info['avg_ms']:.2f} ms", val_width) + " |"
        print(row)

    # Also print total wall clock per scenario (sum of rollout.total + update.total)
    print()
    for r in scenarios:
        t = r['timings']
        roll = t.get('rollout.total', {}).get('avg_ms', 0.0)
        upd = t.get('update.total', {}).get('avg_ms', 0.0)
        print(f"  {r['name']}: iter avg = {(roll + upd) / 1000.0:.3f} s "
              f"(rollout {roll / 1000.0:.3f} s + update {upd / 1000.0:.3f} s)")


def run_profile(output_dir, batch_size, num_epochs, num_warmup, num_timed):
    """Run the 4-scenario baseline benchmark."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("BENCHMARK — baseline (HEAD)")
    print("=" * 60)
    print(f"  batch_size={batch_size}  num_epochs={num_epochs}  "
          f"warmup={num_warmup}  timed={num_timed}")
    print(f"  output: {output_dir}")

    scenarios = [
        ('simple6_cpu_w1',   ['simple_6'],                          False, 1),
        ('simple6_gpu_w1',   ['simple_6'],                          True,  1),
        ('multimap_cpu_w1',  ['simple_6', 'medium_8', 'large_10'],  False, 1),
        ('multimap_cpu_w4',  ['simple_6', 'medium_8', 'large_10'],  False, 4),
        # The realistic training config: CPU workers do rollout (avoid per-step
        # GPU-kernel-launch overhead), GPU does the mega-batch update.
        ('multimap_gpu_w4',  ['simple_6', 'medium_8', 'large_10'],  True,  4),
    ]

    results = []
    for name, maps, use_gpu, workers in scenarios:
        r = _run_profile_scenario(
            name=name, map_names=maps, use_gpu=use_gpu, num_workers=workers,
            output_dir=str(output_dir), batch_size=batch_size, num_epochs=num_epochs,
            num_warmup=num_warmup, num_timed=num_timed,
        )
        results.append(r)

    _print_bench_table(results)

    out_path = output_dir / 'benchmark_baseline.json'
    with open(out_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'config': {
                'batch_size': batch_size,
                'num_epochs': num_epochs,
                'num_warmup': num_warmup,
                'num_timed': num_timed,
            },
            'scenarios': [r for r in results if r is not None],
        }, f, indent=2)
    print(f"\n  saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Multi-map GNN training experiment for Parallel Risk",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--output-dir',
        default='experiments/multi_map_results',
        help='Directory for results JSON and plots',
    )
    parser.add_argument(
        '--num-iterations',
        type=int,
        default=200,
        help='Training iterations for the main 3-map model',
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode: 30 iterations, eval every 10, 20 episodes per map',
    )
    parser.add_argument(
        '--checkpoint-dir',
        default='checkpoints/multi_map_training',
        help='Base directory for training checkpoints',
    )
    parser.add_argument(
        '--skip-transfer',
        action='store_true',
        help='Skip the transfer test (2-map model comparison)',
    )
    parser.add_argument(
        '--mcts-budget',
        type=int,
        default=50,
        help='MCTS simulation budget per move during evaluation',
    )
    parser.add_argument(
        '--profile',
        action='store_true',
        help='Run baseline benchmark (4 scenarios: CPU/GPU × single/multi-map, '
             'single vs 4 workers), save benchmark_baseline.json and exit.',
    )
    parser.add_argument(
        '--profile-batch-size',
        type=int, default=1024,
        help='Batch size for --profile scenarios (kept small for speed).',
    )
    parser.add_argument(
        '--profile-num-epochs',
        type=int, default=3,
        help='Number of SGD epochs per iter for --profile scenarios.',
    )
    parser.add_argument(
        '--profile-iters',
        type=int, default=5,
        help='Number of timed iterations per --profile scenario.',
    )
    parser.add_argument(
        '--profile-warmup',
        type=int, default=1,
        help='Warmup iterations before timing starts (primes CUDA / worker pool).',
    )
    parser.add_argument(
        '--cpu', action='store_true',
        help='Force CPU device even when CUDA is available.',
    )
    parser.add_argument(
        '--num-workers', type=int, default=1,
        help='Number of parallel rollout workers (spawn processes). '
             '1 = sequential in the main process.',
    )
    parser.add_argument(
        '--serial-eval', action='store_true',
        help='Force sequential per-map evaluation (disables the '
             'ProcessPoolExecutor path). Useful for debugging.',
    )
    args = parser.parse_args()

    if args.profile:
        run_profile(
            output_dir=args.output_dir,
            batch_size=args.profile_batch_size,
            num_epochs=args.profile_num_epochs,
            num_warmup=args.profile_warmup,
            num_timed=args.profile_iters,
        )
        return

    if args.quick:
        num_iterations = 30
        eval_interval = 10
        num_episodes = 20
        batch_size = 400    # smaller batch so each iteration completes quickly
        num_epochs = 3
    else:
        num_iterations = args.num_iterations
        eval_interval = 25
        num_episodes = 50
        batch_size = 4096
        num_epochs = 10

    mcts_budget = args.mcts_budget
    output_dir = Path(args.output_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("MULTI-MAP GNN TRAINING EXPERIMENT")
    print("="*60)
    print(f"  Output dir:      {output_dir}")
    print(f"  Checkpoint dir:  {checkpoint_dir}")
    print(f"  Iterations:      {num_iterations}")
    print(f"  Eval interval:   {eval_interval}")
    print(f"  Episodes/map:    {num_episodes}")
    print(f"  MCTS budget:     {mcts_budget}")
    print(f"  Quick mode:      {args.quick}")
    print(f"  Skip transfer:   {args.skip_transfer}")

    ALL_MAPS = ['simple_6', 'medium_8', 'large_10']

    # -----------------------------------------------------------------------
    # Phase 1: Train 3-map model and evaluate at intervals
    # -----------------------------------------------------------------------
    use_gpu = not args.cpu
    num_workers = args.num_workers
    parallel_eval = not args.serial_eval
    print(f"  Device:          {'GPU' if use_gpu else 'CPU'}   Workers: {num_workers}"
          f"   Parallel eval: {parallel_eval}")

    full_trainer, per_map_win_rates, eval_iterations = train_with_eval(
        map_names_to_train=ALL_MAPS,
        num_iterations=num_iterations,
        eval_interval=eval_interval,
        num_episodes=num_episodes,
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir / 'all_3_maps',
        label='3-map model (simple_6 + medium_8 + large_10)',
        verbose=True,
        batch_size=batch_size,
        num_epochs=num_epochs,
        save_weights_path=checkpoint_dir / 'all_3_maps' / 'final.pt',
        mcts_budget=mcts_budget,
        num_workers=num_workers,
        use_gpu=use_gpu,
        parallel_eval=parallel_eval,
    )

    # Final evaluation on each map with more episodes for stable estimates.
    # evaluate_policy_vs_mcts builds observations on CPU, so we snapshot the
    # policy to CPU and rebuild there (avoids the GPU/CPU device mismatch
    # the direct sequential loop would hit when training ran on GPU).
    print(f"\n{'='*60}")
    print("Final evaluation of 3-map model")
    print('='*60)
    final_win_rates = {}

    if parallel_eval and len(ALL_MAPS) > 1:
        # Reuse the parallel-eval worker (spawns CPU processes, one per map).
        state_dict_cpu, model_kwargs = _snapshot_policy_for_eval(full_trainer)
        with ProcessPoolExecutor(
            max_workers=len(ALL_MAPS),
            mp_context=mp.get_context('spawn'),
        ) as ex:
            futures = {
                map_name: ex.submit(
                    _eval_worker,
                    (state_dict_cpu, model_kwargs, map_name,
                     full_trainer.action_budget, num_episodes * 2,
                     mcts_budget, idx * 2000),
                )
                for idx, map_name in enumerate(ALL_MAPS)
            }
            for map_name in ALL_MAPS:
                _, result = futures[map_name].result()
                final_win_rates[map_name] = result['win_rate']
                print(f"  {map_name}: {result['win_rate']:.2%}  "
                      f"({result['wins']}W/{result['losses']}L/{result['draws']}D)")
    else:
        # Serial fallback: move the policy to CPU for eval, then back.
        original_device = full_trainer.device
        full_trainer.policy.to('cpu')
        try:
            for idx, map_name in enumerate(ALL_MAPS):
                result = evaluate_policy_vs_mcts(
                    full_trainer.policy, full_trainer.action_decoder,
                    map_name, full_trainer.action_budget, num_episodes * 2,
                    mcts_budget=mcts_budget, seed_offset=idx * 2000,
                )
                final_win_rates[map_name] = result['win_rate']
                print(f"  {map_name}: {result['win_rate']:.2%}  "
                      f"({result['wins']}W/{result['losses']}L/{result['draws']}D)")
        finally:
            full_trainer.policy.to(original_device)
    full_trainer.policy.train()

    # -----------------------------------------------------------------------
    # Save phase-1 results and plots immediately (before optional transfer test)
    # -----------------------------------------------------------------------
    results = {
        'experiment': 'multi_map_training',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'num_iterations': num_iterations,
            'eval_interval': eval_interval,
            'num_episodes': num_episodes,
            'mcts_budget': mcts_budget,
            'quick_mode': args.quick,
            'maps': ALL_MAPS,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
        },
        'per_map_win_rates': {
            name: [float(r) for r in rates]
            for name, rates in per_map_win_rates.items()
        },
        'eval_iterations': eval_iterations,
        'final_win_rates': {k: float(v) for k, v in final_win_rates.items()},
        'transfer_results': None,
    }

    results_path = output_dir / 'multi_map_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to: {results_path}")

    print("\nGenerating phase-1 plots ...")
    plot_learning_curves(
        per_map_win_rates, eval_iterations,
        output_dir / 'learning_curves.png',
    )
    plot_final_performance(
        final_win_rates,
        output_dir / 'final_performance.png',
    )

    # -----------------------------------------------------------------------
    # Phase 2: Transfer test (optional)
    # -----------------------------------------------------------------------
    if not args.skip_transfer:
        transfer_results = run_transfer_test(
            full_trainer=full_trainer,
            num_iterations=num_iterations,
            eval_interval=eval_interval,
            num_episodes=num_episodes,
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
            verbose=True,
            batch_size=batch_size,
            num_epochs=num_epochs,
            mcts_budget=mcts_budget,
            num_workers=num_workers,
            use_gpu=use_gpu,
            parallel_eval=parallel_eval,
        )
        results['transfer_results'] = {
            key: {
                k: float(v) if isinstance(v, (int, float, np.floating)) else v
                for k, v in val.items()
            }
            for key, val in transfer_results.items()
        }
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        plot_transfer_comparison(
            transfer_results,
            output_dir / 'transfer_comparison.png',
        )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
    print("\nFinal 3-map model performance (vs MCTS):")
    for map_name in ALL_MAPS:
        wr = final_win_rates[map_name]
        tag = "PASS (>50%)" if wr > 0.5 else "below 50%"
        print(f"  {map_name}: {wr:.2%}  [{tag}]")

    if not args.skip_transfer:
        print("\nTransfer test — win rate on large_10 (vs MCTS):")
        print(f"  2-map model (zero-shot): "
              f"{transfer_results['2map_on_large10']['win_rate']:.2%}")
        print(f"  3-map model (trained):   "
              f"{transfer_results['3map_on_large10']['win_rate']:.2%}")

    print(f"\nOutputs written to {output_dir}:")
    saved = ['multi_map_results.json', 'learning_curves.png', 'final_performance.png']
    if not args.skip_transfer:
        saved.append('transfer_comparison.png')
    for fname in saved:
        print(f"  {fname}")
    print(f"\nCheckpoints written to {checkpoint_dir}:")
    print(f"  3-map (final):        {checkpoint_dir / 'all_3_maps' / 'final.pt'}")
    print(f"  3-map (per eval):     {checkpoint_dir / 'all_3_maps' / 'checkpoint_XXXXXX.pt'}")
    if not args.skip_transfer:
        print(f"  2-map (final):        {checkpoint_dir / 'transfer_2map' / 'final.pt'}")
        print(f"  2-map (per eval):     {checkpoint_dir / 'transfer_2map' / 'checkpoint_XXXXXX.pt'}")
    print("="*60)


if __name__ == "__main__":
    main()
