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
import sys
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
                 batch_size=4096, num_epochs=10):
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
            'learning_rate': 1e-4,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_epsilon': 0.2,
            'entropy_coeff': 0.01,
            'value_loss_coeff': 0.5,
            'max_grad_norm': 0.5,
            'use_gpu': False,
        },
        'log_dir': str(Path(output_dir) / 'runs'),
        'checkpoint_dir': str(checkpoint_dir),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_policy_on_map(policy, action_decoder, map_name, action_budget,
                            num_episodes, seed_offset=0):
    """
    Evaluate a GNN policy on a specific map against the masked-random baseline.

    Policy is set to eval() mode on entry; caller should switch it back to
    train() if training will continue.

    Args:
        policy: GCNPolicy instance
        action_decoder: ActionDecoder instance
        map_name: Map to evaluate on (e.g. 'large_10')
        action_budget: Number of actions per turn
        num_episodes: Number of evaluation episodes
        seed_offset: Seed offset so evaluations on different maps don't collide

    Returns:
        dict with win_rate, wins, losses, draws, total_episodes, avg_episode_length
    """
    from parallel_risk import ParallelRiskEnv
    from parallel_risk.training.torchrl.graph_wrapper import GraphObservationWrapper
    from parallel_risk.agents.masked_random_agent import MaskedRandomAgent
    from torch_geometric.data import Batch

    env = ParallelRiskEnv(map_name=map_name, max_turns=50, seed=None,
                          reward_shaping_config=None)
    wrapped_env = GraphObservationWrapper(env, device=torch.device('cpu'))
    masked_random = MaskedRandomAgent(action_budget=action_budget, max_troops=20)

    policy.eval()

    wins = losses = draws = 0
    episode_lengths = []

    for episode in range(num_episodes):
        obs, _ = wrapped_env.reset(seed=42 + seed_offset + episode)
        done = False
        episode_length = 0

        while not done:
            graph_0 = obs.get('agent_0')
            if graph_0 is not None:
                with torch.no_grad():
                    batched = Batch.from_data_list([graph_0])
                    action_logits, _, _ = policy(batched)
                    actions_tensor, _ = action_decoder.decode_actions(
                        action_logits, batched.batch,
                        deterministic=False, return_log_probs=True,
                        observations=[graph_0],
                    )
                    action_array = actions_tensor[0].cpu().numpy()
                    action_0 = {
                        'num_actions': action_budget,
                        'actions': np.vstack([
                            action_array,
                            np.zeros((10 - action_budget, 3)),
                        ]),
                    }
            else:
                action_0 = None

            action_1 = (masked_random.get_action(obs['agent_1'])
                        if 'agent_1' in obs else None)

            actions = {}
            if action_0 is not None:
                actions['agent_0'] = action_0
            if action_1 is not None:
                actions['agent_1'] = action_1

            obs, rewards, terminateds, truncateds, _ = wrapped_env.step(actions)
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
# Training loop with inline evaluation
# ---------------------------------------------------------------------------

def train_with_eval(map_names_to_train, num_iterations, eval_interval,
                    num_episodes, output_dir, checkpoint_dir,
                    label='', verbose=True, batch_size=4096, num_epochs=10):
    """
    Train a multi-map (or single-map) GNN with per-map evaluation at regular
    intervals.  Manages the PPO loop directly so evaluation happens in-memory
    without a checkpoint round-trip.

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

    for iteration in range(num_iterations):
        # Collect and update — mirrors what PPOTrainer.train() does internally
        rollout = trainer.collect_rollout(trainer.batch_size // 2)
        trainer.update_policy(rollout)

        if (iteration + 1) % eval_interval == 0:
            eval_iterations.append(iteration + 1)
            if verbose:
                print(f"\n  -- eval @ iter {iteration+1} --")
            for idx, map_name in enumerate(map_names_to_train):
                result = evaluate_policy_on_map(
                    trainer.policy, trainer.action_decoder,
                    map_name, action_budget, num_episodes,
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

    trainer.writer.close()
    return trainer, per_map_win_rates, eval_iterations


# ---------------------------------------------------------------------------
# Transfer test
# ---------------------------------------------------------------------------

def run_transfer_test(full_trainer, num_iterations, eval_interval,
                      num_episodes, output_dir, checkpoint_dir, verbose=True,
                      batch_size=4096, num_epochs=10):
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

    two_map_trainer, _, _ = train_with_eval(
        map_names_to_train=TWO_MAPS,
        num_iterations=num_iterations,
        eval_interval=eval_interval,
        num_episodes=num_episodes,
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir / 'transfer_2map',
        label='2-map model (simple_6 + medium_8)',
        verbose=verbose,
        batch_size=batch_size,
        num_epochs=num_epochs,
    )

    action_budget = two_map_trainer.action_budget

    print("\n  Evaluating 2-map model on large_10 (zero-shot) ...")
    result_2map = evaluate_policy_on_map(
        two_map_trainer.policy, two_map_trainer.action_decoder,
        'large_10', action_budget, num_episodes * 2,
        seed_offset=5000,
    )

    print("  Evaluating 3-map model on large_10 ...")
    result_3map = evaluate_policy_on_map(
        full_trainer.policy, full_trainer.action_decoder,
        'large_10', action_budget, num_episodes * 2,
        seed_offset=6000,
    )
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
    args = parser.parse_args()

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
    print(f"  Quick mode:      {args.quick}")

    ALL_MAPS = ['simple_6', 'medium_8', 'large_10']

    # -----------------------------------------------------------------------
    # Phase 1: Train 3-map model and evaluate at intervals
    # -----------------------------------------------------------------------
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
    )

    # Final evaluation on each map with more episodes for stable estimates
    print(f"\n{'='*60}")
    print("Final evaluation of 3-map model")
    print('='*60)
    final_win_rates = {}
    for idx, map_name in enumerate(ALL_MAPS):
        result = evaluate_policy_on_map(
            full_trainer.policy, full_trainer.action_decoder,
            map_name, full_trainer.action_budget, num_episodes * 2,
            seed_offset=idx * 2000,
        )
        final_win_rates[map_name] = result['win_rate']
        print(f"  {map_name}: {result['win_rate']:.2%}  "
              f"({result['wins']}W/{result['losses']}L/{result['draws']}D)")
    full_trainer.policy.train()

    # -----------------------------------------------------------------------
    # Phase 2: Transfer test
    # -----------------------------------------------------------------------
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
    )

    # -----------------------------------------------------------------------
    # Save results JSON
    # -----------------------------------------------------------------------
    results = {
        'experiment': 'multi_map_training',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'num_iterations': num_iterations,
            'eval_interval': eval_interval,
            'num_episodes': num_episodes,
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
        'transfer_results': {
            key: {
                k: float(v) if isinstance(v, (int, float, np.floating)) else v
                for k, v in val.items()
            }
            for key, val in transfer_results.items()
        },
    }

    results_path = output_dir / 'multi_map_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to: {results_path}")

    # -----------------------------------------------------------------------
    # Generate plots
    # -----------------------------------------------------------------------
    print("\nGenerating plots ...")
    plot_learning_curves(
        per_map_win_rates, eval_iterations,
        output_dir / 'learning_curves.png',
    )
    plot_final_performance(
        final_win_rates,
        output_dir / 'final_performance.png',
    )
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
    print("\nFinal 3-map model performance:")
    for map_name in ALL_MAPS:
        wr = final_win_rates[map_name]
        tag = "PASS (>50%)" if wr > 0.5 else "below 50%"
        print(f"  {map_name}: {wr:.2%}  [{tag}]")

    print("\nTransfer test — win rate on large_10:")
    print(f"  2-map model (zero-shot): "
          f"{transfer_results['2map_on_large10']['win_rate']:.2%}")
    print(f"  3-map model (trained):   "
          f"{transfer_results['3map_on_large10']['win_rate']:.2%}")

    print(f"\nOutputs written to {output_dir}:")
    for fname in ['multi_map_results.json', 'learning_curves.png',
                  'final_performance.png', 'transfer_comparison.png']:
        print(f"  {fname}")
    print("="*60)


if __name__ == "__main__":
    main()
