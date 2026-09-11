"""End-to-end validation for the MCTS+GNN training loop.

Runs a tiny outer loop (2 iterations, 4 games each, budget=20, 2 workers)
on `simple_6` to prove the pipeline holds together with real spawn-based
workers. Not intended for research results.

Usage:
    PYTHONPATH=. python experiments/validate_mcts_gnn_training.py
"""

from __future__ import annotations

# Load torch first: importing `parallel_risk.training.*` triggers this
# package's __init__ which pulls RLlib, and RLlib's own try_import_torch
# path fails on Windows if torch isn't already in sys.modules.
import torch  # noqa: F401

import argparse
import math
import time
from pathlib import Path

from parallel_risk.training.mcts_gnn.self_play import collect_games
from parallel_risk.training.mcts_gnn.trainer import MCTSGNNTrainer


def build_small_config(output_dir: Path) -> dict:
    return {
        'env': {'map_names': ['simple_6'], 'max_turns': 15, 'action_budget': 5},
        'model': {'type': 'gcn', 'hidden_dim': 32, 'num_layers': 2, 'dropout': 0.0},
        'trainer': {
            'learning_rate': 1e-3,
            'value_loss_coeff': 1.0,
            'entropy_coeff': 0.0,
            'max_grad_norm': 0.5,
        },
        'log_dir': str(output_dir / 'runs'),
        'checkpoint_dir': str(output_dir / 'checkpoints'),
    }


SELF_PLAY_CFG = {
    'dirichlet_alpha': 0.3,
    'noise_frac': 0.25,
    'temperature_turns': 3,
    'max_temperature': 1.0,
}

MCTS_CFG = {
    'simulation_budget': 20,
    'c_puct': 1.4,
    'uct_c': 1.41,
    'pw_alpha': 0.5,
    'max_rollout_turns': 20,
    'action_budget': 5,
}

ENV_CONFIGS = [{'map_name': 'simple_6', 'max_turns': 15}]


def main():
    parser = argparse.ArgumentParser(description="Validate MCTS+GNN loop end-to-end")
    parser.add_argument('--output-dir', default='experiments/mcts_gnn_validation',
                        help='Where to drop the tiny run artifacts')
    parser.add_argument('--num-iterations', type=int, default=2)
    parser.add_argument('--num-games', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--num-epochs', type=int, default=2,
                        help='SGD passes over collected data per iteration')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer = MCTSGNNTrainer(build_small_config(output_dir))
    ckpt_dir = Path(trainer.checkpoint_dir)

    print(f"MCTS+GNN E2E validation")
    print(f"  iterations={args.num_iterations}  games/iter={args.num_games}  "
          f"workers={args.num_workers}  budget={MCTS_CFG['simulation_budget']}")
    print(f"  output: {output_dir}")

    last_losses = []
    for it in range(args.num_iterations):
        t0 = time.perf_counter()
        examples = collect_games(
            policy=trainer.policy,
            model_kwargs=trainer.model_kwargs(),
            env_configs=ENV_CONFIGS,
            mcts_config=MCTS_CFG,
            self_play_config=SELF_PLAY_CFG,
            num_games=args.num_games,
            num_workers=args.num_workers,
            base_seed=it * 10000,
        )
        collect_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        metrics = {}
        for _ in range(args.num_epochs):
            metrics = trainer.update(examples)
        train_s = time.perf_counter() - t1

        for k in ('total_loss', 'policy_loss', 'value_loss'):
            v = metrics.get(k)
            assert v is not None and math.isfinite(v), \
                f"iter {it}: non-finite {k}={v}"

        last_losses.append(metrics['total_loss'])
        print(
            f"  iter {it+1}/{args.num_iterations}  "
            f"examples={len(examples)}  loss={metrics['total_loss']:.3f}  "
            f"policy={metrics['policy_loss']:.3f}  value={metrics['value_loss']:.3f}  "
            f"collect={collect_s:.1f}s train={train_s:.1f}s"
        )

    ckpt_path = ckpt_dir / f"mcts_gnn_iter_{args.num_iterations:06d}.pt"
    trainer.save_checkpoint(ckpt_path, args.num_iterations)
    trainer.close()

    assert ckpt_path.exists(), f"checkpoint not written at {ckpt_path}"
    print(f"\n[OK] E2E validation passed  |  checkpoint: {ckpt_path}")
    print(f"      loss trajectory: {[f'{v:.3f}' for v in last_losses]}")


if __name__ == "__main__":
    main()
