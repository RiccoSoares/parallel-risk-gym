"""ExIt (Expert Iteration) training CLI.

Outer loop:
    for iteration in range(num_iterations):
        examples = collect_games_exit(...)         # MCTS+GNN self-play, per-action data
        for epoch in range(num_epochs):
            trainer.update(examples)               # advantage-weighted actor-critic
        checkpoint every N iterations

Usage:
    python -m parallel_risk.training.mcts_gnn.exit_train \
        --config parallel_risk/training/mcts_gnn/configs/mcts_gnn_exit.yaml \
        --num-iterations 50
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

from parallel_risk.training.mcts_gnn.exit_self_play import collect_games_exit
from parallel_risk.training.mcts_gnn.exit_trainer import ExitTrainer


def _build_trainer_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'env': cfg['env'],
        'model': cfg['model'],
        'trainer': cfg['trainer'],
        'log_dir': cfg.get('logging', {}).get('log_dir', 'runs/exit_training'),
        'checkpoint_dir': cfg.get('logging', {}).get('checkpoint_dir', 'checkpoints/exit_training'),
    }


def _env_configs(cfg: Dict[str, Any]):
    max_turns = int(cfg['env'].get('max_turns', 50))
    return [{'map_name': name, 'max_turns': max_turns} for name in cfg['env']['map_names']]


def _mcts_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    m = dict(cfg['mcts'])
    m['action_budget'] = int(cfg['env'].get('action_budget', 5))
    return m


def run(cfg: Dict[str, Any], num_iterations: int,
        resume: str = None, reset_optimizer: bool = True) -> None:
    trainer = ExitTrainer(_build_trainer_config(cfg))
    start_iteration = 0
    if resume:
        start_iteration = trainer.load_checkpoint(
            resume, load_optimizer=(not reset_optimizer),
        )
        print(f"Resumed from {resume} at iteration {start_iteration}  "
              f"(optimizer_reset={reset_optimizer})")

    env_configs = _env_configs(cfg)
    mcts_cfg = _mcts_config(cfg)
    self_play_cfg = cfg['self_play']
    num_games = int(self_play_cfg.get('num_games_per_iteration', 72))
    num_workers = int(self_play_cfg.get('num_workers', 8))
    games_per_worker = int(self_play_cfg.get('games_per_worker', 1))
    device = str(self_play_cfg.get('device', 'cpu'))
    num_epochs = int(cfg['trainer'].get('num_epochs', 4))
    ckpt_interval = int(cfg.get('outer', {}).get('checkpoint_interval', 10))
    ckpt_dir = Path(cfg.get('logging', {}).get('checkpoint_dir', 'checkpoints/exit_training'))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"ExIt training | maps={cfg['env']['map_names']} | "
          f"iterations={num_iterations} | games/iter={num_games} | "
          f"workers={num_workers} x {games_per_worker} games ({device}) | "
          f"mcts_budget={mcts_cfg['simulation_budget']}")

    for iteration in range(start_iteration, start_iteration + num_iterations):
        iter_start = time.perf_counter()

        collect_start = time.perf_counter()
        examples = collect_games_exit(
            policy=trainer.policy,
            model_kwargs=trainer.model_kwargs(),
            env_configs=env_configs,
            mcts_config=mcts_cfg,
            self_play_config=self_play_cfg,
            num_games=num_games,
            num_workers=num_workers,
            base_seed=iteration * 10000,
            max_regions=trainer.max_regions,
            games_per_worker=games_per_worker,
            device=device,
        )
        collect_time = time.perf_counter() - collect_start

        z_values = np.array([ex['z'] for ex in examples], dtype=np.float32)
        win_frac = float(np.mean(z_values > 0)) if len(z_values) else 0.0

        train_start = time.perf_counter()
        last_metrics: Dict[str, float] = {}
        for _ in range(num_epochs):
            last_metrics = trainer.update(examples)
        train_time = time.perf_counter() - train_start

        iter_time = time.perf_counter() - iter_start
        print(
            f"[iter {iteration+1:4d}]  "
            f"examples={len(examples):5d}  "
            f"win_frac={win_frac:.2f}  "
            f"loss={last_metrics.get('total_loss', float('nan')):.4f}  "
            f"policy={last_metrics.get('policy_loss', float('nan')):.4f}  "
            f"value={last_metrics.get('value_loss', float('nan')):.4f}  "
            f"collect={collect_time:.1f}s  train={train_time:.1f}s  "
            f"total={iter_time:.1f}s"
        )

        trainer.writer.add_scalar('SelfPlay/win_frac', win_frac, iteration)
        trainer.writer.add_scalar('SelfPlay/num_examples', len(examples), iteration)
        trainer.writer.add_scalar('Time/collect_s', collect_time, iteration)
        trainer.writer.add_scalar('Time/train_s', train_time, iteration)

        if (iteration + 1) % ckpt_interval == 0:
            path = ckpt_dir / f"exit_iter_{iteration+1:06d}.pt"
            trainer.save_checkpoint(path, iteration + 1)
            print(f"    saved checkpoint: {path}")

    trainer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="ExIt (Expert Iteration) training")
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument("--num-iterations", type=int, default=None,
                        help="Override outer.num_iterations from the YAML")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from a checkpoint (PPO, AZ or ExIt — same schema)")
    parser.add_argument("--reset-optimizer", dest='reset_optimizer',
                        action='store_true', default=True,
                        help='Discard resumed optimizer state (default; recommended when '
                             'crossing training regimes)')
    parser.add_argument("--keep-optimizer", dest='reset_optimizer',
                        action='store_false')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    num_iterations = args.num_iterations
    if num_iterations is None:
        num_iterations = int(cfg.get('outer', {}).get('num_iterations', 50))

    run(cfg, num_iterations, resume=args.resume,
        reset_optimizer=args.reset_optimizer)


if __name__ == "__main__":
    main()
