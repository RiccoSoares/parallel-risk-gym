"""Distillation trainer for MCTS+GNN training.

Consumes examples produced by `self_play.play_episode` — each an
`(observation graph, visit distribution, outcome value)` tuple — and
updates a `GCNPolicy` with:
  - Cross-entropy on the visit distribution (policy target).
  - MSE on the outcome (value target).

Structure mirrors `parallel_risk/training/torchrl/train.py::PPOTrainer`
so checkpoints are interchangeable via the standard `{iteration,
policy_state_dict, optimizer_state_dict, config}` schema.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Batch

from parallel_risk import ParallelRiskEnv
from parallel_risk.env.map_config import MapRegistry
from parallel_risk.models.action_decoder import ActionDecoder
from parallel_risk.models.gnn_gcn import GCNPolicy


def _feature_dims_for(max_regions: int) -> Dict[str, int]:
    """Same formula as `GNNAgent.from_checkpoint` and the multi-map PPOTrainer.

    Uses `max_regions` (may exceed any single map's region count) so a mixed
    map set can share one policy without shape mismatches.
    """
    return {
        'node_features_dim': 3 + max_regions,
        'global_features_dim': 2 + max_regions,
    }


class MCTSGNNTrainer:
    """Supervised distillation trainer for MCTS+GNN.

    Reuses the checkpoint schema of `PPOTrainer` so a checkpoint can be
    loaded by `MCTSGNNAgent.from_checkpoint` for the next self-play round
    (or by `GNNAgent.from_checkpoint` for evaluation) without any adapter.

    Kept CPU-only in this first landing. GPU support is a follow-up — MCTS
    self-play runs on CPU workers anyway, so the whole loop is CPU-bound.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # Trainer can run on GPU for the update step; self-play workers
        # always run on CPU (MCTS is CPU-bound, and per-call GNN forwards
        # are too small to batch efficiently on GPU).
        use_gpu = bool(config.get('trainer', {}).get('use_gpu', False))
        self.device = torch.device(
            'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        )

        env_cfg = config['env']
        if 'map_names' in env_cfg:
            map_names = list(env_cfg['map_names'])
        elif 'map_name' in env_cfg:
            map_names = [env_cfg['map_name']]
        else:
            raise ValueError("env must contain 'map_name' or 'map_names'")
        self.map_names = map_names

        # Auto-compute max region count across the training set so every
        # map's env_to_graph pads to the same feature width. Mirrors what
        # `PPOTrainer.__init__` does — required for multi-map sets where
        # different maps have different region counts.
        self.max_regions = max(
            len(MapRegistry.get(m).regions) for m in map_names
        )
        dims = _feature_dims_for(self.max_regions)
        self.node_features_dim = dims['node_features_dim']
        self.global_features_dim = dims['global_features_dim']

        model_cfg = config['model']
        train_cfg = config['trainer']
        self.action_budget = int(env_cfg.get('action_budget', 5))

        self.policy = GCNPolicy(
            node_features_dim=self.node_features_dim,
            global_features_dim=self.global_features_dim,
            hidden_dim=int(model_cfg.get('hidden_dim', 128)),
            num_layers=int(model_cfg.get('num_layers', 3)),
            action_budget=self.action_budget,
            max_troops=20,
            dropout=float(model_cfg.get('dropout', 0.1)),
        ).to(self.device)

        self.decoder = ActionDecoder(action_budget=self.action_budget, max_troops=20)

        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=float(train_cfg.get('learning_rate', 1e-4)),
        )

        self.value_loss_coeff = float(train_cfg.get('value_loss_coeff', 1.0))
        self.entropy_coeff = float(train_cfg.get('entropy_coeff', 0.0))
        self.max_grad_norm = float(train_cfg.get('max_grad_norm', 0.5))

        log_dir = config.get('log_dir', 'runs/mcts_gnn_training')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(log_dir=f"{log_dir}/{timestamp}")

        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints/mcts_gnn_training'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.global_step = 0

    # ------------------------------------------------------------------
    # Model-kwargs snapshot for spawning workers
    # ------------------------------------------------------------------

    def model_kwargs(self) -> Dict[str, Any]:
        """Kwargs needed to rebuild a matching `GCNPolicy` in a worker."""
        model_cfg = self.config['model']
        return dict(
            node_features_dim=self.node_features_dim,
            global_features_dim=self.global_features_dim,
            hidden_dim=int(model_cfg.get('hidden_dim', 128)),
            num_layers=int(model_cfg.get('num_layers', 3)),
            action_budget=self.action_budget,
            max_troops=20,
            dropout=float(model_cfg.get('dropout', 0.1)),
        )

    def snapshot_config(self) -> Dict[str, Any]:
        """Config to embed in checkpoints. Includes max_regions in the
        model block so `MCTSGNNAgent.from_checkpoint` reconstructs a
        matching input dim without needing the caller to remember it.
        """
        cfg = {k: v for k, v in self.config.items()}
        cfg['model'] = dict(self.config.get('model', {}))
        cfg['model']['max_regions'] = self.max_regions
        return cfg

    # ------------------------------------------------------------------
    # Distillation update
    # ------------------------------------------------------------------

    def update(self, examples: List[Dict[str, Any]]) -> Dict[str, float]:
        """One gradient step over `examples`. Returns loss metrics."""
        if not examples:
            return {'skipped': 1.0, 'num_examples': 0.0}

        graphs_repeated: List = []
        action_rows: List[List[List[int]]] = []
        weight_rows: List[float] = []
        state_start_indices: List[int] = []
        z_targets: List[float] = []
        row_idx = 0

        for ex in examples:
            vd = ex.get('visit_dist') or {}
            if not vd:
                continue  # nothing to distill for this state
            state_start_indices.append(row_idx)
            z_targets.append(float(ex['z']))
            for key, prob in vd.items():
                if len(key) != self.action_budget:
                    # Actions from MCTSGNNAgent always have exactly
                    # action_budget slots; anything else means the caller
                    # mixed in a different sampler.
                    raise ValueError(
                        f"Expected action key of length {self.action_budget}, "
                        f"got {len(key)}"
                    )
                graphs_repeated.append(ex['graph'])
                action_rows.append([list(row) for row in key])
                weight_rows.append(float(prob))
                row_idx += 1

        if not graphs_repeated:
            return {'skipped': 1.0, 'num_examples': float(len(examples))}

        num_kept = len(z_targets)
        # Move both the batched graph AND the individual Data objects to
        # device: `compute_log_probs` reads per-graph tensors from the
        # observations list for masking, so they must match device with
        # the action logits from `self.policy(mega_batch)`.
        if self.device.type != 'cpu':
            graphs_repeated = [g.to(self.device) for g in graphs_repeated]
        mega_batch = Batch.from_data_list(graphs_repeated).to(self.device)
        actions_flat = torch.tensor(
            action_rows, dtype=torch.long, device=self.device,
        )  # [total_rows, action_budget, 3]
        weights_flat = torch.tensor(
            weight_rows, dtype=torch.float32, device=self.device,
        )  # [total_rows]
        z_tensor = torch.tensor(
            z_targets, dtype=torch.float32, device=self.device,
        )  # [num_kept]

        self.optimizer.zero_grad()
        action_logits, values, _ = self.policy(mega_batch)

        # Value loss: pick the first repeated row per unique state (all
        # repeats give the same value — same graph in). Wastes some head
        # compute but keeps the code path single-forward-pass.
        value_indices = torch.tensor(
            state_start_indices, dtype=torch.long, device=self.device,
        )
        value_preds = values.squeeze(-1)[value_indices]  # [num_kept]
        value_loss = F.mse_loss(value_preds, z_tensor)

        # Policy loss (distillation cross-entropy against visit distribution).
        log_probs = self.decoder.compute_log_probs(
            action_logits, actions_flat, mega_batch.batch,
            observations=graphs_repeated,
        ).sum(dim=1)  # [total_rows]

        # Clamp to a numerical floor. The decoder uses a very negative
        # sentinel (~-1e10) for masked-illegal action components. That
        # rarely fires with a random-init GNN, but a peaked (PPO-trained)
        # policy can put a specific action's mask combo close to -inf,
        # which blows the loss up by ~10 orders of magnitude. Floor of
        # -20 = "probability ~2e-9", effectively "never".
        log_probs = log_probs.clamp(min=-20.0)

        policy_loss = -(weights_flat * log_probs).sum() / num_kept

        entropy_loss = torch.zeros((), device=self.device)
        if self.entropy_coeff > 0.0:
            entropy = self.decoder.compute_entropy(
                action_logits, mega_batch.batch, observations=graphs_repeated,
            ).mean()
            entropy_loss = -entropy

        total_loss = (
            policy_loss
            + self.value_loss_coeff * value_loss
            + self.entropy_coeff * entropy_loss
        )
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        metrics = {
            'total_loss': float(total_loss.item()),
            'policy_loss': float(policy_loss.item()),
            'value_loss': float(value_loss.item()),
            'entropy_loss': float(entropy_loss.item()) if isinstance(entropy_loss, torch.Tensor) else 0.0,
            'num_examples': float(num_kept),
            'total_rows': float(len(graphs_repeated)),
        }
        self.writer.add_scalar('Loss/total', metrics['total_loss'], self.global_step)
        self.writer.add_scalar('Loss/policy', metrics['policy_loss'], self.global_step)
        self.writer.add_scalar('Loss/value', metrics['value_loss'], self.global_step)
        if self.entropy_coeff > 0.0:
            self.writer.add_scalar('Loss/entropy', metrics['entropy_loss'], self.global_step)
        self.global_step += 1
        return metrics

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: Path, iteration: int) -> None:
        """Same schema as `PPOTrainer.train` so checkpoints are interchangeable.

        `config['model']['max_regions']` is included so downstream loaders
        (`MCTSGNNAgent.from_checkpoint`, `GNNAgent.from_checkpoint`) can
        reconstruct the correct input dim without external hints.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'iteration': iteration,
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.snapshot_config(),
        }, path)

    def load_checkpoint(self, path: Path, load_optimizer: bool = True) -> int:
        """Load a checkpoint (PPO or MCTS+GNN — same schema). Returns iteration."""
        ckpt = torch.load(Path(path), map_location=self.device)
        self.policy.load_state_dict(ckpt['policy_state_dict'])
        if load_optimizer and 'optimizer_state_dict' in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            except (ValueError, KeyError):
                # PPO checkpoints may carry an incompatible optimizer state
                # (different LR schedules etc.); ignore rather than fail.
                pass
        return int(ckpt.get('iteration', 0))

    def close(self) -> None:
        self.writer.close()
