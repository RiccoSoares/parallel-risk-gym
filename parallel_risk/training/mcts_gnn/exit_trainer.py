"""Expert Iteration (ExIt) trainer for MCTS+GNN.

Advantage-weighted actor-critic loss:

    A(s) = z(game_outcome) - v_theta(s).detach()   # normalize per batch
    L_policy = -mean(log pi_theta(a_expert | s) * A(s))
    L_value  = MSE(v_theta(s), z)

Where `a_expert` is the action MCTS+GNN actually chose during self-play.
This mirrors REINFORCE-with-baseline / PPO's flat-clip variant: the
policy gradient's sign is set by the actual game outcome, so gradients
remain valid even at cold-start where the AZ distillation target is
degenerate.

Structure mirrors `MCTSGNNTrainer` (same package) for the infra
(device, model construction, optimizer, TensorBoard, checkpoint schema).
The only real difference is `update()`.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Batch

from parallel_risk import ParallelRiskEnv  # noqa: F401 — kept for symmetry / future use
from parallel_risk.env.map_config import MapRegistry
from parallel_risk.models.action_decoder import ActionDecoder
from parallel_risk.models.gnn_gcn import GCNPolicy


def _feature_dims_for(max_regions: int) -> Dict[str, int]:
    return {
        'node_features_dim': 3 + max_regions,
        'global_features_dim': 2 + max_regions,
    }


class ExitTrainer:
    """Expert-Iteration actor-critic trainer for MCTS+GNN.

    Consumes examples produced by `exit_self_play.play_episode_exit`:
        {'graph', 'agent_id', 'action_key', 'map_name', 'z'}

    Checkpoint schema matches `MCTSGNNTrainer` and `PPOTrainer` so weights
    hand off freely between all three training tracks via
    `MCTSGNNAgent.from_checkpoint` / `GNNAgent.from_checkpoint`.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

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

        # Auto-compute max region count across the training set (mirrors
        # `PPOTrainer` and `MCTSGNNTrainer`).
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
        self.entropy_coeff = float(train_cfg.get('entropy_coeff', 0.005))
        self.max_grad_norm = float(train_cfg.get('max_grad_norm', 0.5))

        log_dir = config.get('log_dir', 'runs/exit_training')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(log_dir=f"{log_dir}/{timestamp}")

        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints/exit_training'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.global_step = 0

    # ------------------------------------------------------------------
    # Interop helpers (match MCTSGNNTrainer)
    # ------------------------------------------------------------------

    def model_kwargs(self) -> Dict[str, Any]:
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
        cfg = {k: v for k, v in self.config.items()}
        cfg['model'] = dict(self.config.get('model', {}))
        cfg['model']['max_regions'] = self.max_regions
        return cfg

    # ------------------------------------------------------------------
    # ExIt update
    # ------------------------------------------------------------------

    def update(self, examples: List[Dict[str, Any]]) -> Dict[str, float]:
        if not examples:
            return {'skipped': 1.0, 'num_examples': 0.0}

        graphs: List = []
        action_rows: List[List[List[int]]] = []
        z_targets: List[float] = []

        for ex in examples:
            key = ex.get('action_key')
            if key is None or len(key) != self.action_budget:
                # ExIt examples always come from MCTSGNNAgent, which emits
                # exactly action_budget slots. Anything else means the
                # caller mixed in something incompatible.
                raise ValueError(
                    f"Expected action_key of length {self.action_budget}, "
                    f"got {None if key is None else len(key)}"
                )
            graphs.append(ex['graph'])
            action_rows.append([list(row) for row in key])
            z_targets.append(float(ex['z']))

        N = len(graphs)

        # Move both the batched graph AND the individual Data list to
        # device (compute_log_probs reads per-graph tensors for masking).
        if self.device.type != 'cpu':
            graphs = [g.to(self.device) for g in graphs]
        mega_batch = Batch.from_data_list(graphs).to(self.device)
        actions_flat = torch.tensor(
            action_rows, dtype=torch.long, device=self.device,
        )  # [N, action_budget, 3]
        z_tensor = torch.tensor(
            z_targets, dtype=torch.float32, device=self.device,
        )  # [N]

        self.optimizer.zero_grad()
        action_logits, values, _ = self.policy(mega_batch)
        value_preds = values.squeeze(-1)  # [N]

        # Value loss — MC-return regression (terminal reward only).
        value_loss = F.mse_loss(value_preds, z_tensor)

        # Advantage — detach the baseline so gradients flow only through
        # the value_loss path, not through the policy loss.
        advantages = z_tensor - value_preds.detach()
        # Per-batch normalization (unit variance) stabilizes actor gradients
        # and keeps policy_loss on the same scale across iterations.
        adv_mean = advantages.mean()
        adv_std = advantages.std(unbiased=False)
        advantages = (advantages - adv_mean) / (adv_std + 1e-8)

        # Log-probs of the expert (MCTS-selected) actions under the current policy.
        log_probs = self.decoder.compute_log_probs(
            action_logits, actions_flat, mega_batch.batch,
            observations=graphs,
        ).sum(dim=1)  # [N]

        # Clamp to a numerical floor. The decoder uses a very negative
        # sentinel (~-1e10) for masked-illegal action components; if MCTS
        # ever selects an action the decoder judges masked (validator vs
        # decoder disagreement on edge cases), we'd otherwise blow up
        # policy_loss by ~10 orders of magnitude. Floor of -20 represents
        # "probability ~2e-9" — effectively "never" — without wrecking scale.
        log_probs = log_probs.clamp(min=-20.0)

        policy_loss = -(log_probs * advantages).mean()

        entropy_loss = torch.zeros((), device=self.device)
        if self.entropy_coeff > 0.0:
            entropy = self.decoder.compute_entropy(
                action_logits, mega_batch.batch, observations=graphs,
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
            'advantage_mean_pre_norm': float(adv_mean.item()),
            'advantage_std_pre_norm': float(adv_std.item()),
            'num_examples': float(N),
        }
        self.writer.add_scalar('Loss/total', metrics['total_loss'], self.global_step)
        self.writer.add_scalar('Loss/policy', metrics['policy_loss'], self.global_step)
        self.writer.add_scalar('Loss/value', metrics['value_loss'], self.global_step)
        self.writer.add_scalar('Stats/advantage_mean', metrics['advantage_mean_pre_norm'], self.global_step)
        self.writer.add_scalar('Stats/advantage_std', metrics['advantage_std_pre_norm'], self.global_step)
        if self.entropy_coeff > 0.0:
            self.writer.add_scalar('Loss/entropy', metrics['entropy_loss'], self.global_step)
        self.global_step += 1
        return metrics

    # ------------------------------------------------------------------
    # Checkpoints (same schema as PPOTrainer / MCTSGNNTrainer)
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: Path, iteration: int) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'iteration': iteration,
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.snapshot_config(),
        }, path)

    def load_checkpoint(self, path: Path, load_optimizer: bool = True) -> int:
        ckpt = torch.load(Path(path), map_location=self.device)
        self.policy.load_state_dict(ckpt['policy_state_dict'])
        if load_optimizer and 'optimizer_state_dict' in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            except (ValueError, KeyError):
                pass
        return int(ckpt.get('iteration', 0))

    def close(self) -> None:
        self.writer.close()
