"""
Training script for Parallel Risk with GNN policies.

Implements PPO training with self-play for GNN-based policies.

Usage:
    python -m parallel_risk.training.torchrl.train --config configs/gnn_gcn.yaml
"""

import argparse
import copy
import os
import time
import yaml
import multiprocessing as mp
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Batch

from parallel_risk import ParallelRiskEnv
from parallel_risk.env.reward_shaping import RewardShapingConfig
from parallel_risk.training.torchrl.graph_wrapper import GraphObservationWrapper, env_to_graph
from parallel_risk.training.torchrl.vec_rollout import (
    LockstepRollout, arrays_from_numpy, arrays_to_numpy, build_rollout, map_edge_index,
    merge_arrays,
)
from parallel_risk.models.gnn_gcn import GCNPolicy
from parallel_risk.models.action_decoder import ActionDecoder


# ---------------------------------------------------------------------------
# TimingRecorder — no-op unless .enabled is True. Used by --profile mode.
# ---------------------------------------------------------------------------

class TimingRecorder:
    """
    Records per-section wall-clock. When the trainer's device is CUDA,
    synchronizes before/after so kernel time is measured, not launch time.
    Disabled by default (zero overhead); flip `.enabled = True` to start recording.
    """

    def __init__(self, device: torch.device = None):
        self.enabled = False
        self.device = device
        self._sums = defaultdict(float)
        self._counts = defaultdict(int)

    def reset(self):
        self._sums.clear()
        self._counts.clear()

    @contextmanager
    def section(self, name: str):
        if not self.enabled:
            yield
            return
        cuda = self.device is not None and self.device.type == 'cuda'
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if cuda:
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            self._sums[name] += dt
            self._counts[name] += 1

    def summary(self) -> Dict[str, Dict[str, float]]:
        return {
            name: {
                'total_s': self._sums[name],
                'count': self._counts[name],
                'avg_ms': (self._sums[name] / self._counts[name]) * 1000.0
                          if self._counts[name] else 0.0,
            }
            for name in sorted(self._sums.keys())
        }


# ---------------------------------------------------------------------------
# Parallel rollout worker — module-level so it is picklable by multiprocessing
# ---------------------------------------------------------------------------

# The pool is persistent, so each worker process keeps its policy and lockstep
# engine (envs included) across iterations; only the weights and seeds change.
_WORKER_STATE = {}


def _rollout_worker(args: Dict[str, Any]):
    """
    Collect `num_steps` lockstep steps of `num_envs` envs in a worker process.

    `args` comes from PPOTrainer._build_worker_args. The policy runs on CPU in
    eval mode with one torch thread (several workers share the machine and a
    16-thread worker measured 1.7x slower than a 1-thread one). Returns the
    compact numpy arrays of LockstepRollout.finish (see vec_rollout), which
    the trainer merges and rebuilds on its own device.
    """
    torch.set_num_threads(1)
    key = (
        tuple(args['map_names']), args['max_turns'], args['env_seed'],
        args['use_reward_shaping'], args['action_budget'], args['max_regions'],
        args['num_envs'], tuple(sorted(args['model_kwargs'].items())),
    )
    state = _WORKER_STATE.get(key)
    if state is None:
        policy = GCNPolicy(**args['model_kwargs'])
        engine = LockstepRollout(
            policy, ActionDecoder(action_budget=args['action_budget'], max_troops=20),
            map_names=args['map_names'], num_envs=args['num_envs'],
            action_budget=args['action_budget'], max_regions=args['max_regions'],
            max_turns=args['max_turns'], env_seed=args['env_seed'],
            use_reward_shaping=args['use_reward_shaping'],
        )
        _WORKER_STATE.clear()
        state = _WORKER_STATE[key] = (policy, engine)
    policy, engine = state

    policy.load_state_dict(args['policy_state_dict'])
    policy.eval()
    seed = int(args['seed'])
    # Per-worker, per-iteration streams: action sampling (torch), map choice
    # (rng) and env reset seeds all derive from the worker seed.
    torch.manual_seed(seed)
    engine.rng = np.random.RandomState(seed)
    engine.reset_seed_base = seed
    return arrays_to_numpy(engine.collect(args['num_steps']))


class RunningMeanStd:
    """
    Track running mean and standard deviation for normalization.

    Used to normalize value function targets for stable training.
    """

    def __init__(self, epsilon: float = 1e-8):
        """
        Initialize running statistics.

        Args:
            epsilon: Small constant for numerical stability
        """
        self.mean = 0.0
        self.var = 1.0
        self.count = 0
        self.epsilon = epsilon

    def update(self, x: torch.Tensor):
        """
        Update running statistics with new batch of data.

        Uses Welford's online algorithm for numerical stability.

        Args:
            x: Tensor of values to update statistics with
        """
        batch_mean = torch.mean(x).item()
        batch_var = torch.var(x).item()
        batch_count = x.numel()

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        new_var = M2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize values using running statistics.

        Args:
            x: Tensor to normalize

        Returns:
            Normalized tensor with mean ≈ 0, std ≈ 1
        """
        return (x - self.mean) / (torch.sqrt(torch.tensor(self.var)) + self.epsilon)


class PPOTrainer:
    """
    PPO trainer for GNN policies on Parallel Risk.

    Implements:
    - Data collection with parallel environments
    - PPO loss computation
    - Self-play (both agents use same policy)
    - Gradient updates
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize PPO trainer.

        Args:
            config: Configuration dict with hyperparameters
        """
        self.config = config
        use_gpu = config.get('training', {}).get('use_gpu', config.get('use_gpu', False))
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')

        # Perf knobs for CUDA. TF32 gives large matmul speedup on Ampere+ with
        # negligible precision cost; cudnn.benchmark picks the fastest conv
        # kernel per input shape (safe because our shapes are stable per map).
        if self.device.type == 'cuda':
            torch.set_float32_matmul_precision('high')
            torch.backends.cudnn.benchmark = True

        # Create environment to get observation/action space info
        env_config = config['env']

        # Setup reward shaping if enabled
        reward_shaping_config = None
        if env_config.get('use_reward_shaping', True):  # Default to True
            reward_shaping_config = RewardShapingConfig()  # Uses default (all enabled)

        # Support both map_name (single, backward compat) and map_names (multi-map)
        if 'map_names' in env_config:
            map_names = list(env_config['map_names'])
        elif 'map_name' in env_config:
            map_names = [env_config['map_name']]
        else:
            raise ValueError("env_config must contain either 'map_name' or 'map_names'")

        self.map_names = map_names

        # Compute the max region count across all training maps so every
        # rollout emits observations with the same feature dim. Without this
        # padding, wrapping a mixed-region-count multi-map set crashes the
        # GNN input projection (dim inferred from one map, sees a different
        # map's dim in the next rollout).
        from parallel_risk.env.map_config import MapRegistry
        max_regions = max(len(MapRegistry.get(m).regions) for m in map_names)

        # Env-side action padding must accommodate action_budget > 10.
        # Same computation as later self.action_budget line — hoisted here
        # because env construction happens before self.action_budget is set.
        _env_max_actions = max(10, int(env_config.get('action_budget', 5)))

        # Create one wrapped environment per map
        self.envs = []
        for map_name in map_names:
            _env = ParallelRiskEnv(
                map_name=map_name,
                max_turns=env_config.get('max_turns', 100),
                seed=env_config.get('seed', None),
                reward_shaping_config=reward_shaping_config,
                max_actions_per_turn=_env_max_actions,
            )
            self.envs.append(GraphObservationWrapper(
                _env, device=self.device, max_regions=max_regions,
            ))

        # Backward-compat aliases (point to the first environment)
        self.wrapped_env = self.envs[0]
        self.env = self.envs[0].env
        self.max_regions = max_regions

        # Get graph observation info from first env (dims are now consistent
        # across all maps thanks to the max_regions padding above).
        obs_space = self.envs[0].observation_space
        self.node_features_dim = obs_space['node_features_dim']
        self.global_features_dim = obs_space['global_features_dim']

        # Training hyperparameters
        train_config = config['training']
        self.num_workers = train_config.get('num_workers', 4)
        self.batch_size = train_config.get('batch_size', 2048)
        self.num_epochs = train_config.get('num_epochs', 10)
        self.learning_rate = train_config.get('learning_rate', 3e-4)
        self.gamma = train_config.get('gamma', 0.99)
        self.gae_lambda = train_config.get('gae_lambda', 0.95)
        self.clip_epsilon = train_config.get('clip_epsilon', 0.2)
        self.vf_clip_param = train_config.get('vf_clip_param', 10.0)  # Value clip (matches RLlib default)
        self.entropy_coeff = train_config.get('entropy_coeff', 0.01)
        self.value_loss_coeff = train_config.get('value_loss_coeff', 0.5)
        self.max_grad_norm = train_config.get('max_grad_norm', 0.5)

        # Rollout layout (see vec_rollout.py). num_envs envs are stepped in
        # lockstep per rollout process, so a rollout of S env steps is a
        # [S // num_envs, 2 * num_envs] grid of samples. rollout_device='cuda'
        # collects in the trainer process on the GPU (one batched forward per
        # step; best with num_envs >= 64) instead of in CPU workers.
        self.num_envs = int(train_config.get('num_envs', 16))
        if self.num_envs < 1:
            raise ValueError(f"training.num_envs must be >= 1, got {self.num_envs}")
        self.rollout_device = str(train_config.get('rollout_device', 'cpu'))
        if self.rollout_device not in ('cpu', 'cuda'):
            raise ValueError(f"training.rollout_device must be 'cpu' or 'cuda', got {self.rollout_device!r}")
        if self.rollout_device == 'cuda' and self.device.type != 'cuda':
            print("rollout_device='cuda' requires use_gpu=true and a CUDA device; collecting on CPU")
            self.rollout_device = 'cpu'
        if self.rollout_device == 'cuda' and self.num_workers > 1:
            print(f"rollout_device='cuda': rollouts run in the trainer process on the GPU "
                  f"({self.num_envs} envs in lockstep); num_workers={self.num_workers} is not used for collection")

        # Model configuration
        model_config = config['model']
        self.action_budget = env_config.get('action_budget', 5)

        # Create policy network
        self.policy = GCNPolicy(
            node_features_dim=self.node_features_dim,
            global_features_dim=self.global_features_dim,
            hidden_dim=model_config.get('hidden_dim', 128),
            num_layers=model_config.get('num_layers', 3),
            action_budget=self.action_budget,
            max_troops=20,
            dropout=model_config.get('dropout', 0.1)
        ).to(self.device)

        # Create action decoder (uses autoregressive masking automatically)
        self.action_decoder = ActionDecoder(
            action_budget=self.action_budget,
            max_troops=20,
        )

        # Optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)

        # Running statistics for value normalization (Bug #3 fix)
        self.return_rms = RunningMeanStd()

        # TensorBoard logging
        log_dir = config.get('log_dir', 'runs/gnn_training')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(log_dir=f"{log_dir}/{timestamp}")

        # Checkpoint directory
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints/gnn_training'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training statistics
        self.global_step = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_rewards_per_map = {name: [] for name in self.map_names}

        # Persistent worker pool — created once to amortize spawn overhead
        self._worker_pool = None

        # In-process lockstep engine (built on first use) and, when the trainer
        # policy lives on CUDA but rollouts run on CPU, a CPU copy of the policy
        # whose weights are synced before every collection.
        self._rollout_engine = None
        self._rollout_policy = None
        self._warned_uneven_steps = False

        # Map-constant edge_index per map (order of self.map_names), cached per device.
        self._edge_index_cpu = [map_edge_index(m, self.max_regions) for m in self.map_names]
        self._edge_index_by_device = {}

        # Timing (no-op unless .enabled = True; used by --profile)
        self.timers = TimingRecorder(device=self.device)

    def _build_worker_args(self, worker_seed: int, num_steps: int):
        """Build the args dict for _rollout_worker (num_steps lockstep steps of num_envs envs)."""
        env_config = self.config['env']
        model_kwargs = dict(
            node_features_dim=self.node_features_dim,
            global_features_dim=self.global_features_dim,
            hidden_dim=self.policy.hidden_dim,
            num_layers=self.policy.num_layers,
            action_budget=self.action_budget,
            max_troops=20,
            dropout=self.policy.dropout,
        )
        return {
            'policy_state_dict': {k: v.cpu() for k, v in self.policy.state_dict().items()},
            'model_kwargs': model_kwargs,
            'map_names': list(self.map_names),
            'max_turns': env_config.get('max_turns', 100),
            'env_seed': env_config.get('seed'),
            'use_reward_shaping': env_config.get('use_reward_shaping', True),
            'action_budget': self.action_budget,
            'max_regions': self.max_regions,
            'num_envs': self.num_envs,
            'num_steps': num_steps,
            'seed': worker_seed,
        }

    def _lockstep_steps(self, env_steps: int) -> int:
        """Lockstep steps T so that T * num_envs env steps are collected per process."""
        T = max(1, env_steps // self.num_envs)
        if T * self.num_envs != env_steps and not self._warned_uneven_steps:
            print(f"collect_rollout: {env_steps} env steps is not a multiple of num_envs={self.num_envs}; "
                  f"collecting {T * self.num_envs} per rollout process")
            self._warned_uneven_steps = True
        return T

    def _edge_index_on(self, device: torch.device):
        key = str(device)
        if key not in self._edge_index_by_device:
            self._edge_index_by_device[key] = [ei.to(device) for ei in self._edge_index_cpu]
        return self._edge_index_by_device[key]

    def _finish_rollout(self, arrays):
        """Record episode stats from a merged collection and rebuild the rollout dict on self.device."""
        self.episode_rewards.extend(arrays['episode_rewards'])
        self.episode_lengths.extend(arrays['episode_lengths'])
        for map_name, reward in zip(arrays['episode_map_names'], arrays['episode_rewards']):
            self.episode_rewards_per_map[map_name].append(reward)
        with self.timers.section('rollout.rebuild'):
            return build_rollout(arrays, self._edge_index_on(self.device), self.device)

    def collect_rollout(self, num_steps: int):
        """
        Collect experience by running the policy in the environment.

        `num_steps` env steps (2 samples each) are collected as a lockstep grid
        of [T, B] samples (see vec_rollout.py). With num_workers > 1 and
        rollout_device='cpu' the persistent worker pool collects num_steps //
        num_workers env steps per worker with the current weights, and the
        workers' columns are concatenated; otherwise the trainer process
        collects on rollout_device.

        Returns:
            rollout: dict with lists (length T) of per-step tensors on
                self.device — 'rewards', 'values', 'log_probs', 'dones',
                'next_values', 'terminateds' [B, ...], 'actions' [B, K, 3] —
                'graph_lists' (T lists of B Data) and 'map_names' (one entry
                per completed episode).
        """
        with self.timers.section('rollout.total'):
            if self.num_workers > 1 and self.rollout_device == 'cpu':
                return self._collect_rollout_parallel(num_steps)
            return self._collect_rollout_sequential(num_steps)

    def _collect_rollout_parallel(self, num_steps: int):
        """Pool workers collect on CPU and return compact arrays; merged along the column axis."""
        if self._worker_pool is None:
            ctx = mp.get_context('spawn')
            self._worker_pool = ctx.Pool(processes=self.num_workers)

        T = self._lockstep_steps(max(1, num_steps // self.num_workers))
        worker_args = [
            self._build_worker_args(worker_seed=self.global_step * 100 + w, num_steps=T)
            for w in range(self.num_workers)
        ]
        with self.timers.section('rollout.pool_map'):
            worker_arrays = self._worker_pool.map(_rollout_worker, worker_args)
        return self._finish_rollout(merge_arrays([arrays_from_numpy(a) for a in worker_arrays]))

    def _inprocess_rollout_policy(self):
        """Policy used by the in-process engine: self.policy, or a synced CPU copy of it."""
        if self.rollout_device == self.device.type:
            return self.policy
        if self._rollout_policy is None:
            self._rollout_policy = copy.deepcopy(self.policy).to('cpu')
        self._rollout_policy.load_state_dict(self.policy.state_dict())
        self._rollout_policy.train(self.policy.training)  # same dropout behaviour as self.policy
        return self._rollout_policy

    def _collect_rollout_sequential(self, num_steps: int):
        """In-process collection on rollout_device (num_workers == 1 or rollout_device='cuda')."""
        policy = self._inprocess_rollout_policy()
        if self._rollout_engine is None or self._rollout_engine.policy is not policy:
            env_config = self.config['env']
            self._rollout_engine = LockstepRollout(
                policy, self.action_decoder, map_names=self.map_names, num_envs=self.num_envs,
                action_budget=self.action_budget, max_regions=self.max_regions,
                max_turns=env_config.get('max_turns', 100), env_seed=env_config.get('seed'),
                use_reward_shaping=env_config.get('use_reward_shaping', True),
            )
        with self.timers.section('rollout.collect'):
            arrays = self._rollout_engine.collect(self._lockstep_steps(num_steps))
        return self._finish_rollout(arrays)

    def compute_gae(self, rewards, values, dones, next_values):
        """
        Compute Generalized Advantage Estimation (GAE).

        Properly handles termination vs truncation semantics:
        - Terminated (victory/elimination): next_value=0, GAE stops
        - Truncated (turn limit): next_value=V(s'), GAE stops but bootstraps with value
        - Non-terminal: next_value=V(s'), GAE propagates

        The key distinction is that truncated episodes still have continuation value
        (the game WOULD continue), so we bootstrap with V(s') rather than 0.

        Args:
            rewards: List of reward tensors [batch_size]
            values: List of value tensors [batch_size]
            dones: List of episode boundary flags [batch_size] (True if episode ended)
            next_values: List of next state values [batch_size] (0 for terminated, V(s') for truncated/non-terminal)

        Returns:
            advantages: Tensor of advantages
            returns: Tensor of returns
        """
        # Verify all tensors have the same batch size
        batch_sizes = [r.size(0) for r in rewards]
        if len(set(batch_sizes)) > 1:
            print(f"ERROR: Inconsistent batch sizes in rollout: {batch_sizes}")
            print(f"  Rewards shapes: {[r.shape for r in rewards[:5]]}")
            print(f"  Values shapes: {[v.shape for v in values[:5]]}")
            raise ValueError(f"Inconsistent batch sizes: {batch_sizes}")

        batch_size = rewards[0].size(0)
        advantages = []
        returns = []

        gae = torch.zeros(batch_size, device=self.device)

        # Reverse iteration through trajectory
        for t in reversed(range(len(rewards))):
            # next_val is:
            #   - 0 for terminated states (victory/elimination)
            #   - V(s') for truncated states (turn limit) - allows bootstrapping!
            #   - V(s') for non-terminal states
            next_val = next_values[t]

            # Mask for GAE propagation - don't propagate across episode boundaries
            # (applies to both terminated and truncated episodes)
            mask = 1.0 - dones[t].float()

            # TD error: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
            # For truncated episodes, next_val = V(s') allows proper bootstrapping
            delta = rewards[t] + self.gamma * next_val - values[t]

            # GAE: A_t = δ_t + (γλ) * A_{t+1}
            # Only propagate GAE from future steps if not at episode boundary
            gae = delta + self.gamma * self.gae_lambda * mask * gae

            advantages.insert(0, gae.clone())
            returns.insert(0, gae + values[t])

        advantages = torch.stack(advantages)
        returns = torch.stack(returns)

        return advantages, returns

    def update_policy(self, rollout):
        """Time-wrapped entry point; see _update_policy_impl for the real work."""
        with self.timers.section('update.total'):
            self._update_policy_impl(rollout)

    def _update_policy_impl(self, rollout):
        """
        Update policy using PPO.

        Uses full-batch updates: all T timesteps are batched into a single
        forward pass per epoch (PyG handles variable-size graphs via batch
        indices).  This is O(num_epochs) forward passes instead of
        O(T * num_epochs), giving ~T× speedup on the update step.
        """
        advantages, returns = self.compute_gae(
            rollout['rewards'],
            rollout['values'],
            rollout['dones'],
            rollout['next_values']
        )

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        T = len(rollout['rewards'])
        B = rollout['rewards'][0].size(0)

        # Pre-stack old values (computed at collection time; detached)
        old_log_probs_all = torch.stack(
            [rollout['log_probs'][t].sum(dim=1).detach() for t in range(T)]
        )  # [T, B]
        old_values_all = torch.stack(
            [rollout['values'][t].detach() for t in range(T)]
        )  # [T, B]
        all_actions = torch.stack(rollout['actions'])  # [T, B, action_budget, 3]

        # Build mega-batch ONCE: T*B graphs concatenated via PyG batching
        all_graphs = [g for graph_list in rollout['graph_lists'] for g in graph_list]
        mega_batch = Batch.from_data_list(all_graphs)
        all_actions_flat = all_actions.view(T * B, self.action_budget, 3)

        for epoch in range(self.num_epochs):
            # ONE forward pass through all T*B graphs
            with self.timers.section('update.forward_per_epoch'):
                action_logits, new_values, _ = self.policy(mega_batch)

            # batched_obs=mega_batch: the decoder derives its masks from the
            # batch already built above instead of re-collating all_graphs
            # (which it did twice per epoch). Same tensors, same numbers.
            with self.timers.section('update.log_probs'):
                new_log_probs = self.action_decoder.compute_log_probs(
                    action_logits,
                    all_actions_flat,
                    mega_batch.batch,
                    observations=all_graphs,
                    batched_obs=mega_batch,
                ).sum(dim=1).view(T, B)  # [T, B]

            with self.timers.section('update.entropy'):
                entropy = self.action_decoder.compute_entropy(
                    action_logits,
                    mega_batch.batch,
                    observations=all_graphs,
                    batched_obs=mega_batch,
                ).mean(dim=1).view(T, B)  # [T, B]

            new_values_2d = new_values.squeeze(-1).view(T, B)  # [T, B]

            # Shuffle timestep order for epoch (reduces temporal correlation)
            perm = torch.randperm(T)
            new_lp_flat = new_log_probs[perm].view(-1)
            new_v_flat = new_values_2d[perm].view(-1)
            ent_flat = entropy[perm].view(-1)
            old_lp_flat = old_log_probs_all[perm].view(-1)
            old_v_flat = old_values_all[perm].view(-1)
            adv_flat = advantages[perm].view(-1)
            ret_flat = returns[perm].view(-1)

            # PPO policy loss
            ratio = torch.exp(new_lp_flat - old_lp_flat)
            surr1 = ratio * adv_flat
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv_flat
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value function clipping
            value_pred_clipped = old_v_flat + torch.clamp(
                new_v_flat - old_v_flat, -self.vf_clip_param, self.vf_clip_param
            )
            value_loss = 0.5 * torch.max(
                (new_v_flat - ret_flat) ** 2,
                (value_pred_clipped - ret_flat) ** 2,
            ).mean()

            entropy_loss = -ent_flat.mean()
            loss = policy_loss + self.value_loss_coeff * value_loss + self.entropy_coeff * entropy_loss

            with self.timers.section('update.backward'):
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            with self.timers.section('update.optimizer_step'):
                self.optimizer.step()

            if epoch == self.num_epochs - 1:
                self.writer.add_scalar('Loss/policy', policy_loss.item(), self.global_step)
                self.writer.add_scalar('Loss/value', value_loss.item(), self.global_step)
                self.writer.add_scalar('Loss/entropy', entropy_loss.item(), self.global_step)
                self.writer.add_scalar('Loss/total', loss.item(), self.global_step)
                returns_flat_log = returns[perm].view(-1)
                self.writer.add_scalar('Stats/return_mean', returns_flat_log.mean().item(), self.global_step)
                self.writer.add_scalar('Stats/return_std', returns_flat_log.std().item(), self.global_step)
                clip_fraction = (torch.abs(new_v_flat - old_v_flat) > self.clip_epsilon).float().mean()
                self.writer.add_scalar('Stats/value_clip_fraction', clip_fraction.item(), self.global_step)

        self.global_step += 1

    def train(self, num_iterations: int):
        """
        Main training loop.

        Args:
            num_iterations: Number of training iterations
        """
        print(f"Starting training for {num_iterations} iterations...")
        print(f"Device: {self.device}")
        if len(self.map_names) == 1:
            print(f"Map: {self.map_names[0]}")
        else:
            print(f"Maps ({len(self.map_names)}): {', '.join(self.map_names)}")
        print(f"Policy: GCN ({self.policy.hidden_dim}x{self.policy.num_layers})")
        print()

        for iteration in range(num_iterations):
            # Collect rollout
            rollout = self.collect_rollout(self.batch_size // 2)  # Divide by 2 for 2 agents

            # Update policy
            self.update_policy(rollout)

            # Log statistics
            if len(self.episode_rewards) > 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_length = np.mean(self.episode_lengths[-10:])

                self.writer.add_scalar('Episode/reward', avg_reward, iteration)
                self.writer.add_scalar('Episode/length', avg_length, iteration)

                print(f"Iteration {iteration+1}/{num_iterations} | "
                      f"Reward: {avg_reward:.3f} | Length: {avg_length:.1f} | "
                      f"Episodes: {len(self.episode_rewards)}")

                # Per-map win rate logging (only when training on multiple maps)
                if len(self.map_names) > 1:
                    for map_name in self.map_names:
                        map_rewards = self.episode_rewards_per_map.get(map_name, [])
                        if len(map_rewards) > 0:
                            recent = map_rewards[-10:]
                            win_rate = float(np.mean([1.0 if r > 0 else 0.0 for r in recent]))
                            self.writer.add_scalar(f'train/win_rate_{map_name}', win_rate, iteration)
            else:
                print(f"Iteration {iteration+1}/{num_iterations} | "
                      f"No episodes completed yet | "
                      f"Steps: {self.global_step * self.batch_size // 2}")

            # Save checkpoint
            if (iteration + 1) % 10 == 0:
                checkpoint_path = self.checkpoint_dir / f"checkpoint_{iteration+1:06d}.pt"
                torch.save({
                    'iteration': iteration + 1,
                    'policy_state_dict': self.policy.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'config': self.config,
                }, checkpoint_path)
                print(f"  💾 Saved checkpoint: {checkpoint_path}")

        print("\n✅ Training complete!")
        self.writer.close()
        if self._worker_pool is not None:
            self._worker_pool.terminate()
            self._worker_pool.join()
            self._worker_pool = None


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train Parallel Risk with GNN + PPO")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--num-iterations", type=int, default=1000, help="Training iterations")
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Create trainer
    trainer = PPOTrainer(config)

    # Load checkpoint if provided
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        trainer.policy.load_state_dict(checkpoint['policy_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"✓ Loaded checkpoint from iteration {checkpoint['iteration']}")

    # Train
    trainer.train(args.num_iterations)


if __name__ == "__main__":
    main()
