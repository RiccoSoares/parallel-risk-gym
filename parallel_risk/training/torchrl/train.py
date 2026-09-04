"""
Training script for Parallel Risk with GNN policies.

Implements PPO training with self-play for GNN-based policies.

Usage:
    python -m parallel_risk.training.torchrl.train --config configs/gnn_gcn.yaml
"""

import argparse
import os
import yaml
import multiprocessing as mp
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
from parallel_risk.models.gnn_gcn import GCNPolicy
from parallel_risk.models.action_decoder import ActionDecoder


# ---------------------------------------------------------------------------
# Parallel rollout worker — module-level so it is picklable by multiprocessing
# ---------------------------------------------------------------------------

def _rollout_worker(args):
    """
    Collect environment steps in a separate process.

    Returns a partial rollout dict with the same keys as PPOTrainer.collect_rollout,
    but carrying raw PyG Data objects (picklable) rather than pre-batched Batch
    objects.  The caller is responsible for re-batching before calling
    update_policy.
    """
    (policy_state_dict, model_kwargs, env_configs, action_budget,
     num_steps, base_seed) = args

    import torch
    import numpy as np
    from torch_geometric.data import Batch as _Batch
    from parallel_risk import ParallelRiskEnv
    from parallel_risk.env.reward_shaping import RewardShapingConfig
    from parallel_risk.training.torchrl.graph_wrapper import GraphObservationWrapper
    from parallel_risk.models.gnn_gcn import GCNPolicy as _GCNPolicy
    from parallel_risk.models.action_decoder import ActionDecoder as _ActionDecoder

    device = torch.device('cpu')

    # Reconstruct policy from state dict
    policy = _GCNPolicy(**model_kwargs).to(device)
    policy.load_state_dict(policy_state_dict)
    policy.eval()
    action_decoder = _ActionDecoder(action_budget=action_budget, max_troops=20)

    # Build wrapped environments (one per map)
    envs = []
    map_names = []
    for ecfg in env_configs:
        reward_shaping_config = RewardShapingConfig() if ecfg.get('use_reward_shaping') else None
        raw_env = ParallelRiskEnv(
            map_name=ecfg['map_name'],
            max_turns=ecfg.get('max_turns', 50),
            seed=ecfg.get('seed'),
            reward_shaping_config=reward_shaping_config,
        )
        envs.append(GraphObservationWrapper(raw_env, device=device))
        map_names.append(ecfg['map_name'])

    rollout = {
        'observations': [],
        'graph_lists': [],
        'actions': [],
        'rewards': [],
        'values': [],
        'log_probs': [],
        'dones': [],
        'batches': [],
        'next_values': [],
        'map_names': [],
        'episode_rewards': [],
        'episode_lengths': [],
        'episode_rewards_per_map': {n: [] for n in map_names},
    }

    rng = np.random.RandomState(base_seed)
    env_idx = rng.randint(len(envs))
    current_env = envs[env_idx]
    current_map_name = map_names[env_idx]
    obs, _ = current_env.reset(seed=int(base_seed))
    episode_reward = {agent: 0.0 for agent in obs}
    episode_length = 0
    steps_collected = 0

    while steps_collected < num_steps:
        if len(obs) == 0:
            env_idx = rng.randint(len(envs))
            current_env = envs[env_idx]
            current_map_name = map_names[env_idx]
            obs, _ = current_env.reset(seed=int(base_seed + steps_collected))
            episode_reward = {agent: 0.0 for agent in obs}
            episode_length = 0

        graphs = [obs[agent] for agent in sorted(obs.keys())]
        batched_graph = _Batch.from_data_list(graphs)

        with torch.no_grad():
            action_logits, values, _ = policy(batched_graph)

        actions_tensor, log_probs = action_decoder.decode_actions(
            action_logits, batched_graph.batch,
            deterministic=False, return_log_probs=True, observations=graphs,
        )

        actions_dict = {}
        for i, agent in enumerate(sorted(obs.keys())):
            action_array = actions_tensor[i].cpu().numpy()
            actions_dict[agent] = {
                'num_actions': action_budget,
                'actions': np.vstack([action_array,
                                      np.zeros((10 - action_budget, 3))]),
            }

        next_obs, rewards, terminateds, truncateds, _ = current_env.step(actions_dict)

        for agent in rewards:
            episode_reward[agent] = episode_reward.get(agent, 0.0) + rewards[agent]
        episode_length += 1

        done = terminateds.get('__all__', False) or truncateds.get('__all__', False)
        is_terminated = terminateds.get('__all__', False)
        agent_keys = [k for k in sorted(rewards.keys()) if k != '__all__']

        if is_terminated:
            next_value = torch.zeros(len(agent_keys), device=device)
        else:
            next_graphs = [next_obs[agent] for agent in sorted(next_obs.keys())]
            if next_graphs:
                next_batched = _Batch.from_data_list(next_graphs)
                with torch.no_grad():
                    _, nv, _ = policy(next_batched)
                    next_value = nv.squeeze(-1)
            else:
                next_value = torch.zeros(len(agent_keys), device=device)

        rollout['observations'].append(batched_graph)
        rollout['graph_lists'].append(graphs)
        rollout['actions'].append(actions_tensor)
        rollout['rewards'].append(torch.tensor(
            [rewards[agent] for agent in agent_keys], device=device))
        rollout['values'].append(values.squeeze(-1))
        rollout['log_probs'].append(log_probs)
        rollout['dones'].append(torch.tensor(
            [done for _ in agent_keys], dtype=torch.bool, device=device))
        rollout['batches'].append(batched_graph.batch)
        rollout['next_values'].append(next_value)
        steps_collected += 1

        if done:
            rollout['episode_rewards'].append(episode_reward.get('agent_0', 0.0))
            rollout['episode_lengths'].append(episode_length)
            rollout['map_names'].append(current_map_name)
            rollout['episode_rewards_per_map'][current_map_name].append(
                episode_reward.get('agent_0', 0.0))

            env_idx = rng.randint(len(envs))
            current_env = envs[env_idx]
            current_map_name = map_names[env_idx]
            obs, _ = current_env.reset(seed=int(base_seed + steps_collected))
            episode_reward = {agent: 0.0 for agent in obs}
            episode_length = 0
        else:
            obs = next_obs

    return rollout


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
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.get('use_gpu', False) else 'cpu')

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

        # Create one wrapped environment per map
        self.envs = []
        for map_name in map_names:
            _env = ParallelRiskEnv(
                map_name=map_name,
                max_turns=env_config.get('max_turns', 100),
                seed=env_config.get('seed', None),
                reward_shaping_config=reward_shaping_config
            )
            self.envs.append(GraphObservationWrapper(_env, device=self.device))

        # Backward-compat aliases (point to the first environment)
        self.wrapped_env = self.envs[0]
        self.env = self.envs[0].env

        # Get graph observation info from first env (same across all maps — all have 3 regions)
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

    def _build_worker_args(self, worker_seed: int, steps_each: int):
        """Build args tuple for _rollout_worker."""
        model_kwargs = dict(
            node_features_dim=self.node_features_dim,
            global_features_dim=self.global_features_dim,
            hidden_dim=self.policy.hidden_dim,
            num_layers=self.policy.num_layers,
            action_budget=self.action_budget,
            max_troops=20,
            dropout=self.policy.dropout,
        )
        env_config = self.config['env']
        env_configs = [
            {
                'map_name': name,
                'max_turns': env_config.get('max_turns', 50),
                'seed': env_config.get('seed'),
                'use_reward_shaping': env_config.get('use_reward_shaping', True),
            }
            for name in self.map_names
        ]
        state_dict = {k: v.cpu() for k, v in self.policy.state_dict().items()}
        return (state_dict, model_kwargs, env_configs, self.action_budget,
                steps_each, worker_seed)

    def collect_rollout(self, num_steps: int):
        """
        Collect experience by running the policy in the environment.

        When num_workers > 1, spawns worker processes to collect episodes in
        parallel (each worker gets num_steps // num_workers target steps).
        Workers use independent copies of the current policy weights.

        Args:
            num_steps: Number of environment steps to collect

        Returns:
            rollout: Dict containing collected experience
        """
        if self.num_workers > 1:
            return self._collect_rollout_parallel(num_steps)
        return self._collect_rollout_sequential(num_steps)

    def _collect_rollout_parallel(self, num_steps: int):
        """Parallel version: reuses persistent worker pool across iterations."""
        if self._worker_pool is None:
            ctx = mp.get_context('spawn')
            self._worker_pool = ctx.Pool(processes=self.num_workers)

        steps_each = max(1, num_steps // self.num_workers)
        worker_args = [
            self._build_worker_args(worker_seed=self.global_step * 100 + w, steps_each=steps_each)
            for w in range(self.num_workers)
        ]

        worker_rollouts = self._worker_pool.map(_rollout_worker, worker_args)

        # Merge all worker rollouts into one
        merged = {
            'observations': [],
            'graph_lists': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': [],
            'batches': [],
            'next_values': [],
            'map_names': [],
        }
        for wr in worker_rollouts:
            merged['observations'].extend(wr['observations'])
            merged['graph_lists'].extend(wr['graph_lists'])
            merged['actions'].extend(wr['actions'])
            merged['rewards'].extend(wr['rewards'])
            merged['values'].extend(wr['values'])
            merged['log_probs'].extend(wr['log_probs'])
            merged['dones'].extend(wr['dones'])
            merged['batches'].extend(wr['batches'])
            merged['next_values'].extend(wr['next_values'])
            merged['map_names'].extend(wr['map_names'])
            # Aggregate episode-level stats into trainer state
            self.episode_rewards.extend(wr['episode_rewards'])
            self.episode_lengths.extend(wr['episode_lengths'])
            for map_name, rewards in wr['episode_rewards_per_map'].items():
                if map_name in self.episode_rewards_per_map:
                    self.episode_rewards_per_map[map_name].extend(rewards)

        return merged

    def _collect_rollout_sequential(self, num_steps: int):
        """Original sequential rollout collection (num_workers == 1)."""
        rollout = {
            'observations': [],
            'graph_lists': [],  # Individual graphs for action masking
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': [],
            'batches': [],  # Batch indices for graph data
            'next_values': [],  # Next state values for GAE (Bug #2 fix: store per-timestep)
            'map_names': [],  # episode-level list (one entry per completed episode)
        }

        # Randomly pick starting environment (uniform over maps)
        current_env_idx = np.random.randint(len(self.envs))
        current_env = self.envs[current_env_idx]
        current_map_name = self.map_names[current_env_idx]

        # Reset environment
        obs, _ = current_env.reset()

        episode_reward = {agent: 0.0 for agent in obs.keys()}
        episode_length = 0

        steps_collected = 0
        while steps_collected < num_steps:
            # Check if we need to reset (episode ended)
            if len(obs) == 0:
                current_env_idx = np.random.randint(len(self.envs))
                current_env = self.envs[current_env_idx]
                current_map_name = self.map_names[current_env_idx]
                obs, _ = current_env.reset()
                episode_reward = {agent: 0.0 for agent in obs.keys()}
                episode_length = 0

            # Convert observations to batch
            graphs = [obs[agent] for agent in sorted(obs.keys())]
            batched_graph = Batch.from_data_list(graphs)
            batch_size = len(graphs)

            # Forward pass through policy
            with torch.no_grad():
                action_logits, values, _ = self.policy(batched_graph)

            # Sample actions with masking using graph observations
            actions_tensor, log_probs = self.action_decoder.decode_actions(
                action_logits, batched_graph.batch, deterministic=False, return_log_probs=True,
                observations=graphs
            )

            # Convert actions to environment format
            actions_dict = {}
            for i, agent in enumerate(sorted(obs.keys())):
                # Convert from tensor to numpy and then to tuple format expected by env
                action_array = actions_tensor[i].cpu().numpy()  # [action_budget, 3]
                actions_dict[agent] = {
                    'num_actions': self.action_budget,
                    'actions': np.vstack([action_array, np.zeros((10 - self.action_budget, 3))])  # Pad to 10
                }

            # Step environment
            next_obs, rewards, terminateds, truncateds, infos = current_env.step(actions_dict)

            # Track episode stats
            for agent in rewards.keys():
                episode_reward[agent] += rewards[agent]
            episode_length += 1

            # Check if episode ended
            done = terminateds.get('__all__', False) or truncateds.get('__all__', False)
            is_truncated = truncateds.get('__all__', False)
            is_terminated = terminateds.get('__all__', False)

            # Store experience (before checking done, so we have consistent batch sizes)
            # Only include actual agent keys, not '__all__'
            agent_keys = [k for k in sorted(rewards.keys()) if k != '__all__']

            # Compute next_value for GAE bootstrapping
            # Key distinction:
            #   - Terminated: game naturally ended (victory/elimination) -> bootstrap with 0
            #   - Truncated: game artificially cut off (turn limit) -> bootstrap with V(s')
            # This is critical for learning: truncation means the game WOULD continue,
            # so the value estimate should account for potential future rewards.
            if is_terminated:
                # True termination (victory/elimination): no future value
                next_value = torch.zeros(len(agent_keys), device=self.device)
            else:
                # Non-terminal OR truncated: compute value of next state for bootstrapping
                next_graphs = [next_obs[agent] for agent in sorted(next_obs.keys())]
                next_batched_graph = Batch.from_data_list(next_graphs)
                with torch.no_grad():
                    _, next_value, _ = self.policy(next_batched_graph)
                    next_value = next_value.squeeze(-1)  # [batch_size]

            rollout['observations'].append(batched_graph)
            rollout['graph_lists'].append(graphs)  # Store individual graphs for masking
            rollout['actions'].append(actions_tensor)
            rollout['rewards'].append(torch.tensor([rewards[agent] for agent in agent_keys], device=self.device))
            rollout['values'].append(values.squeeze(-1))  # [batch_size]
            rollout['log_probs'].append(log_probs)  # [batch_size, action_budget]
            # Store episode boundaries (both terminated AND truncated) for GAE propagation masking
            # GAE should not propagate across episode boundaries regardless of termination type
            rollout['dones'].append(torch.tensor([done for _ in agent_keys], dtype=torch.bool, device=self.device))
            rollout['batches'].append(batched_graph.batch)
            rollout['next_values'].append(next_value)  # Stores V(s') for truncated, 0 for terminated

            steps_collected += 1

            if done:
                # Log episode stats
                # In self-play, rewards are symmetric (one wins, one loses)
                # Track agent_0's reward to monitor learning progress
                agent_0_reward = episode_reward.get('agent_0', 0.0)
                self.episode_rewards.append(agent_0_reward)
                self.episode_lengths.append(episode_length)

                # Per-map tracking
                self.episode_rewards_per_map[current_map_name].append(agent_0_reward)
                rollout['map_names'].append(current_map_name)

                # Sample new environment (uniform over maps) for next episode
                current_env_idx = np.random.randint(len(self.envs))
                current_env = self.envs[current_env_idx]
                current_map_name = self.map_names[current_env_idx]

                # Reset for next episode
                obs, _ = current_env.reset()
                episode_reward = {agent: 0.0 for agent in obs.keys()}
                episode_length = 0
            else:
                obs = next_obs

        return rollout

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

        T = len(rollout['observations'])
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
            action_logits, new_values, _ = self.policy(mega_batch)

            new_log_probs = self.action_decoder.compute_log_probs(
                action_logits,
                all_actions_flat,
                mega_batch.batch,
                observations=all_graphs,
            ).sum(dim=1).view(T, B)  # [T, B]

            entropy = self.action_decoder.compute_entropy(
                action_logits,
                mega_batch.batch,
                observations=all_graphs,
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

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
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
