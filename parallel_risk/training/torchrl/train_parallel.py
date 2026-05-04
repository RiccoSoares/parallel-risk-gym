"""
Parallel training script for Parallel Risk with GNN policies.

Uses vectorized environments for faster data collection.

Usage:
    python -m parallel_risk.training.torchrl.train_parallel --config configs/gnn_gcn.yaml --num-workers 4
"""

import argparse
import os
import yaml
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Batch, Data

from parallel_risk.training.torchrl.vec_env import make_vec_env
from parallel_risk.models.gnn_gcn import GCNPolicy
from parallel_risk.models.action_decoder import ActionDecoder


class PPOTrainerParallel:
    """
    PPO trainer with vectorized environments for parallel data collection.

    Key differences from single-env trainer:
    - Runs num_workers environments in parallel
    - Collects batch_size / num_workers steps per environment
    - Aggregates experiences across all environments
    """

    def __init__(self, config: Dict[str, Any], num_workers: int = 4):
        """
        Initialize parallel PPO trainer.

        Args:
            config: Configuration dict with hyperparameters
            num_workers: Number of parallel environments
        """
        self.config = config
        self.num_workers = num_workers
        self.device = torch.device('cuda' if torch.cuda.is_available() and config['training'].get('use_gpu', False) else 'cpu')

        # Environment config
        env_config = config['env']

        # Create vectorized environment
        self.vec_env = make_vec_env(
            num_envs=num_workers,
            map_name=env_config['map_name'],
            max_turns=env_config.get('max_turns', 50),
            use_reward_shaping=env_config.get('use_reward_shaping', True),
            seed=env_config.get('seed', None),
            device=self.device
        )

        # Get observation space info
        obs_space = self.vec_env.observation_space
        self.node_features_dim = obs_space['node_features_dim']
        self.global_features_dim = obs_space['global_features_dim']

        # Training hyperparameters
        train_config = config['training']
        self.batch_size = train_config.get('batch_size', 4096)
        self.minibatch_size = train_config.get('minibatch_size', 256)  # Samples per minibatch
        self.num_epochs = train_config.get('num_epochs', 10)
        self.learning_rate = train_config.get('learning_rate', 3e-4)
        self.gamma = train_config.get('gamma', 0.99)
        self.gae_lambda = train_config.get('gae_lambda', 0.95)
        self.clip_epsilon = train_config.get('clip_epsilon', 0.2)
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

        # Create action decoder
        self.action_decoder = ActionDecoder(
            action_budget=self.action_budget,
            max_troops=20,
            mask_source=False,
            mask_dest=False,
            mask_troops=False,
        )

        # Optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)

        # TensorBoard logging
        log_dir = config.get('logging', {}).get('log_dir', 'runs/gnn_training_parallel')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(log_dir=f"{log_dir}/{timestamp}")

        # Checkpoint directory
        self.checkpoint_dir = Path(config.get('logging', {}).get('checkpoint_dir', 'checkpoints/gnn_training_parallel'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training statistics
        self.global_step = 0
        self.episode_rewards = []
        self.episode_lengths = []

    def collect_rollout(self, num_steps_per_env: int):
        """
        Collect experience from all vectorized environments.

        Args:
            num_steps_per_env: Steps to collect per environment

        Returns:
            rollout: Dict containing aggregated experience from all envs
        """
        rollout = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': [],
            'batches': [],
            'next_values': [],
        }

        # Reset all environments
        obs_list, _ = self.vec_env.reset()

        # Track episode stats per environment
        episode_rewards = [{agent: 0.0 for agent in obs.keys()} for obs in obs_list]
        episode_lengths = [0] * self.num_workers

        for step in range(num_steps_per_env):
            # Flatten observations from all envs into single batch
            all_graphs = []
            env_agent_mapping = []  # Track which graphs belong to which env/agent

            for env_idx, obs in enumerate(obs_list):
                for agent in sorted(obs.keys()):
                    all_graphs.append(obs[agent])
                    env_agent_mapping.append((env_idx, agent))

            batched_graph = Batch.from_data_list(all_graphs)

            # Forward pass through policy
            with torch.no_grad():
                action_logits, values, _ = self.policy(batched_graph)

            # Sample actions with masking
            actions_tensor, log_probs = self.action_decoder.decode_actions(
                action_logits, batched_graph.batch, deterministic=False, return_log_probs=True,
                observations=all_graphs
            )

            # Convert actions back to per-env format
            actions_list = [{} for _ in range(self.num_workers)]
            for i, (env_idx, agent) in enumerate(env_agent_mapping):
                action_array = actions_tensor[i].cpu().numpy()
                actions_list[env_idx][agent] = {
                    'num_actions': self.action_budget,
                    'actions': np.vstack([action_array, np.zeros((10 - self.action_budget, 3), dtype=np.int32)])
                }

            # Step all environments
            next_obs_list, rewards_list, terminateds_list, truncateds_list, infos_list = self.vec_env.step(actions_list)

            # Process results from each environment
            for env_idx in range(self.num_workers):
                rewards = rewards_list[env_idx]
                terminateds = terminateds_list[env_idx]
                truncateds = truncateds_list[env_idx]

                done = terminateds.get('__all__', False) or truncateds.get('__all__', False)

                # Update episode stats (only for active agents)
                for agent in list(episode_rewards[env_idx].keys()):
                    if agent in rewards:
                        episode_rewards[env_idx][agent] += rewards[agent]
                episode_lengths[env_idx] += 1

                if done:
                    # Log episode stats
                    avg_reward = np.mean(list(episode_rewards[env_idx].values()))
                    self.episode_rewards.append(avg_reward)
                    self.episode_lengths.append(episode_lengths[env_idx])

                    # Reset tracking for this env (next_obs_list already has reset obs due to auto-reset)
                    episode_rewards[env_idx] = {agent: 0.0 for agent in next_obs_list[env_idx].keys()}
                    episode_lengths[env_idx] = 0

            # Compute next values for GAE
            agent_keys = [k for k in sorted(rewards_list[0].keys()) if k != '__all__']
            num_agents_per_env = len(agent_keys)
            total_agents = self.num_workers * num_agents_per_env

            # Check termination status per environment
            all_terminated = [terminateds_list[i].get('__all__', False) for i in range(self.num_workers)]
            all_done = [terminateds_list[i].get('__all__', False) or truncateds_list[i].get('__all__', False)
                       for i in range(self.num_workers)]

            # Compute next_value
            if all(all_terminated):
                next_value = torch.zeros(total_agents, device=self.device)
            else:
                next_graphs = []
                for env_idx, obs in enumerate(next_obs_list):
                    for agent in sorted(obs.keys()):
                        next_graphs.append(obs[agent])
                next_batched = Batch.from_data_list(next_graphs)
                with torch.no_grad():
                    _, next_value, _ = self.policy(next_batched)
                    next_value = next_value.squeeze(-1)

                # Zero out values for terminated environments
                for env_idx in range(self.num_workers):
                    if all_terminated[env_idx]:
                        start_idx = env_idx * num_agents_per_env
                        end_idx = start_idx + num_agents_per_env
                        next_value[start_idx:end_idx] = 0.0

            # Store rollout data
            rollout['observations'].append(batched_graph)
            rollout['actions'].append(actions_tensor)

            # Flatten rewards across envs
            flat_rewards = []
            for env_idx in range(self.num_workers):
                for agent in agent_keys:
                    flat_rewards.append(rewards_list[env_idx][agent])
            rollout['rewards'].append(torch.tensor(flat_rewards, device=self.device))

            rollout['values'].append(values.squeeze(-1))
            rollout['log_probs'].append(log_probs)

            # Store done flags (for GAE masking)
            flat_dones = []
            for env_idx in range(self.num_workers):
                done = all_done[env_idx]
                for _ in agent_keys:
                    flat_dones.append(done)
            rollout['dones'].append(torch.tensor(flat_dones, dtype=torch.bool, device=self.device))

            rollout['batches'].append(batched_graph.batch)
            rollout['next_values'].append(next_value)

            # Update observations (handle resets)
            for env_idx in range(self.num_workers):
                if all_done[env_idx]:
                    # Environment already auto-reset via vec_env
                    pass
            obs_list = next_obs_list

        return rollout

    def compute_gae(self, rewards, values, dones, next_values):
        """Compute GAE (same as single-env trainer)."""
        batch_size = rewards[0].size(0)
        advantages = []
        returns = []

        gae = torch.zeros(batch_size, device=self.device)

        for t in reversed(range(len(rewards))):
            next_val = next_values[t]
            mask = 1.0 - dones[t].float()
            delta = rewards[t] + self.gamma * next_val - values[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae

            advantages.insert(0, gae.clone())
            returns.insert(0, gae + values[t])

        return torch.stack(advantages), torch.stack(returns)

    def update_policy(self, rollout):
        """Update policy using PPO with shuffled minibatches."""
        advantages, returns = self.compute_gae(
            rollout['rewards'],
            rollout['values'],
            rollout['dones'],
            rollout['next_values']
        )

        T = len(rollout['observations'])
        B = rollout['rewards'][0].size(0)  # Batch size per timestep (num_workers * num_agents)
        total_samples = T * B

        # Pre-compute all old log probs and values
        old_log_probs = torch.cat([rollout['log_probs'][t].sum(dim=1).detach() for t in range(T)])
        old_values = torch.cat([rollout['values'][t].detach() for t in range(T)])
        all_advantages = advantages.view(-1)
        all_returns = returns.view(-1)

        # Normalize advantages (standard PPO practice, matches RLlib)
        # This ensures consistent gradient scales regardless of reward magnitude
        all_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8)

        # Flatten actions: [T, B, action_budget, 3] -> [T*B, action_budget, 3]
        all_actions = torch.cat([rollout['actions'][t] for t in range(T)], dim=0)

        # Store per-timestep data for minibatch construction
        # We need to track which timestep each sample came from for graph batching
        timestep_indices = torch.arange(T, device=self.device).repeat_interleave(B)
        sample_indices_within_timestep = torch.arange(B, device=self.device).repeat(T)

        for epoch in range(self.num_epochs):
            # Shuffle sample indices for this epoch
            perm = torch.randperm(total_samples, device=self.device)

            # Process minibatches
            num_minibatches = max(1, total_samples // self.minibatch_size)

            for mb_idx in range(num_minibatches):
                start_idx = mb_idx * self.minibatch_size
                end_idx = min(start_idx + self.minibatch_size, total_samples)
                mb_indices = perm[start_idx:end_idx]

                # Get minibatch data
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_old_values = old_values[mb_indices]
                mb_advantages = all_advantages[mb_indices]
                mb_returns = all_returns[mb_indices]
                mb_actions = all_actions[mb_indices]

                # Reconstruct graphs for this minibatch
                # Group samples by their original timestep for efficient batching
                mb_timesteps = timestep_indices[mb_indices]
                mb_sample_idx = sample_indices_within_timestep[mb_indices]

                # Build batched graph from selected samples
                mb_graphs = []
                mb_graph_to_sample = []  # Maps graph index back to minibatch index

                # Get unique timesteps and their samples
                unique_timesteps = mb_timesteps.unique(sorted=True)

                for t in unique_timesteps:
                    t = t.item()
                    # Get the original batched graph for this timestep
                    orig_graph = rollout['observations'][t]

                    # Find which samples from this timestep are in the minibatch
                    mask = (mb_timesteps == t)
                    samples_in_mb = mb_sample_idx[mask]

                    # Extract individual graphs for these samples
                    # orig_graph.batch tells us which nodes belong to which sample
                    for local_idx, sample_idx in enumerate(samples_in_mb):
                        sample_idx = sample_idx.item()
                        # Extract subgraph for this sample
                        node_mask = (orig_graph.batch == sample_idx)
                        subgraph = Data(
                            x=orig_graph.x[node_mask],
                            edge_index=self._extract_subgraph_edges(orig_graph.edge_index, node_mask),
                            num_nodes=node_mask.sum().item()
                        )
                        # Get global features for this sample
                        subgraph.global_features = orig_graph.global_features[sample_idx:sample_idx+1]
                        mb_graphs.append(subgraph)

                # Batch the minibatch graphs
                mb_batched = Batch.from_data_list(mb_graphs)

                # Forward pass
                action_logits, new_values, _ = self.policy(mb_batched)

                new_log_probs = self.action_decoder.compute_log_probs(
                    action_logits, mb_actions, mb_batched.batch
                ).sum(dim=1)

                entropies = self.action_decoder.compute_entropy(
                    action_logits, mb_batched.batch
                ).mean(dim=1)

                new_values = new_values.squeeze(-1)

                # PPO losses
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_pred_clipped = mb_old_values + torch.clamp(
                    new_values - mb_old_values, -self.clip_epsilon, self.clip_epsilon
                )
                value_loss_unclipped = (new_values - mb_returns) ** 2
                value_loss_clipped = (value_pred_clipped - mb_returns) ** 2
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                entropy_loss = -entropies.mean()

                loss = policy_loss + self.value_loss_coeff * value_loss + self.entropy_coeff * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

        # Log final minibatch losses
        self.writer.add_scalar('Loss/policy', policy_loss.item(), self.global_step)
        self.writer.add_scalar('Loss/value', value_loss.item(), self.global_step)
        self.writer.add_scalar('Loss/entropy', entropy_loss.item(), self.global_step)
        self.writer.add_scalar('Loss/total', loss.item(), self.global_step)

        self.global_step += 1

    def _extract_subgraph_edges(self, edge_index, node_mask):
        """Extract edges for a subgraph given a node mask."""
        # Get the indices of nodes in the subgraph
        node_indices = torch.where(node_mask)[0]

        # Create mapping from old indices to new indices
        index_map = torch.full((node_mask.size(0),), -1, dtype=torch.long, device=edge_index.device)
        index_map[node_indices] = torch.arange(len(node_indices), device=edge_index.device)

        # Filter edges where both endpoints are in the subgraph
        src, dst = edge_index
        edge_mask = node_mask[src] & node_mask[dst]

        # Remap edge indices
        new_src = index_map[src[edge_mask]]
        new_dst = index_map[dst[edge_mask]]

        return torch.stack([new_src, new_dst], dim=0)

    def train(self, num_iterations: int, checkpoint_interval: int = 10):
        """Main training loop."""
        print(f"Starting parallel training for {num_iterations} iterations...")
        print(f"Device: {self.device}")
        print(f"Workers: {self.num_workers}")
        print(f"Map: {self.config['env']['map_name']}")
        print(f"Policy: GCN ({self.policy.hidden_dim}x{self.policy.num_layers})")
        print()

        # Steps per env = total batch / (num_workers * 2 agents)
        steps_per_env = self.batch_size // (self.num_workers * 2)

        for iteration in range(num_iterations):
            rollout = self.collect_rollout(steps_per_env)
            self.update_policy(rollout)

            if len(self.episode_rewards) > 0:
                avg_reward = np.mean(self.episode_rewards[-20:])
                avg_length = np.mean(self.episode_lengths[-20:])

                self.writer.add_scalar('Episode/reward', avg_reward, iteration)
                self.writer.add_scalar('Episode/length', avg_length, iteration)

                print(f"Iteration {iteration+1}/{num_iterations} | "
                      f"Reward: {avg_reward:.3f} | Length: {avg_length:.1f} | "
                      f"Episodes: {len(self.episode_rewards)}")

            if (iteration + 1) % checkpoint_interval == 0:
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
        self.vec_env.close()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Parallel Train Parallel Risk with GNN + PPO")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--num-iterations", type=int, default=1000, help="Training iterations")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--checkpoint-interval", type=int, default=10, help="Checkpoint save interval")

    args = parser.parse_args()

    config = load_config(args.config)
    trainer = PPOTrainerParallel(config, num_workers=args.num_workers)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        trainer.policy.load_state_dict(checkpoint['policy_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"✓ Loaded checkpoint from iteration {checkpoint['iteration']}")

    trainer.train(args.num_iterations, checkpoint_interval=args.checkpoint_interval)


if __name__ == "__main__":
    main()
