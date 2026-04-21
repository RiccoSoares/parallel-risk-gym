"""
Training script for Parallel Risk with GNN policies.

Implements PPO training with self-play for GNN-based policies.

Usage:
    python -m parallel_risk.training.torchrl.train --config configs/gnn_gcn.yaml
"""

import argparse
import os
import yaml
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
from parallel_risk.training.torchrl.graph_wrapper import GraphObservationWrapper, env_to_graph
from parallel_risk.models.gnn_gcn import GCNPolicy
from parallel_risk.models.action_decoder import ActionDecoder


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
        self.env = ParallelRiskEnv(
            map_name=env_config['map_name'],
            max_turns=env_config.get('max_turns', 100),
            seed=env_config.get('seed', None)
        )
        self.wrapped_env = GraphObservationWrapper(self.env, device=self.device)

        # Get graph observation info
        obs_space = self.wrapped_env.observation_space
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
            max_troops=20
        )

        # Optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)

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

    def collect_rollout(self, num_steps: int):
        """
        Collect experience by running the policy in the environment.

        Args:
            num_steps: Number of environment steps to collect

        Returns:
            rollout: Dict containing collected experience
        """
        rollout = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': [],
            'batches': [],  # Batch indices for graph data
        }

        # Reset environment
        obs, _ = self.wrapped_env.reset()

        episode_reward = {agent: 0.0 for agent in obs.keys()}
        episode_length = 0

        for step in range(num_steps):
            # Check if we need to reset (episode ended)
            if len(obs) == 0:
                obs, _ = self.wrapped_env.reset()
                episode_reward = {agent: 0.0 for agent in obs.keys()}
                episode_length = 0

            # Convert observations to batch
            graphs = [obs[agent] for agent in sorted(obs.keys())]
            batched_graph = Batch.from_data_list(graphs)

            # Forward pass through policy
            with torch.no_grad():
                action_logits, values, _ = self.policy(batched_graph)

            # Sample actions
            actions_tensor, log_probs = self.action_decoder.decode_actions(
                action_logits, batched_graph.batch, deterministic=False, return_log_probs=True
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
            next_obs, rewards, terminateds, truncateds, infos = self.wrapped_env.step(actions_dict)

            # Store experience
            rollout['observations'].append(batched_graph)
            rollout['actions'].append(actions_tensor)
            rollout['rewards'].append(torch.tensor([rewards[agent] for agent in sorted(rewards.keys())], device=self.device))
            rollout['values'].append(values.squeeze(-1))  # [batch_size]
            rollout['log_probs'].append(log_probs)  # [batch_size, action_budget]
            rollout['dones'].append(torch.tensor([terminateds[agent] for agent in sorted(terminateds.keys())], device=self.device))
            rollout['batches'].append(batched_graph.batch)

            # Track episode stats
            for agent in rewards.keys():
                episode_reward[agent] += rewards[agent]
            episode_length += 1

            # Check if episode ended
            done = terminateds.get('__all__', False) or truncateds.get('__all__', False)

            if done:
                # Log episode stats
                avg_reward = np.mean(list(episode_reward.values()))
                self.episode_rewards.append(avg_reward)
                self.episode_lengths.append(episode_length)

                # Reset
                obs, _ = self.wrapped_env.reset()
                episode_reward = {agent: 0.0 for agent in obs.keys()}
                episode_length = 0
            else:
                obs = next_obs

        return rollout

    def compute_gae(self, rewards, values, dones):
        """
        Compute Generalized Advantage Estimation (GAE).

        Args:
            rewards: List of reward tensors [batch_size]
            values: List of value tensors [batch_size]
            dones: List of done flags [batch_size]

        Returns:
            advantages: Tensor of advantages
            returns: Tensor of returns
        """
        advantages = []
        returns = []

        gae = 0
        next_value = 0

        # Reverse iteration
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t].float()
            delta = rewards[t] + self.gamma * next_value * mask - values[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae

            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])

            next_value = values[t]

        advantages = torch.stack(advantages)
        returns = torch.stack(returns)

        return advantages, returns

    def update_policy(self, rollout):
        """
        Update policy using PPO.

        Args:
            rollout: Collected experience
        """
        # Compute advantages
        advantages, returns = self.compute_gae(
            rollout['rewards'],
            rollout['values'],
            rollout['dones']
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Flatten timestep dimension
        T = len(rollout['observations'])
        B = rollout['rewards'][0].size(0)

        advantages_flat = advantages.view(-1)
        returns_flat = returns.view(-1)
        old_log_probs_flat = torch.cat([lp.sum(dim=1) for lp in rollout['log_probs']])  # Sum over action_budget

        # Multiple epochs of SGD
        for epoch in range(self.num_epochs):
            # Compute new log probs and values
            all_new_log_probs = []
            all_new_values = []
            all_entropies = []

            for t in range(T):
                action_logits, values, _ = self.policy(rollout['observations'][t])

                # Compute log probs for actions taken
                log_probs = self.action_decoder.compute_log_probs(
                    action_logits,
                    rollout['actions'][t],
                    rollout['batches'][t]
                )

                # Compute entropy
                entropy = self.action_decoder.compute_entropy(
                    action_logits,
                    rollout['batches'][t]
                )

                all_new_log_probs.append(log_probs.sum(dim=1))  # Sum over action_budget
                all_new_values.append(values.squeeze(-1))
                all_entropies.append(entropy.mean(dim=1))  # Mean over action_budget

            new_log_probs_flat = torch.cat(all_new_log_probs)
            new_values_flat = torch.cat(all_new_values)
            entropies_flat = torch.cat(all_entropies)

            # PPO policy loss
            ratio = torch.exp(new_log_probs_flat - old_log_probs_flat)
            surr1 = ratio * advantages_flat
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_flat
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = nn.functional.mse_loss(new_values_flat, returns_flat)

            # Entropy bonus
            entropy_loss = -entropies_flat.mean()

            # Total loss
            loss = policy_loss + self.value_loss_coeff * value_loss + self.entropy_coeff * entropy_loss

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # Log
            if epoch == self.num_epochs - 1:  # Log on last epoch
                self.writer.add_scalar('Loss/policy', policy_loss.item(), self.global_step)
                self.writer.add_scalar('Loss/value', value_loss.item(), self.global_step)
                self.writer.add_scalar('Loss/entropy', entropy_loss.item(), self.global_step)
                self.writer.add_scalar('Loss/total', loss.item(), self.global_step)

        self.global_step += 1

    def train(self, num_iterations: int):
        """
        Main training loop.

        Args:
            num_iterations: Number of training iterations
        """
        print(f"Starting training for {num_iterations} iterations...")
        print(f"Device: {self.device}")
        print(f"Map: {self.config['env']['map_name']}")
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
