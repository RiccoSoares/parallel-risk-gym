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
from parallel_risk.env.reward_shaping import RewardShapingConfig
from parallel_risk.training.torchrl.graph_wrapper import GraphObservationWrapper, env_to_graph
from parallel_risk.models.gnn_gcn import GCNPolicy
from parallel_risk.models.action_decoder import ActionDecoder


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

        self.env = ParallelRiskEnv(
            map_name=env_config['map_name'],
            max_turns=env_config.get('max_turns', 100),
            seed=env_config.get('seed', None),
            reward_shaping_config=reward_shaping_config
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

        # Create action decoder with action masking from config
        masking_config = config.get('masking', {})
        self.action_decoder = ActionDecoder(
            action_budget=self.action_budget,
            max_troops=20,
            mask_source=masking_config.get('mask_source', False),
            mask_dest=masking_config.get('mask_dest', False),
            mask_troops=masking_config.get('mask_troops', False),
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
            'graph_lists': [],  # Individual graphs for action masking
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': [],
            'batches': [],  # Batch indices for graph data
            'next_values': [],  # Next state values for GAE (Bug #2 fix: store per-timestep)
        }

        # Reset environment
        obs, _ = self.wrapped_env.reset()

        episode_reward = {agent: 0.0 for agent in obs.keys()}
        episode_length = 0

        steps_collected = 0
        while steps_collected < num_steps:
            # Check if we need to reset (episode ended)
            if len(obs) == 0:
                obs, _ = self.wrapped_env.reset()
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
            next_obs, rewards, terminateds, truncateds, infos = self.wrapped_env.step(actions_dict)

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

                # Reset for next episode
                obs, _ = self.wrapped_env.reset()
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

        Args:
            rollout: Collected experience
        """
        # Compute advantages with proper per-timestep bootstrapping (Bug #2 fix)
        advantages, returns = self.compute_gae(
            rollout['rewards'],
            rollout['values'],
            rollout['dones'],
            rollout['next_values']  # Bug #2 fix: pass per-timestep next_values
        )

        # Normalize advantages (standard PPO practice, matches RLlib)
        # This ensures consistent gradient scales regardless of reward magnitude
        # Note: Only normalize advantages, NOT returns - value network predicts actual returns
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Flatten returns for value loss (unnormalized - value network predicts actual returns)
        returns_flat = returns.view(-1)

        # Flatten timestep dimension
        T = len(rollout['observations'])
        B = rollout['rewards'][0].size(0)

        # Prepare per-timestep tensors for shuffling
        advantages_per_timestep = [advantages[t] for t in range(T)]  # List of [B] tensors
        old_log_probs_per_timestep = [rollout['log_probs'][t].sum(dim=1).detach() for t in range(T)]  # List of [B] tensors
        returns_per_timestep = [returns[t] for t in range(T)]  # List of [B] tensors
        old_values_per_timestep = [rollout['values'][t].detach() for t in range(T)]  # Bug #3 fix: store old values for clipping

        # Multiple epochs of SGD with shuffled timesteps
        # Shuffling reduces overfitting to temporal ordering in small batches
        for epoch in range(self.num_epochs):
            # Shuffle timestep indices for this epoch
            timestep_indices = torch.randperm(T)

            # Compute new log probs and values in shuffled order
            all_new_log_probs = []
            all_new_values = []
            all_entropies = []
            all_advantages = []
            all_old_log_probs = []
            all_returns = []
            all_old_values = []  # Bug #3 fix: track old values

            for idx in timestep_indices:
                t = idx.item()
                action_logits, values, _ = self.policy(rollout['observations'][t])

                # Compute log probs for actions taken (pass observations for masking)
                log_probs = self.action_decoder.compute_log_probs(
                    action_logits,
                    rollout['actions'][t],
                    rollout['batches'][t],
                    observations=rollout['graph_lists'][t]
                )

                # Compute entropy (pass observations for masking)
                entropy = self.action_decoder.compute_entropy(
                    action_logits,
                    rollout['batches'][t],
                    observations=rollout['graph_lists'][t]
                )

                all_new_log_probs.append(log_probs.sum(dim=1))  # Sum over action_budget
                all_new_values.append(values.squeeze(-1))
                all_entropies.append(entropy.mean(dim=1))  # Mean over action_budget

                # Gather corresponding old values in same shuffled order
                all_advantages.append(advantages_per_timestep[t])
                all_old_log_probs.append(old_log_probs_per_timestep[t])
                all_returns.append(returns_per_timestep[t])
                all_old_values.append(old_values_per_timestep[t])  # Bug #3 fix

            new_log_probs_flat = torch.cat(all_new_log_probs)
            new_values_flat = torch.cat(all_new_values)
            entropies_flat = torch.cat(all_entropies)
            advantages_flat = torch.cat(all_advantages)
            old_log_probs_flat = torch.cat(all_old_log_probs)
            returns_flat = torch.cat(all_returns)
            old_values_flat = torch.cat(all_old_values)  # Bug #3 fix

            # PPO policy loss
            ratio = torch.exp(new_log_probs_flat - old_log_probs_flat)
            surr1 = ratio * advantages_flat
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_flat
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value function clipping with separate vf_clip_param (matches RLlib)
            # Uses larger clip range than policy to allow value network to fit properly
            value_pred_clipped = old_values_flat + torch.clamp(
                new_values_flat - old_values_flat, -self.vf_clip_param, self.vf_clip_param
            )
            value_loss_unclipped = (new_values_flat - returns_flat) ** 2
            value_loss_clipped = (value_pred_clipped - returns_flat) ** 2
            value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

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
                # Log return statistics for monitoring
                self.writer.add_scalar('Stats/return_mean', returns_flat.mean().item(), self.global_step)
                self.writer.add_scalar('Stats/return_std', returns_flat.std().item(), self.global_step)
                # Log value clipping statistics
                clip_fraction = (torch.abs(new_values_flat - old_values_flat) > self.clip_epsilon).float().mean()
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
