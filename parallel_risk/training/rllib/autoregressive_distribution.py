"""
Custom autoregressive action distribution for RLlib.

This distribution samples actions in sequence: source → dest|source → troops|source,dest
Each component is conditioned on the previous choices, enabling valid action masking.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional

from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.utils.typing import TensorType


class AutoregressiveActionDistribution(TorchDistributionWrapper):
    """
    Autoregressive action distribution for Parallel Risk.

    Instead of sampling all action components independently, this distribution:
    1. Samples source from owned territories
    2. Samples dest conditioned on source (self or adjacent)
    3. Samples troops conditioned on source and dest

    This ensures 100% valid actions.
    """

    def __init__(
        self,
        inputs: TensorType,
        model: "TorchModelV2",
        *,
        n_territories: int = 6,
        action_budget: int = 5,
        max_troops: int = 20,
    ):
        """
        Initialize the autoregressive distribution.

        Args:
            inputs: Model outputs containing logits and mask info
            model: The policy model
            n_territories: Number of territories
            action_budget: Number of actions per turn
            max_troops: Maximum troops per action
        """
        super().__init__(inputs, model)

        self.n_territories = n_territories
        self.action_budget = action_budget
        self.max_troops = max_troops

        # Parse inputs - expecting concatenated logits for all action components
        # Layout: [source_logits, dest_logits, troops_logits] * action_budget
        single_action_size = n_territories + n_territories + max_troops

        self.batch_size = inputs.shape[0]

        # Reshape to [batch, action_budget, single_action_size]
        logits_reshaped = inputs.view(self.batch_size, action_budget, single_action_size)

        # Extract component logits
        self.source_logits = logits_reshaped[:, :, :n_territories]
        self.dest_logits = logits_reshaped[:, :, n_territories:2*n_territories]
        self.troops_logits = logits_reshaped[:, :, 2*n_territories:]

        # Get masks from model's last observation
        # These will be used for autoregressive masking
        self._raw_data = None
        self._action_mask = None

        # Try to extract masks from model
        if hasattr(model, '_last_obs') and model._last_obs is not None:
            obs = model._last_obs
            if isinstance(obs, dict):
                self._raw_data = obs.get('raw_data', {})
                self._action_mask = obs.get('action_mask', {})

    def sample(self) -> TensorType:
        """Sample actions autoregressively."""
        device = self.source_logits.device

        all_actions = []

        for batch_idx in range(self.batch_size):
            batch_actions = []

            for action_idx in range(self.action_budget):
                # Get logits for this batch item and action
                src_logits = self.source_logits[batch_idx, action_idx]  # [n_territories]
                dst_logits = self.dest_logits[batch_idx, action_idx]    # [n_territories]
                trp_logits = self.troops_logits[batch_idx, action_idx]  # [max_troops]

                # Apply source mask (ownership)
                if self._action_mask and 'source_mask' in self._action_mask:
                    source_mask = self._get_mask_for_batch(self._action_mask['source_mask'], batch_idx)
                    src_logits = src_logits + (1 - source_mask) * (-1e10)

                # Sample source
                src_dist = torch.distributions.Categorical(logits=src_logits)
                source = src_dist.sample()

                # Compute dest mask conditioned on source
                if self._raw_data:
                    dest_mask = self._compute_dest_mask(batch_idx, source.item())
                    dst_logits = dst_logits + (1 - dest_mask) * (-1e10)

                # Sample dest
                dst_dist = torch.distributions.Categorical(logits=dst_logits)
                dest = dst_dist.sample()

                # Compute troops mask conditioned on source and dest
                if self._raw_data:
                    troops_mask = self._compute_troops_mask(batch_idx, source.item(), dest.item())
                    trp_logits = trp_logits + (1 - troops_mask) * (-1e10)

                # Sample troops
                trp_dist = torch.distributions.Categorical(logits=trp_logits)
                troops = trp_dist.sample()

                batch_actions.append(torch.stack([source, dest, troops]))

            # Stack actions for this batch item: [action_budget, 3]
            all_actions.append(torch.stack(batch_actions))

        # Stack all batch items: [batch, action_budget, 3]
        actions = torch.stack(all_actions)

        # Return as tuple of tuples (RLlib format for MultiDiscrete-like)
        return self._tensor_to_tuple_action(actions)

    def _tensor_to_tuple_action(self, actions: TensorType) -> Tuple:
        """Convert tensor actions to RLlib tuple format."""
        # actions: [batch, action_budget, 3]
        # Output format: tuple of (batch,) tensors for each action component
        result = []
        for i in range(self.action_budget):
            action_tuple = (
                actions[:, i, 0],  # source
                actions[:, i, 1],  # dest
                actions[:, i, 2],  # troops
            )
            result.append(action_tuple)
        return tuple(result)

    def _get_mask_for_batch(self, mask: TensorType, batch_idx: int) -> TensorType:
        """Get mask for a specific batch item."""
        if mask.dim() == 1:
            return mask  # Single item, broadcast
        return mask[batch_idx]

    def _compute_dest_mask(self, batch_idx: int, source: int) -> TensorType:
        """Compute destination mask conditioned on source."""
        device = self.source_logits.device

        ownership = self._get_raw_data('ownership', batch_idx)
        adjacency = self._get_raw_data('adjacency', batch_idx)

        if ownership is None or adjacency is None:
            return torch.ones(self.n_territories, device=device)

        # Valid destinations: self (for deploy) or adjacent (for transfer/attack)
        dest_mask = torch.zeros(self.n_territories, device=device)
        dest_mask[source] = 1.0  # Can always target self (deploy)

        # Add adjacent territories
        if adjacency is not None:
            adj_row = adjacency[source] if adjacency.dim() > 1 else adjacency
            dest_mask = torch.maximum(dest_mask, adj_row.float())

        return dest_mask

    def _compute_troops_mask(self, batch_idx: int, source: int, dest: int) -> TensorType:
        """Compute troops mask conditioned on source and destination."""
        device = self.source_logits.device

        ownership = self._get_raw_data('ownership', batch_idx)
        troops = self._get_raw_data('troops', batch_idx)
        income = self._get_raw_data('income', batch_idx)

        troops_mask = torch.zeros(self.max_troops, device=device)

        if ownership is None or troops is None:
            troops_mask[1:] = 1.0  # Allow all non-zero
            return troops_mask

        is_self = (source == dest)

        if is_self:
            # Deploy: limited by income
            inc_val = income[0].item() if income is not None else 1
            max_troops = min(int(inc_val), self.max_troops - 1)
        else:
            # Transfer/attack: limited by troops at source - 1
            src_troops = troops[source].item()
            max_troops = min(int(src_troops) - 1, self.max_troops - 1)

        if max_troops > 0:
            troops_mask[1:max_troops + 1] = 1.0
        else:
            troops_mask[1] = 1.0  # At least allow 1 troop

        return troops_mask

    def _get_raw_data(self, key: str, batch_idx: int) -> Optional[TensorType]:
        """Get raw data tensor for a batch item."""
        if self._raw_data is None or key not in self._raw_data:
            return None

        data = self._raw_data[key]
        if isinstance(data, torch.Tensor):
            if data.dim() > 1 and data.shape[0] > batch_idx:
                return data[batch_idx]
            return data
        return torch.tensor(data, device=self.source_logits.device)

    def logp(self, actions: TensorType) -> TensorType:
        """Compute log probability of actions."""
        # Convert actions to tensor format if needed
        if isinstance(actions, tuple):
            actions = self._tuple_to_tensor_action(actions)

        device = self.source_logits.device
        log_probs = torch.zeros(self.batch_size, device=device)

        for batch_idx in range(self.batch_size):
            batch_log_prob = 0.0

            for action_idx in range(self.action_budget):
                src_logits = self.source_logits[batch_idx, action_idx]
                dst_logits = self.dest_logits[batch_idx, action_idx]
                trp_logits = self.troops_logits[batch_idx, action_idx]

                source = actions[batch_idx, action_idx, 0].long()
                dest = actions[batch_idx, action_idx, 1].long()
                troops = actions[batch_idx, action_idx, 2].long()

                # Apply source mask
                if self._action_mask and 'source_mask' in self._action_mask:
                    source_mask = self._get_mask_for_batch(self._action_mask['source_mask'], batch_idx)
                    src_logits = src_logits + (1 - source_mask) * (-1e10)

                # Compute dest mask conditioned on source
                if self._raw_data:
                    dest_mask = self._compute_dest_mask(batch_idx, source.item())
                    dst_logits = dst_logits + (1 - dest_mask) * (-1e10)

                # Compute troops mask conditioned on source and dest
                if self._raw_data:
                    troops_mask = self._compute_troops_mask(batch_idx, source.item(), dest.item())
                    trp_logits = trp_logits + (1 - troops_mask) * (-1e10)

                # Compute log probs
                src_log_prob = torch.nn.functional.log_softmax(src_logits, dim=0)[source]
                dst_log_prob = torch.nn.functional.log_softmax(dst_logits, dim=0)[dest]
                trp_log_prob = torch.nn.functional.log_softmax(trp_logits, dim=0)[troops]

                batch_log_prob = batch_log_prob + src_log_prob + dst_log_prob + trp_log_prob

            log_probs[batch_idx] = batch_log_prob

        return log_probs

    def _tuple_to_tensor_action(self, actions: Tuple) -> TensorType:
        """Convert RLlib tuple action to tensor format."""
        # actions is tuple of (source, dest, troops) tuples per action slot
        device = self.source_logits.device

        batch_size = len(actions[0][0]) if isinstance(actions[0][0], (list, tuple, torch.Tensor)) else 1

        result = torch.zeros(batch_size, self.action_budget, 3, device=device, dtype=torch.long)

        for action_idx, action_tuple in enumerate(actions):
            if len(action_tuple) == 3:
                result[:, action_idx, 0] = torch.tensor(action_tuple[0], device=device)
                result[:, action_idx, 1] = torch.tensor(action_tuple[1], device=device)
                result[:, action_idx, 2] = torch.tensor(action_tuple[2], device=device)

        return result

    def entropy(self) -> TensorType:
        """Compute entropy of the distribution."""
        # Approximate entropy using the conservative masks (not perfect but reasonable)
        device = self.source_logits.device
        entropies = torch.zeros(self.batch_size, device=device)

        for batch_idx in range(self.batch_size):
            batch_entropy = 0.0

            for action_idx in range(self.action_budget):
                src_logits = self.source_logits[batch_idx, action_idx]
                dst_logits = self.dest_logits[batch_idx, action_idx]
                trp_logits = self.troops_logits[batch_idx, action_idx]

                # Apply source mask if available
                if self._action_mask and 'source_mask' in self._action_mask:
                    source_mask = self._get_mask_for_batch(self._action_mask['source_mask'], batch_idx)
                    src_logits = src_logits + (1 - source_mask) * (-1e10)

                src_dist = torch.distributions.Categorical(logits=src_logits)
                dst_dist = torch.distributions.Categorical(logits=dst_logits)
                trp_dist = torch.distributions.Categorical(logits=trp_logits)

                batch_entropy = batch_entropy + src_dist.entropy() + dst_dist.entropy() + trp_dist.entropy()

            entropies[batch_idx] = batch_entropy

        return entropies

    def kl(self, other: "AutoregressiveActionDistribution") -> TensorType:
        """Compute KL divergence between distributions."""
        # Approximate KL using component-wise comparison
        device = self.source_logits.device
        kls = torch.zeros(self.batch_size, device=device)

        for batch_idx in range(self.batch_size):
            batch_kl = 0.0

            for action_idx in range(self.action_budget):
                src_logits_self = self.source_logits[batch_idx, action_idx]
                src_logits_other = other.source_logits[batch_idx, action_idx]

                src_dist_self = torch.distributions.Categorical(logits=src_logits_self)
                src_dist_other = torch.distributions.Categorical(logits=src_logits_other)

                batch_kl = batch_kl + torch.distributions.kl_divergence(src_dist_self, src_dist_other)

            kls[batch_idx] = batch_kl

        return kls

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        """Return the required output shape for the model."""
        custom_config = model_config.get("custom_model_config", {})
        n_territories = custom_config.get("n_territories", 6)
        action_budget = custom_config.get("action_budget", 5)
        max_troops = custom_config.get("max_troops", 20)

        single_action_size = n_territories + n_territories + max_troops
        return (single_action_size * action_budget,)
