"""
Custom RLlib model with autoregressive action masking.

This model applies action masks during both training and inference to ensure
only valid actions are sampled. Uses autoregressive decoding where:
1. Source is sampled from owned territories
2. Destination is sampled conditioned on source (self or adjacent)
3. Troops is sampled conditioned on source and destination
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from gymnasium import spaces


class AutoregressiveMaskedModel(TorchModelV2, nn.Module):
    """
    Custom model with autoregressive action masking for Parallel Risk.

    Architecture:
    - Shared feature extractor (MLP)
    - Separate heads for source, dest, troops
    - Autoregressive masking during action sampling

    The model produces logits for all action components, and masking is applied
    during the forward pass to ensure only valid actions have non-zero probability.
    """

    def __init__(
        self,
        obs_space: spaces.Space,
        action_space: spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        **kwargs,
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        # Extract dimensions
        self.n_territories = model_config.get("custom_model_config", {}).get(
            "n_territories", 6
        )
        self.action_budget = model_config.get("custom_model_config", {}).get(
            "action_budget", 5
        )
        self.max_troops = model_config.get("custom_model_config", {}).get(
            "max_troops", 20
        )

        # Get hidden layer sizes
        hiddens = model_config.get("fcnet_hiddens", [256, 256])
        activation = model_config.get("fcnet_activation", "relu")

        if activation == "relu":
            self.activation = nn.ReLU
        elif activation == "tanh":
            self.activation = nn.Tanh
        else:
            self.activation = nn.ReLU

        # Get observation size (just the flat observation part)
        if isinstance(obs_space, spaces.Dict):
            obs_size = obs_space["observations"].shape[0]
        else:
            obs_size = obs_space.shape[0]

        # Build shared feature extractor
        layers = []
        prev_size = obs_size
        for hidden_size in hiddens:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(self.activation())
            prev_size = hidden_size

        self.feature_extractor = nn.Sequential(*layers)
        self.feature_dim = prev_size

        # Action heads for each component
        # For each action in the budget, we need source, dest, troops logits
        self.source_head = nn.Linear(self.feature_dim, self.n_territories)
        self.dest_head = nn.Linear(self.feature_dim, self.n_territories)
        self.troops_head = nn.Linear(self.feature_dim, self.max_troops)

        # Value function head
        self.value_head = nn.Linear(self.feature_dim, 1)

        # Store last features for value function
        self._features = None
        self._value = None

    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:
        """Forward pass with action masking.

        Args:
            input_dict: Dict containing observations and masks
            state: RNN state (unused)
            seq_lens: Sequence lengths (unused)

        Returns:
            action_logits: [batch, action_budget * (n_territories + n_territories + max_troops)]
            state: Empty list
        """
        obs = input_dict["obs"]

        # Handle dict observation space
        if isinstance(obs, dict):
            flat_obs = obs["observations"]
            action_mask = obs.get("action_mask", {})
            raw_data = obs.get("raw_data", {})
        else:
            flat_obs = obs
            action_mask = {}
            raw_data = {}

        # Extract features
        features = self.feature_extractor(flat_obs)
        self._features = features

        # Compute logits for each component
        source_logits = self.source_head(features)  # [batch, n_territories]
        dest_logits = self.dest_head(features)      # [batch, n_territories]
        troops_logits = self.troops_head(features)  # [batch, max_troops]

        # Apply masks if available
        if "source_mask" in action_mask:
            source_mask = action_mask["source_mask"]
            # Mask invalid sources with large negative value
            source_logits = source_logits + (1 - source_mask) * (-1e10)

        if "dest_mask" in action_mask:
            dest_mask = action_mask["dest_mask"]
            dest_logits = dest_logits + (1 - dest_mask) * (-1e10)

        if "troops_mask" in action_mask:
            troops_mask = action_mask["troops_mask"]
            troops_logits = troops_logits + (1 - troops_mask) * (-1e10)

        # Compute value
        self._value = self.value_head(features).squeeze(-1)

        # Concatenate logits for all actions in budget
        # Each action needs: source (n_territories) + dest (n_territories) + troops (max_troops)
        single_action_logits = torch.cat([source_logits, dest_logits, troops_logits], dim=-1)

        # Repeat for action budget
        all_action_logits = single_action_logits.unsqueeze(1).repeat(
            1, self.action_budget, 1
        ).reshape(single_action_logits.shape[0], -1)

        return all_action_logits, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        """Return value function output."""
        return self._value

    def get_action_dist_inputs(
        self,
        features: TensorType,
        action_mask: Dict[str, TensorType],
        raw_data: Dict[str, TensorType],
    ) -> Dict[str, TensorType]:
        """Get masked action distribution inputs for autoregressive sampling.

        This is used during inference for proper autoregressive decoding.

        Args:
            features: Extracted features
            action_mask: Conservative action masks
            raw_data: Raw observation data for computing exact masks

        Returns:
            Dict with source_logits, dest_logits, troops_logits
        """
        source_logits = self.source_head(features)
        dest_logits = self.dest_head(features)
        troops_logits = self.troops_head(features)

        # Apply source mask
        if "source_mask" in action_mask:
            source_logits = source_logits + (1 - action_mask["source_mask"]) * (-1e10)

        return {
            "source_logits": source_logits,
            "dest_logits": dest_logits,
            "troops_logits": troops_logits,
            "raw_data": raw_data,
        }


class SimpleMaskedModel(TorchModelV2, nn.Module):
    """
    Simpler masked model that applies conservative masks to all action components.

    This is easier to integrate with RLlib's existing action distributions.
    Uses the same mask for all actions in the budget (conservative masking).
    """

    def __init__(
        self,
        obs_space: spaces.Space,
        action_space: spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        **kwargs,
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        # Extract custom config
        custom_config = model_config.get("custom_model_config", {})
        self.n_territories = custom_config.get("n_territories", 6)
        self.action_budget = custom_config.get("action_budget", 5)
        self.max_troops = custom_config.get("max_troops", 20)

        # Get hidden layer config
        hiddens = model_config.get("fcnet_hiddens", [256, 256])

        # Get observation size from config (most reliable)
        obs_size = custom_config.get("obs_size", None)

        if obs_size is None:
            # Fallback: try to extract from observation space
            if isinstance(obs_space, spaces.Dict):
                if "observations" in obs_space.spaces:
                    obs_size = int(np.prod(obs_space["observations"].shape))
                else:
                    # Last resort: manually calculate for simple_6
                    n_regions = 3  # simple_6 map
                    obs_size = (
                        self.n_territories +  # ownership
                        self.n_territories +  # troops
                        self.n_territories * self.n_territories +  # adjacency
                        1 +  # income
                        1 +  # turn
                        n_regions  # region control
                    )
            else:
                obs_size = int(np.prod(obs_space.shape))

        print(f"SimpleMaskedModel: obs_size={obs_size}, n_territories={self.n_territories}, "
              f"action_budget={self.action_budget}, max_troops={self.max_troops}")

        # Shared feature extractor
        layers = []
        prev_size = obs_size
        for hidden_size in hiddens:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        self.feature_extractor = nn.Sequential(*layers)

        # Output size for one action: source + dest + troops
        single_action_size = self.n_territories + self.n_territories + self.max_troops
        total_action_size = single_action_size * self.action_budget

        # Policy head
        self.policy_head = nn.Linear(prev_size, total_action_size)

        # Value head
        self.value_head = nn.Linear(prev_size, 1)

        self._value = None

    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:
        """Forward pass with action masking."""
        obs = input_dict["obs"]

        # Handle dict observation - RLlib may nest it differently
        if isinstance(obs, dict):
            if "observations" in obs:
                flat_obs = obs["observations"]
            else:
                # Fallback: try to extract from nested structure
                flat_obs = obs.get("obs", obs)
            action_mask = obs.get("action_mask", {})
        else:
            flat_obs = obs
            action_mask = {}

        # Ensure flat_obs is a tensor
        if not isinstance(flat_obs, torch.Tensor):
            flat_obs = torch.tensor(flat_obs, dtype=torch.float32)

        # Extract features
        features = self.feature_extractor(flat_obs)

        # Get raw logits
        raw_logits = self.policy_head(features)

        # Apply masks
        batch_size = flat_obs.shape[0]
        masked_logits = self._apply_masks(raw_logits, action_mask, batch_size)

        # Value
        self._value = self.value_head(features).squeeze(-1)

        return masked_logits, state

    def _apply_masks(
        self,
        logits: TensorType,
        action_mask: Dict[str, TensorType],
        batch_size: int,
    ) -> TensorType:
        """Apply action masks to logits."""
        if not action_mask:
            return logits

        # Get masks
        source_mask = action_mask.get("source_mask")
        dest_mask = action_mask.get("dest_mask")
        troops_mask = action_mask.get("troops_mask")

        # Single action component sizes
        source_size = self.n_territories
        dest_size = self.n_territories
        troops_size = self.max_troops
        single_action_size = source_size + dest_size + troops_size

        # Reshape logits to [batch, action_budget, single_action_size]
        logits_reshaped = logits.view(batch_size, self.action_budget, single_action_size)

        # Apply masks to each action in budget
        for i in range(self.action_budget):
            action_logits = logits_reshaped[:, i, :]

            # Use -1e6 instead of -1e10 to avoid NaN/Inf gradient issues
            MASK_VALUE = -1e6

            # Source logits: [0:source_size]
            if source_mask is not None:
                mask = source_mask.float()
                action_logits[:, :source_size] = action_logits[:, :source_size] + (1 - mask) * MASK_VALUE

            # Dest logits: [source_size:source_size+dest_size]
            if dest_mask is not None:
                mask = dest_mask.float()
                action_logits[:, source_size:source_size+dest_size] = (
                    action_logits[:, source_size:source_size+dest_size] + (1 - mask) * MASK_VALUE
                )

            # Troops logits: [source_size+dest_size:]
            if troops_mask is not None:
                mask = troops_mask.float()
                action_logits[:, source_size+dest_size:] = (
                    action_logits[:, source_size+dest_size:] + (1 - mask) * MASK_VALUE
                )

        return logits_reshaped.view(batch_size, -1)

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        """Return value function output."""
        return self._value
