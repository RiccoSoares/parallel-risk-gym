"""
RLlib wrapper with action masks included in observations.

This wrapper extends the observation space to include action masks that can be
used by a custom model to ensure only valid actions are sampled during training.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
from gymnasium import spaces

from parallel_risk.training.rllib.wrapper import RLlibParallelRiskEnv


class ActionMaskRLlibEnv(RLlibParallelRiskEnv):
    """RLlib wrapper that includes action masks in observations.

    Adds action masks to enable the policy to only sample valid actions:
    - source_mask: [n_territories] - which territories are owned
    - dest_mask: [n_territories] - which destinations are valid (conservative)
    - troops_mask: [max_troops] - which troop counts are valid (conservative)

    The masks are "conservative" meaning they allow any action that could be
    valid for SOME source territory. True autoregressive masking happens
    at action decoding time in the custom model.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize action mask wrapper.

        Args:
            config: Configuration dict with keys:
                - max_troops: Maximum troops per action (default: 20)
                - Plus all keys from RLlibParallelRiskEnv
        """
        super().__init__(config)

        config = config or {}
        self.max_troops = config.get("max_troops", 20)

        # Cache for current raw observation
        self._current_raw_obs = {}

        # Update observation space to include masks
        self._observation_space = self._create_masked_observation_space()

    def _create_masked_observation_space(self) -> spaces.Dict:
        """Create observation space that includes action masks."""
        n_territories = self.env.map_config.n_territories

        # Original flattened observation size
        original_obs_space = super()._create_observation_space()
        obs_size = original_obs_space.shape[0]

        return spaces.Dict({
            "observations": spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
            ),
            "action_mask": spaces.Dict({
                "source_mask": spaces.Box(
                    low=0, high=1, shape=(n_territories,), dtype=np.float32
                ),
                "dest_mask": spaces.Box(
                    low=0, high=1, shape=(n_territories,), dtype=np.float32
                ),
                "troops_mask": spaces.Box(
                    low=0, high=1, shape=(self.max_troops,), dtype=np.float32
                ),
            }),
            # Include raw data needed for autoregressive masking at decode time
            "raw_data": spaces.Dict({
                "ownership": spaces.Box(
                    low=-1, high=1, shape=(n_territories,), dtype=np.float32
                ),
                "troops": spaces.Box(
                    low=0, high=np.inf, shape=(n_territories,), dtype=np.float32
                ),
                "adjacency": spaces.Box(
                    low=0, high=1, shape=(n_territories, n_territories), dtype=np.float32
                ),
                "income": spaces.Box(
                    low=0, high=np.inf, shape=(1,), dtype=np.float32
                ),
            }),
        })

    def _create_masked_observation(
        self, obs_dict: Dict[str, np.ndarray], agent: str
    ) -> Dict[str, Any]:
        """Create observation dict with action masks.

        Args:
            obs_dict: Raw observation dict from ParallelRiskEnv
            agent: Agent name

        Returns:
            Dict with observations, action_mask, and raw_data
        """
        n_territories = self.env.map_config.n_territories

        # Flatten observation
        flat_obs = self._flatten_observation(obs_dict)

        # Compute conservative action masks
        ownership = obs_dict['territory_ownership']
        troops = obs_dict['territory_troops']
        adjacency = obs_dict['adjacency_matrix']
        income = obs_dict['available_income'][0]

        # Source mask: owned territories
        source_mask = (ownership == 1).astype(np.float32)

        # Dest mask (conservative): owned territories + all neighbors of owned
        dest_mask = source_mask.copy()
        for i in range(n_territories):
            if source_mask[i]:
                dest_mask = np.maximum(dest_mask, adjacency[i].astype(np.float32))

        # Troops mask (conservative): valid for max possible action
        # Max troops available is max of:
        # - income (for deploy)
        # - max(troops at owned territories) - 1 (for transfer/attack)
        owned_troops = troops[ownership == 1]
        if len(owned_troops) > 0:
            max_transferable = int(owned_troops.max()) - 1
        else:
            max_transferable = 0
        max_deployable = int(income)
        max_troops_available = max(max_transferable, max_deployable)

        troops_mask = np.zeros(self.max_troops, dtype=np.float32)
        if max_troops_available > 0:
            troops_mask[1:min(max_troops_available + 1, self.max_troops)] = 1.0

        return {
            "observations": flat_obs,
            "action_mask": {
                "source_mask": source_mask,
                "dest_mask": dest_mask,
                "troops_mask": troops_mask,
            },
            "raw_data": {
                "ownership": ownership.astype(np.float32),
                "troops": troops.astype(np.float32),
                "adjacency": adjacency.astype(np.float32),
                "income": np.array([income], dtype=np.float32),
            },
        }

    def reset(self, *, seed=None, options=None):
        """Reset and create masked observations."""
        obs_dict, info_dict = self.env.reset(seed=seed, options=options)

        # Cache raw observations
        self._current_raw_obs = obs_dict.copy()

        # Create masked observations
        observations = {
            agent: self._create_masked_observation(obs, agent)
            for agent, obs in obs_dict.items()
        }

        return observations, info_dict

    def step(self, action_dict):
        """Step and create masked observations."""
        # Convert actions
        env_actions = {
            agent: self._unflatten_action(action)
            for agent, action in action_dict.items()
        }

        # Step environment
        obs_dict, rewards, terminateds, truncateds, infos = self.env.step(env_actions)

        # Cache raw observations
        self._current_raw_obs = obs_dict.copy()

        # Create masked observations
        observations = {
            agent: self._create_masked_observation(obs, agent)
            for agent, obs in obs_dict.items()
        }

        terminateds["__all__"] = all(terminateds.values())
        truncateds["__all__"] = all(truncateds.values())

        return observations, rewards, terminateds, truncateds, infos

    def get_raw_observation(self, agent: str) -> Optional[Dict]:
        """Get raw observation for an agent."""
        return self._current_raw_obs.get(agent)


def make_action_mask_env(config: Optional[Dict[str, Any]] = None):
    """Factory function for action mask environment.

    Args:
        config: Configuration dict

    Returns:
        ActionMaskRLlibEnv instance
    """
    return ActionMaskRLlibEnv(config)
