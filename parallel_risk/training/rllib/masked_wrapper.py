"""
RLlib wrapper with action masking support.

Extends RLlibParallelRiskEnv to apply action masks during action sampling.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
from gymnasium import spaces

from parallel_risk.training.rllib.wrapper import RLlibParallelRiskEnv


class MaskedRLlibParallelRiskEnv(RLlibParallelRiskEnv):
    """RLlib wrapper with action masking.

    Applies masks to prevent invalid actions:
    - Source masking: Only owned territories
    - Destination masking: Adjacent to owned territories (conservative)
    - Troops masking: Within available bounds (conservative)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize masked wrapper.

        Args:
            config: Configuration dict with additional keys:
                - mask_source: Enable source masking (default: True)
                - mask_dest: Enable destination masking (default: False)
                - mask_troops: Enable troops masking (default: False)
        """
        super().__init__(config)

        config = config or {}
        self.mask_source = config.get("mask_source", True)
        self.mask_dest = config.get("mask_dest", False)
        self.mask_troops = config.get("mask_troops", False)

        # Cache for current observation (needed for masked sampling)
        self._current_obs = {}

    def reset(self, *, seed=None, options=None):
        """Reset and cache observations for masking."""
        obs_dict, info_dict = self.env.reset(seed=seed, options=options)

        # Cache original observations for mask computation
        self._current_obs = obs_dict.copy()

        # Flatten observations for RLlib
        observations = {
            agent: self._flatten_observation(obs)
            for agent, obs in obs_dict.items()
        }

        return observations, info_dict

    def step(self, action_dict):
        """Step and update cached observations."""
        # Convert actions
        env_actions = {
            agent: self._unflatten_action(action)
            for agent, action in action_dict.items()
        }

        # Step environment
        obs_dict, rewards, terminateds, truncateds, infos = self.env.step(env_actions)

        # Update cached observations
        self._current_obs = obs_dict.copy()

        # Flatten observations
        observations = {
            agent: self._flatten_observation(obs)
            for agent, obs in obs_dict.items()
        }

        terminateds["__all__"] = all(terminateds.values())
        truncateds["__all__"] = all(truncateds.values())

        return observations, rewards, terminateds, truncateds, infos

    def sample_masked_action(self, agent: str) -> Tuple:
        """Sample a masked action for a specific agent.

        Args:
            agent: Agent name (e.g., "agent_0")

        Returns:
            Tuple of actions (compatible with RLlib action space)
        """
        if agent not in self._current_obs:
            raise ValueError(f"No observation cached for {agent}. Call reset() first.")

        obs = self._current_obs[agent]
        agent_idx = self.env.possible_agents.index(agent)

        # Compute masks
        source_mask = self._compute_source_mask(obs, agent_idx)
        dest_mask = self._compute_dest_mask(obs, agent_idx, source_mask)
        troops_mask = self._compute_troops_mask(obs, agent)

        # Sample actions with masking
        actions = []
        for _ in range(self.action_budget):
            # Sample source
            if self.mask_source:
                source = self._sample_from_mask(source_mask)
            else:
                source = np.random.randint(0, self.env.map_config.n_territories)

            # Sample dest
            if self.mask_dest:
                dest = self._sample_from_mask(dest_mask)
            else:
                dest = np.random.randint(0, self.env.map_config.n_territories)

            # Sample troops
            if self.mask_troops:
                troops = self._sample_from_mask(troops_mask)
            else:
                troops = np.random.randint(0, 20)

            actions.append((source, dest, troops))

        return tuple(actions)

    def _compute_source_mask(self, obs: Dict, agent_idx: int) -> np.ndarray:
        """Compute mask for source territories.

        Args:
            obs: Observation dict
            agent_idx: Agent index (0 or 1)

        Returns:
            Boolean mask [n_territories] where True = valid source
        """
        # Ownership is agent-relative: 1 = owned by this agent
        ownership = obs['territory_ownership']
        return ownership == 1

    def _compute_dest_mask(self, obs: Dict, agent_idx: int, source_mask: np.ndarray) -> np.ndarray:
        """Compute conservative mask for destination territories.

        Conservative approach: Allow destinations that are:
        1. Owned by agent (deploy actions)
        2. Adjacent to ANY owned territory

        Args:
            obs: Observation dict
            agent_idx: Agent index
            source_mask: Source mask (which territories are owned)

        Returns:
            Boolean mask [n_territories] where True = valid destination
        """
        adjacency = obs['adjacency_matrix']
        n_territories = adjacency.shape[0]

        # Destinations owned by agent are always valid (deploy)
        dest_mask = source_mask.copy()

        # Add territories adjacent to ANY owned territory
        for territory in range(n_territories):
            if source_mask[territory]:
                # This territory is owned, add all adjacent territories
                dest_mask |= (adjacency[territory] == 1)

        return dest_mask

    def _compute_troops_mask(self, obs: Dict, agent: str) -> np.ndarray:
        """Compute conservative mask for troop counts.

        Conservative approach: Allow troops that are safe for ANY valid action.
        - Deploy: Limited by income
        - Transfer/attack: Limited by minimum troops across owned territories

        Args:
            obs: Observation dict
            agent: Agent name

        Returns:
            Boolean mask [20] where True = valid troop count
        """
        ownership = obs['territory_ownership']
        troops = obs['territory_troops']
        income = obs['available_income'][0]

        # Owned territories
        owned_mask = (ownership == 1)
        owned_troops = troops[owned_mask]

        # Conservative max for transfers/attacks (must leave 1)
        if len(owned_troops) > 0:
            min_transferable = max(0, owned_troops.min() - 1)
        else:
            min_transferable = 0

        # Safe maximum is the MAXIMUM of:
        # - Available income (for deploy actions)
        # - Minimum transferable troops (for transfer/attack actions)
        # We use max because any action uses EITHER deploy OR transfer, not both
        safe_max = max(income, min_transferable)

        # Create mask: troops from 1 to safe_max are valid
        mask = np.zeros(20, dtype=bool)
        if safe_max > 0:
            mask[1:min(safe_max + 1, 20)] = True

        return mask

    def _sample_from_mask(self, mask: np.ndarray) -> int:
        """Sample uniformly from masked indices.

        Args:
            mask: Boolean mask

        Returns:
            Sampled index
        """
        valid_indices = np.where(mask)[0]
        if len(valid_indices) == 0:
            # Fallback: if no valid options, return 0
            return 0
        return np.random.choice(valid_indices)


def make_masked_rllib_env(config: Optional[Dict[str, Any]] = None):
    """Factory function for masked RLlib environment.

    Args:
        config: Configuration dict with masking options:
            - mask_source: Enable source masking (default: True)
            - mask_dest: Enable destination masking (default: False)
            - mask_troops: Enable troops masking (default: False)

    Returns:
        MaskedRLlibParallelRiskEnv instance
    """
    return MaskedRLlibParallelRiskEnv(config)
