"""
RLlib wrapper with autoregressive action masking support.

Extends RLlibParallelRiskEnv to apply action masks during action sampling.
Uses the same autoregressive masking logic as Phase 2 (GNN/TorchRL).
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
from gymnasium import spaces

from parallel_risk.training.rllib.wrapper import RLlibParallelRiskEnv


class MaskedRLlibParallelRiskEnv(RLlibParallelRiskEnv):
    """RLlib wrapper with autoregressive action masking.

    Applies masks to ensure valid actions using autoregressive logic:
    1. Source: Only owned territories
    2. Dest (conditioned on source): Source itself (deploy) or adjacent (transfer/attack)
    3. Troops (conditioned on source + dest): Income (deploy) or source troops - 1 (transfer)

    This matches the masking logic in Phase 2's ActionDecoder for fair comparison.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize masked wrapper.

        Args:
            config: Configuration dict with additional keys:
                - enable_masking: Enable autoregressive masking (default: True)
                - max_troops: Maximum troops per action (default: 20)
        """
        super().__init__(config)

        config = config or {}
        self.enable_masking = config.get("enable_masking", True)
        self.max_troops = config.get("max_troops", 20)

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

    def get_raw_observation(self, agent: str) -> Optional[Dict]:
        """Get raw (unflattened) observation for an agent.

        Args:
            agent: Agent name (e.g., "agent_0")

        Returns:
            Raw observation dict or None if not available
        """
        return self._current_obs.get(agent)

    def sample_masked_action(self, agent: str) -> Tuple:
        """Sample a masked action using autoregressive logic.

        Masking order:
        1. Source: only owned territories
        2. Dest: source itself (deploy) or adjacent territories (transfer/attack)
        3. Troops: income (deploy) or source troops - 1 (transfer/attack)

        Args:
            agent: Agent name (e.g., "agent_0")

        Returns:
            Tuple of actions (compatible with RLlib action space)
        """
        if agent not in self._current_obs:
            raise ValueError(f"No observation cached for {agent}. Call reset() first.")

        obs = self._current_obs[agent]

        if not self.enable_masking:
            # No masking - random actions
            return self._sample_random_action()

        # Sample actions with autoregressive masking
        actions = []
        for _ in range(self.action_budget):
            # Step 1: Sample source (ownership mask)
            source_mask = self._compute_source_mask(obs)
            valid_sources = np.where(source_mask)[0]

            if len(valid_sources) == 0:
                # No owned territories - use 0 as fallback
                source = 0
            else:
                source = np.random.choice(valid_sources)

            # Step 2: Sample dest (conditioned on source)
            dest_mask = self._compute_dest_mask_for_source(obs, source)
            valid_dests = np.where(dest_mask)[0]

            if len(valid_dests) == 0:
                # No valid destinations - use source as fallback (deploy)
                dest = source
            else:
                dest = np.random.choice(valid_dests)

            # Step 3: Sample troops (conditioned on source + dest)
            troops_mask = self._compute_troops_mask_for_action(obs, source, dest)
            valid_troops = np.where(troops_mask)[0]

            if len(valid_troops) == 0:
                # No valid troop counts - use 1 (minimum)
                troops = 1
            else:
                troops = np.random.choice(valid_troops)

            actions.append((source, dest, troops))

        return tuple(actions)

    def sample_masked_action_raw(self, agent: str) -> Dict:
        """Sample a masked action in raw ParallelRiskEnv format.

        Args:
            agent: Agent name (e.g., "agent_0")

        Returns:
            Action dict with 'num_actions' and 'actions' array
        """
        actions_tuple = self.sample_masked_action(agent)
        actions_array = np.zeros((10, 3), dtype=np.int32)
        for i, (src, dst, troops) in enumerate(actions_tuple):
            actions_array[i] = [src, dst, troops]

        return {
            'num_actions': self.action_budget,
            'actions': actions_array
        }

    def _sample_random_action(self) -> Tuple:
        """Sample completely random action (no masking)."""
        actions = []
        n_territories = self.env.map_config.n_territories
        for _ in range(self.action_budget):
            source = np.random.randint(0, n_territories)
            dest = np.random.randint(0, n_territories)
            troops = np.random.randint(1, self.max_troops)
            actions.append((source, dest, troops))
        return tuple(actions)

    def _compute_source_mask(self, obs: Dict) -> np.ndarray:
        """Compute mask for source territories (owned only).

        Args:
            obs: Observation dict

        Returns:
            Boolean mask [n_territories] where True = valid source
        """
        ownership = obs['territory_ownership']
        return ownership == 1

    def _compute_dest_mask_for_source(self, obs: Dict, source_idx: int) -> np.ndarray:
        """Compute destination mask conditioned on chosen source.

        Valid destinations are:
        - The source itself (for deploy actions)
        - Territories adjacent to the source (for transfer/attack)

        Args:
            obs: Observation dict
            source_idx: Index of the chosen source territory

        Returns:
            Boolean mask [n_territories] where True = valid destination
        """
        adjacency = obs['adjacency_matrix']
        n_territories = adjacency.shape[0]

        dest_mask = np.zeros(n_territories, dtype=bool)

        # Deploy: source == dest
        dest_mask[source_idx] = True

        # Transfer/Attack: adjacent territories
        neighbors = np.where(adjacency[source_idx] == 1)[0]
        dest_mask[neighbors] = True

        return dest_mask

    def _compute_troops_mask_for_action(
        self, obs: Dict, source_idx: int, dest_idx: int
    ) -> np.ndarray:
        """Compute troops mask conditioned on source and destination.

        Args:
            obs: Observation dict
            source_idx: Index of the chosen source territory
            dest_idx: Index of the chosen destination territory

        Returns:
            Boolean mask [max_troops] where True = valid troop count
        """
        troops = obs['territory_troops']
        income = int(obs['available_income'][0])

        if source_idx == dest_idx:
            # Deploy action: limited by income
            max_troops_available = income
        else:
            # Transfer/Attack: limited by source troops (must leave 1)
            max_troops_available = max(0, int(troops[source_idx]) - 1)

        # Create mask: troops from 1 to max_troops_available
        mask = np.zeros(self.max_troops, dtype=bool)
        if max_troops_available > 0:
            mask[1:min(max_troops_available + 1, self.max_troops)] = True

        return mask

    # Legacy methods for backwards compatibility
    def _compute_dest_mask(self, obs: Dict, agent_idx: int, source_mask: np.ndarray) -> np.ndarray:
        """Compute conservative mask for destination territories (legacy).

        DEPRECATED: Use _compute_dest_mask_for_source for autoregressive masking.

        Conservative approach: Allow destinations that are:
        1. Owned by agent (deploy actions)
        2. Adjacent to ANY owned territory
        """
        adjacency = obs['adjacency_matrix']
        n_territories = adjacency.shape[0]

        # Destinations owned by agent are always valid (deploy)
        dest_mask = source_mask.copy()

        # Add territories adjacent to ANY owned territory
        for territory in range(n_territories):
            if source_mask[territory]:
                dest_mask |= (adjacency[territory] == 1)

        return dest_mask

    def _compute_troops_mask(self, obs: Dict, agent: str) -> np.ndarray:
        """Compute conservative mask for troop counts (legacy).

        DEPRECATED: Use _compute_troops_mask_for_action for autoregressive masking.

        Conservative approach: Allow troops that are safe for ANY valid action.
        """
        ownership = obs['territory_ownership']
        troops = obs['territory_troops']
        income = obs['available_income'][0]

        owned_mask = (ownership == 1)
        owned_troops = troops[owned_mask]

        if len(owned_troops) > 0:
            min_transferable = max(0, owned_troops.min() - 1)
        else:
            min_transferable = 0

        safe_max = max(income, min_transferable)

        mask = np.zeros(self.max_troops, dtype=bool)
        if safe_max > 0:
            mask[1:min(int(safe_max) + 1, self.max_troops)] = True

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
            return 0
        return np.random.choice(valid_indices)


def make_masked_rllib_env(config: Optional[Dict[str, Any]] = None):
    """Factory function for masked RLlib environment.

    Args:
        config: Configuration dict with masking options:
            - enable_masking: Enable autoregressive masking (default: True)
            - max_troops: Maximum troops per action (default: 20)

    Returns:
        MaskedRLlibParallelRiskEnv instance
    """
    return MaskedRLlibParallelRiskEnv(config)
