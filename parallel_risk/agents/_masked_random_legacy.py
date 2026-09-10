"""
Legacy MaskedRandomAgentRLlib implementation.

Kept only as the reference for tests/test_sampler_parity.py, which checks that
the fast sampler in masked_random_agent.py returns the same actions AND leaves
np.random in the same state. Mirrors the _action_decoder_legacy.py pattern.
Delete once the fast sampler has been validated in a full experiment.
"""

from typing import Dict, Tuple
import numpy as np


class _MaskedRandomAgentRLlibLegacy:
    """Original MaskedRandomAgentRLlib (three np.random.choice draws per slot).

    Preserved verbatim from commit e6991e7 so RNG-stream parity can be verified
    against the replacement.
    """

    def __init__(
        self,
        n_territories: int,
        adjacency_matrix: np.ndarray,
        action_budget: int = 5,
        max_troops: int = 20,
        max_actions_per_turn: int = None,
    ):
        self.n_territories = n_territories
        self.adjacency_matrix = adjacency_matrix
        self.action_budget = action_budget
        self.max_troops = max_troops
        self.max_actions_per_turn = (
            max_actions_per_turn if max_actions_per_turn is not None
            else max(10, action_budget)
        )

    def get_action(self, observation: Dict[str, np.ndarray]) -> Tuple:
        actions = []

        for _ in range(self.action_budget):
            # Step 1: Sample source (ownership mask)
            source_mask = self._compute_source_mask(observation)
            valid_sources = np.where(source_mask)[0]

            if len(valid_sources) == 0:
                # No owned territories - use 0 as fallback
                source_idx = 0
            else:
                source_idx = np.random.choice(valid_sources)

            # Step 2: Sample dest (conditioned on source)
            dest_mask = self._compute_dest_mask_for_source(observation, source_idx)
            valid_dests = np.where(dest_mask)[0]

            if len(valid_dests) == 0:
                # No valid destinations - use source as fallback (deploy)
                dest_idx = source_idx
            else:
                dest_idx = np.random.choice(valid_dests)

            # Step 3: Sample troops (conditioned on source + dest)
            troops_mask = self._compute_troops_mask_for_action(observation, source_idx, dest_idx)
            valid_troops = np.where(troops_mask)[0]

            if len(valid_troops) == 0:
                # No valid troop counts - use 1 (minimum)
                troops_idx = 1
            else:
                troops_idx = np.random.choice(valid_troops)

            actions.append((source_idx, dest_idx, troops_idx))

        return tuple(actions)

    def get_action_raw(self, observation: Dict[str, np.ndarray]) -> Dict:
        actions_tuple = self.get_action(observation)
        actions_array = np.zeros((self.max_actions_per_turn, 3), dtype=np.int32)
        for i, (src, dst, troops) in enumerate(actions_tuple):
            actions_array[i] = [src, dst, troops]

        return {
            'num_actions': self.action_budget,
            'actions': actions_array
        }

    def _compute_source_mask(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        ownership = observation['territory_ownership']
        return ownership == 1

    def _compute_dest_mask_for_source(
        self, observation: Dict[str, np.ndarray], source_idx: int
    ) -> np.ndarray:
        # Use adjacency from observation if available, otherwise use stored
        if 'adjacency_matrix' in observation:
            adjacency = observation['adjacency_matrix']
        else:
            adjacency = self.adjacency_matrix

        dest_mask = np.zeros(self.n_territories, dtype=bool)

        # Deploy: source == dest
        dest_mask[source_idx] = True

        # Transfer/Attack: adjacent territories
        neighbors = np.where(adjacency[source_idx] == 1)[0]
        dest_mask[neighbors] = True

        return dest_mask

    def _compute_troops_mask_for_action(
        self, observation: Dict[str, np.ndarray], source_idx: int, dest_idx: int
    ) -> np.ndarray:
        troops = observation['territory_troops']
        income = int(observation['available_income'][0])

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

    @classmethod
    def from_env(cls, env, action_budget: int = 5, max_troops: int = 20):
        base_env = env.env if hasattr(env, 'env') else env
        return cls(
            n_territories=base_env.map_config.n_territories,
            adjacency_matrix=base_env.map_config.adjacency_matrix,
            action_budget=action_budget,
            max_troops=max_troops,
        )
