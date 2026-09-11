"""Masked random agent baseline for Parallel Risk.

Uses the same autoregressive masking logic as the GNN agent for fair comparison.

Two variants:
- MaskedRandomAgent: For Phase 2 (TorchRL/GNN) with PyG graph observations
- MaskedRandomAgentRLlib: For Phase 1 (RLlib) with flat dict observations
"""

from typing import Dict, Tuple, Union
import numpy as np
import torch


class MaskedRandomAgent:
    """
    Random agent with autoregressive action masking.

    Generates random but VALID actions using the same masking logic
    as the GNN policy's ActionDecoder. This provides a fair baseline
    for evaluating learned policies.

    Masking order (autoregressive):
    1. Source: only owned territories
    2. Dest: source itself (deploy) or adjacent territories (transfer/attack)
    3. Troops: based on income (deploy) or available troops (transfer/attack)
    """

    def __init__(self, action_budget: int = 5, max_troops: int = 20):
        """
        Initialize masked random agent.

        Args:
            action_budget: Number of actions per turn
            max_troops: Maximum troops per action
        """
        self.action_budget = action_budget
        self.max_troops = max_troops

    def get_action(self, observation) -> dict:
        """
        Generate random but valid action using autoregressive masking.

        Args:
            observation: PyG Data object with node features, edge_index, global_features

        Returns:
            Action dict with 'num_actions' and 'actions' array
        """
        n_territories = observation.num_nodes
        actions = np.zeros((10, 3), dtype=np.int32)  # Padded to 10

        for action_idx in range(self.action_budget):
            # Step 1: Sample source (ownership mask)
            source_mask = self._compute_source_mask(observation)
            valid_sources = torch.where(source_mask)[0].numpy()

            if len(valid_sources) == 0:
                # No owned territories - shouldn't happen but handle gracefully
                break

            source_idx = np.random.choice(valid_sources)

            # Step 2: Sample dest (conditioned on source)
            dest_mask = self._compute_dest_mask_for_source(observation, source_idx)
            valid_dests = torch.where(dest_mask)[0].numpy()

            if len(valid_dests) == 0:
                # No valid destinations - use source as fallback (deploy)
                dest_idx = source_idx
            else:
                dest_idx = np.random.choice(valid_dests)

            # Step 3: Sample troops (conditioned on source + dest)
            troops_mask = self._compute_troops_mask_for_action(observation, source_idx, dest_idx)
            valid_troops = torch.where(troops_mask)[0].numpy()

            if len(valid_troops) == 0:
                # No valid troop counts - use 0 (no-op)
                troops_idx = 0
            else:
                troops_idx = np.random.choice(valid_troops)

            actions[action_idx] = [source_idx, dest_idx, troops_idx]

        return {
            'num_actions': self.action_budget,
            'actions': actions
        }

    def _compute_source_mask(self, observation) -> torch.Tensor:
        """Compute source territory mask (owned territories only)."""
        ownership = observation.x[:, 1]
        return ownership == 1

    def _compute_dest_mask_for_source(self, observation, source_idx: int) -> torch.Tensor:
        """Compute destination mask conditioned on chosen source."""
        n_territories = observation.num_nodes
        edge_index = observation.edge_index

        dest_mask = torch.zeros(n_territories, dtype=torch.bool)

        # Deploy: source == dest
        dest_mask[source_idx] = True

        # Transfer/Attack: adjacent territories
        neighbors = edge_index[1, edge_index[0] == source_idx]
        dest_mask[neighbors] = True

        return dest_mask

    def _compute_troops_mask_for_action(self, observation, source_idx: int, dest_idx: int) -> torch.Tensor:
        """Compute troops mask conditioned on source and destination."""
        # Denormalize troops from log-scaled features
        troops_norm = observation.x[:, 0]
        troops = (torch.exp(troops_norm * torch.log1p(torch.tensor(100.0))) - 1).long()

        # Get income from global features
        gf = observation.global_features
        if gf.dim() == 2:
            income_norm = gf[0, 0]
        else:
            income_norm = gf[0]
        income = (income_norm * 20).long()

        if source_idx == dest_idx:
            # Deploy action: limited by income
            max_troops_available = int(income.item())
        else:
            # Transfer/Attack: limited by source troops (must leave 1)
            max_troops_available = max(0, int(troops[source_idx].item()) - 1)

        # Create mask
        mask = torch.zeros(self.max_troops, dtype=torch.bool)
        if max_troops_available > 0:
            mask[1:min(max_troops_available + 1, self.max_troops)] = True

        return mask


class MaskedRandomAgentRLlib:
    """
    Random agent with autoregressive action masking for RLlib environments.

    Generates random but VALID actions using the same masking logic
    as the GNN policy's ActionDecoder. This provides a fair baseline
    for evaluating learned policies in Phase 1 (RLlib).

    Masking order (autoregressive):
    1. Source: only owned territories
    2. Dest: source itself (deploy) or adjacent territories (transfer/attack)
    3. Troops: based on income (deploy) or available troops (transfer/attack)

    Works with flat dict observations from RLlibParallelRiskEnv.

    Performance note (this is the hot path of MCTS rollouts): the masks are
    never materialised. Each draw uses the identity
    ``np.random.choice(a) == a[np.random.randint(len(a))]`` for a 1-D array
    ``a`` (both consume one bounded draw from the global np.random stream),
    so the actions AND the post-call RNG state are identical to the original
    mask-based loop, which is kept in _masked_random_legacy.py and checked by
    tests/test_sampler_parity.py.

    Destination lists are precomputed from the adjacency matrix given to the
    constructor. The original read ``observation['adjacency_matrix']``; the
    env and RiskSimulator always emit a copy of the map's matrix and every
    in-repo construction (from_env, MCTSAgent, MCTSGNNAgent) passes that same
    map's matrix, so the two are equal by construction.
    """

    def __init__(
        self,
        n_territories: int,
        adjacency_matrix: np.ndarray,
        action_budget: int = 5,
        max_troops: int = 20,
        max_actions_per_turn: int = None,
    ):
        """
        Initialize masked random agent for RLlib.

        Args:
            n_territories: Number of territories in the map
            adjacency_matrix: [n_territories, n_territories] adjacency matrix
            action_budget: Number of actions per turn
            max_troops: Maximum troops per action
            max_actions_per_turn: Env-side padding target for get_action_raw's
                output array. Must be >= action_budget. Default max(10,
                action_budget) matches the env's default while scaling up
                automatically for K > 10.
        """
        self.n_territories = n_territories
        self.adjacency_matrix = adjacency_matrix
        self.action_budget = action_budget
        self.max_troops = max_troops
        self.max_actions_per_turn = (
            max_actions_per_turn if max_actions_per_turn is not None
            else max(10, action_budget)
        )
        # Valid destinations per source, ascending: the source itself (deploy)
        # plus its neighbours. Equals np.where(dest_mask)[0] of the legacy code.
        self._dests_by_source = [
            np.array(sorted({src} | set(np.flatnonzero(adjacency_matrix[src] == 1).tolist())),
                     dtype=np.int64)
            for src in range(n_territories)
        ]
        # Troop counts 0..max_troops-1; indexing it yields the same np.int64
        # scalars the legacy np.where(troops_mask)[0] produced.
        self._troop_values = np.arange(max_troops, dtype=np.int64)

    def get_action(self, observation: Dict[str, np.ndarray]) -> Tuple:
        """
        Generate random but valid action using autoregressive masking.

        Args:
            observation: Dict with keys:
                - 'territory_ownership': [n_territories] array, 1=owned, -1=enemy
                - 'territory_troops': [n_territories] array of troop counts
                - 'adjacency_matrix': [n_territories, n_territories] adjacency
                - 'available_income': [1] array with income value

        Returns:
            Tuple of action_budget actions, each (source, dest, troops)
            Compatible with RLlib action space.
        """
        return tuple(self._sample_actions(observation))

    def get_action_raw(self, observation: Dict[str, np.ndarray]) -> Dict:
        """
        Generate random but valid action in raw ParallelRiskEnv format.

        Args:
            observation: Dict observation from environment

        Returns:
            Action dict with 'num_actions' and 'actions' array
        """
        actions = self._sample_actions(observation)
        actions_array = np.zeros((self.max_actions_per_turn, 3), dtype=np.int32)
        if actions:
            actions_array[:len(actions)] = actions

        return {
            'num_actions': self.action_budget,
            'actions': actions_array
        }

    def _sample_actions(self, observation: Dict[str, np.ndarray]) -> list:
        """action_budget (source, dest, troops) triples, one bounded np.random draw per choice.

        Masking order and fallbacks match the legacy loop exactly:
        1. Source: uniform over owned territories; 0 if none are owned.
        2. Dest: uniform over {source} + neighbours (never empty).
        3. Troops: uniform over 1..min(available, max_troops - 1); 1 if that
           range is empty. available = income for a deploy (source == dest),
           else source troops - 1.
        """
        valid_sources = np.flatnonzero(observation['territory_ownership'] == 1)
        n_sources = len(valid_sources)
        troops = observation['territory_troops'].tolist()
        income = int(observation['available_income'][0])
        dests_by_source = self._dests_by_source
        troop_values = self._troop_values
        max_troop_idx = self.max_troops - 1
        randint = np.random.randint

        actions = []
        for _ in range(self.action_budget):
            source_idx = valid_sources[randint(n_sources)] if n_sources else 0

            dests = dests_by_source[source_idx]
            dest_idx = dests[randint(len(dests))]

            if source_idx == dest_idx:
                max_troops_available = income
            else:
                max_troops_available = max(0, troops[source_idx] - 1)
            n_valid_troops = min(max_troops_available, max_troop_idx)
            if n_valid_troops > 0:
                troops_idx = troop_values[1 + randint(n_valid_troops)]
            else:
                troops_idx = 1

            actions.append((source_idx, dest_idx, troops_idx))

        return actions

    @classmethod
    def from_env(cls, env, action_budget: int = 5, max_troops: int = 20):
        """
        Create a MaskedRandomAgentRLlib from an environment.

        Args:
            env: RLlibParallelRiskEnv or ParallelRiskEnv instance
            action_budget: Number of actions per turn
            max_troops: Maximum troops per action

        Returns:
            MaskedRandomAgentRLlib instance
        """
        # Handle both RLlib wrapper and raw env
        if hasattr(env, 'env'):
            # RLlib wrapper
            base_env = env.env
        else:
            base_env = env

        n_territories = base_env.map_config.n_territories
        adjacency_matrix = base_env.map_config.adjacency_matrix

        return cls(
            n_territories=n_territories,
            adjacency_matrix=adjacency_matrix,
            action_budget=action_budget,
            max_troops=max_troops,
        )
