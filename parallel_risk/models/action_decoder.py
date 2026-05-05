"""
Action decoder for GNN policies.

Converts graph embeddings into Parallel Risk actions.

Challenge: Handle variable-sized graphs (different number of territories)
in a batched setting.

Action format: [source_territory, dest_territory, num_troops]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict
from torch_geometric.data import Batch


class ActionDecoder:
    """
    Decode graph embeddings into Parallel Risk actions.

    Handles the complexity of:
    1. Variable-sized graphs (different number of territories per graph)
    2. Fixed action budget (N actions per turn)
    3. Action format: [source, dest, troops]
    """

    def __init__(
        self,
        action_budget: int = 5,
        max_troops: int = 20,
        mask_source: bool = False,
        mask_dest: bool = False,
        mask_troops: bool = False,
    ):
        """
        Initialize action decoder.

        Args:
            action_budget: Number of actions to output per turn
            max_troops: Maximum troops per action
            mask_source: Enable source territory masking
            mask_dest: Enable destination territory masking
            mask_troops: Enable troops masking
        """
        self.action_budget = action_budget
        self.max_troops = max_troops
        self.mask_source = mask_source
        self.mask_dest = mask_dest
        self.mask_troops = mask_troops

    def decode_actions(
        self,
        action_logits: List[Dict[str, torch.Tensor]],
        batch: torch.Tensor,
        deterministic: bool = False,
        return_log_probs: bool = False,
        observations: List = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode actions from GCN policy logits.

        Args:
            action_logits: List of action dicts (length = action_budget)
                Each dict contains:
                - 'source': [num_nodes] per-node source scores
                - 'dest': [num_nodes] per-node dest scores
                - 'troops': [batch_size, max_troops] troop logits
            batch: [num_nodes] batch assignment for each node
            deterministic: If True, use argmax; if False, sample
            return_log_probs: If True, also return log probabilities
            observations: List of PyG Data observations (for masking)

        Returns:
            actions: [batch_size, action_budget, 3] sampled actions
            log_probs: [batch_size, action_budget] log probabilities (if requested)
        """
        batch_size = int(batch.max().item()) + 1
        device = batch.device

        all_actions = []
        all_log_probs = []

        for action_idx, logits_dict in enumerate(action_logits):
            source_scores = logits_dict['source']  # [num_nodes]
            dest_scores = logits_dict['dest']      # [num_nodes]
            troops_logits = logits_dict['troops']  # [batch_size, max_troops]

            # Sample/select actions for each graph in the batch
            batch_actions = []
            batch_log_probs = []

            for graph_idx in range(batch_size):
                # Get nodes belonging to this graph
                node_mask = (batch == graph_idx)
                graph_source_scores = source_scores[node_mask]  # [n_territories_i] - no clone
                graph_dest_scores = dest_scores[node_mask]      # [n_territories_i] - no clone
                graph_troops_logits = troops_logits[graph_idx]  # [max_troops] - no clone

                n_territories = graph_source_scores.size(0)

                # Apply action masking if enabled (avoid in-place ops for gradient stability)
                if self.mask_source and observations is not None:
                    source_mask = self._compute_source_mask(observations[graph_idx])
                    graph_source_scores = torch.where(
                        source_mask,
                        graph_source_scores,
                        torch.tensor(-1e10, device=graph_source_scores.device, dtype=graph_source_scores.dtype)
                    )

                if self.mask_dest and observations is not None:
                    dest_mask = self._compute_dest_mask(observations[graph_idx])
                    graph_dest_scores = torch.where(
                        dest_mask,
                        graph_dest_scores,
                        torch.tensor(-1e10, device=graph_dest_scores.device, dtype=graph_dest_scores.dtype)
                    )

                if self.mask_troops and observations is not None:
                    troops_mask = self._compute_troops_mask(observations[graph_idx])
                    graph_troops_logits = torch.where(
                        troops_mask,
                        graph_troops_logits,
                        torch.tensor(-1e10, device=graph_troops_logits.device, dtype=graph_troops_logits.dtype)
                    )

                # Sample source territory
                if deterministic:
                    source_idx = torch.argmax(graph_source_scores)
                else:
                    source_dist = torch.distributions.Categorical(logits=graph_source_scores)
                    source_idx = source_dist.sample()

                # Sample dest territory
                if deterministic:
                    dest_idx = torch.argmax(graph_dest_scores)
                else:
                    dest_dist = torch.distributions.Categorical(logits=graph_dest_scores)
                    dest_idx = dest_dist.sample()

                # Sample troops
                if deterministic:
                    troops_idx = torch.argmax(graph_troops_logits)
                else:
                    troops_dist = torch.distributions.Categorical(logits=graph_troops_logits)
                    troops_idx = troops_dist.sample()

                # Combine into action
                # FIX: Use torch.stack() instead of torch.tensor() to preserve gradient connections
                # torch.tensor() creates a new tensor disconnected from the computation graph,
                # causing the sampled indices to lose their gradient connection (critical for PPO)
                action = torch.stack([source_idx, dest_idx, troops_idx])
                batch_actions.append(action)

                # Compute log probability if requested
                if return_log_probs:
                    source_log_prob = F.log_softmax(graph_source_scores, dim=0)[source_idx]
                    dest_log_prob = F.log_softmax(graph_dest_scores, dim=0)[dest_idx]
                    troops_log_prob = F.log_softmax(graph_troops_logits, dim=0)[troops_idx]
                    total_log_prob = source_log_prob + dest_log_prob + troops_log_prob
                    batch_log_probs.append(total_log_prob)

            # Stack actions for this action slot
            actions_tensor = torch.stack(batch_actions)  # [batch_size, 3]
            all_actions.append(actions_tensor)

            if return_log_probs:
                log_probs_tensor = torch.stack(batch_log_probs)  # [batch_size]
                all_log_probs.append(log_probs_tensor)

        # Stack all actions
        actions = torch.stack(all_actions, dim=1)  # [batch_size, action_budget, 3]

        if return_log_probs:
            log_probs = torch.stack(all_log_probs, dim=1)  # [batch_size, action_budget]
            return actions, log_probs
        else:
            return actions, None

    def compute_log_probs(
        self,
        action_logits: List[Dict[str, torch.Tensor]],
        actions: torch.Tensor,
        batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute log probabilities for given actions.

        Useful for computing policy gradients.

        Args:
            action_logits: List of action dicts (length = action_budget)
            actions: [batch_size, action_budget, 3] actions to evaluate
            batch: [num_nodes] batch assignment for each node

        Returns:
            log_probs: [batch_size, action_budget] log probabilities
        """
        batch_size = actions.size(0)
        action_budget = actions.size(1)

        all_log_probs = []

        for action_idx in range(action_budget):
            logits_dict = action_logits[action_idx]
            source_scores = logits_dict['source']  # [num_nodes]
            dest_scores = logits_dict['dest']      # [num_nodes]
            troops_logits = logits_dict['troops']  # [batch_size, max_troops]

            batch_log_probs = []

            for graph_idx in range(batch_size):
                # Get nodes belonging to this graph
                node_mask = (batch == graph_idx)
                graph_source_scores = source_scores[node_mask]  # [n_territories_i]
                graph_dest_scores = dest_scores[node_mask]      # [n_territories_i]
                graph_troops_logits = troops_logits[graph_idx]  # [max_troops]

                # Extract action components
                action = actions[graph_idx, action_idx]  # [3]
                source_idx = action[0].long()
                dest_idx = action[1].long()
                troops_idx = action[2].long()

                # Compute log probabilities
                source_log_prob = F.log_softmax(graph_source_scores, dim=0)[source_idx]
                dest_log_prob = F.log_softmax(graph_dest_scores, dim=0)[dest_idx]
                troops_log_prob = F.log_softmax(graph_troops_logits, dim=0)[troops_idx]

                total_log_prob = source_log_prob + dest_log_prob + troops_log_prob
                batch_log_probs.append(total_log_prob)

            log_probs_tensor = torch.stack(batch_log_probs)  # [batch_size]
            all_log_probs.append(log_probs_tensor)

        log_probs = torch.stack(all_log_probs, dim=1)  # [batch_size, action_budget]
        return log_probs

    def compute_entropy(
        self,
        action_logits: List[Dict[str, torch.Tensor]],
        batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute entropy of action distributions.

        Higher entropy = more exploration.

        Args:
            action_logits: List of action dicts (length = action_budget)
            batch: [num_nodes] batch assignment for each node

        Returns:
            entropy: [batch_size, action_budget] entropy values
        """
        batch_size = int(batch.max().item()) + 1

        all_entropies = []

        for action_idx, logits_dict in enumerate(action_logits):
            source_scores = logits_dict['source']  # [num_nodes]
            dest_scores = logits_dict['dest']      # [num_nodes]
            troops_logits = logits_dict['troops']  # [batch_size, max_troops]

            batch_entropies = []

            for graph_idx in range(batch_size):
                # Get nodes belonging to this graph
                node_mask = (batch == graph_idx)
                graph_source_scores = source_scores[node_mask]  # [n_territories_i]
                graph_dest_scores = dest_scores[node_mask]      # [n_territories_i]
                graph_troops_logits = troops_logits[graph_idx]  # [max_troops]

                # Compute entropy for each component
                source_dist = torch.distributions.Categorical(logits=graph_source_scores)
                dest_dist = torch.distributions.Categorical(logits=graph_dest_scores)
                troops_dist = torch.distributions.Categorical(logits=graph_troops_logits)

                source_entropy = source_dist.entropy()
                dest_entropy = dest_dist.entropy()
                troops_entropy = troops_dist.entropy()

                # Sum entropies (independent distributions)
                total_entropy = source_entropy + dest_entropy + troops_entropy
                batch_entropies.append(total_entropy)

            entropies_tensor = torch.stack(batch_entropies)  # [batch_size]
            all_entropies.append(entropies_tensor)

        entropies = torch.stack(all_entropies, dim=1)  # [batch_size, action_budget]
        return entropies

    def _compute_source_mask(self, observation) -> torch.Tensor:
        """Compute source territory mask from observation.

        Args:
            observation: PyG Data object with node features

        Returns:
            Boolean mask [n_territories] where True = owned by agent
        """
        # Extract ownership from node features
        # Node features: [troops_norm, ownership, in_degree, region_one_hot...]
        # ownership is at index 1
        ownership = observation.x[:, 1]  # [n_territories]

        # ownership == 1 means owned by this agent
        return ownership == 1

    def _compute_dest_mask(self, observation) -> torch.Tensor:
        """Compute conservative destination territory mask.

        Conservative: Allow destinations that are owned or adjacent to any owned territory.

        Args:
            observation: PyG Data object

        Returns:
            Boolean mask [n_territories] where True = valid destination
        """
        n_territories = observation.num_nodes
        ownership = observation.x[:, 1]  # [n_territories]

        # Start with owned territories (valid for deploy)
        dest_mask = (ownership == 1)

        # Add territories adjacent to owned territories
        edge_index = observation.edge_index  # [2, num_edges]

        for territory in range(n_territories):
            if ownership[territory] == 1:
                # Find all neighbors of this owned territory
                neighbors = edge_index[1, edge_index[0] == territory]
                dest_mask[neighbors] = True

        return dest_mask

    def _compute_troops_mask(self, observation) -> torch.Tensor:
        """Compute conservative troops mask.

        Conservative: Safe for both deploy and transfer/attack actions.

        Args:
            observation: PyG Data object

        Returns:
            Boolean mask [max_troops] where True = valid troop count
        """
        # Extract ownership and troops from node features
        ownership = observation.x[:, 1]  # [n_territories]
        troops_norm = observation.x[:, 0]  # [n_territories], normalized
        troops = (troops_norm * 100).long()  # Denormalize

        # Get income from global features
        # global_features: [1, dim] where dim = [income_norm, turn_norm, region_control...]
        # After batching fix, global_features is always 2D
        gf = observation.global_features
        if gf.dim() == 2:
            income_norm = gf[0, 0]  # First element of first (only) row
        else:
            income_norm = gf[0]  # Fallback for 1D
        income = (income_norm * 20).long()  # Denormalize (max income ~20)

        # Owned territories
        owned_mask = (ownership == 1)
        owned_troops = troops[owned_mask]

        # Conservative max for transfers/attacks
        if owned_troops.numel() > 0:
            min_transferable = max(0, int(owned_troops.min().item()) - 1)
        else:
            min_transferable = 0

        # Safe maximum is the MAXIMUM of:
        # - Available income (for deploy actions)
        # - Minimum transferable troops (for transfer/attack actions)
        # We use max because any action uses EITHER deploy OR transfer, not both
        safe_max = max(int(income.item()), min_transferable)

        # Create mask
        mask = torch.zeros(self.max_troops, dtype=torch.bool, device=observation.x.device)
        if safe_max > 0:
            mask[1:min(safe_max + 1, self.max_troops)] = True

        return mask


def convert_to_env_format(actions: torch.Tensor) -> Dict[str, any]:
    """
    Convert batched actions to environment format.

    Args:
        actions: [batch_size, action_budget, 3] tensor

    Returns:
        env_actions: Dict suitable for ParallelRiskEnv.step()
            For RLlib wrapper format (fixed budget)
    """
    batch_size, action_budget, _ = actions.shape

    # For single graph (batch_size = 1), return single agent format
    if batch_size == 1:
        actions_np = actions[0].cpu().numpy()  # [action_budget, 3]
        return tuple(actions_np)  # RLlib expects tuple of actions

    # For batched graphs, return list
    env_actions = []
    for i in range(batch_size):
        actions_np = actions[i].cpu().numpy()  # [action_budget, 3]
        env_actions.append(tuple(actions_np))

    return env_actions
