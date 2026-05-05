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

    Uses autoregressive masking: source → dest|source → troops|source,dest
    This achieves >95% valid actions by conditioning each choice on previous ones.
    """

    def __init__(
        self,
        action_budget: int = 5,
        max_troops: int = 20,
    ):
        """
        Initialize action decoder.

        Args:
            action_budget: Number of actions to output per turn
            max_troops: Maximum troops per action
        """
        self.action_budget = action_budget
        self.max_troops = max_troops

    def decode_actions(
        self,
        action_logits: List[Dict[str, torch.Tensor]],
        batch: torch.Tensor,
        deterministic: bool = False,
        return_log_probs: bool = False,
        observations: List = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode actions from GCN policy logits using autoregressive masking.

        Sampling order: source → dest|source → troops|source,dest
        Each component is conditioned on previous choices for valid actions.

        Args:
            action_logits: List of action dicts (length = action_budget)
                Each dict contains:
                - 'source': [num_nodes] per-node source scores
                - 'dest': [num_nodes] per-node dest scores
                - 'troops': [batch_size, max_troops] troop logits
            batch: [num_nodes] batch assignment for each node
            deterministic: If True, use argmax; if False, sample
            return_log_probs: If True, also return log probabilities
            observations: List of PyG Data observations (required for masking)

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
                graph_source_scores = source_scores[node_mask]  # [n_territories_i]
                graph_dest_scores = dest_scores[node_mask]      # [n_territories_i]
                graph_troops_logits = troops_logits[graph_idx]  # [max_troops]

                obs = observations[graph_idx] if observations is not None else None

                # Step 1: Sample source (ownership mask)
                if obs is not None:
                    source_mask = self._compute_source_mask(obs)
                    masked_source_scores = torch.where(
                        source_mask,
                        graph_source_scores,
                        torch.tensor(-1e10, device=graph_source_scores.device, dtype=graph_source_scores.dtype)
                    )
                else:
                    masked_source_scores = graph_source_scores

                if deterministic:
                    source_idx = torch.argmax(masked_source_scores)
                else:
                    source_dist = torch.distributions.Categorical(logits=masked_source_scores)
                    source_idx = source_dist.sample()

                # Step 2: Sample dest (conditioned on source)
                if obs is not None:
                    dest_mask = self._compute_dest_mask_for_source(obs, source_idx.item())
                    masked_dest_scores = torch.where(
                        dest_mask,
                        graph_dest_scores,
                        torch.tensor(-1e10, device=graph_dest_scores.device, dtype=graph_dest_scores.dtype)
                    )
                else:
                    masked_dest_scores = graph_dest_scores

                if deterministic:
                    dest_idx = torch.argmax(masked_dest_scores)
                else:
                    dest_dist = torch.distributions.Categorical(logits=masked_dest_scores)
                    dest_idx = dest_dist.sample()

                # Step 3: Sample troops (conditioned on source + dest)
                if obs is not None:
                    troops_mask = self._compute_troops_mask_for_action(obs, source_idx.item(), dest_idx.item())
                    masked_troops_logits = torch.where(
                        troops_mask,
                        graph_troops_logits,
                        torch.tensor(-1e10, device=graph_troops_logits.device, dtype=graph_troops_logits.dtype)
                    )
                else:
                    masked_troops_logits = graph_troops_logits

                if deterministic:
                    troops_idx = torch.argmax(masked_troops_logits)
                else:
                    troops_dist = torch.distributions.Categorical(logits=masked_troops_logits)
                    troops_idx = troops_dist.sample()

                # Combine into action
                action = torch.stack([source_idx, dest_idx, troops_idx])
                batch_actions.append(action)

                # Compute log probability if requested
                if return_log_probs:
                    source_log_prob = F.log_softmax(masked_source_scores, dim=0)[source_idx]
                    dest_log_prob = F.log_softmax(masked_dest_scores, dim=0)[dest_idx]
                    troops_log_prob = F.log_softmax(masked_troops_logits, dim=0)[troops_idx]
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
        batch: torch.Tensor,
        observations: List = None,
    ) -> torch.Tensor:
        """
        Compute log probabilities for given actions using autoregressive masking.

        CRITICAL: Uses the same autoregressive masks as decode_actions() to ensure
        correct PPO ratios. Each action's log prob is computed under the distribution
        that was conditioned on the actual choices made.

        Args:
            action_logits: List of action dicts (length = action_budget)
            actions: [batch_size, action_budget, 3] actions to evaluate
            batch: [num_nodes] batch assignment for each node
            observations: List of PyG Data observations (required for masking)

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

                obs = observations[graph_idx] if observations is not None else None

                # Step 1: Source mask (ownership)
                if obs is not None:
                    source_mask = self._compute_source_mask(obs)
                    masked_source_scores = torch.where(
                        source_mask,
                        graph_source_scores,
                        torch.tensor(-1e10, device=graph_source_scores.device, dtype=graph_source_scores.dtype)
                    )
                else:
                    masked_source_scores = graph_source_scores

                # Step 2: Dest mask (conditioned on actual source choice)
                if obs is not None:
                    dest_mask = self._compute_dest_mask_for_source(obs, source_idx.item())
                    masked_dest_scores = torch.where(
                        dest_mask,
                        graph_dest_scores,
                        torch.tensor(-1e10, device=graph_dest_scores.device, dtype=graph_dest_scores.dtype)
                    )
                else:
                    masked_dest_scores = graph_dest_scores

                # Step 3: Troops mask (conditioned on actual source + dest)
                if obs is not None:
                    troops_mask = self._compute_troops_mask_for_action(obs, source_idx.item(), dest_idx.item())
                    masked_troops_logits = torch.where(
                        troops_mask,
                        graph_troops_logits,
                        torch.tensor(-1e10, device=graph_troops_logits.device, dtype=graph_troops_logits.dtype)
                    )
                else:
                    masked_troops_logits = graph_troops_logits

                # Compute log probabilities
                source_log_prob = F.log_softmax(masked_source_scores, dim=0)[source_idx]
                dest_log_prob = F.log_softmax(masked_dest_scores, dim=0)[dest_idx]
                troops_log_prob = F.log_softmax(masked_troops_logits, dim=0)[troops_idx]

                total_log_prob = source_log_prob + dest_log_prob + troops_log_prob
                batch_log_probs.append(total_log_prob)

            log_probs_tensor = torch.stack(batch_log_probs)  # [batch_size]
            all_log_probs.append(log_probs_tensor)

        log_probs = torch.stack(all_log_probs, dim=1)  # [batch_size, action_budget]
        return log_probs

    def compute_entropy(
        self,
        action_logits: List[Dict[str, torch.Tensor]],
        batch: torch.Tensor,
        observations: List = None,
    ) -> torch.Tensor:
        """
        Compute entropy of action distributions using autoregressive masking.

        Higher entropy = more exploration.

        NOTE: For autoregressive distributions, we compute entropy for each component
        under its conditional mask. This gives an approximation that's useful for
        the entropy bonus in PPO, though it's not the true joint entropy.

        Args:
            action_logits: List of action dicts (length = action_budget)
            batch: [num_nodes] batch assignment for each node
            observations: List of PyG Data observations (required for masking)

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

                obs = observations[graph_idx] if observations is not None else None

                # Source entropy (with ownership mask)
                if obs is not None:
                    source_mask = self._compute_source_mask(obs)
                    masked_source_scores = torch.where(
                        source_mask,
                        graph_source_scores,
                        torch.tensor(-1e10, device=graph_source_scores.device, dtype=graph_source_scores.dtype)
                    )
                else:
                    masked_source_scores = graph_source_scores

                source_dist = torch.distributions.Categorical(logits=masked_source_scores)
                source_entropy = source_dist.entropy()

                # For entropy, we compute expected entropy over dest and troops
                # by averaging over the source distribution. This is an approximation
                # but gives a reasonable entropy bonus signal.
                # For simplicity, we use the average mask across likely sources.
                source_probs = F.softmax(masked_source_scores, dim=0)

                # Weighted average dest entropy
                dest_entropy = torch.tensor(0.0, device=graph_dest_scores.device)
                troops_entropy = torch.tensor(0.0, device=graph_troops_logits.device)

                for src_idx in range(graph_source_scores.size(0)):
                    src_prob = source_probs[src_idx]
                    if src_prob < 1e-6:
                        continue

                    if obs is not None:
                        dest_mask = self._compute_dest_mask_for_source(obs, src_idx)
                        masked_dest_scores = torch.where(
                            dest_mask,
                            graph_dest_scores,
                            torch.tensor(-1e10, device=graph_dest_scores.device, dtype=graph_dest_scores.dtype)
                        )
                    else:
                        masked_dest_scores = graph_dest_scores

                    dest_dist = torch.distributions.Categorical(logits=masked_dest_scores)
                    dest_entropy = dest_entropy + src_prob * dest_dist.entropy()

                    # For troops, average over dest choices given this source
                    dest_probs = F.softmax(masked_dest_scores, dim=0)
                    for dst_idx in range(graph_dest_scores.size(0)):
                        dst_prob = dest_probs[dst_idx]
                        if dst_prob < 1e-6:
                            continue

                        if obs is not None:
                            troops_mask = self._compute_troops_mask_for_action(obs, src_idx, dst_idx)
                            masked_troops_logits = torch.where(
                                troops_mask,
                                graph_troops_logits,
                                torch.tensor(-1e10, device=graph_troops_logits.device, dtype=graph_troops_logits.dtype)
                            )
                        else:
                            masked_troops_logits = graph_troops_logits

                        troops_dist = torch.distributions.Categorical(logits=masked_troops_logits)
                        troops_entropy = troops_entropy + src_prob * dst_prob * troops_dist.entropy()

                # Sum entropies (chain rule approximation)
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

    def _compute_dest_mask_for_source(self, observation, source_idx: int) -> torch.Tensor:
        """Compute destination mask conditioned on chosen source.

        Valid destinations are:
        - The source itself (for deploy actions)
        - Territories adjacent to the source (for transfer/attack)

        Args:
            observation: PyG Data object
            source_idx: Index of the chosen source territory

        Returns:
            Boolean mask [n_territories] where True = valid destination
        """
        n_territories = observation.num_nodes
        edge_index = observation.edge_index

        dest_mask = torch.zeros(n_territories, dtype=torch.bool, device=observation.x.device)

        # Deploy: source == dest
        dest_mask[source_idx] = True

        # Transfer/Attack: adjacent territories
        neighbors = edge_index[1, edge_index[0] == source_idx]
        dest_mask[neighbors] = True

        return dest_mask

    def _compute_troops_mask_for_action(self, observation, source_idx: int, dest_idx: int) -> torch.Tensor:
        """Compute troops mask conditioned on chosen source and destination.

        Args:
            observation: PyG Data object
            source_idx: Index of the chosen source territory
            dest_idx: Index of the chosen destination territory

        Returns:
            Boolean mask [max_troops] where True = valid troop count
        """
        # Denormalize troops from log-scaled features
        troops_norm = observation.x[:, 0]
        troops = (torch.exp(troops_norm * torch.log1p(torch.tensor(100.0, device=troops_norm.device))) - 1).long()

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
        mask = torch.zeros(self.max_troops, dtype=torch.bool, device=observation.x.device)
        if max_troops_available > 0:
            # Troops are 1-indexed (troop count 1 to max_troops_available)
            mask[1:min(max_troops_available + 1, self.max_troops)] = True

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
