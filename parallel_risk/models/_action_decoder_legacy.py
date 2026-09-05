"""
Legacy per-graph-loop ActionDecoder implementation.

Kept only as a reference for the parity test at
tests/test_action_decoder_parity.py — production code uses the vectorized
version in action_decoder.py. Delete this file once the vectorized decoder
has been validated in an end-to-end training run.
"""

import math

import torch
import torch.nn.functional as F
from typing import Tuple, List, Dict

_LOG1P_100 = math.log1p(100.0)
_MASK_NEG = -1e10


class _ActionDecoderLegacy:
    """Original ActionDecoder implementation (pre-vectorization).

    Preserved verbatim from the Phase-2 state so numerical parity can be
    verified against the vectorized replacement.
    """

    def __init__(self, action_budget: int = 5, max_troops: int = 20):
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
        batch_size = int(batch.max().item()) + 1

        all_actions = []
        all_log_probs = []

        for action_idx, logits_dict in enumerate(action_logits):
            source_scores = logits_dict['source']
            dest_scores = logits_dict['dest']
            troops_logits = logits_dict['troops']

            batch_actions = []
            batch_log_probs = []

            for graph_idx in range(batch_size):
                node_mask = (batch == graph_idx)
                graph_source_scores = source_scores[node_mask]
                graph_dest_scores = dest_scores[node_mask]
                graph_troops_logits = troops_logits[graph_idx]

                obs = observations[graph_idx] if observations is not None else None

                if obs is not None:
                    source_mask = self._compute_source_mask(obs)
                    masked_source_scores = torch.where(source_mask, graph_source_scores, _MASK_NEG)
                else:
                    masked_source_scores = graph_source_scores

                if deterministic:
                    source_idx = torch.argmax(masked_source_scores)
                else:
                    source_idx = torch.distributions.Categorical(logits=masked_source_scores).sample()

                if obs is not None:
                    dest_mask = self._compute_dest_mask_for_source(obs, source_idx.item())
                    masked_dest_scores = torch.where(dest_mask, graph_dest_scores, _MASK_NEG)
                else:
                    masked_dest_scores = graph_dest_scores

                if deterministic:
                    dest_idx = torch.argmax(masked_dest_scores)
                else:
                    dest_idx = torch.distributions.Categorical(logits=masked_dest_scores).sample()

                if obs is not None:
                    troops_mask = self._compute_troops_mask_for_action(obs, source_idx.item(), dest_idx.item())
                    masked_troops_logits = torch.where(troops_mask, graph_troops_logits, _MASK_NEG)
                else:
                    masked_troops_logits = graph_troops_logits

                if deterministic:
                    troops_idx = torch.argmax(masked_troops_logits)
                else:
                    troops_idx = torch.distributions.Categorical(logits=masked_troops_logits).sample()

                action = torch.stack([source_idx, dest_idx, troops_idx])
                batch_actions.append(action)

                if return_log_probs:
                    source_log_prob = F.log_softmax(masked_source_scores, dim=0)[source_idx]
                    dest_log_prob = F.log_softmax(masked_dest_scores, dim=0)[dest_idx]
                    troops_log_prob = F.log_softmax(masked_troops_logits, dim=0)[troops_idx]
                    total_log_prob = source_log_prob + dest_log_prob + troops_log_prob
                    batch_log_probs.append(total_log_prob)

            all_actions.append(torch.stack(batch_actions))
            if return_log_probs:
                all_log_probs.append(torch.stack(batch_log_probs))

        actions = torch.stack(all_actions, dim=1)

        if return_log_probs:
            log_probs = torch.stack(all_log_probs, dim=1)
            return actions, log_probs
        return actions, None

    def compute_log_probs(
        self,
        action_logits: List[Dict[str, torch.Tensor]],
        actions: torch.Tensor,
        batch: torch.Tensor,
        observations: List = None,
    ) -> torch.Tensor:
        batch_size = actions.size(0)
        action_budget = actions.size(1)
        graph_sizes = torch.bincount(batch).tolist()

        all_log_probs = []
        for action_idx in range(action_budget):
            logits_dict = action_logits[action_idx]
            source_scores_by_graph = logits_dict['source'].split(graph_sizes)
            dest_scores_by_graph = logits_dict['dest'].split(graph_sizes)
            troops_logits = logits_dict['troops']

            batch_log_probs = []
            for graph_idx in range(batch_size):
                graph_source_scores = source_scores_by_graph[graph_idx]
                graph_dest_scores = dest_scores_by_graph[graph_idx]
                graph_troops_logits = troops_logits[graph_idx]

                action = actions[graph_idx, action_idx]
                source_idx = action[0].long()
                dest_idx = action[1].long()
                troops_idx = action[2].long()

                obs = observations[graph_idx] if observations is not None else None

                if obs is not None:
                    source_mask = self._compute_source_mask(obs)
                    masked_source_scores = torch.where(source_mask, graph_source_scores, _MASK_NEG)
                else:
                    masked_source_scores = graph_source_scores

                if obs is not None:
                    dest_mask = self._compute_dest_mask_for_source(obs, source_idx.item())
                    masked_dest_scores = torch.where(dest_mask, graph_dest_scores, _MASK_NEG)
                else:
                    masked_dest_scores = graph_dest_scores

                if obs is not None:
                    troops_mask = self._compute_troops_mask_for_action(obs, source_idx.item(), dest_idx.item())
                    masked_troops_logits = torch.where(troops_mask, graph_troops_logits, _MASK_NEG)
                else:
                    masked_troops_logits = graph_troops_logits

                source_log_prob = F.log_softmax(masked_source_scores, dim=0)[source_idx]
                dest_log_prob = F.log_softmax(masked_dest_scores, dim=0)[dest_idx]
                troops_log_prob = F.log_softmax(masked_troops_logits, dim=0)[troops_idx]

                total_log_prob = source_log_prob + dest_log_prob + troops_log_prob
                batch_log_probs.append(total_log_prob)

            all_log_probs.append(torch.stack(batch_log_probs))

        return torch.stack(all_log_probs, dim=1)

    def compute_entropy(
        self,
        action_logits: List[Dict[str, torch.Tensor]],
        batch: torch.Tensor,
        observations: List = None,
    ) -> torch.Tensor:
        batch_size = int(batch.max().item()) + 1
        graph_sizes = torch.bincount(batch).tolist()

        all_entropies = []
        for action_idx, logits_dict in enumerate(action_logits):
            source_by_graph = logits_dict['source'].split(graph_sizes)
            dest_by_graph = logits_dict['dest'].split(graph_sizes)
            troops_logits = logits_dict['troops']

            batch_entropies = []
            for graph_idx in range(batch_size):
                obs = observations[graph_idx] if observations is not None else None
                graph_src = source_by_graph[graph_idx]
                graph_dst = dest_by_graph[graph_idx]
                graph_troops = troops_logits[graph_idx]

                if obs is not None:
                    source_mask = self._compute_source_mask(obs)
                    masked_src = torch.where(source_mask, graph_src, _MASK_NEG)
                else:
                    masked_src = graph_src

                source_entropy = torch.distributions.Categorical(logits=masked_src).entropy()
                dest_entropy = torch.distributions.Categorical(logits=graph_dst).entropy()
                troops_entropy = torch.distributions.Categorical(logits=graph_troops).entropy()

                total_entropy = source_entropy + dest_entropy + troops_entropy
                batch_entropies.append(total_entropy)

            all_entropies.append(torch.stack(batch_entropies))

        return torch.stack(all_entropies, dim=1)

    def _compute_source_mask(self, observation) -> torch.Tensor:
        ownership = observation.x[:, 1]
        return ownership == 1

    def _compute_dest_mask_for_source(self, observation, source_idx: int) -> torch.Tensor:
        n_territories = observation.num_nodes
        edge_index = observation.edge_index
        dest_mask = torch.zeros(n_territories, dtype=torch.bool, device=observation.x.device)
        dest_mask[source_idx] = True
        neighbors = edge_index[1, edge_index[0] == source_idx]
        dest_mask[neighbors] = True
        return dest_mask

    def _compute_troops_mask_for_action(self, observation, source_idx: int, dest_idx: int) -> torch.Tensor:
        troops_norm = observation.x[:, 0]
        troops = (torch.exp(troops_norm * _LOG1P_100) - 1).long()

        gf = observation.global_features
        if gf.dim() == 2:
            income_norm = gf[0, 0]
        else:
            income_norm = gf[0]
        income = (income_norm * 20).long()

        if source_idx == dest_idx:
            max_troops_available = int(income.item())
        else:
            max_troops_available = max(0, int(troops[source_idx].item()) - 1)

        mask = torch.zeros(self.max_troops, dtype=torch.bool, device=observation.x.device)
        if max_troops_available > 0:
            mask[1:min(max_troops_available + 1, self.max_troops)] = True
        return mask
