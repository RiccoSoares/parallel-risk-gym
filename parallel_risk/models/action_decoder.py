"""
Action decoder for GNN policies.

Converts graph embeddings into Parallel Risk actions.

Vectorized implementation: all per-graph work is done in one batched
Categorical over padded [B, max_nodes] tensors. Handles variable graph
sizes via padding (padded slots get -1e10 logits so softmax weights them
to zero). The old per-graph-loop implementation lives in
_action_decoder_legacy.py and is used only by the parity test.

Action format: [source_territory, dest_territory, num_troops]
"""

import math

import torch
import torch.nn.functional as F
from typing import Tuple, List, Dict, Optional

from torch_geometric.data import Batch


# Precomputed constants used inside hot decoder loops. Recreating these as
# torch.tensor() at every call causes CPU→GPU sync each time when the batched
# graph lives on GPU. _MASK_NEG is a large-but-finite negative used with
# torch.where(mask, x, _MASK_NEG); staying finite avoids NaN in Categorical
# when every entry happens to be masked out.
_LOG1P_100 = math.log1p(100.0)
_MASK_NEG = -1e10


class _BatchGeom:
    """Cache of geometry derived once per (batch, observations) pair.

    Everything here is invariant across the action_budget slots, so we compute
    it once outside the outer loop and re-use.

    Fields:
        B: batch size (number of graphs)
        max_nodes: max graph size across the batch
        graph_sizes: [B]  number of nodes per graph
        ptr: [B+1]        cumulative offsets of nodes per graph
        local_idx: [total_nodes]  position of each node within its graph
        node_valid: [B, max_nodes] bool — True where a real node exists
        arange_B: [B]     cached device arange used for row indexing
        source_mask_bwn: [B, max_nodes] bool — nodes owned by the agent (or
                        None when observations is None)
        adj_bwn2: [B, max_nodes, max_nodes] bool — per-graph adjacency (or None)
        troops_bwn: [B, max_nodes] long — denormalized troop count per node
                    (or None)
        income_b: [B] long — available income per graph (or None)
    """

    def __init__(self, batch: torch.Tensor, observations, batched_obs):
        device = batch.device
        B = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        graph_sizes = torch.bincount(batch, minlength=B)  # [B]
        max_nodes = int(graph_sizes.max().item()) if B > 0 else 0

        ptr = torch.zeros(B + 1, dtype=torch.long, device=device)
        ptr[1:] = torch.cumsum(graph_sizes, dim=0)

        node_arange_total = torch.arange(batch.size(0), device=device, dtype=torch.long)
        local_idx = node_arange_total - ptr[:-1][batch]

        cols = torch.arange(max_nodes, device=device, dtype=torch.long)
        node_valid = cols.unsqueeze(0) < graph_sizes.unsqueeze(1)  # [B, max_nodes]

        arange_B = torch.arange(B, device=device, dtype=torch.long)

        source_mask_bwn = None
        adj_bwn2 = None
        troops_bwn = None
        income_b = None

        # If callers didn't pre-batch, do it internally (fast when graphs
        # already live on the same device).
        if batched_obs is None and observations is not None:
            batched_obs = Batch.from_data_list(list(observations))

        if batched_obs is not None:
            # Ownership mask — feature column 1 is +1 for own, -1 for enemy.
            ownership_flat = batched_obs.x[:, 1] == 1  # [total_nodes]
            source_mask_bwn = torch.zeros((B, max_nodes), dtype=torch.bool, device=device)
            source_mask_bwn[batch, local_idx] = ownership_flat
            source_mask_bwn = source_mask_bwn & node_valid

            # Adjacency matrix per graph. edge_index is GLOBAL because
            # Batch.from_data_list offsets it — convert back to local.
            ei = batched_obs.edge_index
            if ei.numel() > 0:
                edge_src_global = ei[0]
                edge_dst_global = ei[1]
                edge_batch = batch[edge_src_global]
                u_local = edge_src_global - ptr[:-1][edge_batch]
                v_local = edge_dst_global - ptr[:-1][edge_batch]
                adj_bwn2 = torch.zeros((B, max_nodes, max_nodes),
                                       dtype=torch.bool, device=device)
                adj_bwn2[edge_batch, u_local, v_local] = True
            else:
                adj_bwn2 = torch.zeros((B, max_nodes, max_nodes),
                                       dtype=torch.bool, device=device)

            # Denormalized troops per node.
            troops_norm_flat = batched_obs.x[:, 0]
            troops_flat = (torch.exp(troops_norm_flat * _LOG1P_100) - 1).long()
            troops_bwn = torch.zeros((B, max_nodes), dtype=torch.long, device=device)
            troops_bwn[batch, local_idx] = troops_flat

            # Income per graph. global_features may be [B, gf] (from Batch) or
            # [1, gf] (single obs); handle both.
            gf = batched_obs.global_features
            if gf.dim() == 1:
                gf = gf.unsqueeze(0)
            # After Batch.from_data_list, gf is [B*1, gf_dim] — one row per
            # graph. Only the first column carries available_income (see
            # graph_wrapper.env_to_graph).
            if gf.size(0) == B:
                income_b = (gf[:, 0] * 20).long()
            else:
                # Some callers may have collapsed gf; broadcast single-row.
                income_b = (gf[0, 0] * 20).long().expand(B).clone()

        self.B = B
        self.max_nodes = max_nodes
        self.graph_sizes = graph_sizes
        self.ptr = ptr
        self.local_idx = local_idx
        self.node_valid = node_valid
        self.arange_B = arange_B
        self.source_mask_bwn = source_mask_bwn
        self.adj_bwn2 = adj_bwn2
        self.troops_bwn = troops_bwn
        self.income_b = income_b
        self.device = device
        self.batch = batch


def _pad_node_scores(flat_scores: torch.Tensor, geom: _BatchGeom) -> torch.Tensor:
    """Scatter [total_nodes] node scores into [B, max_nodes], padding invalid slots with _MASK_NEG."""
    B, max_nodes = geom.B, geom.max_nodes
    out = torch.full((B, max_nodes), _MASK_NEG,
                     device=flat_scores.device, dtype=flat_scores.dtype)
    out[geom.batch, geom.local_idx] = flat_scores
    return out


def _dest_mask_for(geom: _BatchGeom, source_idx: torch.Tensor) -> torch.Tensor:
    """[B, max_nodes] bool: valid destinations given a chosen source per graph.

    Valid = the source itself (deploy) OR any adjacent node.
    """
    # Gather the adjacency row for each graph's chosen source.
    dest_valid = geom.adj_bwn2[geom.arange_B, source_idx]  # [B, max_nodes]
    # Always allow source == dest (deploy).
    dest_valid = dest_valid.clone()
    dest_valid[geom.arange_B, source_idx] = True
    # Padded positions must never be True.
    return dest_valid & geom.node_valid


def _troops_mask_for(geom: _BatchGeom, source_idx: torch.Tensor,
                     dest_idx: torch.Tensor, max_troops: int) -> torch.Tensor:
    """[B, max_troops] bool: valid troop counts given chosen (source, dest).

    Deploy (source==dest): 1 .. income
    Transfer/Attack:        1 .. troops[source] - 1
    """
    device = source_idx.device
    is_deploy = (source_idx == dest_idx)
    src_troops = geom.troops_bwn[geom.arange_B, source_idx]  # [B]
    max_avail = torch.where(is_deploy, geom.income_b,
                            (src_troops - 1).clamp(min=0))  # [B]
    troops_arange = torch.arange(max_troops, device=device).unsqueeze(0)  # [1, max_troops]
    return (troops_arange >= 1) & (troops_arange <= max_avail.unsqueeze(1))


class ActionDecoder:
    """
    Decode graph embeddings into Parallel Risk actions.

    Handles variable-sized graphs (different number of territories per graph)
    in a batched setting by padding per-graph tensors to max_nodes and using
    a single Categorical over the padded [B, max_nodes] logits per action slot.

    Uses autoregressive masking: source → dest|source → troops|source,dest.
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
        batched_obs: Optional[Batch] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Sample actions from GCN policy logits with autoregressive masking.

        Args:
            action_logits: List of dicts (length = action_budget). Each has
                'source' [total_nodes], 'dest' [total_nodes], 'troops' [B, max_troops].
            batch: [total_nodes] batch assignment for each node.
            deterministic: If True, use argmax; else sample from Categorical.
            return_log_probs: If True, also return log-prob of chosen actions.
            observations: List of PyG Data (one per graph). Used to derive
                ownership/adjacency/troop masks. If None, no masking is applied.
            batched_obs: Optional pre-built Batch of observations. When provided,
                skips the internal Batch.from_data_list call (used by trainers
                that already build the mega-batch).

        Returns:
            actions: [B, action_budget, 3] long tensor.
            log_probs: [B, action_budget] float tensor (or None).
        """
        geom = _BatchGeom(batch, observations, batched_obs)
        B, device = geom.B, geom.device
        max_nodes = geom.max_nodes

        all_actions = []
        all_log_probs = []

        for logits_dict in action_logits:
            src_scores = _pad_node_scores(logits_dict['source'], geom)  # [B, max_nodes]
            dst_scores = _pad_node_scores(logits_dict['dest'], geom)    # [B, max_nodes]
            troops_logits = logits_dict['troops']                        # [B, max_troops]

            # Source: apply ownership mask.
            if geom.source_mask_bwn is not None:
                src_scores = torch.where(geom.source_mask_bwn, src_scores, _MASK_NEG)
            else:
                # No obs → still respect padding.
                src_scores = torch.where(geom.node_valid, src_scores, _MASK_NEG)

            src_dist = torch.distributions.Categorical(logits=src_scores)
            if deterministic:
                source_idx = src_scores.argmax(dim=-1)
            else:
                source_idx = src_dist.sample()
            source_log_prob = src_dist.log_prob(source_idx)

            # Dest: apply adjacency-conditioned mask.
            if geom.adj_bwn2 is not None:
                dest_valid = _dest_mask_for(geom, source_idx)
                dst_scores = torch.where(dest_valid, dst_scores, _MASK_NEG)
            else:
                dst_scores = torch.where(geom.node_valid, dst_scores, _MASK_NEG)

            dst_dist = torch.distributions.Categorical(logits=dst_scores)
            if deterministic:
                dest_idx = dst_scores.argmax(dim=-1)
            else:
                dest_idx = dst_dist.sample()
            dest_log_prob = dst_dist.log_prob(dest_idx)

            # Troops: apply (source,dest)-conditioned mask.
            if geom.troops_bwn is not None:
                troops_mask_bt = _troops_mask_for(geom, source_idx, dest_idx, self.max_troops)
                troops_logits_masked = torch.where(troops_mask_bt, troops_logits, _MASK_NEG)
            else:
                troops_logits_masked = troops_logits

            troops_dist = torch.distributions.Categorical(logits=troops_logits_masked)
            if deterministic:
                troops_idx = troops_logits_masked.argmax(dim=-1)
            else:
                troops_idx = troops_dist.sample()
            troops_log_prob = troops_dist.log_prob(troops_idx)

            all_actions.append(torch.stack([source_idx, dest_idx, troops_idx], dim=-1))  # [B, 3]
            if return_log_probs:
                all_log_probs.append(source_log_prob + dest_log_prob + troops_log_prob)  # [B]

        actions = torch.stack(all_actions, dim=1)  # [B, action_budget, 3]
        if return_log_probs:
            log_probs = torch.stack(all_log_probs, dim=1)  # [B, action_budget]
            return actions, log_probs
        return actions, None

    def compute_log_probs(
        self,
        action_logits: List[Dict[str, torch.Tensor]],
        actions: torch.Tensor,
        batch: torch.Tensor,
        observations: List = None,
        batched_obs: Optional[Batch] = None,
    ) -> torch.Tensor:
        """
        Log-prob of the given actions under the autoregressive-masked distribution.

        Same masking pipeline as decode_actions; the difference is we gather
        log-probs at the passed source/dest/troops indices instead of sampling.
        """
        geom = _BatchGeom(batch, observations, batched_obs)
        B, device = geom.B, geom.device

        action_budget = actions.size(1)
        all_log_probs = []

        for action_idx in range(action_budget):
            logits_dict = action_logits[action_idx]
            src_scores = _pad_node_scores(logits_dict['source'], geom)
            dst_scores = _pad_node_scores(logits_dict['dest'], geom)
            troops_logits = logits_dict['troops']

            source_idx = actions[:, action_idx, 0].long()
            dest_idx   = actions[:, action_idx, 1].long()
            troops_idx = actions[:, action_idx, 2].long()

            if geom.source_mask_bwn is not None:
                src_scores = torch.where(geom.source_mask_bwn, src_scores, _MASK_NEG)
            else:
                src_scores = torch.where(geom.node_valid, src_scores, _MASK_NEG)

            source_log_prob = F.log_softmax(src_scores, dim=-1).gather(
                1, source_idx.unsqueeze(1)).squeeze(1)  # [B]

            if geom.adj_bwn2 is not None:
                dest_valid = _dest_mask_for(geom, source_idx)
                dst_scores = torch.where(dest_valid, dst_scores, _MASK_NEG)
            else:
                dst_scores = torch.where(geom.node_valid, dst_scores, _MASK_NEG)

            dest_log_prob = F.log_softmax(dst_scores, dim=-1).gather(
                1, dest_idx.unsqueeze(1)).squeeze(1)  # [B]

            if geom.troops_bwn is not None:
                troops_mask_bt = _troops_mask_for(geom, source_idx, dest_idx, self.max_troops)
                troops_logits_masked = torch.where(troops_mask_bt, troops_logits, _MASK_NEG)
            else:
                troops_logits_masked = troops_logits

            troops_log_prob = F.log_softmax(troops_logits_masked, dim=-1).gather(
                1, troops_idx.unsqueeze(1)).squeeze(1)  # [B]

            all_log_probs.append(source_log_prob + dest_log_prob + troops_log_prob)

        return torch.stack(all_log_probs, dim=1)  # [B, action_budget]

    def compute_entropy(
        self,
        action_logits: List[Dict[str, torch.Tensor]],
        batch: torch.Tensor,
        observations: List = None,
        batched_obs: Optional[Batch] = None,
    ) -> torch.Tensor:
        """
        Entropy of the action distribution per slot per graph.

        Approximation preserved from the legacy path: source entropy uses the
        ownership mask; dest and troops entropy are computed unmasked. This
        avoids an O(n_sources × n_dests) coupling while still providing a
        useful exploration signal for PPO's entropy bonus.
        """
        geom = _BatchGeom(batch, observations, batched_obs)

        all_entropies = []

        for logits_dict in action_logits:
            src_scores = _pad_node_scores(logits_dict['source'], geom)
            dst_scores = _pad_node_scores(logits_dict['dest'], geom)
            troops_logits = logits_dict['troops']

            if geom.source_mask_bwn is not None:
                src_scores = torch.where(geom.source_mask_bwn, src_scores, _MASK_NEG)
            else:
                src_scores = torch.where(geom.node_valid, src_scores, _MASK_NEG)

            # For dest/troops we keep the "unmasked" approximation, but we
            # STILL respect the padding mask on the node-level scores — the
            # legacy version only ever saw real nodes, so it never assigned
            # probability to padded slots. Applying node_valid keeps parity.
            dst_scores_unmasked = torch.where(geom.node_valid, _pad_node_scores(
                logits_dict['dest'], geom), _MASK_NEG)

            source_entropy = torch.distributions.Categorical(logits=src_scores).entropy()
            dest_entropy   = torch.distributions.Categorical(logits=dst_scores_unmasked).entropy()
            troops_entropy = torch.distributions.Categorical(logits=troops_logits).entropy()

            all_entropies.append(source_entropy + dest_entropy + troops_entropy)

        return torch.stack(all_entropies, dim=1)  # [B, action_budget]

    # ------------------------------------------------------------------
    # Backward-compat helpers (some external callers may still use these).
    # ------------------------------------------------------------------

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

    if batch_size == 1:
        actions_np = actions[0].cpu().numpy()
        return tuple(actions_np)

    env_actions = []
    for i in range(batch_size):
        actions_np = actions[i].cpu().numpy()
        env_actions.append(tuple(actions_np))

    return env_actions
