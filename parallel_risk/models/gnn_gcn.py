"""
Graph Convolutional Network (GCN) policy for Parallel Risk.

Implements a GCN-based actor-critic policy for territorial strategy games.

Architecture:
- Node features: [troops, ownership, in_degree, region_id (one-hot)]
- GCN layers for message passing between territories
- Global pooling for value function (graph-level state value)
- Policy head outputs action distribution over [source, dest, troops]

References:
- Kipf & Welling (2017): Semi-Supervised Classification with Graph Convolutional Networks
- PyTorch Geometric GCN: https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GCNConv
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
from typing import Tuple, Optional


class GCNPolicy(nn.Module):
    """
    GCN-based actor-critic policy network for Parallel Risk.

    Processes graph observations through GCN layers, then outputs:
    - Policy: distribution over actions [source, dest, troops]
    - Value: state value estimate for the current graph
    """

    def __init__(
        self,
        node_features_dim: int,
        global_features_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        action_budget: int = 5,
        max_troops: int = 20,
        dropout: float = 0.1,
    ):
        """
        Initialize GCN policy.

        Args:
            node_features_dim: Number of input node features
            global_features_dim: Number of global features
            hidden_dim: Hidden dimension size
            num_layers: Number of GCN layers
            action_budget: Number of actions to output per turn
            max_troops: Maximum troops per action
            dropout: Dropout probability
        """
        super().__init__()

        self.node_features_dim = node_features_dim
        self.global_features_dim = global_features_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.action_budget = action_budget
        self.max_troops = max_troops
        self.dropout = dropout

        # Input projection
        self.input_proj = nn.Linear(node_features_dim, hidden_dim)

        # GCN layers
        self.conv_layers = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        # Layer normalization after each GCN layer
        # Using LayerNorm instead of BatchNorm to avoid mixing statistics
        # across different game states in batched graphs (critical for self-play)
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])

        # Global features processing
        self.global_proj = nn.Linear(global_features_dim, hidden_dim)

        # FIX: Add LayerNorm before value head to normalize the concatenated inputs
        # (graph_embedding and global_emb may have very different magnitudes)
        self.value_input_norm = nn.LayerNorm(hidden_dim * 2)

        # Value head (critic)
        # Takes graph-level embedding (global pooled nodes + global features)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Policy head (actor) - outputs actions
        # For each action in the budget, we predict:
        #   - source territory (logits over nodes)
        #   - dest territory (logits over nodes)
        #   - num troops (logits over [0, max_troops))

        # We use separate heads for each action in the budget
        self.policy_heads = nn.ModuleList([
            ActionHead(
                node_dim=hidden_dim,
                global_dim=hidden_dim,
                max_troops=max_troops,
                dropout=dropout
            )
            for _ in range(action_budget)
        ])

    def forward(
        self,
        data: Data,
        return_embeddings: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through GCN policy.

        Args:
            data: PyTorch Geometric Data object with:
                - x: node features [num_nodes, node_features_dim]
                - edge_index: graph connectivity [2, num_edges]
                - global_features: graph-level features [global_features_dim]
                - batch: batch assignment for each node (optional, for batched graphs)
            return_embeddings: If True, also return node embeddings

        Returns:
            action_logits: Tensor of shape [batch_size, action_budget, 3, max_logits]
                          where max_logits depends on component:
                          - source/dest: n_territories (varies per graph)
                          - troops: max_troops (fixed)
            value: State value estimate [batch_size, 1]
            embeddings: Node embeddings [num_nodes, hidden_dim] (optional)
        """
        x = data.x  # [num_nodes, node_features_dim]
        edge_index = data.edge_index  # [2, num_edges]
        global_features = data.global_features  # [batch_size, global_features_dim] or [global_features_dim]

        # Handle single graph vs batch
        if hasattr(data, 'batch') and data.batch is not None:
            batch = data.batch
            batch_size = int(batch.max().item()) + 1
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            batch_size = 1

        # Project node features
        x = self.input_proj(x)  # [num_nodes, hidden_dim]
        x = F.relu(x)

        # Apply GCN layers with residual connections
        for i, (conv, ln) in enumerate(zip(self.conv_layers, self.layer_norms)):
            x_residual = x
            x = conv(x, edge_index)  # [num_nodes, hidden_dim]
            x = ln(x)  # LayerNorm: normalizes per node independently
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            # Residual connection (skip connection) for ALL layers
            # FIX: Previously only applied for i > 0, but ALL layers have hidden_dim -> hidden_dim,
            # so dimensionality always matches. Skipping the first layer caused vanishing gradients.
            x = x + x_residual

        node_embeddings = x  # [num_nodes, hidden_dim]

        # Global pooling for value function
        graph_embedding = global_mean_pool(node_embeddings, batch)  # [batch_size, hidden_dim]

        # Process global features
        if hasattr(data, 'global_features'):
            # global_features should be [batch_size, global_features_dim]
            # (graph_wrapper ensures each graph stores it as [1, dim])
            gf = data.global_features
            if gf.dim() == 1:
                # Fallback: single graph without batch dim
                gf = gf.unsqueeze(0)

            # Verify shape consistency
            if gf.size(0) != batch_size:
                raise ValueError(
                    f"Global features batch size mismatch: "
                    f"got {gf.size(0)}, expected {batch_size}"
                )

            global_emb = self.global_proj(gf)  # [batch_size, hidden_dim]
            global_emb = F.relu(global_emb)
        else:
            # No global features, use zeros
            global_emb = torch.zeros(batch_size, self.hidden_dim, device=x.device)

        # Concatenate graph embedding and global features for value
        value_input = torch.cat([graph_embedding, global_emb], dim=-1)  # [batch_size, hidden_dim * 2]
        # FIX: Apply LayerNorm to normalize concatenated inputs before value head
        value_input = self.value_input_norm(value_input)
        value = self.value_head(value_input)  # [batch_size, 1]

        # Generate action logits for each action in the budget
        # Note: For simplicity, we generate all actions independently
        # A more sophisticated approach could use autoregressive generation
        all_action_logits = []
        for action_head in self.policy_heads:
            source_logits, dest_logits, troops_logits = action_head(
                node_embeddings, global_emb, batch
            )
            # Stack into [batch_size, 3, max_logits] where max_logits varies per component
            # We'll need to handle variable-sized graphs carefully
            action_logits = {
                'source': source_logits,  # [batch_size, n_territories]
                'dest': dest_logits,      # [batch_size, n_territories]
                'troops': troops_logits   # [batch_size, max_troops]
            }
            all_action_logits.append(action_logits)

        if return_embeddings:
            return all_action_logits, value, node_embeddings
        else:
            return all_action_logits, value, None


class ActionHead(nn.Module):
    """
    Action head that outputs logits for a single action: [source, dest, troops].

    For source/dest, we output per-node logits (variable size).
    For troops, we output fixed logits over [0, max_troops).
    """

    def __init__(
        self,
        node_dim: int,
        global_dim: int,
        max_troops: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.node_dim = node_dim
        self.global_dim = global_dim
        self.max_troops = max_troops

        # Combined embedding for action selection
        combined_dim = node_dim + global_dim

        # Source territory selection (per-node scoring)
        self.source_scorer = nn.Sequential(
            nn.Linear(combined_dim, node_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(node_dim, 1)
        )

        # Dest territory selection (per-node scoring)
        self.dest_scorer = nn.Sequential(
            nn.Linear(combined_dim, node_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(node_dim, 1)
        )

        # Troops selection (fixed-size output)
        self.troops_head = nn.Sequential(
            nn.Linear(global_dim, node_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(node_dim, max_troops)
        )

    def forward(
        self,
        node_embeddings: torch.Tensor,
        global_emb: torch.Tensor,
        batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate action logits.

        Args:
            node_embeddings: [num_nodes, node_dim]
            global_emb: [batch_size, global_dim]
            batch: [num_nodes] - batch assignment for each node

        Returns:
            source_logits: [batch_size, n_territories] (variable per graph)
            dest_logits: [batch_size, n_territories] (variable per graph)
            troops_logits: [batch_size, max_troops]
        """
        batch_size = int(batch.max().item()) + 1

        # Expand global_emb to match each node
        global_expanded = global_emb[batch]  # [num_nodes, global_dim]

        # Combine node and global features
        combined = torch.cat([node_embeddings, global_expanded], dim=-1)  # [num_nodes, node_dim + global_dim]

        # Score each node as potential source
        source_scores = self.source_scorer(combined).squeeze(-1)  # [num_nodes]

        # Score each node as potential dest
        dest_scores = self.dest_scorer(combined).squeeze(-1)  # [num_nodes]

        # For batched graphs, we need to return per-graph logits
        # This is tricky because each graph has different number of nodes
        # For now, we'll return per-node scores and handle batching externally
        # TODO: Proper batching with padding or dynamic batch handling

        # Troops selection (uses global context only)
        troops_logits = self.troops_head(global_emb)  # [batch_size, max_troops]

        return source_scores, dest_scores, troops_logits


def sample_actions(
    action_logits: list,
    batch: torch.Tensor,
    deterministic: bool = False
) -> torch.Tensor:
    """
    Sample actions from logits.

    Args:
        action_logits: List of action dicts from GCNPolicy forward pass
        batch: Batch assignment for nodes
        deterministic: If True, use argmax instead of sampling

    Returns:
        actions: [batch_size, action_budget, 3] tensor of sampled actions
    """
    # TODO: Implement proper action sampling
    # This requires handling variable-sized graphs in batches
    raise NotImplementedError("Action sampling will be implemented with action decoder")
