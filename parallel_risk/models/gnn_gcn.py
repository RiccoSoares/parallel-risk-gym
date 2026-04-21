"""
Graph Convolutional Network (GCN) policy for Parallel Risk.

TODO Phase 2.2: Implement GCN architecture

Architecture:
- Node features: [troops, ownership, region_id, in_degree]
- GCN layers for message passing
- Global pooling for value function
- Action decoder for policy head

References:
- Kipf & Welling (2017): Semi-Supervised Classification with Graph Convolutional Networks
- PyTorch Geometric GCN: https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GCNConv
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool


class GCNPolicy(nn.Module):
    """
    GCN-based policy network for Parallel Risk.

    TODO: Implement this class
    """

    def __init__(
        self,
        node_features: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 3,
        action_budget: int = 5,
    ):
        """
        Initialize GCN policy.

        Args:
            node_features: Number of input node features
            hidden_dim: Hidden dimension size
            num_layers: Number of GCN layers
            action_budget: Number of actions per turn
        """
        super().__init__()
        raise NotImplementedError("Phase 2.2: GCN policy not yet implemented")

    def forward(self, data):
        """
        Forward pass through GCN policy.

        Args:
            data: PyTorch Geometric Data object

        Returns:
            action_logits: Policy distribution over actions
            value: State value estimate
        """
        raise NotImplementedError("Phase 2.2: GCN policy not yet implemented")
