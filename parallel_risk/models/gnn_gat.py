"""
Graph Attention Network (GAT) policy for Parallel Risk.

TODO Phase 2.2: Implement GAT architecture

GAT learns attention weights over neighbors, which should help identify
strategic territories (chokepoints, region completion targets, etc.)

References:
- Veličković et al. (2018): Graph Attention Networks
- PyTorch Geometric GAT: https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GATConv
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool


class GATPolicy(nn.Module):
    """
    GAT-based policy network for Parallel Risk.

    TODO: Implement this class
    """

    def __init__(
        self,
        node_features: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        action_budget: int = 5,
    ):
        """
        Initialize GAT policy.

        Args:
            node_features: Number of input node features
            hidden_dim: Hidden dimension size
            num_layers: Number of GAT layers
            num_heads: Number of attention heads
            action_budget: Number of actions per turn
        """
        super().__init__()
        raise NotImplementedError("Phase 2.2: GAT policy not yet implemented")

    def forward(self, data):
        """
        Forward pass through GAT policy.

        Args:
            data: PyTorch Geometric Data object

        Returns:
            action_logits: Policy distribution over actions
            value: State value estimate
            attention_weights: Learned attention weights (for visualization)
        """
        raise NotImplementedError("Phase 2.2: GAT policy not yet implemented")
