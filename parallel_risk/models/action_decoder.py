"""
Action decoder for GNN policies.

Converts graph embeddings into action distributions for Parallel Risk.

TODO Phase 2.2: Implement action decoder

Action format: [source_territory, dest_territory, num_troops]
Challenge: Variable-length action space with fixed budget

Approaches to consider:
1. Autoregressive: Sample source, then dest, then troops sequentially
2. Direct: Predict all actions simultaneously
3. Edge-based: Actions correspond to edges in the graph
"""

import torch
import torch.nn as nn


class ActionDecoder(nn.Module):
    """
    Decode graph embeddings into Parallel Risk actions.

    TODO: Implement this class
    """

    def __init__(
        self,
        hidden_dim: int,
        n_territories: int,
        action_budget: int,
        max_troops: int = 20,
    ):
        """
        Initialize action decoder.

        Args:
            hidden_dim: Size of node embeddings
            n_territories: Number of territories (may vary per graph)
            action_budget: Number of actions to output
            max_troops: Maximum troops per action
        """
        super().__init__()
        raise NotImplementedError("Phase 2.2: Action decoder not yet implemented")

    def forward(self, node_embeddings, edge_index, batch):
        """
        Decode actions from node embeddings.

        Args:
            node_embeddings: Node feature matrix after GNN layers
            edge_index: Graph connectivity
            batch: Batch assignment for each node

        Returns:
            action_logits: Distribution over actions
        """
        raise NotImplementedError("Phase 2.2: Action decoder not yet implemented")
