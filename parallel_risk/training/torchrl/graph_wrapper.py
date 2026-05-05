"""
Graph wrapper for Parallel Risk environment.

Converts standard Parallel Risk observations into PyTorch Geometric format
for use with Graph Neural Networks.

Requirements:
    pip install -r requirements/torchrl.txt
"""

from typing import Dict, Any, Optional
import numpy as np

try:
    import torch
    from torch_geometric.data import Data
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    torch = None
    Data = None

from parallel_risk.env.map_config import MapConfig


def env_to_graph(
    obs: Dict[str, Any],
    map_config: MapConfig,
    device: Optional[torch.device] = None
) -> Data:
    """
    Convert environment observation to PyTorch Geometric graph.

    Node features (per territory):
    - troops (normalized)
    - ownership (+1 self, -1 enemy, 0 neutral)
    - in_degree (number of adjacent territories)
    - region_id (one-hot encoded)

    Args:
        obs: Observation dict from ParallelRiskEnv with keys:
            - territory_ownership: (n_territories,) int8 array
            - territory_troops: (n_territories,) int32 array
            - adjacency_matrix: (n_territories, n_territories) int8 array
            - available_income: (1,) int32 array
            - turn_number: (1,) int32 array
            - region_control: (n_regions,) int8 array
        map_config: MapConfig with adjacency information and regions
        device: Torch device (cpu/cuda)

    Returns:
        PyTorch Geometric Data object with:
        - x: node features [n_territories, feature_dim]
        - edge_index: graph connectivity [2, n_edges]
        - global_features: graph-level features [feature_dim]

    Raises:
        ImportError: If PyTorch Geometric is not installed
    """
    if not TORCH_GEOMETRIC_AVAILABLE:
        raise ImportError(
            "PyTorch Geometric is not installed. "
            "Install Phase 2 dependencies: pip install -r requirements/torchrl.txt"
        )
    if device is None:
        device = torch.device('cpu')

    n_territories = map_config.n_territories
    n_regions = len(map_config.regions)

    # Extract observation components
    ownership = obs['territory_ownership']  # Shape: (n_territories,)
    troops = obs['territory_troops']  # Shape: (n_territories,)
    adjacency_matrix = obs['adjacency_matrix']  # Shape: (n_territories, n_territories)
    available_income = obs['available_income']  # Shape: (1,)
    turn_number = obs['turn_number']  # Shape: (1,)
    region_control = obs['region_control']  # Shape: (n_regions,)

    # Create node features
    node_features = []

    # Feature 1: Troop count (log-scaled for better variance preservation)
    # Problem: Linear normalization (troops/100) compresses values too much
    #   - 3 troops → 0.03, 30 troops → 0.30 (tiny differences, std ~0.02)
    # Solution: Use log1p (log(1+x)) to spread out small values
    #   - 3 troops → log(4)≈1.39, 30 troops → log(31)≈3.43 (better separation)
    #   - Then divide by log(101) to keep in reasonable range
    troops_log_normalized = np.log1p(troops) / np.log1p(100.0)
    node_features.append(troops_log_normalized)

    # Feature 2: Ownership (-1, 0, 1)
    node_features.append(ownership)

    # Feature 3: In-degree (number of adjacent territories)
    in_degree = adjacency_matrix.sum(axis=1)
    in_degree_normalized = in_degree / n_territories  # Normalize
    node_features.append(in_degree_normalized)

    # Features 4+: Region membership (multi-hot encoding)
    # Create territory-to-region mapping
    # Note: Territories can belong to multiple regions (e.g., 'center' overlaps with 'north'/'south')
    # We keep raw multi-hot values so the GNN can learn that territories in multiple regions are special
    territory_to_region = np.zeros((n_territories, n_regions), dtype=np.float32)
    for region_idx, (region_name, territories) in enumerate(map_config.regions.items()):
        for territory_id in territories:
            territory_to_region[territory_id, region_idx] = 1.0

    for region_idx in range(n_regions):
        node_features.append(territory_to_region[:, region_idx])

    # Stack all node features
    node_features = np.stack(node_features, axis=1).astype(np.float32)  # Shape: (n_territories, feature_dim)

    # Create edge index (COO format)
    # edge_index[0] = source nodes, edge_index[1] = target nodes
    edge_sources = []
    edge_targets = []
    for i in range(n_territories):
        for j in range(n_territories):
            if adjacency_matrix[i, j] == 1:
                edge_sources.append(i)
                edge_targets.append(j)

    edge_index = np.array([edge_sources, edge_targets], dtype=np.int64)

    # Create global features (graph-level)
    global_features = np.concatenate([
        available_income.astype(np.float32) / 20.0,  # Normalize income (max ~20)
        turn_number.astype(np.float32) / 100.0,  # Normalize turn (max 100)
        region_control.astype(np.float32),  # Region control indicators
    ])

    # Convert to PyTorch tensors
    x = torch.tensor(node_features, dtype=torch.float32, device=device)
    edge_index = torch.tensor(edge_index, dtype=torch.long, device=device)
    global_features = torch.tensor(global_features, dtype=torch.float32, device=device)

    # Ensure global_features is 2D [1, dim] for proper batching
    # PyG Batch.from_data_list() will stack these along dim=0 to get [batch_size, dim]
    if global_features.dim() == 1:
        global_features = global_features.unsqueeze(0)  # [1, dim]

    # Create PyTorch Geometric Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        num_nodes=n_territories,
    )

    # Store global features as graph attribute
    data.global_features = global_features

    # Store metadata
    data.n_territories = n_territories
    data.n_regions = n_regions

    return data


class GraphObservationWrapper:
    """
    Wrapper that converts ParallelRiskEnv observations to graph format.

    This wrapper transforms flat observations into PyTorch Geometric graphs,
    enabling the use of GNN policies.
    """

    def __init__(self, env, device: Optional[torch.device] = None):
        """
        Initialize wrapper around ParallelRiskEnv.

        Args:
            env: ParallelRiskEnv instance
            device: Torch device (cpu/cuda)

        Raises:
            ImportError: If PyTorch Geometric is not installed
        """
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError(
                "PyTorch Geometric is not installed. "
                "Install Phase 2 dependencies: pip install -r requirements/torchrl.txt"
            )
        self.env = env
        self.device = device if device is not None else torch.device('cpu')
        self.map_config = env.map_config

        # Store wrapped environment properties
        self.possible_agents = env.possible_agents
        self.agents = env.agents
        self.metadata = env.metadata

    def reset(self, seed=None, options=None):
        """
        Reset environment and return graph observations.

        Returns:
            observations: Dict[agent_id, Data] - Graph observations per agent
            infos: Dict[agent_id, dict] - Info dictionaries per agent
        """
        obs, infos = self.env.reset(seed=seed, options=options)

        # Convert observations to graphs
        graph_obs = {
            agent: env_to_graph(obs[agent], self.map_config, self.device)
            for agent in obs.keys()
        }

        return graph_obs, infos

    def step(self, actions):
        """
        Step environment with actions and return graph observations.

        Args:
            actions: Dict[agent_id, action] - Actions per agent

        Returns:
            observations: Dict[agent_id, Data] - Graph observations per agent
            rewards: Dict[agent_id, float] - Rewards per agent
            terminateds: Dict[agent_id, bool] - Terminated flags per agent
            truncateds: Dict[agent_id, bool] - Truncated flags per agent
            infos: Dict[agent_id, dict] - Info dictionaries per agent
        """
        obs, rewards, terminateds, truncateds, infos = self.env.step(actions)

        # Convert observations to graphs
        graph_obs = {
            agent: env_to_graph(obs[agent], self.map_config, self.device)
            for agent in obs.keys()
        }

        return graph_obs, rewards, terminateds, truncateds, infos

    def close(self):
        """Close the environment."""
        self.env.close()

    def render(self):
        """Render the environment."""
        return self.env.render()

    @property
    def observation_space(self):
        """
        Return observation space description.

        Note: This is not a standard gymnasium space since graphs have
        variable structure. Instead, we return a dict describing the format.
        """
        n_territories = self.map_config.n_territories
        n_regions = len(self.map_config.regions)
        feature_dim = 3 + n_regions  # troops, ownership, in_degree, + region one-hot

        return {
            'type': 'graph',
            'node_features_dim': feature_dim,
            'global_features_dim': 2 + n_regions,
            'n_territories': n_territories,
            'n_regions': n_regions,
        }

    @property
    def action_space(self):
        """Return action space from wrapped environment."""
        return self.env.action_spaces
