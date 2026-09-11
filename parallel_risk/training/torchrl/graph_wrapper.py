"""
Graph wrapper for Parallel Risk environment.

Converts standard Parallel Risk observations into PyTorch Geometric format
for use with Graph Neural Networks.

Requirements:
    pip install -r requirements/torchrl.txt
"""

from typing import Dict, Any, Optional
import weakref
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

_LOG1P_100 = np.log1p(100.0)


class _StaticGraphParts:
    """Map-constant pieces of a graph observation, built once per (MapConfig, max_regions).

    x_template: float32 [n_territories, 3 + max_regions] with the in-degree
        column and the region multi-hot columns filled in; columns 0 and 1
        (troops, ownership) are overwritten per observation.
    edge_index: int64 [2, n_edges] CPU tensor in row-major scan order of the
        adjacency matrix (same order as the original nested python loop). It
        is shared by every Data object built from this map: nothing in the
        codebase mutates edge_index in place (PyG batching and .to() create
        new tensors).
    """

    __slots__ = ('x_template', 'edge_index')

    def __init__(self, map_config: MapConfig, max_regions: int):
        n_territories = map_config.n_territories
        adjacency = map_config.adjacency_matrix

        x_template = np.zeros((n_territories, 3 + max_regions), dtype=np.float32)
        # Same expression as before (int sum -> float64 -> float32 on store).
        x_template[:, 2] = adjacency.sum(axis=1) / n_territories
        # Region membership is multi-hot: a territory in several regions gets
        # several 1s. Slots beyond this map's region count stay zero.
        for region_idx, territories in enumerate(map_config.regions.values()):
            x_template[territories, 3 + region_idx] = 1.0
        self.x_template = x_template

        sources, targets = np.nonzero(adjacency == 1)
        self.edge_index = torch.from_numpy(
            np.array([sources, targets], dtype=np.int64))


_STATIC_CACHE: Dict[tuple, _StaticGraphParts] = {}


def _static_parts(map_config: MapConfig, max_regions: int) -> _StaticGraphParts:
    """Cached _StaticGraphParts for (map_config, max_regions).

    Keyed by the MapConfig object's identity: MapConfig is an unhashable
    dataclass and MapRegistry.get() returns a fresh object per call, so each
    env/agent that owns a MapConfig gets its own entry. A weakref finalizer
    drops the entry when the MapConfig dies, so a recycled id() can never hit a
    stale entry. The static parts are read from the MapConfig once; do not
    mutate a MapConfig after it has been used to build graphs.
    """
    key = (id(map_config), max_regions)
    parts = _STATIC_CACHE.get(key)
    if parts is None:
        parts = _StaticGraphParts(map_config, max_regions)
        _STATIC_CACHE[key] = parts
        weakref.finalize(map_config, _STATIC_CACHE.pop, key, None)
    return parts


def env_to_graph(
    obs: Dict[str, Any],
    map_config: MapConfig,
    device: Optional[torch.device] = None,
    max_regions: Optional[int] = None,
) -> Data:
    """
    Convert environment observation to PyTorch Geometric graph.

    Node features (per territory):
    - troops (normalized)
    - ownership (+1 self, -1 enemy, 0 neutral)
    - in_degree (number of adjacent territories)
    - region_id (one-hot encoded, padded to max_regions)

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
        max_regions: Pad region one-hot node features and the region_control
            global feature to this width. Required for multi-map training
            where different maps have different region counts so all rollouts
            emit the same feature dimensionality. Defaults to
            len(map_config.regions) (single-map behavior).

    Returns:
        PyTorch Geometric Data object with:
        - x: node features [n_territories, feature_dim]
        - edge_index: graph connectivity [2, n_edges]
        - global_features: graph-level features [1, feature_dim]

    Raises:
        ImportError: If PyTorch Geometric is not installed

    Implementation note: the in-degree column, the region columns and
    edge_index depend only on the map, so they are built once per
    (map_config, max_regions) and cached (see _static_parts). They are derived
    from map_config.adjacency_matrix; obs['adjacency_matrix'] is always a copy
    of that same matrix (ParallelRiskEnv and RiskSimulator both emit one), so
    the result is identical to building everything from the observation.
    tests/test_graph_wrapper.py checks this against a reference build on
    every registered map.
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
    if max_regions is None:
        max_regions = n_regions
    if max_regions < n_regions:
        raise ValueError(
            f"max_regions ({max_regions}) must be >= map's n_regions ({n_regions})"
        )

    static = _static_parts(map_config, max_regions)

    # Node features: [troops, ownership, in_degree, region multi-hot...].
    # Troops are log-scaled (log1p(troops) / log1p(100)) so small counts stay
    # separable: 3 troops -> 0.30, 30 troops -> 0.74 instead of 0.03 / 0.30.
    # The float64 result is rounded to float32 on store, exactly as the
    # previous np.stack(...).astype(np.float32) did.
    node_features = static.x_template.copy()
    node_features[:, 0] = np.log1p(obs['territory_troops']) / _LOG1P_100
    node_features[:, 1] = obs['territory_ownership']

    # Global features [1, 2 + max_regions]: income / 20, turn / 100, then
    # region_control padded with zeros to max_regions so batched multi-map
    # rollouts share one global_features_dim. Income and turn are divided in
    # float32, as before.
    global_features = np.zeros((1, 2 + max_regions), dtype=np.float32)
    global_features[0, 0:1] = obs['available_income'].astype(np.float32) / 20.0
    global_features[0, 1:2] = obs['turn_number'].astype(np.float32) / 100.0
    global_features[0, 2:2 + n_regions] = obs['region_control']

    x = torch.from_numpy(node_features).to(device)
    edge_index = static.edge_index.to(device)
    global_features = torch.from_numpy(global_features).to(device)

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

    def __init__(self, env, device: Optional[torch.device] = None,
                 max_regions: Optional[int] = None):
        """
        Initialize wrapper around ParallelRiskEnv.

        Args:
            env: ParallelRiskEnv instance
            device: Torch device (cpu/cuda)
            max_regions: Pad region features to this width so multi-map training
                over maps with differing region counts emits a consistent
                feature dim. Defaults to len(env.map_config.regions) (single-map).

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
        self.max_regions = (
            max_regions if max_regions is not None else len(self.map_config.regions)
        )

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
            agent: env_to_graph(obs[agent], self.map_config, self.device,
                                max_regions=self.max_regions)
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
            agent: env_to_graph(obs[agent], self.map_config, self.device,
                                max_regions=self.max_regions)
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
        # feature dim uses self.max_regions (may exceed this map's own region
        # count when wrapping in a multi-map training set) so the model sees
        # the same input dim across every map.
        feature_dim = 3 + self.max_regions  # troops, ownership, in_degree, + region one-hot

        return {
            'type': 'graph',
            'node_features_dim': feature_dim,
            'global_features_dim': 2 + self.max_regions,
            'n_territories': n_territories,
            'n_regions': n_regions,
            'max_regions': self.max_regions,
        }

    @property
    def action_space(self):
        """Return action space from wrapped environment."""
        return self.env.action_spaces
