from dataclasses import dataclass
import numpy as np


@dataclass
class MapConfig:
    """Configuration for a game map"""
    n_territories: int
    adjacency_list: dict
    adjacency_matrix: np.ndarray
    initial_ownership: np.ndarray
    regions: dict
    region_bonuses: dict


class MapRegistry:
    """Registry for game maps"""
    _maps = {}

    @classmethod
    def register(cls, name, map_fn):
        """Register a map creation function"""
        cls._maps[name] = map_fn

    @classmethod
    def get(cls, name):
        """Get a map configuration by name"""
        if name not in cls._maps:
            raise ValueError(f"Unknown map name: {name}")
        return cls._maps[name]()

    @classmethod
    def list_maps(cls):
        """List all registered map names"""
        return list(cls._maps.keys())


def create_simple_6_map():
    """Create the default 6-territory grid map

    Map layout:
    0 - 1 - 2  (North Region)
    |   |   |
    3 - 4 - 5  (South Region)

    Center Region: [1, 4]
    """
    adjacency_list = {
        0: [1, 3],
        1: [0, 2, 4],
        2: [1, 5],
        3: [0, 4],
        4: [1, 3, 5],
        5: [2, 4],
    }
    n_territories = 6

    # Build adjacency matrix
    adjacency_matrix = np.zeros((n_territories, n_territories), dtype=np.int8)
    for source, neighbors in adjacency_list.items():
        for dest in neighbors:
            adjacency_matrix[source, dest] = 1

    # Initial ownership: agent_0 gets [0, 1, 5], agent_1 gets [2, 3, 4]
    initial_ownership = np.array([0, 0, 1, 1, 1, 0], dtype=np.int8)

    # Define bonus regions
    regions = {
        'north': [0, 1, 2],
        'south': [3, 4, 5],
        'center': [1, 4],
    }

    # Define bonus troops per region
    region_bonuses = {
        'north': 4,
        'south': 4,
        'center': 2,
    }

    return MapConfig(
        n_territories=n_territories,
        adjacency_list=adjacency_list,
        adjacency_matrix=adjacency_matrix,
        initial_ownership=initial_ownership,
        regions=regions,
        region_bonuses=region_bonuses,
    )


# Register the default map
MapRegistry.register("simple_6", create_simple_6_map)
