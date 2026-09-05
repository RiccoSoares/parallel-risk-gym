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


def create_medium_8_map():
    """Create an 8-territory bridge map with strategic chokepoints.

    Two fully-connected triangular clusters linked by a two-territory bridge.
    Territories 3 and 4 are the chokepoints: controlling them dominates flow
    between sides.

    Layout:
        West Cluster (triangle)   Bridge    East Cluster (triangle)
              [0]                               [7]
             /   \\                            /   \\
           [1]---[2]--[3]---[4]--[5]---[6]
                                  \\___[7]

    Exact adjacency:
        0: [1, 2]          (west apex)
        1: [0, 2, 3]       (west gateway, connects to bridge)
        2: [0, 1, 3]       (west gateway, connects to bridge)
        3: [1, 2, 4]       (west bridge / chokepoint)
        4: [3, 5, 6]       (east bridge / chokepoint)
        5: [4, 6, 7]       (east gateway, connects from bridge)
        6: [4, 5, 7]       (east gateway, connects from bridge)
        7: [5, 6]          (east apex)

    Regions:
        west   = [0, 1, 2]  bonus 4
        bridge = [3, 4]     bonus 2
        east   = [5, 6, 7]  bonus 4

    Starting ownership:
        agent_0: [0, 1, 2, 3]  (west cluster + west bridge)
        agent_1: [4, 5, 6, 7]  (east bridge + east cluster)
    """
    adjacency_list = {
        0: [1, 2],
        1: [0, 2, 3],
        2: [0, 1, 3],
        3: [1, 2, 4],
        4: [3, 5, 6],
        5: [4, 6, 7],
        6: [4, 5, 7],
        7: [5, 6],
    }
    n_territories = 8

    # Build adjacency matrix
    adjacency_matrix = np.zeros((n_territories, n_territories), dtype=np.int8)
    for source, neighbors in adjacency_list.items():
        for dest in neighbors:
            adjacency_matrix[source, dest] = 1

    # agent_0: west cluster [0,1,2] + west bridge [3]
    # agent_1: east bridge [4] + east cluster [5,6,7]
    initial_ownership = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int8)

    regions = {
        'west':   [0, 1, 2],
        'bridge': [3, 4],
        'east':   [5, 6, 7],
    }

    region_bonuses = {
        'west':   4,
        'bridge': 2,
        'east':   4,
    }

    return MapConfig(
        n_territories=n_territories,
        adjacency_list=adjacency_list,
        adjacency_matrix=adjacency_matrix,
        initial_ownership=initial_ownership,
        regions=regions,
        region_bonuses=region_bonuses,
    )


def create_large_10_map():
    """Create a 10-territory hub-and-spoke map with flanking routes.

    Two triangular "continent" clusters in the north and south, connected by
    a narrow two-territory corridor plus two diagonal flank shortcuts (8 and 9).
    The corridor (3-4) is the main highway; flanks (8, 9) reward lateral play.

    Layout:
        [0]---[1]---[2]        North continent (triangle: 0-1-2)
         |              \\
        [3]   [8]        [9]   Corridor (3-4) and flanks (8, 9)
         |    |           |
        [4]  [6]---[5]         South continent (triangle: 5-6-7)
          \\  |×  /
           [7]

    Exact adjacency:
        0: [1, 2, 3]       (north-west, enters corridor)
        1: [0, 2, 8]       (north-center, enters west flank)
        2: [0, 1, 9]       (north-east, enters east flank)
        3: [0, 4]          (corridor north / chokepoint)
        4: [3, 7]          (corridor south / chokepoint)
        5: [6, 7, 9]       (south-east, exits east flank)
        6: [5, 7, 8]       (south-center, exits west flank)
        7: [4, 5, 6]       (south-west, exits corridor)
        8: [1, 6]          (west flank shortcut)
        9: [2, 5]          (east flank shortcut)

    Regions:
        north    = [0, 1, 2]  bonus 4
        south    = [5, 6, 7]  bonus 4
        corridor = [3, 4]     bonus 3
        (flanks 8 and 9 are contested standalone territories)

    Starting ownership:
        agent_0: [0, 1, 2, 3, 8]  (north continent + corridor entrance + west flank)
        agent_1: [4, 5, 6, 7, 9]  (corridor exit + south continent + east flank)
    """
    adjacency_list = {
        0: [1, 2, 3],
        1: [0, 2, 8],
        2: [0, 1, 9],
        3: [0, 4],
        4: [3, 7],
        5: [6, 7, 9],
        6: [5, 7, 8],
        7: [4, 5, 6],
        8: [1, 6],
        9: [2, 5],
    }
    n_territories = 10

    # Build adjacency matrix
    adjacency_matrix = np.zeros((n_territories, n_territories), dtype=np.int8)
    for source, neighbors in adjacency_list.items():
        for dest in neighbors:
            adjacency_matrix[source, dest] = 1

    # agent_0: [0,1,2,3,8] (north + corridor entrance + west flank)
    # agent_1: [4,5,6,7,9] (corridor exit + south + east flank)
    initial_ownership = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 1], dtype=np.int8)

    regions = {
        'north':    [0, 1, 2],
        'corridor': [3, 4],
        'south':    [5, 6, 7],
    }

    region_bonuses = {
        'north':    4,
        'corridor': 3,
        'south':    4,
    }

    return MapConfig(
        n_territories=n_territories,
        adjacency_list=adjacency_list,
        adjacency_matrix=adjacency_matrix,
        initial_ownership=initial_ownership,
        regions=regions,
        region_bonuses=region_bonuses,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MapRegistry.register("simple_6", create_simple_6_map)
MapRegistry.register("basic_6", create_simple_6_map)   # alias for simple_6
MapRegistry.register("medium_8", create_medium_8_map)
MapRegistry.register("large_10", create_large_10_map)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _check_connected(adjacency_list, n_territories):
    """Return True if the graph is connected (all nodes reachable from node 0)."""
    visited = set()
    stack = [0]
    visited.add(0)
    while stack:
        node = stack.pop()
        for neighbor in adjacency_list.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append(neighbor)
    return len(visited) == n_territories


def _check_bidirectional(adjacency_list):
    """Return list of (src, dst) pairs that are not bidirectional."""
    issues = []
    for src, neighbors in adjacency_list.items():
        for dst in neighbors:
            if src not in adjacency_list.get(dst, []):
                issues.append((src, dst))
    return issues


def _check_region_ids(regions, n_territories):
    """Return list of (region_name, territory_id) pairs with out-of-range IDs."""
    issues = []
    for name, territories in regions.items():
        for t in territories:
            if t < 0 or t >= n_territories:
                issues.append((name, t))
    return issues


def validate_all_maps():
    """Validate every registered map for connectivity, bidirectionality, and region ID validity.

    Run directly::

        python parallel_risk/env/map_config.py
    """
    all_ok = True
    for name in sorted(MapRegistry.list_maps()):
        config = MapRegistry.get(name)
        errors = []

        if not _check_connected(config.adjacency_list, config.n_territories):
            errors.append("graph is not connected")

        for src, dst in _check_bidirectional(config.adjacency_list):
            errors.append(f"non-bidirectional edge: {src} -> {dst} (missing reverse)")

        for region_name, tid in _check_region_ids(config.regions, config.n_territories):
            errors.append(f"region '{region_name}' contains invalid territory id {tid}")

        status = "OK" if not errors else "FAILED"
        print(f"  {name}: {status}")
        for err in errors:
            print(f"    ERROR: {err}")
        if errors:
            all_ok = False

    if all_ok:
        print("All maps validated successfully.")
    else:
        print("One or more maps failed validation.")
    return all_ok


if __name__ == "__main__":
    validate_all_maps()
