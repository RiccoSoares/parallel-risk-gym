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

        West triangle              Bridge              East triangle
              [0]                                             [7]
             /   \\                                          /   \\
           [1]---[2]                                       [5]---[6]
             \\  /                                            \\  /
              \\/                                              \\/
              [3] ----------------- [4]
             (west                  (east
              gate)                  gate)

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

        Main highway (corridor)          North triangle (0-1-2)
                                                 [1]
              [0]                                / \\
               |                              [0]---[2]
              [3]                             /       \\
               |                             (edge to [3])
              [4]
               |                          South triangle (5-6-7)
              [7]                                 [7]
                                                  / \\
                                               [6]---[5]
                                                (edge from [4])

        Flank shortcuts (diagonal routes bypassing the corridor):
              [1] --- [8] --- [6]     west flank via node 8
              [2] --- [9] --- [5]     east flank via node 9

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
# Symmetric-map helpers
# ---------------------------------------------------------------------------

def _mirror_half(n, half_adj_list, cross_edges):
    """Build a symmetric adjacency structure from one half of the map.

    Node convention: agent_0 owns [0, n//2), agent_1 owns [n//2, n).
    Mirror pairing: i <-> n-1-i (so 0<->n-1, 1<->n-2, ...).

    Args:
        n: total territories (must be even).
        half_adj_list: {i: [neighbors]} with i and each neighbor in [0, n//2).
            Only lists edges within the agent_0 half. Mirrored edges on the
            agent_1 half are added automatically.
        cross_edges: iterable of (i, j) with i in agent_0 half, j in agent_1 half.
            Each is auto-mirrored to (mirror(i), mirror(j)) unless the pair is
            self-mirroring (i, mirror(i)).

    Returns:
        (adjacency_list, adjacency_matrix, initial_ownership) — regions and
        region_bonuses are left to the caller since some regions may straddle
        the mirror axis.
    """
    assert n % 2 == 0, "n must be even for _mirror_half"
    mirror = lambda i: n - 1 - i
    half = n // 2

    adjacency_list = {i: [] for i in range(n)}

    def add_edge(a, b):
        if b not in adjacency_list[a]:
            adjacency_list[a].append(b)
        if a not in adjacency_list[b]:
            adjacency_list[b].append(a)

    for src, neighbors in half_adj_list.items():
        assert src < half, f"half_adj_list source {src} must be in [0, {half})"
        for dst in neighbors:
            assert dst < half, f"half_adj_list dest {dst} must be in [0, {half})"
            add_edge(src, dst)
            add_edge(mirror(src), mirror(dst))

    for (i, j) in cross_edges:
        assert i < half <= j, (
            f"cross_edges pair ({i}, {j}) must have first index in agent_0 half"
        )
        add_edge(i, j)
        mi, mj = mirror(i), mirror(j)
        add_edge(mi, mj)

    for k in adjacency_list:
        adjacency_list[k] = sorted(adjacency_list[k])

    adjacency_matrix = np.zeros((n, n), dtype=np.int8)
    for src, neighbors in adjacency_list.items():
        for dst in neighbors:
            adjacency_matrix[src, dst] = 1

    initial_ownership = np.zeros(n, dtype=np.int8)
    initial_ownership[half:] = 1

    return adjacency_list, adjacency_matrix, initial_ownership


# ---------------------------------------------------------------------------
# New symmetric maps
# ---------------------------------------------------------------------------

def create_triangle_6_map():
    """Prism graph (K3 x K2): two triangles connected by 3 parallel edges.

    Layout:
              [0]---[1]                          [5]---[4]
                \\  /                              \\  /
                 \\/                                \\/
                 [2] ------ (cross edges) ------ [3]

    Cross edges: 0-5, 1-4, 2-3 (each pair is a mirror-fixed edge).

    All 6 nodes have uniform degree 3 (dense small graph); no chokepoints.
    Contrasts with simple_6 (grid, mixed degrees, 3 regions) at the same size.
    """
    n = 6
    half_adj = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
    cross_edges = [(0, 5), (1, 4), (2, 3)]
    adjacency_list, adjacency_matrix, initial_ownership = _mirror_half(
        n, half_adj, cross_edges
    )
    regions = {
        'left':  [0, 1, 2],
        'right': [3, 4, 5],
    }
    region_bonuses = {'left': 4, 'right': 4}
    return MapConfig(
        n_territories=n,
        adjacency_list=adjacency_list,
        adjacency_matrix=adjacency_matrix,
        initial_ownership=initial_ownership,
        regions=regions,
        region_bonuses=region_bonuses,
    )


def create_ring_6_map():
    """Cycle graph C6: six nodes in a ring, uniform degree 2 (minimum degree).

    Layout:
              [0] --- [1]
             /           \\
          [5]             [2]
             \\           /
              [4] --- [3]

    Sparse cycle; every node has exactly two neighbors. Tests GNN behavior
    on low-degree topology where message passing must traverse a long path.
    """
    n = 6
    half_adj = {0: [1], 1: [0, 2], 2: [1]}   # path 0-1-2 in agent_0 half
    cross_edges = [(0, 5), (2, 3)]           # close the ring via cross edges
    adjacency_list, adjacency_matrix, initial_ownership = _mirror_half(
        n, half_adj, cross_edges
    )
    regions = {
        'left':  [0, 1, 2],
        'right': [3, 4, 5],
    }
    region_bonuses = {'left': 3, 'right': 3}
    return MapConfig(
        n_territories=n,
        adjacency_list=adjacency_list,
        adjacency_matrix=adjacency_matrix,
        initial_ownership=initial_ownership,
        regions=regions,
        region_bonuses=region_bonuses,
    )


def create_star_8_map():
    """Two 'star clusters': each side is a K4 (triangle + apex hub); the two
    hubs are connected by a single bridge.

    Layout:
        Left K4 (0,1,2,3)               Right K4 (4,5,6,7)
              [0]                                [7]
             /|\\                                /|\\
            / | \\                              / | \\
          [1]-|-[2]                          [5]-|-[6]
            \\|/                                \\|/
             [3] ------- (bridge) --------- [4]

    Contrasts with medium_8 (two triangles + separate bridge nodes 3-4):
    here each hub (3, 4) is INSIDE its own K4 rather than being a distinct
    bridge node. Hubs are the only nodes touching the other side.
    """
    n = 8
    half_adj = {
        0: [1, 2, 3],
        1: [0, 2, 3],
        2: [0, 1, 3],
        3: [0, 1, 2],
    }
    cross_edges = [(3, 4)]  # single bridge between the two hubs
    adjacency_list, adjacency_matrix, initial_ownership = _mirror_half(
        n, half_adj, cross_edges
    )
    regions = {
        'left':  [0, 1, 2, 3],
        'right': [4, 5, 6, 7],
        'hubs':  [3, 4],
    }
    region_bonuses = {'left': 3, 'right': 3, 'hubs': 2}
    return MapConfig(
        n_territories=n,
        adjacency_list=adjacency_list,
        adjacency_matrix=adjacency_matrix,
        initial_ownership=initial_ownership,
        regions=regions,
        region_bonuses=region_bonuses,
    )


def create_double_hub_8_map():
    """Two mirrored hubs with cross-spoke bridges (dense mid-degree graph).

    Layout:
        Left web (0,1,2,3)               Right web (4,5,6,7)

              [0] --- [1] --- [2]           [5] --- [6] --- [7]
                \\    |    /                    \\    |    /
                  \\  |  /                        \\  |  /
                    [3] ------- (hubs) ---- [4]

        Cross bridges: 0-7 and 2-5 in addition to the 3-4 hub bridge.

    Each side is a "web" of 3 rim nodes connected to a hub and to two rim
    neighbors. Multiple cross-half routes (through hubs OR rim shortcuts)
    ensure no single chokepoint. Higher connectivity than star_8.
    """
    n = 8
    half_adj = {
        0: [1, 3],
        1: [0, 2, 3],
        2: [1, 3],
        3: [0, 1, 2],
    }
    cross_edges = [(0, 7), (2, 5), (3, 4)]
    adjacency_list, adjacency_matrix, initial_ownership = _mirror_half(
        n, half_adj, cross_edges
    )
    regions = {
        'left':  [0, 1, 2, 3],
        'right': [4, 5, 6, 7],
    }
    region_bonuses = {'left': 5, 'right': 5}
    return MapConfig(
        n_territories=n,
        adjacency_list=adjacency_list,
        adjacency_matrix=adjacency_matrix,
        initial_ownership=initial_ownership,
        regions=regions,
        region_bonuses=region_bonuses,
    )


def create_hex_grid_10_map():
    """2x5 grid with SE diagonal edges (hex-like) — 10 territories, 4 regions.

    Layout:
        [0] - [1] - [2] - [3] - [4]        top row (agent_0)
         | \\  | \\  | \\  | \\  |
        [5] - [6] - [7] - [8] - [9]        bottom row (agent_1)

    Vertical edges: 0-5, 1-6, 2-7, 3-8, 4-9
    Horizontal edges: 0-1, 1-2, 2-3, 3-4, 5-6, 6-7, 7-8, 8-9
    Diagonal edges (SE): 0-6, 1-7, 2-8, 3-9

    Breaks the 3-region schema shared by all previous maps: uses 4 regions
    of different sizes/bonuses. Central column nodes (2, 7) belong to the
    larger regions; edge nodes belong to the smaller regions.
    """
    n = 10
    half_adj = {
        0: [1],
        1: [0, 2],
        2: [1, 3],
        3: [2, 4],
        4: [3],
    }
    # Vertical edges (i, i+5) and SE diagonals (i, i+6):
    cross_edges = [
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9),   # verticals
        (0, 6), (1, 7), (2, 8), (3, 9),           # SE diagonals
    ]
    adjacency_list, adjacency_matrix, initial_ownership = _mirror_half(
        n, half_adj, cross_edges
    )
    regions = {
        'north_left':  [0, 1, 2],
        'north_right': [3, 4],
        'south_left':  [5, 6],
        'south_right': [7, 8, 9],
    }
    region_bonuses = {
        'north_left':  3,
        'north_right': 2,
        'south_left':  2,
        'south_right': 3,
    }
    return MapConfig(
        n_territories=n,
        adjacency_list=adjacency_list,
        adjacency_matrix=adjacency_matrix,
        initial_ownership=initial_ownership,
        regions=regions,
        region_bonuses=region_bonuses,
    )


def create_dense_12_map():
    """Two K6 cliques joined by three mirror-symmetric cross bridges.

    Layout:
        Left K6: nodes {0..5}, all pairs connected (deg 5 within half).
        Right K6: nodes {6..11}, mirror of left.
        Cross bridges: 0-11, 2-9, 4-7 (each is a mirror-fixed edge).

    Dense mid-size graph with no chokepoints — every node has degree 5 or 6.
    Tests whether GNN benefits from dense adjacency (many short paths) vs
    the sparse chokepoint-heavy maps.

    Regions:
        left = [0..5] bonus 4
        right = [6..11] bonus 4
        bridges = [0, 4, 7, 11]  bonus 3  (straddles both halves; can only be
            fully controlled by conquering opponent's bridge nodes)
    """
    n = 12
    # K6 on the agent_0 side
    half_adj = {i: [j for j in range(6) if j != i] for i in range(6)}
    cross_edges = [(0, 11), (2, 9), (4, 7)]
    adjacency_list, adjacency_matrix, initial_ownership = _mirror_half(
        n, half_adj, cross_edges
    )
    regions = {
        'left':    [0, 1, 2, 3, 4, 5],
        'right':   [6, 7, 8, 9, 10, 11],
        'bridges': [0, 4, 7, 11],
    }
    region_bonuses = {'left': 4, 'right': 4, 'bridges': 3}
    return MapConfig(
        n_territories=n,
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
MapRegistry.register("triangle_6", create_triangle_6_map)
MapRegistry.register("ring_6", create_ring_6_map)
MapRegistry.register("star_8", create_star_8_map)
MapRegistry.register("double_hub_8", create_double_hub_8_map)
MapRegistry.register("hex_grid_10", create_hex_grid_10_map)
MapRegistry.register("dense_12", create_dense_12_map)


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


def _check_symmetry(config):
    """Verify the map is symmetric under a player-swapping graph automorphism.

    Returns (ok, message). A valid automorphism pi must:
      (a) map each agent_0 territory to an agent_1 territory and vice versa,
      (b) preserve adjacency: A[i, j] == A[pi(i), pi(j)] for all i, j,
      (c) map each region to another region of equal bonus (regions may permute
          among themselves as long as bonuses match).

    Algorithm: signature-based backtracking. For each node, compute
      (target_owner, degree, sorted bonuses of regions containing it).
    Candidate mappings are restricted to nodes with the swapped owner and
    matching (degree, region-bonus-multiset). This prunes the search
    aggressively; for n <= 20 with balanced ownership it stays well under 10^6
    permutations in practice.
    """
    n = config.n_territories
    ownership = config.initial_ownership
    adj = config.adjacency_matrix

    # Per-node region-bonus multiset (tuple, hashable).
    node_bonuses = [[] for _ in range(n)]
    for name, territories in config.regions.items():
        bonus = config.region_bonuses.get(name, 0)
        for t in territories:
            node_bonuses[t].append(bonus)
    node_bonuses = [tuple(sorted(b)) for b in node_bonuses]

    degrees = [int(adj[i].sum()) for i in range(n)]

    # For each node i, candidate images are nodes j with:
    #   ownership[j] == 1 - ownership[i], same degree, same region-bonus multiset.
    candidates = [[] for _ in range(n)]
    for i in range(n):
        target_owner = 1 - int(ownership[i])
        for j in range(n):
            if (int(ownership[j]) == target_owner
                    and degrees[j] == degrees[i]
                    and node_bonuses[j] == node_bonuses[i]):
                candidates[i].append(j)

    # Early exit: any node with no candidate images kills symmetry.
    for i, cs in enumerate(candidates):
        if not cs:
            return False, (
                f"node {i} (owner={int(ownership[i])}, deg={degrees[i]}, "
                f"region-bonuses={node_bonuses[i]}) has no valid image"
            )

    # Search order: most-constrained-first (fewest candidates) to prune early.
    search_order = sorted(range(n), key=lambda i: len(candidates[i]))

    pi = [-1] * n
    used = [False] * n

    def check_partial_edges(i, image):
        # Every already-assigned node j: if there is an edge i-j (or j-i)
        # in the source, there must be a matching edge image-pi(j) in the target.
        for j in range(n):
            if pi[j] == -1 or j == i:
                continue
            if adj[i, j] != adj[image, pi[j]]:
                return False
            if adj[j, i] != adj[pi[j], image]:
                return False
        return True

    def check_regions(perm):
        # Every region must map to a region of equal bonus.
        # A region R = {t1, t2, ...} maps to {pi(t1), pi(t2), ...}; verify that
        # image set matches some region with the same bonus.
        region_signatures = {}
        for name, territories in config.regions.items():
            key = frozenset(territories)
            region_signatures.setdefault(key, []).append(name)
        # Build set of (frozenset, bonus) pairs for lookup.
        target_regions = {}
        for name, territories in config.regions.items():
            target_regions.setdefault(frozenset(territories), config.region_bonuses.get(name, 0))
        for name, territories in config.regions.items():
            image_set = frozenset(perm[t] for t in territories)
            src_bonus = config.region_bonuses.get(name, 0)
            if image_set not in target_regions or target_regions[image_set] != src_bonus:
                return False
        return True

    def backtrack(k):
        if k == n:
            return check_regions(pi)
        i = search_order[k]
        for image in candidates[i]:
            if used[image]:
                continue
            pi[i] = image
            if check_partial_edges(i, image):
                used[image] = True
                if backtrack(k + 1):
                    return True
                used[image] = False
            pi[i] = -1
        return False

    if backtrack(0):
        return True, f"pi = {pi}"
    return False, "no player-swapping automorphism found"


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

        sym_ok, sym_msg = _check_symmetry(config)
        if not sym_ok:
            errors.append(f"symmetry: {sym_msg}")

        status = "OK" if not errors else "FAILED"
        detail = f"  ({sym_msg})" if sym_ok else ""
        print(f"  {name}: {status}{detail}")
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
