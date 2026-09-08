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
# New medium maps (15-25 territories)
# ---------------------------------------------------------------------------

def create_hex_grid_18_map():
    """3x6 hex-like grid (rectangular grid + SE diagonals).

    Layout (rows top to bottom, mirror axis is horizontal midline):

        row 0 (agent_0):    [ 0] [ 1] [ 2] [ 3] [ 4] [ 5]
        row 1 (mixed):      [ 6] [ 7] [ 8] | [ 9] [10] [11]
        row 2 (agent_1):    [12] [13] [14] [15] [16] [17]

    Mirror pairing: i <-> 17 - i (row 0 <-> row 2 reversed; row 1 splits at
    the vertical midline). Row 1 nodes {6,7,8} start with agent_0; {9,10,11}
    with agent_1, giving a natural front line down the middle.

    Edges (all bidirectional):
        Horizontal (per row): 0-1, 1-2, 2-3, 3-4, 4-5;  6-7, 7-8, 8-9, 9-10,
            10-11;  12-13, 13-14, 14-15, 15-16, 16-17.
        Vertical (row_r - row_{r+1}): (0,6)..(5,11);  (6,12)..(11,17).
        SE diagonals (i, i+7) row 0 -> row 1: 0-7, 1-8, 2-9, 3-10, 4-11.
        SE diagonals row 1 -> row 2: 6-13, 7-14, 8-15, 9-16, 10-17.

    Regions:
        north  = [0..5]   bonus 5 (top row, mirror-pair of south)
        middle = [6..11]  bonus 4 (contested middle row, self-mirror)
        south  = [12..17] bonus 5 (bottom row)
    """
    n = 18
    half_adj = {
        0: [1, 6, 7],
        1: [0, 2, 7, 8],
        2: [1, 3, 8],
        3: [2, 4],
        4: [3, 5],
        5: [4],
        6: [0, 1, 7],
        7: [1, 2, 6, 8],
        8: [2, 7],
    }
    # Cross edges: connect agent_0 half (< 9) to agent_1 half (>= 9).
    # Verticals row1->row2 (partial), row0->row1 middle, SE diagonals.
    cross_edges = [
        # Row 1 middle horizontal
        (8, 9),
        # Verticals row0 -> row1 (right half of row 1)
        (3, 10), (4, 11),
        # Verticals row1 -> row2 (agent_0 row-1 nodes to agent_1 row-2 nodes)
        (6, 12), (7, 13), (8, 14),
        # SE diagonals row0 -> row1 (right half)
        (3, 9), (4, 10), (5, 11),
        # SE diagonals row1 -> row2 (agent_0 row-1 nodes to agent_1 row-2 nodes)
        (6, 13), (7, 14), (8, 15),
    ]
    adjacency_list, adjacency_matrix, initial_ownership = _mirror_half(
        n, half_adj, cross_edges
    )
    regions = {
        'north':  [0, 1, 2, 3, 4, 5],
        'middle': [6, 7, 8, 9, 10, 11],
        'south':  [12, 13, 14, 15, 16, 17],
    }
    region_bonuses = {'north': 5, 'middle': 4, 'south': 5}
    return MapConfig(
        n_territories=n,
        adjacency_list=adjacency_list,
        adjacency_matrix=adjacency_matrix,
        initial_ownership=initial_ownership,
        regions=regions,
        region_bonuses=region_bonuses,
    )


def create_grid_20_map():
    """4x5 rectangular grid (20 territories, no diagonals).

    Layout:
        row 0 (agent_0):  [ 0] [ 1] [ 2] [ 3] [ 4]
        row 1 (agent_0):  [ 5] [ 6] [ 7] [ 8] [ 9]
        row 2 (agent_1):  [10] [11] [12] [13] [14]
        row 3 (agent_1):  [15] [16] [17] [18] [19]

    Mirror pairing i <-> 19 - i pairs row 0 <-> row 3 reversed and
    row 1 <-> row 2 reversed. Front line runs between rows 1 and 2.

    Edges:
        Horizontal: standard row edges per row.
        Vertical:   (0,5), (1,6) ... (14,19).

    Regions (all bonus scaled by size):
        north      = [0..4]    bonus 3
        north_mid  = [5..9]    bonus 4  (agent_0 front line)
        south_mid  = [10..14]  bonus 4  (agent_1 front line, mirror of north_mid)
        south      = [15..19]  bonus 3  (mirror of north)
    """
    n = 20
    half_adj = {
        # row 0 horizontal
        0: [1, 5],
        1: [0, 2, 6],
        2: [1, 3, 7],
        3: [2, 4, 8],
        4: [3, 9],
        # row 1 horizontal
        5: [0, 6],
        6: [1, 5, 7],
        7: [2, 6, 8],
        8: [3, 7, 9],
        9: [4, 8],
    }
    # Cross edges: verticals row1 -> row2 (all self-mirror since 5+14=19 etc.)
    cross_edges = [
        (5, 14), (6, 13), (7, 12), (8, 11), (9, 10),
    ]
    adjacency_list, adjacency_matrix, initial_ownership = _mirror_half(
        n, half_adj, cross_edges
    )
    regions = {
        'north':      [0, 1, 2, 3, 4],
        'north_mid':  [5, 6, 7, 8, 9],
        'south_mid':  [10, 11, 12, 13, 14],
        'south':      [15, 16, 17, 18, 19],
    }
    region_bonuses = {'north': 3, 'north_mid': 4, 'south_mid': 4, 'south': 3}
    return MapConfig(
        n_territories=n,
        adjacency_list=adjacency_list,
        adjacency_matrix=adjacency_matrix,
        initial_ownership=initial_ownership,
        regions=regions,
        region_bonuses=region_bonuses,
    )


def create_corridor_20_map():
    """20-territory corridor + flanks: two wheel-continents joined by a corridor.

    Layout:

        NORTH continent (0..5) = wheel W5 (pentagon 0-1-2-3-4 + hub 5)

              [1]---[2]
              /  \\ / \\
             [0]--[5]--[3]
              \\  / \\ /
              [4]---(3)          (edges 5-0, 5-1, 5-2, 5-3, 5-4 not all drawn)

        Corridor chain: 5 - 6 - 7 - 8 - 9 - 10 - 11 - 12 - 13 - 14
        (agent_0 nodes 6-9, agent_1 nodes 10-13, chokepoint bridge at 9-10.)

        SOUTH continent (14..19) = mirror wheel (hub 14, rim 15-19).

    Regions:
        north     = [0..5]     bonus 5
        south     = [14..19]   bonus 5  (mirror of north)
        corridor  = [6..13]    bonus 4  (self-mirror; owning it grants a solid income)
    """
    n = 20
    half_adj = {
        # Pentagon rim
        0: [1, 4, 5],
        1: [0, 2, 5],
        2: [1, 3, 5],
        3: [2, 4, 5],
        4: [0, 3, 5],
        # Hub 5 (connects to whole pentagon + corridor entry)
        5: [0, 1, 2, 3, 4, 6],
        # Corridor chain (agent_0 side)
        6: [5, 7],
        7: [6, 8],
        8: [7, 9],
        9: [8],
    }
    # Cross edges: corridor bridge in the middle (self-mirror)
    cross_edges = [
        (9, 10),
    ]
    adjacency_list, adjacency_matrix, initial_ownership = _mirror_half(
        n, half_adj, cross_edges
    )
    regions = {
        'north':    [0, 1, 2, 3, 4, 5],
        'corridor': [6, 7, 8, 9, 10, 11, 12, 13],
        'south':    [14, 15, 16, 17, 18, 19],
    }
    region_bonuses = {'north': 5, 'corridor': 4, 'south': 5}
    return MapConfig(
        n_territories=n,
        adjacency_list=adjacency_list,
        adjacency_matrix=adjacency_matrix,
        initial_ownership=initial_ownership,
        regions=regions,
        region_bonuses=region_bonuses,
    )


def create_corridor_22_map():
    """22-territory map with TWO parallel corridors between two continents.

    Two chokepoints instead of one — encourages splitting forces or committing
    to a single axis of advance.

    Layout:

        NORTH continent (0..4) = pentagon with hub structure:
            0-1-2-3-4 (pentagon), plus edges 0-2 and 2-4 for extra connectivity.
            Node 2 (apex) is the internal hub; nodes 2 and 4 gate the corridors.

        Upper corridor (agent_0 side):  2 - 5 - 6 - 7
        Lower corridor (agent_0 side):  4 - 8 - 9 - 10
        Bridge to agent_1: 7 - 14 (upper), 10 - 11 (lower).
        Upper corridor (agent_1 side): 14 - 15 - 16 - 19  (mirror of 7-6-5-2)
        Lower corridor (agent_1 side): 11 - 12 - 13 - 17  (mirror of 10-9-8-4)

        SOUTH continent (17..21) = mirror of north.

    Regions:
        north           = [0..4]        bonus 4
        south           = [17..21]      bonus 4  (mirror)
        upper_corridor  = [5,6,7,14,15,16] bonus 3 (self-mirror)
        lower_corridor  = [8,9,10,11,12,13] bonus 3 (self-mirror)
    """
    n = 22
    half_adj = {
        # North continent (pentagon + 2 chords)
        0: [1, 4],
        1: [0, 2],
        2: [1, 3, 0, 5],   # apex + upper corridor entry
        3: [2, 4],
        4: [0, 3, 2, 8],   # + lower corridor entry (also chord 2-4)
        # Upper corridor
        5: [2, 6],
        6: [5, 7],
        7: [6],
        # Lower corridor
        8: [4, 9],
        9: [8, 10],
        10: [9],
    }
    cross_edges = [
        (7, 14),   # upper corridor bridge (self-mirror: mirror(7)=14)
        (10, 11),  # lower corridor bridge (self-mirror: mirror(10)=11)
    ]
    adjacency_list, adjacency_matrix, initial_ownership = _mirror_half(
        n, half_adj, cross_edges
    )
    regions = {
        'north':           [0, 1, 2, 3, 4],
        'south':           [17, 18, 19, 20, 21],
        'upper_corridor':  [5, 6, 7, 14, 15, 16],
        'lower_corridor':  [8, 9, 10, 11, 12, 13],
    }
    region_bonuses = {
        'north': 4, 'south': 4,
        'upper_corridor': 3, 'lower_corridor': 3,
    }
    return MapConfig(
        n_territories=n,
        adjacency_list=adjacency_list,
        adjacency_matrix=adjacency_matrix,
        initial_ownership=initial_ownership,
        regions=regions,
        region_bonuses=region_bonuses,
    )


def create_hub_spoke_16_map():
    """16-territory dual-hub-per-side map (compact hub-and-spoke).

    Each side has two hubs joined together, with a small ring of 3 spokes
    around each hub. Cross-half connections at all three natural axes.

    Layout (agent_0 half nodes 0..7):

            Hub A (0)                       Hub B (4)
             /|\\                            /|\\
            / | \\                          / | \\
          [1]-[2]-[3]                    [5]-[6]-[7]
                \\____________________________/
                       0-4 internal link (both hubs connected)

    Mirror pairs: 0<->15 (hub A left-right), 4<->11 (hub B), etc.

    Cross bridges (self-mirror): 3-12 (spoke bridge on A side),
        4-11 (hub B bridge), 7-8 (spoke bridge on B side).

    Regions:
        left_A  = [0,1,2,3] bonus 3     ↔  right_A = [12,13,14,15] bonus 3
        left_B  = [4,5,6,7] bonus 3     ↔  right_B = [8,9,10,11] bonus 3
    """
    n = 16
    half_adj = {
        0: [1, 2, 3, 4],   # hub A + link to hub B
        1: [0, 2],
        2: [0, 1, 3],
        3: [0, 2],
        4: [0, 5, 6, 7],   # hub B + link to hub A
        5: [4, 6],
        6: [4, 5, 7],
        7: [4, 6],
    }
    cross_edges = [
        (3, 12),  # spoke-to-spoke bridge on the "A" axis (mirror(3)=12)
        (4, 11),  # hub-B bridge (mirror(4)=11)
        (7, 8),   # spoke-to-spoke bridge on the "B" axis (mirror(7)=8)
    ]
    adjacency_list, adjacency_matrix, initial_ownership = _mirror_half(
        n, half_adj, cross_edges
    )
    regions = {
        'left_A':  [0, 1, 2, 3],
        'left_B':  [4, 5, 6, 7],
        'right_B': [8, 9, 10, 11],
        'right_A': [12, 13, 14, 15],
    }
    region_bonuses = {'left_A': 3, 'left_B': 3, 'right_B': 3, 'right_A': 3}
    return MapConfig(
        n_territories=n,
        adjacency_list=adjacency_list,
        adjacency_matrix=adjacency_matrix,
        initial_ownership=initial_ownership,
        regions=regions,
        region_bonuses=region_bonuses,
    )


def create_dual_hub_20_map():
    """20-territory dual-hub-per-side map with 4-spoke rings.

    Larger version of hub_spoke_16: each side has two hubs, and each hub
    supports a 4-node ring of spokes. Multiple cross-half routes.

    Layout (agent_0 half nodes 0..9):

        Hub A (0)  spokes {1,2,3,4} arranged in a 4-cycle
        Hub B (5)  spokes {6,7,8,9} arranged in a 4-cycle
        Internal link: 0-5 (hub-hub)

    Cross bridges (all self-mirror under i <-> 19-i):
        (0,19): hub A cross-bridge
        (4,15): spoke bridge on A side
        (5,14): hub B cross-bridge
        (9,10): spoke bridge on B side

    Regions:
        left_A  = [0,1,2,3,4] bonus 3  ↔  right_A = [15..19] bonus 3
        left_B  = [5,6,7,8,9] bonus 3  ↔  right_B = [10..14] bonus 3
    """
    n = 20
    half_adj = {
        # Hub A + spoke ring
        0: [1, 2, 3, 4, 5],
        1: [0, 2, 4],
        2: [0, 1, 3],
        3: [0, 2, 4],
        4: [0, 1, 3],
        # Hub B + spoke ring
        5: [0, 6, 7, 8, 9],
        6: [5, 7, 9],
        7: [5, 6, 8],
        8: [5, 7, 9],
        9: [5, 6, 8],
    }
    cross_edges = [
        (0, 19),  # hub A bridge
        (4, 15),  # A spoke bridge
        (5, 14),  # hub B bridge
        (9, 10),  # B spoke bridge
    ]
    adjacency_list, adjacency_matrix, initial_ownership = _mirror_half(
        n, half_adj, cross_edges
    )
    regions = {
        'left_A':  [0, 1, 2, 3, 4],
        'left_B':  [5, 6, 7, 8, 9],
        'right_B': [10, 11, 12, 13, 14],
        'right_A': [15, 16, 17, 18, 19],
    }
    region_bonuses = {'left_A': 3, 'left_B': 3, 'right_B': 3, 'right_A': 3}
    return MapConfig(
        n_territories=n,
        adjacency_list=adjacency_list,
        adjacency_matrix=adjacency_matrix,
        initial_ownership=initial_ownership,
        regions=regions,
        region_bonuses=region_bonuses,
    )


def create_ring_cross_18_map():
    """18-node ring (C18) with mirror-symmetric interior chords.

    Layout:

              0 -- 1 -- 2 -- 3 -- 4 -- 5 -- 6 -- 7 -- 8
              |                                        |
             17                                        9
              |                                        |
             16 - 15 - 14 - 13 - 12 - 11 - 10 -------/

    Ring order: 0-1-...-17-0. Cross chords (all self-mirror under i <-> 17-i):
        (2, 15), (4, 13), (6, 11).

    The chords cut down the "diameter" of the ring while preserving reflective
    symmetry. Regions form three self-mirror arcs of size 6.

    Regions:
        north_arc  = [0,1,2,15,16,17]     bonus 4  (self-mirror)
        middle_arc = [3,4,5,12,13,14]     bonus 4  (self-mirror)
        south_arc  = [6,7,8,9,10,11]      bonus 4  (self-mirror)
    """
    n = 18
    half_adj = {
        0: [1],
        1: [0, 2],
        2: [1, 3],
        3: [2, 4],
        4: [3, 5],
        5: [4, 6],
        6: [5, 7],
        7: [6, 8],
        8: [7],
    }
    cross_edges = [
        (0, 17),  # close the ring on the "north" side
        (8, 9),   # close the ring on the "south" side
        (2, 15),  # north chord
        (4, 13),  # middle chord
        (6, 11),  # south chord
    ]
    adjacency_list, adjacency_matrix, initial_ownership = _mirror_half(
        n, half_adj, cross_edges
    )
    regions = {
        'north_arc':  [0, 1, 2, 15, 16, 17],
        'middle_arc': [3, 4, 5, 12, 13, 14],
        'south_arc':  [6, 7, 8, 9, 10, 11],
    }
    region_bonuses = {'north_arc': 4, 'middle_arc': 4, 'south_arc': 4}
    return MapConfig(
        n_territories=n,
        adjacency_list=adjacency_list,
        adjacency_matrix=adjacency_matrix,
        initial_ownership=initial_ownership,
        regions=regions,
        region_bonuses=region_bonuses,
    )


def create_dense_mesh_16_map():
    """16-territory dense mesh: two K4's per side joined by a perfect matching.

    Layout (agent_0 half nodes 0..7):

        Upper K4:   {0, 1, 2, 3}   (fully connected)
        Lower K4:   {4, 5, 6, 7}   (fully connected)
        Matching:   0-4, 1-5, 2-6, 3-7

    Mirror i <-> 15-i pairs {0..7} with {15..8}. Cross bridges (self-mirror):
        (0, 15), (3, 12), (4, 11), (7, 8).

    Highest per-node degree of any map at its size — many redundant paths,
    no chokepoints. Contrasts with the sparser corridor-based maps.

    Regions:
        left_top  = [0,1,2,3] bonus 3   ↔  right_top = [12,13,14,15] bonus 3
        left_bot  = [4,5,6,7] bonus 3   ↔  right_bot = [8,9,10,11] bonus 3
    """
    n = 16
    half_adj = {
        # Upper K4
        0: [1, 2, 3, 4],
        1: [0, 2, 3, 5],
        2: [0, 1, 3, 6],
        3: [0, 1, 2, 7],
        # Lower K4
        4: [0, 5, 6, 7],
        5: [1, 4, 6, 7],
        6: [2, 4, 5, 7],
        7: [3, 4, 5, 6],
    }
    cross_edges = [
        (0, 15), (3, 12), (4, 11), (7, 8),
    ]
    adjacency_list, adjacency_matrix, initial_ownership = _mirror_half(
        n, half_adj, cross_edges
    )
    regions = {
        'left_top':  [0, 1, 2, 3],
        'left_bot':  [4, 5, 6, 7],
        'right_bot': [8, 9, 10, 11],
        'right_top': [12, 13, 14, 15],
    }
    region_bonuses = {
        'left_top': 3, 'left_bot': 3, 'right_bot': 3, 'right_top': 3,
    }
    return MapConfig(
        n_territories=n,
        adjacency_list=adjacency_list,
        adjacency_matrix=adjacency_matrix,
        initial_ownership=initial_ownership,
        regions=regions,
        region_bonuses=region_bonuses,
    )


def create_bipartite_20_map():
    """20-territory bipartite / split-continent map.

    Two mostly-independent 'continents' each straddle the front line and are
    joined internally by a single edge on each side. Encourages committing to
    one continent early or splitting attention.

    Layout (agent_0 half nodes 0..9):

        Continent A (top): pentagon {0,1,2,3,4} with chords 0-2 and 2-4.
        Continent B (bot): pentagon {5,6,7,8,9} with chords 5-7 and 7-9.
        Inter-continent link (agent_0 side): 4 - 5.

    Cross bridges (self-mirror):
        (0, 19)  -- continent A cross-half bridge
        (9, 10)  -- continent B cross-half bridge

    NOTE: continents are only joined by the 4-5 edge (and its mirror 15-14),
    plus the two cross bridges — so control of a whole continent is a real
    strategic goal, not incidental.

    Regions (mirror i <-> 19-i pairs [0,1,2]<->[17,18,19] and [5,6,7]<->[12,13,14]):
        A_left  = [0,1,2]     bonus 2   ↔  A_right = [17,18,19] bonus 2
        B_left  = [5,6,7]     bonus 2   ↔  B_right = [12,13,14] bonus 2
        (Nodes 3,4,8,9 and their mirrors 10,11,15,16 are "frontier" nodes
         without a regional bonus, incentivizing pushes but not guaranteeing
         income.)
    """
    n = 20
    half_adj = {
        # Continent A pentagon + chords
        0: [1, 4, 2],
        1: [0, 2],
        2: [0, 1, 3, 4],
        3: [2, 4],
        4: [0, 2, 3, 5],   # + inter-continent link
        # Continent B pentagon + chords
        5: [4, 6, 9, 7],
        6: [5, 7],
        7: [5, 6, 8, 9],
        8: [7, 9],
        9: [5, 7, 8],
    }
    cross_edges = [
        (0, 19),   # A cross bridge (self-mirror)
        (9, 10),   # B cross bridge (self-mirror)
    ]
    adjacency_list, adjacency_matrix, initial_ownership = _mirror_half(
        n, half_adj, cross_edges
    )
    regions = {
        'A_left':  [0, 1, 2],
        'A_right': [17, 18, 19],
        'B_left':  [5, 6, 7],
        'B_right': [12, 13, 14],
    }
    region_bonuses = {
        'A_left': 2, 'A_right': 2, 'B_left': 2, 'B_right': 2,
    }
    return MapConfig(
        n_territories=n,
        adjacency_list=adjacency_list,
        adjacency_matrix=adjacency_matrix,
        initial_ownership=initial_ownership,
        regions=regions,
        region_bonuses=region_bonuses,
    )


# ---------------------------------------------------------------------------
# New large maps (28-35 territories)
# ---------------------------------------------------------------------------

def create_grid_30_map():
    """5x6 rectangular grid (30 territories, no diagonals).

    Layout (rows top to bottom):

        row 0 (agent_0):  [ 0] [ 1] [ 2] [ 3] [ 4] [ 5]
        row 1 (agent_0):  [ 6] [ 7] [ 8] [ 9] [10] [11]
        row 2 (mixed):    [12] [13] [14] | [15] [16] [17]
        row 3 (agent_1):  [18] [19] [20] [21] [22] [23]
        row 4 (agent_1):  [24] [25] [26] [27] [28] [29]

    Mirror i <-> 29-i pairs row 0 <-> row 4 (reversed), row 1 <-> row 3
    (reversed), row 2 splits vertically at the midline.

    Standard 4-neighbor grid edges (horizontal + vertical).

    Regions:
        r0 = [0..5]    bonus 3   ↔  r4 = [24..29] bonus 3
        r1 = [6..11]   bonus 3   ↔  r3 = [18..23] bonus 3
        center = [12..17] bonus 4 (self-mirror; owning the middle row is valuable)
    """
    n = 30
    half_adj = {
        # row 0 horizontal + row 0->1 vertical
        0: [1, 6],
        1: [0, 2, 7],
        2: [1, 3, 8],
        3: [2, 4, 9],
        4: [3, 5, 10],
        5: [4, 11],
        # row 1 horizontal + row 1->2 vertical (partial: only 12,13,14)
        6: [0, 7, 12],
        7: [1, 6, 8, 13],
        8: [2, 7, 9, 14],
        9: [3, 8, 10],
        10: [4, 9, 11],
        11: [5, 10],
        # row 2 partial horizontal (12-13, 13-14)
        12: [6, 13],
        13: [7, 12, 14],
        14: [8, 13],
    }
    cross_edges = [
        # row 2 midline horizontal
        (14, 15),
        # row 1 -> row 2 vertical (right half of row 2)
        (9, 15), (10, 16), (11, 17),
        # row 2 -> row 3 vertical (agent_0 row-2 nodes to agent_1 row-3 nodes)
        (12, 18), (13, 19), (14, 20),
    ]
    adjacency_list, adjacency_matrix, initial_ownership = _mirror_half(
        n, half_adj, cross_edges
    )
    regions = {
        'row0':   [0, 1, 2, 3, 4, 5],
        'row1':   [6, 7, 8, 9, 10, 11],
        'center': [12, 13, 14, 15, 16, 17],
        'row3':   [18, 19, 20, 21, 22, 23],
        'row4':   [24, 25, 26, 27, 28, 29],
    }
    region_bonuses = {
        'row0': 3, 'row1': 3, 'center': 4, 'row3': 3, 'row4': 3,
    }
    return MapConfig(
        n_territories=n,
        adjacency_list=adjacency_list,
        adjacency_matrix=adjacency_matrix,
        initial_ownership=initial_ownership,
        regions=regions,
        region_bonuses=region_bonuses,
    )


def create_corridor_28_map():
    """28-territory corridor map with dual corridors + wheel continents.

    Layout:

        NORTH continent (0..6): wheel W6 (hexagon 0-1-2-3-4-5 + inner hub 6).
        Upper corridor (agent_0 side): 5 - 7 - 8 - 9 - 10
        Lower corridor (agent_0 side): 4 - 11 - 12 - 13
        SOUTH continent (21..27): mirror of north (hub 21, rim 22..27).
        Upper corridor (agent_1 side): 17 - 18 - 19 - 20 - 22 (mirror of 10-9-8-7-5)
        Lower corridor (agent_1 side): 14 - 15 - 16 - 23      (mirror of 13-12-11-4)

    Cross bridges (self-mirror under i <-> 27-i):
        (10, 17): upper corridor chokepoint  (mirror(10)=17)
        (13, 14): lower corridor chokepoint  (mirror(13)=14)

    Regions:
        north = [0..6] bonus 5  ↔  south = [21..27] bonus 5
        upper = [7,8,9,10,17,18,19,20] bonus 4 (self-mirror)
        lower = [11,12,13,14,15,16] bonus 3 (self-mirror)
    """
    n = 28
    half_adj = {
        # Hexagon rim
        0: [1, 5, 6],
        1: [0, 2, 6],
        2: [1, 3, 6],
        3: [2, 4, 6],
        4: [3, 5, 6, 11],   # + lower corridor entry
        5: [0, 4, 6, 7],    # + upper corridor entry
        # Wheel hub
        6: [0, 1, 2, 3, 4, 5],
        # Upper corridor (agent_0 side)
        7: [5, 8],
        8: [7, 9],
        9: [8, 10],
        10: [9],
        # Lower corridor (agent_0 side)
        11: [4, 12],
        12: [11, 13],
        13: [12],
    }
    cross_edges = [
        (10, 17),  # upper bridge (self-mirror)
        (13, 14),  # lower bridge (self-mirror)
    ]
    adjacency_list, adjacency_matrix, initial_ownership = _mirror_half(
        n, half_adj, cross_edges
    )
    regions = {
        'north':           [0, 1, 2, 3, 4, 5, 6],
        'south':           [21, 22, 23, 24, 25, 26, 27],
        'upper_corridor':  [7, 8, 9, 10, 17, 18, 19, 20],
        'lower_corridor':  [11, 12, 13, 14, 15, 16],
    }
    region_bonuses = {
        'north': 5, 'south': 5,
        'upper_corridor': 4, 'lower_corridor': 3,
    }
    return MapConfig(
        n_territories=n,
        adjacency_list=adjacency_list,
        adjacency_matrix=adjacency_matrix,
        initial_ownership=initial_ownership,
        regions=regions,
        region_bonuses=region_bonuses,
    )


def create_hub_ring_30_map():
    """30-territory ring + inner hub cluster (mixed hub+ring style).

    Layout:

        Outer ring of 24 nodes: 0-1-2-...-11 (agent_0 arc), 18-19-...-29 (agent_1 arc),
            closed via cross edges 0-29 and 11-18.
        Inner hubs (6 nodes): 12,13,14 (agent_0) and 15,16,17 (agent_1) forming a K3+K3
            with a bridge between the halves.

        Each hub connects to a 4-node arc of the outer ring:
            hub 12 <-> ring {0, 1, 2, 3}
            hub 13 <-> ring {4, 5, 6, 7}
            hub 14 <-> ring {8, 9, 10, 11}
        Mirrored on the agent_1 side (hubs 17, 16, 15 respectively).

    Mirror i <-> 29-i:  ring 0..11 <-> 29..18 (reversed), hubs 12,13,14 <-> 17,16,15.

    Cross bridges (all self-mirror):
        (0, 29), (11, 18)   -- close the outer ring
        (14, 15)            -- hub cluster bridge

    Regions:
        ring_left  = [0..11]      bonus 5
        ring_right = [18..29]     bonus 5   (mirror of ring_left)
        hubs       = [12..17]     bonus 4   (self-mirror inner cluster)
    """
    n = 30
    half_adj = {
        # Outer ring chain 0-11 (agent_0)
        0: [1, 12],
        1: [0, 2, 12],
        2: [1, 3, 12],
        3: [2, 4, 12],
        4: [3, 5, 13],
        5: [4, 6, 13],
        6: [5, 7, 13],
        7: [6, 8, 13],
        8: [7, 9, 14],
        9: [8, 10, 14],
        10: [9, 11, 14],
        11: [10, 14],
        # Inner hubs form a K3 among themselves
        12: [0, 1, 2, 3, 13, 14],
        13: [4, 5, 6, 7, 12, 14],
        14: [8, 9, 10, 11, 12, 13],
    }
    cross_edges = [
        (0, 29),   # close outer ring "north" (self-mirror)
        (11, 18),  # close outer ring "south" (self-mirror)
        (14, 15),  # hub cluster bridge (self-mirror)
    ]
    adjacency_list, adjacency_matrix, initial_ownership = _mirror_half(
        n, half_adj, cross_edges
    )
    regions = {
        'ring_left':  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        'ring_right': [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
        'hubs':       [12, 13, 14, 15, 16, 17],
    }
    region_bonuses = {'ring_left': 5, 'ring_right': 5, 'hubs': 4}
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
# New medium maps (9)
MapRegistry.register("hex_grid_18", create_hex_grid_18_map)
MapRegistry.register("grid_20", create_grid_20_map)
MapRegistry.register("corridor_20", create_corridor_20_map)
MapRegistry.register("corridor_22", create_corridor_22_map)
MapRegistry.register("hub_spoke_16", create_hub_spoke_16_map)
MapRegistry.register("dual_hub_20", create_dual_hub_20_map)
MapRegistry.register("ring_cross_18", create_ring_cross_18_map)
MapRegistry.register("dense_mesh_16", create_dense_mesh_16_map)
MapRegistry.register("bipartite_20", create_bipartite_20_map)
# New large maps (3)
MapRegistry.register("grid_30", create_grid_30_map)
MapRegistry.register("corridor_28", create_corridor_28_map)
MapRegistry.register("hub_ring_30", create_hub_ring_30_map)


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
