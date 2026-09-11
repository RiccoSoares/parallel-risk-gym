"""Render every registered map as a figure showing topology + bonus regions.

Produces one PNG (and optionally PDF/SVG) per map plus a contact-sheet atlas,
for use in papers, docs, and experiment write-ups.

What each figure encodes:
  * Nodes           territories, labelled with their index
  * Node colour     initial owner (agent_0 blue circle / agent_1 red square).
                    Shape is redundant with colour so the figure survives
                    greyscale printing and colour-vision deficiency.
  * Edges           adjacency. Edges whose endpoints start under different
                    owners are drawn heavier + dashed: the opening front line.
  * Coloured blobs  bonus regions, each directly labelled "<name> +<bonus>".

Lives in env/ rather than evaluation/ because it needs only matplotlib and
numpy -- importing parallel_risk.evaluation pulls in Ray and torch.

Usage:
    PYTHONPATH=. python parallel_risk/env/map_viz.py
    PYTHONPATH=. python parallel_risk/env/map_viz.py --maps grid_20 corridor_28
    PYTHONPATH=. python parallel_risk/env/map_viz.py --format pdf --dpi 300
"""

import argparse
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, RegularPolygon

from parallel_risk.env.map_config import MapRegistry

# --------------------------------------------------------------------------
# Palette
#
# Validated with the data-viz palette validator (all-pairs list, light mode,
# surface #fcfcfb) rather than chosen by eye:
#   ownership pair  blue<->red      normal dE 32.3, CVD dE 21.6   PASS
#   regions 1-4     violet/yellow/magenta/green  CVD dE 16.2, normal dE 19.6  PASS
#   regions 1-5     + aqua          CVD dE 6.1 (floor band) -- legal because
#                                   every region carries a direct label.
# Only grid_30 has 5 regions, so slot 5 is rarely in play.
# --------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
EDGE_COLOR = "#b8b7b0"
FRONT_LINE_COLOR = "#52514e"

OWNER_COLORS = {0: "#2a78d6", 1: "#e34948"}
OWNER_LABELS = {0: "agent_0", 1: "agent_1"}

REGION_COLORS = ["#4a3aa7", "#eda100", "#e87ba4", "#008300", "#1baf7a"]

# Geometry, in data units where the median edge is 1.0 long.
NODE_R = 0.20
BLOB_R = 0.34
BLOB_R_STEP = 0.075          # nested radii so overlapping regions stay legible
INCHES_PER_UNIT = 0.62
LABEL_FONTSIZE = 8.5
LABEL_CHAR_W = 0.056         # data units per character at LABEL_FONTSIZE
# A region blob follows its members' short edges only. Long ones (a ring chord,
# a clique diameter) would drag the blob across the map; the edge itself is
# still drawn, so nothing is hidden -- the blob just stops claiming the space.
BLOB_EDGE_MAX = 2.2


# --------------------------------------------------------------------------
# Layouts
#
# Hand-authored per map so the drawing matches the topology the map was
# designed around: a generic force layout renders grid_20 as a tangle and
# hides the very chokepoints these maps exist to test. _auto_layout covers
# any map added later that has no entry here.
# --------------------------------------------------------------------------

def _rows(rows):
    """Grid coords from a list of rows (top row first). None leaves a gap."""
    pos = {}
    n_rows = len(rows)
    for r, row in enumerate(rows):
        for c, node in enumerate(row):
            if node is not None:
                pos[node] = (float(c), float(n_rows - 1 - r))
    return pos


def _ring(nodes, radius, start_deg, step_deg, center=(0.0, 0.0)):
    """Place nodes evenly around a circle, in the given cycle order."""
    pos = {}
    for k, node in enumerate(nodes):
        a = math.radians(start_deg + k * step_deg)
        pos[node] = (center[0] + radius * math.cos(a),
                     center[1] + radius * math.sin(a))
    return pos


def _wheel(hub, rim, radius, start_deg, step_deg, center):
    """A hub at `center` with its rim cycle around it."""
    pos = {hub: center}
    pos.update(_ring(rim, radius, start_deg, step_deg, center))
    return pos


def _chain(nodes, x, y_start, dy):
    """A vertical chain of corridor nodes."""
    return {node: (x, y_start + k * dy) for k, node in enumerate(nodes)}


def _mirror_y(pos, nodes_to_mirror):
    """Reflect nodes across y=0: mirror(i) = n-1-i for the symmetric maps."""
    return {dst: (pos[src][0], -pos[src][1]) for dst, src in nodes_to_mirror.items()}


def _mirror_x(pos, nodes_to_mirror):
    """Reflect nodes across x=0."""
    return {dst: (-pos[src][0], pos[src][1]) for dst, src in nodes_to_mirror.items()}


def _pair_map(n, lo_nodes):
    """{mirror(i): i} for i in lo_nodes, using the i <-> n-1-i convention."""
    return {n - 1 - i: i for i in lo_nodes}


def _layout_simple_6():
    return _rows([[0, 1, 2], [3, 4, 5]])


def _layout_medium_8():
    return {0: (0.0, 0.0), 1: (1.0, 0.62), 2: (1.0, -0.62), 3: (2.0, 0.0),
            4: (3.0, 0.0), 5: (4.0, 0.62), 6: (4.0, -0.62), 7: (5.0, 0.0)}


def _layout_large_10():
    return {0: (0.0, 1.05), 1: (-1.0, 1.85), 2: (1.0, 1.85),
            3: (0.0, 0.35), 4: (0.0, -0.35),
            7: (0.0, -1.05), 6: (-1.0, -1.85), 5: (1.0, -1.85),
            8: (-1.95, 0.0), 9: (1.95, 0.0)}


def _layout_triangle_6():
    pos = {0: (-0.85, 0.85), 1: (0.0, 1.55), 2: (0.85, 0.85)}
    pos.update(_mirror_y(pos, _pair_map(6, [0, 1, 2])))
    return pos


def _layout_ring_6():
    # a0 = {0,1,2} on top, a1 below; mirror is a reflection across y=0.
    return _ring([0, 1, 2, 3, 4, 5], radius=1.0, start_deg=150, step_deg=-60)


def _layout_star_8():
    pos = {3: (0.0, 0.5), 0: (0.0, 1.9), 1: (-0.8, 1.2), 2: (0.8, 1.2)}
    pos.update(_mirror_y(pos, _pair_map(8, [0, 1, 2, 3])))
    return pos


def _layout_double_hub_8():
    pos = {0: (-1.1, 1.7), 1: (0.0, 1.7), 2: (1.1, 1.7), 3: (0.0, 0.6)}
    pos.update(_mirror_y(pos, _pair_map(8, [0, 1, 2, 3])))
    return pos


def _layout_hex_grid_10():
    return _rows([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])


def _layout_dense_12():
    # Two K6 cliques as hexagons. K6 is complete, so the labels are free: put
    # the bridge nodes 0/2/4 on the lower arc so all three bridges come out
    # short and vertical instead of cutting through the cliques.
    pos = _ring([4, 5, 3, 1, 0, 2], radius=1.2, start_deg=330, step_deg=60,
                center=(0.0, 2.05))
    pos.update(_mirror_y(pos, _pair_map(12, range(6))))
    return pos


def _layout_hex_grid_18():
    # 3x6 grid; mirror is a 180-degree rotation, so rows stay in natural order.
    return _rows([list(range(0, 6)), list(range(6, 12)), list(range(12, 18))])


def _layout_grid_20():
    # 4x5 grid; mirror is a reflection across the horizontal midline, so the
    # two agent_1 rows run right-to-left.
    return _rows([[0, 1, 2, 3, 4],
                  [5, 6, 7, 8, 9],
                  [14, 13, 12, 11, 10],
                  [19, 18, 17, 16, 15]])


def _layout_corridor_20():
    # The corridor mouth clears the wheel rim so the two regions stay apart.
    pos = _wheel(5, [0, 1, 2, 3, 4], radius=1.3, start_deg=90, step_deg=-72,
                 center=(0.0, 5.4))
    pos.update(_chain([6, 7, 8, 9], x=0.0, y_start=3.35, dy=-0.95))
    pos.update(_mirror_y(pos, _pair_map(20, range(10))))
    return pos


def _layout_corridor_22():
    # Pentagon with 3 at the base so the two corridor entries (2, 4) face down.
    pos = _ring([0, 1, 2, 3, 4], radius=1.15, start_deg=126, step_deg=-72,
                center=(0.0, 4.3))
    pos.update(_chain([5, 6, 7], x=1.45, y_start=2.6, dy=-1.05))
    pos.update(_chain([8, 9, 10], x=-1.45, y_start=2.6, dy=-1.05))
    pos.update(_mirror_y(pos, _pair_map(22, range(11))))
    return pos


def _layout_hub_spoke_16():
    pos = {0: (-1.8, 2.4), 1: (-3.0, 2.4), 2: (-2.4, 1.2), 3: (-1.2, 1.2),
           4: (1.8, 2.4), 5: (1.2, 1.2), 6: (2.4, 1.2), 7: (3.0, 2.4)}
    pos.update(_mirror_y(pos, _pair_map(16, range(8))))
    return pos


def _layout_dual_hub_20():
    pos = {0: (-2.0, 3.4), 1: (-2.75, 4.15), 2: (-1.25, 4.15),
           3: (-1.25, 2.65), 4: (-2.75, 2.65),
           5: (2.0, 3.4), 6: (1.25, 4.15), 7: (2.75, 4.15),
           8: (2.75, 2.65), 9: (1.25, 2.65)}
    pos.update(_mirror_y(pos, _pair_map(20, range(10))))
    return pos


def _layout_ring_cross_18():
    # C18 with a0 on top; the three chords come out vertical.
    return _ring(list(range(18)), radius=2.6, start_deg=170, step_deg=-20)


def _layout_dense_mesh_16():
    # K4 x K2 per side, a0 left / a1 right. Each K4 is a shallow arc rather
    # than a square: collinear nodes would stack the clique's 6 edges on top
    # of each other, and concentric squares would nest the two regions inside
    # one another. The bridge nodes (0, 3, 4, 7) take the two inner columns so
    # every cross-half edge stays short.
    # The arc has to be deep: a shallow one lets each clique's "skip" chords
    # pass under the intermediate nodes, which reads as a false adjacency.
    dx, curve = -2.6, 0.58
    pos = {}
    for x, top, bot in ((-1.5, 1, 5), (-0.5, 2, 6), (0.5, 0, 4), (1.5, 3, 7)):
        pos[top] = (x + dx, 0.85 - curve * x * x)
        pos[bot] = (x + dx, -0.95 - curve * x * x)
    pos.update(_mirror_x(pos, _pair_map(16, range(8))))
    return pos


def _layout_bipartite_20():
    # Two pentagon continents per side; a0 left, a1 right.
    pos = _ring([0, 1, 2, 3, 4], radius=1.0, start_deg=0, step_deg=72,
                center=(-2.6, 1.6))
    pos.update(_ring([5, 6, 7, 8, 9], radius=1.0, start_deg=72, step_deg=72,
                     center=(-2.6, -1.6)))
    pos.update(_mirror_x(pos, _pair_map(20, range(10))))
    return pos


def _layout_grid_30():
    # 5x6 grid; mirror is a 180-degree rotation, so rows stay in natural order.
    return _rows([list(range(r * 6, r * 6 + 6)) for r in range(5)])


def _layout_corridor_28():
    pos = _ring([3, 2, 1, 0, 5, 4], radius=1.3, start_deg=0, step_deg=60,
                center=(0.0, 4.6))
    pos[6] = (0.0, 4.6)
    pos.update(_chain([7, 8, 9, 10], x=-1.5, y_start=3.5, dy=-1.0))
    pos.update(_chain([11, 12, 13], x=1.5, y_start=3.0, dy=-1.2))
    pos.update(_mirror_y(pos, _pair_map(28, range(14))))
    return pos


def _layout_hub_ring_30():
    # 24-node outer ring in cycle order [0..11, 18..29], a0 on top.
    cycle = list(range(0, 12)) + list(range(18, 30))
    pos = _ring(cycle, radius=3.8, start_deg=180 - 7.5, step_deg=-15)
    pos.update({12: (-1.5, 0.87), 13: (0.0, 1.72), 14: (1.5, 0.87)})
    pos.update(_mirror_y(pos, {17: 12, 16: 13, 15: 14}))
    return pos


_LAYOUTS = {
    "simple_6": _layout_simple_6,
    "basic_6": _layout_simple_6,
    "medium_8": _layout_medium_8,
    "large_10": _layout_large_10,
    "triangle_6": _layout_triangle_6,
    "ring_6": _layout_ring_6,
    "star_8": _layout_star_8,
    "double_hub_8": _layout_double_hub_8,
    "hex_grid_10": _layout_hex_grid_10,
    "dense_12": _layout_dense_12,
    "hex_grid_18": _layout_hex_grid_18,
    "grid_20": _layout_grid_20,
    "corridor_20": _layout_corridor_20,
    "corridor_22": _layout_corridor_22,
    "hub_spoke_16": _layout_hub_spoke_16,
    "dual_hub_20": _layout_dual_hub_20,
    "ring_cross_18": _layout_ring_cross_18,
    "dense_mesh_16": _layout_dense_mesh_16,
    "bipartite_20": _layout_bipartite_20,
    "grid_30": _layout_grid_30,
    "corridor_28": _layout_corridor_28,
    "hub_ring_30": _layout_hub_ring_30,
}


def _auto_layout(cfg):
    """Fallback for maps with no hand-authored layout.

    Lays the graph out with networkx, orients agent_0 above agent_1, then
    enforces the i <-> n-1-i symmetry the _mirror_half maps are built with
    (trying both a reflection and a 180-degree rotation, keeping whichever
    distorts the raw layout less).
    """
    n = cfg.n_territories
    try:
        import networkx as nx
        g = nx.Graph()
        g.add_nodes_from(range(n))
        for src, neighbours in cfg.adjacency_list.items():
            g.add_edges_from((src, dst) for dst in neighbours)
        raw = nx.kamada_kawai_layout(g)
        pos = np.array([raw[i] for i in range(n)], dtype=float)
    except Exception:
        return _ring(list(range(n)), radius=n / (2 * math.pi), start_deg=90,
                     step_deg=-360.0 / n)

    owner = np.asarray(cfg.initial_ownership)
    if (owner == 0).any() and (owner == 1).any():
        axis = pos[owner == 0].mean(0) - pos[owner == 1].mean(0)
        theta = math.atan2(axis[1], axis[0]) - math.pi / 2
        rot = np.array([[math.cos(-theta), -math.sin(-theta)],
                        [math.sin(-theta), math.cos(-theta)]])
        pos = pos @ rot.T
    pos -= pos.mean(0)

    mirrored = pos[::-1].copy()
    reflect = np.stack([mirrored[:, 0], -mirrored[:, 1]], axis=1)
    rotate = -mirrored
    best = min((reflect, rotate), key=lambda cand: np.abs(pos - cand).sum())
    pos = (pos + best) / 2.0
    return {i: (float(pos[i, 0]), float(pos[i, 1])) for i in range(n)}


def layout_for(name, cfg):
    """Node positions for a map, normalised so the median edge length is 1."""
    pos = _LAYOUTS[name]() if name in _LAYOUTS else _auto_layout(cfg)
    missing = set(range(cfg.n_territories)) - set(pos)
    if missing:
        raise ValueError(f"layout for {name!r} is missing nodes {sorted(missing)}")

    lengths = [math.dist(pos[a], pos[b]) for a, b in edges_of(cfg)]
    scale = 1.0 / np.median(lengths) if lengths else 1.0
    return {i: (x * scale, y * scale) for i, (x, y) in pos.items()}


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------

def edges_of(cfg):
    """Unique undirected edges as sorted (low, high) pairs."""
    return sorted({(min(a, b), max(a, b))
                   for a, neighbours in cfg.adjacency_list.items()
                   for b in neighbours})


def _seg_distance(X, Y, p, q):
    """Distance from each grid point to the segment pq."""
    dx, dy = q[0] - p[0], q[1] - p[1]
    len2 = dx * dx + dy * dy
    if len2 == 0.0:
        return np.hypot(X - p[0], Y - p[1])
    t = np.clip(((X - p[0]) * dx + (Y - p[1]) * dy) / len2, 0.0, 1.0)
    return np.hypot(X - (p[0] + t * dx), Y - (p[1] + t * dy))


def _map_grid(pos, pad):
    """A sampling grid covering the whole map, used for every distance field."""
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    x0, x1 = min(xs) - pad, max(xs) + pad
    y0, y1 = min(ys) - pad, max(ys) + pad
    nx = int(np.clip((x1 - x0) * 42, 140, 520))
    ny = int(np.clip((y1 - y0) * 42, 140, 520))
    gx = np.linspace(x0, x1, nx)
    gy = np.linspace(y0, y1, ny)
    return gx, gy, np.meshgrid(gx, gy)


def _erode_x(field, span_px):
    """Sliding horizontal minimum: the worst clearance a label of that half-
    width would cover if centred on each point."""
    if span_px < 1:
        return field
    out = field
    shift = 1
    while shift <= span_px:                       # binary lifting: O(log span)
        left = np.concatenate([np.repeat(out[:, :1], shift, axis=1),
                               out[:, :-shift]], axis=1)
        right = np.concatenate([out[:, shift:],
                                np.repeat(out[:, -1:], shift, axis=1)], axis=1)
        out = np.minimum(out, np.minimum(left, right))
        shift *= 2
    return out


def _distance_field(X, Y, pts, segs):
    """Distance from each grid point to the nearest of `pts` and `segs`."""
    dist = np.full(X.shape, np.inf)
    for p in pts:
        np.minimum(dist, np.hypot(X - p[0], Y - p[1]), out=dist)
    for p, q in segs:
        np.minimum(dist, _seg_distance(X, Y, p, q), out=dist)
    return dist


def _draw_region(ax, cfg, pos, nodes, color, radius, label, grid, clearance):
    """Fill + outline the union of discs and capsules covering a region.

    Built as a distance field so disjoint or concave regions (ring arcs,
    corridors) come out as their true shape rather than a convex hull.

    The label goes to the emptiest point near the region, measured against
    every node, every edge and every label already placed -- an interior
    "centre" would sit on the chokepoint edge of a corridor-shaped region.
    """
    gx, gy, (X, Y) = grid
    members = set(nodes)
    pts = [pos[i] for i in nodes]
    segs = [(pos[a], pos[b]) for a, b in edges_of(cfg)
            if a in members and b in members
            and math.dist(pos[a], pos[b]) <= BLOB_EDGE_MAX]

    dist = _distance_field(X, Y, pts, segs)
    ax.contourf(X, Y, dist, levels=[0.0, radius], colors=[color], alpha=0.15,
                zorder=1)
    ax.contour(X, Y, dist, levels=[radius], colors=[color], linewidths=1.6,
               alpha=0.9, zorder=2)

    # Score the label's whole footprint, not just its centre: a wide label
    # centred in a gap can still have its box land on a chokepoint edge.
    half_w = max(0.25, LABEL_CHAR_W * len(label))
    span_px = int(round(half_w / (gx[1] - gx[0])))
    footprint = _erode_x(clearance, span_px)

    cx = float(np.mean([p[0] for p in pts]))
    cy = float(np.mean([p[1] for p in pts]))
    near = dist <= radius * 2.4
    score = np.where(near, footprint - 0.06 * np.hypot(X - cx, Y - cy), -np.inf)
    iy, ix = np.unravel_index(np.argmax(score), score.shape)
    lx, ly = float(gx[ix]), float(gy[iy])

    ax.text(lx, ly, label, ha="center", va="center", fontsize=LABEL_FONTSIZE,
            color=INK, zorder=6, fontweight="medium",
            bbox=dict(boxstyle="round,pad=0.25", facecolor=SURFACE,
                      edgecolor="none", alpha=0.85))

    # Block this label's footprint so later regions do not land on top of it.
    np.minimum(clearance,
               _seg_distance(X, Y, (lx - half_w, ly), (lx + half_w, ly)),
               out=clearance)


def draw_map(name, cfg, ax, pos=None, show_labels=True):
    """Draw one map onto `ax`. Returns the layout used."""
    pos = pos or layout_for(name, cfg)
    owner = np.asarray(cfg.initial_ownership)
    edges = edges_of(cfg)

    # Regions, largest first so smaller overlapping ones stay readable.
    order = sorted(cfg.regions.items(), key=lambda kv: -len(kv[1]))
    color_of = {n: REGION_COLORS[i % len(REGION_COLORS)]
                for i, n in enumerate(cfg.regions)}
    grid = _map_grid(pos, pad=BLOB_R + 0.9)
    X, Y = grid[2]
    clearance = _distance_field(X, Y, list(pos.values()),
                                [(pos[a], pos[b]) for a, b in edges])
    for depth, (region, members) in enumerate(order):
        _draw_region(ax, cfg, pos, members, color_of[region],
                     BLOB_R + depth * BLOB_R_STEP,
                     f"{region}  +{cfg.region_bonuses[region]}", grid, clearance)

    for a, b in edges:
        front = owner[a] != owner[b]
        ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]],
                color=FRONT_LINE_COLOR if front else EDGE_COLOR,
                linewidth=1.9 if front else 1.3,
                linestyle=(0, (4, 2.5)) if front else "-",
                solid_capstyle="round", zorder=3)

    for node in range(cfg.n_territories):
        x, y = pos[node]
        side = int(owner[node])
        shape_kw = dict(facecolor=OWNER_COLORS[side], edgecolor=SURFACE,
                        linewidth=2.0, zorder=4)
        if side == 0:
            ax.add_patch(Circle((x, y), NODE_R, **shape_kw))
        else:
            ax.add_patch(RegularPolygon((x, y), 4, radius=NODE_R * 1.20,
                                        orientation=math.pi / 4, **shape_kw))
        if show_labels:
            ax.text(x, y, str(node), ha="center", va="center", fontsize=7.5,
                    color="white", fontweight="bold", zorder=5)

    ax.set_aspect("equal")
    ax.axis("off")
    ax.margins(0.10)
    return pos


def map_stats(cfg):
    """One-line summary of a map's size, density and income ceiling."""
    edges = edges_of(cfg)
    degrees = [len(v) for v in cfg.adjacency_list.values()]
    unassigned = cfg.n_territories - len(set().union(*cfg.regions.values()))
    parts = [
        f"{cfg.n_territories} territories",
        f"{len(edges)} edges",
        f"degree {min(degrees)}–{max(degrees)} (mean {np.mean(degrees):.1f})",
        f"{len(cfg.regions)} regions",
        f"income 5+{sum(cfg.region_bonuses.values())}",
    ]
    if unassigned:
        parts.append(f"{unassigned} unassigned")
    return "  ·  ".join(parts)


def _legend_handles(cfg):
    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=9,
               markerfacecolor=OWNER_COLORS[0], markeredgecolor=SURFACE,
               label=f"{OWNER_LABELS[0]} start"),
        Line2D([], [], marker="s", linestyle="none", markersize=9,
               markerfacecolor=OWNER_COLORS[1], markeredgecolor=SURFACE,
               label=f"{OWNER_LABELS[1]} start"),
        Line2D([], [], color=FRONT_LINE_COLOR, linewidth=1.9,
               linestyle=(0, (4, 2.5)), label="front line"),
        Line2D([], [], color=EDGE_COLOR, linewidth=1.3, label="adjacency"),
    ]
    for i, (region, members) in enumerate(cfg.regions.items()):
        handles.append(Line2D(
            [], [], marker="o", linestyle="none", markersize=9,
            markerfacecolor=REGION_COLORS[i % len(REGION_COLORS)],
            markeredgecolor="none", alpha=0.85,
            label=f"{region}: +{cfg.region_bonuses[region]} "
                  f"({len(members)} terr.)"))
    return handles


def render_map(name, out_dir, fmt="png", dpi=200):
    """Render one map to <out_dir>/<name>.<fmt>. Returns the path written."""
    cfg = MapRegistry.get(name)
    pos = layout_for(name, cfg)

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    span_x = max(xs) - min(xs) + 2 * (BLOB_R + 0.9)
    span_y = max(ys) - min(ys) + 2 * (BLOB_R + 0.9)
    # Width has a floor so the header fits; height follows the map's own
    # aspect, otherwise wide-and-flat maps get a band of dead space.
    width = max(6.8, span_x * INCHES_PER_UNIT)
    height = max(2.6, span_y * INCHES_PER_UNIT)

    n_legend_rows = math.ceil((4 + len(cfg.regions)) / 4)
    legend_h = 0.30 * n_legend_rows + 0.20
    header_h = 0.95
    fig = plt.figure(figsize=(width, height + header_h + legend_h),
                     facecolor=SURFACE)
    ax = fig.add_axes([0.02,
                       legend_h / (height + header_h + legend_h),
                       0.96,
                       height / (height + header_h + legend_h)])
    ax.set_facecolor(SURFACE)
    draw_map(name, cfg, ax, pos=pos)

    fig.text(0.02, 1 - 0.34 / (height + header_h + legend_h), name,
             fontsize=17, fontweight="bold", color=INK, ha="left", va="center")
    fig.text(0.02, 1 - 0.66 / (height + header_h + legend_h), map_stats(cfg),
             fontsize=8.5, color=INK_SECONDARY, ha="left", va="center")

    legend = fig.legend(handles=_legend_handles(cfg), loc="lower center",
                        ncol=4, frameon=False, fontsize=8.5,
                        handletextpad=0.6, columnspacing=1.6,
                        bbox_to_anchor=(0.5, 0.004))
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.{fmt}")
    fig.savefig(path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)
    return path


def render_atlas(names, out_dir, paths, dpi=140, cols=3):
    """Contact sheet of the already-rendered per-map PNGs."""
    import matplotlib.image as mpimg

    rows = math.ceil(len(names) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6.0, rows * 5.0),
                             facecolor=SURFACE)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
        ax.set_facecolor(SURFACE)
    for ax, name, path in zip(axes, names, paths):
        ax.imshow(mpimg.imread(path))
    fig.suptitle("Parallel Risk — map atlas", fontsize=19,
                 fontweight="bold", color=INK, y=0.997)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out = os.path.join(out_dir, "atlas.png")
    fig.savefig(out, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)
    return out


def check_layouts(names):
    """Flag layouts that would read as a different graph than they are.

    A node drawn on top of an edge it is not an endpoint of invents an
    adjacency for the reader; two nodes drawn on top of each other hide one.
    Worth running after adding a map, since a new map falls through to
    _auto_layout, which optimises for spread rather than for these.
    """
    problems = 0
    for name in names:
        cfg = MapRegistry.get(name)
        pos = layout_for(name, cfg)
        for node in range(cfg.n_territories):
            for a, b in edges_of(cfg):
                if node in (a, b):
                    continue
                gap = _seg_distance(np.array([pos[node][0]]),
                                    np.array([pos[node][1]]),
                                    pos[a], pos[b])[0]
                if gap < NODE_R * 1.15:
                    print(f"  {name}: node {node} sits {gap:.3f} from edge "
                          f"{a}-{b} (want >= {NODE_R * 1.15:.3f})")
                    problems += 1
        for i in range(cfg.n_territories):
            for j in range(i + 1, cfg.n_territories):
                gap = math.dist(pos[i], pos[j])
                if gap < NODE_R * 2.3:
                    print(f"  {name}: nodes {i} and {j} overlap "
                          f"({gap:.3f} apart, want >= {NODE_R * 2.3:.3f})")
                    problems += 1
    print(f"{'OK: no overlaps' if not problems else f'{problems} problem(s)'} "
          f"across {len(names)} maps")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--maps", nargs="*", default=None,
                        help="map names to render (default: all registered)")
    parser.add_argument("--out-dir", default="docs/maps")
    parser.add_argument("--format", default="png",
                        choices=["png", "pdf", "svg"])
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--no-atlas", action="store_true")
    parser.add_argument("--check-layouts", action="store_true",
                        help="report node/edge overlaps instead of rendering")
    args = parser.parse_args()

    names = args.maps or MapRegistry.list_maps()
    if not args.maps:
        names = [n for n in names if n != "basic_6"]   # alias of simple_6

    if args.check_layouts:
        raise SystemExit(1 if check_layouts(names) else 0)

    paths = []
    for name in names:
        path = render_map(name, args.out_dir, fmt=args.format, dpi=args.dpi)
        paths.append(path)
        print(f"  wrote {path}")

    if not args.no_atlas and args.format == "png" and len(names) > 1:
        print(f"  wrote {render_atlas(names, args.out_dir, paths)}")


if __name__ == "__main__":
    main()
