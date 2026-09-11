# Map Atlas

Rendered figures for every map registered in `parallel_risk/env/map_config.py`.
One PNG per map lives in [`docs/maps/`](maps/), plus a contact sheet at
[`docs/maps/atlas.png`](maps/atlas.png).

## Regenerating

```bash
PYTHONPATH=. python parallel_risk/env/map_viz.py                  # all maps -> docs/maps/
PYTHONPATH=. python parallel_risk/env/map_viz.py --maps grid_20   # just one
PYTHONPATH=. python parallel_risk/env/map_viz.py --format pdf --dpi 300
PYTHONPATH=. python parallel_risk/env/map_viz.py --check-layouts  # sanity check, no rendering
```

The renderer needs only matplotlib and numpy (`networkx` is used only as a
fallback for maps with no hand-authored layout). It lives in `env/` rather than
`evaluation/` so importing it does not pull in Ray and torch.

## Reading a figure

| Mark | Meaning |
|---|---|
| **Blue circle** | territory owned by `agent_0` at reset |
| **Red square** | territory owned by `agent_1` at reset |
| **Grey line** | adjacency (troops may move or attack along it) |
| **Dashed dark line** | *front line* — an edge whose two ends start under different owners |
| **Coloured blob** | a bonus region, labelled `<name> +<bonus>` |

Shape is redundant with colour for ownership, so the figures stay readable in
greyscale and under colour-vision deficiency. Every region carries a direct
label as well as a legend entry, so no region is identified by colour alone.

Each header line reports territory count, edge count, degree range and mean,
region count, and the income ceiling as `5+<sum of region bonuses>` — 5 is the
base `income_per_turn`, and a region's bonus is only paid while an agent holds
**all** of its territories. Every territory starts with 3 troops.

Two caveats worth knowing when reading a blob:

- A region is not necessarily connected. `middle_arc` in `ring_cross_18` and
  `bipartite_20`'s continents render as two lumps because that is what they are.
- A blob hugs only its members' *short* edges. Long ones (a ring chord, a clique
  diameter) would drag the blob across the map, so they are excluded from the
  shape — the edge itself is still drawn.

Some territories belong to no region at all (`large_10`, `bipartite_20`); the
header counts them as `unassigned`. Regions may also overlap: `simple_6`'s
`center`, `star_8`'s `hubs` and `dense_12`'s `bridges` share territories with
the larger regions around them, drawn as nested outlines.

## The maps

All maps are mirror-symmetric between the two agents (`i <-> n-1-i`), so neither
side has a positional advantage. The 12 maps from 16 territories up were added
in commit `879fb6d` to give the action-budget sweep room to differentiate — the
6–12 territory maps were too easy to separate techniques on. The 200-iteration
PPO sweep (`experiments/k_sweep_ppo_200/results_K5_K10.csv`) bears that out:
mean win rate vs. random falls from K=5 to K=10 by 79.3 → 73.3 on the small
maps, 86.7 → 44.4 on the medium ones, and 62.2 → 11.1 on the large ones.

| Map | Terr. | Edges | Mean deg. | Regions | Income | Region bonuses |
|---|---|---|---|---|---|---|
| [`ring_6`](maps/ring_6.png) | 6 | 6 | 2.0 | 2 | 5+6 | left +3, right +3 |
| [`simple_6`](maps/simple_6.png) | 6 | 7 | 2.3 | 3 | 5+10 | north +4, south +4, center +2 |
| [`triangle_6`](maps/triangle_6.png) | 6 | 9 | 3.0 | 2 | 5+8 | left +4, right +4 |
| [`double_hub_8`](maps/double_hub_8.png) | 8 | 13 | 3.2 | 2 | 5+10 | left +5, right +5 |
| [`medium_8`](maps/medium_8.png) | 8 | 11 | 2.8 | 3 | 5+10 | west +4, bridge +2, east +4 |
| [`star_8`](maps/star_8.png) | 8 | 13 | 3.2 | 3 | 5+8 | left +3, right +3, hubs +2 |
| [`hex_grid_10`](maps/hex_grid_10.png) | 10 | 17 | 3.4 | 4 | 5+10 | north_left +3, north_right +2, south_left +2, south_right +3 |
| [`large_10`](maps/large_10.png) | 10 | 13 | 2.6 | 3 | 5+11 | north +4, corridor +3, south +4 |
| [`dense_12`](maps/dense_12.png) | 12 | 33 | 5.5 | 3 | 5+11 | left +4, right +4, bridges +3 |
| [`dense_mesh_16`](maps/dense_mesh_16.png) | 16 | 36 | 4.5 | 4 | 5+12 | left_top +3, left_bot +3, right_bot +3, right_top +3 |
| [`hub_spoke_16`](maps/hub_spoke_16.png) | 16 | 25 | 3.1 | 4 | 5+12 | left_A +3, left_B +3, right_B +3, right_A +3 |
| [`hex_grid_18`](maps/hex_grid_18.png) | 18 | 41 | 4.6 | 3 | 5+14 | north +5, middle +4, south +5 |
| [`ring_cross_18`](maps/ring_cross_18.png) | 18 | 21 | 2.3 | 3 | 5+12 | north_arc +4, middle_arc +4, south_arc +4 |
| [`bipartite_20`](maps/bipartite_20.png) | 20 | 32 | 3.2 | 4 | 5+8 | A_left +2, A_right +2, B_left +2, B_right +2 |
| [`corridor_20`](maps/corridor_20.png) | 20 | 29 | 2.9 | 3 | 5+14 | north +5, corridor +4, south +5 |
| [`dual_hub_20`](maps/dual_hub_20.png) | 20 | 38 | 3.8 | 4 | 5+12 | left_A +3, left_B +3, right_B +3, right_A +3 |
| [`grid_20`](maps/grid_20.png) | 20 | 31 | 3.1 | 4 | 5+14 | north +3, north_mid +4, south_mid +4, south +3 |
| [`corridor_22`](maps/corridor_22.png) | 22 | 28 | 2.5 | 4 | 5+14 | north +4, south +4, upper_corridor +3, lower_corridor +3 |
| [`corridor_28`](maps/corridor_28.png) | 28 | 40 | 2.9 | 4 | 5+17 | north +5, south +5, upper_corridor +4, lower_corridor +3 |
| [`grid_30`](maps/grid_30.png) | 30 | 49 | 3.3 | 5 | 5+16 | row0 +3, row1 +3, center +4, row3 +3, row4 +3 |
| [`hub_ring_30`](maps/hub_ring_30.png) | 30 | 55 | 3.7 | 3 | 5+14 | ring_left +5, ring_right +5, hubs +4 |

`basic_6` is an alias of `simple_6` and is not rendered separately.

## Topology families

The roster deliberately spans structures that stress different things:

- **Chokepoint maps** — `medium_8`, `large_10`, `corridor_20`, `corridor_22`,
  `corridor_28`. Long thin corridors between wheel or pentagon continents. One
  or two bridges carry all traffic, so search-based methods have something to
  find. These are where PPO collapses first as the action budget grows
  (`corridor_20` and `corridor_22` both hit 0% at K=10).
- **Grids** — `simple_6`, `hex_grid_10`, `hex_grid_18`, `grid_20`, `grid_30`.
  Uniform degree, broad fronts, no chokepoints.
- **Dense / clique** — `dense_12` (two K6), `dense_mesh_16` (K4 x K2 per side),
  `triangle_6`. Many redundant paths; high degree stresses the GNN's message
  passing rather than its planning.
- **Hub-and-spoke** — `star_8`, `double_hub_8`, `hub_spoke_16`, `dual_hub_20`,
  `hub_ring_30`. High-degree hubs adjacent to low-degree spokes.
- **Rings** — `ring_6`, `ring_cross_18`. Minimum degree, long paths; message
  passing has to traverse the whole map. `ring_cross_18` adds three chords that
  cut the diameter.
- **Split continents** — `bipartite_20`. Two nearly independent theatres joined
  by one edge per side, with frontier territories that pay no bonus at all.

## Layouts

Positions are hand-authored per map in `map_viz.py` so each drawing matches the
topology the map was designed around. This matters: `grid_20`'s mirror is a
reflection, so its two `agent_1` rows run right-to-left, while `hex_grid_18` and
`grid_30` mirror by 180-degree rotation and keep natural row order. A generic
force layout draws all three as a tangle and hides the chokepoints the maps
exist to test.

A map with no hand-authored entry falls back to `_auto_layout`, which lays the
graph out with networkx, orients `agent_0` above `agent_1`, then enforces the
`i <-> n-1-i` symmetry. **After adding a map, run `--check-layouts`**: it flags a
node drawn on an edge it is not an endpoint of (which invents an adjacency for
the reader) and nodes drawn on top of each other. If a new map trips it, add an
explicit layout to `_LAYOUTS`.
