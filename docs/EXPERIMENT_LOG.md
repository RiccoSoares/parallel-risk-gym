# Experiment Log

Rolling log of what we ran on this branch and what we learned. New
entries at the bottom. Companion to
[docs/RL_TRAINING_ROADMAP.md](RL_TRAINING_ROADMAP.md) (plan) and
[docs/ARCHITECTURAL_IMPROVEMENTS.md](ARCHITECTURAL_IMPROVEMENTS.md)
(optional upgrades). This file only records experiments — design
decisions and rationale live in the roadmap.

Entry template: **Setup → Result → What it changes**.

---

## 1. MCTS+GNN framework implementation (Phase 3)

**Setup.** Two training tracks under
`parallel_risk/training/mcts_gnn/`:
- **AZ distillation** — cross-entropy on the MCTS visit distribution +
  MSE on the terminal outcome.
- **Expert Iteration (ExIt)** — advantage-weighted log-prob on the
  MCTS-selected action, with the value baseline detached.

Shared infrastructure: `MCTSGNNAgent` (PUCT selection, Dirichlet root
noise, temperature-based action sampling, per-slot action decoding),
region-padded warm-start loader, spawn-mode ProcessPool workers.

**Result.** Both trainers run end-to-end on the 9-unique-map roster.
Smoke tests pass. Env plumbing extended to `max_actions_per_turn > 10`
and `max_regions > 3` (region-padded checkpoint loading).

**What it changes.** We have the two AZ-family training modes running
in parallel with PPO. Everything downstream (K-sweep, cold-start
diagnostic, warm-start comparison) is unblocked.

Commit: `10d553d` (framework), `981fded` (K>10 scaling), `502ab75`
(max_regions plumbing).

---

## 2. Cold-start diagnostic — AZ and ExIt

**Setup.** Random-init MCTS+GNN vs MCTS-uniform, both at
budget=40, on the 9-map roster, K=5.

**Result.**
- AZ (visit-distribution distillation): 0/16 wins at cold start. Loss
  is self-referential (target IS the current network's own MCTS visit
  distribution), so no external gradient signal on medium/large maps
  where self-play hits max_turns=40 with all draws.
- ExIt with terminal-only reward: also collapses on cold start.
  Advantages ~ 0 because ~100% draws on the harder maps means
  `z_target - value ≈ 0` everywhere. `use_value_fn=False` didn't
  rescue it either.

**What it changes.** Cold-start on the 9-map roster is not viable
with these techniques at K=5, budget=40, max_turns=40. Both AZ and
ExIt require either warm-start weights (a competent baseline to
generate decisive games) or substantially more compute/search budget
to break the draw regime on the harder maps. See §3.

Related script: `experiments/diagnostic_mcts_gnn_vs_uniform.py`.

---

## 3. Warm-start ExIt on 9 maps

**Setup.** Load a PPO checkpoint (trained on `max_regions=3`) into
the ExIt trainer with region-padded input/global projection weights.
50 iterations, 72 games/iter (~8 per map), MCTS budget=40, eval every
10 iterations against MCTS-uniform-40.

**Result.** Aggregate eval win-rate rose modestly (`+15 pp` over
cold-start baseline) but stayed below 50%. Per-map: small maps
approached parity, medium_8/large_10 stayed <30%. Wall-clock ~103
min.

**What it changes.** Confirmed that warm-start bootstraps ExIt out
of the cold-start hole, but MCTS at budget=40 caps the ceiling.
Directionally the "yellow" outcome from the plan's interpretation
grid (learns something, doesn't reach 50%). Read as evidence that
follow-up should raise the search budget (Tier 1) or use a stronger
starting checkpoint.

Related script: `experiments/mcts_gnn_exit_training_run.py`.

---

## 4. K-sweep PPO (Phase 3.5)

**Setup.** Train PPO+GNN across 21 maps (9 small + 9 medium +
3 large), 200 iterations per K value, K ∈ {5, 10, 15, 20},
`max_actions_per_turn = max(10, K)`, eval vs MCTS-uniform at
matching K.

**Result.** Complete for K=5 and K=10. K=15/K=20 deferred after a
ProcessPoolExecutor over-subscription hang (21 eval workers + 8
rollout workers = deadlock on 8-core machine). Fixed by capping
eval at `min(len(maps), 8)`. Per-map win-rate at 200 iters
(the mean over the 21 maps of the final eval point, from
`experiments/k_sweep_ppo_200/results_K5_K10.csv`, which is the data
behind `dashboard_K5_K10.png`):
- K=5: aggregate **0.800** (small 0.793, medium 0.867, large 0.622)
- K=10: aggregate **0.521** (small 0.733, medium 0.445, large 0.111)
- K=15, K=20: not yet run

*(Correction, 2026-09-10: this entry previously said ~0.72 and ~0.58.
Those numbers do not appear in the results files or the dashboard;
the values above are what the committed data contains.)*

**What it changes.** The K=5 → K=10 drop-off was initially
attributed to "PPO gets harder to train with more action slots".
Q4 (§7) later showed the opponent also got substantially stronger
at K=10, so the drop-off is partly opponent-inflation, not just
training difficulty.

Related script: `experiments/k_sweep_ppo.py`.
Commits: `5ed7dae`, `8c38fce`.

---

## 5. Bauer architectural comparison

**Setup.** Read Bauer's `py-risk` code (paper paywalled; thesis PDF
403'd). Categorized differences between our GCN policy and his
GG-net into superficial / RL-framing artifact / real architectural /
different-game buckets.

**Result.** Real architectural differences:
1. GCN vs GATv2/TransformerConv message passing
2. Mean pooling vs `GlobalAttention` pooling
3. Flat region one-hot vs explicit region-node message passing
4. Factorized softmax vs joint per-order scoring
5. Static intra-turn masks vs sequential state updates between slots

Deferred all of them as optional Tier 1/2/3 upgrades. Ranked by
effort × confidence in
[ARCHITECTURAL_IMPROVEMENTS.md](ARCHITECTURAL_IMPROVEMENTS.md).

**What it changes.** We have a stable menu to draw from if / when
the current architecture becomes the bottleneck. Not on the
critical path today.

Commit: `7ce8e57` (doc).

---

## 6. Q3 — MCTS search-budget calibration

**Setup.** MCTS-uniform(200) vs MCTS-uniform(40) head-to-head, 9
maps, 20 games/map, colors alternated 50/50. Two runs: at K=5 and
at K=10. Metric = (wins + 0.5·draws) / n with Wilson 95% CI.
Wall-clock: 12.6 min (K=5), 21.5 min (K=10).

**Result.**
- Aggregate score for MCTS-200: **0.78 at K=5, 0.77 at K=10** —
  nearly identical average.
- Per-map varies substantially: simple_6 collapses 0.90 → 0.60
  going K=5 → K=10 (small map, more actions ≈ exhausts strategic
  space); large_10 and dense_12 get *stronger* at K=10.
- medium_8 is 100% draws at K=5 (score 0.50), breaks to 14 draws +
  5 wins at K=10 — extra actions/turn end the stalemate.

**What it changes.**
- MCTS-40 is meaningfully weaker than MCTS-200 on most maps.
  Existing PPO-vs-MCTS-40 win-rates are inflated by roughly
  25-30 pp aggregate.
- medium_8's poor results in earlier training runs were not just
  "PPO struggling" — the map is a defensive equilibrium at K=5,
  max_turns=40. Neither MCTS budget wins consistently.
- The Tier-1 upgrade "MCTS budget 40 → 200" is empirically
  supported for eval purposes.

Scripts: `experiments/diagnostic_mcts_budget.py`,
`experiments/diagnostic_mcts_budget/_plot_dashboard.py`.
Results: `experiments/diagnostic_mcts_budget/{full_k5,full_k10}/`.
Commit: `1816864`.

---

## 7. Q4 — MCTS K-strength calibration

**Setup.** MCTS(budget=40, K=10) vs MCTS(budget=40, K=5)
head-to-head, same 9-map roster, 20 games/map, colors alternated
50/50. Both sides asymmetric-K in the same env
(`max_actions_per_turn = 10`). Wall-clock: 6.1 min.

**Result.**
- Aggregate score for K=10 side: **0.87**.
- Wins on 9/9 maps. K=10 side never loses the aggregate on any
  map. Losses are rare (only 3 maps produced any K=5 wins — all
  n=6 maps).
- On the two 10-territory maps: near-perfect for K=10 (~19/20).

**What it changes.**
- Higher K makes MCTS a substantially stronger opponent at fixed
  search budget. The K sweep's drop-off at K=10 (§4) is at least
  partly opponent-inflation, not just PPO training difficulty.
- For a fair K-sweep, opponent K should be fixed (e.g., always
  MCTS-K5) OR agent performance should be reported against a
  common baseline like MaskedRandom.
- K is a bigger opponent-strength lever than search budget: 2× K
  gives ~37 pp lift; 5× search budget gives ~28 pp lift.

Scripts: `experiments/diagnostic_mcts_k_strength.py`.
Results: `experiments/diagnostic_mcts_k_strength/full_b40/`.
Combined plot: `experiments/diagnostic_mcts_budget/q3_q4_combined.png`.
Commit: `1816864`.

---

## Cross-cutting conclusions (as of 2026-09-09)

1. **MCTS-uniform is a moving target across K.** Any K-sweep or
   cross-K comparison must fix a common baseline (fixed-K MCTS, or
   MaskedRandom) for the numbers to be comparable across K values.

2. **medium_8 at K=5, max_turns=40 is a defensive equilibrium.**
   Not a training failure — a structural property of the map + hyper
   combination. Consider dropping it from K=5 cold-start experiments
   or bumping max_turns.

3. **Cold-start AZ/ExIt is not viable on our 9-map roster at
   K=5, budget=40, max_turns=40.** Draws collapse the signal.
   Warm-start unblocks it; higher budget or larger K would too.

4. **Search budget 40 is a weak reference opponent.** Existing
   PPO-vs-MCTS-40 numbers should be read as loose upper bounds.
   For research-grade comparability we should re-evaluate against
   MCTS-200.

5. **Our GNN architecture likely isn't the current bottleneck.**
   The Bauer comparison identified upgrades but the calibration
   experiments (Q3/Q4) suggest the more actionable wins are (a)
   fixing the eval protocol and (b) raising eval-time search budget,
   not swapping GNN backbones.

---

## 8. K-sweep re-run on the optimized code (branch `perf/parallel-gpu`)

**Setup.** The optimization branch changed how PPO collects rollouts:
16 environments advance in lockstep and one batched forward serves them
all, instead of one long trajectory per worker. That changes the sample
composition and the order of random draws by design (the algorithm,
GAE and update are untouched), so old and new PPO numbers are not
bitwise comparable and the K-sweep had to be re-run before anything
else on the branch could be trusted. Same config as §4 — 21 maps, 200
iterations, batch 4096, 10 epochs, eval vs MCTS-50, 15 games per map —
except 12 rollout workers instead of 8. Output in
`experiments/k_sweep_ppo_200_perf/`; the originals are untouched.

**Result.** Both K values came out uniformly stronger, and the K effect
— the headline finding — is preserved exactly.

| | Original | Re-run | Change |
|---|---|---|---|
| K=5 aggregate | 0.800 | 0.927 | +0.127 |
| K=10 aggregate | 0.521 | 0.648 | +0.127 |
| **K=5 − K=10 drop-off** | **+0.279** | **+0.279** | **0.000** |

Per bucket:

| Bucket | K=5 orig → new | K=10 orig → new |
|---|---|---|
| small (6-12) | 0.793 → 0.881 | 0.733 → 0.785 |
| medium (16-22) | 0.867 → 0.956 | 0.445 → 0.659 |
| large (28-30) | 0.622 → **0.978** | 0.111 → 0.200 |

Three maps exceed two sigma at each K (about 1 expected by chance), all
upward. Wall-clock 245 min → 68.7 min (3.6x): K=5 97.6 → 26.6, K=10
147.4 → 42.1.

**What holds.**
1. **The K=5 → K=10 drop-off holds**, at +0.279 in both runs. Read with
   Q4 (§7), which showed the MCTS opponent also strengthens with K, the
   interpretation of §4 is unchanged.
2. **Large maps stay hard at K=10**: 0.111 → 0.200, still the worst
   bucket by far.
3. **The corridor collapse at K=10 holds**: corridor_20 0.00 → 0.13,
   corridor_22 0.00 → 0.27, corridor_28 0.00 → 0.00.

**What does not hold.** The large-map weakness *at K=5* largely
disappears: 0.622 → 0.978, with corridor_28 0.53 → 1.00, grid_30
0.67 → 1.00, hub_ring_30 0.67 → 0.93. That looks partly like an
artifact of the old rollout collection rather than a property of the
maps: one long trajectory per worker makes a batch highly correlated in
time, while 16 lockstep environments decorrelate it and spread coverage
across maps, which is why parallel environments are standard in PPO
implementations. Any claim resting on "PPO degrades on large maps"
needs the qualifier "at K=10"; at K=5 it does not.

**What it does not establish.** This is a *single* training run, so the
21 maps are not 21 independent tests — they are 21 evaluations of one
policy, and the per-map z-scores overstate the evidence because of that
shared dependence. Two changes are also confounded: the rollout
collection and 12 workers instead of 8. Before this is treated as a
finding rather than a strong signal, run 2-3 seeds per K, and ideally
one seed with 8 workers to separate the two changes.

Scripts: `experiments/k_sweep_ppo.py` (unchanged),
`experiments/compare_reruns.py` (new; per-map z-scores against the
sampling noise of 15 eval games, bucket means, drop-off test).
Branch: `perf/parallel-gpu`; `mcts_gnn_selfplay` and `main` untouched.

---

## 9. ExIt at MCTS budget 200, K=10 — warm start vs cold start

**Setup.** The experiment §2 and §3 could not afford: Expert Iteration on
all 21 maps at `simulation_budget=200`, `action_budget=10`,
`max_turns=40`, 50 iterations of 96 self-play games on 12 workers (16
games per worker in lockstep), eval every 10 iterations against
MCTS-uniform at the *same* budget, 12 games per map. Run twice — warm
start from the K=10 PPO checkpoint, and cold start from random init.
3.7 h and 3.5 h respectively on the optimized branch; at the old
self-play rate of 9.0 s/game each would have taken ~13 h.

**Reading the metric.** 7 of the 21 maps draw all 12 eval games at every
point (bipartite_20, the three corridors, dual_hub_20, grid_30,
hub_ring_30) and sit at exactly 0.50 forever, so the 21-map mean is
diluted. The numbers below are over the **14 informative maps**. With 12
games per map the per-map SE is ~0.14 and the SE of the 14-map mean is
~0.037.

| | first eval (iter 10) | last eval (iter 50) | change |
|---|---|---|---|
| warm start | 0.351 | 0.390 | +0.039 (~1.0 SE) |
| cold start | 0.179 | 0.274 | **+0.095 (~2.6 SE)** |

Self-play draw fraction over the last 10 iterations: warm 0.455, cold
0.449. Figure: `experiments/exit_b200_k10/warm_vs_cold.png`.

**What it changes.**

1. **§2 is overturned: cold start IS viable at budget 200 / K=10.** That
   entry concluded cold-start AZ/ExIt "is not viable on our 9-map roster
   at K=5, budget=40, max_turns=40" because self-play produced ~100%
   draws and the advantage signal vanished. At budget 200 with K=10,
   cold-start self-play draws only ~45% of its games and the policy
   improves by 2.6 SE. The collapse was an artifact of the search budget
   and action budget, not a property of the game. This answers the
   roadmap's open item "cold-start bootstrap via higher MCTS budget
   (200+) — the honest test of whether AZ/ExIt can escape cold-start at
   feasible compute": **yes, it escapes.**

2. **But Tier-1 §1.1 (budget 40 → 200) does not buy same-budget parity.**
   The warm-start run gained only ~1 SE and ends at 0.390, still well
   below the 0.50 that would mean matching MCTS-uniform at equal budget.
   §3 saw the same shape at budget 40 / K=5: rises modestly, plateaus
   under 50%. Raising the budget 5x and K 2x reproduced it. Search
   budget was not the bottleneck, so the remaining Tier-1 candidates
   (TransformerConv/GATv2 backbone, attention pooling) or the evaluation
   framing (compute-efficiency: MCTS+GNN at budget B vs MCTS-uniform at
   4B) are the honest next moves, not more search.

3. **Cold start improves more than warm start** (+0.095 vs +0.039) while
   ending lower (0.274 vs 0.390). Consistent with the warm-started
   policy already sitting near this method's ceiling, so ExIt has little
   left to extract, while the random-init policy has room and ExIt does
   move it. Neither approaches parity.

4. **`max_turns=40` wastes a third of the roster.** 7 maps carry no
   signal at all. Either raise `max_turns` for the large maps or drop
   them from this comparison; the earlier decision to defer `max_turns`
   100 should be revisited before the next ExIt run.

**Caveats.** One seed per arm. Per-map changes are mostly inside noise;
only the cold-start aggregate clears 2 SE. The eval opponent is
MCTS-uniform at the same budget, which Q3/Q4 (§6, §7) showed is a strong
reference at K=10.

Scripts: `experiments/mcts_gnn_exit_training_run.py`,
`experiments/plot_exit_warm_vs_cold.py`.
Results: `experiments/exit_b200_k10/`, `experiments/exit_b200_k10_cold/`.

---

## 10. Why MCTS+GNN underperformed PPO: progressive widening proposed random actions

**The puzzle.** §9 left ExIt at 0.390 (warm) and 0.274 (cold) against
MCTS-uniform at the same budget, and the roadmap already recorded a PPO
checkpoint scoring ~94% raw but only 31% when wrapped in MCTS(budget=40).
Wrapping a good policy in search made it *worse*, at any budget.

**Diagnosis.** `DuctMCTS` grows a node's action set by progressive
widening, `visits ** pw_alpha` with `pw_alpha=0.5`, drawing each candidate
from `pw_sampler`. The default is `masked_random`, so at budget 200 a root
holds ~14 *uniformly random* joint actions per agent and the GNN only
ranks that pool — it never proposes anything. At K=10 on a 20-30
territory map the joint space is astronomically large, so the pool never
lands near the policy's mode. Measured on 25 states per map, log-prob of
the policy's own action minus the best of 14 random candidates:

| map | advantage | random beat policy |
|---|---|---|
| simple_6 (6) | +0.85 nats | 10/25 states |
| dense_12 (12) | +4.60 | 3/25 |
| grid_20 (20) | +11.93 | 0/25 |
| hub_ring_30 (30) | +7.89 | 2/25 |

Roughly 150,000x on grid_20. The priors are *not* the problem: softmaxed
they concentrate at 0.68-0.87 on their favourite, far from the uniform
0.071. The GNN ranks decisively; it ranks fourteen random joint actions.
This is the structural difference from AlphaZero, which expands children
using the policy's own top actions.

**The fix, measured without any training.** Took the same K=10 PPO
checkpoint, played it as MCTS+GNN(budget 200) against
MCTS-uniform(budget 200), 8 maps x 12 games, identical seeds and colours,
changing only `pw_sampler`:

| map | random | gnn | delta |
|---|---|---|---|
| simple_6 (6) | 0.417 | 0.667 | +0.250 |
| medium_8 (8) | 0.625 | 0.917 | +0.292 |
| large_10 (10) | 0.375 | **1.000** | +0.625 |
| dense_12 (12) | 0.417 | 0.792 | +0.375 |
| hub_spoke_16 (16) | 0.542 | 0.958 | +0.417 |
| hex_grid_18 (18) | 0.292 | 0.708 | +0.417 |
| grid_20 (20) | 0.500 | 0.875 | +0.375 |
| hub_ring_30 (30) | 0.500 | 0.500 | +0.000 |
| **mean** | **0.458** | **0.802** | **+0.344** |

**What it changes.**

1. **The MCTS+GNN "failure" was a configuration bug, not an algorithmic
   limit.** With policy-proposed candidates the same weights go from
   losing (0.458) to clearly beating MCTS-uniform at equal budget
   (0.802). No training, no architecture change, one flag.
2. **§9, §3 and §2 all trained and evaluated through a crippled search.**
   Their conclusions about ExIt's ceiling, and the Tier-1 verdict in §9
   that "search budget was not the bottleneck", describe
   `pw_sampler='masked_random'` only. The ExIt runs need repeating with
   `gnn` before anything is concluded about the training method.
3. **The 94% -> 31% collapse in the roadmap is explained.**
4. **hub_ring_30 is unaffected** because it draws all 12 games in both
   arms at `max_turns=40` — the ceiling problem of §9 item 4, not a
   failure of the fix. The gain on simple_6 is the smallest non-zero one
   (+0.250), as predicted: random sampling covers a 6-territory space.

**Cost.** GNN proposals need a network forward per candidate: 3.9 s per
decision vs 0.85 s. Worth it at eval; it makes a 50-iteration ExIt run
roughly 12 h instead of 3.7 h.

**Caveats.** One checkpoint, 12 games per map (per-map SE ~0.14, but the
mean shift of +0.344 over 8 maps is ~7 SE). `pw_sampler='gnn'` at *cold
start* is still expected to widen poorly, since an untrained policy is
peaked on noise — the docstring's original warning stands for that case.

Scripts: `experiments/mcts_gnn_exit_training_run.py --pw-sampler`,
scratch diagnostics in the session scratchpad.

---

## Open threads (nothing running now)

- **K=15 / K=20 K-sweep completion** — ~7h combined wall-clock.
  Eval-cap fix is committed; can resume cleanly.
- **Q4 at higher budget** (K=10 vs K=5 at budget=200) — does the K
  advantage persist when both sides have strong search? ~25 min.
- **Q4 across a wider K sweep** (K ∈ {5, 10, 15, 20}) — does the
  K-strength gap plateau? ~1 hour.
- **K-controlled re-plot of the existing K-sweep results** — report
  PPO win-rate against MaskedRandom instead of MCTS, to factor out
  MCTS's K-inflation. Cheap (~20 min).
- **Tier-1 architecture upgrades** — GCN → TransformerConv/GATv2,
  attention pooling, eval budget 40 → 200 (see
  ARCHITECTURAL_IMPROVEMENTS.md).
- **Phase 4** — coevolution (Bauer-style GA) implementation.
