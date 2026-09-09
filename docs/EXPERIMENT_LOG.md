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
eval at `min(len(maps), 8)`. Per-map win-rate at 200 iters:
- K=5: aggregate ~ 0.72
- K=10: aggregate ~ 0.58 (drop-off)
- K=15, K=20: not yet run

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
