# Experiment Proposals

Menu of candidate next experiments, ranked within each cost tier. Draw
from this as a queue — pick items when we decide what to run next.

Companion to [docs/EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) (what we've
already run) and [docs/RL_TRAINING_ROADMAP.md](RL_TRAINING_ROADMAP.md)
(plan).

Entry format: **Question · Setup · Cost · Expected outcome · Files
touched**.

---

## Tier A — Cheap diagnostics (< 1 hour each)

### A1. K-controlled re-plot of the K-sweep
- **Question:** Once we factor out MCTS's K-inflation (Q4 finding),
  does PPO still show the K=5 → K=10 drop-off, or does the drop
  disappear entirely?
- **Setup:** Re-evaluate the existing K-sweep PPO checkpoints against
  a fixed baseline — either MaskedRandom, or MCTS-K5 at all K values.
  Regenerate the K-sweep dashboard with the new metric.
- **Cost:** ~20 min (no new training).
- **Expected outcome:** Cleaner K-sweep interpretation. If drop-off
  persists, it's a real training-difficulty story. If it disappears,
  the drop-off was almost entirely opponent-inflation.
- **Files touched:** new
  `experiments/k_sweep_reeval_common_baseline.py`; consumes existing
  checkpoints under `experiments/k_sweep_ppo_200/`.
- **Priority:** highest — retroactively fixes interpretation of every
  K-sweep run we've done.

### A2. Q4 at budget=200
- **Question:** Does the K=10 > K=5 advantage survive when both sides
  have strong search, or does deeper search saturate the difference?
- **Setup:** Rerun Q4 with `--budget 200`. Same 9 maps, 20 games/map.
- **Cost:** ~25 min.
- **Expected outcome:** If K advantage persists ≥ 0.70, K is a
  structural strength lever independent of search. If drops toward
  0.50, K's edge is a low-search artifact.
- **Files touched:** none — rerun of `diagnostic_mcts_k_strength.py`.
- **Priority:** high — settles whether "MCTS-K10 at strong search" is
  the target opponent for research eval.

### A3. Q4 wider K sweep
- **Question:** Does the K-strength curve plateau, or keep growing?
- **Setup:** Q4-style pairwise at fixed budget=40: `K=10 vs K=5`,
  `K=15 vs K=10`, `K=20 vs K=15`. Small script wrapping
  `diagnostic_mcts_k_strength.py`.
- **Cost:** ~30 min.
- **Expected outcome:** Diminishing returns curve. Informs whether
  K=20 is worth chasing for the research setup.
- **Files touched:** thin wrapper script.
- **Priority:** medium — completes the K-strength picture.

### A4. Draw-map max_turns sensitivity
- **Question:** Are medium_8 / star_8 / dense_12 "genuine draws" or
  "just need more time"?
- **Setup:** Rerun Q3 (MCTS-200 vs MCTS-40) on those 3 maps at
  `max_turns ∈ {40, 80, 120}`.
- **Cost:** ~15 min.
- **Expected outcome:** If draws persist at 120, the maps are
  structurally stalematey — drop from cold-start K=5 experiments. If
  they resolve at 80, bump max_turns.
- **Files touched:** none — CLI variation of
  `diagnostic_mcts_budget.py`.
- **Priority:** medium — cleans up interpretation of cold-start
  failures.

### A5. Q2 — MCTS+GNN vs MCTS-uniform at fair compute
- **Question:** Does the trained GNN policy still fail at high search
  budget, or was the 31% pathology (from earlier diagnostic) a
  low-budget artifact?
- **Setup:** Load an existing trained MCTS+GNN checkpoint. Play
  `MCTS+GNN(budget=200) vs MCTS-uniform(budget=200)`, 9 maps, 20
  games/map.
- **Cost:** ~1 hour.
- **Expected outcome:** If GNN policy wins ≥ 55%, our AZ/ExIt work
  produced a useful artifact at fair compute. If ≤ 50%, the priors
  aren't helping and we need Tier-1 architecture upgrades.
- **Files touched:** thin script combining `mcts_gnn_agent.from_checkpoint`
  with the `diagnostic_mcts_budget.py` matchup loop.
- **Priority:** high — is our GNN actually good?

### A6. Q4 — compute-efficiency ratio (Bauer-style claim)
- **Question:** Does `MCTS+GNN(budget=40)` beat `MCTS-uniform(budget=200)`?
- **Setup:** Load MCTS+GNN checkpoint. Head-to-head at asymmetric
  search budget.
- **Cost:** ~1 hour.
- **Expected outcome:** If yes, we have a defensible "priors are
  worth 5× compute" claim — the standard AZ/ExIt success metric.
- **Files touched:** thin script (variant of A5).
- **Priority:** high — the flagship research metric for MCTS+GNN
  success.

---

## Tier B — Medium-effort experiments (few hours)

### B1. K=15 / K=20 K-sweep completion
- **Question:** Does the K-sweep drop-off keep going at K > 10, or
  plateau?
- **Setup:** Run the existing k_sweep_ppo.py at K=15 and K=20 for
  200 iterations each. The eval-cap fix is committed; no more
  ProcessPool hang.
- **Cost:** ~3.5 h per K value (7 h total).
- **Expected outcome:** Full K-sweep curve. Combined with A1 gives
  the true "how much of the drop-off is training-difficulty" answer
  at all K values.
- **Files touched:** none — rerun of `experiments/k_sweep_ppo.py`.
- **Priority:** medium — completes the K-sweep story.

### B2. Warm-start ExIt at higher MCTS budget
- **Question:** Does ExIt learn faster / plateau higher when
  self-play uses budget=200 instead of 40?
- **Setup:** Rerun warm-start ExIt with `mcts.simulation_budget: 200`
  in the config, same 9 maps, 50 iterations.
- **Cost:** ~9 h (5× current wall-clock).
- **Expected outcome:** Better eval curves if MCTS training-data
  quality was the bottleneck. Same-or-worse curves if the ceiling
  was elsewhere (opponent strength, GNN capacity).
- **Files touched:** `mcts_gnn_exit.yaml` config; no code change.
- **Priority:** medium — the natural "does more search help ExIt?"
  test. Overnight-viable.

### B3. Multi-checkpoint eval battery
- **Question:** Which existing checkpoint is actually strongest?
- **Setup:** All committed MCTS+GNN and PPO+GNN checkpoints played
  head-to-head in a round-robin across the 9-map roster, 5 games per
  matchup per map. Emit a rating (Elo or Bradley-Terry).
- **Cost:** ~1-2 hours depending on checkpoint count.
- **Expected outcome:** Objective ranking; identifies which
  checkpoint is worth building on next.
- **Files touched:** new
  `experiments/checkpoint_round_robin.py`, reuses `run_matchup` from
  `compare_mcts_gnn.py`.
- **Priority:** medium — good hygiene before further training runs.

---

## Tier C — Architecture upgrades (from ARCHITECTURAL_IMPROVEMENTS.md)

### C1. MCTS budget 40 → 200 in eval + training configs
- **Setup:** Config-only change on the eval matchups and warm-start
  ExIt/AZ self-play. See ARCHITECTURAL_IMPROVEMENTS.md §1.1.
- **Cost:** minutes for config change; wall-clock 5× per run.
- **Priority:** high — validated by Q3.

### C2. GCN → GATv2 or TransformerConv
- **Setup:** New file
  `parallel_risk/models/gnn_gat.py` or `gnn_transformer.py`;
  config-driven backbone selection. Retrain warm-start ExIt with
  new backbone. See ARCHITECTURAL_IMPROVEMENTS.md §1.2.
- **Cost:** 1-2 days implementation + 1 training run.
- **Priority:** medium — Tier-1 architecture upgrade with strongest
  literature precedent.

### C3. Attention pooling for value head
- **Setup:** Replace `global_mean_pool` with `GlobalAttention` on
  `GCNPolicy.forward`. See ARCHITECTURAL_IMPROVEMENTS.md §1.3.
- **Cost:** hours implementation; retrain to verify.
- **Priority:** low — combine with C2.

### C4-C6. Tier-2 architectural upgrades
- Region-node message passing (§2.1), intra-turn mask conditioning
  (§2.2), joint (dst, troops) scoring (§2.3). See
  ARCHITECTURAL_IMPROVEMENTS.md for details.
- **Cost:** 3-5 days each.
- **Priority:** medium-low — worth it only if Tier-1 upgrades don't
  move the metric.

### C7-C10. Tier-3 architectural upgrades
- Full joint-triple scoring, per-slot sequential forward passes,
  variable-length action space, env-in-loop decoding. See
  ARCHITECTURAL_IMPROVEMENTS.md §3.
- **Cost:** 1-2 weeks each; likely breaks existing checkpoints.
- **Priority:** low — only after Tier 1 and 2 have been tried.

---

## Tier D — Phase 4 (Coevolution)

### D1. Bauer-style GA population training
- **Question:** Does coevolutionary self-play produce stronger
  policies than PPO or ExIt on this environment?
- **Setup:** PyGAD-based genetic algorithm training loop.
  Population size ~30, ~25 generations. Best individual becomes the
  MCTS prior for the next iteration. Reference:
  `github.com/andenrx/py-risk`.
- **Cost:** ~1 week implementation; multi-day training runs.
- **Priority:** medium — the third major RL technique after PPO and
  MCTS+GNN; central to the research plan.

---

## Recommended sequence (my read as of 2026-09-09)

1. **A1** (~20 min) — cheapest and highest-leverage; retroactively
   fixes interpretation of everything already run.
2. **A5** (~1 h) — is our GNN policy actually useful at fair compute?
   Answer decides whether next work is more training or architecture.
3. **A6** (~1 h) — compute-efficiency ratio; the flagship claim.
4. **A2** (~25 min) — settles the Q4-at-strong-search question.
5. **B1** (7 h, overnight) — complete K-sweep.
6. Then Tier C or D based on what A5/A6 revealed.
