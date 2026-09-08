# RL Training Roadmap

**Last Updated:** 2026-09-08
**Status:** Phase 1 ✓ | Phase 2 ✓ (multi-map + transfer) | MCTS ✓ | Phase 3 (AZ + ExIt) ✓ implementation, research validation ongoing | Phase 4 (Coevolution) planned

## Overview

Long-arc plan: (1) implement the state-of-the-art RL techniques for adversarial games on this environment, (2) invest in engineering so the testbed can support serious research, (3) once (1) and (2) are done, use the resulting infrastructure to conduct a scientific study — likely on topology-invariant policy learning, method comparison across game complexity regimes, or a novel algorithmic contribution.

The techniques we're implementing are:
1. Standard architectures with flat observations (PPO with MLPs) ✓
2. Graph-based observations for multi-map flexibility (PPO with GNNs) ✓
3. Tree search baselines (Decoupled UCT for simultaneous-move games) ✓
4. MCTS + GNN in an AlphaZero-style loop (distillation + Expert Iteration variants) ✓
5. Coevolutionary self-play (Bauer-style population-based training) — planned

## Strategic Considerations

### Why Two Phases?

**Phase 1: Baseline with Flat Observations**
- Validates that the environment is learnable
- Establishes performance benchmarks
- Proves reward shaping and self-play infrastructure
- Lower risk, faster initial results

**Phase 2: Graph Neural Networks**
- Enables training across multiple map sizes simultaneously
- Supports transfer learning between maps
- Future-proofs architecture for arbitrary map topologies
- Research contribution: GNN architectures for turn-based strategy games

**Phase 3: AlphaZero-Style Training**
- Combines MCTS tree search with a learned GNN value/policy network
- MCTS generates high-quality training data (state → policy distribution + value)
- GNN learns to approximate MCTS outputs, then guides search via PUCT
- Iterative improvement: better GNN → better search → better training data
- Enables strong play without the compute cost of deep search at inference

### Graph-Based Observations: Key Motivations

**Current Challenge:** Fixed-size observation spaces only work for a single map size.

**With Graphs:**
- Maps naturally represented as graphs (territories = nodes, adjacency = edges)
- GNNs handle variable-sized inputs (6 territories or 20 territories, same model)
- Message passing captures territorial relationships more naturally
- Transfer learning: model trained on small maps can generalize to larger ones

**Framework Implications:**
- RLlib has limited GNN support → good for Phase 1, not Phase 2
- TorchRL + PyTorch Geometric → flexible for custom GNN architectures
- Requires custom batching for variable-sized graphs

---

## Phase 1: Baseline Training — COMPLETE ✓

### Step 1: Reward Shaping — COMPLETE ✓
- `parallel_risk/env/reward_shaping.py` with 4 configurable components
- Preset configurations (dense, sparse, territorial, aggressive)
- Tests: `tests/test_reward_shaping.py` (8/8 passing)
- Documentation: `docs/REWARD_SHAPING.md`

### Step 2: RLlib Integration — COMPLETE ✓
- `parallel_risk/training/rllib/` wrapper, training script, YAML configs
- Fixed-budget action space, self-play configuration
- Tests: `tests/test_rllib_wrapper.py` (7/7 passing)
- Documentation: `docs/RLLIB_INTEGRATION.md`

### Step 3: Evaluation Harness — COMPLETE ✓
- `parallel_risk/evaluation/` with `evaluate_agent`, `league_evaluator`, `visualize`, `league_visualize`
- Validation experiments: `experiments/validate_learning.py`, `experiments/self_play_league.py`

### Step 4: Baseline Experiments — COMPLETE ✓
- **Result:** 100% win rate vs random by iteration 60
- Learning curves in `experiments/phase1_learning_results/`
- Self-play league results in `experiments/league_results/`

---

## Phase 2: Graph Neural Networks — COMPLETE ✓

### Step 1: Graph Observation Wrapper — COMPLETE ✓
- `parallel_risk/training/torchrl/graph_wrapper.py`
- Tests: `tests/test_graph_wrapper.py` (5/5 passing)

### Step 2: GNN Policy Architectures — COMPLETE ✓
- `parallel_risk/models/gnn_gcn.py` with actor-critic heads
- `parallel_risk/models/action_decoder.py` with autoregressive masking
- Tests: `tests/test_gnn_policy.py` (4/4 passing)

### Step 3: TorchRL Training Loop — COMPLETE ✓
- `parallel_risk/training/torchrl/train.py` with PPO + self-play
- Rollout collection, GAE, TensorBoard, checkpointing
- Tests: `tests/test_training.py` (5/5 passing)

### Step 4: Multi-Map Training & Generalization — COMPLETE ✓

**Completed sub-tasks:**
- [x] GNN agent validates at 99.5% win rate vs random (`experiments/phase2_learning_results/`, `experiments/phase2_revalidation/`)
- [x] Action masking implemented (autoregressive, both RLlib and TorchRL)
- [x] `medium_8` and `large_10` maps added to `map_config.py`
- [x] `PPOTrainer` supports `map_names` (list) and samples uniformly across maps per rollout
- [x] Multi-map training experiment: single GNN trained on all 3 maps simultaneously
- [x] Transfer learning experiment: 2-map (simple_6 + medium_8) evaluated zero-shot on `large_10`
- [x] Visual results: `experiments/multi_map_results/{learning_curves,final_performance,transfer_comparison}.png`

**Results (200 iterations, evaluated vs MCTS-50 over 100 eps/map):**

| Map        | 3-map model | 2-map model (zero-shot) |
|------------|------------:|------------------------:|
| `simple_6` |         94% | —                       |
| `medium_8` |        100% | —                       |
| `large_10` |         98% |                     90% |

- All three maps saturate above 90% (`medium_8` at 100%).
- Zero-shot transfer gap on `large_10`: 8pp (90% → 98% with in-training exposure).
- Full trajectory + config: `experiments/multi_map_results/multi_map_results.json`; write-up in `experiments/multi_map_results/RESULTS.md`.
- Enabled by GPU-friendly update path + parallel workers (commits `3d53b9c`, `3db454f`); full 200-iter + transfer run in <10 min on RTX 5080.

**Success criteria:**
- [x] Single GNN agent achieves >80% win rate on all map sizes simultaneously (94/100/98%)
- [x] Agent trained on small maps achieves >50% win rate on unseen large maps without fine-tuning (90% on `large_10` zero-shot)
- [ ] Convergence speed improves with multi-map pre-training vs. training from scratch per map *(not measured in current run — separate per-map baseline needed for a clean comparison)*

**Caveat:** All three current maps share the same 3-region schema and comparable action budgets. A stronger generalization test would use a map with a different region count or degree distribution (see [Future Work](#future-work)).

---

## MCTS Baseline — COMPLETE ✓

**Decoupled UCT for simultaneous-move games:**
- `parallel_risk/agents/mcts_agent.py` (full implementation)
- Experiments: `experiments/validate_mcts.py`, `experiments/compare_mcts_gnn.py`
- Results: MCTS (budget=200) at 99.5% win rate vs masked-random
- Results: MCTS vs GNN comparison — see note below

**Note on MCTS vs. GNN comparison:** The available comparison (`experiments/mcts_vs_gnn_results_corrected/`) was run against GNN weights from an earlier training run. The best-known GNN checkpoint was trained afterward. A fair comparison between MCTS and the best GNN remains to be run as part of Phase 3 baseline establishment.

---

## Phase 3: MCTS + GNN Training — IMPLEMENTATION COMPLETE ✓, VALIDATION ONGOING

Two coexisting training tracks land on branch `mcts_gnn_selfplay`:
- **AZ-distillation**: `parallel_risk/training/mcts_gnn/{self_play,trainer,train}.py` — MCTS self-play → distil visit distribution + outcome
- **Expert Iteration (ExIt)**: `parallel_risk/training/mcts_gnn/exit_{self_play,trainer,train}.py` — advantage-weighted actor-critic on MCTS-selected actions
- **Shared agent**: `parallel_risk/agents/mcts_gnn_agent.py` — MCTSGNNAgent with configurable PW sampler + value_fn
- **DuctMCTS extensions**: PUCT selection with softmax-normalized priors, Dirichlet root noise (with min_actions widening), temperature-based action sampling, visit-distribution accessor
- **Cross-track interop**: shared checkpoint schema means weights hand off freely between PPO, AZ, and ExIt via `MCTSGNNAgent.from_checkpoint` / `GNNAgent.from_checkpoint`
- **Region padding utility**: `parallel_risk/training/mcts_gnn/checkpoint_utils.py` zero-pads input/global projections when warm-starting from a checkpoint trained with fewer regions

### Sub-step 1: Fair MCTS vs GNN baseline — DEFERRED
The diagnostic script `experiments/diagnostic_mcts_gnn_vs_uniform.py` covers the head-to-head (MCTS-uniform-vs-uniform sanity, random-init MCTS+GNN vs MCTS-uniform, trained MCTS+GNN vs MCTS-uniform). Full budget-sweep comparison at multiple map sizes deferred pending the larger-map roster.

### Sub-steps 2, 3, 4, 5: AZ + ExIt implementation — COMPLETE ✓
All committed on `mcts_gnn_selfplay`. Smoke tests in `tests/test_mcts_gnn_{agent,selfplay,exit}.py` all pass.

### Key findings from the diagnostic journey (documented in the commit message of `10d553d`)

1. **Cold-start MCTS+GNN self-play fails on 9 diverse maps at budget=40**, regardless of loss function (AZ distillation OR ExIt). Root cause is environmental, not algorithmic: at cold-start, both self-play sides play too weakly to conquer within max_turns=40 on medium+ sized maps, so 100% of games truncate as draws (z=0), and the advantage signal collapses. Diagnostic script confirms: mask clamp rate <1%, code correct, value head DOES learn on `simple_6` where cold-start CAN produce decisive games.

2. **Warm-starting from the PPO checkpoint sidesteps cold-start.**
   - Warm-start AZ (single map `simple_6`, 50 iters): eval 25% → 44% vs MCTS(uniform, budget=40) — **+19pp**.
   - Warm-start ExIt (8 maps excluding `hex_grid_10` for ckpt dim compat, 50 iters): mean eval 15.6% → 19.8% (**+4.2pp**), peak 20.8% at iter 30. Substantial per-map gains: `simple_6` +9pp, `large_10` +16pp, `star_8` +17pp. Dashboard evidence: `experiments/mcts_gnn_exit_warm_start/dashboard.png`.
   - 9-map warm-start ExIt convergence run (200 iters, with region-padded PPO checkpoint) in progress at time of writing.

3. **MCTS at budget=40 is a bad wrapper for a good policy on these small state spaces.** PPO checkpoint alone scores ~94% vs MCTS-50, but wrapped in MCTS(budget=40) drops to 31.25% vs MCTS(uniform). The tree isn't deep enough for search-improvement over a good prior; it commits early to prior-preferred moves. This caps the achievable win-rate on `MCTS(GNN, budget=40) vs MCTS(uniform, budget=40)` regardless of training. **To honestly evaluate MCTS+GNN's value, use the roadmap's compute-efficiency metric (`MCTS(GNN, budget=X) vs MCTS(uniform, budget=4X)`) rather than same-budget comparison.**

### Success Criteria (revised, honest)
- [x] AZ + ExIt training frameworks implemented, cross-compatible with PPO checkpoints
- [x] PUCT + Dirichlet + temperature + visit-distribution + softmax-priors landed on `DuctMCTS`
- [x] Warm-start AZ shows real learning on a single map (+19pp)
- [x] Warm-start ExIt shows real learning on 8 maps (+4-17pp per-map)
- [ ] **Convergence run** on all 9 maps with 200 iters — in progress
- [ ] **Raw GNN policy** (no MCTS) from ExIt training evaluated vs MCTS-50 to compare against Phase 2.4 numbers (94/100/98%)
- [ ] **Compute-efficiency metric**: MCTS(GNN, budget=40) vs MCTS(uniform, budget=200) — roadmap's actual "4× efficiency" success criterion. Meaningful because at budget=40 alone MCTS caps a good policy; a stronger opponent gives the GNN room to earn its keep.
- [ ] Cold-start bootstrap via higher MCTS budget (200+) as a follow-up experiment — the honest test of whether AZ/ExIt can escape cold-start on this game at feasible compute.

---

## Phase 4: Coevolutionary Self-Play — PLANNED

### Motivation
Round out the technique matrix with a population-based / evolutionary approach. Complementary to gradient-based methods: no need for well-defined gradients, natural fit for multi-agent competitive games, sidesteps some self-play mode-collapse pathologies via population diversity.

### Reference
- **Bauer, A.** "Artificial intelligence with graph neural networks applied to a risk-like board game." *IEEE Transactions on Games*, v. 16, n. 2, p. 342-351, **2024**.
- **Bauer, A.** `py-risk` repo (2023). <https://github.com/andenrx/py-risk> — read this when nailing down the exact algorithmic recipe.

Modern parallels for background: [Transformer Guided Coevolution](https://arxiv.org/pdf/2410.13769) (2024) and [Coevolutionary Deep RL](https://ieeexplore.ieee.org/document/9308290/) (2020).

### Design space (to be pinned down from Bauer 2024 + py-risk repo)
- **Individual encoding**: full GCN weights, a subset (last layer only?), or hyperparameters
- **Population size**: 8-64
- **Selection**: tournament, ranking, Pareto-based
- **Variation operator**: Gaussian weight noise (ES-style), crossover between parents, or something Bauer-specific
- **Fitness**: round-robin win-rate, Elo, top-K challenger vs incumbent
- **Diversity pressure**: novelty search, quality-diversity, none
- **Inner loop**: purely evolutionary, or hybrid with a short gradient descent per generation

### Infrastructure fit (mostly already in place)
- Checkpoints are the natural "individuals" — same schema across our tracks
- Pairwise evaluation harness exists (used for MCTS+GNN vs MCTS(uniform))
- Spawn-worker parallelism proven for parallel game evaluation
- `MapRegistry` + `max_regions` padding means population members can train/eval on any map subset

### To add
- Population manager (checkpoint pool + fitness bookkeeping)
- Matchmaking scheduler (round-robin or Elo-based challenger selection)
- One or two variation operators (start with Gaussian noise)
- Coevolution training script + config

Rough estimate: 2-3 days of clean work once the specific Bauer recipe is nailed down.

### Success Criteria
- [ ] Coevolution training loop implemented, checkpoints cross-compatible with PPO/AZ/ExIt
- [ ] Population converges to a Pareto front of strategies distinguishable qualitatively
- [ ] Best coevolved agent competitive with best gradient-trained agent on a fair benchmark
- [ ] Compute-efficiency comparison: coevolution vs PPO vs AZ vs ExIt at matched wall-clock

---

## Technique Inventory

| Family | Track | Status | Where |
|---|---|---|---|
| Policy gradient (flat obs) | PPO + MLP via RLlib | ✓ Complete | `parallel_risk/training/rllib/` |
| Policy gradient (graph obs) | PPO + GNN via TorchRL | ✓ Complete | `parallel_risk/training/torchrl/` |
| Tree search baseline | Decoupled UCT | ✓ Complete | `parallel_risk/agents/mcts_agent.py` |
| MCTS + neural, distillation | AZ-style visit-distribution distillation | ✓ Implementation | `parallel_risk/training/mcts_gnn/{self_play,trainer,train}.py` |
| MCTS + neural, actor-critic | Expert Iteration (advantage-weighted) | ✓ Implementation | `parallel_risk/training/mcts_gnn/exit_{self_play,trainer,train}.py` |
| Coevolutionary self-play | Bauer 2024 population-based | ✗ Planned (Phase 4) | TBD |

**Cross-technique interop**: single checkpoint schema (`policy_state_dict` + `optimizer_state_dict` + `config` with embedded `max_regions`). Any checkpoint loads into any of the trainers via `MCTSGNNTrainer.load_checkpoint` / `ExitTrainer.load_checkpoint` / `MCTSGNNAgent.from_checkpoint` / `GNNAgent.from_checkpoint`, with region-width padding handled transparently.

---

## Engineering Roadmap

Once the technique matrix is filled out (Phase 4 lands), the testbed becomes a serious research tool. To get there, invest in the following engineering axes.

### Environment (`parallel_risk/env/`)
- Consistent action-budget handling across configs (currently `action_budget=5` is hardcoded in many places; should scale with map size or be explicitly documented as fixed)
- Env-side game recording (JSON per game with full state trajectory) for offline analysis + replay
- Faster `env.step` if it becomes a bottleneck (profile first)
- Env-level validation on submitted action arrays with clearer error messages
- Metadata schema for maps: standard place to declare "difficulty", "expected turns", "region complexity" for cross-map analysis

### GNN Architectures (`parallel_risk/models/`)
- GCN baseline exists; add **GATv2** (attention-based message passing) as a comparison
- Edge features (currently just adjacency): encode combat-strength differentials, income proximity, etc.
- Deeper networks (currently 3 layers) — measure whether Parallel Risk actually needs long-range message passing
- Explicit hyperparameter sweep infrastructure (Optuna? or just a simple grid script)

### Maps (`parallel_risk/env/map_config.py`)
- Colleague is expanding beyond the current 9 unique maps — good
- **Larger maps (20-50+ territories)** — currently the state space is small enough that MCTS(uniform) is near-optimal and RL techniques don't differentiate. This is the single highest-leverage change for making the RL comparisons interesting.
- **Procedural map generation** — parameterized generator producing arbitrary connected topologies with a target region schema, so we can create a proper zero-shot transfer test set
- **Map difficulty calibration** — for each map, measure a "difficulty score" via how quickly MCTS(uniform, budget=200) vs random converges to a winner. Enables per-map fitness reporting instead of raw win-rate

### Action Space
- Higher `action_budget` (currently 5 slots per turn) — with more slots per turn the strategy space grows; likely necessary for larger maps
- **Adaptive action budget**: budget scales with map size (e.g. `max(5, n_territories // 2)`)
- Consider a **variable-length action space** where the number of slots per turn is dynamically chosen by the policy (currently fixed at 5)

### RL / Training Infrastructure
- **Unified benchmark script** — one entry point that runs any technique against any opponent on any map subset, produces a comparison dashboard. Removes per-technique custom eval scripts.
- **Elo / Glicko ranking system** across snapshots — currently we measure raw win-rates; a proper ranking would let us track continuous improvement over long training runs.
- **Standard hyperparameter naming** across the three trainers (currently PPOTrainer, MCTSGNNTrainer, ExitTrainer have slightly different config schemas)
- **Shared "problem spec" YAML** (map roster + action budget + max_turns + network architecture) with thin per-technique overlays for the loss-specific hyperparameters
- **Batched MCTS leaf evaluation** — queue K pending leaves and evaluate as one batch on GPU. Would substantially speed up MCTS+GNN training + inference and enable higher search budgets in practice.

### Testing + CI
- Cross-map smoke tests (currently many tests hardcode `simple_6`; should iterate over the full registry)
- Determinism tests (given a seed, self-play must produce identical trajectories)
- Regression tests on checkpoint interop (PPO → AZ → ExIt → back)

---

## Research Questions (updated)

1. **Method comparison at scale**: on maps of increasing size and action-budget, at what point does MCTS+GNN training start earning its keep over direct PPO? Where does coevolutionary self-play win over gradient-based?
2. **Topology-invariant learning via GNN**: can a single agent learn from a diverse map roster and zero-shot generalize to unseen topologies with different region schemas / degree distributions? (Phase 2.4 gave a promising 90% on unseen `large_10`; a proper study needs a bigger, more diverse test set.)
3. **Compute-efficiency of learned search priors**: does MCTS(GNN, low budget) match MCTS(uniform, high budget)? What's the crossover budget ratio as a function of map complexity?
4. **Interpretability**: does the trained GNN learn strategy concepts visible in its attention/message-passing patterns (chokepoint recognition, region completion, defensive lines)?
5. **Coevolution vs gradient-based training**: do coevolved policies find qualitatively different strategies (Bauer 2024 vs our AZ/ExIt/PPO)? Are they more robust to distribution shift?
6. **Novel algorithmic direction**: given all four techniques as baselines, is there a hybrid (e.g. PBT-wrapped ExIt, or MCTS-guided coevolution) that clearly beats each individual baseline?

---

## Infrastructure & Tooling

### Experiment Tracking
- TensorBoard: training curves
- Results JSON + matplotlib plots in `experiments/` subdirectories
- All configs checked into git

### Reproducibility
- Fixed random seeds for all experiments
- Version pinning: Python, PyTorch, TorchRL, PyG
- All configs checked into git

---

## Success Metrics

### Phase 1 ✓
- [x] Agent beats random baseline >90% win rate (achieved 100%)
- [x] Agent shows strategic behavior (captures regions, efficient combat)
- [x] Reproducible training in <24 hours
- [x] Documented best practices for reward shaping

### Phase 2 ✓
- [x] GNN agent trains and achieves 99.5% win rate vs random
- [x] Action masking implemented
- [x] GNN agent trains on multiple map sizes simultaneously (94/100/98% on simple/medium/large vs MCTS-50)
- [x] Positive transfer demonstrated: 2-map model → 90% zero-shot on unseen `large_10`

### MCTS ✓
- [x] Decoupled UCT implemented for simultaneous-move games
- [x] 99.5% win rate vs masked-random at budget=200
- [ ] Fair comparison against best GNN checkpoint

### Phase 3 (updated 2026-09-08)
- [x] Self-play data generation pipeline implemented (both AZ + ExIt variants)
- [x] Training loops implemented and validated end-to-end (warm-start required at feasible compute)
- [x] Warm-start AZ shows learning on single map (+19pp vs MCTS-uniform at same budget)
- [x] Warm-start ExIt shows learning on 8 maps (+4-17pp per-map)
- [ ] 9-map convergence run analyzed
- [ ] Raw GNN (no MCTS) from ExIt training compared to PPO baseline
- [ ] Compute-efficiency metric: MCTS(GNN, budget=40) vs MCTS(uniform, budget=200)
- [ ] Paper submission — deferred to after Phase 4 + engineering roadmap complete; the meaningful contribution requires the full technique matrix + a proper empirical study on larger maps

### Phase 4 (Coevolution, targets)
- [ ] Bauer 2024 recipe implemented; population manager + matchmaking + variation operator
- [ ] Coevolution checkpoints cross-compatible with existing tracks
- [ ] Best coevolved agent competitive with best gradient-trained agent on a common benchmark

---

## Future Work

Beyond the roadmap phases above, potential extensions:
- **Stronger generalization test:** add a map with a different region count and/or degree distribution to distinguish schema-transfer from true topology-invariant learning (current 3 maps all use the same 3-region schema).
- **Multi-map convergence-speed measurement:** run per-map baselines from scratch and compare against fine-tuning from the 3-map pre-trained model — cleanly quantifies the pre-training benefit.
- **Larger maps (20+ territories):** stress-test the GNN's inductive bias at scales not seen during training.

---

## Revision History

- **2026-04-07:** Initial roadmap created — two-phase approach
- **2026-04-21:** Phase 1 complete, Phase 2 Steps 1-3 complete
- **2026-09-03:** Phase 2 complete, MCTS complete; updated to reflect Phase 3 AlphaZero goals and Phase 2.4 multi-map as immediate next step
- **2026-09-06:** Phase 2.4 (multi-map + transfer) marked complete; 3-map win rates 94/100/98% and 90% zero-shot transfer on `large_10` recorded; Future Work section added
- **2026-09-08:** Phase 3 implementation landed (branch `mcts_gnn_selfplay`, commits 502ab75 + 10d553d): AZ distillation + Expert Iteration tracks + shared MCTSGNNAgent + PUCT/Dirichlet/temperature helpers + region-padded checkpoint interop. Diagnostic journey documented (cold-start collapse on 9 maps at budget=40 is environmental, warm-start from PPO works). Phase 4 (Coevolutionary self-play, Bauer 2024) added. Technique Inventory + Engineering Roadmap sections added. Research questions expanded to reflect the "engineering testbed → scientific contribution" framing.
