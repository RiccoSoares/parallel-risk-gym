# RL Training Roadmap

**Last Updated:** 2026-09-03  
**Status:** Phase 1 Complete ✓ | Phase 2 Complete ✓ | MCTS Complete ✓ | Phase 3 (AlphaZero) In Progress

## Overview

This document outlines the plan to enable reinforcement learning training on the Parallel Risk environment. The approach is designed to:
1. Establish baseline training capability with standard architectures ✓
2. Transition to graph-based observations for multi-map flexibility ✓
3. Implement MCTS for a strong tree-search baseline ✓
4. Combine MCTS + GNN in an AlphaZero-style loop for the strongest possible agent

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

### Step 4: Multi-Map Training & Generalization — IN PROGRESS ▶

**Completed sub-tasks:**
- [x] GNN agent validates at 99.5% win rate vs random (`experiments/phase2_learning_results/`, `experiments/phase2_revalidation/`)
- [x] Action masking implemented (autoregressive, both RLlib and TorchRL)

**Remaining:**
- [ ] Add medium (8-territory) and large (10-territory) maps to `map_config.py`
- [ ] Update TorchRL training to support multi-map training (sample across maps per rollout)
- [ ] Multi-map training experiment: train single GNN across all maps
- [ ] Transfer learning experiment: pre-train on small, zero-shot / fine-tune on large
- [ ] Measure generalization: agent trained on small maps → how well does it play large maps?
- [ ] Visual results: learning curves per map, cross-map evaluation heatmap

**Success criteria:**
- [ ] Single GNN agent achieves >80% win rate on all map sizes simultaneously
- [ ] Agent trained on small maps achieves >50% win rate on unseen large maps without fine-tuning
- [ ] Convergence speed improves with multi-map pre-training vs. training from scratch per map

---

## MCTS Baseline — COMPLETE ✓

**Decoupled UCT for simultaneous-move games:**
- `parallel_risk/agents/mcts_agent.py` (full implementation)
- Experiments: `experiments/validate_mcts.py`, `experiments/compare_mcts_gnn.py`
- Results: MCTS (budget=200) at 99.5% win rate vs masked-random
- Results: MCTS vs GNN comparison — see note below

**Note on MCTS vs. GNN comparison:** The available comparison (`experiments/mcts_vs_gnn_results_corrected/`) was run against GNN weights from an earlier training run. The best-known GNN checkpoint was trained afterward. A fair comparison between MCTS and the best GNN remains to be run as part of Phase 3 baseline establishment.

---

## Phase 3: AlphaZero-Style Training — PLANNED

### Motivation
MCTS provides strong performance via deep search but is expensive at inference (scales with budget). The GNN is fast but weaker than MCTS. AlphaZero-style training bridges this gap: use MCTS to generate high-quality supervision, train the GNN to internalize it, and use the improved GNN to guide future search.

For a **simultaneous-move** game, we use the **Decoupled UCT** framework already implemented. The AlphaZero loop adapts naturally:
1. MCTS (with GNN prior) plays self-play games
2. Root visit distributions become policy targets; outcome is the value target
3. GNN is trained to predict these targets
4. Updated GNN replaces prior for next iteration of MCTS

### Step 1: Fair MCTS vs. GNN Baseline
- Run MCTS (budget=200) against best GNN checkpoint on all maps
- Establish the performance gap that Phase 3 aims to close
- Visual: Elo comparison chart

### Step 2: MCTS Self-Play Data Generation
- Implement data generation pipeline: MCTS self-play → `(state, policy_distribution, outcome)` tuples
- Policy distribution = normalized visit counts at root
- For simultaneous moves: record both agents' policy distributions per turn
- Config: number of self-play games, MCTS budget per move, parallelism

### Step 3: GNN Supervised Training
- Train GNN to predict (policy distribution, value) from state
- Loss: `L = α * KL(mcts_policy, gnn_policy) + β * MSE(outcome, gnn_value)`
- Evaluate: does GNN policy start matching MCTS visit distributions?

### Step 4: PUCT-Guided MCTS
- Replace uniform prior in UCT with GNN policy network (PUCT formula)
- GNN value network replaces random rollouts for leaf evaluation
- With learned guidance, MCTS should achieve equivalent quality at lower budget

### Step 5: Iterative Self-Play Loop
- Pipeline: self-play → data collection → GNN update → better prior → repeat
- Track: GNN standalone win rate, MCTS(GNN) vs MCTS(uniform) at equal budget
- Termination: convergence in Elo, or fixed number of iterations

### Success Criteria
- [ ] GNN alone (no search) approaches MCTS(budget=50) performance
- [ ] MCTS(GNN, budget=50) beats MCTS(uniform, budget=200) — 4× compute efficiency
- [ ] Self-play loop shows monotonic improvement over iterations
- [ ] Generalizes: AlphaZero agent trained on small maps transfers to large maps

---

## Research Questions

1. Can a single GNN handle 6 to 50 territory maps without retraining? (Phase 2.4)
2. What is the sample efficiency gain from graph inductive bias?
3. Does AlphaZero-style training close the gap between GNN and MCTS? (Phase 3)
4. Do learned PUCT priors produce qualitatively different search trees from uniform UCT?
5. Does the AlphaZero GNN learn interpretable strategy concepts (attention on chokepoints)?

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

### Phase 2 ✓ (partial)
- [x] GNN agent trains and achieves 99.5% win rate vs random
- [x] Action masking implemented
- [ ] GNN agent trains on multiple map sizes simultaneously
- [ ] Positive transfer demonstrated: pre-training on small maps helps on large

### MCTS ✓
- [x] Decoupled UCT implemented for simultaneous-move games
- [x] 99.5% win rate vs masked-random at budget=200
- [ ] Fair comparison against best GNN checkpoint

### Phase 3 (Targets)
- [ ] Self-play data generation pipeline implemented
- [ ] GNN supervised on MCTS data shows improved Elo vs. PPO-trained GNN
- [ ] MCTS(GNN) achieves MCTS(uniform) quality at 4× lower budget
- [ ] Paper submission to RL conference (ICLR, NeurIPS)

---

## Revision History

- **2026-04-07:** Initial roadmap created — two-phase approach
- **2026-04-21:** Phase 1 complete, Phase 2 Steps 1-3 complete
- **2026-09-03:** Phase 2 complete, MCTS complete; updated to reflect Phase 3 AlphaZero goals and Phase 2.4 multi-map as immediate next step
