# RL Training Roadmap

**Last Updated:** 2026-04-07  
**Status:** Phase 1 Complete (Steps 1-2), Phase 2 Planned

## Overview

This document outlines the plan to enable reinforcement learning training on the Parallel Risk environment. The approach is designed to:
1. Establish baseline training capability with standard architectures
2. Transition to graph-based observations for multi-map flexibility
3. Support research-grade experimentation and publication

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

## Phase 1: Baseline Training (4-6 weeks)

### Objectives
- Prove the environment is learnable by RL agents
- Establish baseline performance metrics
- Validate reward shaping effectiveness
- Get self-play infrastructure working

### Step 1: Reward Shaping (Week 1-2)

**Current State:** Sparse rewards (+1 win, -1 loss, 0 otherwise)
- Too delayed for effective learning
- No intermediate signal for progress
- Hard to learn from rare victories

**Proposed Shaped Rewards:**

1. **Territory Control Reward**
   ```python
   territory_percentage = owned_territories / total_territories
   reward += alpha * territory_percentage
   ```
   - Dense signal every step
   - Correlates with winning
   - Risk: might incentivize passive territorial holding

2. **Region Completion Bonus**
   ```python
   for region in completed_regions:
       reward += beta * region_bonus_value
   ```
   - Encourages strategic play (capturing full regions)
   - Aligned with game mechanics (region bonuses = more troops)
   - One-time bonus when region is completed

3. **Troop Advantage Reward**
   ```python
   troop_ratio = my_total_troops / (enemy_total_troops + 1)
   reward += gamma * troop_ratio
   ```
   - Rewards efficient combat and troop management
   - Risk: might incentivize avoiding combat to preserve troops

4. **Strategic Territory Bonus**
   ```python
   for territory in owned_territories:
       reward += delta * territory.connectivity_score
   ```
   - Rewards controlling well-connected territories
   - Encourages map control over isolated territories

**Design Principles:**
- All shaped rewards must correlate with winning
- Coefficients (alpha, beta, gamma, delta) should be tunable
- Terminal win/loss reward (+1/-1) remains dominant
- Test with ablation studies: which combinations work best?

**Validation:**
- Manual play testing: do shaped rewards align with good strategy?
- Correlation analysis: do high shaped rewards predict wins?
- Perverse incentive check: can agent exploit shaped rewards without winning?

**Implementation Location:**
- Add to `parallel_risk/env/parallel_risk_env.py`
- Create `RewardShaper` class for modularity
- Config-driven: enable/disable individual components
- Separate file: `parallel_risk/env/reward_shaping.py`

### Step 2: RLlib Integration (Week 2-3)

**Components:**

1. **Environment Wrapper** (`parallel_risk/training/rllib_wrapper.py`)
   ```python
   from ray.rllib.env import PettingZooEnv
   from parallel_risk import ParallelRiskEnv
   
   def env_creator(config):
       return ParallelRiskEnv(
           map_name=config.get("map_name", "basic_6"),
           reward_shaping_config=config.get("reward_shaping", {})
       )
   ```

2. **Action Space Handling**
   - Current: Dict with `num_actions` + padded array
   - Options for RLlib:
     - **Fixed budget**: Require exactly N actions per turn (simplest)
     - **Autoregressive**: Sample num_actions, then sample each action (complex)
     - **Action masking**: Generate all valid actions, mask invalid (large space)
   
   **Recommendation:** Start with fixed budget (e.g., 5 actions/turn)
   - Simplifies learning problem
   - Still strategically rich
   - Can relax later if needed

3. **Self-Play Configuration**
   ```python
   config = {
       "multiagent": {
           "policies": {"main_policy"},
           "policy_mapping_fn": lambda agent_id: "main_policy",
           "policies_to_train": ["main_policy"],
       },
       "self_play": {
           "policy_pool_size": 10,  # Keep last N policy versions
           "win_rate_threshold": 0.7,  # Add to pool if >70% win rate
       }
   }
   ```

4. **Training Script** (`parallel_risk/training/train.py`)
   - Hyperparameter config (learning rate, batch size, etc.)
   - Checkpointing and logging
   - Weights & Biases integration for experiment tracking

### Step 3: Evaluation Harness (Week 3-4)

**Standardized Metrics:**
- Win rate over training (self-play and vs. random/heuristic baselines)
- Elo rating evolution
- Average episode length (shorter = more decisive play)
- Territory control over time
- Action distribution analysis (deploy vs. transfer vs. attack ratios)

**Tournament System:**
- Round-robin between checkpoints
- Head-to-head comparisons
- Statistical significance testing

**Implementation:** `parallel_risk/evaluation/`
- `tournament.py` - Run matches between policies
- `metrics.py` - Calculate and log performance metrics
- `visualize.py` - Plot learning curves and comparisons

### Step 4: Baseline Experimentation (Week 4-6)

**Experiments to Run:**
1. **Reward shaping ablation:** Which shaped rewards help most?
2. **Architecture search:** MLP depth/width, network capacity
3. **Hyperparameter tuning:** Learning rate, entropy coefficient, GAE lambda
4. **Self-play dynamics:** Policy pool size, opponent sampling strategy

**Success Criteria:**
- Agent consistently beats random baseline (>80% win rate)
- Agent shows strategic behavior (region completion, efficient combat)
- Training converges within reasonable time (<24 hours on single GPU)
- Reproducible results with documented hyperparameters

**Deliverables:**
- Trained baseline agent checkpoint
- Performance benchmarks documented
- Best reward shaping configuration identified
- Training recipe (config + hyperparameters)

## Phase 2: Graph Neural Networks (6-8 weeks)

### Objectives
- Support multi-map training (6, 10, 20 territory maps simultaneously)
- Implement and compare GNN architectures
- Enable transfer learning across map sizes
- Research publication: "GNNs for Territorial Strategy Games"

### Step 1: Graph Observation Wrapper (Week 1-2)

**Convert Environment Observations to PyTorch Geometric Format:**

```python
from torch_geometric.data import Data

def env_to_graph(obs, map_config):
    # Node features: [troops, ownership, region_id, in_degree]
    node_features = torch.tensor([
        [obs['troops'][i], 
         obs['ownership'][i],
         territory_to_region[i],
         map_config.adjacency_matrix[i].sum()]
        for i in range(map_config.n_territories)
    ], dtype=torch.float)
    
    # Edge index: adjacency list as COO format
    edge_index = torch.tensor(
        map_config.adjacency_list_to_coo(), 
        dtype=torch.long
    )
    
    # Optional edge features (e.g., strategic value of connection)
    edge_attr = compute_edge_features(map_config)
    
    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
```

**Batching Strategy:**
- PyTorch Geometric handles variable-sized graphs in batches
- Create DataLoader that samples from multiple map sizes
- Ensure balanced sampling (don't overtrain on one map size)

**Implementation:** `parallel_risk/training/graph_wrapper.py`

### Step 2: GNN Policy Architectures (Week 2-4)

**Baseline Architectures to Compare:**

1. **Graph Convolutional Network (GCN)**
   - Simple message passing
   - Good baseline, easy to implement
   ```python
   from torch_geometric.nn import GCNConv
   
   class GCNPolicy(nn.Module):
       def __init__(self, node_features, hidden_dim, num_layers):
           self.convs = [GCNConv(node_features, hidden_dim)]
           for _ in range(num_layers - 1):
               self.convs.append(GCNConv(hidden_dim, hidden_dim))
           self.action_head = ...
           self.value_head = ...
   ```

2. **Graph Attention Network (GAT)**
   - Learns attention weights for neighbors
   - Can identify strategic territories automatically
   ```python
   from torch_geometric.nn import GATConv
   
   class GATPolicy(nn.Module):
       def __init__(self, node_features, hidden_dim, heads=4):
           self.convs = [GATConv(node_features, hidden_dim, heads=heads)]
           # Multi-head attention for neighbor importance
   ```

3. **GraphSAGE**
   - Sampling-based aggregation
   - Scalable to very large graphs
   - Good for future large-map extensions

**Action Space for GNNs:**

Current action format `[source, dest, troops]` works, but consider:
- **Node-level actions:** Select source node, then edge (neighbor) for dest
- **Edge-level actions:** Directly select edges to act on
- Graph pooling to get graph-level embedding for value function

**Implementation:** `parallel_risk/models/`
- `gnn_policy.py` - Policy network architectures
- `action_decoder.py` - Convert graph embeddings to actions

### Step 3: TorchRL Training Loop (Week 4-6)

**Why TorchRL over RLlib:**
- Better flexibility for custom architectures
- Native PyTorch integration (works well with PyG)
- More control over training loop for research

**Components:**

1. **Data Collection:**
   ```python
   from torchrl.collectors import MultiaSyncDataCollector
   from torchrl.envs import ParallelEnv
   
   # Collect experience from multiple envs with different maps
   collector = MultiAsyncDataCollector(
       create_env_fn=[
           lambda: GraphParallelRiskEnv(map_name="basic_6"),
           lambda: GraphParallelRiskEnv(map_name="large_10"),
       ],
       policy=gnn_policy,
       frames_per_batch=2048,
   )
   ```

2. **Self-Play System:**
   - Port logic from Phase 1 RLlib implementation
   - Policy pool with Elo ratings
   - Opponent sampling strategy (recent + best + random)
   - League-based training (inspired by AlphaStar)

3. **PPO Implementation:**
   ```python
   from torchrl.objectives import ClipPPOLoss
   
   loss_module = ClipPPOLoss(
       actor_network=gnn_policy,
       critic_network=gnn_value,
       clip_epsilon=0.2,
   )
   ```

**Implementation:** `parallel_risk/training/torchrl_trainer.py`

### Step 4: Multi-Map Training & Experiments (Week 6-8)

**Experiments:**

1. **Single-Map Training (Baseline)**
   - Train GCN, GAT, GraphSAGE on basic_6 map only
   - Compare to Phase 1 MLP baseline
   - Expected: Similar or slightly better performance

2. **Multi-Map Training**
   - Train on basic_6 + large_10 simultaneously
   - Test generalization to unseen map sizes
   - Expected: Some performance on new maps without retraining

3. **Transfer Learning**
   - Pre-train on small maps, fine-tune on large
   - Compare to training large map from scratch
   - Expected: Faster convergence, better sample efficiency

4. **Architecture Comparison**
   - GCN vs. GAT vs. GraphSAGE
   - Ablation: number of layers, hidden dimensions, attention heads
   - Identify best architecture for this domain

5. **Attention Visualization**
   - For GAT: visualize learned attention patterns
   - Do agents learn to focus on strategic territories?
   - Qualitative analysis for research publication

**Success Criteria:**
- GNN agent beats MLP baseline on multiple map sizes
- Transfer learning shows positive transfer (>0% improvement)
- Agent trained on small maps achieves >50% win rate on unseen large maps
- Identified best GNN architecture with reproducible results

**Deliverables:**
- Trained GNN agent checkpoints for all architectures
- Comparative analysis (tables, plots)
- Attention visualization and strategic analysis
- Research paper draft sections (methods, results)

## Infrastructure & Tooling

### Experiment Tracking
**Weights & Biases Integration:**
- Log all hyperparameters and config
- Real-time training curves (win rate, loss, rewards)
- Model checkpoints with version control
- Comparison dashboards across experiments

### Reproducibility
- Fixed random seeds for all experiments
- Version pinning: Python, PyTorch, RLlib/TorchRL, PyG
- Docker container for consistent environment
- All configs checked into git

### Compute Requirements
**Phase 1 (RLlib):**
- Single GPU sufficient (e.g., RTX 3090, A100)
- ~24 hours training time per experiment
- CPU-only possible but slower

**Phase 2 (GNN):**
- Single GPU for small experiments
- Multi-GPU beneficial for large-scale multi-map training
- TPUs viable for PyTorch Geometric workloads

## Project Structure (Proposed)

```
parallel_risk/
├── env/                          # Existing environment code
│   ├── parallel_risk_env.py
│   ├── map_config.py
│   ├── combat.py
│   ├── validators.py
│   └── reward_shaping.py         # NEW: Shaped reward implementations
├── training/                     # NEW: Training infrastructure
│   ├── rllib_wrapper.py          # Phase 1: RLlib integration
│   ├── graph_wrapper.py          # Phase 2: PyG graph conversion
│   ├── train_rllib.py            # Phase 1: Training script
│   ├── train_torchrl.py          # Phase 2: Training script
│   └── configs/                  # Hyperparameter configs
│       ├── baseline_ppo.yaml
│       ├── gnn_gcn.yaml
│       └── gnn_gat.yaml
├── models/                       # NEW: Neural network architectures
│   ├── mlp_policy.py             # Phase 1: Standard MLP
│   ├── gnn_policy.py             # Phase 2: GNN policies
│   └── action_decoder.py         # Action space handling
├── evaluation/                   # NEW: Evaluation tools
│   ├── tournament.py             # Head-to-head matches
│   ├── metrics.py                # Performance metrics
│   └── visualize.py              # Plotting and analysis
└── selfplay/                     # NEW: Self-play infrastructure
    ├── policy_pool.py            # Manage past policy versions
    ├── matchmaking.py            # Opponent selection
    └── elo_rating.py             # Elo rating system

docs/
├── RL_TRAINING_ROADMAP.md        # This document
├── REWARD_SHAPING.md             # NEW: Reward design details
└── GNN_ARCHITECTURE.md           # NEW: Phase 2 architecture decisions

experiments/                      # NEW: Experiment logs and results
├── phase1_baseline/
│   ├── reward_ablation/
│   ├── hyperparam_search/
│   └── best_model/
└── phase2_gnn/
    ├── architecture_comparison/
    ├── multi_map_training/
    └── transfer_learning/
```

## Timeline Summary

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| **Phase 1.1:** Reward Shaping | 1-2 weeks | Shaped reward implementation, validation |
| **Phase 1.2:** RLlib Integration | 1-2 weeks | Training pipeline, self-play working |
| **Phase 1.3:** Evaluation | 1 week | Metrics, tournament system |
| **Phase 1.4:** Baseline Experiments | 2-3 weeks | Trained agent, benchmarks, best config |
| **Phase 2.1:** Graph Wrapper | 1-2 weeks | PyG conversion, batching |
| **Phase 2.2:** GNN Architectures | 2 weeks | GCN, GAT, GraphSAGE implementations |
| **Phase 2.3:** TorchRL Training | 2 weeks | Custom training loop, self-play port |
| **Phase 2.4:** Multi-Map Experiments | 2-3 weeks | All experiments, paper results |
| **Total:** | **12-15 weeks** | Full RL training capability + research results |

## Open Questions & Future Decisions

### Phase 1 Decisions Needed:
1. **Action space simplification:** Fixed budget (how many actions?) vs. autoregressive?
2. **Reward shaping coefficients:** How to balance alpha, beta, gamma, delta?
3. **Self-play policy pool size:** 5? 10? 20? Trade-off: diversity vs. compute
4. **Baseline opponents:** Heuristic agent design for comparisons?

### Phase 2 Decisions Needed:
1. **Map size curriculum:** Train on small→large progressively or all simultaneously?
2. **GNN depth:** How many message passing layers? Risk of over-smoothing
3. **Action decoder design:** Node-level, edge-level, or keep current format?
4. **Transfer learning protocol:** What's the best pre-train/fine-tune split?

### Research Questions:
1. Do GNNs learn interpretable strategic concepts (e.g., attention on chokepoints)?
2. Can a single GNN agent handle 6 to 50 territory maps without retraining?
3. What's the sample efficiency gain from graph inductive bias?
4. How does self-play dynamics differ between MLP and GNN agents?

## Success Metrics

### Phase 1 Success:
- [ ] Agent beats random baseline >90% win rate
- [ ] Agent shows strategic behavior (captures regions, efficient combat)
- [ ] Reproducible training in <24 hours
- [ ] Documented best practices for reward shaping

### Phase 2 Success:
- [ ] GNN agent trains on multiple map sizes
- [ ] Positive transfer: pre-training helps on new maps
- [ ] Published-quality results and analysis
- [ ] Open-sourced model checkpoints and code

### Research Impact:
- [ ] Paper submission to RL conference (e.g., ICLR, NeurIPS)
- [ ] Benchmark environment for multi-agent GNN research
- [ ] Reproducible codebase for community use

## Phase 1 Progress

### Completed:
- [x] **Phase 1.1: Reward Shaping** - Fully implemented with 4 configurable components
  - Created `parallel_risk/env/reward_shaping.py` with RewardShaper class
  - Territory control, region completion, troop advantage, strategic position rewards
  - Preset configurations (dense, sparse, territorial, aggressive)
  - Unit tests in `tests/test_reward_shaping.py`
  - Documentation in `docs/REWARD_SHAPING.md`

- [x] **Phase 1.2: RLlib Integration** - Training pipeline ready
  - Created `parallel_risk/training/rllib_wrapper.py` with fixed-budget action space
  - Training script `parallel_risk/training/train_rllib.py` with CLI
  - Configuration system via YAML (`configs/ppo_baseline.yaml`)
  - Unit tests in `tests/test_rllib_wrapper.py`
  - Documentation in `docs/RLLIB_INTEGRATION.md`

### In Progress:
- **Phase 1.3-1.4: Evaluation & Experiments** - Ready to begin full training runs
  - Infrastructure complete, can start baseline training
  - Need to implement tournament system for agent evaluation
  - Need to run ablation studies on reward shaping

## Next Steps

**Immediate:** Phase 1.3 - Evaluation Harness
1. Create `parallel_risk/evaluation/` directory
2. Implement tournament system for head-to-head matches
3. Add Elo rating tracking
4. Create visualization tools for training curves

**Short-term:** Phase 1.4 - Baseline Experiments
1. Run full training with different reward shaping configs
2. Compare sparse vs. dense rewards
3. Measure win rates against random/heuristic baselines
4. Document best configuration and hyperparameters

**Long-term:** Phase 2 - Graph Neural Networks
1. Start with graph observation wrapper (PyG format)
2. Implement GNN policy architectures (GCN, GAT, GraphSAGE)
3. Multi-map training experiments

**References for Implementation:**
- OpenAI Five reward shaping: https://openai.com/research/openai-five
- AlphaStar self-play: https://deepmind.com/blog/article/alphastar-mastering-real-time-strategy-game-starcraft-ii
- GNN for games: "Graph Neural Networks for Multi-Agent Systems" (various papers)
- PyTorch Geometric tutorials: https://pytorch-geometric.readthedocs.io/

## Revision History

- **2026-04-07:** Initial roadmap created
  - Two-phase approach: flat→graph observations
  - Reward shaping as starting point
  - Timeline: 12-15 weeks total
