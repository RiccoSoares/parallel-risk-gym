# TorchRL + GNN Training Guide

**Status:** Phase 2 COMPLETE - Training pipeline fully functional  
**Last Updated:** 2026-04-21

## Overview

Phase 2 introduces Graph Neural Networks (GNNs) for training Parallel Risk agents. Unlike Phase 1's fixed-size MLP approach, GNNs can handle variable-sized maps and learn spatial strategies more efficiently.

## Implementation Status

✅ **Phase 2.1: Graph Observation Wrapper** - COMPLETE
- Converts environment observations to PyTorch Geometric format
- Handles variable-sized graphs
- Test suite passing (5/5)

✅ **Phase 2.2: GNN Policy Architectures** - COMPLETE
- GCN policy with actor-critic heads
- Action decoder with log probs & entropy
- Handles batched variable-sized graphs
- Test suite passing (4/4)

✅ **Phase 2.3: Training Loop** - COMPLETE
- PPO trainer with self-play
- GAE computation
- TensorBoard logging & checkpointing
- Bug-free rollout collection
- Test suite passing (5/5)
- **Full training runs successfully!**

📋 **Phase 2.4: Experiments** - TODO
- Validate learning against random baseline
- Compare GNN vs MLP performance
- Multi-map training
- Transfer learning

## Motivation

**Why GNNs?**
- Maps are naturally graphs (territories = nodes, adjacencies = edges)
- GNNs handle variable-sized inputs (6 territories or 20 territories, same model)
- Message passing captures territorial relationships more naturally
- Transfer learning: model trained on small maps can generalize to larger ones

**Comparison to Phase 1:**
| Aspect | Phase 1 (RLlib + MLP) | Phase 2 (TorchRL + GNN) |
|--------|----------------------|-------------------------|
| Architecture | Flat MLP | Graph Neural Network |
| Map size | Fixed (one map per model) | Variable (multi-map training) |
| Framework | RLlib | TorchRL + PyTorch Geometric |
| Transfer learning | No | Yes |

## Architecture

### Graph Representation

**Node features (per territory):**
- Troop count (normalized)
- Ownership (+1 self, -1 enemy, 0 neutral)
- Region ID (one-hot or embedding)
- In-degree (number of adjacent territories)

**Edge features:**
- Adjacency (binary)
- Optional: Strategic importance, distance

**Graph structure:**
```python
from torch_geometric.data import Data

graph = Data(
    x=node_features,      # [n_territories, feature_dim]
    edge_index=edge_list,  # [2, n_edges] (COO format)
    edge_attr=edge_features # [n_edges, edge_feature_dim]
)
```

### GNN Architectures (Planned)

**1. GCN (Graph Convolutional Network)**
- Simple message passing baseline
- Fast, easy to implement
- Good starting point

**2. GAT (Graph Attention Network)**
- Learns attention weights over neighbors
- Can identify strategic territories automatically
- Interpretable via attention visualization

**3. GraphSAGE**
- Sampling-based aggregation
- Scalable to very large graphs
- Good for future large-map extensions

### Action Decoder

Converting graph embeddings to actions is non-trivial:

**Current action format:** `[source_territory, dest_territory, num_troops]`

**Decoder approaches:**
1. **Node-level:** Select source node, then edge to neighbor
2. **Edge-level:** Directly select edges to act on
3. **Autoregressive:** Sample source, dest, troops sequentially

TODO: Experiment and decide on best approach

## Implementation Roadmap

### ✅ Phase 2.1: Graph Observation Wrapper (COMPLETE)
- ✅ Implement `env_to_graph()` in `training/torchrl/graph_wrapper.py`
- ✅ Handle batching of variable-sized graphs
- ✅ Test with different map sizes
- ✅ Unit tests for graph conversion

### ✅ Phase 2.2: GNN Policy Architectures (COMPLETE)
- ✅ Implement GCN policy in `models/gnn_gcn.py`
- ✅ Implement action decoder in `models/action_decoder.py`
- ✅ Unit tests for forward pass
- ✅ Test on sample graphs
- ⬜ Implement GAT policy in `models/gnn_gat.py` (optional)

### ✅ Phase 2.3: TorchRL Training Loop (COMPLETE)
- ✅ Implement PPO with TorchRL in `training/torchrl/train.py`
- ✅ Self-play policy system
- ✅ Data collection with episode boundary handling
- ✅ Checkpointing and logging
- ✅ TensorBoard integration
- ✅ Fix variable batch size bug

### 📋 Phase 2.4: Experiments (TODO)
- ⬜ Validate learning (train and evaluate vs random)
- ⬜ Single-map training (vs. Phase 1 baseline)
- ⬜ Multi-map training (6 + 10 territory maps)
- ⬜ Transfer learning experiments
- ⬜ Architecture comparison (GCN vs GAT if implemented)
- ⬜ Attention visualization (for GAT)

## Directory Structure

```
parallel_risk/
├── training/
│   ├── rllib/              # Phase 1 (existing)
│   │   ├── wrapper.py
│   │   ├── train.py
│   │   └── configs/
│   └── torchrl/            # Phase 2 (new)
│       ├── graph_wrapper.py
│       ├── train.py
│       └── configs/
│           └── gnn_gcn.yaml
├── models/                 # Phase 2 (new)
│   ├── gnn_gcn.py
│   ├── gnn_gat.py
│   └── action_decoder.py
└── env/                    # Shared between phases

experiments/
├── validate_learning.py    # Phase 1
├── self_play_league.py     # Phase 1
└── gnn_experiments/        # Phase 2 (TODO)
    ├── single_map_training.py
    ├── multi_map_training.py
    └── transfer_learning.py
```

## Installation

```bash
# Install Phase 2 dependencies
pip install -r requirements/torchrl.txt

# Or use installer script
./install_training_deps.sh
# Choose option 2 (TorchRL + GNN)
```

**Dependencies:**
- PyTorch >= 2.0.0
- TorchRL >= 0.3.0
- PyTorch Geometric >= 2.4.0
- TensorBoard >= 2.14.0

## Usage (Fully Implemented)

```bash
# Single-map training with GCN
python -m parallel_risk.training.torchrl.train \
    --config parallel_risk/training/torchrl/configs/gnn_gcn.yaml \
    --num-iterations 1000

# Resume from checkpoint
python -m parallel_risk.training.torchrl.train \
    --config parallel_risk/training/torchrl/configs/gnn_gcn.yaml \
    --num-iterations 1000 \
    --checkpoint checkpoints/gnn_training/checkpoint_000500.pt
```

Expected output:
```
Starting training for 20 iterations...
Device: cpu
Map: simple_6
Policy: GCN (128x3)

Iteration 1/20 | Reward: 0.000 | Length: 64.6 | Episodes: 7
Iteration 10/20 | Reward: 0.000 | Length: 82.9 | Episodes: 61
  💾 Saved checkpoint: checkpoints/gnn_training/checkpoint_000010.pt
Iteration 20/20 | Reward: 0.000 | Length: 68.9 | Episodes: 134

✅ Training complete!
```

## Current State

**✅ Completed:**
- Directory structure created
- Graph wrapper implemented and tested
- GCN policy implemented and tested
- Action decoder implemented and tested
- PPO training loop implemented and tested
- Bug fixes for episode boundaries
- Full end-to-end training working
- Documentation complete

**📋 TODO:**
- Phase 2.4: Validation and experiments
- GAT architecture (optional enhancement)
- Multi-map training support
- Transfer learning experiments

## Comparison to RLlib (Phase 1)

**When to use RLlib (Phase 1):**
- Single fixed-size map
- Standard MLP is sufficient
- Quick baseline experiments
- Leveraging RLlib's built-in features

**When to use TorchRL (Phase 2):**
- Multiple map sizes simultaneously
- Transfer learning between maps
- GNN architecture research
- Need custom training loop control
- Publication-quality experiments

Both frameworks will coexist in the repo and can be used for comparisons.

## Resources

**TorchRL:**
- Documentation: https://pytorch.org/rl/
- Examples: https://github.com/pytorch/rl/tree/main/examples

**PyTorch Geometric:**
- Documentation: https://pytorch-geometric.readthedocs.io/
- Tutorials: https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html

**Papers:**
- GCN: Kipf & Welling (2017) - Semi-Supervised Classification with GCNs
- GAT: Veličković et al. (2018) - Graph Attention Networks
- GraphSAGE: Hamilton et al. (2017) - Inductive Representation Learning on Large Graphs

## Next Steps

1. Start with Phase 2.1: Implement graph wrapper
2. Create unit tests for graph conversion
3. Test with different map sizes
4. Move to Phase 2.2: Implement GCN policy

See `docs/RL_TRAINING_ROADMAP.md` for full context.
