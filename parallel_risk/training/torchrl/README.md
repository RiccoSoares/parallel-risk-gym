# Phase 2 (TorchRL + GNN) - Implementation Status

## Current Status

✅ **Phase 2.1: Graph Observation Wrapper** - COMPLETE
- `graph_wrapper.py`: Converts environment observations to PyTorch Geometric format
- Node features: troops, ownership, in-degree, region membership
- Edge index in COO format
- Global features: income, turn number, region control
- Test suite: `tests/test_graph_wrapper.py` (5/5 passing ✅)

✅ **Phase 2.2: GNN Architectures** - COMPLETE
- `models/gnn_gcn.py`: GCN policy with actor-critic heads
- `models/action_decoder.py`: Action decoder with log prob & entropy computation
- Handles variable-sized graphs in batches
- Test suite: `tests/test_gnn_policy.py` (4/4 passing ✅)

✅ **Phase 2.3: TorchRL Training** - COMPLETE
- `training/torchrl/train.py`: PPO training loop with self-play
- Rollout collection with proper episode boundary handling
- Generalized Advantage Estimation (GAE)
- TensorBoard logging and checkpointing
- Test suite: `tests/test_training.py` (5/5 passing ✅)
- **Bug fixed**: Variable batch sizes resolved ✅

📋 **Phase 2.4: Experiments** - TODO
- Validate learning (compare to random baseline)
- Compare GNN vs MLP (Phase 1) performance
- Multi-map training experiments
- Transfer learning experiments

## Installation

To use Phase 2 components, install TorchRL dependencies:

```bash
./install_training_deps.sh
# Choose option 2 (TorchRL + GNN)
```

Or manually:
```bash
pip install -r requirements/torchrl.txt
```

## Testing

All tests passing:

```bash
# Test graph wrapper
PYTHONPATH=. python tests/test_graph_wrapper.py

# Test GNN policy and action decoder
PYTHONPATH=. python tests/test_gnn_policy.py

# Test training loop
PYTHONPATH=. python tests/test_training.py
```

## Training

Train a GNN policy with PPO:

```bash
# Short test run (20 iterations, ~2 minutes)
python -m parallel_risk.training.torchrl.train \
    --config parallel_risk/training/torchrl/configs/gnn_gcn.yaml \
    --num-iterations 20

# Full training run (1000 iterations, ~1-2 hours)
python -m parallel_risk.training.torchrl.train \
    --config parallel_risk/training/torchrl/configs/gnn_gcn.yaml \
    --num-iterations 1000
```

**Expected output:**
```
Starting training for 20 iterations...
Device: cpu
Map: simple_6
Policy: GCN (128x3)

Iteration 1/20 | Reward: 0.000 | Length: 64.6 | Episodes: 7
Iteration 10/20 | Reward: 0.000 | Length: 82.9 | Episodes: 61
  💾 Saved checkpoint: checkpoints/gnn_training/checkpoint_000010.pt
Iteration 20/20 | Reward: 0.000 | Length: 68.9 | Episodes: 134
  💾 Saved checkpoint: checkpoints/gnn_training/checkpoint_000020.pt

✅ Training complete!
```

## Configuration

Edit `parallel_risk/training/torchrl/configs/gnn_gcn.yaml`:

```yaml
env:
  map_name: "simple_6"
  max_turns: 100
  action_budget: 5

model:
  hidden_dim: 128
  num_layers: 3
  dropout: 0.1

training:
  batch_size: 1024
  num_epochs: 10
  learning_rate: 3.0e-4
  gamma: 0.99
  clip_epsilon: 0.2
  entropy_coeff: 0.01
```

## Usage Example

```python
from parallel_risk import ParallelRiskEnv
from parallel_risk.training.torchrl.graph_wrapper import GraphObservationWrapper

# Create environment with graph wrapper
env = ParallelRiskEnv(map_name="simple_6")
wrapped_env = GraphObservationWrapper(env)

# Reset returns graph observations
graph_obs, infos = wrapped_env.reset()

# Each observation is a PyTorch Geometric Data object
agent_graph = graph_obs['agent_0']
print(f"Nodes: {agent_graph.num_nodes}")
print(f"Node features: {agent_graph.x.shape}")
print(f"Edges: {agent_graph.edge_index.shape}")
```

## Directory Structure

```
parallel_risk/training/torchrl/
├── __init__.py
├── graph_wrapper.py      ✅ COMPLETE
├── train.py              ✅ COMPLETE
├── configs/
│   └── gnn_gcn.yaml      ✅ COMPLETE
└── README.md

parallel_risk/models/
├── __init__.py
├── gnn_gcn.py           ✅ COMPLETE
├── gnn_gat.py           📋 TODO (optional)
└── action_decoder.py    ✅ COMPLETE

tests/
├── test_graph_wrapper.py  ✅ COMPLETE (5/5 passing)
├── test_gnn_policy.py     ✅ COMPLETE (4/4 passing)
└── test_training.py       ✅ COMPLETE (5/5 passing)
```

## Recent Changes

**2026-04-21: Bug Fix - Variable Batch Sizes**
- Fixed episode boundary handling in rollout collection
- Filter out `'__all__'` key from agent dictionaries
- Initialize GAE with proper batched tensors
- Training now runs end-to-end without errors ✅

**2026-04-21: Phase 2.3 Complete**
- Implemented PPO training loop
- GAE computation
- TensorBoard logging
- Checkpointing system

**2026-04-21: Phase 2.2 Complete**
- GCN policy architecture
- Action decoder with log probs & entropy

**2026-04-21: Phase 2.1 Complete**
- Graph observation wrapper
- PyTorch Geometric integration

## Next Steps

1. **Validate learning**: Run longer training and evaluate vs random baseline
2. **Compare to Phase 1**: GNN performance vs MLP baseline
3. **Multi-map training**: Train on multiple map sizes simultaneously
4. **Transfer learning**: Pre-train on small maps, fine-tune on large

## Documentation

- `docs/TORCHRL_GNN_GUIDE.md` - Complete Phase 2 guide
- `docs/RL_TRAINING_ROADMAP.md` - Overall training roadmap
- `CLAUDE.md` - Project overview and quick start
