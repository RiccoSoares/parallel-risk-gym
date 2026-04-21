# Phase 2 (TorchRL + GNN) - Implementation Status

## Current Status

✅ **Phase 2.1: Graph Observation Wrapper** - IMPLEMENTED
- `graph_wrapper.py`: Converts environment observations to PyTorch Geometric format
- Node features: troops, ownership, in-degree, region membership
- Edge index in COO format
- Global features: income, turn number, region control
- Test suite: `tests/test_graph_wrapper.py`

🚧 **Phase 2.2: GNN Architectures** - TODO
- `models/gnn_gcn.py`: GCN policy (stub)
- `models/gnn_gat.py`: GAT policy (stub)
- `models/action_decoder.py`: Action decoder (stub)

🚧 **Phase 2.3: TorchRL Training** - TODO
- `training/torchrl/train.py`: Training loop (stub)

🚧 **Phase 2.4: Experiments** - TODO
- Single-map training
- Multi-map training
- Transfer learning

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

## Testing Graph Wrapper

Once PyTorch Geometric is installed:

```bash
PYTHONPATH=. python tests/test_graph_wrapper.py
```

Expected output:
- Graph creation with correct node/edge structure
- Wrapper works with environment
- Different map sizes supported
- Bidirectional edges verified

## Next Steps

1. Install Phase 2 dependencies
2. Implement GCN policy (`models/gnn_gcn.py`)
3. Implement action decoder (`models/action_decoder.py`)
4. Implement TorchRL training loop (`training/torchrl/train.py`)
5. Run single-map training experiment
6. Compare GNN vs MLP performance

## Usage Example (Once Implemented)

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
├── graph_wrapper.py      ✅ IMPLEMENTED
├── train.py              🚧 TODO
└── configs/
    └── gnn_gcn.yaml      🚧 TODO

parallel_risk/models/
├── __init__.py
├── gnn_gcn.py           🚧 TODO
├── gnn_gat.py           🚧 TODO
└── action_decoder.py    🚧 TODO

tests/
└── test_graph_wrapper.py  ✅ IMPLEMENTED
```

## Documentation

- `docs/TORCHRL_GNN_GUIDE.md` - Complete Phase 2 guide
- `docs/RL_TRAINING_ROADMAP.md` - Overall training roadmap
