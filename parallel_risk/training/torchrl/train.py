"""
TorchRL training script for Parallel Risk with GNN policies.

This script handles:
- Loading GNN policy architecture
- Converting observations to graph format
- Running PPO training loop with TorchRL
- Multi-map training support
- Checkpointing and logging

TODO Phase 2.3: Implement TorchRL training loop

Usage (once implemented):
    python -m parallel_risk.training.torchrl.train --config configs/gnn_gcn.yaml
"""

import argparse
from pathlib import Path


def main():
    """Main training loop."""
    parser = argparse.ArgumentParser(description="Train Parallel Risk with TorchRL + GNN")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--num-iterations", type=int, default=1000, help="Training iterations")

    args = parser.parse_args()

    raise NotImplementedError(
        "Phase 2.3: TorchRL training not yet implemented.\n"
        "For Phase 1 training with RLlib, use:\n"
        "  python -m parallel_risk.training.rllib.train --config parallel_risk/training/rllib/configs/ppo_baseline.yaml"
    )


if __name__ == "__main__":
    main()
