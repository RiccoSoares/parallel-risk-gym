#!/bin/bash
# Install training dependencies for Parallel Risk

set -e

echo "Parallel Risk - Training Dependencies Installer"
echo "================================================"
echo ""
echo "Choose which training framework to install:"
echo "  1) RLlib (Phase 1) - Standard RL with MLPs"
echo "  2) TorchRL + GNN (Phase 2) - Graph Neural Networks"
echo "  3) Both (for development/comparison)"
echo ""

read -p "Enter choice [1-3]: " choice

case $choice in
    1)
        echo ""
        echo "Installing RLlib dependencies (Phase 1)..."
        echo "This may take several minutes as Ray/RLlib are large packages."
        pip install -r requirements/rllib.txt
        echo ""
        echo "✅ RLlib installation complete!"
        echo ""
        echo "To verify:"
        echo "  python -c 'import ray; import ray.rllib; print(\"RLlib ready!\")'"
        echo ""
        echo "To start training:"
        echo "  python -m parallel_risk.training.rllib.train --config parallel_risk/training/rllib/configs/ppo_baseline.yaml"
        ;;
    2)
        echo ""
        echo "Installing TorchRL + PyTorch Geometric dependencies (Phase 2)..."
        pip install -r requirements/torchrl.txt
        echo ""
        echo "✅ TorchRL installation complete!"
        echo ""
        echo "To verify:"
        echo "  python -c 'import torch; import torchrl; import torch_geometric; print(\"TorchRL + PyG ready!\")'"
        echo ""
        echo "Phase 2 training scripts coming soon!"
        ;;
    3)
        echo ""
        echo "Installing both RLlib and TorchRL dependencies..."
        pip install -r requirements/rllib.txt
        pip install -r requirements/torchrl.txt
        echo ""
        echo "✅ All training dependencies installed!"
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "To run tests:"
echo "  python run_tests.py"
