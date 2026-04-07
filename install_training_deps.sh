#!/bin/bash
# Install RLlib training dependencies

echo "Installing RLlib and training dependencies..."
echo "This may take several minutes as Ray/RLlib are large packages."
echo ""

pip install -r requirements.txt

echo ""
echo "✅ Installation complete!"
echo ""
echo "To verify installation:"
echo "  python -c 'import ray; import ray.rllib; print(\"RLlib ready!\")'"
echo ""
echo "To run tests:"
echo "  PYTHONPATH=. python tests/test_rllib_wrapper.py"
echo ""
echo "To start training:"
echo "  python -m parallel_risk.training.train_rllib --config parallel_risk/training/configs/ppo_baseline.yaml --num-iterations 10"
