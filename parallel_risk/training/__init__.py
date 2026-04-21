"""
Training infrastructure for Parallel Risk.

Phase 1 (RLlib): parallel_risk.training.rllib
Phase 2 (TorchRL + GNN): parallel_risk.training.torchrl
"""

# Backward compatibility: expose RLlib wrapper at top level
from parallel_risk.training.rllib.wrapper import (
    RLlibParallelRiskEnv,
    make_rllib_env,
)

# Backward compatibility for old checkpoints
# Old checkpoints reference 'parallel_risk.training.rllib_wrapper'
import sys
sys.modules['parallel_risk.training.rllib_wrapper'] = sys.modules['parallel_risk.training.rllib.wrapper']

__all__ = [
    'RLlibParallelRiskEnv',
    'make_rllib_env',
]
