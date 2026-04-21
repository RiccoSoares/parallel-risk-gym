"""
RLlib training infrastructure for Parallel Risk (Phase 1).

This module provides RLlib integration for training RL agents
on fixed-size maps using standard MLP architectures.

For Phase 2 (GNN with TorchRL), see parallel_risk.training.torchrl
"""

from parallel_risk.training.rllib.wrapper import RLlibParallelRiskEnv, make_rllib_env

__all__ = ['RLlibParallelRiskEnv', 'make_rllib_env']
