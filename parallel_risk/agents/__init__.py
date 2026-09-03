"""Agent implementations for Parallel Risk."""

from parallel_risk.agents.random_agent import RandomAgent
from parallel_risk.agents.masked_random_agent import MaskedRandomAgent, MaskedRandomAgentRLlib
from parallel_risk.agents.mcts_agent import MCTSAgent

try:
    from parallel_risk.agents.gnn_agent import GNNAgent
    __all__ = ["RandomAgent", "MaskedRandomAgent", "MaskedRandomAgentRLlib", "MCTSAgent", "GNNAgent"]
except ImportError:
    __all__ = ["RandomAgent", "MaskedRandomAgent", "MaskedRandomAgentRLlib", "MCTSAgent"]
