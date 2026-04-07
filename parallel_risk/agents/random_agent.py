"""Random agent baseline for Parallel Risk."""

import numpy as np


class RandomAgent:
    """
    Random baseline agent that generates random valid actions.

    Compatible with both raw ParallelRiskEnv and RLlibParallelRiskEnv.
    """

    def __init__(self, n_territories, action_budget=5, mode="rllib"):
        """
        Initialize random agent.

        Args:
            n_territories: Number of territories in the map
            action_budget: Number of actions per turn (for RLlib mode)
            mode: "rllib" or "raw" - determines action format
        """
        self.n_territories = n_territories
        self.action_budget = action_budget
        self.mode = mode

    def get_action(self, observation=None, agent_id=None):
        """
        Generate random action for the agent.

        Args:
            observation: Environment observation (unused, for interface compatibility)
            agent_id: Agent ID (unused, for interface compatibility)

        Returns:
            Action in appropriate format based on mode:
            - "rllib": tuple of action_budget MultiDiscrete actions
            - "raw": dict with 'num_actions' and 'actions' array
        """
        if self.mode == "rllib":
            return self._get_rllib_action()
        else:
            return self._get_raw_action()

    def _get_rllib_action(self):
        """Generate action for RLlib environment (fixed budget)."""
        actions = []
        for _ in range(self.action_budget):
            source = np.random.randint(0, self.n_territories)
            dest = np.random.randint(0, self.n_territories)
            troops = np.random.randint(1, 21)  # 1-20 troops
            actions.append([source, dest, troops])
        return tuple(map(tuple, actions))

    def _get_raw_action(self):
        """Generate action for raw ParallelRiskEnv (variable length)."""
        # Random number of actions (0 to action_budget)
        num_actions = np.random.randint(0, self.action_budget + 1)

        # Generate random action triples
        actions = np.zeros((self.action_budget, 3), dtype=np.int32)
        for i in range(num_actions):
            source = np.random.randint(0, self.n_territories)
            dest = np.random.randint(0, self.n_territories)
            troops = np.random.randint(1, 6)  # 1-5 troops per action
            actions[i] = [source, dest, troops]

        return {
            'num_actions': num_actions,
            'actions': actions
        }
