"""Checkpoint-based agent for Parallel Risk evaluation."""

from pathlib import Path


class CheckpointAgent:
    """
    Agent that loads actions from a trained RLlib checkpoint.

    Interface compatible with RandomAgent for drop-in replacement in evaluation.
    Uses lazy loading to minimize memory usage - checkpoint is only loaded
    when the first action is requested.
    """

    def __init__(self, checkpoint_path: str, action_budget: int = 5):
        """
        Initialize checkpoint agent.

        Args:
            checkpoint_path: Path to RLlib checkpoint directory
            action_budget: Number of actions per turn (for interface compatibility)
        """
        # Ensure we have an absolute path
        self.checkpoint_path = str(Path(checkpoint_path).resolve())
        self.action_budget = action_budget
        self.algorithm = None  # Lazy load on first action
        self._loaded = False

    def _ensure_loaded(self):
        """Load the checkpoint if not already loaded."""
        if not self._loaded:
            from ray.rllib.algorithms.ppo import PPO
            self.algorithm = PPO.from_checkpoint(self.checkpoint_path)
            self._loaded = True

    def get_action(self, observation, agent_id=None):
        """
        Generate action from the trained policy.

        Args:
            observation: Environment observation (flattened for RLlib)
            agent_id: Agent ID (unused, for interface compatibility)

        Returns:
            Action in RLlib format (tuple of action tuples)
        """
        self._ensure_loaded()

        # Compute action from policy (deterministic for evaluation)
        action = self.algorithm.compute_single_action(
            observation,
            policy_id="main_policy",
            explore=False  # Deterministic evaluation
        )

        return action

    def unload(self):
        """
        Explicitly unload the algorithm to free memory.

        Call this between evaluations when using multiple checkpoint agents
        to prevent memory overflow.
        """
        if self.algorithm is not None:
            self.algorithm.stop()
            self.algorithm = None
            self._loaded = False

    def __del__(self):
        """Cleanup on deletion."""
        self.unload()
