"""
RLlib wrapper for Parallel Risk environment.

This module provides wrappers to make ParallelRiskEnv compatible with RLlib.
Key considerations:
1. Action space simplification (variable-length -> fixed)
2. PettingZoo parallel -> RLlib MultiAgent compatibility
3. Self-play support
"""

from typing import Dict, Any, Optional
import numpy as np
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from parallel_risk.env.parallel_risk_env import ParallelRiskEnv
from parallel_risk.env.reward_shaping import RewardShapingConfig


class RLlibParallelRiskEnv(MultiAgentEnv):
    """RLlib-compatible wrapper for ParallelRiskEnv.

    Key features:
    - Converts variable-length action space to fixed budget
    - Compatible with RLlib's MultiAgentEnv interface
    - Supports reward shaping configuration
    - Handles PettingZoo parallel API -> RLlib conversion

    Action space simplification strategies:
    1. FIXED_BUDGET: Agent must submit exactly N actions per turn (default)
    2. AUTOREGRESSIVE: Sample number of actions, then sample each (future work)
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize RLlib wrapper.

        Args:
            config: Configuration dict with keys:
                - map_name: Map to use (default: "simple_6")
                - max_turns: Maximum turns per episode (default: 100)
                - reward_shaping_config: RewardShapingConfig or None
                - action_budget: Fixed number of actions per turn (default: 5)
                - seed: Random seed
        """
        super().__init__()

        config = config or {}
        self.map_name = config.get("map_name", "simple_6")
        self.max_turns = config.get("max_turns", 100)
        self.action_budget = config.get("action_budget", 5)
        self.seed_value = config.get("seed", None)

        # Extract reward shaping config if provided
        reward_config = config.get("reward_shaping_config", None)

        # Create base environment
        self.env = ParallelRiskEnv(
            map_name=self.map_name,
            max_turns=self.max_turns,
            seed=self.seed_value,
            reward_shaping_config=reward_config,
        )

        # Agent IDs
        self._agent_ids = set(self.env.possible_agents)

        # Define simplified action space (fixed budget)
        # Each action is [source, dest, troops]
        # source/dest: territory index [0, n_territories)
        # troops: number of troops [0, max_reasonable_troops]
        n_territories = self.env.map_config.n_territories

        # Action space: MultiDiscrete for each action component
        # This is easier for RL algorithms than Box with continuous values
        single_action_space = spaces.MultiDiscrete([
            n_territories,  # source
            n_territories,  # dest
            20,  # troops (0-19, will be clamped to available)
        ])

        # Fixed budget: agent submits exactly action_budget actions
        self._action_space = spaces.Tuple([single_action_space] * self.action_budget)

        # Observation space: flatten the Dict observation for easier processing
        # Original obs has: territory_ownership, territory_troops, adjacency_matrix, etc.
        self._observation_space = self._create_observation_space()

    def _create_observation_space(self) -> spaces.Space:
        """Create flattened observation space for RLlib.

        Returns a Box space with all observation components flattened.
        """
        n_territories = self.env.map_config.n_territories
        n_regions = len(self.env.map_config.regions)

        # Calculate total observation size:
        # - territory_ownership: n_territories (int8, but we'll use float for NN)
        # - territory_troops: n_territories
        # - adjacency_matrix: n_territories * n_territories
        # - available_income: 1
        # - turn_number: 1
        # - region_control: n_regions

        obs_size = (
            n_territories +  # ownership
            n_territories +  # troops
            n_territories * n_territories +  # adjacency
            1 +  # income
            1 +  # turn
            n_regions  # regions
        )

        # Use Box with reasonable bounds
        # Most values are counts or binary flags
        return spaces.Box(
            low=-1.0,
            high=100.0,
            shape=(obs_size,),
            dtype=np.float32
        )

    def _flatten_observation(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Flatten PettingZoo Dict observation to vector for neural network.

        Args:
            obs_dict: Original observation dict from ParallelRiskEnv

        Returns:
            Flattened observation vector
        """
        return np.concatenate([
            obs_dict['territory_ownership'].astype(np.float32),
            obs_dict['territory_troops'].astype(np.float32),
            obs_dict['adjacency_matrix'].flatten().astype(np.float32),
            obs_dict['available_income'].astype(np.float32),
            obs_dict['turn_number'].astype(np.float32),
            obs_dict['region_control'].astype(np.float32),
        ])

    def _unflatten_action(self, action_tuple) -> Dict[str, np.ndarray]:
        """Convert RLlib action format to ParallelRiskEnv format.

        Args:
            action_tuple: Tuple of MultiDiscrete actions from policy

        Returns:
            Action dict compatible with ParallelRiskEnv
        """
        # Convert tuple of actions to array
        actions_array = np.array([list(a) for a in action_tuple], dtype=np.int32)

        # Create action dict with fixed budget
        return {
            'num_actions': self.action_budget,
            'actions': actions_array,
        }

    def reset(self, *, seed=None, options=None):
        """Reset environment for new episode.

        Returns:
            observations: Dict mapping agent_id -> observation
            infos: Dict mapping agent_id -> info dict
        """
        if seed is not None:
            self.seed_value = seed

        obs_dict, info_dict = self.env.reset(seed=seed, options=options)

        # Flatten observations for each agent
        observations = {
            agent: self._flatten_observation(obs)
            for agent, obs in obs_dict.items()
        }

        return observations, info_dict

    def step(self, action_dict):
        """Execute one environment step.

        Args:
            action_dict: Dict mapping agent_id -> action

        Returns:
            observations: Dict mapping agent_id -> observation
            rewards: Dict mapping agent_id -> reward
            terminateds: Dict mapping agent_id -> done flag
            truncateds: Dict mapping agent_id -> truncated flag
            infos: Dict mapping agent_id -> info dict
        """
        # Convert RLlib actions to env actions
        env_actions = {
            agent: self._unflatten_action(action)
            for agent, action in action_dict.items()
        }

        # Step environment
        obs_dict, rewards, terminateds, truncateds, infos = self.env.step(env_actions)

        # Flatten observations
        observations = {
            agent: self._flatten_observation(obs)
            for agent, obs in obs_dict.items()
        }

        # RLlib expects "__all__" key for done
        terminateds["__all__"] = all(terminateds.values())
        truncateds["__all__"] = all(truncateds.values())

        return observations, rewards, terminateds, truncateds, infos

    @property
    def observation_space(self):
        """Return observation space."""
        return self._observation_space

    @property
    def action_space(self):
        """Return action space."""
        return self._action_space

    def get_agent_ids(self):
        """Return set of agent IDs."""
        return self._agent_ids

    def render(self):
        """Render environment (delegates to base env)."""
        return self.env.render()


def make_rllib_env(config: Optional[Dict[str, Any]] = None):
    """Factory function to create RLlib-compatible environment.

    This is the recommended way to create environments for RLlib.

    Args:
        config: Configuration dict passed to RLlibParallelRiskEnv

    Returns:
        RLlibParallelRiskEnv instance

    Example:
        config = {
            "map_name": "simple_6",
            "action_budget": 5,
            "reward_shaping_config": create_dense_config(),
        }
        env = make_rllib_env(config)
    """
    return RLlibParallelRiskEnv(config)
