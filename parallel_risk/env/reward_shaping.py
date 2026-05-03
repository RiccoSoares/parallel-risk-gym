"""
Reward shaping components for Parallel Risk environment.

This module provides dense reward signals to supplement the sparse terminal rewards
(+1 win, -1 loss). All shaped rewards are designed to correlate with winning while
providing intermediate feedback during training.

Design principles:
- All components must correlate with winning probability
- Terminal rewards remain dominant (scaled appropriately)
- Each component is independently tunable via coefficients
- Components can be enabled/disabled via configuration
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class RewardShapingConfig:
    """Configuration for reward shaping components.

    All coefficients should be << 1.0 to keep terminal rewards dominant.
    Recommended starting values provided as defaults.
    """
    # Enable/disable individual components
    enable_territory_control: bool = False    # Disabled - encourages passive survival
    enable_region_completion: bool = True
    enable_troop_advantage: bool = False      # Disabled - encourages passive survival
    enable_strategic_position: bool = False   # Disabled - encourages passive survival
    enable_territory_conquest: bool = True    # Immediate reward for capturing territories
    enable_territory_loss: bool = True        # Penalty for losing territories

    # Coefficient for each reward component
    # These scale the shaped rewards relative to terminal +1/-1
    territory_control_weight: float = 0.01      # Per-step reward for territory percentage
    region_completion_weight: float = 0.1       # One-time bonus per region
    troop_advantage_weight: float = 0.01        # Per-step reward for troop ratio
    strategic_position_weight: float = 0.005    # Per-step reward for connectivity
    territory_conquest_weight: float = 0.1      # Immediate reward per territory captured
    territory_loss_weight: float = 0.1          # Penalty per territory lost (symmetric with conquest)

    # Terminal reward scale (default 1.0 = unchanged)
    terminal_reward_scale: float = 1.0


class RewardShaper:
    """Computes shaped rewards for training RL agents.

    Usage:
        config = RewardShapingConfig(enable_region_completion=True)
        shaper = RewardShaper(config, map_config)

        # At each step:
        shaped_rewards = shaper.compute_step_rewards(game_state, agents)

        # At terminal state:
        terminal_rewards = shaper.scale_terminal_rewards(base_rewards)
    """

    def __init__(self, config: RewardShapingConfig, map_config):
        """Initialize reward shaper.

        Args:
            config: Reward shaping configuration
            map_config: MapConfig instance with map topology
        """
        self.config = config
        self.map_config = map_config

        # Pre-compute strategic value of each territory (connectivity score)
        self._territory_strategic_values = self._compute_strategic_values()

        # Track previous state for region completion detection
        self._previous_controlled_regions = {}

        # Track previous territory ownership for conquest detection
        self._previous_territory_ownership = None

    def reset(self):
        """Reset internal state at episode start."""
        self._previous_controlled_regions = {}
        self._previous_territory_ownership = None

    def begin_step(self, game_state: Dict):
        """Call at the beginning of each step, before actions are executed.

        Captures the game state before actions are taken, allowing conquest
        detection by comparing pre-action and post-action states.

        Args:
            game_state: Current game state dict before action execution
        """
        # Store ownership before actions are executed
        self._pre_step_ownership = game_state['territory_ownership'].copy()

    def _compute_strategic_values(self) -> np.ndarray:
        """Compute strategic value of each territory based on connectivity.

        More connected territories (higher degree) are more strategically valuable
        as they provide more options for expansion and defense.

        Returns:
            Array of shape (n_territories,) with normalized strategic values [0, 1]
        """
        # Degree centrality: number of adjacent territories
        degrees = self.map_config.adjacency_matrix.sum(axis=1)

        # Normalize to [0, 1]
        if degrees.max() > 0:
            strategic_values = degrees / degrees.max()
        else:
            strategic_values = np.zeros_like(degrees, dtype=np.float32)

        return strategic_values

    def compute_step_rewards(
        self,
        game_state: Dict,
        agents: List[str],
        agent_indices: Dict[str, int]
    ) -> Dict[str, float]:
        """Compute shaped rewards for a single step.

        Args:
            game_state: Current game state dict with keys:
                - territory_ownership: np.ndarray
                - territory_troops: np.ndarray
                - turn_number: int
            agents: List of active agent names
            agent_indices: Dict mapping agent name to index

        Returns:
            Dict mapping agent name to shaped reward (float)
        """
        rewards = {agent: 0.0 for agent in agents}

        for agent in agents:
            agent_idx = agent_indices[agent]

            # 1. Territory control reward
            if self.config.enable_territory_control:
                territory_reward = self._compute_territory_control_reward(
                    game_state, agent_idx
                )
                rewards[agent] += self.config.territory_control_weight * territory_reward

            # 2. Region completion bonus (one-time when completing a region)
            if self.config.enable_region_completion:
                region_reward = self._compute_region_completion_reward(
                    game_state, agent, agent_idx
                )
                rewards[agent] += self.config.region_completion_weight * region_reward

            # 3. Troop advantage reward
            if self.config.enable_troop_advantage:
                troop_reward = self._compute_troop_advantage_reward(
                    game_state, agent_idx, agent_indices
                )
                rewards[agent] += self.config.troop_advantage_weight * troop_reward

            # 4. Strategic position reward
            if self.config.enable_strategic_position:
                strategic_reward = self._compute_strategic_position_reward(
                    game_state, agent_idx
                )
                rewards[agent] += self.config.strategic_position_weight * strategic_reward

            # 5. Territory conquest reward (immediate bonus for capturing territories)
            if self.config.enable_territory_conquest:
                conquest_reward = self._compute_territory_conquest_reward(
                    game_state, agent_idx
                )
                rewards[agent] += self.config.territory_conquest_weight * conquest_reward

            # 6. Territory loss penalty (immediate penalty for losing territories)
            if self.config.enable_territory_loss:
                loss_penalty = self._compute_territory_loss_penalty(
                    game_state, agent_idx
                )
                rewards[agent] -= self.config.territory_loss_weight * loss_penalty

        # Update previous ownership tracking for next step
        self._previous_territory_ownership = game_state['territory_ownership'].copy()

        return rewards

    def _compute_territory_control_reward(
        self,
        game_state: Dict,
        agent_idx: int
    ) -> float:
        """Reward based on percentage of territories controlled.

        Returns value in [0, 1] representing territory control percentage.
        """
        owned_territories = np.sum(game_state['territory_ownership'] == agent_idx)
        percentage = owned_territories / self.map_config.n_territories
        return float(percentage)

    def _compute_region_completion_reward(
        self,
        game_state: Dict,
        agent: str,
        agent_idx: int
    ) -> float:
        """One-time bonus when completing a region.

        Returns sum of region bonuses for newly completed regions.
        Only triggers on the first turn a region is completed.
        """
        # Find currently controlled regions
        current_regions = set()
        for region_name, territories in self.map_config.regions.items():
            if all(game_state['territory_ownership'][t] == agent_idx for t in territories):
                current_regions.add(region_name)

        # Find newly completed regions (not in previous state)
        previous_regions = self._previous_controlled_regions.get(agent, set())
        newly_completed = current_regions - previous_regions

        # Update tracking
        self._previous_controlled_regions[agent] = current_regions

        # Sum bonuses for newly completed regions
        bonus = sum(self.map_config.region_bonuses[region] for region in newly_completed)
        return float(bonus)

    def _compute_troop_advantage_reward(
        self,
        game_state: Dict,
        agent_idx: int,
        agent_indices: Dict[str, int]
    ) -> float:
        """Reward based on troop count advantage over opponent.

        Returns ratio of (my_troops / (enemy_troops + 1)).
        Normalized to roughly [0, 2] range with 1.0 = parity.
        """
        # Find opponent index (assuming 2-player)
        opponent_idx = 1 - agent_idx

        # Count total troops for each player
        ownership = game_state['territory_ownership']
        troops = game_state['territory_troops']

        my_troops = np.sum(troops[ownership == agent_idx])
        enemy_troops = np.sum(troops[ownership == opponent_idx])

        # Compute ratio (add 1 to avoid division by zero)
        ratio = my_troops / (enemy_troops + 1.0)

        # Clip to reasonable range [0, 2]
        ratio = np.clip(ratio, 0.0, 2.0)

        return float(ratio)

    def _compute_strategic_position_reward(
        self,
        game_state: Dict,
        agent_idx: int
    ) -> float:
        """Reward based on strategic value of controlled territories.

        Sum of connectivity scores for owned territories.
        Encourages controlling well-connected territories.

        Returns normalized value roughly [0, 1].
        """
        ownership = game_state['territory_ownership']
        owned_mask = (ownership == agent_idx)

        # Sum strategic values of owned territories
        strategic_score = np.sum(self._territory_strategic_values[owned_mask])

        # Normalize by maximum possible (owning all territories)
        max_possible = np.sum(self._territory_strategic_values)
        if max_possible > 0:
            normalized_score = strategic_score / max_possible
        else:
            normalized_score = 0.0

        return float(normalized_score)

    def _compute_territory_conquest_reward(
        self,
        game_state: Dict,
        agent_idx: int
    ) -> float:
        """Immediate reward for capturing enemy territories.

        Returns the number of territories newly captured this step.
        This provides instant positive feedback for aggressive play.

        Note: Requires begin_step() to be called before actions are executed.
        """
        if not hasattr(self, '_pre_step_ownership') or self._pre_step_ownership is None:
            # First step or begin_step not called - no conquest detection possible
            return 0.0

        current_ownership = game_state['territory_ownership']
        previous_ownership = self._pre_step_ownership

        # Count territories that changed from enemy to this agent
        # (was not owned by agent, now is owned by agent)
        was_not_mine = previous_ownership != agent_idx
        is_mine_now = current_ownership == agent_idx
        newly_captured = np.sum(was_not_mine & is_mine_now)

        return float(newly_captured)

    def _compute_territory_loss_penalty(
        self,
        game_state: Dict,
        agent_idx: int
    ) -> float:
        """Penalty for losing territories to enemies.

        Returns the number of territories lost this step.
        This creates strong disincentive for passive play that allows captures.

        Note: Requires begin_step() to be called before actions are executed.
        """
        if not hasattr(self, '_pre_step_ownership') or self._pre_step_ownership is None:
            # First step or begin_step not called - no loss detection possible
            return 0.0

        current_ownership = game_state['territory_ownership']
        previous_ownership = self._pre_step_ownership

        # Count territories that changed from this agent to enemy
        # (was owned by agent, now is not owned by agent)
        was_mine = previous_ownership == agent_idx
        is_not_mine_now = current_ownership != agent_idx
        territories_lost = np.sum(was_mine & is_not_mine_now)

        return float(territories_lost)

    def scale_terminal_rewards(self, base_rewards: Dict[str, float]) -> Dict[str, float]:
        """Scale terminal rewards (+1/-1) by configured factor.

        Args:
            base_rewards: Dict of agent -> reward (typically +1, -1, or 0)

        Returns:
            Scaled rewards dict
        """
        return {
            agent: reward * self.config.terminal_reward_scale
            for agent, reward in base_rewards.items()
        }

    def get_reward_components_info(
        self,
        game_state: Dict,
        agents: List[str],
        agent_indices: Dict[str, int]
    ) -> Dict[str, Dict[str, float]]:
        """Get detailed breakdown of reward components for debugging/analysis.

        Returns:
            Dict mapping agent to dict of reward component names and values
        """
        info = {agent: {} for agent in agents}

        for agent in agents:
            agent_idx = agent_indices[agent]

            if self.config.enable_territory_control:
                info[agent]['territory_control'] = self._compute_territory_control_reward(
                    game_state, agent_idx
                ) * self.config.territory_control_weight

            if self.config.enable_region_completion:
                info[agent]['region_completion'] = self._compute_region_completion_reward(
                    game_state, agent, agent_idx
                ) * self.config.region_completion_weight

            if self.config.enable_troop_advantage:
                info[agent]['troop_advantage'] = self._compute_troop_advantage_reward(
                    game_state, agent_idx, agent_indices
                ) * self.config.troop_advantage_weight

            if self.config.enable_strategic_position:
                info[agent]['strategic_position'] = self._compute_strategic_position_reward(
                    game_state, agent_idx
                ) * self.config.strategic_position_weight

            if self.config.enable_territory_conquest:
                info[agent]['territory_conquest'] = self._compute_territory_conquest_reward(
                    game_state, agent_idx
                ) * self.config.territory_conquest_weight

            if self.config.enable_territory_loss:
                info[agent]['territory_loss'] = -self._compute_territory_loss_penalty(
                    game_state, agent_idx
                ) * self.config.territory_loss_weight

            info[agent]['total_shaped'] = sum(info[agent].values())

        return info


# Default configurations for common use cases

def create_dense_config() -> RewardShapingConfig:
    """All reward components enabled with default weights."""
    return RewardShapingConfig(
        enable_territory_control=True,
        enable_region_completion=True,
        enable_troop_advantage=True,
        enable_strategic_position=True,
    )


def create_sparse_config() -> RewardShapingConfig:
    """No reward shaping, only terminal rewards."""
    return RewardShapingConfig(
        enable_territory_control=False,
        enable_region_completion=False,
        enable_troop_advantage=False,
        enable_strategic_position=False,
    )


def create_territorial_config() -> RewardShapingConfig:
    """Focus on territory and region control only."""
    return RewardShapingConfig(
        enable_territory_control=True,
        enable_region_completion=True,
        enable_troop_advantage=False,
        enable_strategic_position=False,
        territory_control_weight=0.02,
        region_completion_weight=0.15,
    )


def create_aggressive_config() -> RewardShapingConfig:
    """Focus on troop advantage and strategic positioning."""
    return RewardShapingConfig(
        enable_territory_control=False,
        enable_region_completion=False,
        enable_troop_advantage=True,
        enable_strategic_position=True,
        troop_advantage_weight=0.02,
        strategic_position_weight=0.01,
    )


def create_conquest_config() -> RewardShapingConfig:
    """Focus on territory conquest - immediate reward for capturing territories.

    Designed to encourage aggressive play and discourage defensive strategies.
    Includes territory loss penalty to punish being attacked successfully.
    """
    return RewardShapingConfig(
        enable_territory_control=False,
        enable_region_completion=True,
        enable_troop_advantage=False,
        enable_strategic_position=False,
        enable_territory_conquest=True,
        enable_territory_loss=True,
        region_completion_weight=0.15,
        territory_conquest_weight=0.1,
        territory_loss_weight=0.1,
    )
