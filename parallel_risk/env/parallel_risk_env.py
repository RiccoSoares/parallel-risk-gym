import pettingzoo
from gymnasium import spaces
import numpy as np
import random

from parallel_risk.env.map_config import MapRegistry
from parallel_risk.env.combat import CombatResolver
from parallel_risk.env.validators import ActionValidator
from parallel_risk.env.reward_shaping import RewardShaper, RewardShapingConfig


class ParallelRiskEnv(pettingzoo.ParallelEnv):
    metadata = {'name': 'parallel_risk_env_v0'}

    def __init__(
        self,
        map_name: str = "simple_6",
        max_actions_per_turn: int = 10,
        income_per_turn: int = 5,
        max_turns: int = 100,
        initial_troops_per_territory: int = 3,
        seed: int = None,
        reward_shaping_config: RewardShapingConfig = None
    ):
        self.map_name = map_name
        self.max_actions_per_turn = max_actions_per_turn
        self.income_per_turn = income_per_turn
        self.max_turns = max_turns
        self.initial_troops_per_territory = initial_troops_per_territory
        self._seed = seed

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Initialize agents
        self.possible_agents = ["agent_0", "agent_1"]
        self.agents = self.possible_agents[:]
        self._agent_index = {agent: idx for idx, agent in enumerate(self.possible_agents)}

        # Initialize map from registry
        self.map_config = MapRegistry.get(self.map_name)
        n_territories = self.map_config.n_territories
        # (name, territories, bonus) per region, in map order: used by every region scan
        self._region_items = self.map_config.region_items

        # Initialize reward shaping (None = no shaping, sparse rewards only)
        self.reward_shaper = None
        if reward_shaping_config is not None:
            self.reward_shaper = RewardShaper(reward_shaping_config, self.map_config)

        # Calculate max possible income (base + all region bonuses)
        max_income = income_per_turn + sum(self.map_config.region_bonuses.values())

        # Define observation spaces
        n_regions = len(self.map_config.regions)
        self.observation_spaces = {
            agent: spaces.Dict({
                'territory_ownership': spaces.Box(low=-1, high=1, shape=(n_territories,), dtype=np.int8),
                'territory_troops': spaces.Box(low=0, high=100, shape=(n_territories,), dtype=np.int32),
                'adjacency_matrix': spaces.Box(low=0, high=1, shape=(n_territories, n_territories), dtype=np.int8),
                'available_income': spaces.Box(low=0, high=max_income, shape=(1,), dtype=np.int32),
                'turn_number': spaces.Box(low=0, high=max_turns, shape=(1,), dtype=np.int32),
                'region_control': spaces.Box(low=0, high=1, shape=(n_regions,), dtype=np.int8),
            })
            for agent in self.possible_agents
        }

        # Define action spaces
        self.action_spaces = {
            agent: spaces.Dict({
                'num_actions': spaces.Discrete(max_actions_per_turn + 1),
                'actions': spaces.Box(
                    low=0,
                    high=n_territories,
                    shape=(max_actions_per_turn, 3),
                    dtype=np.int32
                )
            })
            for agent in self.possible_agents
        }

        # Initialize game state (will be properly set in reset())
        self.game_state = None

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.agents = self.possible_agents[:]

        # Initialize game state
        self.game_state = {
            'territory_ownership': self.map_config.initial_ownership.copy(),
            'territory_troops': np.full(
                self.map_config.n_territories,
                self.initial_troops_per_territory,
                dtype=np.int32
            ),
            'turn_number': 0,
            'income_per_turn': self.income_per_turn,
            'available_income': {agent: self.income_per_turn for agent in self.agents},
        }

        # Reset reward shaper if enabled
        if self.reward_shaper is not None:
            self.reward_shaper.reset()

        # Generate initial observations
        observations = {agent: self._get_observation(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        return observations, infos

    def _scan_regions(self, ownership, agent_idx):
        """One pass over the regions for an agent.

        Args:
            ownership: territory ownership as a python list (or array) of agent indices
            agent_idx: agent index to check

        Returns:
            (controlled_region_names, income): region names fully owned by the agent,
            in map order, and base income plus their bonuses
        """
        controlled_regions = []
        income = self.income_per_turn
        for region_name, territories, bonus in self._region_items:
            for territory in territories:
                if ownership[territory] != agent_idx:
                    break
            else:
                controlled_regions.append(region_name)
                income += bonus
        return controlled_regions, income

    def _get_observation(self, agent):
        """Generate observation for a specific agent"""
        agent_idx = self._agent_index[agent]
        controlled_regions, _ = self._scan_regions(
            self.game_state['territory_ownership'].tolist(), agent_idx)
        return self._build_observation(agent, agent_idx, controlled_regions)

    def _build_observation(self, agent, agent_idx, controlled_regions):
        """Observation dict for agent, given its controlled region names (from _scan_regions)."""
        game_state = self.game_state

        # Convert ownership to agent perspective (-1: enemy, 1: self)
        ownership = np.where(
            game_state['territory_ownership'] == agent_idx,
            np.int8(1),
            np.int8(-1)
        )

        # Region control (1 if agent controls region, 0 otherwise)
        region_control = np.zeros(len(self._region_items), dtype=np.int8)
        if controlled_regions:
            for i, (region_name, _territories, _bonus) in enumerate(self._region_items):
                if region_name in controlled_regions:
                    region_control[i] = 1

        return {
            'territory_ownership': ownership,
            'territory_troops': game_state['territory_troops'].copy(),
            'adjacency_matrix': self.map_config.adjacency_matrix.copy(),
            'available_income': np.array([game_state['available_income'][agent]], dtype=np.int32),
            'turn_number': np.array([game_state['turn_number']], dtype=np.int32),
            'region_control': region_control,
        }

    def _check_region_control(self, agent):
        """Check which regions are fully controlled by agent"""
        controlled_regions, _ = self._scan_regions(
            self.game_state['territory_ownership'].tolist(), self._agent_index[agent])
        return controlled_regions

    def _calculate_income(self, agent):
        """Calculate income for agent including region bonuses"""
        _, income = self._scan_regions(
            self.game_state['territory_ownership'].tolist(), self._agent_index[agent])
        return income

    def _check_termination(self):
        """Check if game has ended.

        Returns:
            terminations: Dict of agent -> bool (True if game naturally ended)
            truncations: Dict of agent -> bool (True if episode artificially cut off)
            rewards: Dict of agent -> float (terminal rewards)

        Note: RL semantics distinguish termination (natural game end like victory/elimination)
        from truncation (artificial cutoff like turn limit). This affects value bootstrapping
        in algorithms like PPO - terminated states bootstrap with 0, truncated states should
        bootstrap with V(s') since the game would continue.
        """
        return self._termination_from(self.game_state['territory_ownership'].tolist())

    def _termination_from(self, ownership):
        """_check_termination on a python-list view of territory ownership."""
        possible_agents = self.possible_agents
        no_termination = {a: False for a in possible_agents}
        no_truncation = {a: False for a in possible_agents}
        no_rewards = {a: 0.0 for a in possible_agents}

        # Count territories per agent
        territory_counts = {
            agent: ownership.count(agent_idx)
            for agent_idx, agent in enumerate(possible_agents)
        }

        # Check victory condition (one agent owns all) - TRUE TERMINATION
        for agent, count in territory_counts.items():
            if count == self.map_config.n_territories:
                terminations = {a: True for a in possible_agents}
                rewards = {a: (1.0 if a == agent else -1.0) for a in possible_agents}
                return terminations, no_truncation, rewards

        # Check elimination condition (one agent has 0 territories) - TRUE TERMINATION
        eliminated = [agent for agent, count in territory_counts.items() if count == 0]
        if eliminated:
            remaining = [a for a in possible_agents if a not in eliminated]
            if len(remaining) == 1:
                terminations = {a: True for a in possible_agents}
                rewards = {a: (1.0 if a == remaining[0] else -1.0) for a in possible_agents}
                return terminations, no_truncation, rewards

        # Check turn limit - TRUNCATION (not termination!)
        # Neutral rewards (0, 0) - incentives come from shaped rewards during gameplay:
        # - Territory conquest (+0.1) rewards attacking
        # - Territory loss (-0.08) punishes being captured
        # - This avoids perverse incentives where agents play defensively to avoid turn limit penalties
        if self.game_state['turn_number'] >= self.max_turns:
            truncations = {a: True for a in possible_agents}
            rewards = {a: 0.0 for a in possible_agents}
            return no_termination, truncations, rewards

        # Game continues
        return no_termination, no_truncation, no_rewards

    def step(self, actions):
        """Process one turn of parallel actions"""
        game_state = self.game_state
        agents = self.agents
        agent_index = self._agent_index
        available_income = game_state['available_income']

        # Capture pre-step state for reward shaping (conquest detection)
        if self.reward_shaper is not None:
            self.reward_shaper.begin_step(game_state)

        # Python-list mirrors of the state arrays: the sequential validate/execute
        # loop below runs on python ints and writes back to the arrays once at the end.
        ownership = game_state['territory_ownership'].tolist()
        troops = game_state['territory_troops'].tolist()

        # Calculate income for each agent based on region control
        for agent in agents:
            _, available_income[agent] = self._scan_regions(ownership, agent_index[agent])

        infos = {agent: {'invalid_actions': 0, 'controlled_regions': [], 'income': 0} for agent in agents}

        # Create validator on the mirrored state (it sees the in-place updates below)
        validator = ActionValidator(
            {'territory_ownership': ownership, 'territory_troops': troops,
             'available_income': available_income},
            self.map_config, self.possible_agents)

        # Parse all actions into a flat list with agent attribution
        all_actions = []
        for agent in agents:
            if agent in actions:
                all_actions.extend(validator.parse_actions(agent, actions[agent]))

        # Shuffle actions randomly
        random.shuffle(all_actions)

        # Process actions sequentially
        validate = validator.validate
        for agent, source, dest, count, action_type in all_actions:
            agent_idx = agent_index[agent]
            if not validate(agent_idx, source, dest, count, action_type):
                # Track invalid actions
                infos[agent]['invalid_actions'] += 1
                continue

            if action_type == 'deploy':
                troops[dest] += count
                available_income[agent] -= count

            elif action_type == 'transfer':
                troops[source] -= count
                troops[dest] += count

            else:  # attack
                # Remove troops from source
                troops[source] -= count

                # Resolve combat using CombatResolver
                defending_troops = troops[dest]
                result, surviving_troops = CombatResolver.resolve(count, defending_troops)

                if result == 'attacker_wins':
                    # Capture territory
                    ownership[dest] = agent_idx
                    troops[dest] = surviving_troops
                else:
                    # Defender holds - return surviving attackers to source
                    attacker_casualties = int(defending_troops * 0.6)
                    troops[source] += max(0, count - attacker_casualties)
                    troops[dest] = surviving_troops

        # Write the mirrors back into the state arrays
        game_state['territory_ownership'][:] = ownership
        game_state['territory_troops'][:] = troops

        # Increment turn counter
        game_state['turn_number'] += 1

        # Store region control and income info AFTER actions processed
        controlled_regions = {}
        for agent in agents:
            controlled_regions[agent], infos[agent]['income'] = self._scan_regions(
                ownership, agent_index[agent])
            infos[agent]['controlled_regions'] = controlled_regions[agent]

        # Check termination and truncation conditions
        terminations, truncations, terminal_rewards = self._termination_from(ownership)

        # Compute rewards (shaped + terminal)
        rewards = {agent: 0.0 for agent in self.possible_agents}

        # Add shaped rewards if enabled, with the component breakdown in info for debugging
        if self.reward_shaper is not None:
            shaped_rewards, reward_components = self.reward_shaper.compute_step_rewards_with_info(
                game_state, agents, agent_index
            )
            for agent in agents:
                rewards[agent] += shaped_rewards[agent]
                infos[agent]['reward_components'] = reward_components[agent]

        # Add terminal rewards (scaled if shaper is enabled)
        # Terminal rewards apply for both true termination AND truncation (turn limit)
        episode_ended = any(terminations.values()) or any(truncations.values())
        if episode_ended:
            if self.reward_shaper is not None:
                terminal_rewards = self.reward_shaper.scale_terminal_rewards(terminal_rewards)

            for agent in self.possible_agents:
                rewards[agent] += terminal_rewards[agent]

        # Generate observations for remaining agents (region control already computed above)
        observations = {
            agent: self._build_observation(agent, agent_index[agent], controlled_regions[agent])
            for agent in agents
        }

        # If game ended (either terminated or truncated), clear agents list
        if episode_ended:
            self.agents = []

        # Set __all__ flags correctly
        terminations['__all__'] = any(terminations.values())
        truncations['__all__'] = any(truncations.values())

        return observations, rewards, terminations, truncations, infos

    def render(self):
        """Render current game state"""
        if self.game_state is None:
            print("Game not started")
            return

        print(f"\n=== Turn {self.game_state['turn_number']} ===")
        print("\nTerritory Ownership and Troops:")
        for i in range(self.map_config.n_territories):
            owner_idx = self.game_state['territory_ownership'][i]
            owner = self.possible_agents[owner_idx]
            troops = self.game_state['territory_troops'][i]
            print(f"  Territory {i}: {owner} ({troops} troops)")

        print("\nRegion Control:")
        for agent in self.possible_agents:
            controlled = self._check_region_control(agent)
            if controlled:
                bonuses = sum(self.map_config.region_bonuses[r] for r in controlled)
                print(f"  {agent}: {controlled} (+{bonuses} bonus troops)")
            else:
                print(f"  {agent}: None")

        print("\nIncome (Base + Region Bonuses):")
        for agent in self.possible_agents:
            if agent in self.game_state['available_income']:
                income = self._calculate_income(agent)
                base = self.income_per_turn
                bonus = income - base
                print(f"  {agent}: {income} ({base} base + {bonus} bonus)")
                print(f"    Available: {self.game_state['available_income'][agent]}")

    def observe(self, agent):
        """Return observation for specific agent"""
        return self._get_observation(agent)
