import pettingzoo
from gymnasium import spaces
import numpy as np
import random

class ParallelRiskEnv(pettingzoo.ParallelEnv):
    metadata = {'name': 'parallel_risk_env_v0'}

    def __init__(
        self,
        map_name: str = "simple_6",
        max_actions_per_turn: int = 10,
        income_per_turn: int = 5,
        max_turns: int = 100,
        initial_troops_per_territory: int = 3,
        seed: int = None
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

        # Initialize map
        self.map_config = self._initialize_map()
        n_territories = self.map_config['n_territories']

        # Calculate max possible income (base + all region bonuses)
        max_income = income_per_turn + sum(self.map_config['region_bonuses'].values())

        # Define observation spaces
        n_regions = len(self.map_config['regions'])
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

    def _initialize_map(self):
        """Initialize map structure - simple 6-territory grid"""
        if self.map_name == "simple_6":
            # Map layout:
            # 0 - 1 - 2  (North region)
            # |   |   |
            # 3 - 4 - 5  (South region)
            adjacency_list = {
                0: [1, 3],
                1: [0, 2, 4],
                2: [1, 5],
                3: [0, 4],
                4: [1, 3, 5],
                5: [2, 4],
            }
            n_territories = 6

            # Build adjacency matrix
            adjacency_matrix = np.zeros((n_territories, n_territories), dtype=np.int8)
            for source, neighbors in adjacency_list.items():
                for dest in neighbors:
                    adjacency_matrix[source, dest] = 1

            # Initial ownership: agent_0 gets [0, 1, 5], agent_1 gets [2, 3, 4]
            initial_ownership = np.array([0, 0, 1, 1, 1, 0], dtype=np.int8)

            # Define bonus regions
            regions = {
                'north': [0, 1, 2],
                'south': [3, 4, 5],
                'center': [1, 4],
            }

            # Define bonus troops per region
            region_bonuses = {
                'north': 4,
                'south': 4,
                'center': 2,
            }

            return {
                'n_territories': n_territories,
                'adjacency_list': adjacency_list,
                'adjacency_matrix': adjacency_matrix,
                'initial_ownership': initial_ownership,
                'regions': regions,
                'region_bonuses': region_bonuses,
            }
        else:
            raise ValueError(f"Unknown map name: {self.map_name}")

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.agents = self.possible_agents[:]

        # Initialize game state
        self.game_state = {
            'territory_ownership': self.map_config['initial_ownership'].copy(),
            'territory_troops': np.full(
                self.map_config['n_territories'],
                self.initial_troops_per_territory,
                dtype=np.int32
            ),
            'turn_number': 0,
            'income_per_turn': self.income_per_turn,
            'available_income': {agent: self.income_per_turn for agent in self.agents},
        }

        # Generate initial observations
        observations = {agent: self._get_observation(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        return observations, infos

    def _get_observation(self, agent):
        """Generate observation for a specific agent"""
        agent_idx = self.possible_agents.index(agent)

        # Convert ownership to agent perspective (-1: enemy, 1: self)
        ownership = np.where(
            self.game_state['territory_ownership'] == agent_idx,
            1,
            -1
        ).astype(np.int8)

        # Check region control (1 if agent controls region, 0 otherwise)
        controlled_regions = self._check_region_control(agent)
        region_control = np.zeros(len(self.map_config['regions']), dtype=np.int8)
        for i, region_name in enumerate(self.map_config['regions'].keys()):
            if region_name in controlled_regions:
                region_control[i] = 1

        return {
            'territory_ownership': ownership,
            'territory_troops': self.game_state['territory_troops'].copy(),
            'adjacency_matrix': self.map_config['adjacency_matrix'].copy(),
            'available_income': np.array([self.game_state['available_income'][agent]], dtype=np.int32),
            'turn_number': np.array([self.game_state['turn_number']], dtype=np.int32),
            'region_control': region_control,
        }

    def _classify_action(self, source, dest):
        """Classify action type based on source and destination"""
        if source == dest:
            return 'deploy'
        elif self.game_state['territory_ownership'][source] == self.game_state['territory_ownership'][dest]:
            return 'transfer'
        else:
            return 'attack'

    def _check_region_control(self, agent):
        """Check which regions are fully controlled by agent"""
        agent_idx = self.possible_agents.index(agent)
        controlled_regions = []

        for region_name, territories in self.map_config['regions'].items():
            if all(self.game_state['territory_ownership'][t] == agent_idx for t in territories):
                controlled_regions.append(region_name)

        return controlled_regions

    def _calculate_income(self, agent):
        """Calculate income for agent including region bonuses"""
        base_income = self.income_per_turn

        # Add region bonuses
        controlled_regions = self._check_region_control(agent)
        region_bonus = sum(self.map_config['region_bonuses'][region] for region in controlled_regions)

        return base_income + region_bonus

    def _validate_action(self, action_info):
        """Validate if an action is legal based on current game state"""
        agent = action_info['agent']
        agent_idx = self.possible_agents.index(agent)
        source = action_info['source']
        dest = action_info['dest']
        troops = action_info['troops']
        action_type = action_info['type']

        # Basic bounds checking
        n_territories = self.map_config['n_territories']
        if source < 0 or source >= n_territories or dest < 0 or dest >= n_territories:
            return False
        if troops <= 0:
            return False

        # Source must be owned by agent
        if self.game_state['territory_ownership'][source] != agent_idx:
            return False

        if action_type == 'deploy':
            # Must have enough income
            if troops > self.game_state['available_income'][agent]:
                return False
            return True

        elif action_type == 'transfer':
            # Territories must be adjacent
            if self.map_config['adjacency_matrix'][source, dest] != 1:
                return False
            # Must have enough troops (leave at least 1)
            if troops >= self.game_state['territory_troops'][source]:
                return False
            # Dest must be owned by same agent
            if self.game_state['territory_ownership'][dest] != agent_idx:
                return False
            return True

        elif action_type == 'attack':
            # Territories must be adjacent
            if self.map_config['adjacency_matrix'][source, dest] != 1:
                return False
            # Must have enough troops (leave at least 1)
            if troops >= self.game_state['territory_troops'][source]:
                return False
            # Dest must be owned by opponent
            if self.game_state['territory_ownership'][dest] == agent_idx:
                return False
            return True

        return False

    def _resolve_combat(self, attacking_troops, defending_troops):
        """Resolve combat deterministically"""
        attack_power = attacking_troops
        defense_power = defending_troops * 1.5

        if attack_power > defense_power:
            # Attacker wins
            surviving_troops = max(1, int(attacking_troops - defending_troops * 0.5))
            return 'attacker_wins', surviving_troops
        else:
            # Defender holds
            defender_losses = int(defending_troops * 0.3)
            return 'defender_holds', max(0, defending_troops - defender_losses)

    def _execute_action(self, action_info):
        """Execute a validated action"""
        agent = action_info['agent']
        agent_idx = self.possible_agents.index(agent)
        source = action_info['source']
        dest = action_info['dest']
        troops = action_info['troops']
        action_type = action_info['type']

        if action_type == 'deploy':
            self.game_state['territory_troops'][dest] += troops
            self.game_state['available_income'][agent] -= troops

        elif action_type == 'transfer':
            self.game_state['territory_troops'][source] -= troops
            self.game_state['territory_troops'][dest] += troops

        elif action_type == 'attack':
            # Remove troops from source
            self.game_state['territory_troops'][source] -= troops

            # Resolve combat
            defending_troops = self.game_state['territory_troops'][dest]
            result, surviving_troops = self._resolve_combat(troops, defending_troops)

            if result == 'attacker_wins':
                # Capture territory
                self.game_state['territory_ownership'][dest] = agent_idx
                self.game_state['territory_troops'][dest] = surviving_troops
            else:
                # Defender holds
                self.game_state['territory_troops'][dest] = surviving_troops

    def _check_termination(self):
        """Check if game should terminate and calculate rewards"""
        # Count territories per agent
        territory_counts = {}
        for agent_idx, agent in enumerate(self.possible_agents):
            count = np.sum(self.game_state['territory_ownership'] == agent_idx)
            territory_counts[agent] = count

        # Check victory condition (one agent owns all)
        for agent, count in territory_counts.items():
            if count == self.map_config['n_territories']:
                terminations = {a: True for a in self.possible_agents}
                rewards = {a: (1.0 if a == agent else -1.0) for a in self.possible_agents}
                return terminations, rewards

        # Check elimination condition (one agent has 0 territories)
        eliminated = [agent for agent, count in territory_counts.items() if count == 0]
        if eliminated:
            remaining = [a for a in self.possible_agents if a not in eliminated]
            if len(remaining) == 1:
                terminations = {a: True for a in self.possible_agents}
                rewards = {a: (1.0 if a == remaining[0] else -1.0) for a in self.possible_agents}
                return terminations, rewards

        # Check turn limit
        if self.game_state['turn_number'] >= self.max_turns:
            terminations = {a: True for a in self.possible_agents}
            winner = max(territory_counts, key=territory_counts.get)
            rewards = {a: (0.5 if a == winner else -0.5) for a in self.possible_agents}
            return terminations, rewards

        # Game continues
        return {a: False for a in self.possible_agents}, {a: 0.0 for a in self.possible_agents}

    def step(self, actions):
        """Process one turn of parallel actions"""
        # Calculate income for each agent based on region control
        for agent in self.agents:
            self.game_state['available_income'][agent] = self._calculate_income(agent)

        # Parse all actions into a flat list with agent attribution
        all_actions = []
        infos = {agent: {'invalid_actions': 0, 'controlled_regions': [], 'income': 0} for agent in self.agents}

        for agent in self.agents:
            if agent not in actions:
                continue

            action_dict = actions[agent]
            num_actions = int(action_dict['num_actions'])

            for i in range(num_actions):
                source, dest, troops = action_dict['actions'][i]
                source, dest, troops = int(source), int(dest), int(troops)

                action_type = self._classify_action(source, dest)
                all_actions.append({
                    'agent': agent,
                    'source': source,
                    'dest': dest,
                    'troops': troops,
                    'type': action_type
                })

        # Shuffle actions randomly
        random.shuffle(all_actions)

        # Process actions sequentially
        for action_info in all_actions:
            if self._validate_action(action_info):
                self._execute_action(action_info)
            else:
                # Track invalid actions
                infos[action_info['agent']]['invalid_actions'] += 1

        # Increment turn counter
        self.game_state['turn_number'] += 1

        # Store region control and income info AFTER actions processed
        for agent in self.agents:
            infos[agent]['controlled_regions'] = self._check_region_control(agent)
            infos[agent]['income'] = self._calculate_income(agent)

        # Check termination conditions
        terminations, rewards = self._check_termination()
        truncations = {agent: False for agent in self.possible_agents}

        # Generate observations for remaining agents
        observations = {agent: self._get_observation(agent) for agent in self.agents}

        # If game ended, ensure both agents get terminations
        if any(terminations.values()):
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def render(self):
        """Render current game state"""
        if self.game_state is None:
            print("Game not started")
            return

        print(f"\n=== Turn {self.game_state['turn_number']} ===")
        print("\nTerritory Ownership and Troops:")
        for i in range(self.map_config['n_territories']):
            owner_idx = self.game_state['territory_ownership'][i]
            owner = self.possible_agents[owner_idx]
            troops = self.game_state['territory_troops'][i]
            print(f"  Territory {i}: {owner} ({troops} troops)")

        print("\nRegion Control:")
        for agent in self.possible_agents:
            controlled = self._check_region_control(agent)
            if controlled:
                bonuses = sum(self.map_config['region_bonuses'][r] for r in controlled)
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
