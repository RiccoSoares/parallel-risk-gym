import numpy as np


class ActionValidator:
    """Validates game actions against current state.

    The single authority for what counts as a legal action. `game_state` may
    be the env's numpy-backed state or a mirror of it whose
    'territory_ownership' / 'territory_troops' are python lists: the env and
    RiskSimulator hand it such mirrors during their sequential
    validate/execute loop because python-int list indexing is several times
    faster than numpy scalar indexing. The validator keeps references to the
    state's containers, so it sees in-place updates made while actions are
    processed (later actions are validated against the state left by earlier
    ones). Adjacency is tested against per-source neighbour sets cached on
    the MapConfig (`adjacency_matrix[source, dest] == 1`).
    """

    def __init__(self, game_state, map_config, possible_agents):
        """Initialize validator with current game state

        Args:
            game_state: Current game state dict (or a list-mirrored copy of it)
            map_config: MapConfig object with map data
            possible_agents: List of all possible agent names
        """
        self.game_state = game_state
        self.map_config = map_config
        self.possible_agents = possible_agents
        self._n_territories = map_config.n_territories
        self._adjacency_sets = map_config.adjacency_sets
        self._ownership = game_state['territory_ownership']
        self._troops = game_state['territory_troops']
        self._available_income = game_state['available_income']
        self._agent_index = {agent: idx for idx, agent in enumerate(possible_agents)}

    def classify_action(self, source, dest):
        """Classify action type based on source and destination

        Args:
            source: Source territory ID
            dest: Destination territory ID

        Returns:
            str: 'deploy', 'transfer', or 'attack'
        """
        if source == dest:
            return 'deploy'
        elif self._ownership[source] == self._ownership[dest]:
            return 'transfer'
        else:
            return 'attack'

    def parse_actions(self, agent, action_dict):
        """Parse and classify one agent's submitted action dict.

        Only the first `num_actions` rows are read; the rest is padding.

        Args:
            agent: Agent name the actions belong to
            action_dict: {'num_actions': int, 'actions': array of [source, dest, troops] rows}

        Returns:
            list of (agent, source, dest, troops, action_type) tuples with python ints,
            in submission order
        """
        num_actions = int(action_dict['num_actions'])
        actions_array = action_dict['actions']
        if isinstance(actions_array, np.ndarray) and actions_array.dtype.kind in 'iu':
            # One bulk conversion instead of three int() calls per row
            all_rows = actions_array.tolist()
            rows = [all_rows[i] for i in range(num_actions)]
        else:
            rows = [[int(value) for value in actions_array[i]] for i in range(num_actions)]
        classify = self.classify_action
        return [(agent, source, dest, troops, classify(source, dest))
                for source, dest, troops in rows]

    def validate_action(self, action_info):
        """Validate if an action is legal based on current game state

        Args:
            action_info: Dict with keys: agent, source, dest, troops, type

        Returns:
            bool: True if action is valid, False otherwise
        """
        return self.validate(
            self._agent_index[action_info['agent']],
            action_info['source'],
            action_info['dest'],
            action_info['troops'],
            action_info['type'],
        )

    def validate(self, agent_idx, source, dest, troops, action_type):
        """Validate one action given as python ints (the hot-loop form of validate_action).

        Args:
            agent_idx: Index of the acting agent in possible_agents
            source, dest: Territory IDs
            troops: Number of troops
            action_type: 'deploy', 'transfer' or 'attack' (from classify_action)

        Returns:
            bool: True if action is valid, False otherwise
        """
        # Basic bounds checking
        n_territories = self._n_territories
        if source < 0 or source >= n_territories or dest < 0 or dest >= n_territories:
            return False
        if troops <= 0:
            return False

        # Source must be owned by agent
        if self._ownership[source] != agent_idx:
            return False

        if action_type == 'deploy':
            # Must have enough income
            if troops > self._available_income[self.possible_agents[agent_idx]]:
                return False
            return True

        if action_type == 'transfer':
            # Adjacent, leave at least 1 troop behind, dest owned by same agent
            if dest not in self._adjacency_sets[source]:
                return False
            if troops >= self._troops[source]:
                return False
            if self._ownership[dest] != agent_idx:
                return False
            return True

        if action_type == 'attack':
            # Adjacent, leave at least 1 troop behind, dest owned by opponent
            if dest not in self._adjacency_sets[source]:
                return False
            if troops >= self._troops[source]:
                return False
            if self._ownership[dest] == agent_idx:
                return False
            return True

        return False
