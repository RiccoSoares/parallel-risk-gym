class ActionValidator:
    """Validates game actions against current state"""

    def __init__(self, game_state, map_config, possible_agents):
        """Initialize validator with current game state

        Args:
            game_state: Current game state dict
            map_config: MapConfig object with map data
            possible_agents: List of all possible agent names
        """
        self.game_state = game_state
        self.map_config = map_config
        self.possible_agents = possible_agents

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
        elif self.game_state['territory_ownership'][source] == self.game_state['territory_ownership'][dest]:
            return 'transfer'
        else:
            return 'attack'

    def validate_action(self, action_info):
        """Validate if an action is legal based on current game state

        Args:
            action_info: Dict with keys: agent, source, dest, troops, type

        Returns:
            bool: True if action is valid, False otherwise
        """
        agent = action_info['agent']
        agent_idx = self.possible_agents.index(agent)
        source = action_info['source']
        dest = action_info['dest']
        troops = action_info['troops']
        action_type = action_info['type']

        # Basic bounds checking
        n_territories = self.map_config.n_territories
        if source < 0 or source >= n_territories or dest < 0 or dest >= n_territories:
            return False
        if troops <= 0:
            return False

        # Source must be owned by agent
        if self.game_state['territory_ownership'][source] != agent_idx:
            return False

        if action_type == 'deploy':
            return self._validate_deploy(agent, source, troops)
        elif action_type == 'transfer':
            return self._validate_transfer(agent_idx, source, dest, troops)
        elif action_type == 'attack':
            return self._validate_attack(agent_idx, source, dest, troops)

        return False

    def _validate_deploy(self, agent, source, troops):
        """Validate deployment action"""
        # Must have enough income
        if troops > self.game_state['available_income'][agent]:
            return False
        return True

    def _validate_transfer(self, agent_idx, source, dest, troops):
        """Validate transfer action"""
        # Territories must be adjacent
        if self.map_config.adjacency_matrix[source, dest] != 1:
            return False
        # Must have enough troops (leave at least 1)
        if troops >= self.game_state['territory_troops'][source]:
            return False
        # Dest must be owned by same agent
        if self.game_state['territory_ownership'][dest] != agent_idx:
            return False
        return True

    def _validate_attack(self, agent_idx, source, dest, troops):
        """Validate attack action"""
        # Territories must be adjacent
        if self.map_config.adjacency_matrix[source, dest] != 1:
            return False
        # Must have enough troops (leave at least 1)
        if troops >= self.game_state['territory_troops'][source]:
            return False
        # Dest must be owned by opponent
        if self.game_state['territory_ownership'][dest] == agent_idx:
            return False
        return True
